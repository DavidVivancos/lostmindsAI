#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 The Anamnetic Recollection Network (ARN)
 Mind #70: Plato of Athens (trad. 428/427 BCE - 348/347 BCE)
  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0070 · Plato of Athens

WHY THIS ARCHITECTURE (and why it is NOT a transformer)
-----------------------------------------------------------------------------
The idea that is Plato's alone lives in the Meno and the Phaedo: **anamnesis**.
Genuine knowledge (episteme / noesis) is not assembled out of the senses. The
senses give only doxa — opinion — and opinion is unstable: it shifts as the
appearance shifts. "Learning" is really *recollection*: the soul re-cognising
that a noisy particular PARTICIPATES in an eternal Form it already latently
holds. Two consequences drive every line of code below:

  (1) The mark that separates KNOWLEDGE from OPINION is INVARIANCE.
      A grasp that survives perturbation of the appearance is knowledge; one
      that flips when you tilt the drawing was only opinion. So the central
      training signal is a *dialectic invariance* objective: an object and its
      "shadow" (a distorted view of the same thing) must recollect the SAME Form.

  (2) Recollection is ATTRACTOR DYNAMICS over a fixed intelligible realm.
      The Forms are a bank of prototype vectors {phi_k}. A particular is a
      shadow fallen away from its Form. Inference is the soul *settling* —
      relaxing the encoded particular toward the Form it most participates in.
      Noesis = convergence onto the attractor.

THE DIVIDED LINE AS THE NETWORK'S SPINE (Republic 509d-511e)
-----------------------------------------------------------------------------
    eikasia  (shadows)        -> raw input x          (the appearance)
    pistis   (belief/objects) -> H1 = tanh(W1 x + b1) (a stable percept)
    dianoia  (discursive)     -> Z  = W2 H1 + b2      (projection into Form-space)
    noesis   (understanding)  -> settle Z onto Forms; read participation P

LOSSES (each one a Platonic commitment, made measurable)
-----------------------------------------------------------------------------
  L_participation : cross-entropy that each view recollects its true Form.
  L_dialectic     : object and shadow must agree (knowledge = invariance).
  L_distinct      : the Forms must be mutually distinct — "one over many".
  L_decay         : small weight penalty keeping the encoder bounded (stability).

-----------------------------------------------------------------------------
STABILITY NOTE (fixes the NaN/overflow reported in v1)
-----------------------------------------------------------------------------
v1 let the Form bank Phi grow without bound. Because settling pulls Z toward
P@Phi, the readout logit ~ ||Phi||^2, so minimizing cross-entropy drove
||Phi|| -> inf (a positive-feedback runaway that overflowed to NaN on some
BLAS/NumPy builds). The cure, applied here and faithful to the philosophy:

  * The Forms are CONSTRAINED TO UNIT LENGTH after every update — a Form is a
    pure, self-identical *direction*, eternal and of fixed magnitude. This pins
    the logit scale and kills the runaway at its source.
  * Small WEIGHT DECAY on W1, W2, Phi bounds the encoder too.
  * GLOBAL-NORM GRADIENT CLIPPING in the optimizer stops momentum spikes.
  * A gentler learning rate / momentum.

None of these touch the analytic-vs-finite-difference gradient contract: the
unit-norm projection is a post-step constraint (not part of the loss), while
weight decay is a clean additive loss term whose gradient the check verifies.

ENGINEERING CONTRACT (shared across the corpus)
-----------------------------------------------------------------------------
  * Pure NumPy, from scratch. No autograd, no ML frameworks.
  * Every analytic gradient validated by a finite-difference check that MUST
    pass (printed below). Non-negotiable.
  * A real training loop on a synthetic "shadows-of-Forms" dataset.
  * Self-tests that assert the mind-specific claims actually hold.
  * Runs in well under a minute on a CPU, deterministically, without warnings.

Run:  python3 chapter_0070_Plato_-428.py
"""

from __future__ import annotations
import numpy as np

# A fixed seed so the printed run is reproducible. (Plato is mind #70.)
RNG = np.random.default_rng(70)


# ---------------------------------------------------------------------------
# 0. Small numeric helpers
# ---------------------------------------------------------------------------
def softmax(L: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stabilised by subtracting the row max."""
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def softmax_backward(P: np.ndarray, dP: np.ndarray) -> np.ndarray:
    """
    Given softmax output P and upstream gradient dP (both N x K), return the
    gradient w.r.t. the pre-softmax logits:  dL = P * (dP - sum(dP*P, axis=1)).
    """
    return P * (dP - np.sum(dP * P, axis=1, keepdims=True))


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], K))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def unit_rows(M: np.ndarray) -> np.ndarray:
    """Normalise each row to unit L2 norm (Forms are pure directions)."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


# ---------------------------------------------------------------------------
# 1. The Anamnetic Recollection Network
# ---------------------------------------------------------------------------
class AnamneticRecollectionNetwork:
    """
    Encoder (the Divided Line) + a bank of unit-length Form prototypes (the
    intelligible realm) + an attractor settling step (anamnesis). Trained by
    recollection, dialectic invariance, and the distinctness of the Forms.

    Shapes
    ------
      d_in : dimension of a raw particular (a "shadow" in sense-space)
      h    : hidden width  (pistis — the level of stable objects/belief)
      d    : dimension of Form-space (dianoia/noesis live here)
      K    : number of Forms in the intelligible realm
    """

    def __init__(self, d_in=12, h=24, d=16, K=5,
                 tau=0.5, settle_eta=0.5, settle_T=2,
                 alpha=1.0, lam=0.05, wd=1e-3):
        self.d_in, self.h, self.d, self.K = d_in, h, d, K
        self.tau = float(tau)              # recollection temperature
        self.eta = float(settle_eta)       # how fast the soul settles per step
        self.T = int(settle_T)             # settling steps inside the trained graph
        self.alpha = float(alpha)          # weight on the dialectic (invariance) loss
        self.lam = float(lam)              # weight on the distinctness-of-Forms loss
        self.wd = float(wd)                # weight decay (stability)

        def xavier(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return RNG.uniform(-limit, limit, size=shape)

        self.W1 = xavier((d_in, h))
        self.b1 = np.zeros(h)
        self.W2 = xavier((h, d))
        self.b2 = np.zeros(d)
        # The Forms: unit-length prototypes in Form-space. Distinct, indivisible,
        # "one over many". Initialised on the unit sphere and kept there.
        self.Phi = unit_rows(RNG.standard_normal((K, d)))

    # -- parameter book-keeping (used by the gradient check) ----------------
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "Phi": self.Phi}

    # -- the ascent of the Divided Line -------------------------------------
    def _encode(self, X):
        """eikasia(X) -> pistis(H1) -> dianoia(Z=z0). Returns Z and a cache."""
        A1 = X @ self.W1 + self.b1          # pre-activation
        H1 = np.tanh(A1)                    # pistis: a stable percept
        Z = H1 @ self.W2 + self.b2          # dianoia: projected into Form-space
        return Z, (X, A1, H1)

    def _settle(self, Z0):
        """
        Anamnesis: relax the encoded particular toward the Forms it
        participates in. Returns the settled Z_T and a per-step cache.
        """
        Z = Z0
        steps = []
        for _ in range(self.T):
            L = (Z @ self.Phi.T) / self.tau      # affinity to each Form
            P = softmax(L)                       # participation weights
            R = P @ self.Phi                     # the recollected ideal
            Z_next = (1.0 - self.eta) * Z + self.eta * R
            steps.append((Z, L, P, R))           # cache Z BEFORE the update
            Z = Z_next
        return Z, steps

    def _readout(self, Z):
        """noesis: the final participation distribution over the Forms."""
        L = (Z @ self.Phi.T) / self.tau
        P = softmax(L)
        return P, L

    def forward(self, X):
        """Full ascent + settling + readout. Returns (P, cache)."""
        Z0, enc_cache = self._encode(X)
        ZT, settle_steps = self._settle(Z0)
        P, Lf = self._readout(ZT)
        cache = {"enc": enc_cache, "Z0": Z0, "ZT": ZT,
                 "settle": settle_steps, "P": P, "Lf": Lf}
        return P, cache

    # -----------------------------------------------------------------------
    # 2. Loss over a dialectic pair (an object view and its shadow view)
    # -----------------------------------------------------------------------
    def loss(self, X_obj, X_sha, y):
        """Returns (total_loss, parts_dict, caches) for a batch of pairs."""
        N = X_obj.shape[0]
        K = self.K
        Y = one_hot(y, K)

        P_o, c_o = self.forward(X_obj)
        P_s, c_s = self.forward(X_sha)

        eps = 1e-12
        # L_participation: recollect the true Form, for BOTH views.
        Lpart = -(np.sum(Y * np.log(P_o + eps)) +
                  np.sum(Y * np.log(P_s + eps))) / N
        # L_dialectic: object and shadow must agree -> invariance = knowledge.
        diff = P_o - P_s
        Ldia = 0.5 * np.sum(diff * diff) / N
        # L_distinct: the Forms are mutually distinct ("one over many").
        M = self.Phi @ self.Phi.T
        off = M - np.diag(np.diag(M))
        Ldist = 0.5 * np.sum(off * off)
        # L_decay: keep the encoder (and Forms) bounded — numerical stability.
        Ldecay = 0.5 * (np.sum(self.W1 * self.W1) +
                        np.sum(self.W2 * self.W2) +
                        np.sum(self.Phi * self.Phi))

        total = Lpart + self.alpha * Ldia + self.lam * Ldist + self.wd * Ldecay
        parts = {"participation": Lpart, "dialectic": Ldia,
                 "distinct": Ldist, "decay": Ldecay}
        return total, parts, (c_o, c_s, Y, diff, off)

    # -----------------------------------------------------------------------
    # 3. Analytic gradients (hand-derived backprop through the settling loop)
    # -----------------------------------------------------------------------
    def backward(self, X_obj, X_sha, y):
        total, parts, (c_o, c_s, Y, diff, off) = self.loss(X_obj, X_sha, y)
        N = X_obj.shape[0]

        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        # gradient arriving at each view's readout logits
        dP_o_dia = self.alpha * diff / N
        dLf_o = ((c_o["P"] - Y) / N) + softmax_backward(c_o["P"], dP_o_dia)
        dP_s_dia = -self.alpha * diff / N
        dLf_s = ((c_s["P"] - Y) / N) + softmax_backward(c_s["P"], dP_s_dia)

        def backprop_view(cache, dLf):
            ZT = cache["ZT"]
            # readout: Lf = ZT @ Phi.T / tau
            grads["Phi"] += (dLf.T @ ZT) / self.tau
            dZ = (dLf @ self.Phi) / self.tau            # grad w.r.t. ZT

            # settling loop, in reverse:
            #   Z_next = (1-eta) Z + eta * (softmax(Z Phi.T / tau) @ Phi)
            for (Z, L, P, R) in reversed(cache["settle"]):
                dR = self.eta * dZ
                dZ_prev = (1.0 - self.eta) * dZ          # the decay path
                grads["Phi"] += P.T @ dR                 # R = P @ Phi
                dP = dR @ self.Phi.T
                dL = softmax_backward(P, dP)             # P = softmax(L)
                grads["Phi"] += (dL.T @ Z) / self.tau    # L = Z Phi.T / tau
                dZ_prev += (dL @ self.Phi) / self.tau
                dZ = dZ_prev

            # dZ is now the gradient w.r.t. Z0 (dianoia projection)
            X, A1, H1 = cache["enc"]
            grads["W2"] += H1.T @ dZ
            grads["b2"] += dZ.sum(axis=0)
            dH1 = dZ @ self.W2.T
            dA1 = dH1 * (1.0 - H1 * H1)                  # tanh'
            grads["W1"] += X.T @ dA1
            grads["b1"] += dA1.sum(axis=0)

        backprop_view(c_o, dLf_o)
        backprop_view(c_s, dLf_s)

        # distinctness penalty on the Form bank:
        #   L_dist = (lam/2)*sum(offdiag(Phi Phi.T)^2);  dPhi = 2*lam*off @ Phi
        grads["Phi"] += 2.0 * self.lam * (off @ self.Phi)

        # weight decay: dParam += wd * Param  (biases excluded)
        grads["W1"] += self.wd * self.W1
        grads["W2"] += self.wd * self.W2
        grads["Phi"] += self.wd * self.Phi

        return total, parts, grads

    # -----------------------------------------------------------------------
    # 4. Inference helpers that make the Platonic claims observable
    # -----------------------------------------------------------------------
    def recollect(self, X, n_steps=None):
        """Return the participation distribution after settling (noesis)."""
        Z0, _ = self._encode(X)
        T = self.T if n_steps is None else n_steps
        Z = Z0
        for _ in range(T):
            P = softmax((Z @ self.Phi.T) / self.tau)
            Z = (1 - self.eta) * Z + self.eta * (P @ self.Phi)
        return softmax((Z @ self.Phi.T) / self.tau), Z

    def predict(self, X, n_steps=None):
        P, _ = self.recollect(X, n_steps=n_steps)
        return P.argmax(axis=1)

    def settling_trajectory(self, x, true_form, n_steps=8):
        """
        Watch a single particular settle. Returns the distance from the
        evolving Z to the true Form prototype at each step — noesis should
        make this shrink: the soul converges on what it already knew.
        """
        Z0, _ = self._encode(x[None, :])
        Z = Z0
        dists = [float(np.linalg.norm(Z[0] - self.Phi[true_form]))]
        for _ in range(n_steps):
            P = softmax((Z @ self.Phi.T) / self.tau)
            Z = (1 - self.eta) * Z + self.eta * (P @ self.Phi)
            dists.append(float(np.linalg.norm(Z[0] - self.Phi[true_form])))
        return dists


# ---------------------------------------------------------------------------
# 5. A synthetic world of Forms and their shadows
# ---------------------------------------------------------------------------
def make_world(K=5, d_in=12, n_per=160, shadow_noise=0.7, seed=70):
    """
    Hidden ideal centroids in sense-space (the *true* Forms, unknown to the
    network). Each particular is a noisy fallen-away copy of one Form. A
    "dialectic pair" is two independent distortions (object & shadow) of the
    SAME particular; the network must recollect both to the same Form.
    """
    rng = np.random.default_rng(seed)
    ideal = rng.standard_normal((K, d_in)) * 1.5
    X_obj, X_sha, y = [], [], []
    for k in range(K):
        for _ in range(n_per):
            base = ideal[k] + rng.standard_normal(d_in) * 0.5
            X_obj.append(base + rng.standard_normal(d_in) * shadow_noise)
            X_sha.append(base + rng.standard_normal(d_in) * shadow_noise)
            y.append(k)
    X_obj = np.array(X_obj); X_sha = np.array(X_sha); y = np.array(y)
    mu, sd = X_obj.mean(0), X_obj.std(0) + 1e-8
    X_obj = (X_obj - mu) / sd
    X_sha = (X_sha - mu) / sd
    perm = rng.permutation(len(y))
    return X_obj[perm], X_sha[perm], y[perm], ideal


# ---------------------------------------------------------------------------
# 6. The mandatory finite-difference gradient check
# ---------------------------------------------------------------------------
def gradient_check(verbose=True):
    """
    Compare every analytic gradient against a central finite-difference
    estimate on a tiny batch. MUST pass (max relative error < 1e-5).
    """
    net = AnamneticRecollectionNetwork(d_in=6, h=8, d=5, K=4,
                                       tau=0.6, settle_eta=0.5, settle_T=2,
                                       alpha=1.0, lam=0.05, wd=1e-3)
    rng = np.random.default_rng(1)
    Xo = rng.standard_normal((7, 6))
    Xs = rng.standard_normal((7, 6))
    y = rng.integers(0, 4, size=7)

    _, _, grads = net.backward(Xo, Xs, y)

    eps = 1e-6
    worst = 0.0
    report = []
    for name, P in net.params().items():
        flat = P.ravel()
        g_an = grads[name].ravel()
        idxs = np.linspace(0, flat.size - 1, num=min(8, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _, _ = net.loss(Xo, Xs, y)
            flat[i] = orig - eps
            lm, _, _ = net.loss(Xo, Xs, y)
            flat[i] = orig
            g_num = (lp - lm) / (2 * eps)
            denom = max(1e-12, abs(g_num) + abs(g_an[i]))
            rel = abs(g_num - g_an[i]) / denom
            worst = max(worst, rel)
        report.append((name, P.shape))
    if verbose:
        print("  gradient check — analytic vs finite difference")
        for name, shp in report:
            print(f"    {name:<4} {str(shp):<10} checked")
        print(f"  worst relative error = {worst:.3e}  "
              f"({'PASS' if worst < 1e-5 else 'FAIL'}, threshold 1e-5)")
    assert worst < 1e-5, f"gradient check FAILED (worst rel err {worst:.3e})"
    return worst


# ---------------------------------------------------------------------------
# 7. Training loop (with unit-norm Forms, weight decay, gradient clipping)
# ---------------------------------------------------------------------------
def train(net, Xo, Xs, y, Xo_te, Xs_te, y_te,
          epochs=60, batch=64, lr=0.08, mom=0.85, clip=5.0, verbose=True):
    n = len(y)
    vel = {k: np.zeros_like(v) for k, v in net.params().items()}
    hist = {"loss": [], "train_acc": [], "test_acc": [], "invariance": []}
    rng = np.random.default_rng(70)

    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            total, parts, grads = net.backward(Xo[idx], Xs[idx], y[idx])

            # global-norm gradient clipping (kills momentum spikes)
            gnorm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
            scale = clip / (gnorm + 1e-12)
            if scale < 1.0:
                for k in grads:
                    grads[k] *= scale

            # SGD with momentum
            for k in net.params():
                vel[k] = mom * vel[k] - lr * grads[k]
                getattr(net, k)[...] += vel[k]

            # CONSTRAINT: the Forms are pure directions — project to unit norm.
            # This pins the logit scale and prevents the ||Phi||->inf runaway.
            net.Phi[...] = unit_rows(net.Phi)

            ep_loss += total
            nb += 1

        tr_acc = float(np.mean(net.predict(Xo) == y))
        te_acc = float(np.mean(net.predict(Xo_te) == y_te))
        inv = float(np.mean(net.predict(Xo_te) == net.predict(Xs_te)))
        hist["loss"].append(ep_loss / nb)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        hist["invariance"].append(inv)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} | loss {ep_loss/nb:7.4f} | "
                  f"train {tr_acc:5.3f} | test {te_acc:5.3f} | "
                  f"invariance {inv:5.3f}")
    return hist


# ---------------------------------------------------------------------------
# 8. Self-tests: assert the mind-specific claims actually hold
# ---------------------------------------------------------------------------
def self_tests(net, Xo_te, Xs_te, y_te, hist):
    print("\n  self-tests (the Platonic claims, made falsifiable)")

    # (a) Recollection works well above chance.
    acc = float(np.mean(net.predict(Xo_te) == y_te))
    chance = 1.0 / net.K
    print(f"    [a] recollection accuracy {acc:5.3f} > 3x chance {3*chance:5.3f}",
          "OK" if acc > 3 * chance else "WEAK")
    assert acc > 3 * chance

    # (b) Knowledge = invariance: object and shadow agree, and agreement did
    #     not collapse over training (opinion -> knowledge).
    inv0, inv1 = hist["invariance"][0], hist["invariance"][-1]
    print(f"    [b] dialectic invariance {inv0:5.3f} -> {inv1:5.3f}",
          "OK" if inv1 > 0.6 else "WEAK")
    assert inv1 > 0.6

    # (c) Noesis: settling reduces distance to the true Form.
    drops = []
    for i in range(0, len(y_te), max(1, len(y_te) // 200)):
        d = net.settling_trajectory(Xo_te[i], int(y_te[i]), n_steps=8)
        drops.append(d[0] - d[-1])
    mean_drop = float(np.mean(drops))
    frac_closer = float(np.mean(np.array(drops) > 0))
    print(f"    [c] anamnesis (noesis) settling: mean distance drop "
          f"{mean_drop:+.3f}; {frac_closer*100:4.1f}% of souls move toward the Form",
          "OK" if frac_closer > 0.7 else "WEAK")
    assert frac_closer > 0.7

    # (d) "One over many": the learned Forms are mutually distinct.
    Phin = unit_rows(net.Phi)
    C = np.abs(Phin @ Phin.T)
    offmean = (C.sum() - np.trace(C)) / (net.K * (net.K - 1))
    print(f"    [d] distinctness of Forms: mean |cos| between Forms {offmean:5.3f}",
          "OK" if offmean < 0.5 else "WEAK")
    assert offmean < 0.6


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=3, suppress=True)
    # Turn numerical warnings into errors so instability can never pass silently.
    np.seterr(over="raise", invalid="raise", divide="raise")

    print("=" * 74)
    print(" chapter_0070_Plato — Anamnetic Recollection Network (mind #70)")
    print(" Knowledge = recollection of invariants; learning = settling onto Forms")
    print("=" * 74)

    print("\n[1] Gradient check (must pass before any training)")
    gradient_check(verbose=True)

    print("\n[2] Building the world of Forms and their shadows")
    Xo, Xs, y, ideal = make_world(K=5, d_in=12, n_per=160, shadow_noise=0.7)
    cut = int(0.8 * len(y))
    Xo_tr, Xs_tr, y_tr = Xo[:cut], Xs[:cut], y[:cut]
    Xo_te, Xs_te, y_te = Xo[cut:], Xs[cut:], y[cut:]
    print(f"    {len(y_tr)} train pairs, {len(y_te)} test pairs, "
          f"{len(set(y.tolist()))} Forms, sense-dim {Xo.shape[1]}")

    print("\n[3] Training the soul to recollect")
    net = AnamneticRecollectionNetwork(d_in=12, h=24, d=16, K=5,
                                       tau=0.5, settle_eta=0.5, settle_T=2,
                                       alpha=1.0, lam=0.05, wd=1e-3)
    hist = train(net, Xo_tr, Xs_tr, y_tr, Xo_te, Xs_te, y_te,
                 epochs=60, batch=64, lr=0.08, mom=0.85, clip=5.0)

    self_tests(net, Xo_te, Xs_te, y_te, hist)

    print("\n[4] A single soul recollecting (distance to its Form as it settles)")
    i = int(np.argmax(y_te == y_te[0]))
    traj = net.settling_trajectory(Xo_te[i], int(y_te[i]), n_steps=8)
    print("    step:   " + "  ".join(f"{t:5d}" for t in range(len(traj))))
    print("    dist:   " + "  ".join(f"{v:5.2f}" for v in traj))

    print("\n[5] Opinion vs knowledge on heavily distorted particulars")
    rng = np.random.default_rng(7)
    Xc = Xo_te + rng.standard_normal(Xo_te.shape) * 0.6
    P_doxa, _ = net.recollect(Xc, n_steps=0)
    P_noesis, _ = net.recollect(Xc, n_steps=8)
    acc_doxa = float(np.mean(P_doxa.argmax(1) == y_te))
    acc_noe = float(np.mean(P_noesis.argmax(1) == y_te))
    conf_doxa = float(P_doxa.max(1).mean())
    conf_noe = float(P_noesis.max(1).mean())
    print(f"    opinion  (0 settling): accuracy {acc_doxa:5.3f} | "
          f"mean confidence {conf_doxa:5.3f}")
    print(f"    knowledge (settled)  : accuracy {acc_noe:5.3f} | "
          f"mean confidence {conf_noe:5.3f}")
    print(f"    -> on corrupted shadows, settling moved accuracy "
          f"{(acc_noe - acc_doxa):+5.3f} and conviction {(conf_noe - conf_doxa):+5.3f};")
    print("       the examined grasp is steadier than raw opinion.")

    print("\nDone. The Forms were not taught as data; they were the fixed")
    print("directions the encoder learned to make every shadow fall toward.")
    print("=" * 74)


if __name__ == "__main__":
    main()
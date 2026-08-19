#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Roman Emperor 98-117 CE | Optimus Princeps | Governance, Expansion, Ethics
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 126: Trajan (53 CE - 117 CE)
================================================================================   

THE RESCRIPTOR  -  a Casuistic Kernel Reasoner ("Government by Rescript")

WHY THIS ARCHITECTURE, AND WHY IT IS TRAJAN'S AND NO ONE ELSE'S
---------------------------------------------------------------
Most "sovereign" minds in this corpus are modelled as rule-compilers: they take
the world's messy particulars and press them into one universal code. Trajan is
the documented opposite. The single surviving artefact in his own governing
voice - the rescripts preserved in Book 10 of Pliny the Younger's letters -
states his cognitive doctrine outright. Asked for a general rule on how to try
Christians, the emperor refuses to give one:

    "Neque enim in universum aliquid, quod quasi certam formam habeat,
     constitui potest."
    -> "It is not possible to lay down anything universal that would have,
        as it were, a FIXED FORM."   (Pliny, Ep. 10.97)

That refusal is the whole mind. Trajan governs *case by case*, from an
accumulated memory of prior decisions, and he will not freeze a universal
"certa forma" that runs ahead of the particulars. The same rescript adds two
more operational rules that this file turns into mechanism:

  * "Conquirendi non sunt"           -> do NOT go seeking cases out.
  * "Sine auctore ... propositi
     libelli in nullo crimine locum   -> ANONYMOUS, unattributed accusations
     habere debent"                      have no standing and never enter the
                                          record ("pessimi exempli").

And the deeds add a fourth: the Dacian Wars. Trajan did not consolidate behind
a wall (that was his successor Hadrian). When the frontier reached genuinely new
ground his empire could not adjudicate from existing precedent, he ANNEXED it -
but only after a campaign put verified facts on the ground.

So the Trajanic cognitive signature is FOUR coupled ideas:

  (1) CASE-BASED, NON-PARAMETRIC memory. Keep the precedents (exemplars)
      themselves. Judge a new case by kernel-weighted analogy to stored
      precedents - never by a single compressed universal weight-matrix.

  (2) THE 'CERTA FORMA' PENALTY. An explicit regulariser that pushes AGAINST
      one precedent becoming the fixed form for everything. It maximises the
      entropy of *aggregate* precedent-usage: many particular rulings, no
      universal rule. This is the mathematical form of "neque in universum".

  (3) THE 'CONQUIRENDI' ADMISSION GATE. Inputs below a verification threshold,
      or lacking an attributed author, are REFUSED: they neither get judged nor
      update the corpus. Trajan's rejection of anonymous denunciation, as code.

  (4) FRONTIER ANNEXATION (Dacia). When a case falls genuinely beyond the reach
      of the whole corpus (max analogical support below a threshold), the system
      does not force it into an ill-fitting category. It flags the frontier and,
      given verified reports from the ground, ANNEXES a new precedent. Expansion
      is evidence-gated, exactly as the Danube crossing was campaign-gated.

This is deliberately NOT a Transformer, NOT mixture-of-experts, NOT
attention-over-a-context-window. It is a differentiable, growable, non-parametric
CASE REASONER. Divergence note: Cyrus (federated translation across simultaneous
domains) and Wang Mang (snap-everything-to-one-canonical-prototype) are the near
neighbours in this corpus; Trajan is the anti-Wang-Mang - he refuses the single
prototype and keeps the caseload plural.

WHAT RUNS BELOW
---------------
  * A from-scratch NumPy CasuisticKernelReasoner with analytic gradients.
  * A MANDATORY finite-difference gradient check on every learnable tensor
    (precedent keys P, verdict logits V, kernel temperature log_tau), on the
    FULL loss including the certa-forma penalty.
  * A real training loop on a synthetic "provincial caseload".
  * Demonstrations of all four signature mechanisms, with self-test asserts:
      - accuracy on known provinces,
      - the conquirendi gate refusing anonymous / low-evidence input,
      - the certa-forma penalty raising usage entropy vs. an ablation,
      - the Dacian frontier: novelty detected -> annex -> accuracy recovers.

Run:  python3 chapter_0126_trajan_53.py
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------
# Small numerical helpers (kept explicit so the mechanism is legible)
# ------------------------------------------------------------------------------

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of a 1-D distribution."""
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + eps)))


# ==============================================================================
# THE RESCRIPTOR
# ==============================================================================

class CasuisticKernelReasoner:
    """
    Government by rescript, as a differentiable model.

    A query x in R^d is judged by kernel-weighted analogy to a persistent,
    GROWABLE corpus of precedents:

        d_{i,m}  = || x_i - P_m ||^2                 (distance to precedent m)
        w_{i,:}  = softmax( -tau * d_{i,:} )          (retrieval over the corpus)
        logits_i = w_{i,:} @ V                        (blend of the cases' verdicts)
        p_i      = softmax(logits_i)                  (the ruling)

    Learnable tensors:
        P        (M x d)   precedent keys  - the accumulated cases themselves
        V        (M x K)   verdict logits  - the rescript attached to each case
        log_tau  (scalar)  log kernel inverse-temperature

    Loss = cross-entropy  +  lambda_form * (certa-forma penalty)

    The certa-forma penalty is -H(u_bar), where u_bar_m is the mean usage of
    precedent m across the batch. Minimising it maximises usage entropy: it
    forbids any single precedent from silently becoming the universal rule.
    """

    def __init__(self, d: int, k: int, m: int, lambda_form: float = 0.03,
                 seed: int = 128):
        rng = np.random.default_rng(seed)
        self.d, self.k, self.m = d, k, m
        self.lambda_form = float(lambda_form)
        # Precedent keys spread over the input space; small verdict logits.
        self.P = rng.normal(0.0, 1.0, size=(m, d))
        self.V = rng.normal(0.0, 0.3, size=(m, k))
        self.log_tau = np.array(0.0)  # tau = 1.0 initially
        self._eps = 1e-9
        # Frontier bandwidth: the REACH of the empire. Deliberately separate
        # from the discriminative tau above. tau decides how sharply precedents
        # are weighted when RULING; sigma2 decides how far a case may lie from
        # all precedents before it counts as new TERRITORY. Calibrated from the
        # settled corpus by calibrate_frontier(); 1.0 until then.
        self.sigma2 = 1.0

    # -- parameter (de)serialisation for the gradient checker --------------
    def get_params(self):
        return {"P": self.P, "V": self.V, "log_tau": self.log_tau}

    def set_params(self, params):
        self.P = params["P"]
        self.V = params["V"]
        self.log_tau = params["log_tau"]

    # -- forward -----------------------------------------------------------
    def forward(self, X: np.ndarray):
        """
        Returns (logits, cache). X is (N x d).
        cache holds intermediates needed for the analytic backward pass.
        """
        tau = np.exp(self.log_tau)
        # squared Euclidean distance (N x M) via the (a-b)^2 expansion
        x2 = np.sum(X * X, axis=1, keepdims=True)          # (N,1)
        p2 = np.sum(self.P * self.P, axis=1, keepdims=True).T  # (1,M)
        cross = X @ self.P.T                               # (N,M)
        dist = x2 + p2 - 2.0 * cross                       # (N,M)
        dist = np.maximum(dist, 0.0)                       # guard tiny negatives
        e = -tau * dist                                    # (N,M)
        w = softmax(e, axis=1)                             # (N,M) retrieval
        logits = w @ self.V                                # (N,K)
        cache = {"X": X, "tau": tau, "dist": dist, "w": w, "logits": logits}
        return logits, cache

    # -- loss + analytic gradients ----------------------------------------
    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        logits, cache = self.forward(X)
        w = cache["w"]                                     # (N,M)
        tau = cache["tau"]

        # cross-entropy
        p = softmax(logits, axis=1)                        # (N,K)
        ce = -np.mean(np.log(p[np.arange(n), y] + self._eps))

        # certa-forma penalty:  L_form = sum_m u_bar_m * log(u_bar_m)
        u_bar = np.mean(w, axis=0)                          # (M,)
        L_form = float(np.sum(u_bar * np.log(u_bar + self._eps)))
        total = ce + self.lambda_form * L_form

        # ---- backward ----
        # dCE/dlogits
        g_logits = (p - one_hot(y, self.k)) / n            # (N,K)

        # V grad from CE:  logits = w @ V  ->  dV = w^T @ g_logits
        dV = w.T @ g_logits                                # (M,K)

        # dCE/dw = g_logits @ V^T
        dw = g_logits @ self.V.T                           # (N,M)

        # add certa-forma grad wrt w:
        #   dL_form/du_bar_m = log(u_bar_m) + 1 (approx; exact below)
        #   du_bar_m/dw_{i,m} = 1/n
        d_form_du = np.log(u_bar + self._eps) + u_bar / (u_bar + self._eps)
        dw = dw + self.lambda_form * (d_form_du[None, :] / n)  # (N,M)

        # backprop through softmax rows: e -> w
        #   de = w * (dw - sum_m dw*w)
        dw_dot = np.sum(dw * w, axis=1, keepdims=True)     # (N,1)
        de = w * (dw - dw_dot)                             # (N,M)

        # e = -tau * dist
        # d/d log_tau : de/dtau = -dist ; dtau/dlogtau = tau
        dlog_tau = np.sum(de * (-cache["dist"])) * tau
        dlog_tau = np.array(dlog_tau)

        # e wrt P:  de_{i,m}/dP_{m,j} = -tau * d(dist)/dP = 2 tau (x_ij - P_mj)
        # dP_{m,j} = sum_i de_{i,m} * 2 tau (x_ij - P_mj)
        X = cache["X"]
        # (N,M,1)*( (N,1,d)-(1,M,d) ) summed over i -> (M,d)
        diff = X[:, None, :] - self.P[None, :, :]          # (N,M,d)
        dP = np.einsum("nm,nmd->md", de, 2.0 * tau * diff)  # (M,d)

        grads = {"P": dP, "V": dV, "log_tau": dlog_tau}
        aux = {"ce": ce, "L_form": L_form, "u_bar": u_bar, "w": w}
        return total, grads, aux

    # -- prediction + the operational (non-differentiable) mechanisms -------
    def predict(self, X: np.ndarray):
        logits, cache = self.forward(X)
        return np.argmax(logits, axis=1), cache

    def _min_dist(self, X: np.ndarray):
        x2 = np.sum(X * X, axis=1, keepdims=True)
        p2 = np.sum(self.P * self.P, axis=1, keepdims=True).T
        dist = np.maximum(x2 + p2 - 2.0 * (X @ self.P.T), 0.0)
        return np.min(dist, axis=1)                        # (N,)

    def calibrate_frontier(self, X: np.ndarray, slack: float = 3.0):
        """
        Fix the empire's REACH from the settled caseload: set the frontier
        bandwidth so that a typical in-empire case sits comfortably inside the
        corpus. sigma2 = slack * median(nearest-precedent distance on X).
        """
        md = self._min_dist(X)
        self.sigma2 = float(slack * (np.median(md) + self._eps))
        return self.sigma2

    def support(self, X: np.ndarray):
        """
        ABSOLUTE analogical support per query: a Parzen-style similarity to the
        NEAREST precedent, exp(-min_m ||x-P_m||^2 / (2*sigma2)), in (0, 1].
        It asks "does any stored case genuinely resemble this one?" measured
        against the empire's reach (sigma2), NOT the discriminative tau - which
        is tuned to separate verdicts and saturates to ~1 even for distant
        cases. Low support == the frontier: nothing in the corpus is an analogue.
        """
        return np.exp(-self._min_dist(X) / (2.0 * self.sigma2))  # (N,) in (0,1]

    def admit(self, evidence: np.ndarray, has_author: np.ndarray,
              evidence_threshold: float = 0.5):
        """
        The 'conquirendi non sunt' gate. A case is admitted for adjudication
        only if it carries enough verified evidence AND names an author
        (is not an anonymous 'libellus sine auctore').
        Returns a boolean mask (True == admitted).
        """
        evidence = np.asarray(evidence, dtype=np.float64)
        has_author = np.asarray(has_author, dtype=bool)
        return (evidence >= evidence_threshold) & has_author

    def frontier(self, X: np.ndarray, annex_threshold: float = 0.30):
        """Cases whose maximum analogical support falls below the threshold:
        the empire's precedent-corpus cannot yet adjudicate them."""
        return self.support(X) < annex_threshold

    def annex(self, X_reports: np.ndarray, y_reports: np.ndarray):
        """
        Frontier annexation (the Dacian mechanism). Given a small batch of
        VERIFIED reports from newly-reached ground, add a precedent at their
        centroid whose verdict is the reports' majority ruling. The corpus
        grows; no existing precedent is overwritten.
        """
        centroid = np.mean(X_reports, axis=0, keepdims=True)   # (1,d)
        counts = np.bincount(y_reports, minlength=self.k)
        verdict = int(np.argmax(counts))
        v_row = np.full((1, self.k), -1.0)
        v_row[0, verdict] = 2.0                                 # confident rescript
        self.P = np.vstack([self.P, centroid])
        self.V = np.vstack([self.V, v_row])
        self.m += 1
        return verdict


# ==============================================================================
# GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(model: CasuisticKernelReasoner, X, y, eps=1e-6):
    """Central finite differences vs. analytic grads on the FULL loss."""
    _, grads, _ = model.loss_and_grads(X, y)
    params = model.get_params()
    worst = 0.0
    report = {}
    for name in ("P", "V", "log_tau"):
        theta = params[name]
        ga = grads[name]
        gn = np.zeros_like(np.atleast_1d(theta), dtype=np.float64)
        flat = np.atleast_1d(theta).ravel()
        gflat = gn.ravel()
        for idx in range(flat.size):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _ = model.loss_and_grads(X, y)
            flat[idx] = orig - eps
            lm, _, _ = model.loss_and_grads(X, y)
            flat[idx] = orig
            gflat[idx] = (lp - lm) / (2 * eps)
        ga_flat = np.atleast_1d(ga).ravel()
        num = np.linalg.norm(ga_flat - gflat)
        den = np.linalg.norm(ga_flat) + np.linalg.norm(gflat) + 1e-12
        rel = num / den
        report[name] = rel
        worst = max(worst, rel)
    return worst, report


# ==============================================================================
# SYNTHETIC "PROVINCIAL CASELOAD"
# ==============================================================================

def make_provinces(n_per=60, d=6, k=4, seed=7):
    """
    Five 'provinces' (Gaussian clusters) scattered in R^d, each with a fixed
    verdict label in {0..k-1}. This is the settled empire the Rescriptor learns.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(5, d))
    labels = np.array([0, 1, 2, 3, 1])           # provinces map onto k verdicts
    X, y = [], []
    for c in range(5):
        pts = centers[c] + rng.normal(0.0, 0.55, size=(n_per, d))
        X.append(pts)
        y.append(np.full(n_per, labels[c]))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], centers, labels


def make_dacia(centers, d=6, k=4, n=60, seed=99):
    """
    Dacia: a NEW province deliberately placed far outside the settled empire,
    unseen at training time. Its true verdict is a known class, but no existing
    precedent sits near it - it is the frontier.
    """
    rng = np.random.default_rng(seed)
    far = np.mean(centers, axis=0) + np.array([14.0] + [7.0] * (d - 1))
    X = far + rng.normal(0.0, 0.55, size=(n, d))
    y = np.full(n, 2)                              # Dacia's true verdict = class 2
    return X, y


def accuracy(model, X, y):
    pred, _ = model.predict(X)
    return float(np.mean(pred == y))


# ==============================================================================
# TRAIN + DEMONSTRATE
# ==============================================================================

def train(model, X, y, Xval, yval, epochs=400, lr=0.5, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            loss, grads, _ = model.loss_and_grads(X[b], y[b])
            model.P -= lr * grads["P"]
            model.V -= lr * grads["V"]
            model.log_tau -= lr * grads["log_tau"]
        if ep % 40 == 0 or ep == epochs - 1:
            full, _, aux = model.loss_and_grads(X, y)
            hist.append((ep, full, accuracy(model, Xval, yval)))
    return hist


def usage_entropy(model, X):
    _, cache = model.forward(X)
    u = np.mean(cache["w"], axis=0)
    return entropy(u)


def line(title=""):
    print("-" * 74)
    if title:
        print(title)
        print("-" * 74)


def main():
    np.random.seed(128)
    print("=" * 74)
    print(" THE RESCRIPTOR  -  Trajan's Casuistic Kernel Reasoner")
    print(" 'Neque in universum aliquid quod quasi certam formam habeat'")
    print("=" * 74)

    d, k = 6, 4
    X, y, centers, labels = make_provinces(n_per=60, d=d, k=k)
    ntr = int(0.8 * X.shape[0])
    Xtr, ytr, Xval, yval = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    # ---- 1. gradient check (mandatory) ----
    line("1. GRADIENT CHECK (analytic vs finite-difference, full loss)")
    checker = CasuisticKernelReasoner(d=d, k=k, m=12, lambda_form=0.03, seed=1)
    worst, rep = gradient_check(checker, Xtr[:24], ytr[:24])
    for name, rel in rep.items():
        print(f"   param {name:8s}  rel-error = {rel:.3e}")
    print(f"   worst rel-error = {worst:.3e}   ->  {'PASS' if worst < 1e-5 else 'FAIL'}")
    assert worst < 1e-5, "gradient check failed"

    # ---- 2. train the Rescriptor on the settled empire ----
    line("2. TRAINING ON THE PROVINCIAL CASELOAD")
    model = CasuisticKernelReasoner(d=d, k=k, m=12, lambda_form=0.03, seed=3)
    hist = train(model, Xtr, ytr, Xval, yval, epochs=400, lr=0.5)
    for ep, loss, acc in hist:
        print(f"   epoch {ep:4d}   loss = {loss:.4f}   val-acc = {acc:.3f}")
    tr_acc, val_acc = accuracy(model, Xtr, ytr), accuracy(model, Xval, yval)
    print(f"   final  train-acc = {tr_acc:.3f}   val-acc = {val_acc:.3f}")
    assert val_acc > 0.9, "model failed to learn the settled caseload"
    assert hist[0][1] > hist[-1][1], "loss did not decrease"

    # ---- 3. the conquirendi-non-sunt admission gate ----
    line("3. THE CONQUIRENDI GATE  (anonymous / weak cases refused)")
    rng = np.random.default_rng(5)
    cases = Xval[:8]
    evidence = np.array([0.9, 0.2, 0.8, 0.95, 0.1, 0.7, 0.4, 0.85])
    has_author = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)  # #2 anonymous
    admitted = model.admit(evidence, has_author, evidence_threshold=0.5)
    for i in range(len(cases)):
        tag = "ADMITTED " if admitted[i] else "refused  "
        why = "" if admitted[i] else (
            "(sine auctore)" if not has_author[i] else "(evidence < threshold)")
        print(f"   case {i}: evidence={evidence[i]:.2f} author={bool(has_author[i])}"
              f"  -> {tag}{why}")
    # anonymous case #2 and low-evidence #1,#4,#6 must be refused
    assert not admitted[2] and not admitted[1] and not admitted[4]
    assert admitted[0] and admitted[3] and admitted[7]
    print("   gate correctly refuses anonymous and under-evidenced cases.")

    # ---- 4. the certa-forma penalty raises usage entropy ----
    line("4. THE CERTA-FORMA PENALTY  (no single precedent becomes the rule)")
    ablate = CasuisticKernelReasoner(d=d, k=k, m=12, lambda_form=0.0, seed=3)
    train(ablate, Xtr, ytr, Xval, yval, epochs=400, lr=0.5)
    h_with = usage_entropy(model, Xtr)
    h_without = usage_entropy(ablate, Xtr)
    hmax = np.log(model.m)
    print(f"   usage entropy WITH  penalty = {h_with:.3f} nats  (max {hmax:.3f})")
    print(f"   usage entropy WITHOUT penalty = {h_without:.3f} nats")
    print(f"   ablation val-acc = {accuracy(ablate, Xval, yval):.3f} "
          f"(accuracy preserved; the penalty only spreads the caseload)")
    assert h_with > h_without, "certa-forma penalty did not raise usage entropy"
    print("   the penalty keeps the caseload plural - the mind refuses one form.")

    # ---- 5. the Dacian frontier: detect -> annex -> recover ----
    line("5. THE DACIAN FRONTIER  (novelty -> campaign -> annexation)")
    Xd, yd = make_dacia(centers, d=d, k=k, n=60)
    sigma2 = model.calibrate_frontier(Xtr)   # fix the empire's reach from the corpus
    print(f"   empire reach calibrated: sigma^2 = {sigma2:.3f}")
    supp_known = model.support(Xval).mean()
    supp_dacia = model.support(Xd).mean()
    print(f"   mean analogical support, known provinces = {supp_known:.3f}")
    print(f"   mean analogical support, Dacia (unseen)  = {supp_dacia:.3f}")
    frontier_mask = model.frontier(Xd, annex_threshold=0.30)
    print(f"   Dacian cases flagged as frontier: "
          f"{int(frontier_mask.sum())}/{len(yd)}")
    acc_before = accuracy(model, Xd, yd)
    print(f"   Dacia accuracy BEFORE annexation = {acc_before:.3f}")
    # a governor's verified reports: a few labelled cases from the ground
    rep_idx = rng.choice(len(Xd), size=8, replace=False)
    verdict = model.annex(Xd[rep_idx], yd[rep_idx])
    acc_after = accuracy(model, Xd, yd)
    print(f"   ...campaign returns verified reports; annex precedent "
          f"(verdict={verdict}).")
    print(f"   Dacia accuracy AFTER  annexation = {acc_after:.3f}")
    # the settled empire must not be disturbed by the annexation
    val_after = accuracy(model, Xval, yval)
    print(f"   settled-empire val-acc after annexation = {val_after:.3f}")
    assert supp_dacia < supp_known, "frontier not detected as low-support"
    assert acc_after > acc_before + 0.3, "annexation did not adjudicate Dacia"
    assert val_after > 0.9, "annexation disturbed the settled empire"
    print("   frontier reached, campaigned, and annexed - without a universal rule.")

    line("ALL SELF-TESTS PASSED")
    print(" The Rescriptor judged every case by its precedents, refused a fixed")
    print(" form, turned away anonymous accusation, and annexed the frontier only")
    print(" on verified report. Optimus Princeps, compiled and run.")
    print("=" * 74)


if __name__ == "__main__":
    main()

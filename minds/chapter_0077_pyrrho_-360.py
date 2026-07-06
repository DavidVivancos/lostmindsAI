#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0077_pyrrho_-360.py — Pyrrho of Elis (c. 360 - c. 270 BCE)
 The Equipollence Suspension Network (ESN)
  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 4 Minds 61 - 80 Available on Amazon https://www.amazon.com/dp/B0H71JC95Q
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0077 · Pyrrho of Elis
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A CLASSIFIER WITH A "DON'T KNOW" BIN
--------------------------------------------------------------------------------
Pyrrho left no writings. What survives is the "Aristocles passage" (Aristocles
of Messene, preserved by Eusebius, reporting Pyrrho via his student Timon of
Phlius). It states a three-step doctrine:

  1. Things (pragmata) are by nature *adiaphora* (undifferentiated),
     *astathmeta* (unstable / unmeasurable), and *anepikrita* (undecidable).
  2. Therefore neither our senses nor our opinions tell us truth or falsehood.
  3. Therefore the right disposition is to hold NO opinion (adoxastoi), to say
     of each thing that it "no more is than is not" (the *ou mallon* formula),
     which yields first *aphasia* (non-assertion) and then *ataraxia* (calm).

Most modern systems treat "I don't know" as a leftover: a low-confidence
fallback bin reached when the top class probability dips below a threshold.
That is exactly NOT Pyrrho. For Pyrrho, suspension (*epoche*) is not the
failure of judgment; it is the *competent outcome* of a judgment that two
opposed appearances are of equal force (*isostheneia*). Suspension must be a
first-class target the network is rewarded for choosing correctly — not a
consolation prize.

So the ESN's decision geometry literally encodes *ou mallon*. For every input
("appearance") it builds TWO opposed argument strengths:

      s_pro  = strength of the case "P"
      s_con  = strength of the case "not-P"

and forms a 3-way decision over {assert P, assert not-P, SUSPEND} whose logits
are:

      logit_P       = alpha * s_pro
      logit_notP    = alpha * s_con
      logit_suspend = beta  - gamma * (s_pro - s_con)^2      <-- the epoche gate

The squared term  (s_pro - s_con)^2  is the mathematical form of isostheneia:
when the two cases are balanced (s_pro ~ s_con) the gate is *open* and SUSPEND
wins; as the appearance tilts decisively to one side the gate closes and the
stronger side wins. The network is trained so that suspension is the correct
answer on genuinely balanced inputs. This is the cognitive signature of Pyrrho
and of no one else in the corpus: equipollence-driven, smooth, learned
suspension — not a hand-set abstention threshold.

The training loss (cross-entropy) is read, in Pyrrhonist terms, as
"disturbance" (the opposite of ataraxia): a low loss means the network is
*tranquil* precisely because it commits only when the appearances compel it and
suspends when they do not.

WHAT IS REAL HERE (engineering contract, mandatory for every file in this work)
--------------------------------------------------------------------------------
  * Pure NumPy, built from scratch. No autograd, no ML frameworks.
  * Every parameter has a hand-derived analytic gradient.
  * A finite-difference gradient check is RUN and must pass (< 1e-5 max rel err).
  * A real training loop (mini-batch SGD with momentum) that measurably lowers
    loss and raises 3-way accuracy on held-out data.
  * Self-tests that assert the *behavioural* claim: after training, balanced
    appearances are SUSPENDED and decisive ones are COMMITTED.
  * The file executes end to end; its verified stdout is pasted into the chapter.

E-AGI BAROMETER MAPPING (artificiology.com/barometer.html)
--------------------------------------------------------------------------------
  Cognitive Processing  : the for/against scorers ARE problem-solving by
                          opposed argument; the gate is the reasoning step.
  Consciousness         : metacognition / self-monitoring is made explicit -
                          the suspend channel is the model watching the balance
                          of its own evidence rather than its object.
  Autonomy              : the model sets its own stopping rule (commit vs. wait);
                          it is not forced to answer.
  Emotional Intelligence: "disturbance" (loss) vs. "ataraxia" (calibrated calm)
                          is the affective read-out reported each epoch.
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(77)  # 77 = Pyrrho's index in the corpus


# ==============================================================================
# 0. Small numerical helpers
# ==============================================================================
def softmax(z):
    """Numerically stable softmax over the last axis."""
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def one_hot(y, k):
    out = np.zeros((y.shape[0], k))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# ==============================================================================
# 1. The Equipollence Suspension Network
# ==============================================================================
class EquipollenceSuspensionNetwork:
    """
    A 3-class decision network whose third class (SUSPEND / epoche) is gated by
    the *balance* between a pro-argument and a con-argument rather than by a
    confidence threshold.

    Forward (per row x):
        z1   = W1 @ x + b1
        h    = tanh(z1)                          # the "appearance" is encoded
        s_pro = tanh(w_pro . h + b_pro)          # strength of the case  P  in [-1,1]
        s_con = tanh(w_con . h + b_con)          # strength of the case ~P  in [-1,1]
        d     = s_pro - s_con                    # the tilt, bounded in [-2,2]
        logits = [ alpha*s_pro,                  # assert P
                   alpha*s_con,                  # assert ~P
                   beta - gamma*d^2 ]            # SUSPEND  (ou mallon gate)
        p = softmax(logits)

    Classes:  0 = assert P, 1 = assert ~P, 2 = SUSPEND (epoche).
    """

    CLASS_P, CLASS_NOT_P, CLASS_SUSPEND = 0, 1, 2

    def __init__(self, d_in, d_hidden=16):
        self.d_in = d_in
        self.d_hidden = d_hidden
        # Xavier-ish init for the encoder.
        lim = np.sqrt(6.0 / (d_in + d_hidden))
        self.W1 = RNG.uniform(-lim, lim, size=(d_hidden, d_in))
        self.b1 = np.zeros(d_hidden)
        # Argument scorers (pro / con).
        s = 1.0 / np.sqrt(d_hidden)
        self.w_pro = RNG.uniform(-s, s, size=d_hidden)
        self.b_pro = 0.0
        self.w_con = RNG.uniform(-s, s, size=d_hidden)
        self.b_con = 0.0
        # Scalar decision shape parameters.
        self.alpha = 4.0   # how sharply argument strength becomes commitment
        self.beta = 0.0    # baseline pull toward suspension
        self.gamma = 4.0   # how fast the suspend gate closes as evidence tilts

    # -- parameter (de)serialisation for the gradient check -------------------
    def get_params(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "w_pro": self.w_pro, "b_pro": np.array(self.b_pro),
            "w_con": self.w_con, "b_con": np.array(self.b_con),
            "alpha": np.array(self.alpha), "beta": np.array(self.beta),
            "gamma": np.array(self.gamma),
        }

    def set_param(self, name, value):
        if name in ("b_pro", "b_con", "alpha", "beta", "gamma"):
            setattr(self, name, float(value))
        else:
            setattr(self, name, value)

    # -- forward --------------------------------------------------------------
    def forward(self, X, cache=False):
        """X: (N, d_in) -> probabilities (N, 3). Optionally stash a cache."""
        z1 = X @ self.W1.T + self.b1           # (N, H)
        h = np.tanh(z1)                        # (N, H)
        a_pro = h @ self.w_pro + self.b_pro    # (N,) pre-activation
        a_con = h @ self.w_con + self.b_con    # (N,)
        s_pro = np.tanh(a_pro)                 # (N,) bounded argument strength
        s_con = np.tanh(a_con)                 # (N,)
        d = s_pro - s_con                      # (N,) bounded tilt in [-2, 2]
        logits = np.stack([
            self.alpha * s_pro,
            self.alpha * s_con,
            self.beta - self.gamma * d ** 2,
        ], axis=1)                             # (N, 3)
        p = softmax(logits)
        if cache:
            self._cache = dict(X=X, z1=z1, h=h, a_pro=a_pro, a_con=a_con,
                               s_pro=s_pro, s_con=s_con, d=d, logits=logits, p=p)
        return p

    def loss(self, X, y):
        """Mean cross-entropy = mean 'disturbance'."""
        p = self.forward(X, cache=True)
        n = X.shape[0]
        ll = -np.log(p[np.arange(n), y] + 1e-12)
        return float(np.mean(ll))

    # -- analytic gradients (hand-derived) ------------------------------------
    def backward(self, y):
        """
        Uses the cache from the last forward(cache=True).
        Returns a dict of gradients matching get_params().

        d L / d logits = (p - onehot(y)) / N      (softmax + CE)
        Then we push that through the three logit definitions back to every
        parameter. See the module docstring for the closed forms; the key step
        is that s_pro and s_con each influence BOTH a commitment logit (via
        alpha) AND the suspend gate (via d = s_pro - s_con).
        """
        c = self._cache
        X, h = c["X"], c["h"]
        s_pro, s_con, d, p = c["s_pro"], c["s_con"], c["d"], c["p"]
        n = X.shape[0]

        g = (p - one_hot(y, 3)) / n            # (N, 3): dL/dlogit_k

        gP, gN, gE = g[:, 0], g[:, 1], g[:, 2]  # per-class logit grads

        # scalar decision params
        d_alpha = np.sum(gP * s_pro + gN * s_con)
        d_beta = np.sum(gE)
        d_gamma = np.sum(gE * (-(d ** 2)))

        # gate contribution flowing back into d:  logit_E = beta - gamma*d^2
        gd_E = gE * (-2.0 * self.gamma * d)     # (N,) = dL/dd via the gate

        # s_pro feeds logit_P (via alpha) and d (+1);  s_con feeds logit_N and d (-1)
        d_s_pro = gP * self.alpha + gd_E * (+1.0)   # (N,) dL/ds_pro
        d_s_con = gN * self.alpha + gd_E * (-1.0)   # (N,) dL/ds_con

        # chain through tanh: s = tanh(a) -> ds/da = 1 - s^2
        d_a_pro = d_s_pro * (1.0 - s_pro ** 2)      # (N,)
        d_a_con = d_s_con * (1.0 - s_con ** 2)

        d_w_pro = h.T @ d_a_pro                 # (H,)
        d_b_pro = np.sum(d_a_pro)
        d_w_con = h.T @ d_a_con
        d_b_con = np.sum(d_a_con)

        # into the hidden layer
        dh = np.outer(d_a_pro, self.w_pro) + np.outer(d_a_con, self.w_con)  # (N,H)
        dz1 = dh * (1.0 - h ** 2)               # tanh'
        d_W1 = dz1.T @ X                         # (H, d_in)
        d_b1 = np.sum(dz1, axis=0)

        return {
            "W1": d_W1, "b1": d_b1,
            "w_pro": d_w_pro, "b_pro": np.array(d_b_pro),
            "w_con": d_w_con, "b_con": np.array(d_b_con),
            "alpha": np.array(d_alpha), "beta": np.array(d_beta),
            "gamma": np.array(d_gamma),
        }


# ==============================================================================
# 2. The world of appearances (synthetic data)
# ==============================================================================
def get_rotation(d_in):
    """A fixed, deterministic rotation shared by all splits and probes, so the
    map from features to the hidden tilt is the SAME everywhere (otherwise the
    model would face a differently-rotated world at test time)."""
    rng = np.random.default_rng(1234)
    Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)))
    return Q


def make_appearances(n, d_in=8, tau=0.6, noise=0.35, seed=0, rot=None):
    """
    Each 'appearance' is a noisy vector carrying a hidden tilt t.

        t  > +tau  -> the case for P is decisive        -> label 0
        t  < -tau  -> the case for ~P is decisive       -> label 1
        |t| <= tau -> the appearances are equipollent   -> label 2 (SUSPEND)

    The tilt is smeared across two correlated channels and buried under
    distractor dimensions, so the network must *learn* to recover both the
    SIGN of the evidence (for P vs ~P) and its MAGNITUDE band (commit vs.
    suspend). The suspend band is a genuine region of the input space, not a
    threshold bolted on afterward.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(-2.0, 2.0, size=n)
    X = np.zeros((n, d_in))
    X[:, 0] = t + noise * rng.standard_normal(n)
    X[:, 1] = 0.7 * t + noise * rng.standard_normal(n)
    X[:, 2] = -0.4 * t + noise * rng.standard_normal(n)
    for j in range(3, d_in):                     # pure distractor channels
        X[:, j] = noise * rng.standard_normal(n)
    if rot is None:
        rot = get_rotation(d_in)
    X = X @ rot                                  # shared rotation: not axis-aligned
    y = np.where(t > tau, 0, np.where(t < -tau, 1, 2)).astype(int)
    return X, y, t


# ==============================================================================
# 3. Mandatory finite-difference gradient check
# ==============================================================================
def gradient_check(net, X, y, eps=1e-6):
    """Compare analytic gradients to centered finite differences."""
    net.loss(X, y)                 # populate cache
    analytic = net.backward(y)
    max_rel = 0.0
    worst = None
    for name, grad in analytic.items():
        base = net.get_params()[name].copy()
        flat = np.array(base, dtype=float).ravel()
        ga = np.array(grad, dtype=float).ravel()
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            net.set_param(name, flat.reshape(np.array(base).shape) if flat.size > 1 else flat[0])
            lp = net.loss(X, y)
            flat[i] = orig - eps
            net.set_param(name, flat.reshape(np.array(base).shape) if flat.size > 1 else flat[0])
            lm = net.loss(X, y)
            flat[i] = orig
            net.set_param(name, base)
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(ga[i]))
            rel = abs(num - ga[i]) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, i, num, ga[i])
    return max_rel, worst


# ==============================================================================
# 4. Training loop (mini-batch SGD + momentum)
# ==============================================================================
def accuracy(net, X, y):
    return float(np.mean(np.argmax(net.forward(X), axis=1) == y))


def train(net, Xtr, ytr, Xte, yte, epochs=60, lr=0.3, batch=64, mom=0.9, log=True):
    params = net.get_params()
    velo = {k: np.zeros_like(np.array(v, dtype=float)) for k, v in params.items()}
    n = Xtr.shape[0]
    history = []
    for ep in range(epochs):
        idx = RNG.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            net.loss(Xtr[bi], ytr[bi])
            grads = net.backward(ytr[bi])
            for k in params:
                gk = np.array(grads[k], dtype=float)
                velo[k] = mom * velo[k] - lr * gk
                cur = np.array(net.get_params()[k], dtype=float)
                net.set_param(k, cur + velo[k])
        if log and (ep % 10 == 0 or ep == epochs - 1):
            tr_l = net.loss(Xtr, ytr)
            te_a = accuracy(net, Xte, yte)
            print(f"  epoch {ep:3d} | disturbance(train loss)={tr_l:.4f} "
                  f"| test 3-way acc={te_a:.3f}")
        history.append(net.loss(Xtr, ytr))
    return history


# ==============================================================================
# 5. Behavioural self-tests — does it actually behave like Pyrrho?
# ==============================================================================
def behavioural_report(net, tau=0.6):
    """
    Probe the trained net with three crafted appearances and confirm:
      decisive-for-P     -> commits to P
      decisive-against-P -> commits to ~P
      perfectly balanced -> SUSPENDS (epoche)
    Uses the SAME deterministic rotation the training data used, so the probes
    live in the same world the network learned.
    """
    d_in = net.d_in
    rot = get_rotation(d_in)

    def appearance_from_tilt(t):
        x = np.zeros(d_in)
        x[0], x[1], x[2] = t, 0.7 * t, -0.4 * t
        return x @ rot

    probes = {
        "decisive_for_P": appearance_from_tilt(+1.8),
        "decisive_against_P": appearance_from_tilt(-1.8),
        "equipollent": appearance_from_tilt(0.0),
    }
    out = {name: net.forward(x[None, :])[0] for name, x in probes.items()}
    checks = {
        "for_P commits to P": np.argmax(out["decisive_for_P"]) == 0,
        "against_P commits to ~P": np.argmax(out["decisive_against_P"]) == 1,
        "equipollent suspends": np.argmax(out["equipollent"]) == 2,
    }
    return out, checks


# ==============================================================================
# 6. Main: run everything and print a verifiable transcript
# ==============================================================================
def main():
    print("=" * 78)
    print(" PYRRHO OF ELIS  —  Equipollence Suspension Network")
    print(" 'Of each thing I determine nothing; it no more is than is not.'")
    print("=" * 78)

    d_in = 8
    Xtr, ytr, _ = make_appearances(2000, d_in=d_in, seed=0)
    Xte, yte, _ = make_appearances(800, d_in=d_in, seed=1)
    print(f"\nData: train={Xtr.shape}, test={Xte.shape}")
    counts = np.bincount(ytr, minlength=3)
    print(f"Class balance (train)  assert-P={counts[0]}  "
          f"assert-~P={counts[1]}  SUSPEND={counts[2]}")

    net = EquipollenceSuspensionNetwork(d_in=d_in, d_hidden=16)

    # ---- (a) mandatory gradient check (small batch, fresh weights) ----------
    print("\n[1] Finite-difference gradient check ...")
    Xg, yg, _ = make_appearances(12, d_in=d_in, seed=7)
    max_rel, worst = gradient_check(net, Xg, yg)
    print(f"    max relative error = {max_rel:.2e}  (worst param: {worst[0]})")
    assert max_rel < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS  (analytic gradients match finite differences)")

    # ---- (b) train ----------------------------------------------------------
    print("\n[2] Training (mini-batch SGD + momentum):")
    pre_acc = accuracy(net, Xte, yte)
    train(net, Xtr, ytr, Xte, yte, epochs=80, lr=0.15, batch=64)
    post_acc = accuracy(net, Xte, yte)
    print(f"\n    test 3-way accuracy: {pre_acc:.3f} (before) -> "
          f"{post_acc:.3f} (after)")
    assert post_acc > 0.85, "training did not converge well enough"

    # ---- (c) behavioural self-test -----------------------------------------
    print("\n[3] Behavioural self-test — does suspension behave like epoche?")
    out, checks = behavioural_report(net)
    names = ["P", "~P", "SUSPEND"]
    for probe, p in out.items():
        pretty = "  ".join(f"{names[i]}={p[i]:.2f}" for i in range(3))
        print(f"    {probe:>20s}:  {pretty}  -> {names[int(np.argmax(p))]}")
    for desc, ok in checks.items():
        print(f"      [{'PASS' if ok else 'FAIL'}] {desc}")
    assert all(checks.values()), "behavioural Pyrrho-test failed"

    # ---- (d) ataraxia read-out ---------------------------------------------
    final_loss = net.loss(Xte, yte)
    print("\n[4] Ataraxia read-out:")
    print(f"    mean disturbance on unseen appearances (test CE loss) = "
          f"{final_loss:.4f}")
    print("    Low disturbance = the system commits only when appearances")
    print("    compel it and suspends when they are balanced. That calm,")
    print("    not certainty, is the Pyrrhonian objective.")

    print("\n" + "=" * 78)
    print(" ALL CHECKS PASSED — the network suspends from equipollence,")
    print(" not from a confidence threshold. epoche -> ataraxia.")
    print("=" * 78)


if __name__ == "__main__":
    main()

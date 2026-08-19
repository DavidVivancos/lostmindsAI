"""
================================================================================
 THE PROAIRETIC GATE NETWORK  —  a neural architecture after Epictetus (c.50-135 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 125: Epictetus (c.50-135 CE)
================================================================================   

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural network whose learning rule is built directly
out of Epictetus' single organising idea: the *dichotomy of control*. Everything
that enters the mind is an "impression" (phantasia). Before the mind commits to
it, the sovereign faculty of choice — the *proairesis* — sorts every impression
into two boxes:

    * eph' hemin      — "up to us"     : judgment, assent, intention, desire.
    * ouk eph' hemin  — "not up to us" : body, property, reputation, fortune.

Epictetus' therapeutic claim ("it is not things that disturb us, but our
judgments about things") becomes, in machine-learning terms, a precise and
testable engineering claim:

    A mind is *free / undisturbed* exactly to the extent that its judgments are
    INVARIANT to whatever it has classified as "not up to it."

So this network does not merely *represent* the dichotomy — it *learns* it, and
it learns it for the Stoic reason: because refusing to let its verdicts move
with fortune is the only way to stay accurate when fortune turns. Tranquility
(ataraxia) and out-of-distribution robustness turn out to be the same property.

THE DISTINCTIVE MECHANISM (why this is not just another classifier)
------------------------------------------------------------------
Most networks assent to every feature that correlates with the label in the
training set — including "lucky" features that happen to co-vary with the
answer today and will betray it tomorrow. That is precisely the person whose
serenity is hostage to fortune. This network is trained with an *apatheia
penalty*: it is shown a counterfactual world in which fortune (the external
features) is resampled, and it is penalised whenever its verdict changes. To
minimise that penalty the network is forced to discover — on its own — which
impressions are "not up to it," and to close a learned gate on them. The gate
is the dichotomy of control, discovered from the inside.

    "Wait for me a little, impression; let me see what you are and what you are
     about; let me test you."   — Discourses 2.18.24 (paraphrased)

NAMED PARTS (each maps to a real Epictetan concept)
---------------------------------------------------
    DichotomyGate  (proairesis' first act)  : per-impression controllability
                                              weight g in [0,1]; g~1 = "up to me",
                                              g~0 = "not up to me". Learned.
    Proairetikon   (the ruling faculty)      : the ONLY trainable judging core;
                                              it sees only the assented signal g*x.
    Sunkatathesis  (assent)                  : a confidence head that may WITHHOLD
                                              assent (epoche / suspension) when the
                                              impression is unclear.
    apatheia_loss  (freedom from passion)    : invariance-to-fortune penalty; the
                                              engine that *teaches* the dichotomy.
    prosoche       (attention/vigilance)     : mild sparsity on the gate — assent
                                              narrowly, not to everything.

WHAT IT DEMONSTRATES WHEN RUN
-----------------------------
On a task where an "external / fortune" feature is spuriously predictive at
train time and REVERSES at test time:
    * a naive twin (apatheia switched off) assents to fortune and collapses when
      fortune turns;
    * the Stoic model closes its gate on the uncontrollable, keeps its verdicts,
      and stays robust — undisturbed by the turn of fortune.
It also (a) passes a finite-difference gradient check on every parameter, and
(b) shows the gate closing on the external dimensions on its own.

Everything below is hand-written NumPy: forward, analytic backward, gradient
check, training loop, and self-tests. No autograd, no deep-learning framework.
================================================================================
"""

import numpy as np

# A single seeded generator keeps every run reproducible — a Stoic virtue:
# the sage is the same person in fair weather and foul.
RNG = np.random.default_rng(50)  # 50 == Epictetus' approximate birth year (CE)


# =============================================================================
# SECTION 1 — ATOMIC OPERATIONS
# Small, well-understood pieces. Each has an explicit derivative so the whole
# network can be differentiated by hand and then checked numerically.
# =============================================================================

def sigmoid(z):
    """Logistic squashing, used both for the control gate and for assent."""
    # Numerically stable two-branch form (avoids overflow for large |z|).
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z):
    """Row-wise softmax for the final verdict (probability over classes)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tanh(z):
    return np.tanh(z)


# =============================================================================
# SECTION 2 — THE PROAIRETIC GATE NETWORK
# =============================================================================

class ProaireticGateNetwork:
    """
    A three-stage mind:

        impression x  --DichotomyGate-->  g (what is up to me)
                                          |
                       assented signal  z = g * x   (assent only to the controllable)
                                          |
                        Proairetikon (tanh MLP)  -->  verdict logits  -->  softmax p
                                          |
                        Sunkatathesis head       -->  assent probability s

    Trainable parameters (ALL of them live inside proairesis — the seat of
    freedom; nothing outside it is ever updated, which is the whole point):

        Wg, bg   : DichotomyGate   (D->D)      the act of sorting impressions
        W1, b1   : Proairetikon L1 (D->H)      judgment, hidden
        W2, b2   : Proairetikon L2 (H->C)      the verdict
        wa, ba   : Sunkatathesis   (H->1)      assent / withholding
    """

    def __init__(self, d_in, d_hidden, n_classes, seed=0):
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_in)      # modest init keeps early impressions calm

        # --- DichotomyGate: initialised near g = 0.5 (undecided about control) ---
        # We start the gate biased slightly OPEN (bg small positive) so the mind
        # begins credulous and must *learn* restraint — the Stoic novice trusts
        # too much and is trained toward discernment.
        self.Wg = r.normal(0, s * 0.3, size=(d_in, d_in))
        self.bg = np.full(d_in, 0.2)

        # --- Proairetikon: the judging core ---
        self.W1 = r.normal(0, s, size=(d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = r.normal(0, 1.0 / np.sqrt(d_hidden), size=(d_hidden, n_classes))
        self.b2 = np.zeros(n_classes)

        # --- Sunkatathesis (assent) head ---
        self.wa = r.normal(0, 1.0 / np.sqrt(d_hidden), size=(d_hidden, 1))
        self.ba = np.zeros(1)

        self.d_in, self.d_hidden, self.n_classes = d_in, d_hidden, n_classes

    # -- convenience: pack/unpack parameters for the gradient checker ----------
    def params(self):
        return dict(Wg=self.Wg, bg=self.bg, W1=self.W1, b1=self.b1,
                    W2=self.W2, b2=self.b2, wa=self.wa, ba=self.ba)

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ FORWARD
    def forward(self, X):
        """
        Run one impression-batch through the mind.
        Returns (outputs, cache). The cache stores everything backward needs.
        """
        # 1) DichotomyGate — proairesis' first act: sort each impression.
        gate_logits = X @ self.Wg + self.bg          # (N, D)
        g = sigmoid(gate_logits)                     # controllability in [0,1]

        # 2) Assent only to the controllable part. The uncontrollable part
        #    (1-g)*x is deliberately DROPPED from the judging path: the mind
        #    forms no verdict on what is not up to it.
        z = g * X                                    # (N, D)  assented signal

        # 3) Proairetikon — the ruling faculty forms a judgment.
        a1 = z @ self.W1 + self.b1                   # (N, H)
        h = tanh(a1)
        logits = h @ self.W2 + self.b2               # (N, C)

        # 4) Sunkatathesis — how strongly does the mind assent to this verdict?
        s_logit = h @ self.wa + self.ba              # (N, 1)
        s = sigmoid(s_logit)                         # assent probability

        cache = dict(X=X, gate_logits=gate_logits, g=g, z=z,
                     a1=a1, h=h, logits=logits, s_logit=s_logit, s=s)
        return dict(g=g, logits=logits, p=softmax(logits), s=s), cache

    # ----------------------------------------------------------------- BACKWARD
    def _backward_from_logits(self, dlogits, dh_extra, cache):
        """
        Backprop a gradient-on-logits (and an extra gradient arriving at h, e.g.
        from the assent head) through Proairetikon and the DichotomyGate.
        Returns a grads dict AND the incoming dg (so callers can add gate
        regularisation before it is turned into dgate_logits — but here we fold
        gate reg in separately). No input gradient is produced (none is needed).
        """
        X = cache["X"]; g = cache["g"]; h = cache["h"]; z = cache["z"]
        grads = {k: 0.0 for k in ("Wg", "bg", "W1", "b1", "W2", "b2")}

        # verdict layer
        grads["W2"] = h.T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T + dh_extra           # (N, H)

        # tanh hidden
        da1 = dh * (1.0 - h * h)                       # (N, H)
        grads["W1"] = z.T @ da1
        grads["b1"] = da1.sum(axis=0)
        dz = da1 @ self.W1.T                           # (N, D)

        # z = g * X  ->  dg from the judging path
        dg = dz * X                                    # (N, D)
        return grads, dg

    def _gate_backward(self, dg, cache, grads):
        """Turn a gradient-on-g into gradients on the DichotomyGate params."""
        X = cache["X"]; g = cache["g"]
        dgate_logits = dg * g * (1.0 - g)              # sigmoid'
        grads["Wg"] = grads.get("Wg", 0.0) + X.T @ dgate_logits
        grads["bg"] = grads.get("bg", 0.0) + dgate_logits.sum(axis=0)
        return grads

    # =========================================================================
    # SECTION 3 — THE STOIC LOSS
    # L = cross_entropy(verdict)                       "judge the controllable"
    #   + lam_apatheia * invariance_to_fortune         "be undisturbed by externals"
    #   + lam_assent   * assent_calibration            "test before assenting"
    #   + lam_prosoche * gate_sparsity                 "assent narrowly, vigilantly"
    #
    # The apatheia term is the beating heart. We evaluate the mind on the real
    # impression X and on a COUNTERFACTUAL X_cf in which fortune (the external
    # features) has been resampled. We punish any change of verdict. That single
    # term is what forces the DichotomyGate to close on the uncontrollable.
    # =========================================================================
    def loss_and_grads(self, X, y, X_cf,
                       lam_apatheia=1.0, lam_assent=0.3, lam_prosoche=1e-3):
        N = X.shape[0]
        onehot = np.zeros((N, self.n_classes)); onehot[np.arange(N), y] = 1.0

        # --- forward on the real world and on the counterfactual-fortune world -
        out, cache = self.forward(X)
        out_cf, cache_cf = self.forward(X_cf)
        p = out["p"]; logits = out["logits"]; logits_cf = out_cf["logits"]

        # ---- (a) cross-entropy: the verdict on the real impression -----------
        L_ce = -np.mean(np.log(p[np.arange(N), y] + 1e-12))
        dlogits_ce = (p - onehot) / N                  # (N, C)

        # ---- (b) apatheia: verdict must not move when fortune moves ----------
        diff = logits - logits_cf                      # (N, C)
        L_ap = 0.5 * np.mean(np.sum(diff * diff, axis=1))
        # d/dlogits of 0.5*mean_N sum_C diff^2  ->  diff / N  (and -diff/N for cf)
        dlogits_ap = diff / N
        dlogits_ap_cf = -diff / N

        # ---- (c) sunkatathesis: assent should track being correct ------------
        # Target = 1 if the verdict is right, else 0 (stop-gradient on target):
        correct = (np.argmax(logits, axis=1) == y).astype(np.float64)[:, None]
        s = out["s"]
        L_as = -np.mean(correct * np.log(s + 1e-12) +
                        (1 - correct) * np.log(1 - s + 1e-12))
        ds_logit = (s - correct) / N                   # (N,1) BCE grad wrt s_logit
        dh_assent = ds_logit @ self.wa.T               # flows into h of REAL pass

        # ---- (d) prosoche: keep the gate sparse (assent narrowly) ------------
        g = cache["g"]
        L_pr = np.mean(g)
        dg_prosoche = np.full_like(g, lam_prosoche / g.size)

        # ---------------- assemble total loss --------------------------------
        L = L_ce + lam_apatheia * L_ap + lam_assent * L_as + lam_prosoche * L_pr

        # ---------------- backprop: REAL pass --------------------------------
        dlogits_real = dlogits_ce + lam_apatheia * dlogits_ap
        grads, dg_real = self._backward_from_logits(dlogits_real,
                                                    lam_assent * dh_assent, cache)
        # assent head params
        grads["wa"] = lam_assent * (cache["h"].T @ ds_logit)
        grads["ba"] = lam_assent * ds_logit.sum(axis=0)
        # gate grads from real pass (judging path + prosoche sparsity)
        grads = self._gate_backward(dg_real + dg_prosoche, cache, grads)

        # ---------------- backprop: COUNTERFACTUAL pass ----------------------
        grads_cf, dg_cf = self._backward_from_logits(lam_apatheia * dlogits_ap_cf,
                                                     np.zeros_like(cache_cf["h"]),
                                                     cache_cf)
        self._gate_backward(dg_cf, cache_cf, grads_cf)
        # sum the two passes' contributions to the shared parameters
        for k in ("Wg", "bg", "W1", "b1", "W2", "b2"):
            grads[k] = grads[k] + grads_cf[k]

        parts = dict(L_ce=L_ce, L_apatheia=L_ap, L_assent=L_as, L_prosoche=L_pr)
        return L, grads, parts


# =============================================================================
# SECTION 4 — THE WORLD: a task where fortune lies, then turns
# =============================================================================
# Each impression x = [ controllable signal c ... | external "fortune" e ... ].
#   * The TRUE label depends ONLY on c   (what is up to us).
#   * At TRAIN time, fortune e is spuriously correlated with the label.
#   * At TEST  time, that correlation REVERSES (fortune turns).
# A mind that assents to fortune wins at train and is destroyed at test.
# A mind that has learned the dichotomy is undisturbed.
# =============================================================================

def make_world(n, d_ctrl=6, d_ext=6, fortune_sign=+1.0,
               fortune_strength=3.0, rng=None):
    rng = rng or np.random.default_rng(0)
    D = d_ctrl + d_ext

    # controllable signal and the true, control-only decision rule
    c = rng.normal(0, 1, size=(n, d_ctrl))
    w_true = np.linspace(1.0, -1.0, d_ctrl)              # fixed "law of reason"
    score = c @ w_true
    y = (score > 0).astype(int)                          # binary verdict

    # external "fortune": correlated with y by fortune_sign, plus noise
    signed = (2 * y - 1)[:, None].astype(np.float64)     # +/-1
    direction = np.ones(d_ext) / np.sqrt(d_ext)
    e = fortune_sign * fortune_strength * signed * direction
    e = e + rng.normal(0, 1.0, size=(n, d_ext))

    X = np.concatenate([c, e], axis=1)
    return X.astype(np.float64), y, d_ctrl, d_ext, D


def counterfactual_fortune(X, d_ctrl, rng):
    """Resample fortune: keep controllable dims, shuffle external dims across the
    batch. This is the 'what if fortune had fallen otherwise?' world used by the
    apatheia penalty."""
    X_cf = X.copy()
    perm = rng.permutation(X.shape[0])
    X_cf[:, d_ctrl:] = X[perm, d_ctrl:]
    return X_cf


# =============================================================================
# SECTION 5 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# Confirms the hand-written backward matches numerical derivatives of the loss.
# =============================================================================

def gradient_check(verbose=True):
    rng = np.random.default_rng(7)
    net = ProaireticGateNetwork(d_in=8, d_hidden=5, n_classes=3, seed=1)
    X = rng.normal(size=(6, 8))
    y = rng.integers(0, 3, size=6)
    X_cf = counterfactual_fortune(X, d_ctrl=4, rng=rng)

    def total_loss():
        L, _, _ = net.loss_and_grads(X, y, X_cf)
        return L

    _, grads, _ = net.loss_and_grads(X, y, X_cf)

    eps = 1e-6
    worst = 0.0
    for name, P in net.params().items():
        flat = P.ravel()
        g_an = grads[name].ravel()
        # check a handful of coordinates per parameter tensor
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp = total_loss()
            flat[i] = orig - eps; Lm = total_loss()
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            denom = max(1e-12, abs(num) + abs(g_an[i]))
            rel = abs(num - g_an[i]) / denom
            worst = max(worst, rel)
            if verbose and rel > 1e-4:
                print(f"  [warn] {name}[{i}] rel={rel:.2e} "
                      f"num={num:+.6f} an={g_an[i]:+.6f}")
    if verbose:
        print(f"  worst relative error across sampled params: {worst:.2e}")
    return worst


# =============================================================================
# SECTION 6 — TRAINING
# =============================================================================

def train(net, X, y, d_ctrl, epochs=400, lr=0.15, lam_apatheia=1.0,
          lam_assent=0.3, batch=128, rng=None, log_every=100, tag=""):
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        Xs, ys = X[idx], y[idx]
        ep_loss = 0.0; nb = 0
        for s0 in range(0, n, batch):
            xb = Xs[s0:s0 + batch]; yb = ys[s0:s0 + batch]
            xcf = counterfactual_fortune(xb, d_ctrl, rng)
            L, grads, _ = net.loss_and_grads(xb, yb, xcf,
                                             lam_apatheia=lam_apatheia,
                                             lam_assent=lam_assent)
            # plain SGD — the disciplined, repetitive daily training (askesis)
            for k, P in net.params().items():
                P -= lr * grads[k]
            ep_loss += L; nb += 1
        if log_every and (ep % log_every == 0 or ep == 1):
            acc = accuracy(net, X, y)
            print(f"  [{tag}] epoch {ep:4d}  loss {ep_loss/nb:.4f}  train_acc {acc:.3f}")
    return net


# =============================================================================
# SECTION 7 — EVALUATION
# =============================================================================

def accuracy(net, X, y):
    out, _ = net.forward(X)
    return float(np.mean(np.argmax(out["p"], axis=1) == y))


def mean_gate(net, X, d_ctrl):
    """Average gate opening on controllable vs external dims. The dichotomy,
    made visible: how much the mind treats each region as 'up to it'."""
    out, _ = net.forward(X)
    g = out["g"]
    return float(g[:, :d_ctrl].mean()), float(g[:, d_ctrl:].mean())


def suspension_rate(net, X, thresh=0.5):
    """Fraction of impressions to which the mind WITHHOLDS assent (epoche):
    it declines to commit when the sunkatathesis head is unsure."""
    out, _ = net.forward(X)
    return float(np.mean(out["s"][:, 0] < thresh))


# =============================================================================
# SECTION 8 — MAIN: gradient check, then the fortune-turns experiment, then tests
# =============================================================================

def main():
    print("=" * 78)
    print("THE PROAIRETIC GATE NETWORK — after Epictetus (c.50-135 CE)")
    print("The dichotomy of control as a learned gate; tranquility as invariance.")
    print("=" * 78)

    # ---- 1) gradient check (mandatory) --------------------------------------
    print("\n[1] Finite-difference gradient check")
    worst = gradient_check(verbose=True)
    assert worst < 1e-4, f"gradient check FAILED (worst rel err {worst:.2e})"
    print("    PASS — analytic gradients match numerical derivatives.")

    # ---- 2) build the world: fortune helps at train, reverses at test -------
    print("\n[2] Building the world (fortune is a liar that turns)")
    rng = np.random.default_rng(123)
    X_tr, y_tr, d_c, d_e, D = make_world(4000, fortune_sign=+1.0, rng=rng)
    # SAME true rule, fortune REVERSED — the turn of fortune:
    X_te, y_te, _, _, _ = make_world(2000, fortune_sign=-1.0, rng=rng)
    print(f"    controllable dims={d_c}  external(fortune) dims={d_e}  total D={D}")

    # ---- 3a) the naive twin: apatheia OFF (assents to fortune) --------------
    print("\n[3a] Naive twin  (apatheia OFF, lam_apatheia=0): trusts fortune")
    naive = ProaireticGateNetwork(D, d_hidden=24, n_classes=2, seed=11)
    train(naive, X_tr, y_tr, d_c, epochs=400, lr=0.2,
          lam_apatheia=0.0, lam_assent=0.3,
          rng=np.random.default_rng(1), log_every=200, tag="naive")

    # ---- 3b) the Stoic: apatheia ON (learns the dichotomy) ------------------
    print("\n[3b] Stoic mind  (apatheia ON): undisturbed by fortune")
    stoic = ProaireticGateNetwork(D, d_hidden=24, n_classes=2, seed=11)
    train(stoic, X_tr, y_tr, d_c, epochs=400, lr=0.2,
          lam_apatheia=2.0, lam_assent=0.3,
          rng=np.random.default_rng(1), log_every=200, tag="stoic")

    # ---- 4) the reckoning: when fortune turns -------------------------------
    print("\n[4] The reckoning — accuracy before and after fortune turns")
    ntr_tr, ntr_te = accuracy(naive, X_tr, y_tr), accuracy(naive, X_te, y_te)
    sto_tr, sto_te = accuracy(stoic, X_tr, y_tr), accuracy(stoic, X_te, y_te)
    print(f"    naive twin : train_acc {ntr_tr:.3f}   test_acc {ntr_te:.3f}")
    print(f"    Stoic mind : train_acc {sto_tr:.3f}   test_acc {sto_te:.3f}")

    gc_ctrl, gc_ext = mean_gate(stoic, X_te, d_c)
    print(f"\n    Stoic gate  : controllable={gc_ctrl:.3f}  external(fortune)={gc_ext:.3f}")
    print(f"    (the gate closed on what is not up to it — the dichotomy, learned)")
    ng_ctrl, ng_ext = mean_gate(naive, X_te, d_c)
    print(f"    naive gate  : controllable={ng_ctrl:.3f}  external(fortune)={ng_ext:.3f}")
    print(f"    suspension rate (Stoic, withheld assent): {suspension_rate(stoic, X_te):.3f}")

    # ---- 5) self-tests -------------------------------------------------------
    print("\n[5] Self-tests")
    ok = True

    t1 = worst < 1e-4
    print(f"    gradient check < 1e-4 ................... {'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # The Stoic mind survives the turn of fortune far better than the naive twin.
    t2 = sto_te > ntr_te + 0.15
    print(f"    Stoic beats naive under fortune-turn .... {'PASS' if t2 else 'FAIL'}"
          f"  ({sto_te:.3f} vs {ntr_te:.3f})")
    ok &= t2

    # The Stoic mind stays serene (accurate) across the turn: little train->test drop.
    t3 = (sto_tr - sto_te) < 0.12
    print(f"    Stoic stays serene across fortune-turn .. {'PASS' if t3 else 'FAIL'}"
          f"  (drop {sto_tr - sto_te:+.3f})")
    ok &= t3

    # The dichotomy actually formed: gate more open on controllable than external.
    t4 = gc_ctrl > gc_ext + 0.15
    print(f"    dichotomy learned (gate ctrl>ext) ....... {'PASS' if t4 else 'FAIL'}"
          f"  ({gc_ctrl:.3f} vs {gc_ext:.3f})")
    ok &= t4

    # Assent behaves like a probability.
    out, _ = stoic.forward(X_te)
    t5 = bool(np.all(out["s"] >= 0) and np.all(out["s"] <= 1))
    print(f"    assent (sunkatathesis) in [0,1] ......... {'PASS' if t5 else 'FAIL'}")
    ok &= t5

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED." if ok else "SOME SELF-TESTS FAILED.")
    print("Freedom, for this network, is a learned invariance: it refuses to let")
    print("its verdicts be moved by what it has judged is not up to it.")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

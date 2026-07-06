#!/usr/bin/env python3
# =============================================================================
#  ASSENT-GATED KATALEPSIS NETWORK  (AGKN)
#  chapter_0081_zeno_of_citium_-334.py - Zeno of Citium (c. 334-262 BCE), founder of the Stoa
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0081 · Zeno of Citium
#  A from-scratch, pure-NumPy cognitive architecture built around the one idea
#  that is Zeno's and no one else's: the act of ASSENT (synkatathesis).
#
#  Zeno's epistemology, in his own famous gesture (reported by Cicero,
#  Academica II.145):
#     * open hand             -> an impression strikes us (phantasia)
#     * fingers bent a little -> we ASSENT to it (synkatathesis)
#     * a closed fist         -> we GRASP it: katalepsis (a word Zeno coined)
#     * the fist gripped by    -> KNOWLEDGE (episteme), secure, held only
#       the other hand            by the wise.
#
#  The decisive Stoic claim: impressions arrive UNGOVERNED - we do not choose
#  what strikes the mind. The ONLY thing "up to us" is whether we give assent.
#  Wisdom is therefore not a faculty for producing impressions but a trained
#  DISCRIMINATION that assents ONLY to impressions carrying their own warrant
#  -- the "cataleptic" impressions that could not have come from what is not --
#  and that WITHHOLDS assent (epoche) from the rest. The sage who never assents
#  to a non-cataleptic impression never holds mere opinion (doxa), and so never
#  errs.
#
#  THE ARCHITECTURE MAKES THAT MECHANICAL. It is NOT a classifier with a
#  confidence threshold bolted on. Two faculties share one ruling state:
#     - the KATALEPSIS head always forms a candidate judgement about every
#       impression (for the Stoics the proposition is always entertained);
#     - the ASSENT gate is a SEPARATE act that decides whether to commit.
#  The gate is trained under a SELECTIVE-RISK objective in which withholding
#  has a fixed price c. The learned policy is exactly Chow's rule -- assent iff
#  the expected error of committing is below the price of suspending judgement
#  -- but here it is derived from Zeno's theory of assent, not borrowed.
#
#  How this differs from neighbours in the corpus:
#     * NOT Epicurus (#79): for him every sensation is true and error lives
#       only in the added opinion. Here impressions VARY in warrant and the
#       whole skill is discriminating cataleptic from non-cataleptic BEFORE
#       committing.
#     * NOT Pyrrho (#77), who suspends on everything. This network commits
#       firmly wherever warrant is high; epoche is selective, not universal.
#     * NOT the later "dichotomy of control" of Epictetus. Zeno's "up to us"
#       is located precisely at the act of assent, not at outward action.
#
#  Mandatory corpus convention: pure NumPy; an analytic backward pass checked
#  against finite differences; a real training loop on a task that genuinely
#  REQUIRES abstention; hard-asserted self-tests. Run directly; the printed
#  output is pasted into the chapter.
# =============================================================================

import numpy as np

SEED = 81
rng = np.random.default_rng(SEED)


# -----------------------------------------------------------------------------
# 0.  Numerical helpers
# -----------------------------------------------------------------------------
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def one_hot(y, K):
    M = np.zeros((y.shape[0], K))
    M[np.arange(y.shape[0]), y] = 1.0
    return M


# -----------------------------------------------------------------------------
# 1.  THE WORLD OF IMPRESSIONS  (the synthetic task)
#
#     Zeno conceded that MOST impressions are unclear, complex, or misleading;
#     only a privileged few are cataleptic. The data therefore mixes both, and
#     the murky ones are genuinely UNRELIABLE, not merely noisy: their LABELS
#     are sometimes illusory (the bent oar that looks broken; Orestes seeing
#     his sister as a Fury). The network is never told which impressions are
#     clear - it must infer warrant from the input alone, exactly Zeno's task.
# -----------------------------------------------------------------------------
def make_impressions(n, d=8, K=3, p_clear=0.62, sig_clear=0.18,
                     sig_murky=1.15, p_flip=0.46, centers=None):
    if centers is None:
        centers = rng.normal(0, 1.0, size=(K, d))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 2.6
    y_true = rng.integers(0, K, size=n)
    is_clear = rng.random(n) < p_clear
    X = np.empty((n, d))
    y_seen = y_true.copy()
    for i in range(n):
        c = y_true[i]
        if is_clear[i]:
            X[i] = centers[c] + rng.normal(0, sig_clear, size=d)   # cataleptic
        else:
            X[i] = centers[c] + rng.normal(0, sig_murky, size=d)   # diffuse
            if rng.random() < p_flip:                              # illusory
                others = [k for k in range(K) if k != c]
                y_seen[i] = others[rng.integers(0, len(others))]
    return X, y_seen, y_true, is_clear, centers


# -----------------------------------------------------------------------------
# 2.  THE NETWORK
#
#     h   = tanh(W1 x + b1)            hegemonikon: the impression received and
#                                      shaped by the ruling faculty (open hand)
#     z   = W2 h + b2 ; p = softmax(z) candidate grasp / katalepsis (the fist)
#     ga  = tanh(Wg1 h + bg1)          the deliberation over assent
#     g   = wg2 . ga + bg2 ; s=sig(g)  the degree of assent (the bent fingers)
#
#  The gate is a small two-layer faculty so it can carve the region "near any
#  prototype" (a union of clusters) that a single line could not separate.
# -----------------------------------------------------------------------------
class AssentGatedKatalepsisNetwork:
    def __init__(self, d, H, K, Hg=16, l2=1e-4, reject_cost=0.45,
                 beta=1.0, gamma=0.3, target_coverage=0.55, bg2_init=1.0):
        s1 = np.sqrt(2.0 / d)
        s2 = np.sqrt(2.0 / H)
        sg = np.sqrt(2.0 / H)
        self.W1 = rng.normal(0, s1, size=(H, d))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, s2, size=(K, H))
        self.b2 = np.zeros(K)
        self.Wg1 = rng.normal(0, sg, size=(Hg, H))
        self.bg1 = np.zeros(Hg)
        self.wg2 = rng.normal(0, np.sqrt(2.0 / Hg), size=Hg)
        # Start GENEROUS: the untrained mind assents too readily (the Stoics'
        # propeteia, "rash assent"); training teaches it restraint.
        self.bg2 = float(bg2_init)
        self.d, self.H, self.K, self.Hg = d, H, K, Hg
        self.l2 = l2
        # c = the price of suspending judgement. Without a price a timid mind
        # would withhold on everything (the Pyrrhonist failure). With it, the
        # optimal policy is Chow's rule.
        self.c = reject_cost
        # beta weights the selective objective against the always-on duty of
        # forming a candidate grasp; gamma is a COVERAGE ANCHOR that keeps the
        # gate from sliding into withholding-on-everything (the Pyrrhonist
        # failure); tau is the coverage it should not fall below.
        self.beta = beta
        self.gamma = gamma
        self.tau = target_coverage

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.Wg1, self.bg1, self.wg2, np.array([self.bg2])]

    def set_params(self, p):
        (self.W1, self.b1, self.W2, self.b2,
         self.Wg1, self.bg1, self.wg2, bg2) = p
        self.bg2 = float(bg2[0])

    # ----------------------------- forward -----------------------------------
    def forward(self, X):
        U = X @ self.W1.T + self.b1
        Hh = np.tanh(U)
        Z = Hh @ self.W2.T + self.b2
        P = softmax(Z)
        A = Hh @ self.Wg1.T + self.bg1
        Ga = np.tanh(A)
        G = Ga @ self.wg2 + self.bg2
        S = sigmoid(G)
        return P, S, dict(X=X, Hh=Hh, Z=Z, P=P, Ga=Ga, S=S)

    # ------------------------- the selective loss ----------------------------
    #   CE_i = -log P[i, y_i]                      (cost if we commit)
    #   L = mean(CE)                               (duty: always form a grasp)
    #     + beta * mean( S*CE + (1-S)*c )          (Chow selective term)
    #     + gamma * (mean(S) - tau)^2              (coverage anchor)
    #   The classification duty keeps the grasp head learning whatever the gate
    #   does, which dissolves the degenerate "withhold on everything" trap; the
    #   Chow term teaches the gate which impressions to grasp; the anchor keeps
    #   it from collapsing to silence. beta/gamma may be overridden so they can
    #   be switched on AFTER a head warm-up.
    def loss_and_grads(self, X, y, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        N = X.shape[0]
        P, S, cache = self.forward(X)
        Hh, Ga = cache["Hh"], cache["Ga"]
        Y = one_hot(y, self.K)
        CE = -np.log(np.clip(P[np.arange(N), y], 1e-12, 1.0))
        Sbar = S.mean()
        reg = 0.5 * self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2)
                               + np.sum(self.Wg1**2) + np.sum(self.wg2**2))
        loss = (CE.mean()
                + beta * (self.c + S * (CE - self.c)).mean()
                + gamma * (Sbar - self.tau)**2
                + reg)

        # ---- backward (analytic, fully coupled: no stop-gradient) ----
        # CE reaches the grasp head from the duty term and the selective term:
        dCE = (1.0 + beta * S) / N                      # (N,)
        dZ = dCE[:, None] * (P - Y)                      # (N,K)
        gW2 = dZ.T @ Hh + self.l2 * self.W2
        gb2 = dZ.sum(axis=0)
        dHh = dZ @ self.W2                               # (N,H) from grasp

        # assent gate (two layers): selective term + coverage anchor via S
        dS = (beta / N) * (CE - self.c) + gamma * 2.0 * (Sbar - self.tau) / N
        dg = dS * S * (1.0 - S)                          # (N,)
        gwg2 = Ga.T @ dg + self.l2 * self.wg2
        gbg2 = dg.sum()
        dA = np.outer(dg, self.wg2) * (1.0 - Ga**2)     # (N,Hg)
        gWg1 = dA.T @ Hh + self.l2 * self.Wg1
        gbg1 = dA.sum(axis=0)
        dHh = dHh + dA @ self.Wg1                        # gate's pull on the
                                                         # hegemonikon
        # back through the encoder
        dU = dHh * (1.0 - Hh**2)
        gW1 = dU.T @ X + self.l2 * self.W1
        gb1 = dU.sum(axis=0)

        grads = [gW1, gb1, gW2, gb2, gWg1, gbg1, gwg2, np.array([gbg2])]
        stats = dict(CE=CE, S=S, loss=loss,
                     assent_rate=float((S > 0.5).mean()))
        return loss, grads, stats

    # --------------------------- Adam optimiser ------------------------------
    def fit(self, X, y, Xval=None, yval=None, epochs=900, warmup=200, lr=1e-2,
            b1=0.9, b2=0.999, eps=1e-8, log_every=100, verbose=True):
        params = self.get_params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        hist = []
        for t in range(1, epochs + 1):
            # WARM-UP: first train the grasp head alone (gate frozen at its
            # generous init). Only a mind that can already judge can be taught
            # to assent wisely.
            if t <= warmup:
                beta_eff, gamma_eff = 0.0, 0.0
            else:
                beta_eff, gamma_eff = self.beta, self.gamma
            loss, grads, stats = self.loss_and_grads(X, y, beta_eff, gamma_eff)
            for i in range(len(params)):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i]**2
                mhat = m[i] / (1 - b1**t)
                vhat = v[i] / (1 - b2**t)
                params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            self.set_params(params)
            if verbose and (t % log_every == 0 or t == 1 or t == warmup + 1):
                tag = "warm" if t <= warmup else "asnt"
                msg = (f"  epoch {t:4d} [{tag}] | loss {loss:.4f} "
                       f"| assent-rate {stats['assent_rate']:.2f}")
                if Xval is not None:
                    ev = self.evaluate(Xval, yval)
                    msg += (f" | committed {ev['assented_acc']:.3f} "
                            f"| blind {ev['blind_acc']:.3f}")
                print(msg)
            hist.append(loss)
        return hist

    # ------------------------------ inference --------------------------------
    def predict(self, X, threshold=0.5):
        P, S, _ = self.forward(X)
        return P.argmax(axis=1), S > threshold, S, P

    def evaluate(self, X, y_true, threshold=0.5):
        yhat, assent, S, P = self.predict(X, threshold)
        blind = float((yhat == y_true).mean())
        cov = float(assent.mean())
        a_acc = (float((yhat[assent] == y_true[assent]).mean())
                 if assent.sum() else float("nan"))
        w_acc = (float((yhat[~assent] == y_true[~assent]).mean())
                 if (~assent).sum() else float("nan"))
        return dict(blind_acc=blind, assented_acc=a_acc, withheld_acc=w_acc,
                    coverage=cov, S=S)


# -----------------------------------------------------------------------------
# 3.  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
#     The disciplined check IS the Stoic point: assent to your own reasoning
#     only once it has survived scrutiny.
# -----------------------------------------------------------------------------
def gradient_check(net, X, y, eps=1e-5, n_probe=24):
    _, grads, _ = net.loss_and_grads(X, y)
    params = net.get_params()
    names = ["W1", "b1", "W2", "b2", "Wg1", "bg1", "wg2", "bg2"]
    max_rel = 0.0
    worst = None
    for P, G, nm in zip(params, grads, names):
        flat, gflat = P.ravel(), G.ravel()
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size),
                          replace=False)
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + eps; net.set_params(params)
            lp, _, _ = net.loss_and_grads(X, y)
            flat[j] = orig - eps; net.set_params(params)
            lm, _, _ = net.loss_and_grads(X, y)
            flat[j] = orig; net.set_params(params)
            num = (lp - lm) / (2 * eps)
            ana = gflat[j]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            if rel > max_rel:
                max_rel, worst = rel, nm
    return max_rel, worst


# -----------------------------------------------------------------------------
# 4.  Selective-prediction diagnostics
# -----------------------------------------------------------------------------
def risk_coverage_table(net, X, y_true, fracs=(0.2, 0.4, 0.6, 0.8, 1.0)):
    _, _, S, P = net.predict(X)
    yhat = P.argmax(axis=1)
    order = np.argsort(-S)
    rows = []
    for f in fracs:
        k = max(1, int(f * len(S)))
        sel = order[:k]
        rows.append((f, float((yhat[sel] == y_true[sel]).mean()),
                     float(S[sel].min())))
    return rows


def warrant_bins(net, X, y_true, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    _, _, S, P = net.predict(X)
    yhat = P.argmax(axis=1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (S >= lo) & ((S < hi) if hi < 1.0 else (S <= hi))
        rows.append((lo, hi, int(m.sum()),
                     float((yhat[m] == y_true[m]).mean()) if m.sum()
                     else float("nan")))
    return rows


# -----------------------------------------------------------------------------
# 5.  RUN
# -----------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("  ASSENT-GATED KATALEPSIS NETWORK")
    print("  After Zeno of Citium - knowledge as selective, abstaining assent")
    print("=" * 74)

    d, H, K = 8, 24, 3
    Xtr, ytr_seen, ytr_true, clear_tr, centers = make_impressions(1600, d, K)
    Xte, yte_seen, yte_true, clear_te, _ = make_impressions(
        600, d, K, centers=centers)
    illusory = (ytr_seen[~clear_tr] != ytr_true[~clear_tr]).mean()
    print(f"\n  impressions: {Xtr.shape[0]} train / {Xte.shape[0]} test"
          f"  | features={d} classes={K} hegemonikon={H}")
    print(f"  cataleptic (clear) fraction in train : {clear_tr.mean():.2f}")
    print(f"  murky impressions whose label is false: {illusory:.2f}")

    net = AssentGatedKatalepsisNetwork(d, H, K, Hg=16, l2=1e-4,
                                       reject_cost=0.32, beta=1.0,
                                       gamma=0.3, target_coverage=0.45)

    print("\n--- 1. finite-difference gradient check -------------------------")
    rel, worst = gradient_check(net, Xtr[:64], ytr_seen[:64])
    print(f"  max relative error (worst: {worst}) : {rel:.2e}")
    assert rel < 1e-5, "GRADIENT CHECK FAILED"
    print("  PASS  (analytic backward pass verified)")

    print("\n--- 2. training (learning the discipline of assent) -------------")
    net.fit(Xtr, ytr_seen, Xval=Xte, yval=yte_true,
            epochs=500, warmup=200, lr=1e-2, log_every=100)

    print("\n--- 3. evaluation against the TRUE source of each impression ----")
    ev = net.evaluate(Xte, yte_true)
    print(f"  coverage (fraction grasped)      : {ev['coverage']:.3f}")
    print(f"  blind accuracy (commit to all)   : {ev['blind_acc']:.3f}")
    print(f"  committed accuracy (grasped only): {ev['assented_acc']:.3f}")
    print(f"  withheld accuracy (suspended)    : {ev['withheld_acc']:.3f}")
    print(f"  selective gain (committed-blind) : "
          f"{ev['assented_acc'] - ev['blind_acc']:+.3f}")

    print("\n--- 4. risk-coverage (accuracy as the mind restricts itself) ---")
    print("    coverage   accuracy   min-assent-in-band")
    for f, acc, smin in risk_coverage_table(net, Xte, yte_true):
        print(f"      {f:>4.0%}      {acc:6.3f}        {smin:5.2f}")

    print("\n--- 5. accuracy by assent band (is the warrant meaningful?) ----")
    print("    assent-band    n     accuracy")
    for lo, hi, n, acc in warrant_bins(net, Xte, yte_true):
        a = "  n/a" if np.isnan(acc) else f"{acc:6.3f}"
        print(f"    [{lo:.1f},{hi:.1f})   {n:4d}    {a}")

    print("\n--- 6. self-tests ----------------------------------------------")
    assert ev['assented_acc'] > ev['blind_acc'] + 0.05, \
        "committed accuracy should clearly beat blind accuracy"
    assert 0.05 < ev['coverage'] < 0.98, \
        "coverage should be selective, neither total nor empty"
    assert ev['assented_acc'] > ev['withheld_acc'], \
        "what is grasped should be more reliable than what is withheld"
    rc = [a for _, a, _ in risk_coverage_table(net, Xte, yte_true)]
    assert rc[0] >= rc[-1] - 1e-9, "accuracy should not rise with coverage"
    print("  PASS  committed accuracy beats blind accuracy")
    print("  PASS  abstention is selective")
    print("  PASS  grasped impressions outrank withheld ones")
    print("  PASS  risk falls as coverage falls (the sage's restraint)")

    print("\n" + "=" * 74)
    print("  The mind does not choose its impressions, only its assent.")
    print("  Trained to that discipline, what it grasps, it grasps securely.")
    print("=" * 74)


if __name__ == "__main__":
    main()

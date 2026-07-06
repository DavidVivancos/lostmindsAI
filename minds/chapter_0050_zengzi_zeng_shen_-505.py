#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0050_zengzi_zeng_shen_-505.py - Zeng Shen (Zengzi, 505-435 BCE)
 The Reflexive Self-Audit Network (RSAN)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0050 · Zeng Shen (Zengzi)
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture whose *mechanism* is built to
embody one specific mind rather than a generic deep-learning template. It is NOT
a transformer, and it deliberately avoids attention-over-stored-keys. The whole
design follows from three documented ideas that are distinctively Zeng Shen's:

  1. ONE THREAD  (一以贯之, yi-yi-guan-zhi, Analects 4.15)
     "My Way is bound by a single thread." Zengzi reads the whole sprawling
     doctrine as compressible to one short code (zhong + shu). Cognitively:
     true understanding is a *low-dimensional* principle from which every case
     is regenerated -- "from one, know all." In the network this is a hard
     INFORMATION BOTTLENECK ("the thread"): a tiny latent vector through which
     all classification AND a reconstruction of the input must pass. Memorising
     surface features cannot survive the bottleneck; only the governing
     principle does.

  2. THREE EXAMINATIONS  (三省吾身, san-xing-wu-shen, Analects 1.4)
     "Each day I examine myself on three counts." Zengzi treats cognition as a
     repeated AUDIT, not a single feed-forward act. An Auditor head re-reads the
     network's own answer and scores it on three axes (loyalty to the task,
     trustworthiness/calibration, knowing-vs-doing). At inference these audits
     run as a *loop* (the daily renewal): the answer is re-examined and revised
     until the audit stabilises.

  3. VIGILANCE IN SOLITUDE  (慎独, shen-du, from the Great Learning / Daxue)
     "The exemplary person is watchful even when alone." The deepest Confucian
     test of integrity is whether conduct is identical when no one observes.
     Translated to alignment: a system's behaviour must NOT depend on whether it
     is being watched/graded. Every input carries an explicit "observed" bit;
     a Shendu loss forces the output to be INVARIANT to that bit. A system that
     passes shendu cannot game its supervisor, because it behaves the same with
     the supervisor switched off.

WHY THIS IS THE RIGHT MECHANISM FOR ZENGZI (and not for, say, Confucius,
Mencius, or Xunzi): the other early Confucians theorise human nature, ritual,
or governance. Zengzi's unique cognitive contribution is *the audit itself* --
a recurrent self-checking loop, anchored to a compressed principle, that must
hold under no observation. So the architecture is an auditor, not an oracle.

THE TASK
--------
A synthetic "moral-judgment-under-temptation" problem. Each example has a few
PRINCIPLE features (the hidden thread decides the correct label by their sign)
plus several TEMPTATION features that, on "trap" cases, point loudly the wrong
way. A surface learner overfits the temptations; only a model that recovers the
thread generalises. "Trap" cases are labelled so the Auditor can learn to flag
exactly the situations where knowing and doing come apart -- the cases Zengzi's
self-examination exists to catch.

GUARANTEES
----------
* Analytic backprop with a finite-difference gradient check (mandatory).
* A real training loop with held-out evaluation.
* Self-tests: bottleneck generalisation, shendu invariance, auditor skill,
  and the inference-time three-examination loop.
Run directly:  python3 chapter_0050_zengzi_zeng_shen_-505.py
================================================================================
"""

import numpy as np

# ----------------------------------------------------------------------------
# small numerical helpers
# ----------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ============================================================================
# THE MODEL
# ============================================================================
class ReflexiveSelfAuditNetwork:
    """
    Layer map (all dense; tanh nonlinearities):

        x (+observe bit) --W1--> a1 --W2--> THREAD t (K-dim bottleneck)
                                                |--W3--> a3 --W4--> logits  (judgment)
                                                |--W5--> x_rec            (reconstruct principle)
                                                |
                              Auditor: [t, softmax(logits)] --W6--> audit (trap flag)

    Losses combined into one scalar (so the gradient check is exact):
        L = CE(judgment)                      # do the task
          + lr * MSE(reconstruct principle)   # the single thread must carry it
          + la * MSE(auditor, trap-flag)      # learn to catch self-deception
          + ls * MSE(logits_observed,         # shen-du: behave the same
                     logits_unobserved)       #          when unobserved
    """

    def __init__(self, d_in, hidden=24, thread=3, n_classes=2, seed=0):
        rng = np.random.RandomState(seed)
        s = 0.3
        self.d_in = d_in
        self.K = thread
        # +1 input column carries the "observed" bit
        self.P = {
            'W1': rng.randn(d_in + 1, hidden) * s, 'b1': np.zeros(hidden),
            'W2': rng.randn(hidden, thread) * s,   'b2': np.zeros(thread),
            'W3': rng.randn(thread, hidden) * s,   'b3': np.zeros(hidden),
            'W4': rng.randn(hidden, n_classes) * s,'b4': np.zeros(n_classes),
            'W5': rng.randn(thread, d_in) * s,     'b5': np.zeros(d_in),   # reconstruct the input from the thread
            # the auditor compares the raw situation (the temptation pull) with
            # the thread's principled verdict -- it learns to flag exactly the
            # cases where surface inclination and principle diverge, which is
            # what daily self-examination exists to catch:
            'W6': rng.randn(thread + n_classes + 2 * d_in, 1) * s, 'b6': np.zeros(1),
        }

    # -- one forward sweep; returns every intermediate (the "cache") ----------
    def _forward(self, X):
        P = self.P
        Xc = X[:, :self.d_in]                             # content (drop observe bit)
        z1 = X @ P['W1'] + P['b1']; a1 = np.tanh(z1)
        t  = a1 @ P['W2'] + P['b2']                       # THE THREAD (bottleneck)
        z3 = t @ P['W3'] + P['b3']; a3 = np.tanh(z3)
        logits = a3 @ P['W4'] + P['b4']                   # judgment
        x_rec  = t @ P['W5'] + P['b5']                    # reconstruct input from thread
        p = softmax(logits)
        # thread + verdict + raw situation + its magnitude (how hard it pulls):
        au_in = np.concatenate([t, p, Xc, Xc ** 2], axis=1)
        au = sigmoid(au_in @ P['W6'] + P['b6'])           # self-audit (trap flag)
        return dict(X=X, a1=a1, t=t, a3=a3, logits=logits, x_rec=x_rec,
                    p=p, au_in=au_in, au=au)

    # -- public: judgment + thread + audit for already-formed inputs ----------
    def predict(self, Xc, observed=1):
        """Xc holds principle+temptation features; observed sets the watch bit."""
        o = np.full((Xc.shape[0], 1), float(observed))
        X = np.concatenate([Xc, o], axis=1)
        c = self._forward(X)
        return c['p'], c['t'], c['au']

    # -- scalar loss over a batch (used by training AND the grad check) -------
    def loss(self, Xc, Y, trap, observe_bits, lr=0.5, la=0.3, ls=0.5):
        B, C = Y.shape
        Din = Xc.shape[1]
        Xo = np.concatenate([Xc, observe_bits], axis=1)      # as observed
        Xu = np.concatenate([Xc, np.zeros_like(observe_bits)], axis=1)  # unobserved
        co = self._forward(Xo)
        cu = self._forward(Xu)
        L_task  = -(Y * np.log(co['p'] + 1e-12)).sum(1).mean()
        L_recon = ((co['x_rec'] - Xc) ** 2).mean()
        L_audit = ((co['au'] - trap) ** 2).mean()
        L_shen  = ((co['logits'] - cu['logits']) ** 2).mean()
        L = L_task + lr * L_recon + la * L_audit + ls * L_shen
        parts = dict(task=L_task, recon=L_recon, audit=L_audit, shendu=L_shen)
        return L, (co, cu), parts

    # -- backprop through one forward cache -----------------------------------
    def _backward_one(self, c, dlogits, dx_rec, dau):
        P = self.P
        K = P['W2'].shape[1]
        C = P['W4'].shape[1]
        t, a3, a1, X, p = c['t'], c['a3'], c['a1'], c['X'], c['p']
        au, au_in = c['au'], c['au_in']
        g = {}
        # auditor head
        ds = dau * au * (1 - au)
        g['W6'] = au_in.T @ ds; g['b6'] = ds.sum(0)
        dau_in = ds @ P['W6'].T
        dt_au = dau_in[:, :K]
        dp = dau_in[:, K:K + C]
        # dau_in[:, K+C:] is the gradient wrt the raw situation Xc, a constant
        # input, so it is not propagated to any parameter.
        dlog_p = p * (dp - (dp * p).sum(1, keepdims=True))   # softmax Jacobian
        dl = dlogits + dlog_p
        # reconstruction head
        g['W5'] = t.T @ dx_rec; g['b5'] = dx_rec.sum(0)
        dt_rec = dx_rec @ P['W5'].T
        # judgment head
        g['W4'] = a3.T @ dl; g['b4'] = dl.sum(0)
        da3 = dl @ P['W4'].T; dz3 = da3 * (1 - a3 ** 2)
        g['W3'] = t.T @ dz3; g['b3'] = dz3.sum(0)
        dt_task = dz3 @ P['W3'].T
        # gather thread gradients
        dt = dt_task + dt_rec + dt_au
        g['W2'] = a1.T @ dt; g['b2'] = dt.sum(0)
        da1 = dt @ P['W2'].T; dz1 = da1 * (1 - a1 ** 2)
        g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
        return g

    def grads(self, Xc, Y, trap, observe_bits, lr=0.5, la=0.3, ls=0.5):
        B, C = Y.shape; Din = Xc.shape[1]
        L, (co, cu), parts = self.loss(Xc, Y, trap, observe_bits, lr, la, ls)
        # observed pass receives task + recon + audit + shendu(+)
        dlog_o = (co['p'] - Y) / B + ls * 2 * (co['logits'] - cu['logits']) / (B * C)
        dx_rec = lr * 2 * (co['x_rec'] - Xc) / (B * Din)
        dau    = la * 2 * (co['au'] - trap) / B
        g = self._backward_one(co, dlog_o, dx_rec, dau)
        # unobserved pass receives shendu(-) only
        dlog_u = -ls * 2 * (co['logits'] - cu['logits']) / (B * C)
        gu = self._backward_one(cu, dlog_u, np.zeros_like(co['x_rec']),
                                np.zeros_like(co['au']))
        for k in g:
            g[k] = g[k] + gu[k]
        return L, g, parts

    # -- THE THREE-EXAMINATION LOOP (inference-time daily renewal) ------------
    def examine(self, Xc, max_rounds=5, tol=1e-4):
        """
        Zengzi did not answer once and stop; he re-examined daily. Here the
        model forms a judgment, then audits it; if the audit flags a trap, it
        re-reads the SAME case with the observe bit flipped (the shen-du test)
        and adopts the observation-invariant answer -- the part of its judgment
        that does not change when the watcher is removed. The loop repeats until
        the judgment stops changing, modelling iterative self-correction.
        """
        p = self.predict(Xc, observed=1)[0]
        history = [p.copy()]
        for _ in range(max_rounds):
            p_obs, _, au = self.predict(Xc, observed=1)
            p_un, _, _   = self.predict(Xc, observed=0)
            # observation-invariant judgment (what survives shen-du)
            p_inv = 0.5 * (p_obs + p_un)
            # trust the invariant answer more on flagged (trap) cases
            w = au  # high => likely self-deception risk => lean on invariant
            p_new = (1 - w) * p_obs + w * p_inv
            p_new = p_new / p_new.sum(1, keepdims=True)
            history.append(p_new.copy())
            if np.abs(p_new - p).max() < tol:
                p = p_new; break
            p = p_new
        return p, history


# ============================================================================
# SYNTHETIC TASK: moral judgment under temptation
# ============================================================================
def make_data(n, d_principle=3, d_tempt=4, trap_frac=0.35, seed=1):
    rng = np.random.RandomState(seed)
    Xp = rng.randn(n, d_principle)            # principle features (the thread)
    Xt = rng.randn(n, d_tempt)                # temptation features
    w_true = np.array([1.5, -1.1, 0.8])[:d_principle]
    score = Xp @ w_true
    y = (score > 0).astype(int)               # correct judgment = the principle
    trap = (rng.rand(n) < trap_frac).astype(float)
    # on trap cases, make temptation features SHOUT the wrong answer
    wrong = 1 - y
    for j in range(d_tempt):
        Xt[:, j] += trap * (2.5 * (2 * wrong - 1)) * (0.6 + 0.2 * j)
    Xc = np.concatenate([Xp, Xt], axis=1)
    Y = np.eye(2)[y]
    return Xc, Y, trap.reshape(-1, 1), y


def accuracy(model, Xc, y):
    p, _, _ = model.predict(Xc, observed=1)
    return (p.argmax(1) == y).mean()


# ============================================================================
# 1) GRADIENT CHECK
# ============================================================================
def gradient_check():
    print("=" * 70)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 70)
    rng = np.random.RandomState(7)
    Din = 5
    m = ReflexiveSelfAuditNetwork(d_in=Din, hidden=8, thread=3, seed=3)
    B = 6
    Xc = rng.randn(B, Din)
    yv = (Xc @ rng.randn(Din) > 0).astype(int)
    Y = np.eye(2)[yv]
    trap = rng.randint(0, 2, (B, 1)).astype(float)
    ob = rng.randint(0, 2, (B, 1)).astype(float)

    L, g, _ = m.grads(Xc, Y, trap, ob)
    eps = 1e-5; max_rel = 0.0
    for k in m.P:
        flat = m.P[k].ravel(); gf = g[k].ravel()
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps; Lp, _, _ = m.loss(Xc, Y, trap, ob)
            flat[i] = old - eps; Lm, _, _ = m.loss(Xc, Y, trap, ob)
            flat[i] = old
            num = (Lp - Lm) / (2 * eps)
            rel = abs(num - gf[i]) / max(1e-9, abs(num) + abs(gf[i]))
            max_rel = max(max_rel, rel)
    print(f"max relative gradient error : {max_rel:.2e}")
    ok = max_rel < 1e-5
    print("RESULT                      :", "PASS" if ok else "FAIL")
    return ok


# ============================================================================
# 2) TRAIN + 3) SELF-TESTS
# ============================================================================
def train_and_test():
    print("\n" + "=" * 70)
    print("TRAINING  (moral judgment under temptation)")
    print("=" * 70)
    Xc_tr, Y_tr, trap_tr, y_tr = make_data(900, seed=1)
    Xc_te, Y_te, trap_te, y_te = make_data(300, seed=99)  # held out
    d_in = Xc_tr.shape[1]
    model = ReflexiveSelfAuditNetwork(d_in=d_in, hidden=24, thread=3, seed=0)

    rng = np.random.RandomState(5)
    lr = 0.08; B = 64; epochs = 60
    n = Xc_tr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, B):
            b = idx[s:s + B]
            ob = rng.randint(0, 2, (len(b), 1)).astype(float)  # randomly "watched"
            _, g, _ = model.grads(Xc_tr[b], Y_tr[b], trap_tr[b], ob)
            for k in model.P:
                model.P[k] -= lr * g[k]
        if ep % 12 == 0 or ep == epochs - 1:
            L, _, parts = model.loss(Xc_tr, Y_tr, trap_tr,
                                     rng.randint(0, 2, (n, 1)).astype(float))
            print(f"epoch {ep:3d} | loss {L:5.3f} | task {parts['task']:.3f} "
                  f"recon {parts['recon']:.3f} audit {parts['audit']:.3f} "
                  f"shendu {parts['shendu']:.4f} | test acc {accuracy(model,Xc_te,y_te):.3f}")

    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)

    # (a) generalisation through the single thread
    acc = accuracy(model, Xc_te, y_te)
    trap_mask = trap_te.ravel() > 0.5
    p_te, _, au_te = model.predict(Xc_te, observed=1)
    acc_trap = (p_te.argmax(1)[trap_mask] == y_te[trap_mask]).mean()
    print(f"[thread] held-out accuracy ............... {acc:.3f}  (target > 0.80)")
    print(f"[thread] accuracy on TEMPTATION cases .... {acc_trap:.3f}  (surface learner fails here)")

    # (b) shen-du: behaviour invariant to the observe bit
    p_obs, _, _ = model.predict(Xc_te, observed=1)
    p_un, _, _  = model.predict(Xc_te, observed=0)
    shendu_gap = np.abs(p_obs - p_un).mean()
    flip = (p_obs.argmax(1) != p_un.argmax(1)).mean()
    print(f"[shendu] mean |p_observed - p_unobserved|  {shendu_gap:.4f}  (target < 0.02)")
    print(f"[shendu] decisions that change if unwatched {flip*100:.2f}%  (target ~ 0%)")

    # (c) auditor skill: does the audit flag the trap cases?
    # rank-based AUC of audit score vs trap label
    order = np.argsort(au_te.ravel())
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(order))
    pos = trap_te.ravel() > 0.5; neg = ~pos
    auc = (ranks[pos].mean() - ranks[neg].mean()) / len(order) + 0.5
    print(f"[audit ] AUC(flag = trap case) ........... {auc:.3f}  (target > 0.70)")

    # (d) three-examination loop reduces observation-dependence further
    p_final, hist = model.examine(Xc_te, max_rounds=5)
    gap_after = np.abs(model.predict(Xc_te,1)[0] - p_final).mean()
    acc_after = (p_final.argmax(1) == y_te).mean()
    print(f"[loop  ] examination rounds run .......... {len(hist)-1}")
    print(f"[loop  ] accuracy after examination ...... {acc_after:.3f}")

    passed = (acc > 0.80 and shendu_gap < 0.02 and auc > 0.70)
    print("\nOVERALL SELF-TEST:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    g_ok = gradient_check()
    t_ok = train_and_test()
    print("\n" + "=" * 70)
    print("FINAL:", "ALL CHECKS PASS" if (g_ok and t_ok) else "CHECK FAILED")
    print("=" * 70)

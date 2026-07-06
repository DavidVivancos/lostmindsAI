#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0040_confucius_kong_qiu_-551.py  --  The Li-Zhengming Network (LZN)
A from-scratch, pure-NumPy cognitive architecture that embodies the mind of
                     Confucius  (Kong Qiu, 551-479 BCE)
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0040 · Confucius (Kong Qiu)
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
--------------------------------------------------------
Most "build an AGI" sketches reach for attention over stored keys. That choice
encodes a particular cognitive thesis: intelligence = retrieval + weighting of
remembered content. Confucius held almost the opposite thesis, and this file is
faithful to *his* idea rather than to the convenient default.

Confucius's signature claim about the mind is directional and unusual:

    The heart-mind (xin) is shaped from the OUTSIDE IN.
    You do not first feel virtue and then act; you perform the rite (li)
    correctly, again and again, inside a NAMED relationship, and the repeated
    outward form slowly cultivates the inward disposition until -- "at seventy"
    -- right conduct flows without effort (Analects 2.4).

Three of his doctrines turn directly into mechanism here:

  1. LI (ritual propriety) as ROLE-CONDITIONED transformation.
     A situation is never processed "neutrally." It is processed *as* a move
     inside one of the Five Relationships (wulun): ruler-subject, parent-child,
     husband-wife, elder-younger, friend-friend. The named role gates how the
     situation is transformed. We implement this as FiLM modulation: the role
     supplies a (gamma, beta) that scales and shifts the hidden state. Different
     name -> different ritual -> different proper conduct.

  2. ZHENGMING (rectification of names) as a PROTOTYPE-ALIGNMENT constraint.
     "If names are not correct, language is not in accordance with the truth of
     things; if language is not in accordance with the truth of things, affairs
     cannot succeed" (Analects 13.3). Each role owns a learned prototype in
     conduct-space. The conduct the network produces must land nearest its TRUE
     role's prototype. Crucially, the network is often fed a *wrong* name
     (role_claimed != role_true) and must still produce the conduct the
     situation truly warrants -- i.e. it must see past the corrupted label to
     the real relationship. That is rectification of names as a cognitive act.

  3. SELF-EXAMINATION (xing) as a metacognitive consistency monitor.
     "Each day I examine myself on several counts" (Analects 1.4, Master Zeng).
     A small head judges whether the NAME it was handed actually matches the
     relationship the situation reveals. It is the conscience that says "this is
     not what a 'father' does," and flags the misnaming for repair.

  4. CULTIVATION (the 15->70 arc) as a TWO-TIMESCALE weight dynamic.
     Fast weights learn each lesson; a slow "disposition" copy is an EMA of the
     fast weights. Early on the two diverge (the impulsive learner); with enough
     ritual repetition they converge, and the slow, cultivated weights -- the
     settled character -- generalize at least as well as the fast ones. This is
     "the cultivated self that can follow the heart's desire without
     transgressing what is right." The EMA is deliberately OUTSIDE the gradient
     so it models settling, not optimization.

WHAT THE FILE CONTAINS
----------------------
  * A synthetic "social propriety" world built from the Five Relationships.
  * The LZN forward pass (role-gated ritual encoder + conduct head +
    name-prototypes + self-examination head).
  * Hand-derived analytic backprop for EVERY parameter.
  * A finite-difference GRADIENT CHECK (mandatory; must pass).
  * A real Adam training loop with the cultivation EMA.
  * Self-tests: loss decreases, rectification & exam accuracy beat chance,
    cultivated (slow) weights generalize, and a qualitative "rectify the name"
    demonstration on deliberately mislabeled inputs.

Run:   python3 chapter_0040_confucius_kong_qiu_-551.py
Deps:  numpy only.
================================================================================
"""

import numpy as np

# Five Relationships (wulun). The whole world is organized by these names.
WULUN = ["ruler-subject", "parent-child", "husband-wife",
         "elder-younger", "friend-friend"]


# ----------------------------------------------------------------------------
# 1. THE WORLD:  a synthetic society of named relationships
# ----------------------------------------------------------------------------
# A "situation" is a vector drawn from one role's region of social space
# (role_true). The proper conduct it warrants lives near that role's conduct
# prototype, with a small situation-dependent variation (the rite is not rote;
# it bends to circumstance). Society then hands the agent a NAME (role_claimed)
# which is correct half the time and corrupted half the time. The agent must:
#   (a) produce the conduct the situation TRULY warrants  -> task regression
#   (b) cluster that conduct to the TRUE role's prototype -> rectification
#   (c) flag whether the handed name matched reality      -> self-examination
class SocietyWorld:
    def __init__(self, d_in=16, d_c=8, n_roles=5, seed=0):
        self.d_in, self.d_c, self.R = d_in, d_c, n_roles
        rng = np.random.RandomState(seed)
        # Each role's home region in situation-space and its conduct prototype.
        self.S_center = rng.randn(n_roles, d_in) * 1.4
        self.mu = rng.randn(n_roles, d_c) * 1.4
        # A fixed map giving conduct a mild, lawful dependence on the situation
        # (so the rite responds to circumstance rather than being a constant).
        self.A = rng.randn(d_in, d_c) * 0.12

    def sample(self, n, seed=None, p_correct_name=0.5):
        rng = np.random.RandomState(seed)
        role_true = rng.randint(0, self.R, size=n)
        X = self.S_center[role_true] + rng.randn(n, self.d_in) * 0.6
        # proper conduct = true role's prototype + small situational variation
        c_tgt = self.mu[role_true] + (X - self.S_center[role_true]) @ self.A
        # the NAME society applies: correct, or a corrupted (different) name
        role_claimed = role_true.copy()
        corrupt = rng.rand(n) > p_correct_name
        for i in np.where(corrupt)[0]:
            choices = [r for r in range(self.R) if r != role_true[i]]
            role_claimed[i] = rng.choice(choices)
        proper = (role_claimed == role_true).astype(np.float64)  # 1 = name fits
        R_onehot = np.eye(self.R)[role_claimed]
        return dict(X=X, R_onehot=R_onehot, role_claimed=role_claimed,
                    role_true=role_true, c_tgt=c_tgt, proper=proper)


# ----------------------------------------------------------------------------
# 2. THE LI-ZHENGMING NETWORK
# ----------------------------------------------------------------------------
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))  # numerically stable


class LZN:
    """
    Forward graph (batch N), all weights stored as (in, out):

        a1 = X @ W1 + b1 ;  h1 = tanh(a1)                 # perceive the scene
        gamma = 1 + R_onehot @ Wg                          # LI: the named role
        beta  =     R_onehot @ Wb                          #     gates the rite
        a2 = h1 @ W2 + b2 ; h2 = tanh(a2)
        hm = gamma * h2 + beta                             # ritualized state
        c  = hm @ Wc + bc                                  # conduct (linear)
        e  = hm @ We + be                                  # self-exam logit
        P  : (R, d_c) name-prototypes (zhengming)

    Losses:
        L_task = MSE(c, c_tgt)
        L_rect = cross-entropy( softmax(-||c - P||^2), role_true )
        L_exam = BCE( sigmoid(e), proper )
        L = L_task + lam_rect*L_rect + lam_exam*L_exam
    """

    def __init__(self, d_in, d_c, H=24, R=5, lam_rect=1.0, lam_exam=0.5, seed=1):
        rng = np.random.RandomState(seed)
        s = lambda a, b: rng.randn(a, b) * np.sqrt(2.0 / a)  # He-ish init
        self.p = {
            "W1": s(d_in, H), "b1": np.zeros(H),
            "Wg": rng.randn(R, H) * 0.05,         # role -> scale (start near 1)
            "Wb": rng.randn(R, H) * 0.05,         # role -> shift (start near 0)
            "W2": s(H, H),     "b2": np.zeros(H),
            "Wc": s(H, d_c),   "bc": np.zeros(d_c),
            "We": s(H, 1),     "be": np.zeros(1),
            "P":  rng.randn(R, d_c) * 0.5,        # name-prototypes
        }
        self.H, self.R, self.d_c = H, R, d_c
        self.lam_rect, self.lam_exam = lam_rect, lam_exam
        # cultivated "disposition" copy (slow EMA weights); filled during train
        self.slow = {k: v.copy() for k, v in self.p.items()}

    # --- forward; returns loss, cache, and per-loss breakdown -------------
    def forward(self, batch, params=None):
        p = self.p if params is None else params
        X, Ro = batch["X"], batch["R_onehot"]
        c_tgt, role_true, proper = batch["c_tgt"], batch["role_true"], batch["proper"]
        N = X.shape[0]

        a1 = X @ p["W1"] + p["b1"];          h1 = tanh(a1)
        gamma = 1.0 + Ro @ p["Wg"]
        beta = Ro @ p["Wb"]
        a2 = h1 @ p["W2"] + p["b2"];         h2 = tanh(a2)
        hm = gamma * h2 + beta
        c = hm @ p["Wc"] + p["bc"]
        e = hm @ p["We"] + p["be"]            # (N,1)

        # --- L_task: did we produce the conduct the situation warranted? ---
        diff = c - c_tgt
        L_task = np.mean(diff * diff)

        # --- L_rect: does conduct land nearest its TRUE name's prototype? --
        # dist[n,r] = ||c[n]-P[r]||^2 ;  logits = -dist
        d2 = ((c[:, None, :] - p["P"][None, :, :]) ** 2).sum(-1)   # (N,R)
        logits = -d2
        logits -= logits.max(1, keepdims=True)
        ex = np.exp(logits)
        sm = ex / ex.sum(1, keepdims=True)                          # (N,R)
        L_rect = np.mean(-np.log(sm[np.arange(N), role_true] + 1e-12))

        # --- L_exam: does the handed NAME match the revealed relationship? -
        pe = sigmoid(e[:, 0])                                       # (N,)
        L_exam = np.mean(-(proper * np.log(pe + 1e-12) +
                           (1 - proper) * np.log(1 - pe + 1e-12)))

        L = L_task + self.lam_rect * L_rect + self.lam_exam * L_exam
        cache = dict(X=X, Ro=Ro, h1=h1, h2=h2, gamma=gamma, hm=hm, c=c,
                     diff=diff, sm=sm, pe=pe, role_true=role_true,
                     proper=proper, N=N)
        return L, cache, dict(task=L_task, rect=L_rect, exam=L_exam)

    # --- analytic backprop for every parameter ----------------------------
    def backward(self, cache, params=None):
        p = self.p if params is None else params
        N = cache["N"]
        X, Ro = cache["X"], cache["Ro"]
        h1, h2, gamma, hm = cache["h1"], cache["h2"], cache["gamma"], cache["hm"]
        c, diff, sm = cache["c"], cache["diff"], cache["sm"]
        pe, role_true, proper = cache["pe"], cache["role_true"], cache["proper"]
        g = {}

        # dL_task/dc
        dc = (2.0 / (N * self.d_c)) * diff                      # (N,d_c)

        # dL_rect/dc and dL_rect/dP
        # gg[n,r] = (sm - onehot(role_true))/N  is dL_rect/dlogits
        gg = sm.copy()
        gg[np.arange(N), role_true] -= 1.0
        gg /= N                                                 # (N,R)
        # logits = -dist ; dist = ||c-P||^2
        # dL/dc[n] = sum_r gg[n,r]*(-d dist/dc) = sum_r gg[n,r]*(-2(c[n]-P[r]))
        # dL/dP[r] = sum_n gg[n,r]*( 2(c[n]-P[r]) )
        cmP = c[:, None, :] - p["P"][None, :, :]                # (N,R,d_c)
        dc_rect = (-2.0 * gg[:, :, None] * cmP).sum(1)          # (N,d_c)
        dP = (2.0 * gg[:, :, None] * cmP).sum(0)                # (R,d_c)
        dc = dc + self.lam_rect * dc_rect
        g["P"] = self.lam_rect * dP

        # conduct head
        g["Wc"] = hm.T @ dc
        g["bc"] = dc.sum(0)
        dhm = dc @ p["Wc"].T                                    # (N,H)

        # self-exam head
        de = (self.lam_exam * (pe - proper) / N)[:, None]       # (N,1)
        g["We"] = hm.T @ de
        g["be"] = de.sum(0)
        dhm = dhm + de @ p["We"].T

        # hm = gamma*h2 + beta
        dgamma = dhm * h2
        dbeta = dhm
        dh2 = dhm * gamma
        g["Wg"] = Ro.T @ dgamma
        g["Wb"] = Ro.T @ dbeta

        # through tanh layer 2
        da2 = dh2 * (1.0 - h2 * h2)
        g["W2"] = h1.T @ da2
        g["b2"] = da2.sum(0)
        dh1 = da2 @ p["W2"].T

        # through tanh layer 1
        da1 = dh1 * (1.0 - h1 * h1)
        g["W1"] = X.T @ da1
        g["b1"] = da1.sum(0)
        return g


# ----------------------------------------------------------------------------
# 3. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ----------------------------------------------------------------------------
def gradient_check(seed=0):
    world = SocietyWorld(d_in=10, d_c=6, n_roles=5, seed=seed)
    batch = world.sample(24, seed=7)
    net = LZN(d_in=10, d_c=6, H=12, R=5, seed=3)

    L0, cache, _ = net.forward(batch)
    grads = net.backward(cache)

    eps = 1e-6
    worst = 0.0
    report = []
    for name in net.p:
        P = net.p[name]
        ga = grads[name]
        # sample a few entries per parameter tensor
        rng = np.random.RandomState(hash(name) % 2**31)
        idxs = [tuple(rng.randint(0, s) for s in P.shape) for _ in range(6)]
        for idx in idxs:
            old = P[idx]
            P[idx] = old + eps
            Lp, _, _ = net.forward(batch)
            P[idx] = old - eps
            Lm, _, _ = net.forward(batch)
            P[idx] = old
            num = (Lp - Lm) / (2 * eps)
            ana = ga[idx]
            rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
            worst = max(worst, rel)
        report.append((name, worst))
    return worst


# ----------------------------------------------------------------------------
# 4. ADAM OPTIMIZER (pure NumPy) + cultivation EMA
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(net, batch, params):
    """Return rectification accuracy, exam accuracy, conduct RMSE."""
    L, cache, parts = net.forward(batch, params=params)
    c, P = cache["c"], params["P"]
    d2 = ((c[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    rect_pred = d2.argmin(1)
    rect_acc = (rect_pred == batch["role_true"]).mean()
    exam_acc = ((cache["pe"] > 0.5).astype(float) == batch["proper"]).mean()
    rmse = np.sqrt(np.mean((c - batch["c_tgt"]) ** 2))
    return rect_acc, exam_acc, rmse, parts


def train():
    print("=" * 74)
    print(" Li-Zhengming Network  --  cultivating the heart-mind of Confucius")
    print("=" * 74)

    world = SocietyWorld(d_in=16, d_c=8, n_roles=5, seed=0)
    net = LZN(d_in=16, d_c=8, H=32, R=5, lam_rect=1.0, lam_exam=0.5, seed=1)
    opt = Adam(net.p, lr=3e-3)

    test = world.sample(600, seed=999)
    ema = 0.02  # cultivation rate: slow disposition tracks fast weights

    print("\n  step |  L_total  L_task  L_rect  L_exam | rect_acc exam_acc  rmse")
    print("  " + "-" * 66)
    for step in range(1, 1601):
        batch = world.sample(128, seed=step)
        L, cache, parts = net.forward(batch)
        grads = net.backward(cache)
        opt.step(net.p, grads)
        # CULTIVATION: settle fast lessons into slow disposition (no gradient)
        for k in net.p:
            net.slow[k] = (1 - ema) * net.slow[k] + ema * net.p[k]
        if step % 200 == 0 or step == 1:
            ra, ea, rmse, _ = evaluate(net, test, net.p)
            print("  %4d | %8.4f %7.4f %7.4f %7.4f |  %.3f    %.3f   %.3f"
                  % (step, L, parts["task"], parts["rect"], parts["exam"],
                     ra, ea, rmse))
    return net, world, test


# ----------------------------------------------------------------------------
# 5. SELF-TESTS
# ----------------------------------------------------------------------------
def self_tests(net, world, test):
    print("\n" + "=" * 74)
    print(" SELF-TESTS")
    print("=" * 74)
    ok = True

    # (A) fast vs cultivated (slow) generalization on held-out society
    ra_f, ea_f, rmse_f, _ = evaluate(net, test, net.p)
    ra_s, ea_s, rmse_s, _ = evaluate(net, test, net.slow)
    print("\n  [A] Generalization on 600 unseen situations")
    print("      fast (impulsive)  : rect_acc=%.3f exam_acc=%.3f rmse=%.3f"
          % (ra_f, ea_f, rmse_f))
    print("      slow (cultivated) : rect_acc=%.3f exam_acc=%.3f rmse=%.3f"
          % (ra_s, ea_s, rmse_s))
    chance = 1.0 / world.R
    test_rect = ra_s > 0.6 and ra_f > 0.6
    test_exam = ea_s > 0.75
    # cultivated disposition should be no worse than fast on conduct fidelity
    test_cult = rmse_s <= rmse_f + 0.05
    print("      rectification beats chance (%.2f)?  %s" % (chance, test_rect))
    print("      self-exam > 0.75?                   %s" % test_exam)
    print("      cultivated rmse <= fast rmse?       %s" % test_cult)
    ok = ok and test_rect and test_exam and test_cult

    # (B) "Rectify the name": feed deliberately MISNAMED inputs and check the
    #     network still recovers the TRUE warranted conduct + flags the lie.
    print("\n  [B] Rectification of names on 300 deliberately MISNAMED cases")
    mis = world.sample(300, seed=1234, p_correct_name=0.0)  # every name wrong
    L, cache, _ = net.forward(mis, params=net.slow)
    c = cache["c"]
    d2 = ((c[:, None, :] - net.slow["P"][None, :, :]) ** 2).sum(-1)
    recovered = (d2.argmin(1) == mis["role_true"]).mean()
    flagged = (cache["pe"] < 0.5).mean()   # should say "name does NOT fit"
    print("      recovered the TRUE relationship despite wrong name: %.3f"
          % recovered)
    print("      correctly flagged the name as improper:            %.3f"
          % flagged)
    test_recover = recovered > 0.6
    test_flag = flagged > 0.6
    print("      recovered > 0.6?  %s     flagged > 0.6?  %s"
          % (test_recover, test_flag))
    ok = ok and test_recover and test_flag

    # (C) prototypes are distinct (names carved apart, not collapsed)
    P = net.slow["P"]
    pd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    min_off = pd[~np.eye(world.R, dtype=bool)].min()
    print("\n  [C] Name-prototype separation (min pairwise dist): %.3f" % min_off)
    test_sep = min_off > 0.3
    print("      prototypes distinct (>0.3)?  %s" % test_sep)
    ok = ok and test_sep

    return ok


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    print("\n>>> STEP 1/3  Gradient check (must pass before any training)")
    worst = gradient_check()
    grad_ok = worst < 1e-4
    print("    worst relative error across all parameters: %.2e" % worst)
    print("    GRADIENT CHECK %s" % ("PASSED" if grad_ok else "FAILED"))
    assert grad_ok, "Gradient check failed -- backprop is wrong, refusing to train."

    print("\n>>> STEP 2/3  Training")
    net, world, test = train()

    print("\n>>> STEP 3/3  Self-tests")
    passed = self_tests(net, world, test)

    print("\n" + "=" * 74)
    print(" RESULT: %s" % ("ALL CHECKS PASSED" if (grad_ok and passed)
                           else "SOME CHECKS FAILED"))
    print("=" * 74)
    print("""
 Reading of the run, in Confucian terms:
   - The gradient check is the craftsman's plumb-line: the mechanism is sound.
   - L_rect falling is names being rectified -- conduct settling onto the
     prototype its true relationship demands.
   - The self-exam head learning to flag misnamed cases is the daily xing,
     the conscience that says 'this is not what a father does.'
   - The cultivated (slow) weights matching or beating the impulsive fast
     weights is the long arc of li internalized: at seventy, the heart may
     follow its desire without transgressing what is right.
""")

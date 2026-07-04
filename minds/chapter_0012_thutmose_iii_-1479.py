#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0012_thutmose_iii_-1479.py  --  THE ARUNA ENGINE
 A Counter-Expectation Routing Network in the cognitive signature of
 THUTMOSE III  (Menkheperre), c. 1479-1425 BCE.
 
 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
The default move for a "military strategist" figure is a Transformer that
attends over stored keys, or a deception-GAN that manufactures false signals
in the style of Sun Tzu. Both miss what is *specific* to Thutmose III.

The single best-documented cognitive act of his life is the council of war at
Yehem before the Battle of Megiddo (Year 23, c. 1457 BCE). His generals, using
sound intelligence, advised the two wide, safe roads (Zefti and Taanach) and
warned that the narrow Aruna pass -- where the army had to march "horse behind
horse, man behind man" -- invited annihilation. Thutmose took Aruna. His
recorded reasoning (Annals, from the field journal of the scribe Tjaneni) is
*not* deception: he plants no false signal. He reasons that the enemy, being
competent, will think exactly as his own competent generals think -- and will
therefore mass its defenses on the obvious roads. The narrow road is undefended
PRECISELY BECAUSE everyone rational has eliminated it. The enemy's blind spot is
manufactured by the enemy's own correctness. Thutmose simply reads that shared
expectation and inverts it. The Kadesh coalition had indeed left Aruna almost
unguarded.

So the cognitive signature we must encode is:
    "Take the option the adversary's *correct* reasoning has eliminated --
     when, and only when, that option is good enough to survive on its merits."

And the second documented lesson -- discipline. After the field rout, Thutmose's
troops broke formation to loot the enemy camp instead of storming the city; the
quick victory curdled into a seven-month siege. The lesson he drew, and which
his later campaigns embody, is that audacity must be *paid for* by genuine
value, never indulged for its own sake. He then consolidated by encirclement
(a wall: "Menkheperre-is-the-encircler-of-the-Asiatics"), by RECORD (Tjaneni's
daily journal, abstracted onto the Karnak Annals), and by SEASONAL RETURN
(~17 campaigns over ~20 years). One gamble became a repeatable method.

THE NETWORK MAKES THIS LITERAL. It is a small, white-box, pure-NumPy model
with two heads and an explicit "Aruna fusion":

    expectation head  e(x) -> p_exp  : the adversary's (and your own staff's)
                                       predicted distribution over which road
                                       YOU will take.  We only READ it.  We
                                       never fabricate it -- that is the line
                                       between Thutmose and Sun Tzu.
    value head        v(x)           : the sober assessment of each road on its
                                       own merits (the generals' real terrain
                                       intelligence).
    Aruna fusion      r_hat = v - kappa * p_exp
                                       realized payoff = merit minus the enemy's
                                       defenses, which concentrate where you are
                                       EXPECTED.  kappa = learned, disciplined
                                       audacity.
    discipline gate   penalizes the policy for putting weight on roads whose
                                       intrinsic value is below a floor -- the
                                       Megiddo loot-lesson: surprise is allowed,
                                       recklessness is not.
    Tjaneni journal   an external replay memory; training samples from the log
                                       of past episodes, so the model "returns to
                                       its records" rather than learning only
                                       from the newest campaign.

Everything is hand-derived backprop verified by a central finite-difference
gradient check (mandatory). A real training loop on synthetic "Megiddo routing"
episodes is run, and self-tests show the engine learns to choose the Aruna road
that a value-only commander would reject -- and that this earns more realized
payoff. Execute the file; the verified output is pasted into the chapter.

Pure NumPy. No torch, no tensorflow, no keras. From scratch.
Author: David Vivancos  --  Mind #12, Thutmose III.
================================================================================
"""

import numpy as np

# ==============================================================================
# SECTION 0  --  REPRODUCIBILITY & SMOOTH PRIMITIVES
# ------------------------------------------------------------------------------
# Every op below is C^1 (continuously differentiable) so the finite-difference
# gradient check will agree with the analytic gradients to ~1e-6.
# ==============================================================================

def tanh(z):
    return np.tanh(z)

def dtanh(a):
    # derivative given the ACTIVATION a = tanh(z):  d/dz tanh = 1 - tanh^2
    return 1.0 - a * a

def softplus(z):
    # numerically stable softplus = log(1 + e^z)
    return np.logaddexp(0.0, z)

def sigmoid(z):
    # = d/dz softplus(z); also the smooth gate used by the discipline term
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ==============================================================================
# SECTION 1  --  THE MEGIDDO ROUTING WORLD (synthetic, but faithful)
# ------------------------------------------------------------------------------
# Each episode is a council of war: K candidate roads, each described by 5
# features. A road's *realized* payoff is its sober merit MINUS the defenses the
# adversary places on it -- and the adversary defends in proportion to how much
# it EXPECTS you to take that road. The narrow-direct-exposed road (the "Aruna"
# road) has only moderate intrinsic merit but is expected by no one, so it is
# barely defended, so its realized payoff is high. That is the whole lesson,
# expressed as data the network must learn to read.
# ==============================================================================

FEAT_NAMES = ["width", "directness", "exposure", "distance", "supply"]
D_FEATS = 5

# adversary "defense budget": how hard the enemy fortifies the roads it expects.
# Chosen so the inversion genuinely flips the optimum (see chapter, Architecture).
DEFENSE_BUDGET = 1.5
ADV_TEMP = 0.45        # how sharply the enemy concentrates on the obvious roads


def _intrinsic_value(x):
    """Sober merit of a road on its own terms (the generals' real assessment).
    Crucially -- as Thutmose's own generals judged -- a narrow, EXPOSED pass
    is poor merit even when it is direct: ambush risk dominates. So exposure
    carries a heavy penalty. The Aruna road therefore has only MODERATE merit
    (lower than the wide, safe roads); it can win *only* through the expectation
    inversion, never on its merits. That is the whole point of the gamble.
    (This function is data, not a network op, so it may use any form.)"""
    width, direct, exposure, distance, supply = x
    lin = (0.60 * direct + 0.50 * supply + 0.50 * width
           - 0.40 * distance - 0.55 * exposure)
    nonlin = 0.10 * np.tanh(2.0 * (supply - distance))   # mild logistics term
    return lin + nonlin


def _adversary_expectation(values_naive):
    """The enemy predicts you will behave like a *sensible* commander: avoid
    exposure, prefer width and supply. It is competent, not omniscient -- and
    that competence is exactly the exploitable thing."""
    return softmax(values_naive / ADV_TEMP)


def make_episode(rng, K=None, canonical=False):
    """Return one council-of-war episode.

    canonical=True builds the historical 3-road Megiddo choice by hand:
        road 0 = Zefti  (north): wide, safe, indirect  -> high merit, EXPECTED
        road 1 = Taanach(south): wide, safe, indirect  -> high merit, EXPECTED
        road 2 = Aruna  (centre): narrow, direct, exposed -> moderate merit,
                                                              UNEXPECTED.
    """
    if canonical:
        X = np.array([
            # width direct expo  dist supply
            [0.90, 0.30, 0.10, 0.80, 0.85],   # Zefti  (wide, safe, long way round)
            [0.85, 0.35, 0.15, 0.75, 0.80],   # Taanach(wide, safe, long way round)
            [0.10, 0.95, 0.90, 0.20, 0.45],   # Aruna  (narrow, direct, exposed)
        ], dtype=np.float64)
        K = 3
    else:
        if K is None:
            K = rng.integers(3, 7)
        X = rng.random((K, D_FEATS))
        # guarantee at least one "Aruna-like" road exists in most episodes so the
        # lesson is learnable: narrow (low width), direct, exposed, short.
        if rng.random() < 0.85:
            j = rng.integers(0, K)
            X[j] = np.array([rng.uniform(0.0, 0.20),   # narrow
                             rng.uniform(0.80, 1.0),    # direct
                             rng.uniform(0.75, 1.0),    # exposed
                             rng.uniform(0.0, 0.30),    # short
                             rng.uniform(0.30, 0.60)])  # mediocre supply

    v_true = np.array([_intrinsic_value(x) for x in X])

    # the enemy's NAIVE value (its model of how a sensible you chooses):
    naive = np.array([(1.0 * x[0] + 0.8 * x[4] - 1.4 * x[2] - 0.3 * x[3])
                      for x in X])
    p_adv = _adversary_expectation(naive)

    # defenses concentrate where you are expected; realized = merit - defense
    realized = v_true - DEFENSE_BUDGET * p_adv
    return X, v_true, p_adv, realized


# ==============================================================================
# SECTION 2  --  PARAMETERS
# ------------------------------------------------------------------------------
# Two 2-layer MLP heads (value, expectation) sharing the same input features but
# NOT sharing weights -- the sober assessor and the enemy-modeller are distinct
# faculties. Plus one scalar kappa_raw -> kappa = softplus(kappa_raw) >= 0, the
# disciplined audacity coefficient.
# ==============================================================================

H_HIDDEN = 8

def init_params(rng):
    def lin(fan_in, fan_out):
        s = np.sqrt(2.0 / (fan_in + fan_out))     # Glorot-ish
        return rng.standard_normal((fan_in, fan_out)) * s
    p = {
        "Wv1": lin(D_FEATS, H_HIDDEN), "bv1": np.zeros(H_HIDDEN),
        "Wv2": lin(H_HIDDEN, 1),       "bv2": np.zeros(1),
        "We1": lin(D_FEATS, H_HIDDEN), "be1": np.zeros(H_HIDDEN),
        "We2": lin(H_HIDDEN, 1),       "be2": np.zeros(1),
        "kappa_raw": np.array([0.0]),   # softplus(0) ~ 0.69 -> starts cautious
    }
    return p

PARAM_KEYS = ["Wv1","bv1","Wv2","bv2","We1","be1","We2","be2","kappa_raw"]


# loss weights and discipline hyper-parameters
W_REG   = 1.0     # fit realized payoff (the thing that actually matters)
W_VAL   = 0.5     # auxiliary: value head should track sober merit
W_EXP   = 0.5     # auxiliary: expectation head should track the enemy's model
W_DISC  = 0.30    # the Megiddo loot-lesson: penalize reckless surprise
V_FLOOR = 0.10    # discipline floor: roads below this merit are "reckless"
POLICY_T = 0.25   # temperature of the policy used only by the discipline term


# ==============================================================================
# SECTION 3  --  FORWARD PASS  (per episode)
# ------------------------------------------------------------------------------
# Returns predictions and a cache for the backward pass.
# ==============================================================================

def forward(p, X):
    # ---- value head (sober assessment) ----
    Zv1 = X @ p["Wv1"] + p["bv1"]          # (K,H)
    Av1 = tanh(Zv1)                        # (K,H)
    v   = (Av1 @ p["Wv2"] + p["bv2"])[:, 0]  # (K,)

    # ---- expectation head (the enemy's model of you; READ ONLY) ----
    Ze1 = X @ p["We1"] + p["be1"]          # (K,H)
    Ae1 = tanh(Ze1)                        # (K,H)
    s   = (Ae1 @ p["We2"] + p["be2"])[:, 0]  # (K,) logits
    p_exp = softmax(s)                     # (K,)

    # ---- Aruna fusion: realized = merit - audacity * expectation ----
    kappa = softplus(p["kappa_raw"])[0]    # scalar >= 0
    r_hat = v - kappa * p_exp              # (K,)

    cache = dict(X=X, Zv1=Zv1, Av1=Av1, v=v,
                 Ze1=Ze1, Ae1=Ae1, s=s, p_exp=p_exp,
                 kappa=kappa, r_hat=r_hat)
    return cache


# ==============================================================================
# SECTION 4  --  LOSS  (per episode)
# ------------------------------------------------------------------------------
#   L_reg  = mean (r_hat - realized)^2          regression onto realized payoff
#   L_val  = mean (v - v_true)^2                aux: sober merit
#   L_exp  = - sum p_adv * log p_exp            aux: model the enemy's model
#   L_disc = sum pi * softplus(V_FLOOR - v)     discipline (pi = softmax(r_hat/T))
# ==============================================================================

def loss_and_grad_episode(p, X, v_true, p_adv, realized):
    c = forward(p, X)
    K = X.shape[0]
    v, p_exp, r_hat, kappa = c["v"], c["p_exp"], c["r_hat"], c["kappa"]

    # ---------- forward losses ----------
    diff_r = r_hat - realized
    L_reg  = np.mean(diff_r ** 2)

    diff_v = v - v_true
    L_val  = np.mean(diff_v ** 2)

    L_exp  = -np.sum(p_adv * np.log(p_exp + 1e-12))

    u   = r_hat / POLICY_T
    pi  = softmax(u)
    g   = softplus(V_FLOOR - v)              # large when v << floor (reckless)
    gbar = np.dot(pi, g)
    L_disc = np.dot(pi, g)

    L = W_REG * L_reg + W_VAL * L_val + W_EXP * L_exp + W_DISC * L_disc

    # ---------- backward (hand-derived) ----------
    # dL/dr_hat:  from L_reg, and from L_disc through pi=softmax(r_hat/T)
    dLreg_drhat = W_REG * (2.0 / K) * diff_r
    dLdisc_drhat = W_DISC * (1.0 / POLICY_T) * pi * (g - gbar)
    dL_drhat = dLreg_drhat + dLdisc_drhat            # (K,)

    # r_hat = v - kappa * p_exp
    dL_dv_from_rhat   = dL_drhat * 1.0
    dL_dpexp_from_rhat = dL_drhat * (-kappa)
    dL_dkappa = np.sum(dL_drhat * (-p_exp))          # scalar

    # dL/dv: from r_hat, from L_val, from L_disc part-A (g depends on v)
    dLval_dv  = W_VAL * (2.0 / K) * diff_v
    dLdisc_dv = W_DISC * pi * (-sigmoid(V_FLOOR - v))  # dg/dv = -sigmoid(floor-v)
    dL_dv = dL_dv_from_rhat + dLval_dv + dLdisc_dv     # (K,)

    # dL/ds: from L_exp directly (softmax-CE => p_exp - p_adv), and from p_exp
    # entering r_hat (route through the softmax Jacobian).
    dL_ds_from_exp = W_EXP * (p_exp - p_adv)
    dY = dL_dpexp_from_rhat                            # upstream grad wrt p_exp
    dL_ds_from_rhat = p_exp * (dY - np.dot(dY, p_exp)) # softmax VJP
    dL_ds = dL_ds_from_exp + dL_ds_from_rhat           # (K,)

    # dL/dkappa_raw via softplus: dkappa/dkappa_raw = sigmoid(kappa_raw)
    dkappa_draw = sigmoid(p["kappa_raw"][0])
    dL_dkappa_raw = np.array([dL_dkappa * dkappa_draw])

    # ---- value head params ----
    gv = dL_dv[:, None]                                # (K,1)
    dWv2 = c["Av1"].T @ gv                             # (H,1)
    dbv2 = np.sum(gv, axis=0)                          # (1,)
    dAv1 = gv @ p["Wv2"].T                             # (K,H)
    dZv1 = dAv1 * dtanh(c["Av1"])                      # (K,H)
    dWv1 = c["X"].T @ dZv1                             # (d,H)
    dbv1 = np.sum(dZv1, axis=0)                        # (H,)

    # ---- expectation head params ----
    gs = dL_ds[:, None]                                # (K,1)
    dWe2 = c["Ae1"].T @ gs
    dbe2 = np.sum(gs, axis=0)
    dAe1 = gs @ p["We2"].T
    dZe1 = dAe1 * dtanh(c["Ae1"])
    dWe1 = c["X"].T @ dZe1
    dbe1 = np.sum(dZe1, axis=0)

    grads = {"Wv1": dWv1, "bv1": dbv1, "Wv2": dWv2, "bv2": dbv2,
             "We1": dWe1, "be1": dbe1, "We2": dWe2, "be2": dbe2,
             "kappa_raw": dL_dkappa_raw}
    parts = dict(L=L, L_reg=L_reg, L_val=L_val, L_exp=L_exp, L_disc=L_disc)
    return L, grads, parts


def batch_loss_and_grad(p, batch):
    """Mean loss + grads over a list of episodes (the Tjaneni-journal minibatch)."""
    total = {k: np.zeros_like(v) for k, v in p.items()}
    L_sum = 0.0
    for (X, v_true, p_adv, realized) in batch:
        L, g, _ = loss_and_grad_episode(p, X, v_true, p_adv, realized)
        L_sum += L
        for k in total:
            total[k] += g[k]
    n = len(batch)
    for k in total:
        total[k] /= n
    return L_sum / n, total


# ==============================================================================
# SECTION 5  --  GRADIENT CHECK  (mandatory)
# ------------------------------------------------------------------------------
# Central finite differences over EVERY scalar parameter, compared to the
# analytic gradients. A from-scratch model is only trustworthy if this passes.
# ==============================================================================

def gradient_check(seed=0, eps=1e-6):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    batch = [make_episode(rng) for _ in range(4)]   # small fixed batch

    _, analytic = batch_loss_and_grad(p, batch)

    max_rel = 0.0
    worst = None
    n_checked = 0
    for key in PARAM_KEYS:
        flat = p[key].ravel()
        gflat = analytic[key].ravel()
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = batch_loss_and_grad(p, batch)
            flat[i] = orig - eps
            Lm, _ = batch_loss_and_grad(p, batch)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel, worst = rel, (key, i, num, ana)
            n_checked += 1
    return max_rel, worst, n_checked


# ==============================================================================
# SECTION 6  --  OPTIMIZER (Adam, pure NumPy) + TRAINING with JOURNAL REPLAY
# ==============================================================================

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
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train(seed=1, n_episodes=6000, batch=24, journal_cap=4000, lr=3e-3):
    """Train the Aruna Engine. New councils of war are appended to the Tjaneni
    journal; each step replays a random minibatch from the journal -- the king
    returns to his records rather than over-fitting the latest campaign."""
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    opt = Adam(p, lr=lr)
    journal = []
    hist = []
    for it in range(n_episodes):
        journal.append(make_episode(rng))
        if len(journal) > journal_cap:
            journal.pop(rng.integers(0, len(journal)))   # forget at random
        idx = rng.integers(0, len(journal), size=min(batch, len(journal)))
        mb = [journal[i] for i in idx]
        L, g = batch_loss_and_grad(p, mb)
        opt.step(p, g)
        if (it + 1) % 1000 == 0:
            hist.append((it + 1, L, softplus(p["kappa_raw"])[0]))
    return p, hist


# ==============================================================================
# SECTION 7  --  POLICIES & EVALUATION
# ==============================================================================

def aruna_choice(p, X):
    """The Aruna Engine: choose argmax of predicted realized payoff."""
    return int(np.argmax(forward(p, X)["r_hat"]))

def value_only_choice(p, X):
    """A disciplined-but-blind commander: choose the road of highest sober merit,
    ignoring what the enemy expects. This is the council's recommendation."""
    return int(np.argmax(forward(p, X)["v"]))

def evaluate(p, rng, n=4000):
    tot_aruna = tot_value = tot_rand = tot_oracle = 0.0
    flips = 0
    for _ in range(n):
        X, v_true, p_adv, realized = make_episode(rng)
        a = aruna_choice(p, X)
        b = value_only_choice(p, X)
        r = int(rng.integers(0, X.shape[0]))
        tot_aruna  += realized[a]
        tot_value  += realized[b]
        tot_rand   += realized[r]
        tot_oracle += np.max(realized)
        if a != b:
            flips += 1
    return (tot_aruna / n, tot_value / n, tot_rand / n,
            tot_oracle / n, flips / n)


# ==============================================================================
# SECTION 8  --  MAIN: gradient check -> train -> self-tests
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 72)
    print("THE ARUNA ENGINE  --  Counter-Expectation Routing Network")
    print("Cognitive signature of Thutmose III (Menkheperre)")
    print("=" * 72)

    # ---- 1. mandatory gradient check ----
    print("\n[1] Finite-difference gradient check (central differences)")
    max_rel, worst, n_checked = gradient_check(seed=0, eps=1e-6)
    print(f"    parameters checked : {n_checked}")
    print(f"    max relative error : {max_rel:.3e}")
    print(f"    worst param/index  : {worst[0]}[{worst[1]}] "
          f"num={worst[2]:+.6f} ana={worst[3]:+.6f}")
    ok = max_rel < 1e-4
    print(f"    GRAD CHECK         : {'PASS' if ok else 'FAIL'}  (threshold 1e-4)")
    assert ok, "gradient check failed"

    # ---- 2. train ----
    print("\n[2] Training on synthetic Megiddo councils (journal replay)")
    p, hist = train(seed=1, n_episodes=6000, batch=24, lr=3e-3)
    for (it, L, kappa) in hist:
        print(f"    iter {it:5d}   loss={L:.4f}   kappa(audacity)={kappa:.3f}")

    # ---- 3. held-out evaluation vs baselines ----
    print("\n[3] Held-out realized payoff (mean over 4000 fresh councils)")
    rng = np.random.default_rng(777)
    aruna, value, rand, oracle, flip = evaluate(p, rng, n=4000)
    print(f"    Aruna Engine (read+invert expectation) : {aruna:+.4f}")
    print(f"    Value-only commander (council's choice): {value:+.4f}")
    print(f"    Random road                            : {rand:+.4f}")
    print(f"    Oracle (best possible)                 : {oracle:+.4f}")
    print(f"    Engine vs value-only advantage         : {aruna - value:+.4f}")
    print(f"    fraction of councils where the engine")
    print(f"    OVERRULES the value-only choice        : {flip*100:.1f}%")
    edge = aruna > value + 0.02
    print(f"    SELF-TEST (engine beats value-only)    : "
          f"{'PASS' if edge else 'FAIL'}")
    assert edge

    # ---- 4. the canonical Megiddo decision ----
    print("\n[4] The historical council at Yehem (canonical 3-road episode)")
    Xc, v_true, p_adv, realized = make_episode(np.random.default_rng(0),
                                               canonical=True)
    c = forward(p, Xc)
    labels = ["Zefti (N, wide/safe)", "Taanach (S, wide/safe)",
              "Aruna (centre, narrow/direct)"]
    print(f"    {'road':<30}{'merit v':>9}{'p_exp':>8}{'defended':>10}"
          f"{'realized':>10}")
    for k in range(3):
        print(f"    {labels[k]:<30}{c['v'][k]:>9.3f}{c['p_exp'][k]:>8.3f}"
              f"{DEFENSE_BUDGET*p_adv[k]:>10.3f}{realized[k]:>10.3f}")
    a = aruna_choice(p, Xc)
    b = value_only_choice(p, Xc)
    print(f"\n    value-only commander would take : {labels[b]}")
    print(f"    THE ARUNA ENGINE takes          : {labels[a]}")
    print(f"    learned audacity kappa          : {c['kappa']:.3f} "
          f"(true defense budget {DEFENSE_BUDGET})")
    chose_aruna = (a == 2 and b != 2)
    print(f"    SELF-TEST (engine picks Aruna,")
    print(f"    value-only picks a wide road)   : "
          f"{'PASS' if chose_aruna else 'FAIL'}")
    assert chose_aruna

    print("\n" + "=" * 72)
    print("ALL SELF-TESTS PASSED.")
    print("The engine learned to read the adversary's expectation and invert it")
    print("under discipline -- taking the road competent reasoning eliminated.")
    print("=" * 72)


if __name__ == "__main__":
    main()

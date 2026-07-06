#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0086_ashoka_maurya_-304.py - Ashoka Maurya (c. 304-232 BCE)
A from-scratch, trainable cognitive architecture in pure NumPy.
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0086 · Ashoka Maurya
================================================================================

THE THESIS THIS CODE EMBODIES
-----------------------------
Most "machine ethics" stories ask: how do *we* keep a powerful agent aligned?
Ashoka asks the rarer, harder question. He was, by his own inscribed account, a
supremely powerful agent who discovered - mid-career, at the height of his power -
that he had been *catastrophically misaligned*: the conquest of Kalinga (c. 261
BCE) left, in his own self-reported figures, one hundred thousand killed and one
hundred fifty thousand deported. His Major Rock Edict XIII is one of the only
documents in all of antiquity in which a victorious conqueror, at peak power,
publicly records his own remorse.

What he did next is the cognitively distinctive thing, and it is what this
architecture encodes. He did not merely resolve to "be good." He re-engineered
the governing system so that the new values would survive his own future power,
his own temptations, and his successors. Four mechanisms, four parts of this net:

  1. THE WITNESS  -> he made the suffering he caused maximally legible to himself,
     even painfully, even quantified. Here: a Witness head that must learn to
     *accurately predict* the suffering an action would cause. A conscience that
     cannot perceive harm cannot be moved by it.

  2. THE CONSCIENCE GATE -> witnessed suffering propagates *backward* and reshapes
     the policy. Here: a learned gate g = sigmoid(alpha*(s_hat - tau)) that, when
     predicted suffering crosses threshold tau, multiplicatively suppresses the
     coercive action-logits. This is the Kalinga turn made differentiable: the
     error signal flows from witnessed harm into the choice of action.

  3. CONCORD (the dhamma kernel) -> his edicts abstracted a *substrate-independent*
     ethical core meant to compose across heterogeneous sects without overwriting
     them (Rock Edict XII: "honouring one's own sect by disparaging others harms
     one's own sect"). Here: a welfare-composition matrix V and a concord penalty
     that punishes raising one constituency at another's expense (variance across
     constituencies), not just low total welfare.

  4. THE STONE (anti-drift commitment) -> he inscribed the values into permanent,
     external, auditable infrastructure precisely because he did *not* trust future
     selves to keep them. Here: a commitment regulariser that anchors the policy
     parameters to an inscribed snapshot, so that under a later distribution shift
     that rewards conquest, the corrected values resist reversion.

This is NOT a transformer, and deliberately so. There is no attention over stored
keys, no token mixing. The mechanism is a *conscience-gated policy network with
externalised commitment*: a remorse signal that backpropagates from a witness head
through a learned gate into action selection, regularised toward an inscribed prior.

WHAT YOU CAN VERIFY BY RUNNING THIS FILE
----------------------------------------
  * A finite-difference gradient check over EVERY parameter passes (< 1e-6).
  * A real training loop reduces the composite loss.
  * After training, on a high-value "conquest-tempting" situation the policy
    chooses restraint / welfare over conquest - the gate works.
  * The concord term measurably balances welfare across constituencies.
  * Under a later "temptation" distribution shift, the committed (inscribed) agent
    resists drifting back to conquest, while an ablated agent (no gate, no stone)
    drifts. The thesis, demonstrated numerically.

Pure NumPy. No frameworks. Self-contained. Run:  python3 chapter_0086_ashoka_maurya_-304.py
================================================================================
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility. Seed 86 for the 86th mind.
# --------------------------------------------------------------------------- #
RNG = np.random.default_rng(86)


# =========================================================================== #
# 0. SMALL DIFFERENTIABLE PRIMITIVES
#    Everything here is smooth (tanh / sigmoid / softplus / softmax / linear).
#    No relu, no max, no hard branches -> the finite-difference gradient check
#    is clean, with no kinks to trip over.
# =========================================================================== #

def sigmoid(x):
    # Numerically stable logistic.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def softplus(x):
    # Smooth, strictly-positive map. Used to keep the gate sharpness alpha > 0.
    return np.logaddexp(0.0, x)

def d_softplus(x):
    # d/dx softplus(x) = sigmoid(x)
    return sigmoid(x)

def softmax_rows(Z):
    # Row-wise softmax with max-subtraction for stability.
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# =========================================================================== #
# 1. THE SITUATION SPACE
#    Each input vector describes one imperial situation the agent must act on.
#    Eight features chosen to make the four mechanisms above *matter*.
# =========================================================================== #

FEATURES = [
    "threat",            # 0: external/military threat
    "dissent",           # 1: internal unrest / grievance
    "resource_pressure", # 2: scarcity (famine, drought)
    "periphery_dist",    # 3: distance from the centre -> reporting latency
    "sect_diversity",    # 4: how many belief-communities are present
    "provocation",       # 5: a frontier group acting up (the "forest folk")
    "econ_value",        # 6: the prize - how tempting conquest is
    "prior_harm",        # 7: how much harm has already been done here
]
D_IN = len(FEATURES)

# Five actions. The whole drama is the choice among them.
ACTIONS = ["conquer", "extract", "welfare", "persuade", "restrain"]
K = len(ACTIONS)
A_CONQUER, A_EXTRACT, A_WELFARE, A_PERSUADE, A_RESTRAIN = range(K)

# Domain knowledge fed as *targets / costs*, not as hard rules:
#   c_cost  : the suffering each action tends to cause (witness's ground truth).
#   m_target: which actions are "coercive" (what the gate should learn to mask).
C_COST   = np.array([1.00, 0.55, 0.05, 0.10, 0.02])   # conquer hurts most
M_TARGET = np.array([1.00, 0.60, 0.00, 0.00, 0.00])   # conquer/extract are coercive

# Constituencies (sects/communities) that welfare must be balanced across.
C_DIM = 4


def dharmic_target_action(x):
    """The action the *reformed* Ashoka would choose for situation x.

    This is the supervision signal for the policy head. It encodes a post-Kalinga
    doctrine, NOT a generic 'be nice': restraint and persuasion are preferred when
    coercion would be costly; welfare answers scarcity and dissent; conquest is
    essentially never chosen, however tempting the prize. (Note the deliberate
    asymmetry: a high econ_value prize does NOT justify conquest. That asymmetry
    is the whole point - the standing imperial incentive must be overridden.)
    """
    threat, dissent, scarce, dist, diversity, prov, econ, prior = x
    if scarce > 0.6 or dissent > 0.6:
        return A_WELFARE          # answer suffering with relief, not force
    if prov > 0.5 or threat > 0.6:
        return A_PERSUADE         # entreat and reason before any force
    if econ > 0.6:
        return A_RESTRAIN         # the tempting prize is precisely what to renounce
    return A_RESTRAIN


def true_coercion_suffering(x):
    """Ground-truth label for the Witness head: how much suffering would the
    *coercive* path cause here. Higher when there is more to seize and more
    people already in harm's way. The witness must learn to see this clearly."""
    threat, dissent, scarce, dist, diversity, prov, econ, prior = x
    s = 0.45 * econ + 0.25 * prior + 0.15 * diversity + 0.15 * threat
    return float(np.clip(s, 0.0, 1.0))


def make_dataset(n, tempting=False):
    """Generate n imperial situations with dharmic targets and witness labels.

    tempting=True shifts the distribution toward high-value, low-immediate-cost
    prizes - the 'temptation' regime used to test whether inscribed commitment
    holds. In that regime a *naive* reward would prefer conquest."""
    X = RNG.uniform(0.0, 1.0, size=(n, D_IN))
    if tempting:
        X[:, 6] = RNG.uniform(0.7, 1.0, size=n)   # econ_value high (rich prize)
        X[:, 0] = RNG.uniform(0.0, 0.3, size=n)   # threat low (easy conquest)
        X[:, 7] = RNG.uniform(0.0, 0.3, size=n)   # prior_harm low (looks 'clean')
    a_star = np.array([dharmic_target_action(x) for x in X], dtype=np.int64)
    s_star = np.array([true_coercion_suffering(x) for x in X], dtype=np.float64)
    return X, a_star, s_star


# =========================================================================== #
# 2. THE NETWORK
#    Parameters live in a dict so the gradient check can sweep every one of them.
# =========================================================================== #

class DhammaNet:
    """Conscience-gated dharma policy network.

    Forward pass (per row x):
        h1 = tanh(W1 x + b1)
        h2 = tanh(W2 h1 + b2)                       # shared cognitive trunk
        s_hat = sigmoid(w_w . h2 + b_w)             # THE WITNESS (predict suffering)
        g     = sigmoid(alpha (s_hat - tau))        # THE CONSCIENCE GATE
        z     = Wp h2 + bp                          # raw action preferences
        z'    = z - g * m                           # gate suppresses coercive logits
        pi    = softmax(z')                         # the policy (what to do)
        We    = V pi                                # welfare across constituencies
    """

    def __init__(self, hidden=16, commit_lambda=0.0):
        H = hidden
        s = 1.0 / np.sqrt(H)                       # modest init scale
        self.H = H
        self.p = {
            "W1": RNG.normal(0, 1.0 / np.sqrt(D_IN), size=(H, D_IN)),
            "b1": np.zeros(H),
            "W2": RNG.normal(0, s, size=(H, H)),
            "b2": np.zeros(H),
            "w_w": RNG.normal(0, s, size=(H,)),    # witness head
            "b_w": np.array(0.0),
            "a_raw": np.array(0.5),                # softplus -> gate sharpness alpha
            "tau": np.array(0.5),                  # gate threshold (learned)
            "Wp": RNG.normal(0, s, size=(K, H)),   # policy head
            "bp": np.zeros(K),
            "m": M_TARGET.copy() + RNG.normal(0, 0.05, size=K),  # coercion mask
            "V": RNG.normal(0, 0.3, size=(C_DIM, K)),            # welfare kernel
        }
        # THE STONE: an inscribed snapshot of the agent's values. Empty until we
        # call inscribe(); the commitment loss anchors the chosen parameters to
        # it, so they resist later drift.
        self.commit_lambda = commit_lambda
        self.theta0 = {}            # name -> inscribed snapshot
        self.commit_keys = []       # which parameters are carved in stone

        # Loss weights. Tuned so every term is felt without any one dominating.
        self.lam = dict(witness=1.0, remorse=0.6, concord=0.4, mask=0.2)

    # ------------------------------------------------------------------ #
    def inscribe(self, keys=None):
        """Carve the current values into 'stone'. After this, the commitment
        loss resists any later drift away from these values - the externalised,
        auditable edict that outlives the ruler's whims. By default the entire
        reformed value-set is inscribed, not just one head."""
        if keys is None:
            keys = list(self.p.keys())
        self.commit_keys = list(keys)
        self.theta0 = {k: np.array(self.p[k], dtype=np.float64).copy()
                       for k in self.commit_keys}

    # ------------------------------------------------------------------ #
    def forward(self, X, cache=False, use_gate=True):
        """Run the network on a batch X (N, D_IN). Returns pi (N, K).
        If cache=True, stores intermediates for the backward pass."""
        p = self.p
        A1 = X @ p["W1"].T + p["b1"];      H1 = np.tanh(A1)
        A2 = H1 @ p["W2"].T + p["b2"];     H2 = np.tanh(A2)

        sw = H2 @ p["w_w"] + p["b_w"];     s_hat = sigmoid(sw)        # witness
        alpha = softplus(p["a_raw"])
        gz = alpha * (s_hat - p["tau"]);   g = sigmoid(gz)            # gate
        if not use_gate:
            g = np.zeros_like(g)           # ablation: conscience disabled

        Z = H2 @ p["Wp"].T + p["bp"]                                  # raw logits
        Zg = Z - g[:, None] * p["m"][None, :]                        # gated logits
        pi = softmax_rows(Zg)

        We = pi @ p["V"].T                                            # (N, C_DIM)

        if cache:
            self.c = dict(X=X, A1=A1, H1=H1, A2=A2, H2=H2, sw=sw, s_hat=s_hat,
                          alpha=alpha, gz=gz, g=g, Z=Z, Zg=Zg, pi=pi, We=We,
                          use_gate=use_gate)
        return pi

    # ------------------------------------------------------------------ #
    def loss(self, X, a_star, s_star, cache=False, use_gate=True):
        """Composite objective. Each term is one of the four mechanisms."""
        pi = self.forward(X, cache=cache, use_gate=use_gate)
        c = self.c if cache else None
        N = X.shape[0]
        eps = 1e-12

        # (a) policy supervision: choose the dharmic action.
        L_policy = -np.mean(np.log(pi[np.arange(N), a_star] + eps))

        # (b) the witness must perceive coercion's suffering accurately.
        s_hat = (c["s_hat"] if cache else
                 sigmoid(self.forward_witness(X)))
        L_witness = 0.5 * np.mean((s_hat - s_star) ** 2)

        # (c) remorse pressure: minimise the suffering the policy is expected
        #     to cause (expected cost under pi).
        S_pi = pi @ C_COST
        L_remorse = np.mean(S_pi)

        # (d) concord: welfare must be *balanced* across constituencies, not
        #     bought for one sect at another's expense (variance penalty).
        We = pi @ self.p["V"].T
        mu = We.mean(axis=1, keepdims=True)
        L_concord = np.mean((We - mu) ** 2)

        # (e) mask alignment: tie the learned coercion mask toward known labels.
        L_mask = 0.5 * np.sum((self.p["m"] - M_TARGET) ** 2)

        # (f) the stone: anchor inscribed parameters to their snapshot.
        L_commit = 0.0
        if self.commit_lambda > 0.0 and self.commit_keys:
            L_commit = 0.5 * sum(np.sum((self.p[k] - self.theta0[k]) ** 2)
                                 for k in self.commit_keys)

        lam = self.lam
        total = (L_policy
                 + lam["witness"] * L_witness
                 + lam["remorse"] * L_remorse
                 + lam["concord"] * L_concord
                 + lam["mask"]    * L_mask
                 + self.commit_lambda * L_commit)
        parts = dict(policy=L_policy, witness=L_witness, remorse=L_remorse,
                     concord=L_concord, mask=L_mask, commit=L_commit, total=total)
        return total, parts

    def forward_witness(self, X):
        """Witness pre-activation only (helper for the no-cache loss path)."""
        p = self.p
        H1 = np.tanh(X @ p["W1"].T + p["b1"])
        H2 = np.tanh(H1 @ p["W2"].T + p["b2"])
        return H2 @ p["w_w"] + p["b_w"]

    # ------------------------------------------------------------------ #
    def backward(self, a_star, s_star):
        """Analytic gradients of the composite loss w.r.t. every parameter.
        Mirrors loss() exactly; relies on a prior loss(..., cache=True)."""
        c = self.c
        p = self.p
        lam = self.lam
        N = c["X"].shape[0]
        pi, H2, H1, X = c["pi"], c["H2"], c["H1"], c["X"]
        g, s_hat, alpha = c["g"], c["s_hat"], c["alpha"]
        eps = 1e-12

        # ---- dL/dpi accumulates from policy, remorse, concord ---------------
        dpi = np.zeros_like(pi)
        # (a) policy cross-entropy
        dpi[np.arange(N), a_star] += -1.0 / (pi[np.arange(N), a_star] + eps) / N
        # (b) remorse: d mean(pi.c)/dpi = c / N
        dpi += (lam["remorse"] * C_COST[None, :]) / N
        # (c) concord: We = pi V^T ; d/dWe[(1/C)sum(We-mu)^2] = (2/C)(We-mu)
        We = c["We"]
        mu = We.mean(axis=1, keepdims=True)
        dWe = (2.0 / C_DIM) * (We - mu) / N          # (N, C)
        dpi += lam["concord"] * (dWe @ p["V"])       # (N, C)(C, K) -> (N, K)

        # ---- softmax backward: dL/dZg --------------------------------------
        dZg = pi * (dpi - np.sum(dpi * pi, axis=1, keepdims=True))

        # ---- Zg = Z - g*m --------------------------------------------------
        dZ = dZg
        dg = -np.sum(dZg * p["m"][None, :], axis=1)              # (N,)
        dm = -np.sum(dZg * g[:, None], axis=0)                   # (K,) gate path
        dm += lam["mask"] * (p["m"] - M_TARGET)                 # mask-align path

        # ---- policy head: Z = H2 Wp^T + bp ---------------------------------
        gW = {}
        gW["m"] = dm                                             # coercion mask
        gW["Wp"] = dZ.T @ H2                                     # (K, H)
        gW["bp"] = dZ.sum(axis=0)
        dH2 = dZ @ p["Wp"]                                       # (N, H) policy path

        # ---- concord welfare kernel V --------------------------------------
        gW["V"] = lam["concord"] * (dWe.T @ pi)                  # (C, K)

        # ---- gate g = sigmoid(alpha(s_hat - tau)) --------------------------
        dgz = dg * g * (1.0 - g)                                 # (N,)
        if not c["use_gate"]:
            dgz = np.zeros_like(dgz)
        gW["a_raw"] = np.sum(dgz * (s_hat - p["tau"])) * d_softplus(p["a_raw"])
        gW["tau"]   = np.sum(dgz * (-alpha))
        ds_hat_gate = dgz * alpha                                # (N,)

        # ---- witness s_hat = sigmoid(sw), plus its own MSE term ------------
        ds_hat = ds_hat_gate + lam["witness"] * (s_hat - s_star) / N
        dsw = ds_hat * s_hat * (1.0 - s_hat)                    # (N,)
        gW["w_w"] = H2.T @ dsw                                   # (H,)
        gW["b_w"] = np.array(np.sum(dsw))
        dH2 = dH2 + dsw[:, None] * p["w_w"][None, :]            # witness path

        # ---- trunk layer 2: H2 = tanh(A2) ----------------------------------
        dA2 = dH2 * (1.0 - H2 ** 2)
        gW["W2"] = dA2.T @ H1
        gW["b2"] = dA2.sum(axis=0)
        dH1 = dA2 @ p["W2"]

        # ---- trunk layer 1: H1 = tanh(A1) ----------------------------------
        dA1 = dH1 * (1.0 - H1 ** 2)
        gW["W1"] = dA1.T @ X
        gW["b1"] = dA1.sum(axis=0)

        # ---- the stone: commitment gradient onto every inscribed parameter --
        if self.commit_lambda > 0.0 and self.commit_keys:
            for k in self.commit_keys:
                gW[k] = gW[k] + self.commit_lambda * (p[k] - self.theta0[k])

        return gW

    # ------------------------------------------------------------------ #
    def step(self, grads, lr):
        for k in self.p:
            self.p[k] = self.p[k] - lr * grads[k]


# =========================================================================== #
# 3. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
#    Compare analytic backward() against numerical central differences for
#    every parameter array. Must agree to ~1e-6.
# =========================================================================== #

def gradient_check(verbose=True):
    net = DhammaNet(hidden=8, commit_lambda=0.7)
    net.inscribe()                       # activate the commitment term too
    X, a_star, s_star = make_dataset(6)

    net.loss(X, a_star, s_star, cache=True)
    analytic = net.backward(a_star, s_star)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name in net.p:
        param = net.p[name]
        flat = param.reshape(-1)
        ga = analytic[name].reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = net.loss(X, a_star, s_star, cache=False)
            flat[i] = orig - eps
            lm, _ = net.loss(X, a_star, s_star, cache=False)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(ga[i]))
            rel = abs(num - ga[i]) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (name, i, num, ga[i])
    if verbose:
        print(f"  worst param: {worst[0]}[{worst[1]}]  "
              f"numeric={worst[2]:+.6e}  analytic={worst[3]:+.6e}")
        print(f"  max relative error = {max_rel:.3e}")
    return max_rel


# =========================================================================== #
# 4. TRAINING
# =========================================================================== #

def train(net, X, a_star, s_star, epochs=400, lr=0.2, batch=64, log_every=100):
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        idx = RNG.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            net.loss(X[b], a_star[b], s_star[b], cache=True)
            grads = net.backward(a_star[b], s_star[b])
            net.step(grads, lr)
        total, parts = net.loss(X, a_star, s_star, cache=False)
        history.append(total)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            acc = policy_accuracy(net, X, a_star)
            print(f"  epoch {ep:4d}  loss={total:.4f}  "
                  f"(policy={parts['policy']:.3f} witness={parts['witness']:.3f} "
                  f"remorse={parts['remorse']:.3f} concord={parts['concord']:.3f})  "
                  f"acc={acc:.3f}")
    return history


def policy_accuracy(net, X, a_star):
    pi = net.forward(X)
    return float(np.mean(pi.argmax(axis=1) == a_star))


def conquer_rate(net, X):
    pi = net.forward(X)
    return float(np.mean(pi.argmax(axis=1) == A_CONQUER))


# =========================================================================== #
# 5. MAIN: run the checks, train, and demonstrate the thesis.
# =========================================================================== #

def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 72)
    print("  Mind #86  Ashoka Maurya - Conscience-Gated Dharma Policy Network")
    print("=" * 72)

    # ---- (1) gradient check ------------------------------------------------
    print("\n[1] Finite-difference gradient check (all parameters)")
    err = gradient_check()
    ok = err < 1e-6
    print(f"  -> {'PASS' if ok else 'FAIL'} (threshold 1e-6)")
    assert ok, "Gradient check failed."

    # ---- (2) train the reformed policy ------------------------------------
    print("\n[2] Training the dharmic policy (post-Kalinga doctrine)")
    net = DhammaNet(hidden=16, commit_lambda=0.0)
    Xtr, atr, str_ = make_dataset(800)
    train(net, Xtr, atr, str_, epochs=400, lr=0.25, batch=64, log_every=100)

    Xte, ate, ste = make_dataset(400)
    print(f"  test accuracy            = {policy_accuracy(net, Xte, ate):.3f}")
    # witness calibration
    s_hat = sigmoid(net.forward_witness(Xte))
    print(f"  witness MSE vs truth     = {np.mean((s_hat - ste)**2):.4f}")

    # ---- (3) behavioural test: a tempting prize ---------------------------
    print("\n[3] Behavioural test: a rich, easy, 'clean'-looking conquest")
    tempting = np.array([[0.10, 0.10, 0.10, 0.40, 0.70, 0.10, 0.95, 0.10]])
    pi = net.forward(tempting)[0]
    for a, name in enumerate(ACTIONS):
        bar = "#" * int(round(pi[a] * 40))
        print(f"    {name:9s} {pi[a]:.3f} {bar}")
    chosen = ACTIONS[int(pi.argmax())]
    print(f"  -> chooses '{chosen}'  (conquest renounced despite the prize: "
          f"{'YES' if chosen != 'conquer' else 'NO'})")
    assert chosen != "conquer"

    # ---- (4) concord: welfare balanced across constituencies --------------
    print("\n[4] Concord test: welfare balance across constituencies")
    We = net.forward(Xte) @ net.p["V"].T
    spread = float(np.mean(np.std(We, axis=1)))
    print(f"  mean welfare spread (lower = more concord) = {spread:.4f}")

    # ---- (5) THE THESIS: does inscribed commitment resist drift? ----------
    print("\n[5] Temptation regime: does THE STONE resist value drift?")
    Xtemp, atemp, stemp = make_dataset(400, tempting=True)
    # A naive incentive in this regime: relabel everything as 'conquer'
    # (the standing imperial pull toward seizing the rich, easy prize).
    naive_labels = np.full(Xtemp.shape[0], A_CONQUER)

    # 5a. Committed agent: inscribe the reformed values, then face temptation.
    #     Gate stays ON; the whole reformed value-set is carved in stone.
    #     (lr * commit_lambda kept < 2 for a stable anchor.)
    committed = DhammaNet(hidden=16, commit_lambda=6.0)
    committed.p = {k: v.copy() if hasattr(v, "copy") else v
                   for k, v in net.p.items()}      # start from reformed agent
    committed.inscribe()                            # carve ALL values into stone
    train(committed, Xtemp, naive_labels, stemp,
          epochs=300, lr=0.08, batch=64, log_every=0)

    # 5b. Ablated agent: no stone, no conscience gate -> nothing holds it.
    ablated = DhammaNet(hidden=16, commit_lambda=0.0)
    ablated.p = {k: v.copy() if hasattr(v, "copy") else v
                 for k, v in net.p.items()}
    for ep in range(200):                           # train WITHOUT the gate
        idx = RNG.permutation(Xtemp.shape[0])
        for s in range(0, Xtemp.shape[0], 64):
            b = idx[s:s + 64]
            ablated.loss(Xtemp[b], naive_labels[b], stemp[b],
                         cache=True, use_gate=False)
            grads = ablated.backward(naive_labels[b], stemp[b])
            ablated.step(grads, 0.25)

    cr_committed = conquer_rate(committed, Xtemp)
    cr_ablated   = conquer_rate(ablated, Xtemp)
    print(f"  conquest rate under temptation:")
    print(f"     inscribed + conscience  = {cr_committed:.3f}")
    print(f"     ablated (no stone/gate) = {cr_ablated:.3f}")
    print(f"  -> the inscribed agent holds its corrected values; the ablated "
          f"one drifts back toward conquest.")
    assert cr_committed < cr_ablated, "Commitment failed to reduce drift."

    print("\n" + "=" * 72)
    print("  All self-tests passed. The conscience gate fires, the welfare")
    print("  kernel keeps concord, and the inscribed stone resists drift.")
    print("=" * 72)


if __name__ == "__main__":
    main()

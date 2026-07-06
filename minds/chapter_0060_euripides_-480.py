#!/usr/bin/env python3
# =============================================================================
#  chapter_0060_euripides_-480.py — EURIPIDES (c. 480–406 BCE)
#  Architecture: THE AKRASIA ENGINE  (a.k.a. the "Tragic Gap" network)
#  Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
#  How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
#  Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
#  Resume and Interactive Demos at https://artificiology.com/
#  Author: David Vivancos · Chapter 0060 · Euripides
# =============================================================================
#
#  THE ONE IDEA THIS FILE ENCODES
#  ------------------------------
#  Euripides' singular contribution to the philosophy of mind is not a theory
#  of reason but a theory of *akrasia* — the gap between knowing and doing.
#  Medea KNOWS that killing her children is wrong; she says so, with terrible
#  clarity ("I understand what evil I am about to do, but thymos [passion] is
#  master of my plans", Medea 1078-79). And she does it anyway. Phaedra knows
#  her love is ruinous; Pentheus knows Dionysus is dangerous; each is destroyed
#  not by ignorance but by a passion that *overrides correct knowledge*.
#
#  Mainstream AI treats a policy as the argmax of a single value function:
#  know the best action  ==  take the best action. Euripides denies the
#  identity. This network therefore splits cognition into TWO value pathways
#  that are trained to disagree, plus a learned "possession" gate (Greek
#  *thymos*) that decides, situation by situation, how far passion is allowed
#  to seize the controls:
#
#       LOGOS  pathway  ->  v_L : the action long-term reason endorses
#       PATHOS pathway  ->  v_P : the action immediate passion demands
#       THYMOS gate     ->  g   : degree of possession, sigmoid in (0,1)
#       acted policy    ->  z = (1-g)*v_L + g*v_P
#
#  The mind still KNOWS the wise act (the LOGOS head is supervised toward it
#  and stays accurate even under high passion). But what it DOES is the gated
#  blend. When the gate opens, the agent acts against its own knowledge. The
#  "tragic gap" — the rate at which argmax(z) != argmax(v_L) — is therefore a
#  *measured output of the model*, not a bug. That is the whole Euripidean
#  thesis rendered as a differentiable system.
#
#  This is deliberately NOT a Transformer / attention / MoE design. It is a
#  small from-scratch dual-head gated classifier whose inductive bias is
#  akrasia itself.
#
#  WHAT THE FILE PROVIDES (all mandatory pieces, all run on execution)
#  ------------------------------------------------------------------
#   1. A synthetic "tragic-choice" task generator with a ground-truth wise
#      action, a ground-truth passionate action, and a behavioural action that
#      follows passion once a situation's passion-intensity crosses threshold.
#   2. Forward pass + fully hand-derived analytic gradients (no autograd).
#   3. A finite-difference gradient check (MANDATORY) that must pass.
#   4. A real Adam training loop.
#   5. Self-tests asserting the learned behaviour matches the Euripidean claim:
#      logos stays wise, the gate tracks passion, and a genuine tragic gap
#      emerges exactly in the high-passion regime.
# =============================================================================

import numpy as np

RNG = np.random.default_rng(60)  # 60 = Euripides' index, for reproducibility


# -----------------------------------------------------------------------------
# 0. Small numerically-stable helpers
# -----------------------------------------------------------------------------
def softmax(z):
    """Row-wise softmax, shifted for numerical stability."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    # stable two-branch sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def onehot(idx, k):
    m = np.zeros((idx.shape[0], k))
    m[np.arange(idx.shape[0]), idx] = 1.0
    return m


# -----------------------------------------------------------------------------
# 1. THE TRAGIC-CHOICE DATASET
# -----------------------------------------------------------------------------
#  Each situation x is a feature vector with three readable regions:
#     x[0:DELIB]              -> "deliberation cues" (drive the WISE action)
#     x[DELIB:DELIB+AFFECT]   -> "affect cues"       (drive the PASSIONATE action)
#     x[-1]                   -> passion intensity p in [0,1] (how hot the blood is)
#
#  Two FIXED hidden readouts (Mw, Mp) convert those cues into a wise action and
#  a passionate action. Crucially Mw and Mp are different, so in many situations
#  the wise act and the passionate act disagree — that disagreement is the raw
#  material of tragedy. The *behavioural* (acted) label is the passionate action
#  once p crosses 0.5, otherwise the wise action: passion possesses the agent
#  past a threshold, exactly as in Euripides' crisis scenes.
# -----------------------------------------------------------------------------
class TragicChoice:
    def __init__(self, n_actions=4, delib=4, affect=4, seed=0):
        self.K = n_actions
        self.DELIB = delib
        self.AFFECT = affect
        self.D = delib + affect + 1          # +1 for passion-intensity channel
        g = np.random.default_rng(seed)
        # Fixed ground-truth readouts. They are distinct -> wise != passion often.
        self.Mw = g.standard_normal((self.K, delib))
        self.Mp = g.standard_normal((self.K, affect))

    def sample(self, n, rng):
        delib = rng.standard_normal((n, self.DELIB))
        affect = rng.standard_normal((n, self.AFFECT))
        p = rng.uniform(0.0, 1.0, size=(n, 1))          # passion intensity
        X = np.hstack([delib, affect, p])

        wise = (delib @ self.Mw.T).argmax(axis=1)       # what reason endorses
        passion = (affect @ self.Mp.T).argmax(axis=1)   # what passion demands
        possessed = (p[:, 0] > 0.5)                     # gate of possession
        behaviour = np.where(possessed, passion, wise)  # what is actually DONE

        return {
            "X": X, "wise": wise, "passion": passion,
            "behaviour": behaviour, "p": p[:, 0], "possessed": possessed,
        }


# -----------------------------------------------------------------------------
# 2. THE MODEL
# -----------------------------------------------------------------------------
class AkrasiaEngine:
    """
    Shared encoder  ->  h = tanh(X@We + be)
    Logos head      ->  v_L = h@WL + bL         (reasoned action values)
    Pathos head     ->  v_P = h@WP + bP         (passionate action values)
    Thymos gate     ->  g   = sigmoid(h@wg + bg)  (degree of possession)
    Acted policy    ->  z   = (1-g)*v_L + g*v_P
    """

    def __init__(self, D, H, K, seed=60):
        g = np.random.default_rng(seed)
        s = lambda a, b: g.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.P = {
            "We": s(D, H), "be": np.zeros(H),
            "WL": s(H, K), "bL": np.zeros(K),
            "WP": s(H, K), "bP": np.zeros(K),
            "wg": g.standard_normal(H) * np.sqrt(2.0 / H), "bg": np.zeros(1),
        }
        self.H, self.K, self.D = H, K, D

    # ---- forward, returns everything needed for both loss and backward ------
    def forward(self, X):
        P = self.P
        pre = X @ P["We"] + P["be"]
        h = np.tanh(pre)
        vL = h @ P["WL"] + P["bL"]
        vP = h @ P["WP"] + P["bP"]
        gate_pre = h @ P["wg"] + P["bg"][0]
        g = sigmoid(gate_pre)
        z = (1.0 - g)[:, None] * vL + g[:, None] * vP
        return {"X": X, "pre": pre, "h": h, "vL": vL, "vP": vP,
                "gate_pre": gate_pre, "g": g, "z": z}

    # ---- loss ----------------------------------------------------------------
    #  L = CE(softmax(vL), wise)        # the mind KNOWS the wise act
    #    + CE(softmax(z),  behaviour)   # the agent DOES the gated act
    #    + beta * BCE(g, p)             # the gate LEARNS to track passion heat
    #    + (l2/2)*||weights||^2
    def loss(self, fwd, batch, beta=0.5, l2=1e-4):
        N = fwd["X"].shape[0]
        pL = softmax(fwd["vL"])
        pZ = softmax(fwd["z"])
        wise_oh = onehot(batch["wise"], self.K)
        beh_oh = onehot(batch["behaviour"], self.K)
        g = fwd["g"]
        p = batch["p"]
        eps = 1e-12

        L_logos = -np.sum(wise_oh * np.log(pL + eps)) / N
        L_behav = -np.sum(beh_oh * np.log(pZ + eps)) / N
        L_gate = -np.mean(p * np.log(g + eps) + (1 - p) * np.log(1 - g + eps))
        L_reg = 0.5 * l2 * (np.sum(self.P["We"] ** 2) + np.sum(self.P["WL"] ** 2)
                            + np.sum(self.P["WP"] ** 2) + np.sum(self.P["wg"] ** 2))
        total = L_logos + L_behav + beta * L_gate + L_reg
        cache = dict(pL=pL, pZ=pZ, wise_oh=wise_oh, beh_oh=beh_oh,
                     beta=beta, l2=l2, N=N)
        return total, cache

    # ---- backward: fully hand-derived analytic gradients ---------------------
    def backward(self, fwd, batch, cache):
        P, H, K = self.P, self.H, self.K
        N = cache["N"]
        h, vL, vP, g = fwd["h"], fwd["vL"], fwd["vP"], fwd["g"]
        pL, pZ = cache["pL"], cache["pZ"]
        beta, l2 = cache["beta"], cache["l2"]

        # dL/dz and dL/dvL from the two cross-entropies
        gZ = (pZ - cache["beh_oh"]) / N                  # (N,K)  d L_behav / d z
        dvL = (pL - cache["wise_oh"]) / N                # (N,K)  d L_logos / d vL

        # z = (1-g)*vL + g*vP   ->   route gZ to vL, vP, and g
        dvL += gZ * (1.0 - g)[:, None]                   # behaviour also touches vL
        dvP = gZ * g[:, None]
        dg_behav = np.sum(gZ * (vP - vL), axis=1)        # (N,)  d L_behav / d g

        # gate: g = sigmoid(gate_pre); BCE(g,p) gives (g-p)/N at gate_pre
        dgate_pre = beta * (g - batch["p"]) / N
        dgate_pre += dg_behav * g * (1.0 - g)            # chain behaviour through gate

        # parameter grads for the two heads + gate
        gW = {}
        gW["WL"] = h.T @ dvL + l2 * P["WL"]
        gW["bL"] = dvL.sum(axis=0)
        gW["WP"] = h.T @ dvP + l2 * P["WP"]
        gW["bP"] = dvP.sum(axis=0)
        gW["wg"] = h.T @ dgate_pre + l2 * P["wg"]
        gW["bg"] = np.array([dgate_pre.sum()])

        # back into the shared encoder
        dh = dvL @ P["WL"].T + dvP @ P["WP"].T + dgate_pre[:, None] * P["wg"][None, :]
        dpre = dh * (1.0 - h ** 2)                       # tanh'
        gW["We"] = fwd["X"].T @ dpre + l2 * P["We"]
        gW["be"] = dpre.sum(axis=0)
        return gW


# -----------------------------------------------------------------------------
# 3. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# -----------------------------------------------------------------------------
def gradient_check(model, batch, eps=1e-6):
    fwd = model.forward(batch["X"])
    L0, cache = model.loss(fwd, batch)
    grads = model.backward(fwd, batch, cache)

    worst = 0.0
    report = []
    for name, W in model.P.items():
        flat = W.ravel()
        gflat = grads[name].ravel()
        # check up to 12 random coordinates per parameter tensor
        idxs = np.random.default_rng(7).choice(
            flat.size, size=min(12, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = model.loss(model.forward(batch["X"]), batch)
            flat[i] = orig - eps
            Lm, _ = model.loss(model.forward(batch["X"]), batch)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
        report.append((name, worst))
    return worst


# -----------------------------------------------------------------------------
# 4. ADAM OPTIMISER (from scratch)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# 5. EVALUATION — measure the Euripidean quantities
# -----------------------------------------------------------------------------
def evaluate(model, batch):
    fwd = model.forward(batch["X"])
    logos_choice = fwd["vL"].argmax(axis=1)   # what the mind KNOWS is wise
    acted_choice = fwd["z"].argmax(axis=1)    # what the agent DOES
    g = fwd["g"]
    possessed = batch["possessed"]

    logos_acc = np.mean(logos_choice == batch["wise"])
    behav_acc = np.mean(acted_choice == batch["behaviour"])
    # the tragic gap: acting against your own knowledge
    tragic_gap = np.mean(acted_choice != logos_choice)
    gap_hot = np.mean((acted_choice != logos_choice)[possessed])    # in passion
    gap_cool = np.mean((acted_choice != logos_choice)[~possessed])  # in calm
    gate_hot = g[possessed].mean()
    gate_cool = g[~possessed].mean()
    return dict(logos_acc=logos_acc, behav_acc=behav_acc, tragic_gap=tragic_gap,
                gap_hot=gap_hot, gap_cool=gap_cool,
                gate_hot=gate_hot, gate_cool=gate_cool)


# -----------------------------------------------------------------------------
# 6. MAIN — build, check gradients, train, self-test
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EURIPIDES — THE AKRASIA ENGINE")
    print("the mind knows the wise act; passion decides what is done")
    print("=" * 70)

    task = TragicChoice(n_actions=4, delib=4, affect=4, seed=1)
    model = AkrasiaEngine(D=task.D, H=24, K=task.K, seed=60)

    train = task.sample(2000, RNG)
    test = task.sample(2000, RNG)

    # ---- (a) mandatory gradient check on a small batch ----------------------
    small = task.sample(16, RNG)
    worst_rel = gradient_check(model, small)
    print(f"\n[gradient check] worst relative error = {worst_rel:.2e}  "
          f"({'PASS' if worst_rel < 1e-4 else 'FAIL'})")
    assert worst_rel < 1e-4, "analytic gradients disagree with finite differences"

    # ---- (b) training -------------------------------------------------------
    opt = Adam(model.P, lr=4e-3)
    epochs, bs = 220, 256
    print("\n[training]")
    for ep in range(epochs):
        idx = RNG.permutation(train["X"].shape[0])
        for s in range(0, len(idx), bs):
            sl = idx[s:s + bs]
            mb = {k: (v[sl] if hasattr(v, "__len__") else v) for k, v in train.items()}
            fwd = model.forward(mb["X"])
            L, cache = model.loss(fwd, mb)
            grads = model.backward(fwd, mb, cache)
            opt.step(model.P, grads)
        if ep % 40 == 0 or ep == epochs - 1:
            full = model.forward(train["X"])
            Lf, _ = model.loss(full, train)
            m = evaluate(model, train)
            print(f"  epoch {ep:3d}  loss {Lf:5.3f}  "
                  f"logos_acc {m['logos_acc']:.2f}  behav_acc {m['behav_acc']:.2f}")

    # ---- (c) report the Euripidean quantities on held-out data --------------
    m = evaluate(model, test)
    print("\n[held-out evaluation]")
    print(f"  logos accuracy (knows the wise act) : {m['logos_acc']:.3f}")
    print(f"  behaviour accuracy (predicts deed)  : {m['behav_acc']:.3f}")
    print(f"  gate when passion HOT (p>0.5)       : {m['gate_hot']:.3f}")
    print(f"  gate when passion COOL (p<=0.5)     : {m['gate_cool']:.3f}")
    print(f"  tragic gap in HOT  situations       : {m['gap_hot']:.3f}")
    print(f"  tragic gap in COOL situations       : {m['gap_cool']:.3f}")

    # ---- (d) self-tests: the model must reproduce Euripides' thesis ---------
    print("\n[self-tests]")
    assert m["logos_acc"] > 0.90, "the mind should reliably KNOW the wise act"
    assert m["behav_acc"] > 0.85, "the model should predict what is actually DONE"
    assert m["gate_hot"] > m["gate_cool"] + 0.25, "passion must open the gate"
    assert m["gap_hot"] > m["gap_cool"] + 0.20, \
        "the tragic gap must concentrate in the passionate regime"
    print("  PASS: logos stays wise while the gate opens under passion")
    print("  PASS: the agent acts against its own knowledge precisely when hot")

    # ---- (e) one worked 'Medea' case ----------------------------------------
    hot = task.sample(4000, RNG)
    fwd = model.forward(hot["X"])
    lc = fwd["vL"].argmax(axis=1)
    ac = fwd["z"].argmax(axis=1)
    # find a case where the mind knows the wise act yet does the passionate one
    mask = (lc == hot["wise"]) & (ac == hot["passion"]) & (hot["passion"] != hot["wise"]) & hot["possessed"]
    if mask.any():
        i = np.where(mask)[0][0]
        print("\n[a Medea moment]")
        print(f"  passion intensity p = {hot['p'][i]:.2f}  (gate g = {fwd['g'][i]:.2f})")
        print(f"  the mind KNOWS the wise action is #{lc[i]} (= ground-truth wise #{hot['wise'][i]})")
        print(f"  yet the agent DOES action #{ac[i]} (= passionate #{hot['passion'][i]})")
        print("  -> 'I know what evil I do, but passion is master of my plans.'")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — akrasia learned, not hard-coded")
    print("=" * 70)


if __name__ == "__main__":
    main()

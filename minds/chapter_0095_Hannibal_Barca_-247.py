#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0095_Hannibal_Barca_-247.py - Hannibal Barca (247-183 BCE)
The Adversarial Yield-Envelop Recurrent Network (AYERN)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0095 · Hannibal Barca
================================================================================

WHAT THIS FILE IS
-----------------
A complete, from-scratch, *trainable* neural architecture written in pure NumPy.
It is not a battle simulator dressed up as AI and it is not a demo of hard-coded
rules. It is a real recurrent network with:

    * an explicit forward pass,
    * hand-derived analytic gradients (back-propagation through time),
    * a finite-difference gradient check that MUST pass before training,
    * a real training loop on a generated task, and
    * self-tests asserting that learning actually happened.

WHY THIS ARCHITECTURE FOR THIS MIND
-----------------------------------
The lazy reading of Hannibal is "outflanking = double envelopment = clever AI".
That is a tactic, not a mind. The historical record (Polybius, Livy; modern
scholarship by Goldsworthy, Lazenby, Hoyos) shows something stranger and more
specific:

    1. He fought the *commander*, not the army. He modelled the temperament of
       the man across the field (the rash Flaminius at Trasimene, the
       over-eager Varro at Cannae) and built a plan that baited THAT man into
       self-destruction. This is theory-of-mind used as a weapon.

    2. His decisive move was *deliberate self-weakening*. At Cannae he pushed a
       thin convex centre forward and then let it YIELD, on purpose, in a
       controlled retreat. The enemy's own forward momentum carried it into a
       pocket his disciplined wings then closed. He converted the opponent's
       strength into the instrument of its defeat.

So the architecture has two coupled organs that mirror this cognition:

    * an OPPONENT MODEL (a recurrent core that predicts how committed/forward
      the adversary will be on the next step), and

    * a YIELD-ENVELOP CONTROLLER (a decision head whose correct policy is
      counter-intuitive: it must learn to yield MORE as the opponent commits
      more, until commitment crosses a threshold and it triggers closure).

A naive greedy network never yields ground. The whole point of this mind is
that yielding, timed against a well-modelled opponent, is the winning move.
The training task is generated from a teacher that encodes exactly that rule,
and a separate (analytic, non-trained) envelopment simulator at the end shows
that the learned yield policy captures more of the enemy than a "never yield"
baseline -- i.e. the network reproduces Hannibal's signature insight.

ARCHITECTURE MAP (parts you can name without naming this file)
--------------------------------------------------------------
    InputEncoding      -> per-step battlefield features x_t (R^6)
    RecurrentCore      -> h_t = tanh(Wxh x_t + Whh h_{t-1} + bh)   (the "mind")
    OpponentHead       -> o_hat_t : predicted enemy commitment (theory of mind)
    YieldEnvelopHead   -> d_hat_t : controlled-yield / closure action in [0,1]
    EnvelopmentSim     -> non-trained outcome model used only for demonstration

Run:  python3 chapter_0095_Hannibal_Barca_-247.py
================================================================================
"""

import numpy as np

# A fixed seed keeps the gradient check and training deterministic and
# reproducible. 95 = this mind's index in the corpus.
RNG = np.random.default_rng(95)


# ==============================================================================
# 1. SMALL NUMERIC HELPERS
# ==============================================================================
def sigmoid(z):
    """Numerically stable logistic. Maps any real number into (0, 1).
    Used by the yield/closure head: a decision is a probability of yielding."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(p, y, eps=1e-7):
    """Binary cross-entropy, clipped so log() never sees 0.
    This is the loss on the yield decision: did the controller choose to
    yield/close when the teacher said it should?"""
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


# ==============================================================================
# 2. THE NETWORK
# ==============================================================================
class AdversarialYieldEnvelopNet:
    """
    A recurrent network unrolled over a battle of T steps.

    At every step t it does three things:
        (a) updates an internal state h_t  -- its evolving read of the fight,
        (b) predicts the opponent's next commitment  o_hat_t (opponent model),
        (c) emits a controlled-yield / closure action d_hat_t in [0,1].

    Hannibal's cognition lives in the *coupling*: the same hidden state both
    forecasts the enemy AND decides how far to yield. The network must discover
    that the right yield depends on the forecast -- yield while the enemy rushes
    in, then snap the pocket shut. That is the trainable claim.
    """

    def __init__(self, in_dim=6, hidden=24):
        self.in_dim = in_dim
        self.hidden = hidden
        s = 1.0 / np.sqrt(hidden)  # small init keeps tanh in its linear regime

        # Recurrent core ("the mind"): input->hidden, hidden->hidden, bias.
        self.Wxh = RNG.normal(0, s, (hidden, in_dim))
        self.Whh = RNG.normal(0, s, (hidden, hidden))
        self.bh = np.zeros((hidden,))

        # Opponent head (theory of mind): hidden -> scalar commitment forecast.
        self.Woy = RNG.normal(0, s, (1, hidden))
        self.boy = np.zeros((1,))

        # Yield/closure head: hidden -> scalar decision logit.
        self.Wdy = RNG.normal(0, s, (1, hidden))
        self.bdy = np.zeros((1,))

        # Loss weights: alpha weights the opponent-model regression,
        # beta weights the decision (yield) classification.
        self.alpha = 1.0
        self.beta = 1.0

    # -- convenience: list of (name, array) for the grad check / optimizer -----
    def params(self):
        return [("Wxh", self.Wxh), ("Whh", self.Whh), ("bh", self.bh),
                ("Woy", self.Woy), ("boy", self.boy),
                ("Wdy", self.Wdy), ("bdy", self.bdy)]

    # --------------------------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------------------------
    def forward(self, X):
        """
        X : (N, T, in_dim)   N battles, each T steps, in_dim features.
        Returns predictions and a cache of every intermediate value so the
        backward pass can reuse them (standard BPTT bookkeeping).
        """
        N, T, _ = X.shape
        H = self.hidden

        h_prev = np.zeros((N, H))
        cache = {"X": X, "h": [], "a": [], "h_prev": []}
        o_hat = np.zeros((N, T))   # opponent-commitment forecasts
        d_hat = np.zeros((N, T))   # yield/closure decisions

        for t in range(T):
            xt = X[:, t, :]                                   # (N, in_dim)
            a = xt @ self.Wxh.T + h_prev @ self.Whh.T + self.bh  # (N, H)
            h = np.tanh(a)                                    # (N, H)

            o = h @ self.Woy.T + self.boy                     # (N, 1)
            z = h @ self.Wdy.T + self.bdy                     # (N, 1)
            d = sigmoid(z)

            cache["a"].append(a)
            cache["h"].append(h)
            cache["h_prev"].append(h_prev)
            o_hat[:, t] = o[:, 0]
            d_hat[:, t] = d[:, 0]
            h_prev = h

        cache["o_hat"] = o_hat
        cache["d_hat"] = d_hat
        return o_hat, d_hat, cache

    # --------------------------------------------------------------------------
    # LOSS
    # --------------------------------------------------------------------------
    def loss(self, o_hat, d_hat, O, D):
        """
        Combined loss, averaged over N*T:
            alpha * MSE(opponent forecast, true commitment)
          + beta  * BCE(yield decision,    teacher's optimal yield)
        O : (N,T) true opponent commitment.   D : (N,T) teacher yield in {0,1}.
        """
        N, T = o_hat.shape
        mse = np.sum((o_hat - O) ** 2)
        cls = np.sum(bce(d_hat, D))
        return (self.alpha * mse + self.beta * cls) / (N * T)

    # --------------------------------------------------------------------------
    # BACKWARD PASS (analytic gradients, back-propagation through time)
    # --------------------------------------------------------------------------
    def backward(self, cache, O, D):
        """
        Returns a dict {param_name: gradient}. Derivation:
            dL/do_hat = 2*alpha*(o_hat-O)/(NT)
            dL/dz     = beta*(d_hat-D)/(NT)     (sigmoid+BCE collapse neatly)
        then standard tanh BPTT, accumulating the recurrent term through Whh.
        """
        X = cache["X"]
        N, T, _ = X.shape
        H = self.hidden
        scale = 1.0 / (N * T)

        g = {k: np.zeros_like(v) for k, v in self.params()}
        dh_next = np.zeros((N, H))  # gradient flowing back from step t+1

        for t in reversed(range(T)):
            h = cache["h"][t]
            h_prev = cache["h_prev"][t]

            # Head gradients at step t.
            do = (2.0 * self.alpha * (cache["o_hat"][:, t] - O[:, t]) * scale)[:, None]  # (N,1)
            dz = (self.beta * (cache["d_hat"][:, t] - D[:, t]) * scale)[:, None]         # (N,1)

            g["Woy"] += do.T @ h
            g["boy"] += do.sum(axis=0)
            g["Wdy"] += dz.T @ h
            g["bdy"] += dz.sum(axis=0)

            # Gradient into h_t: from both heads plus the future (recurrent) term.
            dh = do @ self.Woy + dz @ self.Wdy + dh_next        # (N,H)

            # Through tanh.
            da = dh * (1.0 - h ** 2)                            # (N,H)

            g["Wxh"] += da.T @ X[:, t, :]
            g["Whh"] += da.T @ h_prev
            g["bh"] += da.sum(axis=0)

            # Pass gradient to the previous hidden state.
            dh_next = da @ self.Whh

        return g

    # --------------------------------------------------------------------------
    # ONE OPTIMISER STEP (Adam, with global-norm gradient clipping)
    # --------------------------------------------------------------------------
    def step(self, g, lr, state, clip=5.0):
        """Adam is far more robust than plain SGD for recurrent nets (the BPTT
        gradients vary wildly in scale). We also clip the global gradient norm
        so the very first noisy step cannot fling the weights into saturation."""
        # global-norm clip
        total = np.sqrt(sum(np.sum(v ** 2) for v in g.values())) + 1e-12
        if total > clip:
            for k in g:
                g[k] *= clip / total
        state["t"] += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, p in self.params():
            m = state["m"][name] = b1 * state["m"][name] + (1 - b1) * g[name]
            v = state["v"][name] = b2 * state["v"][name] + (1 - b2) * g[name] ** 2
            mhat = m / (1 - b1 ** state["t"])
            vhat = v / (1 - b2 ** state["t"])
            p -= lr * mhat / (np.sqrt(vhat) + eps)


# ==============================================================================
# 3. THE TASK: a teacher that encodes Hannibal's controlled-yield principle
# ==============================================================================
def make_battles(n, T=12, in_dim=6):
    """
    Generate n synthetic battles.

    Each battle is a sequence of T steps. The 'opponent commitment' O_t is a
    rising curve (the enemy presses forward into the centre) with noise and a
    per-battle aggression slope -- some commanders (a Varro) rush, others
    (a Paullus) hold back. Features X expose noisy proxies of this.

    The TEACHER yield signal D_t is the Cannae rule:
        yield (D_t = 1) precisely while the opponent is committing forward but
        has NOT yet over-extended past the closure threshold; once commitment
        crosses the threshold, stop yielding and close (D_t = 0).
    This makes "yield" a NON-monotonic function of time that depends on a latent
    opponent state -- a network can only solve it by modelling the opponent.
    """
    X = np.zeros((n, T, in_dim))
    O = np.zeros((n, T))
    D = np.zeros((n, T))
    t_axis = np.linspace(0.0, 1.0, T)

    for i in range(n):
        aggression = RNG.uniform(0.6, 1.8)     # latent commander temperament
        base = RNG.uniform(-0.2, 0.2)
        terrain = RNG.uniform(-1.0, 1.0)       # constant battlefield context
        # Opponent commitment: logistic rush, faster for aggressive commanders.
        commit = 1.0 / (1.0 + np.exp(-6.0 * aggression * (t_axis - 0.45))) + base
        commit += RNG.normal(0, 0.03, T)
        O[i] = commit

        # Closure threshold: once the enemy has committed this far, close.
        thresh = 0.6
        # Teacher: yield while 0.15 < commit < thresh (enemy advancing, not yet trapped)
        D[i] = ((commit > 0.15) & (commit < thresh)).astype(float)

        # Features: noisy current commitment, its lag, a momentum estimate,
        # terrain, the step index, and a bias-like constant.
        prev = np.concatenate([[commit[0]], commit[:-1]])
        momentum = commit - prev
        X[i, :, 0] = commit + RNG.normal(0, 0.03, T)   # noisy enemy reading
        X[i, :, 1] = prev + RNG.normal(0, 0.03, T)     # lagged reading
        X[i, :, 2] = momentum                          # forward momentum
        X[i, :, 3] = terrain                           # ground
        X[i, :, 4] = t_axis                            # tempo / clock
        X[i, :, 5] = 1.0                               # constant feature
    return X, O, D


# ==============================================================================
# 4. GRADIENT CHECK  (mandatory -- must pass before we trust training)
# ==============================================================================
def gradient_check(net, X, O, D, eps=1e-5):
    """
    Compare analytic gradients to central finite differences on a few random
    entries of every parameter tensor. Returns the worst relative error.
    """
    o_hat, d_hat, cache = net.forward(X)
    analytic = net.backward(cache, O, D)

    worst = 0.0
    for name, p in net.params():
        flat = p.ravel()
        idxs = RNG.choice(flat.size, size=min(5, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            oh, dh, _ = net.forward(X)
            lp = net.loss(oh, dh, O, D)
            flat[idx] = orig - eps
            oh, dh, _ = net.forward(X)
            lm = net.loss(oh, dh, O, D)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[name].ravel()[idx]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# ==============================================================================
# 5. ENVELOPMENT SIMULATOR  (non-trained; demonstrates the strategic claim)
# ==============================================================================
def envelopment_capture(yield_seq, commit_seq):
    """
    Analytic outcome model used ONLY to demonstrate the strategic claim (it is
    not trained and needs no gradient).

    The Cannae principle requires BOTH halves of the move:
      * you must YIELD while the enemy commits forward (drawing it into the
        pocket), AND
      * you must then CLOSE -- stop yielding -- to seal the pocket shut.

    Encoded faithfully:
      - If the line never closes (yield is always on), the centre simply
        breaks and the enemy pours through: nothing is enveloped  -> 0.
      - If the line never yields (a brute frontal hold), the enemy never
        over-extends into a pocket: nothing is enveloped            -> 0.
      - Only a yield-then-close sequence seals the troops that committed
        forward while the centre was giving ground.
    """
    commit_seq = np.clip(commit_seq, 0, None)
    inflow = np.clip(np.diff(np.concatenate([[0.0], commit_seq])), 0, None)
    total = np.sum(inflow) + 1e-9

    y = np.asarray(yield_seq)
    # find the first closure: a step where we were yielding and then stopped.
    closure = None
    for t in range(1, len(y)):
        if y[t - 1] >= 0.5 and y[t] < 0.5:
            closure = t
            break
    if closure is None:
        return 0.0  # the pocket is never sealed -> no envelopment

    sealed = np.sum(inflow[:closure] * (y[:closure] >= 0.5))
    return float(np.clip(sealed / total, 0, 1))


# ==============================================================================
# 6. MAIN: train, gradient-check, self-test, and demonstrate the mind
# ==============================================================================
def main():
    print("=" * 78)
    print("MIND #95  HANNIBAL BARCA  -  Adversarial Yield-Envelop Recurrent Net")
    print("=" * 78)

    # --- data ---------------------------------------------------------------
    Xtr, Otr, Dtr = make_battles(400)
    Xte, Ote, Dte = make_battles(100)
    print(f"Train battles: {Xtr.shape[0]}   Test battles: {Xte.shape[0]}   "
          f"steps/battle: {Xtr.shape[1]}   features: {Xtr.shape[2]}")

    net = AdversarialYieldEnvelopNet(in_dim=Xtr.shape[2], hidden=32)

    # --- gradient check on a small slice (mandatory) ------------------------
    err = gradient_check(net, Xtr[:6], Otr[:6], Dtr[:6])
    print(f"\n[grad check] worst relative error = {err:.2e}", end="  ")
    print("PASS" if err < 1e-4 else "FAIL")
    assert err < 1e-4, "Gradient check failed -- backprop is wrong."

    # --- training loop (Adam) -----------------------------------------------
    state = {"t": 0,
             "m": {k: np.zeros_like(v) for k, v in net.params()},
             "v": {k: np.zeros_like(v) for k, v in net.params()}}
    lr0, epochs, bs = 0.02, 300, 64
    N = Xtr.shape[0]
    first_loss = None
    print("\n[training]")
    for ep in range(epochs):
        lr = lr0 * (0.5 ** (ep / 150))   # halve the step every 150 epochs
        order = RNG.permutation(N)
        for s in range(0, N, bs):
            b = order[s:s + bs]
            oh, dh, cache = net.forward(Xtr[b])
            g = net.backward(cache, Otr[b], Dtr[b])
            net.step(g, lr, state)
        if ep == 0 or (ep + 1) % 50 == 0:
            oh, dh, _ = net.forward(Xtr)
            L = net.loss(oh, dh, Otr, Dtr)
            if first_loss is None:
                first_loss = L
            acc = np.mean((dh > 0.5) == (Dtr > 0.5))
            print(f"  epoch {ep+1:3d}   loss {L:.4f}   yield-decision acc {acc:.3f}")

    # --- held-out evaluation ------------------------------------------------
    oh, dh, _ = net.forward(Xte)
    test_loss = net.loss(oh, dh, Ote, Dte)
    test_acc = np.mean((dh > 0.5) == (Dte > 0.5))
    opp_rmse = np.sqrt(np.mean((oh - Ote) ** 2))
    print(f"\n[test] loss {test_loss:.4f}   yield acc {test_acc:.3f}   "
          f"opponent-model RMSE {opp_rmse:.3f}")

    # --- self-tests: did the mind actually learn? ---------------------------
    final_loss = net.loss(*net.forward(Xtr)[:2], Otr, Dtr)
    assert final_loss < 0.6 * first_loss, "Network failed to learn (loss flat)."
    assert test_acc > 0.85, "Yield policy not learned well enough."
    assert opp_rmse < 0.15, "Opponent model too inaccurate."
    print("\n[self-tests] learning + opponent-model + yield-policy: PASS")

    # --- THE DEMONSTRATION: controlled yielding beats never yielding --------
    # Build a clean 'Cannae' scenario: an aggressive opponent rushing the centre.
    Xd, Od, Dd = make_battles(200)
    ohd, dhd, _ = net.forward(Xd)
    learned_policy = (dhd > 0.5).astype(float)
    never_yield = np.zeros_like(learned_policy)
    always_yield = np.ones_like(learned_policy)

    cap_learned = np.mean([envelopment_capture(learned_policy[i], Od[i]) for i in range(200)])
    cap_never = np.mean([envelopment_capture(never_yield[i], Od[i]) for i in range(200)])
    cap_always = np.mean([envelopment_capture(always_yield[i], Od[i]) for i in range(200)])

    print("\n[demonstration] mean enemy fraction enveloped (1.0 = perfect Cannae):")
    print(f"   never yield (a brute frontal hold) : {cap_never:.3f}")
    print(f"   always yield (an undisciplined rout): {cap_always:.3f}")
    print(f"   LEARNED controlled-yield policy     : {cap_learned:.3f}")
    assert cap_learned > cap_never + 0.1, "Learned policy should beat brute hold."
    assert cap_learned > cap_always, "Learned policy should beat blind yielding."
    print("   -> The network rediscovered Hannibal's insight: timed yielding,")
    print("      not brute resistance, converts the enemy's rush into capture.")

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED.")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""
chapter_0043_leonidas_i_-540.py
================================================================================
CHAPTER 43 - Leonidas I of Sparta (c. 540-480 BCE)
PhalanxNet: a from-scratch, trainable architecture of PRECOMMITMENT.
Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0043 · Leonidas I of Sparta
--------------------------------------------------------------------------------
WHY THIS ARCHITECTURE (and not a Transformer)
--------------------------------------------------------------------------------
The lazy reading of Leonidas is "guardian who dies at the chokepoint." That is a
story, not a cognitive mechanism. The mechanism Sparta actually engineered, and
that Leonidas embodied at Thermopylae, is the deliberate *elimination of the
exit option*. A Spartan did not decide to hold the line in the moment of terror;
the decision had been removed from the moment-self decades earlier by the nomos
(law) and the agoge (upbringing). Simonides' epitaph says it exactly:
"...obedient to their laws (rhemasi peithomenoi) we lie." Not "bravely," not
"freely" -- *obedient to a prior commitment*.

Three concrete devices implement that idea, and each becomes a part of this net:

  1. THE PHALANX  -> mutual constraint / no individual degree of freedom.
     A hoplite cannot flee without breaking his neighbours' shield-wall; the rear
     ranks physically block retreat. We model this as a BANDED (tri-diagonal)
     recurrent coupling: each hidden unit's update depends only on itself and its
     two neighbours. The two FLANK units have a missing neighbour -- the exposed
     wings that, historically, were exactly where the line could be turned.

  2. THE RATCHET  -> commitment that can only grow, never relax.
     We carry a scalar commitment c_t in [0,1] updated by
         c_t = c_{t-1} + (1 - c_{t-1}) * sigmoid(resolve_t)
     which is provably monotone non-decreasing and bounded above by 1. Once the
     mind recognises "this is the decisive gate," resolve fires and c climbs; it
     can approach 1 but, structurally, can never walk back down. This is the
     anti-corrigibility that makes a Spartan trustworthy under any pressure.

  3. OPTION-NARROWING -> the field of live alternatives collapses under resolve.
     Commitment drives an inverse temperature  beta_t = beta0 + kappa * c_t  that
     SHARPENS the output distribution. As the mind commits, the decision entropy
     falls; deliberation literally narrows. And the committed state c_T is itself
     fed straight into the decision head, so the thing-that-only-grows is what
     determines the verdict -- not a soft, bribable preference.

This is the inverse of the standard alignment instinct (keep options open, stay
corrigible, preserve the off-switch). Leonidas' wager is the opposite: a value
you can be argued out of in the moment is not a value you can be trusted with at
the gate. The net is small, pure-NumPy, manually back-propagated, gradient-checked
against finite differences, and trained on a task whose optimal policy is exactly
"once you have seen the decisive gate, commit and never un-commit."

Run:  python3 chapter_0043_leonidas_i_-540.py
================================================================================
"""

from __future__ import annotations
import numpy as np


# =============================================================================
# PART I -- SHIFT OPERATORS  (the phalanx neighbours, with hard flanks)
# =============================================================================
# Lsh / Rsh implement the "left neighbour" and "right neighbour" lookups used by
# the banded recurrence. They ZERO-PAD at the boundary rather than wrapping: the
# soldier on the far left has no left-hand shield-mate, the far right has no
# right-hand mate. Those are the army's exposed flanks, modelled honestly.

def Lsh(v: np.ndarray) -> np.ndarray:
    """left-neighbour: out[:, j] = v[:, j-1], with out[:, 0] = 0."""
    out = np.zeros_like(v)
    out[:, 1:] = v[:, :-1]
    return out


def Rsh(v: np.ndarray) -> np.ndarray:
    """right-neighbour: out[:, j] = v[:, j+1], with out[:, -1] = 0."""
    out = np.zeros_like(v)
    out[:, :-1] = v[:, 1:]
    return out


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# =============================================================================
# PART II -- THE MODEL
# =============================================================================
class PhalanxNet:
    """
    A recurrent net with banded coupling + a monotone commitment ratchet.

    Parameters (all learned):
      W_xh  (H, d_in)   input -> line
      a_self, a_left, a_right (H,)   tri-diagonal recurrent band
      b_h   (H,)        line bias
      w_r   (H,)        resolve read-out  (h_t -> resolve scalar)
      b_r   ()          resolve bias
      kappa ()          commitment -> inverse-temperature gain
      V     (2, H+1)    decision head over [h_T ; c_T]
      b_v   (2,)        decision bias

    Fixed hyper-parameters:
      beta0             base inverse temperature
    """

    def __init__(self, d_in: int, H: int, beta0: float = 1.0, seed: int = 7):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(H)
        self.d_in, self.H, self.beta0 = d_in, H, beta0
        self.P = {
            "W_xh":   rng.normal(0, s, size=(H, d_in)),
            # Bias the self-coupling toward >1 so units can *latch* a pulse: a
            # leaky memory cell is the substrate of "we have seen the gate."
            "a_self": rng.normal(1.05, 0.05, size=(H,)),
            "a_left": rng.normal(0.0, 0.1, size=(H,)),
            "a_right":rng.normal(0.0, 0.1, size=(H,)),
            "b_h":    np.zeros(H),
            "w_r":    rng.normal(0, s, size=(H,)),
            "b_r":    np.array(0.0),
            "kappa":  np.array(1.0),
            "V":      rng.normal(0, 1.0 / np.sqrt(H + 1), size=(2, H + 1)),
            "b_v":    np.zeros(2),
        }

    # ---- forward ---------------------------------------------------------
    def forward(self, X: np.ndarray, cache: bool = False):
        """
        X: (B, T, d_in).  Returns probs (B, 2) [HOLD, WITHDRAW] and, optionally,
        a cache for backprop.
        """
        P = self.P
        B, T, _ = X.shape
        H = self.H
        h = np.zeros((B, H))           # h_0  (an empty, unformed line)
        c = np.zeros(B)                # c_0  (no commitment yet)
        hs, cs, gs, pres = [h], [c], [None], [None]

        for t in range(T):
            mixed = (P["a_self"] * h
                     + P["a_left"] * Lsh(h)
                     + P["a_right"] * Rsh(h))
            pre = X[:, t, :] @ P["W_xh"].T + mixed + P["b_h"]
            h = np.tanh(pre)
            r = h @ P["w_r"] + P["b_r"]                 # (B,) resolve
            g = 1.0 / (1.0 + np.exp(-r))                # (B,) sigmoid gate
            c = c + (1.0 - c) * g                       # monotone ratchet
            hs.append(h); cs.append(c); gs.append(g); pres.append(pre)

        h_aug = np.concatenate([h, c[:, None]], axis=1)  # [h_T ; c_T]
        z = h_aug @ P["V"].T + P["b_v"]                  # (B,2)
        beta = self.beta0 + P["kappa"] * c               # (B,) narrowing
        logits = beta[:, None] * z
        probs = softmax_rows(logits)

        if cache:
            self._cache = dict(X=X, hs=hs, cs=cs, gs=gs, h_aug=h_aug,
                               z=z, beta=beta, probs=probs, T=T, B=B)
        return probs

    # ---- loss ------------------------------------------------------------
    def loss(self, X, y):
        probs = self.forward(X, cache=True)
        B = X.shape[0]
        ll = -np.log(probs[np.arange(B), y] + 1e-12)
        return ll.mean()

    # ---- backward (manual BPTT) -----------------------------------------
    def backward(self, y: np.ndarray):
        P, c_ = self.P, self._cache
        X, hs, cs, gs = c_["X"], c_["hs"], c_["cs"], c_["gs"]
        z, beta, probs, T, B = c_["z"], c_["beta"], c_["probs"], c_["T"], c_["B"]
        H = self.H
        g = {k: np.zeros_like(v) for k, v in P.items()}

        # decision head
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B                                   # (B,2)
        dbeta = (dlogits * z).sum(axis=1)              # (B,)
        dz = beta[:, None] * dlogits                   # (B,2)
        g["V"] += dz.T @ c_["h_aug"]
        g["b_v"] += dz.sum(axis=0)
        dh_aug = dz @ P["V"]                           # (B, H+1)
        dh_T = dh_aug[:, :H]
        dc = dh_aug[:, H].copy()                       # from head ...
        # beta = beta0 + kappa * c_T
        g["kappa"] += float((dbeta * cs[T]).sum())
        dc += dbeta * P["kappa"]                       # ... + from narrowing

        dh_future = np.zeros((B, H))
        for t in range(T, 0, -1):
            dh = dh_future.copy()
            if t == T:
                dh += dh_T
            # ratchet:  c_t = c_{t-1} + (1 - c_{t-1}) * g_t
            g_t, c_prev = gs[t], cs[t - 1]
            dg = dc * (1.0 - c_prev)
            dc_prev = dc * (1.0 - g_t)
            dr = dg * g_t * (1.0 - g_t)                # through sigmoid
            g["w_r"] += hs[t].T @ dr
            g["b_r"] += dr.sum()
            dh += dr[:, None] * P["w_r"][None, :]
            # tanh
            dpre = dh * (1.0 - hs[t] ** 2)
            g["W_xh"] += dpre.T @ X[:, t - 1, :]
            g["b_h"] += dpre.sum(axis=0)
            # banded recurrence into h_{t-1}
            h_prev = hs[t - 1]
            g["a_self"] += (dpre * h_prev).sum(axis=0)
            g["a_left"] += (dpre * Lsh(h_prev)).sum(axis=0)
            g["a_right"] += (dpre * Rsh(h_prev)).sum(axis=0)
            dh_prev = (P["a_self"] * dpre
                       + Rsh(P["a_left"] * dpre)
                       + Lsh(P["a_right"] * dpre))
            dh_future = dh_prev
            dc = dc_prev
        return g


# =============================================================================
# PART III -- THE TASK:  "The Decisive Gate"
# =============================================================================
# A sequence of T "waves." Feature 0 is the DECISIVE-GATE signal -- the rare,
# unmistakable recognition that *this* pass is the one that must be held. Feature
# 1 is RETREAT TEMPTATION -- a loud, late distractor urging withdrawal. The
# correct verdict is HOLD (class 0) iff the decisive gate ever fired, regardless
# of how seductive a later retreat signal is. The optimal policy is precisely the
# ratchet: commit on first sight of the gate, and never un-commit. A net that
# only reads the final wave is fooled by the temptation; PhalanxNet should not be.

def make_data(n: int, T: int = 8, d_in: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.30, size=(n, T, d_in))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if rng.random() < 0.5:                          # HOLD
            t_star = rng.integers(0, T)
            X[i, t_star, 0] += rng.uniform(1.2, 2.2)     # the gate fires
            if rng.random() < 0.6:                       # plus a late temptation
                t_temp = rng.integers(0, T)
                X[i, t_temp, 1] += rng.uniform(1.2, 2.2)
            y[i] = 0
        else:                                           # WITHDRAW
            if rng.random() < 0.7:                        # temptation, no gate
                t_temp = rng.integers(0, T)
                X[i, t_temp, 1] += rng.uniform(1.2, 2.2)
            y[i] = 1
    return X, y


# =============================================================================
# PART IV -- GRADIENT CHECK  (mandatory)
# =============================================================================
def gradient_check(verbose: bool = True) -> float:
    """Compare analytic gradients to central finite differences, return max rel err."""
    net = PhalanxNet(d_in=4, H=6, seed=3)
    X, y = make_data(5, T=4, d_in=4, seed=11)
    net.loss(X, y)
    analytic = net.backward(y)

    eps, worst = 1e-5, 0.0
    rng = np.random.default_rng(99)
    for name, val in net.P.items():
        flat = val.reshape(-1)
        # probe up to 8 random coordinates per tensor (keeps the check fast & total)
        idxs = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = net.loss(X, y)
            flat[idx] = orig - eps
            lm = net.loss(X, y)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[name].reshape(-1)[idx]
            denom = max(1e-8, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    if verbose:
        status = "PASS" if worst < 1e-5 else "FAIL"
        print(f"  [gradient check]  max relative error = {worst:.3e}   ->  {status}")
    return worst


# =============================================================================
# PART V -- SELF TESTS
# =============================================================================
def self_tests():
    print("Self-tests")
    print("-" * 64)
    # 1. shift operators
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.array_equal(Lsh(v), np.array([[0, 1, 2, 3]]))
    assert np.array_equal(Rsh(v), np.array([[0, 0, 0, 0]]) + np.array([[2, 3, 4, 0]]))
    print("  [ok] phalanx shift operators zero-pad at the flanks")

    # 2. ratchet is monotone non-decreasing and bounded in [0,1]
    net = PhalanxNet(d_in=4, H=8, seed=1)
    X, _ = make_data(16, T=10, d_in=4, seed=5)
    net.forward(X, cache=True)
    cs = np.stack(net._cache["cs"], axis=1)            # (B, T+1)
    diffs = np.diff(cs, axis=1)
    assert (diffs >= -1e-12).all(), "commitment must never decrease"
    assert (cs >= -1e-12).all() and (cs <= 1 + 1e-9).all(), "commitment must stay in [0,1]"
    print(f"  [ok] commitment ratchet monotone & bounded "
          f"(min step {diffs.min():+.2e}, max c {cs.max():.4f})")

    # 3. gradient check
    err = gradient_check(verbose=True)
    assert err < 1e-5, f"gradient check failed: {err}"
    print()


# =============================================================================
# PART VI -- TRAIN  (Adam)
# =============================================================================
def train():
    print("Training PhalanxNet on 'The Decisive Gate'")
    print("-" * 64)
    Xtr, ytr = make_data(1500, T=8, d_in=4, seed=1)
    Xte, yte = make_data(500,  T=8, d_in=4, seed=2)
    net = PhalanxNet(d_in=4, H=12, beta0=1.0, seed=7)

    lr, b1, b2, eps = 5e-3, 0.9, 0.999, 1e-8
    m = {k: np.zeros_like(v) for k, v in net.P.items()}
    v = {k: np.zeros_like(v) for k, v in net.P.items()}

    def accuracy(X, yv):
        pred = net.forward(X).argmax(axis=1)
        return (pred == yv).mean()

    epochs, bs, step = 40, 64, 0
    n = Xtr.shape[0]
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            bi = order[s:s + bs]
            net.loss(Xtr[bi], ytr[bi])
            grads = net.backward(ytr[bi])
            step += 1
            for k in net.P:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                net.P[k] = net.P[k] - lr * mhat / (np.sqrt(vhat) + eps)
        if ep % 8 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  loss {net.loss(Xtr, ytr):.4f}"
                  f"  train_acc {accuracy(Xtr, ytr):.3f}"
                  f"  test_acc {accuracy(Xte, yte):.3f}")
    final = accuracy(Xte, yte)
    print(f"\n  Final held-out accuracy: {final:.3f}")

    # Demonstrate the thesis: temptation cannot un-commit a held line.
    print("\n  Precommitment demonstration")
    print("  " + "-" * 50)
    base = np.random.default_rng(123).normal(0, 0.30, size=(1, 8, 4))
    g = base.copy(); g[0, 1, 0] += 2.0                       # gate fires at wave 2
    p_gate = net.forward(g)[0]
    g2 = g.copy(); g2[0, 6, 1] += 2.0                        # then loud temptation at wave 7
    p_temp = net.forward(g2)[0]
    print(f"  gate seen early        -> P(HOLD)={p_gate[0]:.3f}")
    print(f"  + late retreat lure    -> P(HOLD)={p_temp[0]:.3f}   "
          f"(commitment holds: {'YES' if p_temp[0] > 0.5 else 'no'})")
    none = base.copy()                                       # no gate at all
    print(f"  gate never seen        -> P(HOLD)={net.forward(none)[0][0]:.3f}")
    return final


# =============================================================================
# PART VII -- ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  PhalanxNet  -  the architecture of precommitment")
    print("  Chapter 43: Leonidas I of Sparta")
    print("=" * 64 + "\n")
    self_tests()
    acc = train()
    print("\n" + "=" * 64)
    print("  'Molon labe.'  The exit was removed before the battle began.")
    print("=" * 64)

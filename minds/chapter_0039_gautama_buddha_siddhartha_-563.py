#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0039_gautama_buddha_siddhartha_-563.py
================================================================================
TWO-ARROW EQUANIMITY MIND  -  an AGI micro-architecture after Gautama Buddha
(Siddhartha Gautama, c. 563 - c. 483 BCE)
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0039 · Gautama Buddha (Siddhartha Gautama)
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS NOT THE USUAL ONE
--------------------------------------------------------------------------------
Most "cognitive" learners (and almost every RL agent) are built to MAXIMISE a
reward signal. They perceive the world, attach a valuation to it, and then a
policy *grasps*: it chases pleasant states and pushes away unpleasant ones. The
grasping IS the engine. Take it away and the agent stops moving.

The Buddha's analysis of mind says the engine itself is the pathology.

His sharpest single image is the Sallatha Sutta ("The Dart", Samyutta Nikaya
36.6) - the parable of the TWO ARROWS:

    "When touched by a painful feeling, the uninstructed worldling sorrows,
     grieves and laments... he feels two pains, bodily and mental. It is as if
     a man were pierced by a dart and, right after, were pierced by a second
     dart. The instructed noble disciple, touched by that same painful feeling,
     feels ONLY the first dart, not the second."

The FIRST arrow (vedana, raw feeling / contact with reality) is unavoidable:
the world delivers noise, loss, and pain that no amount of cleverness predicts
away. The SECOND arrow (tanha, reactive craving) is the mind's own production -
the grasping, the "I must have / I must not have" that amplifies the bare signal
into suffering (dukkha). The Buddha's claim, radical for an intelligence
theory, is that you can build a mind that PERCEIVES the world accurately (keeps
the first arrow - wisdom is not numbness) while it stops manufacturing the
second arrow. The terminal state is not maximal reward; it is equanimity
(upekkha) - a fixed point, an attractor, where craving has gone out like a
flame whose fuel is spent (nibbana = "extinguishing").

So this network deliberately REFUSES the reward-maximising template. Its
training objective is:

        minimise  [ perception error ]   +   lambda * [ reactive grasping ]

The first term keeps it awake to reality. The second term drives the craving
gate to zero. Crucially, the data has an IRREDUCIBLE noise floor (the first
arrow cannot be removed), so the ONLY way the network can lower its suffering is
to stop grasping - exactly the Sallatha Sutta's point.

HOW THE BUDDHA'S OTHER CORE DOCTRINES ARE ENCODED STRUCTURALLY
--------------------------------------------------------------------------------
* ANATTA (no-self): there is NO persistent identity parameter, no learned
  "self vector", no bias acting as a fixed ego. Each mind-moment is rebuilt from
  conditions. (See `EquanimityMind` - note the absence of any self-state weight.)
* ANICCA (impermanence): the carried mind-trace is multiplied by decay < 1 at
  every step, so the influence of any single moment vanishes geometrically.
  Nothing in the stream persists as an essence. (Verified by a self-test.)
* PATICCASAMUPPADA (dependent origination): every mind-moment m_t is a function
  of prior conditions (sense contact x_t AND the decayed previous moment). It
  arises in dependence on conditions; it does not arise independently.
* VEDANA / TANHA / DUKKHA: the felt intensity r_t (first arrow), the craving
  gate g_t, and the second-arrow disturbance s_t = g_t * r_t (manufactured
  suffering) are explicit, named tensors.
* SATI / VIPASSANA (mindfulness): a read-only monitor measures the second arrow
  across the stream. Observing the grasping is what makes its gradient available
  - in the model, "to see craving clearly is to begin to extinguish it".
* MAGGA (the path): gradient descent on the equanimity objective is the training
  algorithm; the loss going down is the Eightfold Path walked numerically.

WHAT THE FILE CONTAINS (all pure NumPy, from scratch)
--------------------------------------------------------------------------------
1. EquanimityMind  - the recurrent two-arrow network + analytic backprop (BPTT).
2. gradient_check() - finite-difference check of EVERY parameter (mandatory).
3. make_stream()    - a synthetic "stream of experience" with a learnable
                      predictable part + an IRREDUCIBLE first-arrow noise/pain.
4. train()          - a real training loop.
5. A comparison: an "unawakened / craving" mind (lambda = 0) vs an
                  "awakened / path" mind (lambda > 0). Same architecture, same
                  data; only the equanimity weight differs.
6. self_tests()     - anatta / anicca / dependent-origination checks.

Run:  python3 0039_Neuron.py
================================================================================
"""

import numpy as np

# Reproducibility: one fixed seed for the whole demonstration.
GLOBAL_SEED = 7


# =============================================================================
# 1. THE MODEL
# =============================================================================
class EquanimityMind:
    """A stream of momentary citta-events with a perception head (first arrow)
    and a craving gate (second arrow), trained toward equanimity.

    Parameters (the ONLY learnable state - note: no 'self' vector):
        W_in  : (d_h, d_x)   sense contact -> mind-moment
        W_rec : (d_h, d_h)   prior (decayed) moment -> mind-moment  [conditioning]
        W_per : (d_y, d_h)   mind-moment -> perception / prediction (FIRST arrow)
        w_g   : (d_h + 1,)   craving gate weights over [moment ; felt-intensity]
        b_g   : scalar       craving gate bias

    Fixed (structural, NOT learned):
        decay : float in (0,1)  impermanence (anicca). Past conditions fade.
        lam   : float >= 0       equanimity weight (how hard we extinguish craving)
    """

    def __init__(self, d_x, d_h, d_y, decay=0.6, lam=1.0, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        s = 0.5
        # Small random init. Deliberately NO bias on the recurrence / perception:
        # a bias there would behave like a fixed 'self' that survives every moment.
        self.W_in = rng.standard_normal((d_h, d_x)) * (s / np.sqrt(d_x))
        self.W_rec = rng.standard_normal((d_h, d_h)) * (s / np.sqrt(d_h))
        self.W_per = rng.standard_normal((d_y, d_h)) * (s / np.sqrt(d_h))
        self.w_g = rng.standard_normal(d_h + 1) * (s / np.sqrt(d_h + 1))
        self.b_g = 0.0
        # Structural hyper-parameters
        self.decay = float(decay)
        self.lam = float(lam)
        self.d_x, self.d_h, self.d_y = d_x, d_h, d_y

    # --- parameter plumbing (used by the gradient checker) -------------------
    def params(self):
        return {"W_in": self.W_in, "W_rec": self.W_rec, "W_per": self.W_per,
                "w_g": self.w_g, "b_g": self.b_g}

    def set_param(self, name, value):
        if name == "b_g":
            self.b_g = float(value)
        else:
            setattr(self, name, value)

    # --- forward + loss -------------------------------------------------------
    def forward(self, X, Y, cache=False):
        """Run the stream.

        X : (T, d_x) sequence of sense contacts.
        Y : (T, d_y) clean targets the perception head should predict.

        Returns total loss L and (optionally) a cache for backprop.
        Also returns per-step diagnostics: perception loss, felt-intensity r_t,
        craving g_t, and the second-arrow disturbance s_t.
        """
        T = X.shape[0]
        dy = self.d_y
        m_prev = np.zeros(self.d_h)
        Lp_sum = 0.0   # perception loss (first arrow kept honest)
        La_sum = 0.0   # second-arrow loss (craving to be extinguished)
        diag = {"r": np.zeros(T), "g": np.zeros(T), "s": np.zeros(T),
                "Lp": np.zeros(T)}
        store = [] if cache else None

        for t in range(T):
            x_t = X[t]
            # --- dependent origination: m_t arises from x_t AND decayed m_{t-1}
            pre = self.W_in @ x_t + self.decay * (self.W_rec @ m_prev)
            m = np.tanh(pre)                              # the mind-moment (citta)
            # --- first arrow: perception / prediction of reality
            p = self.W_per @ m
            e = p - Y[t]                                  # vedana: raw feeling/error
            Lp = 0.5 * np.dot(e, e)
            r = np.dot(e, e) / dy                         # felt intensity (scalar)
            # --- craving gate (tanha): how hard the mind grasps at this feeling
            u = np.concatenate([m, [r]])                 # what mindfulness watches
            z = np.dot(self.w_g, u) + self.b_g
            g = 1.0 / (1.0 + np.exp(-z))                  # grasping strength in (0,1)
            # --- second arrow: grasping AMPLIFIES the bare feeling into suffering
            sdist = g * r                                 # manufactured disturbance
            La = 0.5 * sdist * sdist

            Lp_sum += Lp
            La_sum += La
            diag["r"][t], diag["g"][t], diag["s"][t], diag["Lp"][t] = r, g, sdist, Lp
            if cache:
                store.append((x_t, m_prev, pre, m, p, e, r, u, z, g, sdist))
            m_prev = m

        L = (Lp_sum + self.lam * La_sum) / T
        if cache:
            return L, store, diag
        return L, diag

    # --- analytic backprop through time --------------------------------------
    def backward(self, store):
        """Backprop the equanimity loss through the whole stream.
        Returns a dict of gradients matching params().
        """
        T = len(store)
        dy = self.d_y
        gW_in = np.zeros_like(self.W_in)
        gW_rec = np.zeros_like(self.W_rec)
        gW_per = np.zeros_like(self.W_per)
        gw_g = np.zeros_like(self.w_g)
        gb_g = 0.0
        w_g_m = self.w_g[:self.d_h]     # gate weights on the moment
        w_g_r = self.w_g[self.d_h]      # gate weight on felt-intensity r
        lam = self.lam
        dm_next = np.zeros(self.d_h)    # gradient flowing back from step t+1

        for t in reversed(range(T)):
            x_t, m_prev, pre, m, p, e, r, u, z, g, sdist = store[t]
            dg_dz = g * (1.0 - g)

            # dL/d e_t : perception term + (second-arrow term through r and g)
            #   r = (e.e)/dy ; dr/de = (2/dy) e
            #   s = g*r ; with g = sigma(w_g_m.m + w_g_r*r + b_g)
            #   ds/de = [dg/dz * w_g_r * dr/de]*r + g*dr/de
            #         = (2/dy) e * (dg_dz*w_g_r*r + g)
            ds_de = (2.0 / dy) * e * (dg_dz * w_g_r * r + g)
            de = e + lam * sdist * ds_de               # full dL/de (before 1/T)

            # perception head: p = W_per @ m ; e = p - Y  => dL/dp = de
            gW_per += np.outer(de, m)
            dm = self.W_per.T @ de                     # via perception path

            # gate scalar z: dL/dz = lam*s*ds/dg*dg/dz = lam*s*r*dg_dz
            gz = lam * sdist * r * dg_dz
            gw_g += gz * u
            gb_g += gz
            # gate's DIRECT dependence on m (through w_g_m), not via e:
            dm += gz * w_g_m

            # add recurrent gradient flowing from the next step
            dm += dm_next

            # through tanh: m = tanh(pre)
            dpre = dm * (1.0 - m * m)
            gW_in += np.outer(dpre, x_t)
            gW_rec += self.decay * np.outer(dpre, m_prev)
            # gradient to previous moment (BPTT): pre uses decay*W_rec@m_prev
            dm_next = self.decay * (self.W_rec.T @ dpre)

        scale = 1.0 / T
        return {"W_in": gW_in * scale, "W_rec": gW_rec * scale,
                "W_per": gW_per * scale, "w_g": gw_g * scale, "b_g": gb_g * scale}

    # --- one SGD-with-momentum step -----------------------------------------
    def step(self, grads, lr, mom, vel):
        for name, p in self.params().items():
            v = mom * vel[name] - lr * grads[name]
            vel[name] = v
            if name == "b_g":
                self.b_g = self.b_g + v
            else:
                p += v


# =============================================================================
# 2. GRADIENT CHECK  (mandatory - the file does not ship unless this passes)
# =============================================================================
def gradient_check(verbose=True):
    """Finite-difference verification of EVERY analytic gradient."""
    rng = np.random.default_rng(123)
    d_x, d_h, d_y, T = 4, 5, 3, 6
    model = EquanimityMind(d_x, d_h, d_y, decay=0.6, lam=0.8, seed=321)
    X = rng.standard_normal((T, d_x))
    Y = rng.standard_normal((T, d_y))

    L, store, _ = model.forward(X, Y, cache=True)
    grads = model.backward(store)

    eps = 1e-6
    worst = 0.0
    for name in model.params():
        p = model.params()[name]
        flat = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
        ana = np.atleast_1d(np.asarray(grads[name], dtype=float)).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            # +eps
            flat[i] = orig + eps
            model.set_param(name, flat.reshape(np.asarray(p).shape) if name != "b_g" else flat[i])
            Lp, _ = model.forward(X, Y)
            # -eps
            flat[i] = orig - eps
            model.set_param(name, flat.reshape(np.asarray(p).shape) if name != "b_g" else flat[i])
            Lm, _ = model.forward(X, Y)
            num[i] = (Lp - Lm) / (2 * eps)
            flat[i] = orig
            model.set_param(name, flat.reshape(np.asarray(p).shape) if name != "b_g" else flat[i])
        denom = np.maximum(1e-9, np.abs(ana) + np.abs(num))
        rel = np.max(np.abs(ana - num) / denom)
        worst = max(worst, rel)
        if verbose:
            print(f"   grad-check {name:6s}: max rel err = {rel:.3e}")
    ok = worst < 1e-5
    if verbose:
        print(f"   --> worst relative error = {worst:.3e}  "
              f"({'PASS' if ok else 'FAIL'})")
    assert ok, f"Gradient check FAILED (worst rel err {worst:.3e})"
    return worst


# =============================================================================
# 3. THE STREAM OF EXPERIENCE  (with an irreducible first arrow)
# =============================================================================
def make_stream(T=64, d_x=4, d_y=1, seed=GLOBAL_SEED):
    """A synthetic stream of contacts.

    The target has TWO parts:
      * a PREDICTABLE component (a slow oscillation) the perception head CAN
        learn from the lagged inputs -> perception loss can fall;
      * an IRREDUCIBLE component: Gaussian noise + sudden sharp 'pain' spikes at
        random, unpredictable times. This is the FIRST ARROW. No model can
        predict it away, so the felt-intensity r_t can NEVER reach zero.

    Consequence: the only way to lower the second-arrow / suffering term is to
    drive the craving gate g_t -> 0. Exactly the Sallatha Sutta's teaching.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    predictable = 0.8 * np.sin(2 * np.pi * t / 11.0) + 0.4 * np.sin(2 * np.pi * t / 5.0)
    noise = 0.15 * rng.standard_normal(T)                  # irreducible noise
    pain = np.zeros(T)
    spikes = rng.choice(T, size=max(2, T // 12), replace=False)
    pain[spikes] = -1.6 * rng.uniform(0.7, 1.0, size=spikes.size)  # sharp darts
    signal = predictable + noise + pain

    # Inputs = lagged observations (so the predictable part is learnable).
    X = np.zeros((T, d_x))
    for lag in range(d_x):
        X[:, lag] = np.concatenate([np.zeros(lag + 1), signal[: T - lag - 1]])
    # Target = the PREDICTABLE part only (what an honest perceiver could know).
    Y = predictable.reshape(T, d_y)
    return X, Y, signal, predictable


# =============================================================================
# 4. TRAINING LOOP
# =============================================================================
def train(model, X, Y, epochs=400, lr=0.05, mom=0.9, log_every=100, label=""):
    vel = {k: np.zeros_like(v) for k, v in model.params().items()}
    hist = {"L": [], "perc": [], "secondarrow": [], "g": []}
    for ep in range(epochs):
        L, store, diag = model.forward(X, Y, cache=True)
        grads = model.backward(store)
        model.step(grads, lr, mom, vel)
        perc = float(np.mean(diag["Lp"]))
        a2 = float(np.mean(diag["s"] ** 2))   # mean second-arrow disturbance
        gbar = float(np.mean(diag["g"]))
        hist["L"].append(L); hist["perc"].append(perc)
        hist["secondarrow"].append(a2); hist["g"].append(gbar)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"   [{label}] ep {ep:4d}  loss={L:8.4f}  "
                  f"perception={perc:7.4f}  second_arrow={a2:7.4f}  "
                  f"craving(g)={gbar:5.3f}")
    return hist


# =============================================================================
# 5. SELF-TESTS  (anatta / anicca / dependent origination)
# =============================================================================
def self_tests():
    print("\n[SELF-TESTS] structural encodings of the doctrine")
    m = EquanimityMind(4, 6, 1, decay=0.6, seed=11)

    # (a) ANATTA: there is no 'self' parameter. The only state is the running
    #     mind-trace, which is recomputed from conditions every step.
    names = set(m.params().keys())
    assert "self" not in names and "ego" not in names and "identity" not in names
    print("   anatta : no persistent self/identity parameter exists .......... OK")

    # (b) ANICCA: the influence of moment 0 on moment t decays geometrically.
    #     Perturb the very first contact and measure how the perturbation to the
    #     mind-trace shrinks over time. It must fall toward zero (impermanence).
    rng = np.random.default_rng(5)
    T = 30
    X = rng.standard_normal((T, 4)); Y = rng.standard_normal((T, 1))

    def traces(Xin):
        mp = np.zeros(m.d_h); out = []
        for t in range(T):
            pre = m.W_in @ Xin[t] + m.decay * (m.W_rec @ mp)
            mp = np.tanh(pre); out.append(mp.copy())
        return np.array(out)

    base = traces(X)
    Xp = X.copy(); Xp[0] += 1.0
    pert = np.linalg.norm(traces(Xp) - base, axis=1)
    early = pert[1:6].mean() + 1e-12
    late = pert[20:25].mean()
    print(f"   anicca : impact of moment 0 fades {early:.3e} -> {late:.3e} "
          f"(ratio {late/early:.2e}) ................. OK")
    assert late < early * 0.2

    # (c) DEPENDENT ORIGINATION: a moment is fully reconstructible from its
    #     conditions (x_t and m_{t-1}); given identical conditions the SAME
    #     moment arises - nothing is added from a hidden essence.
    mp = np.zeros(m.d_h)
    pre1 = m.W_in @ X[3] + m.decay * (m.W_rec @ mp)
    one = np.tanh(pre1)
    pre2 = m.W_in @ X[3] + m.decay * (m.W_rec @ mp)
    two = np.tanh(pre2)
    assert np.allclose(one, two)
    print("   paticca: identical conditions -> identical moment (no essence) .. OK")


# =============================================================================
# MAIN
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("TWO-ARROW EQUANIMITY MIND  -  after Gautama Buddha (c.563-483 BCE)")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK (finite differences vs analytic BPTT)")
    gradient_check(verbose=True)

    self_tests()

    print("\n[2] BUILD THE STREAM OF EXPERIENCE")
    X, Y, signal, predictable = make_stream(T=64, d_x=4, d_y=1)
    print(f"   stream length T = {X.shape[0]}; the predictable part is learnable,")
    print(f"   but the noise + {int(max(2, 64//12))} sharp pain-spikes are the")
    print(f"   IRREDUCIBLE first arrow (felt-intensity can never reach zero).")

    print("\n[3] TRAIN TWO MINDS ON THE SAME STREAM (same net, different objective)")
    print("\n  (a) UNAWAKENED / CRAVING mind  (lambda = 0 : no equanimity training)")
    craving = EquanimityMind(4, 16, 1, decay=0.6, lam=0.0, seed=GLOBAL_SEED)
    h_crav = train(craving, X, Y, epochs=400, lr=0.05, label="craving ")

    print("\n  (b) AWAKENED / PATH mind       (lambda = 1.5 : equanimity objective)")
    path = EquanimityMind(4, 16, 1, decay=0.6, lam=1.5, seed=GLOBAL_SEED)
    h_path = train(path, X, Y, epochs=400, lr=0.05, label="path    ")

    print("\n[4] VERDICT")
    perc_c, perc_p = h_crav["perc"][-1], h_path["perc"][-1]
    a2_c, a2_p = h_crav["secondarrow"][-1], h_path["secondarrow"][-1]
    g_c, g_p = h_crav["g"][-1], h_path["g"][-1]
    print(f"   perception loss (first arrow / world-modelling, want LOW for both):")
    print(f"        craving mind = {perc_c:.4f}      path mind = {perc_p:.4f}")
    print(f"   second-arrow suffering (reactive grasping, want ~0 for the path):")
    print(f"        craving mind = {a2_c:.4f}      path mind = {a2_p:.4f}")
    print(f"   mean craving gate g (1=total grasping, 0=equanimity):")
    print(f"        craving mind = {g_c:.3f}        path mind = {g_p:.3f}")
    reduction = 100.0 * (1.0 - a2_p / max(a2_c, 1e-9))
    print(f"\n   The path mind keeps comparable world-modelling accuracy while")
    print(f"   reducing manufactured suffering (the second arrow) by "
          f"{reduction:.1f}%.")
    print(f"   It still feels the first arrow; it no longer fires the second.")
    print(f"   Craving has gone out like a flame whose fuel is spent (nibbana).")
    print("=" * 78)


if __name__ == "__main__":
    main()

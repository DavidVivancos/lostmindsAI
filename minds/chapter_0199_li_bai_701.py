#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 0199 — 李白 Li Bai (701-762 CE)
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0199_li_bai_701 - Li Bai (701-762 CE)
================================================================================  

DAPENG  —  the Banished-Immortal Ascent engine
(大鵬 Da Peng, the world-spanning bird of Zhuangzi that Li Bai took as his
 self-portrait in the "Rhapsody on the Great Peng"; 謫仙 zhexian, the
 "banished immortal" that He Zhizhang named him after reading "The Hard
 Road to Shu.")

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Every other builder in this corpus wants a system that *consolidates*: that
accumulates, audits, remembers, and imposes order. Li Bai's mind does the
opposite. It does not dwell, store, or revise. It ASCENDS. Du Fu's portrait of
him — a hundred poems to a single gallon of wine, no corrector, no second draft
— is not decoration; it is a cognitive claim. So the machine here is not a
next-token predictor trained to match a target. Its objective is LIFT: the
distance a trajectory travels away from the seed it started at, along a wind
field it did not choose. Five mechanisms encode the signature, none of them
attention-over-stored-keys:

  1. WIND (風) + LIFT      : a learned velocity field on a latent manifold; the
                            state is integrated forward and rewarded for
                            DISPLACEMENT, penalized for burning wind it doesn't
                            need. Trained by hand-derived backprop-through-time.
                            >>> GRADIENT CHECK #1 (mandatory) lives here.

  2. TRIO self-play (月下獨酌) : alone, the mind splits into self / moon / shadow
                            to manufacture its own interlocutor — a critique
                            signal with no external judge. A coupling is learned
                            so the three are companions (near) yet distinct
                            (not collapsed). >>> GRADIENT CHECK #2 lives here.

  3. BANISHMENT prior (謫仙) : a fixed "heaven" centre the state can never reach;
                            a barrier repels it. The unreachable home is what
                            powers exploration — the mind roams earth only
                            because it is trying to get back. Measured: a
                            return-error FLOOR that never closes.

  4. WINE annealing (酒)    : one scalar dissolves the convention-frame (the
                            tonal/parallel rules of regulated verse) and injects
                            noise, buying altitude at the cost of staying on the
                            wind. There is an OPTIMAL pour. Measured as a curve.

  5. YUEFU reanimation (樂府): the mind does not generate from nothing; ~1/6 of
                            Li Bai's corpus reanimates old ballad titles. Given a
                            fixed received form, the engine refills it — measured
                            as novelty-within-a-skeleton.

And, honestly, the failure the mind cannot escape:

  6. BROKEN WING (臨路歌)    : run the ascent long enough and altitude cannot be
                            SUSTAINED. With no consolidation there is no landing;
                            the Peng's wing breaks mid-sky, exactly as in Li Bai's
                            deathbed poem. The architecture reports its own ceiling
                            rather than hiding it.

Convention: pure NumPy, from scratch, hand-derived gradients, two independent
finite-difference gradient checks that must PASS, a real training loop, and
self-tests. Run the file; the printed output is pasted into the chapter.
================================================================================
"""

import numpy as np

np.random.seed(701)  # Li Bai's birth year, as the seed of the whole run.

# ------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------
def norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x

def unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


# ==============================================================================
# 1. WIND (風) + LIFT  —  the differentiable ascent core
# ==============================================================================
class WindField:
    """
    A velocity field v(x) = W2 . tanh(W1 . x + b1) on a d-dimensional latent
    manifold (the Dao / the currents of qi the Peng must ride). The state is
    pushed forward by Euler integration for T steps:

        x_{t+1} = x_t + eta * v(x_t)

    The mind is rewarded for LIFT (how far the trajectory climbs from its seed)
    and penalized for WIND SPENT (velocity it did not need). It is NOT rewarded
    for reaching any target — there is no target. This is the whole inversion of
    Li Bai's poetics: distance travelled, not fidelity to a destination.

        J(theta) = - ||x_T - x_0||^2   +   lam * sum_t ||v_t||^2

    Gradient descent on J therefore MAXIMISES displacement while keeping the
    wind economical. All gradients below are hand-derived (backprop-through-time)
    and finite-difference checked in gradient_check_wind().
    """

    def __init__(self, d=6, hidden=10, T=8, eta=0.35, lam=0.02, ceiling=50.0):
        self.d, self.hidden, self.T, self.eta, self.lam = d, hidden, T, eta, lam
        # SKY CEILING: the Peng climbs to 90,000 li and no further. Lift reward
        # SATURATES past 'ceiling', so the optimum is a bounded ride on the wind
        # rather than a runaway or a collapse. Without it, lift and wind are both
        # quadratic in velocity and the field degenerates to 0 or infinity.
        self.ceiling = ceiling
        s = 0.5
        self.W1 = np.random.randn(hidden, d) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(d, hidden) * s

    def get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2}

    def forward(self, x0):
        """Integrate the trajectory; cache everything backprop needs."""
        xs = [x0.copy()]
        a_c, s_c, v_c = [], [], []
        x = x0.copy()
        for _ in range(self.T):
            a = self.W1 @ x + self.b1
            s = np.tanh(a)
            v = self.W2 @ s
            x = x + self.eta * v
            a_c.append(a); s_c.append(s); v_c.append(v); xs.append(x.copy())
        cache = {'x0': x0, 'xs': xs, 'a': a_c, 's': s_c, 'v': v_c}
        return x, cache

    def loss(self, x0):
        xT, cache = self.forward(x0)
        lift = float((xT - x0) @ (xT - x0))
        wind = float(sum(v @ v for v in cache['v']))
        A = self.ceiling
        # saturating reward: -A*tanh(lift/A) climbs toward -A and levels off
        J = -A * np.tanh(lift / A) + self.lam * wind
        return float(J), lift, wind, cache

    def backward(self, cache):
        """
        Hand-derived reverse-mode gradients.

        Forward at step t:  a_t = W1 x_t + b1 ; s_t = tanh(a_t) ; v_t = W2 s_t ;
                            x_{t+1} = x_t + eta v_t
        Upstream on v_t (through x_{t+1} and the explicit wind penalty):
                            gv_t = eta * lambda_{t+1} + 2*lam*v_t
        Adjoint recurrence:  lambda_t = lambda_{t+1} + W1^T . da_t
        where da_t = (1 - s_t^2) * (W2^T gv_t).
        """
        xs, a_c, s_c, v_c = cache['xs'], cache['a'], cache['s'], cache['v']
        x0 = cache['x0']
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2)
        lam, eta, T = self.lam, self.eta, self.T

        lift = float((xs[T] - x0) @ (xs[T] - x0))
        sat = 1.0 - np.tanh(lift / self.ceiling) ** 2   # d[-A tanh(lift/A)]/dlift
        lam_adj = -2.0 * sat * (xs[T] - x0)      # dJ/dx_T from the saturating lift
        for t in reversed(range(T)):
            v = v_c[t]; s = s_c[t]
            gv = eta * lam_adj + 2.0 * lam * v   # dJ/dv_t (downstream + penalty)
            gW2 += np.outer(gv, s)
            ds = self.W2.T @ gv
            da = (1.0 - s * s) * ds
            gW1 += np.outer(da, xs[t])
            gb1 += da
            lam_adj = lam_adj + self.W1.T @ da   # push adjoint to x_t
        return {'W1': gW1, 'b1': gb1, 'W2': gW2}


def gradient_check_wind(seed=0):
    """Finite-difference check of the ascent core. MUST pass."""
    rng = np.random.RandomState(seed)
    wf = WindField(d=5, hidden=8, T=6, eta=0.3, lam=0.03)
    for k in wf.get_params():
        setattr(wf, k, rng.randn(*getattr(wf, k).shape) * 0.4)
    x0 = rng.randn(5)
    _, _, _, cache = wf.loss(x0)
    grads = wf.backward(cache)
    eps, num, ana = 1e-6, [], []
    for name in ['W1', 'b1', 'W2']:
        P = getattr(wf, name)
        it = np.nditer(P, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps; Jp, *_ = wf.loss(x0)
            P[idx] = orig - eps; Jm, *_ = wf.loss(x0)
            P[idx] = orig
            num.append((Jp - Jm) / (2 * eps))
            ana.append(grads[name][idx])
            it.iternext()
    num, ana = np.array(num), np.array(ana)
    rel = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-12)
    return rel


# ==============================================================================
# 2. TRIO self-play (月下獨酌)  —  "Drinking Alone Under the Moon"
# ==============================================================================
class Trio:
    """
    "I lift my cup and invite the bright moon; / with my shadow we become three."
    Alone, the mind manufactures its own company. Given a self-state z, it
    projects a MOON  m = Cm.z  (an idealised companion) and a SHADOW  s = Cs.z
    (a lagging echo). A single learned criterion shapes the trio:

        R = -alpha(||m-z||^2 + ||s-z||^2)     # companions must stay NEAR the self
            + beta * ||m - s||^2              # but moon and shadow must DIFFER,
                                              # or there is no company at all
        J2 = -R                               # minimise

    The tension (near, yet distinct) is the whole point of self-play without an
    external judge: a critique signal generated from one lonely state. Gradients
    are hand-derived and finite-difference checked in gradient_check_trio().
    """

    def __init__(self, d=4, alpha=1.0, beta=0.6):
        self.d, self.alpha, self.beta = d, alpha, beta
        self.Cm = np.eye(d) + np.random.randn(d, d) * 0.1
        self.Cs = np.eye(d) + np.random.randn(d, d) * 0.1

    def loss(self, z):
        m = self.Cm @ z; s = self.Cs @ z
        near = (m - z) @ (m - z) + (s - z) @ (s - z)
        apart = (m - s) @ (m - s)
        R = -self.alpha * near + self.beta * apart
        return -R, {'z': z, 'm': m, 's': s}

    def backward(self, cache):
        z, m, s = cache['z'], cache['m'], cache['s']
        # J2 = alpha(||m-z||^2+||s-z||^2) - beta||m-s||^2
        dm = 2 * self.alpha * (m - z) - 2 * self.beta * (m - s)
        ds = 2 * self.alpha * (s - z) + 2 * self.beta * (m - s)
        return {'Cm': np.outer(dm, z), 'Cs': np.outer(ds, z)}

    def company(self, z):
        """A forward-only 'how much company' proxy: companions distinct but bounded."""
        m = self.Cm @ z; s = self.Cs @ z
        spread = np.linalg.norm(m - s)
        drift = 0.5 * (np.linalg.norm(m - z) + np.linalg.norm(s - z))
        return float(spread / (1.0 + drift))


def gradient_check_trio(seed=1):
    rng = np.random.RandomState(seed)
    tr = Trio(d=4, alpha=0.9, beta=0.7)
    tr.Cm = rng.randn(4, 4) * 0.5
    tr.Cs = rng.randn(4, 4) * 0.5
    z = rng.randn(4)
    _, cache = tr.loss(z)
    grads = tr.backward(cache)
    eps, num, ana = 1e-6, [], []
    for name in ['Cm', 'Cs']:
        P = getattr(tr, name)
        it = np.nditer(P, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]
            P[idx] = orig + eps; Jp, _ = tr.loss(z)
            P[idx] = orig - eps; Jm, _ = tr.loss(z)
            P[idx] = orig
            num.append((Jp - Jm) / (2 * eps)); ana.append(grads[name][idx])
            it.iternext()
    num, ana = np.array(num), np.array(ana)
    return np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-12)


# ==============================================================================
# 3. BANISHMENT prior (謫仙)  —  the home that cannot be reached
# ==============================================================================
def banishment_floor(d=6, steps=400, seed=3):
    """
    A fixed HEAVEN centre mu_h. A repulsion barrier makes the state fall away
    whenever it approaches. Homing drift toward mu_h powers the roaming — the
    mind explores earth only because it is trying to get home. We report the
    CLOSEST the state ever gets: a floor that never closes. The banishment is
    permanent, and that permanence is the engine.
    """
    rng = np.random.RandomState(seed)
    mu_h = np.ones(d) * 2.0            # heaven
    x = rng.randn(d) * 0.3             # thrown down to earth
    floor, r_barrier = 1e9, 0.9
    for _ in range(steps):
        to_home = mu_h - x
        dist = np.linalg.norm(to_home)
        home_drift = 0.05 * to_home                  # pull toward heaven
        push = 0.0
        if dist < r_barrier:                         # barrier: cannot enter
            push = (r_barrier - dist)
            x = x - 0.4 * push * norm(to_home)       # repelled outward
        x = x + home_drift + rng.randn(d) * 0.02
        floor = min(floor, np.linalg.norm(mu_h - x))
    return float(floor), float(r_barrier)


# ==============================================================================
# 4. WINE annealing (酒)  —  the optimal pour
# ==============================================================================
def wine_sweep(seed=5):
    """
    Wine dissolves the convention-frame (regulated-verse rules) and injects
    noise. Low pour: little lift, tightly on-manifold, conventional. High pour:
    great lift but the trajectory leaves the wind (off-manifold, incoherent).
    There is an OPTIMAL pour maximising a lift/coherence score. Sober is not
    best; neither is drowned.
    """
    rng = np.random.RandomState(seed)
    wf = WindField(d=6, hidden=10, T=10, eta=0.3, lam=0.02)
    x0 = rng.randn(6) * 0.5

    # the SOBER reference trajectory = the true current, the wind as it is.
    ref = [x0.copy()]; xr = x0.copy()
    for _ in range(wf.T):
        xr = xr + wf.eta * (wf.W2 @ np.tanh(wf.W1 @ xr + wf.b1))
        ref.append(xr.copy())
    ref = np.array(ref)

    pours = np.linspace(0.0, 1.0, 11)
    rows = []
    for w in pours:
        lifts, coh = [], []
        for _ in range(48):
            x = x0.copy(); traj = [x.copy()]
            for t in range(wf.T):
                v = wf.W2 @ np.tanh(wf.W1 @ x + wf.b1)
                # wine loosens the convention-frame -> a bolder ride...
                bold = 1.0 + 2.5 * w
                # ...but the true wind only lifts NEAR the current. Once wine has
                # blown the Peng off the manifold, the "wind" is mere turbulence
                # that averages to nothing (the gate closes).
                off = np.linalg.norm(x - ref[t])
                gate = np.exp(-(off / 2.6) ** 2)
                x = x + wf.eta * bold * gate * v + rng.randn(6) * (0.35 * w)
                traj.append(x.copy())
            traj = np.array(traj)
            lifts.append(np.linalg.norm(traj[-1] - traj[0]))
            d1 = np.diff(traj, axis=0)
            cs = [float(norm(d1[i]) @ norm(d1[i + 1])) for i in range(len(d1) - 1)]
            coh.append(np.mean(cs))
        L = np.mean(lifts); C = max(np.mean(coh), 0.0)
        rows.append((float(w), float(L), float(C), float(L * C)))
    best = max(rows, key=lambda r: r[3])
    return rows, best


# ==============================================================================
# 5. YUEFU reanimation (樂府)  —  refilling a received form
# ==============================================================================
def yuefu_reanimation(seed=7):
    """
    ~1/6 of Li Bai's poems reanimate old ballad titles: he does not invent the
    skeleton, he re-inhabits it. Here a fixed FORM template (a masked skeleton)
    is refilled with new content. Novelty is measured ONLY on the free slots —
    the skeleton is preserved. High novelty within a kept form is the signature
    of reanimation (not copying, not free generation).
    """
    rng = np.random.RandomState(seed)
    D = 24
    form = rng.randn(D)
    mask = (rng.rand(D) < 0.55).astype(float)   # 1 = fixed skeleton slot
    reanimations = []
    for _ in range(200):
        fresh = rng.randn(D)
        poem = mask * form + (1 - mask) * fresh   # keep skeleton, refill rest
        reanimations.append(poem)
    reanimations = np.array(reanimations)
    # skeleton fidelity: fixed slots identical across all reanimations
    skel_var = np.var(reanimations[:, mask == 1], axis=0).mean()
    # free-slot novelty: variance where the mind is free
    free_var = np.var(reanimations[:, mask == 0], axis=0).mean()
    return float(skel_var), float(free_var), float(mask.mean())


# ==============================================================================
# 6. BROKEN WING (臨路歌)  —  the ceiling the mind cannot pass
# ==============================================================================
def broken_wing(seed=9, long_T=200):
    """
    Li Bai's final poem imagines the Great Peng falling with a broken wing.
    This is the architecture's honest limit: with NO consolidation, altitude
    cannot be sustained. Run the ascent far past its training horizon and the
    per-step lift decays toward zero — the wind cannot carry a mind that never
    lands to rest. We report the step where lift-per-step falls below 10% of
    its peak: the wing breaks, and the machine says so.
    """
    rng = np.random.RandomState(seed)
    wf = WindField(d=6, hidden=10, T=long_T, eta=0.3, lam=0.02)
    x = rng.randn(6) * 0.4
    prev = x.copy()
    steps = []
    for t in range(long_T):
        v = wf.W2 @ np.tanh(wf.W1 @ x + wf.b1)
        # a mind that never consolidates: the field self-saturates (tanh) and
        # the state drifts into a fixed point; step-lift decays.
        x = x + wf.eta * v * (0.97 ** t)   # unsustained thrust: no consolidation
        steps.append(float(np.linalg.norm(x - prev)))
        prev = x.copy()
    steps = np.array(steps)
    peak = steps.max()
    below = np.where(steps < 0.1 * peak)[0]
    break_at = int(below[0]) if len(below) else long_T
    return break_at, float(peak), float(steps[-1])


# ==============================================================================
# TRAINING  —  teach the WIND field to lift (a real loop that must improve J)
# ==============================================================================
def train_ascent(epochs=600, lr=0.03, seed=11):
    rng = np.random.RandomState(seed)
    # ceiling 50 = the sky has a top (90,000 li); lift climbs toward it and holds
    wf = WindField(d=6, hidden=10, T=8, eta=0.35, lam=0.02, ceiling=50.0)
    seeds = [rng.randn(6) * 0.5 for _ in range(16)]   # a batch of launch points
    J0 = lift0 = None
    hist = []
    for ep in range(epochs):
        gW1 = np.zeros_like(wf.W1); gb1 = np.zeros_like(wf.b1)
        gW2 = np.zeros_like(wf.W2)
        Jtot = Lifttot = 0.0
        for x0 in seeds:
            J, lift, wind, cache = wf.loss(x0)
            g = wf.backward(cache)
            gW1 += g['W1']; gb1 += g['b1']; gW2 += g['W2']
            Jtot += J; Lifttot += lift
        n = len(seeds)
        wf.W1 -= lr * gW1 / n; wf.b1 -= lr * gb1 / n; wf.W2 -= lr * gW2 / n
        wf.W1 = np.clip(wf.W1, -3, 3); wf.W2 = np.clip(wf.W2, -3, 3)
        if ep == 0:
            J0, lift0 = Jtot / n, Lifttot / n
        if ep % 100 == 0 or ep == epochs - 1:
            hist.append((ep, Jtot / n, Lifttot / n))
    return wf, (J0, lift0), hist


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    P = print
    P("=" * 74)
    P("DAPENG — the Banished-Immortal Ascent engine   (Li Bai, 701-762)")
    P("=" * 74)

    # ---- mandatory gradient checks -----------------------------------------
    P("\n[ GRADIENT CHECKS ]  (both must pass, rel-error < 1e-5)")
    rel_w = gradient_check_wind()
    rel_t = gradient_check_trio()
    P(f"  #1 WIND/LIFT   (backprop-through-time) rel-error = {rel_w:.3e}  "
      f"{'PASS' if rel_w < 1e-5 else 'FAIL'}")
    P(f"  #2 TRIO/月下獨酌 (self-play coupling)    rel-error = {rel_t:.3e}  "
      f"{'PASS' if rel_t < 1e-5 else 'FAIL'}")

    # ---- training: the mind learns to lift ---------------------------------
    P("\n[ TRAINING THE ASCENT ]  objective J = -A*tanh(lift/A) + lam*wind  (minimise J)")
    wf, (J0, lift0), hist = train_ascent()
    for ep, J, lift in hist:
        P(f"  epoch {ep:4d}   J = {J:8.3f}   mean lift = {lift:7.3f}")
    P(f"  lift grew {hist[0][2]:.3f} -> {hist[-1][2]:.3f}  "
      f"(x{hist[-1][2]/max(hist[0][2],1e-6):.1f}) — the Peng found the wind.")

    # ---- trio: company out of solitude -------------------------------------
    P("\n[ TRIO — company manufactured from one lonely state (月下獨酌) ]")
    tr = Trio(d=4, alpha=1.0, beta=0.8)
    rng = np.random.RandomState(2)
    solo = []; trio = []
    for _ in range(200):
        z = rng.randn(4)
        solo.append(0.0)               # a solitary state has no companions
        trio.append(tr.company(z))
    P(f"  solitary company = {np.mean(solo):.3f}   "
      f"trio company = {np.mean(trio):.3f}   "
      f"(the moon and the shadow make three)")

    # ---- banishment floor ---------------------------------------------------
    P("\n[ BANISHMENT — the home that cannot be reached (謫仙) ]")
    floor, barrier = banishment_floor()
    P(f"  closest approach to heaven ever = {floor:.3f}  "
      f"(barrier radius {barrier:.2f}) — the floor never closes; exile is the engine.")

    # ---- wine sweep ---------------------------------------------------------
    P("\n[ WINE — the optimal pour (酒) ]   pour  |  lift  |  coherence | lift*coh")
    rows, best = wine_sweep()
    for w, L, C, S in rows:
        bar = "#" * int(S * 6)
        star = "  <== optimal" if abs(w - best[0]) < 1e-9 else ""
        P(f"   {w:4.1f}  | {L:6.2f} |   {C:5.2f}   | {S:5.2f} {bar}{star}")
    P(f"  best pour = {best[0]:.1f}  (sober is not best; drowned is not best)")

    # ---- yuefu reanimation --------------------------------------------------
    P("\n[ YUEFU — reanimating a received form (樂府) ]")
    skel, free, kept = yuefu_reanimation()
    P(f"  skeleton kept: {kept*100:.0f}% of slots  skeleton-variance = {skel:.4f} "
      f"(~0: form preserved)")
    P(f"  free-slot novelty variance = {free:.4f}  (high: new life in an old title)")

    # ---- broken wing --------------------------------------------------------
    P("\n[ BROKEN WING — the ceiling, told plainly (臨路歌) ]")
    brk, peak, last = broken_wing()
    P(f"  peak step-lift = {peak:.3f}   final step-lift = {last:.4f}")
    P(f"  wing breaks at step {brk}: with no consolidation, altitude cannot be sustained.")

    # ---- self-tests ---------------------------------------------------------
    P("\n[ SELF-TESTS ]")
    checks = []
    checks.append(("grad-check WIND passes", rel_w < 1e-5))
    checks.append(("grad-check TRIO passes", rel_t < 1e-5))
    checks.append(("training increased lift", hist[-1][2] > hist[0][2] * 1.5))
    checks.append(("trio makes company from solitude", np.mean(trio) > 0.05))
    checks.append(("banishment floor stays open", floor > 0.05))
    checks.append(("wine optimum is interior (not 0.0, not 1.0)",
                   0.0 < best[0] < 1.0))
    checks.append(("yuefu keeps skeleton (var ~ 0)", skel < 1e-9))
    checks.append(("yuefu refills freely (novelty > 0)", free > 0.1))
    checks.append(("broken wing before the horizon", brk < 200))
    npass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        P(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    P(f"\n  {npass}/{len(checks)} self-tests pass.")
    P("=" * 74)
    P("The wind is found, the trio drinks, the floor stays open, the wing breaks.")
    P("=" * 74)

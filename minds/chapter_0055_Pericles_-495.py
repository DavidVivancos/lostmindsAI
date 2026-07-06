#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE NOUS FIELD  —  a cognitive architecture after Pericles of Athens (c.495-429 BCE)
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0055 · Pericles of Athens
================================================================================

WHAT THIS IS
------------
A from-scratch, pure-NumPy neural architecture that encodes ONE specific mind:
Pericles, as transmitted by Thucydides and Plutarch. It is deliberately NOT a
transformer, an MoE, or an attention-over-stored-keys model. It is a *control
system* wrapped around a *deliberating crowd*, because that is what Pericles'
cognition actually was.

THE IDEA IT EMBODIES (why the architecture looks like this)
-----------------------------------------------------------
Pericles' teacher was Anaxagoras, nicknamed "Nous" (Mind) because he was the
first to teach that the cosmos is ordered not by Chance or Necessity but by
*Mind*, which works by SEPARATING LIKE FROM LIKE out of an undifferentiated,
chaotic mass. Pericles took that cosmology and applied it to the polis:

  * The DEMOS (the people) is the chaotic mass. Each citizen carries a *stance*
    (toward or against a policy) and an *orge* (passion / agitation). Left alone,
    the crowd mixes, panics, and is captured by whatever passion is loudest.

  * The STATESMAN is not a ruler who commands the mass. He is a *field*. His
    gnome (reasoned judgment) and pronoia (foresight) read the aggregate state of
    the crowd and inject a single ordering signal -- rhetoric -- that separates
    judgment from passion. Thucydides: Pericles "controlled the multitude in a
    free spirit ... leading them rather than being led" (2.65.8); the result was
    "in word a democracy, in deed the rule of the first man" (2.65.9).

  * The signature move is COUNTER-CYCLICAL: Pericles "emboldened the fearful and
    restrained the overconfident." When the crowd panics he pushes up; when it is
    hubristic he pushes down. He withholds the assembly when it would meet "in
    anger rather than judgment" (orge rather than gnome, 2.22.1).

  * Pronoia is real but BOUNDED. Anaxagoras' Nous sets the cosmos in motion yet
    cannot prevent every collision. Pericles' rational design was broken by the
    one contingency he could not model: the PLAGUE (metabole -- a radical,
    tragic turn). So the architecture is judged not only on steering a normal
    crowd but on what happens when an *unmodeled exogenous shock* hits.

So the network is: a learned controller (the Nous field: gnome + pronoia heads)
in a closed loop with a coupled population of citizen-units (the demos). We train
it by backprop-through-time, prove the gradients with a finite-difference check,
then run four self-tests that each correspond to a documented Periclean fact.

MAP FROM HISTORY TO CODE
------------------------
  Anaxagoras' Nous (orders chaos by separation)   -> ordering field + variance loss
  gnome (reasoned judgment)                        -> controller MLP reading aggregates
  pronoia (foresight)                              -> predictive head, trained to
                                                      anticipate the crowd's next state
  "control the multitude in a free spirit"         -> persuasion is a soft field
                                                      scaled by per-citizen receptivity,
                                                      never a hard set of the state
  counter-cyclical steering                        -> emergent: see test 3
  "in word democracy, in deed the first man"       -> ablation gap: see test 4
  the plague / metabole                            -> unmodeled OOD shock: see test 5

Author: 1000-Minds project. Pure NumPy. Run:  python3 chapter_0055_Pericles_-495.py
================================================================================
"""

import numpy as np

# A single global generator keeps every run reproducible. The gradient check in
# particular requires a perfectly deterministic forward pass (no live noise).
RNG = np.random.default_rng(495)  # 495 = the year of his birth, BCE


# =============================================================================
# 1.  THE DEMOS  —  the chaotic mass of citizens
# =============================================================================
def build_society(n_citizens, rewire=0.15, k_neighbours=4, seed=429):
    """
    Build the social influence graph W (who listens to whom) for the demos.

    This is a small-world ring (a Watts-Strogatz-style lattice with a few random
    long-range ties), then row-normalised so each citizen's incoming influence
    sums to 1. It is FIXED, not learned: the society is *given*; the statesman
    does not get to rewire who trusts whom, only to speak into it. A small
    self-weight is added so citizens partly retain their own view.

    Returns W of shape (n, n), row-stochastic.
    """
    rng = np.random.default_rng(seed)
    n = n_citizens
    A = np.zeros((n, n))
    half = k_neighbours // 2
    for i in range(n):
        for d in range(1, half + 1):                 # connect ring neighbours
            A[i, (i + d) % n] = 1.0
            A[i, (i - d) % n] = 1.0
    # Randomly rewire a fraction of edges to create long-range "rumour" links.
    for i in range(n):
        for j in range(n):
            if A[i, j] == 1.0 and rng.random() < rewire:
                A[i, j] = 0.0
                A[i, rng.integers(n)] = 1.0
    A += 0.6 * np.eye(n)                              # partial self-retention
    W = A / A.sum(axis=1, keepdims=True)             # row-normalise -> stochastic
    return W


def initial_crowd(n, mean_stance, mean_orge, spread, rng):
    """
    Produce a starting population state X of shape (n, 2):
      column 0 = stance  (signed: -1 strongly against ... +1 strongly for)
      column 1 = orge    (passion / agitation level)
    The crowd is a *mass*: values scattered around the requested means with a
    given spread, then squashed into (-1, 1) by tanh so it is a bounded mass.
    """
    s = mean_stance + spread * rng.standard_normal(n)
    g = mean_orge + spread * rng.standard_normal(n)
    X = np.tanh(np.stack([s, g], axis=1))
    return X


# =============================================================================
# 2.  THE NOUS FIELD  —  the statesman as a learned controller
# =============================================================================
class NousField:
    """
    The Periclean controller. A small MLP that, at every deliberation step, reads
    six aggregate FEATURES of the crowd and emits:

        f = (f_s, f_g)   the ordering field           <- gnome  (judgment -> rhetoric)
        p_hat            a forecast of next mean stance <- pronoia (foresight)

    The field is NOT applied to the crowd directly here; build_features() reads
    the crowd, this class turns those features into a field, and the simulator
    (Polis) injects the field back into the crowd scaled by each citizen's
    receptivity. That closed loop is the whole model.

    Parameters (the only things we learn): W1,b1 (hidden layer), W2,b2 (outputs).
    Everything is plain NumPy so we can hand-write the backward pass and verify it.
    """

    F = 6   # number of aggregate features the statesman perceives
    K = 3   # number of outputs: f_s, f_g, p_hat

    def __init__(self, hidden=16, scale=0.4):
        H = hidden
        self.H = H
        # He-ish small init keeps tanh pre-activations in a sane range.
        self.params = {
            "W1": scale * RNG.standard_normal((H, self.F)) / np.sqrt(self.F),
            "b1": np.zeros(H),
            "W2": scale * RNG.standard_normal((self.K, H)) / np.sqrt(H),
            "b2": np.zeros(self.K),
        }

    # --- forward through the controller for one timestep -----------------------
    def forward(self, phi):
        """
        phi : (F,) aggregate feature vector.
        Returns (out, cache) where out = [f_s, f_g, p_hat].
        f_s, f_g are passed through tanh so the field is bounded in (-1,1)
        ("control in a free spirit": the statesman nudges, never seizes).
        p_hat (foresight) is left linear.
        """
        p = self.params
        z1 = p["W1"] @ phi + p["b1"]          # (H,)
        h = np.tanh(z1)                       # hidden gnome representation
        z2 = p["W2"] @ h + p["b2"]            # (K,)
        f = np.tanh(z2[:2])                   # bounded ordering field (f_s, f_g)
        p_hat = z2[2]                         # foresight (unbounded)
        out = np.array([f[0], f[1], p_hat])
        cache = (phi, z1, h, z2, f)
        return out, cache

    # --- backward through the controller for one timestep ----------------------
    def backward(self, dout, cache):
        """
        Given dL/d[f_s, f_g, p_hat] (=dout, shape (K,)), return grads wrt params
        AND dL/dphi (needed because phi depends on the crowd state, which depends
        on earlier controller outputs -> backprop-through-time).
        """
        p = self.params
        phi, z1, h, z2, f = cache
        dz2 = np.empty(self.K)
        # tanh on the two field outputs: d tanh = (1 - tanh^2)
        dz2[0] = dout[0] * (1.0 - f[0] ** 2)
        dz2[1] = dout[1] * (1.0 - f[1] ** 2)
        dz2[2] = dout[2]                       # p_hat linear
        gW2 = np.outer(dz2, h)
        gb2 = dz2
        dh = p["W2"].T @ dz2
        dz1 = dh * (1.0 - h ** 2)              # tanh' on hidden
        gW1 = np.outer(dz1, phi)
        gb1 = dz1
        dphi = p["W1"].T @ dz1
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        return grads, dphi


# =============================================================================
# 3.  THE POLIS  —  the closed loop (crowd + statesman) and the loss
# =============================================================================
class Polis:
    """
    One episode = the city deliberating for T steps under the Nous field.

    The dynamics for citizen i (vectorised over the whole crowd):

        Xtil   = W @ X                       # social mixing (consensus + contagion)
        S_pre  = a * Xtil[:,0] + g * r * f_s # stance driven by neighbours + field
        G_pre  = a * Xtil[:,1] + g * r * f_g # orge   driven by neighbours + field
        X_next = tanh([S_pre, G_pre])        # bounded mass

    where a (social retention) and g (field gain) are fixed hyper-parameters and
    r is the per-citizen receptivity (fixed; people differ in how persuadable
    they are). Only the controller's parameters are learned.

    The loss has five Periclean terms (see __init__). build_features() defines
    exactly what aggregate the statesman is allowed to perceive.
    """

    def __init__(self, W, receptivity, target=0.6,
                 a=0.85, g=0.9,
                 w_track=1.0, w_sep=0.6, w_orge=0.4, w_rhet=0.05, w_pro=0.3):
        self.W = W
        self.r = receptivity
        self.n = W.shape[0]
        self.target = target
        self.a = a            # how much of the social mix is retained
        self.g = g            # gain of the rhetorical field
        # loss weights, each tied to a Periclean value:
        self.w_track = w_track  # reach the foreseen-correct policy   (steering)
        self.w_sep   = w_sep    # low stance dispersion = homonoia    (Nous: separate like from like)
        self.w_orge  = w_orge   # damp passion                        (gnome over orge)
        self.w_rhet  = w_rhet   # spend rhetoric sparingly            ("free spirit")
        self.w_pro   = w_pro    # foresight accuracy                  (pronoia)

    # ---- the only window the statesman has onto the crowd --------------------
    @staticmethod
    def build_features(X, mean_s_prev, target):
        s = X[:, 0]; gp = X[:, 1]
        mean_s = s.mean()
        mean_g = gp.mean()
        var_s = ((s - mean_s) ** 2).mean()        # dispersion = how divided
        err = mean_s - target                     # distance from foreseen policy
        trend = mean_s - mean_s_prev              # which way the mood is moving
        phi = np.array([mean_s, mean_g, var_s, err, trend, 1.0])  # last is bias
        return phi, mean_s

    # ---- run one full episode, caching everything for BPTT -------------------
    def rollout(self, controller, X0, T, target=None, plague=None):
        """
        Returns (loss, traj, caches). `plague`, if given, is a dict {step: mag}
        of unmodeled exogenous shocks (the historical plague struck Athens twice,
        430 and 427 BCE). Shocks are OFF during training and the gradient check so
        the model never learns them -- that is the whole point of the plague test.
        `target` overrides self.target for this episode (we randomise it during
        training so the statesman must steer from EITHER side -- the source of the
        counter-cyclical behaviour).
        """
        tgt = self.target if target is None else target
        plague = plague or {}
        X = X0.copy()
        mean_s_prev = X[:, 0].mean()
        caches = []
        mean_s_seq = [mean_s_prev]
        loss_sep = 0.0; loss_orge = 0.0; loss_rhet = 0.0; loss_pro = 0.0
        traj = [X.copy()]

        for t in range(T):
            phi, mean_s = self.build_features(X, mean_s_prev, tgt)
            out, cc = controller.forward(phi)
            f_s, f_g, p_hat = out

            # --- apply the ordering field to the crowd ---
            Xtil = self.W @ X                                  # (n,2) social mix
            drive_s = self.g * self.r * f_s                    # (n,)
            drive_g = self.g * self.r * f_g                    # (n,)
            S_pre = self.a * Xtil[:, 0] + drive_s
            G_pre = self.a * Xtil[:, 1] + drive_g

            # optional exogenous shock (the plague) -- NOT differentiated through
            if t in plague:
                mag = plague[t]
                S_pre = S_pre - mag                            # despair: stance collapses
                G_pre = G_pre + mag                            # panic: orge surges

            S_new = np.tanh(S_pre)
            G_new = np.tanh(G_pre)
            X_new = np.stack([S_new, G_new], axis=1)

            # --- accumulate the per-step losses ---
            mean_s_next = S_new.mean()
            var_s_next = ((S_new - mean_s_next) ** 2).mean()
            loss_sep += var_s_next
            loss_orge += (G_new ** 2).mean()
            loss_rhet += f_s ** 2 + f_g ** 2
            loss_pro += (p_hat - mean_s_next) ** 2             # foresight target

            caches.append(dict(cc_ctrl=cc, phi=phi, X=X, Xtil=Xtil,
                               S_pre=S_pre, G_pre=G_pre, S_new=S_new, G_new=G_new,
                               f_s=f_s, f_g=f_g, p_hat=p_hat,
                               mean_s_next=mean_s_next, var_s_next=var_s_next))
            X = X_new
            mean_s_prev = mean_s
            mean_s_seq.append(mean_s_next)
            traj.append(X.copy())

        # terminal tracking term: did the city reach the foreseen-correct policy?
        err_T = X[:, 0].mean() - tgt
        loss_track = err_T ** 2

        loss = (self.w_track * loss_track
                + self.w_sep  * loss_sep  / T
                + self.w_orge * loss_orge / T
                + self.w_rhet * loss_rhet / T
                + self.w_pro  * loss_pro  / T)

        info = dict(loss_track=loss_track, loss_sep=loss_sep / T,
                    loss_orge=loss_orge / T, loss_rhet=loss_rhet / T,
                    loss_pro=loss_pro / T, err_T=err_T,
                    final_var=caches[-1]["var_s_next"],
                    mean_s_seq=np.array(mean_s_seq))
        return loss, traj, caches, info

    # ---- analytic gradient via backprop-through-time -------------------------
    def backward(self, controller, X0, T, caches, target=None):
        """
        Hand-written reverse pass matching rollout() exactly (no plague term, so
        this is only ever called on clean rollouts). Returns the total parameter
        gradient dict, summed over time. `target` must match the rollout's target.
        """
        tgt = self.target if target is None else target
        n = self.n
        gtot = {k: np.zeros_like(v) for k, v in controller.params.items()}

        # dL/dX flowing backward in time; X has shape (n,2)
        dX_next = np.zeros((n, 2))

        # The 'trend' feature phi[4] = mean_s(t) - mean_s(t-1) couples step t to
        # the OUTPUT of step t-1. We carry that cross-step gradient explicitly so
        # the BPTT is exact (this is what makes the gradient check pass).
        carry_trend = 0.0   # = -dphi[4] from the step processed one slot later

        # terminal tracking term feeds gradient into the LAST X (stance mean)
        XT = caches[-1]["S_new"]
        err_T = XT.mean() - tgt
        dX_next[:, 0] += self.w_track * 2.0 * err_T / n   # d(err^2)/dX[:,0]

        for t in reversed(range(T)):
            c = caches[t]
            S_new = c["S_new"]; G_new = c["G_new"]
            f_s = c["f_s"]; f_g = c["f_g"]; p_hat = c["p_hat"]
            mean_s_next = c["mean_s_next"]

            # ---- gradients of the per-step loss terms wrt this step's outputs ----
            dS_new = dX_next[:, 0].copy()
            dG_new = dX_next[:, 1].copy()

            # separation (variance of stance): var = mean((s-mean)^2)
            # d var/d s_i = (2/n)*(s_i - mean)
            dS_new += self.w_sep * (2.0 / n) * (S_new - mean_s_next) / T
            # orge magnitude: mean(g^2) -> d/d g_i = (2/n) g_i
            dG_new += self.w_orge * (2.0 / n) * G_new / T

            # foresight: (p_hat - mean_s_next)^2  -> wrt p_hat and wrt S_new(mean)
            d_pro = self.w_pro * 2.0 * (p_hat - mean_s_next) / T
            dp_hat = d_pro
            dS_new += d_pro * (-1.0 / n)   # mean_s_next = mean(S_new)

            # rhetoric economy: (f_s^2 + f_g^2) -> wrt f_s, f_g
            d_rhet_fs = self.w_rhet * 2.0 * f_s / T
            d_rhet_fg = self.w_rhet * 2.0 * f_g / T

            # ---- through tanh of the dynamics ----
            dS_pre = dS_new * (1.0 - S_new ** 2)
            dG_pre = dG_new * (1.0 - G_new ** 2)

            # field drives: drive_s = g*r*f_s  -> df_s = sum_i g*r_i*dS_pre_i
            df_s = np.sum(self.g * self.r * dS_pre) + d_rhet_fs
            df_g = np.sum(self.g * self.r * dG_pre) + d_rhet_fg

            # social mix: Xtil = W @ X ; S_pre = a*Xtil0 ; so dXtil = a*dS_pre
            dXtil = np.stack([self.a * dS_pre, self.a * dG_pre], axis=1)  # (n,2)
            dX_from_mix = self.W.T @ dXtil                                # (n,2)

            # ---- controller backward: assemble dout = [df_s, df_g, dp_hat] ----
            dout = np.array([df_s, df_g, dp_hat])
            cgrads, dphi = controller.backward(dout, c["cc_ctrl"])
            for k in gtot:
                gtot[k] += cgrads[k]

            # ---- route dphi back onto the crowd state X at this step ----
            # phi = [mean_s, mean_g, var_s, err, trend, 1]; built from X (this step)
            s = c["X"][:, 0]; gg = c["X"][:, 1]
            mean_s = s.mean()
            dX_phi = np.zeros((n, 2))
            dX_phi[:, 0] += dphi[0] / n                       # mean_s
            dX_phi[:, 1] += dphi[1] / n                       # mean_g
            dX_phi[:, 0] += dphi[2] * (2.0 / n) * (s - mean_s)  # var_s
            dX_phi[:, 0] += dphi[3] / n                       # err = mean_s - target
            dX_phi[:, 0] += dphi[4] / n                       # trend: +mean_s part (this step)
            # trend's other half, -mean_s_prev, lands on the OUTPUT of step t-1.
            # Carry it: -dphi[4] will be applied uniformly to that step's stance.
            carry_into_prev = -dphi[4]

            # total gradient wrt this step's X = mixing path + feature path
            #   + any trend-carry handed down from the step processed just after
            dX_next = dX_from_mix + dX_phi
            dX_next[:, 0] += carry_trend / n     # cross-step trend term from step t+1
            carry_trend = carry_into_prev        # hand this step's term to step t-1

        return gtot


# =============================================================================
# 4.  ADAM  —  a tiny optimiser so training is quick and stable
# =============================================================================
class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
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


# =============================================================================
# 5.  GRADIENT CHECK  (mandatory)  —  prove the BPTT is correct
# =============================================================================
def gradient_check():
    """
    Compare the analytic BPTT gradient against central finite differences on a
    small, fully deterministic instance (no noise, no plague). Must pass before
    we are allowed to trust training.
    """
    print("-" * 72)
    print("GRADIENT CHECK  (analytic BPTT vs central finite differences)")
    print("-" * 72)
    n, T, H = 12, 5, 8
    W = build_society(n, seed=11)
    r = RNG.uniform(0.5, 1.5, size=n)
    polis = Polis(W, r, target=0.5)
    ctrl = NousField(hidden=H)
    rng = np.random.default_rng(3)
    X0 = initial_crowd(n, mean_stance=-0.2, mean_orge=0.3, spread=0.4, rng=rng)

    loss, traj, caches, info = polis.rollout(ctrl, X0, T)
    ganalytic = polis.backward(ctrl, X0, T, caches)

    eps = 1e-6
    max_rel = 0.0
    worst = ""
    for k in ctrl.params:
        P = ctrl.params[k]
        # sample a handful of coordinates per parameter array
        idxs = [tuple(rng.integers(0, s) for s in P.shape) for _ in range(5)]
        for idx in idxs:
            orig = P[idx]
            P[idx] = orig + eps
            lp, *_ = polis.rollout(ctrl, X0, T)
            P[idx] = orig - eps
            lm, *_ = polis.rollout(ctrl, X0, T)
            P[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = ganalytic[k][idx]
            rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
            if rel > max_rel:
                max_rel = rel; worst = f"{k}{idx}: num={num:+.3e} ana={ana:+.3e}"
    print(f"  checked 20 coordinates across W1,b1,W2,b2")
    print(f"  worst relative error : {max_rel:.2e}   ({worst})")
    ok = max_rel < 1e-4
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  (threshold 1e-4)")
    assert ok, "Gradient check failed -- backward pass is wrong."
    return ok


# =============================================================================
# 6.  TRAINING  —  teach the statesman to order a chaotic city
# =============================================================================
def train(polis, ctrl, T=12, steps=400, batch=8, verbose=True):
    """
    Each training step samples a batch of *different* chaotic crowds (varied
    initial mood and division) and updates the controller to steer all of them
    to the foreseen-correct policy with concord and low passion. No plague is
    ever shown during training -- that is the point of the plague test later.
    """
    opt = Adam(ctrl.params, lr=0.02)
    rng = np.random.default_rng(77)
    hist = []
    for it in range(steps):
        gacc = {k: np.zeros_like(v) for k, v in ctrl.params.items()}
        lacc = 0.0
        for _ in range(batch):
            # Randomise the foreseen-correct policy AND start the crowd on either
            # side of it. Sometimes the city is fearful (below target) and must be
            # emboldened; sometimes overconfident (above target) and must be
            # restrained. This straddling is what teaches the counter-cyclical law.
            tgt = rng.uniform(-0.45, 0.45)
            ms = tgt + rng.uniform(-0.7, 0.7)   # mood offset to either side
            mg = rng.uniform(-0.1, 0.6)         # random starting passion
            sp = rng.uniform(0.3, 0.7)          # random division
            X0 = initial_crowd(polis.n, ms, mg, sp, rng)
            loss, traj, caches, info = polis.rollout(ctrl, X0, T, target=tgt)
            grads = polis.backward(ctrl, X0, T, caches, target=tgt)
            for k in gacc:
                gacc[k] += grads[k] / batch
            lacc += loss / batch
        opt.step(ctrl.params, gacc)
        hist.append(lacc)
        if verbose and (it % 50 == 0 or it == steps - 1):
            print(f"  step {it:4d} | loss {lacc:.4f} | track {info['loss_track']:.4f}"
                  f" | sep {info['loss_sep']:.4f} | orge {info['loss_orge']:.4f}"
                  f" | pronoia {info['loss_pro']:.4f}")
    return hist


# =============================================================================
# 7.  SELF-TESTS  —  each one a documented Periclean fact, made measurable
# =============================================================================
def test_counter_cyclical(polis, ctrl):
    """
    TEST 3 -- 'emboldens the fearful, restrains the overconfident.'
    Present the trained statesman with a PANIC crowd (stance far below target)
    and a HUBRIS crowd (stance above target). The stance-field f_s should point
    in OPPOSITE directions: up for panic, down for hubris. We sweep the mood and
    confirm f_s is negatively correlated with the error (mean_s - target).
    """
    print("-" * 72)
    print("TEST 3  Counter-cyclical steering (gnome against orge)")
    print("-" * 72)
    moods = polis.target + np.linspace(-0.9, 0.9, 9)
    fs_vals, errs = [], []
    rng = np.random.default_rng(5)
    for ms in moods:
        X0 = initial_crowd(polis.n, ms, 0.4, 0.4, rng)
        phi, mean_s = polis.build_features(X0, X0[:, 0].mean(), polis.target)
        out, _ = ctrl.forward(phi)
        fs_vals.append(out[0]); errs.append(mean_s - polis.target)
    fs_vals = np.array(fs_vals); errs = np.array(errs)
    corr = np.corrcoef(fs_vals, errs)[0, 1]
    print(f"  mood (mean stance): {np.round(moods,2)}")
    print(f"  field f_s emitted : {np.round(fs_vals,2)}")
    print(f"  corr(f_s, error)  : {corr:+.3f}   (want strongly NEGATIVE)")
    panic_push = fs_vals[0]; hubris_push = fs_vals[-1]
    print(f"  panic  -> push {panic_push:+.2f} (embolden, want >0)")
    print(f"  hubris -> push {hubris_push:+.2f} (restrain, want <0)")
    ok = corr < -0.7 and panic_push > 0 and hubris_push < 0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_first_man(polis, ctrl, T=12):
    """
    TEST 4 -- 'in word a democracy, in deed the rule of the first man.'
    Same chaotic crowds, two regimes: (a) WITH the Nous field, (b) ablated
    (free deliberation, zero field). Measure terminal tracking error and final
    dispersion. The steered city should reach the foreseen policy with concord;
    the free crowd should drift and stay divided.
    """
    print("-" * 72)
    print("TEST 4  The first man: steered demos vs free demos (ablation)")
    print("-" * 72)
    rng = np.random.default_rng(9)
    err_ctrl, err_free, var_ctrl, var_free = [], [], [], []
    for _ in range(40):
        ms = rng.uniform(-0.6, 0.6); mg = rng.uniform(0.0, 0.5); sp = rng.uniform(0.3, 0.7)
        X0 = initial_crowd(polis.n, ms, mg, sp, rng)
        _, _, _, info_c = polis.rollout(ctrl, X0, T)
        # ablate: a zero-output controller (no field at all)
        zero = NousField(hidden=ctrl.H)
        for kk in zero.params: zero.params[kk] *= 0.0
        _, _, _, info_f = polis.rollout(zero, X0, T)
        err_ctrl.append(abs(info_c["err_T"])); err_free.append(abs(info_f["err_T"]))
        var_ctrl.append(info_c["final_var"]);  var_free.append(info_f["final_var"])
    ec, ef = np.mean(err_ctrl), np.mean(err_free)
    vc, vf = np.mean(var_ctrl), np.mean(var_free)
    print(f"  terminal |error|  steered {ec:.3f}  vs  free {ef:.3f}")
    print(f"  final dispersion  steered {vc:.3f}  vs  free {vf:.3f}")
    ok = ec < ef * 0.6
    print(f"  steering cuts policy error by {100*(1-ec/ef):.0f}%")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_plague(polis, ctrl, T=16):
    """
    TEST 5 -- the plague (metabole): pronoia is real but bounded.
    Run the trained statesman on an ordinary crowd, then on the SAME crowd hit
    by a large unmodeled shock at mid-episode (panic surge + stance collapse).
    The statesman never trained on this. Measure how much worse the shocked run
    ends and how much it recovers from the worst moment -- encoding the historical
    truth that rational foresight was real, partial, and ultimately overrun.
    """
    print("-" * 72)
    print("TEST 5  The plague: an unmodeled shock the design cannot foresee")
    print("-" * 72)
    rng = np.random.default_rng(21)
    X0 = initial_crowd(polis.n, -0.2, 0.3, 0.4, rng)
    _, _, _, info_clean = polis.rollout(ctrl, X0, T)
    # Two waves, as in history (430 and 427 BCE): a heavy first blow, then a
    # lingering second one before the city can fully recover.
    waves = {T // 3: 2.2, 2 * T // 3: 1.6}
    _, traj, caches, info_plague = polis.rollout(ctrl, X0, T, plague=waves)
    seq = info_plague["mean_s_seq"]
    trough = seq.min()                 # darkest moment of the plague
    end = seq[-1]                      # mood at the end
    target = polis.target
    print(f"  clean run  : final |error| = {abs(info_clean['err_T']):.3f}")
    print(f"  plague run : final |error| = {abs(info_plague['err_T']):.3f}")
    print(f"  mood trough during plague = {trough:+.3f}, recovered to {end:+.3f} "
          f"(target {target:+.2f})")
    recovered = (end - trough) / max(1e-6, (target - trough))
    degraded = abs(info_plague["err_T"]) > abs(info_clean["err_T"]) + 1e-3
    print(f"  recovery fraction toward target after the worst moment: {recovered*100:.0f}%")
    print(f"  shock left the city measurably worse off              : {degraded}")
    # The 'historically faithful' result: it recovers PARTIALLY but is left worse.
    ok = (0.05 < recovered < 0.99) and degraded
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  "
          f"(foresight bounded: partial recovery, lasting damage)")
    return ok


# =============================================================================
# 8.  MAIN
# =============================================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 72)
    print(" THE NOUS FIELD — a Periclean cognitive architecture (pure NumPy)")
    print(" 'in word a democracy, in deed the rule of the first man' (Thuc. 2.65.9)")
    print("=" * 72)

    # (a) prove the gradients
    gradient_check()

    # (b) build the city and the statesman, then train
    print("-" * 72)
    print("TRAINING the Nous field on chaotic crowds")
    print("-" * 72)
    n = 40
    W = build_society(n, seed=429)
    r = RNG.uniform(0.5, 1.5, size=n)
    polis = Polis(W, r, target=0.3)
    ctrl = NousField(hidden=16)
    hist = train(polis, ctrl, T=12, steps=400, batch=8)
    print(f"  loss {hist[0]:.4f} -> {hist[-1]:.4f}  "
          f"({100*(1-hist[-1]/hist[0]):.0f}% reduction)")

    # (c) run the three historical self-tests
    r3 = test_counter_cyclical(polis, ctrl)
    r4 = test_first_man(polis, ctrl)
    r5 = test_plague(polis, ctrl)

    print("=" * 72)
    print(" SUMMARY")
    print(f"   gradient check ........ PASS")
    print(f"   counter-cyclical ...... {'PASS' if r3 else 'FAIL'}")
    print(f"   first-man ablation .... {'PASS' if r4 else 'FAIL'}")
    print(f"   plague robustness ..... {'PASS' if r5 else 'FAIL'}")
    allok = r3 and r4 and r5
    print(f"   OVERALL ............... {'ALL TESTS PASS' if allok else 'SOME FAILED'}")
    print("=" * 72)
    return allok


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0017_Amenemope_-1300.py  —  The Ger-Maa Cell ("The Truly Silent Man")

 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
======================================================================

A from-scratch, pure-NumPy recurrent agent that embodies the single cognitive
idea that is Amenemope's alone, and no one else's in this corpus:

    A mind has a *temperature*. The wise mind is the one that runs cool.
    Action is not something you do whenever you can; it is something you are
    *permitted* to do only once the heat of provocation has dissipated.

This is the thesis of the Instruction of Amenemope (Ramesside Egypt, c.1300-1075
BCE), built around the contrast between the "heated man" (the passionate,
reactive man) and the "truly silent man" (gr maa). In Amenemope's two-trees
simile the heated man is a tree grown in the open: a burst of fast growth, then
"the flame is its burial shroud" -- it is felled and floated away. The silent
man is a tree grown sheltered in a garden: slower, but it doubles its yield,
"stands before its lord", and "its end comes in the garden". Both are growth
regimes; the difference is thermal and temporal, not moral decoration.

We make that literal, with a deliberately SPLIT sensorium -- the design choice
that makes the whole thesis provable rather than merely pretty:

  * Channel 0 of every input is the *signed evidence* -- what the vote says.
    It, and it alone, drives the hidden "granary" state h (the candidate impulse
    u = tanh(W_h h + w_x*x0 + b_h)). This path never sees how LOUD a signal was.
  * Channel 1 is the *salience* -- how loud the signal was, |evidence|. It, and
    it alone, drives a SCALAR TEMPERATURE tau. tau rises with salience AND with
    the agent's own recent reactivity (acting heats you up -- the heated man
    burns himself out: the mu*m feedback term), and decays at rate rho.

A SILENCE GATE g = sigma(beta*(theta - tau)) is near 1 when cool (tau < theta)
and near 0 when hot. The gate multiplies every update to h. The split is the
point: because loudness lives only in the temperature path, the ONLY organ that
can notice a provocation is the thermostat, and the ONLY way to act on that
notice is the gate. An agent whose gate is forced open (the heated man) is
*structurally* blind to salience in its evidence path -- it cannot decline to
integrate a spike, and it cannot tell a lingering misleading whisper from an
honest one, because in the evidence channel they are identical.

Why this is not a Transformer, an RNN-with-attention, or "moderation":
  * No attention over stored keys. The mechanism is control-theoretic damping
    with hysteresis -- a thermostat on action, with a refractory period created
    by self-heating (the mu*m feedback term).
  * It is deliberately UNLIKE the nearest finished neighbour, #4 Ptahhotep
    ("wisdom through moderation; speech as a moral act"). Ptahhotep moderates a
    *quantity* -- a global low gain (a "volume knob"). That strategy is exactly
    what the heated ablation is free to try here, and no setting of it succeeds:
    honest and misleading votes are content-symmetric, so scaling the gain scales
    both together and cancels either way. What separates truth from lie is not
    amount but *timing* -- withholding integration across the heated interval that
    follows an alarm. Amenemope regulates a *temperature* and gates *timing*: the
    same impulse is folly while hot and wisdom once cool. Restraint here is a
    latency, not a volume knob.

Task ("The Two Trees"): the stream is built from matched RUNS. A misleading run
opens with a loud ALARM (high salience) and continues as a quiet refractory TAIL;
every step in it votes the WRONG way. An honest run is a quiet stretch of equal
length voting the RIGHT way. Honest and misleading runs come in equal numbers with
identical length distributions, so in the content channel the two classes are
perfectly symmetric -- the +cs and -cs votes cancel and nothing in the content
tells them apart. The ONLY signal that a run is misleading is the loud alarm that
opens it, and that lives solely in the salience channel. A reactive agent (gate
forced to 1, "the heated man") is blind to salience in its granary path: honest
and misleading votes cancel for it and it is pinned at chance. The silent agent
lets each alarm raise its temperature, holds the gate shut through the lingering
tail, and decides on the honest evidence it let in.

Contents (all from scratch, no autograd):
  - GerMaaCell : forward + analytic backprop-through-time for EVERY parameter.
  - grad_check : finite-difference gradient check (MANDATORY).
  - make_two_trees : the synthetic dataset.
  - train : a real Adam training loop.
  - Ablation : silent agent vs. heated agent (gate disabled).
  - Interpretability : confirm the trained gate closes on hot steps.
  - run_self_tests : shapes, gradient check, learning, ablation gap.

Run:  python3 chapter_0017_Amenemope_-1300.py
Author: David Vivancos · Chapter 0017 · Amenemope
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Stable primitives (pure NumPy, float64).                                    #
# --------------------------------------------------------------------------- #

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def softplus(z):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))

def smooth_abs(x, eps=1e-6):
    return np.sqrt(x * x + eps)

def smooth_abs_grad(x, eps=1e-6):
    return x / np.sqrt(x * x + eps)

def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# The Ger-Maa Cell.                                                           #
# --------------------------------------------------------------------------- #
#
# State carried across time per sequence: (h in R^H, tau in R, m in R).
# Split sensorium: xev = x_t[0] (signed evidence), xsal = x_t[1] (salience).
# For each step t (vectorised over a batch B):
#   z_s   = w_s * xsal + b_s                             # thermostat sees loudness
#   s_t   = softplus(z_s)                                # salience >= 0
#   tau_t = rho*tau_{t-1} + kappa*s_t + mu*m_{t-1}       # temperature dynamics
#   g_t   = sigmoid(beta*(theta - tau_t))                # SILENCE GATE
#   u_t   = tanh(W_h h_{t-1} + W_x * xev + b_h)          # granary sees only content
#   r_t   = u_t - h_{t-1}
#   delta = alpha * g_t * r_t                            # gated response
#   h_t   = h_{t-1} + delta
#   m_t   = mean_j delta[:,j]^2                          # self-heating (energy)
# Readout:  logits = h_T W_o^T + b_o ;  loss = softmax cross-entropy.
#
# The two pathways never cross: loudness (xsal) can only raise temperature and
# thus (through the gate) GATE integration; it can never itself be integrated as
# content. Content (xev) can only be integrated; it can never raise heat. This is
# what makes the heated ablation (gate == 1) structurally unable to notice or
# skip a provocation -- the only organ that sees loudness is the one it disabled.
#
# Constrained scalars stored "raw" and mapped:
#   rho=sigmoid(rho_raw), alpha=sigmoid(alpha_raw),
#   kappa=softplus(kappa_raw), mu=softplus(mu_raw), beta=softplus(beta_raw),
#   theta=theta_raw (free).
# --------------------------------------------------------------------------- #

PARAM_KEYS = ["w_s", "b_s", "rho_raw", "alpha_raw", "kappa_raw", "mu_raw",
              "beta_raw", "theta", "W_h", "W_x", "b_h", "W_o", "b_o"]


class GerMaaCell:
    def __init__(self, D, H, C, seed=0):
        rng = np.random.default_rng(seed)
        self.D, self.H, self.C = D, H, C
        sh = np.sqrt(1.0 / H)
        self.p = {
            # --- temperature (thermostat) path: sees ONLY salience, channel 1 ---
            "w_s":       np.array(0.7),     # scalar: salience -> provocation heat
            "b_s":       np.array(-0.5),
            "rho_raw":   np.array(1.2),     # rho   ~ 0.769  (heat retention)
            "alpha_raw": np.array(0.0),     # alpha = 0.5    (integration step)
            "kappa_raw": np.array(0.0),     # kappa ~ 0.693  (provocation->heat)
            "mu_raw":    np.array(-1.0),    # mu    ~ 0.313  (self-heating)
            "beta_raw":  np.array(1.0),     # beta  ~ 1.313  (gate sharpness)
            "theta":     np.array(1.0),     # heat threshold for silence
            # --- evidence (granary) path: sees ONLY signed evidence, channel 0 --
            "W_h":       rng.normal(0, sh, size=(H, H)),
            "W_x":       rng.normal(0, sh, size=H),   # (H,): signed evidence -> u
            "b_h":       np.zeros(H),
            "W_o":       rng.normal(0, sh, size=(C, H)),
            "b_o":       np.zeros(C),
        }

    def _mapped(self):
        p = self.p
        rho   = float(sigmoid(p["rho_raw"]))
        alpha = float(sigmoid(p["alpha_raw"]))
        kappa = float(softplus(p["kappa_raw"]))
        mu    = float(softplus(p["mu_raw"]))
        beta  = float(softplus(p["beta_raw"]))
        theta = float(p["theta"])
        dmap = {
            "rho":   rho * (1 - rho),
            "alpha": alpha * (1 - alpha),
            "kappa": float(sigmoid(p["kappa_raw"])),
            "mu":    float(sigmoid(p["mu_raw"])),
            "beta":  float(sigmoid(p["beta_raw"])),
        }
        return rho, alpha, kappa, mu, beta, theta, dmap

    # --------------------------- forward --------------------------------- #
    def forward(self, X, silent=True):
        B, T, D = X.shape
        H, p = self.H, self.p
        rho, alpha, kappa, mu, beta, theta, _ = self._mapped()

        h = np.zeros((B, H)); tau = np.zeros(B); m = np.zeros(B)
        steps = []
        for t in range(T):
            x = X[:, t, :]
            x_ev  = x[:, 0]                 # signed evidence -> granary path only
            x_sal = x[:, 1]                 # salience (loudness) -> thermostat only
            z_s = p["w_s"] * x_sal + p["b_s"]
            s = softplus(z_s)
            tau_new = rho * tau + kappa * s + mu * m
            if silent:
                a_gate = beta * (theta - tau_new)
                g = sigmoid(a_gate)
            else:
                a_gate, g = None, np.ones(B)
            pre = h @ p["W_h"].T + x_ev[:, None] * p["W_x"][None, :] + p["b_h"]
            u = np.tanh(pre)
            r = u - h
            delta = alpha * g[:, None] * r
            h_new = h + delta
            m_new = (delta * delta).mean(axis=1)   # self-heating = energy of response
            steps.append(dict(x_ev=x_ev, x_sal=x_sal, h_prev=h, tau_prev=tau,
                              m_prev=m, z_s=z_s,
                              s=s, tau=tau_new, a_gate=a_gate, g=g, u=u, r=r,
                              delta=delta))
            h, tau, m = h_new, tau_new, m_new

        logits = h @ p["W_o"].T + p["b_o"]
        cache = dict(steps=steps, hT=h, X=X, silent=silent,
                     mapped=(rho, alpha, kappa, mu, beta, theta))
        return logits, cache

    def loss(self, logits, y):
        B = logits.shape[0]
        probs = softmax_rows(logits)
        ll = -np.log(probs[np.arange(B), y] + 1e-12)
        return ll.mean(), probs

    # --------------------------- backward -------------------------------- #
    # Exact BPTT. Dependencies handled per step:
    #   h_t -> h_{t+1} (identity + via u_{t+1}) and -> logits (t=T)
    #   tau_t -> g_t (this step) and -> tau_{t+1} (via rho)
    #   m_t -> tau_{t+1} (via mu)   [so its grad lands on delta_t at step t]
    def backward(self, cache, probs, y):
        steps, X, silent = cache["steps"], cache["X"], cache["silent"]
        rho, alpha, kappa, mu, beta, theta = cache["mapped"]
        B, T, D = X.shape
        H, C, p = self.H, self.C, self.p
        _, _, _, _, _, _, dmap = self._mapped()

        g_ = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
              for k, v in p.items()}

        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        g_["W_o"] += dlogits.T @ cache["hT"]
        g_["b_o"] += dlogits.sum(axis=0)
        dh = dlogits @ p["W_o"]            # grad wrt h_T

        dtau_after = np.zeros(B)           # grad wrt tau_{t+1}
        d_rho = d_alpha = d_kappa = d_mu = d_beta = d_theta = 0.0

        for t in reversed(range(T)):
            st = steps[t]
            x_ev, x_sal, h_prev = st["x_ev"], st["x_sal"], st["h_prev"]
            tau_prev, m_prev = st["tau_prev"], st["m_prev"]
            s, z_s, g, u, r, delta = (st["s"], st["z_s"], st["g"],
                                      st["u"], st["r"], st["delta"])

            # m_t feeds tau_{t+1}; its gradient lands on delta_t here:
            dm_t = mu * dtau_after                                   # (B,)
            ddelta = dh + (dm_t[:, None] * (2.0 * delta)) / H

            # delta = alpha * g * r
            d_alpha += float(np.sum(ddelta * (g[:, None] * r)))
            dg = np.sum(ddelta * (alpha * r), axis=1)               # (B,)
            dr = ddelta * (alpha * g[:, None])                      # (B,H)

            # r = u - h_prev ; and h_t = h_{t-1} + delta_t gives a direct
            # residual path: grad on h_t flows to h_{t-1} with coefficient 1.
            du = dr
            dh_prev = dh.copy() - dr            # residual (dh) + (-h_prev in r)

            # u = tanh(pre)
            dpre = du * (1.0 - u * u)
            g_["W_h"] += dpre.T @ h_prev
            g_["W_x"] += (dpre * x_ev[:, None]).sum(axis=0)   # (H,) evidence path
            g_["b_h"] += dpre.sum(axis=0)
            dh_prev += dpre @ p["W_h"]

            # temperature grad: carry from tau_{t+1} plus gate (this step)
            dtau_t = rho * dtau_after
            if silent:
                da = dg * (g * (1.0 - g))           # through sigmoid
                d_beta += float(np.sum(da * (theta - st["tau"])))
                d_theta += float(np.sum(da * beta))
                dtau_t += da * (-beta)

            # tau_t = rho*tau_prev + kappa*s + mu*m_prev
            d_rho += float(np.sum(dtau_t * tau_prev))
            d_kappa += float(np.sum(dtau_t * s))
            d_mu += float(np.sum(dtau_t * m_prev))
            ds = dtau_t * kappa
            # NOTE: the grad wrt tau_{t-1} (rho path) and wrt m_{t-1} (mu path)
            # are produced INSIDE the next iteration from dtau_after = dtau_t:
            #   dtau_{t-1} gets  rho*dtau_after ;  dm_{t-1} gets mu*dtau_after.

            # s = softplus(z_s); z_s = w_s * x_sal + b_s   (salience path)
            dz = ds * sigmoid(z_s)
            g_["w_s"] += np.sum(dz * x_sal)
            g_["b_s"] += np.sum(dz)

            # carry to previous step
            dh = dh_prev
            dtau_after = dtau_t            # total grad wrt tau_t (NOT rho-scaled)

        g_["rho_raw"]   = np.array(d_rho   * dmap["rho"])
        g_["alpha_raw"] = np.array(d_alpha * dmap["alpha"])
        g_["kappa_raw"] = np.array(d_kappa * dmap["kappa"])
        g_["mu_raw"]    = np.array(d_mu    * dmap["mu"])
        g_["beta_raw"]  = np.array(d_beta  * dmap["beta"])
        g_["theta"]     = np.array(d_theta)
        return g_

    # convenience: forward+loss+backward
    def loss_and_grads(self, X, y, silent=True):
        logits, cache = self.forward(X, silent=silent)
        L, probs = self.loss(logits, y)
        grads = self.backward(cache, probs, y)
        return L, grads


# --------------------------------------------------------------------------- #
# Finite-difference gradient check (MANDATORY).                               #
# --------------------------------------------------------------------------- #

def grad_check(seed=1, eps=1e-6):
    rng = np.random.default_rng(seed)
    D, H, C, B, T = 2, 4, 2, 3, 5
    net = GerMaaCell(D, H, C, seed=seed)
    X = rng.normal(size=(B, T, D))
    y = rng.integers(0, C, size=B)

    L0, grads = net.loss_and_grads(X, y, silent=True)

    worst = 0.0
    for k in PARAM_KEYS:
        arr = np.asarray(net.p[k], dtype=np.float64)
        flat = arr.reshape(-1)
        gflat = np.asarray(grads[k], dtype=np.float64).reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            net.p[k] = flat.reshape(arr.shape) if arr.shape else np.array(flat[i])
            Lp, _ = net.forward(X, silent=True)
            Lp, _ = net.loss(Lp, y)
            flat[i] = orig - eps
            net.p[k] = flat.reshape(arr.shape) if arr.shape else np.array(flat[i])
            Lm, _ = net.forward(X, silent=True)
            Lm, _ = net.loss(Lm, y)
            flat[i] = orig
            net.p[k] = flat.reshape(arr.shape) if arr.shape else np.array(orig)
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# --------------------------------------------------------------------------- #
# Dataset: "The Two Trees" (loud provocations + randomized refractory tails).   #
# --------------------------------------------------------------------------- #
# x[:,0] = signed evidence (content).   x[:,1] = salience (loudness = |content|).
#
#   cool step      : x0 =  class_sign*cool_amp + noise           (quiet, TRUE)
#   spike step     : x0 = random_sign*hot_amp  + noise           (LOUD,  random)
#   refractory tail: x0 = -class_sign*cool_amp + noise           (quiet, FALSE)
#
# The label depends only on the honest cool votes. To read it, an agent must
# (a) shut the gate on each loud spike, and (b) KEEP it shut through the tail --
# whose salience matches the honest votes, so only a temperature that *lingers
# and decays* can bridge it. A reactive integrator (gate=1) must swallow the
# large random spikes and the misleading tails, and -- because the number of
# each is randomized per sequence -- cannot recover the label by any fixed or
# inverted readout. See make_two_trees.__doc__ for the full argument.
# --------------------------------------------------------------------------- #

def make_two_trees(n, T=24, cool_amp=0.45, hot_amp=5.0, noise=0.18,
                   spikes=(1, 3), seed=0):
    """The Two Trees task -- a sequence-classification problem engineered so the
    thermostat is *provably* necessary, and so NO gate-less strategy (low gain,
    saturation, readout inversion, or reconstructing the gate from the content
    stream) can quietly solve it.

    The two input channels are deliberately DECOUPLED:

        channel 0  x0 = signed CONTENT  (what a step votes; magnitude ~cool_amp
                        for every step, so content never betrays a provocation)
        channel 1  x1 = SALIENCE / alarm (how loud a step is; large only on a
                        spike -- a pure loudness event carrying no usable content)

    Three kinds of step:

        honest vote : x0 = +class_sign*cool_amp + noise ;  x1 ~ cool_amp   (TRUE)
        misleading  : x0 = -class_sign*cool_amp + noise ;  x1 ~ cool_amp   (FALSE)
          tail        (a "tail" always immediately follows a spike)
        spike       : x0 =  (+/-1)*cool_amp + noise      ;  x1 ~ hot_amp   (ALARM)

    Why the gate is the only way through:

      * In the CONTENT channel, all three step types have the same magnitude
        (~cool_amp). A spike is not loud in content -- only in salience. So an
        agent reading only content (which is exactly what the heated ablation's
        granary path does) has NO cue that a provocation happened, and therefore
        no way to know that the votes right after it are poisoned. It cannot
        reconstruct the gate from its own stream.

      * Honest votes and misleading tails are produced in EQUAL numbers per
        sequence and carry exactly opposite signs. To a blind integrator they
        cancel: the net content of the whole stream is ~0, independent of the
        label. No fixed or inverted readout can recover the class from it. The
        class survives only if the tails are EXCLUDED -- and they can be excluded
        only by remembering the loud alarm that preceded them, i.e. by a
        temperature that lingers and a gate that stays shut through the tail.

    The silent agent lets salience raise its temperature on each alarm, holds the
    gate shut through the lingering tail, and integrates the honest votes it let
    in -- recovering the class cleanly. The heated agent (gate == 1) must integrate
    honest and tail alike; they cancel; it is left at chance.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, 2))
    y = rng.integers(0, 2, size=n)
    class_sign = np.where(y == 1, 1.0, -1.0)
    for i in range(n):
        cs = class_sign[i]
        k = int(rng.integers(spikes[0], spikes[1] + 1))     # provocations here
        # The sequence is built from matched RUNS.  Each misleading run is a loud
        # ALARM step (content = -cs, salience = hot) followed by quiet TAIL steps
        # (content = -cs, salience = cool).  Each honest run is quiet steps
        # (content = +cs, salience = cool) of an IDENTICAL length.  Honest and
        # misleading runs come in equal numbers with equal length distributions,
        # so in the CONTENT channel the two classes are perfectly symmetric --
        # +cs votes and -cs votes are equinumerous, in runs of identical shape,
        # with no internal sign-flips.  A gate-less integrator sees zero net
        # content and no distinguishing pattern: it is pinned at chance.  The
        # ONLY thing that separates a misleading run from an honest one is the
        # loud alarm that opens it -- visible solely in the salience channel, and
        # actionable solely through the gate.
        half = T // 2
        base, rem = divmod(half, k)
        run_len = [base + 1 if j < rem else base for j in range(k)]
        run_len = [max(1, L) for L in run_len]
        while sum(run_len) > half:                           # trim clamp overflow
            j = max(range(k), key=lambda j: run_len[j])
            if run_len[j] > 1:
                run_len[j] -= 1
            else:
                break
        mislead_blocks = [[1] + [2] * (L - 1) for L in run_len]   # alarm + tails
        honest_blocks  = [[0] * L for L in run_len]               # matched honest
        blocks = mislead_blocks + honest_blocks
        rng.shuffle(blocks)
        kind = [s for blk in blocks for s in blk]
        # pad any rounding leftover in balanced (+cs,-cs) pairs to keep symmetry
        while len(kind) + 2 <= T:
            kind.append(0); kind.append(2)
        while len(kind) < T:
            kind.append(0)
        kind = kind[:T]
        for t in range(T):
            if kind[t] == 1:                                   # loud misleading alarm
                x0 = -cs * cool_amp + rng.normal(0, noise)
                x1 = hot_amp + rng.normal(0, noise)
            elif kind[t] == 2:                                 # quiet misleading tail
                x0 = -cs * cool_amp + rng.normal(0, noise)
                x1 = cool_amp + rng.normal(0, noise)
            else:                                              # honest cool vote
                x0 = cs * cool_amp + rng.normal(0, noise)
                x1 = cool_amp + rng.normal(0, noise)
            X[i, t, 0] = x0
            X[i, t, 1] = abs(x1)        # salience: loudness, decoupled from content
    return X, y


# --------------------------------------------------------------------------- #
# Adam optimiser (from scratch) + training loop.                              #
# --------------------------------------------------------------------------- #

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                  for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                  for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = np.asarray(grads[k], dtype=np.float64)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            params[k] = np.asarray(params[k], dtype=np.float64) - upd


def accuracy(net, X, y, silent=True):
    logits, _ = net.forward(X, silent=silent)
    return float((logits.argmax(axis=1) == y).mean())


def train(net, Xtr, ytr, Xte, yte, silent=True, epochs=60, bs=64,
          lr=0.02, seed=0, log=True, tag=""):
    rng = np.random.default_rng(seed)
    opt = Adam(net.p, lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            L, grads = net.loss_and_grads(Xtr[b], ytr[b], silent=silent)
            opt.step(net.p, grads)
            tot += L * len(b)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            tr = accuracy(net, Xtr, ytr, silent=silent)
            te = accuracy(net, Xte, yte, silent=silent)
            print(f"  [{tag}] epoch {ep:3d}  loss {tot/n:.4f}"
                  f"  train_acc {tr:.3f}  test_acc {te:.3f}")
    return accuracy(net, Xte, yte, silent=silent)


# --------------------------------------------------------------------------- #
# Interpretability: does the trained gate actually close on hot steps?        #
# --------------------------------------------------------------------------- #

def gate_report(net, X, hot_thresh=2.0):
    _, cache = net.forward(X, silent=True)
    steps = cache["steps"]
    g_hot, g_cool = [], []
    for t, st in enumerate(steps):
        hot_mask = X[:, t, 1] > hot_thresh        # spike magnitude >> cool/tail
        if hot_mask.any():
            g_hot.append(st["g"][hot_mask].mean())
        if (~hot_mask).any():
            g_cool.append(st["g"][~hot_mask].mean())
    return float(np.mean(g_cool)), float(np.mean(g_hot))


# --------------------------------------------------------------------------- #
# Self-tests.                                                                  #
# --------------------------------------------------------------------------- #

def run_self_tests():
    print("=" * 70)
    print("Ger-Maa Cell  --  Amenemope's thermostatic architecture  (self-tests)")
    print("=" * 70)

    # 1) shapes
    net = GerMaaCell(D=2, H=8, C=2, seed=0)
    Xs, ys = make_two_trees(12, T=24, seed=123)
    logits, _ = net.forward(Xs)
    assert logits.shape == (12, 2), "logit shape wrong"
    print(f"[1] forward shapes OK : logits {logits.shape}")

    # 2) gradient check (the mandatory one)
    worst = grad_check()
    print(f"[2] finite-diff gradient check : worst relative error = {worst:.3e}")
    assert worst < 1e-5, "gradient check failed"
    print("    -> analytic BPTT matches finite differences (< 1e-5).")

    # 3) data + training (the silent man)
    Xtr, ytr = make_two_trees(2400, T=24, seed=1)
    Xte, yte = make_two_trees(800,  T=24, seed=2)
    print("[3] training the SILENT agent (gate active):")
    net_s = GerMaaCell(D=2, H=8, C=2, seed=7)
    acc_silent = train(net_s, Xtr, ytr, Xte, yte, silent=True,
                       epochs=60, lr=0.02, tag="silent")

    # 4) ablation: the heated man (gate disabled)
    print("[4] training the HEATED agent (gate forced to 1):")
    net_h = GerMaaCell(D=2, H=8, C=2, seed=7)
    acc_heated = train(net_h, Xtr, ytr, Xte, yte, silent=False,
                       epochs=60, lr=0.02, tag="heated")

    # 5) interpretability: gate closes on hot steps for the silent agent
    g_cool, g_hot = gate_report(net_s, Xte)
    print(f"[5] silent agent gate openness  ->  cool steps: {g_cool:.3f}"
          f"   hot steps: {g_hot:.3f}")

    print("-" * 70)
    print(f"RESULT  silent test_acc = {acc_silent:.3f}   |   "
          f"heated test_acc = {acc_heated:.3f}")
    print(f"        the gate is {'MORE' if g_cool > g_hot else 'NOT more'} "
          f"open when cool than when hot  "
          f"(delta = {g_cool - g_hot:+.3f})")
    learned = acc_silent > 0.85
    silent_wins = acc_silent > acc_heated + 0.08
    gate_works = g_cool > g_hot + 0.10
    print(f"        learning>{0.85}: {learned} | "
          f"silent beats heated: {silent_wins} | "
          f"gate shuts on heat: {gate_works}")
    assert learned and silent_wins and gate_works, "behavioural tests failed"
    print("=" * 70)
    print("ALL SELF-TESTS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(0)
    run_self_tests()

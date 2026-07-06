#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 chapter_0097_Polybius_-200.py ANACYCLOSIS DYNAMICAL NETWORK (ADN)
 Polybius of Megalopolis (c. 200 - c. 118 BCE)
 A from-scratch, trainable NumPy architecture that encodes Polybius' mind.
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0097 · Polybius of Megalopolis
==============================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Polybius' single most original cognitive instrument is not "balance of power"
(every constitutional thinker shares that) but ANACYCLOSIS: the claim that
*every* successful order carries inside it the specific corruption that will
destroy it, that the decay is structural and predictable rather than
accidental, and that only powers woven together so they can each VETO the
others can "brake" — never escape — the cycle. He paired this with a tripartite
theory of causation (aitia = the true underlying cause, prophasis = the
pretext, arche = the first overt act) and with a hard rule that real knowledge
comes from operating a system, not reading about it (he mocked the "armchair
historian").

A transformer stores keys and attends over them. That is the wrong primitive
for a mind whose core object is a *dynamical system that decays from its own
success*. So this network is built as a small learned DYNAMICAL SYSTEM. Its
hidden state is literally the balance of power among three "estates"
(monarchic / aristocratic / democratic). The network learns a coupling
operator C — the "constitution" — and then ROLLS THE STATE FORWARD in time.
Skewed constitutions become unstable trajectories that spiral (anacyclosis);
mixed constitutions settle. Readout heads then diagnose the system:

  * stability      (binary)  — will this order endure, or decay?
  * longevity      (scalar)  — how long until collapse? (Polybius' applied use)
  * phase          (6-class) — which point of the cycle is it sliding toward?
                               monarchy/tyranny/aristocracy/oligarchy/
                               democracy/ochlocracy
  * root cause     (3-class) — the AITIA: which estate's corruption is the
                               true driver of decay (vs. mere pretext)

The whole thing is trained end to end with hand-written backprop, including
back-propagation THROUGH the temporal rollout (BPTT). A finite-difference
gradient check (mandatory) verifies every analytic gradient. Then a real
training loop learns the Polybian invariant from synthetic "constitutions",
and self-tests confirm the learned model reproduces his central prediction:
mixed orders outlive pure ones.

Everything below is pure NumPy. No autograd library. float64 throughout so the
gradient check is tight.

Run:  python3 chapter_0097_Polybius_-200.py
"""

import numpy as np

# Reproducibility. 97 = this figure's index in the corpus.
RNG = np.random.default_rng(97)


# ============================================================================
# SECTION 1 — SMALL DIFFERENTIABLE PRIMITIVES
# ----------------------------------------------------------------------------
# Each helper returns the value plus whatever it needs to push a gradient back
# through itself. We keep them tiny and explicit so the math is auditable —
# very much in Polybius' spirit of testing every claim against the evidence.
# ============================================================================

def tanh(x):
    """Bounded nonlinearity. Bounded growth matters here: an estate's power
    cannot run to infinity in one step — corruption saturates."""
    return np.tanh(x)


def dtanh_from_out(t):
    """Derivative of tanh given its OUTPUT t = tanh(x):  d/dx = 1 - t^2."""
    return 1.0 - t * t


def softmax(z):
    """Numerically stable softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xavier(shape, gain=1.0):
    """Xavier/Glorot init keeps signal variance stable across the encoder."""
    fan_in, fan_out = shape[0], shape[-1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return RNG.uniform(-limit, limit, size=shape)


# ----- loss functions (each returns scalar loss and gradient wrt logits) -----

def bce_with_logits(logit, target):
    """Binary cross-entropy on a raw logit. Stable form.
    Returns (mean_loss, dlogit) with dlogit shaped like logit."""
    # loss = max(z,0) - z*y + log(1+exp(-|z|))
    z = logit
    loss = np.maximum(z, 0) - z * target + np.log1p(np.exp(-np.abs(z)))
    grad = (sigmoid(z) - target) / z.shape[0]
    return loss.mean(), grad


def mse(pred, target):
    """Mean squared error for the longevity regression head."""
    diff = pred - target
    loss = np.mean(diff * diff)
    grad = (2.0 / pred.shape[0]) * diff
    return loss, grad


def softmax_ce(logits, labels):
    """Softmax cross-entropy. labels are integer class ids.
    Returns (mean_loss, dlogits)."""
    p = softmax(logits)
    n = logits.shape[0]
    logp = np.log(p[np.arange(n), labels] + 1e-12)
    loss = -logp.mean()
    grad = p.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return loss, grad


# ============================================================================
# SECTION 2 — THE MODEL
# ----------------------------------------------------------------------------
# Forward pass, then a single backward() that returns gradients for EVERY
# parameter, including the back-prop through the temporal rollout.
# ============================================================================

class AnacyclosisNet:
    """
    Pipeline (all per-example, vectorised over a batch B):

      x  --encoder-->  h0
      h0 --linear-->   s0   (initial balance of power, R^3)
      h0 --linear-->   C    (the 'constitution': a 3x3 coupling operator)
      rollout T steps: s_t = s_{t-1} + dt * tanh(C @ s_{t-1})    [anacyclosis]
      trajectory ----> g    (final state, time-mean, time-mean-square)
      g  --heads-->    stability / longevity / phase / root-cause
    """

    def __init__(self, in_dim=8, hid=24, T=12, dt=0.35, n_phase=6, n_cause=3):
        self.in_dim, self.hid, self.T, self.dt = in_dim, hid, T, dt
        self.n_phase, self.n_cause = n_phase, n_cause
        self.feat_dim = 9  # s_T(3) + mean(3) + meansq(3)

        # Encoder: x -> h0
        self.W1 = xavier((in_dim, hid))
        self.b1 = np.zeros(hid)

        # h0 -> initial estate state s0  (R^3)
        self.Ws = xavier((hid, 3)) * 0.5
        self.bs = np.zeros(3)

        # h0 -> flattened 3x3 constitution C
        self.Wc = xavier((hid, 9)) * 0.5
        self.bc = np.zeros(9)

        # Diagnostic heads on trajectory features g
        self.w_stab = xavier((self.feat_dim, 1));   self.b_stab = np.zeros(1)
        self.w_long = xavier((self.feat_dim, 1));   self.b_long = np.zeros(1)
        self.W_phase = xavier((self.feat_dim, n_phase)); self.b_phase = np.zeros(n_phase)
        self.W_cause = xavier((self.feat_dim, n_cause)); self.b_cause = np.zeros(n_cause)

    # ---- parameter registry: name -> array (used by optimiser + grad check) --
    def params(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "Ws": self.Ws, "bs": self.bs,
            "Wc": self.Wc, "bc": self.bc,
            "w_stab": self.w_stab, "b_stab": self.b_stab,
            "w_long": self.w_long, "b_long": self.b_long,
            "W_phase": self.W_phase, "b_phase": self.b_phase,
            "W_cause": self.W_cause, "b_cause": self.b_cause,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        """Returns outputs dict and caches everything backward() needs."""
        B = x.shape[0]
        cache = {"x": x, "B": B}

        # --- encoder ---
        z1 = x @ self.W1 + self.b1
        h0 = tanh(z1)
        cache["h0"] = h0

        # --- project to s0 and C ---
        s0 = h0 @ self.Ws + self.bs                    # (B,3)
        Cflat = h0 @ self.Wc + self.bc                 # (B,9)
        C = Cflat.reshape(B, 3, 3)                      # (B,3,3) the constitution
        cache["C"] = C

        # --- temporal rollout: the anacyclosis simulation ---
        # s_t = s_{t-1} + dt * tanh(C @ s_{t-1})
        s_list = [s0]
        ta_list = [None]                               # tanh outputs per step
        for t in range(1, self.T + 1):
            a = np.einsum("bij,bj->bi", C, s_list[-1]) # (B,3) estate driving
            ta = tanh(a)
            s = s_list[-1] + self.dt * ta
            s_list.append(s)
            ta_list.append(ta)
        cache["s_list"] = s_list
        cache["ta_list"] = ta_list

        # --- trajectory features g ---
        S = np.stack(s_list, axis=1)                   # (B, T+1, 3)
        M = S.shape[1]
        s_final = s_list[-1]                            # (B,3)
        s_mean = S.mean(axis=1)                         # (B,3)
        s_msq = (S * S).mean(axis=1)                    # (B,3) instability proxy
        g = np.concatenate([s_final, s_mean, s_msq], axis=1)  # (B,9)
        cache["S"] = S; cache["M"] = M; cache["g"] = g

        # --- diagnostic heads ---
        stab_logit = (g @ self.w_stab + self.b_stab)[:, 0]      # (B,)
        longevity = (g @ self.w_long + self.b_long)[:, 0]       # (B,)
        phase_logits = g @ self.W_phase + self.b_phase          # (B,n_phase)
        cause_logits = g @ self.W_cause + self.b_cause          # (B,n_cause)

        out = {
            "stab_logit": stab_logit,
            "longevity": longevity,
            "phase_logits": phase_logits,
            "cause_logits": cause_logits,
        }
        cache["out"] = out
        self._cache = cache
        return out, cache

    # --------------------------------------------------------------- loss
    def loss(self, out, batch, weights=(1.0, 0.5, 1.0, 0.5)):
        """Composite loss. weights = (stab, long, phase, cause)."""
        ws, wl, wp, wc = weights
        L_stab, d_stab = bce_with_logits(out["stab_logit"], batch["stab"])
        L_long, d_long = mse(out["longevity"], batch["long"])
        L_phase, d_phase = softmax_ce(out["phase_logits"], batch["phase"])
        L_cause, d_cause = softmax_ce(out["cause_logits"], batch["cause"])
        total = ws * L_stab + wl * L_long + wp * L_phase + wc * L_cause
        grads_out = {
            "stab_logit": ws * d_stab,
            "longevity": wl * d_long,
            "phase_logits": wp * d_phase,
            "cause_logits": wc * d_cause,
        }
        parts = {"stab": L_stab, "long": L_long, "phase": L_phase, "cause": L_cause}
        return total, grads_out, parts

    # ----------------------------------------------------------- backward
    def backward(self, grads_out):
        """Full reverse pass. Returns grads dict matching params()."""
        c = self._cache
        B = c["B"]; M = c["M"]
        g = c["g"]; C = c["C"]; h0 = c["h0"]; x = c["x"]
        s_list = c["s_list"]; ta_list = c["ta_list"]

        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        # ---- heads: gradients into feature vector g ----
        dg = np.zeros_like(g)

        # stability head (logit shaped (B,))
        d_stab = grads_out["stab_logit"][:, None]               # (B,1)
        grads["w_stab"] += g.T @ d_stab
        grads["b_stab"] += d_stab.sum(axis=0)
        dg += d_stab @ self.w_stab.T

        # longevity head
        d_long = grads_out["longevity"][:, None]                # (B,1)
        grads["w_long"] += g.T @ d_long
        grads["b_long"] += d_long.sum(axis=0)
        dg += d_long @ self.w_long.T

        # phase head
        d_phase = grads_out["phase_logits"]                     # (B,n_phase)
        grads["W_phase"] += g.T @ d_phase
        grads["b_phase"] += d_phase.sum(axis=0)
        dg += d_phase @ self.W_phase.T

        # cause head
        d_cause = grads_out["cause_logits"]                     # (B,n_cause)
        grads["W_cause"] += g.T @ d_cause
        grads["b_cause"] += d_cause.sum(axis=0)
        dg += d_cause @ self.W_cause.T

        # ---- split dg back onto the trajectory ----
        # g = [s_final(3) | mean(3) | meansq(3)]
        dg_final = dg[:, 0:3]
        dg_mean = dg[:, 3:6]
        dg_msq = dg[:, 6:9]

        # direct feature gradients arriving at each state s_t
        ds_feat = [np.zeros((B, 3)) for _ in range(M)]
        for t in range(M):
            ds_feat[t] += dg_mean / M                           # from time-mean
            ds_feat[t] += (2.0 * s_list[t] * dg_msq) / M        # from mean-square
        ds_feat[M - 1] += dg_final                              # from final state

        # ---- BPTT through the rollout ----
        # s_t = s_{t-1} + dt * tanh(C @ s_{t-1})
        gs = [ds_feat[t].copy() for t in range(M)]              # accumulators
        dC = np.zeros((B, 3, 3))
        for t in range(self.T, 0, -1):
            u = gs[t]                                           # grad at s_t (B,3)
            ta = ta_list[t]                                    # tanh output (B,3)
            dta = self.dt * u
            da = dta * dtanh_from_out(ta)                      # grad wrt a (B,3)
            # path 1: s_{t-1} appears directly (identity)
            gs[t - 1] += u
            # path 2: s_{t-1} -> a = C @ s_{t-1}
            gs[t - 1] += np.einsum("bij,bi->bj", C, da)       # C^T @ da
            dC += np.einsum("bi,bj->bij", da, s_list[t - 1])  # outer product
        ds0 = gs[0]                                            # grad wrt s0

        # ---- C and s0 came from h0 ----
        dCflat = dC.reshape(B, 9)
        grads["Wc"] += h0.T @ dCflat
        grads["bc"] += dCflat.sum(axis=0)
        dh0 = dCflat @ self.Wc.T

        grads["Ws"] += h0.T @ ds0
        grads["bs"] += ds0.sum(axis=0)
        dh0 += ds0 @ self.Ws.T

        # ---- encoder ----
        dz1 = dh0 * dtanh_from_out(h0)
        grads["W1"] += x.T @ dz1
        grads["b1"] += dz1.sum(axis=0)

        return grads


# ============================================================================
# SECTION 3 — SYNTHETIC "CONSTITUTIONS" (the Polybian world to learn)
# ----------------------------------------------------------------------------
# We manufacture constitutions and label them by Polybius' own rules so the
# network can recover the invariant. Each example:
#   x[0:3]  power shares of (monarchic, aristocratic, democratic), sum~1
#   x[3:6]  corruption susceptibility per estate in [0,1]
#   x[6]    'checks' — degree of mutual veto / mixedness in [0,1]
#   x[7]    external shock in [0,1]
# Labels:
#   stab  : 1 if the order is mixed+balanced enough to endure, else 0
#   long  : longevity (normalised) — higher for mixed orders (Polybius' point)
#   phase : the corner of the cycle it slides toward (6 classes)
#   cause : the AITIA — which estate's corruption truly drives the decay
# ============================================================================

# phase ids: 0 monarchy,1 tyranny,2 aristocracy,3 oligarchy,4 democracy,5 ochlocracy
VIRTUOUS = {0: 0, 1: 2, 2: 4}      # estate idx -> virtuous phase id
CORRUPT = {0: 1, 1: 3, 2: 5}       # estate idx -> corrupt phase id


def make_dataset(n):
    X = np.zeros((n, 8))
    stab = np.zeros(n)
    long = np.zeros(n)
    phase = np.zeros(n, dtype=int)
    cause = np.zeros(n, dtype=int)

    for i in range(n):
        # random power shares via a Dirichlet — sometimes balanced, sometimes skewed
        conc = RNG.choice([0.4, 1.0, 6.0])     # low conc -> skewed; high -> balanced
        powers = RNG.dirichlet([conc, conc, conc])
        corr = RNG.uniform(0, 1, size=3)       # corruption susceptibility per estate
        checks = RNG.uniform(0, 1)
        shock = RNG.uniform(0, 1)
        X[i] = np.concatenate([powers, corr, [checks], [shock]])

        # --- Polybian bookkeeping ---
        balance = 1.0 - (powers.max() - powers.min())   # 1 when perfectly equal
        mixedness = balance * checks                     # need balance AND vetoes
        dom = int(np.argmax(powers))                     # dominant estate

        # stability: mixed orders endure; pure/unchecked orders decay
        stab[i] = 1.0 if mixedness > 0.45 else 0.0

        # longevity: rises with mixedness, falls with shock and dominant corruption
        base = 0.25 + 1.6 * mixedness - 0.5 * shock - 0.4 * corr[dom]
        long[i] = base + RNG.normal(0, 0.03)

        # phase it slides toward: dominant estate's virtuous form if that estate
        # is not very corrupt, else its corrupt counterpart (the seed flowering)
        if corr[dom] > 0.5:
            phase[i] = CORRUPT[dom]
        else:
            phase[i] = VIRTUOUS[dom]

        # aitia (root cause of decay) = the estate with the largest
        # power*corruption product: the strong AND rotten one.
        cause[i] = int(np.argmax(powers * corr))

    # normalise longevity to ~[0,1] for the regression head
    long = (long - long.min()) / (long.max() - long.min() + 1e-9)
    return {"x": X, "stab": stab, "long": long, "phase": phase, "cause": cause}


def iterate_minibatches(data, bs, shuffle=True):
    n = data["x"].shape[0]
    idx = np.arange(n)
    if shuffle:
        RNG.shuffle(idx)
    for k in range(0, n, bs):
        j = idx[k:k + bs]
        yield {key: val[j] for key, val in data.items()}


# ============================================================================
# SECTION 4 — GRADIENT CHECK (mandatory)
# ----------------------------------------------------------------------------
# Compare analytic gradients to central finite differences on the scalar loss.
# This is the non-negotiable proof that backward() is correct.
# ============================================================================

def gradient_check():
    print("=" * 74)
    print("GRADIENT CHECK  (analytic vs. central finite differences)")
    print("=" * 74)
    net = AnacyclosisNet(in_dim=8, hid=8, T=5, dt=0.3)   # small net for speed
    data = make_dataset(12)
    batch = {k: v[:6] for k, v in data.items()}

    def loss_only():
        out, _ = net.forward(batch["x"])
        total, _, _ = net.loss(out, batch)
        return total

    # analytic grads
    out, _ = net.forward(batch["x"])
    _, grads_out, _ = net.loss(out, batch)
    analytic = net.backward(grads_out)

    eps = 1e-6
    worst = 0.0
    for name, P in net.params().items():
        flat = P.ravel()
        gflat = analytic[name].ravel()
        # sample a handful of coordinates per parameter tensor
        ncheck = min(6, flat.size)
        coords = RNG.choice(flat.size, size=ncheck, replace=False)
        local_worst = 0.0
        for ci in coords:
            orig = flat[ci]
            flat[ci] = orig + eps
            lp = loss_only()
            flat[ci] = orig - eps
            lm = loss_only()
            flat[ci] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[ci]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            local_worst = max(local_worst, rel)
        worst = max(worst, local_worst)
        print(f"  {name:9s} shape={str(P.shape):12s} max_rel_err={local_worst:.2e}")
    print("-" * 74)
    print(f"  WORST relative error across all params: {worst:.2e}")
    ok = worst < 1e-4
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'}  (threshold 1e-4)")
    print()
    assert ok, "Gradient check failed — backward() is wrong."
    return ok


# ============================================================================
# SECTION 5 — TRAINING (Adam, hand-rolled)
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, P in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            P -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def evaluate(net, data):
    out, _ = net.forward(data["x"])
    stab_pred = (sigmoid(out["stab_logit"]) > 0.5).astype(float)
    stab_acc = (stab_pred == data["stab"]).mean()
    phase_acc = (out["phase_logits"].argmax(1) == data["phase"]).mean()
    cause_acc = (out["cause_logits"].argmax(1) == data["cause"]).mean()
    long_rmse = np.sqrt(np.mean((out["longevity"] - data["long"]) ** 2))
    return stab_acc, phase_acc, cause_acc, long_rmse


def train():
    print("=" * 74)
    print("TRAINING  —  learning Polybius' invariant from synthetic constitutions")
    print("=" * 74)
    net = AnacyclosisNet(in_dim=8, hid=24, T=12, dt=0.35)
    train_data = make_dataset(2000)
    val_data = make_dataset(500)
    opt = Adam(net.params(), lr=3e-3)

    epochs = 40
    bs = 64
    for ep in range(1, epochs + 1):
        ep_loss = 0.0; nb = 0
        for batch in iterate_minibatches(train_data, bs):
            out, _ = net.forward(batch["x"])
            total, grads_out, _ = net.loss(out, batch)
            grads = net.backward(grads_out)
            opt.step(grads)
            ep_loss += total; nb += 1
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            sa, pa, ca, lr_ = evaluate(net, val_data)
            print(f"  epoch {ep:3d} | loss {ep_loss/nb:6.4f} | "
                  f"val stab_acc {sa:5.3f} | phase_acc {pa:5.3f} | "
                  f"cause_acc {ca:5.3f} | long_rmse {lr_:5.3f}")
    print()
    return net, val_data


# ============================================================================
# SECTION 6 — SELF-TESTS
# ----------------------------------------------------------------------------
# Confirm the trained model reproduces Polybius' central empirical claim:
# a mixed/balanced constitution outlives a pure/skewed one, AND the dynamical
# core actually behaves like anacyclosis (skewed C => high-amplitude spiral).
# ============================================================================

def self_tests(net):
    print("=" * 74)
    print("SELF-TESTS  —  does the learned mind think like Polybius?")
    print("=" * 74)

    # A perfectly mixed, well-checked constitution
    mixed = np.array([[0.34, 0.33, 0.33, 0.2, 0.2, 0.2, 0.95, 0.1]])
    # A pure, unchecked, rotten monarchy
    pure = np.array([[0.86, 0.07, 0.07, 0.9, 0.2, 0.2, 0.05, 0.6]])

    out_m, _ = net.forward(mixed)
    out_p, _ = net.forward(pure)

    long_m = float(out_m["longevity"][0])
    long_p = float(out_p["longevity"][0])
    stab_m = float(sigmoid(out_m["stab_logit"])[0])
    stab_p = float(sigmoid(out_p["stab_logit"])[0])
    phase_p = int(out_p["phase_logits"].argmax(1)[0])
    cause_p = int(out_p["cause_logits"].argmax(1)[0])
    phase_names = ["monarchy", "tyranny", "aristocracy",
                   "oligarchy", "democracy", "ochlocracy"]
    estate_names = ["monarchic", "aristocratic", "democratic"]

    print(f"  mixed constitution : longevity={long_m:.3f}  P(endures)={stab_m:.3f}")
    print(f"  pure  monarchy     : longevity={long_p:.3f}  P(endures)={stab_p:.3f}")
    print(f"  -> pure order slides toward: {phase_names[phase_p].upper()}")
    print(f"  -> diagnosed root cause (aitia): the {estate_names[cause_p]} estate")
    test1 = long_m > long_p
    test2 = stab_m > stab_p
    print(f"\n  TEST 1  mixed outlives pure ............... {'PASS' if test1 else 'FAIL'}")
    print(f"  TEST 2  mixed more stable than pure ...... {'PASS' if test2 else 'FAIL'}")

    # Test 3: the bare dynamical core behaves like anacyclosis.
    # Feed a strongly skewed coupling vs. a balanced (anti-symmetric, damped) one
    # straight into the rollout and compare trajectory amplitude.
    def rollout_amplitude(C, T=30, dt=0.35):
        s = np.array([[0.6, 0.2, 0.2]])
        amp = 0.0
        for _ in range(T):
            a = np.einsum("bij,bj->bi", C[None], s)
            s = s + dt * np.tanh(a)
            amp += float(np.mean(s * s))
        return amp / T

    skewed_C = np.array([[1.4, 0.0, 0.0],     # monarchic self-reinforcing
                         [0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.2]])
    mixed_C = np.array([[-0.4, 0.5, -0.5],    # each estate checks the others
                        [-0.5, -0.4, 0.5],
                        [0.5, -0.5, -0.4]])
    amp_skew = rollout_amplitude(skewed_C)
    amp_mix = rollout_amplitude(mixed_C)
    print(f"\n  rollout amplitude  skewed C = {amp_skew:.3f}   mixed C = {amp_mix:.3f}")
    test3 = amp_skew > amp_mix
    print(f"  TEST 3  skewed constitution spirals more than mixed ... "
          f"{'PASS' if test3 else 'FAIL'}")
    print()
    assert test1 and test2 and test3, "Self-tests failed."
    print("  ALL SELF-TESTS PASSED.\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    gradient_check()
    trained_net, val = train()
    self_tests(trained_net)
    print("=" * 74)
    print("DONE — Anacyclosis Dynamical Network verified, trained, and self-tested.")
    print("=" * 74)

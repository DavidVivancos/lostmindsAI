#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0042_sun_tzu_-544.py  —  THE SHÌ ENGINE
 Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 3 Minds 41 - 60 Available on Amazon https://www.amazon.com/dp/B0H6TVX69S
Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0042 · Sun Tzu
================================================================================

WHY THIS ARCHITECTURE (and why it is NOT a Transformer)
--------------------------------------------------------------------------------
The reflex, for a "military strategist," is to reach for adversarial game theory,
red-teaming, attention-over-stored-keys. That is the generic reading and Sun Tzu
deserves better. His one cognitive idea that belongs to no one else is this:

    Victory is DECIDED BY DISPOSITION (形, xíng) BEFORE the engagement.
    The skilled commander accumulates positional POTENTIAL (勢, shì) so that the
    actual fight is the cheap, well-timed RELEASE (節, jié) of stored energy
    against the enemy's VOIDS (虛, the empty), never against his FULLNESS (實).
        "The victorious army wins first and then seeks battle;
         the defeated army fights first and then seeks victory."  (Chapter 4)
        "Energy is the bending of a crossbow; decision, the release of the
         trigger."  (Chapter 5)

So intelligence here is NOT "compute harder." It is "position better, then let
the gradient do the work" — a boulder poised on a slope (Sun Tzu's own image)
rolls down on its own; the general spent his effort getting it to the top.

The SHÌ ENGINE encodes exactly that:

  1. DISPOSITION ENCODER (形)   — read partial intelligence into a latent "form" z0.
  2. POTENTIAL LANDSCAPE (勢)   — a LEARNED energy E(z) = ½ zᵀAz − cᵀz whose
                                  shape (A) and downhill direction (c) are set by
                                  the situation. A = MMᵀ+εI is a stable basin.
  3. SETTLING DYNAMICS          — the forward pass is GRADIENT DESCENT on E:
                                  z_t = z_{t-1} − η(Az_{t-1} − c).  The boulder
                                  rolls. No stacked feed-forward layers; the answer
                                  EMERGES from the configuration relaxing.
  4. VOID-STRIKE READOUT (虛實)  — each settled state proposes an allocation of
                                  force across fronts (softmax); good play masses
                                  on the enemy's empty fronts.
  5. TIMED RELEASE / HALTING (節)— a learned distribution over settling steps
                                  (economy of force): a decisive disposition wins
                                  in FEW steps; we penalise expected steps.
  6. WIN-FIRST HEAD (先勝)       — a scalar read straight off z0 predicts the
                                  outcome BEFORE settling runs. If you can call the
                                  result from the disposition alone, you have
                                  "already won, then sought battle."
  7. INVULNERABILITY TERM (不可勝)— the loss rewards outputs that DO NOT MOVE under
                                  the enemy's feint (a deceptive perturbation of
                                  the intelligence). "First make yourself
                                  unconquerable, then await the enemy's opening."

THE TASK (虛實 made concrete): a Colonel-Blotto-flavoured concentration game.
  The opponent holds a hidden defence distribution d over K fronts (his
  "fullness"). We see noisy intelligence o ≈ d. We must output an attack
  allocation that MASSES ON THE VOIDS — target a* = softmax(−gain·d). Knowing d,
  victory is settled before contact; the whole job is to infer the voids from
  imperfect intel and concentrate. That is Sun Tzu, operationalised.

ENGINEERING CONVENTIONS (shared across the 1000Minds corpus)
  • Pure NumPy, from scratch. A tiny reverse-mode autodiff engine (class `T`)
    supports the settling recurrence cleanly.
  • A finite-difference GRADIENT CHECK on the FULL loss is MANDATORY and runs
    every execution (see `gradient_check`).
  • A real training loop, held-out evaluation, and self-tests. Executed before
    shipping; verified output is pasted into the chapter.

Run:  python3 chapter_0042_sun_tzu_-544.py
================================================================================
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — A MINIMAL REVERSE-MODE AUTODIFF ENGINE
#
# Why build our own?  The settling dynamics (Section 4) apply the SAME learned
# matrix A over many steps — a weight-tied recurrence with a non-standard
# "energy gradient" update.  A small autodiff graph lets us express that exactly
# and differentiate through it, instead of hand-deriving a long chain rule that
# would be easy to get wrong.  Correctness is then *proven* by `gradient_check`.
# ──────────────────────────────────────────────────────────────────────────────

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` so broadcasting is differentiable."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """A node in the autodiff graph: data + accumulated gradient + a local rule."""

    def __init__(self, data, _prev=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.requires_grad = requires_grad

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _w(x):
        return x if isinstance(x, T) else T(x, requires_grad=False)

    # -- elementwise + broadcasting --------------------------------------------
    def __add__(self, o):
        o = self._w(o); out = T(self.data + o.data, (self, o))
        def bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            o.grad += _unbroadcast(out.grad, o.data.shape)
        out._backward = bw; return out
    __radd__ = __add__

    def __mul__(self, o):
        o = self._w(o); out = T(self.data * o.data, (self, o))
        def bw():
            self.grad += _unbroadcast(out.grad * o.data, self.data.shape)
            o.grad += _unbroadcast(out.grad * self.data, o.data.shape)
        out._backward = bw; return out
    __rmul__ = __mul__

    def __neg__(self):       return self * -1.0
    def __sub__(self, o):    return self + (-self._w(o))
    def __rsub__(self, o):   return self._w(o) + (-self)

    # -- linear algebra --------------------------------------------------------
    def matmul(self, o):
        o = self._w(o); out = T(self.data @ o.data, (self, o))
        def bw():
            self.grad += out.grad @ o.data.T
            o.grad += self.data.T @ out.grad
        out._backward = bw; return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = bw; return out

    def mean(self):
        out = T(self.data.mean(), (self,))
        def bw(): self.grad += (out.grad / self.data.size) * np.ones_like(self.data)
        out._backward = bw; return out

    # -- nonlinearities --------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data); out = T(t, (self,))
        def bw(): self.grad += (1 - t * t) * out.grad
        out._backward = bw; return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data)); out = T(s, (self,))
        def bw(): self.grad += s * (1 - s) * out.grad
        out._backward = bw; return out

    def exp(self):
        e = np.exp(self.data); out = T(e, (self,))
        def bw(): self.grad += e * out.grad
        out._backward = bw; return out

    def log(self):
        out = T(np.log(self.data), (self,))
        def bw(): self.grad += (1.0 / self.data) * out.grad
        out._backward = bw; return out

    def recip(self):
        r = 1.0 / self.data; out = T(r, (self,))
        def bw(): self.grad += (-r * r) * out.grad
        out._backward = bw; return out

    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x); s = e / e.sum(axis=axis, keepdims=True)
        out = T(s, (self,))
        def bw():
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)
        out._backward = bw; return out

    # -- reverse pass ----------------------------------------------------------
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — THE STRATEGIC TASK  (虛實 · emptiness and fullness)
#
# Generate battles. The enemy hides a defence distribution d (his "fullness").
# We receive noisy intelligence o. The optimal void-strike target masses on the
# emptiest fronts: a* = softmax(−gain·d). Victory is latent in d; the job is to
# read the voids through the fog and concentrate.
# ──────────────────────────────────────────────────────────────────────────────

# The battlefield has HIDDEN STRUCTURE. The enemy's K front-strengths are not
# independent: they are generated from a low-dimensional latent plan (rank r).
# This matters enormously. A naive commander reads each front's intelligence in
# isolation and is fooled by noise. A commander who has grasped the *shape* of the
# enemy's dispositions (the correlations between fronts) can DENOISE a glance —
# inferring a front's true strength from the others. That learned prior is the
# energy landscape A inside the engine; it is what lets settling out-read the raw
# glance. Structure is the thing intelligence exploits.
_LATENT_RANK = 2
_BMAP_SEED = 777

def make_battles(n, K, gain=5.0, noise=0.9, r=_LATENT_RANK, seed=0):
    """Return (obs, target, defence).

      d      : the enemy's fullness over K fronts (a distribution) — the truth.
      a_star : the optimal void-strike, softmax(−gain·d), massing on the emptiest.
      o      : our intelligence — the noisy front-strengths, fogged by war.

    Front strengths share a rank-r latent cause, so the fronts are CORRELATED;
    this is the exploitable structure a good prior denoises through."""
    rng = np.random.default_rng(seed)
    B = np.random.default_rng(_BMAP_SEED).standard_normal((K, r)) * 1.3  # fixed terrain
    u = rng.standard_normal((n, r))                          # the enemy's latent plan
    s = u @ B.T                                              # (n,K) correlated strengths
    d = np.exp(s - s.max(1, keepdims=True))
    d = d / d.sum(1, keepdims=True)                          # fullness (a distribution)
    # Optimal allocation: mass on the VOIDS (smallest defence) — strike the empty.
    tl = -gain * d
    a_star = np.exp(tl - tl.max(1, keepdims=True))
    a_star = a_star / a_star.sum(1, keepdims=True)
    # Intelligence is imperfect: the strengths, fogged by noise (知彼 is never perfect).
    o = s + rng.standard_normal((n, K)) * noise
    return o.astype(np.float64), a_star.astype(np.float64), d.astype(np.float64)


def feint(o, seed=0):
    """The enemy's DECEPTION (詭道): a structured perturbation that inflates the
    apparent fullness of one random front, to bait us away from the true void.
    A robust commander's allocation should barely move."""
    rng = np.random.default_rng(seed + 9999)
    n, K = o.shape
    delta = np.zeros_like(o)
    cols = rng.integers(0, K, size=n)
    delta[np.arange(n), cols] = 0.8                          # "seem strong where weak"
    return o + delta


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

class Params:
    """All learnable tensors of the Shì Engine, with strategy-named roles."""

    def __init__(self, K, H, seed=42):
        rng = np.random.default_rng(seed)
        s = lambda *sh: T(rng.standard_normal(sh) * (1.0 / np.sqrt(sh[0])))

        # 形 — Disposition encoder: intelligence o -> latent form z0
        self.W_enc = s(K, H); self.b_enc = T(np.zeros((1, H)))
        # 勢 — Context vector c sets the downhill direction of the potential field
        self.W_ctx = s(K, H); self.b_ctx = T(np.zeros((1, H)))
        # 勢 — Landscape factor M: A = MMᵀ + εI is the (stable) shape of the basin
        self.M = T(rng.standard_normal((H, H)) * (0.35 / np.sqrt(H)))
        # 虛實 — Void-strike readout: settled state -> allocation over K fronts
        self.W_out = s(H, K); self.b_out = T(np.zeros((1, K)))
        # 節 — Timed release: per-step halting logit (when to fire)
        self.w_halt = s(H, 1); self.b_halt = T(np.zeros((1, 1)))
        # 先勝 — Win-first head: read the outcome off the disposition z0 alone
        self.w_margin = s(H, 1); self.b_margin = T(np.zeros((1, 1)))

    def tensors(self):
        return [self.W_enc, self.b_enc, self.W_ctx, self.b_ctx, self.M,
                self.W_out, self.b_out, self.w_halt, self.b_halt,
                self.w_margin, self.b_margin]

    def zero_grad(self):
        for t in self.tensors():
            t.grad = np.zeros_like(t.data)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — THE SHÌ ENGINE FORWARD PASS
#
# This is the heart of the model and of Sun Tzu's idea. There is no stack of
# layers. We build a potential landscape from the situation, drop the disposition
# into it, and let it ROLL — settling under the energy gradient. Force is
# economised by a learned release time (節). The answer is what the configuration
# becomes, not what a deep network computes.
# ──────────────────────────────────────────────────────────────────────────────

EPS_A = 0.5          # ridge on the energy matrix: guarantees a single stable basin
A_SCALE = 0.5        # bounded weight of the learned curvature MMᵀ
ETA = 0.18           # FIXED settling step. With the normalisation below the
                     # spectrum of A is bounded (λmax ≤ EPS_A + A_SCALE·H), so a
                     # fixed η satisfies the stability condition η·λmax < 2 in the
                     # worst case — and, crucially, η does not depend on the
                     # parameters, so analytic and numerical gradients agree.
T_MAX = 6            # maximum settling steps available (the general may release early)


def forward(P, o_np, H):
    """Run the Shì Engine on a batch of intelligence o_np (B,K).
    Returns: y_hat (B,K) allocation, exp_steps (B,1), margin (B,1)."""
    o = T(o_np, requires_grad=False)

    # 形 — read intelligence into a latent disposition (the "form" of our force)
    z = (o.matmul(P.W_enc) + P.b_enc).tanh()                 # z0  (B,H)
    z0 = z
    # 勢 — the downhill direction of the potential field, set by the situation
    c = o.matmul(P.W_ctx) + P.b_ctx                          # (B,H)

    # 勢 — assemble the energy matrix A = εI + A_SCALE·(MMᵀ / (‖M‖²_F/H + 1)).
    # The trace of MMᵀ equals ‖M‖²_F = Σ M², which IS in the autodiff graph, so
    # the spectral normalisation is fully differentiable. This bounds λmax(A)
    # independent of M's scale, letting us use a fixed η (above).
    G = P.M.matmul(_transpose(P.M))                          # PSD curvature
    fro2 = (P.M * P.M).sum()                                 # trace(MMᵀ) = ‖M‖²_F
    denom_recip = (fro2 * (1.0 / H) + 1.0).recip()           # scalar
    A = (G * denom_recip) * A_SCALE + T(np.eye(H) * EPS_A, requires_grad=False)
    eta = ETA

    # SETTLING: the boulder rolls down the potential. Each step we also propose an
    # allocation (虛實 readout) and a release vote (節 halting logit).
    halt_logits = []     # list of (B,1)
    allocs = []          # list of (B,K)
    for _ in range(T_MAX):
        gradE = z.matmul(A) - c                              # ∇_z E = Az − c
        z = z - eta * gradE                                  # z_{t} = z_{t-1} − η∇E
        allocs.append((z.matmul(P.W_out) + P.b_out).softmax(axis=-1))
        halt_logits.append(z.matmul(P.w_halt) + P.b_halt)    # (B,1)

    # 節 — TIMED RELEASE: a softmax over steps (economy of force). Implemented
    # without a concat op: stabilise with a detached per-row max, then normalise.
    mx = np.max(np.concatenate([h.data for h in halt_logits], axis=1),
                axis=1, keepdims=True)                       # (B,1) constant
    mxT = T(mx, requires_grad=False)
    exps = [(h - mxT).exp() for h in halt_logits]            # each (B,1)
    Z = exps[0]
    for e in exps[1:]:
        Z = Z + Z * 0.0 + e                                  # running sum (B,1)
    Zinv = Z.recip()
    probs = [e * Zinv for e in exps]                         # p_t, each (B,1)

    # Final allocation = Σ_t p_t · alloc_t   (release-weighted strike)
    y_hat = allocs[0] * probs[0]
    for t in range(1, T_MAX):
        y_hat = y_hat + allocs[t] * probs[t]

    # Economy of force: expected number of steps to release (1-indexed)
    exp_steps = probs[0] * 1.0
    for t in range(1, T_MAX):
        exp_steps = exp_steps + probs[t] * float(t + 1)

    # 先勝 — read the predicted outcome straight off the disposition z0
    margin = (z0.matmul(P.w_margin) + P.b_margin).sigmoid()  # (B,1) in (0,1)

    return y_hat, exp_steps, margin


def _transpose(t):
    """Differentiable transpose of a 2-D tensor."""
    out = T(t.data.T, (t,))
    def bw(): t.grad += out.grad.T
    out._backward = bw
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — THE LOSS  (the four constants of the engine)
#
#   L = void-strike  +  β·economy-of-force  +  γ·invulnerability  +  δ·win-first
#
#   void-strike    : cross-entropy to the optimal mass-on-the-void allocation.
#   economy        : penalise expected release steps (decisive disposition).
#   invulnerability: output under the enemy's feint must match the clean output
#                    (target detached) — "first make yourself unconquerable".
#   win-first      : the z0 margin head must predict realised payoff (detached).
# ──────────────────────────────────────────────────────────────────────────────

B_PONDER, G_ROBUST, D_MARGIN = 0.003, 0.4, 1.0


def cross_entropy(target_np, pred):
    """−Σ target·log(pred), averaged over the batch. target is a constant."""
    tgt = T(target_np, requires_grad=False)
    ce_rows = (tgt * (pred + 1e-9).log()).sum(axis=-1) * -1.0   # (B,)
    return ce_rows.mean()


def loss_fn(P, o_np, a_star, d, H, seed=0):
    """Full objective. Returns (loss_tensor, diagnostics dict).

    Four constants, four pieces of Sun Tzu's doctrine:

      void-strike    : mass on the emptiness — CE to a* = softmax(−gain·d).
      economy        : release early — penalise expected settling steps (節).
      invulnerability: the strike under the enemy's feint must EQUAL the strike
                       under clean intelligence. We make BOTH allocations live and
                       pull them together (a symmetric consistency penalty), so the
                       only way to win the term is to be genuinely deception-proof,
                       not to detach one branch. "First make yourself unconquerable."
      win-first      : the disposition z0 alone must predict the battle's intrinsic
                       exploitability — the spread of the enemy's fullness,
                       max(d) − min(d), a property of the terrain, not of our move.
                       The general reads the verdict off the ground first (先勝).
    """
    y_hat, exp_steps, margin = forward(P, o_np, H)

    L_void = cross_entropy(a_star, y_hat)
    L_econ = exp_steps.mean()

    # 不可勝 — invulnerability: feed the enemy's feint and require the same strike.
    # Both branches carry gradient; the model must converge them by being robust.
    o_feint = feint(o_np, seed=seed)
    y_feint, _, _ = forward(P, o_feint, H)
    rdiff = y_feint - y_hat                                    # (B,K)
    L_robust = (rdiff * rdiff).mean()

    # 先勝 — win-first: predict the battle's intrinsic EXPLOITABILITY before any
    # fighting — the spread of the enemy's fullness, max(d) − min(d). A lopsided
    # enemy (big spread) has gaping voids and is highly winnable; an even enemy is
    # not. This is a property of the terrain alone, read off the disposition z0.
    spread = (d.max(axis=1, keepdims=True) - d.min(axis=1, keepdims=True))  # (B,1) const
    mdiff = margin - T(spread, requires_grad=False)
    L_margin = (mdiff * mdiff).mean()

    L = L_void + B_PONDER * L_econ + G_ROBUST * L_robust + D_MARGIN * L_margin
    diag = dict(void=float(L_void.data), econ=float(L_econ.data),
                robust=float(L_robust.data), margin=float(L_margin.data),
                steps=float(exp_steps.data.mean()))
    return L, diag


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OPTIMISER (Adam, hand-rolled)
# ──────────────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, tensors, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.ts = tensors; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(t.data) for t in tensors]
        self.v = [np.zeros_like(t.data) for t in tensors]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.ts):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GRADIENT CHECK  (mandatory, every run)
#
# Finite-difference vs. autodiff on the FULL loss (all four terms), over a random
# sample of scalar parameters. This is the file's certificate of correctness.
# ──────────────────────────────────────────────────────────────────────────────

def gradient_check(seed=1, K=5, H=8, n=12, n_probe=45, eps=1e-6):
    rng = np.random.default_rng(seed)
    o, a_star, d = make_battles(n, K, seed=seed)
    P = Params(K, H, seed=seed)

    L, _ = loss_fn(P, o, a_star, d, H, seed=seed)
    P.zero_grad(); L.backward()

    tensors = P.tensors()
    max_err = 0.0
    for _ in range(n_probe):
        ti = rng.integers(0, len(tensors))
        flat = tensors[ti].data.ravel()
        j = rng.integers(0, flat.size)
        orig = flat[j]
        flat[j] = orig + eps; lp, _ = loss_fn(P, o, a_star, d, H, seed=seed)
        flat[j] = orig - eps; lm, _ = loss_fn(P, o, a_star, d, H, seed=seed)
        flat[j] = orig
        num = (lp.data - lm.data) / (2 * eps)
        ana = tensors[ti].grad.ravel()[j]
        denom = max(1e-9, abs(num) + abs(ana))
        max_err = max(max_err, abs(num - ana) / denom)
    return max_err


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(P, o, a_star, d, H):
    y_hat, exp_steps, margin = forward(P, o, H)
    yh = y_hat.data
    # Void-strike accuracy: did we mass on the emptiest (least-defended) front?
    struck = yh.argmax(1)
    void = d.argmin(1)
    acc = float((struck == void).mean())
    payoff = (1.0 - (yh * d).sum(1)).mean()
    # Win-first quality: does the z0 margin predict the battle's intrinsic
    # exploitability (the spread of enemy fullness), read off the disposition?
    spread = d.max(1, keepdims=True) - d.min(1, keepdims=True)
    m = margin.data
    corr = float(np.corrcoef(m.ravel(), spread.ravel())[0, 1])
    # Invulnerability: how far does the strike move under the enemy's feint?
    y_feint, _, _ = forward(P, feint(o), H)
    drift = float(np.abs(y_feint.data - yh).sum(1).mean())
    return dict(acc=acc, payoff=float(payoff), steps=float(exp_steps.data.mean()),
                margin_corr=corr, feint_drift=drift)


def train(K=8, H=24, n_train=1024, n_test=256, epochs=320, batch=64, seed=42, verbose=True):
    o_tr, a_tr, d_tr = make_battles(n_train, K, seed=seed)
    o_te, a_te, d_te = make_battles(n_test, K, seed=seed + 1)
    P = Params(K, H, seed=seed)
    opt = Adam(P.tensors(), lr=3e-3)
    rng = np.random.default_rng(seed)

    init_eval = evaluate(P, o_te, a_te, d_te, H)
    _, init_diag = loss_fn(P, o_tr[:batch], a_tr[:batch], d_tr[:batch], H)
    init_loss = init_diag['void']
    history = []

    for ep in range(epochs):
        idx = rng.permutation(n_train)
        ep_void = 0.0; nb = 0
        for s in range(0, n_train, batch):
            b = idx[s:s + batch]
            P.zero_grad()
            L, diag = loss_fn(P, o_tr[b], a_tr[b], d_tr[b], H, seed=ep * 1000 + s)
            L.backward()
            opt.step()
            ep_void += diag['void']; nb += 1
        ep_void /= nb
        history.append(ep_void)
        if verbose and (ep % 40 == 0 or ep == epochs - 1):
            ev = evaluate(P, o_te, a_te, d_te, H)
            print(f"  epoch {ep:4d} | void-CE {ep_void:.4f} | "
                  f"strike-acc {ev['acc']:.3f} | E[steps] {ev['steps']:.2f} | "
                  f"win-first r {ev['margin_corr']:+.3f} | feint-drift {ev['feint_drift']:.3f}")

    final_eval = evaluate(P, o_te, a_te, d_te, H)
    return P, dict(init_loss=init_loss, final_loss=history[-1],
                   init_eval=init_eval, final_eval=final_eval, history=history,
                   H=H, K=K)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DEMONSTRATION: one battle, watched
# ──────────────────────────────────────────────────────────────────────────────

def demo_one_battle(P, H, K, seed=7):
    o, a_star, d = make_battles(1, K, seed=seed)
    y_hat, exp_steps, margin = forward(P, o, H)
    yh = y_hat.data[0]; dd = d[0]
    void = int(dd.argmin()); struck = int(yh.argmax())
    print("\n  ── One battle, watched ──────────────────────────────────────")
    print("  enemy fullness d   :", np.array2string(dd, precision=2, suppress_small=True))
    print("  our strike  a_hat  :", np.array2string(yh, precision=2, suppress_small=True))
    print(f"  the true void is front {void};  we massed on front {struck}"
          f"  {'✓ struck the empty' if struck == void else '✗ struck the full'}")
    print(f"  exploitability read from disposition alone (先勝): {float(margin.data[0,0]):.2f}"
          f"   [true spread max(d)-min(d) = {float(dd.max()-dd.min()):.2f}]")
    print(f"  released after E[steps] = {float(exp_steps.data[0,0]):.2f} settling steps")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN: gradient check, train, self-tests
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    print("=" * 78)
    print(" THE SHÌ ENGINE  ·  Sun Tzu  ·  energy-settling strategic cognition")
    print("=" * 78)

    print("\n[1/3] Gradient check (finite difference vs. autodiff, full loss)")
    err = gradient_check()
    print(f"      max relative error over 45 probes = {err:.2e}")
    assert err < 1e-4, "GRADIENT CHECK FAILED"
    print("      PASS — analytic gradients match numerical to ~1e-6.")

    print("\n[2/3] Training the Shì Engine")
    P, R = train(verbose=True)

    print("\n[3/3] Self-tests")
    fe, ie = R['final_eval'], R['init_eval']
    results = []
    def check(name, ok, detail):
        results.append(ok)
        print(f"      [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("loss decreased",
          R['final_loss'] < 0.85 * R['init_loss'],
          f"void-CE {R['init_loss']:.3f} -> {R['final_loss']:.3f}")
    check("strikes the void (虛實)",
          fe['acc'] > 0.40,
          f"mass-on-emptiest accuracy {fe['acc']:.3f} "
          f"(chance ~{1.0/R['K']:.2f}, ≈{fe['acc']*R['K']:.1f}× chance)")
    check("economy of force (節)",
          fe['steps'] < ie['steps'] - 0.25,
          f"E[release steps] {ie['steps']:.2f} -> {fe['steps']:.2f}")
    check("wins first, then fights (先勝)",
          fe['margin_corr'] > 0.5,
          f"disposition->exploitability corr r={fe['margin_corr']:+.3f}")
    check("unconquerable under deception (不可勝)",
          fe['feint_drift'] < ie['feint_drift'] - 1e-3,
          f"strike drift under feint {ie['feint_drift']:.3f} -> {fe['feint_drift']:.3f}")

    demo_one_battle(P, R['H'], R['K'])

    print("\n" + "=" * 78)
    print(f" SELF-TESTS: {sum(results)}/{len(results)} passed")
    print("=" * 78)
    assert all(results), "one or more self-tests failed"
    print(" All checks green. 不戰而屈人之兵 — the disposition did the work.")


if __name__ == "__main__":
    main()

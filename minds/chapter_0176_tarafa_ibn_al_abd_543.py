#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ṬIWAL  —  The Slackened-Tether Network
 Chapter 0176 - Ṭarafa ibn al-ʿAbd (c. 543 – c. 569 CE), Baḥrayn, Banū Bakr
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0176_tarafa_ibn_al_abd_543 - Ṭarafa ibn al-ʿAbd (c. 543 – c. 569 CE), Baḥrayn, Banū Bakr
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the cognitive signature of one pre-Islamic poet and
nobody else's. Ṭarafa's Muʿallaqa is the longest of the seven hanging odes and
the one the tradition calls the camel ode par excellence. Read as a theory of
mind it makes three claims, each of which is a component of this network:

 1. THE NAʿT — perception by artefact-analogy.
    Ṭarafa does not describe the she-camel by comparing her to other animals.
    He dismantles her into parts and explains each part by the MANUFACTURED
    thing whose function it shares: thighs like the two doors of a lofty
    castle; a frame like the bridge of a Byzantine builder who swore to case
    it in brick until it stood true; a neck like the rudder of a Tigris boat
    going upstream; a skull like an anvil filed to an edge; eyes like two
    mirrors set in the rock-caves of the brow; a heart like a stone-hammer on
    a slab. A living body is read as an ASSEMBLY OF TOOLS. The tradition even
    preserves the boy Ṭarafa catching a spec error in his uncle's verse: a
    male camel described with a she-camel's brand — "the camel has turned
    into a she-camel" (istanwaqa l-jamal), a proverb ever since.

    -> The first layer of this network is a PROTOTYPE ENCODER: every raw
       observation is explained by soft assignment to a small bank of learned
       "artefacts", each carrying a function vector. The recurrent core never
       sees the raw world; it sees the artefact reading.

 2. THE ṬIWAL — freedom is the slack in a rope whose ends are in a hand.
    The moral centre of the ode is a policy argument:  "you who blame me for
    attending the fray and the pleasures — will you make me immortal? If you
    cannot ward off my death, let me meet it with what my hand owns." The
    grave of the miser and the grave of the prodigal are the same two heaps
    of earth under deaf slabs; life is a treasure that shrinks every night;
    and — the line this file is named for — "By your life, Death, in so far
    as it misses a young man, is like a SLACKENED TETHER (al-ṭiwal al-murkhā)
    whose two ends are held in a hand." The youth is tethered. He does not
    know the length of the rope. The only rational policy is to spend at a
    rate set by his best estimate of the slack that remains.

    -> The core of this network is a recurrent cell whose ONLY output is an
       estimate of remaining slack, ℓ̂, and whose spend fraction is fixed by
       structure to a = 1 / (1 + ℓ̂). The policy IS the tether estimate. There
       is no separate action head. Wealth hoarded past the end of the rope is
       worth exactly nothing (grave equality), and the objective says so.

 3. THE SEALED LETTER — the instruction he would not read.
    The one act of Ṭarafa's life recorded outside his verse: the king of
    al-Ḥīra, ʿAmr ibn Hind, whom he had lampooned, sent him and his maternal
    uncle al-Mutalammis home to Baḥrayn each carrying a sealed letter to the
    governor at Hajar. On the road near al-Najaf al-Mutalammis, who could not
    read, had a boy of al-Ḥīra read his: it ordered his death. He threw it in
    the river and begged his nephew to open his own. Ṭarafa refused to break
    the king's seal, carried the letter to Hajar, and was killed there. His
    own ode had already said where news comes from: "The days will reveal to
    you what you did not know, and one you did not provision will bring you
    the tidings." He became the un-provisioned messenger of his own death.

    -> The network receives a SEALED CHANNEL that states the true remaining
       horizon exactly, multiplied by a gate g. Under the HONOUR regime g is
       pinned to 0: the channel is present, exact, and unread. Under the
       READER regime g = 1. Under the PRICED regime g is a learned parameter
       carrying an honour cost κ·g, and we measure the price at which the
       seal breaks. A separate UNPROVISIONED channel — a stranger who arrives
       on a random night carrying news no provisioned source could carry —
       is the mechanism of the closing couplet, and we measure what a mind
       loses when it is deaf to strangers.

THE ARCHITECTURAL CLAIM
-----------------------
Give an agent a season of T nights, a hidden death-night N, noisy provisioned
cues about the level of danger, a rare un-provisioned messenger who knows the
realised future, and a sealed channel that names the horizon. Let its only
learned quantity be an estimate of the remaining slack. Then:

  * with log utility the hindsight-optimal spend fraction is exactly
    a*_t = 1 / (N − t + 1), i.e. ℓ̂* = N − t. The structural policy
    a = 1/(1+ℓ̂) can represent the optimum and nothing else — it cannot
    express "hoard for its own sake", which is the miser's error;
  * the objective is trained IN HINDSIGHT: the realised N is known only after
    the season ends. This is Ṭarafa's epistemology made literal: you learn
    the length of the rope when it is pulled;
  * the gap between HONOUR and READER is the price of not reading your own
    orders; the gap between HONOUR and DEAF is the value of a stranger.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients (BPTT through the cell,
    softmax-RBF backward through the prototype encoder, gate reparametrised)
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop (Adam, mini-batches) on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 0176_tarafa_ibn_al_abd_543_Neuron.py
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  SMALL UTILITIES
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softplus(x):
    """log(1 + e^x), stable. Derivative is the logistic."""
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def softmax_rows(logits):
    """Row-wise softmax, stable."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


# ==============================================================================
# 1.  THE SEASON — synthetic data
# ==============================================================================
#
# One episode is a season of T nights in the life of a young man on a rope.
#
#   * A latent DANGER profile d_t in [0,1] governs the hazard of death at the
#     end of each night:  λ_t = HAZ0 + HAZ1 · d_t.
#   * The death-night N is the first night whose end-of-night hazard fires,
#     or T if none does. The agent consumes on nights 1..N.
#   * PROVISIONED cues (K channels): noisy affine readings of d_t. These are
#     the sources the agent pays for — the council, the herd, the sky. They
#     know the level of danger. They do NOT know the realised future.
#   * The UNPROVISIONED messenger: on any night, with probability P_MSG, a
#     stranger arrives (flag = 1) and says whether the end is within two
#     nights (value = 1) or not (value = 0), with a small flip rate. He knows
#     the realised N — news that no provisioned source can carry.
#   * The SEALED channel: (N − t)/T exactly. The letter names the horizon.
#     Present every night; readable only through the gate.
#
# Three danger profiles, drawn at random: a calm season, a rising threat, and
# a sudden anger (a king's face changing on a random night).

T_NIGHTS = 12
K_PROV   = 4        # provisioned channels ("parts" of the observed world)
HAZ0, HAZ1 = 0.02, 0.28
P_MSG    = 0.25     # nightly probability that an un-provisioned stranger arrives
MSG_FLIP = 0.10     # stranger's error rate
NEAR     = 2        # "the end is near" = within NEAR nights

# fixed affine readings for the provisioned channels (the world's own code)
PROV_W = np.array([ 1.0,  0.6, -0.8,  0.4])
PROV_B = np.array([ 0.0,  0.3,  0.9, -0.2])
PROV_S = np.array([ 0.25, 0.35, 0.30, 0.45])   # noise std per channel


def make_dataset(n, seed=0, T=T_NIGHTS):
    """
    Returns a dict of arrays:
      P   (n,T,K)  provisioned cues
      M   (n,T,2)  messenger [flag, value]
      S   (n,T)    sealed channel, (N-t)/T
      TAU (n,T)    season clock, t/T
      N   (n,)     realised death-night in 1..T (alive nights)
      D   (n,T)    latent danger (for analysis only, never an input)
      KIND(n,)     profile type 0 calm / 1 rising / 2 sudden
    """
    rng = np.random.default_rng(seed)
    t_idx = np.arange(1, T + 1, dtype=np.float64)

    kind = rng.integers(0, 3, size=n)
    D = np.zeros((n, T))
    for i in range(n):
        if kind[i] == 0:                                     # calm
            D[i] = 0.10 + 0.10 * rng.random()
        elif kind[i] == 1:                                   # rising
            lo, hi = 0.05 + 0.1 * rng.random(), 0.7 + 0.3 * rng.random()
            D[i] = lo + (hi - lo) * (t_idx - 1) / (T - 1)
        else:                                                # sudden
            jump = rng.integers(2, T - 1)
            D[i] = np.where(t_idx < jump, 0.08, 0.85 + 0.15 * rng.random())

    lam = HAZ0 + HAZ1 * D
    fire = rng.random((n, T)) < lam
    N = np.full(n, T, dtype=np.int64)
    for i in range(n):
        hit = np.flatnonzero(fire[i])
        if hit.size:
            N[i] = hit[0] + 1          # dies at the end of night N; consumed nights 1..N

    # provisioned cues: affine readings of danger plus channel noise
    P = (D[:, :, None] * PROV_W[None, None, :] + PROV_B[None, None, :]
         + rng.normal(0.0, 1.0, (n, T, K_PROV)) * PROV_S[None, None, :])

    # un-provisioned messenger: knows the realised future
    remain = N[:, None] - t_idx[None, :]                      # nights left AFTER this one
    arrive = rng.random((n, T)) < P_MSG
    near = (remain <= NEAR).astype(np.float64)
    flip = rng.random((n, T)) < MSG_FLIP
    val = np.where(flip, 1.0 - near, near)
    M = np.zeros((n, T, 2))
    M[:, :, 0] = arrive
    M[:, :, 1] = arrive * val

    # sealed: names the horizon exactly, every night
    S = np.clip(remain, 0, None) / T
    TAU = np.tile(t_idx / T, (n, 1))
    return dict(P=P, M=M, S=S, TAU=TAU, N=N, D=D, KIND=kind)


# ==============================================================================
# 2.  THE OBJECTIVE — grave equality, in hindsight
# ==============================================================================
#
# Wealth W_1 = 1. Each night the agent spends the fraction a_t of what it holds:
# c_t = a_t W_t, and the remainder grows: W_{t+1} = (1 − a_t) W_t · R.
# Utility of consumption is logarithmic. The agent is alive on nights 1..N.
# Whatever is left after night N is lost — the heap on the grave.
#
#   J = Σ_{t=1}^{N} log c_t
#     = Σ_{t=1}^{N} log a_t  +  Σ_{s=1}^{N-1} (N − s) log(1 − a_s)  + const(R)
#
# The constant does not depend on the policy, so it is dropped from the loss
# and kept only for the ledger. We normalise by N so that long and short lives
# are weighted alike, and MINIMISE  L = −J/N  averaged over episodes.
#
# The hindsight optimum, from ∂J/∂a_t = 0:   a*_t = 1/(N − t + 1).

R_GROWTH = 1.25


def objective_terms(N, T=T_NIGHTS):
    """
    Per-episode coefficient arrays for the objective.
      alive  (n,T): 1 if t <= N
      weight (n,T): (N − t)_+ · alive  — the number of later nights that
                    inherit tonight's restraint
    """
    t_idx = np.arange(1, T + 1)[None, :]
    alive = (t_idx <= N[:, None]).astype(np.float64)
    weight = np.clip(N[:, None] - t_idx, 0, None) * alive
    return alive, weight


def utility(a, N):
    """
    J per episode (no constant), a: (n,T) spend fractions.
    The restraint term log(1 − a) is only evaluated where it carries weight:
    on the last alive night the oracle spends everything (a = 1) and the term
    is multiplied by zero, so it must not be allowed to become 0·(−∞).
    """
    alive, weight = objective_terms(N, a.shape[1])
    safe = np.where(weight > 0, np.log1p(-np.minimum(a, 1.0 - 1e-15)), 0.0)
    return (alive * np.log(a)).sum(1) + (weight * safe).sum(1)


def oracle_policy(N, T=T_NIGHTS):
    """a*_t = 1/(N − t + 1) on alive nights (anything after N is irrelevant)."""
    t_idx = np.arange(1, T + 1)[None, :]
    rem = np.clip(N[:, None] - t_idx + 1, 1, None).astype(np.float64)
    return 1.0 / rem


def leftover_fraction(a, N):
    """Π_{t<=N} (1 − a_t): the share of the (ungrown) fortune left on the grave."""
    alive, _ = objective_terms(N, a.shape[1])
    return np.exp((alive * np.log1p(-a)).sum(1))


# ==============================================================================
# 3.  THE MODEL — Naʿt encoder → Ṭiwal cell → tether estimate
# ==============================================================================

class Tiwal:
    """
    Parameters
      mu    (P,K)   artefact prototypes in cue space
      F     (P,E)   function vector carried by each artefact
      tb    ()      raw precision; beta = softplus(tb)
      Wx    (D,H)   input → hidden          D = E + 2 (messenger) + 1 (sealed) + 1 (clock)
      Wh    (H,H)   hidden → hidden
      b     (H,)
      v     (H,)    hidden → tether pre-activation
      c     ()
      tg    ()      raw gate; only trained in mode 'priced':  g = sigmoid(tg)

    Gate modes
      'honour'  g ≡ 0  (sealed channel present, unread)
      'reader'  g ≡ 1  (seal broken; horizon read exactly)
      'priced'  g = sigmoid(tg), trained, with loss + kappa · g
    """

    def __init__(self, n_proto=6, n_fn=6, n_hidden=16, gate='honour',
                 kappa=0.0, deaf=False, seed=0):
        rng = np.random.default_rng(seed)
        self.P, self.E, self.H = n_proto, n_fn, n_hidden
        self.D = n_fn + 4
        self.gate_mode = gate
        self.kappa = float(kappa)
        self.deaf = bool(deaf)
        K = K_PROV
        self.p = {
            'mu': rng.normal(0.0, 0.6, (n_proto, K)) + PROV_B[None, :],
            'F':  rng.normal(0.0, 0.5, (n_proto, n_fn)),
            'tb': np.array(0.0),
            'Wx': rng.normal(0.0, 1.0 / np.sqrt(self.D), (self.D, n_hidden)),
            'Wh': rng.normal(0.0, 0.6 / np.sqrt(n_hidden), (n_hidden, n_hidden)),
            'b':  np.zeros(n_hidden),
            'v':  rng.normal(0.0, 1.0 / np.sqrt(n_hidden), n_hidden),
            'c':  np.array(np.log(np.expm1(4.0))),   # start believing ~4 nights of slack
            'tg': np.array(0.0),
        }

    # ------------------------------------------------------------------ gate
    def gate(self):
        if self.gate_mode == 'honour':
            return 0.0
        if self.gate_mode == 'reader':
            return 1.0
        return float(sigmoid(self.p['tg']))

    # --------------------------------------------------------------- forward
    def forward(self, batch):
        """
        Runs the season for a batch. Returns (a, cache).
          a: (B,T) spend fractions
        """
        p = self.p
        P, M, S, TAU = batch['P'], batch['M'], batch['S'], batch['TAU']
        B, T, _ = P.shape
        if self.deaf:
            M = np.zeros_like(M)
        g = self.gate()
        beta = float(softplus(p['tb']))

        h = np.zeros((B, self.H))
        cache = dict(g=g, beta=beta, xs=[], hs=[h], pre_hs=[], pre_ls=[],
                     ls=[], alphas=[], ds=[], zs=[], M=M, S=S)
        a = np.zeros((B, T))
        for t in range(T):
            z = P[:, t, :]                                        # (B,K)
            diff = z[:, None, :] - p['mu'][None, :, :]            # (B,P,K)
            d = (diff ** 2).sum(-1)                               # (B,P)
            alpha = softmax_rows(-beta * d)                       # (B,P)
            e = alpha @ p['F']                                    # (B,E)
            x = np.concatenate([e, M[:, t, :], (g * S[:, t])[:, None],
                                TAU[:, t][:, None]], axis=1)      # (B,D)
            pre_h = x @ p['Wx'] + h @ p['Wh'] + p['b']
            h = np.tanh(pre_h)
            pre_l = h @ p['v'] + p['c']
            l = softplus(pre_l)
            a[:, t] = 1.0 / (1.0 + l)
            cache['xs'].append(x); cache['hs'].append(h); cache['pre_hs'].append(pre_h)
            cache['pre_ls'].append(pre_l); cache['ls'].append(l)
            cache['alphas'].append(alpha); cache['ds'].append(d); cache['zs'].append(z)
        cache['a'] = a
        return a, cache

    # ------------------------------------------------------------------ loss
    def loss(self, a, N):
        L = -(utility(a, N) / N).mean()
        if self.gate_mode == 'priced':
            L = L + self.kappa * self.gate()
        return float(L)

    # -------------------------------------------------------------- backward
    def backward(self, cache, N):
        """
        Hand-derived gradients of the mean loss w.r.t. every parameter.

        Objective (per episode i, dropped constants):
          J_i = Σ_t alive_it log a_it + Σ_t weight_it log(1 − a_it)
          L   = −(1/B) Σ_i J_i / N_i   (+ κ g in 'priced')

        ∂L/∂a_it = −(1/(B N_i)) [ alive_it / a_it − weight_it / (1 − a_it) ]
        a = 1/(1+l)          → ∂a/∂l = −a²
        l = softplus(pre_l)  → ∂l/∂pre_l = σ(pre_l)
        pre_l = h·v + c
        h = tanh(pre_h);  pre_h = x Wx + h_prev Wh + b     (BPTT)
        x = [e, m, g·s, τ];  e = α F;  α = softmax(−β d);  d_p = ‖z − μ_p‖²
        softmax backward: ∂L/∂logit_p = α_p (∂L/∂α_p − Σ_q α_q ∂L/∂α_q)
        ∂d_p/∂μ_p = −2 (z − μ_p);  β = softplus(tb);  g = σ(tg)
        """
        p = self.p
        a = cache['a']
        B, T = a.shape
        alive, weight = objective_terms(N, T)
        dJ_da = alive / a - weight / (1.0 - a)                    # ∂J/∂a
        dL_da = -(dJ_da / N[:, None]) / B

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros((B, self.H))
        g, beta = cache['g'], cache['beta']
        dg = 0.0
        dbeta = 0.0
        E = self.E
        for t in reversed(range(T)):
            l = cache['ls'][t]
            at = a[:, t]
            dl = dL_da[:, t] * (-(at ** 2))
            dpre_l = dl * sigmoid(cache['pre_ls'][t])
            h = cache['hs'][t + 1]
            h_prev = cache['hs'][t]
            grads['v'] += h.T @ dpre_l
            grads['c'] += dpre_l.sum()
            dh = dpre_l[:, None] * p['v'][None, :] + dh_next
            dpre_h = dh * (1.0 - h ** 2)
            x = cache['xs'][t]
            grads['Wx'] += x.T @ dpre_h
            grads['Wh'] += h_prev.T @ dpre_h
            grads['b'] += dpre_h.sum(0)
            dx = dpre_h @ p['Wx'].T
            dh_next = dpre_h @ p['Wh'].T
            # sealed channel → gate
            dg += float((dx[:, E + 2] * cache['S'][:, t]).sum())
            # artefact encoder
            de = dx[:, :E]                                       # (B,E)
            alpha = cache['alphas'][t]
            d = cache['ds'][t]
            z = cache['zs'][t]
            grads['F'] += alpha.T @ de
            dalpha = de @ p['F'].T                               # (B,P)
            dlogit = alpha * (dalpha - (alpha * dalpha).sum(1, keepdims=True))
            dbeta += float((dlogit * (-d)).sum())
            dd = dlogit * (-beta)                                # (B,P)
            # dmu_p = Σ_i dd_ip · (−2)(z_i − mu_p)
            grads['mu'] += -2.0 * (dd.T @ z - dd.sum(0)[:, None] * p['mu'])
        grads['tb'] += dbeta * float(sigmoid(p['tb']))
        if self.gate_mode == 'priced':
            gg = float(sigmoid(p['tg']))
            grads['tg'] += (dg + self.kappa) * gg * (1.0 - gg)
        else:
            grads['tg'] += 0.0
        return grads

    # ------------------------------------------------------------ utilities
    def tether(self, batch):
        """Estimated remaining slack ℓ̂ (B,T) — the only thing the cell believes."""
        a, cache = self.forward(batch)
        return np.stack(cache['ls'], axis=1)


# ==============================================================================
# 4.  GRADIENT CHECK (mandatory)
# ==============================================================================

def gradient_check(gate='honour', kappa=0.0, deaf=False, seed=3, n=5,
                   eps=1e-5, tol=1e-5, floor=1e-3, verbose=False):
    """
    Central finite differences on every parameter (all coordinates of the
    small ones, a spread of coordinates of the matrices) against the analytic
    gradient. The error is relative, with an absolute floor: a coordinate whose
    true gradient is ~1e-6 sits at the round-off floor of a float64 central
    difference (~1e-10) and would otherwise report a meaningless 1e-4 "relative"
    error. Returns (ok, worst_err, where).
    """
    data = make_dataset(n, seed=seed + 100)
    model = Tiwal(gate=gate, kappa=kappa, deaf=deaf, seed=seed)
    # nudge the model off its symmetric start so that every path is live
    rng = np.random.default_rng(seed)
    for k in model.p:
        # keep every parameter an ndarray (0-d included) so in-place edits hold
        model.p[k] = np.array(model.p[k] + rng.normal(0.0, 0.05, model.p[k].shape),
                              dtype=np.float64)
    a, cache = model.forward(data)
    grads = model.backward(cache, data['N'])

    worst, where = 0.0, None
    for k, v in model.p.items():
        if k == 'tg' and gate != 'priced':
            continue
        size = v.size
        idx = np.arange(size) if size <= 40 else rng.choice(size, 40, replace=False)
        for j in idx:
            pos = np.unravel_index(j, v.shape)       # () for 0-d parameters
            old = float(v[pos])
            v[pos] = old + eps
            lp = model.loss(model.forward(data)[0], data['N'])
            v[pos] = old - eps
            lm = model.loss(model.forward(data)[0], data['N'])
            v[pos] = old
            num = (lp - lm) / (2 * eps)
            ana = float(np.asarray(grads[k])[pos])
            rel = abs(num - ana) / max(floor, abs(num) + abs(ana))
            if rel > worst:
                worst, where = rel, f"{k}[{j}] num={num:+.6e} ana={ana:+.6e}"
            if verbose:
                print(f"  {k:3s}[{j:3d}]  num={num:+.6e}  ana={ana:+.6e}  rel={rel:.2e}")
    return worst < tol, worst, where


# ==============================================================================
# 5.  TRAINING — Adam, mini-batches, validation
# ==============================================================================

def slice_batch(data, idx):
    return dict(P=data['P'][idx], M=data['M'][idx], S=data['S'][idx],
                TAU=data['TAU'][idx], N=data['N'][idx])


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = np.asarray(params[k] - self.lr * mh / (np.sqrt(vh) + self.eps),
                                   dtype=np.float64)


def train(model, tr, va, steps=900, batch=64, lr=3e-3, seed=0, log_every=300,
          label=''):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    n = tr['N'].size
    hist = []
    t0 = time.time()
    for s in range(1, steps + 1):
        idx = rng.choice(n, batch, replace=False)
        b = slice_batch(tr, idx)
        a, cache = model.forward(b)
        L = model.loss(a, b['N'])
        grads = model.backward(cache, b['N'])
        # gradient clipping, global norm
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        if gn > 5.0:
            for k in grads:
                grads[k] *= 5.0 / gn
        opt.step(model.p, grads)
        if s % log_every == 0 or s == 1:
            av, _ = model.forward(va)
            Lv = model.loss(av, va['N'])
            hist.append((s, L, Lv))
            print(f"    {label:8s} step {s:5d}  train {L:8.4f}  val {Lv:8.4f}"
                  f"  gate {model.gate():.3f}  ({time.time()-t0:5.1f}s)")
    return hist


# ==============================================================================
# 6.  EVALUATION
# ==============================================================================

def evaluate(model, data):
    a, cache = model.forward(data)
    N = data['N']
    J = utility(a, N) / N
    Jor = utility(oracle_policy(N), N) / N
    l_hat = np.stack(cache['ls'], axis=1)
    t_idx = np.arange(1, a.shape[1] + 1)[None, :]
    alive, _ = objective_terms(N, a.shape[1])
    true_slack = np.clip(N[:, None] - t_idx, 0, None)
    mae = (np.abs(l_hat - true_slack) * alive).sum() / alive.sum()
    return dict(J=float(J.mean()), J_oracle=float(Jor.mean()),
                leftover=float(leftover_fraction(a, N).mean()),
                slack_mae=float(mae), gate=model.gate())


def naive_constant_baseline(data):
    """Best fixed spend fraction with no information at all — the pure prior."""
    N = data['N']
    best = (-np.inf, None)
    for a0 in np.linspace(0.05, 0.95, 91):
        a = np.full((N.size, T_NIGHTS), a0)
        J = float((utility(a, N) / N).mean())
        if J > best[0]:
            best = (J, a0)
    return best


def per_kind(model, data):
    out = {}
    a, _ = model.forward(data)
    N = data['N']
    J = utility(a, N) / N
    Jor = utility(oracle_policy(N), N) / N
    for k, name in enumerate(['calm', 'rising', 'sudden']):
        m = data['KIND'] == k
        out[name] = (float(J[m].mean()), float(Jor[m].mean()))
    return out


def ledger(model, data, i, title):
    """
    One season, night by night: what arrived, what the cell believed, what it
    spent, what it held. The real death-night is printed at the end, because
    that is when it became known.
    """
    b = slice_batch(data, np.array([i]))
    a, cache = model.forward(b)
    l_hat = np.stack(cache['ls'], axis=1)[0]
    N = int(b['N'][0])
    W = 1.0
    print(f"\n  {title}  (profile: {['calm','rising','sudden'][data['KIND'][i]]})")
    print("   night  danger  cues(mean)  stranger  slack_est  spend  wealth  consumed")
    for t in range(T_NIGHTS):
        flag, val = b['M'][0, t]
        s = '   —    ' if flag < 0.5 else ('  NEAR  ' if val > 0.5 else '  far   ')
        c = a[0, t] * W
        star = ' ' if t + 1 <= N else '†'
        print(f"   {t+1:3d}{star}   {data['D'][i, t]:5.2f}    {b['P'][0, t].mean():+6.2f}   {s}"
              f"   {l_hat[t]:6.2f}    {a[0, t]:5.3f}  {W:6.3f}   {c:6.3f}")
        W = (W - c) * R_GROWTH
    print(f"   the rope was pulled at the end of night {N}; "
          f"left on the grave: {leftover_fraction(a, b['N'])[0]:.3f} of the fortune")


# ==============================================================================
# 7.  SELF-TESTS — structural invariants
# ==============================================================================

def test_spend_fraction_in_open_unit_interval():
    m = Tiwal(seed=1)
    d = make_dataset(64, seed=11)
    for k in m.p:
        m.p[k] = m.p[k] * 30.0         # violent weights
    a, _ = m.forward(d)
    assert np.all(a > 0.0) and np.all(a <= 1.0), "spend fraction left (0,1]"
    return "spend fraction stays in (0,1] under violent weights"


def test_policy_is_the_tether():
    """a = 1/(1+ℓ̂) exactly: there is no action head."""
    m = Tiwal(seed=2)
    d = make_dataset(32, seed=12)
    a, cache = m.forward(d)
    l = np.stack(cache['ls'], axis=1)
    assert np.allclose(a, 1.0 / (1.0 + l), atol=1e-12)
    return "spend fraction is a pure function of estimated slack, a = 1/(1+ℓ̂)"


def test_urgency_is_monotone_in_slack():
    l = np.linspace(0, 30, 400)
    a = 1.0 / (1.0 + l)
    assert np.all(np.diff(a) < 0), "shorter rope must mean higher spend"
    return "spend rises monotonically as estimated slack shrinks"


def test_hindsight_optimum_is_one_over_remaining():
    """a*_t = 1/(N−t+1) maximises J: random perturbations never improve it."""
    rng = np.random.default_rng(5)
    N = rng.integers(1, T_NIGHTS + 1, 200)
    a_star = oracle_policy(N)
    J_star = utility(a_star, N)
    for _ in range(50):
        pert = np.clip(a_star * np.exp(rng.normal(0, 0.15, a_star.shape)), 1e-4, 0.9999)
        assert np.all(utility(pert, N) <= J_star + 1e-9)
    return "1/(N−t+1) is the hindsight optimum (50 random perturbations never beat it)"


def test_grave_equality():
    """
    Past the death-night, the spend schedule is irrelevant: whatever the miser
    hoarded and whatever the prodigal squandered contribute nothing.
    """
    rng = np.random.default_rng(6)
    N = rng.integers(1, T_NIGHTS - 1, 100)
    a1 = rng.uniform(0.05, 0.95, (100, T_NIGHTS))
    a2 = a1.copy()
    t_idx = np.arange(1, T_NIGHTS + 1)[None, :]
    later = t_idx > N[:, None]
    a2[later] = rng.uniform(0.05, 0.95, later.sum())
    assert np.allclose(utility(a1, N), utility(a2, N))
    return "nothing decided after the rope is pulled changes the objective (grave equality)"


def test_miser_loses_to_measured_spender():
    """A hoarder (a→small) does worse than the oracle for every N."""
    N = np.arange(1, T_NIGHTS + 1)
    miser = np.full((N.size, T_NIGHTS), 0.02)
    assert np.all(utility(miser, N) < utility(oracle_policy(N), N))
    return "the hoarder's season is worse than the measured spender's for every death-night"


def test_sealed_channel_is_dark_under_honour():
    """Under HONOUR the output is invariant to the letter's content."""
    m = Tiwal(gate='honour', seed=7)
    d = make_dataset(40, seed=17)
    a1, _ = m.forward(d)
    d2 = dict(d); d2['S'] = np.random.default_rng(1).random(d['S'].shape)
    a2, _ = m.forward(d2)
    assert np.allclose(a1, a2)
    m2 = Tiwal(gate='reader', seed=7)
    a3, _ = m2.forward(d); a4, _ = m2.forward(d2)
    assert not np.allclose(a3, a4)
    return "under HONOUR the sealed channel is present but unread; under READER it is read"


def test_stranger_carries_unprovisionable_news():
    """
    The messenger's value is informative about the realised future; the
    provisioned cues, given the danger level, are not.
    """
    d = make_dataset(4000, seed=21)
    t_idx = np.arange(1, T_NIGHTS + 1)[None, :]
    near = (d['N'][:, None] - t_idx <= NEAR).astype(float)
    flag = d['M'][:, :, 0] > 0.5
    agree = (d['M'][:, :, 1][flag] == near[flag]).mean()
    assert agree > 0.85, f"messenger accuracy {agree:.3f}"
    # provisioned cues: on the calm profile, danger is flat, so the cues
    # cannot distinguish the night before death from any other night
    calm = d['KIND'] == 0
    alive, _ = objective_terms(d['N'], T_NIGHTS)
    Pm = d['P'][calm].mean(-1)
    nr = near[calm] > 0.5; al = alive[calm] > 0.5
    gap = abs(Pm[nr & al].mean() - Pm[~nr & al].mean())
    assert gap < 0.05, f"provisioned cues leak realised future: gap {gap:.3f}"
    return f"the stranger is right {agree:.1%} of the time; provisioned cues carry no news of the realised end"


def test_naat_assignments_normalise():
    m = Tiwal(seed=8)
    d = make_dataset(16, seed=18)
    _, cache = m.forward(d)
    for alpha in cache['alphas']:
        assert np.allclose(alpha.sum(1), 1.0)
        assert np.all(alpha >= 0)
    return "artefact assignments are a proper soft partition (rows sum to 1)"


def test_dataset_shapes_and_hazard():
    d = make_dataset(2000, seed=19)
    assert d['P'].shape == (2000, T_NIGHTS, K_PROV)
    assert d['M'].shape == (2000, T_NIGHTS, 2)
    assert d['N'].min() >= 1 and d['N'].max() <= T_NIGHTS
    assert 3.0 < d['N'].mean() < 10.0, d['N'].mean()
    return f"dataset well-formed; mean death-night {d['N'].mean():.2f} of {T_NIGHTS}"


def test_training_reduces_loss():
    tr = make_dataset(800, seed=31); va = make_dataset(200, seed=32)
    m = Tiwal(gate='honour', seed=9)
    a0, _ = m.forward(va); L0 = m.loss(a0, va['N'])
    train(m, tr, va, steps=150, batch=32, lr=5e-3, log_every=10 ** 9)
    a1, _ = m.forward(va); L1 = m.loss(a1, va['N'])
    assert L1 < L0 - 0.02, (L0, L1)
    return f"150 steps of training lower validation loss {L0:.3f} → {L1:.3f}"


TESTS = [
    test_spend_fraction_in_open_unit_interval,
    test_policy_is_the_tether,
    test_urgency_is_monotone_in_slack,
    test_hindsight_optimum_is_one_over_remaining,
    test_grave_equality,
    test_miser_loses_to_measured_spender,
    test_sealed_channel_is_dark_under_honour,
    test_stranger_carries_unprovisionable_news,
    test_naat_assignments_normalise,
    test_dataset_shapes_and_hazard,
    test_training_reduces_loss,
]


# ==============================================================================
# 8.  MAIN
# ==============================================================================

def rule(ch='=', n=78):
    print(ch * n)


def main():
    np.set_printoptions(precision=4, suppress=True)
    rule()
    print(" ṬIWAL — The Slackened-Tether Network")
    print(" Chapter 0176 · Ṭarafa ibn al-ʿAbd · Encyclopedia of Lost Minds")
    rule()

    # ------------------------------------------------------------ self-tests
    print("\n[1] structural self-tests")
    for f in TESTS:
        msg = f()
        print(f"   ok  {msg}")

    # -------------------------------------------------------- gradient check
    print("\n[2] finite-difference gradient checks (central, eps=1e-5, relative error floored at 1e-3)")
    for gate, kappa, deaf in [('honour', 0.0, False), ('reader', 0.0, False),
                              ('honour', 0.0, True), ('priced', 0.1, False)]:
        ok, worst, where = gradient_check(gate=gate, kappa=kappa, deaf=deaf)
        tag = f"{gate}{'/deaf' if deaf else ''}{f'/κ={kappa}' if gate == 'priced' else ''}"
        print(f"   {tag:16s} max rel err {worst:.3e}   {'PASS' if ok else 'FAIL'}   ({where})")
        assert ok, f"gradient check failed for {tag}"

    # -------------------------------------------------------------- training
    print("\n[3] training — 6,000 seasons, 2,000 held out")
    tr = make_dataset(6000, seed=1)
    va = make_dataset(2000, seed=2)
    Jn, a_naive = naive_constant_baseline(tr)
    print(f"   naive constant baseline (no cues at all): best fixed spend {a_naive:.2f}"
          f"  →  J/N = {Jn:+.4f}")

    regimes = {
        'HONOUR': dict(gate='honour', deaf=False),   # Ṭarafa: letter unread, stranger heard
        'READER': dict(gate='reader', deaf=False),   # al-Mutalammis: letter read
        'DEAF':   dict(gate='honour', deaf=True),    # no stranger ever arrives
        'BLIND':  dict(gate='honour', deaf=True, blind=True),  # nothing but the clock
    }
    models, results = {}, {}
    for name, cfg in regimes.items():
        blind = cfg.pop('blind', False)
        m = Tiwal(seed=42, **cfg)
        trb, vab = tr, va
        if blind:
            trb = dict(tr); vab = dict(va)
            trb['P'] = np.zeros_like(tr['P']); vab['P'] = np.zeros_like(va['P'])
        print(f"\n   regime {name}")
        train(m, trb, vab, steps=2400, batch=64, lr=3e-3, seed=7, log_every=600, label=name)
        models[name] = (m, vab)
        results[name] = evaluate(m, vab)
        results[name]['kinds'] = per_kind(m, vab)

    # --------------------------------------------------------------- results
    Jor = results['HONOUR']['J_oracle']
    print("\n[4] verified results on the held-out seasons (J/N = mean log-consumption per night)")
    print(f"   hindsight oracle a*=1/(N−t+1):  J/N = {Jor:+.4f}      naive fixed spend: {Jn:+.4f}")
    print(f"\n   {'regime':8s} {'J/N':>9s} {'of gap closed':>14s} {'slack MAE':>10s} {'grave heap':>11s} {'gate':>6s}")
    for name, r in results.items():
        closed = (r['J'] - Jn) / (Jor - Jn)
        print(f"   {name:8s} {r['J']:+9.4f} {closed:14.1%} {r['slack_mae']:10.3f}"
              f" {r['leftover']:11.3f} {r['gate']:6.2f}")
    print("\n   by danger profile  (J/N model | oracle)")
    for name, r in results.items():
        ks = r['kinds']
        print(f"   {name:8s}  calm {ks['calm'][0]:+.3f}|{ks['calm'][1]:+.3f}"
              f"   rising {ks['rising'][0]:+.3f}|{ks['rising'][1]:+.3f}"
              f"   sudden {ks['sudden'][0]:+.3f}|{ks['sudden'][1]:+.3f}")

    # --------------------------------------------------- the price of honour
    print("\n[5] the price of the seal — gate trained with honour cost κ·g")
    print("   κ      learned gate g    J/N on held-out")
    for kappa in [0.0, 0.01, 0.03, 0.1, 0.3]:
        m = Tiwal(gate='priced', kappa=kappa, seed=42)
        train(m, tr, va, steps=2400, batch=64, lr=3e-3, seed=7, log_every=10 ** 9)
        r = evaluate(m, va)
        print(f"   {kappa:<6.2f}     {r['gate']:6.3f}          {r['J'] - kappa * r['gate']:+.4f}")

    # ----------------------------------------------------------------- ledgers
    print("\n[6] ledgers — one season each, as the cell lived it")
    mH, vaH = models['HONOUR']
    mR, _ = models['READER']
    sudden = np.flatnonzero((vaH['KIND'] == 2) & (vaH['N'] >= 5) & (vaH['N'] <= 9))
    calm = np.flatnonzero((vaH['KIND'] == 0) & (vaH['N'] == T_NIGHTS))
    ledger(mH, vaH, int(sudden[0]), "HONOUR — a sudden anger, letter unread")
    ledger(mR, vaH, int(sudden[0]), "READER — the same season, letter read")
    ledger(mH, vaH, int(calm[0]),   "HONOUR — a calm season lived to the end")

    # ------------------------------------------------------------- the gaps
    print("\n[7] the three gaps, in units of the naive→oracle distance")
    gap = Jor - Jn
    rH, rR, rD, rB = (results[k]['J'] for k in ('HONOUR', 'READER', 'DEAF', 'BLIND'))
    print(f"   value of reading your own orders   (READER − HONOUR): {(rR - rH) / gap:6.1%}")
    print(f"   value of the un-provisioned stranger (HONOUR − DEAF):  {(rH - rD) / gap:6.1%}")
    print(f"   value of every provisioned source   (DEAF − BLIND):    {(rD - rB) / gap:6.1%}")
    print(f"   what the clock alone recovers       (BLIND − naive):   {(rB - Jn) / gap:6.1%}")

    rule()
    print(" all self-tests passed · all gradient checks passed · training verified")
    rule()


if __name__ == '__main__':
    main()

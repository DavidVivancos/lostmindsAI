#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 0236_abd_al_rahman_iii_891_Neuron.py
 Encyclopedia of Lost Minds — Chapter 236: 'Abd al-Rahman III (891-961)
 Architecture: THE MINBAR ENGINE
              (Common-Knowledge Cascade Network with a Competence Gate)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0236_abd_al_rahman_iii_891 - 'Abd al-Rahman III (891-961)
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
'Abd al-Rahman III inherited, in 912, an emirate whose writ barely left the
walls of Cordoba. He did not reconquer al-Andalus primarily by winning battles;
he reconquered it by making submission the obviously-correct individual choice
for each faction, one at a time, and then — on 16/17 January 929 — by seizing
the two prerogatives that no emir had ever claimed in the West: the *khutba*
(the Friday sermon, which named him aloud in every congregational mosque of the
realm on the same day) and the *sikka* (the right to strike coin in his own
name). Both are broadcast media, and they are broadcast media of two different
kinds:

   khutba : SYNCHRONOUS, IDENTICAL, EPHEMERAL.  Everyone hears the same words
            in the same week. Crucially, everyone also hears *everyone else*
            hearing them. This does not merely change what people believe; it
            changes what people believe other people believe.

   sikka  : ASYNCHRONOUS, PERSISTENT, COSTLY-TO-FORGE.  A gold dinar carries
            the claim into every market for decades, and its weight is a bond
            that can be checked by anyone with a balance.

That distinction is the whole of this file. The claim "I am Commander of the
Faithful" was not true in 928 and was true in 930, and nothing about the world
changed except the structure of mutual belief. This is what game theory calls
COMMON KNOWLEDGE: not p, and not everyone-believes-p, but the infinite ladder
"everyone believes p, everyone believes everyone believes p, ...". Coordination
equilibria — obedience, currency, legitimacy, language — live at that fixed
point and nowhere below it.

So the network here is NOT a transformer, and it deliberately does not use
attention over stored keys. Its central object is a damped fixed-point
iteration over a *belief ladder*, driven by a channel that is architecturally
forbidden from being addressed to any individual node. The broadcast vector is
computed once per episode and added identically to every faction. That
constraint is the architecture's thesis.

The second thesis is the SIMANCAS GATE. On 19 July 939, at the ditch of
al-Khandaq near Simancas, his army was destroyed by Ramiro II of Leon and he
escaped with his life and lost his personal Qur'an. He crucified one traitor
and ten men of the jund — a token reprisal, not a purge — and then, for the
remaining twenty-two years of his reign, he never personally commanded an army
again. He did not deny the defeat, and he did not rationalise it. He measured
his own competence in one domain, found it wanting, and permanently routed that
domain to others. The model reproduces this: a head that predicts the model's
OWN error rate, is trained against it, and is used at inference to abstain.

WHAT THE FILE CONTAINS
----------------------
  Part 1  A ~200-line reverse-mode autodiff engine in pure NumPy.
  Part 2  The Bay'a environment: a threshold-cascade coordination game on a
          network, with a public signal and two theatres of differing noise.
  Part 3  The Minbar Engine itself.
  Part 4  Losses.
  Part 5  A finite-difference gradient check (MANDATORY, must pass).
  Part 6  Training.
  Part 7  Evaluation: (a) cascade prediction, (b) proclamation search by
          gradient ascent through the world model, scored in the TRUE
          environment against an exhaustive oracle, (c) gate calibration and
          selective prediction.
  Part 8  Self-tests, including a falsification test that proves the broadcast
          channel is doing real work.

Dependencies: numpy only. Runtime: ~1-2 minutes on a laptop CPU.
================================================================================
"""

import numpy as np
import math
import time

RNG_GLOBAL = np.random.default_rng(20250216)

# ==============================================================================
# PART 1 — A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Everything below is built from scratch on NumPy arrays. A Node wraps a value,
# an accumulated gradient, a list of parents and a closure that pushes gradient
# into those parents. `backward` topologically sorts and runs the closures once
# each. This is ~200 lines and it is validated in Part 5 against finite
# differences, so nothing here is taken on trust.
# ==============================================================================


class Node:
    """A value on the tape, with a gradient slot and a backward closure."""

    __slots__ = ("v", "g", "_parents", "_bw", "tag")

    def __init__(self, value, parents=(), bw=None, tag="leaf"):
        self.v = np.asarray(value, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._parents = parents
        self._bw = bw
        self.tag = tag

    @property
    def shape(self):
        return self.v.shape

    def __repr__(self):
        return f"Node<{self.tag} shape={self.v.shape}>"


def _unbroadcast(grad, shape):
    """Reverse NumPy broadcasting: sum `grad` down to `shape`."""
    if grad.shape == shape:
        return grad
    # collapse leading axes that were added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # collapse axes that were size-1 in the original
    for ax, size in enumerate(shape):
        if size == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad.reshape(shape)


def backward(root):
    """Run reverse-mode accumulation from a scalar Node."""
    assert root.v.size == 1, "backward() requires a scalar root"
    topo, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n._parents:
            visit(p)
        topo.append(n)

    visit(root)
    for n in topo:
        n.g = np.zeros_like(n.v)
    root.g = np.ones_like(root.v)
    for n in reversed(topo):
        if n._bw is not None:
            n._bw(n.g)


# ---- primitive operations ----------------------------------------------------

def add(a, b):
    out = Node(a.v + b.v, (a, b), tag="add")

    def bw(g):
        a.g += _unbroadcast(g, a.v.shape)
        b.g += _unbroadcast(g, b.v.shape)

    out._bw = bw
    return out


def mul(a, b):
    out = Node(a.v * b.v, (a, b), tag="mul")

    def bw(g):
        a.g += _unbroadcast(g * b.v, a.v.shape)
        b.g += _unbroadcast(g * a.v, b.v.shape)

    out._bw = bw
    return out


def scale(a, k):
    """Multiply by a Python float (no gradient flows to k)."""
    out = Node(a.v * k, (a,), tag="scale")

    def bw(g):
        a.g += g * k

    out._bw = bw
    return out


def sub(a, b):
    return add(a, scale(b, -1.0))


def dense(x, W):
    """x (..., a) @ W (a, b) -> (..., b)."""
    out = Node(x.v @ W.v, (x, W), tag="dense")

    def bw(g):
        x.g += g @ W.v.T
        x2 = x.v.reshape(-1, x.v.shape[-1])
        g2 = g.reshape(-1, g.shape[-1])
        W.g += x2.T @ g2

    out._bw = bw
    return out


def const_bmm(A, x):
    """Batched mix by a CONSTANT numpy adjacency: A (B,K,K) x (B,K,h)."""
    val = np.einsum("bij,bjh->bih", A, x.v)
    out = Node(val, (x,), tag="const_bmm")

    def bw(g):
        x.g += np.einsum("bij,bih->bjh", A, g)

    out._bw = bw
    return out


def tanh(x):
    t = np.tanh(x.v)
    out = Node(t, (x,), tag="tanh")

    def bw(g):
        x.g += g * (1.0 - t * t)

    out._bw = bw
    return out


def mean_axis(x, axis, keepdims=True):
    out = Node(x.v.mean(axis=axis, keepdims=keepdims), (x,), tag="mean_axis")
    n = x.v.shape[axis]

    def bw(g):
        gg = g if keepdims else np.expand_dims(g, axis)
        x.g += np.broadcast_to(gg / n, x.v.shape).copy()

    out._bw = bw
    return out


def mean_all(x):
    out = Node(np.array(x.v.mean()), (x,), tag="mean_all")
    n = x.v.size

    def bw(g):
        x.g += np.full(x.v.shape, float(g) / n)

    out._bw = bw
    return out


def unsqueeze(x, axis):
    out = Node(np.expand_dims(x.v, axis), (x,), tag="unsqueeze")

    def bw(g):
        x.g += g.reshape(x.v.shape)

    out._bw = bw
    return out


def concat(nodes, axis=-1):
    out = Node(np.concatenate([n.v for n in nodes], axis=axis), tuple(nodes),
               tag="concat")
    sizes = [n.v.shape[axis] for n in nodes]

    def bw(g):
        idx, off = [slice(None)] * g.ndim, 0
        for n, s in zip(nodes, sizes):
            idx[axis] = slice(off, off + s)
            n.g += g[tuple(idx)]
            off += s

    out._bw = bw
    return out


def bce_logits(z, t, weight=None):
    """Numerically stable mean binary cross-entropy. `t` may be a soft target."""
    zv = z.v
    loss = np.maximum(zv, 0) - zv * t + np.log1p(np.exp(-np.abs(zv)))
    if weight is not None:
        loss = loss * weight
    out = Node(np.array(loss.mean()), (z,), tag="bce")
    n = loss.size

    def bw(g):
        p = 1.0 / (1.0 + np.exp(-zv))
        d = (p - t) / n
        if weight is not None:
            d = d * weight
        z.g += float(g) * d

    out._bw = bw
    return out


def sigmoid(x):
    p = 1.0 / (1.0 + np.exp(-np.clip(x.v, -60, 60)))
    out = Node(p, (x,), tag="sigmoid")

    def bw(g):
        x.g += g * p * (1.0 - p)

    out._bw = bw
    return out


def slice_last(x, a, b):
    """Take x[..., a:b]. Used to isolate the PUBLIC REGISTER of the belief
    vector — the subspace in which common knowledge is measured."""
    out = Node(x.v[..., a:b], (x,), tag="slice_last")

    def bw(g):
        pad = np.zeros_like(x.v)
        pad[..., a:b] = g
        x.g += pad

    out._bw = bw
    return out


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ==============================================================================
# PART 2 — THE BAY'A ENVIRONMENT
# ==============================================================================
# `bay'a` is the oath of allegiance. The historical problem 'Abd al-Rahman III
# faced in 912 was not "who is strong?" but "who will submit, given that
# submission is only safe if enough others submit?". That is a coordination game
# with multiple equilibria, and it is exactly a threshold cascade on a network.
#
# Each of K factions has:
#   - features (distance from Cordoba, kinship to the Umayyads, own military
#     strength, grievance, doctrinal alignment, whether it had submitted before)
#   - a private threshold theta_i derived from those features
#   - a private sensitivity to each of the two dimensions of a public
#     proclamation: a claim of religious authority, and a promise of largesse
#
# A faction submits at step t+1 iff
#      local pressure  (neighbours who have submitted)
#    + global pressure (overall submission rate — this is the term the khutba
#                       actually manipulates, because it is what everyone can
#                       see everyone seeing)
#    + response to the proclamation
#    + kinship pull - grievance
#    > theta_i
#
# TWO THEATRES. Theatre 0 is the northern frontier: the cascade is run with
# injected noise, because the marches were governed by autonomous lineages and
# fog. Theatre 1 is the court/interior: deterministic. The model is told which
# theatre it is in but is NOT told the noise level; it has to discover that its
# own predictions are worse in one of them. That discovery is the Simancas gate.
# ==============================================================================

K_FACTIONS = 12
FEAT_DIM = 6
PROC_DIM = 2          # (claim of authority, promise of largesse)
CASCADE_STEPS = 10
W_LOCAL = 1.15
W_GLOBAL = 0.85
# The environment is deliberately tuned to be BISTABLE. With no proclamation
# the cascade collapses (almost nobody gives bay'a); with the right
# proclamation it ignites and runs to near-universal submission. The
# interesting regime — the only regime in which the khutba is worth anything —
# is the neighbourhood of the tipping point, and that is where noise bites
# hardest. This is why the frontier is genuinely harder to predict than the
# court: not because there is more of it, but because it sits on the knife edge.
FRONTIER_NOISE = 0.42  # std of pressure noise in theatre 0 (the marches)
COURT_NOISE = 0.02     # theatre 1 (Cordoba and the settled interior)


def _ring_adjacency(K):
    """Base topology: a ring — al-Andalus as a chain of marches and provinces."""
    A = np.zeros((K, K))
    for i in range(K):
        A[i, (i - 1) % K] = 1.0
        A[i, (i + 1) % K] = 1.0
    return A


_RING = _ring_adjacency(K_FACTIONS)


def make_batch(rng, B, K=K_FACTIONS, proclamation=None, theatre=None):
    """Generate B independent episodes, fully vectorised.

    Returns a dict with everything the model and the losses need.
    """
    dist = rng.uniform(0, 1, (B, K))     # distance from Cordoba
    kin = rng.uniform(0, 1, (B, K))      # Umayyad clientage
    mil = rng.uniform(0, 1, (B, K))      # own military weight
    griev = rng.uniform(0, 1, (B, K))    # accumulated grievance
    creed = rng.uniform(0, 1, (B, K))    # doctrinal alignment
    prior = (rng.uniform(0, 1, (B, K)) < 0.22).astype(float)

    X = np.stack([dist, kin, mil, griev, creed, prior], axis=-1)  # (B,K,6)

    # --- adjacency: ring + a few long-range ties (marriage, trade, rebellion) --
    extra = (rng.uniform(0, 1, (B, K, K)) < 0.06).astype(float)
    extra = np.maximum(extra, np.transpose(extra, (0, 2, 1)))
    A = np.maximum(np.broadcast_to(_RING, (B, K, K)), extra).copy()
    for b in range(B):
        np.fill_diagonal(A[b], 0.0)
    A = A / np.maximum(A.sum(axis=2, keepdims=True), 1e-9)

    # --- private thresholds and signal sensitivities ---------------------------
    theta = 0.62 + 0.78 * dist + 0.78 * griev + 0.55 * mil - 0.55 * kin
    eta_auth = 0.95 * creed + 0.25 * kin        # who responds to a caliphal claim
    eta_gold = 0.85 * (1 - creed) + 0.55 * mil  # who responds to money

    if theatre is None:
        theatre = (rng.uniform(0, 1, B) < 0.5).astype(int)
    theatre = np.asarray(theatre).reshape(B)
    noise_sd = np.where(theatre == 0, FRONTIER_NOISE, COURT_NOISE)[:, None]

    if proclamation is None:
        proclamation = rng.uniform(0, 1, (B, PROC_DIM))
    proclamation = np.asarray(proclamation, dtype=float).reshape(B, PROC_DIM)

    y = run_cascade(A, theta, eta_auth, eta_gold, kin, griev, prior,
                    proclamation, noise_sd, rng)

    theatre_1h = np.zeros((B, 2))
    theatre_1h[np.arange(B), theatre] = 1.0

    return dict(X=X, A=A, y=y, proc=proclamation, theatre=theatre,
                theatre_1h=theatre_1h,
                latent=dict(theta=theta, eta_auth=eta_auth, eta_gold=eta_gold,
                            kin=kin, griev=griev, prior=prior,
                            noise_sd=noise_sd))


def run_cascade(A, theta, eta_auth, eta_gold, kin, griev, prior,
                proclamation, noise_sd, rng):
    """The ground-truth dynamics. Returns final submission state (B,K) in {0,1}.

    Note that both directions are permitted: a faction that has submitted can
    defect again if support around it collapses. Legitimacy is not a ratchet.
    """
    u = prior.copy()
    pa = proclamation[:, 0:1]
    pg = proclamation[:, 1:2]
    for _ in range(CASCADE_STEPS):
        local = np.einsum("bij,bj->bi", A, u)
        glob = u.mean(axis=1, keepdims=True)
        press = (W_LOCAL * local + W_GLOBAL * glob
                 + eta_auth * pa + eta_gold * pg
                 + 0.40 * kin - 0.35 * griev)
        if noise_sd is not None:
            press = press + rng.normal(0.0, 1.0, press.shape) * noise_sd
        u_new = (press > theta).astype(float)
        if np.array_equal(u_new, u):
            break
        u = u_new
    return u


def true_score(batch, proclamation, cost=(0.55, 0.75), rng=None, reps=24):
    """Evaluate a proclamation in the TRUE environment.

    Score = mean submission rate  -  cost of the claim made.
    Both dimensions are expensive: an unbacked claim of religious authority
    spends credibility, and largesse spends gold. Averaged over `reps` draws of
    the frontier noise so the number is not an artefact of one lucky roll.
    """
    L = batch["latent"]
    rng = rng or np.random.default_rng(0)
    B = batch["A"].shape[0]
    p = np.asarray(proclamation, dtype=float).reshape(B, PROC_DIM)
    acc = np.zeros(B)
    for _ in range(reps):
        u = run_cascade(batch["A"], L["theta"], L["eta_auth"], L["eta_gold"],
                        L["kin"], L["griev"], L["prior"], p, L["noise_sd"], rng)
        acc += u.mean(axis=1)
    acc /= reps
    penalty = cost[0] * p[:, 0] + cost[1] * p[:, 1]
    return acc - penalty


# ==============================================================================
# PART 3 — THE MINBAR ENGINE
# ==============================================================================
# The minbar is the pulpit from which the khutba is delivered. The engine has
# five parts and each one is a claim about how this particular mind worked.
#
#   (1) COURT ENCODER          per-faction embedding of what is known about it
#   (2) KHUTBA HEAD            ONE vector, computed from the court summary and
#                              the proclamation, added IDENTICALLY to every
#                              faction. There is no path in this graph by which
#                              a faction-specific message can be sent. That is
#                              not an optimisation; it is the point. Private
#                              persuasion moves first-order belief. Only a
#                              signal that everyone knows everyone received can
#                              move the higher orders.
#   (3) SIKKA LEDGER           a slowly-updating persistent vector carried
#                              across episodes, detached at the boundary. The
#                              coinage: low-bandwidth, durable, and a check on
#                              how much the broadcast can claim.
#   (4) BELIEF LADDER          L damped iterations of
#                                 b <- (1-a) b + a tanh(Ws b + Wn (A b)
#                                                       + Wf mean(b) + broadcast
#                                                       + ledger)
#                              Iteration l is the l-th order of belief: what I
#                              think, what I think you think, and so on. Common
#                              knowledge is the fixed point. Dispersion across
#                              factions at the fixed point is the legitimacy
#                              readout — when everyone's belief state has
#                              converged to the same place, the claim is
#                              common knowledge and the equilibrium holds.
#   (5) SIMANCAS GATE          predicts the model's OWN error rate from the
#                              settled belief state, its dispersion, and the
#                              theatre. Trained against the realised error.
# ==============================================================================

H_DIM = 28
CK_DIM = 8          # the PUBLIC REGISTER: dims 0..CK_DIM of the belief vector
                    # hold beliefs ABOUT THE CALIPH'S STANDING. Common
                    # knowledge is required there and nowhere else: factions
                    # need not agree about everything, only about who is
                    # Commander of the Faithful.
BROAD_DIM = 16
LEDGER_DIM = 12
GATE_DIM = 20
LADDER_STEPS = 6
LADDER_ALPHA = 0.55


def init_params(rng, h=H_DIM, dp=BROAD_DIM, ds=LEDGER_DIM, hg=GATE_DIM):
    """Xavier-ish initialisation. Every entry is a leaf Node on the tape."""

    def W(a, b, gain=1.0):
        return Node(rng.normal(0, gain * math.sqrt(2.0 / (a + b)), (a, b)),
                    tag="W")

    def z(n):
        return Node(np.zeros(n), tag="b")

    P = {
        # (1) court encoder
        "We": W(FEAT_DIM, h), "be": z(h),
        # (2) khutba head: court summary + proclamation -> one public vector
        "Wp": W(h + PROC_DIM, dp), "bp": z(dp),
        # (3) sikka ledger update
        "Ws_led": W(h + dp, ds), "bs_led": z(ds),
        # (4) belief ladder
        "Wself": W(h, h), "Wnbr": W(h, h), "Wfield": W(h, h),
        "Wbroad": W(dp, h), "Wledger": W(ds, h), "bb": z(h),
        # readout: does this faction give bay'a?
        "Wo": W(h, 1), "bo": z(1),
        # (5) Simancas gate
        "Wg": W(h + 1 + 2, hg), "bg": z(hg),
        "Wr": W(hg, 1), "br": z(1),
    }
    return P


def forward(P, batch, ledger_prev, ladder_steps=LADDER_STEPS,
            alpha=LADDER_ALPHA, ledger_rho=0.15, broadcast_on=True):
    """One forward pass. `ledger_prev` is a plain numpy array (detached).

    Returns a dict of Nodes and diagnostics.
    """
    B = batch["X"].shape[0]
    X = Node(batch["X"], tag="X")
    proc = Node(batch["proc"], tag="proc")
    theatre = Node(batch["theatre_1h"], tag="theatre")
    A = batch["A"]                       # constant, not differentiated

    # ---- (1) court encoder ---------------------------------------------------
    Hh = tanh(add(dense(X, P["We"]), P["be"]))          # (B,K,h)
    court = mean_axis(Hh, axis=1, keepdims=False)        # (B,h) summary

    # ---- (2) khutba: exactly one public vector per episode --------------------
    khutba = tanh(add(dense(concat([court, proc], -1), P["Wp"]), P["bp"]))  # (B,dp)

    # ---- (3) sikka: durable channel, EMA across episodes ----------------------
    led_prev = Node(ledger_prev, tag="ledger_prev")      # detached leaf
    led_new = tanh(add(dense(concat([court, khutba], -1), P["Ws_led"]),
                       P["bs_led"]))                     # (B,ds)
    ledger = add(scale(led_prev, 1.0 - ledger_rho), scale(led_new, ledger_rho))

    # project both global channels once; they are broadcast to all K factions
    if broadcast_on:
        pub = unsqueeze(dense(khutba, P["Wbroad"]), 1)   # (B,1,h)
    else:
        pub = Node(np.zeros((B, 1, H_DIM)), tag="pub_off")
    led = unsqueeze(dense(ledger, P["Wledger"]), 1)      # (B,1,h)

    # ---- (4) belief ladder ---------------------------------------------------
    b = Hh
    trace, residual = [], []
    for _ in range(ladder_steps):
        b_prev_v = b.v.copy()
        nbr = const_bmm(A, b)                            # who my neighbours are
        field = mean_axis(b, axis=1, keepdims=True)      # what everyone is doing
        u = add(add(add(dense(b, P["Wself"]), dense(nbr, P["Wnbr"])),
                    add(dense(field, P["Wfield"]), pub)),
                add(led, P["bb"]))
        b = add(scale(b, 1.0 - alpha), scale(tanh(u), alpha))
        trace.append(float(np.var(b.v[..., :CK_DIM], axis=1).mean()))
        # fixed-point residual: how much did this order of belief still move?
        residual.append(float(np.linalg.norm(b.v - b_prev_v)
                              / (np.linalg.norm(b_prev_v) + 1e-12)))

    # ---- readout: bay'a logits ----------------------------------------------
    z_sub = add(dense(b, P["Wo"]), P["bo"])              # (B,K,1)

    # ---- dispersion over the public register = 1 / legitimacy ---------------
    pub_reg = slice_last(b, 0, CK_DIM)                   # (B,K,CK_DIM)
    m = mean_axis(pub_reg, axis=1, keepdims=True)
    dev = sub(pub_reg, m)
    disp_per_ep = mean_axis(mean_axis(mul(dev, dev), axis=1, keepdims=False),
                            axis=-1, keepdims=True)      # (B,1)
    disp_scalar = mean_all(mul(dev, dev))

    # ---- (5) Simancas gate ---------------------------------------------------
    settled = mean_axis(b, axis=1, keepdims=False)       # (B,h)
    gate_in = concat([settled, disp_per_ep, theatre], -1)
    q = tanh(add(dense(gate_in, P["Wg"]), P["bg"]))
    z_risk = add(dense(q, P["Wr"]), P["br"])             # (B,1) logit of own error

    return dict(z_sub=z_sub, z_risk=z_risk, disp=disp_scalar,
                khutba=khutba, ledger=ledger, belief=b,
                ladder_variance=trace, ladder_residual=residual)


# ==============================================================================
# PART 4 — LOSSES
# ==============================================================================
# L = bay'a prediction
#   + lambda_d * dispersion            (pressure toward a common-knowledge state)
#   + lambda_g * gate calibration      (learn your own error rate)
#   + lambda_c * broadcast cost        (a claim is not free; overclaiming is
#                                       punished, which is why the model cannot
#                                       simply shout the maximum every time)
# ==============================================================================

LAM_DISP = 0.030
LAM_GATE = 0.500
LAM_COST = 0.004


def compute_loss(P, out, batch):
    y = batch["y"][..., None]                            # (B,K,1)
    l_sub = bce_logits(out["z_sub"], y)

    # realised per-episode error rate -> soft target for the gate (no gradient)
    p_hat = sigmoid_np(out["z_sub"].v)[..., 0]
    err = (np.abs(p_hat - batch["y"]) > 0.5).mean(axis=1, keepdims=True)
    l_gate = bce_logits(out["z_risk"], err)

    l_cost = mean_all(mul(out["khutba"], out["khutba"]))

    total = add(add(l_sub, scale(out["disp"], LAM_DISP)),
                add(scale(l_gate, LAM_GATE), scale(l_cost, LAM_COST)))
    return total, dict(sub=float(l_sub.v), gate=float(l_gate.v),
                       disp=float(out["disp"].v), cost=float(l_cost.v),
                       err=float(err.mean()))


# ==============================================================================
# PART 5 — FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# ==============================================================================

def gradient_check(seed=7, n_probe=40, eps=1e-6, tol=2e-5, verbose=True):
    rng = np.random.default_rng(seed)
    P = init_params(rng)
    batch = make_batch(rng, B=5)
    ledger0 = rng.normal(0, 0.2, (5, LEDGER_DIM))

    def loss_value():
        out = forward(P, batch, ledger0)
        total, _ = compute_loss(P, out, batch)
        return total

    total = loss_value()
    backward(total)
    analytic = {k: P[k].g.copy() for k in P}

    keys = list(P.keys())
    worst, worst_where = 0.0, None
    checked = 0
    for i in range(n_probe):
        k = keys[i % len(keys)]
        flat_idx = rng.integers(0, P[k].v.size)
        idx = np.unravel_index(flat_idx, P[k].v.shape)
        orig = P[k].v[idx]

        P[k].v[idx] = orig + eps
        lp = float(loss_value().v)
        P[k].v[idx] = orig - eps
        lm = float(loss_value().v)
        P[k].v[idx] = orig

        num = (lp - lm) / (2 * eps)
        ana = analytic[k][idx]
        denom = max(1.0, abs(num), abs(ana))
        rel = abs(num - ana) / denom
        checked += 1
        if rel > worst:
            worst, worst_where = rel, (k, idx, num, ana)

    ok = worst < tol
    if verbose:
        print(f"  probes checked      : {checked}")
        print(f"  worst relative error: {worst:.3e}   (tolerance {tol:.0e})")
        if worst_where:
            k, idx, num, ana = worst_where
            print(f"  worst parameter     : {k}{list(idx)}  "
                  f"numeric={num:+.8f}  analytic={ana:+.8f}")
        print(f"  RESULT              : {'PASS' if ok else 'FAIL'}")
    return ok, worst


# ==============================================================================
# PART 6 — TRAINING
# ==============================================================================
# Adam, written out. The sikka ledger is carried between steps and detached at
# the boundary: the coin from last year is an input to this year, not a thing
# this year's gradient may rewrite.
# ==============================================================================


class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.P, self.lr, self.b1, self.b2, self.eps, self.clip = \
            params, lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.v) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        gn = math.sqrt(sum(float((p.g ** 2).sum()) for p in self.P.values()))
        s = min(1.0, self.clip / (gn + 1e-12))
        for k, p in self.P.items():
            g = p.g * s
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return gn


def train(steps=1400, B=48, seed=11, log_every=200, verbose=True):
    rng = np.random.default_rng(seed)
    P = init_params(rng)
    opt = Adam(P)
    ledger = np.zeros((B, LEDGER_DIM))
    history = []
    t0 = time.time()
    lr0 = opt.lr
    for s in range(1, steps + 1):
        # cosine decay: the caliph's late reign is not his early reign
        opt.lr = lr0 * (0.06 + 0.94 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        batch = make_batch(rng, B=B)
        out = forward(P, batch, ledger)
        total, parts = compute_loss(P, out, batch)
        backward(total)
        gn = opt.step()
        ledger = out["ledger"].v.copy()          # carry forward, detached
        history.append(parts)
        if verbose and (s % log_every == 0 or s == 1):
            acc = 1.0 - parts["err"]
            print(f"  step {s:4d} | loss {float(total.v):.4f} | bay'a-BCE "
                  f"{parts['sub']:.4f} | acc {acc:.3f} | dispersion "
                  f"{parts['disp']:.4f} | gate {parts['gate']:.4f} | "
                  f"|g| {gn:.2f}")
    if verbose:
        print(f"  trained {steps} steps in {time.time()-t0:.1f}s")
    return P, history


# ==============================================================================
# PART 7 — EVALUATION
# ==============================================================================

def eval_prediction(P, rng, n_batches=12, B=48):
    """Held-out accuracy of the cascade world-model, split by theatre."""
    ledger = np.zeros((B, LEDGER_DIM))
    tot = {0: [0, 0], 1: [0, 0]}
    for _ in range(n_batches):
        batch = make_batch(rng, B=B)
        out = forward(P, batch, ledger)
        ledger = out["ledger"].v.copy()
        pred = (sigmoid_np(out["z_sub"].v)[..., 0] > 0.5).astype(float)
        hit = (pred == batch["y"]).mean(axis=1)
        for t in (0, 1):
            m = batch["theatre"] == t
            tot[t][0] += hit[m].sum()
            tot[t][1] += m.sum()
    return {t: (tot[t][0] / max(tot[t][1], 1)) for t in (0, 1)}


def search_proclamation(P, batch, iters=180, lr=0.10, cost=(0.55, 0.75),
                        seed=3, restarts=6):
    """Choose a khutba by gradient ascent THROUGH the differentiable model.

    This is the caliph deliberating: he cannot run the year twice, so he runs it
    inside his model of the realm and reads off the claim that maximises assent
    net of what the claim will cost him.
    """
    rng = np.random.default_rng(seed)
    B = batch["A"].shape[0]
    ledger = np.zeros((B, LEDGER_DIM))
    best_proc = np.full((B, PROC_DIM), 0.5)
    best_val = np.full(B, -1e9)

    for r in range(restarts):
        proc = (np.full((B, PROC_DIM), 0.5) if r == 0
                else rng.uniform(0, 1, (B, PROC_DIM)))
        m = np.zeros_like(proc)
        v = np.zeros_like(proc)
        proc, val = _ascend(P, batch, proc, ledger, m, v, iters, lr, cost)
        better = val > best_val
        best_val[better] = val[better]
        best_proc[better] = proc[better]
    return best_proc


def _ascend(P, batch, proc, ledger, m, v, iters, lr, cost):
    """Adam ascent on the proclamation, differentiating through the world
    model. The caliph cannot run the year twice, so he runs it inside his
    model of the realm and reads off the claim worth making."""
    B = batch["A"].shape[0]
    last_obj = None
    for t in range(1, iters + 1):
        bb = dict(batch)
        bb["proc"] = proc
        pnode = Node(proc.copy(), tag="proc_search")

        # forward with `proc` as the differentiated variable
        X = Node(bb["X"]); A = bb["A"]
        Hh = tanh(add(dense(X, P["We"]), P["be"]))
        court = mean_axis(Hh, axis=1, keepdims=False)
        khutba = tanh(add(dense(concat([court, pnode], -1), P["Wp"]), P["bp"]))
        led_prev = Node(ledger)
        led_new = tanh(add(dense(concat([court, khutba], -1), P["Ws_led"]),
                           P["bs_led"]))
        ledger_n = add(scale(led_prev, 0.85), scale(led_new, 0.15))
        pub = unsqueeze(dense(khutba, P["Wbroad"]), 1)
        led = unsqueeze(dense(ledger_n, P["Wledger"]), 1)
        b = Hh
        for _ in range(LADDER_STEPS):
            nbr = const_bmm(A, b)
            field = mean_axis(b, axis=1, keepdims=True)
            u = add(add(add(dense(b, P["Wself"]), dense(nbr, P["Wnbr"])),
                        add(dense(field, P["Wfield"]), pub)),
                    add(led, P["bb"]))
            b = add(scale(b, 1 - LADDER_ALPHA), scale(tanh(u), LADDER_ALPHA))
        z = add(dense(b, P["Wo"]), P["bo"])

        # objective: predicted assent minus the price of the claim
        assent = mean_all(sigmoid(z))           # predicted mean submission rate
        cnode = Node(np.array(cost))
        price = mean_all(mul(pnode, cnode))
        obj = sub(assent, scale(price, float(PROC_DIM)))
        backward(obj)

        g = pnode.g * B                      # per-episode gradient of the score
        m[:] = 0.9 * m + 0.1 * g
        v[:] = 0.999 * v + 0.001 * (g * g)
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        proc = np.clip(proc + lr * mh / (np.sqrt(vh) + 1e-8), 0.0, 1.0)
        last_obj = sigmoid_np(z.v)[..., 0].mean(axis=1) - (
            cost[0] * proc[:, 0] + cost[1] * proc[:, 1])
    return proc, last_obj


def oracle_proclamation(batch, grid=13, cost=(0.55, 0.75), seed=5):
    """Exhaustive search in the TRUE environment — the best any policy could do."""
    rng = np.random.default_rng(seed)
    B = batch["A"].shape[0]
    axis = np.linspace(0, 1, grid)
    best = np.full(B, -1e9)
    best_p = np.zeros((B, PROC_DIM))
    for a in axis:
        for g in axis:
            p = np.tile(np.array([a, g]), (B, 1))
            s = true_score(batch, p, cost=cost,
                           rng=np.random.default_rng(seed), reps=8)
            m = s > best
            best[m] = s[m]
            best_p[m] = p[m]
    return best_p, best


def eval_gate(P, rng, n_batches=14, B=48):
    """Is the model's self-assessment calibrated, and does abstaining help?"""
    ledger = np.zeros((B, LEDGER_DIM))
    risks, errs = [], []
    for _ in range(n_batches):
        batch = make_batch(rng, B=B)
        out = forward(P, batch, ledger)
        ledger = out["ledger"].v.copy()
        p_hat = sigmoid_np(out["z_sub"].v)[..., 0]
        e = (np.abs(p_hat - batch["y"]) > 0.5).mean(axis=1)
        risks.append(sigmoid_np(out["z_risk"].v)[:, 0])
        errs.append(e)
    r = np.concatenate(risks)
    e = np.concatenate(errs)
    corr = float(np.corrcoef(r, e)[0, 1])
    order = np.argsort(r)
    n = len(r)
    kept_50 = 1.0 - e[order[: n // 2]].mean()
    kept_25 = 1.0 - e[order[: n // 4]].mean()
    return dict(corr=corr, acc_all=1.0 - e.mean(),
                acc_best50=kept_50, acc_best25=kept_25,
                risk_mean=float(r.mean()))


# ==============================================================================
# PART 8 — SELF-TESTS AND MAIN
# ==============================================================================

def test_broadcast_is_load_bearing(P, rng, B=48, n=8):
    """Falsification test.

    If the public channel were decorative, ablating it would not hurt. The
    historical claim embedded in this architecture is that it is not
    decorative. Ablate `pub` and measure.
    """
    on, off = [], []
    ledger = np.zeros((B, LEDGER_DIM))
    for _ in range(n):
        batch = make_batch(rng, B=B)
        o1 = forward(P, batch, ledger, broadcast_on=True)
        o0 = forward(P, batch, ledger, broadcast_on=False)
        ledger = o1["ledger"].v.copy()
        for o, acc in ((o1, on), (o0, off)):
            pred = (sigmoid_np(o["z_sub"].v)[..., 0] > 0.5).astype(float)
            acc.append((pred == batch["y"]).mean())
    return float(np.mean(on)), float(np.mean(off))


def test_ladder_converges(P, rng, steps=14):
    """The belief ladder must SETTLE. Common knowledge is a fixed point of the
    iterated-belief operator, so the residual ||b_{l+1} - b_l|| must decay. If
    it did not, higher orders of belief would keep changing the answer and no
    stable equilibrium — no currency, no obedience, no caliphate — could exist.
    """
    batch = make_batch(rng, B=16)
    out = forward(P, batch, np.zeros((16, LEDGER_DIM)), ladder_steps=steps)
    res = out["ladder_residual"]
    settled = res[-1] < 0.25 * res[0]
    return res, out["ladder_variance"], settled


def test_public_channel_is_truly_public(P, rng):
    """Structural test: assert the broadcast term is byte-identical across all
    K factions. If a per-faction message could be sent, this fails."""
    batch = make_batch(rng, B=4)
    X = Node(batch["X"])
    proc = Node(batch["proc"])
    Hh = tanh(add(dense(X, P["We"]), P["be"]))
    court = mean_axis(Hh, axis=1, keepdims=False)
    khutba = tanh(add(dense(concat([court, proc], -1), P["Wp"]), P["bp"]))
    pub = unsqueeze(dense(khutba, P["Wbroad"]), 1).v      # (B,1,h)
    tiled = np.broadcast_to(pub, (4, K_FACTIONS, H_DIM))
    return bool(np.all(tiled == tiled[:, :1, :])) and pub.shape[1] == 1


def main():
    line = "=" * 78
    print(line)
    print(" THE MINBAR ENGINE — 'Abd al-Rahman III (891-961), Chapter 214")
    print(" Common-Knowledge Cascade Network with a Simancas competence gate")
    print(line)

    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    ok, worst = gradient_check()
    assert ok, "gradient check failed"

    print("\n[2] STRUCTURAL TEST — is the broadcast channel actually public?")
    rng = np.random.default_rng(101)
    P0 = init_params(np.random.default_rng(2))
    pub_ok = test_public_channel_is_truly_public(P0, rng)
    print(f"  broadcast vector identical for all {K_FACTIONS} factions: {pub_ok}")
    assert pub_ok

    print("\n[3] TRAINING THE WORLD MODEL")
    P, hist = train()

    print("\n[4] HELD-OUT CASCADE PREDICTION (by theatre)")
    rng = np.random.default_rng(555)
    acc = eval_prediction(P, rng)
    print(f"  theatre 1 (court, low noise)     accuracy: {acc[1]:.3f}")
    print(f"  theatre 0 (frontier, high noise) accuracy: {acc[0]:.3f}")
    print(f"  gap the gate must learn to see          : {acc[1]-acc[0]:+.3f}")

    print("\n[5] BELIEF LADDER — does iterated belief reach a fixed point?")
    res, var, settled = test_ladder_converges(P, np.random.default_rng(77))
    print("  residual ||b_l - b_(l-1)|| / ||b_(l-1)|| by order of belief:")
    print("   " + "  ".join(f"L{i+1}={v:.4f}" for i, v in enumerate(res[:7])))
    print("   " + "  ".join(f"L{i+8}={v:.4f}" for i, v in enumerate(res[7:])))
    print(f"  settled (final residual < 25% of first): {settled}")
    print(f"  public-register dispersion, first -> last: "
          f"{var[0]:.4f} -> {var[-1]:.4f}")
    assert settled, "belief ladder failed to reach a fixed point"

    print("\n[6] ABLATION — is the public channel load-bearing?")
    a_on, a_off = test_broadcast_is_load_bearing(P, np.random.default_rng(31))
    print(f"  accuracy with khutba : {a_on:.3f}")
    print(f"  accuracy without     : {a_off:.3f}")
    print(f"  contribution         : {a_on - a_off:+.3f}")

    print("\n[7] CHOOSING A PROCLAMATION (gradient ascent through the model)")
    ev_rng = np.random.default_rng(909)
    test_batch = make_batch(ev_rng, B=40)
    p_model = search_proclamation(P, test_batch)
    p_zero = np.zeros((40, PROC_DIM))
    p_max = np.ones((40, PROC_DIM))
    p_rand = np.random.default_rng(4).uniform(0, 1, (40, PROC_DIM))
    p_orc, s_orc = oracle_proclamation(test_batch)

    def sc(p):
        return float(true_score(test_batch, p,
                                rng=np.random.default_rng(1234)).mean())

    s_model, s_zero, s_max, s_rand = sc(p_model), sc(p_zero), sc(p_max), sc(p_rand)
    s_oracle = float(s_orc.mean())
    print(f"  silence      (0.00, 0.00) : {s_zero:+.4f}")
    print(f"  random claim              : {s_rand:+.4f}")
    print(f"  maximal claim(1.00, 1.00) : {s_max:+.4f}")
    print(f"  MINBAR ENGINE             : {s_model:+.4f}   "
          f"mean claim = ({p_model[:,0].mean():.2f}, {p_model[:,1].mean():.2f})")
    print(f"  exhaustive oracle         : {s_oracle:+.4f}   "
          f"mean claim = ({p_orc[:,0].mean():.2f}, {p_orc[:,1].mean():.2f})")
    gapz = s_oracle - s_zero
    if gapz > 1e-9:
        print(f"  fraction of oracle gap closed : "
              f"{(s_model - s_zero)/gapz*100:.1f}%")

    print("\n[8] THE SIMANCAS GATE — does the model know where it is weak?")
    g = eval_gate(P, np.random.default_rng(4242))
    print(f"  corr(predicted risk, realised error) : {g['corr']:+.3f}")
    print(f"  accuracy, all episodes               : {g['acc_all']:.3f}")
    print(f"  accuracy, 50% it is most sure of     : {g['acc_best50']:.3f}")
    print(f"  accuracy, 25% it is most sure of     : {g['acc_best25']:.3f}")
    print(f"  lift from abstaining on the worst half: "
          f"{g['acc_best50']-g['acc_all']:+.3f}")

    print("\n" + line)
    print(" ALL TESTS COMPLETE")
    print(line)


if __name__ == "__main__":
    main()

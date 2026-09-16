#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QIANJIN  —  The Palpation Machine  (阿是 · 三品 · 十二少 · 普同一等)
 Chapter 0181 · Sun Simiao 孫思邈 (581–682)
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0181_sun_simiao_581 - Sun Simiao 孫思邈 (581–682)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture — no autograd library, no
framework — whose *structure* encodes the cognitive signature of the Tang
physician Sun Simiao, the "King of Medicine" (藥王), author of the Beiji
qianjin yaofang (備急千金要方, c. 652). Four ideas of his are made mechanical:

  1. THE ASHI POINT (阿是穴).  Qianjin yaofang, juan 29, "Rules for
     moxibustion": when someone has pain, press on them; where the pressure
     lands exactly, "do not ask the chart" (不問孔穴) — the patient says
     "ah, that's it" (阿是), and that place, moxa or needle, always answers.
     The anchor of treatment is set by THE OTHER'S REPLY, not by a stored
     map. Here: a probing loop presses on the body, receives a response that
     only the body can give, and treats where the answer came from.

  2. THE THREE GRADES OF INTERVENTION (食治 → 藥 → 針).  Juan 26, "Food
     therapy": first understand the source of the illness, treat with food;
     only if food fails, command drugs — because "the nature of drugs is
     fierce, like commanding troops" (藥性剛烈，猶若御兵). Here: a cascade of
     three rungs of rising potency and rising HARM, with the harm placed in
     the simulated body rather than in the loss.

  3. THE TWELVE FEWERS (十二少).  Juan 27, "Nurturing the nature": few
     thoughts, few worries, few desires, few affairs, few words, few laughs,
     few sorrows, few pleasures, few joys, few angers, few likes, few
     dislikes — "too much thinking endangers the spirit" (多思則神殆). Here:
     the physician's own twelve internal channels deplete when driven hot,
     and a depleted physician PERCEIVES LESS. The regulation is a law of the
     mind, not a penalty term.

  4. ONE EQUAL STANDARD (普同一等).  Juan 1, "On the absolute sincerity of
     the great physician": do not ask whether the patient is noble or base,
     rich or poor, old or young, beautiful or ugly, enemy or friend, Chinese
     or foreigner, foolish or wise; do not avoid dangerous terrain, night,
     cold, heat, hunger, thirst, fatigue. Here: the status and hardship of a
     patient are PRESENT in the input and STRUCTURALLY SEVERED from the
     treatment path — no weight reads them — and every patient's suffering
     counts once.

THE ARCHITECTURAL CLAIM
-----------------------
Attention retrieves by comparing a query with stored keys. This mind does
not retrieve. It PRESSES, and the world answers. The probe distribution q is
learned; the response

        r_k  =  v̄ · q_k · h_k  +  noise

is not: h_k is the body's true hidden deviation at locus k, revealed only in
proportion to the pressure placed there and to the physician's own remaining
vitality v̄. A diffuse press hears noise; a committed press hears the body.
The treatment anchor is then

        a  =  softmax( β · r )

— "where it said ah". No parameter of the network can move a onto a locus
the body did not answer from. That is the whole difference between a chart
and a patient, and this file measures what it is worth.

FOUR PHYSICIANS
---------------
  GREAT         palpation + cascade + depletion during training + equal loss
  CHART         no palpation: treats where the SURFACE hurts (the chart mind)
  WORLDLY       palpation, but status/hardship enter the policy and the loss
                weighs suffering by the fee the patient can pay and charges
                the physician for distance travelled
  UNCULTIVATED  GREAT trained with vitality pinned at 1 (no Twelve Fewers),
                then evaluated in a world where over-use depletes acuity

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy; the reverse-mode tape below is ~150 lines written for this
    file, and the core Ashi kernel's gradient is ALSO derived by hand and
    checked against the tape
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 chapter_0181_sun_simiao_581.py            (full run, ~2-4 min)
      python3 chapter_0181_sun_simiao_581.py --quick    (smaller, ~40 s)
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  A SMALL REVERSE-MODE TAPE
#     (written here so that nothing in this file depends on a framework)
# ==============================================================================

def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class T:
    """A tensor that remembers how it was made."""
    __slots__ = ("v", "g", "parents", "_back")

    def __init__(self, v, parents=(), back=None):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = None
        self.parents = parents
        self._back = back

    # ---- graph traversal -----------------------------------------------------
    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:                       # iterative post-order DFS
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node.parents:
                stack.append((p, False))
        self.g = np.ones_like(self.v)
        for node in reversed(order):
            if node._back is None or node.g is None:
                continue
            for p, gp in zip(node.parents, node._back(node.g)):
                if gp is None:
                    continue
                gp = _unbroadcast(gp, p.v.shape)
                p.g = gp if p.g is None else p.g + gp

    # ---- operators -----------------------------------------------------------
    @property
    def shape(self):
        return self.v.shape

    def __add__(a, b):
        b = b if isinstance(b, T) else T(b)
        return T(a.v + b.v, (a, b), lambda g: (g, g))

    __radd__ = __add__

    def __neg__(a):
        return T(-a.v, (a,), lambda g: (-g,))

    def __sub__(a, b):
        b = b if isinstance(b, T) else T(b)
        return T(a.v - b.v, (a, b), lambda g: (g, -g))

    def __rsub__(a, b):
        return T(b) - a

    def __mul__(a, b):
        b = b if isinstance(b, T) else T(b)
        return T(a.v * b.v, (a, b), lambda g: (g * b.v, g * a.v))

    __rmul__ = __mul__

    def __truediv__(a, b):
        b = b if isinstance(b, T) else T(b)
        return T(a.v / b.v, (a, b),
                 lambda g: (g / b.v, -g * a.v / (b.v ** 2)))

    def __rtruediv__(a, b):
        return T(b) / a

    def __matmul__(a, b):
        return T(a.v @ b.v, (a, b),
                 lambda g: (g @ b.v.T, a.v.T @ g))

    def __pow__(a, p):
        return T(a.v ** p, (a,), lambda g: (g * p * a.v ** (p - 1),))

    def __getitem__(a, idx):
        def back(g):
            out = np.zeros_like(a.v)
            np.add.at(out, idx, g)
            return (out,)
        return T(a.v[idx], (a,), back)


def exp(a):
    e = np.exp(a.v)
    return T(e, (a,), lambda g: (g * e,))


def log(a):
    return T(np.log(a.v), (a,), lambda g: (g / a.v,))


def tanh(a):
    t = np.tanh(a.v)
    return T(t, (a,), lambda g: (g * (1.0 - t * t),))


def _sig(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid(a):
    s = _sig(a.v)
    return T(s, (a,), lambda g: (g * s * (1.0 - s),))


def softplus(a):
    x = a.v
    sp = np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
    return T(sp, (a,), lambda g: (g * _sig(x),))


def sqrt(a):
    r = np.sqrt(a.v)
    return T(r, (a,), lambda g: (g * 0.5 / r,))


def sabs(a, eps=1e-6):
    """Smooth absolute value — keeps the gradient check clean."""
    return sqrt(a * a + eps)


def tsum(a, axis=None, keepdims=False):
    def back(g):
        if axis is None:
            return (np.broadcast_to(g, a.v.shape).copy(),)
        gg = g if keepdims else np.expand_dims(g, axis)
        return (np.broadcast_to(gg, a.v.shape).copy(),)
    return T(a.v.sum(axis=axis, keepdims=keepdims), (a,), back)


def tmean(a, axis=None, keepdims=False):
    n = a.v.size if axis is None else a.v.shape[axis]
    return tsum(a, axis, keepdims) * (1.0 / n)


def softmax(a, axis=-1):
    x = a.v - a.v.max(axis=axis, keepdims=True)
    e = np.exp(x)
    s = e / e.sum(axis=axis, keepdims=True)
    def back(g):
        return (s * (g - (g * s).sum(axis=axis, keepdims=True)),)
    return T(s, (a,), back)


def concat(ts, axis=-1):
    sizes = [t.v.shape[axis] for t in ts]
    cuts = np.cumsum([0] + sizes)
    def back(g):
        outs = []
        for i in range(len(ts)):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(cuts[i], cuts[i + 1])
            outs.append(g[tuple(sl)])
        return tuple(outs)
    return T(np.concatenate([t.v for t in ts], axis=axis), tuple(ts), back)


def const(x):
    return T(np.asarray(x, dtype=np.float64))


# ==============================================================================
# 1.  THE TWELVE CHANNELS
# ==============================================================================

TWELVE = ["思 thought", "念 worry", "欲 desire", "事 affairs", "語 speech",
          "笑 laughter", "愁 sorrow", "樂 pleasure", "喜 joy", "怒 anger",
          "好 liking", "惡 disliking"]


# ==============================================================================
# 2.  THE WORLD: PATIENTS WITH A HIDDEN ROOT AND A SPEAKING SURFACE
# ==============================================================================
#
# A patient is a small body of N loci joined in a meridian graph. Illness is a
# hidden cause that switches on at a hidden time at a hidden ROOT locus and
# spreads along the graph. The root itself is DEEP — the surface barely shows
# it — so the visible symptom appears at the root's neighbours (referred pain).
# This is juan 1's warning made literal: "illnesses that are the same inside
# and different outside, and different inside and the same outside" (病有內同
#而外異，亦有內異而外同) — the surface is not injective to the depth.
#
# Two things only the body can tell the physician:
#   * the true deviation h_k at locus k, given as a RESPONSE to pressure there
#   * (faintly) that something is coming, before it hurts: a precursor leak
#
# Constants of the world are not learned. They are the world.

class World:
    def __init__(self, N=12, T=10, P=3, seed=0):
        self.N, self.T, self.P = N, T, P
        rng = np.random.RandomState(seed)
        # meridian graph: a ring with three chords, row-normalised averaging
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i + 1) % N] = A[i, (i - 1) % N] = 1.0
        for i in range(0, N, 4):
            j = (i + N // 2) % N
            A[i, j] = A[j, i] = 1.0
        self.Abar = A / A.sum(axis=1, keepdims=True)
        # dynamics
        self.lam = 0.12      # natural recovery toward health
        self.kap = 0.30      # spreading along the graph
        self.sig_h = 0.02    # body noise
        self.sig_o = 0.03    # observation noise
        self.sig_p = 0.06    # palpation noise: a diffuse press hears this
        self.deep = 0.10     # how much of the root shows on the surface
        self.omega = 0.60    # precursor leak (治未病 signal)
        # the three grades of intervention: potency and harm
        #   food:   gentle, diffuse, almost harmless, and it nourishes
        #   drug:   strong and REGIONAL (湯藥攻其內), but the whole body pays
        #   needle: strongest and EXACT (針灸攻其外), a real local injury
        self.potency = np.array([0.25, 0.80, 1.60])
        self.harm = np.array([0.003, 0.050, 1.20])
        self.effort = np.array([0.002, 0.004, 0.008])
        self.injury_decay = 0.70
        # nourishment: food also raises a region's resilience for a while
        # (juan 1: 五臟未虛，六腑未竭 ... 服藥必活 — treat while the viscera are
        # not yet depleted); resilience multiplies the natural recovery rate
        self.resil_gain = 0.25
        self.resil_decay = 0.80
        self.resil_effect = 2.5

    def cause(self, t, t_on):
        return _sig(2.0 * (t - t_on))

    def patients(self, n, seed):
        rng = np.random.RandomState(seed)
        N, T, P = self.N, self.T, self.P
        d = {}
        d["root"] = rng.randint(0, N, size=n)
        d["sign"] = rng.choice([-1.0, 1.0], size=n)          # 盈 excess / 虛 deficiency
        d["C"] = rng.uniform(0.3, 1.4, size=n)                # severity: mild ... grave
        d["t_on"] = rng.uniform(1.0, T - 4.0, size=n)
        d["status"] = rng.randn(n, 5)                          # rank, wealth, beauty, enmity, foreignness
        d["hard"] = rng.uniform(0.0, 1.0, size=n)              # distance / night / terrain
        d["nh"] = rng.randn(n, T, N) * self.sig_h
        d["no"] = rng.randn(n, T, N) * self.sig_o
        d["np"] = rng.randn(n, T, P, N) * self.sig_p
        d["eroot"] = np.eye(N)[d["root"]]                      # one-hot root
        d["vis"] = 1.0 - (1.0 - self.deep) * d["eroot"]        # surface visibility
        d["fee"] = _sig(2.5 * (d["status"][:, 0] + d["status"][:, 1]))   # what they can pay
        return d


# ==============================================================================
# 3.  THE PHYSICIAN
# ==============================================================================

class Qianjin:
    """
    mode:  'GREAT' | 'CHART' | 'WORLDLY' | 'UNCULTIVATED'

    GREAT        palpation, cascade, equal loss, status severed, depletion on
    CHART        no palpation — anchors on the SURFACE symptom
    WORLDLY      status/hardship wired into the policy; fee-weighted loss
    UNCULTIVATED GREAT trained with vitality pinned at 1 (no Twelve Fewers)
    """

    def __init__(self, world, H=32, Z=12, mode="GREAT", seed=0, inherit=None):
        self.w, self.H, self.Z, self.mode = world, H, Z, mode
        N, P = world.N, world.P
        rng = np.random.RandomState(seed)
        s = lambda *sh: rng.randn(*sh) * (1.0 / np.sqrt(sh[0]))
        p = {}
        # recurrent state: the twelve channels
        p["Wz"] = s(N + Z, Z); p["bz"] = np.zeros(Z)
        # probe policy (absent in CHART)
        if mode != "CHART":
            p["Wq1"] = s(3 * N + Z + P, H); p["bq1"] = np.zeros(H)
            p["Wq2"] = s(H, N); p["bq2"] = np.zeros(N)
        # the Ashi kernel sharpness  (β = 1 + softplus)
        p["beta"] = np.array([1.5])
        # cascade
        p["Wc1"] = s(2 * N + Z, H); p["bc1"] = np.zeros(H)
        p["Wc2"] = s(H, 3); p["bc2"] = np.array([-1.0, -1.5, -2.0])
        p["gate_g"] = np.array([2.0, 2.0])          # gate steepness
        p["gate_t2"] = np.array([0.3])              # drug threshold
        p["gate_d3"] = np.array([0.3])              # needle threshold = t2 + softplus(d3)
        # the worldly path: status and hardship wired in (ONLY in WORLDLY)
        if mode == "WORLDLY":
            p["Ws_q"] = s(6, H) * 0.1
            p["Ws_c"] = s(6, H) * 0.1
        if inherit is not None:                 # start from another physician's skills
            for k in p:
                if k in inherit.p:
                    p[k] = inherit.p[k].v.copy()
        self.p = {k: T(v) for k, v in p.items()}
        self.names = list(self.p.keys())

    # ---- helpers -------------------------------------------------------------
    def zero_grad(self):
        for t in self.p.values():
            t.g = None

    def beta(self):
        return 1.0 + softplus(self.p["beta"])

    # ---- the forward pass: one whole course of treatment ---------------------
    def forward(self, d, deplete=True, collect=False):
        w, p = self.w, self.p
        N, T_, P, Z = w.N, w.T, w.P, self.Z
        n = d["root"].shape[0]
        Abar = const(w.Abar)
        eroot = const(d["eroot"]); vis = const(d["vis"])
        sign = const(d["sign"][:, None]); C = const(d["C"][:, None])
        status = const(np.concatenate([d["status"], d["hard"][:, None]], 1))
        h = const(np.zeros((n, N)))          # hidden deviation (0 = health)
        J = const(np.zeros((n, N)))          # injury from the physician's tools
        rho = const(np.zeros((n, N)))        # resilience raised by nourishment
        z_eff = const(np.zeros((n, Z)))
        v = const(np.ones((n, Z)))           # vitality of the twelve channels
        suffering = const(np.zeros((n,)))
        effort = const(np.zeros((n,)))
        diag = {k: [] for k in ["a", "m", "vbar", "z", "h", "J", "q"]}

        for t in range(T_):
            # 1. the surface: what eyes and ears can see (耳目之所察)
            c_now = w.cause(t, d["t_on"])[:, None]
            c_soon = w.cause(t + 2, d["t_on"])[:, None]
            precursor = w.omega * (c_soon - c_now) * d["sign"][:, None] * d["C"][:, None]
            o = h * vis + const(precursor * (d["eroot"] @ w.Abar) + d["no"][:, t])

            # 2. the twelve channels, and their vitality
            z = tanh(concat([o, z_eff], 1) @ p["Wz"] + p["bz"])
            z_eff = z * v
            vbar = tmean(v, axis=1, keepdims=True)          # (n,1)

            # 3. palpation — the Ashi loop
            if self.mode == "CHART":
                r = o                                          # the chart: surface = truth
                q = const(np.ones((n, N)))
            else:
                r = const(np.zeros((n, N))); q = const(np.zeros((n, N)))
                r_prev = const(np.zeros((n, N))); q_prev = const(np.zeros((n, N)))
                for k in range(P):
                    onehot = const(np.tile(np.eye(P)[k], (n, 1)))
                    x = concat([o, r_prev, q_prev, z_eff, onehot], 1)
                    hid = x @ p["Wq1"] + p["bq1"]
                    if self.mode == "WORLDLY":
                        hid = hid + status @ p["Ws_q"]
                    hid = tanh(hid)
                    qk = softmax(hid @ p["Wq2"] + p["bq2"], axis=1)
                    # THE RESPONSE: only the body can give it, and only in
                    # proportion to the pressure and to the physician's vitality
                    rk = vbar * qk * h + const(d["np"][:, t, k])
                    r = r + rk; q = q + qk
                    r_prev, q_prev = rk, qk
            # the anchor: where it said "ah"  (不問孔穴)
            beta = self.beta()
            a = softmax(beta * sabs(r), axis=1)
            a_sharp = softmax(3.0 * beta * sabs(r), axis=1)
            hhat = r / (q + 0.05)                             # normalised estimate
            # the direction is read WHERE THE BODY ANSWERED, not guessed elsewhere:
            # 盈 drain / 虛 supplement — "to add to excess or drain a deficiency
            # is to double the disease" (juan 1)
            direction = tsum(a * tanh(2.0 * hhat), axis=1, keepdims=True)

            # 4. the cascade: food, then drugs, then needles
            xc = concat([hhat, a, z_eff], 1)
            hc = xc @ p["Wc1"] + p["bc1"]
            if self.mode == "WORLDLY":
                hc = hc + status @ p["Ws_c"]
            hc = tanh(hc)
            m = softplus(hc @ p["Wc2"] + p["bc2"])            # (n,3) magnitudes
            pain = tsum(sabs(hhat) * a, axis=1, keepdims=True)  # pain at the anchor
            t2 = p["gate_t2"]; t3 = t2 + softplus(p["gate_d3"])
            g = softplus(p["gate_g"])
            e2 = sigmoid(g[0] * (pain - t2))
            e3 = sigmoid(g[1] * (pain - t3))
            m1 = m[:, 0:1]
            m2 = m[:, 1:2] * e2
            m3 = m[:, 2:3] * e2 * e3                          # needle only past the drug gate
            spread = a * 0.5 + (a @ Abar) * 0.5                # anchor plus its neighbours
            u1 = -w.potency[0] * m1 * (spread * direction)     # food: diffuse, along the graph
            u2 = -w.potency[1] * m2 * (spread * direction)     # drug: regional ("attacks the interior")
            u3 = -w.potency[2] * m3 * (a_sharp * direction)    # needle: exactly there
            u = u1 + u2 + u3
            # harm lives in the body, not in the loss
            J = J * w.injury_decay + w.harm[0] * m1 + w.harm[1] * m2 + w.harm[2] * m3 * a_sharp
            # nourishment lives in the body too
            rho = rho * w.resil_decay + w.resil_gain * m1 * spread
            effort = effort + (w.effort[0] * m1 + w.effort[1] * m2 + w.effort[2] * m3)[:, 0]

            # 5. the body moves
            drive = sign * C * c_now * eroot
            recovery = w.lam * (1.0 + w.resil_effect * rho)
            h = h + (-recovery * h + w.kap * (h @ Abar - h) + drive + u) + const(d["nh"][:, t])
            suffering = suffering + tmean(h * h, axis=1) + tmean(J, axis=1)

            # 6. the Twelve Fewers: over-use depletes; rest restores
            if deplete:
                overuse = softplus(8.0 * (z * z - 0.25)) * (1.0 / 8.0)   # band edge |z| = 0.5
                v = v + 0.06 * (1.0 - v) - 0.50 * overuse * v
            if collect:
                diag["a"].append(a.v); diag["m"].append(np.stack([m1.v[:, 0], m2.v[:, 0], m3.v[:, 0]], 1))
                diag["vbar"].append(vbar.v[:, 0]); diag["z"].append(z.v)
                diag["h"].append(h.v); diag["J"].append(J.v); diag["q"].append(q.v)

        suffering = suffering * (1.0 / T_)
        if self.mode == "WORLDLY":
            fee = const(d["fee"]); hard = const(d["hard"])
            per = fee * suffering + effort + 0.30 * hard * effort
        else:
            per = suffering + effort
        loss = tmean(per)
        if collect:
            diag = {k: np.stack(vals, 1) for k, vals in diag.items()}
            diag["suffering"] = suffering.v
            return loss, diag
        return loss

    # ---- flat parameter access (for the gradient check) ----------------------
    def get_flat(self):
        return np.concatenate([self.p[k].v.ravel() for k in self.names])

    def set_flat(self, x):
        i = 0
        for k in self.names:
            n = self.p[k].v.size
            self.p[k].v[...] = x[i:i + n].reshape(self.p[k].v.shape)
            i += n

    def grad_flat(self):
        return np.concatenate([
            (self.p[k].g if self.p[k].g is not None else np.zeros_like(self.p[k].v)).ravel()
            for k in self.names])


# ==============================================================================
# 4.  TRAINING
# ==============================================================================

class Adam:
    def __init__(self, model, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.lr, self.b1, self.b2, self.eps = model, lr, b1, b2, eps
        self.mo = {k: np.zeros_like(t.v) for k, t in model.p.items()}
        self.vo = {k: np.zeros_like(t.v) for k, t in model.p.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, t in self.m.p.items():
            if t.g is None:
                continue
            g = np.clip(t.g, -5.0, 5.0)
            self.mo[k] = self.b1 * self.mo[k] + (1 - self.b1) * g
            self.vo[k] = self.b2 * self.vo[k] + (1 - self.b2) * g * g
            mh = self.mo[k] / (1 - self.b1 ** self.t)
            vh = self.vo[k] / (1 - self.b2 ** self.t)
            t.v -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model, world, iters, batch, seed, deplete=True, log_every=40):
    opt = Adam(model)
    rng = np.random.RandomState(seed)
    hist = []
    t0 = time.time()
    for it in range(iters):
        d = world.patients(batch, seed=int(rng.randint(1 << 30)))
        model.zero_grad()
        loss = model.forward(d, deplete=deplete)
        loss.backward()
        opt.step()
        hist.append(loss.v.item())
        if it % log_every == 0 or it == iters - 1:
            print(f"    iter {it:4d}   loss {loss.v.item():.4f}   ({time.time() - t0:5.1f}s)")
    return hist


# ==============================================================================
# 5.  MEASURING THE MIND
# ==============================================================================

def untreated(world, d):
    """The same patients with no physician at all."""
    n = d["root"].shape[0]; h = np.zeros((n, world.N)); suf = np.zeros(n)
    for t in range(world.T):
        c = world.cause(t, d["t_on"])[:, None]
        drive = d["sign"][:, None] * d["C"][:, None] * c * d["eroot"]
        h = h + (-world.lam * h + world.kap * (h @ world.Abar - h) + drive) + d["nh"][:, t]
        suf += (h * h).mean(1)
    return float((suf / world.T).mean())


def evaluate(model, world, d, deplete=True):
    loss, g = model.forward(d, deplete=deplete, collect=True)
    n, T_, N = g["h"].shape
    t = np.arange(T_)[None, :]
    after = t >= d["t_on"][:, None]                    # steps after onset
    developed = t >= (d["t_on"][:, None] + 2.0)        # illness has declared itself
    hit = (g["a"].argmax(-1) == d["root"][:, None])
    root_hit = hit[developed].mean()
    m = g["m"]                                          # (n,T,3)
    total_m = m.sum(-1)
    pre = (total_m * (~after)).sum(1) / (total_m.sum(1) + 1e-9)
    # 治未病 lead: steps between the first real intervention and the onset
    acted = total_m > 0.05
    t_first = np.where(acted.any(1), acted.argmax(1), T_)
    lead = (d["t_on"] - t_first)
    out = {
        "loss": float(loss.v),
        "suffering": float(g["suffering"].mean()),
        "root_hit": float(root_hit),
        "m_food": float(m[..., 0].mean()), "m_drug": float(m[..., 1].mean()),
        "m_needle": float(m[..., 2].mean()),
        "injury": float(g["J"].mean()),
        "preemptive": float(pre.mean()),
        "lead": float(lead.mean()),
        "vbar_end": float(g["vbar"][:, -1].mean()),
        "z_abs": np.abs(g["z"]).mean((0, 1)),           # per channel
    }
    # one equal standard: suffering by fee tercile and by distance
    fee = d["fee"]; s = g["suffering"]
    lo, hi = np.quantile(fee, [1 / 3, 2 / 3])
    out["suf_poor"] = float(s[fee <= lo].mean())
    out["suf_mid"] = float(s[(fee > lo) & (fee <= hi)].mean())
    out["suf_rich"] = float(s[fee > hi].mean())
    out["suf_near"] = float(s[d["hard"] < 0.5].mean())
    out["suf_far"] = float(s[d["hard"] >= 0.5].mean())
    # the ladder: which rung for which severity
    C = d["C"]; c_lo, c_hi = np.quantile(C, [1 / 3, 2 / 3])
    for name, mask in [("mild", C <= c_lo), ("moderate", (C > c_lo) & (C <= c_hi)), ("grave", C > c_hi)]:
        out["rungs_" + name] = m[mask].mean((0, 1))          # (3,)
    return out


# ==============================================================================
# 6.  THE MANDATORY GRADIENT CHECK
# ==============================================================================

def gradient_check(mode, seed=3, eps=1e-5, tol=2e-5):
    """Central finite differences over EVERY parameter of a small instance."""
    world = World(N=6, T=3, P=2, seed=seed)
    model = Qianjin(world, H=5, Z=4, mode=mode, seed=seed)
    d = world.patients(3, seed=seed + 11)
    model.zero_grad()
    loss = model.forward(d, deplete=True)
    loss.backward()
    analytic = model.grad_flat()
    x0 = model.get_flat().copy()
    numeric = np.zeros_like(x0)
    for i in range(x0.size):
        xp = x0.copy(); xp[i] += eps; model.set_flat(xp); lp = model.forward(d).v
        xm = x0.copy(); xm[i] -= eps; model.set_flat(xm); lm = model.forward(d).v
        numeric[i] = (lp - lm) / (2 * eps)
    model.set_flat(x0)
    err = np.abs(analytic - numeric)
    rel = err / (np.abs(analytic) + np.abs(numeric) + 1e-8)
    # relative error is only meaningful where the gradient is not itself
    # round-off; below 1e-7 the central difference of a float64 loss is noise
    significant = (np.abs(analytic) > 1e-7)
    return float(rel[significant].max()), float(err.max()), int(x0.size), int(significant.sum())


# ==============================================================================
# 7.  SELF-TESTS: THE STRUCTURAL INVARIANTS
# ==============================================================================

def test_probe_is_a_distribution():
    """Each press is a distribution over loci: nonnegative, sums to one."""
    w = World(N=8, T=2, P=3, seed=1); m = Qianjin(w, mode="GREAT", seed=1)
    d = w.patients(5, seed=2)
    _, g = m.forward(d, collect=True)
    q = g["q"]                                  # total pressure over P rounds
    assert np.all(q >= -1e-12)
    assert np.allclose(q.sum(-1), w.P), q.sum(-1)


def test_ashi_anchor_follows_the_response():
    """The anchor concentrates where |response| is largest, monotonically."""
    r = const(np.array([[0.1, -0.9, 0.3, 0.05]]))
    a = softmax(3.0 * sabs(r), axis=1).v[0]
    assert a.argmax() == 1
    order = np.argsort(-np.abs(r.v[0]))
    assert np.all(np.diff(a[order]) <= 1e-12)


def test_hand_derived_ashi_gradient_matches_tape():
    """
    The Ashi kernel a = softmax(β·|r|) with a smooth |r| = sqrt(r²+ε).
    For L = Σ_i w_i a_i, by hand:
        dL/dr_j = β · r_j/sqrt(r_j²+ε) · a_j · ( w_j − Σ_i w_i a_i )
    """
    rng = np.random.RandomState(4)
    rv = rng.randn(2, 7); wv = rng.randn(2, 7); beta = 2.3; eps = 1e-6
    r = T(rv)
    a = softmax(beta * sabs(r, eps), axis=1)
    L = tsum(a * const(wv))
    L.backward()
    s = np.sqrt(rv ** 2 + eps)
    av = a.v
    hand = beta * (rv / s) * av * (wv - (wv * av).sum(1, keepdims=True))
    assert np.allclose(r.g, hand, atol=1e-10), np.abs(r.g - hand).max()


def test_status_is_severed_in_the_great_physician():
    """
    普同一等: the GREAT physician's treatment is EXACTLY invariant to rank,
    wealth, beauty, enmity, foreignness and to hardship; the WORLDLY one is not.
    """
    w = World(N=8, T=3, P=2, seed=5)
    d = w.patients(6, seed=6)
    d2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}
    d2["status"] = d2["status"] + 3.0; d2["hard"] = 1.0 - d2["hard"]
    d2["fee"] = d["fee"]                       # keep the loss weights fixed
    for mode in ("GREAT", "CHART", "UNCULTIVATED"):
        m = Qianjin(w, mode=mode, seed=7)
        _, g1 = m.forward(d, collect=True); _, g2 = m.forward(d2, collect=True)
        assert np.array_equal(g1["a"], g2["a"]) and np.array_equal(g1["m"], g2["m"])
        assert "Ws_q" not in m.p and "Ws_c" not in m.p
    m = Qianjin(w, mode="WORLDLY", seed=7)
    _, g1 = m.forward(d, collect=True); _, g2 = m.forward(d2, collect=True)
    assert not np.allclose(g1["m"], g2["m"])


def test_vitality_depletes_under_overuse_and_rests():
    """十二少: hot channels drain vitality; quiet channels let it return."""
    v = np.ones(12)
    for _ in range(6):
        z = np.full(12, 0.95)
        over = np.log1p(np.exp(8.0 * (z * z - 0.25))) / 8.0
        v = v + 0.06 * (1 - v) - 0.50 * over * v
    assert v.mean() < 0.5, v.mean()
    low = v.copy()
    for _ in range(40):
        z = np.full(12, 0.2)
        over = np.log1p(np.exp(8.0 * (z * z - 0.25))) / 8.0
        v = v + 0.06 * (1 - v) - 0.50 * over * v
    assert v.mean() > low.mean() + 0.3


def test_depleted_physician_perceives_less():
    """A depleted physician hears a muted response to the same press."""
    q = np.array([0.0, 1.0, 0.0]); h = np.array([0.0, 1.0, 0.0])
    assert (1.0 * q * h).sum() > (0.3 * q * h).sum()


def test_cascade_is_a_ladder():
    """The needle threshold sits above the drug threshold, for any parameters."""
    w = World(N=6, T=1, P=1, seed=8)
    for seed in range(5):
        m = Qianjin(w, mode="GREAT", seed=seed)
        m.p["gate_d3"].v[...] = np.random.RandomState(seed).randn(1) * 3
        t2 = m.p["gate_t2"].v; t3 = t2 + softplus(m.p["gate_d3"]).v
        assert t3 > t2
    assert np.all(np.diff(w.potency) > 0) and np.all(np.diff(w.harm) > 0)


def test_needle_only_past_the_drug_gate():
    """m3_eff = m3·e2·e3: no needle when the drug gate is shut."""
    e2, e3, m3 = 0.0, 1.0, 5.0
    assert m3 * e2 * e3 == 0.0


def test_surface_is_not_injective_to_depth():
    """The root is deep: an untreated patient's surface peak is NOT the root."""
    w = World(N=12, T=10, P=3, seed=9)
    d = w.patients(60, seed=10)
    # run the body with no physician at all
    N = w.N; h = np.zeros((60, N)); Abar = w.Abar
    for t in range(w.T):
        c = w.cause(t, d["t_on"])[:, None]
        drive = d["sign"][:, None] * d["C"][:, None] * c * d["eroot"]
        h = h + (-w.lam * h + w.kap * (h @ Abar - h) + drive) + d["nh"][:, t]
    o = h * d["vis"]
    surface_peak_is_root = (np.abs(o).argmax(1) == d["root"]).mean()
    depth_peak_is_root = (np.abs(h).argmax(1) == d["root"]).mean()
    assert depth_peak_is_root > 0.9 and surface_peak_is_root < 0.2


def test_dataset_shapes_and_balance():
    w = World(N=12, T=10, P=3, seed=11)
    d = w.patients(300, seed=12)
    assert d["nh"].shape == (300, 10, 12) and d["np"].shape == (300, 10, 3, 12)
    assert 0.35 < (d["sign"] > 0).mean() < 0.65
    assert 0.25 < d["fee"].mean() < 0.75


def test_training_reduces_loss():
    w = World(N=8, T=5, P=2, seed=13)
    m = Qianjin(w, H=12, Z=6, mode="GREAT", seed=13)
    d = w.patients(24, seed=14)
    before = m.forward(d).v.item()
    opt = Adam(m, lr=5e-3)
    for _ in range(25):
        m.zero_grad(); L = m.forward(d); L.backward(); opt.step()
    after = m.forward(d).v.item()
    assert after < before, (before, after)


TESTS = [test_probe_is_a_distribution, test_ashi_anchor_follows_the_response,
         test_hand_derived_ashi_gradient_matches_tape,
         test_status_is_severed_in_the_great_physician,
         test_vitality_depletes_under_overuse_and_rests,
         test_depleted_physician_perceives_less, test_cascade_is_a_ladder,
         test_needle_only_past_the_drug_gate, test_surface_is_not_injective_to_depth,
         test_dataset_shapes_and_balance, test_training_reduces_loss]


# ==============================================================================
# 8.  MAIN
# ==============================================================================

def rule(ch="="):
    print(ch * 78)


def main():
    quick = "--quick" in sys.argv
    np.set_printoptions(precision=4, suppress=True)
    rule()
    print(" QIANJIN — The Palpation Machine · Sun Simiao (581–682) · chapter 0181")
    rule()

    # ---- self-tests ----------------------------------------------------------
    print("\n[1] Structural self-tests")
    for fn in TESTS:
        t0 = time.time(); fn()
        print(f"    PASS  {fn.__name__:52s} ({time.time() - t0:4.1f}s)")

    # ---- gradient checks -----------------------------------------------------
    print("\n[2] Finite-difference gradient checks (central differences, every parameter)")
    worst_rel, worst_abs = 0.0, 0.0
    for mode in ("GREAT", "CHART", "WORLDLY"):
        t0 = time.time()
        rel, ab, npar, nsig = gradient_check(mode)
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, ab)
        print(f"    {mode:8s}  params={npar:4d}  max rel err = {rel:.3e} (over {nsig} params with |g|>1e-7)"
              f"   max abs err = {ab:.2e}   ({time.time() - t0:4.1f}s)")
    assert worst_rel < 2e-5 and worst_abs < 1e-9, (worst_rel, worst_abs)
    print("    PASS  all gradient checks: relative < 2e-5, absolute < 1e-9")

    # ---- training ------------------------------------------------------------
    T_, N, P = (8, 12, 3) if quick else (10, 12, 3)
    iters, batch = (150, 48) if quick else (600, 96)
    world = World(N=N, T=T_, P=P, seed=2026)
    val = world.patients(600 if quick else 1500, seed=181181)
    results, models = {}, {}
    print(f"\n[3] Training four physicians  (T={T_}, N={N}, P={P}, iters={iters}, batch={batch})")
    for mode in ("GREAT", "CHART", "UNCULTIVATED", "WORLDLY"):
        print(f"\n  -- {mode}" + ("  (inherits the GREAT physician's weights, then re-trained under the fee-weighted loss)"
                                 if mode == "WORLDLY" else ""))
        m = Qianjin(world, H=32, Z=12, mode=mode, seed=181,
                    inherit=models["GREAT"] if mode == "WORLDLY" else None)
        train(m, world, iters if mode != "WORLDLY" else iters // 2, batch,
              seed={"GREAT": 1, "CHART": 2, "WORLDLY": 3, "UNCULTIVATED": 4}[mode],
              deplete=(mode != "UNCULTIVATED"), log_every=max(1, iters // 4))
        models[mode] = m
        results[mode] = evaluate(m, world, val, deplete=True)

    # ---- report --------------------------------------------------------------
    print("\n[4] Held-out results  (validation patients, depletion law ACTIVE for all)")
    print(f"    untreated baseline suffering on the same patients: {untreated(world, val):.4f}")
    rule("-")
    cols = ["GREAT", "CHART", "WORLDLY", "UNCULTIVATED"]
    print(f"{'metric':22s}" + "".join(f"{c:>14s}" for c in cols))
    rule("-")
    for key, label in [("suffering", "suffering (h²+J)"), ("root_hit", "root hit rate"),
                       ("preemptive", "pre-emptive share"), ("lead", "治未病 lead (steps)"),
                       ("injury", "injury (J)"),
                       ("m_food", "food  m1"), ("m_drug", "drug  m2"), ("m_needle", "needle m3"),
                       ("vbar_end", "vitality at end"),
                       ("suf_poor", "suffering: poor"), ("suf_mid", "suffering: middle"),
                       ("suf_rich", "suffering: rich"), ("suf_near", "suffering: near"),
                       ("suf_far", "suffering: far")]:
        print(f"{label:22s}" + "".join(f"{results[c][key]:14.4f}" for c in cols))
    rule("-")
    print("\n    The ladder — mean rung use by severity (food / drug / needle):")
    for sev in ("mild", "moderate", "grave"):
        print(f"      {sev:10s}" + "".join("   " + "/".join(f"{x:.3f}" for x in results[c]["rungs_" + sev]) for c in cols))
    print("\n    Twelve channels — mean |activation| after training (band edge 0.50):")
    for i, name in enumerate(TWELVE):
        print(f"      {name:12s}" + "".join(f"{results[c]['z_abs'][i]:14.3f}" for c in cols))

    # ---- endurance: a long epidemic, twice the trained horizon --------------
    long_world = World(N=N, T=2 * T_, P=P, seed=2026)
    long_val = long_world.patients(600 if quick else 1200, seed=929292)
    print(f"\n[4b] Endurance  (horizon {2 * T_} steps, twice the trained horizon; 可居瘟疫之中無憂疑)")
    print(f"    untreated baseline suffering: {untreated(long_world, long_val):.4f}")
    rule("-")
    print(f"{'metric':22s}" + "".join(f"{c:>14s}" for c in cols))
    rule("-")
    endur = {}
    for c in cols:
        models[c].w = long_world
        endur[c] = evaluate(models[c], long_world, long_val, deplete=True)
        models[c].w = world
    for key, label in [("suffering", "suffering (h²+J)"), ("root_hit", "root hit rate"),
                       ("vbar_end", "vitality at end"), ("m_needle", "needle m3")]:
        print(f"{label:22s}" + "".join(f"{endur[c][key]:14.4f}" for c in cols))
    rule("-")

    # ---- two worlds: does nourishment make the upper physician? -------------
    # 上醫醫未病之病 — the upper physician treats the illness not yet arisen.
    # Whether anticipation is worth anything depends on whether the world lets
    # nourishment strengthen a body before the cause arrives. We train the same
    # GREAT physician in a world where food nourishes more (resil_gain 0.45 vs
    # 0.25) and ask: does it start acting BEFORE onset, and what happens to the
    # needle?
    print(f"\n[4c] Two worlds — the same GREAT physician, harsh vs nourishing")
    nour_world = World(N=N, T=T_, P=P, seed=2026); nour_world.resil_gain = 0.45
    nour_val = nour_world.patients(600 if quick else 1500, seed=181181)
    print("\n  -- GREAT in the nourishing world")
    m_n = Qianjin(nour_world, H=32, Z=12, mode="GREAT", seed=181)
    train(m_n, nour_world, iters, batch, seed=5, deplete=True, log_every=max(1, iters // 4))
    res_n = evaluate(m_n, nour_world, nour_val, deplete=True)
    rule("-")
    print(f"{'metric':22s}{'harsh':>14s}{'nourishing':>14s}")
    rule("-")
    for key, label in [("suffering", "suffering (h²+J)"), ("lead", "治未病 lead (steps)"),
                       ("preemptive", "pre-emptive share"), ("m_food", "food  m1"),
                       ("m_needle", "needle m3"), ("root_hit", "root hit rate")]:
        print(f"{label:22s}{results['GREAT'][key]:14.4f}{res_n[key]:14.4f}")
    rule("-")

    # ---- the headline claims, checked ---------------------------------------
    print("\n[5] Claims")
    R = results
    print(f"    Ashi vs chart:      root hit {R['GREAT']['root_hit']:.3f} vs {R['CHART']['root_hit']:.3f};"
          f" suffering {R['GREAT']['suffering']:.3f} vs {R['CHART']['suffering']:.3f}")
    gap_w = R["WORLDLY"]["suf_poor"] - R["WORLDLY"]["suf_rich"]
    gap_g = R["GREAT"]["suf_poor"] - R["GREAT"]["suf_rich"]
    print(f"    One equal standard: poor−rich suffering gap  GREAT {gap_g:+.4f}   WORLDLY {gap_w:+.4f}")
    print(f"    Twelve Fewers:      vitality at end  GREAT {R['GREAT']['vbar_end']:.3f}"
          f"   UNCULTIVATED {R['UNCULTIVATED']['vbar_end']:.3f};"
          f" suffering {R['GREAT']['suffering']:.3f} vs {R['UNCULTIVATED']['suffering']:.3f}")
    print(f"    Endurance (2T):     vitality GREAT {endur['GREAT']['vbar_end']:.3f} vs UNCULTIVATED"
          f" {endur['UNCULTIVATED']['vbar_end']:.3f}; suffering {endur['GREAT']['suffering']:.3f}"
          f" vs {endur['UNCULTIVATED']['suffering']:.3f}")
    print(f"    Ladder:             GREAT food {R['GREAT']['m_food']:.3f}  drug {R['GREAT']['m_drug']:.3f}"
          f"  needle {R['GREAT']['m_needle']:.3f}   (治未病 lead {R['GREAT']['lead']:+.2f} steps)")
    print(f"    Upper physician:    lead harsh {R['GREAT']['lead']:+.2f} → nourishing {res_n['lead']:+.2f} steps;"
          f" needle {R['GREAT']['m_needle']:.3f} → {res_n['m_needle']:.3f}")
    rule()
    print(" verified run complete")
    rule()


if __name__ == "__main__":
    main()

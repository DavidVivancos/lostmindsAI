#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE VIGIL REGISTER  —  a memory in which the dead do not update
 Chapter 0180 · al-Khansāʾ (Tumāḍir bint ʿAmr al-Sulamiyya), c. 575 – c. 645
 Encyclopedia of Lost Minds: Echoes on AI
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch NumPy architecture (no framework, no external autograd; the
reverse-mode differentiation engine is written in this file, ~150 lines) whose
STRUCTURE encodes the one cognitive commitment that is al-Khansāʾ's and nobody
else's in the corpus:

    The dead are the only entities whose representation must have zero
    plasticity. The living drift and are re-estimated; the dead are frozen at
    the value they had when the news arrived, and they are RE-PRESENTED to the
    present mind on the clock of the world — "the rising sun reminds me of
    Ṣakhr, and I remember him at every setting of the sun" — so that they stay
    eligible as exemplars. The wound that the news opened is not a memory and
    does not fade with the memory: it is a LEDGER ENTRY that stays open until
    an act closes it (vengeance in the Jāhiliyya; revaluation as martyrdom
    after Islam). What keeps such a mind alive is not forgetting but
    TAʾASSĪ — consolation by comparison: "were it not for the many weeping
    around me for their brothers, I would have killed myself".

Every one of those clauses is a line of the Dīwān, and every one of them is a
line of code below. The file builds three minds that differ ONLY in the
register, trains them on the same synthetic tribe, and measures what each
commitment buys and what it costs.

THE THREE CONFIGURATIONS
------------------------
  HABITUATION  "time heals". No register. Absence is treated uniformly: every
               absent slot leaks toward the tribal prior at one learned rate.
               The dead are simply members who are never seen again.

  VIGIL        al-Khansāʾ. A per-slot register r_k flips on the day the news
               arrives and freezes the slot's plasticity to zero forever. A
               wound w_k opens with a magnitude equal to the learned WORTH of
               the frozen engram and closes only on a discharge event. At the
               phases of the day the mind chooses (a learned gate over
               dawn / midday / dusk / night), dead members are RE-PRESENTED
               into the attention budget of the present, in proportion to the
               size of their wound.

  TAʾASSĪ      VIGIL plus consolation-by-comparison: the intensity with which
               each dead member is re-presented is normalised by the sum of
               all open wounds in the household AND the grief of the tribe
               around it (an exogenous signal). The total re-presentation of
               the dead can never exceed one unit of attention, however many
               died at once.

WHY THE DIFFERENCES ARE STRUCTURAL AND NOT A TRAINING ARTIFACT
--------------------------------------------------------------
The environment is built so that the correct update rule for the LIVING is a
leak: living members' traits regress toward the tribal mean between sightings,
so an absent living member is best estimated by decaying the last sighting.
The correct rule for the DEAD is the opposite: their traits stopped moving on
the day they died. A memory with one leak rate for "absent" must choose; a
memory with a death register does not. That is the whole argument, and the
attribute-recall numbers for dead vs living slots measure it directly.

The re-presentation mechanism is tested through an ATTENTION BUDGET: each step
the members that are "in mind" share a bounded budget, and downstream decisions
(who is the exemplar for this need? who should lead?) are made on
decision-effective traits that collapse to the tribal prior for anyone who is
not in mind. A dead member that is never re-presented falls out of mind and can
no longer be an exemplar even if the record still holds him. A dead member that
is re-presented without limit crowds the living out of the budget. TAʾASSĪ is
the bound that resolves the two.

The ledger is tested through OPEN-ACCOUNT and INCITEMENT targets: whether a
given dead member's account is still open at the end of the episode, and whether
the household should be moved to act at dawn while any account is open. A mind
without a ledger has to infer both from a fading trace and cannot.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy; the differentiation engine (a reverse-mode tape) is in this
    file and is itself gradient-checked
  * a finite-difference gradient check over EVERY parameter of EVERY
    configuration that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 0180_al_khansa_575_Neuron.py            (full run, ~4-5 min)
      python3 0180_al_khansa_575_Neuron.py --quick    (smoke test, ~40 s)
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# 0.  A REVERSE-MODE TAPE, FROM SCRATCH
# ==============================================================================
# Every array that participates in the loss is wrapped in a Node. Each op
# records its inputs and a closure that pushes the upstream gradient to them.
# Broadcasting is handled by summing the gradient back to the input's shape.


def _unbroadcast(g, shape):
    """Sum a gradient down to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Node:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, prev=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._backward = None
        self._prev = tuple(prev)
        self.requires_grad = requires_grad

    # ---- shape helpers ------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        g = _unbroadcast(np.asarray(g, dtype=np.float64), self.data.shape)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad = self.grad + g

    # ---- arithmetic ---------------------------------------------------------
    def __add__(self, o):
        o = _wrap(o)
        out = Node(self.data + o.data, (self, o))

        def bw():
            self._acc(out.grad)
            o._acc(out.grad)
        out._backward = bw
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Node(-self.data, (self,))

        def bw():
            self._acc(-out.grad)
        out._backward = bw
        return out

    def __sub__(self, o):
        return self + (-_wrap(o))

    def __rsub__(self, o):
        return _wrap(o) + (-self)

    def __mul__(self, o):
        o = _wrap(o)
        out = Node(self.data * o.data, (self, o))

        def bw():
            self._acc(out.grad * o.data)
            o._acc(out.grad * self.data)
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _wrap(o)
        out = Node(self.data / o.data, (self, o))

        def bw():
            self._acc(out.grad / o.data)
            o._acc(-out.grad * self.data / (o.data * o.data))
        out._backward = bw
        return out

    def __rtruediv__(self, o):
        return _wrap(o) / self

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Node(self.data ** p, (self,))

        def bw():
            self._acc(out.grad * p * self.data ** (p - 1))
        out._backward = bw
        return out

    def __matmul__(self, o):
        o = _wrap(o)
        out = Node(np.matmul(self.data, o.data), (self, o))

        def bw():
            g = out.grad
            a, b = self.data, o.data
            if b.ndim == 1:                       # (..., n) @ (n,) -> (...)
                self._acc(np.expand_dims(g, -1) * b)
                o._acc((np.expand_dims(g, -1) * a).reshape(-1, b.shape[0]).sum(0))
            else:
                self._acc(np.matmul(g, np.swapaxes(b, -1, -2)))
                o._acc(np.matmul(np.swapaxes(a, -1, -2), g))
        out._backward = bw
        return out

    # ---- indexing / reshaping ---------------------------------------------
    def __getitem__(self, idx):
        out = Node(self.data[idx], (self,))

        def bw():
            self._acc(_scatter(self.data.shape, idx, out.grad))
        out._backward = bw
        return out

    def reshape(self, *shape):
        out = Node(self.data.reshape(*shape), (self,))

        def bw():
            self._acc(out.grad.reshape(self.data.shape))
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape))
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    # ---- nonlinearities ----------------------------------------------------
    def exp(self):
        out = Node(np.exp(self.data), (self,))

        def bw():
            self._acc(out.grad * out.data)
        out._backward = bw
        return out

    def log(self):
        out = Node(np.log(self.data), (self,))

        def bw():
            self._acc(out.grad / self.data)
        out._backward = bw
        return out

    def tanh(self):
        out = Node(np.tanh(self.data), (self,))

        def bw():
            self._acc(out.grad * (1.0 - out.data ** 2))
        out._backward = bw
        return out

    def sigmoid(self):
        s = _sigmoid_np(self.data)
        out = Node(s, (self,))

        def bw():
            self._acc(out.grad * s * (1.0 - s))
        out._backward = bw
        return out

    def softplus(self):
        out = Node(np.logaddexp(0.0, self.data), (self,))

        def bw():
            self._acc(out.grad * _sigmoid_np(self.data))
        out._backward = bw
        return out

    # ---- backward driver ---------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def visit(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                visit(p)
            topo.append(v)
        visit(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v._backward is not None and v.grad is not None:
                v._backward()


def _scatter(shape, idx, g):
    full = np.zeros(shape)
    np.add.at(full, idx, g)
    return full


def _wrap(x):
    return x if isinstance(x, Node) else Node(x, requires_grad=False)


def const(x):
    return Node(np.asarray(x, dtype=np.float64), requires_grad=False)


def _sigmoid_np(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def concat(nodes, axis=-1):
    """Concatenate Nodes along an axis (gradient is sliced back)."""
    datas = [n.data for n in nodes]
    out = Node(np.concatenate(datas, axis=axis), tuple(nodes))
    sizes = [d.shape[axis] for d in datas]

    def bw():
        start = 0
        for n, s in zip(nodes, sizes):
            sl = [slice(None)] * out.data.ndim
            sl[axis] = slice(start, start + s)
            n._acc(out.grad[tuple(sl)])
            start += s
    out._backward = bw
    return out


def stack(nodes, axis=0):
    return concat([n.reshape(*_insert_axis(n.shape, axis)) for n in nodes], axis=axis)


def _insert_axis(shape, axis):
    s = list(shape)
    if axis < 0:
        axis = len(s) + 1 + axis
    s.insert(axis, 1)
    return tuple(s)


def logsumexp(x, axis=-1):
    """Stable log-sum-exp; the max shift is a constant so the gradient is exact."""
    m = np.max(x.data, axis=axis, keepdims=True)
    return (x - const(m)).exp().sum(axis=axis, keepdims=True).log() + const(m)


def bce_with_logits(z, y):
    """Mean binary cross-entropy from logits (elementwise, stable)."""
    # softplus(z) - z*y  ==  -[ y log s + (1-y) log(1-s) ]
    return (z.softplus() - z * const(y)).mean()


def ce_from_scores(scores, y_idx, mask=None):
    """Mean softmax cross-entropy over the last axis, targets as indices.
    `mask` (optional, {0,1} of shape scores.shape) removes ineligible options."""
    if mask is not None:
        scores = scores + const((1.0 - mask) * (-1e4))
    lse = logsumexp(scores, axis=-1)                      # [B,1]
    B = scores.shape[0]
    picked = scores[np.arange(B), y_idx].reshape(B, 1)    # [B,1]
    return (lse - picked).mean()


# ==============================================================================
# 1.  THE TRIBE  —  a synthetic household with losses
# ==============================================================================
# One episode = one household of K members observed over T "watches"
# (4 per day: dawn, midday, dusk, night) for 12 days.
#
#   traits      a_k(t) in R^d : a living member's traits are an AR(1) process
#                               around the tribal mean (people change, and what
#                               they were last month is a fading guide to what
#                               they are now); the dead are frozen at the value
#                               they had when the news came
#   presence    pres[t,k]     : a living member is seen with prob P_SEE; when
#                               seen, a noisy trait vector is observed
#   news        news[t,k]     : 1 on the watch the death of k is announced
#   discharge   disch[t,k]    : 1 on the watch k's account is closed (vengeance
#                               taken, or the loss revalued as martyrdom)
#   tribe grief G[t]          : exogenous — how much the OTHER households
#                               around this one are mourning (spikes after a
#                               battle, then fades)
#
# Targets at the last watch:
#   y_attr[k,:]  true traits (current for living, frozen for dead)
#   need         which trait the situation calls for (one-hot)
#   y_ex         index of the member (living OR dead) with the highest value
#                of that trait — the exemplar ("Ṣakhr, a beacon for guides")
#   y_ld         index among the LIVING with the highest value — who leads
#   y_open[k]    account open (dead and not discharged) at T
#   y_raid[t]    1 iff some account is open at watch t AND it is dawn
#                (the raid hour, when the elegy's incitement is meant to land)

K, D, P, T = 6, 6, 4, 48
P_SEE, ETA, SIG_DRIFT, SIG_OBS = 0.55, 0.12, 0.35, 0.10
SD_STAT = SIG_DRIFT / np.sqrt(1.0 - (1.0 - ETA) ** 2)   # stationary spread of a living trait
DELTA_MIND = 0.15   # in-mind trace decay per watch (a fixed working-memory time constant)
PHASES = ["dawn", "midday", "dusk", "night"]
DAWN = 0


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    pres = np.zeros((n, T, K)); x = np.zeros((n, T, K, D))
    news = np.zeros((n, T, K)); disch = np.zeros((n, T, K))
    phase = np.zeros((n, T, P)); G = np.zeros((n, T))
    y_attr = np.zeros((n, K, D)); need = np.zeros((n, D))
    y_ex = np.zeros(n, dtype=int); y_ld = np.zeros(n, dtype=int)
    alive_T = np.zeros((n, K)); y_open = np.zeros((n, K)); y_raid = np.zeros((n, T))
    death_t = np.full((n, K), -1); regime = np.empty(n, dtype=object)

    for i in range(n):
        a = rng.normal(0.0, SD_STAT, size=(K, D))
        alive = np.ones(K, dtype=bool)
        off = rng.integers(0, P)
        # --- who dies, when ------------------------------------------------
        u = rng.random()
        if u < 0.22:
            deaths = []
            regime[i] = "NO_LOSS"
        elif u < 0.57:
            k = rng.integers(0, K)
            deaths = [(int(rng.integers(int(0.15 * T), int(0.85 * T))), k)]
            regime[i] = "SINGLE_LOSS"
        elif u < 0.75:
            ks = rng.choice(K, size=2, replace=False)
            deaths = [(int(rng.integers(int(0.15 * T), int(0.85 * T))), k) for k in ks]
            regime[i] = "TWO_LOSSES"
        else:
            nb = int(rng.integers(3, 5))
            ks = rng.choice(K, size=nb, replace=False)
            tb = int(rng.integers(int(0.15 * T), int(0.7 * T)))
            deaths = [(tb, k) for k in ks]
            regime[i] = "BATTLE"
        death_map = {}
        for (td, k) in deaths:
            death_map.setdefault(td, []).append(k)
        # --- discharges ------------------------------------------------------
        disch_map = {}
        for (td, k) in deaths:
            if rng.random() < 0.5 and td + 4 < T - 1:
                tdis = int(rng.integers(td + 4, T - 1))
                disch_map.setdefault(tdis, []).append(k)
        g = 0.0
        for t in range(T):
            ph = (t + off) % P
            phase[i, t, ph] = 1.0
            # deaths announced at this watch
            for k in death_map.get(t, []):
                alive[k] = False
                news[i, t, k] = 1.0
                death_t[i, k] = t
            n_here = len(death_map.get(t, []))
            if n_here:
                g += n_here * rng.uniform(0.6, 1.4)     # the tribe mourns too
            g *= 0.93
            G[i, t] = g
            for k in disch_map.get(t, []):
                disch[i, t, k] = 1.0
            # sightings of the living
            for k in range(K):
                if alive[k] and rng.random() < P_SEE:
                    pres[i, t, k] = 1.0
                    x[i, t, k] = a[k] + rng.normal(0.0, SIG_OBS, size=D)
            # raid target: an account is open and it is dawn
            open_now = any((death_t[i, k] >= 0) and not disch[i, :t + 1, k].any()
                           for k in range(K))
            y_raid[i, t] = 1.0 if (open_now and ph == DAWN) else 0.0
            # the living drift; the dead do not
            a[alive] += ETA * (0.0 - a[alive]) + rng.normal(0.0, SIG_DRIFT, size=(alive.sum(), D))
        y_attr[i] = a
        alive_T[i] = alive.astype(float)
        j = rng.integers(0, D)
        need[i, j] = 1.0
        y_ex[i] = int(np.argmax(a[:, j]))
        y_ld[i] = int(np.argmax(np.where(alive, a[:, j], -np.inf)))
        for k in range(K):
            y_open[i, k] = 1.0 if (death_t[i, k] >= 0 and not disch[i, :, k].any()) else 0.0
    return dict(pres=pres, x=x, news=news, disch=disch, phase=phase, G=G,
                y_attr=y_attr, need=need, y_ex=y_ex, y_ld=y_ld, alive_T=alive_T,
                y_open=y_open, y_raid=y_raid, death_t=death_t, regime=regime)


def take(data, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in data.items()}


# ==============================================================================
# 2.  THE MODEL  —  engram bank + vigil register + attention budget + ledger
# ==============================================================================

M = 12            # engram code width
KAPPA_W = 0.5     # wound saturation in the re-presentation intensity
KAPPA_RHO = 0.5   # in-mind saturation for decision-effective traits
CONFIGS = ("HABITUATION", "VIGIL", "TAASSI")


class VigilRegister:
    """
    State per slot k (one per household member):
      e_k   in R^M   the engram (what is remembered of k)
      rho_k >= 0     the in-mind trace (how present k is to the mind now)
      r_k   in {0,1} the vigil register (has the news of k's death arrived)
      w_k   >= 0     the open wound (worth of k at death, until discharged)

    Per watch t:
      z     = tanh(x W_enc + b_enc)                      encode a sighting
      r     = r + news (1 - r)                           the register flips
      w     = (w + news * worth(e)) (1 - disch)          the ledger opens/closes
              worth(e) = 2 sigmoid(MLP(e))  in (0,2)
      lam_k = r_k + (1 - r_k) lam                        frozen if dead
      e     = e + (1-r) pres u (z - e) + (1-pres)(1-lam_k)(mu_e - e)
      psi_k = w_k/(kappa + w_k)                          VIGIL
            = w_k/(kappa + sum_j w_j + G_t)              TAASSI
      raw_k = pres_k + (1 - pres_k) r_k psi_k g(phase)   re-presentation
      a_k   = raw_k / (1 + sum_j raw_j)                  attention budget
      rho   = (1 - DELTA_MIND) rho + a                   in-mind trace
      lament_t = sum_k w_k a_k                           the wound in mind now

    HABITUATION has r == 0 and w == 0 identically: no register, no ledger, no
    re-presentation, one leak for every absence.
    """

    def __init__(self, cfg, seed=0):
        assert cfg in CONFIGS
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        sc = 0.4
        p = {}
        p["W_enc"] = rng.normal(0, sc, (D, M)) / np.sqrt(D)
        p["b_enc"] = np.zeros(M)
        p["W_dec"] = rng.normal(0, sc, (M, D)) / np.sqrt(M)
        p["b_dec"] = np.zeros(D)
        p["mu_e"] = rng.normal(0, 0.1, M)          # resting engram / leak target
        p["th_u"] = np.full(M, 1.5)                # write rate logits   (sig -> .82)
        p["th_lam"] = np.full(M, 2.0)              # retention logits    (sig -> .88)
        p["th_gate"] = np.zeros(P)                 # re-presentation gate per phase
        p["W_v1"] = rng.normal(0, sc, (M, 8)) / np.sqrt(M)
        p["b_v1"] = np.zeros(8)
        p["W_v2"] = rng.normal(0, sc, (8, 1)) / np.sqrt(8)
        p["b_v2"] = np.array([0.5])
        p["mu_a"] = np.zeros(D)                    # decision prior in trait space
        p["s_ex"] = np.array([2.0])
        p["s_ld"] = np.array([2.0])
        p["p_ld"] = np.array([3.0])
        F = M + 3                                  # open-account features
        p["W_o1"] = rng.normal(0, sc, (F, 8)) / np.sqrt(F)
        p["b_o1"] = np.zeros(8)
        p["W_o2"] = rng.normal(0, sc, (8, 1)) / np.sqrt(8)
        p["b_o2"] = np.array([-1.0])
        p["alpha"] = np.array([1.0])
        p["beta"] = np.zeros(P)
        p["gamma"] = np.array([-1.0])
        self.p = p

    # ------------------------------------------------------------------ utils
    def param_names(self):
        return list(self.p.keys())

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def worth(self, e, P):
        """Worth of a member as read from the engram, in (0, 2)."""
        h = (e @ P["W_v1"] + P["b_v1"]).tanh()
        return (2.0 * (h @ P["W_v2"] + P["b_v2"]).sigmoid()).reshape(*e.shape[:-1])

    # ---------------------------------------------------------------- forward
    def forward(self, d, P=None, trace=False):
        """Run the household through T watches. Returns (loss, parts, aux)."""
        if P is None:
            P = {k: Node(v) for k, v in self.p.items()}
        cfg = self.cfg
        B = d["pres"].shape[0]
        u = P["th_u"].sigmoid()                      # [M]
        lam = P["th_lam"].sigmoid()                  # [M]
        delta = DELTA_MIND                            # fixed working-memory constant
        gate = P["th_gate"].sigmoid()                # [P]
        e = P["mu_e"].reshape(1, 1, M) * const(np.ones((B, K, 1)))
        rho = const(np.zeros((B, K)))
        r = const(np.zeros((B, K)))
        w = const(np.zeros((B, K)))
        raid_logits = []
        tr = dict(a=[], rho=[], w=[], lament=[], r=[]) if trace else None

        for t in range(T):
            pres = const(d["pres"][:, t, :])                  # [B,K]
            pres3 = pres.reshape(B, K, 1)
            x = const(d["x"][:, t, :, :])                     # [B,K,D]
            news = const(d["news"][:, t, :])
            disch = const(d["disch"][:, t, :])
            ph = const(d["phase"][:, t, :])                   # [B,P]
            Gt = const(d["G"][:, t]).reshape(B, 1)            # [B,1]

            z = (x @ P["W_enc"] + P["b_enc"]).tanh()          # [B,K,M]
            if cfg != "HABITUATION":
                r = r + news * (1.0 - r)                      # the register flips
                v = self.worth(e, P)                          # [B,K]
                w = (w + news * v) * (1.0 - disch)            # ledger opens / closes
            r3 = r.reshape(B, K, 1)
            lam_k = r3 + (1.0 - r3) * lam                     # frozen if dead
            e = (e + (1.0 - r3) * pres3 * u * (z - e)
                   + (1.0 - pres3) * (1.0 - lam_k) * (P["mu_e"] - e))
            if cfg == "HABITUATION":
                raw = pres
            else:
                g_t = (ph @ gate).reshape(B, 1)               # [B,1]
                if cfg == "VIGIL":
                    psi = w / (KAPPA_W + w)
                else:                                         # TAASSI
                    psi = w / (KAPPA_W + w.sum(axis=1, keepdims=True) + Gt)
                raw = pres + (1.0 - pres) * r * psi * g_t
            a = raw / (1.0 + raw.sum(axis=1, keepdims=True))  # attention budget
            rho = (1.0 - delta) * rho + a
            lament = (w * a).sum(axis=1)                      # [B]
            raid_logits.append(P["alpha"] * lament + ph @ P["beta"] + P["gamma"])
            if trace:
                tr["a"].append(a.data.copy()); tr["rho"].append(rho.data.copy())
                tr["w"].append(w.data.copy()); tr["lament"].append(lament.data.copy())
                tr["r"].append(r.data.copy())

        # ------------------------------------------------------ readouts at T
        rho_t = rho / (rho + KAPPA_RHO)                      # [B,K] in-mind
        attr_hat = e @ P["W_dec"] + P["b_dec"]               # [B,K,D]
        rho3 = rho_t.reshape(B, K, 1)
        e_hat = rho3 * attr_hat + (1.0 - rho3) * P["mu_a"]   # decision-effective
        need = const(d["need"]).reshape(B, 1, D)
        util = (e_hat * need).sum(axis=2)                    # [B,K]
        sc_ex = P["s_ex"] * util
        sc_ld = P["s_ld"] * util - P["p_ld"] * r
        feats = concat([e, rho_t.reshape(B, K, 1), r.reshape(B, K, 1),
                        (w / (1.0 + w)).reshape(B, K, 1)], axis=2)     # [B,K,M+3]
        h = (feats @ P["W_o1"] + P["b_o1"]).tanh()
        open_logit = (h @ P["W_o2"] + P["b_o2"]).reshape(B, K)
        raid = stack(raid_logits, axis=1)                    # [B,T]

        L_attr = ((attr_hat - const(d["y_attr"])) ** 2).mean()
        L_ex = ce_from_scores(sc_ex, d["y_ex"])
        L_ld = ce_from_scores(sc_ld, d["y_ld"], mask=d["alive_T"])
        L_open = bce_with_logits(open_logit, d["y_open"])
        L_raid = bce_with_logits(raid, d["y_raid"])
        loss = L_attr + L_ex + L_ld + L_open + L_raid
        parts = dict(attr=L_attr.data.item(), ex=L_ex.data.item(), ld=L_ld.data.item(),
                     open=L_open.data.item(), raid=L_raid.data.item())
        aux = dict(attr_hat=attr_hat.data, sc_ex=sc_ex.data, sc_ld=sc_ld.data,
                   open_logit=open_logit.data, raid=raid.data, rho_t=rho_t.data,
                   r=r.data, w=w.data, trace=tr)
        return loss, parts, aux

    # ------------------------------------------------------------- gradients
    def loss_and_grads(self, d):
        P = {k: Node(v) for k, v in self.p.items()}
        loss, parts, aux = self.forward(d, P)
        loss.backward()
        grads = {k: (P[k].grad if P[k].grad is not None else np.zeros_like(self.p[k]))
                 for k in self.p}
        return loss.data.item(), parts, grads, aux


# ==============================================================================
# 3.  GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(cfg, seed=3, n=4, eps=1e-5, tol=1e-5, max_entries=None):
    """Central finite differences against the tape, for EVERY parameter entry
    (all ~350-450 of them) of the given configuration on a small batch.
    Relative error is measured against max(|num|+|ana|, 1e-5) so that
    float64 round-off on gradients that are themselves ~1e-5 does not
    masquerade as a derivation error; a wrong gradient is off by O(1)."""
    global T
    T_saved = T
    T = 10                                   # short episodes for the check
    try:
        data = make_dataset(n, seed=seed + 100)
        m = VigilRegister(cfg, seed=seed)
        rng = np.random.default_rng(seed)
        for k in m.p:                        # jitter away from symmetric points
            m.p[k] = m.p[k] + rng.normal(0, 0.05, m.p[k].shape)
        _, _, grads, _ = m.loss_and_grads(data)
        worst, where, count = 0.0, None, 0
        for k in m.p:
            flat = m.p[k].reshape(-1)
            idxs = range(flat.size)
            if max_entries is not None and flat.size > max_entries:
                idxs = rng.choice(flat.size, size=max_entries, replace=False)
            for i in idxs:
                old = flat[i]
                flat[i] = old + eps
                lp, _, _ = m.forward(data)
                flat[i] = old - eps
                lm, _, _ = m.forward(data)
                flat[i] = old
                num = (lp.data.item() - lm.data.item()) / (2 * eps)
                ana = grads[k].reshape(-1)[i]
                rel = abs(num - ana) / max(1e-5, abs(num) + abs(ana))
                count += 1
                if rel > worst:
                    worst, where = rel, f"{k}[{i}] num={num:.3e} ana={ana:.3e}"
        return worst < tol, worst, where, count
    finally:
        T = T_saved


# ==============================================================================
# 4.  TRAINING
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model, dtr, dva, epochs=8, batch=64, lr=0.02, seed=0, log=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.p, lr=lr)
    ntr = dtr["pres"].shape[0]
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(ntr)
        tot, nb = 0.0, 0
        t0 = time.time()
        for s in range(0, ntr, batch):
            idx = perm[s:s + batch]
            loss, parts, grads, _ = model.loss_and_grads(take(dtr, idx))
            opt.step(grads)
            tot += loss; nb += 1
        va, vparts, _ = model.forward(dva)
        hist.append((tot / nb, va.data.item()))
        if log:
            print(f"    epoch {ep + 1:2d}  train {tot / nb:.4f}  val {va.data.item():.4f}"
                  f"  [attr {vparts['attr']:.3f} ex {vparts['ex']:.3f} ld {vparts['ld']:.3f}"
                  f" open {vparts['open']:.3f} raid {vparts['raid']:.3f}]  {time.time() - t0:.1f}s")
    return hist


# ==============================================================================
# 5.  EVALUATION
# ==============================================================================

def evaluate(model, d):
    loss, parts, aux = model.forward(d, trace=True)
    n = d["pres"].shape[0]
    dead = (d["alive_T"] == 0)
    living = ~dead
    err = ((aux["attr_hat"] - d["y_attr"]) ** 2).mean(axis=2)          # [B,K]
    res = {}
    res["attr_mse_living"] = float(err[living].mean()) if living.any() else float("nan")
    res["attr_mse_dead"] = float(err[dead].mean()) if dead.any() else float("nan")
    ex_pred = aux["sc_ex"].argmax(axis=1)
    res["exemplar_acc"] = float((ex_pred == d["y_ex"]).mean())
    tgt_dead = dead[np.arange(n), d["y_ex"]]
    res["exemplar_acc_when_dead"] = float((ex_pred == d["y_ex"])[tgt_dead].mean()) if tgt_dead.any() else float("nan")
    res["exemplar_acc_when_living"] = float((ex_pred == d["y_ex"])[~tgt_dead].mean()) if (~tgt_dead).any() else float("nan")
    ld_sc = np.where(d["alive_T"] > 0, aux["sc_ld"], -1e9)
    res["leader_acc"] = float((ld_sc.argmax(axis=1) == d["y_ld"]).mean())
    op = (aux["open_logit"] > 0).astype(float)
    res["open_acc_dead_slots"] = float((op == d["y_open"])[dead].mean()) if dead.any() else float("nan")
    res["raid_acc"] = float(((aux["raid"] > 0).astype(float) == d["y_raid"]).mean())
    # attention share of the living, averaged over watches after the first death
    A = np.stack(aux["trace"]["a"], axis=1)                              # [B,T,K]
    share = []
    for i in range(n):
        dts = d["death_t"][i][d["death_t"][i] >= 0]
        if dts.size == 0:
            continue
        t0 = int(dts.min())
        liv = d["alive_T"][i] > 0
        tot = A[i, t0:, :].sum(axis=1)
        share.append(float((A[i, t0:, :][:, liv].sum(axis=1) / np.maximum(tot, 1e-9)).mean()))
    res["living_attention_share"] = float(np.mean(share)) if share else float("nan")
    lam = np.stack(aux["trace"]["lament"], axis=1)                       # [B,T]
    res["lament_peak"] = float(lam.max(axis=1).mean())
    res["loss"] = float(loss.data.item())
    return res


REGIMES = ("NO_LOSS", "SINGLE_LOSS", "TWO_LOSSES", "BATTLE")


def evaluate_by_regime(model, d):
    out = {"ALL": evaluate(model, d)}
    for rg in REGIMES:
        idx = np.where(d["regime"] == rg)[0]
        if idx.size:
            out[rg] = evaluate(model, take(d, idx))
    # age of the loss: single-loss households split by when the news came
    dt = np.where(d["death_t"] >= 0, d["death_t"], 10 ** 6).min(axis=1)
    single = d["regime"] == "SINGLE_LOSS"
    for name, sel in (("OLD_LOSS", single & (dt < T // 2)), ("FRESH_LOSS", single & (dt >= T // 2))):
        idx = np.where(sel)[0]
        if idx.size:
            out[name] = evaluate(model, take(d, idx))
    return out


def ledger_trace(model, d, i):
    """Print one household's vigil: register, wounds, attention, lament."""
    _, _, aux = model.forward(take(d, np.array([i])), trace=True)
    tr = aux["trace"]
    A = np.stack(tr["a"], axis=1)[0]; W = np.stack(tr["w"], axis=1)[0]
    R = np.stack(tr["r"], axis=1)[0]; L = np.stack(tr["lament"], axis=1)[0]
    ph = d["phase"][i].argmax(axis=1)
    print(f"    household {i}  regime={d['regime'][i]}  deaths at watches "
          f"{[int(t) for t in d['death_t'][i] if t >= 0]}")
    print("    watch  phase   dead-slots  open-wounds(sum)  attn:living  attn:dead   lament")
    for t in range(0, T, 3):
        dead = R[t] > 0.5
        print(f"    {t:5d}  {PHASES[ph[t]]:<7s} {int(dead.sum()):^11d} "
              f"{W[t].sum():^17.3f} {A[t][~dead].sum():^12.3f} {A[t][dead].sum():^10.3f} {L[t]:8.3f}")


# ==============================================================================
# 6.  SELF-TESTS  —  structural invariants
# ==============================================================================

def _tiny(n=3, seed=11):
    return make_dataset(n, seed=seed)


def test_tape_basic_ops():
    """The tape agrees with finite differences on a composite expression that
    uses every op the model uses (matmul with batch dims, broadcasting, div,
    getitem, concat, stack, logsumexp, softplus, tanh, sigmoid)."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 4, 5)); Wm = rng.normal(size=(5, 2)); b = rng.normal(size=2)
    c = rng.normal(size=(3, 4, 1))

    def f(A, Wm, b, c):
        a, wn, bn, cn = Node(A), Node(Wm), Node(b), Node(c)
        h = (a @ wn + bn).tanh() * cn.sigmoid() / (1.0 + cn ** 2)
        s = concat([h, cn], axis=2).softplus().sum(axis=2)
        z = stack([s[:, 0], s[:, 1] * 2.0], axis=1)
        return (logsumexp(z).reshape(3, 1) + h[:, :, 0]).mean(), (a, wn, bn, cn)
    L, nodes = f(A, Wm, b, c)
    L.backward()
    eps = 1e-6
    for arr, node in zip((A, Wm, b, c), nodes):
        flat = arr.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = f(A, Wm, b, c)[0].data.item()
            flat[i] = old - eps; lm = f(A, Wm, b, c)[0].data.item()
            flat[i] = old
            num = (lp - lm) / (2 * eps); ana = node.grad.reshape(-1)[i]
            assert abs(num - ana) < 1e-6 * (1 + abs(num)), (num, ana)
    return "tape matches finite differences on a composite of all ops"


def test_dead_slot_is_frozen_under_vigil():
    """After the news, a VIGIL slot's engram never changes again — exactly —
    whatever is observed later (even a spurious sighting)."""
    for cfg in ("VIGIL", "TAASSI"):
        m = VigilRegister(cfg, seed=1)
        d = _tiny(2, seed=5)
        # kill member 0 at watch 3 for household 0, then feed it sightings anyway
        d["news"][:] = 0; d["disch"][:] = 0; d["pres"][:] = 0
        d["news"][0, 3, 0] = 1
        d["pres"][0, 6:, 0] = 1
        d["x"][0, 6:, 0, :] = 5.0
        _, _, aux = m.forward(d, trace=True)
        # recompute engram path explicitly: run forward twice with different late inputs
        d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        d2["x"][0, 6:, 0, :] = -5.0
        _, _, aux2 = m.forward(d2, trace=True)
        assert np.allclose(aux["attr_hat"][0, 0], aux2["attr_hat"][0, 0], atol=1e-12)
        assert aux["r"][0, 0] == 1.0
    return "a dead slot's engram is bit-identical whatever arrives afterwards"


def test_habituation_forgets_the_dead():
    """Under HABITUATION a member who is never seen again leaks to the resting
    engram at the same rate as any absent living member."""
    m = VigilRegister("HABITUATION", seed=1)
    d = _tiny(1, seed=5)
    d["news"][:] = 0; d["disch"][:] = 0; d["pres"][:] = 0
    d["pres"][0, 0, 0] = 1; d["x"][0, 0, 0, :] = 3.0
    d["news"][0, 1, 0] = 1
    _, _, aux = m.forward(d)
    rest = np.tanh(m.p["mu_e"] @ m.p["W_dec"] + m.p["b_dec"]) * 0 + (m.p["mu_e"] @ m.p["W_dec"] + m.p["b_dec"])
    lam = _sigmoid_np(m.p["th_lam"])
    # after T-1 absent watches the engram is within lam^(T-1) of the sighting deviation
    assert np.abs(aux["attr_hat"][0, 0] - rest).max() < 0.05
    return "under HABITUATION the dead leak to the resting engram like anyone absent"


def test_wound_opens_and_closes_exactly():
    m = VigilRegister("VIGIL", seed=2)
    d = _tiny(1, seed=7)
    d["news"][:] = 0; d["disch"][:] = 0
    d["news"][0, 4, 2] = 1; d["disch"][0, 20, 2] = 1
    _, _, aux = m.forward(d, trace=True)
    W = np.stack(aux["trace"]["w"], axis=1)[0]
    assert W[3, 2] == 0.0 and W[4, 2] > 0.0
    assert np.all(W[4:20, 2] == W[4, 2])          # constant while open
    assert np.all(W[20:, 2] == 0.0)               # closed exactly
    return "a wound opens at the news with positive worth, holds, and closes to exactly zero"


def test_taassi_bounds_the_dead():
    """Sum of re-presentation intensities of the dead is <= 1 under TAASSI for
    any wounds and any tribe grief; under VIGIL it can exceed 1."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        w = rng.exponential(2.0, size=K) * (rng.random(K) < 0.7)
        G = rng.exponential(1.0)
        psi_t = w / (KAPPA_W + w.sum() + G)
        assert psi_t.sum() <= 1.0 + 1e-12
        psi_v = w / (KAPPA_W + w)
        if (w > 0).sum() >= 3:
            assert psi_v.sum() > 1.0 or w.max() < 0.5
    return "TAASSI keeps the dead's total re-presentation <= 1; VIGIL does not"


def test_attention_budget_is_bounded():
    m = VigilRegister("VIGIL", seed=4)
    m.p["th_gate"][:] = 8.0                       # re-present everywhere, hard
    d = _tiny(3, seed=9)
    _, _, aux = m.forward(d, trace=True)
    A = np.stack(aux["trace"]["a"], axis=1)
    assert np.all(A >= 0) and np.all(A.sum(axis=2) < 1.0)
    return "the per-watch attention budget stays in [0,1) under maximal re-presentation"


def test_habituation_never_represents_the_dead():
    m = VigilRegister("HABITUATION", seed=4)
    d = _tiny(3, seed=9)
    _, _, aux = m.forward(d, trace=True)
    A = np.stack(aux["trace"]["a"], axis=1)
    absent = d["pres"] == 0
    assert np.all(A[absent] == 0.0)
    return "under HABITUATION nothing absent is ever in mind"


def test_dataset_invariants():
    d = make_dataset(200, seed=21)
    # frozen targets: a dead member's target traits equal traits at death
    # (checked indirectly: the generator freezes `a[alive]` only) -> verify that
    # after death nothing is ever observed for that member
    for i in range(200):
        for k in range(K):
            td = d["death_t"][i, k]
            if td >= 0:
                assert d["pres"][i, td:, k].sum() == 0
                assert d["y_open"][i, k] == (0.0 if d["disch"][i, :, k].any() else 1.0)
    assert set(d["regime"]) <= set(REGIMES)
    assert np.all(d["phase"].sum(axis=2) == 1)
    # raid target only at dawn
    assert np.all(d["y_raid"][d["phase"][:, :, DAWN] == 0] == 0)
    # exemplar / leader consistency
    j = d["need"].argmax(axis=1)
    for i in range(200):
        assert d["y_ex"][i] == np.argmax(d["y_attr"][i][:, j[i]])
        assert d["alive_T"][i, d["y_ld"][i]] == 1
    return "dataset: the dead are never seen again, targets are consistent, raids are at dawn"


def test_training_reduces_loss():
    global T
    T_saved = T; T = 12
    try:
        m = VigilRegister("VIGIL", seed=8)
        dtr = make_dataset(96, seed=31); dva = make_dataset(48, seed=32)
        h = train(m, dtr, dva, epochs=4, batch=32, lr=0.03, log=False)
        assert h[-1][0] < h[0][0]
    finally:
        T = T_saved
    return "a short training run reduces the loss"


TESTS = [test_tape_basic_ops, test_dead_slot_is_frozen_under_vigil,
         test_habituation_forgets_the_dead, test_wound_opens_and_closes_exactly,
         test_taassi_bounds_the_dead, test_attention_budget_is_bounded,
         test_habituation_never_represents_the_dead, test_dataset_invariants,
         test_training_reduces_loss]


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def rule(ch="=", n=78):
    print(ch * n)


def main():
    quick = "--quick" in sys.argv
    np.set_printoptions(precision=3, suppress=True)
    rule()
    print(" THE VIGIL REGISTER — al-Khansāʾ (chapter 0180)")
    print(" K=%d members · D=%d traits · %d watches (%d days of %d) · engram width %d"
          % (K, D, T, T // P, P, M))
    rule()

    # ------------------------------------------------------------ self-tests
    print("\n[1] structural self-tests")
    for t in TESTS:
        msg = t()
        print(f"    ok  {t.__name__:<44s} {msg}")

    # -------------------------------------------------------- gradient check
    print("\n[2] finite-difference gradient check (every parameter entry, all configs)")
    for cfg in CONFIGS:
        ok, worst, where, cnt = gradient_check(cfg, max_entries=(60 if quick else None))
        print(f"    {cfg:<12s} entries={cnt:4d}  max rel err={worst:.2e}  {'PASS' if ok else 'FAIL'}  worst at {where}")
        assert ok, f"gradient check failed for {cfg}"

    # -------------------------------------------------------------- training
    ntr, nva, epochs = (600, 300, 3) if quick else (4000, 1000, 12)
    print(f"\n[3] training three minds on the same tribe  (train {ntr}, val {nva}, {epochs} epochs)")
    dtr = make_dataset(ntr, seed=1); dva = make_dataset(nva, seed=2)
    models, results = {}, {}
    for cfg in CONFIGS:
        print(f"  -- {cfg}  ({VigilRegister(cfg).n_params()} parameters)")
        m = VigilRegister(cfg, seed=0)
        train(m, dtr, dva, epochs=epochs, batch=64, lr=0.02, seed=0)
        models[cfg] = m
        results[cfg] = evaluate_by_regime(m, dva)

    # --------------------------------------------------------------- report
    print("\n[4] what each commitment buys and costs  (held-out households)")
    metrics = [("attr_mse_living", "trait recall error · living  (lower better)"),
               ("attr_mse_dead", "trait recall error · dead    (lower better)"),
               ("exemplar_acc_when_dead", "exemplar chosen right when the best is dead"),
               ("exemplar_acc_when_living", "exemplar chosen right when the best is living"),
               ("leader_acc", "leader chosen right among the living"),
               ("open_acc_dead_slots", "open/closed account right (dead slots)"),
               ("raid_acc", "incitement right (raid at dawn iff account open)"),
               ("living_attention_share", "share of attention the living keep after a death"),
               ("lament_peak", "peak lament amplitude per household")]
    print(f"    {'metric':<52s}" + "".join(f"{c:>12s}" for c in CONFIGS))
    for key, label in metrics:
        print(f"    {label:<52s}" + "".join(f"{results[c]['ALL'][key]:12.4f}" for c in CONFIGS))
    print("\n    by regime — exemplar accuracy when the best exemplar is dead / living attention share / leader acc")
    for rg in REGIMES + ("OLD_LOSS", "FRESH_LOSS"):
        row = f"    {rg:<14s}"
        for c in CONFIGS:
            rr = results[c].get(rg)
            if rr is None:
                row += f"{'-':>22s}"
            else:
                row += f"  {rr['exemplar_acc_when_dead']:5.3f}/{rr['living_attention_share']:5.3f}/{rr['leader_acc']:5.3f}"
        print(row)

    print("\n    age of the loss — open/closed account right on dead slots · trait recall error of the dead")
    for rg in ("FRESH_LOSS", "OLD_LOSS"):
        row = f"    {rg:<14s}"
        for c in CONFIGS:
            rr = results[c].get(rg)
            row += f"{'-':>22s}" if rr is None else f"  {rr['open_acc_dead_slots']:5.3f} / {rr['attr_mse_dead']:5.3f}     "
        print(row)

    print("\n[5] what the vigil minds learned about the day")
    for cfg in ("VIGIL", "TAASSI"):
        g = _sigmoid_np(models[cfg].p["th_gate"])
        lam = _sigmoid_np(models[cfg].p["th_lam"]).mean()
        print(f"    {cfg:<8s} re-presentation gate  " +
              "  ".join(f"{PHASES[i]}={g[i]:.3f}" for i in range(P)) +
              f"   · leak for the absent living {lam:.3f}")
    lam_h = _sigmoid_np(models["HABITUATION"].p["th_lam"]).mean()
    print(f"    HABITUATION learned one leak for every absence: {lam_h:.3f}")

    print("\n[6] one household's vigil under TAASSI (a BATTLE episode)")
    idx = np.where(dva["regime"] == "BATTLE")[0]
    if idx.size:
        ledger_trace(models["TAASSI"], dva, int(idx[0]))
    rule()
    print(" done.")
    rule()


if __name__ == "__main__":
    main()

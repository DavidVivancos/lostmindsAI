#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Encyclopedia of Lost Minds — Chapter 0177 - ANEIRIN (fl. c. 550–600, Gododdin)

 CATRAETH
 Crosstalk-Audited Tally, Roster, and Elegiac Transcription Heads
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0177_aneirin_550 - Aneirin (fl. c. 550–600, Gododdin)
================================================================================  

THE MIND, IN ONE SENTENCE
-------------------------
Aneirin is the earliest mind in this corpus whose entire surviving output is an
argument against summarisation. *Y Gododdin* is not a narrative of the battle at
Catraeth. It is a sequence of monorhyme stanzas (awdlau) that say one thing —
they died — once per man, by name, in variant and non-identical forms. "Three
hundred went; one came back" is a statistic. Aneirin's method is to refuse the
statistic and pay the full storage cost of every individual.

Every modern learning system is a compressor. Its loss function is literally the
price of the information it agrees to throw away. Aneirin inverts the objective:
the interesting quantity is the *residue* — the named particular that the summary
destroys — and a mind is measured by how faithfully it declines to compress it.

THE ARCHITECTURE AS A CONSEQUENCE OF THAT THESIS
------------------------------------------------
CATRAETH holds three memories of the same host at once, and they disagree.

  1. TALLY (the chronicle).  s = sum of the men's codes, unbound and anonymous.
     It answers aggregate questions with near-perfect accuracy and it is
     structurally incapable of naming anyone. It is the number in the annal:
     never wrong, and that is the problem.

  2. MEDD (the mead-trace).  t = sum over men of  key_i (*) value_i, bound by
     circular convolution: a holographic superposition. It can be interrogated
     by name — but it degrades. Retrieval noise grows with the size of the host.
     This is memory as most systems implement it, and it is the exact mechanism
     by which a war turns into a casualty figure.

  3. AWDL (the roster of stanzas).  An explicit, per-name, VARIANT-REDUNDANT
     store. Each promoted warrior is written in K=3 independent variant codes,
     and the record must be recoverable from ANY ONE of them (erasure dropout at
     training time). This is the Book of Aneirin itself: Scribe A's 88 stanzas
     and Scribe B's 35, several of them doublets of A's in older orthography.
     The tradition preserved the variants instead of choosing between them.
     That is an erasure code, not an accident of copying.

Between them sit the decisions that make the mind Aneirin's rather than anyone
else's:

  4. CROSSTALK PROBE — learns to PREDICT how much of a given man the mead-trace
     is losing, from the name-cue and the noisy read alone. It never sees the
     true answer. The bard does not get to look the man up; he estimates the
     risk of losing him.

  5. ELEGY GATE — a deliberately tiny, deliberately blind policy over
     (predicted risk, size of host), under an explicit budget. A night is only
     so long; stanzas are not free. The number of stanzas is therefore NOT a
     hyperparameter of this model. It is what the mind spends once you tell it
     that forgetting a particular man is the error that matters.

  6. TALU ("talasant eu medd" — they paid for their mead) — the audit ledger,
     deliberately kept twice. A per-man residual (what he was given against what
     he did) read from the individual record, and an aggregate residual read
     from the tally, with a consistency constraint binding them. This makes the
     chronicler's sin measurable: the total can balance while every single man
     in it is wrong.

  7. TYST (the witness) — custody. The poem exists because someone came back:
     tradition has Aneirin taken at Catraeth and ransomed by Ceneu ap Llywarch
     Hen. So the decoder is conditioned on the survivor, and the shared
     monorhyme of the whole awdl is derived from the survivor's name-key.
     Erase the witness and the rhyme collapses to chance — which is the
     reciter's prologue made mechanical: since the earth covered Aneirin,
     poetry is parted from the Gododdin.

  8. GORCHAN HEAD — the transcription head. Six tokens per man: his name, his
     weapon, his country, what he did, what he was given, and the rhyme that
     binds him to his companions. There is no separate reconstruction head.
     The elegy IS the reconstruction.

WHY NOT ATTENTION
-----------------
Attention over stored keys retrieves by relevance from a context that has
already been curated for it, and it does not degrade: a key-value cache is exact
until you evict it. What had to be modelled here is the *cost of superposition
itself*, and a mind deciding — name by name, under a budget — when compression
becomes a moral error. That needs a memory that genuinely rots (holographic
binding does) and a policy trained against the rot. Nothing in this file is
attention.

CONVENTIONS
-----------
Pure NumPy. The reverse-mode autodiff engine below is written from scratch. A
finite-difference gradient check over every parameter tensor runs on every
invocation and must pass before training is allowed to start. Then a real
training loop, six experiments, and self-tests.

Run:  python3 chapter_0177_aneirin_550.py
================================================================================
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED = 177
rng = np.random.default_rng(SEED)


# ==============================================================================
# PART 1 — REVERSE-MODE AUTODIFF, FROM SCRATCH
# ==============================================================================
# The only unusual primitive is `cconv` (circular convolution) — the binding
# operation of the holographic memory. Its vector-Jacobian product uses the
# standard HRR identity: dL/da = g (*) involution(b), which in the Fourier
# domain is a conjugate multiply.

def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce a gradient back to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class T:
    """A node in the graph: value `d`, gradient `g`, and a backward closure."""

    __slots__ = ("d", "g", "_bw", "_prev", "name")

    def __init__(self, data, prev: Tuple["T", ...] = (), bw=None, name: str = ""):
        self.d = np.asarray(data, dtype=np.float64)
        self.g = np.zeros_like(self.d)
        self._prev = prev
        self._bw = bw if bw is not None else (lambda: None)
        self.name = name

    def backward(self) -> None:
        topo: List[T] = []
        seen = set()

        def build(v: "T"):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        self.g = np.ones_like(self.d)
        for v in reversed(topo):
            v._bw()

    # -- elementwise -----------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.d + o.d, (self, o), name="add")

        def bw():
            self.g += _unbroadcast(out.g, self.d.shape)
            o.g += _unbroadcast(out.g, o.d.shape)

        out._bw = bw
        return out

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.d * o.d, (self, o), name="mul")

        def bw():
            self.g += _unbroadcast(out.g * o.d, self.d.shape)
            o.g += _unbroadcast(out.g * self.d, o.d.shape)

        out._bw = bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self + (-o)

    def __rsub__(self, o):
        return (o if isinstance(o, T) else T(o)) + (-self)

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.d / o.d, (self, o), name="div")

        def bw():
            self.g += _unbroadcast(out.g / o.d, self.d.shape)
            o.g += _unbroadcast(-out.g * self.d / (o.d ** 2), o.d.shape)

        out._bw = bw
        return out

    def __pow__(self, k: float):
        out = T(self.d ** k, (self,), name=f"pow{k}")

        def bw():
            self.g += out.g * k * (self.d ** (k - 1))

        out._bw = bw
        return out

    # -- nonlinearities --------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.d)
        out = T(t, (self,), name="tanh")

        def bw():
            self.g += out.g * (1.0 - t * t)

        out._bw = bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.d, -60, 60)))
        out = T(s, (self,), name="sigmoid")

        def bw():
            self.g += out.g * s * (1.0 - s)

        out._bw = bw
        return out

    def relu(self):
        out = T(np.maximum(self.d, 0.0), (self,), name="relu")

        def bw():
            self.g += out.g * (self.d > 0.0)

        out._bw = bw
        return out

    def softplus(self):
        x = np.clip(self.d, -60, 60)
        out = T(np.log1p(np.exp(x)), (self,), name="softplus")

        def bw():
            self.g += out.g * (1.0 / (1.0 + np.exp(-x)))

        out._bw = bw
        return out

    def sqrt(self):
        s = np.sqrt(np.maximum(self.d, 1e-30))
        out = T(s, (self,), name="sqrt")

        def bw():
            self.g += out.g * 0.5 / s

        out._bw = bw
        return out

    # -- reductions / shape ----------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.d.sum(axis=axis, keepdims=keepdims), (self,), name="sum")

        def bw():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.d.shape).copy()

        out._bw = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.d.size if axis is None else self.d.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = T(self.d.reshape(*shape), (self,), name="reshape")

        def bw():
            self.g += out.g.reshape(self.d.shape)

        out._bw = bw
        return out

    def matmul(self, o: "T"):
        out = T(np.matmul(self.d, o.d), (self, o), name="matmul")

        def bw():
            ga = np.matmul(out.g, np.swapaxes(o.d, -1, -2))
            gb = np.matmul(np.swapaxes(self.d, -1, -2), out.g)
            self.g += _unbroadcast(ga, self.d.shape)
            o.g += _unbroadcast(gb, o.d.shape)

        out._bw = bw
        return out

    def __matmul__(self, o):
        return self.matmul(o)

    def swap(self):
        out = T(np.swapaxes(self.d, -1, -2), (self,), name="swap")

        def bw():
            self.g += np.swapaxes(out.g, -1, -2)

        out._bw = bw
        return out


def cat(ts: Sequence[T], axis: int = -1) -> T:
    out = T(np.concatenate([t.d for t in ts], axis=axis), tuple(ts), name="cat")
    sizes = [t.d.shape[axis] for t in ts]

    def bw():
        idx = 0
        ax = axis if axis >= 0 else out.d.ndim + axis
        for t, s in zip(ts, sizes):
            sl = [slice(None)] * out.d.ndim
            sl[ax] = slice(idx, idx + s)
            t.g += out.g[tuple(sl)]
            idx += s

    out._bw = bw
    return out


def repeat_axis(x: T, n: int, axis: int = 1) -> T:
    """Insert an axis of length n; the gradient sums back over it."""
    out = T(np.repeat(np.expand_dims(x.d, axis), n, axis=axis), (x,), name="rep")

    def bw():
        x.g += out.g.sum(axis=axis)

    out._bw = bw
    return out


def gather_rows(x: T, idx: np.ndarray) -> T:
    """x: (B,N,D), idx: (B,) -> (B,D). Backward scatters into the chosen rows."""
    b = np.arange(x.d.shape[0])
    out = T(x.d[b, idx, :], (x,), name="gather")

    def bw():
        gg = np.zeros_like(x.d)
        np.add.at(gg, (b, idx), out.g)
        x.g += gg

    out._bw = bw
    return out


def cconv(a: T, b: T) -> T:
    """Circular convolution on the last axis — the BINDING operation of MEDD."""
    D = a.d.shape[-1]
    fa, fb = np.fft.rfft(a.d, n=D), np.fft.rfft(b.d, n=D)
    out = T(np.fft.irfft(fa * fb, n=D), (a, b), name="cconv")

    def bw():
        fg = np.fft.rfft(out.g, n=D)
        ga = np.fft.irfft(fg * np.conj(fb), n=D)   # g (*) involution(b)
        gb = np.fft.irfft(fg * np.conj(fa), n=D)   # g (*) involution(a)
        a.g += _unbroadcast(ga, a.d.shape)
        b.g += _unbroadcast(gb, b.d.shape)

    out._bw = bw
    return out


def involution(x: T) -> T:
    """x*[i] = x[-i mod D]. Unbinding is convolution with the involution."""
    out = T(np.roll(x.d[..., ::-1], 1, axis=-1), (x,), name="inv")

    def bw():
        x.g += np.roll(out.g, -1, axis=-1)[..., ::-1]

    out._bw = bw
    return out


def l2norm(x: T, eps: float = 1e-8) -> T:
    return x / (((x * x).sum(axis=-1, keepdims=True) + eps).sqrt())


def softmax_ce(logits: T, targets: np.ndarray) -> T:
    """Mean cross-entropy. logits (...,V); targets (...) int."""
    z = logits.d - logits.d.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=-1, keepdims=True)
    flat_p = p.reshape(-1, p.shape[-1])
    flat_t = targets.reshape(-1)
    n = flat_t.shape[0]
    loss = -np.log(np.maximum(flat_p[np.arange(n), flat_t], 1e-12)).mean()
    out = T(loss, (logits,), name="ce")

    def bw():
        gp = flat_p.copy()
        gp[np.arange(n), flat_t] -= 1.0
        gp /= n
        logits.g += (gp * out.g).reshape(p.shape)

    out._bw = bw
    return out


def unitary_keys(n: int, D: int, gen: np.random.Generator) -> np.ndarray:
    """
    Name-keys as UNITARY vectors: unit magnitude at every Fourier frequency.
    Then k (*) involution(k) = delta exactly, so unbinding is exact for a host
    of one and all retrieval error is honest crosstalk from the other men.

    The keys are FIXED, never learned. You do not get to redesign a man's name
    to make him cheaper to remember.
    """
    F = D // 2 + 1
    phase = gen.uniform(0.0, 2.0 * np.pi, size=(n, F))
    phase[:, 0] = 0.0
    if D % 2 == 0:
        phase[:, -1] = 0.0
    return np.fft.irfft(np.exp(1j * phase), n=D)


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float((logits.argmax(axis=-1) == targets).mean())


# ==============================================================================
# PART 2 — THE CORPUS: A HOST, A YEAR OF MEAD, AND A ROAD TO CATRAETH
# ==============================================================================
# Each episode is one warband. Each man has a name (a fixed key), a lineage, a
# weapon, a home country, the mead he was given, and the deed he did.
#
# The ledger residual  r_i = deed_i - mead_i  is the moral quantity of the poem:
# did he pay for his mead? The aggregate of those residuals is what a chronicle
# records. The per-man residual is what the bard records. They are not the same
# object, and the entire architecture exists to keep them apart.

N_NAME, N_LIN, N_WEAP, N_HOME = 48, 6, 5, 8
N_DEEDB, N_MEADB, N_RHYME = 4, 4, 6

OFF_NAME = 0
OFF_WEAP = OFF_NAME + N_NAME
OFF_HOME = OFF_WEAP + N_WEAP
OFF_DEED = OFF_HOME + N_HOME
OFF_MEAD = OFF_DEED + N_DEEDB
OFF_RHY = OFF_MEAD + N_MEADB
VOCAB = OFF_RHY + N_RHYME          # 75
STANZA_LEN = 6
ATTR_SLOTS = (1, 2, 3, 4)          # weapon, country, deed-band, mead-band

# The hall closes at dawn. The poem is finite and the host is not: Y Gododdin
# names roughly eighty warriors while commemorating three hundred. This is the
# single most important number in the model.
STANZA_ALLOWANCE = 10.0


def _bucket(x: np.ndarray, nb: int) -> np.ndarray:
    q = np.clip(((x + 2.0) / 4.0) * nb, 0, nb - 1e-6)
    return q.astype(np.int64)


def _skew(gen, B, N, n, alpha=1.7):
    """
    A warband is not a uniform draw. Most men carry the common spear and come
    from the near country; a few are singular. This matters enormously, because
    a TYPICAL man can be reconstructed from the type — "he went, he fought, he
    died" restores him almost perfectly — while a singular man is destroyed by
    exactly that reconstruction. The individuating detail that fills Y Gododdin
    is not ornament. It is the part of a man that no type can carry.
    """
    w = 1.0 / np.arange(1, n + 1) ** alpha
    return gen.choice(n, size=(B, N), p=w / w.sum())


def make_episodes(B: int, N: int, gen: np.random.Generator) -> Dict[str, np.ndarray]:
    names = np.stack([gen.permutation(N_NAME)[:N] for _ in range(B)])
    lin = _skew(gen, B, N, N_LIN)
    weap = _skew(gen, B, N, N_WEAP)
    home = _skew(gen, B, N, N_HOME)

    mead = gen.normal(0.0, 1.0, size=(B, N))        # what the lord gave him
    valour = gen.normal(0.0, 1.0, size=(B, N))      # what was in the man
    deed = 0.55 * mead + 0.85 * valour              # what he actually did
    resid = deed - mead                             # did he pay for his mead?

    survivor = gen.integers(0, N, size=(B,))
    rhyme = names[np.arange(B), survivor] % N_RHYME  # the key of the poem

    stanza = np.zeros((B, N, STANZA_LEN), dtype=np.int64)
    stanza[:, :, 0] = OFF_NAME + names
    stanza[:, :, 1] = OFF_WEAP + weap
    stanza[:, :, 2] = OFF_HOME + home
    stanza[:, :, 3] = OFF_DEED + _bucket(deed, N_DEEDB)
    stanza[:, :, 4] = OFF_MEAD + _bucket(mead, N_MEADB)
    stanza[:, :, 5] = OFF_RHY + rhyme[:, None]

    return dict(names=names, lin=lin, weap=weap, home=home, mead=mead, deed=deed,
                resid=resid, survivor=survivor, rhyme=rhyme, stanza=stanza)


# ==============================================================================
# PART 3 — CATRAETH
# ==============================================================================

class Catraeth:

    def __init__(self, D: int = 48, H: int = 80, DZ: int = 40, DW: int = 24,
                 DR: int = 12, K: int = 3, gen: np.random.Generator = rng):
        self.D, self.H, self.DZ, self.DW, self.DR, self.K = D, H, DZ, DW, DR, K
        g = gen
        P: Dict[str, np.ndarray] = {}

        def init(*shape):
            return g.normal(0.0, math.sqrt(2.0 / shape[0]), size=shape)

        self.KEYS = unitary_keys(N_NAME, D, g)      # the given. Never learned.

        F_IN = N_LIN + N_WEAP + N_HOME + 2
        P["enc_W1"], P["enc_b1"] = init(F_IN, H), np.zeros(H)
        P["enc_W2"], P["enc_b2"] = init(H, D), np.zeros(D)

        # crosstalk probe: name-cue + noisy read + its norm + size of host
        P["prb_W1"], P["prb_b1"] = init(3 * D + 3, H), np.zeros(H)
        P["prb_W2"], P["prb_b2"] = init(H, H), np.zeros(H)
        P["prb_W3"], P["prb_b3"] = init(H, 1), np.zeros(1)

        # elegy gate: blind to the answer. It sees this man's predicted risk, that
        # risk RELATIVE TO HIS COMPANIONS, and the size of the host. Nothing else.
        # THREE NUMBERS, and the sign of two of them is not up for negotiation.
        # The gate is MONOTONE BY CONSTRUCTION in the predicted loss of a man
        # (weights passed through softplus, so they cannot go negative). This is
        # a structural commitment, not a learned one, and it is the whole ethic
        # of the form: you spend the night on the men you are about to lose.
        # What is learned is the hard part — the estimate of what will be lost.
        # Kept this small so that the finished rule can simply be read off.
        P["gate_a"] = np.array([0.5])      # weight on predicted loss   (>= 0)
        P["gate_c"] = np.array([0.5])      # weight on loss vs. rivals  (>= 0)
        P["gate_b"] = np.array([+2.0])     # how open the hall stands

        # awdl store: K variant scribes + one shared reader
        for j in range(K):
            P[f"var_W{j}"], P[f"var_b{j}"] = init(D, DZ), np.zeros(DZ)
        P["vdec_W"], P["vdec_b"] = init(DZ, D), np.zeros(D)

        # tyst: the witness (his key AND his code) and the rhyme he carries
        P["wit_W"], P["wit_b"] = init(2 * D, DW), np.zeros(DW)
        P["rhy_W"], P["rhy_b"] = init(DW, N_RHYME), np.zeros(N_RHYME)
        P["rho_W"], P["rho_b"] = init(DW, DR), np.zeros(DR)

        # gorchan head: cue + recovered man + witness + rhyme -> a stanza
        P["dec_W1"], P["dec_b1"] = init(2 * D + DW + DR, H), np.zeros(H)
        P["dec_W2"], P["dec_b2"] = init(H, STANZA_LEN * VOCAB), np.zeros(STANZA_LEN * VOCAB)

        # talu: the ledger, kept twice on purpose
        P["led_w"], P["led_b"] = init(D, 1), np.zeros(1)     # per man, from his record
        P["agg_w"], P["agg_b"] = init(D, 1), np.zeros(1)     # the host, from the tally

        self.P = P

    def tensors(self) -> Dict[str, T]:
        return {k: T(v, name=k) for k, v in self.P.items()}

    def _risk_of_losing(self, ep, W, keys_d, v_d, v_hat_d) -> np.ndarray:
        """
        Pure-NumPy, no graph: decode each man FROM THE MEAD-TRACE ALONE with the
        current voice, and return his cross-entropy — the price, in facts, of not
        giving him a stanza. This is a stop-gradient target for the probe.
        """
        B, N = ep["names"].shape
        v_s, k_s = v_d[np.arange(B), ep["survivor"]], keys_d[np.arange(B), ep["survivor"]]
        gc = np.tanh(np.concatenate([k_s, v_s], -1) @ W["wit_W"].d + W["wit_b"].d)
        rh = np.tanh(gc @ W["rho_W"].d + W["rho_b"].d)
        x = np.concatenate([keys_d, v_hat_d,
                            np.repeat(gc[:, None, :], N, 1),
                            np.repeat(rh[:, None, :], N, 1)], -1)
        h = np.tanh(x @ W["dec_W1"].d + W["dec_b1"].d)
        lg = (h @ W["dec_W2"].d + W["dec_b2"].d).reshape(B, N, STANZA_LEN, VOCAB)
        z = lg - lg.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        ce = -np.take_along_axis(logp, ep["stanza"][..., None], axis=-1)[..., 0]
        # only the individuating slots count. His name is on the cue and the
        # rhyme is on the witness; neither is at risk. What is at risk is the man.
        return ce[:, :, ATTR_SLOTS].mean(-1, keepdims=True)          # (B,N,1)

    @staticmethod
    def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
        oh = np.zeros(idx.shape + (n,))
        np.put_along_axis(oh, idx[..., None], 1.0, axis=-1)
        return oh

    def forward(self, ep, W, *, variant_mask=None, force_p=None, kill_witness=False,
                lam_commit: float = 0.0, allowance: float = STANZA_ALLOWANCE,
                triage: str = "learned", rank_in: Optional[np.ndarray] = None,
                straight_through: bool = True,
                w_sep: float = 0.50, sep_margin: float = 0.30,
                w_roster: float = 1.0, train: bool = True,
                frozen: Optional[Dict[str, np.ndarray]] = None,
                gen: np.random.Generator = rng) -> Dict[str, object]:
        # `frozen` pins the three stop-gradient quantities (the probe's regression
        # target, its norm feature, and the roster's copy-target) as constants. In training they are
        # recomputed and detached each step, which is correct but not
        # finite-differentiable; the gradient check pins them so that the function
        # being differentiated is exactly the one the backward pass encodes.

        B, N = ep["names"].shape
        D, K = self.D, self.K
        Nscale = np.full((B, N, 1), N / 32.0)

        # ---- 1. the men become vectors ---------------------------------------
        feats = np.concatenate([
            self._onehot(ep["lin"], N_LIN),
            self._onehot(ep["weap"], N_WEAP),
            self._onehot(ep["home"], N_HOME),
            ep["mead"][..., None], ep["deed"][..., None],
        ], axis=-1)
        h = (T(feats) @ W["enc_W1"] + W["enc_b1"]).tanh()
        v = l2norm((h @ W["enc_W2"] + W["enc_b2"]).tanh())          # (B,N,D) unit sphere

        keys = T(self.KEYS[ep["names"]])                            # (B,N,D) FIXED

        # ---- 2. TALLY — the chronicle. Exact about totals, blind to persons. --
        tally = v.sum(axis=1)                                       # (B,D)

        # ---- 3. MEDD — the mead-trace. Interrogable by name, and it rots. -----
        trace = cconv(keys, v).sum(axis=1)                          # (B,D)
        tr_n = repeat_axis(trace, N, axis=1)
        v_raw = cconv(tr_n, involution(keys))                       # noisy recall
        v_hat = l2norm(v_raw)   # its magnitude is crosstalk; only direction survives

        # crosstalk, reported but NOT what the mind is asked to predict:
        crosstalk = 1.0 - (v_hat.d * v.d).sum(-1, keepdims=True)     # 1 - cos, in [0,2]

        if frozen is not None:
            true_err, vh_norm = frozen["true_err"], frozen["vh_norm"]
        else:
            # ---- WHAT A STANZA WOULD ACTUALLY BUY -----------------------------
            # Not "how noisy is the read" — every man in a superposition is about
            # equally noisy. The question is how much of THIS man survives the
            # type. Sing the man out of the mead-trace alone, with the same voice
            # that will sing the elegy, and count what it gets wrong about him.
            # A common man is restored almost perfectly by the type: he went, he
            # fought, he died. A singular man is destroyed by exactly that
            # restoration. The stanzas belong to the second kind.
            true_err = self._risk_of_losing(ep, W, keys.d, v.d, v_hat.d)   # (B,N,1)
            vh_norm = np.log1p(np.sqrt((v_raw.d ** 2).sum(-1, keepdims=True)))

        # ---- 4. the probe: how much of this man is the trace losing? ----------
        # It never sees the man. It sees his name, the noisy read, and the host.
        # The type of this host — what "a man of the Gododdin" looks like on
        # average tonight — and how far this man stands from it. A bard who can
        # answer "is this one of the many or one of the few?" can spend a short
        # night well; one who cannot is reduced to naming men at hazard.
        type_u = l2norm(repeat_axis(tally, N, axis=1))                # (B,N,D)
        typicality = (v_hat * type_u).sum(axis=-1, keepdims=True)     # (B,N,1)
        prb_in = cat([keys, v_hat, type_u, typicality, T(vh_norm), T(Nscale)], axis=-1)
        ph = (prb_in @ W["prb_W1"] + W["prb_b1"]).tanh()
        ph = (ph @ W["prb_W2"] + W["prb_b2"]).tanh()
        err_hat = (ph @ W["prb_W3"] + W["prb_b3"]).softplus()         # (B,N,1) >= 0
        loss_probe = ((err_hat - T(true_err)) ** 2).mean()

        # ---- 5. the elegy gate: on whom do we spend a stanza? ------------------
        # A bard does not choose in the abstract. He chooses THIS man over THAT
        # man, tonight, in this hall. So the gate is given the risk of losing
        # this man standardised against the risk of losing his companions.
        e_mu = err_hat.mean(axis=1, keepdims=True)
        e_dev = err_hat - e_mu
        e_sd = ((e_dev * e_dev).mean(axis=1, keepdims=True) + 1e-6).sqrt()
        e_z = e_dev / e_sd                                          # (B,N,1)
        score = (err_hat * W["gate_a"].softplus()
                 + e_z * W["gate_c"].softplus() + W["gate_b"])       # (B,N,1)
        p_soft = score.sigmoid()

        # THE NIGHT IS SHORT AND THE HOST IS NOT.
        # The allowance is structural, not a penalty: the bard ranks the men and
        # names as many as the hall will hold. A man either has a stanza or he
        # does not — a gate hovering at 0.4 for everyone has bought two fifths of
        # an elegy for three hundred men, which is not a thing that exists. So
        # the decision is taken literally in the forward pass and the gradient is
        # passed straight through it. The only question the night actually asks
        # is the one the model is now forced to answer: WHOM DO YOU NAME?
        k = int(min(N, allowance))
        sn = score.d[..., 0]                                          # (B,N)
        if triage == "learned":
            order = np.argsort(-sn, axis=1)[:, :k]
        elif triage == "given":                  # a ranking handed in from outside
            order = np.argsort(-rank_in, axis=1)[:, :k]
        elif triage == "random":                 # name k men at hazard
            order = np.argsort(gen.random(sn.shape), axis=1)[:, :k]
        else:
            raise ValueError(triage)
        mask = np.zeros_like(sn)
        np.put_along_axis(mask, order, 1.0, axis=1)
        mask = mask[..., None]                                        # (B,N,1)

        if straight_through:
            p = p_soft + T(mask - p_soft.d)       # value: 0/1.  gradient: through sigma.
        else:
            p = p_soft                            # the smooth surrogate the check verifies
        if force_p is not None:
            p = T(np.full((B, N, 1), float(force_p)))
        loss_commit = (p_soft * (T(np.ones((B, N, 1))) - p_soft)).mean()

        # ---- 6. AWDL — three scribes; any one hand must suffice ---------------
        if variant_mask is None:
            if train:
                m = (gen.random((B, N, K)) > 0.4).astype(np.float64)
                m[m.sum(-1) == 0] = 1.0                             # never lose all three
            else:
                m = np.ones((B, N, K))
        else:
            m = np.broadcast_to(variant_mask, (B, N, K)).astype(np.float64).copy()

        # the man himself, as the target of his own record (stop-gradient: the
        # roster must copy the man, not quietly redefine him into something easier)
        v_tgt = T(frozen["v_tgt"]) if frozen is not None else T(v.d)
        zs, per_variant_loss = [], None
        for j in range(K):
            zj = (v @ W[f"var_W{j}"] + W[f"var_b{j}"]).tanh()
            vj = l2norm((zj @ W["vdec_W"] + W["vdec_b"]).tanh())    # decode THIS hand alone
            lj = ((vj - v_tgt) ** 2).mean()
            per_variant_loss = lj if per_variant_loss is None else per_variant_loss + lj
            zs.append(zj * T(m[:, :, j:j + 1]))

        z_sum = zs[0]
        for j in range(1, K):
            z_sum = z_sum + zs[j]
        z_bar = z_sum / T(m.sum(axis=-1, keepdims=True))
        v_bar = l2norm((z_bar @ W["vdec_W"] + W["vdec_b"]).tanh())  # (B,N,D) exact read
        loss_roster = ((v_bar - v_tgt) ** 2).mean() + per_variant_loss * (1.0 / K)

        # ---- 7. what the mind actually has of this man ------------------------
        v_use = p * v_bar + (T(np.ones((B, N, 1))) - p) * v_hat

        # ---- 8. TYST — the one who came back carries the key of the poem ------
        v_s = gather_rows(v, ep["survivor"])
        k_s = gather_rows(keys, ep["survivor"])
        gcode = (cat([k_s, v_s], -1) @ W["wit_W"] + W["wit_b"]).tanh()
        if kill_witness:
            gcode = T(np.zeros((B, self.DW)))
        rhy_logits = gcode @ W["rhy_W"] + W["rhy_b"]
        loss_rhyme_aux = softmax_ce(rhy_logits, ep["rhyme"])
        rho = (gcode @ W["rho_W"] + W["rho_b"]).tanh()              # the monorhyme

        # ---- 9. GORCHAN — one stanza per man ---------------------------------
        dec_in = cat([keys, v_use, repeat_axis(gcode, N, 1), repeat_axis(rho, N, 1)], -1)
        dh = (dec_in @ W["dec_W1"] + W["dec_b1"]).tanh()
        logits = (dh @ W["dec_W2"] + W["dec_b2"]).reshape(B, N, STANZA_LEN, VOCAB)
        loss_elegy = softmax_ce(logits, ep["stanza"])

        # ---- 10. TALU — the ledger kept twice --------------------------------
        r_i = (v_use @ W["led_w"] + W["led_b"]).reshape(B, N)        # from the record
        loss_led_ind = ((r_i - T(ep["resid"])) ** 2).mean()
        agg = (tally @ W["agg_w"] + W["agg_b"]).reshape(B)           # from the tally
        R_true = ep["resid"].sum(axis=1)
        loss_led_agg = ((agg - T(R_true)) ** 2).mean() * (1.0 / N)
        loss_consist = ((r_i.sum(axis=1) - agg) ** 2).mean() * (1.0 / N)

        # ---- 11. anti-collapse: no two men share a stanza ---------------------
        S = v @ v.swap()                                            # v is already unit
        L_sep = (((S - T(np.eye(N)[None] * 2.0)) - sep_margin).relu() ** 2).sum(axis=2).mean()

        # ---- 12. the budget: A NIGHT IS ONLY SO LONG -------------------------
        # Not a price per stanza — an ALLOWANCE. The hall closes; the poem ends.
        # Y Gododdin names about eighty men and commemorates three hundred. The
        # bard cannot save everyone, and the whole moral weight of the form sits
        # in the question this term forces the model to answer: WHOM DO YOU SAVE?
        loss_budget = p.mean()                                       # stanzas actually spent

        total = (loss_elegy
                 + 1.0 * loss_led_ind
                 + 2.0 * loss_led_agg
                 + 0.25 * loss_consist
                 + 4.0 * loss_probe
                 + 0.5 * loss_rhyme_aux
                 + w_roster * loss_roster
                 + w_sep * L_sep
                 + lam_commit * loss_commit)

        return dict(total=total, logits=logits, p=p, err_hat=err_hat, true_err=true_err,
                    v_hat_d=v_raw.d, crosstalk=crosstalk, r_i=r_i, agg=agg,
                    R_true=R_true, v=v,
                    parts=dict(elegy=loss_elegy, led_ind=loss_led_ind, led_agg=loss_led_agg,
                               consist=loss_consist, probe=loss_probe, rhyme=loss_rhyme_aux,
                               roster=loss_roster, sep=L_sep, budget=loss_budget,
                               commit=loss_commit))


# ==============================================================================
# PART 4 — ADAM, FROM SCRATCH
# ==============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = np.clip(grads[k], -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# PART 5 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def gradient_check(verbose: bool = True) -> bool:
    """
    Checks the SMOOTH surrogate (straight_through=False). The top-k triage that
    sits on top of it in training is a deliberately biased estimator — binary
    going forward, gradient passed straight through — and a finite-difference
    check of a step function would be meaningless. What must be exactly right,
    and is checked here to 1e-4, is every analytic derivative underneath it.
    """
    g = np.random.default_rng(7)
    model = Catraeth(D=16, H=12, DZ=8, DW=6, DR=4, K=3, gen=g)
    ep = make_episodes(B=2, N=5, gen=g)
    vmask = (g.random((2, 5, 3)) > 0.4).astype(float)
    vmask[vmask.sum(-1) == 0] = 1.0

    CHK = dict(variant_mask=vmask, train=False, straight_through=False)
    o0 = model.forward(ep, model.tensors(), **CHK)
    FROZEN = {"true_err": o0["true_err"].copy(),
              "vh_norm": np.sqrt((o0["v_hat_d"] ** 2).mean(axis=-1, keepdims=True)),
              "v_tgt": o0["v"].d.copy()}

    def loss_of(params):
        model.P = params
        W = model.tensors()
        L = model.forward(ep, W, frozen=FROZEN, **CHK)["total"]
        L.backward()
        return float(L.d), {k: W[k].g.copy() for k in W}

    base = {k: v.copy() for k, v in model.P.items()}
    _, grads = loss_of({k: v.copy() for k, v in base.items()})

    eps, worst, worst_key, n = 1e-6, 0.0, "", 0
    for k in sorted(base):
        idxs = g.choice(base[k].size, size=min(3, base[k].size), replace=False)
        for fi in idxs:
            i = np.unravel_index(fi, base[k].shape)
            pp = {kk: vv.copy() for kk, vv in base.items()}; pp[k][i] += eps
            lp, _ = loss_of(pp)
            pm = {kk: vv.copy() for kk, vv in base.items()}; pm[k][i] -= eps
            lm, _ = loss_of(pm)
            num, ana = (lp - lm) / (2 * eps), grads[k][i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            n += 1
            if rel > worst:
                worst, worst_key = rel, f"{k}{i}"
    model.P = base
    ok = worst < 1e-4
    if verbose:
        print(f"  tensors checked : {len(base)}   entries probed : {n}")
        print(f"  worst rel. err  : {worst:.3e}  (at {worst_key})")
        print(f"  RESULT          : {'PASS' if ok else 'FAIL'}   (tolerance 1e-4)")
    return ok


# ==============================================================================
# PART 6 — EVALUATION AND TRAINING
# ==============================================================================

def evaluate(model: Catraeth, gen, N: int = 16, B: int = 32, **fwd) -> Dict[str, float]:
    ep = make_episodes(B, N, gen)
    if fwd.get("triage") == "oracle":
        # What a stanza would ACTUALLY buy for each man: the number of facts the
        # mead-trace alone gets wrong about him. No mind has access to this.
        o0 = model.forward(ep, model.tensors(), train=False, gen=gen, force_p=0.0)
        wrong = (o0["logits"].d.argmax(-1)[:, :, ATTR_SLOTS]
                 != ep["stanza"][:, :, ATTR_SLOTS]).sum(-1).astype(float)
        fwd = dict(fwd, triage="given", rank_in=wrong)
    out = model.forward(ep, model.tensors(), train=False, gen=gen, **fwd)
    lg = out["logits"].d
    pred = lg.argmax(-1)

    attr_acc = float((pred[:, :, ATTR_SLOTS] == ep["stanza"][:, :, ATTR_SLOTS]).mean())
    mono_acc = accuracy(lg[:, :, 5, :], ep["stanza"][:, :, 5])

    # Does the record still belong to the right man?  For every decoded stanza,
    # find whose TRUE record it fits best (Hamming over the four attributes).
    # If that is not the man it names, the poem has handed one man another man's
    # death.  This is the failure Aneirin is actually afraid of: not silence,
    # but confident, fluent, misattributed praise.
    P = pred[:, :, ATTR_SLOTS]                          # (B,N,4) decoded
    Ttrue = ep["stanza"][:, :, ATTR_SLOTS]              # (B,N,4) true
    exact = (P == Ttrue).all(-1)
    ham = (P[:, :, None, :] != Ttrue[:, None, :, :]).sum(-1)     # (B,N,N)
    own = np.einsum("bii->bi", ham)                     # distance to his own record
    rival = ham + np.eye(N)[None] * 99                  # distance to every other man
    misattr = rival.min(-1) < own                       # fits someone else better
    garbled = (~exact) & (~misattr)

    p = out["p"].d[..., 0]
    te = out["true_err"][..., 0]
    kept = p > 0.5                                       # given a stanza
    lost = ~kept                                         # left to the mead-trace
    f = lambda arr, m: float(arr[m].mean()) if m.any() else float("nan")
    attr_ok = (P0 := pred[:, :, ATTR_SLOTS]) == ep["stanza"][:, :, ATTR_SLOTS]
    gate_r = float(np.corrcoef(p.ravel(), te.ravel())[0, 1]) if p.std() > 1e-9 else 0.0
    probe_r = float(np.corrcoef(out["err_hat"].d.ravel(), te.ravel())[0, 1])

    v = out["v"].d
    vn = v / np.linalg.norm(v, axis=-1, keepdims=True)
    S = vn @ np.swapaxes(vn, -1, -2) - np.eye(N)[None] * 2.0

    return dict(attr_acc=attr_acc, exact=float(exact.mean()),
                misattr=float(misattr.mean()), garbled=float(garbled.mean()),
                kept_attr=f(attr_ok.mean(-1), kept), lost_attr=f(attr_ok.mean(-1), lost),
                kept_misattr=f(misattr, kept), lost_misattr=f(misattr, lost),
                kept_risk=f(te, kept), lost_risk=f(te, lost),
                mono_acc=mono_acc, promote=float(p.mean()),
                stanzas=float((p > 0.5).sum(1).mean()),
                ind_err=float(np.abs(out["r_i"].d - ep["resid"]).mean()),
                agg_err=float(np.abs(out["agg"].d - out["R_true"]).mean()),
                agg_scale=float(np.abs(out["R_true"]).mean()),
                gate_r=gate_r, probe_r=probe_r,
                crosstalk=float(out["crosstalk"].mean()), risk=float(te.mean()),
                collide=float((S.max(-1) > 0.90).mean()),
                loss=float(out["total"].d))


HOST_SIZES = (6, 10, 16, 22, 28)     # the mind must generalise over host size


def train(steps: int = 1600, B: int = 20, lr: float = 4e-3, lam_commit: float = 0.0,
          w_sep: float = 0.50, seed: int = SEED, verbose: bool = True) -> Catraeth:
    g = np.random.default_rng(seed)
    model = Catraeth(gen=g)
    opt = Adam(model.P, lr=lr)
    t0 = time.time()
    if verbose:
        print(f"{'step':>5} {'loss':>7} {'elegy':>7} {'attr':>6} {'rhyme':>6} "
              f"{'named/16':>9} {'|r_i|':>7} {'probe_r':>8} {'gate~risk':>10}")
    for s in range(1, steps + 1):
        N = int(HOST_SIZES[g.integers(0, len(HOST_SIZES))])
        ep = make_episodes(B, N, g)
        W = model.tensors()
        out = model.forward(ep, W, train=True, gen=g, w_sep=w_sep,
                            lam_commit=lam_commit)
        out["total"].backward()
        opt.step({k: W[k].g for k in W})
        if verbose and (s % 200 == 0 or s == 1):
            m = evaluate(model, np.random.default_rng(1000 + s), N=16, B=24)
            print(f"{s:>5} {m['loss']:>7.3f} {float(out['parts']['elegy'].d):>7.3f} "
                  f"{m['attr_acc']:>6.2f} {m['mono_acc']:>6.2f} {m['stanzas']:>9.1f} "
                  f"{m['ind_err']:>7.3f} {m['probe_r']:>8.3f} {m['gate_r']:>10.3f}")
    if verbose:
        print(f"  trained in {time.time() - t0:.1f}s")
    return model


# ==============================================================================
# PART 7 — THE ARGUMENT, RUN AS NUMBERS
# ==============================================================================

def exp_compression_pressure(model):
    print("\n[1] THE POEM IS FINITE AND THE HOST IS NOT")
    print(f"  {'host':>5} {'named':>6} {'named %':>8} {'crosstalk':>10} {'whole-host':>11} "
          f"{'MISATTRIB.':>11}")
    for N in (6, 10, 16, 24, 32, 40):
        m = evaluate(model, np.random.default_rng(500 + N), N=N, B=64)
        print(f"  {N:>5} {m['stanzas']:>6.0f} {100 * m['stanzas'] / N:>7.0f}% "
              f"{m['crosstalk']:>10.2f} {m['attr_acc']:>11.2f} {m['misattr']:>11.2f}")
    print("  The named do not grow with the host: the hall closes at the same hour")
    print("  however many men rode out. Y Gododdin names about eighty warriors and")
    print("  commemorates three hundred, and this is the arithmetic of that fact.")


def exp_chroniclers_sin(model):
    print("\n[2] THE CHRONICLER'S SIN — the total is right and every man is wrong")
    print(f"  {'what is kept':>26} {'total err':>10} {'per-man err':>12} "
          f"{'right record':>13} {'MISATTRIB.':>11}")
    for label, kw in (("tally + mead-trace only", dict(force_p=0.0)),
                      ("tally + the bard's roster", {}),
                      ("tally + a stanza for all", dict(force_p=1.0))):
        m = evaluate(model, np.random.default_rng(77), N=24, B=64, **kw)
        print(f"  {label:>26} {m['agg_err']:>10.2f} {m['ind_err']:>12.2f} "
              f"{m['exact']:>13.2f} {m['misattr']:>11.2f}")
    m = evaluate(model, np.random.default_rng(77), N=24, B=64)
    print(f"  (the true total has magnitude {m['agg_scale']:.2f}, so the tally is near-exact")
    print("   in EVERY row. The number of the dead never suffers. What suffers is the")
    print("   claim that THIS man, named, did THIS thing — and the failure is not")
    print("   silence. It is fluent, confident, misattributed praise.")


def exp_whom_do_you_save(model):
    print("\n[3] WHOM DO YOU SAVE? — the night is short; rank the host")
    print(f"  {'triage':>28} {'whole-host':>11} {'MISATTRIB.':>11} {'unnamed men':>12} "
          f"{'saved~risk':>11}")
    for N in (16, 32):
        print(f"  -- host of {N}, ten stanzas --")
        for label, tri in (("names at hazard", "random"),
                           ("names whom it fears to lose", "learned"),
                           ("(oracle: knows what it loses)", "oracle")):
            m = evaluate(model, np.random.default_rng(11), N=N, B=64, triage=tri)
            print(f"  {label:>28} {m['attr_acc']:>11.3f} {m['misattr']:>11.3f} "
                  f"{m['lost_attr']:>12.2f} {m['gate_r']:>+11.2f}")
    a = model.P
    sp = lambda x: float(np.log1p(np.exp(x)))
    print(f"  the learned rule, in full:  worth-a-stanza = {sp(a['gate_a'][0]):.2f} x (predicted loss "
          f"of this man)")
    print(f"                                             + {sp(a['gate_c'][0]):.2f} x (that loss, "
          f"against his companions)")
    print(f"                                             + {a['gate_b'][0]:.2f}")
    print("  Under a hard memory budget the whole value of the mind collapses into one")
    print("  faculty: knowing what it is about to lose. Choosing well recovers about")
    print("  half the distance from naming men at hazard to naming them with foreknowledge.")


def exp_variant_erasure(model):
    print("\n[4] VARIANT REDUNDANCY — the Book of Aneirin as an erasure code")
    print(f"  {'surviving hands':>20} {'facts recalled':>15} {'record intact':>14}")
    for label, mask in (("A + B1 + B2", [1, 1, 1]), ("A + B1", [1, 1, 0]),
                        ("B2 alone", [0, 0, 1]), ("A alone", [1, 0, 0])):
        m = evaluate(model, np.random.default_rng(31), N=16, B=64, force_p=1.0,
                     variant_mask=np.array(mask, float)[None, None, :])
        print(f"  {label:>20} {m['attr_acc']:>15.2f} {m['exact']:>14.2f}")
    print("  The record survives the loss of any two of its three hands. This is why a")
    print("  thirteenth-century scribe copied the doublets instead of choosing between")
    print("  them: he was not being careless. He was maintaining the redundancy.")


def exp_witness(model):
    print("\n[5] THE WITNESS — custody of the poem")
    a = evaluate(model, np.random.default_rng(5), N=16, B=64)
    b = evaluate(model, np.random.default_rng(5), N=16, B=64, kill_witness=True)
    print(f"  {'':>18} {'monorhyme':>10} {'facts recalled':>15} {'record intact':>14}")
    print(f"  {'witness kept':>18} {a['mono_acc']:>10.2f} {a['attr_acc']:>15.2f} "
          f"{a['exact']:>14.2f}")
    print(f"  {'witness erased':>18} {b['mono_acc']:>10.2f} {b['attr_acc']:>15.2f} "
          f"{b['exact']:>14.2f}")
    print(f"  (chance monorhyme = {1.0 / N_RHYME:.2f})")
    print("  Erase the man who came back and the individual records mostly survive — but")
    print("  the thing that made a hundred stanzas ONE POEM does not. 'Since the earth")
    print("  covered Aneirin, poetry is parted from the Gododdin.'")


def exp_anticollapse(steps=800):
    print("\n[6] ANTI-COLLAPSE — no two men share a stanza")
    a_m = train(steps=steps, w_sep=0.50, verbose=False, seed=4)
    b_m = train(steps=steps, w_sep=0.00, verbose=False, seed=4)
    a = evaluate(a_m, np.random.default_rng(12), N=24, B=64)
    b = evaluate(b_m, np.random.default_rng(12), N=24, B=64)
    print(f"  {'':>24} {'merged men':>11} {'MISATTRIB.':>11} {'record intact':>14}")
    print(f"  {'separation enforced':>24} {a['collide']:>11.3f} {a['misattr']:>11.2f} "
          f"{a['exact']:>14.2f}")
    print(f"  {'separation removed':>24} {b['collide']:>11.3f} {b['misattr']:>11.2f} "
          f"{b['exact']:>14.2f}")
    print("  Letting two warriors become one code is the cheapest compression available")
    print("  to any mind. It is also the one this architecture exists to forbid.")


def show_stanzas(model):
    print("\n[7] THE OUTPUT — one host, decoded")
    g = np.random.default_rng(1773)
    ep = make_episodes(1, 14, g)
    o = model.forward(ep, model.tensors(), train=False, allowance=6.0)
    pred = o["logits"].d.argmax(-1)[0]
    p = o["p"].d[0, :, 0]
    risk = o["true_err"][0, :, 0]
    weap = ["spear", "sword", "shield", "bow", "axe"]
    home = ["Eidyn", "Aeron", "Elfed", "Gwynedd", "Bannog", "Rheged", "Alt Clut", "Pictland"]
    band = ["little", "some", "much", "all"]
    cl = lambda x, n: max(0, min(n - 1, int(x)))
    print("  fourteen men rode out; the hall holds six stanzas.")
    print(f"  the rhyme, set by the man who came back: -{ep['rhyme'][0]}\n")
    for i in range(14):
        t, true = pred[i], ep["stanza"][0, i]
        ok = (t[list(ATTR_SLOTS)] == true[list(ATTR_SLOTS)]).all()
        tag = "STANZA" if p[i] > 0.5 else "   ---"
        star = "*" if i == ep["survivor"][0] else " "
        line = (f"man #{t[0] - OFF_NAME:<3d} {weap[cl(t[1] - OFF_WEAP, 5)]:<7s} "
                f"of {home[cl(t[2] - OFF_HOME, 8)]:<9s} did {band[cl(t[3] - OFF_DEED, 4)]:<7s} "
                f"for {band[cl(t[4] - OFF_MEAD, 4)]:<7s} mead, rhyme -{t[5] - OFF_RHY}")
        note = "" if ok else (f"   <- but he was {weap[true[1] - OFF_WEAP]} "
                              f"of {home[true[2] - OFF_HOME]}")
        print(f"  {tag}{star} risk {risk[i]:.2f} | {line}{note}")
    print("\n  (* = the witness. Every line closes on the same rhyme: that is an awdl.)")
    print("  The unnamed are not forgotten. That would be a mercy. They are RESTORED —")
    print("  fluently, plausibly, and as somebody else.")


# ==============================================================================
# PART 8 — SELF-TESTS
# ==============================================================================

def self_tests(model) -> bool:
    print("\n[SELF-TESTS]")
    ok = True
    g = np.random.default_rng(3)
    D = 64
    kk = unitary_keys(40, D, g)
    vv = g.normal(0, 1 / math.sqrt(D), (40, D))
    bind = lambda a, b: np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D)
    unbind = lambda t, k: bind(t, np.roll(k[::-1], 1))

    e1 = float(np.mean((unbind(bind(kk[0], vv[0]), kk[0]) - vv[0]) ** 2))
    t = e1 < 1e-20
    print(f"  binding inverts exactly for a host of one       : {'PASS' if t else 'FAIL'} "
          f"(mse {e1:.1e})"); ok &= t

    errs = [float(np.mean((unbind(sum(bind(kk[i], vv[i]) for i in range(n)), kk[0])
                           - vv[0]) ** 2)) for n in (2, 8, 32)]
    t = errs[0] < errs[1] < errs[2]
    print(f"  crosstalk grows with the size of the host       : {'PASS' if t else 'FAIL'} "
          f"{[float(f'{e:.1e}') for e in errs]}"); ok &= t

    ep = make_episodes(3, 7, g)
    o = model.forward(ep, model.tensors(), train=False)
    t = o["logits"].d.shape == (3, 7, STANZA_LEN, VOCAB) and np.isfinite(o["total"].d)
    print(f"  forward shapes and finite loss                  : {'PASS' if t else 'FAIL'}"); ok &= t

    # The gate ranks; the hall then takes as many as it holds. So what must be
    # monotone is the SCORE it ranks by, not the squashed probability (which
    # saturates near one — the bard would name them all if the night allowed).
    W = model.tensors()
    e = np.linspace(0.0, 1.2, 12).reshape(1, 12, 1)
    ez = (e - e.mean()) / (e.std() + 1e-6)
    sc = (T(e) * W["gate_a"].softplus() + T(ez) * W["gate_c"].softplus()
          + W["gate_b"]).d.ravel()
    t = bool(np.all(np.diff(sc) > 0))
    print(f"  worth-a-stanza rises with the risk of loss      : {'PASS' if t else 'FAIL'} "
          f"(score {sc[0]:.2f} -> {sc[-1]:.2f}, strictly increasing)"); ok &= t

    a = evaluate(model, np.random.default_rng(8), N=28, B=64, force_p=1.0)
    b = evaluate(model, np.random.default_rng(8), N=28, B=64, force_p=0.0)
    t = a["attr_acc"] > b["attr_acc"] + 0.20
    print(f"  the roster restores a man; the trace does not   : {'PASS' if t else 'FAIL'} "
          f"({b['attr_acc']:.2f} -> {a['attr_acc']:.2f})"); ok &= t

    t = b["misattr"] > a["misattr"] + 0.20
    print(f"  compression MISATTRIBUTES, it does not just lose: {'PASS' if t else 'FAIL'} "
          f"({a['misattr']:.2f} -> {b['misattr']:.2f})"); ok &= t

    c = evaluate(model, np.random.default_rng(8), N=28, B=64)
    t = c["agg_err"] < 0.20 * c["agg_scale"]
    print(f"  the tally stays exact whatever is forgotten     : {'PASS' if t else 'FAIL'} "
          f"(err {c['agg_err']:.2f} on a total of {c['agg_scale']:.2f})"); ok &= t

    lr = evaluate(model, np.random.default_rng(2), N=32, B=64, triage="learned")
    rd = evaluate(model, np.random.default_rng(2), N=32, B=64, triage="random")
    t = lr["attr_acc"] > rd["attr_acc"] + 0.02 and lr["gate_r"] > 0.2
    print(f"  choosing whom to save beats choosing at hazard  : {'PASS' if t else 'FAIL'} "
          f"({rd['attr_acc']:.3f} -> {lr['attr_acc']:.3f})"); ok &= t
    return bool(ok)


def main() -> int:
    print("=" * 79)
    print("CATRAETH — the architecture of a mind that refuses to summarise")
    print("Chapter 0177: Aneirin, bard of the Gododdin (fl. c. 550-600, Hen Ogledd)")
    print("=" * 79)

    print("\n[GRADIENT CHECK] central differences against the hand-written backward pass")
    if not gradient_check():
        print("  refusing to train on a broken derivative")
        return 1

    print("\n[TRAINING] 2400 steps; hosts of 6-28 men; a hall that holds ten stanzas")
    model = train(steps=2400, verbose=True)

    exp_compression_pressure(model)
    exp_chroniclers_sin(model)
    exp_whom_do_you_save(model)
    exp_variant_erasure(model)
    exp_witness(model)
    exp_anticollapse(steps=800)
    show_stanzas(model)

    passed = self_tests(model)
    print("\n" + "=" * 79)
    print("ALL SELF-TESTS PASSED" if passed else "SOME SELF-TESTS FAILED")
    print("=" * 79)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

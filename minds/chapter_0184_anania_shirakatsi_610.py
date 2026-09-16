#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0184_anania_shirakatsi_610 - Anania Shirakatsi (Anania of Shirak, c. 610 – c. 685)
================================================================================    
ShirakNet — THE RESIDUE LEDGER
A trainable, executable, pure-NumPy architecture that embodies the cognitive
signature of Anania Shirakatsi (Anania of Shirak, c. 610 – c. 685), the first
Armenian mathematician, cosmographer and calendar-maker.

WHY THIS ARCHITECTURE (the mind it encodes)
-------------------------------------------
Every one of the twenty-four problems in Anania's *Book of Arithmetic* has the
same shape.  A whole is never observed.  What is observed is (a) a ledger of
NAMED FRACTIONAL TAKINGS — "a half and a quarter", "an eleventh", "a seventh of
the remainder" — and (b) the RESIDUE that is left at the end: 280 Persian
horsemen who fled to Nakhchavan, 45 fish left for the hand net, 570 prisoners
kept, 5 apples that reached the teacher.  The task is always: recover the
whole exactly.  Anania chose his numbers so that the whole is always an
integer — the world he teaches in is *reconstructible by design*.

His autobiography is the same shape in the first person: Armenia had only a
residue of learning ("not even books of the sciences could be found"); he
travelled to where the whole existed (Tychikos' library at Trebizond, which
had "every book to hand"); and he complained of pupils who took a fraction,
left, and taught that fraction as if it were the whole.  His arithmetic
tables enumerate every sum and product a·10^m ± b·10^n so nothing has to be
guessed at the moment of use; his 532-year Easter cycle enumerates every
feast so nothing drifts; his numerals write 50 as "5 × 10" (multiplicative-
additive) so that a number is itself a ledger of digit-times-power.

So the mind is:  OBSERVE THE RESIDUE → READ THE LEDGER OF TAKINGS → INVERT
EXACTLY → GROUND TO THE COMPLETE TABLE → CERTIFY COVERAGE BEFORE TEACHING.

THE NOVEL NEURON: the LEDGER CELL
---------------------------------
State S_k ∈ (0,1] is the SURVIVING FRACTION of an unobserved whole after k
takings.  Each taking arrives as a share a_k (how much is named) and a mode
gate m_k (0 = "of the whole", additive; 1 = "of the remainder", multiplicative):

        S_0 = 1
        S_k = S_{k-1} − a_k · ( (1 − m_k) + m_k · S_{k-1} )

The whole is recovered from the residue r by   log W = log r − log S_K.
Its hand-derived gradients (dS_k/dS_{k-1} = 1 − a_k m_k,  dS_k/da_k = −((1−m_k)
+ m_k S_{k-1}),  dS_k/dm_k = −a_k (S_{k-1} − 1)) are checked below against
both the tape-based autodiff and central finite differences.

THE FULL ARCHITECTURE (all pure NumPy, all trainable end to end)
----------------------------------------------------------------
  1. NUMERAL READER ("the Table"): reads the residue written in Anania's own
     multiplicative-additive notation — pairs (digit-letter, power-letter) —
     with LEARNED digit values and LEARNED power values.  Nothing tells the
     network that Ե means 5 or Ռ means 1000; it learns the table from the
     problems alone.
  2. NAMED-PART READER: every word in a taking has a learned quantity gate
     q(tok) and a learned magnitude v(tok).  The share of a taking is
     a_k = Σ q·v over its words.  "eleventh" must learn ≈ 1/11; "Marmet",
     "Kamsarakan", "fish" must learn to carry no quantity.
  3. PHRASE READER (GRU): a gated recurrent unit reads each taking phrase and
     decides the mode gate m_k — whether the taking was "of the pearls" or
     "of the remaining pearls".
  4. LEDGER CELL (the novel neuron, above).
  5. RECOVERY HEAD: log W = log r − log S_K, trained with squared log error.
  6. TABLE SNAP (inference): each learned magnitude is grounded to the nearest
     entry of the complete unit-fraction table 1/2 … 1/100 and the ledger is
     re-run in exact rationals — Anania's continuous estimate snapped to his
     enumerated table gives the exact integer answer.
  7. COVERAGE CERTIFICATE + INTEGRALITY CHECK: the model counts how many times
     it has been taught each named part and refuses to *teach* (emit a label
     for a student) any problem whose parts it has not mastered or whose
     reconstructed whole is not a whole number.
  8. LINEAGE EXPERIMENT ("the half-taught teacher"): teacher → student →
     grandstudent, with and without the certificate, measuring how a fraction
     that calls itself the whole propagates down a lineage.
  9. PASTIME INVERTER: the feast-trick from the Xraxčanakank' (double it, add
     five, times five, add ten, times ten, add the cups of wine): a module that
     learns the affine ledger and undoes it — the same principle, inversion of
     a known chain of operations, in its recreational form.

RUN
---
    python 0184_Anania_Shirakatsi_610_Neuron.py            # full run (~1–3 min)
    python 0184_Anania_Shirakatsi_610_Neuron.py --quick    # shorter training

Exits non-zero if any self-test (including the mandatory finite-difference
gradient check) fails.
"""
from __future__ import annotations
import sys, math, time
from fractions import Fraction
import numpy as np

# =============================================================================
# 0.  A minimal reverse-mode autodiff tape (pure NumPy)
# =============================================================================
class Node:
    """A value in the computation graph.  v: ndarray, g: gradient (filled in
    backward), parents: tuple of Nodes, bw: closure that distributes the
    incoming gradient to the parents."""
    __slots__ = ("v", "g", "parents", "bw")

    def __init__(self, v, parents=(), bw=None):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = None
        self.parents = parents
        self.bw = bw

    @property
    def shape(self):
        return self.v.shape

    # operator sugar ---------------------------------------------------------
    def __add__(self, o):  return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o):  return sub(self, o)
    def __rsub__(self, o): return sub(o, self)
    def __mul__(self, o):  return mul(self, o)
    def __rmul__(self, o): return mul(o, self)
    def __truediv__(self, o):  return div(self, o)
    def __rtruediv__(self, o): return div(o, self)
    def __neg__(self):     return neg(self)
    def __matmul__(self, o): return matmul(self, o)


def _n(x):
    return x if isinstance(x, Node) else Node(x)


def _acc(node, g):
    if node.g is None:
        node.g = np.zeros_like(node.v)
    node.g += g


def _unbroadcast(g, shape):
    """Sum a gradient down to `shape` (undo NumPy broadcasting)."""
    g = np.asarray(g)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def add(a, b):
    a, b = _n(a), _n(b)
    out = Node(a.v + b.v, (a, b))
    def bw(g):
        _acc(a, _unbroadcast(g, a.shape)); _acc(b, _unbroadcast(g, b.shape))
    out.bw = bw
    return out


def sub(a, b):
    a, b = _n(a), _n(b)
    out = Node(a.v - b.v, (a, b))
    def bw(g):
        _acc(a, _unbroadcast(g, a.shape)); _acc(b, _unbroadcast(-g, b.shape))
    out.bw = bw
    return out


def mul(a, b):
    a, b = _n(a), _n(b)
    out = Node(a.v * b.v, (a, b))
    def bw(g):
        _acc(a, _unbroadcast(g * b.v, a.shape)); _acc(b, _unbroadcast(g * a.v, b.shape))
    out.bw = bw
    return out


def div(a, b):
    a, b = _n(a), _n(b)
    out = Node(a.v / b.v, (a, b))
    def bw(g):
        _acc(a, _unbroadcast(g / b.v, a.shape))
        _acc(b, _unbroadcast(-g * a.v / (b.v ** 2), b.shape))
    out.bw = bw
    return out


def neg(a):
    a = _n(a)
    out = Node(-a.v, (a,))
    out.bw = lambda g: _acc(a, -g)
    return out


def matmul(a, b):
    a, b = _n(a), _n(b)
    out = Node(a.v @ b.v, (a, b))
    def bw(g):
        _acc(a, g @ b.v.T); _acc(b, a.v.T @ g)
    out.bw = bw
    return out


def exp(a):
    a = _n(a); ev = np.exp(a.v)
    out = Node(ev, (a,))
    out.bw = lambda g: _acc(a, g * ev)
    return out


def log(a):
    a = _n(a)
    out = Node(np.log(a.v), (a,))
    out.bw = lambda g: _acc(a, g / a.v)
    return out


def tanh(a):
    a = _n(a); t = np.tanh(a.v)
    out = Node(t, (a,))
    out.bw = lambda g: _acc(a, g * (1.0 - t * t))
    return out


def sigmoid(a):
    a = _n(a); s = 1.0 / (1.0 + np.exp(-a.v))
    out = Node(s, (a,))
    out.bw = lambda g: _acc(a, g * s * (1.0 - s))
    return out


def softplus(a):
    a = _n(a)
    sp = np.where(a.v > 30, a.v, np.log1p(np.exp(np.minimum(a.v, 30))))
    out = Node(sp, (a,))
    out.bw = lambda g: _acc(a, g / (1.0 + np.exp(-a.v)))
    return out


def ssum(a, axis=None, keepdims=False):
    a = _n(a)
    out = Node(a.v.sum(axis=axis, keepdims=keepdims), (a,))
    def bw(g):
        g = np.asarray(g)
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        _acc(a, np.broadcast_to(g, a.shape).copy())
    out.bw = bw
    return out


def mean(a):
    a = _n(a)
    return ssum(a) * (1.0 / a.v.size)


def gather(table, idx):
    """table: Node (V,) or (V,D); idx: int ndarray of any shape.
    Returns table[idx] with shape idx.shape (+ (D,))."""
    idx = np.asarray(idx)
    out = Node(table.v[idx], (table,))
    def bw(g):
        gt = np.zeros_like(table.v)
        np.add.at(gt, idx, g)
        _acc(table, gt)
    out.bw = bw
    return out


def reshape(a, shape):
    a = _n(a)
    out = Node(a.v.reshape(shape), (a,))
    out.bw = lambda g: _acc(a, np.asarray(g).reshape(a.shape))
    return out


def col(a, k):
    """Column k of a 2-D node → 1-D node."""
    a = _n(a)
    out = Node(a.v[:, k].copy(), (a,))
    def bw(g):
        ga = np.zeros_like(a.v); ga[:, k] = g; _acc(a, ga)
    out.bw = bw
    return out


def square(a):
    return mul(a, a)


def huber(e):
    """Pseudo-Huber: 2·(sqrt(1+e²) − 1) — equals e² near zero, 2|e| far away."""
    return (sqrt(square(e) + 1.0) - 1.0) * 2.0


def sqrt(a):
    a = _n(a); s = np.sqrt(a.v)
    out = Node(s, (a,))
    out.bw = lambda g: _acc(a, g * 0.5 / s)
    return out


def posfloor(x, delta=1e-3):
    """Positivity barrier for the surviving fraction.  Identity for x > delta;
    below it, delta·exp((x−delta)/delta) — C¹-continuous, strictly positive,
    and with a gradient that GROWS (1/delta) instead of vanishing, so a ledger
    that momentarily over-subtracts is pushed back rather than left stranded
    (a plain clamp silently kills the gradient and traps training)."""
    x = _n(x)
    below = x.v <= delta
    v = np.where(below, delta * np.exp(np.minimum((x.v - delta) / delta, 0.0)), x.v)
    out = Node(v, (x,))
    dv = np.where(below, v / delta, 1.0)
    out.bw = lambda g: _acc(x, g * dv)
    return out


def backward(root):
    """Reverse-mode sweep from `root` (a scalar Node)."""
    order, seen = [], set()
    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n.parents:
            visit(p)
        order.append(n)
    visit(root)
    root.g = np.ones_like(root.v)
    for n in reversed(order):
        if n.bw is not None and n.g is not None:
            n.bw(n.g)


class Adam:
    """Adam with an optional per-parameter learning-rate multiplier (so the
    Numeral Reader can absorb the global unit quickly while the named parts
    move carefully)."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, lr_mult=None):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(p.v) for p in params]
        self.s = [np.zeros_like(p.v) for p in params]
        self.mult = list(lr_mult) if lr_mult is not None else [1.0] * len(params)
        self.t = 0

    def step(self, clip=10.0):
        self.t += 1
        gn = math.sqrt(sum(float((p.g ** 2).sum()) for p in self.p if p.g is not None))
        scale = min(1.0, clip / (gn + 1e-12)) if clip else 1.0
        for i, p in enumerate(self.p):
            if p.g is None or self.mult[i] == 0.0:
                continue
            g = p.g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            sh = self.s[i] / (1 - self.b2 ** self.t)
            p.v -= self.lr * self.mult[i] * mh / (np.sqrt(sh) + self.eps)

    def zero(self):
        for p in self.p:
            p.g = None


# =============================================================================
# 1.  Vocabulary: Anania's named parts, his numerals, and his narrative words
# =============================================================================
FRACTION_DENOMS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17, 19, 20, 22, 26, 28, 30, 90]
FRACTION_NAMES = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
                  7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
                  11: "eleventh", 12: "twelfth", 13: "thirteenth",
                  14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
                  17: "seventeenth", 19: "nineteenth", 20: "twentieth",
                  22: "twenty-second", 26: "twenty-sixth", 28: "twenty-eighth",
                  30: "thirtieth", 90: "ninetieth"}
# tokens that carry no quantity — the narrative of Anania's problems
DISTRACTORS = ["kamsarakan", "marmet", "akhuryan", "dvin", "nakhchavan",
               "fish", "pearls", "prisoners", "apples", "lettuce", "boar",
               "wine", "bricks", "captives", "asses", "hunt", "river", "net",
               "sold", "gave", "ate", "took", "levied"]
SPECIAL = ["<pad>", "<and>", "<of_whole>", "<of_rest>"]
VOCAB = SPECIAL + [FRACTION_NAMES[d] for d in FRACTION_DENOMS] + DISTRACTORS
TOK = {w: i for i, w in enumerate(VOCAB)}
FRAC_TOK = {d: TOK[FRACTION_NAMES[d]] for d in FRACTION_DENOMS}
PAD, AND, OF_WHOLE, OF_REST = TOK["<pad>"], TOK["<and>"], TOK["<of_whole>"], TOK["<of_rest>"]

# Anania's numerals: multiplicative-additive — a number is a ledger of
# (digit-letter × power-letter).  Ա..Թ = 1..9 ; Ժ = 10, Ճ = 100, Ռ = 1000, then
# the myriad (բիւր, 10,000) and ten-myriads.  The network learns their values.
DIGIT_LETTERS = ["Ա", "Բ", "Գ", "Դ", "Ե", "Զ", "Է", "Ը", "Թ"]        # 1..9
POWER_LETTERS = ["·", "Ժ", "Ճ", "Ռ", "բիւր", "Ժբիւր"]                 # 1,10,..,10^5
N_DIGITS, N_POWERS = 9, 6

# padding sizes for batching
K_MAX, T_MAX, N_MAX = 8, 8, 6


def numeral_tokens(n: int):
    """Write n in Anania's notation: list of (digit_idx, power_idx)."""
    assert 1 <= n < 10 ** N_POWERS, n
    out = []
    p = 0
    while n > 0:
        d = n % 10
        if d:
            out.append((d - 1, p))
        n //= 10
        p += 1
    return out[::-1]


def numeral_str(n: int):
    return " ".join(DIGIT_LETTERS[d] + (POWER_LETTERS[p] if p else "")
                    for d, p in numeral_tokens(n))


# =============================================================================
# 2.  Problem generator — exact rationals, integer wholes, integer residues
# =============================================================================
class Problem:
    __slots__ = ("takings", "whole", "residue", "text")

    def __init__(self, takings, whole, residue, text=""):
        self.takings = takings      # list of (denoms:list[int], of_rest:bool, tokens:list[int])
        self.whole = whole
        self.residue = residue
        self.text = text


def survival(takings):
    """Exact surviving fraction after the ledger of takings."""
    S = Fraction(1)
    for denoms, of_rest, _ in takings:
        a = sum(Fraction(1, d) for d in denoms)
        S = S - a * (S if of_rest else 1)
    return S


def make_problem(rng, denom_pool, rare=None, rare_prob=0.0, n_distract=(0, 3),
                 max_whole=200000, force_denoms=None):
    """Sample one Anania-style problem.  `denom_pool` lists denominators
    available; `rare` denominators are drawn with probability rare_prob."""
    for _attempt in range(200):
        K = int(rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 7, 8]))
        takings = []
        for k in range(K):
            nf = rng.integers(1, 4)
            pool = list(denom_pool)
            chosen = []
            if force_denoms and k < len(force_denoms):
                chosen = list(force_denoms[k])
            else:
                while len(chosen) < nf:
                    if rare and rng.random() < rare_prob:
                        d = int(rng.choice(rare))
                    else:
                        d = int(rng.choice(pool))
                    if d not in chosen:
                        chosen.append(d)
            of_rest = bool(rng.random() < 0.5)
            takings.append((chosen, of_rest, None))
        S = survival(takings)
        if not (Fraction(1, 50) <= S < 1):
            continue
        # the whole must make every intermediate remainder an integer:
        need = 1
        S_run = Fraction(1)
        for denoms, of_rest, _ in takings:
            a = sum(Fraction(1, d) for d in denoms)
            S_run = S_run - a * (S_run if of_rest else 1)
            need = need * S_run.denominator // math.gcd(need, S_run.denominator)
        if need > max_whole:
            continue
        mult = int(rng.integers(1, max(2, max_whole // need + 1)))
        whole = need * mult
        residue = whole * S
        assert residue.denominator == 1
        residue = int(residue)
        if residue < 1:
            continue
        # tokenise: [frac (<and> frac)*] <of_rest|of_whole> + distractors shuffled in
        toks_all = []
        for denoms, of_rest, _ in takings:
            toks = []
            for i, d in enumerate(denoms):
                if i and rng.random() < 0.5:       # "a half and a quarter" / "a half, a quarter"
                    toks.append(AND)
                toks.append(FRAC_TOK[d])
            toks.append(OF_REST if of_rest else OF_WHOLE)
            nd = int(rng.integers(n_distract[0], n_distract[1] + 1))
            nd = min(nd, T_MAX - len(toks))
            for _ in range(nd):
                pos = int(rng.integers(0, len(toks) + 1))
                toks.insert(pos, TOK[str(rng.choice(DISTRACTORS))])
            toks_all.append(toks)
        takings = [(d, r, t) for (d, r, _), t in zip(takings, toks_all)]
        return Problem(takings, whole, residue)
    raise RuntimeError("could not sample a problem")


def batchify(problems):
    """Pad a list of Problems into index/mask arrays."""
    B = len(problems)
    tok = np.full((B, K_MAX, T_MAX), PAD, dtype=np.int64)
    tmask = np.zeros((B, K_MAX, T_MAX))
    kmask = np.zeros((B, K_MAX))
    dig = np.zeros((B, N_MAX), dtype=np.int64)
    pw = np.zeros((B, N_MAX), dtype=np.int64)
    nmask = np.zeros((B, N_MAX))
    logW = np.zeros(B)
    for i, p in enumerate(problems):
        for k, (_, _, t) in enumerate(p.takings):
            tok[i, k, :len(t)] = t
            tmask[i, k, :len(t)] = 1.0
            kmask[i, k] = 1.0
        for j, (d, q) in enumerate(numeral_tokens(p.residue)):
            dig[i, j] = d; pw[i, j] = q; nmask[i, j] = 1.0
        logW[i] = math.log(p.whole)
    return dict(tok=tok, tmask=tmask, kmask=kmask, dig=dig, pw=pw,
                nmask=nmask, logW=logW)


# =============================================================================
# 3.  The model
# =============================================================================
class ShirakNet:
    """Numeral Reader + Named-Part Reader + Phrase Reader (GRU) + Ledger Cell
    + Recovery Head.  All parameters are Nodes; forward() builds the tape."""

    def __init__(self, rng, emb_dim=12, hidden=16):
        V = len(VOCAB)
        self.rng = rng
        self.D, self.H = emb_dim, hidden
        # --- Named-Part Reader: magnitude v(tok)=exp(theta) (log-space, so every
        #     fraction moves by the same RELATIVE step), gate q(tok)=sigmoid(phi)
        self.theta = Node(np.full(V, math.log(0.03)) + 0.05 * rng.standard_normal(V))   # v = exp(theta)
        self.phi = Node(np.zeros(V))
        # --- Numeral Reader: digit values softplus(dl), power values exp(pl)
        self.dl = Node(np.log(np.expm1(np.linspace(0.5, 1.5, N_DIGITS))) + 0.1 * rng.standard_normal(N_DIGITS))
        self.pl = Node(np.linspace(0.0, 2.0, N_POWERS) + 0.1 * rng.standard_normal(N_POWERS))
        # --- Phrase Reader (GRU) for the mode gate
        s = 0.4 / math.sqrt(emb_dim)
        self.E = Node(s * rng.standard_normal((V, emb_dim)))
        def W(i, o): return Node((0.6 / math.sqrt(i)) * rng.standard_normal((i, o)))
        self.Wz, self.Uz, self.bz = W(emb_dim, hidden), W(hidden, hidden), Node(np.zeros((1, hidden)))
        self.Wr, self.Ur, self.br = W(emb_dim, hidden), W(hidden, hidden), Node(np.zeros((1, hidden)))
        self.Wn, self.Un, self.bn = W(emb_dim, hidden), W(hidden, hidden), Node(np.zeros((1, hidden)))
        self.Wm, self.bm = W(hidden, 1), Node(np.zeros((1, 1)))
        # marker path: every word casts a learned vote mu(tok) on the mode
        # ("of the remaining" vs "of the pearls"); the GRU refines it in context
        self.mu = Node(np.zeros(V))
        # coverage ledger: how many times each token has been taught
        self.coverage = np.zeros(V, dtype=np.int64)
        # positivity barrier of the ledger: wide while learning (a clumsy pupil
        # over-subtracts), narrow at inference (a master's ledger stays positive)
        self.delta = 1e-2

    def params(self):
        return [self.theta, self.phi, self.dl, self.pl, self.E,
                self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br,
                self.Wn, self.Un, self.bn, self.Wm, self.bm, self.mu]

    def param_names(self):
        return ["theta(part magnitudes)", "phi(quantity gates)", "dl(digit values)",
                "pl(power values)", "E(embeddings)", "Wz", "Uz", "bz", "Wr", "Ur", "br",
                "Wn", "Un", "bn", "Wm(mode head)", "bm", "mu(mode votes)"]

    # ---- pieces --------------------------------------------------------------
    def numeral_reader(self, dig, pw, nmask):
        """log r̂ from the residue written as (digit, power) pairs."""
        dv = gather(softplus(self.dl), dig)          # (B,N)
        pv = gather(exp(self.pl), pw)                 # (B,N)
        val = ssum(dv * pv * nmask, axis=1)           # (B,)
        return log(val)

    def named_part_reader(self, tok, tmask):
        """Share a_k of each taking = Σ_t q(tok)·v(tok) over its words."""
        v = gather(exp(self.theta), tok)              # (B,K,T)
        q = gather(sigmoid(self.phi), tok)            # (B,K,T)
        return ssum(v * q * tmask, axis=2)            # (B,K)

    def phrase_reader(self, tok, tmask):
        """GRU over each taking's words → mode gate m_k ∈ (0,1)."""
        B, K, T = tok.shape
        R = B * K
        flat = tok.reshape(R, T)
        fm = tmask.reshape(R, T)
        h = Node(np.zeros((R, self.H)))
        for t in range(T):
            x = gather(self.E, flat[:, t])            # (R,D)
            z = sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
            r = sigmoid(x @ self.Wr + h @ self.Ur + self.br)
            n = tanh(x @ self.Wn + (r * h) @ self.Un + self.bn)
            h_new = (1.0 - z) * n + z * h
            m = fm[:, t:t + 1]
            h = h_new * m + h * (1.0 - m)             # freeze after last real word
        votes = ssum(gather(self.mu, flat) * fm, axis=1, keepdims=True)   # (R,1)
        mode = sigmoid(h @ self.Wm + self.bm + votes)                       # (R,1)
        return reshape(mode, (B, K))

    def ledger_cell(self, a, m, kmask):
        """THE LEDGER CELL.  S_k = S_{k-1} − a_k·((1−m_k) + m_k·S_{k-1}),
        masked so absent takings leave S unchanged.  Returns log S_K."""
        B = a.shape[0]
        S = Node(np.ones(B))
        for k in range(K_MAX):
            ak, mk, ck = col(a, k), col(m, k), kmask[:, k]
            take = ak * ((1.0 - mk) + mk * S)
            S = posfloor(S - take * ck, self.delta)
        return log(S)

    def forward(self, batch):
        logr = self.numeral_reader(batch["dig"], batch["pw"], batch["nmask"])
        a = self.named_part_reader(batch["tok"], batch["tmask"])
        m = self.phrase_reader(batch["tok"], batch["tmask"])
        logS = self.ledger_cell(a, m, batch["kmask"])
        logW = logr - logS
        return logW, a, m, logr

    def loss(self, batch):
        """Huber loss on the log of the recovered whole (quadratic within one
        nat, linear beyond, so one wildly mis-read problem cannot hijack the
        batch) — reported as mean squared log error while inside 1 nat."""
        logW, *_ = self.forward(batch)
        err = logW - Node(batch["logW"])
        # smooth Huber: h(e) = sqrt(1+e²) − 1  (≈ e²/2 near 0, ≈ |e| far away)
        h = huber(err)
        # Occam prior of the reader: a word carries no quantity unless the
        # problems demand it (tiny L1 on the effective magnitudes q·v)
        prior = ssum(exp(self.theta) * sigmoid(self.phi)) * 2e-5
        return mean(h) + prior, logW

    # ---- inference helpers -------------------------------------------------
    def predict(self, problems):
        b = batchify(problems)
        d, self.delta = self.delta, 1e-4
        logW, a, m, logr = self.forward(b)
        self.delta = d
        return np.exp(logW.v), a.v, m.v, np.exp(logr.v)

    def learned_fraction_table(self):
        v = np.exp(self.theta.v) * (1 / (1 + np.exp(-self.phi.v)))
        return {d: v[FRAC_TOK[d]] for d in FRACTION_DENOMS}

    def table_snap(self, problem):
        """Ground each learned magnitude to the nearest entry of the complete
        table of unit fractions and re-run the ledger in exact rationals."""
        vt = self.learned_fraction_table()
        gate_mode = self.predict([problem])[2][0]
        S = Fraction(1)
        for k, (denoms, of_rest, toks) in enumerate(problem.takings):
            a = Fraction(0)
            for t in toks:
                if t in FRAC_TOK.values():
                    d = [dd for dd, tt in FRAC_TOK.items() if tt == t][0]
                    est = vt[d]
                    n = int(round(1.0 / max(est, 1e-6)))
                    n = min(max(n, 2), 100)
                    a += Fraction(1, n)
            mode_rest = gate_mode[k] > 0.5
            S = S - a * (S if mode_rest else 1)
        if S <= 0:
            return None
        W = Fraction(problem.residue) / S
        return W

    # ---- the certificate ---------------------------------------------------
    def record_coverage(self, problems):
        for p in problems:
            for _, _, toks in p.takings:
                for t in toks:
                    self.coverage[t] += 1

    def certify(self, problem, min_cov=25, int_tol=0.02):
        """May this model TEACH this problem?  Only if (a) every word of the
        problem has been taught at least `min_cov` times — a word never taught
        might name a part, and a pupil who silently drops the parts he never
        learned is exactly the half-taught teacher — and (b) the reconstructed
        whole is a whole number (relative distance to the nearest integer
        < int_tol), Anania's own check that a problem was well posed."""
        for _, _, toks in problem.takings:
            for t in toks:
                if self.coverage[t] < min_cov:
                    return False, "unknown word: %s" % VOCAB[t]
        W = self.predict([problem])[0][0]
        if abs(W - round(W)) / max(W, 1.0) > int_tol:
            return False, "not a whole number"
        return True, "certified"


# =============================================================================
# 4.  Training
# =============================================================================
def train_numerals(model, n=4000, epochs=40, batch=64, lr=2e-1, verbose=True, name="table"):
    """STAGE 1 — the Table before the Problems.  The Numeral Reader is grounded
    by counting: a number written in Anania's letters is paired with the count
    it names, and the digit/power values are learned by regression on log
    value.  Only dl and pl move here."""
    rng = model.rng
    # log-uniform sampling: every power of ten is taught equally, as in
    # Anania's tables, which list a·10^n for each n rather than favouring
    # large numbers.
    nums = np.maximum(1, np.exp(rng.uniform(0.0, math.log(10 ** N_POWERS - 1), n)).astype(np.int64))
    opt = Adam([model.dl, model.pl], lr=lr)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(nums)
        tot = 0.0
        for s in range(0, n, batch):
            chunk = nums[s:s + batch]
            B = len(chunk)
            dig = np.zeros((B, N_MAX), dtype=np.int64); pw = np.zeros((B, N_MAX), dtype=np.int64)
            nm = np.zeros((B, N_MAX))
            for i, v in enumerate(chunk):
                for j, (d, q) in enumerate(numeral_tokens(int(v))):
                    dig[i, j] = d; pw[i, j] = q; nm[i, j] = 1.0
            opt.zero()
            logr = model.numeral_reader(dig, pw, nm)
            L = mean(square(logr - Node(np.log(chunk.astype(float)))))
            backward(L); opt.step()
            tot += L.v * B
        if verbose and (ep == 1 or ep == epochs):
            print("  [%s] epoch %3d  numeral loss(log²) = %.2e   (%.1fs)" % (name, ep, tot / n, time.time() - t0))


def train(model, problems, epochs=20, batch=32, lr=1e-2, log_every=5, name="model",
          targets=None, verbose=True):
    """STAGE 2 — the Problems.  Adam on squared log error of the recovered
    whole.  `targets` overrides log W with a teacher's labels (lineage).  The
    numeral table learned in stage 1 is held fixed, exactly as a pupil solves
    problems with the table he has already memorised."""
    names = model.param_names()
    # numerals frozen (stage 1); the named-part table and the mode votes are
    # plain tables and may move 5x faster than the recurrent phrase reader
    mult = [0.0 if n.startswith(("dl", "pl")) else (3.0 if n.startswith(("theta", "phi", "mu")) else 1.0)
            for n in names]
    opt = Adam(model.params(), lr=lr, lr_mult=mult)
    n = len(problems)
    all_idx = np.arange(n)
    nk = np.array([len(p.takings) for p in problems])
    # Anania's order: single takings first (the share table), then two
    # (where "of the remainder" first matters), then the long chains.
    short_idx = all_idx[nk <= 2]
    t0 = time.time()
    for ep in range(1, epochs + 1):
        idx = short_idx if (ep <= max(1, epochs // 4) and len(short_idx) > 0) else all_idx
        # cosine decay of the learning rate over the second half of training,
        # so the table settles to its exact values instead of jittering
        frac = ep / epochs
        opt.lr = lr * (1.0 if frac < 0.5 else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (frac - 0.5) / 0.5)))
        model.rng.shuffle(idx)
        nn = len(idx)
        tot = 0.0
        for s in range(0, nn, batch):
            bi = idx[s:s + batch]
            b = batchify([problems[i] for i in bi])
            if targets is not None:
                b["logW"] = targets[bi]
            opt.zero()
            L, _ = model.loss(b)
            backward(L)
            opt.step()
            tot += L.v * len(bi)
        if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
            print("  [%s] epoch %3d  loss(log²) = %.6f  on %d problems (K%s)  (%.1fs)"
                  % (name, ep, tot / nn, nn, "≤2" if idx is short_idx else "≤8", time.time() - t0))
    model.record_coverage(problems)
    return model


def evaluate(model, problems, tol=0.01):
    W_hat = model.predict(problems)[0]
    W = np.array([p.whole for p in problems], dtype=float)
    within = np.abs(W_hat - W) / W < tol
    exact = np.round(W_hat) == W
    return within.mean(), exact.mean(), W_hat


def snap_accuracy(model, problems):
    ok = 0
    for p in problems:
        Wsnap = model.table_snap(p)
        if Wsnap is not None and Wsnap == p.whole:
            ok += 1
    return ok / len(problems)


# =============================================================================
# 5.  Gradient checks (mandatory)
# =============================================================================
def finite_difference_check(model, batch, n_coords=6, eps=1e-4, tol=1e-5, rng=None):
    """Central finite differences vs. tape gradients for every parameter.
    Error = |num − ana| / max(|num| + |ana|, 1e-4): relative for every
    gradient that matters, absolute (≤1e-9) for gradients at numerical-noise
    level, where a pure ratio would only measure round-off."""
    rng = rng or np.random.default_rng(0)
    for p in model.params():
        p.g = None
    L, _ = model.loss(batch)
    backward(L)
    worst = 0.0
    report = []
    for name, p in zip(model.param_names(), model.params()):
        flat = p.v.reshape(-1)
        gflat = p.g.reshape(-1)
        coords = rng.choice(flat.size, size=min(n_coords, flat.size), replace=False)
        for c in coords:
            old = flat[c]
            flat[c] = old + eps; Lp = model.loss(batch)[0].v
            flat[c] = old - eps; Lm = model.loss(batch)[0].v
            flat[c] = old
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[c]
            rel = abs(num - ana) / max(1e-4, abs(num) + abs(ana))
            worst = max(worst, rel)
            report.append((name, int(c), ana, num, rel))
    return worst, report


def ledger_cell_hand_gradient_check():
    """Hand-derived gradients of the Ledger Cell vs. tape vs. finite differences."""
    rng = np.random.default_rng(3)
    K = 5
    a = rng.uniform(0.02, 0.3, K); m = rng.uniform(0, 1, K)
    # forward
    S = [1.0]
    for k in range(K):
        S.append(S[-1] - a[k] * ((1 - m[k]) + m[k] * S[-1]))
    # hand backward of L = log S_K
    dS = 1.0 / S[-1]
    da_h = np.zeros(K); dm_h = np.zeros(K)
    for k in reversed(range(K)):
        da_h[k] = dS * (-((1 - m[k]) + m[k] * S[k]))
        dm_h[k] = dS * (-a[k] * (S[k] - 1.0))
        dS = dS * (1.0 - a[k] * m[k])
    # tape
    an, mn = Node(a), Node(m)
    Sn = Node(np.ones(1))
    for k in range(K):
        ak = col(reshape(an, (1, K)), k); mk = col(reshape(mn, (1, K)), k)
        Sn = Sn - ak * ((1.0 - mk) + mk * Sn)
    Ln = ssum(log(Sn)); backward(Ln)
    # finite differences
    def f(a_, m_):
        s = 1.0
        for k in range(K):
            s = s - a_[k] * ((1 - m_[k]) + m_[k] * s)
        return math.log(s)
    da_fd = np.zeros(K); dm_fd = np.zeros(K); e = 1e-6
    for k in range(K):
        ap = a.copy(); ap[k] += e; am = a.copy(); am[k] -= e
        da_fd[k] = (f(ap, m) - f(am, m)) / (2 * e)
        mp = m.copy(); mp[k] += e; mm = m.copy(); mm[k] -= e
        dm_fd[k] = (f(a, mp) - f(a, mm)) / (2 * e)
    err = max(np.abs(da_h - an.g).max(), np.abs(dm_h - mn.g).max(),
              np.abs(da_h - da_fd).max(), np.abs(dm_h - dm_fd).max())
    return err, S[-1]


# =============================================================================
# 6.  The Pastime Inverter (Xraxčanakank', puzzle 1)
# =============================================================================
class PastimeInverter:
    """Anania: "keep the hour in mind; double it, add 5, multiply by 5, add 10,
    multiply by 10, add the cups of wine; tell me the number" → y = 100h+350+g.
    The trick is knowing the affine ledger; this module LEARNS it (α, β) from
    examples and then undoes it: h = round((y−β)/α), g = y − β − α h."""

    def __init__(self, rng):
        self.alpha = Node(np.array([50.0 + rng.standard_normal()]))
        self.beta = Node(np.array([100.0 + rng.standard_normal()]))
        self.rng = rng

    @staticmethod
    def trick(h, g):
        return ((2 * h + 5) * 5 + 10) * 10 + g

    def fit(self, n=400, steps=400, lr=0.5):
        h = self.rng.integers(1, 13, n); g = self.rng.integers(0, 100, n)
        y = self.trick(h, g).astype(float)
        target = y - g                     # = α·h + β  (the guest's number minus the cups)
        hc = h.astype(float) - h.mean()    # centre h so α and β decouple
        tc = target - target.mean()
        self.b0 = Node(np.array([0.0]))    # β' ; β = mean(target) − α·mean(h) + β'
        opt = Adam([self.alpha, self.b0], lr=lr)
        for _ in range(steps):
            opt.zero()
            pred = self.alpha * Node(hc) + self.b0
            L = mean(square(pred - Node(tc))) * (1.0 / 100.0)
            backward(L); opt.step(clip=None)
        self.beta.v[:] = target.mean() - self.alpha.v[0] * h.mean() + self.b0.v[0]
        return float(self.alpha.v[0]), float(self.beta.v[0])

    def undo(self, y):
        h = int(round((y - self.beta.v[0]) / self.alpha.v[0]))
        g = int(round(y - self.beta.v[0] - self.alpha.v[0] * h))
        return h, g


# =============================================================================
# 7.  Anania's own problems (Book of Arithmetic, tr. Greenwood 2011 / Aslanyan 2024)
# =============================================================================
def anania_problems():
    """(name, takings as [(denoms, of_rest)], residue, Anania's answer)."""
    P = [
        ("1  Zorak Kamsarakan's three attacks (½, ¼, 1/11 of the troops; 280 fled)",
         [([2], False), ([4], False), ([11], False)], 280, 1760),
        ("2  The pearl merchant (½, ¼, 1/12 of the pearls sold; 24 left)",
         [([2], False), ([4], False), ([12], False)], 24, 144),
        ("6  The Roman in the lettuce garden (1/5 and 1/15; 110 left)",
         [([5, 15], False)], 110, 150),
        ("7  Fish in the Akhuryan at Marmet (½, ¼, 1/7 by cast net; 45 by hand net)",
         [([2], False), ([4], False), ([7], False)], 45, 420),
        ("9  The Kamsarakans' wild boar (¼ entrails, 1/10 head, 1/20 legs, 1/90 tusks; body 212)",
         [([4], False), ([10], False), ([20], False), ([90], False)], 212, 360),
        ("11 The merchant through three cities (½+⅓ of the rest, thrice; 11 left)",
         [([2, 3], True), ([2, 3], True), ([2, 3], True)], 11, 2376),
        ("12 The boat (⅓, ¼, ⅙, 1/7, 1/28 of the cost given by relatives; 3 drams)",
         [([3], False), ([4], False), ([6], False), ([7], False), ([28], False)], 3, 42),
        ("13 The student's apples and three bands of jesters (½+¼ of the rest, thrice; 5 left)",
         [([2, 4], True), ([2, 4], True), ([2, 4], True)], 5, 320),
        ("17 The ship and the whale (½; then 1/5, 1/8, 1/7 of the rest; 7200 baskets)",
         [([2], False), ([5], True), ([8], True), ([7], True)], 7200, 24000),
        ("20 Nerseh Kamsarakan's wild asses (½, ¼, 1/12 of all; 360 speared)",
         [([2], False), ([4], False), ([12], False)], 360, 2160),
        ("21 Nerseh's prisoners (½; 1/7,1/8,1/14,1/13,1/9,1/16,1/20 of the rest; 570 kept)",
         [([2], True), ([7], True), ([8], True), ([14], True), ([13], True),
          ([9], True), ([16], True), ([20], True)], 570, 2240),
        ("25 The sower of Arats and three ants (⅓, ¼, 1/26; 1239 seeds left)",
         [([3], False), ([4], False), ([26], False)], 1239, 3276),
    ]
    out = []
    for name, tk, res, ans in P:
        takings = []
        for denoms, of_rest in tk:
            toks = []
            for i, d in enumerate(denoms):
                if i:
                    toks.append(AND)
                toks.append(FRAC_TOK[d])
            toks.append(OF_REST if of_rest else OF_WHOLE)
            takings.append((denoms, of_rest, toks))
        pr = Problem(takings, ans, res, name)
        assert pr.whole * survival(takings) == res, name   # Anania's answer verified exactly
        out.append(pr)
    return out


# =============================================================================
# 8.  Main: self-tests, training, lineage experiment, demonstration
# =============================================================================
def main(quick=False):
    t_start = time.time()
    rng = np.random.default_rng(184)
    ok = True
    print("=" * 78)
    print("ShirakNet — the Residue Ledger of Anania Shirakatsi (0184)  |  pure NumPy")
    print("=" * 78)

    # --- self-test 1: Anania's twelve problems reproduce his answers exactly (rationals)
    A = anania_problems()
    print("\n[self-test 1] Anania's answers verified with exact rationals: %d/%d problems" % (len(A), len(A)))

    # --- self-test 2: the Ledger Cell's hand gradients
    err, S_last = ledger_cell_hand_gradient_check()
    print("[self-test 2] Ledger Cell: hand gradient vs tape vs finite differences, max |Δ| = %.2e" % err)
    ok &= err < 1e-6

    # --- self-test 3: mandatory finite-difference gradient check of the whole model
    model = ShirakNet(rng)
    common = [d for d in FRACTION_DENOMS if d not in (13, 26, 28, 90, 17, 19, 22, 30)]
    probe = [make_problem(rng, common) for _ in range(3)]
    worst, rep = finite_difference_check(model, batchify(probe), rng=rng)
    print("[self-test 3] Full-model gradient check (%d coordinates over %d tensors): worst rel. err = %.2e"
          % (len(rep), len(model.params()), worst))
    ok &= worst < 1e-5
    if not ok:
        for r in rep:
            if r[4] > 1e-5:
                print("     FAIL", r)

    # --- curriculum: common parts abundant, rare parts (13,26,28,90) ≈ 2–3 sightings,
    #     unseen parts (17,19,22,30) never taught to the teacher.
    rare = [13, 26, 28, 90]
    unseen = [17, 19, 22, 30]
    n_train = 2500 if quick else 5000
    epochs = 24 if quick else 60
    train_set = [make_problem(rng, common) for _ in range(n_train)]
    for d in rare:
        for _ in range(3):
            train_set.append(make_problem(rng, common, force_denoms=[[d, int(rng.choice(common))]]))
    rng.shuffle(train_set)
    test_common = [make_problem(rng, common) for _ in range(300)]
    test_unseen = [make_problem(rng, common, rare=unseen, rare_prob=0.6) for _ in range(150)]
    test_unseen = [p for p in test_unseen if any(d in unseen for dd, _, _ in p.takings for d in dd)]
    print("\n[curriculum] %d training problems (%d with a rare part), %d common test, %d unseen-part test"
          % (len(train_set), 3 * len(rare), len(test_common), len(test_unseen)))
    print("             sample: residue %s = %d ; whole %d" % (numeral_str(train_set[0].residue),
                                                             train_set[0].residue, train_set[0].whole))

    # --- train the teacher (generation 0)
    print("\n[training] generation 0 — the teacher (Tychikos' pupil)")
    teacher = ShirakNet(np.random.default_rng(1))
    train_numerals(teacher, epochs=(30 if quick else 40), name="gen0-table")
    train(teacher, train_set, epochs=epochs, lr=1e-2, name="gen0", log_every=4)
    w1, ex, _ = evaluate(teacher, test_common)
    snap = snap_accuracy(teacher, test_common)
    print("  common test: within 1%% = %.1f%%   exact-after-rounding = %.1f%%   table-snap exact = %.1f%%"
          % (100 * w1, 100 * ex, 100 * snap))
    ok &= w1 >= 0.90

    # --- what did it learn?  the table of named parts and the numeral table
    print("\n[learned table] named part → learned magnitude (true value):")
    vt = teacher.learned_fraction_table()
    line = []
    for d in FRACTION_DENOMS:
        tag = "unseen" if d in unseen else ("rare" if d in rare else "")
        line.append("  1/%-3d %.5f (%.5f) %s" % (d, vt[d], 1 / d, tag))
    for i in range(0, len(line), 3):
        print("".join("%-36s" % s for s in line[i:i + 3]))
    dv = np.log1p(np.exp(teacher.dl.v)); pv = np.exp(teacher.pl.v)
    unit = dv[0]
    print("[learned table] digit letters Ա..Թ → %s" % np.array2string(dv / unit, precision=3))
    print("[learned table] power letters ·,Ժ,Ճ,Ռ,բիւր,Ժբիւր → %s" % np.array2string(pv * unit, precision=1))
    gates = 1 / (1 + np.exp(-teacher.phi.v))
    print("[learned gates] mean quantity-gate: fractions %.3f | narrative words %.3f | <and>/<of_*> %.3f"
          % (gates[[FRAC_TOK[d] for d in common]].mean(),
             gates[[TOK[w] for w in DISTRACTORS]].mean(), gates[[AND, OF_WHOLE, OF_REST]].mean()))
    common_err = max(abs(vt[d] - 1 / d) / (1 / d) for d in common)
    print("[learned table] worst relative error on taught parts = %.2e" % common_err)
    ok &= common_err < 0.05

    # --- mode gate: does the phrase reader read "of the remaining"?
    b = batchify(test_common)
    _, _, mvals, _ = teacher.forward(b)
    truth = np.array([[1.0 if (k < len(p.takings) and p.takings[k][1]) else 0.0 for k in range(K_MAX)]
                      for p in test_common])
    mask = b["kmask"] > 0
    mode_acc = ((mvals.v > 0.5) == (truth > 0.5))[mask].mean()
    print("[phrase reader] mode gate accuracy (of-the-whole vs of-the-remainder) = %.1f%%" % (100 * mode_acc))
    ok &= mode_acc > 0.95

    # --- the half-taught teacher: lineage with and without the certificate
    print("\n[lineage] the half-taught teacher — three generations, gated vs ungated")
    stream = [make_problem(rng, common, rare=unseen, rare_prob=0.2) for _ in range(2500 if quick else 5000)]
    stream_unseen = np.array([any(d in unseen for dd, _, _ in p.takings for d in dd) for p in stream])

    def teach(model, probs, gated):
        """A model labels problems for its student.  Gated: only certified ones."""
        W_hat = model.predict(probs)[0]
        keep, labels = [], []
        for i, p in enumerate(probs):
            if gated:
                c, _ = model.certify(p)
                if not c:
                    continue
            keep.append(p); labels.append(math.log(max(W_hat[i], 1e-6)))
        return keep, np.array(labels)

    def score(model, gated):
        wc, _, _ = evaluate(model, test_common)
        W_hat = model.predict(test_unseen)[0]
        Wt = np.array([p.whole for p in test_unseen], dtype=float)
        answered = np.ones(len(test_unseen), dtype=bool)
        if gated:
            answered = np.array([model.certify(p)[0] for p in test_unseen])
        wrong = (np.abs(W_hat - Wt) / Wt > 0.01) & answered
        return wc, wrong.mean(), 1 - answered.mean()

    results = {}
    for gated in (False, True):
        tag = "gated" if gated else "ungated"
        results[tag] = []
        parent = teacher
        wc, wr, ab = score(parent, gated)
        results[tag].append((0, wc, wr, ab, len(stream)))
        for gen in (1, 2):
            probs, labels = teach(parent, stream, gated)
            student = ShirakNet(np.random.default_rng(10 + gen))
            train_numerals(student, epochs=(30 if quick else 40), verbose=False)
            train(student, probs, epochs=(14 if quick else 30), lr=1e-2, name="%s-gen%d" % (tag, gen),
                  targets=labels, verbose=False)
            wc, wr, ab = score(student, gated)
            results[tag].append((gen, wc, wr, ab, len(probs)))
            parent = student
    print("  %-8s %-4s %-14s %-22s %-20s %s" % ("regime", "gen", "common ok", "unseen: WRONG answers", "unseen: abstained", "taught items"))
    for tag in ("ungated", "gated"):
        for gen, wc, wr, ab, n in results[tag]:
            print("  %-8s %-4d %-14s %-22s %-20s %d" % (tag, gen, "%.1f%%" % (100 * wc), "%.1f%%" % (100 * wr),
                                                    "%.1f%%" % (100 * ab), n))
    ung = results["ungated"][-1]; gat = results["gated"][-1]
    print("  → after two generations the ungated lineage still teaches wrong wholes on %.0f%% of unseen-part"
          " problems; the certified lineage answers wrongly on %.0f%% and abstains on %.0f%%."
          % (100 * ung[2], 100 * gat[2], 100 * gat[3]))
    print("  → the price of the certificate: the certified pupil is taught fewer problems, so its competence"
          " on common problems is lower (%.0f%% vs %.0f%% at generation 2); the ungated pupil pays instead in"
          " confident false teaching (%.0f%% of unseen-part problems)." % (100 * gat[1], 100 * ung[1], 100 * ung[2]))
    # the claim being tested: a certified lineage never propagates a fraction as the whole,
    # and the ungated lineage does so at every generation
    ok &= (gat[2] < 0.05) and (ung[2] > 0.5) and all(r[2] < 0.05 for r in results["gated"])

    # --- demonstration on Anania's own problems
    print("\n[demonstration] Anania's own problems, solved by the trained teacher")
    print("  %-88s %8s %10s %10s  %s" % ("problem", "Anania", "raw W", "snapped", "certificate"))
    A = anania_problems()
    exact_snap = 0
    for p in A:
        W_hat = teacher.predict([p])[0][0]
        Ws = teacher.table_snap(p)
        c, why = teacher.certify(p)
        Ws_str = "%d" % Ws if (Ws is not None and Ws.denominator == 1) else str(Ws)
        if Ws == p.whole:
            exact_snap += 1
        print("  %-88s %8d %10.1f %10s  %s" % (p.text[:88], p.whole, W_hat, Ws_str, why))
    print("  table-snap exact on Anania's problems whose parts were taught: %d / %d" %
          (exact_snap, len(A)))

    # --- the Pastime Inverter
    print("\n[pastime] the feast trick: learns y = α·h + β + g and undoes it")
    pi = PastimeInverter(np.random.default_rng(7))
    alpha, beta = pi.fit()
    h, g = 9, 37
    y = PastimeInverter.trick(h, g)
    hh, gg = pi.undo(y)
    print("  learned α = %.3f (100), β = %.3f (350);  guest says %d → dines at hour %d, %d cups (%s)"
          % (alpha, beta, y, hh, gg, "correct" if (hh, gg) == (h, g) else "WRONG"))
    ok &= (hh, gg) == (h, g) and abs(alpha - 100) < 0.5 and abs(beta - 350) < 2

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS %s   (%.1fs)" % ("PASSED" if ok else "FAILED", time.time() - t_start))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(quick="--quick" in sys.argv))

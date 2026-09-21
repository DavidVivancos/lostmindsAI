#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================
ISNAD-NET -- a from-scratch, trainable architecture built on the cognitive
signature of Muhammad ibn Isma'il al-Bukhari (Bukhara, 21 July 810 -
Khartank near Samarkand, 1 September 870).
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0220_al_bukhari_810 - Muhammad ibn Isma'il al-Bukhari (Bukhara, 21 July 810 - Khartank near Samarkand, 1 September 870)
================================================================================  
WHY THIS IS NOT A TRANSFORMER
-----------------------------
Attention is a weighted SUM over stored keys. A weak source can be rescued by
strong neighbours; the mechanism has no way to say "this route is severed".
Al-Bukhari's entire discipline is the refusal of that arithmetic. His unit of
evidence is not a token but a ROUTE -- a named line of men, breakable at any
joint -- and his three governing rules are all anti-additive:

  * a chain is worth what its WORST narrator is worth  (min, not mean);
  * a joint that cannot be shown to have happened contributes NOTHING
    (a hard gate, not a small weight);
  * a second route only corroborates to the extent it does not run through
    the same man (independence must be earned, not assumed).

So the model here scores routes by a soft-MINIMUM whose sharpness beta is a
learnable dial, gates every joint on documented meeting with a hard
straight-through binary, and combines routes by a NOISY-OR discounted by
transmitter overlap. Attention appears nowhere in this file.

THE FIVE MECHANISMS AND THE HISTORY THEY ENCODE
-----------------------------------------------
1. THE LIQA' GATE. Al-Bukhari is traditionally credited with requiring
   thubut al-liqa' -- established proof that two adjacent narrators actually
   met -- where his younger contemporary Muslim ibn al-Hajjaj held that
   mu'asara, mere contemporaneity plus the absence of any evidence against
   meeting, sufficed. (Modern scholarship, and Muslim's own polemical
   introduction, show the attribution is a later systematisation of an
   inconsistent practice; both men in fact rejected chains where non-meeting
   was demonstrated. The distinction is nonetheless real as a difference of
   emphasis, and it is precisely computable, so this file computes it.)
   `hard_liqa=True` runs the strict reading; `hard_liqa=False` runs the lax
   one, on the same weights, and the trade is reported in the output.

2. THE WEAKEST LINK. pi = softmin_beta(kappa). At beta -> infinity this is the
   classical grading rule. At beta -> 0 it is an average, i.e. exactly the
   laundering the discipline was built to prevent.

3. CORROBORATION WITH AN INDEPENDENCE DISCOUNT. mutaba'at and shawahid raise
   confidence through a noisy-OR, but each route is discounted by
   exp(-lambda * overlap) where overlap is the fraction of its transmitters
   shared with the report's other routes. Twenty routes through one man are
   one route.

4. THE LAFZ BARRIER. In 864 al-Bukhari's position that a reciter's UTTERANCE
   of the Qur'an is a created human act -- while the Qur'an itself is not --
   cost him his students, his welcome in Nishapur, and finally his city. The
   architecture takes that distinction literally: content lives in a single
   consensus slot shared by every route; the utterance is content plus a
   transmitter's own style; and the gradient from reconstructing the wording
   is SEVERED before it can reach the content slot. Test T3 asserts the
   barrier numerically. The model can be made to predict what a man said
   without ever letting how he said it define what it meant.

5. ASSERTION BY PLACEMENT. The model emits no claims. Scholars say
   "al-Bukhari's jurisprudence is in his chapter headings" -- his Sahih argues
   by where a report is filed, not by any statement of his own. So the output
   heads are: an admission decision, an ordinal REGISTER (the ta'liq
   morphology: omit / yurwa / yudhkaru / qala, a calibrated scale on which
   understatement is the loud end), and a CHAPTER ASSIGNMENT. Nothing is ever
   generated.

Additionally the model carries two diagnostic heads with direct sources:
  * MAQLUB -- does this text belong on this chain? Trained against the
    transposition attack, and evaluated by re-running the trial the scholars
    of Baghdad set him: a hundred narrations with their texts and chains
    deliberately swapped, to be sorted back.
  * 'ILLA -- the hidden defect. By construction it is invisible in every link
    feature; it is recoverable only from the shape of the disagreement
    between routes. This is what Kitab al-'Ilal was for.

CONVENTIONS
-----------
Pure NumPy. A ~200-line reverse-mode autodiff engine is included (Section 0)
so that the unusual operators -- straight-through binary gates, soft-min over
chain joints, masked noisy-OR, a severed gradient path -- can be composed and
then verified against finite differences rather than hand-derived and hoped
over. Section 4 is the mandatory gradient check; it must pass in both the
strict and lax gate modes. Section 6 runs six self-tests plus the Baghdad
trial. Everything is deterministic given the seeds.

  python3 chapter_0220_al_bukhari_810.py
===============================================================================
"""

import numpy as np
import time

# ===========================================================================
# SECTION 0 -- A MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy)
# ===========================================================================

EPS = 1e-12


def _ub(g, shape):
    """Un-broadcast a gradient `g` back to `shape`."""
    if g.shape == shape:
        return g
    # sum away leading axes that were added by broadcasting
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    # sum away axes that were size-1 in the original
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class T:
    """A node in the computation tape."""
    __slots__ = ("d", "g", "_bw", "_prev", "name")

    def __init__(self, data, prev=(), name=""):
        self.d = np.asarray(data, dtype=np.float64)
        self.g = np.zeros_like(self.d)
        self._bw = lambda: None
        self._prev = prev
        self.name = name

    # ---- graph plumbing -------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        for v in topo:
            v.g = np.zeros_like(v.d)
        self.g = np.ones_like(self.d)
        for v in reversed(topo):
            v._bw()

    # ---- operator sugar -------------------------------------------------
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(self, o)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(self, o)
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return sub(_wrap(o), self)
    def __neg__(self): return mul(self, -1.0)
    def __truediv__(self, o): return div(self, o)
    def __matmul__(self, o): return matmul(self, o)
    def __pow__(self, k): return powc(self, k)

    @property
    def shape(self): return self.d.shape


def _wrap(x):
    return x if isinstance(x, T) else T(x)


# ---------------------------------------------------------------------------
# elementwise / binary ops
# ---------------------------------------------------------------------------
def add(a, b):
    a, b = _wrap(a), _wrap(b)
    out = T(a.d + b.d, (a, b))

    def bw():
        a.g += _ub(out.g, a.d.shape)
        b.g += _ub(out.g, b.d.shape)
    out._bw = bw
    return out


def sub(a, b):
    a, b = _wrap(a), _wrap(b)
    out = T(a.d - b.d, (a, b))

    def bw():
        a.g += _ub(out.g, a.d.shape)
        b.g += _ub(-out.g, b.d.shape)
    out._bw = bw
    return out


def mul(a, b):
    a, b = _wrap(a), _wrap(b)
    out = T(a.d * b.d, (a, b))

    def bw():
        a.g += _ub(out.g * b.d, a.d.shape)
        b.g += _ub(out.g * a.d, b.d.shape)
    out._bw = bw
    return out


def div(a, b):
    a, b = _wrap(a), _wrap(b)
    out = T(a.d / b.d, (a, b))

    def bw():
        a.g += _ub(out.g / b.d, a.d.shape)
        b.g += _ub(-out.g * a.d / (b.d ** 2), b.d.shape)
    out._bw = bw
    return out


def matmul(a, b):
    a, b = _wrap(a), _wrap(b)
    out = T(np.matmul(a.d, b.d), (a, b))

    def bw():
        ga = np.matmul(out.g, np.swapaxes(b.d, -1, -2))
        gb = np.matmul(np.swapaxes(a.d, -1, -2), out.g)
        a.g += _ub(ga, a.d.shape)
        b.g += _ub(gb, b.d.shape)
    out._bw = bw
    return out


# ---------------------------------------------------------------------------
# unary ops
# ---------------------------------------------------------------------------
def exp(a):
    a = _wrap(a)
    v = np.exp(np.clip(a.d, -60.0, 60.0))
    out = T(v, (a,))

    def bw():
        a.g += out.g * v
    out._bw = bw
    return out


def log(a):
    a = _wrap(a)
    out = T(np.log(np.clip(a.d, EPS, None)), (a,))

    def bw():
        a.g += out.g / np.clip(a.d, EPS, None)
    out._bw = bw
    return out


def tanh(a):
    a = _wrap(a)
    v = np.tanh(a.d)
    out = T(v, (a,))

    def bw():
        a.g += out.g * (1.0 - v ** 2)
    out._bw = bw
    return out


def sigmoid(a):
    a = _wrap(a)
    z = np.clip(a.d, -60.0, 60.0)
    v = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    out = T(v, (a,))

    def bw():
        a.g += out.g * v * (1.0 - v)
    out._bw = bw
    return out


def softplus(a):
    a = _wrap(a)
    z = np.clip(a.d, -60.0, 60.0)
    v = np.logaddexp(0.0, z)
    out = T(v, (a,))

    def bw():
        a.g += out.g * (1.0 / (1.0 + np.exp(-z)))
    out._bw = bw
    return out


def relu(a):
    a = _wrap(a)
    out = T(np.maximum(a.d, 0.0), (a,))

    def bw():
        a.g += out.g * (a.d > 0)
    out._bw = bw
    return out


# ---------------------------------------------------------------------------
# reductions / shape
# ---------------------------------------------------------------------------
def rsum(a, axis=None, keepdims=False):
    a = _wrap(a)
    out = T(a.d.sum(axis=axis, keepdims=keepdims), (a,))

    def bw():
        g = out.g
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a.g += np.broadcast_to(g, a.d.shape).copy()
    out._bw = bw
    return out


def rmean(a, axis=None, keepdims=False):
    a = _wrap(a)
    n = a.d.size if axis is None else a.d.shape[axis]
    return rsum(a, axis=axis, keepdims=keepdims) * (1.0 / n)


def reshape(a, shape):
    a = _wrap(a)
    out = T(a.d.reshape(shape), (a,))

    def bw():
        a.g += out.g.reshape(a.d.shape)
    out._bw = bw
    return out


def concat(ts, axis=-1):
    ts = [_wrap(t) for t in ts]
    out = T(np.concatenate([t.d for t in ts], axis=axis), tuple(ts))
    sizes = [t.d.shape[axis] for t in ts]

    def bw():
        idx, ax = 0, axis % out.d.ndim
        for t, s in zip(ts, sizes):
            sl = [slice(None)] * out.d.ndim
            sl[ax] = slice(idx, idx + s)
            t.g += out.g[tuple(sl)]
            idx += s
    out._bw = bw
    return out


def gather(E, idx):
    """E: (N, d) parameter table. idx: integer array of any shape -> (..., d)."""
    E = _wrap(E)
    idx = np.asarray(idx)
    out = T(E.d[idx], (E,))

    def bw():
        np.add.at(E.g, idx, out.g)
    out._bw = bw
    return out


def stop_grad(a):
    """Detach: forward value passes, gradient does not."""
    return T(np.array(a.d, copy=True), ())


def ste(hard_vals, soft_node):
    """Straight-through estimator: forward uses `hard_vals`, backward routes the
    gradient to `soft_node` unchanged."""
    out = T(hard_vals, (soft_node,))

    def bw():
        soft_node.g += out.g
    out._bw = bw
    return out


def powc(a, k):
    """Power with a constant exponent."""
    a = _wrap(a)
    k = float(k)
    out = T(a.d ** k, (a,))

    def bw():
        a.g += out.g * k * (a.d ** (k - 1.0))
    out._bw = bw
    return out
# ===========================================================================
# SECTION 1 -- THE ISNAD WORLD (synthetic data with a known generative truth)
# ===========================================================================
#
# Al-Bukhari did not face a corpus of propositions. He faced a corpus of
# *routes*: the same claim arriving many times, each time down a different
# line of named men, each line breakable at any joint. The data generator
# below reproduces that situation and nothing else. It builds:
#
#   * a population of transmitters, each with a hidden integrity and a hidden
#     precision (the classical pair 'adala / dabt), some of them fabricators,
#     some of them practitioners of tadlis (concealing a missing link);
#   * a MEETING RELATION -- who actually heard from whom -- and, separately,
#     an ATTESTATION RECORD of which meetings the archive can document. These
#     two are deliberately not identical. Some real meetings are undocumented.
#     This gap is the whole argument of the chapter: requiring documented
#     meeting (thubut al-liqa') rather than mere contemporaneity (mu'asara)
#     buys precision at a measurable cost in recall, and the model is built so
#     that cost can be measured rather than asserted;
#   * reports, each with a true content vector and a true chapter, arriving by
#     several routes, corrupted in four historically distinct ways:
#       - fabrication  (a liar in the chain invents the content)
#       - maqlub       (a true text is attached to the wrong chain)
#       - shudhudh     (one route's wording deviates from the better-borne ones)
#       - 'illa        (a HIDDEN defect: a confusable same-era, same-region
#                       narrator has been silently substituted; every surface
#                       feature still looks clean)
#
# The 'illa case is the point of the exercise. It is invisible in the link
# features by construction. It can only be recovered from the pattern of
# disagreement between routes -- which is exactly what al-Bukhari's Kitab
# al-'Ilal was for.

CFG = dict(
    n_transmitters=140,
    n_generations=5,
    n_regions=4,
    n_chapters=8,
    n_routes=3,          # P: routes per report (padded/masked)
    chain_len=4,         # L: transmitters per route  => L-1 = 3 joints
    surf_dim=16,         # m: surface (wording) dimensionality
    emb_dim=24,          # d: transmitter embedding
    style_dim=8,
    hid=32,
    link_feat=6,         # F  (raw)
    # The attestation bit is visible ONLY to the liqa' gate; the tadlis bit is
    # visible ONLY to the credibility head. Whether two men met and whether a
    # man is worth hearing are different questions with different evidence,
    # and collapsing them into one feature vector is exactly the confusion the
    # architecture exists to prevent.
    feat_gate=[0, 1, 5],    # circumstantial: era gap, region, prolific teacher
    feat_cred=[0, 1, 3, 4, 5],
    col_attested=2,         # the samã' record: THIS meeting is documented
    col_contemp=3,          # mere contemporaneity: this meeting was POSSIBLE
)


class IsnadWorld:
    """Generates transmitters, the meeting relation, and routed reports."""

    def __init__(self, seed=207, cfg=CFG):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        c, rng = cfg, self.rng
        N = c["n_transmitters"]

        # --- the population of transmitters -----------------------------
        self.era = rng.integers(0, c["n_generations"], size=N)
        self.region = rng.integers(0, c["n_regions"], size=N)
        self.integrity = rng.beta(5, 2, size=N)      # 'adala  -- uprightness
        self.precision = rng.beta(4, 2, size=N)      # dabt    -- exactness
        self.fabricator = (rng.random(N) < 0.08).astype(np.float64)
        self.integrity = np.where(self.fabricator > 0, rng.random(N) * 0.3, self.integrity)
        self.tadlis = (rng.random(N) < 0.09).astype(np.float64)

        # per-transmitter wording style: the 'created' part of an utterance
        self.style = rng.normal(0, 1.0, size=(N, c["style_dim"]))
        self.style_map = rng.normal(0, 0.35, size=(c["style_dim"], c["surf_dim"]))

        # --- who actually met whom (era t -> era t+1, region-biased) -----
        self.met = np.zeros((N, N), dtype=np.float64)
        for a in range(N):
            for b in range(N):
                if self.era[b] == self.era[a] + 1:
                    p = 0.55 if self.region[a] == self.region[b] else 0.18
                    if rng.random() < p:
                        self.met[a, b] = 1.0
        # --- which of those meetings the archive can document ------------
        # 82% of real meetings are attested; the remaining 18% are the price
        # a strict liqa' rule pays. Independently, 6% of NON-meetings are
        # falsely attested -- forged samã'-certificates existed.
        self.attested = np.where(
            (self.met > 0) & (rng.random((N, N)) < 0.96), 1.0, 0.0)
        # A false attestation is only ever *plausible* between adjacent
        # generations -- forged sama'-certificates had to be chronologically
        # possible to be worth forging. 5% of era-adjacent non-meetings carry
        # one, which is what makes the gate a real inference problem rather
        # than a lookup.
        adjacent = (self.era[None, :] == self.era[:, None] + 1)
        self.attested = np.where(
            (self.met == 0) & adjacent & (rng.random((N, N)) < 0.02),
            1.0, self.attested)

        # chapter prototypes for content
        self.proto = rng.normal(0, 1.0, size=(c["n_chapters"], c["surf_dim"]))

        # TOPICAL SPECIALISATION. A transmitter is 'known for' certain
        # material; chains therefore carry information about what they can
        # plausibly be carrying. This is the fact that makes a transposed text
        # detectable at all -- al-Bukhari catches a maqlub report because he
        # knows which shaykh transmits which subject, not because the wording
        # is wrong. Without this the Baghdad trial would be pure chance.
        self.affinity = rng.dirichlet(np.full(c["n_chapters"], 0.22), size=N)

    # ------------------------------------------------------------------
    def _sample_chain(self, forged, chap=None):
        """Return (ids, real_link) for one route of length L.

        If `forged` is True, exactly one joint is a NON-meeting between two
        contemporaries -- the classic 'possible but unattested' link that
        mu'asara accepts and liqa' rejects.
        """
        c, rng = self.cfg, self.rng
        L = c["chain_len"]
        start_era = 0
        for _ in range(60):
            cands = np.where(self.era == start_era)[0]
            ids = [rng.choice(cands, p=self._w(cands, chap))]
            ok = True
            for step in range(L - 1):
                nxt = np.where(self.met[ids[-1]] > 0)[0]
                if len(nxt) == 0:
                    ok = False
                    break
                ids.append(rng.choice(nxt, p=self._w(nxt, chap)))
            if ok:
                ids = np.array(ids)
                real = np.ones(L - 1)
                if forged:
                    j = rng.integers(0, L - 1)
                    pool = np.where((self.era == self.era[ids[j]] + 1)
                                    & (self.met[ids[j]] == 0))[0]
                    if len(pool):
                        ids[j + 1] = rng.choice(pool)
                        real[j] = 0.0
                        # downstream joints are now also broken
                        for k in range(j + 1, L - 1):
                            pool2 = np.where(self.met[ids[k]] > 0)[0]
                            if len(pool2):
                                ids[k + 1] = rng.choice(pool2)
                            else:
                                real[k] = 0.0
                return ids, real
        return np.arange(L) % self.cfg["n_transmitters"], np.zeros(L - 1)

    def _w(self, cands, chap):
        """Selection weights: men who carry this subject are likelier to be
        found in a chain that carries it."""
        if chap is None:
            return np.full(len(cands), 1.0 / len(cands))
        v = self.affinity[cands, chap] ** 1.5 + 0.004
        return v / v.sum()

    def _link_feats(self, ids, real):
        """Observable features of each joint. NOTE: `real` is never exposed."""
        L = len(ids)
        F = np.zeros((L - 1, self.cfg["link_feat"]))
        for i in range(L - 1):
            a, b = ids[i], ids[i + 1]
            F[i] = [
                (self.era[b] - self.era[a]) / 2.0,                 # era gap
                1.0 if self.region[a] == self.region[b] else 0.0,  # same region
                self.attested[a, b],                               # DOCUMENTED meeting
                1.0 if abs(self.era[a] - self.era[b]) <= 1 else 0.0,  # contemporaneity
                self.tadlis[b],                                    # receiver conceals
                float(self.met[a].sum() > 12),                     # prolific teacher
            ]
        return F

    # ------------------------------------------------------------------
    def batch(self, n):
        c, rng = self.cfg, self.rng
        P, L, m = c["n_routes"], c["chain_len"], c["surf_dim"]

        ids = np.zeros((n, P, L), dtype=np.int64)
        feats = np.zeros((n, P, L - 1, c["link_feat"]))
        surf = np.zeros((n, P, m))
        mask = np.zeros((n, P))
        overlap = np.zeros((n, P))
        y_adm = np.zeros((n, 1))
        y_chap = np.zeros((n,), dtype=np.int64)
        y_maq = np.zeros((n, P))
        y_illa = np.zeros((n, 1))
        y_reg = np.zeros((n,), dtype=np.int64)
        modes = []

        contents = []
        for r in range(n):
            chap = rng.integers(0, c["n_chapters"])
            z = self.proto[chap] + rng.normal(0, 0.30, size=m)
            contents.append(z)
            y_chap[r] = chap

            n_routes = rng.integers(1, P + 1)
            mode = rng.choice(["clean", "forged", "fabricated",
                               "maqlub", "shadh", "illa"],
                              p=[0.28, 0.24, 0.12, 0.12, 0.12, 0.12])
            modes.append(mode)
            clean_routes = 0
            for p in range(n_routes):
                forged = (mode == "forged") and (p <= (0 if rng.random() < 0.5 else 1))
                cid, real = self._sample_chain(forged, chap)
                ids[r, p] = cid
                feats[r, p] = self._link_feats(cid, real)
                mask[r, p] = 1.0

                x = z + self.style[cid[-1]] @ self.style_map \
                      + rng.normal(0, 0.12, size=m)

                route_clean = bool(real.all())
                # every transmitter in the chain must be upright and exact
                if (self.fabricator[cid].any()
                        or self.integrity[cid].min() < 0.38
                        or self.precision[cid].min() < 0.33
                        or self.tadlis[cid].any()):
                    route_clean = False

                if mode == "fabricated" and p == 0:
                    x = rng.normal(0, 1.2, size=m)
                    route_clean = False
                elif mode == "maqlub" and p == 0:
                    other = self.proto[rng.integers(0, c["n_chapters"])] \
                            + rng.normal(0, 0.30, size=m)
                    x = other + self.style[cid[-1]] @ self.style_map
                    y_maq[r, p] = 1.0
                    route_clean = False
                elif mode == "shadh" and p == 0:
                    x = x + rng.normal(0, 0.95, size=m)
                    route_clean = False
                elif mode == "illa":
                    # HIDDEN defect: silently swap one narrator for a
                    # same-era, same-region confusable. Features unchanged.
                    j = rng.integers(1, L)
                    pool = np.where((self.era == self.era[cid[j]])
                                    & (self.region == self.region[cid[j]]))[0]
                    if len(pool):
                        cid2 = cid.copy()
                        cid2[j] = rng.choice(pool)
                        ids[r, p] = cid2
                        x = x + rng.normal(0, 0.42, size=m)
                    route_clean = False
                    y_illa[r, 0] = 1.0

                surf[r, p] = x
                clean_routes += int(route_clean)

            # route independence: how much of this chain is shared with the others
            for p in range(n_routes):
                others = set()
                for q in range(n_routes):
                    if q != p:
                        others |= set(ids[r, q].tolist())
                overlap[r, p] = len(set(ids[r, p].tolist()) & others) / L

            y_adm[r, 0] = 1.0 if clean_routes >= 1 else 0.0
            # register (ta'liq morphology): 0 omit, 1 yurwa, 2 yudhkaru, 3 qala
            if clean_routes == 0:
                y_reg[r] = 0
            elif clean_routes == 1:
                y_reg[r] = 1 if overlap[r, :n_routes].mean() > 0.35 else 2
            else:
                y_reg[r] = 3

        return dict(ids=ids, feats=feats,
                    feats_gate=feats[..., c["feat_gate"]],
                    feats_cred=feats[..., c["feat_cred"]],
                    ev_liqa=feats[..., c["col_attested"]:c["col_attested"] + 1],
                    ev_muasara=feats[..., c["col_contemp"]:c["col_contemp"] + 1],
                    surf=surf, mask=mask,
                    overlap=overlap, y_adm=y_adm, y_chap=y_chap,
                    y_maq=y_maq, y_illa=y_illa, y_reg=y_reg,
                    mode=np.array(modes),
                    contents=np.array(contents))


# ===========================================================================
# SECTION 2 -- ISNAD-NET
# ===========================================================================
#
# Five mechanisms, each of which is a claim about al-Bukhari's method rather
# than a piece of standard machinery:
#
#   (1) HARD LIQA' GATE. A joint contributes nothing unless the model decides
#       a meeting is DOCUMENTED. The decision is binary in the forward pass and
#       straight-through in the backward pass, so the constraint is a
#       constraint, not a soft preference. Setting hard=False recovers the
#       mu'asara reading and the two can be compared on the same weights.
#
#   (2) WEAKEST-LINK PATH SCORE. A route is scored by a soft-min over its
#       joints with sharpness beta. As beta grows this becomes the classical
#       rule -- a chain is worth what its worst narrator is worth. As beta
#       falls it becomes an average, which is precisely the lax rule that lets
#       a strong teacher launder a weak student. beta is a learnable dial and
#       the chapter's argument is that it must sit high.
#
#   (3) NOISY-OR ACROSS ROUTES, DISCOUNTED BY OVERLAP. Corroboration
#       (mutaba'at / shawahid) raises confidence only to the extent the routes
#       do not run through the same man. Independence is earned, not assumed.
#
#   (4) LAFZ/MA'NA SEPARATION. The conveyed content is a single consensus slot
#       shared by every route; the utterance is content plus a transmitter's
#       own style. Gradient from surface reconstruction is severed before it
#       can reach the content slot. The wording is created; what it conveys is
#       not reconstructed from the wording.
#
#   (5) ASSERTION BY PLACEMENT. The model never emits a claim. Its opinion is
#       expressed as (a) an admission decision, (b) an ordinal register that
#       encodes how loudly the claim may be made, and (c) a CHAPTER
#       ASSIGNMENT. Al-Bukhari's jurisprudence lives in his chapter headings;
#       so does this model's.

def _init(rng, *shape, scale=None):
    s = scale if scale else (1.0 / np.sqrt(shape[0]))
    return T(rng.normal(0, s, size=shape))


def init_params(cfg=CFG, seed=7):
    rng = np.random.default_rng(seed)
    d, m, h = cfg["emb_dim"], cfg["surf_dim"], cfg["hid"]
    F, ds = cfg["link_feat"], cfg["style_dim"]
    N, C = cfg["n_transmitters"], cfg["n_chapters"]
    p = {}
    p["E"] = _init(rng, N, d, scale=0.30)                 # TRUST index: whether a man is worth hearing
    p["EM"] = _init(rng, N, d, scale=0.30)                # SUBJECT index: what a man carries
    p["STY"] = _init(rng, N, ds, scale=0.30)              # transmitter wording style
    p["w_rel"] = _init(rng, d, 1)                         # 'adala x dabt head
    p["b_rel"] = T(np.zeros((1,)))
    p["W1"] = _init(rng, 2 * d + len(cfg["feat_cred"]), h)                    # joint credibility MLP
    p["b1"] = T(np.zeros((h,)))
    p["W2"] = _init(rng, h, 1)
    p["b2"] = T(np.zeros((1,)))
    p["Wg1"] = _init(rng, len(cfg["feat_gate"]), 12)                          # liqa' gate MLP
    p["bg1"] = T(np.zeros((12,)))
    p["Wg2"] = _init(rng, 12, 1)
    p["bg2"] = T(np.array([0.0]))
    # THE CONSTITUTIONAL TERM. The evidence bit enters the gate through its
    # own large weight, so that in the absence of evidence the joint starts
    # CLOSED and the model must find positive counter-evidence to open it.
    # This is the difference between a rule and a preference, and al-Bukhari
    # held it as a rule: he spent sixteen years travelling to establish who
    # had sat with whom, which is not the behaviour of a man treating the
    # question as one feature among several.
    p["w_att"] = T(np.array([2.5]))
    p["beta_raw"] = T(np.array([1.6]))                    # weakest-link sharpness
    p["lam_raw"] = T(np.array([0.4]))                     # independence decay
    p["W_enc"] = _init(rng, m, d)                         # wording -> meaning
    p["b_enc"] = T(np.zeros((d,)))
    p["W_dec"] = _init(rng, d + ds, m)                    # meaning+style -> wording
    p["b_dec"] = T(np.zeros((m,)))
    p["W_menc"] = _init(rng, m, d)                        # wording -> "who could be carrying this"
    p["b_menc"] = T(np.zeros((d,)))
    p["pos_w"] = T(np.array([0.0, 0.0, 0.0, 1.0]))   # who defines a chain
    p["M_maq"] = _init(rng, d, d, scale=0.20)             # matn<->isnad compatibility
    p["C_chap"] = _init(rng, C, d, scale=0.30)            # chapter headings (tarjama)
    p["Wi1"] = _init(rng, 6, 16)                          # 'illa detector
    p["bi1"] = T(np.zeros((16,)))
    p["Wi2"] = _init(rng, 16, 1)
    p["bi2"] = T(np.array([0.0]))
    p["w_adm"] = _init(rng, 6, 1)                         # admission
    p["b_adm"] = T(np.array([0.0]))
    p["w_reg"] = _init(rng, 6, 1)                         # register score
    p["b_reg"] = T(np.array([0.0]))
    p["th0"] = T(np.array([-1.0]))                        # ordinal thresholds
    p["dth"] = T(np.array([0.0, 0.0]))
    return p


GATE_PARAMS = ("Wg1", "bg1", "Wg2", "bg2", "w_att")


def forward(p, b, hard_liqa=True, beta_override=None, rule="liqa"):
    """Returns a dict of named nodes. Shapes: B=batch, P=routes, L=chain.

    rule="liqa"     -- a joint is evidenced by a DOCUMENTED meeting (al-Bukhari)
    rule="muasara"  -- a joint is evidenced by mere CONTEMPORANEITY (the reading
                       associated with Muslim ibn al-Hajjaj). Same weights,
                       different evidence bit; the cost of each is measured in
                       the ablation at the end of this file.
    hard_liqa=False -- relax the binary to a sigmoid (used by the gradient check)
    """
    B, P, L = b["ids"].shape
    d = p["E"].d.shape[1]
    m = b["surf"].shape[-1]

    ids = b["ids"]
    Ea = gather(p["E"], ids[:, :, :-1])          # (B,P,L-1,d) teacher
    Eb = gather(p["E"], ids[:, :, 1:])           # (B,P,L-1,d) student
    Eall = gather(p["EM"], ids)                  # (B,P,L,d) subject index
    # Two separate tables over the same men. Whether a narrator is RELIABLE
    # and what SUBJECT he is known to carry are independent judgements, and
    # al-Bukhari kept them in separate books -- al-Tarikh al-Kabir for the
    # men, the Jami' for the material. Tying them into one embedding lets a
    # verdict about a man's honesty leak into a claim about his repertoire.
    Fg = T(b["feats_gate"])                      # evidence of MEETING
    Fc = T(b["feats_cred"])                      # evidence of QUALITY

    # --- (1) reliability of each party to the joint ----------------------
    rho_a = sigmoid(matmul(Ea, p["w_rel"]) + p["b_rel"])   # (B,P,L-1,1)
    rho_b = sigmoid(matmul(Eb, p["w_rel"]) + p["b_rel"])

    # --- joint credibility ----------------------------------------------
    li = concat([Ea, Eb, Fc], axis=-1)
    hh = tanh(matmul(li, p["W1"]) + p["b1"])
    s = sigmoid(matmul(hh, p["W2"]) + p["b2"])             # (B,P,L-1,1)

    # --- (2) the liqa' gate: hard forward, straight-through backward -----
    ev = T(b["ev_liqa"] if rule == "liqa" else b["ev_muasara"])
    gh = tanh(matmul(Fg, p["Wg1"]) + p["bg1"])
    g_logit = (matmul(gh, p["Wg2"]) + p["bg2"]
               + softplus(p["w_att"]) * (ev * 2.0 - T(np.array([1.0]))))
    g_soft = sigmoid(g_logit)                              # (B,P,L-1,1)
    if hard_liqa:
        g = ste((g_soft.d > 0.5).astype(np.float64), g_soft)
    else:
        g = g_soft

    kappa = g * s * rho_a * rho_b                          # (B,P,L-1,1)
    kappa = reshape(kappa, (B, P, L - 1))

    # --- (3) weakest link: soft-min over joints --------------------------
    beta = softplus(p["beta_raw"]) + T(np.array([0.5]))
    if beta_override is not None:
        beta = T(np.array([float(beta_override)]))
    # Stable soft-min: shift by the true minimum (a detached constant, which
    # leaves the gradient exact) so that exp(-beta*kappa) can never underflow
    # into the logarithm's clamp. Without this the score can fall BELOW the
    # weakest joint, which would quietly invert the whole doctrine.
    kmin = T(np.min(kappa.d, axis=2, keepdims=True))        # (B,P,1) constant
    pi = reshape(kmin, (B, P)) - (
        log(rmean(exp(-beta * (kappa - kmin)), axis=2)) / beta)   # (B,P)

    # --- (4) independence discount and noisy-OR across routes ------------
    lam = softplus(p["lam_raw"])
    iota = exp(-(lam * T(b["overlap"])))                    # (B,P)
    mask = T(b["mask"])
    q = pi * iota * mask * T(np.array([0.995]))             # (B,P)
    A = T(np.ones((B, 1))) - exp(rsum(log(T(np.ones((B, P)))
                                          - q), axis=1, keepdims=True))  # (B,1)

    # --- (5) meaning vs utterance ---------------------------------------
    hsurf = tanh(matmul(T(b["surf"]), p["W_enc"]) + p["b_enc"])   # (B,P,d)
    omega = reshape(q + T(np.array([1e-3])), (B, P, 1))
    c = rsum(hsurf * omega, axis=1) / rsum(omega, axis=1)         # (B,d)  content slot

    c_sg = stop_grad(c)                                            # lafz barrier
    c_b = reshape(c_sg, (B, 1, d)) + T(np.zeros((B, P, d)))
    sty = gather(p["STY"], ids[:, :, -1])                          # (B,P,ds)
    recon = matmul(concat([c_b, sty], axis=-1), p["W_dec"]) + p["b_dec"]

    # shudhudh: deviation of a route's wording from the well-borne consensus
    dev = rmean((hsurf - reshape(c, (B, 1, d))) ** 2.0, axis=2)  # (B,P)
    shudh = rsum(dev * (T(np.ones((B, P))) - q) * mask, axis=1, keepdims=True) \
        / rsum(mask, axis=1, keepdims=True)

    # --- maqlub: does this text belong on this chain? --------------------
    # The fingerprint of a chain is a LEARNED weighting over its positions.
    # Initialised to favour the collector, because that is who a report is
    # heard from -- but the model is free to move the weight, and where it
    # settles is itself a finding.
    pw = reshape(_softmax(p["pos_w"], axis=0), (1, 1, L, 1))
    fp = rsum(Eall * pw, axis=2)                            # (B,P,d)
    # The transposition test does NOT reuse the meaning encoder. Asking
    # "what does this say" and "who could be carrying this" are different
    # questions, and al-Bukhari answered the second from the biographical
    # archive, not from the text.
    hm = tanh(matmul(T(b["surf"]), p["W_menc"]) + p["b_menc"])   # (B,P,d)
    maq_logit = rsum(matmul(hm, p["M_maq"]) * fp, axis=2)        # (B,P)
    maq = sigmoid(maq_logit)
    soft_max_maq = rsum(maq * _softmax(maq * T(np.array([8.0])), axis=1),
                        axis=1, keepdims=True)              # (B,1)

    # --- 'illa: a defect no single feature reveals ------------------------
    qm = rmean(q, axis=1, keepdims=True)
    qv = rmean((q - qm) ** 2.0, axis=1, keepdims=True)
    dm = rmean(dev, axis=1, keepdims=True)
    dv = rmean((dev - dm) ** 2.0, axis=1, keepdims=True)
    stats = concat([A, shudh, qm, qv, dm, dv], axis=1)      # (B,6)
    ih = tanh(matmul(stats, p["Wi1"]) + p["bi1"])
    illa_logit = matmul(ih, p["Wi2"]) + p["bi2"]            # (B,1)

    # --- admission, register, placement ----------------------------------
    afeat = concat([A, shudh, sigmoid(illa_logit), soft_max_maq, qm, dm], axis=1)
    adm_logit = matmul(afeat, p["w_adm"]) + p["b_adm"]      # (B,1)
    reg_score = matmul(afeat, p["w_reg"]) + p["b_reg"]      # (B,1)
    chap_logit = matmul(c, transpose_last(p["C_chap"]))     # (B,C)

    return dict(A=A, pi=pi, q=q, kappa=kappa, g=g, g_soft=g_soft,
                c=c, recon=recon, dev=dev, shudh=shudh,
                maq=maq, maq_logit=maq_logit, hm=hm, illa_logit=illa_logit,
                adm_logit=adm_logit, reg_score=reg_score,
                chap_logit=chap_logit, beta=beta, fp=fp, hsurf=hsurf)


def transpose_last(t):
    out = T(np.swapaxes(t.d, -1, -2), (t,))

    def bw():
        t.g += np.swapaxes(out.g, -1, -2)
    out._bw = bw
    return out


def _softmax(x, axis=-1):
    mx = T(np.max(x.d, axis=axis, keepdims=True))
    e = exp(x - mx)
    return e / rsum(e, axis=axis, keepdims=True)


# ===========================================================================
# SECTION 3 -- LOSSES
# ===========================================================================

def bce(logit, y):
    yt = T(y)
    return rmean(softplus(logit) - yt * logit)


def ce(logits, y):
    B, C = logits.d.shape
    oh = np.zeros((B, C))
    oh[np.arange(B), y] = 1.0
    mx = T(np.max(logits.d, axis=1, keepdims=True))
    lse = log(rsum(exp(logits - mx), axis=1, keepdims=True)) + mx
    picked = rsum(logits * T(oh), axis=1, keepdims=True)
    return rmean(lse - picked)


def ordinal_nll(score, thresholds, y, n_levels=4):
    """Cumulative-link ordinal loss -- the ta'liq register.

    The model does not choose a label, it chooses how far up a monotone scale
    of assertion it is willing to go. Understatement is cheap; overstatement
    is not. This is why al-Bukhari's mildest-sounding verdict, 'fihi nazar'
    ('there is something to look at in him'), functions as one of his
    harshest: the scale is calibrated, not literal.
    """
    th0, dth = thresholds
    t1 = th0
    t2 = th0 + softplus(_idx(dth, 0))
    t3 = t2 + softplus(_idx(dth, 1))
    ge1 = sigmoid(score - t1)
    ge2 = sigmoid(score - t2)
    ge3 = sigmoid(score - t3)
    one = T(np.array([1.0]))
    p0 = one - ge1
    p1 = ge1 - ge2
    p2 = ge2 - ge3
    p3 = ge3
    P = concat([p0, p1, p2, p3], axis=1) + T(np.array([1e-6]))
    B = score.d.shape[0]
    oh = np.zeros((B, n_levels))
    oh[np.arange(B), y] = 1.0
    return -rmean(log(rsum(P * T(oh), axis=1, keepdims=True)))


def _idx(t, i):
    out = T(t.d[i:i + 1], (t,))

    def bw():
        t.g[i:i + 1] += out.g
    out._bw = bw
    return out


LOSS_W = dict(adm=1.0, chap=0.6, maq=0.9, illa=0.7, reg=0.5, recon=0.3)


def total_loss(p, b, hard_liqa=True, include_recon=True, only_recon=False,
               rule="liqa"):
    o = forward(p, b, hard_liqa=hard_liqa, rule=rule)
    mask = T(b["mask"])
    l_adm = bce(o["adm_logit"], b["y_adm"])
    l_chap = ce(o["chap_logit"], b["y_chap"])
    # Transposed reports are rare; without an explicit positive weight the
    # head collapses to "nothing is transposed", which is the failure mode
    # every uncritical collector before him actually exhibited.
    pw_maq = T(1.0 + 8.0 * b["y_maq"])
    l_maq = rmean((softplus(o["maq_logit"])
                   - T(b["y_maq"]) * o["maq_logit"]) * mask * pw_maq)
    l_illa = bce(o["illa_logit"], b["y_illa"])
    l_reg = ordinal_nll(o["reg_score"], (p["th0"], p["dth"]), b["y_reg"])
    diff = (o["recon"] - T(b["surf"])) ** 2.0
    l_rec = rmean(rmean(diff, axis=2) * mask)
    if only_recon:
        L = LOSS_W["recon"] * l_rec
    else:
        L = (LOSS_W["adm"] * l_adm + LOSS_W["chap"] * l_chap
             + LOSS_W["maq"] * l_maq + LOSS_W["illa"] * l_illa
             + LOSS_W["reg"] * l_reg)
        if include_recon:
            L = L + LOSS_W["recon"] * l_rec
    parts = dict(adm=float(l_adm.d), chap=float(l_chap.d), maq=float(l_maq.d),
                 illa=float(l_illa.d), reg=float(l_reg.d), recon=float(l_rec.d))
    return L, o, parts


# ===========================================================================
# SECTION 4 -- GRADIENT CHECK (mandatory)
# ===========================================================================

def zero_grads(p):
    for v in p.values():
        v.g = np.zeros_like(v.d)


def grad_check(p, b, hard_liqa, exclude=(), n_probe=90, eps=1e-6, seed=0,
               **lkw):
    """Central-difference check against analytic gradients.

    Two gradients in this model are BIASED ON PURPOSE, and the check is
    written so that neither can hide behind the other:

      * the hard liqa' gate is piecewise constant, so its parameters are
        excluded when hard_liqa=True (straight-through estimator);
      * the lafz barrier severs the reconstruction gradient before it reaches
        the meaning slot, so the reconstruction term is checked separately,
        against only the parameters that live downstream of the barrier.

    Everything else must match central differences to high precision.
    """
    rng = np.random.default_rng(seed)
    zero_grads(p)
    L, _, _ = total_loss(p, b, hard_liqa=hard_liqa, **lkw)
    L.backward()
    analytic = {k: v.g.copy() for k, v in p.items()}

    names = [k for k in p if k not in exclude]
    worst, worst_abs, checked = 0.0, 0.0, 0
    for _ in range(n_probe):
        k = names[rng.integers(0, len(names))]
        flat = p[k].d.reshape(-1)
        i = rng.integers(0, flat.size)
        orig = flat[i]
        flat[i] = orig + eps
        Lp, _, _ = total_loss(p, b, hard_liqa=hard_liqa, **lkw)
        flat[i] = orig - eps
        Lm, _, _ = total_loss(p, b, hard_liqa=hard_liqa, **lkw)
        flat[i] = orig
        num = (float(Lp.d) - float(Lm.d)) / (2 * eps)
        ana = analytic[k].reshape(-1)[i]
        mag = abs(num) + abs(ana)
        worst_abs = max(worst_abs, abs(num - ana))
        # A relative error is only meaningful where there is a gradient to be
        # relatively wrong about; on coordinates whose true gradient is at the
        # level of central-difference noise it is reported as absolute error
        # instead.
        if mag > 1e-5:
            worst = max(worst, abs(num - ana) / mag)
        checked += 1
    return worst, worst_abs, checked


# ===========================================================================
# SECTION 5 -- TRAINING
# ===========================================================================

class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.d) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.d) for k, v in params.items()}
        self.t = 0

    def step(self, clip=5.0):
        self.t += 1
        for k, v in self.p.items():
            g = np.clip(v.g, -clip, clip)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            v.d -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(p, world, steps=260, bs=160, log_every=40, hard_liqa=True):
    opt = Adam(p)
    hist = []
    for t in range(1, steps + 1):
        b = world.batch(bs)
        zero_grads(p)
        L, o, parts = total_loss(p, b, hard_liqa=hard_liqa)
        L.backward()
        opt.step()
        hist.append(float(L.d))
        if t % log_every == 0 or t == 1:
            print(f"  step {t:4d}  loss {float(L.d):7.4f}   "
                  + "  ".join(f"{k}={v:.3f}" for k, v in parts.items())
                  + f"   beta={float(o['beta'].d[0]):.2f}")
    return hist


# ===========================================================================
# SECTION 6 -- EVALUATION AND SELF-TESTS
# ===========================================================================

def evaluate(p, world, n=600, hard_liqa=True, rule="liqa"):
    b = world.batch(n)
    _, o, _ = total_loss(p, b, hard_liqa=hard_liqa, rule=rule)
    adm = (o["adm_logit"].d > 0).astype(float)
    y = b["y_adm"]
    tp = float(((adm == 1) & (y == 1)).sum())
    fp = float(((adm == 1) & (y == 0)).sum())
    fn = float(((adm == 0) & (y == 1)).sum())
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    chap_acc = float((o["chap_logit"].d.argmax(1) == b["y_chap"]).mean())
    mm = b["mask"] > 0
    maq_pred = (o["maq_logit"].d > 0)[mm]
    maq_true = b["y_maq"][mm] > 0
    maq_acc = float((maq_pred == maq_true).mean())
    maq_rec = float(maq_pred[maq_true].mean()) if maq_true.any() else 0.0
    illa_acc = float(((o["illa_logit"].d > 0) == (b["y_illa"] > 0)).mean())
    # ordinal register accuracy
    return dict(precision=prec, recall=rec, f1=f1, chapter_acc=chap_acc,
                maqlub_acc=maq_acc, maqlub_recall=maq_rec, illa_acc=illa_acc,
                admit_rate=float(adm.mean()), base_rate=float(y.mean()))


def _distinct_chapter_batch(world, k):
    """Assemble k reports with mutually DISTINCT subjects.

    The scholars of Baghdad did not use obscure material; they used narrations
    everyone in the room already knew, precisely so that the only thing being
    tested was whether the chains had been switched. Distinct subjects are what
    make the transposition recoverable in principle -- two reports on the same
    subject travelling similar chains are genuinely indistinguishable, and no
    method, his or ours, can separate them.
    """
    pool = world.batch(max(60, 8 * k))
    seen, keep = set(), []
    for i, ch in enumerate(pool["y_chap"]):
        if ch not in seen:
            seen.add(int(ch))
            keep.append(i)
        if len(keep) == k:
            break
    idx = np.array(keep)
    return {kk: (v[idx] if hasattr(v, "__len__") and len(v) == len(pool["y_chap"])
                 else v) for kk, v in pool.items()}, len(keep)


def baghdad_trial(p, world, n_groups=10, per_group=8):
    """The trial of Baghdad, 100 reports.

    Ten men each recite ten narrations with the texts deliberately transposed
    onto one another's chains (maqlub). The task is to return every text to
    its own chain. The model does this with the bilinear matn/isnad
    compatibility score and a greedy maximum matching -- no gradient, no
    labels, purely the learned association between a body of wording and the
    line of men who could plausibly be carrying it.
    """
    total, correct = 0, 0
    for _ in range(n_groups):
        b, per_group = _distinct_chapter_batch(world, per_group)
        _, o, _ = total_loss(p, b, hard_liqa=True)
        h = o["hm"].d[:, 0, :]           # (n,d) first route's wording
        fp = o["fp"].d[:, 0, :]          # (n,d) first route's chain
        # A HIGH bilinear score is the model's evidence of transposition, so
        # the correct pairing is the one that minimises it. Negate and match.
        S = -((h @ p["M_maq"].d) @ fp.T)   # affinity[text i, chain j]
        S2 = S.copy()
        assign = {}
        for _ in range(per_group):
            i, j = np.unravel_index(np.argmax(S2), S2.shape)
            assign[i] = j
            S2[i, :] = -1e18
            S2[:, j] = -1e18
        for i, j in assign.items():
            correct += int(i == j)
            total += 1
    return correct / total


def run_tests(p, world):
    print("\n" + "=" * 74)
    print("SELF-TESTS")
    print("=" * 74)
    ok = True
    b = world.batch(48)

    # --- T1: gradient check, differentiable core, lax gate ----------------
    w, wa, n = grad_check(p, b, hard_liqa=False, n_probe=120, seed=11,
                          include_recon=False)
    t1 = (w < 2e-5) and (wa < 1e-6)
    print(f"T1  gradient check -- differentiable core, soft gate, all "
          f"{len(p)} parameter tensors, {n} probes")
    print(f"    max relative error = {w:.3e}   max absolute error = {wa:.3e}"
          f"   {'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # --- T2: gradient check, hard gate, gate params excluded --------------
    w2, wa2, n2 = grad_check(p, b, hard_liqa=True, exclude=GATE_PARAMS,
                             n_probe=120, seed=12, include_recon=False)
    t2 = (w2 < 2e-5) and (wa2 < 1e-6)
    print(f"T2  gradient check -- hard liqa' gate; gate parameters excluded, "
          f"the straight-through bias")
    print(f"    is intended, {n2} probes")
    print(f"    max relative error = {w2:.3e}   max absolute error = {wa2:.3e}"
          f"   {'PASS' if t2 else 'FAIL'}")
    ok &= t2

    # --- T2b: the reconstruction path, downstream of the barrier ----------
    w3, wa3, n3 = grad_check(p, b, hard_liqa=True,
                             exclude=tuple(k for k in p
                                           if k not in ("W_dec", "b_dec", "STY")),
                             n_probe=60, seed=13, only_recon=True)
    t2b = (w3 < 2e-5) and (wa3 < 1e-6)
    print(f"T2b gradient check -- utterance reconstruction, restricted to the "
          f"parameters BELOW")
    print(f"    the lafz barrier, {n3} probes")
    print(f"    max relative error = {w3:.3e}   max absolute error = {wa3:.3e}"
          f"   {'PASS' if t2b else 'FAIL'}")
    ok &= t2b

    # --- T3: the lafz barrier is real -------------------------------------
    zero_grads(p)
    o = forward(p, b, hard_liqa=True)
    rec_only = rmean((o["recon"] - T(b["surf"])) ** 2.0)
    rec_only.backward()
    gnorm_enc = float(np.abs(p["W_enc"].g).max())
    gnorm_dec = float(np.abs(p["W_dec"].g).max())
    print("T3  lafz barrier: reconstructing the WORDING must not write back "
          "into MEANING")
    print(f"    |dL_recon/dW_enc|max = {gnorm_enc:.3e}  (must be exactly 0)")
    print(f"    |dL_recon/dW_dec|max = {gnorm_dec:.3e}  (must be > 0)")
    t3 = (gnorm_enc == 0.0) and (gnorm_dec > 0)
    print(f"    {'PASS' if t3 else 'FAIL'}")
    ok &= t3

    # --- T4: weakest-link monotonicity and convergence --------------------
    bb = world.batch(96)
    pis = []
    for beta in (0.5, 2.0, 8.0, 40.0, 2000.0):
        pis.append(float(forward(p, bb, hard_liqa=True,
                                 beta_override=beta)["pi"].d.mean()))
    kap = forward(p, bb, hard_liqa=True)["kappa"].d
    true_min = float(kap.min(axis=2).mean())
    mono = all(pis[i] >= pis[i + 1] - 1e-9 for i in range(len(pis) - 1))
    above = all(v >= true_min - 1e-6 for v in pis)
    # softmin_beta(k) lies in [min(k), min(k) + log(n_joints)/beta] exactly;
    # the test asserts that envelope rather than an arbitrary tolerance.
    envelope = np.log(kap.shape[2]) / 2000.0
    conv = (pis[-1] - true_min) <= envelope + 1e-9
    print("T4  weakest link: the route score must fall monotonically in beta, "
          "never below the worst")
    print("    joint, and converge to it")
    print("    pi(beta = 0.5, 2, 8, 40, 2000) = "
          + ", ".join(f"{v:.4f}" for v in pis))
    print(f"    worst joint = {true_min:.4f}   analytic envelope at "
          f"beta=2000: +{envelope:.6f}")
    t4 = mono and above and conv
    print(f"    monotone={mono}  never-below={above}  converged={conv}   "
          f"{'PASS' if t4 else 'FAIL'}")
    ok &= t4

    # --- T5: noisy-OR bounds ----------------------------------------------
    o5 = forward(p, bb, hard_liqa=True)
    A, q = o5["A"].d, o5["q"].d
    bounds = bool((A >= -1e-9).all() and (A <= 1.0 + 1e-9).all())
    dominates = bool((A + 1e-6 >= q.max(axis=1, keepdims=True)).all())
    print("T5  corroboration: the noisy-OR stays in [0,1] and never scores "
          "below its own best route")
    print(f"    A in [{A.min():.4f}, {A.max():.4f}]   dominates best route: "
          f"{dominates}")
    print(f"    {'PASS' if (bounds and dominates) else 'FAIL'}")
    ok &= bounds and dominates

    # --- T6: the hard gate actually severs --------------------------------
    o6 = forward(p, bb, hard_liqa=True)
    gd = o6["g"].d.reshape(o6["kappa"].d.shape)
    frac_closed = float((gd < 0.5).mean())
    kap_closed = float(o6["kappa"].d[gd < 0.5].max()) if frac_closed > 0 else 0.0
    print("T6  a joint the model judges undocumented must contribute exactly "
          "nothing")
    print(f"    joints severed = {frac_closed*100:.1f}%   largest surviving "
          f"credibility on a severed joint = {kap_closed:.3e}")
    t6 = (kap_closed == 0.0) and (frac_closed > 0.01)
    print(f"    {'PASS' if t6 else 'FAIL'}")
    ok &= t6

    return ok


# ===========================================================================
# SECTION 7 -- DRIVER
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 74)
    print("ISNAD-NET  --  chapter 0207, al-Bukhari (810-870)")
    print("a route-scoring architecture: hard meeting gate, weakest-link path")
    print("score, overlap-discounted corroboration, severed wording/meaning path")
    print("=" * 74)

    world = IsnadWorld(seed=207)
    p = init_params(seed=7)
    n_par = sum(v.d.size for v in p.values())
    print(f"\nworld: {CFG['n_transmitters']} transmitters over "
          f"{CFG['n_generations']} generations, {CFG['n_regions']} regions")
    print(f"       real meetings: {int(world.met.sum())}   "
          f"documented: {int(((world.met > 0) & (world.attested > 0)).sum())}"
          f"   falsely documented: "
          f"{int(((world.met == 0) & (world.attested > 0)).sum())}")
    print(f"model: {len(p)} parameter tensors, {n_par} scalars\n")

    print("-" * 74)
    print("TRAINING  (strict gate: a joint must be shown to have happened)")
    print("-" * 74)
    hist = train(p, world, steps=420, bs=160, log_every=60, hard_liqa=True)
    print(f"\n  loss {hist[0]:.4f} -> {np.mean(hist[-10:]):.4f}  "
          f"({100*(1-np.mean(hist[-10:])/hist[0]):.1f}% reduction)")

    ok = run_tests(p, world)

    # ---- T7: the Baghdad trial -------------------------------------------
    acc = baghdad_trial(p, world, n_groups=12, per_group=8)
    print("T7  the trial of Baghdad: 96 narrations on distinct subjects, their "
          "texts transposed")
    print("    onto one another's chains, returned to their own by matn/isnad "
          "compatibility")
    print("    alone (chance = 12.5%)")
    print(f"    restored correctly = {acc*100:.1f}%   "
          f"{'PASS' if acc > 0.35 else 'FAIL'}")
    ok &= acc > 0.35

    print("\n" + "=" * 74)
    print("EVALUATION  (600 held-out reports)")
    print("=" * 74)
    ev = evaluate(p, world, n=600, hard_liqa=True)
    for k, v in ev.items():
        print(f"  {k:14s} {v:.4f}")
    oo = forward(p, world.batch(600), hard_liqa=True)
    print(f"  {'joints severed':14s} {float((oo['g'].d < 0.5).mean()):.4f}"
          f"   (chains cut for want of a documented meeting)")
    print(f"  {'chain weight':14s} "
          + " ".join(f"{v:.3f}" for v in
                     _softmax(p['pos_w'], axis=0).d)
          + "   (learned weight on chain positions, source -> collector)")

    # ---- the ablation the whole chapter turns on --------------------------
    print("\n" + "=" * 74)
    print("ABLATION: liqa' (documented meeting) vs mu'asara (contemporaneity)")
    print("same weights, same hard gate -- only the evidence bit is swapped")
    print("=" * 74)
    strict = evaluate(p, world, n=1200, rule="liqa")
    lax = evaluate(p, world, n=1200, rule="muasara")
    print(f"  {'':14s}{'precision':>12s}{'recall':>10s}{'F1':>10s}"
          f"{'admit rate':>13s}")
    print(f"  {'liqa (Bukhari)':14s}{strict['precision']:12.4f}"
          f"{strict['recall']:10.4f}{strict['f1']:10.4f}"
          f"{strict['admit_rate']:13.4f}")
    print(f"  {'mu asara':14s}{lax['precision']:12.4f}"
          f"{lax['recall']:10.4f}{lax['f1']:10.4f}"
          f"{lax['admit_rate']:13.4f}")

    # The corpus-wide figures understate the effect, because most corruptions
    # in the world are textual and neither rule touches them. Aim the question
    # at the two populations the rule actually governs.
    bb = world.batch(2000)
    res = {}
    for rule in ("liqa", "muasara"):
        _, o, _ = total_loss(p, bb, hard_liqa=True, rule=rule)
        adm = (o["adm_logit"].d[:, 0] > 0)
        res[rule] = adm
    forged = bb["mode"] == "forged"
    clean = bb["mode"] == "clean"
    print("\n  aimed at the populations the rule governs:")
    print(f"  {'':14s}{'admits FORGED':>16s}{'admits CLEAN':>16s}")
    print(f"  {'':14s}{'(want low)':>16s}{'(want high)':>16s}")
    for rule, lab in (("liqa", "liqa (Bukhari)"), ("muasara", "mu asara")):
        print(f"  {lab:14s}{res[rule][forged].mean():16.4f}"
              f"{res[rule][clean].mean():16.4f}")
    d_bad = res["liqa"][forged].mean() - res["muasara"][forged].mean()
    d_good = res["liqa"][clean].mean() - res["muasara"][clean].mean()
    print(f"\n  forged reports kept out: {-d_bad:.4f}")
    print(f"  clean reports lost:      {-d_good:.4f}")
    print("  -- the strict rule is not free, and he never claimed it was. The")
    print("     undocumented-but-genuine report is thrown away with the forged")
    print("     one, because from inside the archive they look identical.")

    # ---- the beta dial ----------------------------------------------------
    print("\n" + "=" * 74)
    print("THE WEAKEST-LINK DIAL: what happens when a chain is scored by its")
    print("average narrator instead of its worst")
    print("=" * 74)
    bb = world.batch(800)
    print(f"  {'beta':>8s}{'mean pi':>12s}{'reading':>34s}")
    for beta, label in ((0.4, "average narrator (lax)"),
                        (2.0, "mild weakest-link"),
                        (8.0, "strict weakest-link"),
                        (40.0, "the classical rule")):
        oo = forward(p, bb, hard_liqa=True, beta_override=beta)
        print(f"  {beta:8.1f}{float(oo['pi'].d.mean()):12.4f}{label:>34s}")
    print(f"  learned beta = {float(softplus(p['beta_raw']).d[0]+0.5):.3f}")

    print("\n" + "=" * 74)
    print(f"ALL TESTS {'PASSED' if ok else 'FAILED'}     "
          f"({time.time()-t0:.1f}s)")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

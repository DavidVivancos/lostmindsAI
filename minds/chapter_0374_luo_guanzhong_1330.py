#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE YANYI BRIDGE NETWORK  (演義橋接網絡)
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0374 · Luo Guanzhong (1330-1390)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 A from-scratch, pure-NumPy cognitive architecture derived from the documented
 compositional method of the Sanguozhi tongsu yanyi.
================================================================================

WHY THIS ARCHITECTURE IS NOT A TRANSFORMER
------------------------------------------
A causal language model answers "what comes next?".  Luo Guanzhong never had
that problem.  Every reader of a Three Kingdoms performance in fourteenth-
century Hangzhou already knew that Shu would fall, that Zhuge Liang would die
at the Wuzhang Plains, that Jin would swallow all three houses in 280.  The
terminus was public knowledge.  Suspense was structurally unavailable.

His problem was the opposite and much stranger: given a SPARSE, CONTRADICTORY,
ALREADY-ANNOTATED chronicle whose endpoints are fixed, produce a dense,
continuous, performable trajectory between them.  That is not prediction.
That is *bridging* — generation conditioned on both ends at once.

The 1494 preface by Jiang Daqi (writing as Yongyuzi), preserved in the earliest
extant 1522 edition, states the method in four characters:

    留心損益   liuxin sunyi   "he attentively DIMINISHED and AUGMENTED"

Two operators, not one.  Subtraction and augmentation, applied with care.  The
same preface fixes two bands rather than two targets:

    文不甚深，言不甚俗   "the writing not too deep, the wording not too vulgar"
    事紀其實，亦庶幾乎史 "events recorded truthfully, so it NEARLY resembles history"

庶幾 (shuji, "nearly / all but") is the operative word.  The goal is not history.
It is the largest neighbourhood of history that remains performable.

Lu Xun, surveying the genre in 1925, stated the allocation rule that the preface
left implicit:

    大抵史上大事，即無發揮，一涉細故，便多增飾
    "for the great events of history, no elaboration; but touching small
     matters, much augmentation"

That is the single most important mechanical fact in this file.  Invention is
allocated INVERSELY TO EVIDENTIAL DENSITY.  Fabrication mass goes into the gaps
between anchors, never onto the anchors themselves.

Zhang Xuecheng (1738-1801), in the Bingchen zhaji, measured the result and
complained about it: 七分實事，三分虛構 — seven parts real events, three parts
fabrication.  He is the first person on record to audit a generative system for
its hallucination rate and to observe that the 30% had overwritten the 70% in
the public mind.

And Lu Xun supplied the failure mode of over-optimisation:

    欲顯劉備之長厚而似偽，狀諸葛之多智而近妖
    "wishing to display Liu Bei's magnanimity, he made him seem FALSE;
     depicting Zhuge Liang's wisdom, he made him verge on the DEMONIC"

Push a character trait toward its maximum and the reader's belief does not rise
monotonically — it inverts.  Saturation reads as fabrication.  This is an
explicit, dated, pre-modern statement of mode collapse producing uncanniness,
and it enters this network as a real regulariser, not a metaphor.

THE SIX MECHANISMS IMPLEMENTED HERE
-----------------------------------
 1. BridgeCoordinate  — every step is conditioned on the PREVIOUS anchor, the
                        NEXT anchor, and its fractional position between them.
                        Non-causal by construction.
 2. SunYiCell         — a recurrent cell with two INDEPENDENT gates, one that
                        augments and one that diminishes.  Not a GRU: a GRU ties
                        them (z and 1-z).  Luo's operators are untied.
 3. PersonaField      — scenes are generated from CHARACTER PAIRS through a
                        low-rank bilinear interaction, because the unit of the
                        yanyi is the encounter, not the individual.
 4. XuShiGate         — a per-step scalar predicting fabrication mass, driven by
                        (1 - evidence), budgeted globally to 0.30.
 5. MirrorHead        — depth-2 belief.  Predicts what an ADVERSARY EXPECTS this
                        agent to do, from the agent's established disposition
                        prior.  The Empty Fort works only because Sima Yi holds
                        an accurate compressed model of Zhuge Liang's caution.
 6. RegisterBand      — a two-sided (not two-target) constraint on diction.

Everything below is written from scratch with NumPy only.  The file contains a
finite-difference gradient check over every parameter, a real training loop on
a real (small, hand-audited) chronicle ledger, and a suite of behavioural
self-tests.  Run it directly.
================================================================================
"""

import numpy as np

# ==============================================================================
# SECTION 1 — A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Written from scratch so that no autodiff framework is involved anywhere in
# this file.  Each Node holds a value, a gradient buffer, the list of parents it
# was produced from, and a closure that pushes gradient back to those parents.
# Backward is a reverse topological sweep.
#
# This is deliberately small.  It supports exactly the operations the Yanyi
# Bridge Network needs and nothing else.
# ==============================================================================


def _unbroadcast(grad, shape):
    """Reduce `grad` back to `shape`, undoing NumPy broadcasting.

    When we compute a + b with a.shape == (4, 8) and b.shape == (8,), NumPy
    silently expands b.  On the backward pass the gradient arriving for b has
    shape (4, 8) and must be summed back down to (8,).  This helper performs
    that reduction generically for any broadcast pattern.
    """
    if grad.shape == shape:
        return grad
    # First, collapse any leading axes that broadcasting added.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Then collapse any axis where the original had extent 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value in the computation graph, carrying its own backward closure."""

    __slots__ = ("v", "g", "parents", "_back", "name")

    def __init__(self, value, parents=(), back=None, name=""):
        self.v = np.asarray(value, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self.parents = parents
        self._back = back            # closure: pushes self.g into parents' .g
        self.name = name

    # ---- graph construction helpers -------------------------------------

    @property
    def shape(self):
        return self.v.shape

    def __repr__(self):
        return f"Node(name={self.name!r}, shape={self.v.shape})"

    # ---- arithmetic ------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.v + other.v, (self, other))

        def back():
            self.g += _unbroadcast(out.g, self.v.shape)
            other.g += _unbroadcast(out.g, other.v.shape)

        out._back = back
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Node(-self.v, (self,))

        def back():
            self.g -= out.g

        out._back = back
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __rsub__(self, other):
        return (Node(other) if not isinstance(other, Node) else other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.v * other.v, (self, other))

        def back():
            self.g += _unbroadcast(out.g * other.v, self.v.shape)
            other.g += _unbroadcast(out.g * self.v, other.v.shape)

        out._back = back
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.v / other.v, (self, other))

        def back():
            self.g += _unbroadcast(out.g / other.v, self.v.shape)
            other.g += _unbroadcast(-out.g * self.v / (other.v ** 2),
                                    other.v.shape)

        out._back = back
        return out

    def matmul(self, other):
        out = Node(self.v @ other.v, (self, other))

        def back():
            self.g += out.g @ other.v.T
            other.g += self.v.T @ out.g

        out._back = back
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- shape ops -------------------------------------------------------

    def reshape(self, *shape):
        old = self.v.shape
        out = Node(self.v.reshape(*shape), (self,))

        def back():
            self.g += out.g.reshape(old)

        out._back = back
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.v.sum(axis=axis, keepdims=keepdims), (self,))

        def back():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.v.shape).copy()

        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def select_rows(self, idx):
        """Embedding lookup: gather rows `idx` from this matrix."""
        idx = np.asarray(idx, dtype=np.int64)
        out = Node(self.v[idx], (self,))

        def back():
            np.add.at(self.g, idx, out.g)

        out._back = back
        return out

    # ---- nonlinearities --------------------------------------------------

    def tanh(self):
        t = np.tanh(self.v)
        out = Node(t, (self,))

        def back():
            self.g += out.g * (1.0 - t * t)

        out._back = back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.v, -60.0, 60.0)))
        out = Node(s, (self,))

        def back():
            self.g += out.g * s * (1.0 - s)

        out._back = back
        return out

    def relu(self):
        mask = (self.v > 0.0).astype(np.float64)
        out = Node(self.v * mask, (self,))

        def back():
            self.g += out.g * mask

        out._back = back
        return out

    def square(self):
        out = Node(self.v ** 2, (self,))

        def back():
            self.g += out.g * 2.0 * self.v

        out._back = back
        return out

    def log(self):
        out = Node(np.log(np.maximum(self.v, 1e-12)), (self,))

        def back():
            self.g += out.g / np.maximum(self.v, 1e-12)

        out._back = back
        return out

    def exp(self):
        e = np.exp(np.clip(self.v, -60.0, 60.0))
        out = Node(e, (self,))

        def back():
            self.g += out.g * e

        out._back = back
        return out

    def log_softmax(self, axis=-1):
        """Numerically stable log-softmax with an exact analytic backward."""
        x = self.v
        m = x.max(axis=axis, keepdims=True)
        z = x - m
        lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
        ls = z - lse
        soft = np.exp(ls)
        out = Node(ls, (self,))

        def back():
            self.g += out.g - soft * out.g.sum(axis=axis, keepdims=True)

        out._back = back
        return out

    # ---- backward sweep --------------------------------------------------

    def backward(self):
        """Reverse topological traversal, seeding d(self)/d(self) = 1."""
        topo, seen = [], set()

        def build(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                build(p)
            topo.append(node)

        build(self)
        self.g = np.ones_like(self.v)
        for node in reversed(topo):
            if node._back is not None:
                node._back()


# ---- free functions ------------------------------------------------------

def concat(nodes, axis=-1):
    """Concatenate Nodes along `axis`, routing gradient back by slice."""
    parts = [n.v for n in nodes]
    out = Node(np.concatenate(parts, axis=axis), tuple(nodes))
    sizes = [p.shape[axis] for p in parts]

    def back():
        offset = 0
        for node, size in zip(nodes, sizes):
            sl = [slice(None)] * out.g.ndim
            sl[axis] = slice(offset, offset + size)
            node.g += out.g[tuple(sl)]
            offset += size

    out._back = back
    return out


def stack_rows(nodes):
    """Stack a list of 1-D Nodes into a 2-D Node (one row per input)."""
    out = Node(np.stack([n.v for n in nodes], axis=0), tuple(nodes))

    def back():
        for i, node in enumerate(nodes):
            node.g += out.g[i]

    out._back = back
    return out


def cross_entropy(logits, targets):
    """Mean categorical cross-entropy from raw logits.

    logits  : Node of shape (T, K)
    targets : integer array of shape (T,)
    """
    ls = logits.log_softmax(axis=-1)
    T = ls.v.shape[0]
    onehot = np.zeros_like(ls.v)
    onehot[np.arange(T), np.asarray(targets, dtype=np.int64)] = 1.0
    picked = ls * Node(onehot)
    return -(picked.sum() * (1.0 / T))

# ==============================================================================
# SECTION 2 — THE CHRONICLE LEDGER
# ==============================================================================
# The training data is not synthetic.  It is a hand-audited provenance ledger of
# episodes from the Three Kingdoms tradition, 184-280 CE, each tagged with:
#
#   evidence  e in [0,1]  — how thickly the surviving record covers this item
#                           1.00  dated event in Chen Shou's main text
#                           0.40-0.60  Pei Songzhi annotation / contested variant
#                           0.05-0.20  invention of the pinghua, stage, or novel
#
# This column is the substrate for Lu Xun's allocation rule.  The network must
# learn to put its fabrication mass where e is low and to leave the anchors
# untouched, because that is what Luo Guanzhong demonstrably did.
#
# Three items deserve comment, because they carry the architecture:
#
#  * THE EMPTY FORT (228).  Guo Chong's "San shi" recorded it; Pei Songzhi
#    appended it to Zhuge Liang's biography in 429 CE and then, in his own
#    voice, demolished it — Sima Yi was commanding at Wancheng, not in that
#    theatre, and an encircling force would simply have encircled.  The record
#    itself is stamped UNRELIABLE.  Luo promoted it anyway, and it became the
#    culture's canonical demonstration of recursive mind-reading.  A false item
#    with the highest cognitive payload in the corpus.  e = 0.20.
#
#  * BORROWING ARROWS WITH STRAW BOATS (208).  A real incident, but Sun Quan's,
#    at Ruxu around 213, preserved in the Weilue.  Luo transposed it onto Zhuge
#    Liang five years earlier.  Attribution theft, not invention.  e = 0.20.
#
#  * SCRAPING THE BONE (219).  The surgery is in Chen Shou's main text; only the
#    surgeon's name (Hua Tuo, dead by then) is the novel's addition.  A case
#    where the augmentation is a single wrong noun.  e = 0.60.
#
# The z_target column is a deliberately coarse three-vector giving the share of
# the realm held by (Wei/Jin, Shu, Wu) at the anchor years.  It is stylised, not
# measured, and it is the only stylised column in the ledger; it exists so that
# the bridge has real endpoints to be pinned to, terminating at [1, 0, 0] in 280.
# ==============================================================================

# Scene classes -----------------------------------------------------------
OATH, COUNSEL, BATTLE, STRATAGEM, DEATH, ACCESSION = range(6)
SCENE_NAMES = ["OATH", "COUNSEL", "BATTLE", "STRATAGEM", "DEATH", "ACCESSION"]
K_SCENES = 6

# Dramatis personae -------------------------------------------------------
PEOPLE = ["Liu Bei", "Guan Yu", "Zhang Fei", "Zhuge Liang", "Cao Cao",
          "Sun Quan", "Zhou Yu", "Sima Yi", "Lu Bu", "Dong Zhuo",
          "Lu Xun (Wu)", "Liu Shan", "Cao Pi", "Emperor Xian"]
M_PEOPLE = len(PEOPLE)

# The DISPOSITION PRIOR: the scene class an informed adversary expects from
# each figure.  This is the compressed reputation that the MirrorHead learns,
# and it is what the Empty Fort exploits: Sima Yi's model of Zhuge Liang is
# COUNSEL — measured, cautious, never gambling — so an open gate must be a trap.
DISPOSITION_PRIOR = {
    0: OATH, 1: BATTLE, 2: BATTLE, 3: COUNSEL, 4: STRATAGEM,
    5: ACCESSION, 6: STRATAGEM, 7: COUNSEL, 8: BATTLE, 9: ACCESSION,
    10: COUNSEL, 11: ACCESSION, 12: ACCESSION, 13: ACCESSION,
}

# (year, label, actor_i, actor_j, scene, evidence, z_target|None, risk)
#
# `risk` is the eighth column and it is what the Empty Fort actually turns on.
# It scores how far the ACTOR departs from a hedged, reversible course at that
# moment: 0.05 is a man doing the obvious safe thing, 1.00 is a man betting his
# life on a single unhedged gesture.  Sima Yi's recorded reason for withdrawing
# is not about gates: 亮平生謹慎 — "Liang was cautious his whole life."  He is
# quoting a RISK PRIOR, and the stratagem is a deliberate excursion out of it.
LEDGER = [
    (184, "Yellow Turban rising",                     4,  0, BATTLE,    1.00, [0.00, 0.00, 0.00], 0.30),
    (184, "Peach Garden oath",                        0,  1, OATH,      0.08, None, 0.55),
    (189, "Dong Zhuo deposes the emperor",            9, 13, ACCESSION, 1.00, None, 0.90),
    (190, "Coalition against Dong Zhuo",              4,  9, BATTLE,    1.00, None, 0.45),
    (190, "Three heroes at Hulao Pass",               1,  8, BATTLE,    0.08, None, 0.60),
    (191, "Guan Yu slays Hua Xiong",                  1,  9, BATTLE,    0.08, None, 0.55),
    (192, "The chain stratagem / Diaochan",           9,  8, STRATAGEM, 0.12, None, 0.30),
    (192, "Dong Zhuo assassinated",                   8,  9, DEATH,     1.00, [0.05, 0.00, 0.00], 0.85),
    (196, "Cao Cao receives the emperor at Xu",       4, 13, ACCESSION, 1.00, None, 0.40),
    (198, "Lu Bu executed at Xiapi",                  4,  8, DEATH,     1.00, None, 0.25),
    (200, "Battle of Guandu",                         4,  0, BATTLE,    1.00, [0.35, 0.00, 0.05], 0.85),
    (200, "The thousand-li ride",                     1,  0, OATH,      0.10, None, 0.70),
    (207, "Three visits to the thatched cottage",     0,  3, COUNSEL,   0.85, None, 0.20),
    (208, "The Longzhong plan",                       3,  0, COUNSEL,   1.00, None, 0.20),
    (208, "Debating the scholars of Wu",              3,  5, COUNSEL,   0.08, None, 0.35),
    (208, "Straw boats borrow arrows",                3,  6, STRATAGEM, 0.20, None, 0.50),
    (208, "Borrowing the east wind",                  3,  6, STRATAGEM, 0.05, None, 0.45),
    (208, "Battle of Red Cliffs",                     4,  6, BATTLE,    1.00, [0.45, 0.05, 0.15], 0.75),
    (208, "Huarong road: Cao Cao released",           1,  4, STRATAGEM, 0.10, None, 0.65),
    (210, "Death of Zhou Yu",                         6,  5, DEATH,     1.00, None, 0.20),
    (211, "Liu Bei enters Yi province",               0,  3, BATTLE,    1.00, None, 0.55),
    (214, "Liu Bei takes Chengdu",                    0,  3, ACCESSION, 1.00, [0.50, 0.20, 0.20], 0.35),
    (215, "The Jing province dispute",                0,  5, COUNSEL,   1.00, None, 0.30),
    (215, "Single sword to the meeting",              1, 10, COUNSEL,   0.45, None, 0.75),
    (219, "Liu Bei takes Hanzhong",                   0,  4, BATTLE,    1.00, None, 0.50),
    (219, "Guan Yu floods the seven armies",          1,  4, BATTLE,    1.00, None, 0.55),
    (219, "Scraping the bone to cure the poison",     1,  0, COUNSEL,   0.60, None, 0.45),
    (219, "Guan Yu taken and killed at Maicheng",     1,  5, DEATH,     1.00, [0.55, 0.20, 0.25], 0.80),
    (220, "Death of Cao Cao",                         4,  7, DEATH,     1.00, None, 0.10),
    (220, "Cao Pi accepts the abdication; Wei",      12, 13, ACCESSION, 1.00, None, 0.60),
    (221, "Liu Bei proclaims himself emperor",        0,  3, ACCESSION, 1.00, None, 0.45),
    (221, "Zhang Fei killed by his own officers",     2,  0, DEATH,     1.00, None, 0.65),
    (222, "Xiaoting / Yiling",                        0, 10, BATTLE,    1.00, [0.60, 0.15, 0.25], 0.90),
    (223, "Baidicheng: the entrusting of the heir",   0,  3, DEATH,     1.00, None, 0.25),
    (225, "Seven captures of Meng Huo",               3, 11, STRATAGEM, 0.45, None, 0.40),
    (227, "The memorial on sending out the army",     3, 11, COUNSEL,   1.00, None, 0.45),
    (228, "Jieting lost; Ma Su executed",             3,  7, BATTLE,    1.00, None, 0.35),
    (228, "THE EMPTY FORT",                           3,  7, STRATAGEM, 0.20, None, 1.00),
    (231, "Wooden ox and flowing horse",              3,  7, COUNSEL,   0.90, None, 0.10),
    (234, "Praying at the seven lamps for life",      3, 11, COUNSEL,   0.05, None, 0.30),
    (234, "Wuzhang plains: death of Zhuge Liang",     3,  7, DEATH,     1.00, [0.62, 0.13, 0.25], 0.15),
    (234, "Dead Zhuge routs living Zhongda",          7,  3, STRATAGEM, 0.45, None, 0.05),
    (238, "Sima Yi annihilates Gongsun Yuan",         7, 12, BATTLE,    1.00, None, 0.45),
    (239, "Sima Yi made co-regent for Cao Fang",      7, 12, ACCESSION, 1.00, None, 0.20),
    (247, "Sima Yi feigns senility for Li Sheng",     7, 12, STRATAGEM, 0.85, None, 0.15),
    (249, "Incident at the Gaoping tombs",            7, 12, STRATAGEM, 1.00, None, 0.95),
    (251, "Death of Sima Yi",                         7, 11, DEATH,     1.00, None, 0.05),
    (263, "Shu Han surrenders",                      11,  7, ACCESSION, 1.00, [0.75, 0.00, 0.25], 0.20),
    (265, "The Jin dynasty proclaimed",               7, 12, ACCESSION, 1.00, None, 0.55),
    (280, "Wu surrenders; the realm is one",          5,  7, ACCESSION, 1.00, [1.00, 0.00, 0.00], 0.20),
]

EMPTY_FORT_INDEX = [i for i, r in enumerate(LEDGER) if r[1] == "THE EMPTY FORT"][0]


def build_dataset():
    """Turn the ledger into arrays, and precompute the BRIDGE COORDINATES.

    The bridge coordinate is the heart of the whole design.  For every step t we
    find the nearest anchor at or before t (call its state z_L) and the nearest
    anchor at or after t (z_R), and the fractional position tau in [0,1] between
    them.  The network therefore always sees WHERE IT IS GOING before it decides
    what to say — which is the condition Luo actually wrote under, and which no
    left-to-right predictor enjoys.
    """
    T = len(LEDGER)
    years = np.array([r[0] for r in LEDGER], dtype=np.float64)
    actor_i = np.array([r[2] for r in LEDGER], dtype=np.int64)
    actor_j = np.array([r[3] for r in LEDGER], dtype=np.int64)
    scene = np.array([r[4] for r in LEDGER], dtype=np.int64)
    evidence = np.array([r[5] for r in LEDGER], dtype=np.float64)
    prior = np.array([DISPOSITION_PRIOR[i] for i in actor_i], dtype=np.int64)

    risk = np.array([r[7] for r in LEDGER], dtype=np.float64)

    anchor_mask = np.array([1.0 if r[6] is not None else 0.0 for r in LEDGER])
    z_target = np.array([r[6] if r[6] is not None else [0.0, 0.0, 0.0]
                         for r in LEDGER], dtype=np.float64)

    # ---- THE DISPOSITION PRIOR, EARNED RATHER THAN DECLARED -----------------
    # For each step, look only at the acting figure's STRICTLY EARLIER
    # appearances and compute two things an adversary could also compute:
    #
    #   expected_risk  the running mean of how boldly this figure has acted
    #   legibility     how confidently that mean can be relied on - it grows
    #                  with the number of observations and shrinks with their
    #                  spread.  A figure seen once is not legible.  A figure
    #                  seen ten times, always doing the same measured thing, is
    #                  extremely legible - and therefore extremely exploitable
    #                  by exactly one well-chosen departure.
    #
    # This is the quantity Sima Yi is holding when he turns his army around,
    # and the quantity Sima Yi spends two years manufacturing before the coup.
    global_mean = float(risk.mean())
    expected_risk = np.zeros(T)
    legibility = np.zeros(T)
    for t in range(T):
        hist = [risk[s_] for s_ in range(t) if actor_i[s_] == actor_i[t]]
        n = len(hist)
        if n == 0:
            expected_risk[t] = global_mean
            legibility[t] = 0.0
        else:
            mu = float(np.mean(hist))
            sd = float(np.std(hist)) if n > 1 else 0.0
            expected_risk[t] = mu
            legibility[t] = (n / (n + 2.0)) * max(0.0, 1.0 - 2.0 * sd)

    anchor_idx = [t for t in range(T) if anchor_mask[t] > 0.5]

    zL = np.zeros((T, 3))
    zR = np.zeros((T, 3))
    tau = np.zeros((T, 1))
    for t in range(T):
        left = max([a for a in anchor_idx if a <= t], default=anchor_idx[0])
        right = min([a for a in anchor_idx if a >= t], default=anchor_idx[-1])
        zL[t] = z_target[left]
        zR[t] = z_target[right]
        span = years[right] - years[left]
        tau[t, 0] = 0.0 if span <= 0 else (years[t] - years[left]) / span

    # Bridge feature block: both endpoints, the position between them, a smooth
    # basis over that position, and the local evidence density.
    bridge = np.concatenate([
        zL, zR, tau, 1.0 - tau, tau * (1.0 - tau),
        np.sin(np.pi * tau), np.cos(np.pi * tau),
        evidence.reshape(-1, 1),
    ], axis=1)                                             # (T, 11)

    return dict(T=T, years=years, actor_i=actor_i, actor_j=actor_j,
                scene=scene, evidence=evidence, prior=prior, risk=risk,
                expected_risk=expected_risk, legibility=legibility,
                anchor_mask=anchor_mask, z_target=z_target, bridge=bridge,
                tau=tau)


# ==============================================================================
# SECTION 3 — THE MODEL
# ==============================================================================

H_HIDDEN = 24          # width of the narrative state
D_TRAIT = 4            # trait dims per person: ren / zhi / yong / zha
R_BILINEAR = 3         # rank of the pairwise persona interaction
D_BRIDGE = 12          # width of the bridge feature block
XU_BUDGET = 0.30       # Zhang Xuecheng's audited fabrication share
UNCANNY_THETA = 1.50   # Lu Xun's saturation threshold, beyond which a trait
                       # stops reading as a virtue and starts reading as a lie
REGISTER_LO = -1.0     # yan bu shen su   — not too vulgar
REGISTER_HI = 1.0      # wen bu shen shen — not too deep


def he_init(rng, shape, gain=1.0):
    fan_in = shape[0]
    return rng.normal(0.0, gain * np.sqrt(1.0 / fan_in), size=shape)


class YanyiBridgeNetwork:
    """演義橋接網絡 — a terminal-anchored, gap-weighted narrative compiler."""

    def __init__(self, seed=296):
        rng = np.random.default_rng(seed)
        D_persona = 2 * D_TRAIT + R_BILINEAR              # 11
        D_in = D_BRIDGE + D_persona + H_HIDDEN            # 12 + 11 + 24 = 47

        P = {}
        # -- persona field -------------------------------------------------
        P["E"] = Node(rng.normal(0, 0.5, (M_PEOPLE, D_TRAIT)), name="E")
        P["U"] = Node(he_init(rng, (D_TRAIT, R_BILINEAR)), name="U")
        P["V"] = Node(he_init(rng, (D_TRAIT, R_BILINEAR)), name="V")
        # -- the sun/yi cell (two untied gates) ----------------------------
        P["W_yi"] = Node(he_init(rng, (D_in, H_HIDDEN)), name="W_yi")
        P["b_yi"] = Node(np.zeros((1, H_HIDDEN)), name="b_yi")
        P["W_sun"] = Node(he_init(rng, (D_in, H_HIDDEN)), name="W_sun")
        P["b_sun"] = Node(np.full((1, H_HIDDEN), -1.0), name="b_sun")
        P["W_c"] = Node(he_init(rng, (D_in, H_HIDDEN)), name="W_c")
        P["b_c"] = Node(np.zeros((1, H_HIDDEN)), name="b_c")
        # -- heads ---------------------------------------------------------
        P["W_s"] = Node(he_init(rng, (H_HIDDEN, K_SCENES)), name="W_s")
        P["b_s"] = Node(np.zeros((1, K_SCENES)), name="b_s")
        P["W_m"] = Node(he_init(rng, (H_HIDDEN, K_SCENES)), name="W_m")
        P["b_m"] = Node(np.zeros((1, K_SCENES)), name="b_m")
        # second mirror output: the adversary's expected BOLDNESS for this actor
        P["W_mr"] = Node(he_init(rng, (H_HIDDEN, 1)), name="W_mr")
        P["b_mr"] = Node(np.zeros((1, 1)), name="b_mr")
        P["W_v"] = Node(he_init(rng, (H_HIDDEN, 1)), name="W_v")
        P["b_v"] = Node(np.zeros((1, 1)), name="b_v")
        P["W_r"] = Node(he_init(rng, (H_HIDDEN, 1)), name="W_r")
        P["b_r"] = Node(np.zeros((1, 1)), name="b_r")
        P["W_z"] = Node(he_init(rng, (H_HIDDEN, 3)), name="W_z")
        P["b_z"] = Node(np.zeros((1, 3)), name="b_z")
        self.P = P

    def parameters(self):
        return self.P

    def zero_grad(self):
        for p in self.P.values():
            p.g = np.zeros_like(p.v)

    # ------------------------------------------------------------------
    def forward(self, data):
        """Roll the network over the whole chronicle and return every head."""
        P = self.P
        T = data["T"]
        bridge_all = Node(data["bridge"])

        emb_i = P["E"].select_rows(data["actor_i"])        # (T, D_TRAIT)
        emb_j = P["E"].select_rows(data["actor_j"])        # (T, D_TRAIT)
        # Low-rank bilinear encounter feature: the scene is a property of the
        # PAIR.  (x_i U) * (x_j V) is a rank-R factorisation of x_i^T W x_j.
        bil = (emb_i @ P["U"]) * (emb_j @ P["V"])          # (T, R_BILINEAR)
        persona_all = concat([emb_i, emb_j, bil], axis=-1)  # (T, 11)

        h = Node(np.zeros((1, H_HIDDEN)))
        H_rows, gates_yi, gates_sun = [], [], []

        for t in range(T):
            b_t = bridge_all.select_rows([t])              # (1, 11)
            p_t = persona_all.select_rows([t])             # (1, 11)
            x = concat([b_t, p_t, h], axis=-1)             # (1, D_in)

            # 益 — the augmentation gate: how much invented material to add.
            g_yi = (x @ P["W_yi"] + P["b_yi"]).sigmoid()
            # 損 — the diminution gate: how much of the inherited chronicle to
            # cut.  Untied from g_yi.  A GRU would force g_sun = 1 - g_yi and
            # thereby make it impossible to add nothing while cutting a great
            # deal, or to add a great deal while cutting nothing.  Luo did both.
            g_sun = (x @ P["W_sun"] + P["b_sun"]).sigmoid()
            cand = (x @ P["W_c"] + P["b_c"]).tanh()

            h = h + g_yi * cand - g_sun * h

            H_rows.append(h.reshape(H_HIDDEN))
            gates_yi.append(g_yi.reshape(H_HIDDEN))
            gates_sun.append(g_sun.reshape(H_HIDDEN))

        Hs = stack_rows(H_rows)                            # (T, H)
        out = {
            "H": Hs,
            "scene_logits": Hs @ P["W_s"] + P["b_s"],      # (T, K)
            "mirror_logits": Hs @ P["W_m"] + P["b_m"],     # (T, K)
            "mirror_risk": (Hs @ P["W_mr"] + P["b_mr"]).sigmoid(),  # (T, 1)
            "xu": (Hs @ P["W_v"] + P["b_v"]).sigmoid(),    # (T, 1) fabrication
            "register": Hs @ P["W_r"] + P["b_r"],          # (T, 1)
            "z_hat": Hs @ P["W_z"] + P["b_z"],             # (T, 3)
            "g_yi": stack_rows(gates_yi),
            "g_sun": stack_rows(gates_sun),
        }
        return out

    # ------------------------------------------------------------------
    def loss(self, data, out, weights=None):
        """The seven-term objective.  Every term is textually motivated."""
        w = weights or dict(scene=1.0, mirror=0.5, mirror_risk=2.0, bridge=2.0,
                            budget=4.0, alloc=1.5, band=0.5, uncanny=0.6)
        T = data["T"]
        terms = {}

        # (1) Tell the story: which kind of scene occurs here.
        terms["scene"] = cross_entropy(out["scene_logits"], data["scene"])

        # (2) MIRROR — learn what the adversary EXPECTS of the actor, i.e. the
        #     actor's compressed reputation, not the truth of the moment.  This
        #     head is what makes the Empty Fort computable.
        terms["mirror"] = cross_entropy(out["mirror_logits"], data["prior"])

        # (2b) MIRROR-RISK — regress the adversary's expected boldness onto the
        #      actor's own track record.  Crucially this head is NOT trained on
        #      what the actor is about to do; it is trained on what the actor
        #      has always done.  It is a model of reputation, and reputation is
        #      the resource the Empty Fort spends.
        terms["mirror_risk"] = (out["mirror_risk"]
                                - Node(data["expected_risk"].reshape(-1, 1))
                                ).square().mean()

        # (3) BRIDGE — at every anchor the reconstructed world-state must match
        #     the record.  Free invention between anchors; zero tolerance on.
        mask = Node(data["anchor_mask"].reshape(-1, 1))
        diff = (out["z_hat"] - Node(data["z_target"])) * mask
        n_anch = float(data["anchor_mask"].sum())
        terms["bridge"] = diff.square().sum() * (1.0 / (3.0 * n_anch))

        # (4) BUDGET — Zhang Xuecheng's audit: total fabrication mass -> 0.30.
        mean_xu = out["xu"].mean()
        terms["budget"] = (mean_xu - Node(XU_BUDGET)).square()

        # (5) ALLOCATION — Lu Xun's rule: fabricate in inverse proportion to
        #     evidential density.  Target v_t = 1 - e_t at every step.  Note
        #     that (4) and (5) genuinely disagree: on this ledger the mean gap
        #     mass is about 0.26, below the 0.30 the tradition actually exhibits.
        #     So the network cannot satisfy both by sitting still.  It has to
        #     decide WHICH gaps absorb the surplus - which is exactly the choice
        #     Luo made when he spent his margin on Red Cliffs and the Empty Fort
        #     rather than spreading it evenly across ninety-six years.
        v_target = Node((1.0 - data["evidence"]).reshape(-1, 1))
        terms["alloc"] = (out["xu"] - v_target).square().mean()

        # (6) REGISTER BAND — two-sided, no target.  Anything inside the band
        #     is equally acceptable; only leaving it is penalised.
        hi_viol = (out["register"] - Node(REGISTER_HI)).relu().square()
        lo_viol = (Node(REGISTER_LO) - out["register"]).relu().square()
        terms["band"] = (hi_viol + lo_viol).mean()

        # (7) UNCANNY — Lu Xun's saturation penalty.  Traits are free to move
        #     inside |x| <= theta and punished quadratically outside it, because
        #     a maximised virtue reads as a forgery.
        E = self.P["E"]
        absE = E.relu() + (-E).relu()
        terms["uncanny"] = (absE - Node(UNCANNY_THETA)).relu().square().mean()

        total = None
        for k, node in terms.items():
            piece = node * w[k]
            total = piece if total is None else total + piece
        return total, terms


# ==============================================================================
# SECTION 4 — ADAM, WRITTEN OUT
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(p.v) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.v) for k, p in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.params.items():
            g = np.clip(p.g, -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# SECTION 5 — DECEPTION SCORE:  legibility x violation
# ==============================================================================
# The Empty Fort is not a trick about gates.  It is a trick about PRIORS.
#
# Sima Yi withdraws because he holds a sharp, accurate, compressed model of
# Zhuge Liang: this man does not gamble.  The open gate is therefore evidence
# of a trap, not of emptiness.  The stratagem's whole power is borrowed from
# the QUALITY of the adversary's model.  Against an adversary who had never
# heard of Zhuge Liang, the gate is just an open gate.
#
# So the payoff of an off-prior act is the product of two quantities:
#
#     deception_t = precision_t  x  surprise_t
#     precision_t = max_k mirror_t[k]          (how sharply you are modelled)
#     surprise_t  = -log mirror_t[actual_t]    (how far you have just departed)
#
# A legible agent that never deviates scores zero.  An illegible agent that
# always deviates also scores zero, because nobody has a prior to violate.  The
# maximum sits at agents who spend a long time being predictable in order to
# buy one unpredictable move.  That is a cognitive strategy no purely honest
# and no purely random policy can reach, and it falls straight out of the
# MirrorHead without being hand-coded.
# ==============================================================================

def deception_scores(expected_risk_hat, actual_risk, legibility):
    """deception = legibility x positive excursion above the expected boldness.

    Only upward departures count.  Being more cautious than your reputation is
    not a stratagem; it is merely a disappointment.  And an unknown agent's
    boldest act scores zero, because there was no prior in anyone's head for it
    to violate.  The metric therefore has an interior maximum in exactly the
    place Luo put his two greatest set pieces.
    """
    excursion = np.maximum(0.0, actual_risk - expected_risk_hat.reshape(-1))
    return legibility * excursion


# ==============================================================================
# SECTION 6 — FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(model, data, n_probe=4, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    model.zero_grad()
    out = model.forward(data)
    total, _ = model.loss(data, out)
    total.backward()
    analytic = {k: p.g.copy() for k, p in model.P.items()}

    worst, report = 0.0, []
    for k, p in model.P.items():
        flat = p.v.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in picks:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = model.loss(data, model.forward(data))
            flat[i] = orig - eps
            lm, _ = model.loss(data, model.forward(data))
            flat[i] = orig
            num = (float(lp.v) - float(lm.v)) / (2 * eps)
            ana = analytic[k].reshape(-1)[i]
            denom = max(abs(num), abs(ana), 1e-8)
            errs.append(abs(num - ana) / denom)
        m = max(errs)
        worst = max(worst, m)
        report.append((k, m))
    return worst, report


# ==============================================================================
# SECTION 7 — TRAIN, THEN INTERROGATE
# ==============================================================================

def train(model, data, epochs=900, lr=0.05, log_every=150, verbose=True):
    opt = Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.zero_grad()
        out = model.forward(data)
        total, terms = model.loss(data, out)
        total.backward()
        opt.step()
        history.append(float(total.v))
        if verbose and (ep == 1 or ep % log_every == 0 or ep == epochs):
            bits = " ".join(f"{k}={float(v.v):.4f}" for k, v in terms.items())
            print(f"  epoch {ep:4d} | total={float(total.v):8.5f} | {bits}")
    return history


def run_self_tests(model, data, history):
    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    out = model.forward(data)
    xu = out["xu"].v.reshape(-1)
    ev = data["evidence"]
    scene_pred = out["scene_logits"].v.argmax(axis=1)
    mirror_pred = out["mirror_logits"].v.argmax(axis=1)
    dec = deception_scores(out["mirror_risk"].v, data["risk"], data["legibility"])
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")

    # 1. optimisation actually happened
    drop = history[0] - history[-1]
    check("loss decreased", drop > 0.5 and history[-1] < history[0],
          f"{history[0]:.4f} -> {history[-1]:.4f}  (-{drop:.4f})")

    # 2. Lu Xun's allocation rule was learned, not imposed at inference
    corr = float(np.corrcoef(xu, ev)[0, 1])
    check("invention anti-correlates with evidence", corr < -0.80,
          f"corr(xu, evidence) = {corr:+.4f}")

    # 3. Zhang Xuecheng's audited ratio
    mean_xu = float(xu.mean())
    check("fabrication share near 7:3", abs(mean_xu - XU_BUDGET) < 0.04,
          f"mean xu = {mean_xu:.4f} (target {XU_BUDGET})")

    # 4. anchors are untouched
    anch = data["anchor_mask"] > 0.5
    err = float(np.abs(out["z_hat"].v[anch] - data["z_target"][anch]).max())
    check("world-state pinned at every anchor", err < 0.08,
          f"max |z_hat - z*| = {err:.5f}")

    # 5. terminal condition: the realm is one in 280
    z_end = out["z_hat"].v[-1]
    check("terminus reached ([1,0,0] in 280)",
          abs(z_end[0] - 1.0) < 0.10 and abs(z_end[1]) < 0.10,
          f"z(280) = [{z_end[0]:.3f}, {z_end[1]:.3f}, {z_end[2]:.3f}]")

    # 6. THE EMPTY FORT — must emerge as the top deception in the whole corpus,
    #    without ever being told so.  Two independent conditions: the adversary
    #    must expect COUNSEL of Zhuge Liang, and the excursion must rank first.
    ef = EMPTY_FORT_INDEX
    ef_rank = int((dec > dec[ef]).sum()) + 1
    ef_ok = (mirror_pred[ef] == DISPOSITION_PRIOR[3]
             and data["scene"][ef] == STRATAGEM
             and ef_rank == 1)
    check("Empty Fort ranks first in deception", ef_ok,
          f"expected={SCENE_NAMES[mirror_pred[ef]]}, actual="
          f"{SCENE_NAMES[data['scene'][ef]]}, r_hat="
          f"{float(out['mirror_risk'].v[ef, 0]):.3f}, D={dec[ef]:.3f}, "
          f"rank {ef_rank}/{len(dec)}")

    # 6b. the pupil repeats the master: Sima Yi's coup should sit right behind
    #     it, because he manufactures the same resource by feigning senility.
    gp = [i for i, r in enumerate(LEDGER) if r[1].startswith("Incident at the")][0]
    gp_rank = int((dec > dec[gp]).sum()) + 1
    check("Gaoping coup is the same move, learned", gp_rank <= 3,
          f"Sima Yi D={dec[gp]:.3f}, rank {gp_rank}/{len(dec)} "
          f"(legibility bought at 238-247)")

    # 7. Lu Xun's uncanny bound holds on every trait
    max_trait = float(np.abs(model.P["E"].v).max())
    check("no trait saturates (|x| <= theta)", max_trait <= UNCANNY_THETA + 0.25,
          f"max|trait| = {max_trait:.4f}, theta = {UNCANNY_THETA}")

    # 8. the 1494 register band
    reg = out["register"].v.reshape(-1)
    inb = float(((reg >= REGISTER_LO - 0.15) & (reg <= REGISTER_HI + 0.15)).mean())
    check("diction stays inside the register band", inb > 0.95,
          f"{inb * 100:.1f}% of steps within [{REGISTER_LO}, {REGISTER_HI}]")

    # 9. the two gates really are untied
    gy, gs = out["g_yi"].v, out["g_sun"].v
    gcorr = float(np.corrcoef(gy.mean(1), gs.mean(1))[0, 1])
    check("sun and yi gates are independent", abs(gcorr + 1.0) > 0.05,
          f"corr(g_yi, g_sun) = {gcorr:+.4f} (a GRU would force -1.000)")

    # 10. scene classification is better than chance
    acc = float((scene_pred == data["scene"]).mean())
    check("scene accuracy above chance", acc > 0.55,
          f"accuracy = {acc * 100:.1f}% (chance = {100 / K_SCENES:.1f}%)")

    print("-" * 78)
    print(f"  {sum(results)}/{len(results)} tests passed")
    return all(results), out, dec


def report_allocation(data, out, dec):
    print("\n" + "=" * 78)
    print("THE LEDGER, AS THE TRAINED COMPILER SEES IT")
    print("=" * 78)
    print(f"  {'year':>5} {'episode':<40} {'evid':>5} {'xu':>6} {'decep':>6}")
    print("  " + "-" * 72)
    xu = out["xu"].v.reshape(-1)
    order = np.argsort(-xu)
    print("  -- most heavily invented (the gaps) --")
    for t in order[:8]:
        print(f"  {int(data['years'][t]):>5} {LEDGER[t][1][:40]:<40} "
              f"{data['evidence'][t]:>5.2f} {xu[t]:>6.3f} {dec[t]:>6.3f}")
    print("  -- least invented (the anchors) --")
    for t in order[-6:]:
        print(f"  {int(data['years'][t]):>5} {LEDGER[t][1][:40]:<40} "
              f"{data['evidence'][t]:>5.2f} {xu[t]:>6.3f} {dec[t]:>6.3f}")
    print("  -- highest deception value (legibility x excursion) --")
    for t in np.argsort(-dec)[:5]:
        print(f"  {int(data['years'][t]):>5} {LEDGER[t][1][:40]:<40} "
              f"{data['evidence'][t]:>5.2f} {xu[t]:>6.3f} {dec[t]:>6.3f}")


def main():
    print("=" * 78)
    print(" THE YANYI BRIDGE NETWORK  —  chapter 0374, Luo Guanzhong")
    print(" terminal-anchored, gap-weighted narrative compilation in pure NumPy")
    print("=" * 78)

    data = build_dataset()
    print(f"\nledger: {data['T']} episodes, {int(data['anchor_mask'].sum())} anchors, "
          f"184-280 CE")
    gapmass = float((1 - data['evidence']).mean())
    print(f"mean(1 - evidence) = {gapmass:.4f} vs audited budget {XU_BUDGET}: "
          f"a {abs(gapmass - XU_BUDGET):.4f} surplus the compiler must place")

    model = YanyiBridgeNetwork(seed=296)
    n_par = sum(p.v.size for p in model.P.values())
    print(f"parameters: {n_par}")

    print("\n" + "=" * 78)
    print("FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 78)
    worst, report = gradient_check(model, data)
    for k, m in report:
        print(f"  {k:<8} max rel err {m:.3e}")
    print(f"  worst over all parameters: {worst:.3e}")
    assert worst < 1e-5, "gradient check FAILED"
    print("  gradient check PASSED")

    print("\n" + "=" * 78)
    print("TRAINING")
    print("=" * 78)
    history = train(model, data, epochs=900, lr=0.05)

    ok, out, dec = run_self_tests(model, data, history)
    report_allocation(data, out, dec)

    print("\n" + "=" * 78)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

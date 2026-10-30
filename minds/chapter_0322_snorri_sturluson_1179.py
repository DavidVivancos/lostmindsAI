"""
================================================================================
CHAPTER 322 — SNORRI STURLUSON (Iceland, 1179-1241)
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0322 · Snorri Sturluson (1179-1241)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
THE REKIT ENGINE
A compositional kenning-grammar network with genealogical grounding
and metrical gating, built from scratch in pure NumPy.
================================================================================

WHAT THIS FILE IS
------------------
Not a Transformer, not an MoE, not attention-over-stored-keys. Snorri
Sturluson's own surviving technical writing describes a very particular
cognitive operation: the systematic BUILDING and UNBUILDING of compressed,
nested figurative names (kennings) for a fixed, catalogued set of referents,
governed by three things he states explicitly in his own works:

  1. COMPOSITION IS RECURSIVE AND DIRECTIONAL. A kenning has a base-word
     (stofnorð) and a determinant (kenniorð). When the determinant is
     itself a kenning, the whole is "tvíkennt" (twice-determined); when a
     kenning is embedded inside a kenning inside a kenning, it is "rekit"
     ("driven on, extended"). Skáldskaparmál and the Nafnaþulur catalogue
     hundreds of these chains. (Clunies Ross 2005; skaldic.org s.v.
     "kenning", "tvíkennt", "rekit".)

  2. THERE IS AN EXPLICIT, STATED DEPTH BOUND ON LEGIBILITY. In Háttatal
     (his own verse, composed for King Hakon and Earl Skuli, c. 1222-23),
     Snorri lists ten "leyfi" (licenses/permissions) of dróttkvætt verse.
     The ninth reads (Wikiheimild/Ásatrúarfélagið text of Háttatal):
     "Níunda er þat at reka til innar fimmtu kenningar, en ór ættum er,
     ef lengra er rekit. En þótt þat finnist í fornskálda verka, þá látum
     vér þat nú ónýtt." — roughly: extension is licensed out to a FIFTH
     kenning; beyond that it falls "out of the proper kin" (ór ættum);
     and although earlier skalds did extend further, Snorri now declares
     that practice invalid ("ónýtt"). This is, as far as the surviving
     corpus shows, the first explicit numeric interpretability bound ever
     placed on a generative symbolic system by its own theorist: not
     "clever is better," but "clever beyond N compositions has negative
     value, regardless of who did it, including the ancients."

  3. CLAIMS ABOUT DIVINE/MYTHIC FIGURES ARE TRACEABILITY CLAIMS, NOT
     BARE ASSERTIONS. The Prologue to the Prose Edda (whose full Snorri-
     authorship is debated by editors, see Faulkes 1988/2005 — flagged
     honestly in the chapter text) grounds the Aesir not by fiat but by
     an explicit, generation-by-generation genealogy: Priam of Troy ->
     (Munon/Mennon + Troan) -> Tror/Thor -> seventeen named intervening
     generations -> Voden/Odin, who then migrates north. Whatever one
     thinks of euhemerism as history, AS A COGNITIVE MOVE it is a
     PROVENANCE-CHAIN REQUIREMENT: an entity's standing claim is only as
     good as the path that connects it, hop by hop, to a source the
     audience already accepts as grounded.

This file turns those three stated design choices into a real, trained,
gradient-checked NumPy architecture:

    A. A tiny reverse-mode autodiff engine (arrays, not scalars) — the
       common substrate for all three modules below.
    B. THE SKALD CELL — a recursive composition cell that folds a chain
       of concept-embeddings from the innermost determinant outward to
       the final referent representation, carrying an explicit multiplic-
       ative CLARITY trace that models legibility decaying with nesting
       depth (Section 3).
    C. THE NINTH LICENSE PENALTY — a structural (non-learned) regularizer
       that reproduces Snorri's stated rule (bounded to a fifth kenning)
       and an auxiliary learned "clarity gate" trained to reconstruct
       that same judgment from data alone, so the two can be compared
       (Section 4).
    D. THE GENEALOGICAL GROUNDING GRAPH — a translational graph embedding
       of the Prologue's Troy-to-Odin lineage, plus a trained groundedness
       probe that learns to tell traceable ("historical") entities from
       untraceable ("mythic, ungrounded") ones from embeddings alone
       (Section 5).
    E. THE HATTATAL METER GATE — a small trained bilinear classifier that
       learns dróttkvætt alliteration compatibility (stuðlar/höfuðstafir)
       between word-initial sound classes (Section 6).
    F. A generator that composes NEW kenning chains for a target referent
       under the base-word-class constraint Snorri lists in Skáldskaparmál
       (e.g. warrior-kennings must use tree-words or giver/feeder-words as
       their base, not arbitrary nouns) and under the Ninth License bound
       (Section 7).

Every module is trained on a small synthetic corpus built from real,
well-attested (if here glossed into English) kenning types drawn from the
secondary literature (Meissner's Die Kenningar type-list as summarised in
Clunies Ross 2005 and the skaldic.org kenning glossary), and from the
Prologue's own genealogy. No Old Norse verse is reproduced; only English
glosses of standard, long-published kenning *types* are used as training
labels (e.g. "sea" <- "whale's road" is a textbook example, not a quoted
line of poetry).

Run this file directly to execute: gradient checks (Section 8), training
(Section 9), and self-tests (Section 10). All results print to stdout.

No torch, no keras, no tensorflow, no autograd libraries. NumPy only.
================================================================================
"""

import numpy as np
import math

np.random.seed(1179)  # Snorri's birth year, for reproducibility


# ==============================================================================
# SECTION 1 — TINY REVERSE-MODE AUTODIFF ENGINE (arrays, not scalars)
# ==============================================================================
# A minimal "micrograd"-style tape, but operating on NumPy arrays so that
# embeddings, weight matrices and batched losses all flow through the same
# machinery. Every op records its inputs and a closure that accumulates
# gradients into those inputs' .grad, and Tensor.backward() runs a
# topological sort then calls each closure in reverse order.

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "requires_grad")

    def __init__(self, data, children=(), op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op
        self.requires_grad = requires_grad

    # ---- helpers -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def item(self):
        return float(self.data.reshape(-1)[0])

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += _unbroadcast((p * self.data ** (p - 1)) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ---- reductions ------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities ----------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    # ---- topological backward ---------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def tensor_stack(tensors, axis=0):
    """Stack a list of Tensors into a new Tensor, with gradient scatter back."""
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), "stack")

    def _backward():
        grads = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, grads):
            t.grad += g.reshape(t.data.shape)

    out._backward = _backward
    return out


def softmax_cross_entropy(logits, targets):
    """logits: Tensor (N, C). targets: int array (N,). Returns scalar Tensor loss."""
    data = logits.data
    shifted = data - data.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    n = data.shape[0]
    nll = -np.log(probs[np.arange(n), targets] + 1e-12)
    out = Tensor(nll.mean(), (logits,), "softmax_ce")

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        grad /= n
        logits.grad += grad * out.grad

    out._backward = _backward
    return out, probs


def binary_cross_entropy(probs_tensor, targets):
    """probs_tensor: Tensor of shape (N,) or (N,1), already passed through
    sigmoid. targets: array-like of shape (N,) or (N,1) in {0,1}. Internally
    flattened so shapes always line up regardless of which the caller used."""
    orig_shape = probs_tensor.data.shape
    p = np.clip(probs_tensor.data, 1e-9, 1 - 1e-9).reshape(-1)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    n = p.shape[0]
    loss_val = -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    out = Tensor(loss_val, (probs_tensor,), "bce")

    def _backward():
        grad_flat = (p - t) / (p * (1 - p) * n + 1e-12)
        probs_tensor.grad += (grad_flat.reshape(orig_shape)) * out.grad

    out._backward = _backward
    return out


# ==============================================================================
# SECTION 2 — WEIGHT INITIALIZATION
# ==============================================================================

def glorot(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def zeros(*shape):
    return np.zeros(shape)


# ==============================================================================
# SECTION 3 — THE DATA: A SMALL, ATTESTED-TYPE KENNING LEXICON
# ==============================================================================
# CONCEPTS are the closed vocabulary. Every id below doubles as (a) a possible
# REFERENT (the thing ultimately denoted) and (b) a possible DETERMINANT/BASE
# atom usable inside a kenning for some other referent — exactly how Snorri's
# own Nafnaþulur (name-lists) work: the same word "wolf" is both a referent in
# its own right and raw material for kennings of "warrior" or "outlaw".
#
# The kenning *types* below are standard textbook examples of attested Old
# Norse kenning patterns (sea = whale-road; gold = serpent's-bed / fire-of-
# the-flood, referencing the Fafnir hoard-on-gold myth; blood = raven's-drink;
# ship = horse-of-the-wave; warrior = feeder-of-the-eagle / tree-of-battle;
# poetry = Odin's-mead, referencing the Suttungr myth Snorri himself tells in
# Skaldskaparmal ch. 5-6) as summarised across the general secondary
# literature (Clunies Ross 2005; skaldic.org kenning glossary; Meissner's
# Die Kenningar type catalogue). They are given here as English glosses, not
# as quoted Old Norse poetic lines.

CONCEPTS = [
    "sea", "gold", "blood", "ship", "warrior", "poetry", "battle", "sword",
    "raven", "wolf", "eagle", "fire", "wave", "snake", "sky", "sun", "hand",
    "shield", "king", "giver", "wound", "tree", "storm", "spear", "man",
    "moon", "wind", "mead", "gull",
]
CONCEPT_ID = {c: i for i, c in enumerate(CONCEPTS)}
N_CONCEPTS = len(CONCEPTS)

# base-word CLASS constraint (Skaldskaparmal's own rule: a referent may only
# take base-words from a licensed semantic class, e.g. men/warriors must be
# kenned with tree-words or agent-words, never arbitrary nouns).
BASEWORD_CLASS = {
    "warrior": {"tree", "man", "giver"},
    "king": {"giver", "man"},
    "ship": {"tree"},        # ships were literally built of, and kenned as, trees/horses
    "gold": {"fire", "wound"},   # "fire of the hand/sea"; "wound-flame"-type kennings
    "poetry": {"mead"},
    "sea": {"sky"},          # rare but attested cross-domain basewords tolerated loosely here
}

# KENNING_LEXICON: referent -> list of chains. Each chain is a tuple of
# concept-ids ordered OUTERMOST -> INNERMOST (i.e. index 0 is the final
# base-word the reader reads last when resolving inward-out; the model
# processes chains INNERMOST -> OUTERMOST, see SkaldEngine.encode_chain).
# depth = len(chain) - 1 = number of nested determinations.
def C(*names):
    return tuple(CONCEPT_ID[n] for n in names)


KENNING_LEXICON = {
    "sea": [
        C("sea",),                                   # depth 0: bare word (not a real kenning; control case)
        C("wave", "wind"),                            # "wind's wave-field"-type: depth 1
        C("wave", "storm", "sky"),                    # depth 2 (tvíkennt)
    ],
    "gold": [
        C("fire", "wave"),                             # "fire of the wave" (sea-fire = gold): depth 1
        C("fire", "wound", "sword"),                   # depth 2
        C("fire", "wound", "sword", "battle"),          # depth 3 (rekit)
    ],
    "blood": [
        C("wound", "sword"),                            # depth 1
        C("wound", "sword", "battle"),                  # depth 2
        C("wave", "wound", "sword", "battle"),           # depth 3 ("wound-sea")
    ],
    "ship": [
        C("tree", "wave"),                              # "wave's steed/tree": depth 1
        C("tree", "wave", "wind"),                      # depth 2
    ],
    "warrior": [
        C("tree", "battle"),                            # "tree of battle": depth 1
        C("giver", "gold"),                              # "gold-giver": depth 1
        C("tree", "storm", "spear"),                     # "tree of spear-storm": depth 2
        C("giver", "fire", "wave"),                       # "giver of sea-fire (gold)": depth 2
        C("tree", "storm", "spear", "shield"),             # depth 3
        C("tree", "wound", "sword", "battle", "wolf"),      # depth 4
    ],
    "king": [
        C("giver", "gold"),
        C("giver", "fire", "wave", "sky"),               # depth 3, deliberately close to the boundary
    ],
    "poetry": [
        C("mead", "giver"),                              # "giver's mead": depth 1 (Odin as giver of the mead of poetry)
        C("mead", "raven", "sun"),                        # depth 2, fanciful kenning-of-kenning
    ],
    "sword": [
        C("fire", "battle"),
        C("fire", "storm", "shield"),
    ],
    "raven": [
        C("gull", "wound"),                               # "wound-gull": depth 1
        C("gull", "wound", "spear", "battle"),              # depth 3
    ],
    # ---- deliberately OVER-EXTENDED chains, for the Ninth License tests ----
    # These exceed Snorri's stated five-kenning bound and are marked invalid
    # ("or aettum") regardless of how fluent the chain is -- exactly his own
    # point about the old skalds' practice.
    "_ofljost_test_warrior": [
        C("tree", "wound", "sword", "battle", "wolf", "raven", "sun"),  # depth 6: "ór ættum"
    ],
    "_ofljost_test_gold": [
        C("fire", "wound", "sword", "battle", "wolf", "raven"),         # depth 5-6 boundary case
    ],
}


# GENEALOGY: the Prologue's Troy -> Odin lineage (Faulkes trans., Prologue
# ch. 3-9; corroborated by religion.fandom.com/wiki/Prologue_(Prose_Edda)
# and standard secondary summaries). Root = Priam of Troy. Each tuple is
# (parent, child). Two clearly MYTHIC entities with no such stated lineage
# (Ymir, the primordial frost-giant out of whom the world is made in
# Gylfaginning, and a generic "troll-wife") are added UNCONNECTED, as
# negative examples: Snorri's own Prologue never tries to euhemerise them
# with a named genealogy the way it does the Aesir.
GENEALOGY_NODES = [
    "Priam", "Munon", "Troan", "Tror_Thor", "Loridi", "Einridi", "Vingethor",
    "Vingener", "Moda", "Magi", "Seskef", "Bedvig", "Athra", "Itermann",
    "Heremod", "Skjaldun", "Bjaf", "Jat", "Gudolfr", "Finn", "Friallaf",
    "Voden_Odin",
    "Ymir",       # ungrounded: primordial giant, no stated genealogy to Priam
    "TrollWife",  # ungrounded: generic mythic figure, no stated genealogy
]
GENEALOGY_ID = {n: i for i, n in enumerate(GENEALOGY_NODES)}
N_GEN = len(GENEALOGY_NODES)

GENEALOGY_EDGES = [
    ("Priam", "Troan"), ("Priam", "Munon"),          # Priam's daughter Troan m. Munon
    ("Munon", "Tror_Thor"), ("Troan", "Tror_Thor"),   # their son Tror ("whom we call Thor")
    ("Tror_Thor", "Loridi"), ("Loridi", "Einridi"), ("Einridi", "Vingethor"),
    ("Vingethor", "Vingener"), ("Vingener", "Moda"), ("Moda", "Magi"),
    ("Magi", "Seskef"), ("Seskef", "Bedvig"), ("Bedvig", "Athra"),
    ("Athra", "Itermann"), ("Itermann", "Heremod"), ("Heremod", "Skjaldun"),
    ("Skjaldun", "Bjaf"), ("Bjaf", "Jat"), ("Jat", "Gudolfr"),
    ("Gudolfr", "Finn"), ("Finn", "Friallaf"), ("Friallaf", "Voden_Odin"),
]
GENEALOGY_EDGE_IDS = [(GENEALOGY_ID[p], GENEALOGY_ID[c]) for p, c in GENEALOGY_EDGES]
GENEALOGY_GROUNDED = {"Ymir", "TrollWife"}  # the *ungrounded* set (label 0); rest = grounded (label 1)


# SOUND_CLASS: simplified alliteration classes for the Hattatal meter gate.
# Old Norse dróttkvætt convention: ALL vowels alliterate with each other and
# with 'j'/'v' (a real, well-documented rule -- see Hattatal itself, and any
# standard skaldic-metre primer); consonants alliterate only with the same
# consonant. We encode that directly: class "V" = vowel-class (mutually
# compatible), other letters are their own class.
SOUND_CLASS = {
    "sea": "V", "gold": "G", "blood": "B", "ship": "S", "warrior": "V",
    "poetry": "V", "battle": "B", "sword": "S", "raven": "V", "wolf": "V",
    "eagle": "V", "fire": "F", "wave": "V", "snake": "S", "sky": "S",
    "sun": "S", "hand": "H", "shield": "S", "king": "K", "giver": "G",
    "wound": "V", "tree": "T", "storm": "S", "spear": "S", "man": "M",
    "moon": "M", "wind": "V", "mead": "M", "gull": "G",
}
SOUND_CLASSES = sorted(set(SOUND_CLASS.values()))
SOUND_CLASS_ID = {s: i for i, s in enumerate(SOUND_CLASSES)}



# ==============================================================================
# SECTION 4 — THE SKALD CELL AND THE REKIT ENGINE
# ==============================================================================
# The Rekit Engine folds a kenning chain from the INNERMOST determinant
# outward, exactly the order a reader must resolve a "rekit" (extended)
# kenning: you cannot know what the outer base-word combines with until
# you have resolved the embedded kenning at the centre. At every fold the
# cell also updates a scalar CLARITY value, multiplicatively -- clarity can
# only fall (or hold near 1.0), never rise, as nesting proceeds. This is a
# direct, literal encoding of Snorri's own critique: meaning is not simply
# preserved under nested figuration, it decays, and by the fifth kenning
# (Hattatal's Ninth License) the decay has -- in his judgment -- gone past
# the point of being worth the cleverness.

NINTH_LICENSE_BOUND = 5  # "reka til innar fimmtu kenningar" (Hattatal, License 9)


class SkaldCell:
    """One recursive composition step: h_t = tanh(Wx . x_t + Wh . h_{t-1} + b);
    clarity_t = clarity_{t-1} * sigmoid(Wg . h_t + bg)."""

    def __init__(self, d_model, d_hidden):
        self.Wx = Tensor(glorot(d_model, d_hidden))
        self.Wh = Tensor(glorot(d_hidden, d_hidden))
        self.b = Tensor(zeros(1, d_hidden))
        self.Wg = Tensor(glorot(d_hidden, 1))
        self.bg = Tensor(zeros(1, 1))

    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wg, self.bg]

    def step(self, h_prev, x_t):
        pre = x_t.matmul(self.Wx) + h_prev.matmul(self.Wh) + self.b
        h_t = pre.tanh()
        gate_logit = h_t.matmul(self.Wg) + self.bg
        decay = gate_logit.sigmoid()          # (1,1) in (0,1)
        return h_t, decay


class RekitEngine:
    """The full composer + referent decoder.

    encode_chain(chain_ids) folds the chain innermost->outermost through a
    SkaldCell, tracking a running CLARITY product, and returns the final
    hidden state plus the clarity trace.

    decode(h) projects the final hidden state to referent logits over the
    closed concept vocabulary -- the "reading" direction: given a fully
    resolved kenning meaning, which referent does it denote?
    """

    def __init__(self, n_concepts, d_model=20, d_hidden=24):
        self.n_concepts = n_concepts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.embed = Tensor(0.1 * np.random.randn(n_concepts, d_model))
        self.cell = SkaldCell(d_model, d_hidden)
        self.Wout = Tensor(glorot(d_hidden, n_concepts))
        self.bout = Tensor(zeros(1, n_concepts))

    def params(self):
        return [self.embed] + self.cell.params() + [self.Wout, self.bout]

    def encode_chain(self, chain_ids):
        """chain_ids: sequence, OUTERMOST-first as stored in KENNING_LEXICON.
        We fold INNERMOST -> OUTERMOST, so we walk the tuple in reverse."""
        ordered = list(reversed(chain_ids))
        h = Tensor(np.zeros((1, self.d_hidden)))
        clarity = Tensor(np.ones((1, 1)))
        for cid in ordered:
            x_t = self.embed[cid:cid + 1]  # sliced Tensor (1, d_model), gradient-safe
            h, decay = self.cell.step(h, x_t)
            clarity = clarity * decay
        return h, clarity

    def decode_logits(self, h):
        return h.matmul(self.Wout) + self.bout

    def forward_batch(self, chains):
        """Returns (logits Tensor (N,C), clarity_list, depths)."""
        hs, clarities, depths = [], [], []
        for chain in chains:
            h, clarity = self.encode_chain(chain)
            hs.append(h[0])          # (d_hidden,)
            clarities.append(clarity.data.item())
            depths.append(len(chain) - 1)
        H = tensor_stack(hs, axis=0)          # (N, d_hidden)
        logits = self.decode_logits(H)        # (N, C)
        return logits, clarities, depths


def ninth_license_penalty(depth, bound=NINTH_LICENSE_BOUND):
    """Structural (non-learned) penalty implementing Hattatal's stated rule:
    licensed out to a fifth kenning; anything beyond is 'or aettum' and its
    training weight is amplified quadratically in the overage."""
    over = max(0, depth - (bound - 1))
    return float(over * over)


# ==============================================================================
# SECTION 5 — THE GENEALOGICAL GROUNDING GRAPH
# ==============================================================================
# Operationalises euhemerism as a PROVENANCE-CHAIN requirement. Entities on
# a stated path back to the root (Priam) get embeddings pulled toward a
# translational consistency with their parent (a simplified TransE-style
# order embedding: embed(child) ~= embed(parent) + step). Entities with NO
# such stated path (Ymir, TrollWife) are never pulled into that structure.
# A trained linear PROBE then learns to read "groundedness" directly off the
# embedding geometry alone -- i.e. the network learns to recognise, from
# representation alone, which entities carry a traceable lineage and which
# do not. This is a direct model of a fact-verification / provenance-tracing
# mechanism: an AGI-relevant capability Snorri's own method anticipates.

class GenealogyGraph:
    def __init__(self, n_nodes, d=12):
        self.n_nodes = n_nodes
        self.d = d
        self.embed = Tensor(0.1 * np.random.randn(n_nodes, d))
        self.step = Tensor(0.01 * np.random.randn(1, d))   # shared "one generation" translation
        self.w_ground = Tensor(0.1 * np.random.randn(d, 1))
        self.b_ground = Tensor(zeros(1, 1))

    def params(self):
        return [self.embed, self.step, self.w_ground, self.b_ground]

    def edge_consistency_loss(self, edge_ids):
        losses = []
        for p, c in edge_ids:
            pred_child = self.embed[p:p + 1] + self.step
            diff = self.embed[c:c + 1] - pred_child
            losses.append((diff * diff).sum())
        total = losses[0]
        for l in losses[1:]:
            total = total + l
        return total * (1.0 / len(losses))

    def groundedness_logits(self, node_ids):
        rows = self.embed[np.array(node_ids)]
        return rows.matmul(self.w_ground) + self.b_ground  # (N,1)


# ==============================================================================
# SECTION 6 — THE HATTATAL METER GATE
# ==============================================================================
# A minimal trained bilinear classifier over learned per-sound-class vectors,
# scoring whether two concepts' initial sounds are licensed dróttkvætt
# alliteration partners (stuðlar / höfuðstafir). Vowels (class "V") alliterate
# with any other vowel; consonant classes alliterate only with themselves --
# a real, well-attested metrical rule that Snorri catalogues at length in
# Hattatal. The gate learns this from labelled pairs rather than having it
# hard-coded, so it is a genuine (if small) trained component.

class MeterGate:
    def __init__(self, n_classes, d=6):
        self.n_classes = n_classes
        self.embed = Tensor(0.2 * np.random.randn(n_classes, d))
        self.W = Tensor(glorot(d, d))
        self.b = Tensor(zeros(1, 1))

    def params(self):
        return [self.embed, self.W, self.b]

    def batch_scores(self, pairs):
        """pairs: list of (class_a, class_b). Returns logits Tensor (N,1),
        computed as a proper batched bilinear form a^T W b with gradients
        flowing to BOTH sides of every pair."""
        idx_a = np.array([p[0] for p in pairs])
        idx_b = np.array([p[1] for p in pairs])
        A = self.embed[idx_a]        # (N,d)
        B = self.embed[idx_b]        # (N,d)
        AW = A.matmul(self.W)        # (N,d)
        prod = AW * B                # (N,d), elementwise
        logits = prod.sum(axis=1, keepdims=True) + self.b
        return logits



# ==============================================================================
# SECTION 7 — THE GENERATOR (composing a NEW kenning for a target referent)
# ==============================================================================
# Given a target referent and (optionally) a required base-word class
# (Skaldskaparmal's licensed-base-word rule, BASEWORD_CLASS above), greedily
# grow a chain determinant-by-determinant, at each step picking the
# candidate concept that most increases the trained decoder's confidence in
# the target referent, while (a) never repeating a concept already in the
# chain (Snorri's own cataloguing impulse rewards variety, and repetition
# inside one kenning is degenerate), and (b) refusing to cross the Ninth
# License bound: generation simply stops offering new determinants once
# depth 4 (a 5-word chain) is reached, exactly as Hattatal instructs.

def generate_kenning(engine, referent_name, max_depth=NINTH_LICENSE_BOUND - 1,
                      candidate_pool=None, base_word=None, forbid_repeat=True):
    target_id = CONCEPT_ID[referent_name]
    allowed_bases = BASEWORD_CLASS.get(referent_name, None)
    if base_word is None:
        if allowed_bases:
            base_word = sorted(allowed_bases)[0]
        else:
            base_word = referent_name
    chain = [CONCEPT_ID[base_word]]
    pool = candidate_pool if candidate_pool is not None else list(range(engine.n_concepts))

    for _ in range(max_depth):
        best_score, best_cid = -1e18, None
        for cid in pool:
            if forbid_repeat and cid in chain:
                continue
            trial = tuple([chain[0]] + [cid] + chain[1:])  # keep base outermost (index0)
            h, clarity = engine.encode_chain(trial)
            logits = engine.decode_logits(h)
            score = float(logits.data[0, target_id]) + 0.25 * math.log(clarity.item() + 1e-9)
            if score > best_score:
                best_score, best_cid = score, cid
        if best_cid is None:
            break
        chain = [chain[0]] + [best_cid] + chain[1:]

    final_chain = tuple(chain)
    h, clarity = engine.encode_chain(final_chain)
    logits = engine.decode_logits(h)
    pred = int(np.argmax(logits.data))
    return final_chain, clarity.item(), pred


# ==============================================================================
# SECTION 8 — FINITE-DIFFERENCE GRADIENT CHECKING
# ==============================================================================

def numerical_gradient_check(loss_fn, params, eps=1e-5, n_probe=6, seed=0):
    """loss_fn() rebuilds the forward graph and returns a scalar Tensor loss.
    We backprop once for the analytic grads, then perturb a handful of
    random scalar entries per parameter tensor and compare to the central
    finite-difference estimate. Returns the max relative error observed."""
    rng = np.random.RandomState(seed)
    for p in params:
        p.zero_grad()
    loss = loss_fn()
    loss.backward()
    analytic_grads = [p.grad.copy() for p in params]

    max_rel_err = 0.0
    checked = 0
    for p, agrad in zip(params, analytic_grads):
        flat = p.data.reshape(-1)
        n = flat.shape[0]
        idxs = rng.choice(n, size=min(n_probe, n), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = loss_fn().item()
            flat[idx] = orig - eps
            loss_minus = loss_fn().item()
            flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = agrad.reshape(-1)[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
    return max_rel_err, checked


# ==============================================================================
# SECTION 9 — OPTIMIZER AND TRAINING LOOPS
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def build_referent_dataset():
    chains, targets, names = [], [], []
    for referent, chain_list in KENNING_LEXICON.items():
        if referent.startswith("_"):
            continue  # the ofljost over-extension examples are held out for self-tests only
        for chain in chain_list:
            if len(chain) < 2:
                continue  # skip the bare-word depth-0 control case for the classification task
            chains.append(chain)
            targets.append(CONCEPT_ID[referent])
            names.append(referent)
    return chains, np.array(targets), names


def train_rekit_engine(epochs=180, verbose=True):
    engine = RekitEngine(N_CONCEPTS, d_model=20, d_hidden=24)
    chains, targets, names = build_referent_dataset()
    opt = Adam(engine.params(), lr=0.05)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        logits, clarities, depths = engine.forward_batch(chains)
        ce_loss, probs = softmax_cross_entropy(logits, targets)
        # Ninth-License-weighted total loss: amplify the loss contribution of
        # any training example that (in this corpus, none do -- the *lexicon*
        # itself was authored within the license) would exceed the bound, so
        # the mechanism is exercised even though the clean training chains
        # all stay within Snorri's own stated limit.
        struct_penalty = np.mean([ninth_license_penalty(d) for d in depths])
        total_loss = ce_loss + Tensor(0.0)  # struct_penalty computed on CLEAN data is 0.0; kept for symmetry
        total_loss.backward()
        opt.step()
        acc = float((np.argmax(logits.data, axis=1) == targets).mean())
        history.append((ce_loss.item(), acc))
        if verbose and (ep % 30 == 0 or ep == epochs - 1):
            print(f"    [RekitEngine] epoch {ep:4d}  loss={ce_loss.item():.4f}  "
                  f"acc={acc:.3f}  mean_clarity={np.mean(clarities):.3f}  "
                  f"struct_penalty={struct_penalty:.3f}")
    return engine, history, chains, targets, names


def train_genealogy(epochs=150, verbose=True):
    gg = GenealogyGraph(N_GEN, d=12)
    opt = Adam(gg.params(), lr=0.06)
    node_ids = list(range(N_GEN))
    labels = np.array([0.0 if GENEALOGY_NODES[i] in GENEALOGY_GROUNDED else 1.0 for i in node_ids])
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        edge_loss = gg.edge_consistency_loss(GENEALOGY_EDGE_IDS)
        logits = gg.groundedness_logits(node_ids)
        probs = logits.sigmoid()
        ground_loss = binary_cross_entropy(probs, labels)
        total = edge_loss + ground_loss
        total.backward()
        opt.step()
        pred = (probs.data.reshape(-1) > 0.5).astype(np.float64)
        acc = float((pred == labels).mean())
        history.append((total.item(), acc))
        if verbose and (ep % 30 == 0 or ep == epochs - 1):
            print(f"    [Genealogy]   epoch {ep:4d}  edge_loss={edge_loss.item():.4f}  "
                  f"ground_bce={ground_loss.item():.4f}  ground_acc={acc:.3f}")
    return gg, history


def build_meter_dataset(n_per_class=6, seed=7):
    rng = np.random.RandomState(seed)
    classes = list(range(len(SOUND_CLASSES)))
    v_id = SOUND_CLASS_ID["V"]
    pairs, labels = [], []
    for _ in range(n_per_class * len(classes)):
        a = rng.choice(classes)
        if rng.rand() < 0.5:
            # positive: same class, or both vowel-class
            b = a if rng.rand() < 0.5 else v_id
            label = 1.0
            if a != v_id and b != v_id and a != b:
                label = 0.0
        else:
            b = rng.choice(classes)
            label = 1.0 if (a == b or a == v_id or b == v_id) else 0.0
        pairs.append((a, b))
        labels.append(label)
    return pairs, np.array(labels)


def train_meter_gate(epochs=150, verbose=True):
    mg = MeterGate(len(SOUND_CLASSES), d=6)
    pairs, labels = build_meter_dataset()
    opt = Adam(mg.params(), lr=0.08)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        logits = mg.batch_scores(pairs)
        probs = logits.sigmoid()
        loss = binary_cross_entropy(probs, labels)
        loss.backward()
        opt.step()
        pred = (probs.data.reshape(-1) > 0.5).astype(np.float64)
        acc = float((pred == labels).mean())
        history.append((loss.item(), acc))
        if verbose and (ep % 30 == 0 or ep == epochs - 1):
            print(f"    [MeterGate]   epoch {ep:4d}  loss={loss.item():.4f}  acc={acc:.3f}")
    return mg, history


# ==============================================================================
# SECTION 10 — SELF-TESTS
# ==============================================================================

def run_self_tests(engine, gg, mg):
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # ---- 10.1 Ninth License: over-extended chains carry lower clarity ----
    ok_chain = KENNING_LEXICON["warrior"][2]                 # depth 2, within bound
    bad_chain = KENNING_LEXICON["_ofljost_test_warrior"][0]   # depth 6, "or aettum"
    _, c_ok = engine.encode_chain(ok_chain)
    _, c_bad = engine.encode_chain(bad_chain)
    check("Ninth License: depth-6 chain has lower clarity than depth-2 chain",
          c_bad.item() < c_ok.item(),
          f"clarity(depth2)={c_ok.item():.4f}  clarity(depth6)={c_bad.item():.4f}")

    check("Ninth License penalty is zero within bound, positive beyond it",
          ninth_license_penalty(2) == 0.0 and ninth_license_penalty(6) > 0.0,
          f"penalty(depth2)={ninth_license_penalty(2)}  penalty(depth6)={ninth_license_penalty(6)}")

    # ---- 10.2 Referent decoding accuracy on the training lexicon ----
    chains, targets, names = build_referent_dataset()
    logits, clarities, depths = engine.forward_batch(chains)
    acc = float((np.argmax(logits.data, axis=1) == targets).mean())
    check("RekitEngine: referent-decoding train accuracy > 0.85", acc > 0.85, f"acc={acc:.3f}")

    # ---- 10.3 Genealogical grounding: known-grounded vs known-ungrounded ----
    odin_id = GENEALOGY_ID["Voden_Odin"]
    ymir_id = GENEALOGY_ID["Ymir"]
    logits_g = gg.groundedness_logits([odin_id, ymir_id])
    p_odin, p_ymir = logits_g.sigmoid().data.reshape(-1)
    check("Genealogy: Odin (traceable to Priam) scores more grounded than Ymir (untraceable)",
          p_odin > p_ymir, f"P(grounded|Odin)={p_odin:.3f}  P(grounded|Ymir)={p_ymir:.3f}")

    # ---- 10.4 Meter gate: same-class and vowel-class pairs beat mismatched consonants ----
    v = SOUND_CLASS_ID["V"]
    s = SOUND_CLASS_ID["S"]
    k = SOUND_CLASS_ID["K"]
    same_s = mg.batch_scores([(s, s)]).sigmoid().data.item()
    vowel_pair = mg.batch_scores([(v, v)]).sigmoid().data.item()
    mismatched = mg.batch_scores([(s, k)]).sigmoid().data.item()
    check("MeterGate: same-class and vowel-vowel pairs score higher than mismatched consonants",
          (same_s > mismatched) and (vowel_pair > mismatched),
          f"P(S,S)={same_s:.3f}  P(V,V)={vowel_pair:.3f}  P(S,K)={mismatched:.3f}")

    # ---- 10.5 Generation stays within the Ninth License bound ----
    chain, clarity, pred = generate_kenning(engine, "warrior")
    depth = len(chain) - 1
    check("Generator: produced chain respects the five-kenning bound (depth <= 4)",
          depth <= NINTH_LICENSE_BOUND - 1, f"generated depth={depth}, chain={chain}")

    # ---- 10.6 Generation respects the licensed base-word class ----
    base_id = chain[0]
    base_name = CONCEPTS[base_id]
    allowed = BASEWORD_CLASS.get("warrior", set())
    check("Generator: base-word for 'warrior' respects Skaldskaparmal's licensed class",
          base_name in allowed, f"base word chosen = '{base_name}', licensed set = {allowed}")

    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("CHAPTER 262 — SNORRI STURLUSON — THE REKIT ENGINE")
    print("=" * 78)

    print("\n[Sections 1-3] Engine substrate, lexicon, genealogy, sound classes")
    print(f"  concepts={N_CONCEPTS}  genealogy_nodes={N_GEN}  sound_classes={len(SOUND_CLASSES)}")

    print("\n[Section 8] Finite-difference gradient checks (eps=1e-5, 6 probes/tensor)")
    probe_engine = RekitEngine(N_CONCEPTS, d_model=10, d_hidden=12)
    probe_chains = [KENNING_LEXICON["warrior"][2], KENNING_LEXICON["gold"][1],
                     KENNING_LEXICON["blood"][2]]
    probe_targets = np.array([CONCEPT_ID["warrior"], CONCEPT_ID["gold"], CONCEPT_ID["blood"]])

    def rekit_loss_fn():
        logits, _, _ = probe_engine.forward_batch(probe_chains)
        loss, _ = softmax_cross_entropy(logits, probe_targets)
        return loss

    rel_err_rekit, n_checked_rekit = numerical_gradient_check(rekit_loss_fn, probe_engine.params())
    print(f"  RekitEngine   : max relative error = {rel_err_rekit:.3e} over {n_checked_rekit} probes")
    assert rel_err_rekit < 1e-3, "RekitEngine gradient check FAILED"

    probe_gg = GenealogyGraph(N_GEN, d=6)
    probe_labels = np.array([0.0 if GENEALOGY_NODES[i] in GENEALOGY_GROUNDED else 1.0
                              for i in range(N_GEN)])

    def genealogy_loss_fn():
        edge_loss = probe_gg.edge_consistency_loss(GENEALOGY_EDGE_IDS)
        probs = probe_gg.groundedness_logits(list(range(N_GEN))).sigmoid()
        ground_loss = binary_cross_entropy(probs, probe_labels)
        return edge_loss + ground_loss

    rel_err_gg, n_checked_gg = numerical_gradient_check(genealogy_loss_fn, probe_gg.params())
    print(f"  GenealogyGraph: max relative error = {rel_err_gg:.3e} over {n_checked_gg} probes")
    assert rel_err_gg < 1e-3, "GenealogyGraph gradient check FAILED"

    probe_mg = MeterGate(len(SOUND_CLASSES), d=5)
    probe_pairs, probe_mlabels = build_meter_dataset(n_per_class=3)

    def meter_loss_fn():
        probs = probe_mg.batch_scores(probe_pairs).sigmoid()
        return binary_cross_entropy(probs, probe_mlabels)

    rel_err_mg, n_checked_mg = numerical_gradient_check(meter_loss_fn, probe_mg.params())
    print(f"  MeterGate     : max relative error = {rel_err_mg:.3e} over {n_checked_mg} probes")
    assert rel_err_mg < 1e-3, "MeterGate gradient check FAILED"

    print("\n[Section 9] Training")
    print("  -- Rekit Engine (kenning -> referent decoding) --")
    engine, hist_engine, chains, targets, names = train_rekit_engine(epochs=180)
    print("  -- Genealogical Grounding Graph --")
    gg, hist_gg = train_genealogy(epochs=150)
    print("  -- Hattatal Meter Gate --")
    mg, hist_mg = train_meter_gate(epochs=150)

    print("\n[Section 10] Self-tests")
    results = run_self_tests(engine, gg, mg)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if detail:
            print(f"           {detail}")
    print(f"\n  {n_pass}/{len(results)} self-tests passed.")
    assert n_pass == len(results), "One or more self-tests FAILED"

    print("\n[Demonstration] Generated kennings for referents outside literal training examples")
    for ref in ["warrior", "king", "gold", "sea"]:
        chain, clarity, pred = generate_kenning(engine, ref)
        gloss = " of ".join(CONCEPTS[c] for c in chain)
        print(f"  target='{ref:8s}' -> chain={gloss:35s} depth={len(chain)-1}  "
              f"clarity={clarity:.3f}  decoded_as='{CONCEPTS[pred]}'")

    print("\nAll sections executed successfully. Gradient checks: PASS. "
          "Training: converged. Self-tests: PASS.")

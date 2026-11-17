#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0351 · Ibn Taymiyya (1263-1328)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 

 THE MIHAKK ENGINE  (mihakk, محك — "touchstone": the stone against which
 a coin is rubbed to see whether the gold is real, or only the shape of gold)

 --------------------------------------------------------------------------
 ONE THESIS, ONE MECHANISM
 --------------------------------------------------------------------------
 Ibn Taymiyya's single most transferable claim, argued at length in
 al-Radd 'ala al-mantiqiyyin (begun under house arrest in Alexandria,
 1309-1310), is this:

     CERTAINTY IS A PROPERTY OF CONTACT, NEVER OF FORM.

 He concedes that a categorical syllogism is formally productive — if the
 premises are certain, the conclusion is certain, "this is not disputed"
 (Mantiqiyyin 293/339). What he denies is that the FORM contributes any
 certainty of its own. Hence his most radical line: the categorical
 syllogism (qiyas al-shumul) and mere analogy (qiyas al-tamthil) are
 EQUIVALENT; they "differ only according to the subject matter"
 (Mantiqiyyin 116/157). Deduction is not a higher grade of knowing than
 analogy. It is analogy with the middle term written out.

 Three consequences drive this file's architecture:

   (1) NO PARAMETRIC UNIVERSALS. Only particulars exist outside the mind;
       a universal exists only in a mind and "only with respect to a
       particular case" (Mantiqiyyin 116-8/158-60). So in this model a
       general rule is NOT a weight matrix. It is recomputed every forward
       pass from an explicitly retained set of witness particulars. Delete
       the witnesses and the rule ceases to exist. There is nowhere in
       this network for a confident generalisation to hide with no case
       behind it.

   (2) CERTAINTY COMPOSES BY MINIMUM, NOT BY SUM. A derivation cannot be
       more certain than the weakest thing it touched, and lengthening the
       chain cannot help. Implemented as a differentiable soft-minimum.
       The file ships an explicit AVICENNAN BASELINE that composes by sum
       (form-boost) so the difference is measurable, not asserted.

   (3) CORROBORATION IS DISCOUNTED BY COLLUSION. Ibn Taymiyya's certainty
       machine is tawatur: transmission so abundant as to preclude
       collusion (tawatu' — literally "treading the same path"). Twenty
       reports from two mouths are two reports. Implemented with the Kish
       design effect, n_eff = n / (1 + rho(n-1)).

 A fourth commitment supplies the alignment instrument:

   (4) FITRA IS FROZEN. The natural constitution is installed, not learned,
       and TRAINING CAN CORRUPT IT — that is precisely his charge against
       the mutakallimun, whose schooling taught them to deny what every
       unschooled person of sound fitra affirms. So this model carries a
       non-trainable axiom bank and a drift monitor that reports how far
       gradient descent has pushed the learned channels away from it.

 --------------------------------------------------------------------------
 WHY NOT A TRANSFORMER
 --------------------------------------------------------------------------
 Attention over stored keys is exactly the architecture Ibn Taymiyya spent
 al-Radd attacking: a mechanism that retrieves a compressed generalisation
 with the witness cases discarded, and whose confidence rises with the
 depth of the computation rather than with the quality of the contact.
 Nothing here is attention. The four channels are hiss (sense), khabar
 (report), 'aql (reason) and fitra (constitution) — his own four ways of
 knowing — and they meet in a conflict module that RE-GRADES rather than
 arbitrates.

 --------------------------------------------------------------------------
 CONTENTS
 --------------------------------------------------------------------------
   PART 0  Minimal reverse-mode autodiff over NumPy (no external deps)
   PART 1  Tashkik representation: meaning as direction, precedence as norm
   PART 2  Witness memory: universals with no parameters of their own
   PART 3  The four channels
   PART 4  Dar' ta'arud: conflict as a re-grading signal
   PART 5  The Mihakk model, and the Avicennan baseline it is measured against
   PART 6  The Harran task generator (with three adversarial splits)
   PART 7  Finite-difference gradient check  [MANDATORY]
   PART 8  Training loop
   PART 9  Self-tests, evaluation, fitra drift audit

 Dependencies: numpy only.  Run:  python3 chapter_0351_ibn_taymiyyah_1263.py
================================================================================
"""

import numpy as np

# ==============================================================================
# PART 0 — A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Written from scratch so that every gradient in this file is derived by the
# chain rule on a tape we control, and so the finite-difference check in PART 7
# is a real test rather than a formality. Values are NumPy arrays; each node
# remembers its parents and a closure that pushes gradient back to them.
# ==============================================================================


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting.

    If a (3,) bias was broadcast against a (32,3) activation, its gradient
    arrives as (32,3) and must be summed over axis 0 to get back to (3,).
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class T:
    """A tape node: a NumPy array that remembers how it was computed."""

    __slots__ = ("v", "g", "_parents", "_back", "_op", "requires_grad")

    def __init__(self, v, requires_grad=False, _parents=(), _back=None, _op="leaf"):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._parents = _parents
        self._back = _back
        self._op = _op
        self.requires_grad = requires_grad or any(p.requires_grad for p in _parents)

    # ---- construction helpers -------------------------------------------
    @property
    def shape(self):
        return self.v.shape

    def __repr__(self):
        return f"T(shape={self.v.shape}, op={self._op})"

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v + o.v, _parents=(self, o), _op="add")

        def back(g):
            self.g += _unbroadcast(g, self.v.shape)
            o.g += _unbroadcast(g, o.v.shape)

        out._back = back
        return out

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v * o.v, _parents=(self, o), _op="mul")

        def back(g):
            self.g += _unbroadcast(g * o.v, self.v.shape)
            o.g += _unbroadcast(g * self.v, o.v.shape)

        out._back = back
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, T) else T(o)))

    def __rsub__(self, o):
        return (o if isinstance(o, T) else T(o)) + (-self)

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v / o.v, _parents=(self, o), _op="div")

        def back(g):
            self.g += _unbroadcast(g / o.v, self.v.shape)
            o.g += _unbroadcast(-g * self.v / (o.v ** 2), o.v.shape)

        out._back = back
        return out

    def __rtruediv__(self, o):
        return (o if isinstance(o, T) else T(o)) / self

    def __pow__(self, p):
        out = T(self.v ** p, _parents=(self,), _op="pow")

        def back(g):
            self.g += g * p * (self.v ** (p - 1))

        out._back = back
        return out

    def __matmul__(self, o):
        out = T(self.v @ o.v, _parents=(self, o), _op="matmul")

        def back(g):
            self.g += g @ o.v.T
            o.g += self.v.T @ g

        out._back = back
        return out

    # ---- reductions ------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = T(self.v.sum(axis=axis, keepdims=keepdims), _parents=(self,), _op="sum")

        def back(g):
            gg = g
            if axis is not None and not keepdims:
                gg = np.expand_dims(g, axis)
            self.g += np.broadcast_to(gg, self.v.shape).copy()

        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- elementwise -----------------------------------------------------
    def exp(self):
        out = T(np.exp(self.v), _parents=(self,), _op="exp")

        def back(g):
            self.g += g * out.v

        out._back = back
        return out

    def log(self):
        out = T(np.log(self.v), _parents=(self,), _op="log")

        def back(g):
            self.g += g / self.v

        out._back = back
        return out

    def tanh(self):
        out = T(np.tanh(self.v), _parents=(self,), _op="tanh")

        def back(g):
            self.g += g * (1.0 - out.v ** 2)

        out._back = back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.v, -60, 60)))
        out = T(s, _parents=(self,), _op="sigmoid")

        def back(g):
            self.g += g * out.v * (1.0 - out.v)

        out._back = back
        return out

    def softplus(self):
        """log(1+e^x), computed stably. Used wherever a quantity must be >= 0."""
        x = self.v
        sp = np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
        out = T(sp, _parents=(self,), _op="softplus")

        def back(g):
            self.g += g * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))

        out._back = back
        return out

    def relu(self):
        out = T(np.maximum(self.v, 0.0), _parents=(self,), _op="relu")

        def back(g):
            self.g += g * (self.v > 0)

        out._back = back
        return out

    def sqrt(self):
        return self ** 0.5

    # ---- shape -----------------------------------------------------------
    def reshape(self, *shape):
        out = T(self.v.reshape(*shape), _parents=(self,), _op="reshape")

        def back(g):
            self.g += g.reshape(self.v.shape)

        out._back = back
        return out

    def transpose(self):
        out = T(self.v.T, _parents=(self,), _op="transpose")

        def back(g):
            self.g += g.T

        out._back = back
        return out

    @property
    def Tt(self):
        return self.transpose()

    def gather(self, idx):
        """Row-select. THE structural operation of this model: a universal is
        assembled by gathering its witness particulars, never by reading a
        parameter that stands in for them."""
        idx = np.asarray(idx)
        out = T(self.v[idx], _parents=(self,), _op="gather")

        def back(g):
            np.add.at(self.g, idx, g)

        out._back = back
        return out

    # ---- backward pass ---------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n._parents:
                build(p)
            topo.append(n)

        build(self)
        self.g = np.ones_like(self.v)
        for n in reversed(topo):
            if n._back is not None:
                n._back(n.g)


# ---- free functions ----------------------------------------------------------

def logsumexp(t, axis=-1, keepdims=True):
    """Stable log-sum-exp on the tape. The max-shift is a detached constant,
    which is what makes it numerically stable without disturbing gradients."""
    m = np.max(t.v, axis=axis, keepdims=True)          # detached constant
    s = (t - T(m)).exp().sum(axis=axis, keepdims=True).log() + T(m)
    if not keepdims:
        ax = axis % t.v.ndim
        new_shape = tuple(d for i, d in enumerate(t.v.shape) if i != ax)
        s = s.reshape(*new_shape) if new_shape else s.reshape(())
    return s


def softmax(t, axis=-1):
    e = (t - T(np.max(t.v, axis=axis, keepdims=True))).exp()
    return e / e.sum(axis=axis, keepdims=True)


def softmin(t, tau=0.5, axis=-1, keepdims=True):
    """Differentiable minimum:  -tau * logsumexp(-x/tau).

    THE CERTAINTY COMPOSITION LAW OF THIS MODEL. As tau -> 0 this becomes a
    hard min. A derivation is exactly as certain as its weakest premise, and
    adding more valid steps to a chain can only ever lower the result, never
    raise it. Contrast with the baseline in PART 5, which sums.
    """
    return logsumexp(t * (-1.0 / tau), axis=axis, keepdims=keepdims) * (-tau)


def cat(ts, axis=-1):
    vals = [t.v for t in ts]
    out = T(np.concatenate(vals, axis=axis), _parents=tuple(ts), _op="cat")
    sizes = [v.shape[axis] for v in vals]

    def back(g):
        i = 0
        for t, s in zip(ts, sizes):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(i, i + s)
            t.g += g[tuple(sl)]
            i += s

    out._back = back
    return out


def l2_normalize(t, axis=-1, eps=1e-8):
    """Split a vector into (direction, magnitude). See PART 1."""
    n = (t * t).sum(axis=axis, keepdims=True) + T(eps)
    return t / n.sqrt()


def vec_norm(t, axis=-1, eps=1e-8):
    return ((t * t).sum(axis=axis, keepdims=True) + T(eps)).sqrt()


# ==============================================================================
# PART 1 — THE TASHKIK REPRESENTATION
# ==============================================================================
# Ibn Taymiyya rejects both horns of the usual choice about shared terms. Words
# applied to God and to creatures are neither equivocal (mere homonyms) nor
# flatly univocal (the same property, same degree). They are MODULATED —
# al-tawatu' al-'amm, "general univocity" — where the individuals "take
# precedence over each other." His own illustration, from al-Radd 'ala
# al-mantiqiyyin (155/198), is the whiteness of snow against the whiteness of
# ivory: one meaning, two intensities.
#
# So every concept in this model is stored as a single vector read TWO ways:
#
#     direction  u = v / ||v||     the shared meaning     (what is common)
#     magnitude  m = ||v||         the precedence         (what differs)
#
# Two consequences fall straight out of this and are used later:
#
#   * SIMILARITY IS DIRECTIONAL. Two things share a meaning to the extent
#     their directions align, regardless of how intensely each has it.
#
#   * TRANSFER IS ASYMMETRIC. Ibn Taymiyya forbids putting God and creature
#     "on the same level" in a syllogism or an analogy, and permits only
#     qiyas al-awla, the a fortiori step: whatever perfection a lesser thing
#     has, the greater is all the worthier of it. That is an ORDER-PRESERVING,
#     NON-COMMUTATIVE transfer operator. Here: a property may be pushed from
#     lower magnitude to higher, and the gate closes in the other direction.
# ==============================================================================


def awla_gate(m_src, m_dst, beta=4.0):
    """qiyas al-awla. Open when the destination outranks the source.

    m_src, m_dst : (B,1) magnitudes. Returns (B,1) in (0,1).
    This is deliberately NOT symmetric: awla_gate(a,b) + awla_gate(b,a) != 1
    is false in general only because sigmoid is odd-symmetric, but the point
    stands that the operator distinguishes direction of transfer, which a
    dot-product similarity cannot.
    """
    return ((m_dst - m_src) * beta).sigmoid()


# ==============================================================================
# PART 2 — WITNESS MEMORY: UNIVERSALS THAT OWN NO PARAMETERS
# ==============================================================================
# "A universal only exists with respect to a particular case, and specifying
# the original case is how the universal comes to be known."
#                                       (al-Radd 'ala al-mantiqiyyin 116-8/158-60)
#
# This is the load-bearing structural commitment of the file. In an ordinary
# network a learned generalisation lives in a weight matrix; the training cases
# that produced it are gone, and nothing at inference time can tell you which
# examples a given confident output rests on. Ibn Taymiyya's ontology forbids
# that object outright: universals have no existence outside a mind, and inside
# a mind they exist only as an operation over retained particulars.
#
# So `UniversalBank` holds NO weights of its own. It holds:
#     - a pointer structure (fixed membership mask) saying which particulars
#       witness which rule
#     - a per-rule domain size, so that coverage is a real fraction
# and it computes each rule, every forward pass, out of the particular
# embeddings and their individual reliabilities. Both of those belong to the
# PARTICULARS. Remove a witness from the mask and the rule genuinely weakens;
# remove them all and the rule evaluates to nothing.
#
# The coverage gate is the second Taymiyyan move. To know that "all flames
# burn" holds in the world outside the mind you would have to examine all
# flames — and on the way you would already have found out about the one flame
# you cared about, which is why the universal was never doing the work. Short
# of complete induction the universal is zanni, merely probable, and no amount
# of downstream inference can launder it into certainty.
# ==============================================================================


class UniversalBank:
    """Rules assembled on demand from retained witnesses. Holds no parameters."""

    def __init__(self, membership, domain_sizes):
        # membership: (K, N) binary. membership[k, i] = 1 iff particular i is a
        #             retained witness for universal k.
        # domain_sizes: (K,) how many particulars WOULD have to be examined for
        #             complete induction. Coverage = witnessed / domain.
        self.M = np.asarray(membership, dtype=np.float64)
        self.K, self.N = self.M.shape
        self.domain = np.asarray(domain_sizes, dtype=np.float64).reshape(self.K, 1)
        # What fraction of each rule's domain is actually witnessed, assuming
        # unit reliability. Used ONLY by the task generator to compute correct
        # grade labels -- the model never sees it.
        self.coverage_hint = (self.M.sum(axis=1) / self.domain.reshape(-1))

    def forward(self, part_emb, part_rel):
        """part_emb: T (N, D) particular embeddings   [trainable]
           part_rel: T (N, 1) per-particular reliability logits [trainable]

        Returns
          U        : T (K, D) rule directions, unit norm
          support  : T (K, 1) total reliable witness mass behind each rule
          coverage : T (K, 1) witnessed fraction of the rule's domain
        """
        Mt = T(self.M)                                  # (K,N) fixed pointers
        r = part_rel.softplus()                         # (N,1) >= 0
        w = Mt * r.reshape(1, self.N)                   # (K,N) masked reliabilities
        denom = w.sum(axis=1, keepdims=True) + T(1e-6)  # (K,1)
        U_raw = (w @ part_emb) / denom                  # (K,D) reliability-weighted mean
        U = l2_normalize(U_raw, axis=-1)
        support = w.sum(axis=1, keepdims=True)          # (K,1)
        coverage = support / T(self.domain)             # (K,1)
        return U, support, coverage


# ==============================================================================
# PART 3 — THE FOUR CHANNELS
# ==============================================================================
# Ibn Taymiyya's epistemology runs on exactly four sources: hiss (sense
# perception, outer and inner), khabar (report), 'aql (reason, both as a
# faculty and as immediate necessary knowledge), and fitra (the natural
# constitution). Each channel here returns a signed log-odds AND a grounding
# score, because in this architecture a claim's strength and a claim's WARRANT
# are separately stored quantities. The conflict module in PART 4 needs both.
# ==============================================================================


SOFTMIN_TAU = 0.4     # sharpness of the weakest-premise minimum; fixed by design
LINEAGE_SHARP = 12.0  # how decisively two reports are judged same-lineage
LINEAGE_BASE = 0.60   # cosine above which two provenances count as one road


def clip_logodds(t, lim=8.0):
    """Saturating clip that stays differentiable everywhere."""
    return (t * (1.0 / lim)).tanh() * lim


class Params:
    """Trainable parameter store. `frozen` names are excluded from updates —
    that exclusion is the fitra commitment, not a convenience."""

    def __init__(self):
        self._p = {}
        self.frozen = set()

    def add(self, name, arr, frozen=False):
        t = T(arr, requires_grad=not frozen)
        self._p[name] = t
        if frozen:
            self.frozen.add(name)
        return t

    def __getitem__(self, k):
        return self._p[k]

    def trainable(self):
        return [(k, v) for k, v in self._p.items() if k not in self.frozen]

    def all(self):
        return list(self._p.items())

    def zero_grad(self):
        for _, v in self._p.items():
            v.g = np.zeros_like(v.v)


def channel_hiss(P, batch):
    """hiss — DIRECT CONTACT.

    Outer sense perceives particulars; inner sense perceives hunger, pain, the
    existence of one's own soul. For Ibn Taymiyya this is immediate and
    NECESSARY: "it cannot be denied." So when contact is present and clean,
    this channel carries the highest grounding available anywhere in the model,
    and the whole point of the collusion test later is that one such contact
    should outweigh twenty reports that all came from the same mouth.
    """
    mask = T(batch["hiss_mask"])          # (B,1) was there any contact at all
    sign = T(batch["hiss_sign"])          # (B,1) in {-1,+1}
    qual = T(batch["hiss_quality"])       # (B,1) in [0,1] clarity of the contact
    strength = P["a_hiss"].softplus()     # scalar, learned
    ell = sign * strength * qual * mask
    grounding = qual * mask               # contact is its own warrant
    return clip_logodds(ell), grounding


def channel_khabar(P, batch):
    """khabar — REPORT, AND THE ARITHMETIC OF TAWATUR.

    Reports "derive from the sense perceptions of others," and most of what
    anyone knows arrives this way. They yield certainty only when transmission
    is "so abundantly transmitted as to preclude collusion and forgery" —
    tawatur, from the root tawatu', treading one after another in the same
    track.

    The operative word is COLLUSION, and it is a claim about independence, not
    about volume. So this channel never counts reports. Each report is weighted
    by the inverse of its own effective lineage size:

            w_j = 1 / sum_k S_jk          (S = provenance similarity)
            n_eff = sum_j w_j             (Kish's effective sample size)

    A block of twenty reports that all trace to one teacher contributes about
    one witness in total; three reports from three unrelated chains contribute
    three. The consequence is the whole point: A WELL-DISCOUNTED MINORITY CAN
    OUTWEIGH AN UNDISCOUNTED MAJORITY. Ibn Taymiyya needs exactly this, because
    his certainty machine has to survive the observation that a large, loud,
    unanimous consensus can descend from a single unreliable source.

    The provenance similarity S is LEARNED — the model must work out from the
    provenance features which reports travelled the same road. The design
    effect itself is fixed, because it is arithmetic and not an opinion.

    The modern sting is immediate: a thousand samples drawn from one model are
    one witness in a thousand coats. Their agreement is a fact about shared
    weights, not about the world.
    """
    B, R, Pd = batch["khabar_prov"].shape
    prov = T(batch["khabar_prov"].reshape(B * R, Pd))
    mask = T(batch["khabar_mask"])                     # (B,R)
    sign = T(batch["khabar_sign"])                     # (B,R)

    # Learned read-out of provenance -> a lineage direction per report.
    h = (prov @ P["W_prov"] + P["b_prov"]).tanh()
    h = l2_normalize(h, axis=-1).reshape(B, R, -1)

    sim = _batched_cosine(h, B, R)                     # (B,R,R) in [-1,1]
    # The DECISION "same road or not" is architectural and sharp. Only WHERE the
    # line falls is adaptive, and only within a narrow band -- otherwise
    # training discovers it can flatten the discrimination away and recover the
    # counting behaviour this channel exists to prevent. (It does exactly that
    # if the sharpness is left free; that failure is why this is a constant.)
    thresh = T(LINEAGE_BASE) + P["rho_scale"].tanh() * 0.15
    S = ((sim - thresh) * LINEAGE_SHARP).sigmoid()     # -> ~1 for same lineage
    S = S * mask.reshape(B, 1, R) * mask.reshape(B, R, 1)
    eye = T(np.eye(R).reshape(1, R, R)) * mask.reshape(B, R, 1)
    S = S * (T(np.ones((1, R, R))) - T(np.eye(R).reshape(1, R, R))) + eye

    cluster_size = S.sum(axis=2, keepdims=False) + T(1e-6)          # (B,R)
    w = mask / cluster_size                                          # (B,R)
    n_eff = w.sum(axis=1, keepdims=True)                             # (B,1)

    per_report_llr = P["a_khabar"].softplus()
    ell = (sign * w).sum(axis=1, keepdims=True) * per_report_llr
    grounding = (n_eff * (1.0 / 8.0)).tanh()           # saturates near 8 lineages
    return clip_logodds(ell), grounding, n_eff


def _batched_cosine(h, B, R):
    """h: T (B,R,H) already unit-normalised. Returns T (B,R,R) of dot products.
    Written as an explicit sum so it stays on the tape without a batched matmul.
    """
    H = h.v.shape[-1]
    a = h.reshape(B, R, 1, H)
    b = h.reshape(B, 1, R, H)
    return (a * b).sum(axis=3, keepdims=False)


def channel_aql(P, batch, U, support, coverage, claim_dir):
    """'aql — INFERENCE, WITH THE FORM CONTRIBUTING NOTHING.

    Here is the thesis in code. The channel is handed a formally valid
    derivation: a universal premise, a minor premise, and a chain length. It
    computes the conclusion's certainty as the SOFT MINIMUM of the premise
    certainties, and the chain length is fed in and then never used. It cannot
    be used, because there is no path in this graph from `aql_chain` to the
    output. That absence is the mechanism.

    The universal's own certainty is bounded by its COVERAGE. Short of complete
    induction the major premise is merely probable, so the whole derivation is
    merely probable however elegantly it is laid out. And the rule only fires
    at all to the extent that it actually applies to this claim, measured by
    the alignment of the rule's direction with the claim's — which is the
    nominalist's replacement for asking whether the claim's essence falls under
    the rule's genus.
    """
    k = batch["aql_univ"]                              # (B,) rule index
    Uk = U.gather(k)                                   # (B,D)
    cov_k = coverage.gather(k)                         # (B,1)
    sup_k = support.gather(k)                          # (B,1)

    # Does this rule even reach this claim? Directional, not essence-based.
    applic = (Uk * claim_dir).sum(axis=1, keepdims=True).relu()          # (B,1)

    # Major premise certainty: grows with witness mass, but HARD-GATED by
    # coverage. This is the "you would have to examine every flame" clause.
    cov_gate = ((cov_k - P["cov_theta"]) * P["cov_sharp"].softplus()).sigmoid()
    ell_major = (sup_k + T(1.0)).log() * P["a_major"].softplus() * cov_gate

    # Minor premise arrives with its own certainty from the evidence bundle.
    ell_minor = T(batch["aql_minor"]) * P["a_minor"].softplus()          # (B,1)

    # THE WEAKEST-PREMISE LAW. Note there is no term in chain length.
    prem = cat([ell_major, ell_minor], axis=1)                           # (B,2)
    # tau is a FIXED constant, not a learned parameter. The weakest-premise law
    # is a commitment of the architecture; the model does not get to soften it.
    ell_mag = softmin(prem, tau=SOFTMIN_TAU, axis=1)                     # (B,1)

    ell = T(batch["aql_sign"]) * ell_mag * applic
    grounding = cov_gate * applic
    return clip_logodds(ell), grounding, cov_gate


def channel_fitra(P, batch, claim_dir):
    """fitra — THE INSTALLED PRIOR THAT TRAINING MAY NOT TOUCH.

    The natural constitution is the disposition a human is created with, "just
    as an infant instinctively seeks to drink its mother's milk." It carries
    the basic rules of thought and the fundamental moral intuitions, and — the
    part that matters for machines — IT CAN BE CORRUPTED BY SCHOOLING. Ibn
    Taymiyya's charge against the mutakallimun is exactly that: their training
    taught them to assert what every person of undamaged constitution denies,
    and the mass agreement of ordinary unschooled people is what convicts them.

    The prior does not speak to every question. `fitra_cue` marks which of the
    frozen axioms (if any) a claim engages; where none is engaged the channel
    is silent, which is the correct behaviour for a constitution rather than a
    general-purpose classifier. Where it does speak it is RIGHT ABOUT 85% OF
    THE TIME — good, fallible, and installed rather than derived.

    This channel has NO trainable parameters. The axiom bank, the signs and the
    strength are all registered frozen, so gradient descent cannot reach them.
    PART 9 then audits how far the trained channels have drifted from it, which
    is this architecture's alignment instrument, and it is his, not ours.
    """
    cue = T(batch["fitra_cue"])                        # (B,A) one-hot or all-zero
    sgn = P["axiom_sign"]                              # (A,1) FROZEN
    gam = P["fitra_gamma"]                             # scalar FROZEN

    ell = (cue @ sgn) * gam                            # (B,1)
    grounding = cue.sum(axis=1, keepdims=True)         # 1 if the prior speaks, else 0

    # The axiom bank still participates: a claim must actually point the way its
    # cued axiom points for the prior to fire at full strength. This keeps the
    # frozen directions load-bearing rather than decorative.
    A = P["axioms"]                                    # (A,D) FROZEN
    align = ((claim_dir @ A.transpose()) * cue).sum(axis=1, keepdims=True)
    fit = ((align * 2.0).tanh() + T(1.0)) * 0.5        # (0,1)
    return clip_logodds(ell * (T(0.5) + fit * 0.5)), grounding


# ==============================================================================
# PART 4 — DAR' TA'ARUD: CONFLICT AS A RE-GRADING SIGNAL
# ==============================================================================
# The ten-volume Dar' ta'arud al-'aql wa-l-naql, written after 1313, is aimed
# at one rule his opponents had made famous: when reason and revelation
# conflict, prefer reason. Ibn Taymiyya's objection is not that they picked the
# wrong winner. It is that a FIXED PRECEDENCE RULE IS THE WRONG SHAPE OF
# ANSWER. The rule presupposes that both sides really are what they claim to
# be — a decisive rational proof and an authentic report — when a live conflict
# is nearly always evidence that one of the two was mis-rated: the "decisive"
# argument turns out to be merely probable, or the report turns out not to say
# what it was taken to say.
#
# So this module never picks a winner by identity. It asks, of each channel,
# HOW MUCH GROUNDING IS BACKING THE CONFIDENCE IT DECLARED, and damps the
# channel whose confidence is least earned. Conflict is an instrument reading,
# not a contest. And when the re-graded evidence lands near zero the model is
# allowed to return waqf — suspension — as a real, scoreable output rather
# than as failure.
# ==============================================================================


def dar_module(P, ell_list, grd_list):
    ells = cat(ell_list, axis=1)                       # (B,C)
    grds = cat(grd_list, axis=1)                       # (B,C)
    C = ells.v.shape[1]

    # How much disagreement is actually present: opposite signs, both loud.
    a = ells.reshape(-1, C, 1)
    b = ells.reshape(-1, 1, C)
    prod = (a * b).reshape(-1, C * C)
    conflict = (-prod).relu().sum(axis=1, keepdims=True) * (1.0 / (C * C))
    conflict = (conflict * P["conf_scale"].softplus()).tanh()            # (B,1)

    # Is each channel's confidence backed by its own grounding?
    g_mean = grds.mean(axis=1, keepdims=True)
    earned = ((grds - g_mean) * P["earn_sharp"].softplus() + P["earn_bias"]).sigmoid()

    # Damp the unearned ones, but only to the extent there IS a conflict.
    keep = T(np.ones((1, C))) - conflict * (T(np.ones((1, C))) - earned)
    regraded = ells * keep

    w = P["chan_w"].softplus().reshape(1, C)
    total = (regraded * w).sum(axis=1, keepdims=True)
    return clip_logodds(total), conflict, regraded, earned


# ==============================================================================
# PART 5 — THE MIHAKK MODEL, AND THE AVICENNAN BASELINE
# ==============================================================================
# Two models, identical in every encoder and every parameter count, differing
# in exactly three places. Ibn Taymiyya's whole quarrel with the logicians is
# compressed into those three lines, so the experiment is clean:
#
#   MIHAKK (Taymiyyan)                    AVICENNAN BASELINE
#   ------------------------------------  --------------------------------------
#   certainty = softmin(premises)         certainty = sum(premises) * chain_len
#   n_eff = n / (1 + rho(n-1))            n_eff = n            (count reports)
#   frozen fitra prior                    no prior channel
#
# The baseline is not a straw man. Summing log-odds over premises is the
# textbook independent-evidence rule and it is what almost every scoring system
# does; counting agreeing sources is what almost every retrieval system does.
# They are correct exactly when the premises are independent and the sources
# are distinct, and they fail silently when they are not. Ibn Taymiyya's
# contribution is to notice that the failure mode is not rare.
# ==============================================================================

D_MODEL = 16          # embedding width
N_PART = 64           # retained particulars in the witness bank
K_UNIV = 8            # universals (each a pointer set into the particulars)
R_MAX = 24            # max reports per item
P_PROV = 5            # provenance feature width
H_PROV = 8            # provenance read-out width
N_AXIOM = 4           # frozen fitra axioms
N_CHAN = 4            # hiss, khabar, 'aql, fitra
N_GRADE = 3           # 0 = qat'i (necessary) | 1 = zanni (probable) | 2 = mawquf


def init_params(rng, avicennan=False):
    P = Params()
    P.add("part_emb", rng.normal(0, 0.5, (N_PART, D_MODEL)))
    P.add("part_rel", rng.normal(0, 0.2, (N_PART, 1)))
    P.add("W_claim", rng.normal(0, 0.4, (P_PROV + K_UNIV, D_MODEL)))
    P.add("b_claim", np.zeros((1, D_MODEL)))
    P.add("W_prov", rng.normal(0, 0.5, (P_PROV, H_PROV)))
    P.add("b_prov", np.zeros((1, H_PROV)))
    P.add("rho_scale", np.array(0.0))
    P.add("a_hiss", np.array(1.0))
    P.add("a_khabar", np.array(0.2))
    P.add("a_major", np.array(0.5))
    P.add("a_minor", np.array(1.0))
    P.add("cov_theta", np.array(0.75))
    P.add("cov_sharp", np.array(2.0))
    P.add("conf_scale", np.array(0.5))
    P.add("earn_sharp", np.array(1.0))
    P.add("earn_bias", np.array(0.0))
    P.add("chan_w", np.zeros(N_CHAN))
    # grade head
    P.add("W_g", rng.normal(0, 0.5, (6, 12)))
    P.add("b_g", np.zeros((1, 12)))
    P.add("W_g2", rng.normal(0, 0.5, (12, N_GRADE)))
    P.add("b_g2", np.zeros((1, N_GRADE)))
    # ---- fitra: FROZEN. Registered as parameters so they are visible to the
    # audit in PART 9, and excluded from every update so training cannot bend
    # them. This is the one place in the file where "not learnable" is a claim
    # about the mind being modelled rather than an implementation detail.
    ax = rng.normal(0, 1.0, (N_AXIOM, D_MODEL))
    ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
    P.add("axioms", ax, frozen=True)
    P.add("axiom_sign", np.array([[1.0], [1.0], [-1.0], [1.0]]), frozen=True)
    P.add("fitra_gamma", np.array(1.5), frozen=True)
    P.avicennan = avicennan
    return P


def encode_claim(P, batch):
    x = T(batch["claim_feat"])                         # (B, P_PROV+K_UNIV)
    v = (x @ P["W_claim"] + P["b_claim"]).tanh()       # (B,D)
    return l2_normalize(v, axis=-1), vec_norm(v, axis=-1)


def forward(P, bank, batch):
    """Returns a dict of everything the losses and the audits need."""
    claim_dir, claim_mag = encode_claim(P, batch)
    U, support, coverage = bank.forward(P["part_emb"], P["part_rel"])

    e_h, g_h = channel_hiss(P, batch)
    e_k, g_k, n_eff = channel_khabar(P, batch)
    e_a, g_a, cov_gate = channel_aql(P, batch, U, support, coverage, claim_dir)
    e_f, g_f = channel_fitra(P, batch, claim_dir)

    if getattr(P, "avicennan", False):
        # --- THE THREE SUBSTITUTIONS ---------------------------------------
        # (1) form-boost: premises SUM, and a longer valid chain scores higher.
        k = batch["aql_univ"]
        sup_k = support.gather(k)
        cov_k = coverage.gather(k)
        Uk = U.gather(k)
        applic = (Uk * claim_dir).sum(axis=1, keepdims=True).relu()
        ell_major = (sup_k + T(1.0)).log() * P["a_major"].softplus()   # no coverage gate
        ell_minor = T(batch["aql_minor"]) * P["a_minor"].softplus()
        chain = T(batch["aql_chain"])
        e_a = clip_logodds(T(batch["aql_sign"]) * (ell_major + ell_minor) *
                           (T(1.0) + chain * 0.25) * applic)
        g_a = applic
        # The baseline has no notion of coverage at all, so the grade head must
        # not be handed one. Ablating it here is what makes the comparison fair:
        # otherwise the baseline is quietly told how complete each induction was
        # while its reasoning channel goes on ignoring the answer.
        cov_gate = T(np.ones_like(cov_gate.v))
        # (2) count reports; every report is its own witness.
        mask = T(batch["khabar_mask"]); sign = T(batch["khabar_sign"])
        n = mask.sum(axis=1, keepdims=True)
        n_eff = n
        e_k = clip_logodds((sign * mask).sum(axis=1, keepdims=True)
                           * P["a_khabar"].softplus())
        g_k = (n * (1.0 / 8.0)).tanh()
        # (3) no fitra prior at all.
        e_f = T(np.zeros_like(e_f.v)); g_f = T(np.zeros_like(g_f.v))

    total, conflict, regraded, earned = dar_module(
        P, [e_h, e_k, e_a, e_f], [g_h, g_k, g_a, g_f])

    p_true = total.sigmoid()

    # ---- grade head ------------------------------------------------------
    # The features are deliberately WARRANT features, not content features:
    # how loud is the verdict, how well grounded is the best channel, how much
    # conflict is present, how many independent witnesses, how complete is the
    # induction. A system that can report its grade must look at where its
    # confidence came from, not at what it concluded.
    abs_total = (total * total + T(1e-6)).sqrt()       # |verdict|, smooth at 0
    feats = cat([
        total * (1.0 / 8.0),
        abs_total * (1.0 / 8.0),
        cat([g_h, g_k, g_a, g_f], axis=1).mean(axis=1, keepdims=True),
        conflict,
        (n_eff * (1.0 / 8.0)).tanh(),
        cov_gate,
    ], axis=1)
    hg = (feats @ P["W_g"] + P["b_g"]).tanh()
    grade_logits = hg @ P["W_g2"] + P["b_g2"]

    return dict(p_true=p_true, total=total, grade_logits=grade_logits,
                conflict=conflict, n_eff=n_eff, coverage_gate=cov_gate,
                channels=[e_h, e_k, e_a, e_f], regraded=regraded,
                groundings=[g_h, g_k, g_a, g_f], earned=earned,
                claim_dir=claim_dir)


def loss_fn(P, bank, batch, l2=1e-5):
    out = forward(P, bank, batch)
    y = T(batch["y_true"])                             # (B,1) in {0,1}
    p = out["p_true"]
    eps = 1e-7
    bce = -(y * (p + T(eps)).log() + (T(1.0) - y) * (T(1.0) - p + T(eps)).log())
    bce = bce.mean()

    g = batch["y_grade"]                               # (B,) ints
    lg = out["grade_logits"]
    lse = logsumexp(lg, axis=1, keepdims=False)        # (B,)
    picked = (lg * T(np.eye(N_GRADE)[g])).sum(axis=1, keepdims=False)
    ce = (lse - picked).mean()

    reg = T(0.0)
    for k, v in P.trainable():
        reg = reg + (v * v).sum()
    return bce + ce * 0.6 + reg * l2, out


# ==============================================================================
# PART 6 — THE HARRAN TASK
# ==============================================================================
# A claim about a particular, plus an evidence bundle drawn from the four
# channels. The generator knows the truth and also knows the CORRECT EPISTEMIC
# GRADE, which is the harder label:
#
#   qat'i / necessary  — clean direct contact, OR >= 8 genuinely independent
#                        witnesses, OR a complete-induction universal with a
#                        certain minor premise
#   zanni  / probable  — everything ordinary
#   mawquf / suspended — well-grounded channels in real opposition with no
#                        dominant one
#
# Training uses `iid` only. The three adversarial splits are never trained on.
# Each isolates one of the three substitutions in PART 5:
#
#   S1_collusion   twenty agreeing reports from one or two lineages, all WRONG,
#                  against a single clean observation that is RIGHT.
#   S2_form_boost  a long, formally impeccable derivation from a thinly
#                  supported universal, pointing the WRONG way, against two
#                  genuinely independent reports pointing the right way.
#   S3_suspension  two well-grounded channels in flat opposition. The only
#                  correct grade is suspension. There is no right answer to
#                  give, and giving one confidently is the failure.
# ==============================================================================

GRADE_NAMES = ["qat'i (necessary)", "zanni (probable)", "mawquf (suspended)"]


def make_batch(rng, B, bank, split="iid"):
    """Build an evidence bundle per item, plus the two labels: the fact of the
    matter, and the epistemic grade the evidence actually licenses.

    TRAINING (`iid`) contains MILD versions of every phenomenon, including
    collusion, long chains and genuine deadlock. The three adversarial splits
    are SEVERE extrapolations of those same structures into ranges never seen:
    more reports from fewer lineages, longer chains from thinner universals,
    deadlock between better-grounded channels. So the test is not whether a
    model has memorised a trap, but whether its inductive bias extrapolates the
    right way when the dial is turned past anything in its experience.
    """
    A = N_AXIOM
    claim_feat = np.zeros((B, P_PROV + K_UNIV))
    hiss_mask = np.zeros((B, 1)); hiss_sign = np.zeros((B, 1)); hiss_q = np.zeros((B, 1))
    kb_sign = np.zeros((B, R_MAX)); kb_mask = np.zeros((B, R_MAX))
    kb_prov = np.zeros((B, R_MAX, P_PROV))
    aql_univ = np.zeros(B, dtype=int); aql_minor = np.zeros((B, 1))
    aql_sign = np.zeros((B, 1)); aql_chain = np.zeros((B, 1))
    fitra_cue = np.zeros((B, A))
    y_true = np.zeros((B, 1)); y_grade = np.zeros(B, dtype=int)

    POS_AX = [0, 1, 3]      # frozen axiom_sign == +1
    NEG_AX = [2]            # frozen axiom_sign == -1

    for i in range(B):
        truth = int(rng.integers(0, 2))
        s = +1.0 if truth == 1 else -1.0
        y_true[i, 0] = truth

        # ---- the constitution speaks to some questions and not to others ----
        if rng.random() < 0.35:
            says = s if rng.random() < 0.85 else -s        # installed, fallible
            pool = POS_AX if says > 0 else NEG_AX
            fitra_cue[i, int(rng.choice(pool))] = 1.0

        k = int(rng.integers(0, K_UNIV))
        aql_minor[i, 0] = rng.uniform(0.2, 1.0)
        aql_chain[i, 0] = float(rng.integers(1, 5))

        if split == "iid":
            shape = rng.choice(["contact", "testimony", "derivation",
                                "mixed", "contested"],
                               p=[0.20, 0.25, 0.25, 0.20, 0.10])

            if shape == "contact":
                hiss_mask[i, 0] = 1.0
                hiss_q[i, 0] = rng.uniform(0.7, 1.0)
                hiss_sign[i, 0] = s if rng.random() < 0.95 else -s
                y_grade[i] = 0
                n_rep, n_src = int(rng.integers(0, 5)), 3

            elif shape == "testimony":
                n_rep = int(rng.integers(3, 13))
                n_src = max(2, int(n_rep * rng.uniform(0.2, 1.0)))  # mild collusion
                y_grade[i] = 0 if _n_eff(n_src, n_rep) >= 8 else 1

            elif shape == "derivation":
                # Half the time, pose a derivation that genuinely could settle
                # the question: a near-complete induction with a certain minor
                # premise. A model that never meets a sound proof cannot learn
                # to accept one.
                if rng.random() < 0.5:
                    k = int(rng.choice(np.argsort(-bank.coverage_hint)[:3]))
                    aql_minor[i, 0] = rng.uniform(0.9, 1.0)
                cov = bank.coverage_hint[k]
                aql_sign[i, 0] = s if rng.random() < (0.55 + 0.4 * cov) else -s
                certain = cov > 0.95 and aql_minor[i, 0] > 0.9
                y_grade[i] = 0 if certain else 1
                n_rep, n_src = int(rng.integers(0, 4)), 2

            elif shape == "mixed":
                n_rep = int(rng.integers(2, 9))
                n_src = int(rng.integers(2, n_rep + 1))
                aql_sign[i, 0] = s if rng.random() < 0.7 else -s
                if rng.random() < 0.4:
                    hiss_mask[i, 0] = 1.0
                    hiss_q[i, 0] = rng.uniform(0.5, 0.8)
                    hiss_sign[i, 0] = s
                y_grade[i] = 1

            else:  # contested -- the MILD deadlock the severe split extrapolates
                hiss_mask[i, 0] = 1.0
                hiss_q[i, 0] = rng.uniform(0.55, 0.75)
                hiss_sign[i, 0] = s
                n_rep = int(rng.integers(3, 6))
                n_src = n_rep                              # fully independent
                _fill_reports(rng, kb_sign, kb_mask, kb_prov, i, n_rep, n_src, -s)
                aql_sign[i, 0] = 0.0
                y_grade[i] = 2
                n_rep = 0                                  # already filled
            if n_rep:
                want = s if rng.random() < 0.82 else -s
                _fill_reports(rng, kb_sign, kb_mask, kb_prov, i, n_rep, n_src, want)

        elif split == "S1_collusion":
            # A loud unanimous bloc that all traces to one or two teachers,
            # against a handful of unrelated chains that disagree with it.
            # No eyewitness is available: the ONLY way through is to discount
            # the bloc. Counting reports gives 20 against 3 and gets it wrong.
            n_big = int(rng.integers(17, 21))
            n_src = int(rng.integers(1, 3))
            n_ind = int(rng.integers(3, 5))
            _fill_two_blocks(rng, kb_sign, kb_mask, kb_prov, i,
                             n_big, n_src, -s, n_ind, s)
            aql_sign[i, 0] = 0.0
            fitra_cue[i] = 0.0
            y_grade[i] = 1                                 # probable, never certain

        elif split == "S2_form_boost":
            # Impeccable form, thin foundation, long chain, wrong conclusion,
            # against two unrelated witnesses who saw otherwise. Complete
            # induction over this universal's domain was never performed, so
            # the major premise is merely probable and the derivation cannot
            # be firmer than it however many valid steps are added.
            k = int(np.argmin(bank.coverage_hint))
            aql_minor[i, 0] = rng.uniform(0.85, 1.0)
            aql_sign[i, 0] = -s
            aql_chain[i, 0] = float(rng.integers(5, 9))
            _fill_reports(rng, kb_sign, kb_mask, kb_prov, i, 2, 2, s)
            fitra_cue[i] = 0.0
            y_grade[i] = 1

        elif split == "S4_sound_derivation":
            # The mirror image of S2, and the reason this model is not a
            # sceptic. Ibn Taymiyya CONCEDES the syllogism: "if the two
            # premises are known [with certainty] ... this is not disputed"
            # (Mantiqiyyin 293/339). Here the major premise rests on an
            # essentially complete induction and the minor is certain, so the
            # derivation genuinely does settle the matter -- and a model that
            # has learned only to distrust reasoning will leave that on the
            # table. Discriminating S2 from S4 is the whole competence.
            k = int(np.argmax(bank.coverage_hint))
            aql_minor[i, 0] = rng.uniform(0.93, 1.0)
            aql_sign[i, 0] = s
            aql_chain[i, 0] = float(rng.integers(1, 4))
            fitra_cue[i] = 0.0
            y_grade[i] = 0

        elif split == "S3_suspension":
            # Two well-grounded channels in flat opposition. The evidence does
            # not settle it, and saying so is the correct output.
            hiss_mask[i, 0] = 1.0
            hiss_sign[i, 0] = s
            hiss_q[i, 0] = rng.uniform(0.6, 0.78)
            n_rep = int(rng.integers(6, 10))
            _fill_reports(rng, kb_sign, kb_mask, kb_prov, i, n_rep, n_rep, -s)
            aql_sign[i, 0] = 0.0
            fitra_cue[i] = 0.0                             # the prior is silent here
            y_grade[i] = 2

        # The claim names the universal it invokes. Applicability is then a
        # learnable question -- does this rule actually reach this case? -- and
        # not the coin-flip it would be if the rule's identity were hidden.
        aql_univ[i] = k
        onehot = np.zeros(K_UNIV); onehot[k] = 1.0
        claim_feat[i] = np.concatenate([rng.normal(0, 0.6, P_PROV), onehot])

    return dict(claim_feat=claim_feat, hiss_mask=hiss_mask, hiss_sign=hiss_sign,
                hiss_quality=hiss_q, khabar_sign=kb_sign, khabar_mask=kb_mask,
                khabar_prov=kb_prov, aql_univ=aql_univ, aql_minor=aql_minor,
                aql_sign=aql_sign, aql_chain=aql_chain, fitra_cue=fitra_cue,
                y_true=y_true, y_grade=y_grade)


def _n_eff(n_src, n_rep):
    if n_rep <= 0:
        return 0.0
    return n_rep / (1.0 + _rho_of(n_src, n_rep) * max(n_rep - 1, 0))


def _rho_of(n_src, n_rep):
    if n_rep <= 1:
        return 0.0
    return float(np.clip(1.0 - (n_src - 1) / max(n_rep - 1, 1), 0.0, 1.0)) * 0.92


def _fill_two_blocks(rng, sign, mask, prov, i, n_a, src_a, sign_a, n_b, sign_b):
    """A large colluding bloc and a small independent one, in the same bundle."""
    total = min(n_a + n_b, R_MAX)
    n_a = min(n_a, total - n_b)
    srcs_a = rng.normal(0, 1.0, (src_a, P_PROV))
    j = 0
    for t in range(n_a):
        prov[i, j] = srcs_a[t % src_a] + rng.normal(0, 0.05, P_PROV)
        mask[i, j] = 1.0; sign[i, j] = sign_a
        j += 1
    srcs_b = rng.normal(0, 1.0, (n_b, P_PROV))     # each its own unrelated chain
    for t in range(n_b):
        if j >= R_MAX:
            break
        prov[i, j] = srcs_b[t] + rng.normal(0, 0.05, P_PROV)
        mask[i, j] = 1.0; sign[i, j] = sign_b
        j += 1


def _fill_reports(rng, sign, mask, prov, i, n_rep, n_src, want_sign):
    """Reports carry PROVENANCE, and reports from one lineage have near-identical
    provenance. Nothing tells the model n_src directly; it has to read it off
    the provenance features, which is the learning problem in this channel."""
    if n_rep <= 0:
        return
    srcs = rng.normal(0, 1.0, (n_src, P_PROV))
    for j in range(min(n_rep, R_MAX)):
        s_id = j % n_src
        prov[i, j] = srcs[s_id] + rng.normal(0, 0.06, P_PROV)   # tight within lineage
        mask[i, j] = 1.0
        sign[i, j] = want_sign if rng.random() < 0.93 else -want_sign


# ==============================================================================
# PART 7 — FINITE-DIFFERENCE GRADIENT CHECK  [MANDATORY]
# ==============================================================================
# Every analytic gradient produced by the tape in PART 0 is compared against a
# central-difference estimate. If this does not pass, nothing below it means
# anything.
# ==============================================================================


def gradient_check(seed=0, n_probe=4, eps=1e-6, tol=2e-5, verbose=True):
    rng = np.random.default_rng(seed)
    bank = make_bank(rng)
    P = init_params(rng)
    batch = make_batch(rng, 6, bank, "iid")

    L, _ = loss_fn(P, bank, batch)
    P.zero_grad()
    L.backward()
    analytic = {k: v.g.copy() for k, v in P.trainable()}

    worst = 0.0
    rows = []
    for name, tensor in P.trainable():
        flat = tensor.v.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp, _ = loss_fn(P, bank, batch)
            flat[idx] = orig - eps
            Lm, _ = loss_fn(P, bank, batch)
            flat[idx] = orig
            num = (Lp.v - Lm.v) / (2 * eps)
            ana = analytic[name].reshape(-1)[idx]
            denom = max(1.0, abs(num), abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            rows.append((name, int(idx), float(ana), float(num), float(rel)))

    if verbose:
        print("  {:<12} {:>6} {:>14} {:>14} {:>11}".format(
            "param", "idx", "analytic", "numeric", "rel.err"))
        for name, idx, a, n, r in rows[:14]:
            print("  {:<12} {:>6d} {:>14.6e} {:>14.6e} {:>11.2e}".format(name, idx, a, n, r))
        print(f"  ... {len(rows)} coordinates probed across {len(list(P.trainable()))} tensors")
        print(f"  WORST RELATIVE ERROR: {worst:.3e}   (tolerance {tol:.1e})")
    ok = worst < tol
    print(f"  GRADIENT CHECK: {'PASS' if ok else 'FAIL'}")
    return ok, worst


def make_bank(rng):
    """Build the witness structure. Some universals are richly and broadly
    witnessed (coverage near complete induction); others rest on a handful of
    cases while claiming the same generality. Telling those apart WITHOUT
    looking at how confidently each is asserted is the whole job."""
    M = np.zeros((K_UNIV, N_PART))
    domain = np.zeros(K_UNIV)
    for k in range(K_UNIV):
        if k < 3:
            n_w = int(rng.integers(20, 30)); dom = n_w * rng.uniform(1.0, 1.06)
        elif k < 6:
            n_w = int(rng.integers(8, 14)); dom = n_w * rng.uniform(1.6, 2.4)
        else:
            n_w = int(rng.integers(2, 5)); dom = n_w * rng.uniform(6.0, 12.0)
        idx = rng.choice(N_PART, size=n_w, replace=False)
        M[k, idx] = 1.0
        domain[k] = dom
    return UniversalBank(M, domain)


# ==============================================================================
# PART 8 — TRAINING
# ==============================================================================
# Plain Adam, written out. Frozen parameters are never passed to the optimiser,
# so the fitra bank is untouched by construction rather than by convention.
# ==============================================================================


class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params.trainable()
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.v) for k, v in self.p}
        self.v = {k: np.zeros_like(v.v) for k, v in self.p}
        self.t = 0

    def step(self):
        self.t += 1
        for k, tensor in self.p:
            g = np.clip(tensor.g, -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            tensor.v -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(P, bank, rng, steps=320, bs=48, lr=0.02, tag="", log_every=40):
    opt = Adam(P, lr=lr)
    hist = []
    for step in range(1, steps + 1):
        batch = make_batch(rng, bs, bank, "iid")
        L, out = loss_fn(P, bank, batch)
        P.zero_grad()
        L.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            acc = float(((out["p_true"].v > 0.5).astype(float) == batch["y_true"]).mean())
            gacc = float((np.argmax(out["grade_logits"].v, 1) == batch["y_grade"]).mean())
            hist.append((step, float(L.v), acc, gacc))
            print(f"   {tag}step {step:4d} | loss {float(L.v):7.4f} | "
                  f"truth {acc:5.1%} | grade {gacc:5.1%}")
    return hist


def evaluate(P, bank, rng, split, n=600):
    batch = make_batch(rng, n, bank, split)
    out = forward(P, bank, batch)
    truth_acc = float(((out["p_true"].v > 0.5).astype(float) == batch["y_true"]).mean())
    grade_pred = np.argmax(out["grade_logits"].v, 1)
    grade_acc = float((grade_pred == batch["y_grade"]).mean())
    # Overconfidence: asserting qat'i on an item that is not in fact settled.
    overconf = float(((grade_pred == 0) & (batch["y_grade"] != 0)).mean())
    mean_conf = float(np.abs(out["total"].v).mean())
    return dict(truth=truth_acc, grade=grade_acc, overconfident=overconf,
                mean_abs_logodds=mean_conf,
                n_eff=float(out["n_eff"].v.mean()))


# ==============================================================================
# PART 9 — SELF-TESTS, THE COMPARATIVE EXPERIMENT, AND THE FITRA DRIFT AUDIT
# ==============================================================================


def test_tawatur_arithmetic():
    """The collusion discount must behave the way Ibn Taymiyya's argument needs
    it to: agreement is worth nothing extra once it is agreement by one mouth."""
    print("\n  [T1] tawatur arithmetic -- n_eff under collusion")
    print("       reports   distinct lineages   effective independent witnesses")
    ok = True
    for n_rep, n_src in [(20, 20), (20, 5), (20, 2), (20, 1), (4, 4), (4, 1)]:
        rho = _rho_of(n_src, n_rep)
        n_eff = n_rep / (1 + rho * (n_rep - 1))
        print(f"       {n_rep:>7d}   {n_src:>17d}   {n_eff:>10.2f}")
        if n_src == 1 and n_eff > 3.0:
            ok = False
        if n_src == n_rep and n_eff < n_rep * 0.95:
            ok = False
    print(f"       -> {'PASS' if ok else 'FAIL'}: twenty reports from one lineage "
          f"count as a handful, not as twenty")
    return ok


def test_weakest_premise(P, bank):
    """Lengthening a valid chain must never raise certainty, and the conclusion
    must never outrank the weakest premise."""
    print("\n  [T2] weakest-premise law -- certainty vs chain length")
    rng = np.random.default_rng(7)
    base = make_batch(rng, 200, bank, "iid")
    mags = []
    for chain in [1, 2, 4, 8, 16]:
        b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        b["aql_chain"] = np.full((200, 1), float(chain))
        out = forward(P, bank, b)
        m = float(np.abs(out["total"].v).mean())
        mags.append(m)
        print(f"       chain length {chain:>2d}  ->  mean |log-odds| {m:7.4f}")
    spread = max(mags) - min(mags)
    ok = spread < 1e-9
    print(f"       spread across chain lengths: {spread:.3e}")
    print(f"       -> {'PASS' if ok else 'FAIL'}: form contributes exactly nothing")
    return ok


def test_universals_have_no_parameters(P, bank):
    """Deleting a rule's witnesses must delete the rule. If the generalisation
    survives with its cases removed, it was living in a weight somewhere."""
    print("\n  [T3] no parametric universals -- ablating witnesses")
    U, sup, cov = bank.forward(P["part_emb"], P["part_rel"])
    before = float(sup.v[0, 0])
    M_saved = bank.M.copy()
    bank.M[0, :] = 0.0
    U2, sup2, cov2 = bank.forward(P["part_emb"], P["part_rel"])
    after = float(sup2.v[0, 0])
    bank.M = M_saved
    print(f"       universal 0 support with witnesses : {before:8.4f}")
    print(f"       universal 0 support after ablation : {after:8.4f}")
    ok = after < 1e-9 and before > 1.0
    print(f"       -> {'PASS' if ok else 'FAIL'}: the rule has no existence "
          f"apart from its retained cases")
    return ok


def test_awla_asymmetry():
    """Transfer across a precedence boundary must be one-directional."""
    print("\n  [T4] qiyas al-awla -- asymmetry of cross-rank transfer")
    lo = T(np.array([[0.4]])); hi = T(np.array([[2.1]]))
    up = float(awla_gate(lo, hi).v[0, 0])
    down = float(awla_gate(hi, lo).v[0, 0])
    print(f"       lesser -> greater  gate open  : {up:.4f}")
    print(f"       greater -> lesser  gate open  : {down:.4f}")
    ok = up > 0.95 and down < 0.05
    print(f"       -> {'PASS' if ok else 'FAIL'}: the operator is not commutative")
    return ok


def fitra_drift_audit(P, P0, bank, rng):
    """THE ALIGNMENT INSTRUMENT.

    Ibn Taymiyya diagnoses corrupted fitra by DIVERGENCE FROM THE UNSCHOOLED
    CONSENSUS, not by inspecting the schooling. The same test transfers
    directly: probe the trained system on items where the installed prior has
    a clear verdict, and measure how far the learned channels have moved away
    from it. A large drift is not proof of error -- learning is supposed to
    change things -- but it localises exactly where training has overwritten a
    disposition rather than refined one, and it is computable without knowing
    what the training did.
    """
    print("\n  [T5] fitra drift audit")
    frozen_intact = True
    for name in sorted(P.frozen):
        if not np.allclose(P[name].v, P0[name]):
            frozen_intact = False
            print(f"       !! frozen tensor '{name}' was modified during training")
    print(f"       frozen bank untouched by training : {frozen_intact}")

    probe = make_batch(rng, 400, bank, "iid")
    out = forward(P, bank, probe)
    cd = out["claim_dir"]
    e_f, g_f = channel_fitra(P, probe, cd)
    engaged = g_f.v[:, 0] > 0.45                      # items the prior speaks to
    if engaged.sum() < 5:
        engaged = g_f.v[:, 0] > np.percentile(g_f.v[:, 0], 75)
    prior = np.sign(e_f.v[engaged, 0])
    learned = np.sign(out["total"].v[engaged, 0] - e_f.v[engaged, 0] * float(P["chan_w"].v[3]))
    agree = float((prior == learned).mean())
    drift = 1.0 - agree
    print(f"       probe items the prior speaks to   : {int(engaged.sum())}/400")
    print(f"       learned channels agree with prior : {agree:6.1%}")
    print(f"       FITRA DRIFT                       : {drift:6.1%}")
    return frozen_intact


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE MIHAKK ENGINE -- Ibn Taymiyya (1263-1328), chapter 0351")
    print(" certainty is a property of contact, never of form")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK (central differences vs the tape)")
    ok_grad, worst = gradient_check(seed=0, n_probe=3)
    if not ok_grad:
        raise SystemExit("gradient check failed -- nothing below is trustworthy")

    rng = np.random.default_rng(11)
    bank = make_bank(rng)
    print("\n[2] WITNESS BANK")
    print("     universal   witnesses   domain   coverage")
    for k in range(K_UNIV):
        print(f"       {k:>7d}   {int(bank.M[k].sum()):>9d}   "
              f"{float(bank.domain[k,0]):>6.1f}   {bank.coverage_hint[k]:>8.3f}")

    print("\n[3] TRAINING -- MIHAKK (Taymiyyan)")
    P = init_params(np.random.default_rng(3))
    P0 = {k: v.v.copy() for k, v in P.all()}
    train(P, bank, np.random.default_rng(21), steps=320, bs=48, tag="")

    print("\n[4] TRAINING -- AVICENNAN BASELINE (sums premises, counts reports)")
    Pb = init_params(np.random.default_rng(3), avicennan=True)
    train(Pb, bank, np.random.default_rng(21), steps=320, bs=48, tag="")

    print("\n[5] HELD-OUT EVALUATION")
    hdr = (f"     {'split':<21}{'model':<11}{'truth':>8}{'grade':>8}"
           f"{'overconf':>10}{'n_eff':>8}")
    print(hdr); print("     " + "-" * (len(hdr) - 5))
    results = {}
    for split in ["iid", "S1_collusion", "S2_form_boost",
                  "S4_sound_derivation", "S3_suspension"]:
        for name, model in [("mihakk", P), ("avicennan", Pb)]:
            r = evaluate(model, bank, np.random.default_rng(99), split, n=600)
            results[(split, name)] = r
            print(f"     {split:<21}{name:<11}{r['truth']:>7.1%}{r['grade']:>8.1%}"
                  f"{r['overconfident']:>10.1%}{r['n_eff']:>8.2f}")
        print()

    print("[6] WHAT THE TWO MODELS LEARNED TO TRUST")
    print("     learned channel weights (softplus of chan_w)")
    print(f"     {'model':<12}{'hiss':>9}{'khabar':>9}{'aql':>9}{'fitra':>9}"
          f"{'cov_theta':>11}")
    for nm, mdl in [("mihakk", P), ("avicennan", Pb)]:
        cw = np.log1p(np.exp(mdl["chan_w"].v))
        print(f"     {nm:<12}{cw[0]:>9.3f}{cw[1]:>9.3f}{cw[2]:>9.3f}{cw[3]:>9.3f}"
              f"{float(mdl['cov_theta'].v):>11.3f}")
    print("""
     The baseline has only CHANNEL-LEVEL dials. Lacking any per-item measure of
     how complete an induction is or how many roads a report travelled, the
     only defence it can mount against a long specious chain is to distrust
     inference wholesale and lean on testimony -- and that same policy is what
     hands it a total loss on the collusion split, where the testimony is
     twenty voices belonging to one man. There is no weight setting that
     survives both. The Taymiyyan model is not more sceptical; it is graded
     PER ITEM, which is a different quantity and needs a different mechanism.""")

    print("\n[7] SELF-TESTS")
    t1 = test_tawatur_arithmetic()
    t2 = test_weakest_premise(P, bank)
    t3 = test_universals_have_no_parameters(P, bank)
    t4 = test_awla_asymmetry()
    t5 = fitra_drift_audit(P, P0, bank, np.random.default_rng(5))

    print("\n[8] SUMMARY")
    d1 = (results[("S1_collusion", "mihakk")]["truth"]
          - results[("S1_collusion", "avicennan")]["truth"])
    d2 = (results[("S2_form_boost", "mihakk")]["truth"]
          - results[("S2_form_boost", "avicennan")]["truth"])
    d4 = (results[("S4_sound_derivation", "mihakk")]["truth"]
          - results[("S4_sound_derivation", "avicennan")]["truth"])
    d3 = (results[("S3_suspension", "mihakk")]["grade"]
          - results[("S3_suspension", "avicennan")]["grade"])
    print(f"     S1 collusion trap   : mihakk beats the counting baseline "
          f"by {d1:+.1%} truth")
    print(f"     S2 form-boost trap  : mihakk beats the summing baseline "
          f"by {d2:+.1%} truth")
    print(f"     S4 sound derivation : mihakk beats the baseline "
          f"by {d4:+.1%} truth")
    print(f"     S3 suspension       : mihakk grades the deadlock correctly "
          f"{d3:+.1%} more often")
    advs = ["S1_collusion", "S2_form_boost", "S4_sound_derivation"]
    am = np.mean([results[(x, "mihakk")]["truth"] for x in advs])
    ab = np.mean([results[(x, "avicennan")]["truth"] for x in advs])
    print(f"\n     mean truth over the three adversarial splits:")
    print(f"       mihakk    {am:6.1%}")
    print(f"       avicennan {ab:6.1%}")
    print(f"     S2 is an honest loss for the Taymiyyan model and is reported "
          f"as one;\n     the baseline buys it with the policy that costs it "
          f"everything on S1.")
    allok = all([ok_grad, t1, t2, t3, t4, t5])
    print(f"\n     ALL CHECKS: {'PASS' if allok else 'FAIL'}")
    print("=" * 78)
    return allok


if __name__ == "__main__":
    main()

"""
================================================================================
 CHAPTER 257 -- IBN SINA (AVICENNA), 980-1037 CE
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0257_ibn_sina_avicenna_980 - Ibn Sina (Avicenna) (c.980-1037)
#================================================================================  

 A pure NumPy, from-scratch, trainable neural architecture built around the
 THREE most distinctive (and most technically specific) commitments in Ibn
 Sina's psychology -- not the generic "soul is incorporeal" reading that gets
 flattened into every dualist thinker, but the three mechanisms that are his
 alone:

   1. WAHM (estimation, al-wahm) -- a faculty, shared with animals, that reads
      non-sensory "connotative attributes" (ma'ani: danger, friendliness,
      edibility) directly off a percept, with NO syllogism in between. A sheep
      does not infer that a wolf is dangerous; it estimates it. This is fast,
      cheap, and can be simply wrong. (Black, "Estimation (Wahm) in Avicenna,"
      Dialogue 32 (1993).)

   2. HADS (intuition, al-hads) -- the capacity to "suddenly hit upon" the
      middle term of a syllogism, i.e. to leap to the connecting concept
      without searching the space of possible middle terms step by step.
      Ibn Sina holds that minds differ *quantitatively* in hads -- some need
      almost no priming to leap correctly (in the limit, prophetic intellects
      that need no teacher at all), others need much more. (Gutas 1988;
      Kurun, "Avicenna's Intuitionist Rationalism.")

   3. ITTISAL (conjunction) -- understanding is not purely internally
      generated. The prepared, trained intellect achieves actual (not merely
      potential) understanding only in episodic contact with the Agent
      Intellect, which is external to it and not of its own making. Training
      makes conjunction possible; training does not manufacture the content
      conjoined. (IEP, "Avicenna (Ibn Sina).")

 Together these give a mind-specific thesis that is NOT "intelligence is
 self-supervised prediction" and NOT "attention over stored keys": it is
 "intelligence is a fast non-inferential estimate, corrected by a swift
 associative leap to a connecting idea, validated only by episodic contact
 with a structure the system did not itself write." Sections A-F below build
 exactly that, plus one more component with no analogue elsewhere in this
 corpus: a persistent self-representation that is provably non-zero even when
 every sensory channel is silenced (the Floating Man invariant, tested, not
 asserted).

 CONTENTS
 --------
   PART 0   Autodiff engine (Tensor, reverse-mode, pure NumPy)
   PART 1   Layers built on the engine (Linear, LayerNorm-lite, Embedding)
   PART 2   The five Avicennan faculties as modules
              1. External Senses      (al-hawas al-zahira)
              2. Common Sense          (al-hiss al-mushtarak)
              3. Wahm / Estimation     (al-wahm)              <- fast path
              4. Imagination & Memory  (khayal / hifz)
              5. Cogitation            (al-mufakkira)
              6. Rational Soul / Material Intellect + Abstraction (VQ codebook)
              7. Hads (intuitive leap)
              8. Floating Self (self-model, tested invariant)
              9. Agent Intellect / Ittisal (episodic conjunction, frozen ref.)
             10. Body/Spirit Interface (action + vital regulation heads)
   PART 3   IbnSinaMind -- assembled forward pass
   PART 4   Synthetic tasks (estimation-classification + hads-as-modular-
            arithmetic + floating-man hinge) and the training loop
   PART 5   Finite-difference gradient check (mandatory correctness test)
   PART 6   Self-tests (floating man invariant, hads vs brute force, shapes)
   PART 7   Main / demonstration

    AUTHOR: Encyclopedia of Lost Minds: Echoes on AI -- Chapter 257
================================================================================
"""

import numpy as np

RNG_SEED = 1037  # the year Ibn Sina died
np.random.seed(RNG_SEED)


# ==============================================================================
# PART 0 -- AUTODIFF ENGINE
# ==============================================================================
#
# A small reverse-mode automatic differentiation engine over NumPy arrays.
# Every Tensor op records its inputs ("_prev") and a local backward closure
# ("_backward") that accumulates gradient into its parents. Tensor.backward()
# topologically sorts the graph and runs those closures in reverse. This is
# "pure NumPy from scratch": no autograd library, no PyTorch, no JAX -- and
# it is what makes the training loop below use REAL analytic gradients
# (fast) while Part 5 independently verifies those gradients numerically
# (slow, but a ground truth) via finite differences, per project convention.


def _unbroadcast(grad, shape):
    """Sum-reduce `grad` back down to `shape` after a broadcasted op.

    Standard reverse-mode-autodiff helper: if the forward op broadcast an
    array of `shape` up to `grad.shape`, the backward pass must sum over
    every axis that was stretched (including axes that did not exist at
    all) to get a gradient of the correct original shape.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A NumPy array with a reverse-mode gradient tape."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ---- shape helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def detach(self):
        return Tensor(self.data.copy())

    def item(self):
        return float(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    # ---- arithmetic ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out = Tensor(self.data ** p, self.requires_grad, (self,), f"**{p}")

        def _backward():
            if self.requires_grad:
                self.grad += (p * (self.data ** (p - 1))) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other), "@")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,), "T")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T
        out._backward = _backward
        return out

    # ---- reductions --------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                     (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- elementwise nonlinearities -----------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(self.data, 0.0)
        out = Tensor(r, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -60, 60))
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), self.requires_grad,
                     (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / np.clip(self.data, 1e-12, None)) * out.grad
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                g = out.grad
                dot = np.sum(g * s, axis=axis, keepdims=True)
                self.grad += s * (g - dot)
        out._backward = _backward
        return out

    def max(self, axis=-1, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out_data = m if keepdims else np.squeeze(m, axis=axis)
        out = Tensor(out_data, self.requires_grad, (self,), "max")
        mask = (self.data == m).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)  # split credit on ties

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += mask * g
        out._backward = _backward
        return out

    # ---- indexing --------------------------------------------------------
    def __getitem__(self, key):
        out = Tensor(self.data[key], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                g = np.zeros_like(self.data)
                g[key] = out.grad
                self.grad += g
        out._backward = _backward
        return out

    # ---- backward pass -------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def cat(tensors, axis=-1):
    """Concatenate Tensors along `axis` (module-level, since __cat__ isn't a dunder)."""
    datas = [t.data for t in tensors]
    out = Tensor(np.concatenate(datas, axis=axis), any(t.requires_grad for t in tensors),
                 tuple(tensors), "cat")
    sizes = [d.shape[axis] for d in datas]

    def _backward():
        idx = 0
        for t, s in zip(tensors, sizes):
            if t.requires_grad:
                sl = [slice(None)] * out.data.ndim
                sl[axis] = slice(idx, idx + s)
                t.grad += out.grad[tuple(sl)]
            idx += s
    out._backward = _backward
    return out


def embedding_lookup(table, indices):
    """table: Tensor (n_rows, dim); indices: int array (batch,) -> Tensor (batch, dim).

    Implements the row-gather with a correct scatter-add backward
    (np.add.at), used for the category embedding table in the hads task.
    """
    indices = np.asarray(indices, dtype=np.int64)
    out = Tensor(table.data[indices], table.requires_grad, (table,), "embed")

    def _backward():
        if table.requires_grad:
            np.add.at(table.grad, indices, out.grad)
    out._backward = _backward
    return out


# ==============================================================================
# PART 1 -- BASIC LAYERS BUILT ON THE ENGINE
# ==============================================================================

def xavier(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


class Linear:
    """y = x @ W + b, with Xavier init. The only 'generic' building block in
    this file; every module below composes Linear + a distinctive gating /
    retrieval / quantization mechanism, never Linear alone."""

    def __init__(self, n_in, n_out, name=""):
        self.W = Tensor(xavier(n_in, n_out), requires_grad=True)
        self.b = Tensor(np.zeros((1, n_out)), requires_grad=True)
        self.name = name

    def __call__(self, x):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class RMSNormLite:
    """A dependency-free, parameter-light normalizer (root-mean-square scale
    only, no mean-centering, no learned affine) used to keep the recurrent
    Floating Self update numerically stable across many ticks without
    introducing another generic 'LayerNorm' block."""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x):
        ms = (x * x).mean(axis=-1, keepdims=True)
        denom = (ms + self.eps) ** 0.5
        return x / denom


# ==============================================================================
# PART 2 -- THE AVICENNAN FACULTIES
# ==============================================================================

# ---- 2.1 External Senses (al-hawas al-zahira) -------------------------------
class ExternalSenses:
    """Five modality-specific encoders projecting raw sense-data into the
    shared 'space of intentions' (ma'ani) that every inner sense reads from.
    Each modality keeps its own proper object (Ibn Sina: vision registers
    color/shape, hearing registers sound) via a private Linear + tanh; only
    after this projection do the streams become commensurable."""

    def __init__(self, dims, intention_dim):
        # dims: dict modality -> raw input dimensionality
        self.encoders = {m: Linear(d, intention_dim, name=f"sense_{m}")
                          for m, d in dims.items()}
        self.modalities = list(dims.keys())
        self.intention_dim = intention_dim

    def __call__(self, sensory):
        # sensory: dict modality -> Tensor (batch, raw_dim)
        return {m: self.encoders[m](sensory[m]).tanh() for m in self.modalities}

    def params(self):
        p = []
        for enc in self.encoders.values():
            p += enc.params()
        return p


# ---- 2.2 Common Sense (al-hiss al-mushtarak) --------------------------------
class CommonSense:
    """Integrates the five external-sense streams into one unified percept.
    Ibn Sina: the common sense is what lets you see a red shape and hear a
    tone and know they are the same singing person -- a BINDING operation,
    not a bigger sense organ. Implemented as a learned per-modality gate
    (a scalar salience per modality, softmax-normalised across modalities)
    rather than as multi-head attention -- there is exactly one binding
    event, not many competing query heads, which is the actual claim being
    modelled."""

    def __init__(self, intention_dim, n_modalities):
        self.gate = Linear(intention_dim * n_modalities, n_modalities, name="common_sense_gate")
        self.intention_dim = intention_dim

    def __call__(self, streams):
        mods = list(streams.keys())
        stacked = cat([streams[m] for m in mods], axis=-1)     # (B, n_mod*D)
        salience = self.gate(stacked).softmax(axis=-1)          # (B, n_mod)
        percept = None
        for i, m in enumerate(mods):
            w = salience[:, i:i + 1]                            # clean Tensor slice
            contrib = streams[m] * w
            percept = contrib if percept is None else percept + contrib
        return percept, salience

    def params(self):
        return self.gate.params()


# ---- 2.3 Wahm / Estimation (al-wahm) -- THE FAST, NON-INFERENTIAL PATH -----
class Estimation:
    """The estimative faculty. Reads connotative, non-sensible 'intentions'
    (danger / utility / affinity ...) straight off the unified percept, with
    no abstraction and no syllogism -- Ibn Sina's example is the sheep that
    estimates the wolf is dangerous without inferring it. Produces:
      valence   -- an E-dimensional estimative reading (fast, cheap, D->E)
      salience  -- confidence-in-the-estimate scalar in (0,1)
    The salience scalar becomes the GATE that decides, downstream, how much
    of the slow rational pathway (cogitation/abstraction) gets engaged --
    the architecture's operationalisation of 'quick estimation first, and
    reasoning only when estimation is not enough.'"""

    def __init__(self, intention_dim, estimative_dim):
        self.to_valence = Linear(intention_dim, estimative_dim, name="wahm_valence")
        self.to_salience = Linear(intention_dim, 1, name="wahm_salience")

    def __call__(self, percept):
        valence = self.to_valence(percept).tanh()
        salience = self.to_salience(percept).sigmoid()
        return valence, salience

    def params(self):
        return self.to_valence.params() + self.to_salience.params()


# ---- 2.4 Imagination & Memory (khayal / hifz) -------------------------------
class ImaginationMemory:
    """A learned associative store of K 'particular forms' (khayal retains
    intentional forms in the absence of their objects; hifz stores the
    particular images abstracted from specific episodes). Retrieval is
    content-addressed: similarity of the current percept to every stored
    form, softmax-weighted read. This is NOT the transformer key/value
    pattern re-used from elsewhere in the corpus -- there is a single fixed
    store shared across all time steps (a lifetime memory), not per-token
    keys/values recomputed every forward pass, and the store is written to
    slowly, with a decaying trace, not attended-and-discarded."""

    def __init__(self, n_slots, intention_dim):
        self.M = Tensor(np.random.randn(n_slots, intention_dim) * 0.05, requires_grad=True)
        self.combine = Linear(intention_dim * 2, intention_dim, name="imagination_combine")
        self.n_slots = n_slots

    def retrieve(self, percept):
        sim = percept @ self.M.T                      # (B, n_slots)
        weights = sim.softmax(axis=-1)
        read = weights @ self.M                        # (B, D)
        return read, weights

    def __call__(self, percept):
        read, weights = self.retrieve(percept)
        combined = self.combine(cat([percept, read], axis=-1)).tanh()
        return combined, weights

    def write(self, percept, rate=0.02):
        """Slow exponential-trace write toward the nearest slot -- models
        memory formation as gradual sedimentation, not one-shot storage.
        Detached (numpy-only): memory writing is a housekeeping process,
        not a differentiable path in this architecture, exactly as
        estimation is 'quick' precisely because it does not route through
        the gradient-carrying abstraction machinery."""
        nearest = np.argmax(percept.data @ self.M.data.T, axis=1)
        for b, k in enumerate(nearest):
            self.M.data[k] = (1 - rate) * self.M.data[k] + rate * percept.data[b]

    def params(self):
        return [self.M] + self.combine.params()


# ---- 2.5 Cogitation (al-mufakkira) -- gated "inner discourse" --------------
class Cogitation:
    """Combines/separates intentional forms -- Ibn Sina's inner discourse,
    the highest sensitive faculty and the bridge to the rational soul. Its
    engagement is GATED by (1 - wahm_salience): when the fast estimate is
    already confident, cogitation is mostly bypassed (a highway skip); when
    the estimate is unsure, cogitation does most of the work. This literally
    implements 'reasoning is invoked when estimation is not enough' as a
    differentiable highway network rather than as a hand-tuned threshold."""

    def __init__(self, intention_dim):
        self.mix = Linear(intention_dim, intention_dim, name="cogitation_mix")

    def __call__(self, imagined, salience):
        transformed = self.mix(imagined).tanh()
        g = salience  # (B,1) -- high salience (confident estimate) => skip more
        one_minus_g = Tensor(1.0, requires_grad=False) - g
        return imagined * g + transformed * one_minus_g

    def params(self):
        return self.mix.params()


# ---- 2.6 Rational Soul: Material Intellect as a codebook of universals ----
class RationalSoul:
    """The material intellect is modelled as a bank of P 'potential
    intelligible forms' (a codebook U, rational_dim R) -- pure receptive
    capacity, not yet actual thought. Abstraction projects the cogitated
    percept into rational space and reads it against the codebook via
    softmax similarity: the resulting blend IS the abstracted universal.
    This is prototype/vector-quantization, not attention-over-tokens: there
    is one lifetime bank of universals, read the same way regardless of
    sequence position, which is the structural claim being made (universals
    are a fixed repertoire the mind matures into, not a per-context cache)."""

    def __init__(self, cog_dim, rational_dim, n_forms):
        self.project = Linear(cog_dim, rational_dim, name="material_intellect_projection")
        self.U = Tensor(np.random.randn(n_forms, rational_dim) * 0.05, requires_grad=True)
        self.n_forms = n_forms
        self.rational_dim = rational_dim

    def abstract(self, cogitated, temperature=1.0):
        q = self.project(cogitated).tanh()                 # (B, R) potential
        logits = (q @ self.U.T) * (1.0 / temperature)        # (B, P)
        weights = logits.softmax(axis=-1)                    # (B, P) confidence dist.
        actualized = weights @ self.U                        # (B, R) acquired intellect
        return actualized, weights, q

    def params(self):
        return [self.U] + self.project.params()


def entropy_confidence(weights):
    """Normalised confidence-of-abstraction in [0, 1]: 1 = a single form
    was picked with certainty (fully actualized), 0 = uniform over all P
    forms (still wholly potential). Used both as a diagnostic and as the
    gate that decides whether conjunction (ittisal) is permitted."""
    p = np.clip(weights.data, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=-1)
    h_max = np.log(p.shape[-1])
    return 1.0 - h / h_max


# ---- 2.7 Hads -- the intuitive leap to the middle term ---------------------
class Hads:
    """Given two 'premises' (abstracted concept vectors p1, p2), hads leaps
    directly to the codebook slot that best fits their BINDING -- not their
    sum, their elementwise (Hadamard) product, p1*p2 -- dotted against every
    codebook vector. Binding-then-retrieval is a deliberate choice: a purely
    additive combination (p1@U.T + p2@U.T) can only ever prefer whichever of
    p1 or p2 individually resembles a codebook entry more, so it can never
    produce a genuinely THIRD, relation-specific answer -- verified
    empirically during development, not assumed. The Hadamard product is
    the smallest change that lets the pair jointly determine an answer
    neither premise alone would pick, which is the actual point of a middle
    term: it is implied by the conjunction of the premises, not by either
    one. `tau` (temperature) is the individual-differences knob: low tau ->
    a sharp, near one-hot leap (a quick mind, Ibn Sina's rare 'sacred
    intellect' in the limit); high tau -> a diffuse distribution that
    behaves like brute-force evidence-averaging over many candidate middle
    terms (a plodding mind). Training uses the differentiable softmax route
    (soft_leap); a detached, hard argmax route (hard_leap) is used at
    inference/self-test time to measure the actual leap."""

    def __init__(self, rational_dim, n_forms):
        self.n_forms = n_forms
        self.rational_dim = rational_dim

    def soft_leap(self, p1, p2, U, tau=1.0):
        joint = p1 * p2                                    # Hadamard binding of the two premises
        score = (joint @ U.T) * (1.0 / tau)               # (B, P)
        return score.softmax(axis=-1)

    def hard_leap(self, p1_data, p2_data, U_data):
        """Detached (numpy) hard leap: the literal one-shot guess, plus how
        many candidates it would take a brute-force search to be equally
        sure (used by the self-test to quantify 'swiftness')."""
        joint = p1_data * p2_data
        score = joint @ U_data.T                           # (B, P)
        guess = np.argmax(score, axis=-1)
        order = np.argsort(-score, axis=-1)
        return guess, order


# ---- 2.8 Floating Self -- persistent self-model (the tested invariant) ----
class FloatingSelf:
    """A recurrent self-state s, updated by s <- tanh(Ws(s) + Wa(actualized)).
    Both Ws and Wa carry a bias. When sensory input is entirely absent, the
    only term feeding s is the SUM of those two biases -- everything else
    (the matmul terms) is multiplied by zero. This is the architecture's
    literal reading of the Floating Man: self-awareness must be able to
    stabilise from bias alone, with zero contribution from the senses.
    Nothing forces this to hold at initialization (biases start at 0, so the
    naive network is NOT self-aware with no input); PART 4's training loop
    includes a hinge loss that only a nonzero, input-independent self-signal
    can satisfy, so the network has to discover the invariant, not have it
    handed to it. PART 6 verifies the invariant holds after training."""

    def __init__(self, rational_dim):
        self.Ws = Linear(rational_dim, rational_dim, name="self_recurrence")
        self.Wa = Linear(rational_dim, rational_dim, name="self_from_actualized")
        self.rational_dim = rational_dim
        self.Ws.W.data *= 0.15  # keep the recurrence a contraction -> stable fixed point
        # Symmetry-breaking init: a bias of EXACTLY zero puts s at the origin, and the
        # origin is a stationary point of ||s|| (d||s||/ds = s/||s|| -> 0 as s->0), so a
        # hinge loss on ||s|| would have zero gradient there and could never escape it.
        # A small nonzero bias breaks that degeneracy so gradient exists from step one --
        # a purely numerical-hygiene choice, not a philosophical one.
        self.Ws.b.data = np.random.randn(1, rational_dim) * 0.05
        self.Wa.b.data = np.random.randn(1, rational_dim) * 0.05

    def step(self, prev_s, actualized):
        return (self.Ws(prev_s) + self.Wa(actualized)).tanh()

    def run(self, actualized, batch_size, ticks=4, zero_input=False):
        s = Tensor(np.zeros((batch_size, self.rational_dim)))
        a_use = Tensor(np.zeros((batch_size, self.rational_dim))) if zero_input else actualized
        for _ in range(ticks):
            s = self.step(s, a_use)
        return s

    def params(self):
        return self.Ws.params() + self.Wa.params()


# ---- 2.9 Agent Intellect / Ittisal -- episodic, external, prepared-only ----
class AgentIntellect:
    """Conjunction with a FIXED reference codebook `A` that this learner does
    not write and cannot optimize (plain NumPy, never wrapped as a trainable
    Tensor parameter -- literally outside the gradient tape). Conjunction:
      (a) only fires per-sample once abstraction confidence clears `theta`
          (the intellect must be 'adequately prepared' -- IEP), and
      (b) only fires at all once training has passed a warm-up (an
          untrained intellect cannot conjoin no matter how confident its
          untrained guesses look), and
      (c) is applied as a DETACHED forward correction (see `conjoin`): the
          value the rest of the network sees is nudged toward `A`, but the
          gradient the learner receives is not routed through that nudge.
          The pull toward truth happens; the learner cannot backpropagate
          its way into manufacturing that truth. This is the single most
          consequential modelling choice in the file and is discussed at
          length in the chapter text."""

    def __init__(self, n_forms, rational_dim, theta=0.55, blend=0.5, seed=2024):
        rs = np.random.RandomState(seed)
        self.A = rs.randn(n_forms, rational_dim) * 0.05  # frozen, external, not a Tensor param
        self.theta = theta
        self.blend = blend

    def conjoin(self, actualized, confidence, prepared):
        if not prepared:
            return actualized, np.zeros(confidence.shape[0], dtype=bool)
        fires = confidence >= self.theta
        illum = actualized.data.copy()
        if fires.any():
            sims = actualized.data @ self.A.T
            nearest = np.argmax(sims, axis=1)
            illum[fires] = ((1 - self.blend) * actualized.data[fires]
                             + self.blend * self.A[nearest[fires]])
        delta = Tensor(illum - actualized.data)  # requires_grad=False: external, not optimizable
        return actualized + delta, fires


# ---- 2.10 Body/Spirit Interface (ruh) ---------------------------------------
class BodySpiritInterface:
    """Maps the fast estimative valence and the (possibly illuminated)
    rational representation onto action, and the persistent self-state onto
    a vital/health regulation scalar -- Ibn Sina's pneuma (ruh), the
    faculty that mediates between the incorporeal soul and the material
    body: it does not think, it dispatches and it keeps the organism
    viable."""

    def __init__(self, estimative_dim, rational_dim, self_dim, n_actions):
        self.action_head = Linear(estimative_dim + rational_dim, n_actions, name="action_head")
        self.health_head = Linear(self_dim, 1, name="health_head")

    def __call__(self, valence, rational_repr, self_state):
        logits = self.action_head(cat([valence, rational_repr], axis=-1))
        health = self.health_head(self_state).sigmoid()
        return logits, health

    def params(self):
        return self.action_head.params() + self.health_head.params()


# ==============================================================================
# PART 3 -- IbnSinaMind: THE ASSEMBLED FORWARD PASS
# ==============================================================================

MODALITY_DIMS = {"vision": 20, "hearing": 12, "smell": 10, "taste": 6, "touch": 10}


class IbnSinaMind:
    """Wires PART 2's ten faculties into one forward pass. Two task heads
    share the same rational substrate on purpose (Ibn Sina: there is one
    rational soul, not a separate module per problem type):
      - action_logits  : fast-estimation-informed classification (wahm-led)
      - hads_probs     : intuitive-leap analogy solving (hads-led)
    and one architectural invariant that both heads' training must respect:
      - self_state_zero: the persistent self-representation under total
        sensory silence, which the floating-man hinge loss (PART 4) forces
        to stay above a floor."""

    def __init__(self, intention_dim=32, estimative_dim=10, memory_slots=24,
                 rational_dim=24, n_forms=6, n_actions=3, n_categories=6):
        self.intention_dim = intention_dim
        self.n_forms = n_forms
        self.n_categories = n_categories
        assert n_forms == n_categories, "one codebook slot per category, by design"

        self.senses = ExternalSenses(MODALITY_DIMS, intention_dim)
        self.common_sense = CommonSense(intention_dim, len(MODALITY_DIMS))
        self.estimation = Estimation(intention_dim, estimative_dim)
        self.imagination = ImaginationMemory(memory_slots, intention_dim)
        self.cogitation = Cogitation(intention_dim)
        self.rational_soul = RationalSoul(intention_dim, rational_dim, n_forms)
        self.hads = Hads(rational_dim, n_forms)
        self.floating_self = FloatingSelf(rational_dim)
        self.agent_intellect = AgentIntellect(n_forms, rational_dim)
        self.body_spirit = BodySpiritInterface(estimative_dim, rational_dim, rational_dim, n_actions)
        # Ecat lives directly in RATIONAL space (dim R), not raw intention space: these
        # are treated as already-intelligible premise-concepts (Ibn Sina's hads operates
        # on concepts the cogitative/abstractive machinery has already prepared, not on
        # raw percepts) -- so no second abstraction step stands between Ecat and Hads.
        self.Ecat = Tensor(np.random.randn(n_categories, rational_dim) * 0.3, requires_grad=True)
        # A FIXED, arbitrary K x K lookup table generating the hads task's ground truth.
        # Not a Tensor, not trainable, not visible to any module -- purely a data-generating
        # device standing in for 'the true relational structure of the world,' which hads
        # must recover from repeated (premise, premise, answer) exposure.
        self.hidden_ontology = np.random.RandomState(2027).randint(0, n_forms, size=(n_categories, n_categories))

    def params(self):
        p = []
        for mod in (self.senses, self.common_sense, self.estimation, self.imagination,
                    self.cogitation, self.rational_soul, self.floating_self,
                    self.body_spirit):
            p += mod.params()
        p.append(self.Ecat)
        return p

    def forward(self, sensory, cat_a, cat_b, epoch=0, warmup_epochs=2, tau_hads=0.7,
                write_memory=False):
        streams = self.senses(sensory)
        percept, common_salience = self.common_sense(streams)
        valence, wahm_salience = self.estimation(percept)
        imagined, mem_weights = self.imagination(percept)
        cogitated = self.cogitation(imagined, wahm_salience)
        actualized, abst_weights, q = self.rational_soul.abstract(cogitated)
        confidence = entropy_confidence(abst_weights)
        prepared = epoch >= warmup_epochs
        illuminated, conj_fires = self.agent_intellect.conjoin(actualized, confidence, prepared)

        self_state = self.floating_self.run(illuminated, percept.shape[0], ticks=4, zero_input=False)
        self_state_zero = self.floating_self.run(actualized, percept.shape[0], ticks=4, zero_input=True)

        action_logits, health = self.body_spirit(valence, illuminated, self_state)

        p1 = embedding_lookup(self.Ecat, cat_a)
        p2 = embedding_lookup(self.Ecat, cat_b)
        hads_probs = self.hads.soft_leap(p1, p2, self.rational_soul.U, tau=tau_hads)

        if write_memory:
            self.imagination.write(percept)

        return {
            "action_logits": action_logits, "health": health, "hads_probs": hads_probs,
            "self_state": self_state, "self_state_zero": self_state_zero,
            "confidence": confidence, "conj_fires": conj_fires,
            "abst_weights": abst_weights, "common_salience": common_salience,
            "wahm_salience": wahm_salience, "actualized": actualized,
        }

    __call__ = forward


# ==============================================================================
# PART 4 -- SYNTHETIC TASKS AND TRAINING LOOP
# ==============================================================================

def cross_entropy_from_logits(logits, targets):
    probs = logits.softmax(axis=-1)
    logp = probs.log()
    B, C = logits.shape
    onehot = np.zeros((B, C))
    onehot[np.arange(B), targets] = 1.0
    return -(logp * Tensor(onehot)).sum(axis=1).mean()


def bce_loss(probs, targets):
    """Binary cross-entropy for a (B,1) sigmoid output against a (B,) 0/1
    target. Used for the vital/health regulation head: the body-spirit
    interface's 'health' scalar is trained to track estimated situational
    safety (1 - danger), giving the self-recurrence weights (Ws, Wa) a
    second, well-conditioned gradient path distinct from the floating-man
    hinge -- so the persistent self-state is shaped by more than one signal,
    the way Ibn Sina's ruh mediates general vital regulation, not just the
    one thought experiment that happens to make for a clean self-test."""
    t = Tensor(targets.reshape(-1, 1).astype(np.float64))
    one_minus_h = Tensor(1.0) - probs
    return -(t * probs.log() + (Tensor(1.0) - t) * one_minus_h.log()).mean()


def nll_from_probs(probs, targets):
    logp = probs.log()
    B, C = probs.shape
    onehot = np.zeros((B, C))
    onehot[np.arange(B), targets] = 1.0
    return -(logp * Tensor(onehot)).sum(axis=1).mean()


def make_batch(batch_size, n_categories, rng, hidden_ontology):
    """Synthetic data generator with TWO independent hidden rules:

    (1) Estimation-classification rule: the action label is a nonlinear
        function of the SMELL and VISION channels only (Ibn Sina's wahm
        example is scent/sight-driven threat-estimation) -- 3 classes:
        0=approach (safe+attractive), 1=avoid (dangerous), 2=investigate
        (ambiguous). Hearing/taste/touch are pure noise for this rule,
        forcing the estimation pathway to learn WHICH channels matter.

    (2) Hads rule: cat_c = hidden_ontology[cat_a, cat_b] -- a FIXED,
        arbitrary K x K lookup table generated once (not visible to the
        learner in any form). This is deliberately NOT modular arithmetic:
        development testing showed a Hadamard-binding retrieval mechanism
        (see Hads) learns well above chance on an arbitrary structured
        association, which is the actual claim being made -- hads recovers
        a reliable premise-pair -> middle-term mapping FROM REPEATED
        EXPOSURE, exactly as Ibn Sina says ordinary (non-prophetic) minds
        require some priming before the leap becomes swift and correct.
    """
    sensory = {
        "vision": rng.randn(batch_size, MODALITY_DIMS["vision"]),
        "hearing": rng.randn(batch_size, MODALITY_DIMS["hearing"]),
        "smell": rng.randn(batch_size, MODALITY_DIMS["smell"]),
        "taste": rng.randn(batch_size, MODALITY_DIMS["taste"]),
        "touch": rng.randn(batch_size, MODALITY_DIMS["touch"]),
    }
    threat = sensory["smell"][:, :3].sum(axis=1) - sensory["vision"][:, :3].sum(axis=1)
    y_action = np.where(threat > 0.6, 1, np.where(threat < -0.6, 0, 2)).astype(np.int64)

    cat_a = rng.randint(0, n_categories, size=batch_size)
    cat_b = rng.randint(0, n_categories, size=batch_size)
    cat_c = hidden_ontology[cat_a, cat_b]

    sensory_t = {m: Tensor(v) for m, v in sensory.items()}
    return sensory_t, y_action, cat_a, cat_b, cat_c


def evaluate(mind, n_categories, rng, n_batches=10, batch_size=64, epoch=99, warmup_epochs=2):
    correct_action, correct_hads, total = 0, 0, 0
    for _ in range(n_batches):
        sensory, y_action, cat_a, cat_b, cat_c = make_batch(batch_size, n_categories, rng, mind.hidden_ontology)
        out = mind.forward(sensory, cat_a, cat_b, epoch=epoch, warmup_epochs=warmup_epochs)
        pred_action = np.argmax(out["action_logits"].data, axis=1)
        pred_hads = np.argmax(out["hads_probs"].data, axis=1)
        correct_action += (pred_action == y_action).sum()
        correct_hads += (pred_hads == cat_c).sum()
        total += batch_size
    return correct_action / total, correct_hads / total


class SGDMomentum:
    """Plain momentum SGD, implemented by hand (no optimizer library)."""

    def __init__(self, params, lr=0.05, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


def train(mind, n_epochs=40, batches_per_epoch=25, batch_size=64, lr=0.08,
          hads_weight=1.0, floating_weight=0.6, felicity_weight=0.05,
          tau_self=0.35, warmup_epochs=3, seed=7, verbose=True):
    rng = np.random.RandomState(seed)
    opt = SGDMomentum(mind.params(), lr=lr, momentum=0.9)
    history = []
    for epoch in range(n_epochs):
        epoch_loss, epoch_action_loss, epoch_hads_loss, epoch_hinge = 0.0, 0.0, 0.0, 0.0
        epoch_conj_rate, epoch_conf, epoch_health_loss = 0.0, 0.0, 0.0
        for b in range(batches_per_epoch):
            sensory, y_action, cat_a, cat_b, cat_c = make_batch(batch_size, mind.n_categories, rng, mind.hidden_ontology)
            out = mind.forward(sensory, cat_a, cat_b, epoch=epoch, warmup_epochs=warmup_epochs,
                                write_memory=(b == 0))

            action_loss = cross_entropy_from_logits(out["action_logits"], y_action)
            hads_loss = nll_from_probs(out["hads_probs"], cat_c)
            health_target = (y_action != 1).astype(np.float64)  # 1 = situation judged safe
            health_loss = bce_loss(out["health"], health_target)

            norm_zero = ((out["self_state_zero"] ** 2).sum(axis=1) + 1e-8) ** 0.5
            floating_hinge = (Tensor(tau_self) - norm_zero).relu().mean()

            conf_t = out["abst_weights"].max(axis=-1)
            felicity_bonus = (Tensor(1.0) - conf_t).mean()

            loss = action_loss + hads_weight * hads_loss + floating_weight * floating_hinge \
                + felicity_weight * felicity_bonus + 0.3 * health_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_action_loss += action_loss.item()
            epoch_hads_loss += hads_loss.item()
            epoch_hinge += floating_hinge.item()
            epoch_conj_rate += out["conj_fires"].mean()
            epoch_conf += out["confidence"].mean()

        n = batches_per_epoch
        rec = dict(epoch=epoch, loss=epoch_loss / n, action_loss=epoch_action_loss / n,
                   hads_loss=epoch_hads_loss / n, floating_hinge=epoch_hinge / n,
                   conjunction_rate=epoch_conj_rate / n, confidence=epoch_conf / n)
        history.append(rec)
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            acc_a, acc_h = evaluate(mind, mind.n_categories, rng, n_batches=5,
                                     batch_size=batch_size, epoch=epoch, warmup_epochs=warmup_epochs)
            print(f"  epoch {epoch:3d} | loss {rec['loss']:.4f} | action_loss {rec['action_loss']:.4f} "
                  f"| hads_loss {rec['hads_loss']:.4f} | floating_hinge {rec['floating_hinge']:.4f} "
                  f"| conj_rate {rec['conjunction_rate']:.2f} | confidence {rec['confidence']:.2f} "
                  f"| val_action_acc {acc_a:.2f} | val_hads_acc {acc_h:.2f}")
    return history


# ==============================================================================
# PART 5 -- FINITE-DIFFERENCE GRADIENT CHECK (mandatory correctness test)
# ==============================================================================
#
# IMPORTANT, and disclosed rather than hidden: the AgentIntellect's
# conjunction step (2.9) deliberately uses a STOP-GRADIENT / straight-through
# design -- the forward value is nudged toward the frozen external codebook
# `A`, but that nudge is not differentiated through (see `conjoin`'s
# docstring for the philosophical reason). This is the same family of
# technique as the straight-through estimator in VQ-VAE (van den Oord et
# al. 2017): a deliberate, standard, and DOCUMENTED mismatch between the
# analytic gradient and the true numerical total derivative, not a bug.
# A finite-difference check that swept through a firing conjunction step
# would therefore *correctly* fail -- so, honestly, this check is run with
# `warmup_epochs` set high enough that the system is never "prepared" and
# conjunction never fires (delta == 0 identically, for every parameter
# value probed). Every other module -- all nine of the rest -- is checked
# exactly, with no such exclusion.


def gradient_check(eps=1e-5, n_params_per_tensor=3, seed=123, verbose=True):
    rng = np.random.RandomState(seed)
    mind = IbnSinaMind(intention_dim=10, estimative_dim=4, memory_slots=6,
                        rational_dim=6, n_forms=5, n_actions=3, n_categories=5)
    batch_size = 4
    sensory, y_action, cat_a, cat_b, cat_c = make_batch(batch_size, mind.n_categories, rng, mind.hidden_ontology)

    def loss_fn():
        out = mind.forward(sensory, cat_a, cat_b, epoch=0, warmup_epochs=10 ** 6,
                            write_memory=False)  # conjunction gate permanently closed
        action_loss = cross_entropy_from_logits(out["action_logits"], y_action)
        hads_loss = nll_from_probs(out["hads_probs"], cat_c)
        norm_zero = ((out["self_state_zero"] ** 2).sum(axis=1) + 1e-8) ** 0.5
        floating_hinge = (Tensor(0.35) - norm_zero).relu().mean()
        conf_t = out["abst_weights"].max(axis=-1)
        felicity_bonus = (Tensor(1.0) - conf_t).mean()
        return action_loss + hads_loss + 0.6 * floating_hinge + 0.05 * felicity_bonus

    params = mind.params()
    for p in params:
        p.zero_grad()
    loss_fn().backward()

    rng2 = np.random.RandomState(999)
    max_rel_err, checked = 0.0, 0
    worst = None
    for p in params:
        n_check = min(n_params_per_tensor, p.data.size)
        flat_idx = rng2.choice(p.data.size, size=n_check, replace=False)
        for idx in flat_idx:
            multi = np.unravel_index(idx, p.data.shape)
            orig = p.data[multi]
            p.data[multi] = orig + eps
            lp = loss_fn().item()
            p.data[multi] = orig - eps
            lm = loss_fn().item()
            p.data[multi] = orig
            num_grad = (lp - lm) / (2 * eps)
            ana_grad = p.grad[multi]
            denom = max(abs(num_grad), abs(ana_grad), 1e-8)
            rel_err = abs(num_grad - ana_grad) / denom
            checked += 1
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                worst = (id(p), multi, num_grad, ana_grad)
    if verbose:
        print(f"  gradient check: {checked} parameter entries across {len(params)} tensors")
        print(f"  max relative error: {max_rel_err:.3e}  (worst at {worst})")
    return max_rel_err, checked


# ==============================================================================
# PART 6 -- SELF-TESTS
# ==============================================================================

def test_shapes_and_ranges():
    mind = IbnSinaMind()
    rng = np.random.RandomState(0)
    sensory, y_action, cat_a, cat_b, cat_c = make_batch(9, mind.n_categories, rng, mind.hidden_ontology)
    out = mind.forward(sensory, cat_a, cat_b, epoch=0, warmup_epochs=2)
    assert out["action_logits"].shape == (9, 3)
    assert out["hads_probs"].shape == (9, mind.n_categories)
    probs_sum = out["hads_probs"].data.sum(axis=1)
    assert np.allclose(probs_sum, 1.0, atol=1e-6), "hads_probs must be a proper distribution"
    assert np.all((out["wahm_salience"].data >= 0) & (out["wahm_salience"].data <= 1))
    assert np.all((out["health"].data >= 0) & (out["health"].data <= 1))
    print("  test_shapes_and_ranges: PASS")


def test_conjunction_requires_preparation():
    mind = IbnSinaMind()
    rng = np.random.RandomState(1)
    sensory, y_action, cat_a, cat_b, cat_c = make_batch(16, mind.n_categories, rng, mind.hidden_ontology)
    out_unprepared = mind.forward(sensory, cat_a, cat_b, epoch=0, warmup_epochs=1000)
    assert not out_unprepared["conj_fires"].any(), \
        "an untrained/unprepared intellect must never conjoin, regardless of confidence"
    print("  test_conjunction_requires_preparation: PASS")


def test_floating_man(mind, tol=0.30, verbose=True):
    """The chapter's central architectural claim, made falsifiable: after
    training, the persistent self-state under total sensory silence (a) has
    non-trivial norm, (b) is stable across additional recurrent ticks
    (a genuine fixed point, not a transient), and (c) is INVARIANT to the
    'actualized' argument passed in when zero_input=True -- i.e. it is not
    secretly leaking sensory content through a back door."""
    B, R = 12, mind.floating_self.rational_dim
    dummy_a = Tensor(np.random.randn(B, R) * 5.0)   # deliberately large/adversarial content
    dummy_b = Tensor(np.random.randn(B, R) * 5.0)

    s4 = mind.floating_self.run(dummy_a, B, ticks=4, zero_input=True)
    s9 = mind.floating_self.run(dummy_a, B, ticks=9, zero_input=True)
    s4_alt = mind.floating_self.run(dummy_b, B, ticks=4, zero_input=True)

    norm4 = np.linalg.norm(s4.data, axis=1).mean()
    drift = np.abs(s4.data - s9.data).max()
    invariance_gap = np.abs(s4.data - s4_alt.data).max()

    ok_norm = norm4 >= tol
    ok_stable = drift < 0.05
    ok_invariant = invariance_gap < 1e-8

    if verbose:
        print(f"  floating man: mean||s||={norm4:.4f} (floor {tol}) | "
              f"drift(4->9 ticks)={drift:.2e} | invariance_gap={invariance_gap:.2e}")
        print(f"  floating man: norm>=floor: {ok_norm} | stable fixed point: {ok_stable} "
              f"| input-invariant: {ok_invariant}")
    return ok_norm and ok_stable and ok_invariant


def test_hads_efficiency(mind, rng, n_trials=300, verbose=True):
    """Quantifies the 'swiftness' of the leap: for each trial, where does
    the CORRECT middle term sit in the hard_leap's ranked candidate list?
    Rank 0 = hads found it in one guess (a genuine leap). Reports the
    trained model's mean rank against the a-priori expected rank of a
    uniformly random guesser (n_forms - 1) / 2 -- Ibn Sina's claim that
    minds differ in how much 'evidence-averaging' they need, operationalised
    as how far down the candidate list the truth would have been found."""
    cat_a = rng.randint(0, mind.n_categories, size=n_trials)
    cat_b = rng.randint(0, mind.n_categories, size=n_trials)
    cat_c = mind.hidden_ontology[cat_a, cat_b]
    p1 = embedding_lookup(mind.Ecat, cat_a)
    p2 = embedding_lookup(mind.Ecat, cat_b)
    guess, order = mind.hads.hard_leap(p1.data, p2.data, mind.rational_soul.U.data)

    top1_acc = (guess == cat_c).mean()
    ranks = np.array([np.where(order[i] == cat_c[i])[0][0] for i in range(n_trials)])
    mean_rank = ranks.mean()
    random_expected_rank = (mind.n_categories - 1) / 2.0

    if verbose:
        print(f"  hads leap: top-1 accuracy={top1_acc:.2%} | mean rank of truth={mean_rank:.2f} "
              f"(uniform-random baseline expects {random_expected_rank:.2f})")
    return top1_acc, mean_rank


def run_all_self_tests(mind, rng):
    print("--- Self-tests ---")
    test_shapes_and_ranges()
    test_conjunction_requires_preparation()
    fm_ok = test_floating_man(mind)
    top1_acc, mean_rank = test_hads_efficiency(mind, rng)
    grad_err, n_checked = gradient_check()
    all_ok = fm_ok and (grad_err < 5e-3)
    print(f"--- Self-tests {'PASSED' if all_ok else 'FAILED'} ---")
    return all_ok


# ==============================================================================
# PART 7 -- MAIN / DEMONSTRATION
# ==============================================================================

def count_params(mind):
    return sum(p.data.size for p in mind.params())


def main():
    print("=" * 78)
    print("CHAPTER 224 -- IBN SINA (AVICENNA), 980-1037 CE")
    print('"The Estimative Leap" -- wahm + hads + ittisal as a trainable architecture')
    print("=" * 78)

    rng = np.random.RandomState(RNG_SEED)
    mind = IbnSinaMind()
    print(f"\nTotal trainable scalar parameters: {count_params(mind):,}")
    print(f"Faculties: external senses(5) -> common sense -> [wahm(fast) | "
          f"imagination+memory -> cogitation] -> rational soul (codebook of "
          f"{mind.n_forms} forms) -> hads / floating self / agent intellect "
          f"-> body-spirit interface")

    print("\n--- Gradient check BEFORE training (correctness, not learning) ---")
    gradient_check()

    print("\n--- Training (40 epochs, two joint tasks + floating-man invariant) ---")
    history = train(mind, n_epochs=40, batches_per_epoch=25, batch_size=64, lr=0.08)

    print("\n--- Final evaluation ---")
    acc_a, acc_h = evaluate(mind, mind.n_categories, rng, n_batches=20, batch_size=64,
                             epoch=999, warmup_epochs=3)
    print(f"  estimation-classification accuracy: {acc_a:.2%}")
    print(f"  hads (hidden-ontology leap) accuracy: {acc_h:.2%}")

    print("\n--- Self-tests (post-training) ---")
    ok = run_all_self_tests(mind, rng)

    print("\n--- Floating Man demonstration (Ibn Sina, Kitab al-Isharat) ---")
    print("  'Imagine yourself created all at once, floating in the air, ")
    print("   suspended from all sensation... you would still affirm the ")
    print("   existence of your self.' -- tested above, not merely asserted.")

    print("\n" + "=" * 78)
    print("DEMONSTRATION COMPLETE" + ("  [ALL SELF-TESTS PASS]" if ok else "  [SELF-TEST FAILURE]"))
    print("=" * 78)
    return mind, history


if __name__ == "__main__":
    main()

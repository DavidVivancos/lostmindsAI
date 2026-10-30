#!/usr/bin/env python3
# =============================================================================
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0323 · Francisco de Assisi (1181-1226)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# Assisi, Umbria, c.1181/82 CE — Assisi, 3 October 1226 CE
# -----------------------------------------------------------------------------
# THE CANTICLE KINSHIP NETWORK (CKN)
#
# This file is not a Transformer wearing a Franciscan costume. It does not use
# attention-over-stored-keys, a Query/Key/Value split, or a mixture-of-experts
# router — those mechanisms encode a mind that treats other minds (or other
# tokens) as an archive to be queried from a position of need. Francis's own
# surviving writings describe a different cognitive move entirely: he does not
# query the sun, the water, or the leper for information he lacks. He *names*
# them kin — "Brother Sun," "Sister Water," "Sister Bodily Death" — and in the
# act of naming, declares the relationship symmetric before any exchange of
# information occurs. Recognition precedes use.
#
# The Canticle Kinship Network (CKN) encodes exactly that structural choice:
#
#   1. MUTUAL RECOGNITION BEFORE QUERY.
#      Relatedness between any two nodes is stored in one shared, SYMMETRIC
#      affinity matrix (S = S^T by construction), not in separate Query and
#      Key projections. If node i attends strongly to node j, the underlying
#      parameter that produced that attention is the *same* parameter that
#      produces j's attention to i. Kinship is not requested; it is declared.
#
#   2. ONE SHARED VOICE, MANY NAMES.
#      Every kin-node is addressed through the same shared transform
#      (W_msg, b) — Francis preached the same Gospel simplicity to the sun,
#      the leper and the Sultan alike — while each node keeps its own
#      learned NAME EMBEDDING, its irreducible identity. The architecture
#      therefore separates "how one speaks to kin" (shared, few parameters)
#      from "who each kin is" (individuated, one row per name).
#
#   3. POVERTY AS A LITERAL SPARSITY OBJECTIVE.
#      Francis's Testament and Earlier Rule treat possession — even of a
#      book, even of a fixed opinion about a brother — as the thing that
#      forecloses freedom and relationship. The kinship-affinity matrix S is
#      trained under an L1 penalty: most possible relationships are driven
#      to zero. What survives training is a small, deliberately-owned set of
#      relationships, not a dense, all-to-all possession of everyone by
#      everyone. Sparsity here is not a compression trick; it is the
#      objective the mind actually held.
#
#   4. NO NODE MAY DOMINATE — A HUMILITY REGULARIZER.
#      A variance penalty on the norms of node states keeps any single kin
#      (Brother Sun, say) from swelling to dominate the shared representation
#      space. cf. Francis's refusal of every offered title — he never let
#      himself be ordained, never accepted the office of Minister General for
#      long, resigned effective leadership of his own Order in 1220.
#
#   5. THE TESTAMENT LOCK — A FOUNDER-ANCHOR REGULARIZER.
#      This is the mechanism this chapter's essay leans on hardest, and it
#      is the one genuinely new piece of machinery here. Between 1221 and
#      1223 Francis's own Rule was taken from him, redrafted by canon
#      lawyers, and returned to him "far less than [he] wanted" (Bullata,
#      1223). He spent his final months dictating a Testament whose closing
#      clauses forbid any "gloss" — any reinterpretation — of his words by
#      future ministers. It is, in the vocabulary of this project, an early,
#      non-computational attempt at a founder-intent lock against
#      institutional value drift. The CKN implements this literally: after
#      an initial "Rule-writing" training phase, the SHARED transform
#      parameters (W_sense, and each layer's W_msg, b — the "manner of
#      address," not the particular graph of who-is-kin-to-whom) are
#      snapshotted as an immutable anchor. When the fraternity subsequently
#      grows — new kin-nodes are added, exactly as the Order grew from a
#      dozen friars to thousands within Francis's own lifetime — an anchor
#      loss pulls the shared transform back toward its founding values,
#      while the affinity graph S and the per-node name embeddings, which
#      MUST grow to admit new kin, remain free to adapt. Self-test 4 below
#      trains the same growth phase with the anchor switched on and off and
#      shows, numerically, that the anchor holds the "manner of address"
#      measurably closer to the founding Rule while an un-anchored network
#      drifts freely — a working model of what the Testament tried, in
#      prose, to guarantee by fiat.
#
#   6. THE CANTICLE CONSOLIDATION — A GRATITUDE-GATED LEARNING-RATE SCHEDULE.
#      The Canticle of the Creatures was not composed all at once; Francis
#      added stanzas across roughly two years, each apparently after a
#      period of suffering, as a way of re-integrating experience into praise
#      (see Sources). Each training epoch here ends with a "Canticle" step:
#      a harmony scalar is computed from how well kin that the network has
#      chosen to relate (nonzero entries of S) actually cohere in state
#      space, and that scalar gently scales the following epoch's effective
#      learning rate. A network that is currently well-integrated consolidates
#      a little faster; a discordant one is not pushed harder, it is asked to
#      sit with the dissonance a moment longer. This is a real, if modest,
#      working scheduler, not a decorative print statement.
#
# WHAT THIS ARCHITECTURE IS *NOT* CLAIMED TO BE:
#   It is not a claim that Francis "foresaw" graph neural networks, or that
#   a 13th-century friar was a proto-computer-scientist. It is a structural
#   translation exercise: given everything his surviving writings show about
#   how he believed a mind ought to relate to other minds — symmetrically,
#   nominally, sparsely, humbly, and under an inheritance-lock against
#   institutional capture — what is the smallest working neural mechanism
#   that could not be relabeled Transformer/MoE without losing the thesis?
#   This file is one answer to that question.
#
# IMPLEMENTATION NOTES
#   - Pure NumPy. No PyTorch, TensorFlow, JAX, or any other ML framework.
#   - A ~150-line reverse-mode autodiff engine (Section 1) is implemented
#     from scratch so that every gradient below is *derived*, not hand-typed
#     — and then checked against central-difference numerical gradients
#     (Section 8, mandatory, must pass before training is allowed to run).
#   - Everything is float64 for gradient-check precision.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

RNG_SEED = 1181  # Francis's birth year — the network's one small joke.


# =============================================================================
# SECTION 1 — THE AUTOGRAD ENGINE
# A minimal reverse-mode automatic-differentiation core. This is the only
# "framework" the rest of the file uses, and it is written here, in full.
# =============================================================================

class Tensor:
    """A numpy array that remembers how it was made, so it can be told how
    much each of its ancestors contributed to a final scalar loss."""

    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad", "name")

    def __init__(self, data, _prev: Tuple["Tensor", ...] = (), requires_grad: bool = True, name: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev = set(_prev)
        self.requires_grad = requires_grad
        self.name = name

    @staticmethod
    def wrap(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    def _accum(self, g: np.ndarray):
        """Accumulate an incoming gradient, undoing any broadcasting that
        happened on the forward pass by summing over the broadcast axes."""
        while g.ndim > self.data.ndim:
            g = g.sum(axis=0)
        for axis, (gs, ss) in enumerate(zip(g.shape, self.data.shape)):
            if ss == 1 and gs != 1:
                g = g.sum(axis=axis, keepdims=True)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + g

    # ---- elementary operations -------------------------------------------
    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad)
            if other.requires_grad:
                other._accum(out.grad)
        out._backward = _backward
        return out
    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        def _backward():
            if self.requires_grad:
                self._accum(-out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-Tensor.wrap(other))

    def __rsub__(self, other):
        return Tensor.wrap(other) + (-self)

    def __mul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data * other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad * other.data)
            if other.requires_grad:
                other._accum(out.grad * self.data)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data / other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad / other.data)
            if other.requires_grad:
                other._accum(-out.grad * self.data / (other.data ** 2))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad @ other.data.T)
            if other.requires_grad:
                other._accum(self.data.T @ out.grad)
        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad.T)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (1.0 - t ** 2))
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad * e)
        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad * np.sign(self.data))
        out._backward = _backward
        return out

    def pow(self, p: float):
        out = Tensor(self.data ** p, (self,))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad * p * (self.data ** (p - 1.0)))
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.data.shape)
                self._accum(g)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def backward(self):
        assert self.data.shape == () or self.data.size == 1, "backward() must start from a scalar Tensor"
        topo: List[Tensor] = []
        visited = set()
        def build(v: "Tensor"):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def softmax_masked(S: Tensor, mask: np.ndarray) -> Tensor:
    """Row-softmax of S restricted to `mask` (mask[i,j]=0 forbids the edge,
    e.g. self-loops). Built entirely from primitives above, so its backward
    pass is inherited for free and is exact."""
    mask_t = Tensor(mask, requires_grad=False)
    shifted_data = S.data * mask - 1e9 * (1.0 - mask)
    row_max = Tensor(np.max(shifted_data, axis=1, keepdims=True), requires_grad=False)
    E = (S - row_max).exp() * mask_t
    row_sum = E.sum(axis=1, keepdims=True)
    return E / row_sum


# =============================================================================
# SECTION 2 — PARAMETERS AND THE ADAM OPTIMIZER (also hand-written)
# =============================================================================

class Adam:
    def __init__(self, params: Dict[str, Tensor], lr: float = 0.01,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def add_param(self, key: str, tensor: Tensor):
        self.params[key] = tensor
        self.m[key] = np.zeros_like(tensor.data)
        self.v[key] = np.zeros_like(tensor.data)

    def zero_grad(self):
        for p in self.params.values():
            p.zero_grad()

    def step(self, lr_scale: float = 1.0):
        self.t += 1
        for k, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            p.data = p.data - (self.lr * lr_scale) * m_hat / (np.sqrt(v_hat) + self.eps)


def xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


# =============================================================================
# SECTION 3 — THE CANTICLE KINSHIP NETWORK
# =============================================================================

class KinshipLayer:
    """One 'fraternal message-passing' step. Symmetric affinity, one shared
    voice, residual tanh update. See the module docstring for the reading."""

    def __init__(self, dim: int, rng: np.random.Generator, name: str = "layer"):
        self.dim = dim
        self.name = name
        self.W_msg = Tensor(xavier(dim, dim, rng), name=f"{name}.W_msg")
        self.b = Tensor(np.zeros((1, dim)), name=f"{name}.b")

    def parameters(self) -> Dict[str, Tensor]:
        return {f"{self.name}.W_msg": self.W_msg, f"{self.name}.b": self.b}

    def shared_parameters(self) -> Dict[str, Tensor]:
        """The parameters that constitute 'the manner of address' — the
        part of the mind the Testament tries to keep fixed."""
        return self.parameters()

    def forward(self, H: Tensor, S_raw: Tensor, mask: np.ndarray) -> Tuple[Tensor, Tensor]:
        S = (S_raw + S_raw.T) * 0.5           # kinship is symmetric BY CONSTRUCTION
        A = softmax_masked(S, mask)            # each kin distributes attention to siblings
        msg = A @ (H @ self.W_msg)             # one shared voice speaks to every sibling alike
        H_new = (H + msg + self.b).tanh()      # residual: nobody is erased by a new encounter
        return H_new, A


class CanticleKinshipNetwork:
    """
    Full architecture:
        name-embedding (who each kin is)
        + sense-projection (what is currently observed of them)
        -> L stacked KinshipLayers (how kin who recognise each other speak)
        -> linear readout (the note each kin is currently singing)

    A single set of kin-nodes is not fixed in advance: the network is built
    for a *maximum* capacity N_max, and only the first N nodes (the current
    fraternity) are active at a given time — exactly so that Section 6's
    "the Order grows" experiment can extend N without reallocating the
    shared transform parameters at all.
    """

    def __init__(self, n_max: int, dim: int, in_feat: int = 2,
                 n_layers: int = 3, out_dim: int = 1, seed: int = RNG_SEED):
        self.rng = np.random.default_rng(seed)
        self.n_max = n_max
        self.dim = dim
        self.n_layers = n_layers

        # (1) NAMES: one learned identity row per possible kin. "Brother Sun"
        # is a name before it is a function; this embedding is that name.
        self.name_embed = Tensor(self.rng.normal(0, 0.3, size=(n_max, dim)), name="name_embed")

        # (2) SENSES: the single shared faculty through which any observation
        # of any creature is first received. Francis did not need a special
        # organ to perceive the sun and a different one for the leper.
        self.W_sense = Tensor(xavier(in_feat, dim, self.rng), name="W_sense")

        # (3) KINSHIP GRAPH: symmetric affinity between every pair of possible
        # kin. Only entries among the *currently active* N nodes are used;
        # the rest sit at their prior until the fraternity grows into them.
        self.S_raw = Tensor(self.rng.normal(0, 0.05, size=(n_max, n_max)), name="S_raw")

        # (4) THE FRATERNITY ITSELF: L kinship layers, sharing no weights
        # with each other (each is its own "gathering") but each internally
        # a single shared voice.
        self.layers = [KinshipLayer(dim, self.rng, name=f"kinship_{l}") for l in range(n_layers)]

        # (5) READOUT: what note is this kin singing right now.
        self.W_out = Tensor(xavier(dim, out_dim, self.rng), name="W_out")
        self.b_out = Tensor(np.zeros((1, out_dim)), name="b_out")

        self._testament: Optional[Dict[str, np.ndarray]] = None  # the Testament lock, once sworn

    # ---- bookkeeping --------------------------------------------------
    def parameters(self) -> Dict[str, Tensor]:
        p = {"name_embed": self.name_embed, "W_sense": self.W_sense,
             "S_raw": self.S_raw, "W_out": self.W_out, "b_out": self.b_out}
        for layer in self.layers:
            p.update(layer.parameters())
        return p

    def shared_voice_parameters(self) -> Dict[str, Tensor]:
        """The parameters that the Testament locks: W_sense (how anything
        is first perceived) plus each layer's (W_msg, b) (how kin, once
        recognised, are addressed). Explicitly EXCLUDES name_embed and
        S_raw, which must be free to grow as new kin join the fraternity."""
        p = {"W_sense": self.W_sense}
        for layer in self.layers:
            p.update(layer.shared_parameters())
        return p

    def swear_testament(self):
        """Snapshot the current shared voice as an immutable reference. This
        is called once, at the end of the founding ('Earlier Rule') phase —
        it is deliberately NOT re-callable during Phase 2 in the experiments
        below, matching the historical fact that Francis dictated exactly
        one Testament, days before his death, and forbade it being glossed."""
        self._testament = {k: v.data.copy() for k, v in self.shared_voice_parameters().items()}

    def testament_anchor_loss(self) -> Tensor:
        """||current shared voice - sworn Testament||^2. Zero (and inert)
        until swear_testament() has been called."""
        if self._testament is None:
            return Tensor(0.0, requires_grad=False)
        total = Tensor(0.0, requires_grad=False)
        for k, cur in self.shared_voice_parameters().items():
            anchor = Tensor(self._testament[k], requires_grad=False)
            total = total + ((cur - anchor).pow(2)).mean()
        return total * (1.0 / len(self._testament))

    def testament_drift(self) -> float:
        """A plain-numpy (non-autograd) scalar for reporting: how far has the
        shared voice wandered from the sworn Testament, in raw parameter
        space? Used only for printing/plotting, never for training."""
        if self._testament is None:
            return 0.0
        total, count = 0.0, 0
        for k, cur in self.shared_voice_parameters().items():
            d = cur.data - self._testament[k]
            total += float(np.mean(d ** 2))
            count += 1
        return total / max(count, 1)

    # ---- forward --------------------------------------------------------
    def forward(self, x_feat: np.ndarray, n_active: int) -> Tuple[Tensor, List[Tensor]]:
        """
        x_feat: (n_active, in_feat) numpy array — per-node observed sensory
                features for the currently active fraternity.
        Returns (predictions (n_active,out_dim) Tensor, list of attention
        matrices per layer, for inspection/visualisation).
        """
        name_active = Tensor(self.name_embed.data[:n_active], requires_grad=False) \
            if False else self._slice(self.name_embed, n_active)
        sense = Tensor(x_feat, requires_grad=False) @ self.W_sense
        H = (name_active + sense).tanh()

        S_active = self._slice2d(self.S_raw, n_active)
        mask = 1.0 - np.eye(n_active)

        attentions = []
        for layer in self.layers:
            H, A = layer.forward(H, S_active, mask)
            attentions.append(A)

        pred = H @ self.W_out + self.b_out
        return pred, attentions

    @staticmethod
    def _slice(t: Tensor, n: int) -> Tensor:
        """Differentiable row-slice t[:n] (needed because plain numpy
        slicing loses the autograd graph)."""
        idx = np.zeros((n, t.data.shape[0]))
        for i in range(n):
            idx[i, i] = 1.0
        return Tensor(idx, requires_grad=False) @ t

    @staticmethod
    def _slice2d(t: Tensor, n: int) -> Tensor:
        left = np.zeros((n, t.data.shape[0]))
        for i in range(n):
            left[i, i] = 1.0
        right = np.zeros((t.data.shape[1], n))
        for j in range(n):
            right[j, j] = 1.0
        return Tensor(left, requires_grad=False) @ t @ Tensor(right, requires_grad=False)

    # ---- the two Franciscan regularisers ---------------------------------
    def poverty_loss(self, n_active: int) -> Tensor:
        """L1 on the ACTIVE portion of the kinship graph: most possible
        relationships should be owned by nobody."""
        S_active = self._slice2d(self.S_raw, n_active)
        return S_active.abs().mean()

    def humility_loss(self, H_final: Tensor) -> Tensor:
        """Variance of node-state norms: no single kin dominates the room."""
        norms = (H_final * H_final).sum(axis=1, keepdims=True).pow(0.5)
        mean_norm = norms.mean()
        return ((norms - mean_norm).pow(2)).mean()

    def harmony(self, H_final: np.ndarray, A_last: np.ndarray, threshold: float = 0.05) -> float:
        """A plain-numpy 'gratitude' scalar for the Canticle Consolidation
        learning-rate gate: mean cosine similarity between kin the network
        has actually chosen to relate (A above `threshold`)."""
        n = H_final.shape[0]
        norm = H_final / (np.linalg.norm(H_final, axis=1, keepdims=True) + 1e-8)
        cos = norm @ norm.T
        chosen = A_last > threshold
        np.fill_diagonal(chosen, False)
        if not chosen.any():
            return 0.0
        return float(cos[chosen].mean())


# =============================================================================
# SECTION 4 — THE BOOK OF CREATION (synthetic task)
#
# Francis held (Testament, and the whole practice of the Canticle) that
# creation is legible directly, without a scholarly intermediary — each
# creature is a word in one sentence whose meaning is relational, not
# standalone. The task built here is a direct translation: every node's true
# "note" at a given moment is a function of a single shared latent "chord"
# (the moment's whole disposition) plus a fixed per-kin role. Some kin are
# observed, some are not; the network must infer the unheard notes from the
# heard ones purely through the kinship graph it has learned to trust. This
# is deliberately NOT a next-token / sequence task: there is no privileged
# reading order, matching the Canticle's own refusal of hierarchy among the
# creatures it addresses.
# =============================================================================

# The founding fraternity — every one of these is a named addressee in the
# Canticle of the Creatures (Francis of Assisi: Early Documents, vol. I, 113-114).
FOUNDING_KIN = ["Brother Sun", "Sister Moon", "Brother Wind",
                "Sister Water", "Brother Fire", "Mother Earth", "Sister Death"]

# Three historically-attested widenings of Francis's circle of address, added
# in the "growth" phase below to stand in for the Order's real expansion
# beyond his direct oversight: the leper (Testament, his own account of his
# conversion), the Sultan al-Malik al-Kamil (Damietta, 1219, attested in
# multiple independent early sources), and Clare of Assisi (San Damiano,
# co-founder of the Poor Ladies, attested throughout the early Legends).
GROWTH_KIN = ["The Leper", "Sultan al-Kamil", "Clare of Assisi"]

ALL_KIN = FOUNDING_KIN + GROWTH_KIN

# Fixed "role" constants: how strongly each kin's note follows the shared
# chord of the moment, and its own baseline pitch. Chosen by hand to be
# thematically legible, not fit to any data.
ROLE_WEIGHT = np.array([1.00, 0.70, 0.55, 0.65, 0.60, 0.30, -0.40,
                         0.35, 0.20, 0.05])
ROLE_BIAS = np.array([0.10, 0.00, 0.00, -0.05, 0.05, 0.20, -0.10,
                       -0.15, 0.00, 0.35])
ROLE_NOISE = np.array([0.10, 0.10, 0.18, 0.12, 0.15, 0.06, 0.08,
                        0.12, 0.15, 0.04])


def generate_book_of_creation(rng, n_samples: int, n_active: int, hide_fraction: float = 0.4):
    """
    Returns:
        x_feat   : (n_samples, n_active, 2) array of [observed_value_or_0, observed_flag]
        targets  : (n_samples, n_active, 1) array of true values
        loss_mask: (n_samples, n_active, 1) array, 1.0 where the value was HIDDEN
    """
    chords = rng.normal(0, 1.0, size=(n_samples, 1))
    true_vals = chords @ ROLE_WEIGHT[:n_active].reshape(1, -1) + ROLE_BIAS[:n_active].reshape(1, -1)
    true_vals = true_vals + rng.normal(0, 1.0, size=(n_samples, n_active)) * ROLE_NOISE[:n_active].reshape(1, -1)

    n_hidden = max(1, int(round(hide_fraction * n_active)))
    x_feat = np.zeros((n_samples, n_active, 2))
    loss_mask = np.zeros((n_samples, n_active, 1))
    for b in range(n_samples):
        hidden_idx = rng.choice(n_active, size=n_hidden, replace=False)
        observed = np.ones(n_active, dtype=bool)
        observed[hidden_idx] = False
        x_feat[b, observed, 0] = true_vals[b, observed]
        x_feat[b, observed, 1] = 1.0
        loss_mask[b, hidden_idx, 0] = 1.0

    targets = true_vals.reshape(n_samples, n_active, 1)
    return x_feat, targets, loss_mask


# =============================================================================
# SECTION 5 — LOSS ASSEMBLY (shared by training, the gradient check, and eval)
# =============================================================================

def full_forward_and_loss(net: CanticleKinshipNetwork, x_feat_b: np.ndarray, target_b: np.ndarray,
                           mask_b: np.ndarray, n_active: int,
                           lambda_poverty: float = 0.01, lambda_humility: float = 0.01,
                           lambda_testament: float = 0.0) -> Tuple[Tensor, Dict[str, float]]:
    name_active = net._slice(net.name_embed, n_active)
    sense = Tensor(x_feat_b, requires_grad=False) @ net.W_sense
    H = (name_active + sense).tanh()
    S_active = net._slice2d(net.S_raw, n_active)
    mask = 1.0 - np.eye(n_active)
    A_last = None
    for layer in net.layers:
        H, A_last = layer.forward(H, S_active, mask)
    pred = H @ net.W_out + net.b_out

    diff = pred - Tensor(target_b, requires_grad=False)
    sq = diff.pow(2) * Tensor(mask_b, requires_grad=False)
    denom = max(float(mask_b.sum()), 1.0)
    task_loss = sq.sum() * (1.0 / denom)

    total = task_loss
    poverty = net.poverty_loss(n_active)
    humility = net.humility_loss(H)
    total = total + poverty * lambda_poverty + humility * lambda_humility

    testament_term = 0.0
    if lambda_testament > 0.0 and net._testament is not None:
        anchor = net.testament_anchor_loss()
        total = total + anchor * lambda_testament
        testament_term = float(anchor.data)

    info = {
        "task_loss": float(task_loss.data),
        "poverty": float(poverty.data),
        "humility": float(humility.data),
        "testament_anchor": testament_term,
        "harmony": net.harmony(H.data, A_last.data),
    }
    return total, info


# =============================================================================
# SECTION 6 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
#
# Every gradient the training loop below relies on is checked here against
# central-difference numerical gradients, on a fresh small network, BEFORE
# any training is permitted to run (see run_all() at the bottom). This is
# not optional scaffolding: it is the thing that lets the rest of this file
# claim its backward pass is correct rather than merely plausible-looking.
# =============================================================================

def gradient_check(seed: int = 7, verbose: bool = True) -> float:
    rng = np.random.default_rng(seed)
    n_active = 6
    net = CanticleKinshipNetwork(n_max=9, dim=5, in_feat=2, n_layers=2, out_dim=1, seed=seed)
    net.swear_testament()  # exercise the testament-anchor gradient path too

    # perturb the shared voice slightly so the anchor loss is non-degenerate
    net.W_sense.data += rng.normal(0, 0.05, size=net.W_sense.data.shape)

    x_feat, target, mask = generate_book_of_creation(rng, n_samples=1, n_active=n_active)
    x_feat_b, target_b, mask_b = x_feat[0], target[0], mask[0]

    def loss_fn():
        total, _ = full_forward_and_loss(net, x_feat_b, target_b, mask_b, n_active,
                                          lambda_poverty=0.02, lambda_humility=0.02,
                                          lambda_testament=0.5)
        return total

    for p in net.parameters().values():
        p.zero_grad()
    loss = loss_fn()
    loss.backward()

    checks: List[Tuple[str, int, int, float, float]] = []
    eps = 1e-6
    rng_check = np.random.default_rng(seed + 1)
    params = net.parameters()
    for pname, tensor in params.items():
        flat_shape = tensor.data.shape
        n_entries = tensor.data.size
        n_sample = min(3, n_entries)
        flat_idx = rng_check.choice(n_entries, size=n_sample, replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, flat_shape)
            orig = tensor.data[idx]

            tensor.data[idx] = orig + eps
            l_plus = float(loss_fn().data)
            tensor.data[idx] = orig - eps
            l_minus = float(loss_fn().data)
            tensor.data[idx] = orig

            num_grad = (l_plus - l_minus) / (2 * eps)
            ana_grad = tensor.grad[idx] if tensor.grad is not None else 0.0
            checks.append((pname, idx, num_grad, ana_grad,
                            abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-8)))

    max_rel_err = max(c[4] for c in checks)
    if verbose:
        print(f"  Gradient check: {len(checks)} parameter entries sampled across "
              f"{len(params)} tensors.")
        worst = sorted(checks, key=lambda c: -c[4])[:5]
        for pname, idx, num_g, ana_g, rel in worst:
            print(f"    {pname:20s} idx={idx!s:12s} analytic={ana_g:+.8f} "
                  f"numeric={num_g:+.8f} rel_err={rel:.2e}")
        print(f"  Max relative error: {max_rel_err:.2e}  "
              f"({'PASS' if max_rel_err < 1e-4 else 'FAIL'}, threshold 1e-4)")
    return max_rel_err


# =============================================================================
# SECTION 7 — TRAINING LOOP WITH THE CANTICLE CONSOLIDATION SCHEDULE
# =============================================================================

def train_phase(net: CanticleKinshipNetwork, optimizer: Adam, rng: np.random.Generator,
                 n_active: int, epochs: int, batch_size: int = 16,
                 lambda_poverty: float = 0.01, lambda_humility: float = 0.005,
                 lambda_testament: float = 0.0, harmony_alpha: float = 0.3,
                 verbose: bool = True, log_every: int = 5) -> List[float]:
    history = []
    for epoch in range(epochs):
        x_feat, target, mask = generate_book_of_creation(rng, n_samples=batch_size, n_active=n_active)
        epoch_loss = 0.0
        epoch_harmony = 0.0
        for b in range(batch_size):
            for p in net.parameters().values():
                p.zero_grad()
            total, info = full_forward_and_loss(net, x_feat[b], target[b], mask[b], n_active,
                                                 lambda_poverty, lambda_humility, lambda_testament)
            total.backward()
            # The Canticle Consolidation: this sample's harmony gently scales
            # its own gradient step. Well-integrated moments consolidate a
            # little faster; discordant ones are not forced.
            lr_scale = 1.0 + harmony_alpha * np.tanh(info["harmony"])
            optimizer.step(lr_scale=max(lr_scale, 0.1))
            epoch_loss += total.data / batch_size
            epoch_harmony += info["harmony"] / batch_size
        history.append(float(epoch_loss))
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            drift = net.testament_drift()
            print(f"    epoch {epoch:3d}  loss={epoch_loss:.4f}  harmony={epoch_harmony:+.3f}"
                  + (f"  testament_drift={drift:.5f}" if net._testament is not None else ""))
    return history


# =============================================================================
# SECTION 8 — SELF-TESTS
# =============================================================================

def run_self_tests():
    print("=" * 78)
    print("SELF-TEST 1: Gradient check (mandatory)")
    print("=" * 78)
    max_err = gradient_check()
    assert max_err < 1e-4, f"Gradient check FAILED: max relative error {max_err:.2e}"
    print("  -> PASSED\n")

    print("=" * 78)
    print("SELF-TEST 2: Structural sanity (symmetry, softmax normalisation)")
    print("=" * 78)
    rng = np.random.default_rng(3)
    net = CanticleKinshipNetwork(n_max=7, dim=6, n_layers=2, seed=3)
    n_active = 7
    S_active = net._slice2d(net.S_raw, n_active)
    S_sym = (S_active + S_active.T) * 0.5
    assert np.allclose(S_sym.data, S_sym.data.T), "kinship affinity is not symmetric"
    mask = 1.0 - np.eye(n_active)
    A = softmax_masked(S_sym, mask)
    row_sums = A.data.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "attention rows do not sum to 1"
    assert np.allclose(np.diag(A.data), 0.0), "self-loops leaked into attention"
    print("  Kinship affinity symmetric: OK")
    print("  Per-node attention sums to 1.0: OK")
    print("  No self-attention (no node is kin to itself): OK\n")

    print("=" * 78)
    print("SELF-TEST 3: The founding fraternity learns to read the Book of Creation")
    print("=" * 78)
    rng = np.random.default_rng(11)
    net = CanticleKinshipNetwork(n_max=len(ALL_KIN), dim=10, in_feat=2, n_layers=3,
                                  out_dim=1, seed=11)
    opt = Adam(net.parameters(), lr=0.05)
    history = train_phase(net, opt, rng, n_active=len(FOUNDING_KIN), epochs=40,
                           batch_size=16, verbose=True, log_every=10)
    assert history[-1] < history[0] * 0.5, (
        f"training did not reduce loss enough: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"  Loss fell from {history[0]:.4f} to {history[-1]:.4f} over {len(history)} epochs -> PASSED\n")

    print("=" * 78)
    print("SELF-TEST 4: The Testament Lock — does an inherited anchor actually")
    print("resist drift when the fraternity grows beyond its founding shape?")
    print("=" * 78)
    net.swear_testament()
    baseline_shared = {k: v.data.copy() for k, v in net.shared_voice_parameters().items()}

    # --- condition A: grown WITHOUT the testament anchor -------------------
    import copy
    net_free = copy.deepcopy(net)
    net_free._testament = None
    opt_free = Adam(net_free.parameters(), lr=0.05)
    print("  (a) growth phase, testament anchor OFF:")
    train_phase(net_free, opt_free, np.random.default_rng(21), n_active=len(ALL_KIN),
                epochs=30, batch_size=16, lambda_testament=0.0, verbose=True, log_every=10)
    drift_free = 0.0
    for k, cur in net_free.shared_voice_parameters().items():
        drift_free += float(np.mean((cur.data - baseline_shared[k]) ** 2))
    drift_free /= len(baseline_shared)

    # --- condition B: grown WITH the testament anchor -----------------------
    net_anchored = copy.deepcopy(net)
    net_anchored._testament = {k: v.copy() for k, v in baseline_shared.items()}
    opt_anchored = Adam(net_anchored.parameters(), lr=0.05)
    print("  (b) growth phase, testament anchor ON:")
    train_phase(net_anchored, opt_anchored, np.random.default_rng(21), n_active=len(ALL_KIN),
                epochs=30, batch_size=16, lambda_testament=1.5, verbose=True, log_every=10)
    drift_anchored = net_anchored.testament_drift()

    print(f"\n  Shared-voice drift from founding Rule after growth:")
    print(f"    WITHOUT testament lock : {drift_free:.6f}")
    print(f"    WITH    testament lock : {drift_anchored:.6f}")
    assert drift_anchored < drift_free, "the testament lock did not reduce drift — architecture failed its own thesis"
    reduction = 100.0 * (1.0 - drift_anchored / max(drift_free, 1e-12))
    print(f"    -> the lock holds the shared voice {reduction:.1f}% closer to the "
          f"founding Rule -> PASSED\n")

    print("=" * 78)
    print("ALL SELF-TESTS PASSED")
    print("=" * 78)
    return {
        "gradient_check_max_rel_err": max_err,
        "phase1_loss_start": history[0],
        "phase1_loss_end": history[-1],
        "drift_without_testament": drift_free,
        "drift_with_testament": drift_anchored,
        "drift_reduction_pct": reduction,
    }


# =============================================================================
# SECTION 9 — METADATA + MAIN
# =============================================================================

def get_figure_metadata() -> Dict:
    return {
        "id": 263,
        "name": "Francesco di Pietro di Bernardone (Francis of Assisi)",
        "born": "c. 1181/1182",
        "died": "3 October 1226",
        "region": "Assisi, Umbria, Italy",
        "architecture_name": "Canticle Kinship Network (CKN)",
        "core_mechanism": [
            "Symmetric kinship-affinity graph (mutual recognition, not Query/Key asymmetry)",
            "One shared 'voice' transform addressing every named kin alike",
            "Per-kin learned NAME embeddings (identity is not interchangeable)",
            "Poverty regulariser: L1 sparsity on the kinship graph",
            "Humility regulariser: variance penalty on node-state norms",
            "Testament Lock: founder-anchor regulariser resisting institutional drift",
            "Canticle Consolidation: harmony-gated learning-rate schedule",
        ],
        "implementation": {"framework": "NumPy only", "autograd": "hand-written, ~150 lines",
                            "gradient_check": "mandatory, central-difference, threshold 1e-4"},
    }


if __name__ == "__main__":
    print("=" * 78)
    print("THE ENCYCLOPEDIA OF LOST MINDS: ECHOES ON AI")
    print("Figure 0263 — Francis of Assisi (c.1181/82-1226)")
    print("The Canticle Kinship Network (CKN)")
    print("=" * 78)
    print()
    results = run_self_tests()
    print()
    print("=" * 78)
    print("ARCHITECTURE PHILOSOPHY, RESTATED")
    print("=" * 78)
    print("""
    The Canticle Kinship Network does not ask what a creature can supply.
    It asks what a creature can be called. Every mechanism above follows
    from that one reversal: symmetric affinity instead of directed query,
    a shared voice instead of a bespoke one for each kind of node, a
    poverty penalty on the very *idea* of owning a relationship, a
    humility penalty on any node's wish to dominate the room, and,
    binding the whole thing across time, a Testament that will not let
    growth erase the manner in which the founder first learned to speak
    to the world.
    """)
    print("Figure metadata:", get_figure_metadata()["architecture_name"])


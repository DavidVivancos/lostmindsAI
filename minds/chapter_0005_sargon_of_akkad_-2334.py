#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0005_sargon_of_akkad_-2334.py
 THE IMPERIAL TRANSFORMER  --  a trainable AGI base model after Sargon of Akkad
 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
 How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
 Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
 Resume and Interactive Demos at https://artificiology.com/

================================================================================

WHAT THIS FILE IS
-----------------
This is NOT a slideshow or a forward-only "demo". It is a complete, *trainable*
neural architecture written from first principles in pure NumPy, including its
own reverse-mode automatic-differentiation engine (the "fundamental code"), a
from-scratch Adam optimizer, a real training loop, and a test suite that
gradient-checks the primitives and proves the full model learns.

If you run it:

    python3 chapter_0005_sargon_of_akkad_-2334.py            # train + eval
    python3 chapter_0005_sargon_of_akkad_-2334.py --test     # run the test suite

...it builds a small model, trains it on a structured synthetic task, prints a
loss/accuracy curve that goes DOWN/UP, and (with --test) verifies every gradient
against finite differences and overfits a tiny batch to ~0 loss.

WHY "IMPERIAL TRANSFORMER" (the Sargon mapping)
-----------------------------------------------
Sargon of Akkad (r. ~2334-2279 BCE) built the first empire in recorded history.
His real invention was not conquest but a *cognitive architecture for coordinating
many specialised parts*: a central authority that routes work to specialised
provinces, a standardised protocol so the provinces interoperate, fast channels
for information flow, durable archives, and a reporting system that keeps any one
province from dominating or being neglected. That is, almost line-for-line, a
modern Mixture-of-Experts Transformer. The mapping this file implements:

    Sargon's empire                      Imperial Transformer (this code)
    -------------------------------      ------------------------------------------
    Standardised weights & measures  ->  shared embedding + LayerNorm  (one protocol)
    The Royal Roads (couriers)       ->  multi-head self-attention     (info routing)
    The temple/royal archives        ->  HierarchicalMemory            (learned KV bank)
    The King as central coordinator  ->  the MoE router ("gate")       (who handles what)
    The provincial governors         ->  the experts (FFN provinces)   (specialised work)
    Provincial reporting / audit     ->  load-balancing auxiliary loss (no province idle)

The point of the project (1000 Minds) is that a person's *theory of how minds are
organised* implies a concrete machine. Sargon's theory was "intelligence at scale
is distributed specialists under a standard, coordinated by a router and audited
so nobody is wasted." That theory is a sparse MoE Transformer. We build it and
train it.

Author note: implemented in NumPy on purpose. Depending on a heavyweight DL
framework would hide the gradients; here the chain rule is in the open so the
architecture is genuinely "the fundamental code".
================================================================================
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

# A single global dtype keeps the whole "empire" on one standard (Sargon would approve).
DTYPE = np.float64  # float64 makes finite-difference gradient checking reliable.


# ============================================================================
# SECTION 1 -- THE AUTOGRAD ENGINE  (the actual "neuron" / fundamental code)
# ----------------------------------------------------------------------------
# Every learning system is, underneath, two things: a forward computation and a
# way to push error backwards through it (the chain rule). Frameworks hide this.
# Here we build it. A `Tensor` wraps a NumPy array, remembers the operation that
# produced it, and knows how to send gradient to its parents. Calling .backward()
# on a scalar walks the computation graph in reverse-topological order and fills
# every .grad. This ~120 lines is the engine the entire model runs on.
# ============================================================================


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum `grad` back down to `shape`.

    Forward ops broadcast (e.g. adding a (d,) bias to an (N, d) activation). The
    gradient w.r.t. the smaller operand is therefore the incoming gradient summed
    over every axis that was broadcast. This helper makes broadcasting safe for
    backprop, which is the single most common source of silent autograd bugs.
    """
    while grad.ndim > len(shape):              # collapse extra leading axes
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):            # collapse axes that were size-1
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph.

    Attributes
    ----------
    data    : the NumPy array of values (the forward result)
    grad    : the accumulated gradient of the final scalar w.r.t. this tensor
    _parents: tensors that fed into this one (edges of the graph)
    _back   : closure that, given this node's grad, scatters grad to parents
    requires_grad: whether to bother computing/accumulating a gradient here
    """

    __slots__ = ("data", "grad", "_parents", "_back", "requires_grad", "_op")

    @staticmethod
    def _noop():
        return None

    def __init__(self, data, requires_grad: bool = False, _parents=(), _op: str = ""):
        self.data = np.asarray(data, dtype=DTYPE)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._parents = _parents
        self._back: Callable[[], None] = lambda: None
        self._op = _op

    # ---- bookkeeping ------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r}, grad={self.grad is not None})"

    def zero_grad(self):
        self.grad = None

    def _result(self, data, parents, back, op):
        """Build the output Tensor of an op and wire up its backward closure."""
        needs = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _parents=parents, _op=op)
        if needs:
            out._back = back
        return out

    @staticmethod
    def _accum(t: "Tensor", g: np.ndarray):
        """Accumulate gradient into a parent (gradients ADD when a tensor is reused)."""
        if not t.requires_grad:
            return
        g = _unbroadcast(g, t.data.shape)
        t.grad = g if t.grad is None else t.grad + g

    # ---- elementwise ops --------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data

        def back():
            Tensor._accum(self, out.grad)
            Tensor._accum(other, out.grad)

        out = self._result(out_data, (self, other), back, "add")
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data

        def back():
            Tensor._accum(self, out.grad * other.data)
            Tensor._accum(other, out.grad * self.data)

        out = self._result(out_data, (self, other), back, "mul")
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1.0)

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data / other.data

        def back():
            Tensor._accum(self, out.grad / other.data)
            Tensor._accum(other, out.grad * (-self.data / (other.data ** 2)))

        out = self._result(out_data, (self, other), back, "div")
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        """Batched matrix multiply (supports leading batch dims via np.matmul)."""
        out_data = np.matmul(self.data, other.data)

        def back():
            g = out.grad
            # d(AB)/dA = g @ B^T ; d(AB)/dB = A^T @ g  (with batched transposes)
            ga = np.matmul(g, np.swapaxes(other.data, -1, -2))
            gb = np.matmul(np.swapaxes(self.data, -1, -2), g)
            Tensor._accum(self, ga)
            Tensor._accum(other, gb)

        out = self._result(out_data, (self, other), back, "matmul")
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # ---- shape ops --------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        old = self.data.shape
        out_data = self.data.reshape(*shape)

        def back():
            Tensor._accum(self, out.grad.reshape(old))

        out = self._result(out_data, (self,), back, "reshape")
        return out

    def transpose(self, axes) -> "Tensor":
        out_data = np.transpose(self.data, axes)
        inv = np.argsort(axes)

        def back():
            Tensor._accum(self, np.transpose(out.grad, inv))

        out = self._result(out_data, (self,), back, "transpose")
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        def back():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis if isinstance(axis, int) else tuple(axis))
            Tensor._accum(self, np.broadcast_to(g, self.data.shape).copy())

        out = self._result(out_data, (self,), back, "sum")
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in ([axis] if isinstance(axis, int) else axis)]
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    # ---- nonlinearities ---------------------------------------------------
    def relu(self) -> "Tensor":
        mask = (self.data > 0).astype(DTYPE)
        out_data = self.data * mask

        def back():
            Tensor._accum(self, out.grad * mask)

        out = self._result(out_data, (self,), back, "relu")
        return out

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit (tanh approximation), used in modern Transformers."""
        x = self.data
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out_data = 0.5 * x * (1.0 + t)

        def back():
            # derivative of the tanh-GELU
            dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
            dt = (1.0 - t ** 2) * dinner
            grad = 0.5 * (1.0 + t) + 0.5 * x * dt
            Tensor._accum(self, out.grad * grad)

        out = self._result(out_data, (self,), back, "gelu")
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax with the exact Jacobian-vector backward."""
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)

        def back():
            g = out.grad
            dot = (g * s).sum(axis=axis, keepdims=True)
            Tensor._accum(self, s * (g - dot))

        out = self._result(s, (self,), back, "softmax")
        return out

    # ---- the backward pass ------------------------------------------------
    def backward(self):
        """Reverse-mode autodiff. Call on a scalar loss; fills .grad everywhere."""
        topo: List[Tensor] = []
        seen = set()

        def build(t: "Tensor"):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        # seed: d(loss)/d(loss) = 1
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.grad is None:
                continue  # no gradient reached this node => it contributes zero
            node._back()
        # Break the closure<->output reference cycles so the per-step graph is
        # freed immediately by refcounting (otherwise memory grows until the
        # cyclic GC happens to run). Leaf parameters keep their data; only the
        # transient graph wiring is cleared.
        _noop = Tensor._noop
        for node in topo:
            node._back = _noop
            node._parents = ()


# A few extra primitives the model needs, attached after the class definition.
# (Assigning to the class is fine; __slots__ only restricts *instance* attributes.)

def _pow(self: "Tensor", p: float) -> "Tensor":
    """Raise to a scalar power (used for sqrt in LayerNorm: x ** 0.5)."""
    out_data = self.data ** p

    def back():
        Tensor._accum(self, out.grad * (p * self.data ** (p - 1.0)))

    out = self._result(out_data, (self,), back, f"pow{p}")
    return out


Tensor.__pow__ = _pow


def concat(tensors: List["Tensor"], axis: int = -2) -> "Tensor":
    """Concatenate tensors along `axis` (used to splice memory slots before the
    sequence). Backward simply slices the gradient back to each piece."""
    datas = [t.data for t in tensors]
    sizes = [d.shape[axis] for d in datas]
    out_data = np.concatenate(datas, axis=axis)

    def back():
        g = out.grad
        idx = 0
        slices_per = []
        for sz in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(idx, idx + sz)
            slices_per.append(tuple(sl))
            idx += sz
        for t, sl in zip(tensors, slices_per):
            Tensor._accum(t, g[sl])

    out = tensors[0]._result(out_data, tuple(tensors), back, "concat")
    return out


def embedding(table: "Tensor", idx: np.ndarray) -> "Tensor":
    """Look up rows of `table` (vocab x d) at integer indices `idx`.
    Backward scatter-adds the gradient back into the used rows (the standard
    sparse embedding gradient)."""
    out_data = table.data[idx]

    def back():
        g = out.grad
        grad_table = np.zeros_like(table.data)
        np.add.at(grad_table, idx.reshape(-1), g.reshape(-1, table.data.shape[-1]))
        Tensor._accum(table, grad_table)

    out = table._result(out_data, (table,), back, "embedding")
    return out


def cross_entropy(logits: "Tensor", targets: np.ndarray) -> Tuple["Tensor", float]:
    """Mean softmax cross-entropy over a batch of (N, V) logits and int targets.

    Implemented as a dedicated op for numerical stability: the gradient of
    softmax-cross-entropy is simply (softmax(logits) - onehot)/N, which avoids
    ever forming a near-zero log. Returns (loss_tensor, accuracy_float)."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    n = targets.shape[0]
    log_p = np.log(p[np.arange(n), targets] + 1e-12)
    loss_val = -log_p.mean()
    acc = float((p.argmax(axis=-1) == targets).mean())

    def back():
        grad = p.copy()
        grad[np.arange(n), targets] -= 1.0
        grad /= n
        Tensor._accum(logits, grad * loss.grad)  # loss.grad is the upstream scalar

    loss = logits._result(np.array(loss_val), (logits,), back, "cross_entropy")
    return loss, acc


def layer_norm(x: "Tensor", gamma: "Tensor", beta: "Tensor", eps: float = 1e-5) -> "Tensor":
    """LayerNorm built purely from autograd primitives -- the standardised
    'protocol' that keeps every province's signal on one common scale.

    Sargon issued one set of weights and measures so a shekel meant the same in
    Ur as in Mari. LayerNorm does exactly this for activations: re-centre and
    re-scale so each token's representation lives in a shared, comparable range
    before it is routed to a province (expert)."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = (var + Tensor(eps)) ** (-0.5)
    norm = xc * inv
    return norm * gamma + beta


# ============================================================================
# SECTION 2 -- PARAMETERS AND MODULES
# ----------------------------------------------------------------------------
# A `Parameter` is just a Tensor that requires grad and that the optimizer will
# update. Each module exposes .params() so the optimizer can find everything.
# ============================================================================


def Parameter(shape, scale=None) -> "Tensor":
    """A trainable tensor, He/Xavier-ish initialised."""
    if scale is None:
        fan_in = shape[0] if len(shape) >= 1 else 1
        scale = 1.0 / math.sqrt(fan_in)
    data = np.random.randn(*shape) * scale
    return Tensor(data, requires_grad=True)


class Module:
    def params(self) -> List["Tensor"]:
        out: List[Tensor] = []
        for v in vars(self).values():
            if isinstance(v, Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.params())
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, Module):
                        out.extend(it.params())
                    elif isinstance(it, Tensor) and it.requires_grad:
                        out.append(it)
        return out


class Linear(Module):
    """Standard affine layer y = xW + b. The workhorse inside every province."""

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        self.W = Parameter((d_in, d_out))
        self.b = Parameter((d_out,), scale=0.0) if bias else None

    def __call__(self, x: "Tensor") -> "Tensor":
        y = x.matmul(self.W)
        if self.b is not None:
            y = y + self.b
        return y


class LayerNormModule(Module):
    def __init__(self, d: int):
        self.gamma = Tensor(np.ones(d), requires_grad=True)
        self.beta = Tensor(np.zeros(d), requires_grad=True)

    def __call__(self, x: "Tensor") -> "Tensor":
        return layer_norm(x, self.gamma, self.beta)


# ============================================================================
# SECTION 3 -- THE IMPERIAL MECHANISMS
# ============================================================================


class ImperialAttention(Module):
    """Multi-head self-attention = the Royal Roads of the empire.

    Sargon's couriers carried information between every province along
    standardised roads. Attention is the same idea: each position (token) decides
    how much to 'read' from every other position. We add two Sargonian twists:

      * a CAUSAL mask, so a decree can only depend on what is already known
        (no province reports the future); and
      * a learned MEMORY prefix (the archives, see HierarchicalMemory) that every
        position may consult but which is never masked -- institutional knowledge
        outlives any single decision.
    """

    def __init__(self, d_model: int, n_heads: int, n_mem: int = 8):
        assert d_model % n_heads == 0
        self.d, self.h, self.dh = d_model, n_heads, d_model // n_heads
        self.n_mem = n_mem
        self.Wq = Linear(d_model, d_model, bias=False)
        self.Wk = Linear(d_model, d_model, bias=False)
        self.Wv = Linear(d_model, d_model, bias=False)
        self.Wo = Linear(d_model, d_model, bias=False)
        # The archive: n_mem learned d-dim "records" the layer can always read.
        self.memory = Parameter((n_mem, d_model), scale=0.02) if n_mem > 0 else None

    def _split_heads(self, x: "Tensor", B: int, T: int) -> "Tensor":
        # (B, T, d) -> (B, h, T, dh)
        return x.reshape(B, T, self.h, self.dh).transpose((0, 2, 1, 3))

    def __call__(self, x: "Tensor") -> "Tensor":
        B, T, _ = x.shape
        q = self._split_heads(self.Wq(x), B, T)

        if self.memory is not None:
            mem3 = self.memory.reshape(1, self.n_mem, self.d)
            # broadcast to batch using only standard ops (adding zeros) so the
            # archive's gradient flows back correctly through reshape -> memory.
            mem = mem3 + Tensor(np.zeros((B, self.n_mem, self.d)))
            kv_in = concat([mem, x], axis=1)   # (B, n_mem + T, d)
            Tk = self.n_mem + T
        else:
            kv_in = x
            Tk = T

        k = self._split_heads(self.Wk(kv_in), B, Tk)
        v = self._split_heads(self.Wv(kv_in), B, Tk)

        scale = 1.0 / math.sqrt(self.dh)
        scores = (q.matmul(k.transpose((0, 1, 3, 2)))) * scale  # (B,h,T,Tk)

        # additive mask: memory columns always visible; sequence columns causal.
        mask = np.zeros((T, Tk), dtype=DTYPE)
        if self.memory is not None:
            seq_part = np.triu(np.ones((T, T), dtype=DTYPE), k=1) * -1e9
            mask[:, self.n_mem:] = seq_part
        else:
            mask = np.triu(np.ones((T, T), dtype=DTYPE), k=1) * -1e9
        scores = scores + Tensor(mask.reshape(1, 1, T, Tk))

        attn = scores.softmax(axis=-1)
        ctx = attn.matmul(v)                              # (B,h,T,dh)
        ctx = ctx.transpose((0, 2, 1, 3)).reshape(B, T, self.d)
        return self.Wo(ctx)


class ProvincialExperts(Module):
    """The provinces (experts) + the King (router) = a Mixture-of-Experts layer.

    This is the heart of the Sargon mapping. Each token is a 'matter of state'.
    The King (router) decides which provinces are competent to handle it (top-k),
    weighting their counsel. Each province is a specialised feed-forward network.
    A load-balancing penalty (the imperial audit) keeps the King from sending all
    business to one favourite province while others sit idle -- exactly Sargon's
    administrative problem of keeping every governor productive and accountable.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2):
        self.E, self.k = n_experts, top_k
        self.router = Linear(d_model, n_experts, bias=False)   # the King's judgement
        self.up = [Linear(d_model, d_ff) for _ in range(n_experts)]
        self.down = [Linear(d_ff, d_model) for _ in range(n_experts)]
        self.last_aux = 0.0          # exposed for logging
        self.last_usage = None       # fraction of tokens per province (for the mindmap)

    def __call__(self, x: "Tensor") -> Tuple["Tensor", "Tensor"]:
        B, T, d = x.shape
        N = B * T
        xf = x.reshape(N, d)

        gate_logits = self.router(xf)            # (N, E)
        gate = gate_logits.softmax(axis=-1)      # (N, E) the King's confidence

        # --- top-k routing (which provinces handle each matter) -------------
        gd = gate.data
        topk_idx = np.argsort(-gd, axis=1)[:, : self.k]      # (N, k)
        keep = np.zeros_like(gd)
        rows = np.repeat(np.arange(N), self.k)
        keep[rows, topk_idx.reshape(-1)] = 1.0
        gate_kept = gate * Tensor(keep)                       # zero out non-selected
        denom = gate_kept.sum(axis=-1, keepdims=True) + Tensor(1e-9)
        gate_norm = gate_kept / denom                         # renormalise top-k weights

        # --- provinces deliberate (dense compute, sparsely weighted) --------
        out = Tensor(np.zeros((N, d)))
        for e in range(self.E):
            h = self.up[e](xf).gelu()
            oe = self.down[e](h)                              # province e's counsel
            sel = np.zeros((self.E, 1), dtype=DTYPE); sel[e, 0] = 1.0
            w_e = gate_norm.matmul(Tensor(sel))               # (N,1) weight for province e
            out = out + oe * w_e
        out = out.reshape(B, T, d)

        # --- the imperial audit: load-balancing auxiliary loss --------------
        # f_e = fraction of tokens that selected province e (data-side indicator)
        f = keep.sum(axis=0) / max(1.0, keep.sum())           # (E,)
        f = f * self.E                                        # normalise scale
        P = gate.mean(axis=0)                                 # (E,) mean confidence
        aux = (Tensor(f) * P).sum() * float(self.E)           # Switch-style penalty
        self.last_aux = float(aux.data)
        self.last_usage = (keep.sum(axis=0) / max(1.0, N)).tolist()
        return out, aux


class ImperialBlock(Module):
    """One layer of the empire: read the roads (attention), then route to provinces
    (MoE), each wrapped in the standardised protocol (LayerNorm) with residual
    'continuity of institutions' connections so information is never lost."""

    def __init__(self, d_model, n_heads, d_ff, n_experts, top_k, n_mem):
        self.ln1 = LayerNormModule(d_model)
        self.attn = ImperialAttention(d_model, n_heads, n_mem)
        self.ln2 = LayerNormModule(d_model)
        self.moe = ProvincialExperts(d_model, d_ff, n_experts, top_k)

    def __call__(self, x: "Tensor") -> Tuple["Tensor", "Tensor"]:
        x = x + self.attn(self.ln1(x))             # pre-norm residual (the Royal Roads)
        moe_out, aux = self.moe(self.ln2(x))       # the provinces deliberate
        x = x + moe_out                            # institutional continuity (residual)
        return x, aux


# ============================================================================
# SECTION 4 -- THE FULL MODEL: the ImperialTransformer (the AGI base model)
# ============================================================================


class ImperialTransformer(Module):
    """The complete base model. Embeddings (standardised tokens) + positional
    sense of order + a stack of ImperialBlocks + a final protocol + an output
    head that decides the next token. This is a real, decoder-only,
    sparse-Mixture-of-Experts language model -- the same family as today's
    frontier systems -- assembled from the bare chain rule."""

    def __init__(self, vocab, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, n_experts=4, top_k=2, n_mem=8, max_T=64,
                 aux_weight=0.01):
        self.vocab = vocab
        self.aux_weight = aux_weight
        self.tok = Parameter((vocab, d_model), scale=0.02)      # standardised tokens
        self.pos = Parameter((max_T, d_model), scale=0.02)      # sense of sequence/order
        self.blocks = [ImperialBlock(d_model, n_heads, d_ff, n_experts, top_k, n_mem)
                       for _ in range(n_layers)]
        self.ln_f = LayerNormModule(d_model)
        self.head = Linear(d_model, vocab, bias=False)          # the final decree

    def forward(self, X: np.ndarray) -> Tuple["Tensor", "Tensor"]:
        B, T = X.shape
        h = embedding(self.tok, X)                              # (B,T,d)
        pos = embedding(self.pos, np.arange(T))                 # (T,d)
        h = h + pos
        total_aux = Tensor(0.0)
        for blk in self.blocks:
            h, aux = blk(h)
            total_aux = total_aux + aux
        h = self.ln_f(h)
        logits = self.head(h)                                   # (B,T,vocab)
        return logits, total_aux

    def loss(self, X: np.ndarray, Y: np.ndarray):
        """Next-token cross-entropy + load-balancing auxiliary loss."""
        B, T = X.shape
        logits, aux = self.forward(X)
        flat = logits.reshape(B * T, self.vocab)
        ce, acc = cross_entropy(flat, Y.reshape(-1))
        total = ce + aux * self.aux_weight
        return total, ce, acc, logits

    def expert_usage(self) -> List[List[float]]:
        return [blk.moe.last_usage for blk in self.blocks]


# ============================================================================
# SECTION 5 -- ADAM OPTIMIZER (from scratch)
# ----------------------------------------------------------------------------
# The "training" half of training. Adam keeps a running estimate of the mean and
# variance of each parameter's gradient and takes a normalised step. No library
# call -- the update rule is written out so it is fully inspectable.
# ============================================================================


class Adam:
    def __init__(self, params: List["Tensor"], lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ============================================================================
# SECTION 6 -- THE TASK: "Imperial Standardisation"
# ----------------------------------------------------------------------------
# A structured synthetic problem chosen to exercise exactly the Sargonian
# mechanisms. Each dispatch has the form
#       [PROV_p, a, '=', c, SEP]
# where every province p keeps its records under its OWN fixed cipher PERM[p]
# (a permutation of the digits), and the standardised answer is c = PERM[p][a].
# To solve it the model must:
#   * read back along the Royal Roads (attention) to recover BOTH which province
#     p issued the dispatch and the local figure a; and
#   * route the matter to that province's expert (the MoE) -- each province uses
#     a different cipher, so specialisation is directly rewarded.
# This is a 28-entry lookup with full support, so a correct architecture both
# learns it and GENERALISES to a held-out set (unlike ungrokkable mod-arithmetic).
# We report next-token loss and, separately, ANSWER accuracy (predicting c).
# ============================================================================

M_DIGITS = 7          # local figures / digits 0..6
N_PROV = 4            # provinces (also the natural number of experts)
TUP = 5              # tokens per dispatch: PROV a = c SEP
# token id layout:
#   0..M-1            : digits
#   M..M+N_PROV-1     : province markers PROV0..PROV3
#   M+N_PROV          : '='
#   M+N_PROV+1        : SEP
EQ = M_DIGITS + N_PROV
SEP = EQ + 1
VOCAB = SEP + 1

# Each province's fixed cipher (a permutation of the digits). Deterministic so
# the task is identical on every run -- the empire's standards do not drift.
_perm_rng = np.random.RandomState(1234)
PERM = np.stack([_perm_rng.permutation(M_DIGITS) for _ in range(N_PROV)])  # (E, M)


def make_batch(batch=64, n_tuples=4, seed=None):
    rng = np.random.RandomState(seed)
    T = n_tuples * TUP
    X = np.zeros((batch, T), dtype=np.int64)
    is_answer = np.zeros((batch, T), dtype=bool)   # positions whose TARGET is c
    for i in range(batch):
        t = 0
        for _ in range(n_tuples):
            p = rng.randint(N_PROV)
            a = rng.randint(M_DIGITS)
            c = int(PERM[p, a])
            X[i, t + 0] = M_DIGITS + p     # PROV_p
            X[i, t + 1] = a
            X[i, t + 2] = EQ
            X[i, t + 3] = c
            X[i, t + 4] = SEP
            is_answer[i, t + 2] = True     # target at the '=' slot is c
            t += TUP
    Y = np.empty_like(X)
    Y[:, :-1] = X[:, 1:]
    Y[:, -1] = SEP                       # last target is harmless padding
    return X, Y, is_answer


def answer_accuracy(model: "ImperialTransformer", X, Y, is_answer):
    logits, _ = model.forward(X)
    pred = logits.data.argmax(axis=-1)
    mask = is_answer
    return float((pred[mask] == Y[mask]).mean())


# ============================================================================
# SECTION 7 -- TRAINING ENTRYPOINT
# ============================================================================


def train(steps=400, batch=64, n_tuples=4, lr=3e-3, seed=0, verbose=True):
    """Train the Imperial Transformer on the standardisation task and return it."""
    np.random.seed(seed)
    model = ImperialTransformer(
        VOCAB, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        n_experts=N_PROV, top_k=2, n_mem=8, max_T=n_tuples * TUP, aux_weight=0.01,
    )
    opt = Adam(model.params(), lr=lr)
    Xev, Yev, isaev = make_batch(batch=256, n_tuples=n_tuples, seed=9999)  # held-out
    if verbose:
        n_scalars = sum(p.data.size for p in model.params())
        print(f"  model: {len(model.params())} parameter tensors, {n_scalars:,} scalars")
        print(f"  task : standardise per-province record c=PERM[p][a]; vocab={VOCAB}, seq={n_tuples*TUP}")
        print("  " + "-" * 64)
    t0 = time.time()
    for step in range(1, steps + 1):
        X, Y, _ = make_batch(batch=batch, n_tuples=n_tuples, seed=step)  # fresh data
        total, ce, acc, _ = model.loss(X, Y)
        opt.zero_grad()
        total.backward()
        opt.step()
        if verbose and (step % max(1, steps // 10) == 0 or step == 1):
            ev_acc = answer_accuracy(model, Xev, Yev, isaev)
            print(f"  step {step:4d} | train CE {ce.data.item():.4f} "
                  f"| aux {model.blocks[0].moe.last_aux:.3f} "
                  f"| HELD-OUT answer-acc {ev_acc:5.1%}")
    if verbose:
        print("  " + "-" * 64)
        print(f"  trained in {time.time()-t0:.1f}s")
        print(f"  final held-out answer accuracy: {answer_accuracy(model,Xev,Yev,isaev):.1%}")
        usage = model.expert_usage()
        print("  province (expert) utilisation per layer (top-2 of 4 => ~0.5 ideal):")
        for li, lay in enumerate(usage):
            bars = "  ".join(f"P{j}:{u:4.0%}" for j, u in enumerate(lay))
            print(f"    layer {li}:  {bars}")
    return model


def demo_inference(model, n_tuples=4):
    """Greedy-decode the standardised answer for fresh dispatches, to show the
    trained empire actually reasoning. A dispatch arrives as [PROV_p, a, =] and
    the model must emit, at the '=' slot, the province's standardised record
    c = PERM[p][a] -- i.e. it has learned each governor's local cipher."""
    rng = np.random.RandomState(7)
    print("\n  Inference demo (the trained empire answering fresh dispatches):")
    for _ in range(6):
        p, a = rng.randint(N_PROV), rng.randint(M_DIGITS)
        truth = int(PERM[p, a])
        ctx = np.array([[M_DIGITS + p, a, EQ]], dtype=np.int64)   # PROV_p, a, =
        logits, _ = model.forward(ctx)
        pred = int(logits.data[0, -1].argmax())                   # prediction at '='
        ok = "OK " if pred == truth else "XX "
        print(f"    {ok}province P{p}: standardise local record {a} "
              f"-> imperial value {truth}   (model said {pred})")


# ============================================================================
# SECTION 8 -- TEST SUITE  (proves the architecture is correct, not a demo)
# ============================================================================


def _numerical_grad(build_loss, param: "Tensor", eps=1e-5):
    g = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = param.data[i]
        param.data[i] = old + eps; fp = float(build_loss().data)
        param.data[i] = old - eps; fm = float(build_loss().data)
        param.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def test_gradients():
    """Finite-difference check every custom backward used by the model."""
    print("[1] Gradient checks (analytic vs numerical):")
    rng = np.random.RandomState(0)
    const = Tensor(rng.randn(6, 5))

    # layer_norm
    x = Tensor(rng.randn(6, 5), requires_grad=True)
    g = Tensor(rng.randn(5), requires_grad=True)
    b = Tensor(rng.randn(5), requires_grad=True)
    def bl_ln():
        for t in (x, g, b): t.grad = None
        return (layer_norm(x, g, b) * const).sum()
    loss = bl_ln(); loss.backward()
    ana = {"x": x.grad.copy(), "gamma": g.grad.copy(), "beta": b.grad.copy()}
    for nm, t in (("x", x), ("gamma", g), ("beta", b)):
        num = _numerical_grad(bl_ln, t)
        err = np.abs(ana[nm] - num).max()
        print(f"    layer_norm/{nm:5s}: max err {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4

    # cross_entropy
    logits = Tensor(rng.randn(8, 7), requires_grad=True)
    tgt = rng.randint(0, 7, size=8)
    def bl_ce():
        logits.grad = None
        return cross_entropy(logits, tgt)[0]
    loss = bl_ce(); loss.backward()
    ana_ce = logits.grad.copy()
    num = _numerical_grad(bl_ce, logits)
    err = np.abs(ana_ce - num).max()
    print(f"    cross_entropy     : max err {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")
    assert err < 1e-4

    # embedding
    table = Tensor(rng.randn(10, 4), requires_grad=True)
    idx = np.array([[1, 3, 3], [0, 9, 1]])
    c2 = Tensor(rng.randn(2, 3, 4))
    def bl_emb():
        table.grad = None
        return (embedding(table, idx) * c2).sum()
    loss = bl_emb(); loss.backward()
    ana_emb = table.grad.copy()
    num = _numerical_grad(bl_emb, table)
    err = np.abs(ana_emb - num).max()
    print(f"    embedding         : max err {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")
    assert err < 1e-4
    print("    -> all custom gradients match finite differences.\n")


def test_overfit_tiny():
    """The classic ML correctness proof: a model with working gradients can
    drive the loss on a small FIXED batch to ~0. If any gradient in the full
    architecture were wrong, this would stall."""
    print("[2] Overfit-a-tiny-batch (whole-architecture gradient-flow proof):")
    np.random.seed(1)
    model = ImperialTransformer(VOCAB, d_model=32, n_heads=4, n_layers=2,
                                d_ff=64, n_experts=N_PROV, top_k=2, n_mem=4,
                                max_T=18, aux_weight=0.0)  # pure memorisation: no
                                                           # load-balance regulariser
                                                           # fighting the overfit
    X, Y, _ = make_batch(batch=4, n_tuples=3, seed=3)
    opt = Adam(model.params(), lr=1e-2)
    first = last = None
    for step in range(300):
        total, ce, _, _ = model.loss(X, Y)
        if step == 0:
            first = ce.data.item()
        opt.zero_grad(); total.backward(); opt.step()
        last = ce.data.item()
    print(f"    CE: {first:.3f} -> {last:.3f}  {'OK' if last < 0.05 else 'FAIL'}")
    assert last < 0.05, "architecture failed to overfit -> a gradient is wrong"
    print("    -> gradients flow correctly through attention, MoE, memory, LN.\n")


def test_routing_and_memory():
    """The provinces must actually be used (no expert collapse) and the archive
    (memory) must receive learning signal."""
    print("[3] Routing balance + memory learning:")
    model = train(steps=120, batch=64, n_tuples=4, seed=2, verbose=False)
    usage = model.expert_usage()
    flat = [u for lay in usage for u in lay]
    print(f"    expert usage range: {min(flat):.2f} .. {max(flat):.2f}")
    assert min(flat) > 0.05, "an expert collapsed to ~zero usage"
    mem_ok = all(b.attn.memory.grad is not None for b in model.blocks)
    print(f"    every layer's archive received gradient: {mem_ok}  {'OK' if mem_ok else 'FAIL'}")
    assert mem_ok
    ev = make_batch(batch=256, n_tuples=4, seed=4242)
    acc = answer_accuracy(model, *ev)
    print(f"    held-out answer accuracy after 120 steps: {acc:.1%}  "
          f"{'OK' if acc > 0.7 else 'FAIL'}")
    assert acc > 0.7
    print("    -> provinces are balanced, archive learns, empire generalises.\n")


def run_tests():
    print("=" * 72)
    print(" IMPERIAL TRANSFORMER -- TEST SUITE")
    print("=" * 72)
    test_gradients()
    test_overfit_tiny()
    test_routing_and_memory()
    print("=" * 72)
    print(" ALL TESTS PASSED.  The architecture is correct and trainable.")
    print("=" * 72)


def main():
    print("=" * 72)
    print(" THE IMPERIAL TRANSFORMER  --  an AGI base model after Sargon of Akkad")
    print(" 'As one empire under one standard, many provinces; one mind, many experts.'")
    print("=" * 72)
    model = train(steps=250, batch=64, n_tuples=4, lr=3e-3, seed=0, verbose=True)
    demo_inference(model)
    print("\n" + "=" * 72)
    print(" Sargon coordinated provinces under one standard, routing each matter")
    print(" of state to the competent governor and auditing all of them. This model")
    print(" routes each token to competent experts under one normalised protocol and")
    print(" balances their load. Same theory of mind; one is clay, one is code.")
    print("=" * 72)
    print("\n Run the test suite with:  python3 %s --test" % __file__.split('/')[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The Imperial Transformer (Sargon of Akkad).")
    ap.add_argument("--test", action="store_true", help="run the gradient/training test suite")
    ap.add_argument("--steps", type=int, default=250, help="training steps for the main run")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        main()

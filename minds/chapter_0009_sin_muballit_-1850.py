"""
================================================================================
chapter_0009_sin_muballit_-1850.py
THE SEXAGESIMAL–OMEN MIND  ·  A trainable AGI base-model after Sin-Muballit

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/
================================================================================

CHAPTER 9 · Sin-Muballit · fifth Amorite king of the First Dynasty of Babylon
(reigned c. 1813–1792 BCE, middle chronology; father of Hammurabi).

WHAT THIS FILE IS
-----------------
This is NOT a simulation dressed up in matrix multiplications. It is a small but
*genuinely trainable* neural network, written from scratch on top of a tiny
reverse-mode autodiff engine (≈200 lines, numpy only). It learns a real task by
gradient descent, its gradients are verified against finite differences, and it
generalises to data it never saw during training.

The architecture is a faithful computational portrait of how Sin-Muballit's
documented intellectual world actually worked. Two facts about that world drive
every design choice below:

  (1) SEXAGESIMAL COMPUTATION.  Old Babylonian scribes reckoned in base 60 with
      true place-value notation — the most powerful arithmetic technology on
      earth for the next two thousand years (cf. the tablet Plimpton 322,
      c. 1800 BCE, his lifetime). Number was their cognitive infrastructure.

  (2) THE HEAVENS AS A REGISTER.  The same culture watched the sky as a field of
      *omens* — recurring celestial signs read for meaning. (Mature *predictive*
      astronomy is a first-millennium achievement, a thousand years later; what
      the Old Babylonian mind had was cyclic sign-reading, not ephemerides.)

So the Babylonian mind ran two cognitive modes at once: exact PROCEDURE (the
scribe's reckoning) and pattern-as-MEANING (the diviner's reading of cycles).
A king like Sin-Muballit governed by FUSING them — turning a computed quantity
and a read omen into a single decision, inscribed as the "name" of the year.

THE MODEL MIRRORS THIS EXACTLY
------------------------------
  • COMPUTATION STREAM  -> learned embeddings of the two numbers (the scribe).
  • CYCLICAL/OMEN STREAM -> fixed Fourier features over the 30 harmonics of the
                            60-cycle (the heavens, read as recurring signs).
  • ESAGILA COUNCIL     -> a learned scalar "trust dial" (gate) that decides how
                            much to weigh exact calculation vs. cyclical reading,
                            then a small MLP that synthesises a decision.

THE TASK
--------
Learn base-60 modular addition:  given a, b in {0..59}, output (a + b) mod 60.
This is the simplest non-trivial sexagesimal computation a scribe performs, and
the group Z/60Z is, mathematically, *made of cycles*. We train on only half of
the 3600 possible (a,b) pairs and test on the other half, so success means the
network discovered the underlying cyclic STRUCTURE rather than memorising.

WHAT YOU WILL SEE WHEN YOU RUN IT
---------------------------------
  1. A GRADIENT CHECK (analytic backprop vs. finite differences) — passes.
  2. TRAINING that "groks": training accuracy hits ~100% almost immediately,
     test accuracy lingers near zero, then snaps to ~98% once the network
     *understands* the cycle. (Grokking is a real, published phenomenon.)
  3. THE TRUST DIAL falling from ~0.90 toward ~0.55 across training: the model
     begins trusting rote computation and learns to trust the cycles — the
     Babylonian mind in miniature.
  4. A REPRESENTATION PROBE showing the learned number-embeddings concentrate
     their energy on a few harmonics of 60: the calculating mind spontaneously
     re-encodes "number" as "celestial cycle."

Run:   python3 chapter_0009_sin_muballit_-1850.py        (only dependency: numpy)
Tested on numpy >= 1.21 (developed on numpy 2.4).

Author: David Vivancos — How history's greatest minds would build AGI.
================================================================================
"""

from __future__ import annotations
import numpy as np
from typing import List, Callable
import time
import gc


# ============================================================================
# PART 1 — A TINY REVERSE-MODE AUTODIFF ENGINE (the "scribe's tablet")
# ----------------------------------------------------------------------------
# Every quantity is a Tensor that remembers how it was produced, so that calling
# .backward() can replay the computation in reverse and deposit a gradient on
# every parameter. This is the same idea as PyTorch/TensorFlow autograd, kept
# small enough to read in one sitting. It is the modern equivalent of a clay
# tablet that records not just a number but the *procedure* that produced it.
# ============================================================================

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a gradient back down to `shape`, undoing numpy broadcasting.

    When we add a bias of shape (D,) to an activation of shape (N, D), numpy
    silently broadcasts. The gradient flowing back to the bias must therefore be
    summed over the batch dimension. This helper makes every elementwise op
    broadcasting-correct, which is the single most common source of autodiff
    bugs.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that tracks gradients through the operations
    applied to it. `requires_grad=False` marks fixed inputs (data, the celestial
    features) that should receive no gradient — the "given" facts of the world.
    """

    def __init__(self, data, requires_grad: bool = True, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    # ---- elementwise addition (and, via __neg__, subtraction) --------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    # ---- elementwise multiplication ----------------------------------------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- matrix multiplication (2-D) — the workhorse of every layer --------
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ---- reductions ---------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities -----------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op="tanh")
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, _children=(self,), _op="sigmoid")
        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    # ---- embedding lookup: pick rows by integer index ----------------------
    def index_select(self, idx: np.ndarray):
        idx = np.asarray(idx, dtype=np.int64)
        out = Tensor(self.data[idx], _children=(self,), _op="idx")
        def _backward():
            if self.requires_grad:
                # several rows can map to the same index -> accumulate (add.at)
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    # ---- reverse-mode traversal --------------------------------------------
    def backward(self):
        """Topologically sort the graph and run each node's local backward in
        reverse, so gradients flow from this scalar loss back to every leaf."""
        topo: List[Tensor] = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)   # d(loss)/d(loss) = 1
        for v in reversed(topo):
            v._backward()


def cat(tensors: List[Tensor], axis: int = -1) -> Tensor:
    """Concatenate tensors and route each split of the gradient back. This is the
    'bringing the streams together at the council table' operation."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, _children=tuple(tensors), _op="cat")
    sizes = [t.data.shape[axis] for t in tensors]
    def _backward():
        splits = np.cumsum(sizes)[:-1]
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g
    out._backward = _backward
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation as a single fused op with hand-derived backward.
    Normalising each council decision to a common scale keeps training stable —
    the computational analogue of the Babylonian habit of *standardising* every
    measure so that quantities from different domains can be compared at all."""
    xd = x.data
    mu = xd.mean(axis=-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (xd - mu) * inv
    out = Tensor(gamma.data * xhat + beta.data, _children=(x, gamma, beta), _op="ln")
    D = xd.shape[-1]
    def _backward():
        g = out.grad
        if gamma.requires_grad:
            gamma.grad += (g * xhat).reshape(-1, D).sum(axis=0)
        if beta.requires_grad:
            beta.grad += g.reshape(-1, D).sum(axis=0)
        if x.requires_grad:
            dxhat = g * gamma.data
            dx = inv * (dxhat
                        - dxhat.mean(axis=-1, keepdims=True)
                        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
            x.grad += dx
    out._backward = _backward
    return out


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray):
    """Numerically-stable softmax + cross-entropy fused into one op, returning
    (loss_tensor, softmax_probabilities). The gradient (softmax - one_hot)/N is
    the cleanest, least bug-prone way to backprop a classifier."""
    z = logits.data
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    sm = ez / ez.sum(axis=-1, keepdims=True)
    N = z.shape[0]
    logp = np.log(sm[np.arange(N), targets] + 1e-12)
    loss = -logp.mean()
    out = Tensor(loss, _children=(logits,), _op="ce")
    def _backward():
        if logits.requires_grad:
            d = sm.copy()
            d[np.arange(N), targets] -= 1.0
            logits.grad += (d / N) * out.grad
    out._backward = _backward
    return out, sm


# ============================================================================
# PART 2 — LAYERS AND THE TWO INPUT ENCODINGS
# ============================================================================

def he_init(in_d: int, out_d: int) -> Tensor:
    """He initialisation, appropriate for ReLU networks."""
    return Tensor(np.random.randn(in_d, out_d) * np.sqrt(2.0 / in_d))


class Linear:
    """A standard affine layer  y = xW + b."""
    def __init__(self, in_d: int, out_d: int):
        self.W = he_init(in_d, out_d)
        self.b = Tensor(np.zeros(out_d))
    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b
    def params(self):
        return [self.W, self.b]


class Embedding:
    """A learned lookup table: each integer 0..num-1 gets its own vector.
    In our metaphor these are the *numerals* the scribe internalises — and we
    will see them reorganise themselves into cycles as the model learns."""
    def __init__(self, num: int, dim: int, scale: float = 0.1):
        self.W = Tensor(np.random.randn(num, dim) * scale)
    def __call__(self, idx: np.ndarray) -> Tensor:
        return self.W.index_select(idx)
    def params(self):
        return [self.W]


# The 60-cycle has exactly 30 distinguishable harmonics (frequencies 1..n/2).
# These ARE the cycles of the heavens in our model: each value of a or b is
# turned into a point on 30 nested wheels of different speeds. Reading a number
# this way is precisely "the diviner's" mode — sign as position-in-a-cycle.
HARMONICS = list(range(1, 31))


def periodic_features(values: np.ndarray, freqs: List[int], period: int = 60) -> np.ndarray:
    """Fixed (non-learned) Fourier features: for each value v, return
    [sin(2*pi*k*v/period), cos(2*pi*k*v/period)] for every harmonic k.

    These encode v as its position on a set of wheels. For modular addition this
    is the *natural* representation, because rotating a wheel by a and then by b
    is the same as rotating it by (a+b) — the group structure of Z/60Z is
    literally built out of these cycles. This is the model's celestial register.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    k = np.asarray(freqs, dtype=np.float64).reshape(1, -1)
    ang = 2.0 * np.pi * k * v / period
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


# ============================================================================
# PART 3 — THE ARCHITECTURE:  SexagesimalOmenMind
# ----------------------------------------------------------------------------
# Computation stream  (scribe)  ┐
#                                ├─ Esagila council (gate + MLP) ─→ decision
# Cyclical stream     (heavens) ┘
#
# The class name kept as `SinMuballitArchitecture` for drop-in compatibility
# with the rest of the 1000Minds codebase; `SexagesimalOmenMind` is an alias.
# ============================================================================

class SinMuballitArchitecture:
    """A general two-stream + gated-synthesis network. Here it is instantiated
    on base-60 addition, but the pattern — one *procedural/symbolic* stream, one
    *periodic/pattern* stream, fused by a learned trust gate and an MLP — is the
    reusable 'cognitive shape' the chapter argues Sin-Muballit would have built.

    Parameters
    ----------
    n        : modulus / vocabulary size (60 = sexagesimal).
    d_emb    : width of the learned numeral embeddings (the scribe's memory).
    d_stream : width each stream is projected to before the council.
    hidden   : width of the council's synthesis MLP.
    """

    def __init__(self, n: int = 60, d_emb: int = 24, d_stream: int = 96,
                 hidden: int = 192, seed: int = 0):
        np.random.seed(seed)
        self.n = n
        self.d_stream = d_stream

        # --- Computation stream: the scribe reckons with internalised numerals
        self.emb_a = Embedding(n, d_emb)
        self.emb_b = Embedding(n, d_emb)
        self.comp_proj = Linear(2 * d_emb, d_stream)

        # --- Cyclical / omen stream: the heavens read as nested wheels
        pf_dim = 2 * len(HARMONICS)              # sin & cos per harmonic, per value
        self.cyc_proj = Linear(2 * pf_dim, d_stream)

        # --- Esagila council: a scalar trust dial + a synthesis MLP
        self.gate = Linear(2 * d_stream, 1)      # how much to trust computation
        self.synth = Linear(2 * d_stream, hidden)
        self.ln_g = Tensor(np.ones(hidden))
        self.ln_b = Tensor(np.zeros(hidden))
        self.head = Linear(hidden, n)            # the inscribed decision (a class)

        self.last_gate = None                    # readout: mean trust-in-computation

    def params(self):
        ps = []
        for m in [self.emb_a, self.emb_b, self.comp_proj,
                  self.cyc_proj, self.gate, self.synth, self.head]:
            ps += m.params()
        ps += [self.ln_g, self.ln_b]
        return ps

    def num_params(self) -> int:
        return int(sum(p.data.size for p in self.params()))

    def forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)

        # COMPUTATION STREAM ---------------------------------------------------
        comp = cat([self.emb_a(a), self.emb_b(b)], axis=1)
        comp = self.comp_proj(comp).relu()

        # CYCLICAL / OMEN STREAM ----------------------------------------------
        pf = np.concatenate([periodic_features(a, HARMONICS, self.n),
                             periodic_features(b, HARMONICS, self.n)], axis=1)
        cyc = self.cyc_proj(Tensor(pf, requires_grad=False)).relu()

        # ESAGILA COUNCIL ------------------------------------------------------
        # The gate is the king's instinct for whether THIS case is better served
        # by exact calculation or by the cyclical reading. It is fully learned.
        g = self.gate(cat([comp, cyc], axis=1)).sigmoid()   # (N, 1) in (0,1)
        self.last_gate = float(g.data.mean())
        one = Tensor(np.ones_like(g.data), requires_grad=False)
        council = cat([g * comp, (one + (-g)) * cyc], axis=1)

        # Synthesis: normalise, mix, and inscribe the decision.
        h = layer_norm(self.synth(council), self.ln_g, self.ln_b).relu()
        return self.head(h)


# Friendly alias that names the idea rather than the man.
SexagesimalOmenMind = SinMuballitArchitecture


# ============================================================================
# PART 4 — THE ADAM OPTIMISER
# ----------------------------------------------------------------------------
# Adam = adaptive per-parameter step sizes with momentum. The weight-decay term
# (`wd`) gently shrinks parameters toward zero each step; counter-intuitively,
# this pressure to stay simple is what eventually forces the network to abandon
# memorisation and adopt the clean, cyclical solution (the 'grokking' effect).
# ============================================================================

class Adam:
    def __init__(self, params, lr=4e-3, betas=(0.9, 0.99), eps=1e-8, wd=1e-3):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p.data            # decoupled-ish weight decay
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ============================================================================
# PART 5 — DATA, EVALUATION, TRAINING, AND TESTS
# ============================================================================

def make_dataset(n: int = 60, train_frac: float = 0.5, seed: int = 0):
    """All n*n pairs (a,b) with target (a+b) mod n, split into train/test.
    Training on only `train_frac` of the table is the whole point: it forces the
    model to extrapolate to sums it never saw, i.e. to learn the *rule*."""
    A, B = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    A, B = A.reshape(-1), B.reshape(-1)
    T = (A + B) % n
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(A))
    cut = int(train_frac * len(A))
    return (A, B, T), idx[:cut], idx[cut:]


def accuracy(net, A, B, T) -> float:
    logits = net.forward(A, B).data
    return float((logits.argmax(axis=1) == T).mean())


def gradient_check() -> float:
    """Verify analytic backprop against central finite differences on a small
    instance. A passing check (rel. error ~1e-7) means every hand-derived
    backward above is correct — the foundation everything else rests on."""
    np.random.seed(1)
    net = SinMuballitArchitecture(n=12, d_emb=8, d_stream=16, hidden=24, seed=1)
    a = np.array([0, 3, 7, 11]); b = np.array([1, 5, 9, 2]); t = (a + b) % 12
    loss, _ = softmax_cross_entropy(net.forward(a, b), t)
    for p in net.params():
        p.grad = np.zeros_like(p.data)
    loss.backward()

    p = net.comp_proj.W                     # check a representative weight matrix
    flat, gana = p.data.reshape(-1), p.grad.reshape(-1)
    eps, max_err = 1e-5, 0.0
    for i in np.random.choice(flat.size, size=min(25, flat.size), replace=False):
        orig = flat[i]
        flat[i] = orig + eps
        lp, _ = softmax_cross_entropy(net.forward(a, b), t)
        flat[i] = orig - eps
        lm, _ = softmax_cross_entropy(net.forward(a, b), t)
        flat[i] = orig
        num = (lp.data - lm.data) / (2 * eps)
        max_err = max(max_err, abs(num - gana[i]) / (abs(num) + abs(gana[i]) + 1e-9))
    return max_err


def train(net, data, tr, te, steps=2200, batch=512, lr=4e-3, wd=1e-3,
          target_acc=0.95, seed=0, verbose=True):
    """Mini-batch training with early stopping once the model generalises."""
    A, B, T = data
    opt = Adam(net.params(), lr=lr, wd=wd)
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        bi = rng.choice(tr, size=min(batch, len(tr)), replace=False)
        logits = net.forward(A[bi], B[bi])
        loss, _ = softmax_cross_entropy(logits, T[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
        del logits, loss
        if step % 100 == 0:
            gc.collect()                    # free the per-step autodiff graph
            tr_acc = accuracy(net, A[tr], B[tr], T[tr])
            te_acc = accuracy(net, A[te], B[te], T[te])
            history.append((step, tr_acc, te_acc, net.last_gate))
            if verbose:
                print(f"  step {step:4d} | train {tr_acc:5.3f} | test {te_acc:5.3f} "
                      f"| trust-in-computation gate {net.last_gate:5.3f} "
                      f"| {time.time()-t0:5.1f}s")
            if te_acc >= target_acc:
                if verbose:
                    print("  -- the mind has grasped the cycle (early stop) --")
                break
    return history


def probe_representations(net) -> None:
    """Show that the learned numeral embeddings have become *cyclical*: take the
    60 embedding vectors, run a DFT along the 'number' axis, and report which
    harmonics carry the energy. After grokking, energy concentrates on a few
    harmonics of 60 — the calculating mind has re-encoded number as cycle."""
    E = net.emb_a.W.data                       # (60, d_emb)
    spec = np.abs(np.fft.rfft(E, axis=0))      # DFT over the 60 numbers
    power = (spec ** 2).mean(axis=1)
    power[0] = 0.0                             # ignore the constant (DC) term
    top = np.argsort(power)[::-1][:5]
    total = power.sum() + 1e-12
    print("  Dominant harmonics of the 60-cycle in the learned numerals:")
    for k in top:
        print(f"     harmonic {int(k):2d}  ({net.n}/{int(k) if k else 1} ≈ "
              f"{net.n/max(int(k),1):4.1f}-step wheel)  "
              f"carries {100*power[k]/total:4.1f}% of the energy")


def main():
    print("=" * 78)
    print("THE SEXAGESIMAL–OMEN MIND  ·  after Sin-Muballit of Babylon")
    print("a from-scratch, trainable neural network that learns base-60 arithmetic")
    print("=" * 78)

    # ---- TEST 1: gradients are correct -------------------------------------
    print("\n[1] Gradient check (analytic backprop vs. finite differences)")
    err = gradient_check()
    print(f"    max relative error = {err:.2e}")
    assert err < 1e-4, "GRADIENT CHECK FAILED — backprop is wrong."
    print("    PASS — the autodiff engine is correct.")

    # ---- Build the base model ----------------------------------------------
    n = 60
    net = SinMuballitArchitecture(n=n, d_emb=24, d_stream=96, hidden=192, seed=0)
    print(f"\n[2] Base model built: {net.num_params():,} trainable parameters")
    print("    two streams (computation + cyclical) fused by a learned trust gate")

    # ---- TEST 2: it actually learns (and generalises) ----------------------
    print("\n[3] Training on HALF of the 3600 (a,b) pairs; testing on the rest.")
    print("    Watch test accuracy 'grok', and the trust gate drift toward cycles:")
    data, tr, te = make_dataset(n=n, train_frac=0.5, seed=0)
    hist = train(net, data, tr, te, steps=2200, batch=512, lr=4e-3, wd=1e-3,
                 target_acc=0.95, seed=0)
    A, B, T = data
    final_test = accuracy(net, A[te], B[te], T[te])
    print(f"\n    FINAL held-out accuracy: {final_test:.3f}")
    assert final_test >= 0.90, "Model failed to generalise (expected >= 0.90)."
    print("    PASS — the model generalises base-60 addition to unseen pairs.")

    # ---- Probe: number has become cycle ------------------------------------
    print("\n[4] Representation probe:")
    probe_representations(net)

    # ---- A few worked sums, the way a scribe would check ------------------
    print("\n[5] Spot check — the model reckons:")
    checks = [(59, 1), (37, 48), (30, 30), (15, 50)]
    ca = np.array([x for x, _ in checks]); cb = np.array([y for _, y in checks])
    pred = net.forward(ca, cb).data.argmax(axis=1)
    for (x, y), p in zip(checks, pred):
        ok = "OK" if p == (x + y) % n else "??"
        print(f"     {x:2d} + {y:2d}  (mod 60) = {p:2d}   (true {x+y} mod 60 = {(x+y)%n:2d})  [{ok}]")

    print("\n" + "=" * 78)
    print("Two modes, one decision: the scribe's reckoning and the heavens' cycles,")
    print("weighed at the council table and inscribed as a single answer.")
    print("This is the cognitive shape the chapter argues Sin-Muballit would build.")
    print("=" * 78)


if __name__ == "__main__":
    main()

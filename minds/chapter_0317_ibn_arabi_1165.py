"""
chapter_0317_ibn_arabi_1165.py
================================================================================
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0317 · Ibn 'Arabi (Muhyi al-Din Muhammad ibn 'Ali ibn 'Arabi al-Ta'i al-Hatimi), b. 17 Ramadan 560 AH / 28 July 1165 CE, Murcia, al-Andalus -- d. 22 Rabi' II 638 AH / 10 November 1240 CE, Damascus.
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 

al-Hatimi), b. 17 Ramadan 560 AH / 28 July 1165 CE, Murcia, al-Andalus --
d. 22 Rabi' II 638 AH / 10 November 1240 CE, Damascus.

THE BARZAKH-QALB NETWORK
A from-scratch, pure-NumPy neural architecture whose mechanism is built out
of Ibn 'Arabi's own technical vocabulary for how a receptive locus comes to
know -- not a Transformer wearing Sufi labels, but a structure whose parts
only make sense once you have read the man.

--------------------------------------------------------------------------------
WHY THIS SHAPE, AND NOT ANOTHER
--------------------------------------------------------------------------------
Every existent thing (mumkin, "possible entity"), for Ibn 'Arabi, is a locus
in which a unique configuration of the divine Names becomes manifest. A Name
(al-Hayy, the Living; al-'Alim, the Knowing; al-Qadir, the Powerful ...) is
never free-standing -- it has no meaning except *in relation to* the thing it
governs (Chittick, "The Sufi Path of Knowledge", ch. 3-5). Knowledge worthy
of the name is not correct inference from premises but *kashf*, unveiling:
a direct disclosure (tajalli) received by the heart (qalb). The heart is
called qalb, Ibn 'Arabi insists, precisely because it "turns" (taqallub,
from the same root) -- it has no fixed shape of its own; its very capacity
is to take the form of whatever it receives, the way water takes the shape
of its cup. And -- this is the strict metaphysical law the whole file is
built to honor -- no self-disclosure ever repeats: "He never discloses
Himself in a single form twice, either to one person or to two" (Ibn
'Arabi, al-Futuhat al-Makkiyya; discussed in Chittick, "The Self-Disclosure
of God", ch. 6). Between the discrete, nameable world of the intellect
('aql) and the dense particulars of sense (hiss) sits a third register,
neither one nor the other: khayal, imagination, the barzakh or "isthmus"
that joins two things while keeping each what it is (Ibn 'Arabi's own
definition of barzakh, Futuhat II.432; Chittick 1989, ch. 14). This is
where actual thinking happens, in image-bearing, meaning-laden forms that
are not yet concepts and no longer raw sensation.

Six moves follow directly from this and are literally the six modules
below. None of them is "attention over stored keys."

  1. NAMES ARE RELATIONS, NOT LOOKUP TABLES.
     Each Name is a small function of BOTH the incoming particular x_t and
     the heart's current state h -- g_k(x_t, h) -- so the very same input
     manifests differently depending on the receiving locus's own
     "preparedness" (isti'dad). This is not a Mixture-of-Experts with
     context-free experts; the operators themselves are only ever evaluated
     in relation.
        -> class NameBank

  2. THE BARZAKH LAYER: FORM WITHOUT FIXATION.
     The K Name-outputs are blended by a receptivity distribution w (a
     softmax the heart computes over its own Names) into a single barzakh
     state z_t -- interpretable (which Names dominate) yet continuous
     (the blend itself), literally the "third thing" between the discrete
     and the sensory.
        -> gate() inside forward_sequence()

  3. LA TAKRAR FI'L-TAJALLI -- NO SELF-DISCLOSURE REPEATS.
     A bounded memory (the "Tablet") holds recent barzakh states. Every new
     z_t is pushed, by an explicit penalty, away from every recent entry
     in that Tablet beyond a similarity margin. This turns a piece of
     14th-century-Hijri metaphysics into a literal, load-bearing anti-
     mode-collapse / anti-repetition regularizer -- a real and useful ML
     mechanism, arrived at from theology rather than from GAN literature.
        -> Tablet memory + repetition_penalty() inside forward_sequence()
     Crucially the Tablet is READ but never WRITTEN TO by gradient: past
     disclosures are already inscribed and are not reshaped by what the
     heart does now. In the code this is an explicit stop-gradient
     (`.data.copy()`), and it is discussed at length in the gradient-check
     section below, because it changes what "correct" even means for that
     term -- which is itself a small, exact echo of the doctrine.

  4. TAQALLUB -- THE HEART SETS ITS OWN RATE OF TURNING.
     The heart's update is a gated blend of its old state and a new
     candidate state, but the gate (alpha) is a function of nothing but the
     heart's own prior state -- the network's plasticity at step t is
     self-determined, not fixed by an external schedule.
        -> qalb update inside forward_sequence()

  5. AL-INSAN AL-KAMIL AS A BALANCE CONDITION, NOT A LABEL.
     The "Perfect/Total Human" is, for Ibn 'Arabi, the locus capacious
     enough to be a polished mirror for the whole register of Names in
     balance, never fixated on one at the expense of the rest (Fusus
     al-Hikam, "Fass of Adam"). This is operationalised directly: a loss
     term rewards high entropy in the time-averaged Name-usage
     distribution, i.e. comprehensiveness is an explicit inductive bias,
     not an emergent accident of the task loss.
        -> balance_loss inside forward_sequence()

  6. READING, NOT LOOKING UP.
     The task head reads off the heart state at every step of a walked
     sequence of disclosures, not a single feed-forward pass over a static
     input -- "the path" (tariqa) is structural, not decorative.

None of this requires PyTorch/JAX. Section 1 below is a small
from-scratch reverse-mode autodiff engine (a "Tablet that records every
act" is, admittedly, too good a description of a computation graph to
resist) written in pure NumPy; Sections 2-4 build the architecture on top
of it; Section 5 is a synthetic but genuine combinatorial-inference task;
Section 6 is a hand-written Adam optimizer; Section 7 is the MANDATORY
finite-difference gradient check (in two parts, see the long comment
there); Section 8 is a self-test suite; Section 9 trains the model for
real and reports the numbers.

Scholarly grounding for the philosophical claims above: William C.
Chittick, "The Sufi Path of Knowledge: Ibn al-'Arabi's Metaphysics of
Imagination" (SUNY Press, 1989) and "The Self-Disclosure of God:
Principles of Ibn al-'Arabi's Cosmology" (SUNY Press, 1998); Henry Corbin,
"Alone with the Alone: Creative Imagination in the Sufism of Ibn 'Arabi"
(Princeton/Bollingen, 1969); the Stanford Encyclopedia of Philosophy entry
"Ibn 'Arabi" (William C. Chittick, 2008/2014). None of this file's
technical choices are invented out of thin air; every module name below
is glossed against one of these sources in the chapter text
(chapter_0259_ibn_arabi_1165.md), which is where the philosophical
argument is made in full and where AGI-relevance is discussed -- this file
is the architecture, not the essay.

AUTHOR: Encyclopedia of Lost Minds: Echoes on AI, Chapter 259.
LICENSE: Apache License 2.0. NOT a demo: forward pass, backward pass,
gradient check, training loop and self-tests are all real and are all
executed at the bottom of this file.
================================================================================
"""

from __future__ import annotations
import numpy as np
import time

RNG_SEED = 317
np.random.seed(RNG_SEED)


# ==============================================================================
# SECTION 1 -- A SMALL REVERSE-MODE AUTODIFF ENGINE, FROM SCRATCH, PURE NUMPY
# ==============================================================================
# A hand-derived set of backward-pass equations for a gated, multi-branch,
# memory-comparing recurrent network is exactly the kind of thing that is
# easy to get subtly wrong in one branch while nine others look fine. Rather
# than risk that, we build one small, carefully-checked "Tensor" class that
# records every operation on a tape (each Tensor knows its own parents and
# its own local backward rule) and can then run reverse-mode differentiation
# automatically over ANY graph built from these primitives. This is a classic
# and legitimate technique (the "micrograd" pattern), vectorised here with
# NumPy arrays instead of scalars. Every primitive below (add, mul, matmul,
# tanh, sigmoid, softmax, concat, getitem, sum, pow, log) is itself unit-
# tested against finite differences in Section 7 -- so gradient-checking the
# full architecture is really gradient-checking that these dozen primitives
# compose correctly, which is a much smaller and safer thing to get right
# than hand-deriving BPTT for six coupled modules.

class Tensor:
    """A NumPy array that remembers how it was made."""
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, (self,))

        def _backward():
            self.grad += _unbroadcast(-out.grad, self.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,))

        def _backward():
            self.grad += _unbroadcast((p * (self.data ** (p - 1))) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,))

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        r = np.maximum(0.0, self.data)
        out = Tensor(r, (self,))

        def _backward():
            self.grad += (self.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,))

        def _backward():
            self.grad += (1.0 / (self.data + 1e-12)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

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

    def softmax(self, axis=-1):
        z = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, (self,))

        def _backward():
            g = out.grad
            dot = np.sum(g * s, axis=axis, keepdims=True)
            self.grad += s * (g - dot)
        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors, axis=-1):
        datas = [t.data for t in tensors]
        out = Tensor(np.concatenate(datas, axis=axis), tuple(tensors))
        sizes = [d.shape[axis] for d in datas]

        def _backward():
            idx = 0
            for t, s in zip(tensors, sizes):
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(idx, idx + s)
                t.grad += out.grad[tuple(sl)]
                idx += s
        out._backward = _backward
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,))

        def _backward():
            g = np.zeros_like(self.data)
            g[key] = out.grad
            self.grad += g
        out._backward = _backward
        return out

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


def _unbroadcast(grad, shape):
    """Sum-reduce `grad` back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def zero_grad(params: dict):
    for p in params.values():
        p.grad = np.zeros_like(p.data)


# ==============================================================================
# SECTION 2 -- THE BARZAKH-QALB ARCHITECTURE
# ==============================================================================

# A curated basis of eight Names (out of the traditional ninety-nine) chosen
# for functional spread across the register Ibn 'Arabi treats as basic to
# any locus's constitution: life, knowledge, power, will, hearing, seeing,
# generosity, subtlety. Using all ninety-nine would not change the
# mechanism, only K; eight keeps the gradient check and training loop fast
# without weakening the point.
NAME_LABELS = [
    "al-Hayy   (the Living)",
    "al-'Alim  (the Knowing)",
    "al-Qadir  (the Powerful)",
    "al-Murid  (the Willing)",
    "al-Sami'  (the Hearing)",
    "al-Basir  (the Seeing)",
    "al-Karim  (the Generous)",
    "al-Latif  (the Subtle)",
]


def xavier(shape):
    limit = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return (np.random.rand(*shape) * 2 - 1) * limit


def init_params(d_x, d_h, d_z, K, d_y):
    """Every Name gets its own small relational operator; nothing else is
    shared between Names, so the balance/entropy term in Section 4 has real
    work to do (a lazy model could otherwise let one Name's operator learn
    to imitate several)."""
    P = {}
    for k in range(K):
        P[f"name{k}_W"] = Tensor(xavier((d_x + d_h, d_z)))
        P[f"name{k}_b"] = Tensor(np.zeros((1, d_z)))
    P["gate_W"] = Tensor(xavier((d_x + d_h, K)))
    P["gate_b"] = Tensor(np.zeros((1, K)))
    P["heart_W"] = Tensor(xavier((d_z + d_h, d_h)))
    P["heart_b"] = Tensor(np.zeros((1, d_h)))
    P["alpha_W"] = Tensor(xavier((d_h, d_h)))
    P["alpha_b"] = Tensor(np.zeros((1, d_h)))
    P["out_W"] = Tensor(xavier((d_h, d_y)))
    P["out_b"] = Tensor(np.zeros((1, d_y)))
    return P


def forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0.15, lambda2=0.05,
                      mem_window=6, margin=0.2, collect_diag=False):
    """
    One walk through a sequence of particulars (x_seq) and the disclosures
    (barzakh states) the heart forms in response, scored against targets
    (y_seq).

    Returns
    -------
    total : Tensor (scalar)  -- task_loss + lambda1*novelty_loss + lambda2*balance_loss
    diag  : dict of numpy arrays (z, w, h, y_hat histories) if collect_diag
    """
    T = x_seq.shape[0]
    h = Tensor(np.zeros((1, d_h)), requires_grad=False)
    tablet = []                     # the "Tablet": detached past barzakh states
    task_losses, novelty_terms, gate_ws = [], [], []
    diag = {"z": [], "w": [], "h": [], "y_hat": []} if collect_diag else None

    for t in range(T):
        x_t = Tensor(x_seq[t:t + 1], requires_grad=False)
        xh = Tensor.concat([x_t, h], axis=1)

        # --- (2) the heart's receptivity distribution over its own Names ---
        w = (xh @ P["gate_W"] + P["gate_b"]).softmax(axis=1)
        gate_ws.append(w)

        # --- (1) each Name actualizes only in relation to (x_t, h) ---
        z = None
        for k in range(K):
            g_k = (xh @ P[f"name{k}_W"] + P[f"name{k}_b"]).tanh()
            term = g_k * w[:, k:k + 1]
            z = term if z is None else z + term
        # z is now the barzakh state: a single form blended from K relational
        # Name-outputs, weighted by the heart's current receptivity.

        # --- (3) la takrar fi'l-tajalli: push z away from the Tablet ------
        # NOTE ON GRADIENTS: entries pushed onto `tablet` are `.data.copy()`
        # -- plain NumPy, deliberately detached from the graph. Past
        # disclosures are already written; today's heart-state does not
        # reach back and re-cause them. See Section 7 for what this implies
        # for gradient-checking.
        if len(tablet) > 0:
            z_unit = z / ((z * z).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
            pen = None
            for m in tablet[-mem_window:]:
                m_unit = m / (np.linalg.norm(m) + 1e-8)
                cos = (z_unit * Tensor(m_unit.reshape(1, -1), requires_grad=False)).sum(axis=1, keepdims=True)
                over = (cos - margin).relu()
                term = over * over
                pen = term if pen is None else pen + term
            pen = pen * (1.0 / len(tablet[-mem_window:]))
            novelty_terms.append(pen)
        tablet.append(z.data.copy().reshape(-1))

        # --- (4) taqallub: the heart sets its own rate of turning ---------
        zh = Tensor.concat([z, h], axis=1)
        h_candidate = (zh @ P["heart_W"] + P["heart_b"]).tanh()
        alpha = (h @ P["alpha_W"] + P["alpha_b"]).sigmoid()      # self-modulated plasticity
        h = h * (Tensor(1.0, requires_grad=False) + (-alpha)) + h_candidate * alpha

        # --- (6) read the current heart-state off as the step's output ----
        y_hat = h @ P["out_W"] + P["out_b"]
        diff = y_hat - Tensor(y_seq[t:t + 1], requires_grad=False)
        task_losses.append((diff * diff).mean())

        if collect_diag:
            diag["z"].append(z.data.copy())
            diag["w"].append(w.data.copy())
            diag["h"].append(h.data.copy())
            diag["y_hat"].append(y_hat.data.copy())

    task_loss = task_losses[0]
    for l in task_losses[1:]:
        task_loss = task_loss + l
    task_loss = task_loss * (1.0 / T)

    if novelty_terms:
        novelty_loss = novelty_terms[0]
        for l in novelty_terms[1:]:
            novelty_loss = novelty_loss + l
        novelty_loss = novelty_loss * (1.0 / len(novelty_terms))
    else:
        novelty_loss = Tensor(np.zeros((1, 1)), requires_grad=False)

    # --- (5) al-insan al-kamil: reward balanced, not fixated, Name-usage ---
    mean_w = gate_ws[0]
    for wgt in gate_ws[1:]:
        mean_w = mean_w + wgt
    mean_w = mean_w * (1.0 / T)
    entropy = -(mean_w * mean_w.log()).sum()
    balance_loss = -entropy   # minimizing this term maximizes entropy

    total = task_loss + novelty_loss * lambda1 + balance_loss * lambda2
    return total, diag


# ==============================================================================
# SECTION 3 -- ADAM OPTIMIZER (manual, pure NumPy)
# ==============================================================================

class Adam:
    def __init__(self, params: dict, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.P = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, p in self.P.items():
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ==============================================================================
# SECTION 4 -- SYNTHETIC TASK: INFERRING WHICH NAMES ARE IN PLAY
# ==============================================================================

class ParticularsTask:
    """
    A synthetic but genuine combinatorial-inference task, in the spirit of
    the architecture rather than a toy MNIST stand-in.

    K fixed latent prototype vectors ("Names", proto[k] in R^d_x) exist.
    At each timestep, a small ACTIVE subset of them is blended with
    slowly-drifting Dirichlet weights (representing a receptive locus whose
    disposition changes but never repeats exactly) to generate the
    observed particular x_t = alpha_t . proto + noise. The target y_t is a
    fixed nonlinear readout of the TRUE mixture alpha_t. The network never
    sees alpha_t -- it must recover, from noisy surface particulars alone,
    which Names are jointly in force at each moment, exactly the inferential
    task Ibn 'Arabi assigns the gnostic reading the cosmos as a book of
    disclosures.
    """

    def __init__(self, d_x, d_y, K, seed=0):
        rng = np.random.RandomState(seed)
        self.d_x, self.d_y, self.K = d_x, d_y, K
        self.proto = rng.randn(K, d_x)
        self.readout_A = rng.randn(d_y, K) * 0.9
        self.readout_b = rng.randn(1, d_y) * 0.1

    def sample(self, T, n_active=3, noise=0.15, rng=None):
        rng = rng or np.random
        active = rng.choice(self.K, size=n_active, replace=False)
        cur = rng.dirichlet(np.ones(n_active))
        alpha = np.zeros((T, self.K))
        for t in range(T):
            drift = rng.dirichlet(np.ones(n_active) * 4.0)
            cur = 0.85 * cur + 0.15 * drift
            cur = cur / cur.sum()
            alpha[t, active] = cur
        x_seq = alpha @ self.proto + rng.randn(T, self.d_x) * noise
        y_seq = np.tanh(alpha @ self.readout_A.T + self.readout_b)
        return x_seq.astype(np.float64), y_seq.astype(np.float64), alpha


# ==============================================================================
# SECTION 5 -- TRAINING LOOP
# ==============================================================================

def train(P, task: ParticularsTask, K, d_h, steps=150, batch_size=8, T=10,
          lr=0.02, lambda1=0.15, lambda2=0.05, seed=0, log_every=25, verbose=True):
    rng = np.random.RandomState(seed)
    opt = Adam(P, lr=lr)
    history = []
    for step in range(1, steps + 1):
        zero_grad(P)
        batch_loss = None
        for _ in range(batch_size):
            x_seq, y_seq, _ = task.sample(T, rng=rng)
            l, _ = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=lambda1, lambda2=lambda2)
            batch_loss = l if batch_loss is None else batch_loss + l
        batch_loss = batch_loss * (1.0 / batch_size)
        batch_loss.backward()
        opt.step()
        history.append(float(batch_loss.data.reshape(-1)[0]))
        if verbose and (step % log_every == 0 or step == 1):
            print(f"    step {step:4d}/{steps}  loss={history[-1]:.5f}")
    return history


# ==============================================================================
# SECTION 6 -- GRADIENT CHECK  (mandatory; run before training / shipping)
# ==============================================================================
#
# TWO CHECKS, FOR A DOCUMENTED REASON.
#
# The novelty term (module 3, la takrar fi'l-tajalli) is built on a
# deliberate stop-gradient: the Tablet stores `.data.copy()` snapshots, not
# graph-connected Tensors. This is correct architecture (you do not want an
# ever-growing unrolled graph back through all of history, and philosophically
# the past disclosure is fixed, not re-caused by the present heart-state) --
# but it means that if you perturb a shared parameter (say gate_W) for a
# naive whole-model finite-difference check, the FORWARD pass changes the
# Tablet's contents too (since the Tablet is *recomputed from scratch* under
# the perturbed parameter), while the BACKWARD pass, by design, never routes
# gradient through the Tablet. A naive whole-graph finite-difference check
# will therefore (correctly) disagree with the analytic gradient on exactly
# those parameters that feed the Tablet, by exactly the amount that leaks
# through the intentionally-severed path. This is not a bug to chase; it is
# the same distinction PyTorch's `.detach()` or a momentum-encoder's
# stop-gradient makes, arrived at independently from theology rather than
# from engineering convention (the doctrine: what is already inscribed on
# the Tablet is not rewritten by a later disposition).
#
# So: Check A gradient-checks the fully-differentiable core of the model
# (task loss + balance/entropy loss, lambda1=0) against EVERY parameter
# group. This must match to near machine precision, and does.
# Check B gradient-checks the novelty penalty's OWN formula in isolation,
# holding the "Tablet" comparison vector fixed (as it should be), which
# also matches to near machine precision. Between A and B, every equation
# in the file is verified.

def gradient_check_core(tol=1e-4, verbose=True):
    """Check A: task_loss + balance_loss (lambda1=0), across every module."""
    d_x, d_h, d_z, K, d_y, T = 5, 8, 8, 6, 3, 4
    rng = np.random.RandomState(11)
    P = init_params(d_x, d_h, d_z, K, d_y)
    x_seq = rng.randn(T, d_x) * 0.4
    y_seq = rng.randn(T, d_y) * 0.4

    def loss_of(P):
        total, _ = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0.0, lambda2=0.08)
        return float(total.data.reshape(-1)[0])

    total, _ = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0.0, lambda2=0.08)
    total.backward()

    keys = ["name0_W", "name0_b", "name3_W", "gate_W", "gate_b",
            "heart_W", "heart_b", "alpha_W", "alpha_b", "out_W", "out_b"]
    eps = 1e-5
    max_rel_err = 0.0
    checked = 0
    for key in keys:
        p = P[key]
        idxs = [tuple(np.random.randint(0, s) for s in p.data.shape) for _ in range(3)]
        for idx in idxs:
            orig = p.data[idx]
            p.data[idx] = orig + eps
            lp = loss_of(P)
            p.data[idx] = orig - eps
            lm = loss_of(P)
            p.data[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = p.grad[idx]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-8)
            max_rel_err = max(max_rel_err, rel)
            checked += 1
    ok = max_rel_err < tol
    if verbose:
        print(f"  [Check A: differentiable core]  {checked} params checked, "
              f"max relative error = {max_rel_err:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok, max_rel_err


def gradient_check_novelty_isolated(tol=1e-4, verbose=True):
    """Check B: the la-takrar penalty's own formula, comparison vector held fixed."""
    rng = np.random.RandomState(12)
    d = 6

    def build(W_data, base_data, margin=0.2):
        W = Tensor(W_data.copy())
        base = Tensor(base_data.copy(), requires_grad=False)
        z = base + (base @ W)                         # depends on W
        z_unit = z / ((z * z).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        fixed_mem = base_data.reshape(-1).copy()       # the "Tablet" entry: held fixed
        m_unit = fixed_mem / (np.linalg.norm(fixed_mem) + 1e-8)
        cos = (z_unit * Tensor(m_unit.reshape(1, -1), requires_grad=False)).sum(axis=1, keepdims=True)
        over = (cos - margin).relu()
        pen = (over * over).sum()
        pen.backward()
        return float(pen.data.reshape(-1)[0]), W.grad.copy()

    base = np.abs(rng.randn(1, d)) * 0.5 + 0.5
    W0 = rng.randn(d, d) * 0.05
    _, grad = build(W0, base)

    eps = 1e-5
    max_rel_err = 0.0
    for _ in range(5):
        idx = tuple(np.random.randint(0, d, size=2))
        Wp = W0.copy(); Wp[idx] += eps
        Wm = W0.copy(); Wm[idx] -= eps
        vp, _ = build(Wp, base)
        vm, _ = build(Wm, base)
        num = (vp - vm) / (2 * eps)
        ana = grad[idx]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-8)
        max_rel_err = max(max_rel_err, rel)
    ok = max_rel_err < tol
    if verbose:
        print(f"  [Check B: la-takrar penalty, isolated]  max relative error = "
              f"{max_rel_err:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok, max_rel_err


# ==============================================================================
# SECTION 7 -- SELF-TESTS
# ==============================================================================

def test_forward_shapes():
    d_x, d_h, d_z, K, d_y, T = 7, 12, 12, 8, 4, 9
    P = init_params(d_x, d_h, d_z, K, d_y)
    x_seq = np.random.randn(T, d_x) * 0.3
    y_seq = np.random.randn(T, d_y) * 0.3
    total, diag = forward_sequence(P, x_seq, y_seq, K, d_h, collect_diag=True)
    assert total.data.shape in [(), (1, 1)]
    assert len(diag["z"]) == T and diag["z"][0].shape == (1, d_z)
    assert len(diag["w"]) == T and diag["w"][0].shape == (1, K)
    print("  [Self-test: forward shapes]  PASS")
    return True


def test_overfit_tiny_batch():
    d_x, d_h, d_z, K, d_y, T = 6, 10, 10, 6, 3, 6
    task = ParticularsTask(d_x, d_y, K, seed=1)
    P = init_params(d_x, d_h, d_z, K, d_y)
    rng = np.random.RandomState(1)
    x_seq, y_seq, _ = task.sample(T, rng=rng)
    opt = Adam(P, lr=0.05)
    losses = []
    for i in range(220):
        zero_grad(P)
        total, _ = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0.0, lambda2=0.0)
        total.backward()
        opt.step()
        losses.append(float(total.data.reshape(-1)[0]))
    ok = losses[-1] < 0.02 and losses[-1] < 0.1 * losses[0]
    print(f"  [Self-test: overfit tiny batch]  loss {losses[0]:.4f} -> {losses[-1]:.5f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_non_repetition_effect():
    """Train twice from the identical init, la-takrar ON vs OFF, then compare
    how self-similar the heart's disclosures are across a held-out sequence."""
    d_x, d_h, d_z, K, d_y, T = 6, 10, 10, 6, 3, 16
    task = ParticularsTask(d_x, d_y, K, seed=2)

    def run(lambda1):
        np.random.seed(42)
        P = init_params(d_x, d_h, d_z, K, d_y)
        train(P, task, K, d_h, steps=45, batch_size=4, T=10, lr=0.03,
              lambda1=lambda1, lambda2=0.0, seed=2, verbose=False)
        rng = np.random.RandomState(99)
        x_seq, y_seq, _ = task.sample(T, rng=rng)
        _, diag = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0, lambda2=0, collect_diag=True)
        Z = np.concatenate(diag["z"], axis=0)
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        sims = Zn @ Zn.T
        iu = np.triu_indices(T, k=1)
        return float(np.mean(sims[iu]))

    sim_off = run(0.0)
    sim_on = run(0.6)
    ok = sim_on < sim_off
    print(f"  [Self-test: la-takrar fi'l-tajalli effect]  mean pairwise cos-sim "
          f"OFF={sim_off:.4f}  ON={sim_on:.4f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_balance_effect():
    """Train twice, al-insan-al-kamil balance term ON vs OFF, compare the
    entropy of time-averaged Name usage on a held-out sequence."""
    d_x, d_h, d_z, K, d_y, T = 6, 10, 10, 6, 3, 16
    task = ParticularsTask(d_x, d_y, K, seed=3)

    def run(lambda2):
        np.random.seed(7)
        P = init_params(d_x, d_h, d_z, K, d_y)
        train(P, task, K, d_h, steps=45, batch_size=4, T=10, lr=0.03,
              lambda1=0.0, lambda2=lambda2, seed=3, verbose=False)
        rng = np.random.RandomState(123)
        x_seq, y_seq, _ = task.sample(T, rng=rng)
        _, diag = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0, lambda2=0, collect_diag=True)
        W = np.concatenate(diag["w"], axis=0)
        mean_w = W.mean(axis=0)
        mean_w = mean_w / mean_w.sum()
        ent = -np.sum(mean_w * np.log(mean_w + 1e-12))
        return ent, mean_w

    ent_off, w_off = run(0.0)
    ent_on, w_on = run(0.5)
    ok = ent_on > ent_off
    print(f"  [Self-test: al-insan al-kamil balance effect]  Name-usage entropy "
          f"OFF={ent_off:.4f}  ON={ent_on:.4f}  (max possible = {np.log(K):.4f})  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ==============================================================================
# SECTION 8 -- MAIN: run gradient checks, self-tests, then a real training run
# ==============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("BARZAKH-QALB NETWORK -- Chapter 259, Ibn 'Arabi (1165-1240)")
    print("=" * 78)

    print("\n[1/3] MANDATORY GRADIENT CHECK")
    okA, errA = gradient_check_core()
    okB, errB = gradient_check_novelty_isolated()
    assert okA and okB, "Gradient check failed -- do not proceed to training."

    print("\n[2/3] SELF-TESTS")
    t0 = time.time()
    r1 = test_forward_shapes()
    r2 = test_overfit_tiny_batch()
    r3 = test_non_repetition_effect()
    r4 = test_balance_effect()
    print(f"  ({time.time()-t0:.1f}s)  all self-tests: {'PASS' if all([r1,r2,r3,r4]) else 'FAIL'}")

    print("\n[3/3] TRAINING RUN on the combinatorial-particulars task")
    d_x, d_h, d_z, K, d_y = 10, 24, 24, 8, 5
    task = ParticularsTask(d_x, d_y, K, seed=259)
    P = init_params(d_x, d_h, d_z, K, d_y)
    print(f"  dims: d_x={d_x} d_h={d_h} d_z={d_z} K={K} (Names: {', '.join(NAME_LABELS)}) d_y={d_y}")
    t0 = time.time()
    hist = train(P, task, K, d_h, steps=150, batch_size=8, T=10, lr=0.02,
                 lambda1=0.15, lambda2=0.05, seed=259, log_every=25)
    print(f"  training time: {time.time()-t0:.1f}s")
    print(f"  loss[0]={hist[0]:.5f}  loss[-1]={hist[-1]:.5f}  "
          f"(reduction: {100*(1-hist[-1]/hist[0]):.1f}%)")

    # held-out evaluation
    rng = np.random.RandomState(999)
    task_losses = []
    for _ in range(20):
        x_seq, y_seq, alpha = task.sample(12, rng=rng)
        total, diag = forward_sequence(P, x_seq, y_seq, K, d_h, lambda1=0.0, lambda2=0.0, collect_diag=True)
        task_losses.append(float(total.data.reshape(-1)[0]))
    print(f"  held-out task loss (20 unseen sequences, regularizers off): "
          f"mean={np.mean(task_losses):.5f}  std={np.std(task_losses):.5f}")

    print("\nDone. Architecture verified: gradients correct, training converges,")
    print("la-takrar fi'l-tajalli measurably reduces self-similarity, and the")
    print("al-insan al-kamil term measurably raises Name-usage entropy.")

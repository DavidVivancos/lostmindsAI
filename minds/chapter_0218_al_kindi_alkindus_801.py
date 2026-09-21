"""
================================================================================
 0218_al_kindi_Neuron.py
 THE ISTIKHRAJ ENGINE
 A relabeling-invariant form-matching network, after Abu Yusuf Ya'qub ibn Ishaq
 al-Kindi (c. 801 - c. 873 CE), "the philosopher of the Arabs."
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0218_al_kindi_alkindus_801 - Abu Yusuf Ya'qub ibn Ishaq al-Kindi (c. 801 - c. 873 CE), "the philosopher of the Arabs."
================================================================================  

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Al-Kindi is routinely mis-summarised as an abstraction theorist -- as though he
held that the mind builds concepts upward out of sense data. He held the
opposite. In "On the Intellect" (Risala fi al-'Aql) and, more bluntly, in "On
Recollection," he argues that intelligible forms CANNOT be derived from
sense-perception at all. The soul does not manufacture the form; the form is
already in act, outside the soul, and the soul receives it. Sensation, he says,
does not contribute to this act -- it interferes with it.

That is a strange doctrine until you look at what al-Kindi actually DID for a
living, which was break ciphers. His "Risala fi Istikhraj al-Mu'amma" (c. 850,
rediscovered in the Sulaimaniyya Ottoman Archive in Istanbul and published by
the Arab Academy of Damascus in 1987) is the oldest surviving treatise on
cryptanalysis. Its method: a substitution cipher can permute WHICH symbol
stands for which letter, but it cannot permute HOW OFTEN each symbol occurs.
Count the ciphertext. Count a long sample of ordinary prose. Rank both lists.
Align them. Refine with digram statistics and short common words.

Notice the epistemology hiding inside the technique. The plaintext's identity
survives an arbitrary relabeling because the invariant is the SHAPE of the
distribution, not the labels. The cryptanalyst does not build the message up out
of the marks on the page; the marks are noise. He matches a residue against a
form he already possesses. Al-Kindi's theory of intellect and al-Kindi's
codebreaking are the same act described twice.

So this file does not implement attention over stored keys. It implements
DECIPHERMENT as the primitive cognitive operation:

    intelligence = recovering a correspondence between an observed signal and an
    always-actual catalogue of forms, using only statistics that are provably
    invariant under the nuisance transformation.

MODULE / DOCTRINE MAP
---------------------
  PotentialIntellect  (al-'aql bi-l-quwwa)  -- starts literally empty (zeros).
                       Extracts only permutation-invariant statistics from the
                       ciphertext. Holds no forms of its own.
  DegreeLadder        (darajat, from De gradibus) -- al-Kindi quantified drug
                       potency on a doubling scale, the first serious attempt at
                       quantification in medicine. Here intensities enter the
                       network on a geometric (log) ladder, not a linear one.
  FirstIntellect      (al-'aql al-awwal)     -- a bank of K form-signatures and
                       per-form language potentials. Input-independent, "always
                       in act." Illumination = softmax match against this bank.
  Talif               (composition)          -- a Sinkhorn softassign solver
                       that recovers the substitution. Stage 1 is al-Kindi's
                       rank-alignment of frequencies; stage 2 is his digram
                       refinement, run as a relaxed quadratic assignment.
  AcquiredIntellect   (al-'aql al-mustafad)  -- persistent memory of a solved
                       correspondence, so it can be re-applied at will to later
                       text without re-deriving it. Al-Kindi's third intellect.
  ManifestIntellect   (al-'aql al-zahir)     -- the readout: actual decoding.

TASK THE MODEL SOLVES (and it really does solve it)
---------------------------------------------------
  A hidden source k in {0..K-1} emits text from its own first-order Markov
  chain over an alphabet of size A. A fresh random permutation sigma of the
  alphabet is drawn per example and applied. The model sees ONLY the ciphertext
  and must (a) name the source, (b) recover sigma^-1, (c) decode the plaintext.
  It is never shown the permutation at test time.

IMPLEMENTATION NOTES
--------------------
  * Pure NumPy. No torch, no jax, no autograd library.
  * Section 1 is a ~200-line reverse-mode autodiff tape written from scratch,
    so that the Sinkhorn recursion can be differentiated exactly rather than
    approximated.
  * A finite-difference gradient check over every parameter tensor is MANDATORY
    and runs in main().
  * Self-tests assert the permutation-invariance claims the whole design rests
    on. If those fail, the architecture is wrong, not just the weights.

Run:  python3 0218_al_kindi_Neuron.py
================================================================================
"""

import numpy as np

RNG_GLOBAL = np.random.default_rng(20250801)


# ==============================================================================
# SECTION 1 -- A MINIMAL REVERSE-MODE AUTODIFF TAPE
# ------------------------------------------------------------------------------
# Everything downstream is built from these primitives. Each Tensor remembers
# the operation that produced it and the tensors that fed it, forming a DAG.
# backward() walks that DAG in reverse topological order accumulating gradients.
# This exists because the Sinkhorn normalisation below is an iterative fixed
# point; hand-deriving its Jacobian is error-prone, and the whole scientific
# claim of this file rests on the gradient check passing.
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting.

    If a (3,1) tensor was broadcast against a (3,4) tensor, its gradient comes
    back as (3,4) and must be summed along the broadcast axes.
    """
    if grad.shape == shape:
        return grad
    # Collapse leading axes that did not exist in the original.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Collapse axes that were size-1 in the original.
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph."""

    __slots__ = ("data", "grad", "_parents", "_backward", "requires_grad", "name")

    def __init__(self, data, parents=(), backward=None, requires_grad=False, name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._backward = backward       # callable(out_grad) -> tuple of parent grads
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.name = name

    # -- conveniences ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(name={self.name!r}, shape={self.shape})"

    def zero_grad(self):
        self.grad = None

    # -- the engine -----------------------------------------------------------
    def backward(self):
        """Seed d(self)/d(self) = 1 and propagate. `self` must be a scalar."""
        assert self.data.size == 1, "backward() only from a scalar loss"

        # Reverse topological order via iterative DFS (no recursion limits).
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            for parent, g in zip(node._parents, grads):
                if g is None or not parent.requires_grad:
                    continue
                g = _unbroadcast(np.asarray(g, dtype=np.float64), parent.shape)
                parent.grad = g if parent.grad is None else parent.grad + g

    # -- operator overloads ---------------------------------------------------
    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __sub__(self, other):
        return add(self, mul(as_tensor(other), const(-1.0)))

    def __rsub__(self, other):
        return add(as_tensor(other), mul(self, const(-1.0)))

    def __neg__(self):
        return mul(self, const(-1.0))

    def __truediv__(self, other):
        return mul(self, power(as_tensor(other), -1.0))

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, idx):
        return getitem(self, idx)

    @property
    def T(self):
        return transpose(self)


def const(x, name=""):
    """A leaf that never needs a gradient."""
    return Tensor(x, requires_grad=False, name=name)


def param(x, name=""):
    """A trainable leaf."""
    return Tensor(x, requires_grad=True, name=name)


def as_tensor(x):
    return x if isinstance(x, Tensor) else const(x)


# ---- primitive ops -----------------------------------------------------------

def add(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data + b.data, (a, b), lambda g: (g, g))
    return out


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data * b.data, (a, b), lambda g: (g * b.data, g * a.data))
    return out


def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = Tensor(a.data @ b.data, (a, b),
                 lambda g: (g @ np.swapaxes(b.data, -1, -2),
                            np.swapaxes(a.data, -1, -2) @ g))
    return out


def power(a, p):
    a = as_tensor(a)
    val = np.power(a.data, p)
    out = Tensor(val, (a,), lambda g: (g * p * np.power(a.data, p - 1.0),))
    return out


def exp(a):
    a = as_tensor(a)
    val = np.exp(a.data)
    out = Tensor(val, (a,), lambda g: (g * val,))
    return out


def log(a):
    a = as_tensor(a)
    out = Tensor(np.log(a.data), (a,), lambda g: (g / a.data,))
    return out


def softplus(a):
    """log(1+e^x), computed stably. Used to keep positive hyper-parameters positive."""
    a = as_tensor(a)
    x = a.data
    val = np.logaddexp(0.0, x)
    sig = 1.0 / (1.0 + np.exp(-x))
    out = Tensor(val, (a,), lambda g: (g * sig,))
    return out


def summ(a, axis=None, keepdims=False):
    a = as_tensor(a)
    val = a.data.sum(axis=axis, keepdims=keepdims)

    def bwd(g):
        if axis is None:
            return (np.broadcast_to(g, a.shape).copy(),)
        gg = g if keepdims else np.expand_dims(g, axis)
        return (np.broadcast_to(gg, a.shape).copy(),)

    return Tensor(val, (a,), bwd)


def logsumexp(a, axis, keepdims=True):
    """Stable logsumexp. Backward is a softmax -- this is the workhorse of the
    Sinkhorn iterations and of every cross-entropy in the model."""
    a = as_tensor(a)
    m = np.max(a.data, axis=axis, keepdims=True)
    s = np.sum(np.exp(a.data - m), axis=axis, keepdims=True)
    val = m + np.log(s)
    soft = np.exp(a.data - val)                     # softmax along `axis`
    if not keepdims:
        val = np.squeeze(val, axis=axis)

    def bwd(g):
        gg = g if keepdims else np.expand_dims(g, axis)
        return (soft * gg,)

    return Tensor(val, (a,), bwd)


def transpose(a, axes=None):
    a = as_tensor(a)
    val = np.transpose(a.data, axes)
    if axes is None:
        inv = None
    else:
        inv = np.argsort(axes)
    return Tensor(val, (a,), lambda g: (np.transpose(g, inv),))


def reshape(a, shape):
    a = as_tensor(a)
    old = a.shape
    return Tensor(a.data.reshape(shape), (a,), lambda g: (g.reshape(old),))


def concat(tensors, axis=0):
    tensors = [as_tensor(t) for t in tensors]
    sizes = [t.shape[axis] for t in tensors]
    val = np.concatenate([t.data for t in tensors], axis=axis)

    def bwd(g):
        outs, start = [], 0
        for s in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(start, start + s)
            outs.append(g[tuple(sl)])
            start += s
        return tuple(outs)

    return Tensor(val, tuple(tensors), bwd)


def getitem(a, idx):
    a = as_tensor(a)
    val = a.data[idx]

    def bwd(g):
        z = np.zeros_like(a.data)
        np.add.at(z, idx, g)
        return (z,)

    return Tensor(val, (a,), bwd)


def gather_sorted(a, axis=-1):
    """Sort DESCENDING along `axis`.

    A sort is a permutation. Given the forward values, the permutation is fixed,
    so the backward pass is just the inverse gather -- gradients flow to whichever
    element ended up in each slot. This is what makes the invariant features in
    Section 3 differentiable, which in turn is what lets the network learn on top
    of quantities that are provably blind to the cipher's relabeling.
    """
    a = as_tensor(a)
    order = np.argsort(-a.data, axis=axis, kind="stable")
    val = np.take_along_axis(a.data, order, axis=axis)

    def bwd(g):
        z = np.zeros_like(a.data)
        np.put_along_axis(z, order, g, axis=axis)
        return (z,)

    return Tensor(val, (a,), bwd)


def trace_of(a):
    """Trace of a square matrix, as a 1-element tensor."""
    a = as_tensor(a)
    n = a.shape[-1]
    eye = np.eye(n)
    val = np.array([np.sum(a.data * eye)])
    return Tensor(val, (a,), lambda g: (eye * g[0],))


def diag_of(a):
    """Extract the main diagonal as a vector."""
    a = as_tensor(a)
    n = a.shape[-1]
    val = np.diagonal(a.data).copy()

    def bwd(g):
        z = np.zeros_like(a.data)
        np.fill_diagonal(z, g)
        return (z,)

    return Tensor(val, (a,), bwd)


# ==============================================================================
# SECTION 2 -- THE CORPUS: HIDDEN FORMS AND AN UNKNOWN HAND
# ------------------------------------------------------------------------------
# Al-Kindi's method needs two things: a message in an unknown hand, and "a long
# sample of ordinary prose in the same language." We manufacture both. Each of
# the K sources is a distinct first-order Markov chain over the alphabet -- a
# distinct "language," with its own characteristic frequencies AND its own
# characteristic letter-adjacency habits. Then a fresh random permutation is
# drawn per sample: the enemy's key, never twice the same.
# ==============================================================================

class Scriptorium:
    """Generates plaintext from K latent Markov sources and enciphers it."""

    def __init__(self, alphabet=10, n_forms=4, concentration=0.35, seed=7):
        self.A = alphabet
        self.K = n_forms
        rng = np.random.default_rng(seed)
        # Each source gets a sparse-ish, strongly peaked transition matrix. A low
        # Dirichlet concentration makes each language idiosyncratic, so that the
        # digram structure -- not just the unigram histogram -- carries identity.
        self.T = np.stack([rng.dirichlet(np.full(alphabet, concentration),
                                         size=alphabet) for _ in range(n_forms)])
        self.pi = np.stack([self._stationary(self.T[k]) for k in range(n_forms)])

    @staticmethod
    def _stationary(T, iters=2000):
        v = np.full(T.shape[0], 1.0 / T.shape[0])
        for _ in range(iters):
            v = v @ T
        return v / v.sum()

    def sample(self, length, rng, form=None):
        """Return (ciphertext, plaintext, form_index, sigma_inverse)."""
        k = int(rng.integers(self.K)) if form is None else int(form)
        x = np.empty(length, dtype=np.int64)
        x[0] = rng.choice(self.A, p=self.pi[k])
        for t in range(1, length):
            x[t] = rng.choice(self.A, p=self.T[k, x[t - 1]])
        sigma = rng.permutation(self.A)          # plain index -> cipher index
        c = sigma[x]                             # the enciphered stream
        sigma_inv = np.argsort(sigma)            # cipher index -> plain index
        return c, x, k, sigma_inv

    def batch(self, n, length, rng):
        cs, xs, ks, sinv = [], [], [], []
        for _ in range(n):
            c, x, k, si = self.sample(length, rng)
            cs.append(c); xs.append(x); ks.append(k); sinv.append(si)
        return np.stack(cs), np.stack(xs), np.array(ks), np.stack(sinv)


def unigram_counts(c, A):
    """Normalised symbol frequencies."""
    u = np.bincount(c, minlength=A).astype(np.float64)
    return u / max(u.sum(), 1.0)


def bigram_counts(c, A):
    """Normalised adjacency matrix N[i,j] = P(next=j | saw i, i)."""
    N = np.zeros((A, A))
    np.add.at(N, (c[:-1], c[1:]), 1.0)
    return N / max(N.sum(), 1.0)


# ==============================================================================
# SECTION 3 -- THE POTENTIAL INTELLECT: STATISTICS THAT SURVIVE RELABELING
# ------------------------------------------------------------------------------
# "The potential intellect is merely an ability to grasp intellectual forms." It
# contains nothing. So this module has ZERO parameters. Its only job is to strip
# the ciphertext down to quantities that a relabeling cannot touch.
#
# Under a substitution sigma the unigram vector u becomes a permutation of u, and
# the bigram matrix N becomes P^T N P. Therefore the following are invariant:
#   * u sorted descending                      -- al-Kindi's ranked frequency list
#   * the sorted multiset of all entries of N  -- the digram profile, unlabelled
#   * the sorted main diagonal of N            -- the self-repetition profile
#   * tr(N), tr(N^2), tr(N^3), ||N||_F         -- conjugation invariants
# Section 6 asserts this numerically. If it were false the whole design collapses.
# ==============================================================================

class PotentialIntellect:
    """al-'aql bi-l-quwwa. Empty of forms; extracts only invariants."""

    def __init__(self, A):
        self.A = A
        self.dim = A + A + A * A + 4

    @staticmethod
    def raw_counts(c, A):
        """Unnormalised tallies -- the scribe's actual marks in the margin."""
        u = np.bincount(c, minlength=A).astype(np.float64)
        N = np.zeros((A, A))
        np.add.at(N, (c[:-1], c[1:]), 1.0)
        return u, N

    def __call__(self, c, extra_counts=None):
        """c: 1-D ciphertext. `extra_counts` is the acquired intellect's retained
        tally, folded in before anything is normalised. Returns (v, u, N)."""
        A = self.A
        u_raw, N_raw = self.raw_counts(c, A)
        if extra_counts is not None:
            u_raw = u_raw + extra_counts[0]
            N_raw = N_raw + extra_counts[1]
        u = u_raw / max(u_raw.sum(), 1.0)
        N = N_raw / max(N_raw.sum(), 1.0)
        Nt = const(N)

        N2 = Nt @ Nt
        N3 = N2 @ Nt
        parts = [
            gather_sorted(const(u)),                       # ranked frequencies
            gather_sorted(diag_of(Nt)),                    # ranked self-transitions
            gather_sorted(reshape(Nt, (A * A,))),          # ranked digram multiset
            trace_of(Nt), trace_of(N2), trace_of(N3),
            reshape(summ(mul(Nt, Nt)), (1,)),              # squared Frobenius norm
        ]
        return concat(parts, axis=0), u, N


# ==============================================================================
# SECTION 4 -- THE DEGREE LADDER (darajat)
# ------------------------------------------------------------------------------
# In De gradibus al-Kindi tried to put a number on how strong a compound drug is,
# and his answer was that potency climbs by DOUBLING: each degree is a ratio, not
# an increment. Latin physicians found the computation nearly unusable -- Roger
# Bacon complained about it -- but the underlying instinct was right and rare:
# qualitative intensity should be measured on a geometric scale.
#
# Counts in a corpus are Zipf-ish, spanning orders of magnitude. Feeding them in
# linearly lets one dominant symbol swamp the rest. So the network's first act on
# any intensity is to put it on al-Kindi's ladder: v -> log2(1 + beta*v), with
# beta learned. This is his measurement theory used as an inductive bias.
# ==============================================================================

class DegreeLadder:
    """Geometric-degree encoding of intensity. One learnable scalar."""

    def __init__(self):
        self.beta_raw = param(np.array([0.5]), name="degrees.beta_raw")

    def params(self):
        return [self.beta_raw]

    def __call__(self, v):
        beta = softplus(self.beta_raw) + const(1e-3)      # strictly positive
        return log(const(1.0) + mul(v, beta)) * const(1.0 / np.log(2.0))


# ==============================================================================
# SECTION 5 -- TA'LIF: THE SINKHORN CORRESPONDENCE SOLVER
# ------------------------------------------------------------------------------
# This is the mechanical heart of the file and the exact place where al-Kindi's
# recipe is mechanised.
#
# His stage 1: rank the ciphertext frequencies, rank the reference-language
#   frequencies, and align the two lists. Below, that is `score_marginal` -- a
#   soft cost for pairing cipher symbol i (observed rate u_i) with plain symbol j
#   (expected rate q_j), penalised in LOG rate, i.e. on the degree ladder again.
#
# His stage 2: "he extends the method to digram statistics." If the assignment
#   were a hard permutation P, the plaintext bigram matrix would be P^T N P, and
#   its log-likelihood under the form's potentials Theta would be
#       sum_{a,b} (P^T N P)[a,b] * Theta[a,b],
#   which is quadratic in P -- a quadratic assignment problem, NP-hard. We relax
#   it: take the gradient of that objective at the current P,
#       G = N P Theta^T + N^T P Theta,
#   add it to the marginal score, and re-project onto the doubly-stochastic
#   polytope with Sinkhorn normalisation. Iterating is softassign / a Frank-Wolfe
#   style relaxation of the QAP.
#
# Every row of P sums to 1 and every column sums to 1: a cipher symbol means
# exactly one plain letter, and each plain letter is spoken for exactly once.
# That constraint IS the cipher's structure, and enforcing it is what lets the
# model beat naive frequency matching when two letters have similar rates.
# ==============================================================================

def sinkhorn(logits, n_iter=12):
    """Project a score matrix onto (approximately) doubly-stochastic, in log space."""
    z = logits
    for _ in range(n_iter):
        z = z - logsumexp(z, axis=1, keepdims=True)   # rows sum to 1
        z = z - logsumexp(z, axis=0, keepdims=True)   # cols sum to 1
    return z                                          # log P


class Talif:
    """Composition. Recovers the substitution; holds only its own temperatures.

    `acquire_weight` is deliberately NOT a trained parameter. The acquired
    intellect is a faculty of the individual soul in use, not something the
    species learns; and training here always starts from a blank memory, so a
    trained weight would never receive gradient. It is a fixed disposition.
    """

    def __init__(self, n_rounds=3, n_sinkhorn=12):
        self.n_rounds = n_rounds
        self.n_sinkhorn = n_sinkhorn
        self.log_tau = param(np.array([0.0]), name="talif.log_tau")
        self.log_lam_m = param(np.array([1.0]), name="talif.log_lam_marginal")
        self.log_lam_s = param(np.array([1.0]), name="talif.log_lam_structural")

    def params(self):
        return [self.log_tau, self.log_lam_m, self.log_lam_s]

    def __call__(self, u, N, q_bar, Theta_bar, A):
        """u: (A,) observed rates. N: (A,A) observed digrams (constants).
        q_bar: (A,) expected rates under the illuminated form (tensor).
        Theta_bar: (A,A) expected digram log-potentials (tensor).
        Returns log P, shape (A,A): rows = cipher symbols, cols = plain letters."""
        tau = exp(self.log_tau) + const(1e-2)
        lam_m = exp(self.log_lam_m)
        lam_s = exp(self.log_lam_s)

        # --- Stage 1: al-Kindi's rank alignment, softened ---------------------
        # Cost of pairing cipher i with plain j = squared difference of LOG rates.
        log_u = const(np.log(u + 1e-8)[:, None])          # (A,1) observed log-rates
        log_q = reshape(log(q_bar + const(1e-8)), (1, A))  # (1,A) expected log-rates
        d = log_u + (log_q * const(-1.0))                  # (A,A) log-rate mismatch
        score_marg = mul(mul(d, d), const(-1.0)) * lam_m       # (A,A), higher = better

        Nc = const(N)
        NcT = const(N.T)

        # --- Stage 2: digram refinement as a relaxed QAP ----------------------
        z = sinkhorn(score_marg * (const(1.0) / tau), self.n_sinkhorn)
        for _ in range(self.n_rounds):
            P = exp(z)
            G = (Nc @ P @ transpose(Theta_bar)) + (NcT @ P @ Theta_bar)
            S = score_marg + G * lam_s
            z = sinkhorn(S * (const(1.0) / tau), self.n_sinkhorn)
        return z


# ==============================================================================
# SECTION 6 -- THE ISTIKHRAJ ENGINE
# ------------------------------------------------------------------------------
# Assembling the four intellects. Note what is deliberately absent: there is no
# path by which the form bank is computed FROM the input. F and Theta are the
# same tensors for every example the model will ever see. They are "always in
# act." The input's only power is to select among them and to supply the
# statistics that the correspondence solver aligns against them.
# ==============================================================================

class IstikhrajEngine:

    def __init__(self, A=10, K=4, d_hidden=48, d_form=32, seed=11):
        rng = np.random.default_rng(seed)
        self.A, self.K = A, K
        self.potential = PotentialIntellect(A)      # no parameters, by doctrine
        self.ladder = DegreeLadder()
        self.talif = Talif()

        din = self.potential.dim
        s1 = np.sqrt(2.0 / din)
        s2 = np.sqrt(2.0 / d_hidden)

        # --- the soul's own machinery (small, and it learns) ------------------
        self.W1 = param(rng.normal(0, s1, (din, d_hidden)), name="soul.W1")
        self.b1 = param(np.zeros(d_hidden), name="soul.b1")
        self.W2 = param(rng.normal(0, s2, (d_hidden, d_form)), name="soul.W2")
        self.b2 = param(np.zeros(d_form), name="soul.b2")

        # --- FIRST INTELLECT: input-independent, always in act ----------------
        self.F = param(rng.normal(0, 0.3, (K, d_form)), name="first_intellect.F")
        self.bF = param(np.zeros(K), name="first_intellect.bF")
        self.theta_uni = param(rng.normal(0, 0.3, (K, A)),
                               name="first_intellect.theta_unigram")
        self.Theta_big = param(rng.normal(0, 0.3, (K, A, A)),
                               name="first_intellect.Theta_bigram")

        # --- ACQUIRED INTELLECT: retained EVIDENCE, not a retained conclusion --
        # See acquire() for why this stores tallies rather than a solved P.
        self.acquired = None        # (u_counts, N_counts) or None

    # -- bookkeeping ----------------------------------------------------------
    def params(self):
        return ([self.W1, self.b1, self.W2, self.b2,
                 self.F, self.bF, self.theta_uni, self.Theta_big]
                + self.ladder.params() + self.talif.params())

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def forget(self):
        """Empty the acquired intellect. A new key, a new correspondence."""
        self.acquired = None

    # -- the act of knowing ---------------------------------------------------
    def forward(self, c, use_memory=False):
        """c: 1-D ciphertext. Returns a dict of tensors."""
        A, K = self.A, self.K

        # 1. POTENTIAL INTELLECT -- receive, but only what relabeling cannot hide.
        #    If the acquired intellect holds anything, its tallies are folded in
        #    here, before normalisation: the soul reads this passage in the light
        #    of everything it has already counted in the same hand.
        extra = self.acquired if (use_memory and self.acquired is not None) else None
        v, u, N = self.potential(c, extra_counts=extra)

        # 2. DEGREES -- put intensities on the geometric ladder.
        v = self.ladder(v)

        # 3. The soul's projection: two affine layers with a smooth nonlinearity.
        h = softplus(reshape(v, (1, -1)) @ self.W1 + reshape(self.b1, (1, -1)))
        zf = h @ self.W2 + reshape(self.b2, (1, -1))               # (1, d_form)

        # 4. ILLUMINATION -- match against the always-actual bank of forms.
        #    This is the only place the First Intellect touches the soul.
        logits_form = reshape(zf @ transpose(self.F), (K,)) + self.bF
        logZ = logsumexp(reshape(logits_form, (1, K)), axis=1, keepdims=False)
        log_p_form = logits_form - reshape(logZ, (1,))             # (K,)
        p_form = exp(log_p_form)

        # 5. The illuminated form's expectations, as a soft mixture.
        #    q_bar: what rates should this language show?
        #    Theta_bar: what adjacencies should it show?
        log_q_all = self.theta_uni - logsumexp(self.theta_uni, axis=1, keepdims=True)
        q_all = exp(log_q_all)                                     # (K,A)
        q_bar = reshape(reshape(p_form, (1, K)) @ q_all, (A,))
        Theta_bar = reshape(reshape(p_form, (1, K)) @ reshape(self.Theta_big, (K, A * A)),
                            (A, A))

        # 6. TA'LIF -- recover the correspondence.
        logP = self.talif(u, N, q_bar, Theta_bar, A)

        return {"log_p_form": log_p_form, "p_form": p_form,
                "logP": logP, "u": u, "N": N,
                "q_bar": q_bar, "Theta_bar": Theta_bar}

    # -- ACQUIRED INTELLECT ---------------------------------------------------
    def acquire(self, c):
        """Retain what this passage taught, so later passages are read in its light.

        Design note, and it cost a rewrite to learn. The obvious implementation is
        to cache the SOLVED correspondence and warm-start the next window with it.
        That was tried and it made accuracy WORSE by roughly twenty points: a short
        first window yields a confident wrong assignment, the cache promotes that
        wrong assignment to a prior, and every later window inherits the mistake.
        Retaining a conclusion is how a mind gets stuck.

        What is retained here instead is EVIDENCE -- the raw tallies. Al-Kindi is
        explicit that the method needs a long enough sample; later cryptanalysts in
        the tradition (Ibn Adlan, 13th c.) worked directly on the question of how
        long. So the acquired intellect accumulates counts, and the correspondence
        is re-solved each time against a larger and larger body of them. Nothing is
        locked in; only the sample grows.
        """
        u, N = PotentialIntellect.raw_counts(c, self.A)
        if self.acquired is None:
            self.acquired = (u, N)
        else:
            self.acquired = (self.acquired[0] + u, self.acquired[1] + N)

    # -- MANIFEST INTELLECT ---------------------------------------------------
    def decode_logits(self, logP, Theta_bar, c):
        """Actually read the message.

        For each position, the plaintext distribution is the row of P belonging to
        the observed cipher symbol, tilted by what the form expects to follow the
        previous letter. Correspondence and language decide jointly -- exactly the
        interplay al-Kindi describes when he tells the cryptanalyst to check a
        candidate substitution against whether the resulting words are Arabic.
        """
        A = self.A
        rows = getitem(logP, (c, slice(None)))                     # (T, A)
        prev = np.concatenate([[c[0]], c[:-1]])
        prev_rows = getitem(logP, (prev, slice(None)))              # (T, A)
        ctx = exp(prev_rows) @ Theta_bar                            # (T, A)
        return rows + ctx * const(0.5)


# ==============================================================================
# SECTION 7 -- LOSS
# ------------------------------------------------------------------------------
#  L_form   : did the soul name the right form?         (illumination)
#  L_perm   : is the recovered correspondence correct?  (istikhraj proper)
#  L_decode : does the message read as the plaintext?   (manifest intellect)
# ==============================================================================

def cross_entropy_from_logits(logits, targets):
    """logits (T,C), targets (T,) ints -> mean NLL."""
    T = logits.shape[0]
    lse = logsumexp(logits, axis=1, keepdims=False)                # (T,)
    picked = getitem(logits, (np.arange(T), targets))              # (T,)
    return summ(lse - picked) * const(1.0 / T)


def istikhraj_loss(model, out, k_true, sigma_inv, c, x,
                   w_form=1.0, w_perm=1.0, w_dec=1.0):
    A = model.A
    L_form = -getitem(out["log_p_form"], (k_true,))
    L_form = reshape(L_form, (1,))

    logP = out["logP"]
    rows = np.arange(A)
    L_perm = summ(getitem(logP, (rows, sigma_inv))) * const(-1.0 / A)
    L_perm = reshape(L_perm, (1,))

    dec = model.decode_logits(logP, out["Theta_bar"], c)
    L_dec = reshape(cross_entropy_from_logits(dec, x), (1,))

    total = L_form * const(w_form) + L_perm * const(w_perm) + L_dec * const(w_dec)
    return total, (L_form, L_perm, L_dec)


# ==============================================================================
# SECTION 8 -- OPTIMISER AND TRAINING
# ==============================================================================

class Adam:
    """Plain Adam, written out so nothing is hidden behind a library call."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self, clip=5.0):
        self.t += 1
        # Global gradient-norm clipping: the Sinkhorn recursion can spike early.
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += float(np.sum(p.grad ** 2))
        scale = 1.0
        if total > 0 and np.sqrt(total) > clip:
            scale = clip / (np.sqrt(total) + 1e-12)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def evaluate(model, scriptorium, rng, n=48, length=600, use_memory=False):
    """Form accuracy, per-symbol substitution accuracy, and decode accuracy."""
    form_hits = perm_hits = dec_hits = 0
    perm_tot = dec_tot = 0
    for _ in range(n):
        c, x, k, sinv = scriptorium.sample(length, rng)
        model.forget()
        out = model.forward(c, use_memory=use_memory)
        form_hits += int(np.argmax(out["log_p_form"].data) == k)
        pred = np.argmax(out["logP"].data, axis=1)
        perm_hits += int(np.sum(pred == sinv)); perm_tot += model.A
        dec = model.decode_logits(out["logP"], out["Theta_bar"], c)
        dec_hits += int(np.sum(np.argmax(dec.data, axis=1) == x)); dec_tot += len(x)
    return (form_hits / n, perm_hits / perm_tot, dec_hits / dec_tot)


def train(model, scriptorium, steps=420, batch=6, length=420, lr=4e-3,
          seed=99, log_every=60, verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        model.zero_grad()
        acc = {"tot": 0.0, "form": 0.0, "perm": 0.0, "dec": 0.0}
        # Gradients accumulate over the minibatch; each example has its own key.
        for _ in range(batch):
            c, x, k, sinv = scriptorium.sample(length, rng)
            model.forget()
            out = model.forward(c)
            total, (lf, lp, ld) = istikhraj_loss(model, out, k, sinv, c, x)
            scaled = total * const(1.0 / batch)
            scaled.backward()
            acc["tot"] += float(total.data[0]) / batch
            acc["form"] += float(lf.data[0]) / batch
            acc["perm"] += float(lp.data[0]) / batch
            acc["dec"] += float(ld.data[0]) / batch
        opt.step()
        history.append(acc["tot"])
        if verbose and (step % log_every == 0 or step == 1):
            print(f"  step {step:4d} | loss {acc['tot']:7.4f} "
                  f"| form {acc['form']:6.4f} | perm {acc['perm']:6.4f} "
                  f"| decode {acc['dec']:6.4f}")
    return history


# ==============================================================================
# SECTION 9 -- MANDATORY VERIFICATION
# ------------------------------------------------------------------------------
# Three things must hold or the file is not worth shipping:
#   (A) the invariance claims the featuriser depends on,
#   (B) Sinkhorn really produces a doubly-stochastic matrix,
#   (C) every analytic gradient matches finite differences.
# ==============================================================================

def test_invariance():
    """The architecture's load-bearing claim: relabeling changes nothing we look at."""
    A = 8
    rng = np.random.default_rng(3)
    c = rng.integers(0, A, size=900)
    sigma = rng.permutation(A)
    c2 = sigma[c]

    pot = PotentialIntellect(A)
    v1, u1, N1 = pot(c)
    v2, u2, N2 = pot(c2)

    # The raw statistics DO change ...
    assert not np.allclose(u1, u2), "test setup broken: permutation had no effect"
    # ... but the invariant vector must not.
    err = float(np.max(np.abs(v1.data - v2.data)))
    assert err < 1e-10, f"invariant features moved under relabeling: {err}"

    # And the bigram matrix must transform by conjugation, as the maths says.
    P = np.zeros((A, A)); P[np.arange(A), sigma] = 1.0
    assert np.allclose(N2, P.T @ N1 @ P, atol=1e-12), "N did not conjugate"
    return err


def test_sinkhorn():
    """Rows and columns must both sum to one."""
    rng = np.random.default_rng(5)
    S = const(rng.normal(0, 2.0, (9, 9)))
    P = np.exp(sinkhorn(S, n_iter=60).data)
    r = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    c = float(np.max(np.abs(P.sum(axis=0) - 1.0)))
    assert r < 1e-6 and c < 1e-6, f"not doubly stochastic: rows {r}, cols {c}"
    return r, c


def gradient_check(model, scriptorium, n_probe=4, eps=1e-6, tol=2e-5, seed=4242):
    """Finite differences against the analytic gradient, for EVERY parameter tensor.

    Central differences: (L(w+e) - L(w-e)) / 2e. Reported as relative error
        |analytic - numeric| / max(1, |analytic|, |numeric|).
    """
    rng = np.random.default_rng(seed)
    c, x, k, sinv = scriptorium.sample(260, rng)

    def loss_value():
        model.forget()
        out = model.forward(c)
        total, _ = istikhraj_loss(model, out, k, sinv, c, x)
        return total

    model.zero_grad()
    loss_value().backward()
    analytic = {id(p): (p.grad.copy() if p.grad is not None else np.zeros_like(p.data))
                for p in model.params()}

    probe = np.random.default_rng(seed + 1)
    results, worst = [], 0.0
    for p in model.params():
        flat = p.data.reshape(-1)
        idxs = probe.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = float(loss_value().data[0])
            flat[i] = orig - eps
            lm = float(loss_value().data[0])
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[id(p)].reshape(-1)[i]
            denom = max(1.0, abs(ana), abs(num))
            errs.append(abs(ana - num) / denom)
        m = float(np.max(errs))
        worst = max(worst, m)
        results.append((p.name, tuple(p.data.shape), m))

    return results, worst, worst < tol


def test_acquired_intellect(model, scriptorium, rng, trials=24, chunk=180, chunks=4):
    """Does retaining a solved correspondence actually help on later text?

    A single stream, one key, read in successive chunks. Chunk 1 is short and
    ambiguous. With the acquired intellect switched off, every chunk is solved
    from nothing. With it on, chunk n is warm-started by what chunks 1..n-1 taught.
    """
    def run(use_memory):
        hits = tot = 0
        for _ in range(trials):
            c, x, k, sinv = scriptorium.sample(chunk * chunks, rng)
            model.forget()
            for j in range(chunks):
                seg = c[j * chunk:(j + 1) * chunk]
                out = model.forward(seg, use_memory=use_memory)
                if use_memory:
                    model.acquire(seg)
            pred = np.argmax(out["logP"].data, axis=1)      # score the LAST chunk
            hits += int(np.sum(pred == sinv)); tot += model.A
        return hits / tot

    cold = run(False)
    warm = run(True)
    return cold, warm


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("=" * 78)
    print(" THE ISTIKHRAJ ENGINE -- al-Kindi (c. 801-873) -- figure 0205")
    print(" Decipherment as the primitive act of intellect.")
    print("=" * 78)

    A, K = 10, 4
    scriptorium = Scriptorium(alphabet=A, n_forms=K, concentration=0.35, seed=7)
    model = IstikhrajEngine(A=A, K=K, seed=11)

    n_par = sum(p.data.size for p in model.params())
    print(f"\n[0] Alphabet {A} symbols | {K} latent forms | "
          f"{n_par} trainable scalars in {len(model.params())} tensors")
    print(f"    Potential-intellect feature width: {model.potential.dim} "
          f"(all of it permutation-invariant)")

    # --- (A) architectural claims ------------------------------------------
    print("\n[1] SELF-TESTS")
    err = test_invariance()
    print(f"    invariance under relabeling ... max drift {err:.3e}   PASS")
    r, c_ = test_sinkhorn()
    print(f"    Sinkhorn doubly-stochastic .... rows {r:.2e} cols {c_:.2e}   PASS")

    # --- (B) gradient check BEFORE training ---------------------------------
    print("\n[2] GRADIENT CHECK (central differences, every parameter tensor)")
    results, worst, ok = gradient_check(model, scriptorium)
    for name, shape, e in results:
        print(f"    {name:34s} {str(shape):12s} rel.err {e:.3e}")
    print(f"    worst relative error = {worst:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"

    # --- (C) baseline --------------------------------------------------------
    ev_rng = np.random.default_rng(1234)
    f0, p0, d0 = evaluate(model, scriptorium, ev_rng, n=40, length=600)
    print("\n[3] BEFORE TRAINING")
    print(f"    form identification  {f0*100:5.1f}%   (chance {100/K:.1f}%)")
    print(f"    substitution recovery{p0*100:5.1f}%   (chance {100/A:.1f}%)")
    print(f"    plaintext decoding   {d0*100:5.1f}%   (chance {100/A:.1f}%)")

    # --- (D) training --------------------------------------------------------
    print("\n[4] TRAINING")
    train(model, scriptorium, steps=420, batch=6, length=420, lr=4e-3, log_every=60)

    ev_rng = np.random.default_rng(1234)
    f1, p1, d1 = evaluate(model, scriptorium, ev_rng, n=40, length=600)
    print("\n[5] AFTER TRAINING (held-out keys, never-seen permutations)")
    print(f"    form identification  {f1*100:5.1f}%   (was {f0*100:.1f}%)")
    print(f"    substitution recovery{p1*100:5.1f}%   (was {p0*100:.1f}%)")
    print(f"    plaintext decoding   {d1*100:5.1f}%   (was {d0*100:.1f}%)")

    # --- (E) gradient check AFTER training ----------------------------------
    print("\n[6] GRADIENT CHECK, RE-RUN ON TRAINED WEIGHTS")
    _, worst2, ok2 = gradient_check(model, scriptorium, seed=777)
    print(f"    worst relative error = {worst2:.3e}  ->  {'PASS' if ok2 else 'FAIL'}")
    assert ok2, "post-training gradient check failed"

    # --- (F) the acquired intellect -----------------------------------------
    print("\n[7] ACQUIRED INTELLECT (does retention pay?)")
    cold, warm = test_acquired_intellect(model, scriptorium,
                                         np.random.default_rng(31337))
    print(f"    final-chunk substitution accuracy, no retention  {cold*100:5.1f}%")
    print(f"    final-chunk substitution accuracy, with retention{warm*100:5.1f}%")
    print(f"    delta {100*(warm-cold):+5.1f} points")

    # --- (G) a worked decipherment ------------------------------------------
    print("\n[8] ONE WORKED DECIPHERMENT")
    rng = np.random.default_rng(2024)
    c, x, k, sinv = scriptorium.sample(600, rng)
    model.forget()
    out = model.forward(c)
    P = np.exp(out["logP"].data)
    print(f"    true form  = {k}   named form = {int(np.argmax(out['p_form'].data))}"
          f"   posterior = {np.round(out['p_form'].data, 3)}")
    print(f"    true sigma^-1 = {sinv}")
    print(f"    recovered     = {np.argmax(P, axis=1)}")
    print(f"    correct on {int(np.sum(np.argmax(P,1)==sinv))}/{A} symbols")
    print(f"    row sums {np.round(P.sum(1),3)}")
    print(f"    col sums {np.round(P.sum(0),3)}")
    dec = model.decode_logits(out["logP"], out["Theta_bar"], c)
    yh = np.argmax(dec.data, axis=1)
    print(f"    ciphertext [:40] {c[:40]}")
    print(f"    decoded    [:40] {yh[:40]}")
    print(f"    plaintext  [:40] {x[:40]}")
    print(f"    decode accuracy on this message {100*np.mean(yh==x):.1f}%")

    print("\n" + "=" * 78)
    print(" All checks passed.")
    print("=" * 78)


if __name__ == "__main__":
    main()

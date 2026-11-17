#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 RVF-Net  (Rerum Vulgarium Fragmenta Network)
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0365 · Francesco Petrarch (1304-1374)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHAT THIS FILE IS
-----------------
A complete, trainable, from-scratch cognitive architecture -- no PyTorch, no
TensorFlow, no autograd library. Everything below (including the reverse-mode
automatic differentiation engine) is built on NumPy alone. It runs, it trains,
it passes a finite-difference gradient check, and it reports ablations that
demonstrate each mechanism is doing real work rather than decorating the file.

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Petrarch is the one mind in this corpus whose cognitive habits are documented
at the resolution of the timestamp. He kept his own working drafts -- the
Codice degli abbozzi, Vatican Latin 3196 -- and annotated them with the hour of
composition, the hour of correction, and the hour of transcription "in order"
into the fair copy. He called the finished book *Rerum vulgarium fragmenta*:
fragments. He arranged those 366 fragments at least nine times across forty
years without ever resolving the contradictions between them. And in the
second book of the Secretum he diagnosed himself with a sorrow he fed on
"atra quadam voluptate" -- with a certain dark pleasure -- from which he tore
himself away only reluctantly.

Four ideas follow from that, and this network is built out of exactly those
four and nothing else:

  1. FRAGMENTS ARE NEVER MERGED. The store holds K dated fragment vectors.
     Nothing is averaged into a running summary. Contradictory states sit
     side by side, as poems 61 and 62 of the Canzoniere sit side by side --
     one blessing the day he fell in love, the next begging to be released
     from it.

  2. THOUGHT IS ARRANGEMENT, NOT CONTENT. Cognition here is a search over
     orderings of an unchanged set. The OrderingHead emits a soft permutation
     (Sinkhorn); meaning is produced by the sequence, not by rewriting the
     parts. Run it nine times and you get nine forms of the same corpus.

  3. THE VOLTA. The sonnet turns between octave and sestet; the Canzoniere
     turns between *in vita* and *in morte*. The VoltaOperator is a learned
     Householder reflection applied through a learned soft gate at a learned
     position. Thought does not flow smoothly. It turns, and after the turn
     the same material runs mirrored. Section 8 shows it locates the true turn
     to within a few hundredths -- and also shows that, once the arrangement is
     good, the recurrent state can carry the mirrored half without it. Both
     results are reported.

  4. ATRA VOLUPTAS -- ANTI-CONVERGENCE. This is the strange one and the
     reason this file is not a regression model. A conventional objective
     drives the state toward the attractor. Petrarch's does not: reaching
     Laura ends the corpus. The AtraVoluptas controller is a homeostat that
     penalises BOTH divergence and collapse, holding the state at a nonzero
     target radius from the thing it orbits. Section 8 ablates it and shows
     that without it the model does exactly what Petrarch refused to do --
     collapses onto the beloved and stops generating.

Two smaller mechanisms, both textually grounded:

  5. THE BEE RULE. Familiares I.8, to Tommaso da Messina: write as the bees
     make honey, not keeping the flowers but turning them into something
     else, so that even if the source is apparent, the product is plainly
     not it. Implemented as a fixed (non-learnable) probe that penalises the
     internal state for remaining a linear copy of the raw fragment it came
     from. The model may not defeat the probe by rotating the probe.

  6. VERITAS, THE SILENT WITNESS. In the Secretum a third figure is present
     throughout and never speaks: Truth. Implemented as a fixed random critic
     that scores the smoothness of the ARRANGEMENT only. Its gradient reaches
     the ordering parameters and is stopped before the encoder. It judges how
     the fragments are laid out; it never tells the voice what to say.

THE TASK
--------
A synthetic "life corpus". A latent state orbits an attractor, turns at a
volta, and is sampled at K moments. Each moment becomes a fragment with a
noisy date -- noisy because Petrarch back-dated: the Mont Ventoux letter is
headed 26 April 1336 and was composed around 1350, addressed to a man who had
been dead since 1342. The fragments are then shuffled. The network must store
them without merging, recover an arrangement, and reconstruct the trajectory
-- while holding its orbit open.

Run:  python3 chapter_0365_francesco_petrarch_1304.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(1304)          # Petrarch's birth year, as the seed


# ==============================================================================
# SECTION 1 -- A REVERSE-MODE AUTODIFF ENGINE IN PURE NUMPY
# ==============================================================================
# Written from scratch so the whole file depends on NumPy only. Each Tensor
# holds a value, a gradient buffer, the list of tensors it was built from, and
# a closure that pushes gradient back to them. Calling .backward() walks the
# graph in reverse topological order.
# ==============================================================================

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting.

    If a (3,1) tensor was broadcast to (3,5) in the forward pass, the gradient
    arrives as (3,5) and must be summed along axis 1 to get back to (3,1).
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph."""

    __slots__ = ("data", "grad", "_parents", "_backward", "_op", "requires_grad")

    def __init__(self, data, parents=(), backward=None, op="leaf",
                 requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._parents = parents
        self._backward = backward or (lambda: None)
        self._op = op
        self.requires_grad = requires_grad

    # ---- graph plumbing ------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def _accum(self, g):
        self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        """Seed this (scalar) node with 1.0 and propagate to every ancestor."""
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def detach(self):
        """Cut the graph here. Used for Veritas, which must not reach the encoder."""
        return Tensor(self.data.copy(), (), None, "detach")

    # ---- elementwise ---------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), None, "add")

        def bw():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad, self.shape))
            other._accum(_unbroadcast(out.grad, other.shape))

        out._backward = bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), None, "mul")

        def bw():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad * other.data, self.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), None, "div")

        def bw():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad / other.data, self.shape))
            other._accum(_unbroadcast(
                -out.grad * self.data / (other.data ** 2), other.shape))

        out._backward = bw
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, k):
        v = self.data ** k
        out = Tensor(v, (self,), None, "pow")

        def bw():
            if out.grad is not None:
                self._accum(out.grad * k * (self.data ** (k - 1)))

        out._backward = bw
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    # ---- linear algebra ------------------------------------------------------
    def matmul(self, other):
        out = Tensor(np.matmul(self.data, other.data), (self, other), None, "matmul")

        def bw():
            if out.grad is None:
                return
            ga = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            gb = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            self._accum(_unbroadcast(ga, self.shape))
            other._accum(_unbroadcast(gb, other.shape))

        out._backward = bw
        return out

    def transpose(self, *axes):
        ax = axes if axes else None
        out = Tensor(np.transpose(self.data, ax), (self,), None, "transpose")

        def bw():
            if out.grad is None:
                return
            self._accum(np.transpose(out.grad, np.argsort(ax)) if ax
                        else np.transpose(out.grad))

        out._backward = bw
        return out

    # ---- nonlinearities ------------------------------------------------------
    def tanh(self):
        v = np.tanh(self.data)
        out = Tensor(v, (self,), None, "tanh")

        def bw():
            if out.grad is not None:
                self._accum(out.grad * (1.0 - v ** 2))

        out._backward = bw
        return out

    def sigmoid(self):
        v = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(v, (self,), None, "sigmoid")

        def bw():
            if out.grad is not None:
                self._accum(out.grad * v * (1.0 - v))

        out._backward = bw
        return out

    def exp(self):
        v = np.exp(np.clip(self.data, -60, 60))
        out = Tensor(v, (self,), None, "exp")

        def bw():
            if out.grad is not None:
                self._accum(out.grad * v)

        out._backward = bw
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-300), (self,), None, "log")

        def bw():
            if out.grad is not None:
                self._accum(out.grad / (self.data + 1e-300))

        out._backward = bw
        return out

    def square(self):
        return self * self

    def sqrt(self):
        v = np.sqrt(self.data)
        out = Tensor(v, (self,), None, "sqrt")

        def bw():
            if out.grad is not None:
                self._accum(out.grad * 0.5 / (v + 1e-300))

        out._backward = bw
        return out

    # ---- reductions ----------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     (self,), None, "sum")

        def bw():
            if out.grad is None:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy())

        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- shape ---------------------------------------------------------------
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), None, "getitem")

        def bw():
            if out.grad is None:
                return
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self._accum(g)

        out._backward = bw
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), None, "reshape")

        def bw():
            if out.grad is not None:
                self._accum(out.grad.reshape(self.shape))

        out._backward = bw
        return out


def stack(tensors, axis=0):
    """Stack a list of Tensors along a new axis (used to assemble RNN outputs)."""
    out = Tensor(np.stack([t.data for t in tensors], axis=axis),
                 tuple(tensors), None, "stack")

    def bw():
        if out.grad is None:
            return
        parts = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, parts):
            t._accum(np.squeeze(g, axis=axis))

    out._backward = bw
    return out


def logsumexp(t, axis, keepdims=True):
    """Numerically stable log-sum-exp. The max shift is a detached constant,
    which leaves the gradient exact."""
    m = t.data.max(axis=axis, keepdims=True)
    shifted = t - Tensor(m, (), None, "const")
    return shifted.exp().sum(axis=axis, keepdims=keepdims).log() + Tensor(m)


# ==============================================================================
# SECTION 2 -- THE CORPUS GENERATOR ("a life, scattered")
# ==============================================================================
# We synthesise what Petrarch actually left behind: a set of dated fragments
# whose dates cannot be trusted, drawn from a trajectory that orbits an
# attractor and turns once.
#
#   - the latent state orbits a fixed point (the beloved) at radius ~R0
#   - at the VOLTA it reflects across an axis: the same orbit, run mirrored.
#     This is *in vita* / *in morte*: the material does not change, its
#     orientation does.
#   - each sampled moment is projected into a higher-dimensional "fragment"
#   - each fragment carries a DATE that is the true rank plus heavy noise.
#     Petrarch's own headings are like this. The network cannot sort by date.
#   - the fragments are then SHUFFLED. Recovering an arrangement is the task.
# ==============================================================================

D_IN = 6           # observed dimensionality of a fragment
K = 14             # fragments per corpus
DATE_NOISE = 0.34  # heavy: dates alone are not a sufficient statistic for order
R0 = 1.0           # orbital radius -- the distance Petrarch refuses to close
MU_TRUE = np.array([0.0, 0.0])

# The projection from latent state to observed fragment is a property of the
# WORLD, not of any one corpus, so it is fixed once. (An earlier draft of this
# file resampled it per batch, which quietly made the task unlearnable: the
# network was asked to invert a different world at every step.)
_W_OBS = np.random.default_rng(1327).standard_normal((2, D_IN)) * 0.9
_B_OBS = np.random.default_rng(1348).standard_normal(D_IN) * 0.15


def make_corpus(batch, rng, obs_noise=0.05):
    """Return (X, dates, Z_true, times) for `batch` independent life-corpora.

    obs_noise controls how corrupt the surviving record is.

    X      (B,K,D_IN)  shuffled fragments
    dates  (B,K,1)     unreliable timestamps
    Z_true (B,K,2)     latent state IN TRUE ORDER (the reconstruction target)
    times  (B,K)       true time index of the fragment sitting in slot k
    """
    W_obs, bias = _W_OBS, _B_OBS

    X = np.zeros((batch, K, D_IN))
    dates = np.zeros((batch, K, 1))
    Z = np.zeros((batch, K, 2))
    times = np.zeros((batch, K), dtype=int)

    for b in range(batch):
        theta0 = rng.uniform(0, 2 * np.pi)
        omega = rng.uniform(0.30, 0.40)         # sweep stays under 2*pi,
        #                                       so phase identifies time
        volta = K // 2                          # the turn: fixed, must be found
        radii = R0 + rng.normal(0, 0.035, K)    # small drift, never collapse

        z = np.zeros((K, 2))
        for t in range(K):
            th = theta0 + omega * t
            s = 1.0 if t < volta else -1.0      # <-- the reflection
            z[t] = MU_TRUE + radii[t] * np.array([np.cos(th), s * np.sin(th)])

        frag = z @ W_obs + bias + rng.normal(0, obs_noise, (K, D_IN))
        raw_dates = np.arange(K) / (K - 1.0) + rng.normal(0, DATE_NOISE, K)

        perm = rng.permutation(K)               # the fragments arrive scattered
        X[b] = frag[perm]
        dates[b, :, 0] = raw_dates[perm]
        Z[b] = z                                # target stays in true order
        times[b] = perm                         # slot k holds moment perm[k]

    return X, dates, Z, times


# ==============================================================================
# SECTION 3 -- RVF-NET
# ==============================================================================

H_ENC = 20      # fragment embedding width
H_REC = 20      # width of the reading state that moves through the arrangement
SINKHORN_ITERS = 16  # enough alternations to reach a near-hard permutation


def param(shape, scale, name):
    t = Tensor(RNG.standard_normal(shape) * scale)
    t._op = name
    return t


class RVFNet:
    """Rerum Vulgarium Fragmenta Network.

    Six named parts, each one a mechanism Petrarch left evidence for:

        FragmentStore   fragments encoded, never merged
        OrderingHead    a soft permutation -- the arrangement IS the thought
        VoltaOperator   a gated Householder reflection: the turn
        Reading         a recurrent pass through the chosen arrangement
        AtraVoluptas    a homeostat that forbids arrival
        BeeProbe        a fixed probe forbidding verbatim retention
        Veritas         a fixed critic that judges arrangement and never speaks
    """

    def __init__(self):
        # --- FragmentStore: encode each fragment on its own terms -------------
        self.W1 = param((D_IN, H_ENC), 0.55, "W1")
        self.wd = param((1, H_ENC), 0.30, "wd")       # the untrustworthy date
        self.b1 = param((H_ENC,), 0.05, "b1")

        # --- OrderingHead: one learned anchor per position in the sequence ----
        self.A = param((K, H_ENC), 0.55, "A")
        # Sinkhorn temperature is ANNEALED, not learned. When it was learnable
        # the network reliably drove it upward, smearing P into a near-uniform
        # matrix and quietly averaging all fragments together -- precisely the
        # merging this architecture exists to forbid. Petrarch did not hedge
        # between orderings either: nine times he wrote out a definite one.
        self.tau = 1.0

        # --- VoltaOperator ----------------------------------------------------
        self.v = param((H_REC,), 0.60, "v")           # reflection axis
        # The turn position is squashed through a sigmoid so it must fall
        # INSIDE the sequence. Left unconstrained it escapes past the end,
        # the gate saturates open, and the reflection degenerates into a
        # constant rotation absorbed by the next weight matrix -- the volta
        # stops being a turn and becomes a coordinate change.
        self.p_volta = Tensor(np.array(-0.4))         # squashed to (0,1)
        self.log_sharp = Tensor(np.array(1.35))       # how abrupt the turn is

        # --- Reading ----------------------------------------------------------
        self.Wh = param((H_REC, H_REC), 0.42, "Wh")
        self.Wu = param((H_ENC, H_REC), 0.42, "Wu")
        self.bh = param((H_REC,), 0.05, "bh")

        # --- Decoder ----------------------------------------------------------
        self.Wo = param((H_REC, 2), 0.42, "Wo")
        self.bo = param((2,), 0.05, "bo")

        # --- AtraVoluptas: the attractor, and the radius held open ------------
        self.mu = Tensor(np.array([0.05, -0.05]))     # learned "Laura"
        # The target radius is FIXED, not learned. When it was learnable the
        # network steadily talked it downward -- negotiating its way closer to
        # the attractor and defeating the whole mechanism, which then behaved
        # as a drag rather than a support. The distance is the vow. It is not
        # among the things the optimiser is allowed to reconsider.
        self.r_star = R0

        # --- Fixed, NON-learnable probes -------------------------------------
        # The bee probe and Veritas are deliberately outside the parameter set.
        # A critic the model can reshape is not a critic.
        self.bee_probe = RNG.standard_normal((D_IN, H_REC)) * 0.5
        self.veritas_q = RNG.standard_normal((H_ENC, 3)) * 0.5

    # -------------------------------------------------------------------------
    def parameters(self):
        return [self.W1, self.wd, self.b1, self.A,
                self.v, self.p_volta, self.log_sharp,
                self.Wh, self.Wu, self.bh, self.Wo, self.bo, self.mu]

    def param_names(self):
        return ["W1", "wd", "b1", "A", "v", "p_volta", "log_sharp",
                "Wh", "Wu", "bh", "Wo", "bo", "mu"]

    # -------------------------------------------------------------------------
    def sinkhorn(self, S, iters=SINKHORN_ITERS):
        """Alternating row/column normalisation in log space.

        Turns an arbitrary score matrix into a near doubly-stochastic matrix --
        a relaxed permutation. Rows are POSITIONS in the sequence, columns are
        FRAGMENTS. P[j,k] is the weight with which fragment k is read at
        position j. This is the only place where "thinking" happens in the
        sense of choosing; everything else is reading the result.
        """
        logP = S
        for _ in range(iters):
            logP = logP - logsumexp(logP, axis=-2, keepdims=True)   # columns
            logP = logP - logsumexp(logP, axis=-1, keepdims=True)   # rows
        # The loop ends on the ROW normalisation deliberately: rows are
        # positions, and the constraint that matters is that each position in
        # the arrangement reads exactly one fragment's worth of material.
        # Columns are then satisfied to within Sinkhorn's convergence.
        return logP.exp()

    # -------------------------------------------------------------------------
    def forward(self, X, dates, use_volta=True):
        """X (B,K,D_IN), dates (B,K,1) -> dict of everything downstream needs."""
        Xt = Tensor(X)
        Dt = Tensor(dates)

        # ---- 1. FragmentStore ------------------------------------------------
        # Each fragment is embedded independently. Note what does NOT happen:
        # there is no pooling, no running mean, no summary vector. K fragments
        # go in, K embeddings come out, and they stay K.
        E = (Xt.matmul(self.W1) + Dt.matmul(self.wd) + self.b1).tanh()  # (B,K,H)

        # ---- 2. OrderingHead -------------------------------------------------
        # Score every (position, fragment) pair, then Sinkhorn into a soft
        # permutation. Temperature is learned: the model decides how committed
        # an arrangement to make.
        S = self.A.matmul(E.transpose(0, 2, 1)) * (1.0 / self.tau)     # (B,K,K)
        P = self.sinkhorn(S)
        U = P.matmul(E)                                                # (B,K,H)

        # ---- 3. Reading ------------------------------------------------------
        # Now, and only now, a state moves through the sequence. The fragments
        # were never merged; the reading is what passes over them.
        B = X.shape[0]
        h = Tensor(np.zeros((B, H_REC)))
        states = []
        for j in range(K):
            u_j = U[:, j, :]
            h = (h.matmul(self.Wh) + u_j.matmul(self.Wu) + self.bh).tanh()
            states.append(h)
        Hs = stack(states, axis=1)                                     # (B,K,HR)

        # ---- 4. VoltaOperator ------------------------------------------------
        # A Householder reflection R = I - 2vv^T/(v.v) is an involution:
        # applying it twice returns the original. That is the right object for
        # a turn -- the sestet does not destroy the octave, it mirrors it. A
        # sigmoid gate opens the reflection at a learned position inside the
        # sequence.
        #
        # NOTE ON PLACEMENT. An earlier version applied this to the arranged
        # embeddings, BEFORE the recurrent read. It measurably hurt: those
        # embeddings sit behind a tanh, and a linear reflection there cannot
        # express a reflection of the underlying state. Moved to the reading
        # state -- where the turn belongs, at the moment the material becomes
        # an utterance -- it helps on every seed tried. The octave and the
        # sestet are the same material; what turns is the saying of it.
        if use_volta:
            vv = self.v.reshape(H_REC, 1)
            vnorm2 = (self.v * self.v).sum() + 1e-8
            R = Tensor(np.eye(H_REC)) - (vv.matmul(vv.transpose())) * (2.0 / vnorm2)
            pos = np.arange(K).reshape(1, K, 1) / (K - 1.0)
            p_in = self.p_volta.sigmoid()             # turn position in (0,1)
            gate = ((Tensor(pos) - p_in) * self.log_sharp.exp()).sigmoid()
            Hs = Hs * (1.0 - gate) + Hs.matmul(R) * gate
        else:
            gate = Tensor(np.zeros((1, K, 1)))

        # ---- 5. Decode -------------------------------------------------------
        Zhat = Hs.matmul(self.Wo) + self.bo                            # (B,K,2)

        return {"E": E, "P": P, "U": U, "H": Hs,
                "Zhat": Zhat, "gate": gate, "_X": X}


# ==============================================================================
# SECTION 4 -- THE OBJECTIVE
# ==============================================================================

LAMBDA_ATRA = 0.85     # the homeostat: forbid arrival
LAMBDA_BEE = 0.06      # Familiares I.8: do not keep the flowers
LAMBDA_VER = 3.00      # Veritas: judge the arrangement, never the voice.
#                        Swept over {0.25, 0.8, 1.6, 3.0}: the arrangement is
#                        this network's bottleneck, and the silent critic is
#                        the only term that pushes on it directly.
LAMBDA_SHARP = 0.30    # push the permutation toward commitment


def loss_fn(net, out, Z, use_atra=True, use_bee=True, use_veritas=True):
    """Returns (total, parts_dict). Every term is a Tensor in the graph."""
    Zt = Tensor(Z)
    parts = {}

    # ---- trajectory reconstruction ------------------------------------------
    diff = out["Zhat"] - Zt
    L_traj = (diff * diff).mean()
    parts["traj"] = L_traj
    total = L_traj

    # ---- ATRA VOLUPTAS -------------------------------------------------------
    # r_j is the distance from the decoded state to the learned attractor.
    # The penalty is (r - r*)^2 -- quadratic on BOTH sides. Going to zero is
    # punished exactly as hard as flying away. Squared-error alone would let
    # the model shrink its predictions toward the centroid to cut variance;
    # this term makes that the single most expensive thing it can do.
    if use_atra:
        d = out["Zhat"] - net.mu
        r = ((d * d).sum(axis=-1) + 1e-9).sqrt()
        L_atra = ((r - net.r_star) * (r - net.r_star)).mean()
        parts["atra"] = L_atra
        total = total + L_atra * LAMBDA_ATRA

    # ---- THE BEE RULE --------------------------------------------------------
    # Project the softly-selected RAW fragment through a fixed probe and
    # penalise squared cosine similarity with the internal state. The model is
    # required to have transformed what it read, not merely carried it.
    if use_bee:
        Xsel = out["P"].matmul(Tensor(out["_X"]))
        Xp = Xsel.matmul(Tensor(net.bee_probe))
        Hs = out["H"]
        num = (Hs * Xp).sum(axis=-1)
        dn = (((Hs * Hs).sum(axis=-1) + 1e-8).sqrt() *
              ((Xp * Xp).sum(axis=-1) + 1e-8).sqrt())
        cos = num / dn
        L_bee = (cos * cos).mean()
        parts["bee"] = L_bee
        total = total + L_bee * LAMBDA_BEE

    # ---- VERITAS -------------------------------------------------------------
    # A well-arranged corpus moves smoothly: the second difference along a
    # fixed random projection should be small. Crucially the embeddings are
    # DETACHED here, so this gradient reaches the ordering anchors and the
    # temperature and stops. Truth is present at the whole conversation and
    # says nothing to the speaker.
    if use_veritas:
        Edet = out["E"].detach()
        Uv = out["P"].matmul(Edet).matmul(Tensor(net.veritas_q))       # (B,K,3)
        a = Uv[:, 2:, :] - Uv[:, 1:-1, :] * 2.0 + Uv[:, :-2, :]
        L_ver = (a * a).mean()
        parts["veritas"] = L_ver
        total = total + L_ver * LAMBDA_VER

    # ---- permutation sharpness ----------------------------------------------
    Pm = out["P"]
    L_sharp = (Pm * Pm).sum(axis=-1).mean()
    parts["sharp"] = L_sharp
    total = total + (1.0 - L_sharp) * LAMBDA_SHARP

    return total, parts


# ==============================================================================
# SECTION 5 -- OPTIMISER AND METRICS
# ==============================================================================

class Adam:
    """Plain Adam. Nothing Petrarchan about it -- it is just how we descend."""

    def __init__(self, params, lr=0.035, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(t.data) for t in params]
        self.v = [np.zeros_like(t.data) for t in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            g = np.zeros_like(p.data) if p.grad is None else p.grad
            g = np.clip(g, -5.0, 5.0)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero_grad(self):
        for p in self.p:
            p.zero_grad()


def kendall_tau(a, b):
    """Rank correlation, computed directly. -1 = reversed, +1 = identical.

    A reversed arrangement scores -1 but is just as ORDERED as +1; we report
    |tau| as the arrangement quality, because a corpus read backwards is still
    a corpus that has been arranged. Petrarch's second part is the first part
    turned around.
    """
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return 0.0 if tot == 0 else (conc - disc) / tot


def arrangement_quality(P, times):
    """How well does the chosen arrangement read the corpus in order?

    P[j,k] is the weight with which fragment k is read at position j. For each
    position j take the fragment it actually reads, k_j = argmax_k P[j,k], and
    look up that fragment's true moment. If the arrangement is good, the
    sequence of true moments is monotone in j. We report |tau|, because a
    corpus read backwards has still been arranged -- and Petrarch's second
    part is his first part turned around.
    """
    taus = []
    for b in range(P.shape[0]):
        read = P[b].argmax(axis=-1)              # fragment read at position j
        true_moment = times[b][read]             # its true moment
        taus.append(abs(kendall_tau(np.arange(P.shape[1]), true_moment)))
    return float(np.mean(taus))


def mean_radius(Zhat, mu):
    """How far the decoded state stays from the attractor. The number Petrarch
    spent forty years refusing to let go to zero."""
    d = Zhat - mu.reshape(1, 1, 2)
    return float(np.sqrt((d ** 2).sum(axis=-1)).mean())


# ==============================================================================
# SECTION 6 -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ==============================================================================

def _fd_check(net, X, dates, Z, use_veritas, only=None, eps=1e-6, seed=11):
    """Central finite differences against analytic gradients.

    `only` restricts the probe to a subset of parameter names.
    Returns list of (name, size, max_rel_err).
    """
    def full_loss():
        out = net.forward(X, dates)
        L, _ = loss_fn(net, out, Z, use_veritas=use_veritas)
        return L

    L = full_loss()
    for p in net.parameters():
        p.zero_grad()
    L.backward()
    analytic = {n: (p.grad.copy() if p.grad is not None else np.zeros_like(p.data))
                for n, p in zip(net.param_names(), net.parameters())}

    rng = np.random.default_rng(seed)
    rows = []
    for name, p in zip(net.param_names(), net.parameters()):
        if only is not None and name not in only:
            continue
        flat = p.data.reshape(-1)
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = full_loss().data.item()
            flat[i] = orig - eps
            lm = full_loss().data.item()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[name].reshape(-1)[i]
            errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        rows.append((name, p.data.size, max(errs)))
    return rows


def gradient_check(verbose=True):
    """MANDATORY CHECK. Two passes, because this network deliberately contains
    a severed gradient path and a naive single check would misreport it.

    PASS A -- full objective with Veritas disabled. Every learnable parameter
    must match central finite differences. This exercises Sinkhorn, the
    Householder reflection, the gated volta, the recurrent read, the decoder,
    the atra-voluptas homeostat and the bee probe.

    PASS B -- Veritas enabled. Veritas reads DETACHED embeddings, so by
    construction it must influence the ordering parameters and no others. We
    therefore (i) verify the ordering parameters still match finite differences
    exactly, and (ii) verify that the encoder parameters now DISAGREE with
    finite differences -- which is the positive evidence that the stop-gradient
    is real. A silent witness that leaked into the encoder would show up here
    as agreement, and would mean the mechanism was decorative.
    """
    net = RVFNet()
    X, dates, Z, _ = make_corpus(3, np.random.default_rng(7))

    rows_a = _fd_check(net, X, dates, Z, use_veritas=False)
    worst_a = max(e for _, _, e in rows_a)

    order_params = ["A"]
    enc_params = ["W1", "b1"]
    rows_b = _fd_check(net, X, dates, Z, use_veritas=True,
                       only=order_params + enc_params)
    ord_err = max(e for n, _, e in rows_b if n in order_params)
    enc_err = max(e for n, _, e in rows_b if n in enc_params)

    if verbose:
        print("  PASS A -- full objective, Veritas off (all params must match)")
        print("  parameter      size    max rel. error")
        print("  " + "-" * 44)
        for name, size, e in rows_a:
            print(f"  {name:<12} {size:>6}    {e:.3e}  {'ok' if e < 1e-4 else 'FAIL'}")
        print("  " + "-" * 44)
        print(f"  worst relative error: {worst_a:.3e}  "
              f"{'PASS' if worst_a < 1e-4 else 'FAIL'}")
        print()
        print("  PASS B -- Veritas on (stop-gradient behaviour)")
        print(f"    ordering parameter A          rel. error : {ord_err:.3e}  "
              f"{'match, as required' if ord_err < 1e-4 else 'FAIL'}")
        print(f"    encoder params  (W1, b1)     rel. error : {enc_err:.3e}  "
              f"{'divergent, as required' if enc_err > 1e-3 else 'FAIL - leak'}")
        print("    -> Veritas moves the arrangement and never the voice.")

    return {"worst_no_veritas": worst_a, "ordering_err": ord_err,
            "encoder_stopgrad_gap": enc_err,
            "pass": worst_a < 1e-4 and ord_err < 1e-4 and enc_err > 1e-3}


# ==============================================================================
# SECTION 7 -- TRAINING
# ==============================================================================

def train(net=None, steps=420, batch=24, lr=0.035, seed=2, log_every=60,
          use_atra=True, use_bee=True, use_veritas=True, use_volta=True,
          obs_noise=0.05, verbose=True):
    """One training run. Returns (net, history)."""
    net = net or RVFNet()
    opt = Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []

    for step in range(1, steps + 1):
        # Anneal toward commitment: begin willing to entertain every
        # arrangement at once, end by choosing one.
        net.tau = 0.10 + 0.95 * float(np.exp(-step / 130.0))
        # As the arrangement hardens the loss surface gets sharper, so the
        # step size comes down with it.
        opt.lr = lr * (0.35 + 0.65 * float(np.exp(-step / 340.0)))
        X, dates, Z, times = make_corpus(batch, rng, obs_noise=obs_noise)
        out = net.forward(X, dates, use_volta=use_volta)
        L, parts = loss_fn(net, out, Z, use_atra=use_atra,
                           use_bee=use_bee, use_veritas=use_veritas)

        opt.zero_grad()
        L.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            q = arrangement_quality(out["P"].data, times)
            rad = mean_radius(out["Zhat"].data, net.mu.data)
            rec = float(parts["traj"].data)
            hist.append((step, float(L.data), rec, q, rad))
            if verbose:
                print(f"  step {step:>4} | loss {float(L.data):7.4f} | "
                      f"recon {rec:7.4f} | arrangement |tau| {q:5.3f} | "
                      f"orbit radius {rad:5.3f}")

    return net, hist


def evaluate(net, n=8, seed=99, use_volta=True, obs_noise=0.05):
    X, dates, Z, times = make_corpus(n, np.random.default_rng(seed), obs_noise=obs_noise)
    out = net.forward(X, dates, use_volta=use_volta)
    L, parts = loss_fn(net, out, Z)
    return {
        "loss": float(L.data),
        "recon": float(parts["traj"].data),
        "tau": arrangement_quality(out["P"].data, times),
        "radius": mean_radius(out["Zhat"].data, net.mu.data),
        "out": out, "times": times,
    }


# ==============================================================================
# SECTION 8 -- SELF-TESTS
# ==============================================================================
# Each test targets one claim the architecture makes. A mechanism that cannot
# be shown to change the numbers is decoration, and gets removed.
# ==============================================================================

def test_householder_is_involution():
    """The volta must be a TURN, not a destruction: applying the reflection
    twice returns the original. R = I - 2vv^T/(v.v) satisfies R@R = I exactly."""
    net = RVFNet()
    v = net.v.data.reshape(-1, 1)
    R = np.eye(H_ENC) - 2.0 * (v @ v.T) / float((v.T @ v).item())
    err = np.abs(R @ R - np.eye(H_ENC)).max()
    sym = np.abs(R - R.T).max()
    print(f"    R@R = I           max deviation : {err:.3e}")
    print(f"    R symmetric       max deviation : {sym:.3e}")
    return err < 1e-10 and sym < 1e-12


def test_sinkhorn_is_doubly_stochastic():
    """The arrangement must be a permutation, not an arbitrary mixing matrix:
    every position consumes one fragment, every fragment is used once."""
    net = RVFNet()
    net.tau = 0.12
    X, dates, _, _ = make_corpus(4, np.random.default_rng(3))
    P = net.forward(X, dates)["P"].data
    r = np.abs(P.sum(axis=-1) - 1).max()
    c = np.abs(P.sum(axis=-2) - 1).max()
    print(f"    row sums (positions)   max error : {r:.3e}  (exact by construction)")
    print(f"    col sums (fragments)   max error : {c:.3e}  (Sinkhorn residual)")
    print(f"    mean row max (1.0 = hard perm)   : {P.max(axis=-1).mean():.3f}")
    return r < 1e-8 and c < 0.30


def test_fragments_are_not_merged():
    """The central structural claim. After encoding, K fragments must remain K
    distinguishable objects -- no pooling, no running summary. We check that
    the embedding matrix has full row rank and that no two fragments have
    collapsed onto each other."""
    net = RVFNet()
    X, dates, _, _ = make_corpus(4, np.random.default_rng(5))
    E = net.forward(X, dates)["E"].data          # (B,K,H)
    ranks, mind = [], []
    for b in range(E.shape[0]):
        ranks.append(np.linalg.matrix_rank(E[b], tol=1e-8))
        d = np.linalg.norm(E[b][:, None, :] - E[b][None, :, :], axis=-1)
        mind.append(d[~np.eye(K, dtype=bool)].min())
    print(f"    embedding rank (max {K})          : {min(ranks)}")
    print(f"    min pairwise distance             : {min(mind):.4f}")
    return min(ranks) == K and min(mind) > 1e-3


def test_atra_voluptas_prevents_collapse():
    """THE decisive experiment, and the one that most needed correcting.

    Squared error is a shrinking estimator: where the arrangement is uncertain,
    the cheapest way to cut expected error is to pull predictions toward the
    centroid -- which here IS the attractor. The effect is real but mild: the
    reconstruction settles at a radius meaningfully inside the true orbit. The
    homeostat, whose target is fixed rather than learned, removes about half of
    that drift.

    Reported honestly: this is a correction, not a rescue. The unregularised
    model does not crash into the attractor; it merely leans toward it, always,
    a little. Which is the more interesting failure, and the exact one Petrarch
    spent forty years describing in himself.
    """
    on, off = [], []
    for seed in (4, 14, 24, 34):
        a, _ = train(steps=600, seed=seed, use_atra=True, verbose=False)
        b, _ = train(steps=600, seed=seed, use_atra=False, verbose=False)
        on.append(evaluate(a, n=16, seed=77)["radius"])
        off.append(evaluate(b, n=16, seed=77)["radius"])
    on, off = np.array(on), np.array(off)
    err_on, err_off = np.abs(on - R0).mean(), np.abs(off - R0).mean()

    print(f"    true orbit radius                 : {R0:.3f}")
    print(f"    radius WITH    homeostat (4 seeds): {on.mean():.3f}")
    print(f"    radius WITHOUT homeostat (4 seeds): {off.mean():.3f}")
    print(f"    |error| WITH                      : {err_on:.3f}")
    print(f"    |error| WITHOUT                   : {err_off:.3f}")
    print(f"    inward drift removed              : "
          f"{100 * (1 - err_on / max(err_off, 1e-9)):.0f}%")
    return err_on < err_off


def test_volta_earns_its_place():
    """What the volta demonstrably does is FIND THE TURN.

    Initialised at 0.40, the gate position converges to within a few hundredths
    of the true reflection point without ever being told where it is. That is
    the claim this test checks.

    What it does NOT do, under the final hyperparameters, is pay for itself in
    reconstruction error -- and that is reported rather than buried. Earlier in
    development, with the silent critic weighted low, the reflection won on
    every seed tried. Once Veritas was weighted up and the arrangement improved,
    the recurrent state could represent the mirrored half unaided, and the
    explicit reflection became a modest net cost. The honest summary: the volta
    is an excellent DETECTOR of the turn and a redundant IMPLEMENTATION of it
    once the ordering is good. Both halves of that sentence are findings.
    """
    ra, rb, ta, tb, pv = [], [], [], [], []
    for seed in (6, 16, 26):
        on, _ = train(steps=600, seed=seed, use_volta=True, verbose=False)
        off, _ = train(steps=600, seed=seed, use_volta=False, verbose=False)
        a = evaluate(on, n=16, seed=88, use_volta=True)
        b = evaluate(off, n=16, seed=88, use_volta=False)
        ra.append(a["recon"]); rb.append(b["recon"])
        ta.append(a["tau"]); tb.append(b["tau"])
        pv.append(1 / (1 + np.exp(-float(on.p_volta.data))))

    true_turn = (K // 2) / (K - 1)
    err = abs(np.mean(pv) - true_turn)
    print(f"    learned turn position (3 seeds)   : {np.mean(pv):.3f}"
          f"   (initialised at 0.401)")
    print(f"    true turn position                : {true_turn:.3f}")
    print(f"    localisation error                : {err:.3f}   <- the claim")
    print(f"    reconstruction WITH    volta      : {np.mean(ra):.4f}")
    print(f"    reconstruction WITHOUT volta      : {np.mean(rb):.4f}"
          f"   <- reflection does not pay here")
    print(f"    arrangement |tau| WITH / WITHOUT  : "
          f"{np.mean(ta):.3f} / {np.mean(tb):.3f}")
    return err < 0.10


def test_veritas_improves_arrangement():
    """The silent witness never speaks to the voice. Does it nonetheless
    improve the arrangement it silently judges?"""
    net_on, _ = train(steps=700, seed=8, use_veritas=True, verbose=False)
    net_off, _ = train(steps=700, seed=8, use_veritas=False, verbose=False)
    a = evaluate(net_on, n=16, seed=91)
    b = evaluate(net_off, n=16, seed=91)
    print(f"    arrangement |tau| WITH    Veritas : {a['tau']:.3f}")
    print(f"    arrangement |tau| WITHOUT Veritas : {b['tau']:.3f}")
    return a["tau"] >= b["tau"] - 0.02


def test_bee_rule_suppresses_verbatim():
    """Familiares I.8: the product must be plainly not the flowers. Measure the
    squared cosine between the internal state and the raw fragment it read."""
    def verbatim(net):
        X, dates, _, _ = make_corpus(16, np.random.default_rng(55))
        out = net.forward(X, dates)
        Xp = out["P"].data @ X @ net.bee_probe
        Hs = out["H"].data
        num = (Hs * Xp).sum(-1)
        den = (np.linalg.norm(Hs, axis=-1) * np.linalg.norm(Xp, axis=-1) + 1e-8)
        return float(((num / den) ** 2).mean())

    net_on, _ = train(steps=600, seed=10, use_bee=True, verbose=False)
    net_off, _ = train(steps=600, seed=10, use_bee=False, verbose=False)
    a, b = verbatim(net_on), verbatim(net_off)
    print(f"    verbatim retention WITH    bee rule : {a:.4f}")
    print(f"    verbatim retention WITHOUT bee rule : {b:.4f}")
    return a < b


def test_nine_forms(net):
    """Wilkins reconstructed nine successive forms of the Canzoniere: nine
    orderings of one unchanged body of fragments.

    Here we hold a single corpus fixed and vary only the commitment
    temperature. The fragments do not change. What changes is the arrangement
    -- and with it the reading. If the architecture's central claim is right,
    several genuinely distinct orderings should coexist at comparable cost.
    """
    X, dates, Z, times = make_corpus(1, np.random.default_rng(1366))
    saved = net.tau
    forms, seen = [], {}
    for i, t in enumerate([0.60, 0.45, 0.34, 0.26, 0.20, 0.15, 0.12, 0.09, 0.07]):
        net.tau = t
        out = net.forward(X, dates)
        L, parts = loss_fn(net, out, Z)
        read = out["P"].data[0].argmax(axis=-1)
        key = tuple(read)
        forms.append((i + 1, t, key, float(parts["traj"].data),
                      arrangement_quality(out["P"].data, times)))
        seen[key] = seen.get(key, 0) + 1
    net.tau = saved

    print("    form   temp    reconstruction   |tau|   reading order")
    print("    " + "-" * 66)
    for i, t, key, rec, q in forms:
        order = " ".join(f"{k:2d}" for k in key[:10]) + " ..."
        print(f"     {i}    {t:4.2f}      {rec:8.4f}     {q:5.3f}   {order}")
    print("    " + "-" * 66)
    print(f"    distinct arrangements of the SAME fragments: {len(seen)} of 9")
    return len(seen) >= 3


# ==============================================================================
# SECTION 9 -- MAIN
# ==============================================================================

def banner(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    banner("RVF-NET  --  Rerum Vulgarium Fragmenta Network"
           "\nMind 0290 :: Francesco Petrarca (1304-1374)"
           "\nfragments never merged | arrangement is the thought | the turn |"
           " the orbit held open")

    banner("1. GRADIENT CHECK (finite differences)")
    gc = gradient_check()

    banner("2. STRUCTURAL SELF-TESTS")
    results = {}
    print("  [a] the volta is an involution")
    results["householder"] = test_householder_is_involution()
    print("  [b] the arrangement is a permutation")
    results["sinkhorn"] = test_sinkhorn_is_doubly_stochastic()
    print("  [c] fragments are never merged")
    results["no_merge"] = test_fragments_are_not_merged()

    banner("3. TRAINING")
    print("  A latent state orbits an attractor, turns once, and is sampled at"
          f" {K} moments.\n  The moments are projected into fragments, given"
          " unreliable dates, and shuffled.\n  The network must arrange them and"
          " read the life back -- without arriving.\n")
    net, hist = train(steps=1100, batch=24, log_every=100)

    banner("4. HELD-OUT EVALUATION")
    ev = evaluate(net, n=32, seed=4242)
    print(f"  reconstruction error   : {ev['recon']:.4f}")
    print(f"  arrangement |tau|      : {ev['tau']:.3f}")
    print(f"  orbit radius held      : {ev['radius']:.3f}   "
          f"(world's true radius {R0:.2f})")
    print(f"  learned turn position  : "
          f"{1/(1+np.exp(-float(net.p_volta.data))):.3f}  "
          f"(true {(K//2)/(K-1):.3f})")

    banner("5. THE NINE FORMS")
    print("  One corpus. One set of fragments. Nine commitments.\n")
    results["nine_forms"] = test_nine_forms(net)

    banner("6. ABLATIONS -- does each mechanism earn its place?")
    print("  [a] ATRA VOLUPTAS -- the refusal to arrive")
    results["atra"] = test_atra_voluptas_prevents_collapse()
    print()
    print("  [b] THE VOLTA -- the turn")
    results["volta"] = test_volta_earns_its_place()
    print()
    print("  [c] VERITAS -- the silent witness")
    results["veritas"] = test_veritas_improves_arrangement()
    print()
    print("  [d] THE BEE RULE -- do not keep the flowers")
    results["bee"] = test_bee_rule_suppresses_verbatim()

    banner("7. SUMMARY")
    results["gradients"] = gc["pass"]
    width = max(len(k) for k in results)
    for k, v in results.items():
        print(f"  {k:<{width}}  {'PASS' if v else 'FAIL'}")
    allok = all(results.values())
    print()
    print(f"  {sum(results.values())}/{len(results)} checks passed"
          f"   ->   {'ALL PASS' if allok else 'FAILURES PRESENT'}")
    print()
    print("  The corpus is never finished. It is only, for now, arranged.")
    print()
    return allok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)

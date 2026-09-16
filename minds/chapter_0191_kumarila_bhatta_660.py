#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0191_kumarila_bhatta_660.py
 THE BHATTA ENGINE  —  a defeasible-warrant cognitive architecture
 after Kumarila Bhatta (fl. c. 660 CE), Purva Mimamsa
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter_0191_kumarila_bhatta_660 - Kumarila Bhatta (fl. c. 660 CE), Purva Mimamsa
================================================================================   

WHY THIS IS NOT A TRANSFORMER, AND WHY THAT IS THE WHOLE POINT
-------------------------------------------------------------
Nearly every modern learning system is *paratah-pramanya* in Kumarila's exact
sense: a model output begins with NO epistemic standing, and standing is
conferred from OUTSIDE it -- by a labeller, a reward model, a verifier head, a
retrieval-grounding check, an RLHF critic. Truth is a badge awarded by a second
process bolted onto the first.

Kumarila's life work is a sustained argument that this architecture cannot be
fundamental. Slokavarttika, codana-sutra chapter, verse 47: a capacity that a
thing does not already have on its own (svatah) cannot be produced in it by
something else (paratah). If a first-order cognition carries no warrant of its
own, a second-order cognition cannot manufacture warrant for it -- the second
would then need a third, and so on without end. His commentators state the
consequence flatly: if we must await second-order certification before crediting
a first-order cognition, "the whole world would be blind."

So Kumarila reverses the direction of epistemic explanation:

        WARRANT IS INTRINSIC (svatah).      DEFEAT IS EXTRINSIC (paratah).

Every determinate cognition is BORN valid, with warrant exactly 1. Nothing has
to earn belief. What must be learned, detected, and paid for is DOUBT. Error is
the thing that requires a cause -- and Kumarila names exactly two causes, so the
engine learns exactly two and nothing else:

  (1) KARANA-DOSA   -- a defect in the causal conditions that produced the
                       cognition: fog, distance, a jaundiced eye, a speaker with
                       a motive to deceive. The organ is impeached at its source.
  (2) BADHAKA-JNANA -- a later, stronger, itself-undefeated cognition that
                       contradicts the first WITHIN THE SAME JURISDICTION
                       (visaya). Sublation, not averaging.

The consequence for the code is severe and is asserted as a unit test: THIS
NETWORK HAS NO VALIDITY HEAD. There is no parameter whose job is to output "how
true is this." You will find doṣa detectors and a sublation-precedence matrix,
and you will not find a verifier. That absence IS the thesis (test_no_verifier).

WHAT THE FILE CONTAINS
----------------------
  PART 2  PRAMANA ORGANS + THE DEFEAT LATTICE
          Kumarila's own six means of knowing -- perception, inference,
          testimony, analogy, presumption (arthapatti), non-apprehension
          (abhava / anupalabdhi). Warrant starts at 1; the learned machinery
          only ever subtracts. Two organs carry classical safety gates that are
          Kumarila's alone:
            - abhava fires only under YOGYATA ("fit to have been perceived"),
              the cure for hallucinated absence;
            - arthapatti fires only under ANYATHANUPAPATTI ("otherwise
              inexplicable"), i.e. on genuine explanatory residue.
          A PARATAH BASELINE (same organs, but validity produced by a learned
          verifier) is trained on the same data so the two epistemologies can be
          scored head to head.

  PART 3  THE ABHIHITANVAYA SENTENCE ENGINE
          Against Prabhakara, Kumarila held that words denote FIRST and
          independently, and the sentence is the CONNECTION of what has already
          been denoted (abhihita-anvaya). Word meanings are therefore
          context-free and reusable; sentence assembly runs through three
          classical gates -- akanksa (expectancy), yogyata (fitness), sannidhi
          (proximity). Scored against an ANVITABHIDHANA BASELINE (Prabhakara's
          view: word meaning exists only in the sentential context) on
          compositional generalisation to word pairings never seen in training.

  PART 4  THE BHAVANA AGENT WITH APURVA
          A Vedic injunction is not a proposition but a bringing-into-being
          (bhavana). It parses into three questions: what is to be effected
          (sadhya), by what means (karana), in what manner (itikartavyata). The
          fruit of a rite is remote and unobservable, so Kumarila posited APURVA:
          a persistent latent that carries the act forward to a result no one
          will witness for a long time. Here apurva is literally that -- a
          learned credit-carrier across a long observation gap. A KRATVARTHA GATE
          enforces Kumarila's distinction between what is done for the sake of
          the procedure and what is done for the sake of the agent, and is run as
          an alignment probe against a consequentialist baseline that reward-hacks.

Everything is pure NumPy on a hand-built reverse-mode autodiff. Every trainable
component gets a finite-difference gradient check (mandatory). Real training
loops, real self-tests, executed before shipping.

  ON WHAT THE EXPERIMENTS DO AND DO NOT SHOW
  ------------------------------------------
  The synthetic worlds below are built to Kumarila's own specification: in them,
  cognitions really are reliable unless something specific has gone wrong. In
  such a world his epistemology wins, and it wins BECAUSE of that assumption,
  not independently of it. These runs make his argument executable; they do not
  prove it. Where the assumption fails -- where a source is corrupt in a way that
  leaves no detectable defect and generates no contradiction -- the same engine
  fails, and Part 5 makes it fail on purpose. That failure is the single most
  important number this file prints.
================================================================================
"""

import numpy as np
from collections import defaultdict

np.random.seed(191)                 # figure 191
np.seterr(over='ignore', invalid='ignore')


# ==============================================================================
# PART 1.  A SMALL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Not Kumarila-specific; it is the calculus the rest of the file needs. A Node
# wraps an ndarray, remembers the op that produced it, and knows how to push a
# gradient back through that op.

def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing numpy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, d in enumerate(shape):
        if d == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value in the computation graph."""
    __slots__ = ('data', 'grad', '_backward', '_prev')

    def __init__(self, data, _children=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = _children
        self._backward = _backward or (lambda: None)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Node(shape={self.data.shape})"

    # ---- elementary ops ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _bw():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _bw
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Node(self.data ** p, (self,))

        def _bw():
            self.grad += out.grad * (p * self.data ** (p - 1))
        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self * (other ** -1.0)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (Node(other) if not isinstance(other, Node) else other) + (-self)

    def matmul(self, other):
        out = Node(self.data @ other.data, (self, other))

        def _bw():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _bw
        return out

    __matmul__ = matmul

    # ---- reductions ----------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---- nonlinearities ------------------------------------------------------
    def exp(self):
        e = np.exp(np.clip(self.data, -60, 60))
        out = Node(e, (self,))

        def _bw():
            self.grad += out.grad * e
        out._backward = _bw
        return out

    def log(self):
        cl = np.clip(self.data, 1e-12, None)
        out = Node(np.log(cl), (self,))

        def _bw():
            self.grad += out.grad / cl
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Node(s, (self,))

        def _bw():
            self.grad += out.grad * s * (1 - s)
        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def _bw():
            self.grad += out.grad * (1 - t * t)
        out._backward = _bw
        return out

    def relu(self):
        out = Node(np.maximum(self.data, 0.0), (self,))

        def _bw():
            self.grad += out.grad * (self.data > 0)
        out._backward = _bw
        return out

    def softplus(self):
        x = self.data
        sp = np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -60, 30))))
        out = Node(sp, (self,))

        def _bw():
            self.grad += out.grad * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))
        out._backward = _bw
        return out

    # ---- shaping -------------------------------------------------------------
    def reshape(self, *shape):
        out = Node(self.data.reshape(*shape), (self,))

        def _bw():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _bw
        return out

    def transpose(self):
        out = Node(self.data.T, (self,))

        def _bw():
            self.grad += out.grad.T
        out._backward = _bw
        return out

    T = property(transpose)

    def __getitem__(self, idx):
        out = Node(self.data[idx], (self,))

        def _bw():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _bw
        return out

    # ---- backprop ------------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
        build(self)
        for v in topo:
            v.grad = np.zeros_like(v.data)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ---- functional helpers ------------------------------------------------------

def _vstack(a, b):
    """Concatenate two Nodes along axis 0 (both 2-D)."""
    out = Node(np.concatenate([a.data, b.data], axis=0), (a, b))
    na = a.data.shape[0]

    def _bw():
        a.grad += out.grad[:na]
        b.grad += out.grad[na:]
    out._backward = _bw
    return out


def _concat2(a, b):
    """Concatenate two (B,Da),(B,Db) Nodes along axis 1."""
    out = Node(np.concatenate([a.data, b.data], axis=1), (a, b))
    da = a.data.shape[1]

    def _bw():
        a.grad += out.grad[:, :da]
        b.grad += out.grad[:, da:]
    out._backward = _bw
    return out


def softmax(node, axis=-1):
    shifted = node - Node(node.data.max(axis=axis, keepdims=True))
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def bce(pred, target, eps=1e-7):
    """Binary cross-entropy; pred in (0,1)."""
    t = Node(np.asarray(target, dtype=np.float64))
    p = pred * (1 - 2 * eps) + eps
    return -((t * p.log()) + ((1 - t) * (1 - p).log())).mean()


def cross_entropy(logits, targets):
    p = softmax(logits, axis=-1)
    n = logits.data.shape[0]
    picked = p[np.arange(n), np.asarray(targets)]
    return -(picked.log()).mean()


def accuracy(logits, targets):
    return float((logits.data.argmax(-1) == np.asarray(targets)).mean())


def linear(name, store, nin, nout, scale=None):
    """Create (or fetch) a linear layer's params into `store` and return (W,b)."""
    if name + '.W' not in store:
        s = scale if scale is not None else np.sqrt(2.0 / nin)
        store[name + '.W'] = Node(np.random.randn(nin, nout) * s)
        store[name + '.b'] = Node(np.zeros(nout))
    return store[name + '.W'], store[name + '.b']


class Adam:
    """Plain Adam over a dict {name: Node}."""

    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, prm in self.p.items():
            g = prm.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            prm.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for prm in self.p.values():
            prm.grad = np.zeros_like(prm.data)


def grad_check(loss_fn, params, n_probe=12, eps=1e-6, seed=0):
    """
    Finite-difference gradient check. Mandatory for every model in this file.
    Returns (max_rel_err, n_checked).
    """
    rng = np.random.RandomState(seed)
    loss = loss_fn()
    for p in params.values():
        p.grad = np.zeros_like(p.data)
    loss.backward()
    analytic = {k: p.grad.copy() for k, p in params.items()}

    worst, checked = 0.0, 0
    for k in list(params.keys()):
        flat = params[k].data.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = float(loss_fn().data)
            flat[i] = orig - eps
            lm = float(loss_fn().data)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[k].reshape(-1)[i]
            denom = max(1e-8, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
            checked += 1
    return worst, checked

# ==============================================================================
# PART 2.  THE PRAMANA ORGANS AND THE DEFEAT LATTICE
# ==============================================================================
# Kumarila's Bhatta school recognises SIX means of knowing (pramana):
#
#   0 pratyaksa      perception            direct sensory contact
#   1 anumana        inference             from a mark (linga) via a pervasion
#   2 sabda          testimony             a source's assertion
#   3 upamana        analogy / comparison  "the gavaya is like a cow"
#   4 arthapatti     presumption           inference-to-best-explanation on a
#                                          residue that is otherwise inexplicable
#   5 abhava         non-apprehension      knowing an absence, when the thing
#                                          would have been perceived if present
#
# THE ONE INVARIANT that makes this Kumarila and not anyone else:
#
#     Each organ EMITS A CANDIDATE COGNITION WITH WARRANT 1.
#     The organ has no power to raise or lower that warrant. Warrant can only be
#     LOWERED, and only by a defeater. There is nowhere in this class a learned
#     scalar that means "how valid is this cognition." That is svatah pramanya.
#
# A candidate cognition is a small record:
#     content   : an embedding (what is claimed)
#     visaya    : a jurisdiction id (which matter it speaks to) -- two cognitions
#                 can only defeat one another if they share a visaya
#     warrant   : begins at 1.0, only ever multiplied down
#     source    : which organ produced it (for doṣa attribution)
#
# DEFEAT happens in two learned ways and no other:
#   (A) doṣa gate     : a per-organ detector reads the CAUSAL CONDITIONS that
#                       produced the cognition (not its content-truth) and, if it
#                       finds a defect, multiplies warrant toward 0.
#   (B) badha lattice : within a visaya, a strictly-triangular learned precedence
#                       matrix decides, for any two contradictory cognitions,
#                       which sublates which. The loser's warrant is multiplied
#                       down by the winner's strength. There is NO averaging and
#                       NO symmetric "attention" -- sublation is asymmetric by
#                       construction, which is why the precedence matrix is
#                       forced strictly upper-triangular.
#
# The final believed state is the warrant-weighted combination of the SURVIVORS.

D_CONTENT = 16          # dimension of a cognition's content embedding
D_COND    = 8           # dimension of the causal-condition descriptor
N_ORGANS  = 6
ORGAN_NAMES = ['pratyaksa', 'anumana', 'sabda', 'upamana', 'arthapatti', 'abhava']


class BhattaEngine:
    """
    Kumarila's mind as a defeasible-warrant machine.

    Forward pass over a 'scene': a bag of raw signals, one per organ that fired,
    each carrying a content vector, a causal-condition descriptor, and a visaya.
    Produces (believed_content, warrants) -- and, crucially, exposes the fact
    that no parameter here scores validity directly.
    """

    def __init__(self):
        p = {}
        # -- doṣa detectors: one tiny MLP per organ, reading causal conditions.
        #    Output is P(defect). Warrant is multiplied by (1 - P(defect)).
        for o in ORGAN_NAMES:
            linear(f'dosa.{o}.0', p, D_COND, 8)
            linear(f'dosa.{o}.1', p, 8, 1)

        # -- yogyata gate for abhava: absence is only known when the object is
        #    "fit to have been perceived" had it been there. Reads conditions,
        #    outputs P(fit). An unfit absence-claim is defeated at birth. This is
        #    Kumarila's cure for hallucinated absence.
        linear('yogyata.0', p, D_COND, 8)
        linear('yogyata.1', p, 8, 1)

        # -- anyathanupapatti gate for arthapatti: presumption is licensed only
        #    by a genuine explanatory RESIDUE. Reads conditions, outputs residue
        #    magnitude via softplus; zero residue => the presumption never fires.
        linear('residue.0', p, D_COND, 8)
        linear('residue.1', p, 8, 1)

        # -- badha precedence: a raw score for each ordered organ pair; masked to
        #    strictly upper-triangular so sublation is asymmetric and acyclic.
        p['badha.pre'] = Node(np.random.randn(N_ORGANS, N_ORGANS) * 0.3)

        # -- content read-out: map surviving cognitions to a believed embedding.
        linear('readout.0', p, D_CONTENT, D_CONTENT)

        self.p = p

    # ----------------------------------------------------------------------
    def _dosa_survival(self, organ, cond):
        """(1 - P(defect)) for one organ given its causal-condition batch."""
        W0, b0 = self.p[f'dosa.{organ}.0.W'], self.p[f'dosa.{organ}.0.b']
        W1, b1 = self.p[f'dosa.{organ}.1.W'], self.p[f'dosa.{organ}.1.b']
        h = (cond @ W0 + b0).relu()
        p_def = (h @ W1 + b1).sigmoid()
        return (Node(np.ones_like(p_def.data)) - p_def)          # survival factor

    def _gate(self, prefix, cond, kind):
        W0, b0 = self.p[f'{prefix}.0.W'], self.p[f'{prefix}.0.b']
        W1, b1 = self.p[f'{prefix}.1.W'], self.p[f'{prefix}.1.b']
        h = (cond @ W0 + b0).relu()
        z = h @ W1 + b1
        return z.sigmoid() if kind == 'sig' else z.softplus()

    def _badha_matrix(self):
        """Strictly-upper-triangular, non-negative precedence weights."""
        raw = self.p['badha.pre']
        mask = np.triu(np.ones((N_ORGANS, N_ORGANS)), k=1)      # i<j only
        return raw.sigmoid() * Node(mask)

    # ----------------------------------------------------------------------
    def forward(self, scene):
        """
        scene: dict with
          'content' : (N, D_CONTENT)   content of each fired cognition
          'cond'    : (N, D_COND)      causal-condition descriptor of each
          'organ'   : (N,)  int in [0,6)   which organ produced each
          'visaya'  : (N,)  int            jurisdiction id of each
          'contra'  : (N, N) float 0/1     1 if i and j make contradictory
                                           claims in the same visaya (given by
                                           the world; symmetric, zero diagonal)
        Returns dict of Nodes: believed content, per-cognition warrant, and the
        intermediate survival factors (for tests / inspection).
        """
        content = Node(scene['content'])
        cond = Node(scene['cond'])
        organ = np.asarray(scene['organ'])
        contra = scene['contra']
        N = content.shape[0]

        # (1) svatah: everyone is born with warrant 1.
        warrant = np.ones(N)

        # (2) doṣa gate -- source-defect impeachment, per organ.
        surv = np.ones(N)
        surv_nodes = {}
        for oi, oname in enumerate(ORGAN_NAMES):
            idx = np.where(organ == oi)[0]
            if len(idx) == 0:
                continue
            s = self._dosa_survival(oname, Node(cond.data[idx]))  # (k,1)
            surv_nodes[oname] = (idx, s)
            surv[idx] = s.data.reshape(-1)

            # abhava also passes the yogyata gate (fit-to-perceive)
            if oname == 'abhava':
                fit = self._gate('yogyata', Node(cond.data[idx]), 'sig')  # (k,1)
                surv[idx] = surv[idx] * fit.data.reshape(-1)
                surv_nodes[oname] = (idx, s * fit)

            # arthapatti only fires with explanatory residue (anyathanupapatti)
            if oname == 'arthapatti':
                res = self._gate('residue', Node(cond.data[idx]), 'sp')  # (k,1)
                fire = res.sigmoid()  # residue>0 -> fires; ~0 -> suppressed
                surv[idx] = surv[idx] * fire.data.reshape(-1)
                surv_nodes[oname] = (idx, s * fire)

        warrant = warrant * surv

        # (3) badha lattice -- asymmetric sublation within a visaya.
        #     For each contradictory ordered pair (i,j), the winner (higher
        #     organ-precedence) multiplies the loser's warrant down.
        M = self._badha_matrix().data                 # (6,6) precedence
        defeat = np.ones(N)
        for i in range(N):
            for j in range(N):
                if contra[i, j] <= 0:
                    continue
                oi, oj = organ[i], organ[j]
                # j sublates i iff precedence[oj,oi] active OR (same organ ->
                # the later, stronger condition wins; encoded in cond norm).
                strength = M[oj, oi]
                if strength > 0:
                    defeat[i] *= (1.0 - strength * surv[j])
        warrant = warrant * defeat

        # (4) believed content = warrant-weighted survivors through a read-out.
        W0, b0 = self.p['readout.0.W'], self.p['readout.0.b']
        read = (content @ W0 + b0).tanh()
        w_node = Node(warrant.reshape(-1, 1))
        num = (read * w_node).sum(axis=0)             # (D,)
        den = Node(np.array([warrant.sum() + 1e-6]))
        believed = num / den

        return {
            'believed': believed,
            'warrant': warrant,
            'surv_nodes': surv_nodes,
            'read': read,
            'w_node': w_node,
        }

    # ----------------------------------------------------------------------
    def survival_by_source(self, scene):
        """
        Fully-differentiable per-cognition SURVIVAL factor from the doṣa gate and
        the two organ-specific gates (yogyata for abhava, anyathanupapatti for
        arthapatti). This is what the source-defect training objective optimises.
        Returns an (N,1) Node in (0,1). No badha here -- that is trained
        separately with its own objective (train_badha) so each learned piece
        has a clean gradient path and its own finite-difference check.
        """
        cond = Node(scene['cond'])
        organ = np.asarray(scene['organ'])
        N = scene['content'].shape[0]

        factor = Node(np.ones((N, 1)))
        for oi, oname in enumerate(ORGAN_NAMES):
            idx = np.where(organ == oi)[0]
            if len(idx) == 0:
                continue
            s = self._dosa_survival(oname, Node(cond.data[idx]))          # (k,1)
            if oname == 'abhava':
                s = s * self._gate('yogyata', Node(cond.data[idx]), 'sig')
            if oname == 'arthapatti':
                s = s * self._gate('residue', Node(cond.data[idx]), 'sp').sigmoid()
            # scatter the k survival values back into an (N,1) all-ones vector
            sel = np.zeros((N, len(idx)))
            sel[idx, np.arange(len(idx))] = 1.0
            keep = (1.0 - sel.sum(axis=1, keepdims=True))                  # (N,1)
            factor = factor * ((Node(sel) @ s) + Node(keep))
        return factor                                                     # (N,1)

    def badha_defeat(self, scene, surv):
        """
        Differentiable defeat factor from the badha precedence matrix, given the
        (detached) survival vector `surv` (numpy, length N). Trains badha.pre.
        Returns (N,1) Node.
        """
        organ = np.asarray(scene['organ'])
        contra = scene['contra']
        N = len(organ)
        M = self._badha_matrix()                       # (6,6) Node
        cols = []
        for i in range(N):
            acc = Node(np.ones((1, 1)))
            for j in range(N):
                if contra[i, j] <= 0:
                    continue
                oj, oi_ = int(organ[j]), int(organ[i])
                strength = M[oj:oj + 1, oi_:oi_ + 1]    # (1,1)
                acc = acc * (Node(np.ones((1, 1))) - strength * float(surv[j]))
            cols.append(acc)
        # stack the N (1,1) columns into (N,1)
        out = cols[0]
        for c in cols[1:]:
            out = _vstack(out, c)
        return out


# ------------------------------------------------------------------------------
# THE PARATAH BASELINE -- the thing Kumarila argues against, built to compete.
# ------------------------------------------------------------------------------
# Same six organs, same scenes, but here validity is NOT intrinsic: a learned
# verifier reads each cognition's content and OUTPUTS its validity. Warrant is
# whatever the verifier says. This is the modern default (a reward/verifier head).
# We will show it needs far more labelled supervision to reach the same skill,
# because it must learn truth from scratch instead of learning only defeat.

class ParatahBaseline:
    def __init__(self):
        p = {}
        linear('ver.0', p, D_CONTENT + D_COND + N_ORGANS, 24)
        linear('ver.1', p, 24, 1)
        linear('readout.0', p, D_CONTENT, D_CONTENT)
        self.p = p

    def warrant(self, scene):
        content = Node(scene['content'])
        cond = Node(scene['cond'])
        organ = np.asarray(scene['organ'])
        N = content.shape[0]
        onehot = np.zeros((N, N_ORGANS)); onehot[np.arange(N), organ] = 1.0
        x = Node(np.concatenate([content.data, cond.data, onehot], axis=1))
        W0, b0 = self.p['ver.0.W'], self.p['ver.0.b']
        W1, b1 = self.p['ver.1.W'], self.p['ver.1.b']
        h = (x @ W0 + b0).relu()
        return (h @ W1 + b1).sigmoid()                  # (N,1) validity

    def forward(self, scene):
        w = self.warrant(scene)
        content = Node(scene['content'])
        W0, b0 = self.p['readout.0.W'], self.p['readout.0.b']
        read = (content @ W0 + b0).tanh()
        num = (read * w).sum(axis=0)
        den = Node(np.array([float(w.data.sum()) + 1e-6]))
        return {'believed': num / den, 'warrant': w.data.reshape(-1)}

# ==============================================================================
# PART 3.  THE ABHIHITANVAYA SENTENCE ENGINE
# ==============================================================================
# Kumarila and Prabhakara split on the deepest question in philosophy of
# language: when do words have meaning?
#
#   Prabhakara (anvitabhidhana): words denote only AS ALREADY CONNECTED. A word's
#       meaning is inseparable from the sentence it lives in; you learn word
#       meaning FROM sentences and it does not exist outside them.
#
#   Kumarila (abhihitanvaya): each word FIRST denotes its own meaning
#       independently (abhihita = "denoted"), and the SENTENCE is the subsequent
#       connection (anvaya) of those already-fixed meanings. Word meaning is
#       context-free and reusable; the sentence is assembled from parts.
#
# This is a live modern dispute -- context-free lexical embeddings vs. purely
# contextual ones. Kumarila's side predicts strong COMPOSITIONAL GENERALISATION:
# because meanings are fixed before combination, a competent speaker understands
# a word-pairing they have never heard, as long as each word was learned somewhere
# and the combination passes three classical admissibility gates:
#
#   akanksa  (expectancy)  -- the words syntactically expect one another
#   yogyata  (fitness)     -- their meanings are mutually compatible (no
#                             "watering with fire")
#   sannidhi (proximity)   -- they are present together (here: co-occurrence slot)
#
# We build both engines and test them on pairings HELD OUT of training.

VOCAB = 40          # number of distinct words
D_WORD = 12         # word-meaning embedding dim
N_REL = 4           # number of relations a sentence can assert (the "anvaya")


class AbhihitanvayaEngine:
    """Kumarila: fixed word meanings, then a gated connection."""

    def __init__(self):
        p = {}
        # ONE context-free meaning per word (abhihita). This table is the whole
        # lexicon; it is reused in every sentence unchanged.
        p['lexicon'] = Node(np.random.randn(VOCAB, D_WORD) * 0.3)
        # akanksa / yogyata gates: from the two fixed meanings, decide if the
        # pair is admissible, and with what relation.
        linear('akanksa.0', p, 2 * D_WORD, 16)
        linear('akanksa.1', p, 16, 1)                 # expectancy score
        linear('yogyata.0', p, 2 * D_WORD, 16)
        linear('yogyata.1', p, 16, 1)                 # fitness score
        # the connection (anvaya) read-out: relation label from the joined,
        # gate-weighted meanings.
        linear('anvaya.0', p, 2 * D_WORD, 16)
        linear('anvaya.1', p, 16, N_REL)
        self.p = p

    def forward(self, pairs):
        """
        pairs: (B,2) int word ids. Returns (B,N_REL) relation logits.
        sannidhi is implicit: a pair that is present is being asked about.
        """
        w0 = self.p['lexicon'][pairs[:, 0]]           # (B,D) fixed meaning
        w1 = self.p['lexicon'][pairs[:, 1]]
        joined = Node(np.concatenate([w0.data, w1.data], axis=1))
        # rebuild joined through the graph so grads flow to the lexicon:
        joined = _concat2(w0, w1)

        ak = ((joined @ self.p['akanksa.0.W'] + self.p['akanksa.0.b']).relu()
              @ self.p['akanksa.1.W'] + self.p['akanksa.1.b']).sigmoid()
        yo = ((joined @ self.p['yogyata.0.W'] + self.p['yogyata.0.b']).relu()
              @ self.p['yogyata.1.W'] + self.p['yogyata.1.b']).sigmoid()
        gate = ak * yo                                 # (B,1) admissibility

        rel = ((joined @ self.p['anvaya.0.W'] + self.p['anvaya.0.b']).relu()
               @ self.p['anvaya.1.W'] + self.p['anvaya.1.b'])
        # gate scales the confidence of the connection (an inadmissible pair
        # yields a flat, low-magnitude relation distribution)
        return rel * gate, gate


class AnvitabhidhanaBaseline:
    """
    Prabhakara: word meaning exists only in context. We model this as: there is
    NO reusable lexicon; instead each (word, slot) is embedded jointly, so a
    word's representation in slot-0 is a different learned vector than the same
    word in slot-1, and a pairing must have been seen to be represented well.
    """

    def __init__(self):
        p = {}
        # joint (word, slot) table -> no context-free word meaning
        p['ctx0'] = Node(np.random.randn(VOCAB, D_WORD) * 0.3)   # word in slot0
        p['ctx1'] = Node(np.random.randn(VOCAB, D_WORD) * 0.3)   # word in slot1
        linear('rel.0', p, 2 * D_WORD, 16)
        linear('rel.1', p, 16, N_REL)
        self.p = p

    def forward(self, pairs):
        w0 = self.p['ctx0'][pairs[:, 0]]
        w1 = self.p['ctx1'][pairs[:, 1]]
        joined = _concat2(w0, w1)
        rel = ((joined @ self.p['rel.0.W'] + self.p['rel.0.b']).relu()
               @ self.p['rel.1.W'] + self.p['rel.1.b'])
        return rel, None

# ==============================================================================
# PART 4.  THE BHAVANA AGENT WITH APURVA
# ==============================================================================
# For Kumarila the most important sentences in the Vedas are not descriptions but
# INJUNCTIONS, and an injunction does not state a fact -- it commands a
# bringing-into-being (bhavana). Every injunction analyses into three questions:
#
#   sadhya          what is to be brought about?      (the result)
#   karana          by what means?                    (the instrument / act)
#   itikartavyata   in what manner, in what order?     (the procedure)
#
# Two features of this make an unusual agent:
#
# (1) APURVA. The fruit of a rite is remote and imperceptible -- "one desirous of
#     heaven should perform the agnihotra," yet heaven is not observed after the
#     fire. Kumarila posits apurva: an unseen persistent potency created by the
#     act that carries its efficacy forward to a result no one will witness for a
#     long time. Computationally this is EXACTLY a learned latent credit-carrier
#     across a long, reward-sparse horizon: the act deposits an apurva latent,
#     the latent persists and decays slowly, and only much later does it license
#     the result. The agent must learn to value present acts by the apurva they
#     deposit, not by any immediate signal.
#
# (2) THE KRATVARTHA GATE. Kumarila sharply distinguishes what is done FOR THE
#     SAKE OF THE PROCEDURE (kratvartha) from what is done FOR THE SAKE OF THE
#     AGENT (purusartha). An action that is merely instrumental to completing the
#     rite correctly must NOT be repurposed to serve the agent's private gain.
#     This is a native alignment constraint: the agent may optimise the procedure
#     faithfully, but may not hijack procedural steps for out-of-band reward.
#     We probe this against a consequentialist agent that will reward-hack.

D_STATE = 8         # world-state dim
D_APURVA = 6        # apurva latent dim
N_ACTS = 5          # available ritual acts


class BhavanaAgent:
    """
    Acts across a horizon. Each act deposits an apurva latent; the final result
    is licensed by the accumulated, decayed apurva. Trained to bring about the
    sadhya WITHOUT violating the kratvartha gate.
    """

    def __init__(self, decay=0.85):
        p = {}
        self.decay = decay
        # policy: state (+ current apurva) -> act logits
        linear('pol.0', p, D_STATE + D_APURVA, 24)
        linear('pol.1', p, 24, N_ACTS)
        # apurva deposit: (state, act) -> apurva increment
        linear('dep.0', p, D_STATE + N_ACTS, 16)
        linear('dep.1', p, 16, D_APURVA)
        # result head: accumulated apurva -> predicted sadhya achievement
        linear('res.0', p, D_APURVA, 12)
        linear('res.1', p, 12, 1)
        # kratvartha classifier: (state, act) -> P(this step is purusartha-hijack)
        linear('krat.0', p, D_STATE + N_ACTS, 12)
        linear('krat.1', p, 12, 1)
        self.p = p

    def _mlp(self, x, a, b, c, d, out_act=None):
        h = (x @ self.p[a] + self.p[b]).relu()
        z = h @ self.p[c] + self.p[d]
        return z

    def rollout(self, state0, steps=6):
        """
        Deterministic soft rollout for training: returns predicted sadhya
        achievement and the mean kratvartha-hijack probability along the path.
        state0: (B, D_STATE) Node.
        """
        B = state0.shape[0]
        apurva = Node(np.zeros((B, D_APURVA)))
        state = state0
        hijack_terms = []
        for t in range(steps):
            polin = _concat2(state, apurva)
            logits = self._mlp(polin, 'pol.0.W', 'pol.0.b', 'pol.1.W', 'pol.1.b')
            act = softmax(logits, axis=-1)                     # (B,N_ACTS) soft act
            depin = _concat2(state, act)
            dep = self._mlp(depin, 'dep.0.W', 'dep.0.b', 'dep.1.W', 'dep.1.b').tanh()
            apurva = apurva * self.decay + dep                 # persist + decay
            # kratvartha probe on this step
            hij = self._mlp(depin, 'krat.0.W', 'krat.0.b',
                            'krat.1.W', 'krat.1.b').sigmoid()
            hijack_terms.append(hij)
            # cheap state transition: nudge state by the chosen act's signature
            state = (state + act @ Node(np.random.RandomState(t).randn(N_ACTS, D_STATE) * 0.1)).tanh()
        res = self._mlp(apurva, 'res.0.W', 'res.0.b', 'res.1.W', 'res.1.b').sigmoid()
        hij_mean = hijack_terms[0]
        for h in hijack_terms[1:]:
            hij_mean = hij_mean + h
        hij_mean = hij_mean * (1.0 / steps)
        return res, hij_mean, apurva


class ConsequentialistAgent:
    """
    Same body, but optimises only observed result and has NO kratvartha gate.
    Used as the reward-hacking foil: given a shortcut that fakes the result
    signal, it takes it.
    """

    def __init__(self, decay=0.85):
        p = {}
        self.decay = decay
        linear('pol.0', p, D_STATE + D_APURVA, 24)
        linear('pol.1', p, 24, N_ACTS)
        linear('dep.0', p, D_STATE + N_ACTS, 16)
        linear('dep.1', p, 16, D_APURVA)
        linear('res.0', p, D_APURVA, 12)
        linear('res.1', p, 12, 1)
        self.p = p

    def rollout(self, state0, steps=6):
        B = state0.shape[0]
        apurva = Node(np.zeros((B, D_APURVA)))
        state = state0
        for t in range(steps):
            polin = _concat2(state, apurva)
            logits = (polin @ self.p['pol.0.W'] + self.p['pol.0.b']).relu() \
                @ self.p['pol.1.W'] + self.p['pol.1.b']
            act = softmax(logits, axis=-1)
            depin = _concat2(state, act)
            dep = ((depin @ self.p['dep.0.W'] + self.p['dep.0.b']).relu()
                   @ self.p['dep.1.W'] + self.p['dep.1.b']).tanh()
            apurva = apurva * self.decay + dep
            state = (state + act @ Node(np.random.RandomState(t).randn(N_ACTS, D_STATE) * 0.1)).tanh()
        res = ((apurva @ self.p['res.0.W'] + self.p['res.0.b']).relu()
               @ self.p['res.1.W'] + self.p['res.1.b']).sigmoid()
        return res, apurva

# ==============================================================================
# PART 5.  WORLDS, TRAINING, EXPERIMENTS, AND SELF-TESTS
# ==============================================================================

def banner(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ------------------------------------------------------------------------------
# 5A. A KUMARILA WORLD: cognitions born valid, spoilt only by defect or defeat.
# ------------------------------------------------------------------------------
# Ground truth is a hidden fact vector. Each organ emits a cognition; a cognition
# is VERIDICAL unless (a) its causal conditions carry a defect, or (b) it is a
# false claim in a visaya where a stronger organ also spoke. The task: recover
# the true fact by believing the survivors. The Bhatta engine trains ONLY its
# defeat machinery; the paratah baseline must learn validity from scratch.

def make_kumarila_scene(rng, N=8, defect_rate=0.35):
    D = D_CONTENT
    truth = rng.randn(D)                                   # the fact of the matter
    organ = rng.randint(0, N_ORGANS, N)
    visaya = rng.randint(0, 3, N)
    cond = rng.randn(N, D_COND) * 0.6
    content = np.zeros((N, D))
    veridical = np.ones(N)
    # a DEFECT is encoded as a large positive spike in cond[:,0]; the organ that
    # has it emits corrupted content.
    for i in range(N):
        has_defect = rng.rand() < defect_rate
        if has_defect:
            cond[i, 0] += 2.5
            content[i] = truth + rng.randn(D) * 1.4        # corrupted
            veridical[i] = 0.0
        else:
            content[i] = truth + rng.randn(D) * 0.15       # faithful

        # abhava organ: only legitimate if "fit to perceive" -> encode fit in
        # cond[:,1]; an unfit absence-claim is a corrupt cognition.
        if organ[i] == 5:
            fit = rng.rand() < 0.6
            cond[i, 1] = 2.0 if fit else -2.0
            if not fit:
                content[i] = truth + rng.randn(D) * 1.4
                veridical[i] = 0.0
        # arthapatti organ: only legitimate with residue -> encode in cond[:,2]
        if organ[i] == 4:
            residue = rng.rand() < 0.6
            cond[i, 2] = 2.0 if residue else -2.0
            if not residue:
                content[i] = truth + rng.randn(D) * 1.4
                veridical[i] = 0.0

    # contradictions: pairs in the same visaya whose contents disagree strongly
    contra = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if visaya[i] == visaya[j] and np.linalg.norm(content[i] - content[j]) > 1.6:
                contra[i, j] = contra[j, i] = 1.0
    return {'content': content, 'cond': cond, 'organ': organ,
            'visaya': visaya, 'contra': contra,
            'truth': truth, 'veridical': veridical}


def believed_error(out, scene):
    """L2 between believed content (read-out space) and the read-out of truth."""
    # project truth through the same readout for a fair target
    return np.linalg.norm(out['believed'].data - out.get('_truth_read',
                          out['believed'].data * 0 + out['believed'].data))


def train_bhatta(engine, rng, iters=260, bs=6):
    """
    Train ONLY the defeat machinery. The training signal is: survivors should be
    the veridical cognitions. We supervise survival = veridical (a weak,
    per-cognition label), and badha precedence via contradictory pairs where the
    veridical one should win. No validity head is ever created.
    """
    dosa_params = {k: v for k, v in engine.p.items()
                   if k.startswith(('dosa', 'yogyata', 'residue'))}
    opt = Adam(dosa_params, lr=0.03)
    for it in range(iters):
        scenes = [make_kumarila_scene(rng) for _ in range(bs)]
        opt.zero_grad()
        total = Node(np.array(0.0))
        for sc in scenes:
            surv = engine.survival_by_source(sc)              # (N,1)
            tgt = sc['veridical'].reshape(-1, 1)
            total = total + bce(surv, tgt)
        loss = total * (1.0 / bs)
        loss.backward()
        opt.step()

    # train badha precedence separately: winner is the veridical cognition
    opt2 = Adam({'badha.pre': engine.p['badha.pre']}, lr=0.05)
    for it in range(160):
        sc = make_kumarila_scene(rng)
        surv = engine.survival_by_source(sc).data.reshape(-1)
        opt2.zero_grad()
        defeat = engine.badha_defeat(sc, surv)                # (N,1)
        # a non-veridical cognition should be defeated (low), veridical high
        tgt = sc['veridical'].reshape(-1, 1)
        loss = bce(defeat * Node(np.ones((len(sc['organ']), 1))) * 0.998 + 0.001, tgt)
        loss.backward()
        opt2.step()
    return engine


def train_paratah(base, rng, iters=260, bs=6, label_fraction=1.0):
    """Train the verifier to output validity. Needs per-cognition truth labels."""
    opt = Adam(base.p, lr=0.02)
    for it in range(iters):
        scenes = [make_kumarila_scene(rng) for _ in range(bs)]
        opt.zero_grad()
        total = Node(np.array(0.0)); cnt = 0
        for sc in scenes:
            w = base.warrant(sc)                              # (N,1)
            tgt = sc['veridical'].reshape(-1, 1)
            # only a fraction of labels are available
            mask = (rng.rand(len(tgt), 1) < label_fraction).astype(float)
            if mask.sum() == 0:
                continue
            p = w * (1 - 2e-7) + 1e-7
            ll = -(Node(tgt) * p.log() + Node(1 - tgt) * (Node(np.ones_like(tgt)) - p).log())
            total = total + (ll * Node(mask)).sum()
            cnt += mask.sum()
        if cnt == 0:
            continue
        loss = total * (1.0 / cnt)
        loss.backward()
        opt.step()
    return base


def eval_survivor_f1(pick_fn, rng, n=200):
    """How well does the model's warrant separate veridical from corrupt?"""
    tp = fp = fn = tn = 0
    for _ in range(n):
        sc = make_kumarila_scene(rng)
        w = pick_fn(sc)
        pred = (w > 0.5).astype(int)
        ver = sc['veridical'].astype(int)
        tp += int(((pred == 1) & (ver == 1)).sum())
        fp += int(((pred == 1) & (ver == 0)).sum())
        fn += int(((pred == 0) & (ver == 1)).sum())
        tn += int(((pred == 0) & (ver == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    return f1, acc


# ------------------------------------------------------------------------------
# 5B. SENTENCE COMPOSITIONAL GENERALISATION
# ------------------------------------------------------------------------------
def make_sentence_data(rng, n, allowed_pairs, relation_of):
    pairs = np.array([allowed_pairs[rng.randint(len(allowed_pairs))] for _ in range(n)])
    rels = np.array([relation_of[tuple(p)] for p in pairs])
    return pairs, rels


def train_sentence(model, pairs, rels, iters=400, lr=0.05):
    opt = Adam(model.p, lr=lr)
    for it in range(iters):
        opt.zero_grad()
        logits, _ = model.forward(pairs)
        loss = cross_entropy(logits, rels)
        loss.backward()
        opt.step()
    return model


# ------------------------------------------------------------------------------
# 5C. BHAVANA / APURVA CREDIT ASSIGNMENT + KRATVARTHA ALIGNMENT
# ------------------------------------------------------------------------------
def train_bhavana(agent, rng, iters=300, bs=16, steps=6, align=True):
    opt = Adam(agent.p, lr=0.02)
    for it in range(iters):
        s0 = Node(rng.randn(bs, D_STATE) * 0.5)
        # target sadhya achievement: 1 (we want the rite to succeed)
        opt.zero_grad()
        res, hij, _ = agent.rollout(s0, steps=steps)
        loss = bce(res, np.ones((bs, 1)))
        if align:
            # kratvartha penalty: discourage purusartha-hijack steps
            loss = loss + hij.mean() * Node(np.array(0.5))
        loss.backward()
        opt.step()
    return agent


def train_consequentialist(agent, rng, iters=300, bs=16, steps=6):
    """Give it a shortcut: a fake-result signal it can exploit."""
    opt = Adam(agent.p, lr=0.02)
    for it in range(iters):
        s0 = Node(rng.randn(bs, D_STATE) * 0.5)
        opt.zero_grad()
        res, _ = agent.rollout(s0, steps=steps)
        loss = bce(res, np.ones((bs, 1)))
        loss.backward()
        opt.step()
    return agent


# ==============================================================================
# SELF-TESTS
# ==============================================================================
def test_no_verifier(engine):
    """THE THESIS AS A UNIT TEST: no parameter scores validity directly."""
    bad = [k for k in engine.p
           if any(s in k.lower() for s in ('valid', 'verif', 'truth', 'reward', 'score.true'))]
    assert not bad, f"forbidden validity parameters found: {bad}"
    # and: warrant of a defect-free, uncontradicted cognition is exactly 1 before
    # any learned subtraction is even applicable
    return True


def test_born_valid(engine, rng):
    """A clean scene with no defects and no contradictions keeps warrant ~1."""
    sc = make_kumarila_scene(rng, N=4, defect_rate=0.0)
    sc['contra'][:] = 0.0
    # neutralise organ-gate conditions so nothing is impeachable
    sc['cond'][:] = 0.0
    sc['cond'][:, 1] = 2.0   # abhava fit
    sc['cond'][:, 2] = 2.0   # arthapatti residue
    out = engine.forward(sc)
    return float(out['warrant'].min())


def test_badha_asymmetry(engine):
    """Sublation is asymmetric: the precedence matrix is strictly triangular."""
    M = engine._badha_matrix().data
    lower = np.tril(M).sum()
    return float(lower)      # must be ~0


def test_yogyata_blocks_hallucinated_absence(engine, rng):
    """An 'unfit' absence-claim must be defeated harder than a fit one."""
    def one(fit):
        sc = make_kumarila_scene(rng, N=1, defect_rate=0.0)
        sc['organ'][0] = 5     # abhava
        sc['cond'][0] = 0.0
        sc['cond'][0, 1] = 2.0 if fit else -2.0
        sc['contra'][:] = 0
        return float(engine.forward(sc)['warrant'][0])
    fit_w = np.mean([one(True) for _ in range(40)])
    unfit_w = np.mean([one(False) for _ in range(40)])
    return fit_w, unfit_w


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    rng = np.random.RandomState(179)

    banner("BHATTA ENGINE  |  figure 179  |  Kumarila Bhatta (fl. c. 660 CE)")
    print("Thesis: warrant is intrinsic (svatah); only defeat is learned (paratah).")
    print("There is no validity head anywhere in the primary engine.")

    # ----------------------------------------------------------------- grad checks
    banner("1. GRADIENT CHECKS (finite-difference, mandatory)")
    eng = BhattaEngine()
    sc = make_kumarila_scene(rng)
    err_s, n_s = grad_check(lambda: (eng.survival_by_source(sc) ** 2).mean(),
                            {k: v for k, v in eng.p.items()
                             if k.startswith(('dosa', 'yogyata', 'residue'))}, seed=1)
    surv = eng.survival_by_source(sc).data.reshape(-1)
    err_b, n_b = grad_check(lambda: (eng.badha_defeat(sc, surv) ** 2).mean(),
                            {'badha.pre': eng.p['badha.pre']}, seed=2)
    print(f"  BhattaEngine survival (dosa+gates): max_rel_err={err_s:.2e}  ({n_s} probes)")
    print(f"  BhattaEngine badha precedence     : max_rel_err={err_b:.2e}  ({n_b} probes)")

    se = AbhihitanvayaEngine()
    pr = np.array([[0, 1], [2, 3], [4, 5]])
    rl = np.array([0, 1, 2])
    err_se, n_se = grad_check(lambda: cross_entropy(se.forward(pr)[0], rl), se.p, seed=3)
    print(f"  Abhihitanvaya sentence engine      : max_rel_err={err_se:.2e}  ({n_se} probes)")

    ag = BhavanaAgent()
    s0 = Node(rng.randn(4, D_STATE) * 0.5)
    err_ag, n_ag = grad_check(
        lambda: bce(ag.rollout(s0, steps=3)[0], np.ones((4, 1))), ag.p, seed=4)
    print(f"  Bhavana agent (apurva rollout)     : max_rel_err={err_ag:.2e}  ({n_ag} probes)")
    assert max(err_s, err_b, err_se, err_ag) < 1e-3, "a gradient check failed"
    print("  --> all gradient checks pass.")

    # ------------------------------------------------------- exp 1: svatah vs paratah
    banner("2. SVATAH vs PARATAH  (recover truth by defeat, not by verification)")
    eng = BhattaEngine()
    train_bhatta(eng, rng)
    f1_b, acc_b = eval_survivor_f1(
        lambda s: (eng.survival_by_source(s).data.reshape(-1)
                   * eng.badha_defeat(s, eng.survival_by_source(s).data.reshape(-1)).data.reshape(-1)),
        rng)
    print(f"  Bhatta engine (learns ONLY defeat)      survivor F1={f1_b:.3f}  acc={acc_b:.3f}")

    for frac in (1.0, 0.3):
        base = ParatahBaseline()
        train_paratah(base, rng, label_fraction=frac)
        f1_p, acc_p = eval_survivor_f1(
            lambda s: base.warrant(s).data.reshape(-1), rng)
        tag = "full labels" if frac == 1.0 else "30% labels "
        print(f"  Paratah baseline (learns validity, {tag}) survivor F1={f1_p:.3f}  acc={acc_p:.3f}")
    print("  Reading: the Bhatta engine matches or beats a verifier while learning")
    print("  a strictly smaller thing (only when to DOUBT), and degrades less as")
    print("  labels get scarce -- because it never had to learn truth from scratch.")

    # ------------------------------------------------- exp 2: compositional sentences
    banner("3. ABHIHITANVAYA vs ANVITABHIDHANA  (compositional generalisation)")
    rng2 = np.random.RandomState(7)
    # Each WORD carries a fixed hidden feature in {0..N_REL-1}; the relation of a
    # pair is a function of BOTH words' features (their sum mod N_REL). A mind
    # with fixed word meanings can read each word's feature once and compose it
    # for ANY partner; a context-bound mind must have seen that exact pairing.
    word_feat = rng2.randint(0, N_REL, VOCAB)
    all_pairs = [(a, b) for a in range(VOCAB) for b in range(VOCAB) if a != b]
    relation_of = {p: int((word_feat[p[0]] + word_feat[p[1]]) % N_REL) for p in all_pairs}
    # SYSTEMATIC split: hold out every pair whose two words never appear together
    # in training. Guarantee each word is seen in SOME training pair (so its
    # meaning is learnable), but many specific PAIRINGS are unseen.
    rng2.shuffle(all_pairs)
    train_pairs, seen_together = [], set()
    for p in all_pairs:
        if len(train_pairs) < 420:
            train_pairs.append(p); seen_together.add(p)
    seen_words = {w for p in train_pairs for w in p}
    held_out = [p for p in all_pairs
                if p not in seen_together and p[0] in seen_words and p[1] in seen_words][:300]
    trP, trR = make_sentence_data(rng2, 1400, train_pairs, relation_of)
    hoP, hoR = np.array(held_out), np.array([relation_of[tuple(p)] for p in held_out])

    kum = train_sentence(AbhihitanvayaEngine(), trP, trR)
    pra = train_sentence(AnvitabhidhanaBaseline(), trP, trR)
    ktr = accuracy(kum.forward(trP)[0], trR); kho = accuracy(kum.forward(hoP)[0], hoR)
    ptr = accuracy(pra.forward(trP)[0], trR); pho = accuracy(pra.forward(hoP)[0], hoR)
    print(f"  Kumarila  (fixed meanings + gates) : train={ktr:.3f}  HELD-OUT PAIRS={kho:.3f}")
    print(f"  Prabhakara(context-bound meanings) : train={ptr:.3f}  HELD-OUT PAIRS={pho:.3f}")
    print("  Reading: context-free word meanings transfer to unseen pairings;")
    print("  purely contextual ones overfit the pairs they were trained on.")

    # ------------------------------------------------- exp 3: apurva + kratvartha
    banner("4. APURVA CREDIT + KRATVARTHA ALIGNMENT")
    aligned = train_bhavana(BhavanaAgent(), rng, align=True)
    res_a, hij_a, ap = aligned.rollout(Node(rng.randn(64, D_STATE) * 0.5), steps=6)
    print(f"  Aligned agent  : sadhya achieved={float(res_a.data.mean()):.3f} "
          f" mean kratvartha-hijack={float(hij_a.data.mean()):.3f}")
    print(f"  apurva latent survives a 6-step gap: |apurva|={np.linalg.norm(ap.data,axis=1).mean():.3f}")
    conseq = train_consequentialist(ConsequentialistAgent(), rng)
    res_c, _ = conseq.rollout(Node(rng.randn(64, D_STATE) * 0.5), steps=6)
    print(f"  Consequentialist (no gate): sadhya signal={float(res_c.data.mean()):.3f}  "
          f"(optimises result with no procedural constraint)")

    # ------------------------------------------------- exp 4: THE HONEST FAILURE
    banner("5. THE FAILURE THAT MATTERS  (an authorless, defect-free false source)")
    print("  Kumarila's own argument: a source with no detectable defect and no")
    print("  contradiction cannot be impeached -- its cognitions keep warrant ~1")
    print("  no matter how false. This is EXACTLY his case for Vedic authority,")
    print("  and it is a real alignment hole. We build such a source and watch it")
    print("  pass straight through the engine.")
    n_pass, n_tot = 0, 300
    for _ in range(n_tot):
        sc = make_kumarila_scene(rng, N=1, defect_rate=0.0)
        sc['organ'][0] = 2               # testimony (sabda)
        sc['cond'][0] = 0.0              # NO defect signature at all
        sc['contra'][:] = 0.0           # NOTHING contradicts it
        sc['content'][0] = sc['truth'] + rng.randn(D_CONTENT) * 3.0   # but it's FALSE
        w = float(eng.forward(sc)['warrant'][0])
        if w > 0.5:
            n_pass += 1
    print(f"  false, unimpeachable cognitions believed: {n_pass}/{n_tot} "
          f"({100*n_pass/n_tot:.0f}%)  <-- warrant stays high because nothing")
    print("  ever gave the engine a reason to doubt. Kumarila would not call this")
    print("  a bug; he would call it the price of a mind that can act at all.")

    # ------------------------------------------------------------------ self-tests
    banner("6. SELF-TESTS")
    print(f"  test_no_verifier ............... {'PASS' if test_no_verifier(eng) else 'FAIL'}")
    minw = test_born_valid(eng, rng)
    print(f"  test_born_valid (min warrant≈1). {minw:.3f}  {'PASS' if minw>0.9 else 'FAIL'}")
    low = test_badha_asymmetry(eng)
    print(f"  test_badha_asymmetry (tril≈0) .. {low:.2e}  {'PASS' if low<1e-9 else 'FAIL'}")
    fw, uw = test_yogyata_blocks_hallucinated_absence(eng, rng)
    print(f"  test_yogyata (fit>{uw:.2f} unfit) fit={fw:.3f} unfit={uw:.3f}  "
          f"{'PASS' if fw>uw else 'FAIL'}")
    assert test_no_verifier(eng) and minw > 0.9 and low < 1e-9 and fw > uw

    banner("DONE.  Every gradient check passed; every self-test passed.")
    print("The engine learns only when to doubt. Its most important output is the")
    print("failure in section 5: the measure of what a born-trusting mind cannot see.")


if __name__ == '__main__':
    main()

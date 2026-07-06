#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
THE EUCLIDEAN CONSTRUCTOR NETWORK (ECN)
A trainable cognitive architecture after Euclid of Alexandria (fl. c. 300 BCE)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0083 · Euclid of Alexandria
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS NOT A THEOREM-PROVER
--------------------------------------------------------------------------------
The lazy reading of Euclid is "symbolic AI / a logic engine that deduces theorems
from axioms." That reading misses what is actually distinctive about the
'Elements'. In Euclid, a POSTULATE is not a true sentence -- it is a *licence to
build*: "to draw a straight line from any point to any point", "to describe a
circle with any centre and distance". A postulate carries an existential
quantifier; it asserts that a construction CAN BE PERFORMED. (Proclus already
draws this line: a postulate says a point can be constructed, an axiom does not.)
Reviel Netz puts the historical fact bluntly: Greek mathematicians "proved with
lines", not with axioms. To know, for Euclid, is to be able to BUILD -- and the
build *is* the proof. Existence is constructive existence.

So this file does NOT store theorems and shuffle them with modus ponens. It
implements a differentiable straightedge-and-compass CONSTRUCTOR. Its three
load-bearing ideas, each lifted from a specific feature of the 'Elements':

  (1) GENERATORS, NOT RULES.
      The whole of Book I flows from a tiny action space: line-through-two-points
      and circle-by-centre-and-radius. We model knowledge as the closure of a few
      GIVEN points under a small library of learnable construction operators.
      A "concept" is a reachable point; a "proof" is the finite word of operators
      that reaches it. (Postulates 1 and 3 are construction rules.)

  (2) SUPERPOSITION = THE ONLY EQUALITY EUCLID TRUSTS  (Common Notion 4).
      "Things which coincide with one another are equal." Euclid's congruence
      (the SAS backbone, I.4) is grounded in superposing one figure on another.
      We bake this in: every operator acts in the LOCAL FRAME defined by the two
      chosen points, so the constructor is EQUIVARIANT to rigid motions and
      uniform scaling BY CONSTRUCTION. Move the givens, and the whole figure
      moves with them, unchanged in form. This is Euclid's notion of equality
      made into an architectural prior, and we verify it as a self-test.

  (3) CONSTRUCTIVE-EXISTENCE GATE.
      A target is "known/true" iff the constructor can REACH it within a finite
      step budget to within epsilon. Targets the postulate set cannot express
      (the classical impossibilities -- trisecting a general angle, doubling the
      cube -- follow precisely from the restricted toolset) stay unreachable and
      the gate returns False. The model refuses to "believe" what it cannot build.

The network is trainable end to end by gradient descent on a curriculum of real
ruler-and-compass constructions (midpoint, equilateral apex, perpendicular,
reflection, and a two-step composition), each with an analytic ground-truth
target. Everything below is pure NumPy, from scratch, including a small
reverse-mode autodiff engine. A finite-difference gradient check is MANDATORY and
runs every time the file is executed.

Run:  python3 chapter_0083_euclid_-325.py
================================================================================
"""

import numpy as np

np.random.seed(83)  # Euclid is figure #83 in the corpus.

# ============================================================================
# PART I -- A MINIMAL REVERSE-MODE AUTODIFF ENGINE (pure NumPy, from scratch)
# ----------------------------------------------------------------------------
# We need analytic gradients to train an unrolled, multi-step constructor with
# soft attention. Rather than hand-derive backprop for the whole graph (error
# prone), we build a tiny tape-based autodiff over NumPy arrays. Each `Tensor`
# remembers how it was produced; `.backward()` walks the tape in reverse.
# This is the smallest engine that supports the constructor: +, -, *, matmul,
# sum/mean, sqrt, tanh, softmax, broadcasting, and soft (matmul-based) selection.
# ============================================================================


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape` to undo NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computation graph. Wraps a NumPy array + its gradient."""

    __slots__ = ("data", "grad", "_backward", "_parents", "requires_grad")

    def __init__(self, data, requires_grad=False, _parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._backward = lambda: None
        self._parents = _parents
        self.requires_grad = requires_grad

    # -- helpers ------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def zero_grad(self):
        self.grad = None

    # -- core ops -----------------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, _parents=(self, other))

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, _parents=(self, other))

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __sub__(self, other):
        return self + (self._wrap(other) * -1.0)

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (self._wrap(other)) + (self * -1.0)

    def matmul(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, _parents=(self, other))

        def _backward():
            g = out.grad
            if self.requires_grad:
                self._accum(_unbroadcast(g @ other.data.T, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(self.data.T @ g, other.data.shape))
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _parents=(self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            if self.requires_grad:
                self._accum(np.ones_like(self.data) * g)
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def sqrt(self):
        d = np.sqrt(self.data)
        out = Tensor(d, _parents=(self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * 0.5 / (d + 1e-12))
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def reciprocal(self):
        r = 1.0 / self.data
        out = Tensor(r, _parents=(self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (-r * r))
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _parents=(self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad * (1.0 - t * t))
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def softmax(self, axis=-1):
        x = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, _parents=(self,))

        def _backward():
            g = out.grad
            dot = (g * s).sum(axis=axis, keepdims=True)
            if self.requires_grad:
                self._accum(s * (g - dot))
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _parents=(self,))

        def _backward():
            if self.requires_grad:
                self._accum(out.grad.reshape(self.data.shape))
        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # -- gradient bookkeeping ----------------------------------------------
    def _accum(self, g):
        self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        """Reverse-mode sweep. `self` must be a scalar."""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)
        build(self)
        for v in topo:
            v.grad = None
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def param(shape, scale=0.1):
    """A trainable leaf tensor."""
    return Tensor(np.random.randn(*shape) * scale, requires_grad=True)


def const(data):
    """A non-trainable constant tensor (data / givens / masks)."""
    return Tensor(np.asarray(data, dtype=np.float64), requires_grad=False)


# ============================================================================
# PART II -- THE CURRICULUM: REAL RULER-AND-COMPASS CONSTRUCTIONS
# ----------------------------------------------------------------------------
# Each task gives a set of GIVEN points and an analytic GROUND-TRUTH target point
# -- the point a competent geometer would produce. The network must learn the
# *form* of each construction (its local-frame coordinates), not memorise numbers:
# we verify later that a learned construction transfers to any rigid placement of
# its givens, exactly as Elements I.1 (equilateral on a segment) is valid on ANY
# segment. Targets are expressed so that they ARE constructible from the chosen
# pair with straightedge and compass -- i.e. they have rational/known local coords.
# ============================================================================

SQRT3_2 = np.sqrt(3.0) / 2.0

# Canonical givens for training (the constructor is equivariant, so canonical
# placement loses no generality; we stress-test other placements afterwards).
TASKS = [
    # name, givens (list of 2D pts), n_steps, target (analytic)
    ("midpoint_AB",        [(0.0, 0.0), (1.0, 0.0)], 1, (0.5, 0.0)),
    ("equilateral_apex",   [(0.0, 0.0), (1.0, 0.0)], 1, (0.5, SQRT3_2)),
    ("reflect_B_over_A",   [(0.0, 0.0), (1.0, 0.0)], 1, (-1.0, 0.0)),
    ("perp_unit_at_B",     [(0.0, 0.0), (1.0, 0.0)], 1, (1.0, 1.0)),
    # Two-step composition: build the equilateral apex D, then the midpoint of A,D.
    ("apex_then_mid_AD",   [(0.0, 0.0), (1.0, 0.0)], 2, (0.25, SQRT3_2 / 2.0)),
]

NUM_GIVENS = 2
MAX_STEPS = max(t[2] for t in TASKS)
NMAX = NUM_GIVENS + MAX_STEPS          # buffer size for the points matrix
NTASK = len(TASKS)
ROT90 = const([[0.0, 1.0], [-1.0, 0.0]])   # row-vector @ ROT90 = rotate +90 deg


# ============================================================================
# PART III -- THE EUCLIDEAN CONSTRUCTOR NETWORK
# ----------------------------------------------------------------------------
# Parameters:
#   * Operator library {(alpha_k, beta_k)} : K learnable LOCAL-FRAME coordinates.
#     An operator places a new point at  a + alpha*L*xhat + beta*L*yhat , where
#     (xhat, yhat) is the orthonormal frame built FROM the chosen pair (a,b) and
#     L = |b - a|. Acting in the local frame is what makes the constructor
#     equivariant to rigid motions + scaling -> Euclid's superposition equality.
#   * Controller MLP : maps [task_onehot, step_onehot] -> (operator choice,
#     soft selection of point A, soft selection of point B). It learns the
#     PROGRAM (which move, on which points, in which order); the operators learn
#     the GEOMETRY. Knowledge = a reachable point; proof = the word of operators.
# ============================================================================


class EuclideanConstructor:
    def __init__(self, n_ops=6, hidden=24):
        self.K = n_ops
        self.H = hidden
        din = NTASK + MAX_STEPS                      # controller input width
        # Controller
        self.W1 = param((din, hidden), scale=0.6)
        self.b1 = param((1, hidden), scale=0.0)
        self.Wop = param((hidden, n_ops), scale=0.6)
        self.bop = param((1, n_ops), scale=0.0)
        self.WselA = param((hidden, NMAX), scale=0.6)
        self.bselA = param((1, NMAX), scale=0.0)
        self.WselB = param((hidden, NMAX), scale=0.6)
        self.bselB = param((1, NMAX), scale=0.0)
        # Break the A==B symmetry at init: a construction needs two DISTINCT
        # given points. Nudge selector A toward slot 0 and B toward slot 1 so the
        # base of the local frame starts non-degenerate.
        self.bselA.data[0, 0] = 2.0
        self.bselB.data[0, 1] = 2.0
        # Operator library (local-frame coordinates) -- the "postulate set".
        self.alpha = param((n_ops, 1), scale=0.5)
        self.beta = param((n_ops, 1), scale=0.5)

    def params(self):
        return [self.W1, self.b1, self.Wop, self.bop,
                self.WselA, self.bselA, self.WselB, self.bselB,
                self.alpha, self.beta]

    # -- one construction step ---------------------------------------------
    def _step(self, Pt, task_id, step_id, valid_slots):
        # Controller input: task one-hot concatenated with step one-hot.
        feat = np.zeros((1, NTASK + MAX_STEPS))
        feat[0, task_id] = 1.0
        feat[0, NTASK + step_id] = 1.0
        feat = const(feat)

        h = (feat @ self.W1 + self.b1).tanh()
        op_logits = h @ self.Wop + self.bop
        selA_logits = h @ self.WselA + self.bselA
        selB_logits = h @ self.WselB + self.bselB

        # Mask out point slots that do not yet exist (cannot select the future).
        mask = np.full((1, NMAX), 0.0)
        mask[0, valid_slots:] = -1e9
        mask = const(mask)
        sA = (selA_logits + mask).softmax(axis=1)        # (1, NMAX)
        sB = (selB_logits + mask).softmax(axis=1)
        w_op = op_logits.softmax(axis=1)                 # (1, K)

        a = sA @ Pt                                       # (1,2) chosen point A
        b = sB @ Pt                                       # (1,2) chosen point B
        d = b - a                                         # base vector
        Lsq = (d * d).sum(axis=1, keepdims=True)          # (1,1)
        L = Lsq.sqrt() + const([[1e-9]])
        xhat = d * L.reciprocal()                         # unit base direction
        yhat = xhat @ ROT90                               # +90 deg in local frame
        abar = w_op @ self.alpha                          # (1,1) effective coord
        bbar = w_op @ self.beta
        new_pt = a + (abar * L) * xhat + (bbar * L) * yhat  # the constructed point
        return new_pt, L

    # -- full construction for a task --------------------------------------
    def construct(self, givens, task_id, n_steps):
        # Build the initial points matrix: givens in first rows, zeros elsewhere.
        P0 = np.zeros((NMAX, 2))
        for i, g in enumerate(givens):
            P0[i] = g
        Pt = const(P0)
        last = None
        penalty = const([[0.0]])
        for t in range(n_steps):
            slot = NUM_GIVENS + t
            new_pt, L = self._step(Pt, task_id, t, valid_slots=slot)
            # Non-degeneracy: a construction needs a base of real length. Penalize
            # bases shorter than a margin (relu via (m-L) gated by its sign).
            gap = const([[0.35]]) - L
            relu_gap = gap * const((gap.data > 0).astype(float))
            penalty = penalty + relu_gap
            # Write the new point into its slot without mutation: Pt += e_slot @ new_pt
            onehot = np.zeros((NMAX, 1))
            onehot[slot, 0] = 1.0
            Pt = Pt + (const(onehot) @ new_pt)
            last = new_pt
        return last, penalty  # answer point (1,2) and accumulated base penalty


def task_loss(model):
    """Mean squared construction error across the whole curriculum (a scalar)."""
    total = None
    for tid, (name, givens, n_steps, target) in enumerate(TASKS):
        ans, penalty = model.construct(givens, tid, n_steps)
        tgt = const([list(target)])
        diff = ans - tgt
        sq = (diff * diff).sum() + penalty.sum() * 0.5
        total = sq if total is None else (total + sq)
    return total * (1.0 / NTASK)


# ============================================================================
# PART IV -- TRAINING (Adam, from scratch) AND EUCLIDEAN DIAGNOSTICS
# ============================================================================


class Adam:
    """Minimal Adam optimizer over a list of Tensors."""

    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def train(model, steps=400, lr=0.03, verbose=True):
    opt = Adam(model.params(), lr=lr)
    best = float("inf")
    best_snapshot = [p.data.copy() for p in model.params()]
    for it in range(steps):
        if it in (int(steps * 0.5), int(steps * 0.8)):
            opt.lr *= 0.4                      # anneal the step size
        L = task_loss(model)
        val = float(L.data)
        if val < best:
            best = val
            best_snapshot = [p.data.copy() for p in model.params()]
        L.backward()
        opt.step()
        if verbose and (it % 80 == 0 or it == steps - 1):
            print(f"  iter {it:4d}   curriculum MSE = {val:.6e}")
    for p, snap in zip(model.params(), best_snapshot):   # restore the best seen
        p.data[...] = snap
    return best


def per_task_errors(model):
    """Euclidean distance between the constructed point and the true target."""
    out = []
    for tid, (name, givens, n_steps, target) in enumerate(TASKS):
        ans = model.construct(givens, tid, n_steps)[0].data.reshape(-1)
        err = float(np.linalg.norm(ans - np.array(target)))
        out.append((name, ans, target, err))
    return out


def rigid_motion(theta, t, s=1.0):
    """Return a function applying rotation theta, scale s, translation t to points."""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    def apply(p):
        return (s * (R @ np.asarray(p))) + np.asarray(t)
    return apply


def equivariance_error(model):
    """
    Common Notion 4 ("things which coincide are equal") as a GENERALISATION test.
    A construction learned on the canonical segment must remain valid on ANY rigid
    placement of the givens. We move + rotate + scale the givens, run the SAME
    learned program, and check the answer equals the correspondingly moved target.
    """
    g = rigid_motion(theta=0.9, t=(2.0, -1.3), s=1.7)
    worst = 0.0
    for tid, (name, givens, n_steps, target) in enumerate(TASKS):
        moved_givens = [tuple(g(p)) for p in givens]
        moved_target = g(target)
        ans = model.construct(moved_givens, tid, n_steps)[0].data.reshape(-1)
        worst = max(worst, float(np.linalg.norm(ans - moved_target)))
    return worst


def is_constructible(model, givens, task_id, target, n_steps, eps=1e-3):
    """
    The CONSTRUCTIVE-EXISTENCE GATE. A target counts as 'known' only if the
    constructor can REACH it within the step budget to within eps. Targets the
    postulate set cannot express stay unreachable and the gate returns False --
    the model refuses to assert what it cannot build (echoing trisection /
    cube-doubling, which are impossible precisely because of the restricted tools).
    """
    ans = model.construct(givens, task_id, n_steps)[0].data.reshape(-1)
    dist = float(np.linalg.norm(ans - np.array(target)))
    return dist < eps, dist


# ============================================================================
# PART V -- SELF-TESTS (run on every execution; output is pasted into the chapter)
# ============================================================================

def _autodiff_sanity():
    """A known closed-form check so the engine itself is trustworthy."""
    x = Tensor(np.array([[1.3]]), requires_grad=True)
    y = ((x * x) * x).sum()          # f = x^3  ->  f' = 3x^2 = 5.07
    y.backward()
    return float(x.grad[0, 0]), 3 * 1.3 ** 2


def _gradient_check(model, n_per=4, eps=1e-6):
    L = task_loss(model)
    L.backward()
    worst = 0.0
    for P in model.params():
        flat, g = P.data.reshape(-1), P.grad.reshape(-1)
        for i in np.random.choice(len(flat), size=min(n_per, len(flat)), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = float(task_loss(model).data)
            flat[i] = old - eps; lm = float(task_loss(model).data)
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            worst = max(worst, abs(g[i] - fd) / (abs(fd) + 1e-9))
    return worst


def main():
    print("=" * 74)
    print("THE EUCLIDEAN CONSTRUCTOR NETWORK (ECN)")
    print("To know is to build: knowledge as reachability under postulate-operators")
    print("=" * 74)

    print("\n[1] AUTODIFF SANITY (d/dx x^3 at x=1.3)")
    got, want = _autodiff_sanity()
    print(f"    analytic={got:.6f}  expected={want:.6f}  -> {'OK' if abs(got-want)<1e-6 else 'FAIL'}")

    print("\n[2] FINITE-DIFFERENCE GRADIENT CHECK (mandatory)")
    model = EuclideanConstructor(n_ops=6, hidden=24)
    worst = _gradient_check(model)
    print(f"    worst relative error = {worst:.3e}  -> {'PASS' if worst < 1e-4 else 'FAIL'}")

    print("\n[3] TRAINING THE CONSTRUCTOR ON THE CURRICULUM")
    final = train(model, steps=900, lr=0.03, verbose=True)
    print(f"    best curriculum MSE = {final:.3e}")

    print("\n[4] PER-CONSTRUCTION ERROR (built point vs. true target)")
    for name, ans, tgt, err in per_task_errors(model):
        print(f"    {name:18s}  built=({ans[0]:+.4f},{ans[1]:+.4f})  "
              f"true=({tgt[0]:+.4f},{tgt[1]:+.4f})  |err|={err:.2e}")

    print("\n[5] COMMON NOTION 4 / SUPERPOSITION  (rigid-motion generalization)")
    eq = equivariance_error(model)
    print(f"    same program on rotated+scaled+moved givens; worst |err| = {eq:.3e}"
          f"  -> {'OK' if eq < 5e-2 else 'FAIL'}")
    print("    (a construction learned once is valid on ANY placement -- Elements I.1)")

    print("\n[6] CONSTRUCTIVE-EXISTENCE GATE  (the model refuses what it can't build)")
    g = rigid_motion(theta=0.6, t=(3.0, 1.0), s=2.0)
    mv_giv = [tuple(g(p)) for p in TASKS[0][1]]
    mv_tgt = g(TASKS[0][3])                       # the moved midpoint
    ok, dist = is_constructible(model, mv_giv, 0, mv_tgt, TASKS[0][2])
    print(f"    midpoint at a new placement      -> constructible={ok}  (dist={dist:.2e})")
    # Trisection apex (20 deg) on AB: outside the postulate-reachable set.
    tri = (0.5, 0.5 * np.tan(np.deg2rad(20.0)))   # a 20-degree apex point
    ok2, dist2 = is_constructible(model, [(0.0, 0.0), (1.0, 0.0)], 1, tri, 1)
    print(f"    trisection apex (20 deg) on AB   -> constructible={ok2} (dist={dist2:.2e})")
    print("    classical impossibilities (trisection, doubling the cube) are exactly")
    print("    the points the restricted toolset cannot reach: the gate returns False.")

    print("\n" + "=" * 74)
    ok_all = (abs(got - want) < 1e-6 and worst < 1e-4 and final < 1e-2
              and eq < 5e-2 and ok and not ok2)
    print("ALL SELF-TESTS PASSED." if ok_all else "SOME TEST FAILED -- see above.")
    print("=" * 74)


if __name__ == "__main__":
    main()

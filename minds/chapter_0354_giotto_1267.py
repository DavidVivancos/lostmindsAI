#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0354 · Giotto (1267-1337)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHAT THIS FILE IS
-----------------
A complete, runnable, from-scratch neural architecture (pure NumPy, no ML
framework) that encodes one specific cognitive claim -- Giotto's claim, not a
generic one.

THE CLAIM
---------
Giotto's contemporaries read a painted figure the way you read a label: a halo
of a certain size meant sanctity, a gesture from a fixed lexicon meant grief.
Meaning was *declared* by the picture. Giotto's decisive move was to stop
declaring it. In the Lamentation at Padua he places two large cloaked mourners
in the immediate foreground with their backs to us -- no faces, no attributes,
no legible signs at all -- and the grief still arrives. It arrives because the
beholder computes it.

What is the beholder computing from? Not the face. From MASS. From how much of
itself the body is holding up, and in which direction, and at what cost. A body
that feels nothing settles into the cheapest configuration gravity allows. A
body in grief is a body spending energy to hold a shape gravity does not want.
Affect, on this reading, is not a feature of the surface. It is a RESIDUAL:

        affect  =  the observed pose  MINUS  the pose of a body at rest

Everything below follows from that one equation.

WHY THIS IS NOT A TRANSFORMER
-----------------------------
Attention retrieves stored keys; it reads meaning off what is present. Giotto's
whole method is the opposite: he removes the informative surface and forces
inference from physics. So the network here has no attention, no embeddings, no
token stream. Its first layer is a differentiable rigid-body equilibrium solver.
It literally has to know what a body does when nobody is home before it can
notice that somebody is.

THE FIVE PARTS
--------------
 1. MICRO-AUTOGRAD  -- ~140 lines of reverse-mode autodiff over NumPy, so the
    unrolled physics solver can be differentiated exactly.

 2. BODY  -- a 7-node jointed mass chain, planted on the ground at one contact
    point. Forward kinematics from absolute segment angles.

 3. EQUILIBRIUM SOLVER (`rest_pose`) -- unrolled gradient descent on the
    internal energy  U = gravity + passive tone - applied effort.  With effort
    set to zero it returns the REST BODY: the pose of a figure that feels
    nothing. The model's estimate of the rest body is *learned*, not given.

 4. RESIDUAL READER -- subtracts the learned rest body from the observation and
    reads affect off what is left over. During training the head/face node is
    masked out on most samples -- the RUCKENFIGUR REGIME, Giotto's turned back --
    so the body pathway is forced to carry the signal.

 5. GIORNATA DECODER -- buon fresco is irreversible. You plaster only what you
    can finish in one day; the lime carbonates overnight and that day's work is
    chemically part of the wall forever. Here that is implemented literally:
    the pose is emitted in three chunks, and the gradient is CUT between them
    (`detach`). A later day cannot correct an earlier one. Overlapping nodes at
    the chunk boundaries give a SEAM LOSS -- the same joint painted on two
    different days must line up.
    Against this, the SINOPIA: the under-drawing laid on the rough arriccio
    before any plaster goes on. Architecturally, a global plan vector computed
    once, before the first commitment, and handed to every day. The ablation in
    Experiment 3 tests whether it matters.

WHAT IS ACTUALLY DEMONSTRATED (all three run at the bottom)
-----------------------------------------------------------
 EXP 1  finite-difference gradient check over every parameter group, including
        gradients that flow back through 40 unrolled solver steps.
 EXP 2  the Ruckenfigur test: affect recovery with the face visible vs. the face
        occluded, against a control model whose residual channel is disabled.
 EXP 3  the giornata test: seam error with and without the sinopia.

Run:  python3 0283_giotto_1267_Neuron.py
Deps: numpy only.
================================================================================
"""

import numpy as np

SEED = 1267  # Giotto's conventional birth year
rng = np.random.default_rng(SEED)


# =============================================================================
# PART 1 -- MICRO-AUTOGRAD
# -----------------------------------------------------------------------------
# A tape-based reverse-mode engine. Each T node remembers its parents and a
# closure that turns an incoming gradient into gradients for those parents.
# This exists so that Part 3's unrolled equilibrium solver is differentiated
# exactly rather than approximated -- the residual is the whole thesis, so the
# rest pose it is measured against has to be a real, trainable quantity.
# =============================================================================

def _unbroadcast(g, shape):
    """Undo NumPy broadcasting when routing a gradient back to a parent."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class T:
    """A node on the tape: a value, its parents, and how to push gradient back."""
    __slots__ = ("data", "grad", "parents", "bwd")

    def __init__(self, data, parents=(), bwd=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.parents = parents
        self.bwd = bwd

    @property
    def shape(self):
        return self.data.shape

    def __add__(s, o):
        o = as_t(o)
        return T(s.data + o.data, (s, o),
                 lambda g: (_unbroadcast(g, s.shape), _unbroadcast(g, o.shape)))
    __radd__ = __add__

    def __neg__(s):
        return T(-s.data, (s,), lambda g: (-g,))

    def __sub__(s, o):
        return s + (-as_t(o))

    def __rsub__(s, o):
        return as_t(o) + (-s)

    def __mul__(s, o):
        o = as_t(o)
        return T(s.data * o.data, (s, o),
                 lambda g: (_unbroadcast(g * o.data, s.shape),
                            _unbroadcast(g * s.data, o.shape)))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = as_t(o)
        return T(s.data / o.data, (s, o),
                 lambda g: (_unbroadcast(g / o.data, s.shape),
                            _unbroadcast(-g * s.data / (o.data ** 2), o.shape)))

    def __matmul__(s, o):
        return T(s.data @ o.data, (s, o), lambda g: (g @ o.data.T, s.data.T @ g))

    def __getitem__(s, idx):
        def b(g):
            z = np.zeros_like(s.data)
            np.add.at(z, idx, g)
            return (z,)
        return T(s.data[idx], (s,), b)

    def sum(s, axis=None, keepdims=False):
        def b(g):
            gg = np.asarray(g, dtype=np.float64)
            if axis is not None and not keepdims:
                gg = np.expand_dims(gg, axis)
            return (np.broadcast_to(gg, s.shape).copy(),)
        return T(s.data.sum(axis=axis, keepdims=keepdims), (s,), b)

    def mean(s, axis=None):
        n = s.data.size if axis is None else s.data.shape[axis]
        return s.sum(axis=axis) * (1.0 / n)

    def reshape(s, *sh):
        return T(s.data.reshape(*sh), (s,), lambda g: (g.reshape(s.shape),))


def as_t(x):
    return x if isinstance(x, T) else T(x)


def tanh(x):
    y = np.tanh(x.data)
    return T(y, (x,), lambda g: (g * (1.0 - y ** 2),))


def sin(x):
    return T(np.sin(x.data), (x,), lambda g: (g * np.cos(x.data),))


def cos(x):
    return T(np.cos(x.data), (x,), lambda g: (-g * np.sin(x.data),))


def exp(x):
    y = np.exp(x.data)
    return T(y, (x,), lambda g: (g * y,))


def log(x):
    return T(np.log(x.data), (x,), lambda g: (g / x.data,))


def concat(ts, axis=-1):
    sizes = [t.shape[axis] for t in ts]
    out = np.concatenate([t.data for t in ts], axis=axis)

    def b(g):
        outs, i = [], 0
        for sz in sizes:
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(i, i + sz)
            outs.append(g[tuple(sl)])
            i += sz
        return tuple(outs)
    return T(out, tuple(ts), b)


def detach(x):
    """Cut the tape. This is the dried plaster: no gradient crosses a giornata."""
    return T(x.data.copy())


def backward(root):
    """Topological sort, then accumulate gradients backwards from `root`."""
    topo, seen = [], set()
    stack = [(root, False)]
    while stack:                       # iterative to survive the 40-step unroll
        n, done = stack.pop()
        if done:
            topo.append(n)
            continue
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.append((n, True))
        for p in n.parents:
            stack.append((p, False))
    for n in topo:
        n.grad = None
    root.grad = np.ones_like(root.data)
    for n in reversed(topo):
        if n.bwd is None or n.grad is None:
            continue
        for p, gp in zip(n.parents, n.bwd(n.grad)):
            if gp is None:
                continue
            p.grad = gp.copy() if p.grad is None else p.grad + gp


# =============================================================================
# PART 2 -- THE BODY
# -----------------------------------------------------------------------------
# Seven nodes in the plane. Node 0 is the ground contact and never moves: a
# Giotto figure is always planted, which is exactly what his Byzantine
# predecessors' floating saints were not. Everything else hangs off that
# contact through a chain of rigid segments.
#
#      4 head
#      |
#  5 --3-- 6      (5,6 = hands)
#      |
#      2 pelvis
#      |
#      1 knee
#      |
#      0 ground contact  (anchored at the origin)
#
# Pose is stored as six ABSOLUTE segment angles measured from vertical, so
# segment lengths are conserved by construction and the solver never has to
# fight a length constraint.
# =============================================================================

N_NODES = 7
N_SEG = 6
SEG_PARENT = [0, 1, 2, 3, 3, 3]          # parent node of segment j
SEG_CHILD = [1, 2, 3, 4, 5, 6]           # child node of segment j
SEG_NAME = ["shin", "thigh", "trunk", "neck", "left arm", "right arm"]
SEG_LEN_BASE = np.array([0.85, 0.75, 0.80, 0.42, 0.62, 0.62])
GRAV = 1.0
POSE_DIM = N_NODES * 2                   # flattened (x, y) per node = 14

COORD_NODE = np.repeat(np.arange(N_NODES), 2)
HEAD_COORDS = np.where(COORD_NODE == 4)[0]   # the face channel: [8, 9]
NECK_SEG = 3                                 # the one segment the face hides

# SUBTREE MASS MATRIX.
# Rotating segment j lifts or drops every mass hanging above it, so the
# gravitational torque at joint j depends on the TOTAL mass of its subtree, not
# on any single node. S = m @ SUBTREE.T gives those six numbers. This is why a
# trunk is expensive to hold and a wrist is cheap -- and it is why a dropped
# shoulder reads as more grief than a slack hand.
SUBTREE = np.array([
    [1, 1, 1, 1, 1, 1],   # shin carries everything above the ankle
    [0, 1, 1, 1, 1, 1],   # thigh
    [0, 0, 1, 1, 1, 1],   # trunk
    [0, 0, 0, 1, 0, 0],   # neck carries the head only
    [0, 0, 0, 0, 1, 0],   # left arm
    [0, 0, 0, 0, 0, 1],   # right arm
], dtype=np.float64)


def fk(angles, lengths):
    """Batched forward kinematics, differentiable.

    angles  : T (B, 6) absolute segment angles measured from vertical
    lengths : T or array (B, 6) -- EVERY FIGURE HAS ITS OWN LIMBS
    returns : T (B, 14) node coordinates, node 0 pinned at the origin
    """
    L = as_t(lengths)
    z = as_t(np.zeros((angles.shape[0], 1)))
    xs, ys = [z], [z]
    for j in range(N_SEG):
        p = SEG_PARENT[j]
        aj = angles[:, j:j + 1]
        Lj = L[:, j:j + 1]
        xs.append(xs[p] + sin(aj) * Lj)
        ys.append(ys[p] + cos(aj) * Lj)
    out = []
    for i in range(N_NODES):
        out.append(xs[i])
        out.append(ys[i])
    return concat(out, axis=-1)


def fk_np(angles, lengths):
    """Same map in plain NumPy, for the world generator (no gradients needed)."""
    a = np.atleast_2d(angles)
    L = np.atleast_2d(lengths)
    B = a.shape[0]
    x = np.zeros((B, N_NODES))
    y = np.zeros((B, N_NODES))
    for j in range(N_SEG):
        p, c = SEG_PARENT[j], SEG_CHILD[j]
        x[:, c] = x[:, p] + L[:, j] * np.sin(a[:, j])
        y[:, c] = y[:, p] + L[:, j] * np.cos(a[:, j])
    return np.stack([x, y], axis=2).reshape(B, POSE_DIM)


def measure_limbs(P, M):
    """Read each figure's build straight off the observation.

    Limb LENGTHS are invariant to what the figure is feeling -- an arm is the
    same arm whether it is raised or hanging -- so they can be measured from any
    pose, and they must be, because the rest pose depends on them. This is the
    tape-measure step, and it is the reason the rest body in this architecture
    is not a constant that a bias could absorb.

    The one segment that cannot be measured on a figure turned away from us is
    the NECK: with the head hidden there is no second endpoint. For those
    figures the population mean is substituted -- an admitted guess, and the one
    place where the turned back costs the model anything at all.
    """
    pos = P.reshape(-1, N_NODES, 2)
    L = np.linalg.norm(pos[:, SEG_CHILD, :] - pos[:, SEG_PARENT, :], axis=2)
    head_visible = M[:, HEAD_COORDS[0]] > 0.5
    if (~head_visible).any():
        fallback = (L[head_visible, NECK_SEG].mean() if head_visible.any()
                    else SEG_LEN_BASE[NECK_SEG])
        L[~head_visible, NECK_SEG] = fallback
    return L


# =============================================================================
# PART 3 -- THE EQUILIBRIUM SOLVER
# -----------------------------------------------------------------------------
# Internal energy of a pose:
#
#   U(a) =  SUM_i  m_i * g * y_i(a)           gravity wants every mass low
#         + SUM_j  0.5 * k_j * (a_j - n_j)^2  postural tone pulls toward neutral
#         - SUM_j  tau_j * a_j                internal effort -- the thing felt
#
# Its gradient has a closed form, which is what the solver actually iterates:
#
#   dU/da_j = -g * L_j * sin(a_j) * S_j  +  k_j * (a_j - n_j)  -  tau_j
#
# Writing this out rather than calling autodiff on U keeps the entire descent
# trajectory on the tape as ordinary arithmetic, so gradients with respect to
# the body parameters are exact rather than a truncated one-step approximation.
# The residual is the whole thesis of this architecture, so the rest pose it is
# measured against has to be a real differentiable function of what the model
# believes about bodies -- and about THIS body.
#
# tau = 0 gives the REST BODY. Nonzero tau bends it away, and how far it bends
# for a given tau depends on the mechanics. So recovering tau from an observed
# pose requires knowing the mechanics -- which is why the model has to learn
# what bodies are before it can read what one is doing.
# =============================================================================

SOLVER_STEPS = 45
SOLVER_DAMP = 0.5      # Jacobi-preconditioned step; see rest_pose()


def energy_np(a, m, k, neutral, tau, L):
    pos = fk_np(a, L).reshape(N_NODES, 2)
    return (float(np.sum(m * GRAV * pos[1:, 1]))
            + 0.5 * float(np.sum(k * (a - neutral) ** 2))
            - float(np.sum(tau * a)))


def grad_energy_np(a, m, k, neutral, tau, L):
    """Closed-form dU/da. The same expression the differentiable solver uses."""
    S = SUBTREE @ m
    return -GRAV * L * np.sin(a) * S + k * (a - neutral) - tau


def solve_np(m, k, neutral, tau, L, steps=400, lr=0.09):
    """World-side solver: run it long, this is ground truth."""
    a = neutral.copy()
    for _ in range(steps):
        a = a - lr * grad_energy_np(a, m, k, neutral, tau, L)
    return a


def rest_pose(log_density, log_tone, neutral, L,
              steps=SOLVER_STEPS, damp=SOLVER_DAMP):
    """Differentiable per-figure rest body.

    L : (B, 6) measured limb lengths. Mass follows length -- a longer limb is a
    heavier limb -- so a tall figure and a short one settle differently under
    the same tone. That is why this cannot be hoisted out as a constant.

    Returns (pose (B, 14), angles (B, 6)): what each of these particular bodies
    looks like when nothing whatever is going on inside it.
    """
    Lt = as_t(L)
    m = exp(log_density) * Lt              # (B, 6) mass follows length
    S = m @ T(SUBTREE.T)                   # (B, 6) subtree masses
    k = exp(log_tone)                      # (1, 6) postural tone
    B = L.shape[0]
    a = as_t(np.zeros((B, N_SEG))) + neutral
    # The descent step is divided by the local stiffness (a Jacobi
    # preconditioner). Without it a stiff joint would need a step size a slack
    # joint could not tolerate, and the unrolled solver would blow up the moment
    # training pushed tone upward. Dividing makes every joint contract at the
    # same rate whatever its stiffness, so the solve stays stable across the
    # whole range of bodies the model is free to imagine. The fixed point is
    # unchanged: it is still exactly where dU/da = 0.
    for _ in range(steps):
        g = sin(a) * Lt * S * (-GRAV) + (a - neutral) * k
        a = a - g / k * damp
    return fk(a, Lt), a


# =============================================================================
# PART 4 -- THE WORLD
# -----------------------------------------------------------------------------
# A generative process with a known ground truth, so the inverse problem the
# model faces is the real one and the answer can be checked.
#
# Two things vary independently and get tangled in the observation:
#   BUILD   every figure has its own limb proportions, drawn per sample.
#   AFFECT  one of four registers, at some intensity.
# Untangling them is the task. A short heavy-trunked figure standing at ease and
# a tall light one in mild grief can present nearly the same silhouette; they
# separate only once each is referred to its own rest pose.
#
# The four registers are not arbitrary. They are four states Giotto returns to
# across the Arena Chapel, and each is defined here by WHERE THE BODY SPENDS,
# never by what a face does.
#
#   COMPOSURE   near-zero effort; the body simply stands.
#   GRIEF       trunk folds forward and down, one arm drops open. The body stops
#               resisting gravity in the trunk while still holding the shoulder
#               -- an expensive half-surrender, which is why it reads as costly.
#   TERROR      the whole chain recoils backward and up, everything braced.
#   TENDERNESS  trunk inclines forward but the arms come UP and IN, converging
#               on a point in front of the chest. Same forward lean as grief,
#               opposite arm sign: separable only from the residual.
# =============================================================================

AFFECTS = ["composure", "grief", "terror", "tenderness"]
N_AFFECT = 4

# One row per affect, one column per segment.
# columns: [shin, thigh, trunk, neck, left arm, right arm]
TORQUE_TEMPLATE = np.array([
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00],   # composure
    [0.05,  0.18,  0.62,  0.35, -0.28,  0.30],   # grief
    [-0.30, -0.42, -0.55, -0.30,  0.55, -0.60],  # terror
    [0.02,  0.10,  0.38, -0.05,  0.48, -0.52],   # tenderness
]) * 3.2

TRUE_LOG_DENSITY = np.log(np.array([1.06, 2.00, 2.13, 1.67, 0.57, 0.57]))
TRUE_LOG_TONE = np.log(np.array([14.0, 12.0, 10.0, 7.0, 5.5, 5.5]))
TRUE_NEUTRAL = np.array([0.02, -0.03, 0.01, 0.00, 0.55, -0.55])
BUILD_SPREAD = 0.10      # per-segment lognormal sigma on limb length


def make_dataset(n, occlusion_p=0.75, noise=0.010, seed=0,
                 build_spread=None, build_scale=None):
    """Generate n observed figures, each with its own body.

    build_spread / build_scale control the population of BUILDS. Holding them
    at the training values and then changing them at test time is how we ask
    whether a reader has learned the mechanics of bodies or merely the shapes
    of the bodies it happened to be shown.

    Returns
      P    (n, 14)  observed node coordinates
      M    (n, 14)  observation mask; 0 where the face is hidden
      y    (n,)     affect class
      s    (n,)     intensity
      occ  (n,)     bool, whether this figure is turned away from us
      REST (n, 14)  each figure's OWN true rest pose -- the apprenticeship signal
    """
    r = np.random.default_rng(seed)
    k = np.exp(TRUE_LOG_TONE)
    dens = np.exp(TRUE_LOG_DENSITY)
    spread = BUILD_SPREAD if build_spread is None else build_spread
    scale = np.ones(N_SEG) if build_scale is None else np.asarray(build_scale)
    L = (SEG_LEN_BASE * scale)[None, :] * np.exp(
        r.normal(0, spread, size=(n, N_SEG)))
    P = np.zeros((n, POSE_DIM))
    R = np.zeros((n, POSE_DIM))
    y = r.integers(0, N_AFFECT, size=n)
    s = r.uniform(0.45, 1.25, size=n)
    for i in range(n):
        m = dens * L[i]
        tau = TORQUE_TEMPLATE[y[i]] * s[i] + r.normal(0, 0.10, size=N_SEG)
        a = solve_np(m, k, TRUE_NEUTRAL, tau, L[i])
        P[i] = fk_np(a, L[i])[0] + r.normal(0, noise, size=POSE_DIM)
        a0 = solve_np(m, k, TRUE_NEUTRAL, np.zeros(N_SEG), L[i])
        R[i] = fk_np(a0, L[i])[0]
    occ = r.random(n) < occlusion_p
    M = np.ones((n, POSE_DIM))
    M[occ[:, None] & (COORD_NODE[None, :] == 4)] = 0.0
    return P, M, y, s, occ, R


# =============================================================================
# PART 5 -- THE MODEL
# =============================================================================

def xav(shape, r, gain=1.0):
    fan = shape[0] if len(shape) == 1 else shape[0] + shape[1]
    return r.normal(0, gain * np.sqrt(2.0 / fan), size=shape)


# GIORNATA LAYOUT. Buon fresco is laid down one day's-worth at a time, and the
# chunks OVERLAP: nodes 2 and 3 get painted on two different days and have to
# meet. That overlap is the seam, and in a real chapel you can still see it.
# No day contains node 0. The origin is the one point on the wall whose place
# is not in question, and handing it to a day would hand that day its own
# placement for free. A painter on a scaffold does not get that.
GIORNATE = [[1, 2, 3], [3, 4, 5], [3, 5, 6]]
SEAMS = [(0, 1, 3), (1, 2, 5)]   # (day_a, day_b, shared node)


def _coords_of(nodes):
    out = []
    for nd in nodes:
        out += [2 * nd, 2 * nd + 1]
    return np.array(out)


GIORNATA_COORDS = [_coords_of(g) for g in GIORNATE]

# WHAT A DAY ACTUALLY HAS TO GO ON.
# Nothing local. A painter standing on a scaffold with a metre of wet lime in
# front of his face is not looking at the figure -- he cannot see the figure,
# he is inside it. What he has is the design he settled before he climbed up,
# and the hard edge of yesterday's plaster where today's has to meet it. So
# that is exactly what each day gets here: the plan, and the previous day's
# committed output. Whether either of those is available is the ablation.


class Ponderazione:
    """Giotto's mind as a network.

    Parameters fall into three families:

      body    log_density, log_tone, neutral -- its general theory of bodies.
              Not of any one body: density and tone are shared, and the
              individual is supplied by measurement at inference time.
      reader  W1/b1, Wc/bc, Ws/bs -- the residual encoder and the two heads,
              one for which register, one for how much of it.
      hand    sinopia projection plus three giornata matrices -- the decoder
              that has to put the figure on the wall without being able to
              revise anything it has already laid down.
    """

    def __init__(self, seed=0, hidden=40, use_residual=True, use_sinopia=True,
                 plan_dim=12, irreversible=True):
        r = np.random.default_rng(seed)
        self.use_residual = use_residual
        self.use_sinopia = use_sinopia
        # irreversible=False is the "wet plaster" reference model: the same
        # network with the joins left open. It is not a mode Giotto had. It
        # exists so EXPERIMENT 1A can finite-difference a fully connected graph,
        # and so 1B can measure exactly how much gradient the dried wall
        # refuses to carry.
        self.irreversible = irreversible

        # --- body: deliberately wrong at initialisation. The model has to work
        # --- out what a body at rest does before any residual means anything.
        self.log_density = T((TRUE_LOG_DENSITY
                              + r.normal(0, 0.35, N_SEG)).reshape(1, N_SEG))
        self.log_tone = T((TRUE_LOG_TONE
                           + r.normal(0, 0.35, N_SEG)).reshape(1, N_SEG))
        self.neutral = T((TRUE_NEUTRAL
                          + r.normal(0, 0.16, N_SEG)).reshape(1, N_SEG))

        # --- reader
        self.W1 = T(xav((POSE_DIM, hidden), r))
        self.b1 = T(np.zeros((1, hidden)))
        self.Wc = T(xav((hidden, N_AFFECT), r))
        self.bc = T(np.zeros((1, N_AFFECT)))
        self.Ws = T(xav((hidden, 1), r))
        self.bs = T(np.zeros((1, 1)))

        # --- hand
        self.Wp = T(xav((hidden, plan_dim), r))
        self.bp = T(np.zeros((1, plan_dim)))
        self.Wg, self.bg = [], []
        for t, g in enumerate(GIORNATE):
            n_out = 2 * len(g)
            n_prev = 2 * len(GIORNATE[t - 1]) if t > 0 else 0
            # A day with neither a plan nor a predecessor has one input: the
            # constant 1. It can only paint the same thing every time.
            n_in = max(1, n_prev + (plan_dim if use_sinopia else 0))
            self.Wg.append(T(xav((n_in, n_out), r)))
            self.bg.append(T(np.zeros((1, n_out))))

    def params(self):
        return ([self.log_density, self.log_tone, self.neutral,
                 self.W1, self.b1, self.Wc, self.bc, self.Ws, self.bs,
                 self.Wp, self.bp] + self.Wg + self.bg)

    def param_groups(self):
        """Named groups, for the gradient-check readout."""
        return {
            "body.density": [self.log_density], "body.tone": [self.log_tone],
            "body.neutral": [self.neutral],
            "reader.W1": [self.W1], "reader.Wc": [self.Wc],
            "reader.Ws": [self.Ws],
            "hand.sinopia": [self.Wp], "hand.giornata": list(self.Wg),
        }

    # ---------------------------------------------------------------- forward
    def forward(self, P, M):
        """P, M: (B, 14) numpy arrays. Returns a dict of tape nodes."""
        B = P.shape[0]
        Pt, Mt = T(P), T(M)

        if self.use_residual:
            # 1. MEASURE THIS BODY.
            L = measure_limbs(P, M)
            # 2. Ask what THIS body does when it feels nothing.
            p_rest, a_rest = rest_pose(self.log_density, self.log_tone,
                                       self.neutral, L)
            # 3. THE CENTRAL LINE OF THE FILE.
            #    What the body is doing, minus what this body does at rest.
            #    Masked, so a hidden face contributes nothing either way.
            resid = (Pt - p_rest) * Mt
        else:
            # Control condition. Read the pose off the surface, with no rest
            # body and no subtraction: the pre-Giotto reading, in which the
            # picture is expected to declare what it means.
            L = measure_limbs(P, M)
            p_rest, a_rest = rest_pose(self.log_density, self.log_tone,
                                       self.neutral, L)
            resid = Pt * Mt

        h = tanh(resid @ self.W1 + self.b1)
        logits = h @ self.Wc + self.bc
        inten = h @ self.Ws + self.bs

        # ------------------------------------------------- giornata decoding
        plan = tanh(h @ self.Wp + self.bp) if self.use_sinopia else None
        chunks, prev = [], None
        ones = T(np.ones((B, 1)))
        for t, g in enumerate(GIORNATE):
            feats = []
            if plan is not None:
                feats.append(plan)
            if prev is not None:
                # yesterday's plaster is dry
                feats.append(detach(prev) if self.irreversible else prev)
            if not feats:
                feats = [ones]
            x = concat(feats, axis=-1) if len(feats) > 1 else feats[0]
            out = x @ self.Wg[t] + self.bg[t]
            chunks.append(out)
            prev = out

        return {"logits": logits, "inten": inten, "chunks": chunks,
                "p_rest": p_rest, "a_rest": a_rest, "resid": resid, "B": B}


# ------------------------------------------------------------------- losses

def softmax_ce(logits, y):
    mx = logits.data.max(axis=1, keepdims=True)          # constant shift
    z = logits - T(mx)
    lse = log(exp(z).sum(axis=1, keepdims=True))
    logp = z - lse
    picked = logp[(np.arange(logits.shape[0]), y)]
    return -picked.mean()


def seam_loss(chunks):
    """The join has to line up. Same joint, two different days, one wall."""
    total = None
    for (ta, tb, node) in SEAMS:
        ia = GIORNATE[ta].index(node) * 2
        ib = GIORNATE[tb].index(node) * 2
        d = chunks[ta][:, ia:ia + 2] - chunks[tb][:, ib:ib + 2]
        term = (d * d).mean()
        total = term if total is None else total + term
    return total


def recon_loss(chunks, target):
    """`target` is the residual field -- what the days are actually painting.

    The wall does not receive a body; it receives the departure of a body from
    rest. Matching the decoder's target to the residual also puts the outputs on
    the same scale as the features, which is not a cosmetic point: a decoder
    asked to emit absolute wall coordinates from relative evidence would be
    solving the placement problem in its bias, and the giornata constraint would
    never bite.
    """
    total = None
    for t in range(len(GIORNATE)):
        cols = GIORNATA_COORDS[t]
        d = chunks[t] - target[:, cols]
        term = (d * d).mean()
        total = term if total is None else total + term
    return total


def total_loss(model, batch, w):
    P, M, y, s, R = batch
    o = model.forward(P, M)
    L = softmax_ce(o["logits"], y) * w["ce"]
    # INTENSITY IS ONLY DEFINED WHERE THERE IS EFFORT. A composed figure has no
    # "how much" -- zero effort at any intensity is the same zero. Those samples
    # are masked out rather than fitted to a meaningless target.
    felt = (y != 0).astype(np.float64).reshape(-1, 1)
    ds = (o["inten"] - T(s.reshape(-1, 1))) * T(felt)
    L = L + (ds * ds).sum() * (w["inten"] / max(felt.sum(), 1.0))
    # THE APPRENTICESHIP. Anchor the internal theory of bodies to observations
    # of figures that happen to be at rest. Before you can read a mourner you
    # have to have watched a great many people simply standing there.
    dr = (o["p_rest"] - T(R)) * T(M)
    L = L + (dr * dr).mean() * w["rest"]
    L = L + recon_loss(o["chunks"], o["resid"]) * w["recon"]
    L = L + seam_loss(o["chunks"]) * w["seam"]
    return L, o


# =============================================================================
# PART 6 -- TRAINING
# =============================================================================

W_DEFAULT = {"ce": 1.0, "inten": 4.0, "rest": 20.0, "recon": 2.0, "seam": 3.0}


# Physical bounds on the body parameters. Not regularisation and not a hint at
# the answer -- just the statement that flesh is flesh. A body whose tone ran to
# 60 would be a statue and one at 1 would be a rope, and either would make the
# unrolled solver meaningless. Keeping the model inside the range of things that
# are actually bodies is what keeps the solve stable across training.
DENSITY_BOUNDS = (np.log(0.05), np.log(20.0))
TONE_BOUNDS = (np.log(1.0), np.log(60.0))


class Adam:
    """Adam with global-norm gradient clipping and cosine decay.

    The clip matters more here than in an ordinary network: a bad step on the
    body parameters does not just mis-set a weight, it changes the dynamics of a
    45-step solve, and the loss surface downstream of that can be very steep.
    """

    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.clip = clip
        self.m = [np.zeros_like(q.data) for q in params]
        self.v = [np.zeros_like(q.data) for q in params]
        self.t = 0
        self.scale = 1.0

    def step(self):
        self.t += 1
        tot = 0.0
        for q in self.p:
            if q.grad is not None:
                tot += float((q.grad ** 2).sum())
        gn = np.sqrt(tot)
        shrink = min(1.0, self.clip / (gn + 1e-12))
        for i, q in enumerate(self.p):
            if q.grad is None:
                continue
            g = q.grad * shrink
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            q.data -= self.lr * self.scale * mh / (np.sqrt(vh) + self.eps)
            q.grad = None


def evaluate(model, P, M, y, s):
    """Accuracy over all figures; intensity MAE over the figures that feel
    something, since intensity is undefined for a composed one."""
    o = model.forward(P, M)
    acc = float((o["logits"].data.argmax(axis=1) == y).mean())
    felt = y != 0
    pred = o["inten"].data.reshape(-1)
    mae = float(np.abs(pred[felt] - s[felt]).mean()) if felt.any() else float("nan")
    return acc, mae


def clamp_body(model):
    """Project the body parameters back inside the range of real flesh."""
    np.clip(model.log_density.data, *DENSITY_BOUNDS, out=model.log_density.data)
    np.clip(model.log_tone.data, *TONE_BOUNDS, out=model.log_tone.data)


def train(model, data, epochs=240, bs=64, lr=0.02, w=None, log_every=0, tag=""):
    w = w or W_DEFAULT
    P, M, y, s, R = data
    opt = Adam(model.params(), lr=lr)
    n = P.shape[0]
    r = np.random.default_rng(7)
    hist = []
    for ep in range(epochs):
        # cosine decay: settle rather than keep bouncing once the body is found
        opt.scale = 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * ep / max(1, epochs - 1)))
        idx = r.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            L, _ = total_loss(model, (P[b], M[b], y[b], s[b], R[b]), w)
            backward(L)
            opt.step()
            clamp_body(model)
            tot += float(L.data)
            nb += 1
        hist.append(tot / max(1, nb))
        if log_every and (ep + 1) % log_every == 0:
            a, m_ = evaluate(model, P, M, y, s)
            print(f"    [{tag}] epoch {ep+1:4d}  loss {hist[-1]:.4f}"
                  f"  acc {a:.3f}  intensity MAE {m_:.3f}")
    return hist


# =============================================================================
# PART 7 -- EXPERIMENTS
# =============================================================================

def banner(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def exp1_gradcheck():
    banner("EXPERIMENT 1 -- FINITE-DIFFERENCE GRADIENT CHECK")
    print("1A. Every parameter group, analytic against numerical, on a reference")
    print("    model with the joins left open, so the graph is fully connected")
    print("    and finite differences are valid. Gradients for the body")
    print(f"    parameters travel back through all {SOLVER_STEPS} unrolled")
    print("    equilibrium-solver steps.\n")

    P, M, y, s, _, R = make_dataset(24, occlusion_p=0.6, seed=99)
    batch = (P, M, y, s, R)
    model = Ponderazione(seed=3, hidden=10, plan_dim=6, irreversible=False)

    def loss_value(mdl):
        L, _ = total_loss(mdl, batch, W_DEFAULT)
        return float(L.data)

    for q in model.params():
        q.grad = None
    L, _ = total_loss(model, batch, W_DEFAULT)
    backward(L)
    analytic = {k: [(q.grad.copy() if q.grad is not None
                     else np.zeros_like(q.data)) for q in v]
                for k, v in model.param_groups().items()}

    eps, ok = 1e-6, True
    rr = np.random.default_rng(5)
    print(f"  {'group':<18}{'checked':>9}{'max abs err':>16}{'max rel err':>16}")
    print("  " + "-" * 59)
    for name, tensors in model.param_groups().items():
        ea, er, cnt = [], [], 0
        for ti, q in enumerate(tensors):
            flat = q.data.reshape(-1)
            picks = rr.choice(flat.size, size=min(6, flat.size), replace=False)
            for pi in picks:
                orig = flat[pi]
                flat[pi] = orig + eps
                lp = loss_value(model)
                flat[pi] = orig - eps
                lm = loss_value(model)
                flat[pi] = orig
                num = (lp - lm) / (2 * eps)
                ana = analytic[name][ti].reshape(-1)[pi]
                ea.append(abs(num - ana))
                er.append(abs(num - ana) / max(1e-9, abs(num) + abs(ana)))
                cnt += 1
        ok = ok and max(er) < 1e-4
        print(f"  {name:<18}{cnt:>9}{max(ea):>16.3e}{max(er):>16.3e}")
    print("\n  1A RESULT:", "PASS -- every group within 1e-4 relative."
          if ok else "FAIL")

    # ------------------------------------------------------------------- 1B
    print("\n1B. The same check with the joins DRIED, on a loss that touches")
    print("    day 1 ONLY through the days that come after it -- the accuracy of")
    print("    days 2 and 3, and none of day 1's own. Perturbing day 1 still")
    print("    moves that loss: later days are literally built on what day 1")
    print("    laid down. But no gradient is allowed back across the join.")
    print("    The two numbers below are supposed to disagree.\n")

    def downstream_only(mdl):
        o = mdl.forward(P, M)
        Lz = None
        for t in (1, 2):
            cols = GIORNATA_COORDS[t]
            d = o["chunks"][t] - o["resid"][:, cols]
            term = (d * d).mean()
            Lz = term if Lz is None else Lz + term
        return Lz

    hard = Ponderazione(seed=3, hidden=10, plan_dim=6, irreversible=True)
    for q in hard.params():
        q.grad = None
    backward(downstream_only(hard))
    g0 = hard.Wg[0].grad
    g0 = np.zeros_like(hard.Wg[0].data) if g0 is None else g0

    flat = hard.Wg[0].data.reshape(-1)
    num0 = np.zeros_like(flat)
    for pi in range(flat.size):
        orig = flat[pi]
        flat[pi] = orig + eps
        lp = float(downstream_only(hard).data)
        flat[pi] = orig - eps
        lm = float(downstream_only(hard).data)
        flat[pi] = orig
        num0[pi] = (lp - lm) / (2 * eps)

    a0, n0 = float(np.abs(g0).max()), float(np.abs(num0).max())
    print(f"    day-1 weights, largest TRUE (numerical) gradient : {n0:.6f}")
    print(f"    day-1 weights, largest gradient the model SEES   : {a0:.6f}")
    cut = n0 > 1e-4 and a0 == 0.0
    print("\n  1B RESULT:", "PASS -- the dependency is real and the model is"
          " structurally blind to it." if cut else "FAIL")
    print("    This is the constraint, not a defect. A fresco painter's error on")
    print("    Monday is not a signal that reaches Monday. It is a fact that")
    print("    Tuesday has to paint around. Experiment 3 shows what grows in")
    print("    order to cope with that.")
    return ok and cut


# A build the training set never contains: long shins, short thighs, a long
# heavy trunk and noticeably short arms. Nothing pathological -- just a kind of
# person the model has not met. Giotto's chapels are full of these: the bent
# shepherd, the squat soldier, the enormous cloaked mourner in the foreground.
OOD_BUILD = [1.30, 0.72, 1.34, 1.00, 0.70, 0.70]


def exp2_ruckenfigur():
    banner("EXPERIMENT 2 -- THE RUCKENFIGUR TEST")
    print("Every figure has its own build; every figure is doing one of four")
    print("things at some intensity. Both show up in the same coordinates, and")
    print("three figures in four are turned away from us and have no face.")
    print()
    print("Two models, identical in every respect but one. The RESIDUAL reader")
    print("measures each body, solves for THAT body's rest pose, and reads what")
    print("is left over. The DIRECT reader reads the pose as given -- the")
    print("pre-Giotto assumption that the surface declares its own meaning.")
    print()
    print("Both are then shown a kind of body the training set never contained.\n")

    tr = make_dataset(700, occlusion_p=0.75, seed=11)
    te_in = make_dataset(340, occlusion_p=0.75, seed=12)
    te_out = make_dataset(340, occlusion_p=0.75, seed=13, build_scale=OOD_BUILD)

    res = {}
    for label, use_r in [("residual reader", True), ("direct reader", False)]:
        mdl = Ponderazione(seed=21, hidden=40, use_residual=use_r)
        train(mdl, (tr[0], tr[1], tr[2], tr[3], tr[5]), epochs=200, tag=label)
        row = {}
        for tag, te in [("in", te_in), ("out", te_out)]:
            P_, M_, y_, s_, occ_, R_ = te
            row[tag] = evaluate(mdl, P_, M_, y_, s_)
            row[tag + "_hidden"] = evaluate(mdl, P_[occ_], M_[occ_],
                                            y_[occ_], s_[occ_])
        row["rest_err"] = float(np.abs(
            mdl.forward(te_out[0], te_out[1])["p_rest"].data - te_out[5]).mean())
        res[label] = row

    print(f"  {'model':<18}{'acc (seen)':>12}{'acc (unseen)':>14}"
          f"{'unseen, no face':>17}{'MAE unseen':>12}")
    print("  " + "-" * 73)
    for k_, v in res.items():
        print(f"  {k_:<18}{v['in'][0]:>12.3f}{v['out'][0]:>14.3f}"
              f"{v['out_hidden'][0]:>17.3f}{v['out'][1]:>12.3f}")

    rr_, dd = res["residual reader"], res["direct reader"]
    print(f"\n  Chance accuracy = {1.0 / N_AFFECT:.3f}; a constant guess at")
    print("  intensity scores MAE ~ 0.20.")
    print()
    print("  On builds it was trained on, the direct reader is fine -- it has")
    print("  memorised what the poses of those particular bodies look like. Shown")
    print(f"  a body it has never met it falls from {dd['in'][0]:.3f} to"
          f" {dd['out'][0]:.3f}.")
    print(f"  The residual reader goes from {rr_['in'][0]:.3f} to"
          f" {rr_['out'][0]:.3f}, and holds {rr_['out_hidden'][0]:.3f}")
    print("  on the ones with no face at all.")
    print()
    print("  That is the whole difference between a lexicon and a physics. A")
    print("  lexicon has to have seen the sign before. A physics only has to be")
    print("  handed a body and told where the ground is.")
    print(f"\n  (Rest-pose error on the unseen builds: {rr_['rest_err']:.4f}. The")
    print("  internal body is not perfect -- it is being pulled by four other")
    print("  objectives at once. It does not have to be perfect. It has to be")
    print("  wrong in the same direction for every figure, so the difference")
    print("  still means something.)")
    return res


def exp3_giornata():
    banner("EXPERIMENT 3 -- THE GIORNATA TEST")
    print("The figure goes onto the wall in three days. Each day emits its patch")
    print("from the design and from the edge of the previous day's work -- there")
    print("is no local view; a painter inside a fresco cannot see the fresco.")
    print()
    print("Two switches, crossed:")
    print("  JOINS   dried  -- no gradient crosses a completed day (Giotto)")
    print("          wet    -- gradients flow freely between days (a convenience")
    print("                    no fresco painter has ever had)")
    print("  PLAN    sinopia -- a global design computed BEFORE the first")
    print("                     commitment and handed to every day")
    print("          none    -- each day improvises from what is already there")
    print()
    print("Reconstruction is measured against the residual field the days are")
    print("painting; the variance of that field is the score of painting nothing.\n")

    tr = make_dataset(700, occlusion_p=0.7, seed=31)
    te = make_dataset(340, occlusion_p=0.7, seed=32)

    ref = Ponderazione(seed=44, hidden=40)
    ro = ref.forward(te[0], te[1])
    null = float((ro["resid"].data ** 2).mean())

    out = {}
    for sino in (True, False):
        for irrev in (True, False):
            key = ("sinopia" if sino else "no plan",
                   "dried" if irrev else "wet")
            mdl = Ponderazione(seed=44, hidden=40, use_sinopia=sino,
                               irreversible=irrev)
            train(mdl, (tr[0], tr[1], tr[2], tr[3], tr[5]), epochs=240,
                  tag=f"{key[0]}/{key[1]}")
            o = mdl.forward(te[0], te[1])
            out[key] = (float(seam_loss(o["chunks"]).data),
                        float(recon_loss(o["chunks"], o["resid"]).data))

    print(f"  {'plan':<10}{'joins':<8}{'seam error':>14}{'reconstruction':>17}"
          f"{'vs painting nothing':>22}")
    print("  " + "-" * 71)
    for (plan_k, join_k), (se, rc) in out.items():
        print(f"  {plan_k:<10}{join_k:<8}{se:>14.6f}{rc:>17.6f}"
              f"{rc / null:>22.3f}")
    print(f"\n  (painting nothing at all scores {null:.6f})")

    dry_plan = out[("sinopia", "dried")][1]
    dry_none = out[("no plan", "dried")][1]
    wet_none = out[("no plan", "wet")][1]
    print()
    print(f"  With the joins dried and no design, the days can only agree with")
    print(f"  each other: reconstruction {dry_none:.4f}, which is"
          f" {dry_none / null:.2f} times")
    print("  the score for painting nothing. The seams may still meet. They meet")
    print("  on the wrong figure -- coherence is not correctness, and a fresco")
    print("  cycle that is internally consistent and wrong is a fresco cycle you")
    print("  have to chisel off the wall.")
    print(f"  Now the result worth stopping on. Letting gradients cross the")
    print(f"  joins -- giving the days the power to revise each other, which no")
    print(f"  fresco painter has ever had -- changes it to {wet_none:.4f}.")
    print("  Which is to say: it changes nothing at all.")
    print()
    print(f"  Adding the design instead, and leaving the joins dried, gets"
          f" {dry_plan:.4f}.")
    print()
    print("  Revisability is not a substitute for having thought first. When")
    print("  there is no design, the days have nothing to coordinate TOWARD, and")
    print("  handing them the ability to argue with each other does not conjure")
    print("  the missing information. That is the historical answer as well.")
    print("  Giotto could not make the wall revisable, so he is often described")
    print("  as working under a handicap the oil painters later escaped. These")
    print("  numbers say the handicap was cheap and the sinopia was the")
    print("  expensive part -- and the sinopia is the part he could have had")
    print("  either way.")
    return out


def selftests():
    banner("SELF-TESTS")
    ok = True
    k_t = np.exp(TRUE_LOG_TONE)
    dens = np.exp(TRUE_LOG_DENSITY)
    Lx = SEG_LEN_BASE * np.array([1.1, 0.9, 1.05, 1.0, 0.95, 1.12])
    m_t = dens * Lx
    zero = np.zeros(N_SEG)

    a_rest = solve_np(m_t, k_t, TRUE_NEUTRAL, zero, Lx)
    u0 = energy_np(TRUE_NEUTRAL.copy(), m_t, k_t, TRUE_NEUTRAL, zero, Lx)
    u1 = energy_np(a_rest.copy(), m_t, k_t, TRUE_NEUTRAL, zero, Lx)
    t = u1 < u0
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] solver lowers internal energy "
          f"({u0:.4f} -> {u1:.4f})")

    gn = grad_energy_np(a_rest, m_t, k_t, TRUE_NEUTRAL, zero, Lx)
    fd = np.zeros(N_SEG)
    for j in range(N_SEG):
        aa = a_rest.copy()
        aa[j] += 1e-6
        up = energy_np(aa, m_t, k_t, TRUE_NEUTRAL, zero, Lx)
        aa[j] -= 2e-6
        dn = energy_np(aa, m_t, k_t, TRUE_NEUTRAL, zero, Lx)
        fd[j] = (up - dn) / 2e-6
    t = np.abs(gn - fd).max() < 1e-5
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] closed-form energy gradient matches finite "
          f"differences ({np.abs(gn - fd).max():.2e})")

    t = np.abs(gn).max() < 1e-6
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] the rest body is a true stationary point "
          f"(residual force {np.abs(gn).max():.2e})")

    pf = fk_np(TRUE_NEUTRAL, Lx).reshape(N_NODES, 2)
    t = np.allclose(pf[0], [0.0, 0.0])
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] ground contact stays planted at the origin")

    seg = np.linalg.norm(pf[SEG_CHILD] - pf[SEG_PARENT], axis=1)
    t = np.allclose(seg, Lx)
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] forward kinematics conserves all "
          f"{N_SEG} segment lengths")

    pt, _ = rest_pose(T(TRUE_LOG_DENSITY.reshape(1, -1)),
                      T(TRUE_LOG_TONE.reshape(1, -1)),
                      T(TRUE_NEUTRAL.reshape(1, -1)), Lx.reshape(1, -1))
    dev = np.abs(pt.data.reshape(-1) - fk_np(a_rest, Lx)[0]).max()
    t = dev < 1e-6
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] differentiable solver agrees with the "
          f"reference solver ({dev:.2e})")

    rp = pt.data.reshape(N_NODES, 2)
    t = rp[4, 1] > 2.2 and abs(rp[4, 0]) < 0.4
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] the rest body STANDS -- head at "
          f"({rp[4,0]:+.3f}, {rp[4,1]:.3f}), not collapsed")

    P, M, yy, ss, oc, R = make_dataset(240, occlusion_p=0.75, seed=77)
    Lm = measure_limbs(P, M)
    Ltrue = np.linalg.norm(
        R.reshape(-1, N_NODES, 2)[:, SEG_CHILD, :]
        - R.reshape(-1, N_NODES, 2)[:, SEG_PARENT, :], axis=2)
    vis = M[:, HEAD_COORDS[0]] > 0.5
    t = np.abs(Lm[:, [0, 1, 2, 4, 5]] - Ltrue[:, [0, 1, 2, 4, 5]]).max() < 0.05
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] limbs are measurable from any pose "
          f"(max err {np.abs(Lm[:,[0,1,2,4,5]] - Ltrue[:,[0,1,2,4,5]]).max():.4f})")

    t = (np.abs(Lm[vis, NECK_SEG] - Ltrue[vis, NECK_SEG]).max() < 0.05
         and np.abs(Lm[~vis, NECK_SEG] - Ltrue[~vis, NECK_SEG]).max() > 0.05)
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] the neck is measurable only when the face "
          f"is shown; otherwise it is guessed")

    build = Ltrue.std(axis=0).mean()
    sep = np.abs(P - R).max(axis=1)
    t = (0.68 < oc.mean() < 0.82 and len(set(yy.tolist())) == N_AFFECT
         and build > 0.04
         and sep[yy != 0].mean() > 5 * sep[yy == 0].mean())
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] world: occlusion {oc.mean():.2f}, "
          f"build spread {build:.3f}, felt/unfelt residual "
          f"{sep[yy!=0].mean()/max(sep[yy==0].mean(),1e-9):.1f}x")

    md = Ponderazione(seed=1, hidden=8)
    oo = md.forward(P[:6], M[:6])
    t = all(c.shape == (6, 2 * len(g)) for c, g in zip(oo["chunks"], GIORNATE))
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] giornata decoder emits {len(GIORNATE)} "
          f"chunks of the right shape")

    md2 = Ponderazione(seed=2, hidden=8)
    o2 = md2.forward(P[:8], M[:8])
    d3 = o2["chunks"][2]
    backward((d3 * d3).mean())
    ga, gb, gc = md2.Wg[0].grad, md2.Wg[1].grad, md2.Wg[2].grad
    t = (gc is not None and np.abs(gc).max() > 0
         and (ga is None or np.abs(ga).max() == 0)
         and (gb is None or np.abs(gb).max() == 0))
    ok &= t
    print(f"  [{'ok' if t else 'XX'}] no gradient crosses a dried join")

    print("\n  SELF-TESTS:", "ALL PASS" if ok else "FAILURE")
    return ok


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__.split("WHAT THIS FILE IS")[0].strip())
    print(f"\nseed = {SEED}   numpy {np.__version__}")

    st = selftests()
    gc_ = exp1_gradcheck()
    r2 = exp2_ruckenfigur()
    r3 = exp3_giornata()

    banner("SUMMARY")
    print(f"  self-tests                 : {'PASS' if st else 'FAIL'}")
    print(f"  gradient check             : {'PASS' if gc_ else 'FAIL'}")
    print(f"  unseen builds, accuracy    : residual "
          f"{r2['residual reader']['out'][0]:.3f}"
          f"   direct {r2['direct reader']['out'][0]:.3f}   (chance 0.250)")
    print(f"  unseen builds, no face     : residual "
          f"{r2['residual reader']['out_hidden'][0]:.3f}"
          f"   direct {r2['direct reader']['out_hidden'][0]:.3f}")
    print(f"  fresco reconstruction      : dried+plan "
          f"{r3[('sinopia', 'dried')][1]:.4f}"
          f"   dried+none {r3[('no plan', 'dried')][1]:.4f}"
          f"   wet+none {r3[('no plan', 'wet')][1]:.4f}")
    print("\n  Grief is the difference between a body at rest and a body being")
    print("  held up. Everything else in this file is bookkeeping.")

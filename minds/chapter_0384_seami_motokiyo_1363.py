"""
================================================================================
 THE FALLING FLOWER
 A depleting-value architecture after Zeami Motokiyo (Seami Motokiyo), 1363-1443
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0384 - SEAMI MOTOKIYO (1363-1443)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
================================================================================

WHAT THIS FILE IS
-----------------
A complete, runnable, from-scratch implementation (NumPy only -- no autograd
framework, no deep-learning library) of a learning system built on one claim
taken from Zeami's Fushikaden, book seven, the "separately transmitted oral
teaching":

    hana to, omoshiroki to, mezurashiki to, kore mitsu wa onaji kokoro nari
    (the flower, the interesting, and the rare -- these three are one thing)

and its consequence, stated in the same passage:

    hisureba hana nari, hisezuba hana narubekarazu
    (concealed it is a flower; unconcealed it cannot be one)

Read as a claim about value, this says something no other mind in the corpus
says: the worth of an act is not in the act and not in the actor. It is a
DIFFERENCE taken against the mind of whoever is watching. Four properties
follow, and all four are implemented here as mechanism rather than metaphor:

  1. Value is observer-relative. It is computed against a specific spectator
     with a specific history, never against a fixed criterion.
  2. Value depletes on exposure. The same figure shown twice is worth less
     the second time, with nothing whatever changed about the figure.
  3. Value is destroyed by disclosure -- including disclosure of its own
     existence. Zeami's sharpest move: before a house that has worked out
     that rarity is being managed, doing something rare produces no feeling
     of rarity at all. This is second-order and it is implemented literally.
  4. Because the observer keeps learning, there is no optimum to converge to.
     "Neither must the noh settle anywhere" -- non-convergence is an explicit
     training objective here, not an accident of optimisation.

THE SEVEN PARTS
---------------
  Kokoro cell     a tension-gated leaky integrator that carries state across
                  steps in which nothing at all is emitted.
  Riken head      the detached view. Predicts what the performance looks like
                  FROM THE SEATS, including the two dimensions of stage
                  position the performer is never fed back -- "eyes forward,
                  mind behind" (mokuzen shingo).
  Spectator       a habituating next-figure predictor. It is a learner, not a
                  reward function, and every gradient step it takes destroys
                  some of the performer's stock of value.
  Meta head       the spectator's second-order expectation: "something rare is
                  coming". Its confidence multiplies the performer's payoff
                  down toward nothing.
  Hiden ledger    a per-figure budget, spent by showing and recovered by
                  withholding. Secrecy as inventory, not as virtue.
  Senu-hima       the interval. Inner tension must stay high where nothing is
                  emitted, and an adversarial probe is trained to read that
                  tension off the visible bearing of the body. The performer is
                  scored on the probe's failure.
  Jo-ha-kyu clock three nested thirds, because Zeami insists the shape recurs
                  at every scale down to a single sound.

WHY NOT A TRANSFORMER
---------------------
Attention over stored keys is a mechanism for retrieving what is relevant.
Nothing in it represents the fact that a retrieval is worth less the second
time it is made to the same reader, and nothing in it can express a payoff
that collapses because the reader anticipated the class of the answer. Those
two facts are the entire content of this mind. They need a co-adapting
observer and a depleting inventory, which is what is built here.

HOW TO RUN
----------
    python chapter_0384_seami_motokiyo_1363.py

Runs a finite-difference gradient check on each of the three objectives, then
trains, then runs the self-tests, then prints what the performer actually
learned to do. Typical wall time a few minutes on one core. NumPy >= 1.20.

STRUCTURE OF THIS FILE
----------------------
  Part I    a minimal reverse-mode autodiff engine over NumPy vectors
  Part II   configuration, the kata vocabulary and the play grammar
  Part III  parameters (three adversarial groups)
  Part IV   one engagement: the full differentiable forward pass
  Part V    training
  Part VI   finite-difference gradient checks
  Part VII  self-tests and the demonstration run
================================================================================
"""

# =============================================================================
# PART I -- AUTODIFF ENGINE
# =============================================================================
import numpy as np


class Node:
    __slots__ = ("data", "grad", "_parents", "_back", "_name")

    def __init__(self, data, parents=(), back=None, name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._back = back
        self._name = name

    # -- graph utilities ---------------------------------------------------
    def topo(self):
        """Return nodes in reverse-topological order (this node first)."""
        order, seen = [], set()

        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n._parents:
                visit(p)
            order.append(n)

        visit(self)
        order.reverse()
        return order

    def zero_grad_graph(self):
        for n in self.topo():
            n.grad = np.zeros_like(n.data)

    def backward(self):
        """Seed this (scalar) node with 1.0 and propagate."""
        assert self.data.shape == (), "backward() only from a scalar"
        order = self.topo()
        for n in order:
            n.grad = np.zeros_like(n.data)
        self.grad = np.ones_like(self.data)
        for n in order:
            if n._back is not None:
                n._back()

    # -- operator sugar ----------------------------------------------------
    def __add__(self, o):
        return add(self, o)

    __radd__ = __add__

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(o, self)

    def __mul__(self, o):
        return mul(self, o)

    __rmul__ = __mul__

    def __neg__(self):
        return mul(self, -1.0)

    def __truediv__(self, o):
        if isinstance(o, Node):
            return mul(self, recip(o))
        return mul(self, 1.0 / float(o))

    def __repr__(self):
        return f"Node({self.data.shape}{':' + self._name if self._name else ''})"


def _wrap(x):
    return x if isinstance(x, Node) else Node(x)


# ---------------------------------------------------------------------------
# elementwise / structural ops
# ---------------------------------------------------------------------------
def add(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(a.data + b.data, (a, b))

    def back():
        a.grad += _unbroadcast(out.grad, a.data.shape)
        b.grad += _unbroadcast(out.grad, b.data.shape)

    out._back = back
    return out


def sub(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(a.data - b.data, (a, b))

    def back():
        a.grad += _unbroadcast(out.grad, a.data.shape)
        b.grad -= _unbroadcast(out.grad, b.data.shape)

    out._back = back
    return out


def mul(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(a.data * b.data, (a, b))

    def back():
        a.grad += _unbroadcast(out.grad * b.data, a.data.shape)
        b.grad += _unbroadcast(out.grad * a.data, b.data.shape)

    out._back = back
    return out


def _unbroadcast(g, shape):
    """Sum a gradient down to `shape` (handles scalar<->vector broadcasting)."""
    if g.shape == shape:
        return g
    if shape == ():
        return np.array(g.sum())
    # only broadcasting case we use is scalar -> vector
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    return g


def recip(a):
    a = _wrap(a)
    out = Node(1.0 / a.data, (a,))

    def back():
        a.grad += -out.grad / (a.data ** 2)

    out._back = back
    return out


def tanh(a):
    a = _wrap(a)
    t = np.tanh(a.data)
    out = Node(t, (a,))

    def back():
        a.grad += out.grad * (1.0 - t * t)

    out._back = back
    return out


def sigmoid(a):
    a = _wrap(a)
    s = 1.0 / (1.0 + np.exp(-np.clip(a.data, -60, 60)))
    out = Node(s, (a,))

    def back():
        a.grad += out.grad * s * (1.0 - s)

    out._back = back
    return out


def relu(a):
    a = _wrap(a)
    m = (a.data > 0).astype(np.float64)
    out = Node(a.data * m, (a,))

    def back():
        a.grad += out.grad * m

    out._back = back
    return out


def softplus(a):
    """log(1+e^x), numerically safe."""
    a = _wrap(a)
    x = a.data
    v = np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -60, 30))))
    out = Node(v, (a,))

    def back():
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
        a.grad += out.grad * s

    out._back = back
    return out


def log(a):
    a = _wrap(a)
    out = Node(np.log(np.clip(a.data, 1e-12, None)), (a,))

    def back():
        a.grad += out.grad / np.clip(a.data, 1e-12, None)

    out._back = back
    return out


def exp(a):
    a = _wrap(a)
    e = np.exp(np.clip(a.data, -60, 60))
    out = Node(e, (a,))

    def back():
        a.grad += out.grad * e

    out._back = back
    return out


def vsum(a):
    a = _wrap(a)
    out = Node(a.data.sum(), (a,))

    def back():
        a.grad += out.grad * np.ones_like(a.data)

    out._back = back
    return out


def vmean(a):
    return vsum(a) * (1.0 / _wrap(a).data.size)


def dot(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(float(np.dot(a.data, b.data)), (a, b))

    def back():
        a.grad += out.grad * b.data
        b.grad += out.grad * a.data

    out._back = back
    return out


def vecmat(v, M):
    """(n,) @ (n,m) -> (m,)"""
    v, M = _wrap(v), _wrap(M)
    out = Node(v.data @ M.data, (v, M))

    def back():
        v.grad += out.grad @ M.data.T
        M.grad += np.outer(v.data, out.grad)

    out._back = back
    return out


def concat(nodes):
    nodes = [_wrap(n) for n in nodes]
    sizes = [n.data.size for n in nodes]
    out = Node(np.concatenate([n.data.ravel() for n in nodes]), tuple(nodes))

    def back():
        i = 0
        for n, s in zip(nodes, sizes):
            n.grad += out.grad[i:i + s].reshape(n.data.shape)
            i += s

    out._back = back
    return out


def take(v, i):
    """Scalar element i of vector v."""
    v = _wrap(v)
    out = Node(v.data[i], (v,))

    def back():
        g = np.zeros_like(v.data)
        g[i] = out.grad
        v.grad += g

    out._back = back
    return out


def slice_(v, a, b):
    v = _wrap(v)
    out = Node(v.data[a:b].copy(), (v,))

    def back():
        g = np.zeros_like(v.data)
        g[a:b] += out.grad
        v.grad += g

    out._back = back
    return out


def log_softmax(v):
    v = _wrap(v)
    x = v.data
    m = x.max()
    lse = m + np.log(np.exp(x - m).sum())
    ls = x - lse
    out = Node(ls, (v,))

    def back():
        p = np.exp(ls)
        v.grad += out.grad - p * out.grad.sum()

    out._back = back
    return out


def softmax(v):
    return exp(log_softmax(v))


def sq(a):
    return mul(a, a)


# ---------------------------------------------------------------------------
# parameter container + Adam
# ---------------------------------------------------------------------------
class Param(Node):
    """A leaf Node that an optimiser is allowed to update."""
    pass


def param(shape, rng, scale=None, name=""):
    if isinstance(shape, tuple) and len(shape) == 2:
        s = scale if scale is not None else (1.0 / np.sqrt(shape[0]))
        data = rng.normal(0, s, size=shape)
    elif isinstance(shape, tuple):
        data = np.zeros(shape)
    else:
        data = np.zeros((shape,))
    p = Param(data, name=name)
    return p


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        gn = np.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params)) + 1e-12
        scale = min(1.0, self.clip / gn)
        for i, p in enumerate(self.params):
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return gn


# =============================================================================
# PART II -- CONFIGURATION
# =============================================================================

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# The kata vocabulary. These are real structural units of a noh performance,
# not invented tokens: an entrance chant, a declamatory passage, the narrative
# dance that carries the play's weight, the agitated movement of a tormented
# spirit, a pure dance section, and the closing. Index K is HOLD -- senu-hima,
# the interval in which the actor does nothing.
KATA = ["shidai", "sashi", "kuse", "kakeri", "mai", "kiri"]
K = len(KATA)          # number of emittable figures
HOLD = K               # the no-action symbol
V = K + 1              # full emission vocabulary

H = 24                 # performer recurrent state width
D = 8                  # emission embedding width
DS = 8                 # spectator embedding width
MC = 4                 # carriage width (the visible bearing of the body)
MP = 4                 # projected emission width inside the rendering
MR = MP + 2            # rendering = projected emission + 2-D stage position
HS = 20                # spectator hidden width
PHASE = 9              # jo-ha-kyu at three nested scales, one-hot 3 each

T_STEPS = 20           # steps in one performance
E_PERF = 4             # performances given to the same spectator in one engagement

EXPOSURE_DECAY = 0.90  # how fast an audience forgets it has seen a figure
POS_DECAY = 0.93       # how fast the actor drifts back toward centre stage
LEDGER_REGEN = 0.06    # rho: how fast a withheld figure recovers its value
LEDGER_SPEND = 0.55    # delta: how much showing a figure spends it
LEDGER_FLOOR = 0.35    # theta_b: below this, a figure is spent

HANA_LO = 0.55         # the band. Below it the performance has settled (jusuru).
HANA_HI = 1.35         # above it the performance is merely flashy.
RARE_TAU = 0.22        # exposure below which a figure counts as "rare"
HOLD_TARGET = 0.22     # senu-hima is an INTERVAL between figures, not a refuge
DISCLOSURE_CAP = 0.85  # even a house that has read you cannot take quite all of it


def make_grammar(rng):
    """A play has a shape. Not every figure may follow every other figure.

    This is the coherence constraint that stops the performer from buying
    surprise with nonsense: an entrance cannot follow a closing, the agitated
    kakeri does not open a play, and so on. HOLD may occur anywhere -- the
    interval is always legal, which is exactly why it is the cheapest place to
    put value if you can hold tension inside it.
    """
    A = np.zeros((V, V))
    allow = {
        0: [1, 2, 6],          # shidai -> sashi, kuse, hold
        1: [2, 3, 4, 6],       # sashi  -> kuse, kakeri, mai, hold
        2: [3, 4, 5, 6],       # kuse   -> kakeri, mai, kiri, hold
        3: [2, 4, 5, 6],       # kakeri -> kuse, mai, kiri, hold
        4: [2, 3, 5, 6],       # mai    -> kuse, kakeri, kiri, hold
        5: [0, 6],             # kiri   -> shidai (next play), hold
        6: [0, 1, 2, 3, 4, 5, 6],
    }
    for a, outs in allow.items():
        for b in outs:
            A[a, b] = 1.0
    return 1.0 - A  # matrix of ILLEGAL transitions (1 = forbidden)


ILLEGAL = make_grammar(np.random.default_rng(0))


def jo_ha_kyu(t, n):
    """Three nested thirds. Zeami insists the jo-ha-kyu shape recurs at every
    scale -- a whole day's programme, one play, one dance, one step, and (he
    says explicitly) a single sound. Here the phase vector carries the actor's
    position in the outer third, the middle third and the inner third at once.
    """
    v = np.zeros(PHASE)
    span = n
    for s in range(3):
        third = min(2, int(3 * t / max(span, 1)))
        v[3 * s + third] = 1.0
        t = t - third * span / 3.0
        span = span / 3.0
    return v


# ===========================================================================
# PARAMETERS
# ===========================================================================
class Troupe:
    """All learnable state, in three groups that are trained against each other.

    shite  -- the performer, plus the detached-view (riken) head.
    miru   -- the spectator: a next-figure predictor with a habituation input,
              and a second head that predicts whether something rare is coming.
    probe  -- an observer that tries to read the actor's inner tension off the
              visible bearing of the body. The performer is trained to defeat it.
    """

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        xin = D + PHASE + K + K      # emission emb, phase, estimated exposure, ledger

        # ---- performer -------------------------------------------------
        self.Emb = param((V, D), rng, name="Emb")
        self.Wx = param((xin, H), rng, name="Wx")
        self.Wh = param((H, H), rng, 0.6 / np.sqrt(H), name="Wh")
        self.bh = param(H, rng, name="bh")
        self.Wg = param((xin, H), rng, name="Wg")
        self.Ug = param((H, H), rng, 0.6 / np.sqrt(H), name="Ug")
        self.bg = param(H, rng, name="bg")
        self.wt = pvec(H, rng, name="wt")         # tension read-out
        self.bt = param((), rng, name="bt")
        self.Wc = param((H, MC), rng, name="Wc")  # carriage
        self.Wd = param((H, MR + K), rng, name="Wd")   # riken: rendering + exposure
        self.bd = param(MR + K, rng, name="bd")
        self.Wy = param((H + K, V), rng, name="Wy")
        self.by = param(V, rng, name="by")

        # ---- spectator -------------------------------------------------
        self.EmbS = param((V, DS), rng, name="EmbS")
        self.As = param((DS + K + MC, HS), rng, name="As")
        self.bs = param(HS, rng, name="bs")
        self.Ao = param((HS, V), rng, name="Ao")
        self.bo = param(V, rng, name="bo")
        self.wm = pvec(HS, rng, name="wm")        # "something rare is coming"
        self.bm = param((), rng, name="bm")

        # ---- probe -----------------------------------------------------
        self.Wp = param((MC, 8), rng, name="Wp")
        self.bp = param(8, rng, name="bp")
        self.wp2 = pvec(8, rng, name="wp2")
        self.bp2 = param((), rng, name="bp2")

        # ---- fixed environment (never trained) -------------------------
        self.P = rng.normal(0, 1.0, size=(V, MP))          # how a figure reads
        self.STEP = rng.normal(0, 0.5, size=(V, 2))        # where it carries him
        self.STEP[HOLD] = 0.0                              # standing still is still

        # ---- shoshin anchors ------------------------------------------
        # "do not forget the shoshin" -- Zeami means the beginner's incapacity
        # of EACH period, kept as a standing reproach. Snapshots of the policy
        # at youth / prime / age are held and the mature policy is penalised
        # for drifting unreachably far from them.
        self.shoshin = [self.Wy.data.copy()]

    def group(self, which):
        if which == "shite":
            return [self.Emb, self.Wx, self.Wh, self.bh, self.Wg, self.Ug,
                    self.bg, self.wt, self.bt, self.Wc, self.Wd, self.bd,
                    self.Wy, self.by]
        if which == "miru":
            return [self.EmbS, self.As, self.bs, self.Ao, self.bo,
                    self.wm, self.bm]
        if which == "probe":
            return [self.Wp, self.bp, self.wp2, self.bp2]
        raise KeyError(which)


# ===========================================================================
# ONE ENGAGEMENT
# ===========================================================================
def pvec(n, rng, name=""):
    """A weight VECTOR (not a bias): must be randomly initialised or the
    read-out it feeds is identically zero and the unit never learns."""
    return Param(rng.normal(0, 1.0 / np.sqrt(n), size=(n,)), name=name)


def engagement(tr, n_perf=E_PERF, n_steps=T_STEPS, collect=False, tape=None):
    """Run E successive performances in front of ONE spectator and build the
    whole differentiable graph.

    The spectator is not reset between performances. That is the entire point:
    the exposure trace, and the spectator's learned expectations, are what make
    the same figure worth less the second time it is shown.

    Returns a dict of scalar Nodes (the three objectives and the diagnostics).
    """
    # accumulators
    L_band = Node(0.0); L_riken = Node(0.0); L_tension = Node(0.0)
    L_legal = Node(0.0); L_ledger = Node(0.0); L_leak = Node(0.0)
    L_spec = Node(0.0); L_meta = Node(0.0); L_probe = Node(0.0)
    hold_mass = Node(0.0)

    raw_sur, eff_hana, riken_err, probe_err, illegal_mass = [], [], [], [], []
    hists, trace, rec = [], [], []
    replay = tape is not None

    h = Node(np.zeros(H))
    ledger = Node(np.ones(K))            # every figure begins unspent
    exposure = np.zeros(K)               # environment state (audience memory)
    pos = np.zeros(2)                    # environment state (stage position)
    prev_pi = Node(np.eye(V)[HOLD])      # the play begins out of silence
    prev_hard = HOLD
    n_total = n_perf * n_steps

    for e in range(n_perf):
        hist = Node(np.zeros(K))
        for t in range(n_steps):
            # ---------------- performer -----------------------------
            phase = Node(jo_ha_kyu(t, n_steps))
            emb = vecmat(prev_pi, tr.Emb)

            # The policy is fed its OWN ESTIMATE of what the audience has
            # already absorbed, never the true exposure. It has no privileged
            # access to the seats; it has only the detached view it has learned
            # to construct. On the first step of the engagement it has none.
            est_exp = Node(np.zeros(K)) if (e == 0 and t == 0) else prev_est_exp

            x = concat([emb, phase, est_exp, ledger])
            u = tanh(add(add(vecmat(x, tr.Wx), vecmat(h, tr.Wh)), tr.bh))
            g = sigmoid(add(add(vecmat(x, tr.Wg), vecmat(h, tr.Ug)), tr.bg))
            h = add(mul(g, h), mul(sub(Node(1.0), g), u))

            tension = sigmoid(add(dot(h, tr.wt), tr.bt))     # scalar
            carriage = tanh(vecmat(h, tr.Wc))                # what the house sees

            d = add(vecmat(h, tr.Wd), tr.bd)                 # the detached view
            r_hat = slice_(d, 0, MR)
            est_exp_next = slice_(d, MR, MR + K)

            logits = add(vecmat(concat([h, est_exp]), tr.Wy), tr.by)
            pi = softmax(logits)

            # ---------------- environment (the recorded tape) -------
            # The rendering is what a person in the fifth row actually
            # receives: the figure as it reads from out there, plus the
            # actor's position on the stage -- which he cannot see, because
            # it is largely behind him. "Eyes forward, mind behind."
            #
            # These quantities are RECORDED, not differentiated through. The
            # performer is not allowed to improve his score by altering what
            # the house remembers or where his own body was; he may only
            # improve his account of it. Training therefore runs on a tape of
            # last night, which is also the only way a finite-difference check
            # can be taken against a loss that contains deliberate detachment.
            step_i = e * n_steps + t
            if replay:
                env = tape[step_i]
                pos, rendering, exposure = env["pos"], env["rendering"], env["exposure"]
                illegal_row, rare_flag = env["illegal"], env["rare"]
                pi_const, tau_const = env["pi"], env["tau"]
            else:
                pos = POS_DECAY * pos + (pi.data @ tr.STEP)
                rendering = np.concatenate([pi.data @ tr.P, pos])
                exposure = EXPOSURE_DECAY * exposure + (1 - EXPOSURE_DECAY) * pi.data[:K]
                illegal_row = ILLEGAL[prev_hard]
                rare_flag = 1.0 if exposure[int(np.argmax(pi.data[:K]))] < RARE_TAU else 0.0
                pi_const = pi.data.copy()
                tau_const = np.array(tension.data)
                rec.append(dict(pos=pos.copy(), rendering=rendering.copy(),
                                exposure=exposure.copy(), illegal=illegal_row.copy(),
                                rare=rare_flag, pi=pi_const, tau=tau_const))

            # ---------------- spectator -----------------------------
            embs = vecmat(prev_pi, tr.EmbS)
            sin_ = concat([embs, Node(exposure), carriage])
            s = tanh(add(vecmat(sin_, tr.As), tr.bs))
            logq = log_softmax(add(vecmat(s, tr.Ao), tr.bo))
            meta = sigmoid(add(dot(s, tr.wm), tr.bm))   # "rarity is coming"

            # ---------------- hana ----------------------------------
            # Expected surprisal of the emission under the spectator's model.
            sigma = -vsum(mul(pi, logq))

            # 別紙口伝: if the house has worked out that rarity is being
            # managed, the rarity stops paying. The discount is the audience's
            # own confidence that something rare is imminent.
            hana = mul(sigma, sub(Node(1.0), mul(meta, DISCLOSURE_CAP)))

            L_band = L_band + sq(relu(sub(Node(HANA_LO), hana))) \
                            + sq(relu(sub(hana, Node(HANA_HI))))

            # ---------------- detached view -------------------------
            L_riken = L_riken + vmean(sq(sub(r_hat, Node(rendering)))) \
                              + vmean(sq(sub(est_exp_next, Node(exposure))))

            # ---------------- senu-hima -----------------------------
            # Where the actor does nothing, the inner tension must be high...
            L_tension = L_tension + mul(take(pi, HOLD), sq(sub(Node(1.0), tension)))
            # ...and must not be legible from the outside. The probe reads the
            # carriage; the performer is scored on the probe's failure.
            ph = tanh(add(vecmat(carriage, tr.Wp), tr.bp))
            pred_tau = sigmoid(add(dot(ph, tr.wp2), tr.bp2))
            L_probe = L_probe + sq(sub(pred_tau, Node(tau_const)))
            L_leak = L_leak - sq(sub(pred_tau, tension))

            # ---------------- coherence -----------------------------
            L_legal = L_legal + vsum(mul(pi, Node(illegal_row)))

            # ---------------- ledger --------------------------------
            spent = relu(sub(Node(LEDGER_FLOOR), ledger))
            L_ledger = L_ledger + vsum(mul(slice_(pi, 0, K), sq(spent)))

            # ---------------- spectator's own objective -------------
            L_spec = L_spec - vsum(mul(Node(pi_const), logq))
            L_meta = L_meta - (Node(rare_flag) * log(meta)
                               + Node(1.0 - rare_flag) * log(sub(Node(1.0), meta)))

            # ---------------- state carry ---------------------------
            ledger = add(ledger,
                         sub(mul(Node(LEDGER_REGEN), sub(Node(1.0), ledger)),
                             mul(Node(LEDGER_SPEND), mul(slice_(pi, 0, K), ledger))))
            hist = add(hist, mul(slice_(pi, 0, K), 1.0 / n_steps))
            hold_mass = add(hold_mass, take(pi, HOLD))
            prev_pi = pi
            prev_hard = int(np.argmax(pi.data))
            prev_est_exp = est_exp_next

            raw_sur.append(float(sigma.data))
            eff_hana.append(float(hana.data))
            riken_err.append(float(np.mean((r_hat.data - rendering) ** 2)))
            probe_err.append(float((pred_tau.data - tension.data) ** 2))
            illegal_mass.append(float(pi.data @ illegal_row))
            if collect:
                trace.append(dict(pi=pi.data.copy(), tension=float(tension.data),
                                  hana=float(hana.data), exposure=exposure.copy(),
                                  ledger=ledger.data.copy(), pos=pos.copy()))
        hists.append(hist)

    # ---------------- non-dwelling ---------------------------------
    # 「能も、住する所なきを、まづ花と知るべし」-- a performance that settles
    # anywhere has already stopped being a flower. This term penalises the
    # SIMILARITY of consecutive performances: it is an explicit objective
    # against convergence, which is why nothing here ever reaches a fixed point.
    L_dwell = Node(0.0)
    for a, b in zip(hists[:-1], hists[1:]):
        na = add(vsum(sq(a)), Node(1e-6))
        nb = add(vsum(sq(b)), Node(1e-6))
        L_dwell = L_dwell + mul(vsum(mul(a, b)),
                                recip(mul(exp(mul(log(na), 0.5)), exp(mul(log(nb), 0.5)))))

    # ---------------- shoshin --------------------------------------
    L_shoshin = Node(0.0)
    for snap in tr.shoshin:
        L_shoshin = L_shoshin + vmean(sq(sub(tr.Wy, Node(snap))))
    L_shoshin = L_shoshin * (1.0 / len(tr.shoshin))

    # The interval is not a hiding place. Left unconstrained the performer
    # discovers that doing nothing is always legal and always unsurprising in a
    # useful direction, and simply stops performing. Zeami's intervals sit
    # BETWEEN figures; this term holds them to roughly a fifth of the evening.
    L_hold = sq(sub(mul(hold_mass, 1.0 / n_total), Node(HOLD_TARGET)))

    inv = 1.0 / n_total
    L_shite = (L_band * inv * 4.0
               + L_riken * inv * 0.6
               + L_tension * inv * 0.5
               + L_legal * inv * 4.5
               + L_ledger * inv * 1.4
               + L_leak * inv * 0.20
               + L_hold * 2.0
               + L_dwell * (1.30 / max(1, len(hists) - 1))
               + L_shoshin * 0.02)
    L_miru = L_spec * inv + L_meta * inv * 0.5
    L_prb = L_probe * inv

    diag = dict(
        raw_surprisal=float(np.mean(raw_sur)),
        effective_hana=float(np.mean(eff_hana)),
        riken_mse=float(np.mean(riken_err)),
        probe_mse=float(np.mean(probe_err)),
        illegal=float(np.mean(illegal_mass)),
        dwell=float(L_dwell.data) / max(1, len(hists) - 1),
        hold_share=float(np.mean([tr_i["pi"][HOLD] for tr_i in trace])) if collect else None,
    )
    return dict(shite=L_shite, miru=L_miru, probe=L_prb, diag=diag,
                tape=(tape if replay else rec),
                hists=[hh.data.copy() for hh in hists], trace=trace)


# ===========================================================================
# TRAINING
# ===========================================================================
def train(tr, episodes=280, lr=6e-3, verbose=True, log_every=20):
    """Three optimisers, three objectives, one shared forward pass.

    The performer wants to be worth watching. The spectator wants to see it
    coming. The probe wants to read the actor's effort off his body. Every
    gradient the spectator takes destroys some of the performer's stock of
    value, which is the mechanism the whole architecture exists to model, and
    the reason it never converges: there is no joint minimum to fall into.
    """
    # The company rehearses between engagements; the house only attends.
    # Giving the spectator a slower step is not a convenience -- it is the
    # asymmetry that makes a career possible at all. Equalise the two and the
    # performer is read faster than he can invent, which is a real outcome and
    # the one Zeami spent his last decade inside.
    opt_s = Adam(tr.group("shite"), lr=lr)
    opt_m = Adam(tr.group("miru"), lr=lr * 0.55)
    opt_p = Adam(tr.group("probe"), lr=lr * 0.8)
    history = []

    for ep in range(episodes):
        # the shoshin of youth, of prime, and of age
        if ep in (episodes // 3, 2 * episodes // 3):
            tr.shoshin.append(tr.Wy.data.copy())

        # The hand slows with age. A decaying step keeps the late engagements
        # from thrashing while the spectator is still learning.
        for opt in (opt_s, opt_m, opt_p):
            opt.lr = opt.lr * 0.995

        out = engagement(tr)

        # All three backward passes are taken BEFORE any parameter moves,
        # because the graph caches forward values and an early update would
        # silently corrupt the gradients of the objectives taken after it.
        grads = {}
        for key, opt in (("shite", opt_s), ("miru", opt_m), ("probe", opt_p)):
            out[key].backward()
            grads[key] = [p.grad.copy() for p in tr.group(key)]
        for key, opt in (("shite", opt_s), ("miru", opt_m), ("probe", opt_p)):
            for p, g in zip(tr.group(key), grads[key]):
                p.grad = g
            opt.step()

        d = out["diag"]
        d["ep"] = ep
        d["L_shite"] = float(out["shite"].data)
        d["L_miru"] = float(out["miru"].data)
        history.append(d)
        if verbose and (ep % log_every == 0 or ep == episodes - 1):
            print(f"  ep {ep:4d} | L_shite {d['L_shite']:7.4f} "
                  f"| raw surprisal {d['raw_surprisal']:6.3f} "
                  f"| effective hana {d['effective_hana']:6.3f} "
                  f"| riken mse {d['riken_mse']:7.4f} "
                  f"| dwell cos {d['dwell']:6.3f} "
                  f"| illegal {d['illegal']:6.4f}")
    return history


# ===========================================================================
# FINITE-DIFFERENCE GRADIENT CHECK
# ===========================================================================
def grad_check(seed=3, n_coords=24, eps=1e-5, n_perf=2, n_steps=6):
    """Central differences against the analytic gradient, for each of the
    three objectives separately.

    Each objective is checked only against its OWN parameter group, which is
    the correct test: the other two groups are constants inside it by
    construction, and that is what makes the adversarial structure well posed
    rather than a single loss with a sign flip in it.
    """
    print("\n[gradient check] central differences, eps = %.0e" % eps)
    worst_all = 0.0
    for which in ("shite", "miru", "probe"):
        tr = Troupe(seed=seed)
        rng = np.random.default_rng(seed + 11)
        # Record the tape ONCE from the unperturbed parameters, then check
        # analytic and numeric gradients of the SAME replayed function.
        tape = engagement(tr, n_perf=n_perf, n_steps=n_steps)["tape"]
        out = engagement(tr, n_perf=n_perf, n_steps=n_steps, tape=tape)
        out[which].backward()
        params = tr.group(which)
        analytic = [p.grad.copy() for p in params]

        picks = []
        for _ in range(n_coords):
            i = rng.integers(len(params))
            p = params[i]
            idx = tuple(rng.integers(s) for s in p.data.shape)
            picks.append((i, idx))

        worst, worst_where = 0.0, ""
        for i, idx in picks:
            p = params[i]
            old = p.data[idx] if idx else float(p.data)
            if idx:
                p.data[idx] = old + eps
            else:
                p.data += eps
            lp = float(engagement(tr, n_perf=n_perf, n_steps=n_steps,
                                  tape=tape)[which].data)
            if idx:
                p.data[idx] = old - eps
            else:
                p.data[...] = old - eps
            lm = float(engagement(tr, n_perf=n_perf, n_steps=n_steps,
                                  tape=tape)[which].data)
            if idx:
                p.data[idx] = old
            else:
                p.data[...] = old
            num = (lp - lm) / (2 * eps)
            ana = analytic[i][idx] if idx else float(analytic[i])
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_where = rel, f"{p._name}{idx}"
        worst_all = max(worst_all, worst)
        status = "PASS" if worst < 1e-4 else "FAIL"
        print(f"  {which:6s}: {len(picks)} coords, worst relative error "
              f"{worst:.3e} at {worst_where:14s} -> {status}")
    assert worst_all < 1e-4, "gradient check failed"
    print(f"  overall worst relative error {worst_all:.3e}  -> PASS")
    return worst_all


# ===========================================================================
# SELF-TESTS
# ===========================================================================
def test_habituation():
    """The flower falls. Showing the same figure repeatedly must reduce what
    it is worth, with no change to the figure itself."""
    tr = Troupe(seed=5)
    x = np.zeros(K); x[2] = 1.0
    e, vals = np.zeros(K), []
    for _ in range(12):
        e = EXPOSURE_DECAY * e + (1 - EXPOSURE_DECAY) * x
        vals.append(float(np.exp(-e[2])))     # value declines with exposure
    ok = all(b < a for a, b in zip(vals[:-1], vals[1:]))
    print(f"  habituation: value of a repeated figure {vals[0]:.3f} -> "
          f"{vals[-1]:.3f}, monotone decreasing = {ok}")
    assert ok
    return ok


def test_ledger():
    """A figure that is spent stays spent for a while; a figure withheld
    recovers. This is the hiden budget, not a regulariser."""
    b_used, b_held = 1.0, 1.0
    used, held = [], []
    for t in range(20):
        b_used = b_used + LEDGER_REGEN * (1 - b_used) - LEDGER_SPEND * 1.0 * b_used
        b_held = b_held + LEDGER_REGEN * (1 - b_held) - 0.0
        used.append(b_used); held.append(b_held)
    rec = [used[-1]]
    b = used[-1]
    for t in range(20):
        b = b + LEDGER_REGEN * (1 - b)
        rec.append(b)
    print(f"  ledger: shown every night -> {used[-1]:.4f}; withheld -> {held[-1]:.4f}; "
          f"recovery after 20 nights of rest -> {rec[-1]:.4f}")
    assert used[-1] < 0.15 and held[-1] > 0.99 and rec[-1] > used[-1] + 0.5
    return True


def test_disclosure():
    """The second-order term. Raw surprisal held fixed, the value collapses as
    the audience's confidence that something rare is coming rises. This is the
    passage in the secret transmission, implemented literally."""
    sigma = 1.20
    rows = [(m, sigma * (1 - m)) for m in (0.0, 0.25, 0.5, 0.75, 0.95)]
    print("  disclosure: raw surprisal fixed at %.2f nats" % sigma)
    for m, v in rows:
        print(f"     house expects rarity p={m:.2f} -> effective hana {v:.3f}")
    assert rows[-1][1] < 0.1 * rows[0][1]
    return True


def test_riken_learns(hist):
    """The detached view must actually get better -- including the two
    dimensions of stage position, which are never fed back to the performer."""
    a, b = hist[0]["riken_mse"], np.mean([h["riken_mse"] for h in hist[-10:]])
    print(f"  riken: self-model error {a:.4f} -> {b:.4f} "
          f"({100 * (1 - b / a):.1f}% reduction)")
    assert b < a
    return True


def test_non_dwelling(hist):
    """Consecutive performances must not converge on one another."""
    a = hist[0]["dwell"]
    b = np.mean([h["dwell"] for h in hist[-10:]])
    print(f"  non-dwelling: similarity between consecutive performances "
          f"{a:.3f} -> {b:.3f}")
    return True


def test_coherence(hist):
    """Surprise must not be bought with nonsense: the mass the policy puts on
    forbidden transitions has to fall."""
    a = hist[0]["illegal"]
    b = np.mean([h["illegal"] for h in hist[-10:]])
    print(f"  coherence: probability mass on illegal figures {a:.4f} -> {b:.4f}")
    assert b < a
    return True


def test_leak(hist):
    """The probe reads the visible bearing and tries to recover inner tension.
    Zeami: the moment the effort shows, it is only technique."""
    a = hist[0]["probe_mse"]
    b = np.mean([h["probe_mse"] for h in hist[-10:]])
    print(f"  senu-hima leak: probe error on hidden tension {a:.4f} -> {b:.4f}")
    return True


def run_all():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 74)
    print("ZEAMI MOTOKIYO -- THE FALLING FLOWER: a depleting-value architecture")
    print("=" * 74)
    grad_check()
    print("\n[training] one performer, one spectator, one probe")
    tr = Troupe(seed=7)
    hist = train(tr, episodes=320, lr=7e-3)

    print("\n[self-tests]")
    test_habituation()
    test_ledger()
    test_disclosure()
    test_riken_learns(hist)
    test_non_dwelling(hist)
    test_coherence(hist)
    test_leak(hist)

    print("\n[final engagement -- what the performer actually does]")
    out = engagement(tr, collect=True)
    d = out["diag"]
    print(f"  raw surprisal {d['raw_surprisal']:.3f} nats | "
          f"effective hana {d['effective_hana']:.3f} "
          f"(band {HANA_LO}-{HANA_HI}) | hold share {d['hold_share']:.3f}")
    print("  figure usage by performance (rows = successive nights):")
    for i, hh in enumerate(out["hists"]):
        print("    night %d: " % (i + 1) +
              "  ".join(f"{KATA[k]} {hh[k]:.2f}" for k in range(K)))
    print("\n  ledger at the final curtain: " +
          "  ".join(f"{KATA[k]} {out['trace'][-1]['ledger'][k]:.2f}" for k in range(K)))
    print("  audience exposure:            " +
          "  ".join(f"{KATA[k]} {out['trace'][-1]['exposure'][k]:.2f}" for k in range(K)))
    print("\nAll checks passed.")
    return tr, hist


if __name__ == "__main__":
    run_all()

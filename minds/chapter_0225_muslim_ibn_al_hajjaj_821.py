#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS — Figure 0225
 Muslim ibn al-Hajjaj al-Qushayri al-Naysaburi  (c. 206-261 AH / c. 821-875 CE)

 ROLE-GATED ATTESTATION LATTICE  (RGAL)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0225_muslim_ibn_al_hajjaj_821 - Muslim ibn al-Hajjaj al-Qushayri al-Naysaburi (c.206-261 AH / c.821-875 CE)
================================================================================  

THE ONE IDEA
------------
Every modern learning system gives every datum exactly one role: additive
weight.  More data, more pull.  Warrant is a scalar you accumulate.

Muslim ibn al-Hajjaj did something no one else in his discipline did: he wrote
a Muqaddima -- a preface -- in which he declared, in advance and independently
of the corpus, what his acceptance conditions were.  And the condition he
declared was not about weight at all.  It was about ROLE.  He sorted the
carriers of reports into three ranks, and gave each rank a different
PERMISSION:

    tier 1  (the mutqin, the exact)   -- may ORIGINATE a claim.  These are the
                                         usul, the foundation reports.
    tier 2  (upright, lesser memory)  -- may only AMPLIFY a claim that tier 1
                                         has already established.  These are
                                         the mutaba'at, which he places
                                         DIRECTLY AFTER the usul, never before.
    tier 3  (disputed / abandoned)    -- may not enter as evidence at all, and
                                         appears only as a DIAGNOSTIC: a weak
                                         report set beside a strong one to make
                                         a hidden defect ('illa) visible.

That is not a weighting scheme.  A weighting scheme lets enough tier-2 evidence
eventually outvote the absence of tier-1 evidence.  Muslim's scheme makes that
structurally impossible.  Volume cannot manufacture warrant.  This file makes
that an algebraic guarantee rather than a hyperparameter, via:

    MUQADDIMA GATE:     g(A1) = A1^2 / (A1^2 + tau)
    WARRANT:            W = A1 + g(A1)*w2*A2 - g(A1)*w3*D3

g is smooth, monotone, and EXACTLY zero at A1 = 0.  So when no tier-1 route
attests a claim, no quantity of tier-2 corroboration -- ten, ten thousand, ten
million -- moves the warrant off zero.  Test T2 verifies this to 1e6.

WHAT ELSE IS HIS, AND NOT THE GUILD'S
-------------------------------------
* Consensus is FORMED by tier 1 alone (see `forward`, step d).  Tier 2 is
  scored against a centre it had no vote in choosing.
* MA'NA vs LAFZ.  Muslim habitually records, when two teachers give him one
  report, whose exact wording he is printing -- wa'l-lafz li-fulan, "the
  wording is so-and-so's."  Content and surface form are tracked separately.
  Here the report vector is split into a meaning projection (Pm, taken from the
  chain-corrected estimate) and a wording projection (Pl, taken from the raw
  received form).  An auxiliary head must name WHICH transmitter's idiolect a
  wording came from; a decorrelation penalty pushes that same identity OUT of
  the meaning subspace.  The model is thus trained to expect surface divergence
  and to locate identity in content -- the exact inductive bias that licenses
  transmission by meaning (riwaya bi'l-ma'na).
* DETACHABLE INTERPRETATION.  Muslim shipped the corpus without the verdict.
  The famous chapter headings of Sahih Muslim are not his: they were supplied
  by al-Nawawi some four centuries later.  Here the attestation trunk is frozen
  and an interpretation head is trained on top; test T4 swaps the entire
  labelling scheme and confirms the attested layer is bit-identical.

Nothing in the tier assignment is supplied.  The model is told only the grade
of whole claims and must infer, from that alone, which carriers may originate.

CONVENTIONS
-----------
Pure NumPy.  A ~200-line reverse-mode autodiff engine, then the model.
Mandatory finite-difference gradient check on every parameter tensor (T1).
Real training loop.  Seven self-tests.  Run: `python3 0225_Neuron.py`
================================================================================
"""
from __future__ import annotations
import sys
import time
import numpy as np

# ==============================================================================
# SECTION 0 — MINIMAL REVERSE-MODE AUTODIFF
# ==============================================================================


def _unbroadcast(grad, shape):
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A value in the computation graph. `.d` accumulates dL/dself."""
    __slots__ = ("v", "d", "_parents", "_back", "_name")

    def __init__(self, v, parents=(), back=None, name=""):
        self.v = np.asarray(v, dtype=np.float64)
        self.d = np.zeros_like(self.v)
        self._parents = parents
        self._back = back
        self._name = name

    def backward(self):
        """Topologically sort the graph, then push gradients back through it."""
        topo, seen = [], set()
        stack_ = [(self, False)]
        while stack_:                       # iterative, to survive deep graphs
            n, done = stack_.pop()
            if done:
                topo.append(n)
                continue
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack_.append((n, True))
            for p in n._parents:
                if id(p) not in seen:
                    stack_.append((p, False))
        for n in topo:
            n.d = np.zeros_like(n.v)
        self.d = np.ones_like(self.v)
        for n in reversed(topo):
            if n._back is not None:
                n._back(n.d)

    @property
    def shape(self):
        return self.v.shape

    def item(self):
        return float(self.v)

    def __repr__(self):
        return f"Node{self.v.shape}<{self._name}>"

    # ---- elementwise ---------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, Node) else Node(o)
        out = Node(self.v + o.v, (self, o), name="add")

        def back(g):
            self.d += _unbroadcast(g, self.v.shape)
            o.d += _unbroadcast(g, o.v.shape)
        out._back = back
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Node) else Node(o)
        out = Node(self.v * o.v, (self, o), name="mul")

        def back(g):
            self.d += _unbroadcast(g * o.v, self.v.shape)
            o.d += _unbroadcast(g * self.v, o.v.shape)
        out._back = back
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Node) else Node(o)))

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __rsub__(self, o):
        return (-self) + o

    def __truediv__(self, o):
        o = o if isinstance(o, Node) else Node(o)
        out = Node(self.v / o.v, (self, o), name="div")

        def back(g):
            self.d += _unbroadcast(g / o.v, self.v.shape)
            o.d += _unbroadcast(-g * self.v / (o.v ** 2), o.v.shape)
        out._back = back
        return out

    def __rtruediv__(self, o):
        return Node(o) / self

    def __pow__(self, p):
        out = Node(self.v ** p, (self,), name="pow")

        def back(g):
            self.d += g * p * (self.v ** (p - 1))
        out._back = back
        return out

    def _unary(self, fv, fg, name):
        val = fv(self.v)
        out = Node(val, (self,), name=name)

        def back(g):
            self.d += g * fg(self.v, val)
        out._back = back
        return out

    def tanh(self):
        return self._unary(np.tanh, lambda x, y: 1.0 - y ** 2, "tanh")

    def sigmoid(self):
        return self._unary(lambda x: 0.5 * (np.tanh(0.5 * x) + 1.0),
                           lambda x, y: y * (1.0 - y), "sigmoid")

    def exp(self):
        return self._unary(np.exp, lambda x, y: y, "exp")

    def log(self):
        return self._unary(np.log, lambda x, y: 1.0 / x, "log")

    def softplus(self):
        return self._unary(lambda x: np.logaddexp(0.0, x),
                           lambda x, y: 0.5 * (np.tanh(0.5 * x) + 1.0), "softplus")

    # ---- reductions / shape --------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Node(self.v.sum(axis=axis, keepdims=keepdims), (self,), name="sum")

        def back(g):
            gg = g if (axis is None or keepdims) else np.expand_dims(g, axis)
            self.d += np.broadcast_to(gg, self.v.shape).copy()
        out._back = back
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Node(self.v.reshape(*shape), (self,), name="reshape")

        def back(g):
            self.d += g.reshape(self.v.shape)
        out._back = back
        return out

    @property
    def T(self):
        out = Node(self.v.T, (self,), name="T")

        def back(g):
            self.d += g.T
        out._back = back
        return out

    def __matmul__(self, o):
        o = o if isinstance(o, Node) else Node(o)
        out = Node(self.v @ o.v, (self, o), name="matmul")

        def back(g):
            a, b = self.v, o.v
            self.d += _unbroadcast(g @ b.T if b.ndim > 1 else np.outer(g, b), a.shape)
            o.d += _unbroadcast(a.T @ g if a.ndim > 1 else np.outer(a, g), b.shape)
        out._back = back
        return out

    def __getitem__(self, idx):
        out = Node(self.v[idx], (self,), name="index")

        def back(g):
            np.add.at(self.d, idx, g)
        out._back = back
        return out


def stack(nodes, axis=0):
    out = Node(np.stack([n.v for n in nodes], axis=axis), tuple(nodes), name="stack")

    def back(g):
        for i, n in enumerate(nodes):
            n.d += np.take(g, i, axis=axis).reshape(n.v.shape)
    out._back = back
    return out


def logsumexp(x: Node, axis=None):
    """Stable LSE.  The max shift is a constant and cancels exactly."""
    m = np.max(x.v, axis=axis, keepdims=True)
    z = (x - Node(m)).exp().sum(axis=axis, keepdims=True).log() + Node(m)
    return z.reshape(()) if axis is not None and x.v.ndim == 1 else z


def softmin(x: Node, beta: float, axis=None):
    """Differentiable minimum: -LSE(-beta*x)/beta  ->  min(x) as beta -> inf.
    A chain of transmission is exactly as strong as its weakest carrier."""
    return logsumexp(x * (-beta), axis=axis) * (-1.0 / beta)


def softmax_rows(x: Node):
    m = np.max(x.v, axis=-1, keepdims=True)
    e = (x - Node(m)).exp()
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits: Node, target: int):
    z = logits - Node(np.array(float(np.max(logits.v))))
    return z.exp().sum().log() - z[target]


# ==============================================================================
# SECTION 1 — THE CHAIN FORGE
# ==============================================================================
SOUND, DEFECTIVE, FABRICATED = 0, 1, 2
GRADE_NAMES = ["sahih (sound)", "mu'allal (hidden defect)", "mawdu' (fabricated)"]


class ChainForge:
    """Generates claims that reach the model ONLY through chains of transmitters.

    The model never observes a claim directly -- only reports of it, each having
    passed through two to three carriers, each of whom bent it a little in a way
    peculiar to himself and then re-said it in his own words.

    Ground truth withheld from the model: every transmitter's true tier, true
    noise level, and true distortion operator.  These must be inferred from
    whole-claim grade labels alone, which is precisely the epistemic position of
    a ninth-century critic holding a notebook and no access to the Prophet.
    """

    def __init__(self, n_transmitters=40, D=12, Dm=6, seed=7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.T, self.D, self.Dm, self.Dl = n_transmitters, D, Dm, D - Dm

        # An orthonormal split of R^D: MA'NA (what is said) vs LAFZ (how).
        Qm, _ = np.linalg.qr(rng.normal(size=(D, D)))
        self.Bm, self.Bl = Qm[:, :Dm], Qm[:, Dm:]

        # tier 0 = mutqin (may originate) | 1 = middling (may amplify)
        # tier 2 = matruk (may only serve as a probe)
        self.tier = rng.choice([0, 1, 2], size=self.T, p=[0.40, 0.35, 0.25])
        self.noise = np.array([0.05, 0.18, 0.55])[self.tier] * rng.uniform(0.7, 1.3, self.T)

        # Personal distortion: transmission is identity plus a rank-1 personal tic.
        amp = np.array([0.04, 0.12, 0.30])[self.tier]
        self.U0 = rng.normal(size=(self.T, D, 1)) * amp[:, None, None]
        self.V0 = rng.normal(size=(self.T, D, 1)) / np.sqrt(D)

        # Each transmitter's idiolect: a fixed vector confined to the LAFZ subspace.
        self.lafz = self.Bl @ rng.normal(size=(self.Dl, self.T)) * 0.9
        self.by_tier = {k: np.where(self.tier == k)[0] for k in (0, 1, 2)}

    def _carry(self, m, chain):
        """Push a source vector down a chain, accumulating each carrier's damage.
        The wording that survives at the end is the LAST carrier's, not the source's."""
        h = m.copy()
        for t in chain:
            h = h + (self.U0[t] @ (self.V0[t].T @ h))
            h = h + self.rng.normal(size=self.D) * self.noise[t]
        return h + self.lafz[:, chain[-1]]

    def _chain(self, tiers, L):
        pool = np.concatenate([self.by_tier[k] for k in tiers if len(self.by_tier[k])])
        return list(self.rng.choice(pool, size=min(L, len(pool)), replace=False))

    def make_claim(self, label):
        rng = self.rng
        m = self.Bm @ rng.normal(size=self.Dm)
        routes = []
        if label == SOUND:
            for _ in range(int(rng.integers(3, 6))):
                routes.append(self._chain((0,) if rng.random() < 0.7 else (0, 1),
                                          int(rng.integers(2, 4))))
            contents = [m] * len(routes)
        elif label == DEFECTIVE:
            # A genuine tier-1 core, PLUS low-tier routes carrying a different
            # content.  The flaw is not in any single report: it is the shape of
            # the disagreement between routes that ought to have agreed.
            for _ in range(int(rng.integers(2, 4))):
                routes.append(self._chain((0,), int(rng.integers(2, 4))))
            n_core = len(routes)
            for _ in range(int(rng.integers(2, 4))):
                routes.append(self._chain((1, 2), int(rng.integers(2, 4))))
            drift = self.Bm @ rng.normal(size=self.Dm) * 1.6
            contents = [m] * n_core + [m + drift] * (len(routes) - n_core)
        else:  # FABRICATED: nothing upright anywhere in the lattice
            for _ in range(int(rng.integers(2, 6))):
                routes.append(self._chain((2,), int(rng.integers(2, 4))))
            contents = [m + self.Bm @ rng.normal(size=self.Dm) * 0.9 for _ in routes]
        return {"routes": routes,
                "x": np.array([self._carry(c, ch) for c, ch in zip(contents, routes)]),
                "last": np.array([ch[-1] for ch in routes]),
                "y": label}

    def corpus(self, n):
        labels = self.rng.choice([SOUND, DEFECTIVE, FABRICATED], size=n,
                                 p=[0.40, 0.35, 0.25])
        return [self.make_claim(int(l)) for l in labels]


# ==============================================================================
# SECTION 2 — THE MODEL
# ==============================================================================
class RoleGatedAttestationLattice:
    """Warrant is TYPED BY ROLE, not summed by weight."""

    N_FEATS = 7

    def __init__(self, T, D, Dm=6, K=5, Q=1, seed=0):
        rng = np.random.default_rng(seed)
        self.T, self.D, self.Dm, self.Dl, self.Q = T, D, Dm, D - Dm, Q
        n = lambda *s: Node(rng.normal(size=s) * 0.25)
        self.p = {
            "Emb":   n(T, K),                                   # who each carrier is
            "Wrank": n(K, 3),                                   # -> his PERMISSION
            "Wprec": n(K, 1),                                   # -> his precision (dabt)
            "Ud":    Node(rng.normal(size=(T, D, Q)) * 0.05),   # his personal tic ...
            "Vd":    Node(rng.normal(size=(T, D, Q)) * 0.05),   # ... rank-Q, undoable
            "Pm":    n(D, Dm),                                  # ma'na projection
            "Pl":    n(D, D - Dm),                                # lafz projection
            "Wlafz": n(D - Dm, T),                              # "whose wording is this?"
            "Wcls":  n(self.N_FEATS, 3),
            "bcls":  Node(np.zeros(3)),
            "w2":    Node(np.array(0.0)),      # softplus -> amplification strength
            "w3":    Node(np.array(0.0)),      # softplus -> impugnment strength
            "ltau":  Node(np.array(-0.5)),     # softplus -> Muqaddima gate knee
        }
        self.beta = 6.0          # sharpness of the weakest-link minimum
        self.kappa, self.mid = 5.0, 0.35

    def params(self):
        return self.p

    # ---- THE MUQADDIMA GATE -------------------------------------------
    @staticmethod
    def muqaddima_gate(A1, tau):
        """g(A1) = A1^2/(A1^2 + tau).

        Smooth, monotone increasing, bounded in [0,1) -- and EXACTLY zero when
        A1 is exactly zero, not merely small.  Everything that may only amplify
        or only impugn is multiplied by this.  A mutaba'a therefore cannot lift
        a claim that has no asl beneath it, however many mutaba'at there are.
        This is the preface, compiled."""
        a2 = A1 * A1
        return a2 / (a2 + tau)

    def warrant(self, A1, A2, D3, w2, w3, tau):
        g = self.muqaddima_gate(A1, tau)
        return A1 + g * (w2 * A2) - g * (w3 * D3), g

    def _transmitters(self):
        E = self.p["Emb"]
        tiers = softmax_rows(E @ self.p["Wrank"])              # (T,3) permissions
        prec = (E @ self.p["Wprec"]).reshape(self.T).softplus() + 0.05
        return tiers, prec

    # ---- forward on one claim ------------------------------------------
    def forward(self, claim, tiers, prec):
        Ud, Vd = self.p["Ud"], self.p["Vd"]
        X = claim["x"]
        us, vs, rhos, g1s, g2s, g3s = [], [], [], [], [], []

        for r, chain in enumerate(claim["routes"]):
            # (a) Walk the chain BACKWARDS, undoing each carrier's tic in the
            #     reverse of the order it was applied.  C_t = I - U_t V_t^T is
            #     the first-order inverse of that carrier's transmission map,
            #     and shares its parameters -- to model a man is to be able to
            #     subtract him.
            h = Node(X[r])
            for t in reversed(chain):
                h = h - (Ud[t] @ (Vd[t].T @ h.reshape(self.D, 1))).reshape(self.D)
            h = h.tanh()
            us.append((h.reshape(1, self.D) @ self.p["Pm"]).reshape(self.Dm))
            # (b) Wording is read off the RECEIVED form, uncorrected: the lafz
            #     genuinely belongs to the last man, and is not an error.
            vs.append((Node(X[r]).reshape(1, self.D) @ self.p["Pl"]).reshape(self.Dl))

            # (c) Serial evidence aggregates by MINIMUM.
            rhos.append(softmin(stack([prec[t] for t in chain]), self.beta).reshape(()))

            # (d) A chain's permission is its weakest carrier's permission.
            #     P(all may originate) and P(all at least middling), as products.
            g1 = stack([tiers[t][0] for t in chain]).log().sum().exp()
            g12 = stack([tiers[t][0] + tiers[t][1] for t in chain]).log().sum().exp()
            g1s.append(g1)
            g2s.append(g12 - g1)          # may amplify but not originate
            g3s.append(1.0 - g12)         # may only probe

        R = len(us)
        # (e) CONSENSUS IS FORMED BY TIER ONE ALONE.  Tier-2 routes are scored
        #     against a centre they had no vote in choosing.  This is the usul /
        #     mutaba'at ordering, made arithmetic.
        a = [g1s[i] * rhos[i] for i in range(R)]
        asum = sum(a[1:], a[0]) + 1e-6
        mhat = sum([a[i] * us[i] for i in range(1, R)], a[0] * us[0]) / asum
        mn = ((mhat * mhat).sum() + 1e-8) ** 0.5

        aligns = []
        for i in range(R):
            un = ((us[i] * us[i]).sum() + 1e-8) ** 0.5
            cos = (us[i] * mhat).sum() / (un * mn)
            aligns.append(((cos - self.mid) * self.kappa).sigmoid())

        A1 = sum([g1s[i] * rhos[i] * aligns[i] for i in range(1, R)],
                 g1s[0] * rhos[0] * aligns[0])
        A2 = sum([g2s[i] * rhos[i] * aligns[i] for i in range(1, R)],
                 g2s[0] * rhos[0] * aligns[0])
        D3 = sum([g3s[i] * rhos[i] * (1.0 - aligns[i]) for i in range(1, R)],
                 g3s[0] * rhos[0] * (1.0 - aligns[0]))

        w2, w3 = self.p["w2"].softplus(), self.p["w3"].softplus()
        tau = self.p["ltau"].softplus() + 1e-3
        W, g = self.warrant(A1, A2, D3, w2, w3, tau)

        spread = sum([a[i] * (1.0 - aligns[i]) for i in range(1, R)],
                     a[0] * (1.0 - aligns[0])) / asum
        feats = stack([A1, A2, D3, g, W, spread,
                       Node(np.array(R / 6.0))]).reshape(1, self.N_FEATS)
        return {
            "feats": feats,
            "logits": (feats @ self.p["Wcls"]).reshape(3) + self.p["bcls"],
            "lafz": [(vs[i].reshape(1, self.Dl) @ self.p["Wlafz"]).reshape(self.T)
                     for i in range(R)],
            "u": us, "A1": A1, "A2": A2, "D3": D3, "W": W, "gate": g,
        }


# ==============================================================================
# SECTION 3 — LOSS AND TRAINING
# ==============================================================================
def claim_loss(M, claim, tiers, prec, w_lafz=0.45, w_dec=0.12):
    """Grade CE + the wa'l-lafz auxiliary + a ma'na/lafz decorrelation penalty.

    The decorrelation term penalises how much of a report's MEANING projection
    is explained by the identity of the man who last spoke it.  Pressed from
    both sides -- identity pulled INTO the wording subspace by the auxiliary
    head, pushed OUT of the meaning subspace by this penalty -- the split
    becomes the one that licenses transmission by meaning."""
    o = M.forward(claim, tiers, prec)
    L = cross_entropy(o["logits"], claim["y"])
    for r, ll in enumerate(o["lafz"]):
        L = L + w_lafz * cross_entropy(ll, int(claim["last"][r]))
    R = len(o["u"])
    if R > 1:
        ubar = sum(o["u"][1:], o["u"][0]) * (1.0 / R)
        dec = sum([((o["u"][i] - ubar) * (o["u"][i] - ubar)).sum() for i in range(1, R)],
                  ((o["u"][0] - ubar) * (o["u"][0] - ubar)).sum())
        L = L + w_dec * dec * (1.0 / R)
    return L, o


class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
        self.P, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.S = {k: [np.zeros_like(v.v), np.zeros_like(v.v)] for k, v in params.items()}
        self.t = 0

    def zero_grad(self):
        for v in self.P.values():
            v.d = np.zeros_like(v.v)

    def step(self, only=None):
        self.t += 1
        for k, v in self.P.items():
            if only is not None and k not in only:
                continue
            m, s = self.S[k]
            m[...] = self.b1 * m + (1 - self.b1) * v.d
            s[...] = self.b2 * s + (1 - self.b2) * v.d * v.d
            mh = m / (1 - self.b1 ** self.t)
            sh = s / (1 - self.b2 ** self.t)
            v.v -= self.lr * mh / (np.sqrt(sh) + self.eps)


def evaluate(M, data):
    tiers, prec = M._transmitters()
    ok = 0
    per = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for c in data:
        pred = int(np.argmax(M.forward(c, tiers, prec)["logits"].v))
        ok += pred == c["y"]
        per[c["y"]][1] += 1
        per[c["y"]][0] += pred == c["y"]
    return ok / len(data), {k: (v[0] / max(1, v[1]), v[1]) for k, v in per.items()}


def train(M, train_set, test_set, epochs=14, batch=30, lr=0.02, seed=0, log=print):
    opt = Adam(M.params(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    hist = []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(train_set))
        tot = 0.0
        for b in range(0, len(train_set), batch):
            chunk = [train_set[i] for i in idx[b:b + batch]]
            tiers, prec = M._transmitters()
            L = Node(np.array(0.0))
            for c in chunk:
                li, _ = claim_loss(M, c, tiers, prec)
                L = L + li
            L = L * (1.0 / len(chunk))
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += L.item()
        acc, _ = evaluate(M, test_set)
        hist.append((ep, tot / max(1, len(train_set) // batch), acc))
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            log(f"    epoch {ep:2d}/{epochs}   loss {hist[-1][1]:7.4f}"
                f"   test acc {acc:6.3f}   [{time.time()-t0:5.1f}s]")
    return hist


# ==============================================================================
# SECTION 4 — SELF-TESTS
# ==============================================================================
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}" + (f"   {detail}" if detail else ""))
    return ok


def t1_gradient_check():
    """MANDATORY: analytic gradients vs central finite differences."""
    print("\nT1  FINITE-DIFFERENCE GRADIENT CHECK")
    forge = ChainForge(n_transmitters=8, D=6, Dm=3, seed=3)
    data = forge.corpus(6)
    M = RoleGatedAttestationLattice(forge.T, forge.D, forge.Dm, K=3, Q=1, seed=2)
    P = M.params()

    def total():
        tiers, prec = M._transmitters()
        L = Node(np.array(0.0))
        for c in data:
            li, _ = claim_loss(M, c, tiers, prec)
            L = L + li
        return L

    L = total()
    for v in P.values():
        v.d = np.zeros_like(v.v)
    L.backward()
    analytic = {k: v.d.copy() for k, v in P.items()}

    rng = np.random.default_rng(11)
    eps, worst, worst_k = 1e-6, 0.0, ""
    for k, v in P.items():
        flat = v.v.reshape(-1)
        picks = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in picks:
            o = flat[i]
            flat[i] = o + eps
            lp = total().item()
            flat[i] = o - eps
            lm = total().item()
            flat[i] = o
            num = (lp - lm) / (2 * eps)
            ana = analytic[k].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_k = rel, f"{k}[{i}]"
        print(f"      {k:<7} max rel err so far {worst:.3e}")
    return check("gradient check (all 13 parameter tensors)", worst < 1e-5,
                 f"worst rel err {worst:.3e} at {worst_k}")


def t2_muqaddima_gate():
    """Volume of tier-2 evidence cannot originate warrant.  Ever."""
    print("\nT2  THE MUQADDIMA GATE — corroboration cannot originate")
    M = RoleGatedAttestationLattice(8, 6, 3, seed=0)
    tau = Node(np.array(0.5))
    z = Node(np.array(0.0))
    vals = []
    for mag in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        W, g = M.warrant(z, Node(np.array(mag)), Node(np.array(0.0)),
                         Node(np.array(3.0)), Node(np.array(3.0)), tau)
        vals.append(abs(W.item()))
        print(f"      A1 = 0, A2 = {mag:>10.0e}  ->  warrant = {W.item():.1f}")
    exact = max(vals) == 0.0
    # ... and with even a whisper of tier-1 attestation, the same A2 does lift it.
    Wl, _ = M.warrant(Node(np.array(0.05)), Node(np.array(10.0)), Node(np.array(0.0)),
                      Node(np.array(3.0)), Node(np.array(3.0)), tau)
    W0, _ = M.warrant(Node(np.array(0.05)), Node(np.array(0.0)), Node(np.array(0.0)),
                      Node(np.array(3.0)), Node(np.array(3.0)), tau)
    print(f"      A1 = 0.05, A2 = 0    ->  warrant = {W0.item():.6f}")
    print(f"      A1 = 0.05, A2 = 10   ->  warrant = {Wl.item():.6f}  (lifted)")
    return check("no quantity of tier-2 evidence originates warrant",
                 exact and Wl.item() > W0.item(),
                 "warrant is exactly 0.0 up to A2 = 1e6")


def t3_volume_attack(M, forge, data):
    """The decisive comparison.  Take the SAME measured evidence from real
    held-out claims and aggregate it two ways:

        control (additive)  W = A1 + w2*A2 - w3*D3        <- every modern system
        RGAL (role-gated)   W = A1 + g(A1)*(w2*A2 - w3*D3)

    Then flood each claim with corroboration -- multiply its tier-2 mass by up
    to 64x, as a forger with a printing press or a synthetic-data pipeline
    would.  The control is capturable: enough corroboration lifts fabricated
    claims past sound ones.  The gate makes the same attack a no-op."""
    print("\nT3  VOLUME ATTACK — additive control vs role-gated lattice")
    tiers, prec = M._transmitters()
    w2 = float(M.p["w2"].softplus().v)
    w3 = float(M.p["w3"].softplus().v)
    tau = float(M.p["ltau"].softplus().v) + 1e-3

    ev = {0: [], 2: []}
    for c in data:
        if c["y"] in ev:
            o = M.forward(c, tiers, prec)
            ev[c["y"]].append((o["A1"].item(), o["A2"].item(), o["D3"].item()))

    # One "manufactured corroboration" = a fully-aligned route through a
    # narrator the model rates squarely tier-2, contributing DELTA to A2.
    # This is what a forger buys, and what a synthetic-data pipeline produces
    # for free: unlimited upright-looking agreement with no new source behind it.
    DELTA = 1.0

    def w_ctrl(A1, A2, D3, k):
        return A1 + w2 * (A2 + k * DELTA) - w3 * D3

    def w_rgal(A1, A2, D3, k):
        g = A1 * A1 / (A1 * A1 + tau)
        return A1 + g * (w2 * (A2 + k * DELTA)) - g * (w3 * D3)

    print(f"      learned amplification w2 = {w2:.3f}, impugnment w3 = {w3:.3f}, "
          f"gate knee tau = {tau:.3f}")
    print("      forged    | additive control          | role-gated lattice")
    print("      routes    | sound   forged   captured | sound   forged   captured")
    cap_ctrl = cap_rgal = 0
    for k in (0, 8, 64, 512, 4096):
        row = []
        for fn in (w_ctrl, w_rgal):
            s = np.mean([fn(*e, 0) for e in ev[0]])          # sound, never flooded
            f = [fn(*e, k) for e in ev[2]]                   # forged, flooded
            captured = int(np.sum(np.array(f) >= s))
            row.append((s, float(np.mean(f)), captured))
        cap_ctrl = max(cap_ctrl, row[0][2])
        cap_rgal = max(cap_rgal, row[1][2])
        print(f"      +{k:<5d}   | {row[0][0]:6.2f} {row[0][1]:8.2f} {row[0][2]:9d} "
              f"| {row[1][0]:6.2f} {row[1][1]:8.2f} {row[1][2]:9d}")
    n = len(ev[2])
    print(f"      ({n} fabricated claims; 'captured' = forged warrant reaches the")
    print("       mean warrant of genuinely attested claims)")
    return check("volume attack captures the additive control, not the lattice",
                 cap_ctrl > 0 and cap_rgal == 0,
                 f"control captured {cap_ctrl}/{n} at scale; lattice {cap_rgal}/{n} at any scale")


def t4_detachable_interpretation(M, forge, data):
    """Muslim shipped the corpus; al-Nawawi supplied the headings 400 years on.
    The attested layer must survive a total change of interpretive scheme."""
    print("\nT4  DETACHABLE INTERPRETATION LAYER")
    tiers, prec = M._transmitters()
    trunk_before = np.array([[M.forward(c, tiers, prec)[k].item()
                              for k in ("A1", "A2", "D3", "gate", "W")] for c in data])
    feats = [M.forward(c, tiers, prec)["feats"].v.copy() for c in data]

    accs = []
    # Two genuinely different, genuinely learnable readings of the same frozen
    # attestation: one filing by how densely a claim is attested, one by grade.
    schemes = (("scheme A: 4 books, filed by attestation density", 4,
                np.array([min(3, len(c["routes"]) - 2) for c in data])),
               ("scheme B: 3 chapters, filed by juristic grade", 3,
                np.array([c["y"] for c in data])))
    for scheme, n_cls, labels in schemes:
        rng = np.random.default_rng(9)
        Wh = Node(rng.normal(size=(M.N_FEATS, n_cls)) * 0.2)
        bh = Node(np.zeros(n_cls))
        opt = Adam({"Wh": Wh, "bh": bh}, lr=0.05)
        for _ in range(40):
            L = Node(np.array(0.0))
            for f, y in zip(feats, labels):
                lg = (Node(f) @ Wh).reshape(n_cls) + bh
                L = L + cross_entropy(lg, int(y))
            L = L * (1.0 / len(feats))
            opt.zero_grad()
            L.backward()
            opt.step()
        pred = [int(np.argmax(((Node(f) @ Wh).reshape(n_cls) + bh).v)) for f in feats]
        acc = float(np.mean(np.array(pred) == labels))
        accs.append(acc)
        print(f"      trained head on {scheme:<32} acc {acc:.3f}")

    tiers, prec = M._transmitters()
    trunk_after = np.array([[M.forward(c, tiers, prec)[k].item()
                             for k in ("A1", "A2", "D3", "gate", "W")] for c in data])
    identical = np.array_equal(trunk_before, trunk_after)
    print(f"      attestation outputs bit-identical after both: {identical}")
    return check("interpretation is re-trainable without touching attestation",
                 identical, "5 attestation quantities x %d claims, exact match" % len(data))


def t5_learning(hist):
    print("\nT5  LEARNING")
    first, last = hist[0][2], max(h[2] for h in hist)
    print(f"      test accuracy {first:.3f} (epoch 1)  ->  {last:.3f} (best)")
    return check("model learns well above the 40% majority-class baseline",
                 last > 0.80 and last > first + 0.15, f"best {last:.3f} vs base 0.400")


def t6_tier_discovery(M, forge):
    """The model was never told who was reliable.  Did it work it out?"""
    print("\nT6  TIER DISCOVERY — narrator criticism learned from grades alone")
    tiers, _ = M._transmitters()
    p1 = tiers.v[:, 0]
    means = {}
    for k, nm in ((0, "mutqin  (truly may originate)"),
                  (1, "middling(truly may amplify) "),
                  (2, "matruk  (truly probe-only)  ")):
        m = forge.tier == k
        means[k] = float(p1[m].mean())
        print(f"      {nm}  n={m.sum():2d}   learned P(may originate) = {means[k]:.3f}")
    corr = float(np.corrcoef(p1, -forge.tier)[0, 1])
    print(f"      correlation with withheld ground truth: {corr:+.3f}")
    print("      note: the model separates the EXCLUDED tier sharply while the")
    print("      tier-1/tier-2 boundary stays soft -- which is where the historical")
    print("      disputes actually sat.  Nobody argued about the abandoned.")
    return check("excluded tier identified without supervision",
                 means[2] < 0.25 and means[0] > 0.6 and corr > 0.5,
                 f"matruk {means[2]:.3f} vs mutqin {means[0]:.3f}")


def t7_lafz_disentanglement(M, forge, data):
    """Identity should live in the wording subspace, not the meaning subspace."""
    print("\nT7  MA'NA / LAFZ DISENTANGLEMENT")
    tiers, prec = M._transmitters()
    U, V, Y = [], [], []
    for c in data:
        o = M.forward(c, tiers, prec)
        for r in range(len(o["u"])):
            U.append(o["u"][r].v)
            V.append((c["x"][r] @ M.p["Pl"].v))
            Y.append(int(c["last"][r]))
    U, V, Y = np.array(U), np.array(V), np.array(Y)

    def probe_acc(Z):
        """Closed-form ridge probe: can transmitter identity be read off Z?"""
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        oh = np.eye(forge.T)[Y]
        Wp = np.linalg.solve(Zb.T @ Zb + 1e-2 * np.eye(Zb.shape[1]), Zb.T @ oh)
        return float(np.mean(np.argmax(Zb @ Wp, 1) == Y))

    chance = 1.0 / forge.T
    a_l, a_m = probe_acc(V), probe_acc(U)
    print(f"      transmitter identity from LAFZ subspace : {a_l:.3f}")
    print(f"      transmitter identity from MA'NA subspace: {a_m:.3f}   (chance {chance:.3f})")
    return check("wording carries the speaker; meaning is scrubbed of him",
                 a_l > a_m * 1.5 and a_l > 0.2, f"{a_l:.3f} vs {a_m:.3f}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print(" ROLE-GATED ATTESTATION LATTICE — Figure 0209")
    print(" Muslim ibn al-Hajjaj al-Naysaburi (c. 821-875 CE)")
    print(" 'Warrant is typed by role, not summed by weight.'")
    print("=" * 78)

    t1_gradient_check()
    t2_muqaddima_gate()

    print("\nBUILDING CORPUS")
    forge = ChainForge(n_transmitters=40, D=12, Dm=6, seed=7)
    tr, te = forge.corpus(360), forge.corpus(180)
    print(f"  {len(tr)} training claims / {len(te)} held-out, {forge.T} transmitters")
    print(f"  routes reach the model only through chains of length 2-3;")
    print(f"  true tiers, noise and distortion operators are WITHHELD.")

    print("\nTRAINING")
    M = RoleGatedAttestationLattice(forge.T, forge.D, forge.Dm, K=5, Q=1, seed=1)
    hist = train(M, tr, te, epochs=14, batch=30, lr=0.02, seed=0)

    acc, per = evaluate(M, te)
    print(f"\n  final held-out accuracy {acc:.3f}   (majority-class baseline 0.400)")
    for k in (0, 1, 2):
        print(f"    {GRADE_NAMES[k]:<26} {per[k][0]:.3f}  (n={per[k][1]})")

    t5_learning(hist)
    t6_tier_discovery(M, forge)
    t3_volume_attack(M, forge, te)
    t7_lafz_disentanglement(M, forge, te[:60])
    t4_detachable_interpretation(M, forge, te[:40])

    print("\nWARRANT BY TRUE GRADE (held-out)")
    tiers, prec = M._transmitters()
    for k in (0, 1, 2):
        sub = [M.forward(c, tiers, prec) for c in te if c["y"] == k]
        print(f"  {GRADE_NAMES[k]:<26} A1={np.mean([o['A1'].item() for o in sub]):6.3f}"
              f"  A2={np.mean([o['A2'].item() for o in sub]):6.3f}"
              f"  D3={np.mean([o['D3'].item() for o in sub]):6.3f}"
              f"  gate={np.mean([o['gate'].item() for o in sub]):5.3f}"
              f"  W={np.mean([o['W'].item() for o in sub]):6.3f}")

    print("\n" + "=" * 78)
    ok = sum(1 for _, r in RESULTS if r)
    for nm, r in RESULTS:
        print(f"  [{'PASS' if r else 'FAIL'}]  {nm}")
    print(f"\n  {ok}/{len(RESULTS)} tests passed")
    print("=" * 78)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

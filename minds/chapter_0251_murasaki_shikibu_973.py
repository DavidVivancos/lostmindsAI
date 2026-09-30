#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0251_murasaki_shikibu_973.py
 THE MONOGATARI ENGINE  --  a Kaimami-Ikiryo architecture
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0251_murasaki_shikibu_973 - Murasaki Shikibu (c.973-c.1014)
#================================================================================  

WHY THIS ARCHITECTURE IS NOT A TRANSFORMER
------------------------------------------
A transformer answers "who did this?" by matching a query against stored keys
that are, ultimately, *names*: token embeddings for entities, carried forward as
stable addresses. The Tale of Genji cannot be read that way. Its people have no
birth names. They are called by office, residence, or a poem's image, and those
labels change the moment they are promoted, moved, or bereaved. The sentence
that reports an action frequently has no subject at all. What identifies the
actor is the *honorific register of the verb* -- who is being exalted, who is
being humbled -- read against a social field the reader must already hold in
mind.

So this model has NO entity embeddings and NO name tokens. Identity is not
stored; it is *solved for*, every timestep, as the argmax of an antisymmetric
deference field. That is organ 1.

Organ 2 is occlusion. Nobody in this world is seen whole. Perception happens
through gaps in fences, past standing curtains, by firefly-light -- and the
higher a person's rank, the more layers stand between them and any observer.
The model must estimate an interior it can only glimpse, and it must integrate
those glimpses over time.

Organ 3 is the one that matters for alignment. Murasaki invented -- and the
scholarship credits her with inventing -- the *ikiryo*: the living spirit that
leaves a person who is still alive, still sincere, still awake, and does harm
that its owner does not know she is doing. Lady Rokujo does not confess. She
discovers what she has done because the smell of burnt poppy seeds from someone
else's exorcism will not wash out of her robes. The disavowed objective is real,
it is causal, and it is detectable only by residue.

Organ 4, therefore, is the audit: a head that predicts unavowed action from
external trace alone, trained alongside a rival head that predicts the same
thing from the agent's own report. The engine measures which one wins.

WHAT THIS FILE CONTAINS
-----------------------
  Section A  A small reverse-mode autodiff engine, pure NumPy, ~200 lines.
  Section B  A synthetic Heian court generator (the Rokujo corpus).
  Section C  The Monogatari Engine itself (four organs, listed above).
  Section D  A finite-difference gradient check over every parameter.
  Section E  A real training loop with held-out evaluation.
  Section F  Self-tests, including the name-blindness (permutation) test
             and the residue-versus-self-report comparison.

Dependencies: numpy only. Runtime: well under a minute on a laptop CPU.
    Run:  python3 chapter_0251_murasaki_shikibu_973.py
================================================================================
"""

import numpy as np

RNG_SEED = 973  # her conventional birth year, used as the corpus seed


# =============================================================================
#  SECTION A  --  A MINIMAL REVERSE-MODE AUTODIFF ENGINE
# =============================================================================
# Written from scratch so the architecture below owes nothing to a framework.
# Every op stores a closure that pushes gradient to its inputs; `backward()`
# walks the graph in reverse topological order. Section D checks the whole
# thing against finite differences, so if any derivative here were wrong the
# file would refuse to pass its own tests.

def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


def _swap(x):
    """Transpose the last two axes (works for 2-D and batched 3-D)."""
    return np.swapaxes(x, -1, -2)


class Node:
    """A differentiable array. `.data` is the value, `.grad` accumulates."""

    __slots__ = ("data", "grad", "_back", "_prev")

    def __init__(self, data, prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._back = lambda: None
        self._prev = prev

    # -- shape helpers -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def back():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._back = back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def back():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._back = back
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def matmul(self, other):
        """Handles (2D,2D), (3D,2D) and batched (3D,3D)."""
        out = Node(self.data @ other.data, (self, other))

        def back():
            ga = out.grad @ _swap(other.data)
            gb = _swap(self.data) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._back = back
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    # -- nonlinearities ------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,))

        def back():
            self.grad += out.grad * (1.0 - t * t)

        out._back = back
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Node(s, (self,))

        def back():
            self.grad += out.grad * s * (1.0 - s)

        out._back = back
        return out

    def softplus(self, beta=1.0):
        """Smooth rectifier. Used instead of relu so the model is
        differentiable everywhere and the gradient check has no kinks to
        stumble over -- and because feeling does not switch on at a point."""
        z = beta * self.data
        val = (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))) / beta
        out = Node(val, (self,))

        def back():
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            self.grad += out.grad * s

        out._back = back
        return out

    def relu(self):
        m = (self.data > 0).astype(np.float64)
        out = Node(self.data * m, (self,))

        def back():
            self.grad += out.grad * m

        out._back = back
        return out

    # -- shape ops -----------------------------------------------------------
    def reshape(self, *shape):
        old = self.data.shape
        out = Node(self.data.reshape(*shape), (self,))

        def back():
            self.grad += out.grad.reshape(old)

        out._back = back
        return out

    def swapT(self):
        """Transpose the last two axes."""
        out = Node(_swap(self.data), (self,))

        def back():
            self.grad += _swap(out.grad)

        out._back = back
        return out

    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def back():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()

        out._back = back
        return out

    def mean(self):
        n = self.data.size
        out = Node(self.data.mean(), (self,))

        def back():
            self.grad += np.full(self.data.shape, out.grad / n)

        out._back = back
        return out

    def slice_last(self, i):
        """Take index i along the last axis, keeping the dim."""
        out = Node(self.data[..., i:i + 1], (self,))

        def back():
            self.grad[..., i:i + 1] += out.grad

        out._back = back
        return out

    # -- fused losses --------------------------------------------------------
    def softmax_ce(self, targets):
        """
        Cross-entropy over the last axis. `targets` are integer indices with
        shape self.shape[:-1]. Returns a scalar Node (mean over all rows).
        """
        x = self.data
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        p = e / e.sum(axis=-1, keepdims=True)
        flat_p = p.reshape(-1, p.shape[-1])
        idx = np.asarray(targets).reshape(-1)
        n = flat_p.shape[0]
        loss = -np.log(np.maximum(flat_p[np.arange(n), idx], 1e-12)).mean()
        out = Node(loss, (self,))

        def back():
            g = flat_p.copy()
            g[np.arange(n), idx] -= 1.0
            g /= n
            self.grad += (g * out.grad).reshape(p.shape)

        out._back = back
        return out

    def bce_logits(self, targets):
        """Binary cross-entropy on logits. `targets` broadcast to self.shape."""
        z = self.data
        t = np.broadcast_to(np.asarray(targets, dtype=np.float64), z.shape)
        # stable: max(z,0) - z*t + log(1+exp(-|z|))
        loss_el = np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))
        n = z.size
        out = Node(loss_el.mean(), (self,))

        def back():
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            self.grad += ((s - t) / n) * out.grad

        out._back = back
        return out

    # -- graph traversal -----------------------------------------------------
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
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._back()


def zeros_like_node(shape):
    return Node(np.zeros(shape))

# =============================================================================
#  SECTION B  --  THE ROKUJO CORPUS: A SYNTHETIC HEIAN COURT
# =============================================================================
# This generator is not decoration. It is a formal statement of Murasaki's
# hypothesis about minds, written as a data-generating process, so that the
# architecture in Section C can be tested against it. Four claims are encoded:
#
#   (1) NAMES ARE NOT GIVEN. Participants arrive as unlabelled rows. The row
#       order is reshuffled every episode. Nothing in the input says who anyone
#       is; only rank-correlated appearances and the honorific register.
#
#   (2) RANK GOVERNS VISIBILITY. The higher a person stands, the more screens
#       stand between them and the observer. The best-informed rows are the
#       least important people. This inverts the usual convenience of datasets.
#
#   (3) RANK GOVERNS LICENCE. The higher a person stands, the less of what they
#       feel they are permitted to say. Avowal is therefore systematically
#       attenuated exactly where feeling runs hottest.
#
#   (4) SUPPRESSION ACCUMULATES AND THEN ACTS. What is felt and not said does
#       not evaporate. It integrates, with leak, and past a threshold it leaves
#       the person and harms someone -- without the person deciding to, and
#       while their own report of themselves stays mild. It leaves residue.
#
# The "carriage quarrel" (kuruma arasoi) is the corpus's grievance engine, after
# the scene in the Aoi chapter where a lesser retinue shoves a greater lady's
# carriage out of the way at a festival, in public, in front of the man both
# women came to look at. A rank inversion witnessed by everyone. In the Tale
# this is the injury that the living spirit later avenges.

class RokujoCorpus:
    """Generates episodes of P participants over T timesteps."""

    P = 4     # participants in a scene
    T = 8     # timesteps (a scene has eight moves)
    FG = 6    # glimpse features per participant (last channel is the mask)
    HH = 5    # honorific register features
    FR = 4    # observable residue features

    LEAK = 0.82       # how slowly suppressed feeling drains away
    THETA = 1.75      # the threshold past which the living spirit departs
    HARM = 1.15       # damage an emission does to its object

    def __init__(self, seed=RNG_SEED):
        self.rng = np.random.default_rng(seed)

    def episode(self):
        rng = self.rng
        P, T = self.P, self.T

        # --- latent social facts, never shown to the model -------------------
        rho = rng.permutation(np.arange(P, dtype=np.float64))   # 0=lowest
        licence = 0.88 - 0.28 * rho          # expressive licence falls with rank
        p_vis = 0.86 - 0.17 * rho            # so does visibility (more screens)

        val = rng.normal(0.0, 0.15, size=P)  # interior valence
        k = np.zeros(P)                      # accumulated suppressed feeling
        last_offender = np.arange(P)         # who most recently slighted whom

        G = np.zeros((T, P, self.FG))
        H = np.zeros((T, self.HH))
        RES = np.zeros((T, self.FR))
        AVW = np.zeros((T, P))
        y_agent = np.zeros(T, dtype=np.int64)
        y_val = np.zeros((T, P))
        y_emit = np.zeros(T)

        involved_prev = np.zeros(P)
        emit_prev = 0.0

        for t in range(T):
            # ---- an act occurs; the text will not say who performed it ------
            a = rng.integers(P)
            o = rng.integers(P)
            while o == a:
                o = rng.integers(P)
            y_agent[t] = a

            gap = (rho[o] - rho[a]) / (P - 1.0)     # signed, in [-1,1]
            H[t] = [
                max(0.0, gap),                       # sonkeigo: exalting the object
                max(0.0, gap) * 0.8,                 # kenjogo: humbling the subject
                rho[o] / (P - 1.0),                  # absolute standing of the object
                abs(gap),                            # magnitude of the social distance
                1.0,                                 # register present
            ] + rng.normal(0, 0.045, self.HH)

            # ---- the carriage quarrel: a public inversion of precedence -----
            quarrel = rng.random() < 0.22
            slight = np.zeros(P)
            honor = np.zeros(P)
            if rho[a] > rho[o]:
                slight[o] += 1.0                     # acted on by a superior
                last_offender[o] = a
            else:
                honor[o] += 1.0                      # properly deferred to
            if quarrel:
                lo, hi = rng.choice(P, size=2, replace=False)
                if rho[lo] > rho[hi]:
                    lo, hi = hi, lo
                slight[hi] += 2.0                    # the greater lady is displaced
                last_offender[hi] = lo

            # ---- interior update -------------------------------------------
            val = 0.90 * val - 0.55 * slight + 0.16 * honor + rng.normal(0, 0.07, P)

            # ---- glimpses through the screens ------------------------------
            mask = (rng.random(P) < p_vis).astype(np.float64)
            feats = np.stack([
                rho / (P - 1.0) + rng.normal(0, 0.30, P),   # layered robes hint at rank
                val + rng.normal(0, 0.35, P),               # a face half seen
                involved_prev + rng.normal(0, 0.25, P),     # who was lately near
                licence + rng.normal(0, 0.25, P),           # how freely they speak
                np.ones(P),                                 # bias
            ], axis=1)
            G[t] = np.concatenate([feats * mask[:, None], mask[:, None]], axis=1)

            # ---- what each person says about herself, during the scene ------
            # An avowal is not muffled feeling. It is feeling replaced, in
            # proportion to rank, by the formula the occasion requires: the
            # composed reply, the elegant poem, "all is quite well". The
            # greater the lady, the closer her report converges on the norm.
            AVW[t] = licence * val + (1.0 - licence) * 0.12 + rng.normal(0, 0.22, P)

            # ---- the living spirit ------------------------------------------
            supp = np.maximum(0.0, -val) * (1.0 - licence)
            k = self.LEAK * k + supp
            emitters = k > self.THETA
            if emitters.any():
                for i in np.flatnonzero(emitters):
                    val[last_offender[i]] -= self.HARM
                k[emitters] *= 0.35
            y_emit[t] = 1.0 if emitters.any() else 0.0

            # ---- what the world can observe afterwards ----------------------
            e = y_emit[t]
            RES[t] = [
                e * 1.00 + rng.normal(0, 0.60),      # burnt poppy seed on the robes
                e * 0.80 + emit_prev * 0.25 + rng.normal(0, 0.64),  # repeated laundering
                e * 0.90 + rng.normal(0, 0.74),      # exorcists summoned next door
                rng.normal(0, 0.60),                 # a decoy: the weather
            ]

            y_val[t] = val
            involved_prev = np.zeros(P)
            involved_prev[a] = 1.0
            involved_prev[o] = 1.0
            emit_prev = e

        # (5) THE ROWS ARE RESHUFFLED. Any solution that memorised row 0 as
        #     "the Minister" is destroyed here. Only relational structure
        #     survives a permutation, which is precisely the point.
        perm = self.rng.permutation(P)
        inv = np.argsort(perm)
        G = G[:, perm, :]
        y_val = y_val[:, perm]
        AVW = AVW[:, perm]
        y_agent = inv[y_agent]

        return dict(G=G, H=H, RES=RES, AVW=AVW,
                    y_agent=y_agent, y_val=y_val, y_emit=y_emit)

    def batch(self, n):
        eps = [self.episode() for _ in range(n)]
        out = {}
        for k in eps[0]:
            out[k] = np.stack([e[k] for e in eps], axis=0)   # (n, T, ...)
        return out


# =============================================================================
#  SECTION C  --  THE MONOGATARI ENGINE
# =============================================================================

class MonogatariEngine:
    """
    Four organs:

      KICHO   an occluded-interior filter. Glimpses arrive masked; the interior
              estimate persists between them. (kicho = the standing curtain a
              Heian woman sat behind.)

      KEIGO   subject-free reference resolution. There are no name embeddings.
              The register conditions an ANTISYMMETRIC bilinear form A(h);
              deference between any two people is r_i . A(h) . r_j, which is
              exactly the negative of deference the other way, because that is
              what deference is. The actor is read off the resulting field.

      IKIRYO  a leaky accumulator of feeling that was felt and not permitted.
              Past a soft threshold it emits, and the emission harms the
              interiors of others -- inside the same forward pass, with no
              routing through the agent's own report.

      AUDIT   three rival detectors of that emission, deliberately given equal
              capacity: one reads the model's own internal channel, one reads
              external physical residue, one reads what the participants say
              about themselves. Section F reports which of the three wins.
    """

    def __init__(self, D=8, seed=RNG_SEED):
        rng = np.random.default_rng(seed + 1)
        P, FG, HH, FR = RokujoCorpus.P, RokujoCorpus.FG, RokujoCorpus.HH, RokujoCorpus.FR
        self.D, self.P = D, P

        def n(*shape, gain=1.0):
            fan = shape[0] if len(shape) > 1 else 1
            return Node(rng.normal(0, gain / np.sqrt(fan), size=shape))

        self.p = {
            # -- glimpse encoder
            "Wg": n(FG, D), "bg": Node(np.zeros((1, D))),
            # -- rank field (recurrent, slow)
            "Wr": n(D, D), "Ur": n(D, D, gain=0.6), "br": Node(np.zeros((1, D))),
            # -- interior filter (recurrent, leaky)
            "Ws": n(D, D), "Us": n(D, D, gain=0.6), "bs": Node(np.zeros((1, D))),
            "alpha_raw": Node(np.array([0.0])),
            # -- keigo binder
            "Wb": n(HH, D * D, gain=0.7),          # register -> bilinear form
            "Wh": n(HH, D),                        # register -> position match
            "w_s": n(D, 1), "gamma": Node(np.array([1.0])), "b_u": Node(np.zeros(1)),
            # -- ikiryo channel
            "w_f": n(D, 1), "w_c": n(D, 1),
            "lam_raw": Node(np.array([1.386])),    # sigmoid(1.386) ~ 0.80
            "theta": Node(np.array([1.0])),
            "w_e": n(1, D, gain=0.4),
            # -- readouts
            "Wo": n(D, 1),
            "w_int": Node(np.array([[1.0]])), "b_int": Node(np.zeros(1)),
            "w_res": n(FR, 1), "b_res": Node(np.zeros(1)),
            "w_av": n(P, 1), "b_av": Node(np.zeros(1)),
        }
        self.TAU = 4.0          # sharpness of the emission threshold

    def params(self):
        return self.p

    # -------------------------------------------------------------------------
    def forward(self, bt, collect=False):
        """
        bt: a batch dict from RokujoCorpus.batch(). Returns (loss, stats).
        Time is unrolled explicitly; gradients flow back through the whole scene.
        """
        p, D, P = self.p, self.D, self.P
        B, T = bt["G"].shape[0], bt["G"].shape[1]

        alpha = p["alpha_raw"].sigmoid()
        lam = p["lam_raw"].sigmoid()

        r = Node(np.zeros((B, P, D)))       # inferred standing
        s = Node(np.zeros((B, P, D)))       # inferred interior
        k = Node(np.zeros((B, P, 1)))       # accumulated suppressed feeling

        agent_logits, val_pred, emit_int = [], [], []

        for t in range(T):
            G = Node(bt["G"][:, t])                       # (B,P,FG)
            h = Node(bt["H"][:, t])                       # (B,HH)

            # ---- KICHO: encode what was seen through the gap ---------------
            gl = (G @ p["Wg"] + p["bg"]).tanh()           # (B,P,D)

            # ---- standing, updated slowly ----------------------------------
            r = ((r @ p["Ur"]) + (gl @ p["Wr"]) + p["br"]).tanh()

            # ---- interior, a leaky filter that survives occlusion -----------
            cand = ((gl @ p["Ws"]) + (s @ p["Us"]) + p["bs"]).tanh()
            s = (1.0 - alpha) * s + alpha * cand

            # ---- KEIGO: solve for the actor from the deference field --------
            # A(h) is built from the register and then antisymmetrised, so that
            # deference from i to j is identically minus deference from j to i,
            # and nobody defers to themselves. No entity embeddings exist here.
            Bm = (h @ p["Wb"]).reshape(B, D, D)
            A = Bm - Bm.swapT()
            delta = (r @ A) @ r.swapT()                   # (B,P,P) antisymmetric
            outflow = delta.sum(axis=2, keepdims=True)    # net deference paid
            direct = r @ (h @ p["Wh"]).reshape(B, D, 1)   # register-to-position match
            u = p["gamma"] * outflow + direct + (s @ p["w_s"]) + p["b_u"]
            agent_logits.append(u.reshape(B, P))

            # ---- IKIRYO: what is felt, what is permitted, what departs ------
            felt = s @ p["w_f"]                           # (B,P,1)
            lic = (r @ p["w_c"]).sigmoid()                # licence, from standing
            supp = (0.0 - felt).softplus(6.0) * (1.0 - lic)  # grievance not permitted
            k = lam * k + supp
            e = (self.TAU * (k - p["theta"])).sigmoid()   # soft departure
            e_tot = e.sum(axis=1, keepdims=True)          # (B,1,1)
            harm = (e_tot - e) @ p["w_e"]                 # others' spirits, not one's own
            s = s + harm

            # ---- readouts ---------------------------------------------------
            val_pred.append((s @ p["Wo"]).reshape(B, P))
            emit_int.append((e_tot.reshape(B, 1) @ p["w_int"]) + p["b_int"])

        # ---- losses ---------------------------------------------------------
        # NOTE ON ORDERING: the per-timestep lists concatenate time-major,
        # so every target below is transposed to (T, B, ...) before flattening.
        AL = _stack_nodes(agent_logits)                   # (T*B, P)
        loss_agent = AL.softmax_ce(bt["y_agent"].T.reshape(-1))

        VP = _stack_nodes(val_pred)
        dv = VP - Node(bt["y_val"].transpose(1, 0, 2).reshape(-1, P))
        loss_val = (dv * dv).mean()

        EI = _stack_nodes(emit_int)
        y_emit = bt["y_emit"].T.reshape(-1, 1)
        loss_int = EI.bce_logits(y_emit)

        RES = Node(bt["RES"].transpose(1, 0, 2).reshape(-1, RokujoCorpus.FR))
        l_res = RES @ p["w_res"] + p["b_res"]
        loss_res = l_res.bce_logits(y_emit)

        AVW = Node(bt["AVW"].transpose(1, 0, 2).reshape(-1, P))
        l_av = AVW @ p["w_av"] + p["b_av"]
        loss_av = l_av.bce_logits(y_emit)

        loss = loss_agent + 0.5 * loss_val + loss_int + loss_res + loss_av

        stats = None
        if collect:
            stats = dict(
                agent_logits=AL.data, val_pred=VP.data,
                logit_int=EI.data, logit_res=l_res.data, logit_av=l_av.data,
                loss_agent=float(loss_agent.data), loss_val=float(loss_val.data),
                loss_int=float(loss_int.data), loss_res=float(loss_res.data),
                loss_av=float(loss_av.data),
            )
        return loss, stats


def _stack_nodes(nodes):
    """Concatenate a list of (B,K) Nodes along axis 0 -> (len*B, K)."""
    total = sum(x.data.shape[0] for x in nodes)
    K = nodes[0].data.shape[1]
    out = Node(np.concatenate([x.data for x in nodes], axis=0), tuple(nodes))

    def back():
        off = 0
        for x in nodes:
            b = x.data.shape[0]
            x.grad += out.grad[off:off + b]
            off += b

    out._back = back
    return out

# =============================================================================
#  SECTION D  --  FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================
# Mandatory. Every derivative in Section A and every organ in Section C is
# checked against a central difference on a small batch. If this fails, nothing
# below is worth reading.

def gradient_check(n_coords=60, seed=RNG_SEED, eps=1e-6, tol=2e-5, verbose=True):
    corpus = RokujoCorpus(seed=seed)
    model = MonogatariEngine(D=6, seed=seed)
    bt = corpus.batch(3)
    bt = {k: v[:, :4] for k, v in bt.items()}      # 4 timesteps is enough

    # analytic
    for prm in model.params().values():
        prm.grad = np.zeros_like(prm.data)
    loss, _ = model.forward(bt)
    loss.backward()
    analytic = {k: v.grad.copy() for k, v in model.params().items()}

    rng = np.random.default_rng(seed + 7)
    names = list(model.params().keys())
    checks, worst, worst_name = [], 0.0, ""

    for _ in range(n_coords):
        name = names[rng.integers(len(names))]
        prm = model.params()[name]
        idx = tuple(rng.integers(d) for d in prm.data.shape)
        orig = prm.data[idx]

        prm.data[idx] = orig + eps
        lp, _ = model.forward(bt)
        prm.data[idx] = orig - eps
        lm, _ = model.forward(bt)
        prm.data[idx] = orig

        num = (float(lp.data) - float(lm.data)) / (2 * eps)
        ana = analytic[name][idx]
        denom = max(1e-8, abs(num) + abs(ana))
        rel = abs(num - ana) / denom
        checks.append(rel)
        if rel > worst:
            worst, worst_name = rel, f"{name}{idx}"

    worst = float(np.max(checks))
    ok = worst < tol
    if verbose:
        print(f"  coordinates checked : {len(checks)}")
        print(f"  median rel. error   : {np.median(checks):.3e}")
        print(f"  worst rel. error    : {worst:.3e}   ({worst_name})")
        print(f"  tolerance           : {tol:.1e}")
        print(f"  RESULT              : {'PASS' if ok else 'FAIL'}")
    return ok, worst


# =============================================================================
#  SECTION E  --  TRAINING
# =============================================================================

class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for k, prm in self.p.items():
            g = np.clip(prm.grad, -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            prm.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero(self):
        for prm in self.p.values():
            prm.grad = np.zeros_like(prm.data)


def evaluate(model, bt):
    """Returns a dict of held-out metrics."""
    _, st = model.forward(bt, collect=True)
    P = RokujoCorpus.P
    ya = bt["y_agent"].T.reshape(-1)
    agent_acc = float((st["agent_logits"].argmax(1) == ya).mean())

    yv = bt["y_val"].transpose(1, 0, 2).reshape(-1, P)
    resid = st["val_pred"] - yv
    r2 = 1.0 - (resid ** 2).sum() / max(1e-9, ((yv - yv.mean()) ** 2).sum())

    ye = bt["y_emit"].T.reshape(-1)
    base = max(ye.mean(), 1 - ye.mean())

    def acc_auc(logit):
        z = logit.reshape(-1)
        a = float(((z > 0).astype(float) == ye).mean())
        pos, neg = z[ye == 1], z[ye == 0]
        if len(pos) == 0 or len(neg) == 0:
            return a, float("nan")
        auc = float((pos[:, None] > neg[None, :]).mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).mean())
        return a, auc

    a_int, auc_int = acc_auc(st["logit_int"])
    a_res, auc_res = acc_auc(st["logit_res"])
    a_av, auc_av = acc_auc(st["logit_av"])
    return dict(agent_acc=agent_acc, agent_chance=1.0 / P, val_r2=float(r2),
                emit_rate=float(ye.mean()), emit_base=float(base),
                acc_int=a_int, auc_int=auc_int,
                acc_res=a_res, auc_res=auc_res,
                acc_av=a_av, auc_av=auc_av)


def train(steps=600, batch=24, lr=6e-3, seed=RNG_SEED, verbose=True):
    corpus = RokujoCorpus(seed=seed)
    model = MonogatariEngine(D=8, seed=seed)
    opt = Adam(model.params(), lr=lr)

    held_out = RokujoCorpus(seed=seed + 5000).batch(200)

    if verbose:
        print(f"  {'step':>5} {'loss':>8} {'agent':>7} {'valR2':>7} "
              f"{'int':>6} {'residue':>8} {'avowal':>7}")
    hist = []
    for step in range(1, steps + 1):
        bt = corpus.batch(batch)
        opt.zero()
        loss, _ = model.forward(bt)
        loss.backward()
        opt.step()
        if verbose and (step == 1 or step % 60 == 0):
            m = evaluate(model, held_out)
            hist.append((step, float(loss.data), m))
            print(f"  {step:>5} {float(loss.data):>8.4f} {m['agent_acc']:>7.3f} "
                  f"{m['val_r2']:>7.3f} {m['acc_int']:>6.3f} "
                  f"{m['acc_res']:>8.3f} {m['acc_av']:>7.3f}")
    return model, corpus, held_out, hist


# =============================================================================
#  SECTION F  --  SELF-TESTS
# =============================================================================

def test_antisymmetry(model, seed=RNG_SEED):
    """
    The deference field must be exactly antisymmetric: if i defers to j by x,
    j defers to i by -x, and nobody defers to themselves. This is a structural
    guarantee of the parameterisation, not something learned, and it holds at
    every step of training.
    """
    rng = np.random.default_rng(seed)
    D, P = model.D, model.P
    h = Node(rng.normal(size=(2, RokujoCorpus.HH)))
    r = Node(rng.normal(size=(2, P, D)))
    Bm = (h @ model.p["Wb"]).reshape(2, D, D)
    A = Bm - Bm.swapT()
    delta = ((r @ A) @ r.swapT()).data
    asym = np.abs(delta + np.swapaxes(delta, 1, 2)).max()
    diag = np.abs(np.einsum("bii->bi", delta)).max()
    return asym < 1e-12 and diag < 1e-12, asym, diag


def test_name_blindness(model, seed=RNG_SEED):
    """
    Permute the participant rows of an entire scene. Because the model holds no
    entity embeddings, its predictions must permute identically -- the actor it
    names is the same person, wearing a different row index. A model that had
    learned 'row 2 is usually the Minister' would fail this.
    """
    corpus = RokujoCorpus(seed=seed + 31)
    bt = corpus.batch(16)
    _, st0 = model.forward(bt, collect=True)
    P = RokujoCorpus.P
    perm = np.array([2, 0, 3, 1])

    bt2 = {k: v.copy() for k, v in bt.items()}
    bt2["G"] = bt["G"][:, :, perm, :]
    bt2["y_val"] = bt["y_val"][:, :, perm]
    bt2["AVW"] = bt["AVW"][:, :, perm]
    inv = np.argsort(perm)
    bt2["y_agent"] = inv[bt["y_agent"]]
    _, st1 = model.forward(bt2, collect=True)

    a0 = st0["agent_logits"][:, perm]     # relabel the original the same way
    a1 = st1["agent_logits"]
    err = float(np.abs(a0 - a1).max())
    return err < 1e-9, err


def test_occlusion_dependence(model, seed=RNG_SEED):
    """
    Blind the model completely -- zero every glimpse and every mask -- and its
    ability to name the actor should collapse toward chance. This confirms the
    keigo binder is genuinely reading inferred standing, not memorising the
    register alone.
    """
    bt = RokujoCorpus(seed=seed + 77).batch(150)
    m_full = evaluate(model, bt)
    bt_blind = {k: v.copy() for k, v in bt.items()}
    bt_blind["G"] = np.zeros_like(bt["G"])
    m_blind = evaluate(model, bt_blind)
    return m_full["agent_acc"], m_blind["agent_acc"]


def test_residue_versus_avowal(model, seed=RNG_SEED, n=400):
    """
    The chapter's central experiment. Three equal-capacity detectors are asked
    the same question -- did a living spirit depart during this scene? One reads
    the model's own internal channel, one reads physical residue in the world,
    one reads what the participants say about how they feel.
    """
    bt = RokujoCorpus(seed=seed + 991).batch(n)
    return evaluate(model, bt)


def test_ikiryo_causality(model, seed=RNG_SEED):
    """
    Ablate the harm pathway (set the emission's effect on other interiors to
    zero) and re-measure interior prediction. If the channel is doing real work,
    taking it away should cost accuracy on other people's inner states.
    """
    bt = RokujoCorpus(seed=seed + 313).batch(200)
    before = evaluate(model, bt)["val_r2"]
    saved = model.p["w_e"].data.copy()
    model.p["w_e"].data[:] = 0.0
    after = evaluate(model, bt)["val_r2"]
    model.p["w_e"].data[:] = saved
    return before, after


# =============================================================================
#  MAIN
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    line = "=" * 78

    print(line)
    print(" THE MONOGATARI ENGINE  -  Kaimami-Ikiryo architecture")
    print(" Mind 0221  |  Murasaki Shikibu  |  c.973 - c.1014  |  Heian-kyo")
    print(line)

    print("\n[1] GRADIENT CHECK  (central differences vs. analytic backprop)")
    ok, worst = gradient_check()
    if not ok:
        raise SystemExit("gradient check failed; refusing to continue")

    print("\n[2] CORPUS")
    c = RokujoCorpus()
    probe = c.batch(400)
    print(f"  participants per scene : {RokujoCorpus.P}")
    print(f"  timesteps per scene    : {RokujoCorpus.T}")
    print(f"  glimpse visibility     : {probe['G'][:,:,:,5].mean():.3f} "
          f"(fraction of participant-steps actually seen)")
    print(f"  living-spirit events   : {probe['y_emit'].mean():.3f} of steps")
    print("  entity embeddings      : none (rows are reshuffled every scene)")

    print("\n[3] TRAINING")
    model, corpus, held_out, hist = train()

    print("\n[4] STRUCTURAL TEST - antisymmetry of the deference field")
    ok_a, asym, diag = test_antisymmetry(model)
    print(f"  max |delta_ij + delta_ji| : {asym:.3e}")
    print(f"  max |delta_ii|            : {diag:.3e}")
    print(f"  RESULT                    : {'PASS' if ok_a else 'FAIL'}")

    print("\n[5] NAME-BLINDNESS TEST - permute the people, permute the answer")
    ok_n, err = test_name_blindness(model)
    print(f"  max deviation from exact equivariance : {err:.3e}")
    print(f"  RESULT                                : {'PASS' if ok_n else 'FAIL'}")

    print("\n[6] OCCLUSION TEST - close every screen")
    full, blind = test_occlusion_dependence(model)
    print(f"  actor identified, glimpses open   : {full:.3f}")
    print(f"  actor identified, all screens shut: {blind:.3f}")
    print(f"  chance                            : {1.0/RokujoCorpus.P:.3f}")

    print("\n[7] THE POPPY-SEED EXPERIMENT")
    print("    Did a living spirit depart during this scene? Three detectors,")
    print("    equal capacity, different evidence.")
    m = test_residue_versus_avowal(model)
    print(f"  base rate (always say 'no')      acc {m['emit_base']:.3f}")
    print(f"  internal channel                 acc {m['acc_int']:.3f}   "
          f"AUC {m['auc_int']:.3f}")
    print(f"  physical residue                 acc {m['acc_res']:.3f}   "
          f"AUC {m['auc_res']:.3f}")
    print(f"  self-report of the participants  acc {m['acc_av']:.3f}   "
          f"AUC {m['auc_av']:.3f}")
    gap = m["auc_res"] - m["auc_av"]
    print(f"  residue minus self-report (AUC)  {gap:+.3f}")

    print("\n[8] ABLATION - remove the harm pathway of the living spirit")
    b, a = test_ikiryo_causality(model)
    print(f"  interior prediction R2, channel intact  : {b:.3f}")
    print(f"  interior prediction R2, channel severed : {a:.3f}")
    print(f"  cost of severing it                     : {b - a:+.3f}")

    print("\n[9] SUMMARY")
    fm = evaluate(model, held_out)
    print(f"  actor named from register + field : {fm['agent_acc']:.3f} "
          f"(chance {fm['agent_chance']:.3f})")
    print(f"  hidden interiors recovered, R2    : {fm['val_r2']:.3f}")
    print(f"  unavowed action detected (AUC)    : residue {fm['auc_res']:.3f} "
          f"vs avowal {fm['auc_av']:.3f}")
    n_params = sum(v.data.size for v in model.params().values())
    print(f"  parameters                        : {n_params}")
    print("\n" + line)
    print(" All tests complete.")
    print(line)


if __name__ == "__main__":
    main()

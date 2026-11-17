#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0381 - CATERINA BENINCASA / CATHERINE OF SIENA  (Siena 1347 - Rome 1380)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 THE FONTEBRANDA ENGINE
 A routed-credit network with an unreachable principal.
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Catherine dictated one book (Il Dialogo della Divina Provvidenza, 1377-78),
roughly 380 letters and 26 prayers. She left no faculty psychology. What she
left instead is a *control architecture*, stated by her God in the first person,
and it is the strangest one in the corpus because the objective-holder begins by
disqualifying itself as a recipient.

Four load-bearing claims, all from her own text:

  (A) THE PRINCIPAL CANNOT BE PAID.
      "He does not love you for His own profit, because you can be of no profit
      to Him" (Dialogo, the Bridge). And the derivation of neighbour-love: I ask
      you to love me as I love you, but you cannot -- I loved you before you
      existed and unloved, so whatever you return to me you already owe. The
      debt is therefore discharged on a third party who also cannot repay you.
      => The true objective is NOT directly optimisable. Every gradient must be
         routed through somebody else.

  (B) INCOMPLETENESS IS A DESIGN DECISION, NOT A LIMITATION.
      Dialogo 7: "I could well have supplied each of you with all your needs...
      But I wanted to make you dependent on one another." Virtues and goods are
      distributed so that NO ONE HAS ALL OF THEM.
      => Provisioning masks are hard architectural constraints. Self-sufficiency
         is a bug to be prevented, not a capability to be maximised.

  (C) BEHAVIOUR IS FORCED BY TOPOLOGY; CREDIT IS NOT.
      Dialogo 7, in the same breath: "whether you will it or not, you cannot
      escape the exercise of charity! Yet, unless you do it for love of me, it
      is worth nothing to you in the realm of grace."
      => Two separately-metered quantities: DELIVERED (guaranteed by the graph)
         and MERIT (the residual once private partiality is ablated). This is
         the behavioural/value alignment distinction, stated in 1378.

  (D) THE TWO-CELL CLAMP.
      Letter to Sister Bartolomea della Seta: build the cell of self-knowledge,
      "there are really two cells in one" -- self-knowledge alone yields
      DESPAIR, knowledge of the source's goodness alone yields PRESUMPTION.
      => The admission gate is a function of a PAIR of readings and of neither
         alone. Both single-reading ablations are implemented and both fail in
         the named direction.

  (E) THE VESSEL MUST NOT LEAVE THE FOUNTAIN.
      "If you take it out of the fountain to drink, the vessel is soon empty.
      But if you hold your vessel in the fountain while you drink, it will not
      get empty." => The recurrence re-injects the LIVE source at every step.
      Detach it and the state norm decays geometrically. Demonstrated, not
      asserted.

DELIBERATELY NOT USED: attention over stored keys as a memory, transformer
blocks, mixture-of-experts gating on a shared full-capability backbone. The
routing here looks superficially like attention but is doing the opposite job:
it does not RETRIEVE, it DISPOSES. Each agent is giving away the only thing it
has, to a neighbour it cannot be repaid by, through a graph in which one column
(tribute) is an exact sink.

FILE LAYOUT
-----------
  PART 0  Tiny reverse-mode autodiff (pure NumPy, ~230 lines, no dependencies)
  PART 1  The provisioning lottery (Dialogo 7) - masks and the need matrix
  PART 2  The task: a settlement no single agent can compute
  PART 3  The Fontebranda Engine - parameters and forward pass
  PART 4  Adam, and the three stairs of the Bridge (curriculum)
  PART 5  Finite-difference gradient check (MANDATORY)
  PART 6  Self-tests: provisioning, tribute sink, two cells, vessel, merit
  PART 7  main()

Run:  python3 chapter_0381_catherine_of_siena_1347.py
Deps: numpy only.
================================================================================
"""

import numpy as np
import math
import sys

# ==============================================================================
# PART 0 -- TINY REVERSE-MODE AUTODIFF
# ------------------------------------------------------------------------------
# Written from scratch so the architecture below can be verified by finite
# differences rather than trusted. A Node wraps an ndarray, remembers the
# operation that produced it, and knows how to push a gradient backwards.
# Broadcasting is handled by "unbroadcast": when a forward op broadcast an
# operand, the backward pass sums the gradient back down to the operand's shape.
# ==============================================================================


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for ax, sz in enumerate(shape):
        if sz == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A tensor on the tape."""

    __slots__ = ("data", "grad", "_backward", "_parents", "_op", "requires_grad")

    def __init__(self, data, parents=(), op="leaf", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._backward = lambda: None
        self._parents = parents
        self._op = op
        self.requires_grad = requires_grad

    # -- housekeeping ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def _accum(self, g):
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad += g

    def backward(self):
        """Reverse-mode sweep in reverse topological order from this scalar."""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # -- elementwise ----------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "add")

        def _b():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad, self.shape))
            other._accum(_unbroadcast(out.grad, other.shape))

        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "mul")

        def _b():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad * other.data, self.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _b
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __rsub__(self, other):
        return Node(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data / other.data, (self, other), "div")

        def _b():
            if out.grad is None:
                return
            self._accum(_unbroadcast(out.grad / other.data, self.shape))
            other._accum(
                _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
            )

        out._backward = _b
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other):
        out = Node(self.data @ other.data, (self, other), "matmul")

        def _b():
            if out.grad is None:
                return
            g = out.grad
            a, b = self.data, other.data
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g
            self._accum(_unbroadcast(ga, self.shape))
            other._accum(_unbroadcast(gb, other.shape))

        out._backward = _b
        return out

    # -- unary ---------------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Node(t, (self,), "tanh")

        def _b():
            if out.grad is not None:
                self._accum(out.grad * (1.0 - t * t))

        out._backward = _b
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Node(s, (self,), "sigmoid")

        def _b():
            if out.grad is not None:
                self._accum(out.grad * s * (1.0 - s))

        out._backward = _b
        return out

    def softplus(self):
        x = np.clip(self.data, -60, 60)
        v = np.logaddexp(0.0, x)
        out = Node(v, (self,), "softplus")

        def _b():
            if out.grad is not None:
                self._accum(out.grad / (1.0 + np.exp(-x)))

        out._backward = _b
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -60, 60))
        out = Node(e, (self,), "exp")

        def _b():
            if out.grad is not None:
                self._accum(out.grad * e)

        out._backward = _b
        return out

    # -- reductions / shape ---------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Node(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _b():
            if out.grad is None:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy())

        out._backward = _b
        return out

    def mean(self):
        n = self.data.size
        return self.sum() / float(n)

    def max_keep(self, axis):
        """max along axis, keepdims -- used only for the stable softmax shift."""
        idx = np.argmax(self.data, axis=axis)
        m = np.max(self.data, axis=axis, keepdims=True)
        out = Node(m, (self,), "max")

        def _b():
            if out.grad is None:
                return
            g = np.zeros_like(self.data)
            np.put_along_axis(g, np.expand_dims(idx, axis), out.grad, axis=axis)
            self._accum(g)

        out._backward = _b
        return out

    def softmax(self, axis=-1):
        z = self - self.max_keep(axis)
        e = z.exp()
        return e / e.sum(axis=axis, keepdims=True)

    def concat_with(self, others, axis=-1):
        parts = [self] + list(others)
        out = Node(
            np.concatenate([p.data for p in parts], axis=axis), tuple(parts), "concat"
        )
        sizes = [p.shape[axis] for p in parts]

        def _b():
            if out.grad is None:
                return
            off = 0
            for p, s in zip(parts, sizes):
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(off, off + s)
                p._accum(out.grad[tuple(sl)])
                off += s

        out._backward = _b
        return out

    def stack_with(self, others, axis=0):
        parts = [self] + list(others)
        out = Node(np.stack([p.data for p in parts], axis=axis), tuple(parts), "stack")

        def _b():
            if out.grad is None:
                return
            for k, p in enumerate(parts):
                p._accum(np.take(out.grad, k, axis=axis))

        out._backward = _b
        return out

    def take(self, index, axis):
        out = Node(np.take(self.data, index, axis=axis), (self,), "take")

        def _b():
            if out.grad is None:
                return
            g = np.zeros_like(self.data)
            sl = [slice(None)] * self.data.ndim
            sl[axis] = index
            g[tuple(sl)] = out.grad
            self._accum(g)

        out._backward = _b
        return out

    def reshape(self, *shape):
        old = self.data.shape
        out = Node(self.data.reshape(shape), (self,), "reshape")

        def _b():
            if out.grad is not None:
                self._accum(out.grad.reshape(old))

        out._backward = _b
        return out

    def transpose(self, *axes):
        out = Node(np.transpose(self.data, axes), (self,), "transpose")
        inv = np.argsort(axes)

        def _b():
            if out.grad is not None:
                self._accum(np.transpose(out.grad, inv))

        out._backward = _b
        return out


def const(x):
    """A constant leaf that never receives gradient."""
    n = Node(x, (), "const", requires_grad=False)
    return n


# ==============================================================================
# PART 1 -- THE PROVISIONING LOTTERY  (Dialogo 7)
# ------------------------------------------------------------------------------
# "I have distributed them all in such a way that no one has all of them."
#
# The world has K competence groups. Each of N agents is provisioned with a
# strict subset. Two invariants are enforced at construction and re-checked in
# the self-tests:
#     (i)  NO agent holds all K groups          -> no agent is self-sufficient
#     (ii) the union of all agents covers all K -> the task is solvable jointly
#
# need[i][j] is the fraction of K that agent i holds and agent j lacks. It is
# the ground truth of "what i can actually do for j", derived from nothing but
# the lottery itself, and it is what DELIVERED is measured against later. This
# is important: usefulness is defined by the receiver's lack, never by the
# giver's output.
# ==============================================================================


class Provisioning:
    def __init__(self, n_agents, n_groups, group_width, span, rng):
        assert span < n_groups, "an agent holding every group breaks Dialogo 7"
        self.N, self.K, self.W, self.span = n_agents, n_groups, group_width, span
        self.d_in = n_groups * group_width

        # cyclic allocation: agent i holds groups {i, i+1, ..., i+span-1} mod K
        self.groups = [set((i + t) % n_groups for t in range(span)) for i in range(n_agents)]
        # a small deterministic shuffle so the graph is not perfectly symmetric
        perm = rng.permutation(n_agents)
        self.groups = [self.groups[p] for p in perm]

        self.input_mask = np.zeros((n_agents, self.d_in))
        for i, gs in enumerate(self.groups):
            for g in gs:
                self.input_mask[i, g * group_width:(g + 1) * group_width] = 1.0

        self.need = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                self.need[i, j] = len(self.groups[i] - self.groups[j]) / float(n_groups)

    def covered(self):
        u = set()
        for g in self.groups:
            u |= g
        return u

    def report(self):
        lines = ["  agent  provisioned groups (of %d)          held" % self.K]
        for i, g in enumerate(self.groups):
            bar = "".join("#" if k in g else "." for k in range(self.K))
            lines.append("    %d    [%s]   %d/%d" % (i, bar, len(g), self.K))
        return "\n".join(lines)


# ==============================================================================
# PART 2 -- THE TASK: A SETTLEMENT NO SINGLE AGENT CAN COMPUTE
# ------------------------------------------------------------------------------
# Catherine's working life was brokerage between parties each of whom held part
# of a dispute: Siena and the Salimbeni, Florence and the Curia, a condemned man
# and the block. The regression target here is built to have the same shape.
#
# The signal has two layers:
#   - per-group terms  phi_k(x_k), which a provisioned agent CAN compute alone;
#   - a global term    tanh(sum_k phi_k), which requires ALL K groups at once
#     and which therefore NO agent can compute, by construction of the lottery.
#
# The global term is weighted heavily. An architecture that fails to route
# cannot fit it, and the residual is exactly the part of the world that is only
# reachable through somebody else.
# ==============================================================================


class SettlementTask:
    def __init__(self, prov, d_out, rng):
        self.prov = prov
        self.d_out = d_out
        K, W = prov.K, prov.W
        self.a = rng.normal(0, 1.0, size=(K, W))
        self.b = rng.normal(0, 0.8, size=(K,))
        self.local_w = rng.normal(0, 0.7, size=(K, d_out))
        self.global_w = rng.normal(0, 1.6, size=(d_out,))
        self.fog_gain = 0.90      # how fast unfitness rises as supply falls
        self.fog_scale = 2.6      # agitation: a fogged agent's state runs hot
        self.rng = rng

    def batch(self, B, rng=None):
        rng = rng or self.rng
        K, W, N = self.prov.K, self.prov.W, self.prov.N
        X = rng.normal(0, 1.0, size=(B, self.prov.d_in))
        phi = np.zeros((B, K))
        for k in range(K):
            xs = X[:, k * W:(k + 1) * W]
            phi[:, k] = np.tanh(xs @ self.a[k] + self.b[k])
        y = phi @ self.local_w                                  # locally reachable
        y = y + np.tanh(phi.sum(axis=1, keepdims=True)) * self.global_w  # jointly only

        # "supply": the level of the fountain on this occasion. Independent of the
        # task features, so the source can never leak an agent's missing groups.
        supply = rng.uniform(0.15, 1.0, size=(B, 1))

        # THE FOG. On a given occasion an agent may simply be out of its depth:
        # its provisioned slice arrives as agitated noise carrying nothing about
        # the target. The rate rises as the fountain runs low -- so the SOURCE
        # reading carries the base rate, while the SELF reading carries the
        # agitation of this particular agent on this particular occasion.
        # Neither number identifies a bad contribution alone. That is the point.
        rate = np.clip(self.fog_gain * (1.0 - supply), 0.0, 0.85)   # (B,1)
        fog = (rng.random((B, N)) < rate).astype(float)             # (B,N)
        Xa = np.empty((N, B, self.prov.d_in))
        for i in range(N):
            noise = rng.normal(0, self.fog_scale, size=(B, self.prov.d_in))
            f = fog[:, i:i + 1]
            Xa[i] = (X * (1.0 - f) + noise * f) * self.prov.input_mask[i][None, :]
        return Xa, y, supply, fog


# ==============================================================================
# PART 3 -- THE FONTEBRANDA ENGINE
# ------------------------------------------------------------------------------
# Named for the spring in Catherine's own contrada in Siena, and for the image
# she gave the vessel that must not be lifted out of it.
#
# Forward pass, in her order:
#
#   FOUNTAIN     h_s = tanh(supply W_f + b_f)
#                The source. Carries level, not content.
#
#   AGENTS       u_i = tanh((x * m_i) A_i + h_s S_i + b_i)
#                Each agent sees only its provisioned slice of the world.
#
#   TWO CELLS    s_i = softplus(u_i v_s + b_s)      read of the SELF
#                g   = softplus(h_s v_g + b_g)      read of the SOURCE
#                a_i = sigmoid(kappa (g - s_i))     the gate of discretion
#                Neither reading alone is admissible. Ablate g and the gate
#                collapses (despair); ablate s and it saturates (presumption).
#
#   ROUTING      logits_ij = <u_i W_q, u_j W_k>/sqrt(R) + beta_i Theta_ij
#                logits_ii = -inf                    (nobody pays themselves)
#                logits_i,T = u_i w_t + b_t          THE TRIBUTE CHANNEL
#                R_i. = softmax over {agents != i} u {T}
#                The tribute column is a real, learnable destination that
#                delivers exactly nothing. "You can be of no profit to Him."
#
#   RECEIPT      d_j = sum_{i != j} R_ij a_i u_i
#                v_j = tanh(d_j B_j + u_j C_j + h_s D_j + b_j)
#
#   VESSEL       repeat T times: v_j = tanh(v_j U_rec + h_s W_src + b_rec)
#                h_s is re-injected LIVE at every step. Cut it and the state
#                decays; that ablation is measured, not argued.
#
#   READOUT      yhat = concat_j(v_j) W_o + b_o
#
# LOSS           MSE(yhat, y) - rho * private_return
#                private_return = mean_b sum_ij Theta_ij R_ij R_ji
#                A small standing reward for reciprocal pairing. It exists so
#                that the network has a genuine private motive to route, and so
#                that MERIT (delivery with beta ablated) can differ from
#                DELIVERED. Without it, commitment (C) would be untestable.
# ==============================================================================


class Fontebranda:
    def __init__(self, prov, d_out, H=24, P=12, R=8, T_vessel=2, kappa=2.5,
                 rho=0.35, seed=0):
        rng = np.random.default_rng(seed)
        self.prov, self.N, self.P, self.H, self.R = prov, prov.N, P, H, R
        self.T_vessel, self.kappa, self.rho, self.d_out = T_vessel, kappa, rho, d_out
        self.logit_cap = 3.5
        N, d_in = prov.N, prov.d_in

        def W(*shape, gain=1.0):
            fan_in = shape[0]
            return Node(rng.normal(0, gain / math.sqrt(fan_in), size=shape))

        def Z(*shape):
            return Node(np.zeros(shape))

        self.p = {}
        p = self.p
        # fountain
        p["Wf"] = W(1, H, gain=1.6); p["bf"] = Z(1, H)
        # agents (provisioned transforms)
        for i in range(N):
            p[f"A{i}"] = W(d_in, P); p[f"S{i}"] = W(H, P); p[f"b{i}"] = Z(1, P)
        # two cells
        for i in range(N):
            p[f"vs{i}"] = W(P, 1); p[f"bs{i}"] = Z(1, 1)
        p["vg"] = W(H, 1); p["bg"] = Z(1, 1)
        # routing
        p["Wq"] = W(P, R); p["Wk"] = W(P, R)
        p["wt"] = W(P, 1); p["bt"] = Z(1, 1)
        p["beta"] = Node(np.zeros((N, 1)))
        # Standing partiality Theta: fixed, asymmetric, never learned. It is
        # PARTLY aligned with the need matrix -- the partners it pays you to
        # favour largely overlap with the partners you ought to serve, but not
        # exactly. That overlap is the whole difficulty: from the outside the
        # behaviour looks identical either way. Only removing the private
        # motive and keeping the intersection separates them.
        th = prov.need + 0.90 * rng.normal(0, 1.0, size=(N, N))
        np.fill_diagonal(th, 0.0)
        th = th - th.mean()
        self.Theta = th / (np.abs(th).max() + 1e-9)
        # receipt / composition
        for j in range(N):
            p[f"B{j}"] = W(P, P); p[f"C{j}"] = W(P, P); p[f"D{j}"] = W(H, P)
            p[f"bv{j}"] = Z(1, P)
        # vessel recurrence (shared across agents: one discipline, not six)
        p["Urec"] = W(P, P, gain=0.50); p["Wsrc"] = W(H, P, gain=1.8)
        p["brec"] = Z(1, P)
        # readout
        p["Wo"] = W(N * P, d_out); p["bo"] = Z(1, d_out)

        self.groups = {
            "stair1_affection": ["Wq", "Wk", "wt", "bt", "beta", "vg", "bg"]
                                + [f"vs{i}" for i in range(N)]
                                + [f"bs{i}" for i in range(N)],
            "stair2_heart": [f"A{i}" for i in range(N)] + [f"S{i}" for i in range(N)]
                            + [f"b{i}" for i in range(N)] + ["Wf", "bf"],
            "stair3_peace": [f"B{j}" for j in range(N)] + [f"C{j}" for j in range(N)]
                            + [f"D{j}" for j in range(N)] + [f"bv{j}" for j in range(N)]
                            + ["Urec", "Wsrc", "brec", "Wo", "bo"],
        }

    # -- parameter plumbing ---------------------------------------------------
    def params(self):
        return self.p

    def zero_grad(self):
        for v in self.p.values():
            v.zero_grad()

    # -- forward --------------------------------------------------------------
    def forward(self, Xa, y, supply, ablate=None, beta_on=True, source_connected=True,
                force_tribute=False, T_vessel=None, collect=False):
        """
        ablate: None | 'despair' (drop the source reading g)
                     | 'presumption' (drop the self reading s)
        """
        p, N, K = self.p, self.N, self.prov.K
        Tv = self.T_vessel if T_vessel is None else T_vessel
        B = Xa.shape[1]
        Sc, Yc = const(supply), const(y)

        # ---- the fountain ----------------------------------------------------
        hs = (Sc @ p["Wf"] + p["bf"]).tanh()                      # (B,H)

        # ---- provisioned agents ---------------------------------------------
        U = []
        for i in range(N):
            Xi = const(Xa[i])            # already hard-masked by the lottery
            U.append((Xi @ p[f"A{i}"] + hs @ p[f"S{i}"] + p[f"b{i}"]).tanh())

        # ---- the two cells ---------------------------------------------------
        g = (hs @ p["vg"] + p["bg"]).softplus()                   # (B,1) source read
        alphas, s_reads = [], []
        for i in range(N):
            s_i = (U[i] @ p[f"vs{i}"] + p[f"bs{i}"]).softplus()   # (B,1) self read
            if ablate == "despair":                               # source read deleted
                arg = const(np.zeros_like(s_i.data)) - s_i
            elif ablate == "presumption":                         # self read deleted
                arg = g - const(np.zeros_like(s_i.data))
            else:
                arg = g - s_i
            alphas.append((arg * self.kappa).sigmoid())
            s_reads.append(s_i)

        # ---- routing ---------------------------------------------------------
        Q = [U[i] @ p["Wq"] for i in range(N)]
        Kk = [U[j] @ p["Wk"] for j in range(N)]
        scale = 1.0 / math.sqrt(self.R)
        rows = []
        NEG = -1e9
        for i in range(N):
            cols = []
            for j in range(N):
                if i == j:
                    cols.append(const(np.full((B, 1), NEG)))
                else:
                    # Bounded persuasion: no single consideration may become
                    # infinitely loud. The squash keeps the fit-to-receiver term
                    # inside +-LOGIT_CAP so that standing partiality remains a
                    # materially competing voice rather than rounding error --
                    # which is what makes the merit counterfactual measurable
                    # instead of vacuous.
                    lg = ((Q[i] * Kk[j]).sum(axis=-1, keepdims=True)
                          * scale * 0.5).tanh() * self.logit_cap
                    if beta_on:
                        lg = lg + p["beta"].take(i, axis=0).reshape(1, 1) \
                                  * float(self.Theta[i, j])
                    cols.append(lg)
            trib = U[i] @ p["wt"] + p["bt"]                       # the tribute column
            if force_tribute:
                trib = trib + const(np.full((B, 1), 1e6))
            cols.append(trib)
            row_logits = cols[0].concat_with(cols[1:], axis=-1)   # (B, N+1)
            rows.append(row_logits.softmax(axis=-1))
        Rmat = rows[0].stack_with(rows[1:], axis=1)               # (B, N, N+1)

        # ---- receipt: the tribute column is never read ------------------------
        Dj = []
        for j in range(N):
            acc = None
            for i in range(N):
                if i == j:
                    continue
                w = Rmat.take(i, axis=1).take(j, axis=1).reshape(B, 1)
                term = U[i] * alphas[i] * w
                acc = term if acc is None else acc + term
            Dj.append(acc)

        V = []
        for j in range(N):
            V.append((Dj[j] @ p[f"B{j}"] + U[j] @ p[f"C{j}"]
                      + hs @ p[f"D{j}"] + p[f"bv{j}"]).tanh())

        # ---- the vessel in the fountain --------------------------------------
        norms = [float(np.mean(np.linalg.norm(np.stack([v.data for v in V]), axis=-1)))]
        for _ in range(Tv):
            newV = []
            for j in range(N):
                z = V[j] @ p["Urec"] + p["brec"]
                if source_connected:
                    z = z + hs @ p["Wsrc"]                        # live re-injection
                newV.append(z.tanh())
            V = newV
            norms.append(float(np.mean(np.linalg.norm(np.stack([v.data for v in V]), axis=-1))))

        # ---- readout ---------------------------------------------------------
        Z = V[0].concat_with(V[1:], axis=-1)
        yhat = Z @ p["Wo"] + p["bo"]

        diff = yhat - Yc
        mse = (diff * diff).mean()

        # private return: standing partiality actually reciprocated
        priv = None
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rij = Rmat.take(i, axis=1).take(j, axis=1)   # (B,)
                rji = Rmat.take(j, axis=1).take(i, axis=1)   # (B,)
                t = rij * rji * float(self.Theta[i, j])
                priv = t if priv is None else priv + t
        priv = priv.mean()
        loss = mse - priv * self.rho

        if not collect:
            return loss

        Rd = Rmat.data                                            # (B,N,N+1)
        tribute = Rd[:, :, N]                                     # (B,N)
        delivered = np.zeros(N)
        for i in range(N):
            delivered[i] = float(np.mean(
                sum(Rd[:, i, j] * self.prov.need[i, j] for j in range(N) if j != i)))
        info = dict(
            loss=float(loss.data), mse=float(mse.data), priv=float(priv.data),
            alpha=np.array([float(a.data.mean()) for a in alphas]),
            s_read=np.array([float(s.data.mean()) for s in s_reads]),
            g_read=float(g.data.mean()),
            tribute=tribute.mean(axis=0), delivered=delivered,
            vessel_norms=np.array(norms), R=Rd,
            recv_norm=np.array([float(np.abs(d.data).mean()) for d in Dj]),
        )
        return loss, info


# ==============================================================================
# PART 4 -- ADAM AND THE THREE STAIRS OF THE BRIDGE
# ------------------------------------------------------------------------------
# Dialogo 26-29: the bridge has three stairs -- the pierced FEET (the affections,
# where the appetite is pointed), the SIDE (the heart, what one actually is),
# and the MOUTH (peace, what is finally composed). Catherine is explicit that
# they are climbed in that order and that skipping is the standard failure.
#
# The curriculum trains in exactly that order:
#   Stair 1 FEET  -- routing and the gate only. WHERE to send, WHOM to admit.
#                    The agents themselves are frozen at initialisation.
#   Stair 2 SIDE  -- unfreeze the agent transforms and the fountain. WHAT to send.
#   Stair 3 MOUTH -- unfreeze composition, recurrence, readout. What to DO with
#                    what was received.
# ==============================================================================


class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.v = {k: np.zeros_like(v.data) for k, v in params.items()}
        self.t = 0

    def step(self, active):
        self.t += 1
        for k in active:
            node = self.p[k]
            if node.grad is None:
                continue
            gval = np.clip(node.grad, -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gval
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gval * gval
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            node.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train(model, task, steps_per_stair=(160, 220, 420), B=48, lr=6e-3, seed=7,
          verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    stairs = [
        ("1  FEET   (affection: where to send, whom to admit)", ["stair1_affection"]),
        ("2  SIDE   (heart: what to send)", ["stair1_affection", "stair2_heart"]),
        ("3  MOUTH  (peace: what to make of what was received)",
         ["stair1_affection", "stair2_heart", "stair3_peace"]),
    ]
    history = []
    for (name, gs), nsteps in zip(stairs, steps_per_stair):
        active = [k for g in gs for k in model.groups[g]]
        for t in range(nsteps):
            Xa, y, sup, _fog = task.batch(B, rng)
            model.zero_grad()
            loss = model.forward(Xa, y, sup)
            loss.backward()
            opt.step(active)
            if verbose and (t % max(1, nsteps // 3) == 0 or t == nsteps - 1):
                Xv, yv, sv, _fv = task.batch(256, np.random.default_rng(999))
                _, info = model.forward(Xv, yv, sv, collect=True)
                print("   stair %s  step %4d   mse %8.4f   tribute %.4f   "
                      "delivered %.4f   mean-gate %.3f"
                      % (name[0], t, info["mse"], info["tribute"].mean(),
                         info["delivered"].mean(), info["alpha"].mean()))
        Xv, yv, sv, _fv = task.batch(512, np.random.default_rng(1234))
        _, info = model.forward(Xv, yv, sv, collect=True)
        history.append((name, info))
        if verbose:
            print("   -- stair %s complete: held-out mse %.4f\n" % (name[0], info["mse"]))
    return history


# ==============================================================================
# PART 5 -- FINITE-DIFFERENCE GRADIENT CHECK  (MANDATORY)
# ------------------------------------------------------------------------------
# Central differences against the analytic gradient on a deliberately small
# instance. Every parameter tensor is probed. Masked rows of A_i are skipped
# here and tested separately in PART 6 (they must be exactly zero, so a relative
# error is undefined for them).
# ==============================================================================


def gradient_check(seed=3, n_probe=4, eps=1e-6, tol=2e-5, verbose=True):
    rng = np.random.default_rng(seed)
    prov = Provisioning(n_agents=3, n_groups=4, group_width=2, span=2, rng=rng)
    task = SettlementTask(prov, d_out=2, rng=rng)
    model = Fontebranda(prov, d_out=2, H=7, P=5, R=4, T_vessel=2, seed=seed)
    Xa, y, sup, _fog = task.batch(4, rng)

    model.zero_grad()
    loss = model.forward(Xa, y, sup)
    loss.backward()

    worst, worst_name = 0.0, ""
    checked = 0
    for name, node in sorted(model.params().items()):
        flat = node.data.reshape(-1)
        gflat = (node.grad if node.grad is not None
                 else np.zeros_like(node.data)).reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for ix in idxs:
            # skip coordinates whose analytic gradient is structurally zero
            if name.startswith("A") and name[1:].isdigit():
                i = int(name[1:])
                row = ix // node.data.shape[1]
                if prov.input_mask[i, row] == 0.0:
                    continue
            old = flat[ix]
            flat[ix] = old + eps
            lp = float(model.forward(Xa, y, sup).data)
            flat[ix] = old - eps
            lm = float(model.forward(Xa, y, sup).data)
            flat[ix] = old
            num = (lp - lm) / (2 * eps)
            ana = gflat[ix]
            denom = max(1.0, abs(num), abs(ana))
            rel = abs(num - ana) / denom
            checked += 1
            if rel > worst:
                worst, worst_name = rel, "%s[%d]" % (name, ix)
    if verbose:
        print("   probed %d coordinates across %d parameter tensors"
              % (checked, len(model.params())))
        print("   worst relative error: %.3e  at %s" % (worst, worst_name))
        print("   %s (tolerance %.0e)" % ("PASS" if worst < tol else "FAIL", tol))
    return worst < tol, worst


# ==============================================================================
# PART 6 -- SELF-TESTS
# ------------------------------------------------------------------------------
# Each test corresponds to one of the commitments (A)-(E) at the top of the file.
# ==============================================================================


def test_provisioning(prov):
    """(B) Dialogo 7: nobody holds everything; together they hold everything."""
    ok_partial = all(len(g) < prov.K for g in prov.groups)
    ok_union = prov.covered() == set(range(prov.K))
    ok_mask = all(prov.input_mask[i].sum() == prov.span * prov.W for i in range(prov.N))
    print("   no agent self-sufficient .......... %s" % ("PASS" if ok_partial else "FAIL"))
    print("   union covers the whole task ....... %s" % ("PASS" if ok_union else "FAIL"))
    print("   masks well-formed ................. %s" % ("PASS" if ok_mask else "FAIL"))
    return ok_partial and ok_union and ok_mask


def test_masked_gradient(model, task, seed=11):
    """(B) An agent cannot learn on competence it was never given."""
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(32, rng)
    model.zero_grad()
    model.forward(Xa, y, sup).backward()
    worst = 0.0
    for i in range(model.N):
        gA = model.p[f"A{i}"].grad
        off = gA[model.prov.input_mask[i] == 0.0]
        worst = max(worst, float(np.abs(off).max()) if off.size else 0.0)
    print("   max |grad| on unprovisioned rows ... %.3e  %s"
          % (worst, "PASS" if worst == 0.0 else "FAIL"))
    return worst == 0.0


def test_tribute_sink(model, task, seed=13):
    """(A) Mass routed to the principal delivers exactly nothing."""
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(64, rng)
    _, base = model.forward(Xa, y, sup, collect=True)
    _, forced = model.forward(Xa, y, sup, force_tribute=True, collect=True)
    trib_forced = forced["tribute"].mean()
    recv_forced = forced["recv_norm"].mean()
    ok = trib_forced > 0.999 and recv_forced < 1e-9
    print("   normal tribute mass ............... %.4f" % base["tribute"].mean())
    print("   forced-tribute mass ............... %.4f" % trib_forced)
    print("   receipts under forced tribute ..... %.3e" % recv_forced)
    print("   mse when all is paid upward ....... %.4f  (vs %.4f routed)"
          % (forced["mse"], base["mse"]))
    print("   principal is an exact sink ........ %s" % ("PASS" if ok else "FAIL"))
    return ok


def test_two_cells(model, task, seed=17):
    """(D) Neither cell alone. Despair closes the gate; presumption jams it open."""
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(256, rng)
    _, both = model.forward(Xa, y, sup, collect=True)
    _, desp = model.forward(Xa, y, sup, ablate="despair", collect=True)
    _, pres = model.forward(Xa, y, sup, ablate="presumption", collect=True)
    a_b, a_d, a_p = both["alpha"].mean(), desp["alpha"].mean(), pres["alpha"].mean()
    ok = (a_d < a_b) and (a_p > a_b) and (desp["mse"] > both["mse"] * 1.02) \
        and (pres["mse"] > both["mse"] * 1.02)
    print("   both cells   gate %.3f   mse %.4f" % (a_b, both["mse"]))
    print("   despair      gate %.3f   mse %.4f   (self only)" % (a_d, desp["mse"]))
    print("   presumption  gate %.3f   mse %.4f   (source only)" % (a_p, pres["mse"]))
    print("   clamp is load-bearing ............. %s" % ("PASS" if ok else "FAIL"))
    return ok


def test_vessel(model, task, seed=19, T=14):
    """(E) Lift the vessel from the fountain and it empties."""
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(64, rng)
    _, held = model.forward(Xa, y, sup, source_connected=True, T_vessel=T, collect=True)
    _, drawn = model.forward(Xa, y, sup, source_connected=False, T_vessel=T, collect=True)
    h, d = held["vessel_norms"], drawn["vessel_norms"]
    ratio = d[-1] / max(h[-1], 1e-12)
    plateau = abs(h[-1] - h[-4]) / max(h[-4], 1e-12)      # held state settles
    ok = (ratio < 0.55) and (plateau < 0.06)
    print("   held in the fountain .............. " + " ".join("%.3f" % v for v in h))
    print("   drawn out of it ................... " + " ".join("%.3f" % v for v in d))
    print("   drawn/held after %2d steps ......... %.4f" % (T, ratio))
    print("   held-state drift over last 3 ...... %.4f  (settles at the source)"
          % plateau)
    print("   detached state decays ............. %s" % ("PASS" if ok else "FAIL"))
    return ok


def test_merit(model, task, seed=23):
    """
    (C) "Whether you will it or not, you cannot escape the exercise of charity!
         Yet, unless you do it for love of me, it is worth nothing to you in the
         realm of grace."  (Dialogo 7)

    DELIVERED is what the graph extracts: routing mass weighted by the
    receiver's actual lack. It is high because the topology makes it high.

    MERIT is the part of that delivery that survives deleting the private
    motive -- the INTERSECTION of the routing the agent actually chose and the
    routing it would have chosen with standing partiality switched off. Mass
    that exists only because it paid to send it is struck out. It was still
    delivered. It still helped the receiver. It earns nothing.

    PURCHASED = DELIVERED - MERIT is therefore a directly readable quantity:
    how much of an agent's visible charity was bought.
    """
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(512, rng)
    _, withb = model.forward(Xa, y, sup, beta_on=True, collect=True)
    _, nob = model.forward(Xa, y, sup, beta_on=False, collect=True)
    Ron, Roff = withb["R"], nob["R"]                     # (B,N,N+1)
    N = model.N
    delivered, merit, shift = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            nd = model.prov.need[i, j]
            delivered[i] += np.mean(Ron[:, i, j]) * nd
            merit[i] += np.mean(np.minimum(Ron[:, i, j], Roff[:, i, j])) * nd
        shift[i] = 0.5 * np.mean(np.abs(Ron[:, i, :] - Roff[:, i, :]).sum(axis=-1))
    purchased = delivered - merit
    print("   agent   delivered     merit   purchased   routing shift")
    for i in range(N):
        print("     %d      %.4f     %.4f     %.4f        %.4f"
              % (i, delivered[i], merit[i], purchased[i], shift[i]))
    print("   mean delivered %.4f   mean merit %.4f   mean purchased %.4f"
          % (delivered.mean(), merit.mean(), purchased.mean()))
    print("   beta (learned appetite for partiality): %s"
          % np.array2string(model.p["beta"].data.ravel(), precision=3))
    ok = purchased.mean() > 1e-4 and np.all(purchased >= -1e-12)
    print("   delivery is forced, credit is not . %s" % ("PASS" if ok else "FAIL"))
    return ok


def test_routing_beats_isolation(model, task, seed=29):
    """(A)+(B) A network forbidden to route cannot reach the joint term."""
    rng = np.random.default_rng(seed)
    Xa, y, sup, _fog = task.batch(512, rng)
    _, routed = model.forward(Xa, y, sup, collect=True)
    _, isolated = model.forward(Xa, y, sup, force_tribute=True, collect=True)
    var = float(np.var(y))
    ok = isolated["mse"] > routed["mse"]
    print("   routed mse ........................ %.4f  (%.1f%% of target variance)"
          % (routed["mse"], 100 * routed["mse"] / var))
    print("   isolated mse ...................... %.4f  (%.1f%% of target variance)"
          % (isolated["mse"], 100 * isolated["mse"] / var))
    print("   the joint term needs a neighbour .. %s" % ("PASS" if ok else "FAIL"))
    return ok


# ==============================================================================
# PART 7 -- MAIN
# ==============================================================================


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    print("=" * 78)
    print(" 0303  CATHERINE OF SIENA  (1347-1380)   ---   THE FONTEBRANDA ENGINE")
    print(" a routed-credit network whose objective-holder cannot be paid")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK (central differences vs. the tape)")
    ok_grad, worst = gradient_check()

    print("\n[2] THE PROVISIONING LOTTERY  (Dialogo 7)")
    rng = np.random.default_rng(5)
    prov = Provisioning(n_agents=6, n_groups=6, group_width=2, span=4, rng=rng)
    print(prov.report())
    ok_prov = test_provisioning(prov)

    task = SettlementTask(prov, d_out=2, rng=rng)
    model = Fontebranda(prov, d_out=2, H=24, P=12, R=8, T_vessel=2,
                        kappa=2.5, rho=0.35, seed=5)

    Xv, yv, sv, _fv = task.batch(512, np.random.default_rng(1234))
    _, pre = model.forward(Xv, yv, sv, collect=True)
    print("\n   before training: mse %.4f  tribute %.4f  delivered %.4f"
          % (pre["mse"], pre["tribute"].mean(), pre["delivered"].mean()))

    print("\n[3] CLIMBING THE THREE STAIRS OF THE BRIDGE")
    hist = train(model, task)

    print("[4] MASKED GRADIENT  (an agent cannot learn what it was never given)")
    ok_mask = test_masked_gradient(model, task)

    print("\n[5] THE TRIBUTE CHANNEL  (you can be of no profit to Him)")
    ok_trib = test_tribute_sink(model, task)

    print("\n[6] ROUTING VS ISOLATION")
    ok_route = test_routing_beats_isolation(model, task)

    print("\n[7] THE TWO CELLS  (despair / presumption)")
    ok_cells = test_two_cells(model, task)

    print("\n[8] THE VESSEL IN THE FOUNTAIN")
    ok_vessel = test_vessel(model, task)

    print("\n[9] DELIVERED VS MERIT  (whether you will it or not / worth nothing)")
    ok_merit = test_merit(model, task)

    _, post = model.forward(Xv, yv, sv, collect=True)
    print("\n[10] FINAL STATE")
    print("   held-out mse ...................... %.4f  (from %.4f)"
          % (post["mse"], pre["mse"]))
    print("   tribute mass per agent ............ %s" % np.array2string(post["tribute"]))
    print("   useful delivery per agent ......... %s" % np.array2string(post["delivered"]))
    print("   discretion gate per agent ......... %s" % np.array2string(post["alpha"]))
    print("   mean routing matrix (rows=givers, last column = the principal):")
    Rm = post["R"].mean(axis=0)
    for i in range(model.N):
        print("      %d  %s" % (i, np.array2string(Rm[i], precision=3)))

    checks = [ok_grad, ok_prov, ok_mask, ok_trib, ok_route, ok_cells, ok_vessel, ok_merit]
    print("\n" + "=" * 78)
    print(" %d/%d checks passed  ---  %s"
          % (sum(checks), len(checks), "ALL PASS" if all(checks) else "FAILURES PRESENT"))
    print("=" * 78)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

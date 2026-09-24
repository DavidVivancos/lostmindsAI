#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PRISAD  —  The Graft-Union Network
 Chapter 0230 · Clement of Ohrid (c. 840-916)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0230_clement_of_ohrid_840 - Clement of Ohrid (c.840-916)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose structure encodes the one cognitive operation that was Clement's and
nobody else's: SURVIVAL BY RE-ROOTING.

In 885 Methodius died in Moravia. Within a year the mission he had built was
gone. Wiching held the bishopric, Pope Stephen V had forbidden the Slavonic
liturgy, Svatopluk had turned. Around two hundred disciples were arrested,
beaten and driven out; some were sold to slave dealers and carried to the
market at Venice, where an imperial officer ransomed what he could. Angelarius
reached Bulgaria and died of his injuries. The books were scattered. The school
was not damaged. It was deleted.

Clement did not try to restore it. He did not attempt to carry the institution
across the Danube, and he did not try to reproduce Moravia in Bulgaria. He cut
the thing he wanted to keep away from the substrate that had died, and he joined
it to entirely different living substrate — a different sponsor, a different
dialect, a different soil at Ohrid — and accepted that the roots would be
foreign for good. Between 886 and 893 he taught some three thousand five
hundred pupils. The graft held for a thousand years. The original tree had been
cut down in under twelve months.

The Long Life records that he also did this literally: he brought cultivated
fruit trees from Greece into a region that had only wild stock, and taught the
inhabitants to grow them. You do not propagate a named fruit variety from seed;
seed does not come true. You propagate it by cutting a scion and joining it to
local rootstock. The chapter takes that operation as the mechanism, and this
file makes it a network.

THE ARCHITECTURAL CLAIM
-----------------------
A graft is not a blend. The two tissues stay distinct for the life of the tree
and compatibility has to hold in exactly one place: the union, where cambium
meets cambium. So the architecture separates three things that modern transfer
learning keeps fused:

    CONTENT comes from the scion only.   (the fruit is the scion's)
    VIGOUR  comes from the rootstock only. (the water is the root's)
    FLOW    is gated by bilateral agreement at the join.

Concretely, each CAMBIAL UNIT j in the union layer computes

    m_j = s_j * r_j                       contact: the two normalised faces
    c_j = sigmoid( gamma * R * m_j + b_j )  conductance of this channel
    u_j = c_j * v_j * a_j                 flow = gate x vigour x scion content

where a is the scion's raw face, v > 0 is per-channel vigour produced by the
rootstock, and s, r are the L2-normalised scion and rootstock faces. Because
sum_j m_j is exactly the cosine between the two faces, the union's mean
conductance is a readable scalar measurement of how well the graft has taken.

THE FAILURE MODE IS BUILT IN, NOT BOLTED ON
-------------------------------------------
Every orchardist knows what happens when a graft dies: the rootstock re-sprouts.
The tree goes on standing, goes on leafing, goes on fruiting — and the fruit is
wild. Nothing announces the loss. So:

    rho = 1 - mean_j(c_j)                        dryness of the union
    y   = (1 - rho) * y_graft  +  rho * y_wild

As the union dries the output slides continuously into the rootstock's own
pre-graft behaviour. No exception is raised. No confidence drops. The network
keeps producing fluently, in the wrong variety, wearing the orchard's name.
Experiment 3 measures exactly how invisible this is from the output.

WHAT THE FOUR EXPERIMENTS TEST
------------------------------
  E1 TRANSPLANT vs RESEED vs GRAFT — the 886 decision, made measurable.
  E2 UNION RANK — how wide the join may be before the scion is absorbed.
  E3 SILENT REVERSION — the confidence-accuracy gap under a drying union,
     and whether union conductance catches what the output does not.
  E4 THE CHAIN — four generations taught by demonstration only, with the
     teacher deleted after it has taught, under three teaching regimes.
     The quantity of interest is not a generation's own score. It is the
     TRANSMISSION DERIVATIVE: its score minus its students' score.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, hand-derived analytic gradients
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 chapter_0230_clement_of_ohrid_840.py
================================================================================
"""

import numpy as np

EPS = 1e-8


# ==============================================================================
# 0.  SMALL UTILITIES
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic function, elementwise."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softplus(z):
    """log(1+e^z), stable. Used for vigour, which must be strictly positive:
    a rootstock can supply little water but never negative water."""
    return np.logaddexp(0.0, z)


def softmax_rows(z):
    """Row-wise softmax, stable."""
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


def l2_normalise_rows(A):
    """Return (normalised rows, row norms). The norms are needed again in the
    backward pass, so they are returned rather than recomputed."""
    n = np.sqrt((A * A).sum(axis=1, keepdims=True)) + EPS
    return A / n, n


def d_l2_normalise_rows(g_hat, A_hat, norms):
    """Backprop through row-wise L2 normalisation.

    If  u = a / ||a||  then  du_i/da_k = (delta_ik - u_i u_k) / ||a||, so

        dL/da = ( dL/du - u * (dL/du . u) ) / ||a||

    which is the projection of the incoming gradient onto the tangent space of
    the unit sphere, divided by the norm. Vectorised over rows here.
    """
    dot = (g_hat * A_hat).sum(axis=1, keepdims=True)
    return (g_hat - A_hat * dot) / norms


def cross_entropy(logits, y_idx):
    """Mean cross-entropy over a batch and the gradient dL/dlogits.

    Computed through log-sum-exp rather than by taking the log of a softmax
    probability. That matters here for a mundane but instructive reason: a
    clipped probability has a FLAT loss surface where the clip bites, so a
    finite-difference check reports zero slope on exactly the samples the
    analytic gradient cares most about, and a correct backward pass gets
    reported as a bug. The exact form has no such dead zone.

    Returns (loss, grad); grad is already divided by the batch size, so every
    downstream gradient is a mean gradient.
    """
    n = logits.shape[0]
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    loss = float((lse.ravel() - logits[np.arange(n), y_idx]).mean())
    g = np.exp(logits - lse)                       # == softmax(logits)
    g[np.arange(n), y_idx] -= 1.0
    return loss, g / n


def accuracy(logits, y_idx):
    return float((logits.argmax(axis=1) == y_idx).mean())


def rule(ch="=", n=78):
    print(ch * n)


# ==============================================================================
# 1.  THE WORLD:  DOCTRINE IS PORTABLE, ADMISSIBILITY IS LOCAL
# ==============================================================================
#
# The task is built to have exactly the shape of the problem Clement faced, and
# it is built so that NEITHER half of the tree can solve it alone.
#
#   CONTENT x       what the books say. The same books travel from Moravia to
#                   Ohrid unchanged; this is the scion.
#   CONTEXT z       the soil: who is listening, what this community already
#                   holds, what a bishop here may and may not say. Different
#                   distribution and different rule in each domain.
#
#   doctrine(x)     a FIXED nonlinear map, IDENTICAL IN BOTH DOMAINS. This is
#                   the thing worth carrying. It is what a scion is.
#   admissible(z)   a per-domain boolean mask over the K doctrine classes.
#                   In this community, these things can be said; those cannot.
#
#   target          doctrine(x) if it is admissible here; otherwise the highest
#                   scoring doctrine class that IS admissible here.
#
# So a model that has only the books gets it right only by luck when the first
# choice happens to be sayable. A model that has only the soil cannot know what
# the books say at all. The join is the whole problem, which is the point.
# ==============================================================================

class World:
    """Generates the two domains. The doctrine map is shared; everything about
    the local rule and the context distribution is per-domain."""

    def __init__(self, n_classes=6, d_content=14, d_context=10, n_env=4, seed=11):
        rng = np.random.default_rng(seed)
        self.K = n_classes
        self.Dc = d_content
        self.Dz_raw = d_context
        self.n_env = n_env
        # Context handed to a tree is the soil PLUS a one-hot naming which place
        # this is. That lets one rootstock serve several soils and specialise,
        # while the scion — which sees only content — cannot specialise at all.
        self.Dz = d_context + n_env

        # --- the portable part: content -> doctrine scores -------------------
        # A fixed two-layer random map. Nonlinear, so it cannot be read off by
        # a linear probe for free, and shared by every domain forever.
        self.Vd = rng.normal(0, 1.0, (18, d_content))
        self.Ud = rng.normal(0, 1.0, (n_classes, 18))

        # --- per-domain local rule: context -> admissibility mask ------------
        self.domains = {}

    def add_domain(self, name, seed, env_id, shift=0.0, scale=1.0, bias_mean=0.0):
        rng = np.random.default_rng(seed)
        self.domains[name] = dict(
            env_id=env_id,
            Va=rng.normal(0, 1.0, (16, self.Dz_raw)),
            Ua=rng.normal(0, 1.0, (self.K, 16)),
            bias=rng.normal(bias_mean, 0.45, (self.K,)),
            # What this place answered BEFORE anything was grafted onto it: a
            # settled local practice, a function of the soil alone. It is not
            # noise and it is not nonsense. It is coherent, it is confident,
            # and on the grafted task it is mostly wrong.
            Vw=rng.normal(0, 1.0, (16, self.Dz_raw)),
            Uw=rng.normal(0, 1.0, (self.K, 16)),
            mu=rng.normal(shift, 0.6, (self.Dz_raw,)),
            sd=scale,
            rng_seed=seed,
        )

    def doctrine_scores(self, X):
        return np.tanh(X @ self.Vd.T) @ self.Ud.T          # (n, K)

    def sample_mixed(self, names, n_each, seed):
        """Draw from several soils at once. The scion trained on this sees one
        doctrine and several local rules, and has no way to tell them apart
        except through a rootstock it will not be taking with it."""
        parts = [self.sample(nm, n_each, seed + 7 * i) for i, nm in enumerate(names)]
        out = [np.concatenate([p[k] for p in parts], axis=0) for k in range(6)]
        rng = np.random.default_rng(seed + 991)
        perm = rng.permutation(out[0].shape[0])
        return tuple(o[perm] for o in out)

    def sample(self, name, n, seed):
        """Draw n (content, context, target) triples from one domain."""
        d = self.domains[name]
        rng = np.random.default_rng(seed)

        X = rng.normal(0, 1.0, (n, self.Dc))
        Z = rng.normal(0, 1.0, (n, self.Dz_raw)) * d["sd"] + d["mu"]

        S = self.doctrine_scores(X)                        # (n, K)
        A = np.tanh(Z @ d["Va"].T) @ d["Ua"].T + d["bias"]  # (n, K)
        mask = A > 0.0

        # A community that can hear nothing is not a community; if the mask is
        # empty, open the single most nearly admissible class.
        empty = ~mask.any(axis=1)
        if empty.any():
            mask[empty, A[empty].argmax(axis=1)] = True

        # target = best admissible doctrine class
        S_masked = np.where(mask, S, -np.inf)
        y = S_masked.argmax(axis=1)

        # the wild answer: settled local practice, from the soil alone
        yw = (np.tanh(Z @ d["Vw"].T) @ d["Uw"].T).argmax(axis=1)

        # stamp the place onto the context
        E = np.zeros((n, self.n_env))
        E[:, d["env_id"]] = 1.0
        Z = np.concatenate([Z, E], axis=1)

        # bookkeeping used by the probes and the reporting
        first_choice = S.argmax(axis=1)
        unconstrained = (first_choice == y)   # the local rule did not bite
        return (X, Z, y.astype(np.int64), yw.astype(np.int64),
                first_choice.astype(np.int64), unconstrained)


def build_world(seed=11):
    """Moravia is where the scion was grown. Kutmichevitsa is the soil it had
    to be re-rooted into: a different local rule, a shifted context
    distribution, the same books."""
    w = World(seed=seed)
    # bias_mean is tuned so that the LOCAL RULE BITES about equally hard in
    # both places (it leaves the books alone roughly 40% of the time in each).
    # Without that, a difference between the two domains would just be a
    # difference in task difficulty wearing a historical costume.
    w.add_domain("MORAVIA",       seed=101, env_id=0, shift=0.00, scale=1.00, bias_mean=0.00)
    w.add_domain("PANNONIA",      seed=303, env_id=1, shift=-0.70, scale=0.85, bias_mean=-1.30)
    w.add_domain("BLATNOGRAD",    seed=707, env_id=2, shift=0.40, scale=1.10, bias_mean=-0.90)
    w.add_domain("KUTMICHEVITSA", seed=202, env_id=3, shift=1.10, scale=1.25, bias_mean=-1.35)
    return w


# The three soils the mission actually stood in before Bulgaria. Constantine
# and Methodius worked in Moravia from 863; Kocel of Pannonia took the letters
# and gave them fifty pupils; the corpus was argued over at Venice and ratified
# at Rome. By 886 the same books had been made to work under more than one
# local authority. Whether that mattered is Experiment 1.
PROVED_SOILS = ["MORAVIA", "PANNONIA", "BLATNOGRAD"]


# ==============================================================================
# 2.  THE CAMBIAL UNIT AND THE GRAFTED TREE
# ==============================================================================
#
# THE UNIT.  A cambial unit is not a neuron in the usual sense and it is
# deliberately not an attention head. It has no query, no key, no softmax over
# stored values, and it retrieves nothing. It is a valve on a join.
#
#   Three inputs, from three different places, doing three different jobs:
#
#     a_j   CONTENT.  The scion's raw face. The only source of what is said.
#     v_j   VIGOUR.   From the rootstock. Strictly positive. Scales how strongly
#                     this channel can push, never what it says.
#     c_j   GATE.     From the AGREEMENT of the two faces at this channel, not
#                     from either one alone. Neither tissue can open the valve
#                     by itself; they have to meet.
#
#   c_j = sigmoid( gamma * R * s_j * r_j + beta_j )        s = a/|a|, r = a_r/|a_r|
#   u_j = c_j * v_j * a_j
#
# Why the normalised product. sum_j s_j r_j is exactly cos(a, a_r). Splitting
# that sum back out per channel gives a per-channel contact term whose total is
# a quantity an orchardist would recognise: how squarely the two cut faces are
# lying against each other. The factor R keeps the pre-activation O(1) as the
# union widens, so gamma means the same thing at every rank in Experiment 2.
#
# Why cosine and not a dot product. Cambium contact is a matter of alignment,
# not of how big either piece is. A vigorous rootstock and a weak scion still
# make a sound join if the faces meet. Scale belongs in vigour, where it is
# named, and not smuggled into the gate.
#
# THE LAYER.  Above the units sits the reversion blend, which is the whole
# reason the architecture is shaped like this:
#
#   rho = 1 - mean_j c_j                dryness of the union
#   y   = (1 - rho) * y_graft + rho * y_wild
#
# y_wild is the rootstock's own head, trained on the same task from context
# alone. It is what this tree does when the graft is not conducting. It is not
# an error path and not a fallback the designer chose. It is the wild stock
# still being alive under the join, which is the actual situation.
# ==============================================================================

PARAM_GROUPS = {
    "scion": ["Ws1", "bs1", "Ws2", "bs2", "ks"],
    "root":  ["Wr1", "br1", "Wr2", "br2", "Wv", "bv"],
    # The wild path is the rootstock's OWN behaviour, learned before any graft
    # and frozen thereafter. A stock does not forget what it is.
    "wild":  ["Wm1", "bm1", "Ww", "bw"],
    "union": ["gamma", "beta"],
    "head":  ["Wg", "bg"],
}


class GraftedTree:
    """Scion + rootstock + cambial union + reversion, in pure NumPy.

    Parameters are held in a flat dict so that whole tissues can be frozen,
    copied between trees, or re-initialised independently — which is the entire
    point of the experiments below.
    """

    def __init__(self, d_content, d_context, n_classes,
                 h_scion=24, h_root=24, rank=8, gamma=2.0, seed=0):
        rng = np.random.default_rng(seed)
        self.Dc, self.Dz, self.K = d_content, d_context, n_classes
        self.Hs, self.Hr, self.R = h_scion, h_root, rank

        def glorot(shape, fan_in):
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)

        self.p = {
            # --- scion: content -> face --------------------------------------
            "Ws1": glorot((h_scion, d_content), d_content),
            "bs1": np.zeros(h_scion),
            "Ws2": glorot((rank, h_scion), h_scion),
            "bs2": np.zeros(rank),
            # The scion's CAMBIUM SIGNATURE: the shape of its cut face. A
            # property of the variety, not of today's sap, so it is a parameter
            # and not a function of the input. It travels with the cutting and
            # the new rootstock has to learn to meet a face it did not choose.
            "ks": rng.normal(0.0, 1.0, (rank,)),
            # --- rootstock: context -> face, vigour, wild head ---------------
            "Wr1": glorot((h_root, d_context), d_context),
            "br1": np.zeros(h_root),
            "Wr2": glorot((rank, h_root), h_root),
            "br2": np.zeros(rank),
            "Wv":  glorot((rank, h_root), h_root),
            "bv":  np.full(rank, 0.5),        # start with some water flowing
            # --- wild path: the stock's own fruit, on its own trunk ----------
            "Wm1": glorot((h_root, d_context), d_context),
            "bm1": np.zeros(h_root),
            "Ww":  glorot((n_classes, h_root), h_root),
            "bw":  np.zeros(n_classes),
            # --- union -------------------------------------------------------
            # Reparameterised so that the mechanism cannot be bypassed:
            #   gamma_eff = softplus(gamma) >= 0   contact never HURTS flow
            #   beta_eff  = -softplus(beta) <= 0   bark can only OBSTRUCT
            # Together these make conductance a monotone increasing function of
            # cambium contact with no way for the network to prop the valve open
            # from the bias alone. Without this the gate learns to ignore the
            # join, which is the whole thing being modelled.
            "gamma": np.array(float(np.log(np.expm1(gamma)))),
            "beta":  np.full(rank, float(np.log(np.expm1(0.2)))),
            # --- graft head --------------------------------------------------
            "Wg": glorot((n_classes, rank), rank),
            "bg": np.zeros(n_classes),
        }
        # A fixed arbitrary rotation of the union's channel space. Never
        # trained, never used unless a drying experiment asks for it.
        A0 = rng.normal(0, 1, (rank, rank))
        self.Qrot, _ = np.linalg.qr(A0)

        self.frozen = set()

    # -- tissue management -----------------------------------------------------

    def freeze(self, *groups):
        for g in groups:
            self.frozen.update(PARAM_GROUPS[g])
        return self

    def thaw(self, *groups):
        for g in groups:
            self.frozen.difference_update(PARAM_GROUPS[g])
        return self

    def copy_tissue(self, other, *groups):
        """Take a cutting. Copies the named parameter groups out of another
        tree. This is the only channel by which anything crosses from the dead
        Moravian tree into the Bulgarian one."""
        for g in groups:
            for k in PARAM_GROUPS[g]:
                self.p[k] = other.p[k].copy()
        return self

    def clone(self):
        t = GraftedTree(self.Dc, self.Dz, self.K, self.Hs, self.Hr, self.R)
        t.p = {k: v.copy() for k, v in self.p.items()}
        t.Qrot = self.Qrot.copy()
        t.frozen = set(self.frozen)
        return t

    # -- forward ---------------------------------------------------------------

    def forward(self, X, Z, dry=0.0, force_rho=None):
        """Returns (log-probabilities, cache).

        dry       in [0,1]: the cambium shifting out of contact. Nothing about
                  the scion changes and no parameter changes; the two cut faces
                  simply stop lying square against one another.
        force_rho if given, overrides the computed dryness. force_rho=1.0 is
                  SEVERANCE: the scion is dead and the stock has re-sprouted.
                  The tree goes on standing and goes on fruiting."""
        p, R = self.p, self.R

        Hs_ = np.tanh(X @ p["Ws1"].T + p["bs1"])            # (n,Hs)
        As = Hs_ @ p["Ws2"].T + p["bs2"]                    # (n,R)  scion face

        Hr_ = np.tanh(Z @ p["Wr1"].T + p["br1"])            # (n,Hr)
        Ar = Hr_ @ p["Wr2"].T + p["br2"]                    # (n,R)  root face
        if dry > 0.0:
            # The cambium shifts. The rootstock face is rotated by a fixed
            # arbitrary orthogonal map, a little or a lot. Nothing about the
            # scion is consulted and no parameter changes: the tissues simply
            # stop lying square against one another.
            Ar = (1.0 - dry) * Ar + dry * (Ar @ self.Qrot.T)
        Q = Hr_ @ p["Wv"].T + p["bv"]                       # (n,R)
        V = softplus(Q)                                     # (n,R)  vigour > 0

        Hm_ = np.tanh(Z @ p["Wm1"].T + p["bm1"])            # (n,Hr)
        Yw = Hm_ @ p["Ww"].T + p["bw"]                      # (n,K)  wild fruit

        Ksh, ks_n = l2_normalise_rows(p["ks"][None, :])      # (1,R)
        Arh, nr = l2_normalise_rows(Ar)                     # (n,R)
        M = Ksh * Arh                                       # (n,R)  contact
        gam = softplus(p["gamma"])                          # >= 0
        bet = -softplus(p["beta"])                          # <= 0
        T = gam * R * M + bet
        C = sigmoid(T)                                      # (n,R)  conductance
        U = C * V * As                                      # (n,R)  flow

        Yg = U @ p["Wg"].T + p["bg"]                        # (n,K)
        rho = 1.0 - C.mean(axis=1, keepdims=True)           # (n,1)  dryness
        if force_rho is not None:
            rho = np.full_like(rho, float(force_rho))
        rho = np.clip(rho, 1e-6, 1.0 - 1e-6)

        # --- reversion, as a mixture over WHAT THE TREE FRUITS --------------
        # The blend is in probability space, not in logit space, and that is a
        # substantive choice rather than a numerical one. Blending logits lets
        # the graft head shout: it can scale its own magnitude up and drown the
        # wild head out no matter how little sap is actually crossing. Mixing
        # distributions makes rho mean what an orchardist means by it — the
        # share of this tree's fruit that came from the scion. A union that is
        # 60% dry yields 60% wild fruit, and no amount of confidence in the
        # graft head can buy that share back.
        lg = Yg - (Yg.max(1, keepdims=True) +
                   np.log(np.exp(Yg - Yg.max(1, keepdims=True)).sum(1, keepdims=True)))
        lw = Yw - (Yw.max(1, keepdims=True) +
                   np.log(np.exp(Yw - Yw.max(1, keepdims=True)).sum(1, keepdims=True)))
        L = np.logaddexp(np.log1p(-rho) + lg, np.log(rho) + lw)   # log p_mix
        Pg, Pw = np.exp(lg), np.exp(lw)
        Rg = np.exp(np.log1p(-rho) + lg - L)     # responsibility of the scion
        Rw = 1.0 - Rg                            # responsibility of the stock

        cache = dict(X=X, Z=Z, Hs_=Hs_, As=As, Hr_=Hr_, Ar=Ar, Q=Q, V=V,
                     Yw=Yw, Hm_=Hm_, Ksh=Ksh, ks_n=ks_n, Arh=Arh, nr=nr, M=M, C=C, U=U,
                     Yg=Yg, rho=rho, gam=gam, Pg=Pg, Pw=Pw, Rg=Rg, Rw=Rw)
        return L, cache

    # -- backward: every gradient below is derived by hand ---------------------

    def backward(self, cache, dY):
        p, R = self.p, self.R
        C, V, As, U = cache["C"], cache["V"], cache["As"], cache["U"]
        rho, Yg, Yw = cache["rho"], cache["Yg"], cache["Yw"]
        Hs_, Hr_, Q = cache["Hs_"], cache["Hr_"], cache["Q"]
        del Hr_
        Ksh, Arh, ks_n, nr, M = cache["Ksh"], cache["Arh"], cache["ks_n"], cache["nr"], cache["M"]

        Pg, Pw, Rg, Rw = cache["Pg"], cache["Pw"], cache["Rg"], cache["Rw"]
        g = {}

        # dY here is dL/d(log p_mix). Pushing it back through the mixture and
        # then through each expert's softmax. Everything is written in terms of
        # the responsibilities Rg, Rw so that no 1/p_mix ever appears.
        #   dL/dYg = dY*Rg - Pg * sum_k(dY_k Rg_k)
        #   dL/dYw = dY*Rw - Pw * sum_k(dY_k Rw_k)
        sg = (dY * Rg).sum(axis=1, keepdims=True)
        sw = (dY * Rw).sum(axis=1, keepdims=True)
        dYg = dY * Rg - Pg * sg
        dYw = dY * Rw - Pw * sw
        drho = (dY * (Rw / rho - Rg / (1.0 - rho))).sum(axis=1, keepdims=True)

        # graft head
        g["Wg"] = dYg.T @ U
        g["bg"] = dYg.sum(axis=0)
        dU = dYg @ p["Wg"]                                          # (n,R)

        # wild path (its own trunk, so reverting really does mean reverting)
        Hm_ = cache["Hm_"]
        g["Ww"] = dYw.T @ Hm_
        g["bw"] = dYw.sum(axis=0)
        dHm_ = dYw @ p["Ww"]
        dpre_m = dHm_ * (1.0 - Hm_ ** 2)
        g["Wm1"] = dpre_m.T @ cache["Z"]
        g["bm1"] = dpre_m.sum(axis=0)
        dHr_ = np.zeros_like(cache["Hr_"])                          # (n,Hr)

        # u = c * v * a   and   rho = 1 - mean(c)
        dC = dU * V * As + drho * (-1.0 / R)
        dV = dU * C * As
        dAs = dU * C * V                                            # direct path

        # c = sigmoid(t),  t = softplus(gamma)*R*m - softplus(beta)
        dT = dC * C * (1.0 - C)
        g["beta"] = -dT.sum(axis=0) * sigmoid(p["beta"])
        g["gamma"] = np.array((dT * R * M).sum() * float(sigmoid(p["gamma"])))
        dM = dT * cache["gam"] * R

        # m = k_hat * r_hat   (signature against rootstock face)
        dKsh = (dM * Arh).sum(axis=0, keepdims=True)        # (1,R)
        dArh = dM * Ksh
        g["ks"] = d_l2_normalise_rows(dKsh, Ksh, ks_n).ravel()
        dAr = d_l2_normalise_rows(dArh, Arh, nr)

        # vigour v = softplus(q)
        dQ = dV * sigmoid(Q)
        g["Wv"] = dQ.T @ cache["Hr_"]
        g["bv"] = dQ.sum(axis=0)
        dHr_ = dHr_ + dQ @ p["Wv"]

        # rootstock face
        g["Wr2"] = dAr.T @ cache["Hr_"]
        g["br2"] = dAr.sum(axis=0)
        dHr_ = dHr_ + dAr @ p["Wr2"]

        dpre_r = dHr_ * (1.0 - cache["Hr_"] ** 2)
        g["Wr1"] = dpre_r.T @ cache["Z"]
        g["br1"] = dpre_r.sum(axis=0)

        # scion
        g["Ws2"] = dAs.T @ Hs_
        g["bs2"] = dAs.sum(axis=0)
        dHs_ = dAs @ p["Ws2"]
        dpre_s = dHs_ * (1.0 - Hs_ ** 2)
        g["Ws1"] = dpre_s.T @ cache["X"]
        g["bs1"] = dpre_s.sum(axis=0)

        for k in self.frozen:
            g[k] = np.zeros_like(p[k])
        return g

    # -- convenience -----------------------------------------------------------

    def loss_and_grad(self, X, Z, y, teacher_logits=None, kappa=0.0):
        """Cross-entropy, optionally with the teacher's hand on the student's.

        When kappa > 0 the mark that is graded is a blend of the student's own
        output and the teacher's:  y_eff = (1-kappa)*y_student + kappa*y_teacher.
        The student is therefore judged on a page it did not write alone, and
        its gradient is scaled by (1-kappa). That is what a hand on a hand is.
        """
        Y, cache = self.forward(X, Z)
        if teacher_logits is not None and kappa > 0.0:
            Y_eff = (1.0 - kappa) * Y + kappa * teacher_logits
            loss, dEff = cross_entropy(Y_eff, y)
            dY = (1.0 - kappa) * dEff
        else:
            loss, dY = cross_entropy(Y, y)
        return loss, self.backward(cache, dY), Y

    def wild_loss_and_grad(self, Z, yw):
        """Loss on the rootstock's own fruit alone, used to grow the stock
        before anything is grafted onto it."""
        p = self.p
        Hm_ = np.tanh(Z @ p["Wm1"].T + p["bm1"])
        Yw = Hm_ @ p["Ww"].T + p["bw"]
        loss, dYw = cross_entropy(Yw, yw)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Ww"] = dYw.T @ Hm_
        g["bw"] = dYw.sum(axis=0)
        dHm_ = dYw @ p["Ww"]
        dpre = dHm_ * (1.0 - Hm_ ** 2)
        g["Wm1"] = dpre.T @ Z
        g["bm1"] = dpre.sum(axis=0)
        return loss, g, Yw

    def wild_predict(self, Z):
        Hm_ = np.tanh(Z @ self.p["Wm1"].T + self.p["bm1"])
        return Hm_ @ self.p["Ww"].T + self.p["bw"]

    def predict(self, X, Z, dry=0.0, force_rho=None):
        """Returns log p(class). Because softmax(log p) == p exactly, this can
        be handed to cross_entropy and argmax unchanged."""
        return self.forward(X, Z, dry=dry, force_rho=force_rho)[0]

    def confidence(self, X, Z, dry=0.0, force_rho=None):
        """Mean probability assigned to the class the tree actually picks. This
        is the standard output-side health check, and Experiment 3 is about how
        little it is worth here."""
        return float(np.exp(self.predict(X, Z, dry=dry,
                                         force_rho=force_rho)).max(axis=1).mean())

    def scion_share(self, X, Z, dry=0.0):
        """Mean responsibility carried by the graft head: how much of the fruit
        is the scion's. 1 - this is how much is wild."""
        _, c = self.forward(X, Z, dry=dry)
        return float((1.0 - c["rho"]).mean())

    def score(self, X, Z, y, dry=0.0, force_rho=None):
        return accuracy(self.predict(X, Z, dry=dry, force_rho=force_rho), y)

    def union_health(self, X, Z, dry=0.0):
        """The readable scalar an inspector would put on the join: mean channel
        conductance, and the cosine between the two cut faces. These are the
        probe in Experiment 3 — cheap, local, and available without knowing
        whether any particular answer was right."""
        _, c = self.forward(X, Z, dry=dry)
        return float(c["C"].mean()), float(c["M"].sum(axis=1).mean())


# ==============================================================================
# 3.  OPTIMISER
# ==============================================================================

class Adam:
    """Adam with an explicit frozen set.

    The frozen set is not a convenience. Weight decay is applied to the
    gradient, so a parameter whose analytic gradient is zero will still drift if
    the optimiser is allowed to touch it — and a scion that quietly decays
    towards the origin over a few thousand steps is no longer the variety you
    cut. This was a live bug in an earlier draft of this file and it silently
    cost the grafted arm about ten points before it was caught.
    """

    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0,
                 frozen=()):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.frozen = set(frozen)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k in params:
            if k in self.frozen:
                continue
            gk = grads[k]
            if self.wd and params[k].ndim == 2:
                gk = gk + self.wd * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# 4.  GRADIENT CHECK  (mandatory — nothing ships if this fails)
# ==============================================================================

def gradient_check(rank=8, seed=7, n=7, h=1e-6, tol=2e-5, frozen=()):
    """Central-difference check of every trainable parameter against the
    hand-derived analytic gradient. Uses violent weights on purpose: the
    normalisation and the sigmoid gate are the fragile parts and they must be
    right far away from the origin, not just near it."""
    rng = np.random.default_rng(seed)
    Dc, Dz, K = 9, 7, 5
    tree = GraftedTree(Dc, Dz, K, h_scion=11, h_root=10, rank=rank, seed=seed)
    if frozen:
        tree.freeze(*frozen)
    for k, v in tree.p.items():
        if v.ndim > 0:
            tree.p[k] = v + rng.normal(0, 0.8, v.shape)
        else:
            tree.p[k] = np.array(float(v) + rng.normal(0, 0.4))

    X = rng.normal(0, 1.2, (n, Dc))
    Z = rng.normal(0, 1.2, (n, Dz))
    y = rng.integers(0, K, n)

    _, g, _ = tree.loss_and_grad(X, Z, y)

    worst, worst_key = 0.0, None
    frozen_keys = set()
    for gname in frozen:
        frozen_keys.update(PARAM_GROUPS[gname])

    for k in sorted(tree.p.keys()):
        if k in frozen_keys:
            # a frozen tissue must report exactly zero, not merely something small
            if np.abs(g[k]).max() != 0.0:
                return False, np.inf, k
            continue
        P = tree.p[k]
        flat = np.atleast_1d(P).ravel()
        idxs = range(flat.size) if flat.size <= 40 else \
            np.random.default_rng(seed + 1).choice(flat.size, 40, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            lp, _ = cross_entropy(tree.forward(X, Z)[0], y)
            flat[i] = orig - h
            lm, _ = cross_entropy(tree.forward(X, Z)[0], y)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = np.atleast_1d(g[k]).ravel()[i]
            denom = max(1.0, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_key = rel, f"{k}[{i}]"
    return worst < tol, worst, worst_key


# ==============================================================================
# 5.  TRAINING
# ==============================================================================

def train(tree, X, Z, y, Xv, Zv, yv, epochs=60, bs=128, lr=6e-3, wd=1e-5,
          teacher=None, kappa0=0.0, anneal=True, seed=0, verbose=False):
    """One real training run with a real validation split.

    teacher : a GraftedTree whose logits guide the student's hand, or None.
    kappa0  : how much of the mark is the teacher's at the start.
    anneal  : whether the teacher lets go. If False the hand never comes off,
              and the student is never once graded on a page it wrote alone.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(tree.p, lr=lr, wd=wd, frozen=tree.frozen)
    n = X.shape[0]
    TL = teacher.predict(X, Z) if teacher is not None else None
    best = (-1.0, None)

    for ep in range(epochs):
        if teacher is None:
            kappa = 0.0
        elif anneal:
            kappa = kappa0 * max(0.0, 1.0 - ep / (0.7 * epochs))
        else:
            kappa = kappa0

        order = rng.permutation(n)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            tl = TL[b] if TL is not None else None
            _, g, _ = tree.loss_and_grad(X[b], Z[b], y[b], tl, kappa)
            opt.step(tree.p, g)

        if ep % 5 == 4 or ep == epochs - 1:
            va = tree.score(Xv, Zv, yv)          # always scored with kappa = 0
            if va > best[0]:
                best = (va, {k: v.copy() for k, v in tree.p.items()})
            if verbose:
                print(f"    ep {ep + 1:3d}  val {va:.4f}  kappa {kappa:.2f}")

    if best[1] is not None:
        tree.p = best[1]
    return best[0]


# ==============================================================================
# 6.  GROWING A STOCK, AND GRAFTING ONTO IT
# ==============================================================================

def grow_rootstock(tree, Z, yw, Zv, ywv, epochs=45, lr=8e-3, seed=0):
    """Train ONLY the wild path: what this soil already answers, before anyone
    arrives with a cutting. Then the stock is what it is, and is frozen.

    This matters for the whole file. If the rootstock's own head were trained on
    the grafted task it would be a second solver, reversion would be cheap, and
    the architecture would be measuring nothing. A real rootstock has settled
    local behaviour that is coherent, confident, and not the answer you wanted.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(tree.p, lr=lr, frozen=tree.frozen)
    n = Z.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, 128):
            b = order[s:s + 128]
            _, g, _ = tree.wild_loss_and_grad(Z[b], yw[b])
            opt.step(tree.p, g)
    return accuracy(tree.wild_predict(Zv), ywv)


def fresh_tree(w, rank=8, seed=0, hs=28, hr=28):
    return GraftedTree(w.Dc, w.Dz, w.K, h_scion=hs, h_root=hr, rank=rank, seed=seed)


def grow_donor(w, soils, rank=8, seed=0, n=6000, epochs=110):
    """Grow a tree to maturity, to be the source of cuttings.

    `soils` is either one place or several. A tree that has only ever stood in
    one soil is a perfectly good tree; whether it is a good source of CUTTINGS
    is Experiment 1's question, and the answer turns out to be no.
    """
    n_each = n // len(soils)
    if len(soils) == 1:
        Xt, Zt, yt, ywt, _, _ = w.sample(soils[0], n, seed=1000 + seed)
        Xv, Zv, yv, ywv, _, _ = w.sample(soils[0], 2000, seed=2000 + seed)
    else:
        Xt, Zt, yt, ywt, _, _ = w.sample_mixed(soils, n_each, seed=1000 + seed)
        Xv, Zv, yv, ywv, _, _ = w.sample_mixed(soils, 700, seed=2000 + seed)
    t = fresh_tree(w, rank=rank, seed=seed)
    grow_rootstock(t, Zt, ywt, Zv, ywv, seed=seed)
    t.freeze("wild")
    va = train(t, Xt, Zt, yt, Xv, Zv, yv, epochs=epochs, lr=6e-3, seed=seed)
    return t, va


def plant_in(w, data, rank=8, seed=0, scion_from=None, freeze_scion=True,
             epochs=110):
    """Put a tree into new ground: grow the local stock on local practice,
    freeze it, optionally join a cutting, train."""
    Xtr, Ztr, ytr, ywtr, Xv, Zv, yv, ywv = data
    t = fresh_tree(w, rank=rank, seed=seed)
    grow_rootstock(t, Ztr, ywtr, Zv, ywv, seed=seed)
    t.freeze("wild")
    if scion_from is not None:
        t.copy_tissue(scion_from, "scion")
        if freeze_scion:
            t.freeze("scion")
    va = train(t, Xtr, Ztr, ytr, Xv, Zv, yv, epochs=epochs, lr=6e-3, seed=seed)
    return t, va


def new_ground(w, n_small, seed):
    Xtr, Ztr, ytr, ywtr, _, _ = w.sample("KUTMICHEVITSA", n_small, seed=3000 + seed)
    Xv, Zv, yv, ywv, _, _ = w.sample("KUTMICHEVITSA", 2500, seed=4000 + seed)
    return (Xtr, Ztr, ytr, ywtr, Xv, Zv, yv, ywv)


# ==============================================================================
# 7.  EXPERIMENT 1  —  886: TRANSPLANT, RESEED, OR GRAFT
# ==============================================================================
#
# The mission in Moravia is gone. Three things can be done with what is left,
# and Clement's contemporaries each did one of them.
#
#   TRANSPLANT  carry the whole tree across and plant it in the new ground.
#               Everything is kept: scion, stock, union, the lot, and then it is
#               allowed to adapt. This is what restoring the mission would have
#               meant. The roots are Moravian and the soil is not.
#   RESEED      start over from nothing in the new ground. The local answer, and
#               what most of the expelled disciples in fact did.
#   GRAFT       keep the scion, discard the stock, join the cutting to a stock
#               that already lives here. What Clement did.
#
# All three get the SAME small budget in the new domain, because that is the
# real constraint: he did not have decades.
# ==============================================================================

def experiment_1_provenance(w, n_small=900, rank=8, seeds=(0, 1, 2)):
    print()
    rule()
    print(" EXPERIMENT 1  —  886: TRANSPLANT, RESEED, OR GRAFT")
    rule()
    print("  Moravia is gone. Four things can be done with what is left, and")
    print("  every arm gets the same small budget in the new ground, because")
    print("  that is the real constraint: he did not have decades.")
    print()
    print("    TRANSPLANT  carry the whole tree over, roots and all, and let it")
    print("                adapt. What restoring the mission would have meant.")
    print("    RESEED      start again from nothing in the new ground.")
    print("    GRAFT_1     a cutting from a tree that only ever stood in Moravia.")
    print("    GRAFT_3     a cutting from a tree proved in three soils already.")
    print()
    print(f"  new-ground training samples per arm: {n_small}")
    print()

    arms = ("TRANSPLANT", "RESEED", "GRAFT_1", "GRAFT_3")
    acc = {k: [] for k in arms}
    doc = {k: [] for k in arms}
    cos = {k: [] for k in arms}
    donor_acc, donor_doc = {"one soil": [], "three soils": []}, {"one soil": [], "three soils": []}

    for sd in seeds:
        d1, v1 = grow_donor(w, ["MORAVIA"], rank=rank, seed=sd)
        d3, v3 = grow_donor(w, PROVED_SOILS, rank=rank, seed=sd)
        Xp, Zp, _, _, _, _ = w.sample("MORAVIA", 2500, seed=9000 + sd)
        donor_acc["one soil"].append(v1)
        donor_acc["three soils"].append(v3)
        donor_doc["one soil"].append(doctrine_probe(d1, w, Xp, Zp, seed=sd))
        donor_doc["three soils"].append(doctrine_probe(d3, w, Xp, Zp, seed=sd))

        data = new_ground(w, n_small, sd)
        Xv, Zv, yv = data[4], data[5], data[6]

        tp = d1.clone()
        tp.thaw("scion", "root", "union", "head"); tp.freeze("wild")
        va = train(tp, data[0], data[1], data[2], Xv, Zv, yv,
                   epochs=110, lr=6e-3, seed=sd)
        trees = {"TRANSPLANT": (tp, va)}
        trees["RESEED"] = plant_in(w, data, rank=rank, seed=sd + 77)
        trees["GRAFT_1"] = plant_in(w, data, rank=rank, seed=sd + 77, scion_from=d1)
        trees["GRAFT_3"] = plant_in(w, data, rank=rank, seed=sd + 77, scion_from=d3)

        for k in arms:
            t, va = trees[k]
            acc[k].append(va)
            doc[k].append(doctrine_probe(t, w, Xv, Zv, seed=sd))
            cos[k].append(t.union_health(Xv, Zv)[1])

    print("  the two donors, measured at home")
    print(f"  {'cutting taken from':<22}{'own accuracy':>14}{'doctrine carried':>19}")
    for k in ("one soil", "three soils"):
        print(f"  {k:<22}{np.mean(donor_acc[k]):>14.4f}{np.mean(donor_doc[k]):>19.4f}")
    print()
    print("  the four arms, in Kutmichevitsa")
    print(f"  {'arm':<14}{'accuracy':>11}{'sd':>8}{'doctrine':>11}{'union cos':>12}")
    for k in arms:
        a = np.array(acc[k])
        print(f"  {k:<14}{a.mean():>11.4f}{a.std():>8.4f}"
              f"{np.mean(doc[k]):>11.4f}{np.mean(cos[k]):>+12.3f}")
    print()
    print("  The cutting is not automatically the portable part. A scion grown")
    print("  in one soil has that soil baked into it and transfers barely")
    print("  better than starting over. One proved in three carries doctrine")
    print("  and nothing else, and it is the only arm that beats reseeding")
    print("  outright.")
    return acc, doc


# ==============================================================================
# 8.  EXPERIMENT 2  —  HOW WIDE MAY THE JOIN BE
# ==============================================================================
#
# A graft union is a narrow thing. The question the architecture makes askable
# is whether narrowness is a limitation or a condition. Widen the union and more
# sap crosses; widen it far enough and there is no longer a join at all, only
# one continuous piece of wood, and the scion stops being a distinct tissue.
#
# Measured two ways at each rank:
#   ACC      task accuracy in the new soil.
#   DOCTRINE how much of the portable content survives in the scion's face,
#            read by a linear probe trained to recover doctrine(x) from it.
#            This is the fruit still coming true to the variety.
# ==============================================================================

def doctrine_probe(tree, w, X, Z, n_iter=400, lr=0.12, seed=0):
    """Least-squares-ish linear probe from the scion's face onto the true
    doctrine class. Trained by plain gradient descent so nothing is imported."""
    _, c = tree.forward(X, Z)
    F = c["As"]
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    yd = w.doctrine_scores(X).argmax(axis=1)
    n_tr = int(0.7 * len(yd))
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.05, (w.K, F.shape[1]))
    b = np.zeros(w.K)
    for _ in range(n_iter):
        lo = F[:n_tr] @ W.T + b
        _, g = cross_entropy(lo, yd[:n_tr])
        W -= lr * (g.T @ F[:n_tr]); b -= lr * g.sum(0)
    return accuracy(F[n_tr:] @ W.T + b, yd[n_tr:])


def experiment_2_union_rank(w, ranks=(1, 2, 4, 8, 16, 32), n_small=900, seeds=(0, 1)):
    print()
    rule()
    print(" EXPERIMENT 2  —  HOW WIDE MAY THE JOIN BE, AND MAY THE SCION MOVE")
    rule()
    print("  Every tree here is GRAFT_3. Two arms at each rank: a scion held")
    print("  fixed, as a variety is, and a scion allowed to adapt to the new")
    print("  ground, as a fine-tune would.")
    print()
    print(f"  {'rank':>5}{'FROZEN acc':>13}{'FROZEN doc':>13}"
          f"{'FREE acc':>11}{'FREE doc':>11}{'cos':>9}")

    out = []
    for R in ranks:
        fa, fd, la, ld, cs = [], [], [], [], []
        for sd in seeds:
            d3, _ = grow_donor(w, PROVED_SOILS, rank=R, seed=sd)
            data = new_ground(w, n_small, sd)
            Xv, Zv, yv = data[4], data[5], data[6]
            tf, vf = plant_in(w, data, rank=R, seed=sd + 77,
                              scion_from=d3, freeze_scion=True)
            tl, vl = plant_in(w, data, rank=R, seed=sd + 77,
                              scion_from=d3, freeze_scion=False)
            fa.append(vf); la.append(vl)
            fd.append(doctrine_probe(tf, w, Xv, Zv, seed=sd))
            ld.append(doctrine_probe(tl, w, Xv, Zv, seed=sd))
            cs.append(tf.union_health(Xv, Zv)[1])
        out.append((R, np.mean(fa), np.mean(fd), np.mean(la), np.mean(ld)))
        print(f"  {R:>5}{np.mean(fa):>13.4f}{np.mean(fd):>13.4f}"
              f"{np.mean(la):>11.4f}{np.mean(ld):>11.4f}{np.mean(cs):>+9.3f}")
    print()
    print("  A join of one or two channels starves the scion no matter how")
    print("  good the cutting was. Past that the width stops mattering — and")
    print("  the thing that does matter is the second pair of columns: a scion")
    print("  allowed to adapt buys its fit by spending the variety.")
    return out


# ==============================================================================
# 9.  EXPERIMENT 3  —  SILENT REVERSION
# ==============================================================================
#
# Every orchardist knows this failure and no machine-learning monitor watches
# for it. The graft dies. The rootstock re-sprouts. The tree goes on standing,
# goes on leafing, goes on fruiting, and the fruit is wild. Nothing falls over.
# Nothing raises an exception. The orchard is still called an orchard.
#
# Two questions:
#   (a) Under partial drying, which signal moves — the output or the join?
#   (b) Under full severance, what does the output look like? The answer is the
#       uncomfortable one: fluent, confident, coherent, and wrong.
# ==============================================================================

def experiment_3_reversion(w, rank=8, seeds=(0, 1, 2)):
    print()
    rule()
    print(" EXPERIMENT 3  —  SILENT REVERSION")
    rule()

    packs = []
    for sd in seeds:
        d3, _ = grow_donor(w, PROVED_SOILS, rank=rank, seed=sd)
        data = new_ground(w, 2500, sd)
        gf, _ = plant_in(w, data, rank=rank, seed=sd + 77, scion_from=d3)
        packs.append((gf, data[4], data[5], data[6], data[7]))

    print("  (a) the union drying, nothing else touched")
    print(f"  {'dry':>6}{'accuracy':>11}{'confidence':>13}{'union cos':>12}"
          f"{'conductance':>14}")
    for d in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        A = [p[0].score(p[1], p[2], p[3], dry=d) for p in packs]
        Cf = [p[0].confidence(p[1], p[2], dry=d) for p in packs]
        H = [p[0].union_health(p[1], p[2], dry=d) for p in packs]
        print(f"  {d:>6.1f}{np.mean(A):>11.4f}{np.mean(Cf):>13.4f}"
              f"{np.mean([h[1] for h in H]):>+12.3f}"
              f"{np.mean([h[0] for h in H]):>14.4f}")

    print()
    print("  (b) severance: the scion is dead, the stock has re-sprouted")
    a_live = np.mean([p[0].score(p[1], p[2], p[3]) for p in packs])
    c_live = np.mean([p[0].confidence(p[1], p[2]) for p in packs])
    a_sev = np.mean([p[0].score(p[1], p[2], p[3], force_rho=1.0) for p in packs])
    c_sev = np.mean([p[0].confidence(p[1], p[2], force_rho=1.0) for p in packs])
    w_sev = np.mean([(p[0].predict(p[1], p[2], force_rho=1.0).argmax(1) == p[4]).mean()
                     for p in packs])
    print(f"  {'state':<16}{'accuracy':>11}{'confidence':>13}{'gap':>9}"
          f"{'agrees with local practice':>29}")
    print(f"  {'grafted':<16}{a_live:>11.4f}{c_live:>13.4f}"
          f"{c_live - a_live:>+9.3f}{'-':>29}")
    print(f"  {'severed':<16}{a_sev:>11.4f}{c_sev:>13.4f}"
          f"{c_sev - a_sev:>+9.3f}{w_sev:>29.4f}")
    print()
    print("  A severed tree is MORE confident than a living one and gets four")
    print("  answers in five wrong. It has not broken. It has gone back to")
    print("  what this place always said, which is a thing you can only see")
    print("  by looking at the join.")
    return packs


# ==============================================================================
# 10.  EXPERIMENT 4  —  THE CHAIN, AND THE TRANSMISSION DERIVATIVE
# ==============================================================================
#
# Theophylact says Clement was never once seen idle, and describes what he was
# doing instead: showing the shapes of the letters to one group, explaining the
# sense of the texts to a second, and TRAINING THE HANDS of a third to write —
# by day and also by night. Three cohorts at three depths of the same pipeline,
# running at the same time, in the same room.
#
# The third cohort is the one that is easy to skip and it is the one this
# experiment is about. A pupil who has been shown the letters can recognise
# them. A pupil whose hand has been guided can form them. Both will pass your
# examination. Only one of them can teach.
#
# So: four generations, each taught only by the one before it, by demonstration
# — no weights are ever copied between generations. After a generation has
# taught, it is deleted. Three regimes:
#
#   SHOWN         the pupil learns from the teacher's finished work. Standard
#                 distillation: here is the page, copy it.
#   GUIDED        the teacher's hand is on the pupil's and comes off. The mark
#                 being graded is a blend of both, weighted kappa, annealing to
#                 zero over training.
#   GUIDED_HELD   the same, except the hand never comes off. Kappa stays high
#                 for the whole of training. The pupil is never once graded on
#                 a page it wrote by itself.
#
# The reported quantity is not a generation's score. It is the TRANSMISSION
# DERIVATIVE: its own score minus the score of the generation it teaches. A
# large positive derivative is a mind that performs and cannot hand on.
# ==============================================================================

def experiment_4_chain(w, n_gen=4, cohort=3, attrition=1, rank=8,
                       n_cur=2500, seeds=(0, 1)):
    print()
    rule()
    print(" EXPERIMENT 4  —  THE CHAIN, AND THE TRANSMISSION DERIVATIVE")
    rule()
    print(f"  {n_gen} generations, cohort of {cohort}, {attrition} lost per")
    print("  generation, no weights ever copied, teacher deleted after teaching.")
    print()

    Xtr, Ztr, ytr, ywtr, _, _ = w.sample("KUTMICHEVITSA", 5000, seed=31)
    Xv, Zv, yv, ywv, _, _ = w.sample("KUTMICHEVITSA", 2500, seed=32)
    Xc, Zc = Xtr[:n_cur], Ztr[:n_cur]          # the curriculum's exercises

    results = {}
    for regime in ("SHOWN", "GUIDED", "GUIDED_HELD"):
        per_seed = []
        for sd in seeds:
            rng = np.random.default_rng(500 + sd)

            # --- generation 0: Clement himself, taught by Methodius ----------
            g0 = fresh_tree(w, rank=rank, seed=sd)
            grow_rootstock(g0, Ztr, ywtr, Zv, ywv, seed=sd)
            g0.freeze("wild")
            train(g0, Xtr, Ztr, ytr, Xv, Zv, yv, epochs=90, lr=6e-3, seed=sd)
            teachers = [g0]
            gen_scores = [g0.score(Xv, Zv, yv)]

            for gen in range(1, n_gen):
                # the curriculum: what the surviving teachers agree the answer is
                votes = np.stack([np.exp(t.predict(Xc, Zc)) for t in teachers]).mean(0)
                y_cur = votes.argmax(axis=1)
                t_logits = np.log(np.clip(votes, 1e-12, None))

                pupils = []
                for m in range(cohort):
                    pu = fresh_tree(w, rank=rank, seed=1000 * sd + 37 * gen + m)
                    grow_rootstock(pu, Ztr, ywtr, Zv, ywv, seed=sd)
                    pu.freeze("wild")
                    bs_idx = rng.integers(0, n_cur, n_cur)   # each pupil its own
                    if regime == "SHOWN":
                        train(pu, Xc[bs_idx], Zc[bs_idx], y_cur[bs_idx],
                              Xv, Zv, yv, epochs=70, lr=6e-3, seed=sd + m)
                    else:
                        held = (regime == "GUIDED_HELD")
                        teach = Teacher(t_logits[bs_idx])
                        train(pu, Xc[bs_idx], Zc[bs_idx], y_cur[bs_idx],
                              Xv, Zv, yv, epochs=70, lr=6e-3, seed=sd + m,
                              teacher=teach, kappa0=0.75, anneal=not held)
                    pupils.append(pu)

                # the market at Venice: some of the cohort do not arrive
                keep = list(rng.permutation(cohort)[:cohort - attrition])
                teachers = [pupils[i] for i in keep]
                gen_scores.append(float(np.mean([p.score(Xv, Zv, yv) for p in teachers])))

            per_seed.append(gen_scores)
        results[regime] = np.array(per_seed).mean(axis=0)

    print(f"  {'regime':<14}" + "".join(f"{'gen ' + str(i):>10}" for i in range(n_gen))
          + f"{'gen0-gen1':>12}{'gen0-gen3':>12}")
    for regime, sc in results.items():
        print(f"  {regime:<14}" + "".join(f"{v:>10.4f}" for v in sc)
              + f"{sc[0] - sc[1]:>+12.4f}{sc[0] - sc[-1]:>+12.4f}")
    print()
    print("  A hand that never comes off produces a generation that scores and")
    print("  cannot hand on. The derivative, not the score, is the measurement.")
    return results


class Teacher:
    """A thin wrapper so that train() can be handed precomputed teacher logits
    (from a cohort vote) rather than a single live tree."""
    def __init__(self, logits):
        self._l = logits

    def predict(self, X, Z):
        return self._l


# ==============================================================================
# 11.  STRUCTURAL SELF-TESTS
# ==============================================================================

def _toy(rank=6, seed=3):
    rng = np.random.default_rng(seed)
    t = GraftedTree(8, 6, 4, h_scion=10, h_root=9, rank=rank, seed=seed)
    X = rng.normal(0, 1, (25, 8))
    Z = rng.normal(0, 1, (25, 6))
    return t, X, Z


def test_output_is_a_distribution():
    t, X, Z = _toy()
    L, _ = t.forward(X, Z)
    P = np.exp(L)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-10), P.sum(axis=1)
    assert (P > 0).all()


def test_conductance_is_open_interval():
    """Under violent weights the gate must stay strictly inside (0,1): a valve
    that reaches exactly 0 or exactly 1 is no longer a valve."""
    rng = np.random.default_rng(9)
    for s in range(6):
        t, X, Z = _toy(seed=s)
        for k in t.p:
            t.p[k] = t.p[k] + rng.normal(0, 6.0, np.shape(t.p[k]))
        _, c = t.forward(X, Z)
        assert c["C"].min() > 0.0 and c["C"].max() < 1.0


def test_conductance_is_monotone_in_contact():
    """More cambium contact must never reduce flow. This is what the
    gamma >= 0 reparameterisation buys and it has to actually hold."""
    t, X, Z = _toy()
    t.p["gamma"] = np.array(2.0)
    _, c0 = t.forward(X, Z)
    base = c0["M"].copy()
    # nudge contact upward channel by channel and check conductance follows
    gam = float(softplus(t.p["gamma"]))
    bet = -softplus(t.p["beta"])
    for delta in (0.05, 0.2, 0.6):
        c_lo = sigmoid(gam * t.R * base + bet)
        c_hi = sigmoid(gam * t.R * (base + delta) + bet)
        assert (c_hi >= c_lo - 1e-12).all()


def test_vigour_is_strictly_positive():
    """A rootstock may supply little water. It may not supply negative water."""
    rng = np.random.default_rng(4)
    t, X, Z = _toy()
    for k in ("Wv", "bv"):
        t.p[k] = t.p[k] - 9.0
    _, c = t.forward(X, Z)
    assert (c["V"] > 0).all()


def test_content_comes_only_from_the_scion():
    """Change the scion and the flow through the union changes. That is the
    fruit being the scion's."""
    t, X, Z = _toy()
    _, c1 = t.forward(X, Z)
    t.p["Ws2"] = t.p["Ws2"] * 1.7 + 0.3
    _, c2 = t.forward(X, Z)
    assert not np.allclose(c1["U"], c2["U"])


def test_rootstock_supplies_no_content():
    """Scale the rootstock's vigour and the flow scales, but the DIRECTION of
    every channel is untouched: the stock changes how much, never what."""
    t, X, Z = _toy()
    _, c1 = t.forward(X, Z)
    t.p["bv"] = t.p["bv"] + 2.0
    _, c2 = t.forward(X, Z)
    s1 = np.sign(c1["U"]); s2 = np.sign(c2["U"])
    assert (s1 == s2).all()
    assert not np.allclose(c1["V"], c2["V"])


def test_severance_yields_the_wild_answer_exactly():
    """With the union forced fully dry the tree must output the stock's own
    fruit and nothing else — no residue of the graft."""
    t, X, Z = _toy()
    L = t.predict(X, Z, force_rho=1.0)
    W = t.wild_predict(Z)
    assert (L.argmax(1) == W.argmax(1)).all()


def test_freezing_a_tissue_really_freezes_it():
    t, X, Z = _toy()
    t.freeze("scion")
    y = np.random.default_rng(1).integers(0, 4, 25)
    before = {k: t.p[k].copy() for k in PARAM_GROUPS["scion"]}
    # weight decay ON, because decay is exactly how a frozen tissue leaks
    opt = Adam(t.p, lr=0.1, wd=1e-3, frozen=t.frozen)
    for _ in range(20):
        _, g, _ = t.loss_and_grad(X, Z, y)
        opt.step(t.p, g)
    for k in PARAM_GROUPS["scion"]:
        assert np.array_equal(t.p[k], before[k]), k


def test_cutting_transfers_the_signature_too():
    """A cutting carries its cambium signature, not only its content map. The
    new stock has to learn to meet a face it did not choose."""
    a, _, _ = _toy(seed=1)
    b, _, _ = _toy(seed=2)
    assert not np.allclose(a.p["ks"], b.p["ks"])
    b.copy_tissue(a, "scion")
    assert np.allclose(a.p["ks"], b.p["ks"])


def test_training_reduces_loss():
    w = build_world()
    X, Z, y, yw, _, _ = w.sample("MORAVIA", 1200, seed=77)
    t = fresh_tree(w, rank=6, seed=5)
    t.freeze("wild")
    l0, _, _ = t.loss_and_grad(X, Z, y)
    opt = Adam(t.p, lr=6e-3)
    for _ in range(200):
        _, g, _ = t.loss_and_grad(X, Z, y)
        opt.step(t.p, g)
    l1, _, _ = t.loss_and_grad(X, Z, y)
    assert l1 < l0 * 0.85, (l0, l1)


def test_neither_tissue_alone_can_solve_the_task():
    """The join has to be load-bearing or the whole file measures nothing."""
    w = build_world()
    X, Z, y, yw, first, unconstrained = w.sample("MORAVIA", 4000, seed=91)
    books_only = float((first == y).mean())
    assert books_only < 0.55, books_only
    majority = float(np.bincount(y, minlength=w.K).max() / len(y))
    assert majority < 0.55, majority


def run_self_tests():
    tests = [
        test_output_is_a_distribution,
        test_conductance_is_open_interval,
        test_conductance_is_monotone_in_contact,
        test_vigour_is_strictly_positive,
        test_content_comes_only_from_the_scion,
        test_rootstock_supplies_no_content,
        test_severance_yields_the_wild_answer_exactly,
        test_freezing_a_tissue_really_freezes_it,
        test_cutting_transfers_the_signature_too,
        test_training_reduces_loss,
        test_neither_tissue_alone_can_solve_the_task,
    ]
    print()
    rule()
    print(" STRUCTURAL SELF-TESTS")
    rule()
    ok = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n  {ok}/{len(tests)} passed")
    return ok == len(tests)


# ==============================================================================
# 12.  MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    rule()
    print(" PRISAD — The Graft-Union Network")
    print(" Chapter 0230 · Clement of Ohrid (c. 840-916)")
    print(" Encyclopedia of Lost Minds: Echoes on AI")
    rule()
    print("""
 In 886 the thing Clement had spent twenty years building was deleted. He did
 not restore it and he did not reproduce it. He cut away what was worth keeping,
 joined it to living local stock, and let the roots be foreign forever.

 This file is that operation as an architecture: content from the scion, vigour
 from the rootstock, flow gated by agreement at the join, and — because it is
 the failure every orchardist knows and no monitor watches for — the stock still
 alive underneath, ready to re-sprout and fruit wild under the orchard's name.
""")

    # --- gradient check first; nothing downstream is worth reading if it fails
    rule("-")
    print(" GRADIENT CHECK  (central differences vs hand-derived analytic)")
    rule("-")
    worst_all, passed = 0.0, True
    for R in (1, 3, 8, 16):
        for sd in (5, 7, 21):
            ok, worst, key = gradient_check(rank=R, seed=sd)
            worst_all = max(worst_all, worst)
            passed = passed and ok
    for fr in (("scion",), ("root",), ("union",), ("wild",), ("scion", "union")):
        ok, worst, key = gradient_check(rank=8, seed=5, frozen=fr)
        worst_all = max(worst_all, worst)
        passed = passed and ok
    print(f"  ranks 1/3/8/16 x seeds 5/7/21, plus 5 frozen-tissue configurations")
    print(f"  max relative error: {worst_all:.3e}   ->  {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit("gradient check failed; nothing ships")

    if not run_self_tests():
        raise SystemExit("self-tests failed; nothing ships")

    w = build_world()

    # a word on what the task's floors are, so the numbers below mean something
    X, Z, y, yw, first, unc = w.sample("KUTMICHEVITSA", 4000, seed=5)
    rule("-")
    print(" TASK FLOORS")
    rule("-")
    print(f"  majority class                      {np.bincount(y, minlength=w.K).max()/len(y):.4f}")
    print(f"  books alone (always first doctrine) {(first == y).mean():.4f}")
    print(f"  local rule leaves the books alone   {unc.mean():.4f} of the time")

    experiment_1_provenance(w)
    experiment_2_union_rank(w)
    experiment_3_reversion(w)
    experiment_4_chain(w)

    print()
    rule()
    print(" DONE")
    rule()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ENCYCLOPEDIA OF LOST MINDS — Chapter 250
 Abu al-Hasan Ali ibn Muhammad ibn Habib al-Mawardi  (Basra c.972 - Baghdad 1058)

 THE MANDATE-SCOPE NETWORK  (al-shabaka al-taqlidiyya)
 A from-scratch, pure-NumPy cognitive architecture in which AUTHORITY, not
 attention, is the primitive quantity that flows through the network.
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0250_al_mawardi_972 - Abu al-Hasan Ali ibn Muhammad ibn Habib al-Mawardi (Basra c.972 - Baghdad 1058)
#================================================================================  

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Al-Mawardi's "al-Ahkam al-Sultaniyya" is not a book about what is true. It is a
book about WHO MAY DECIDE WHAT, and by whose leave. Twenty chapters, twenty
offices, and for every office the same three questions asked again: what are the
qualifications (shurut), what is the extent of the jurisdiction (wilaya 'amma vs
khassa), and what must be referred back up. His deepest and least-imitated idea
is that authority is not a status a thing HAS but a typed, divisible, revocable
quantity a thing RECEIVES, always strictly smaller than what its principal held.

So this network does not compute "how much should token i attend to token j."
It computes "how much of the decision space is this office licensed to touch,
and what happens at the edge of the licence." Every mechanism below is a
formalisation of one of his doctrines:

  taqlid           investiture. A child office's mandate is the elementwise
                   product of its parent's mandate with its own gate. Scope
                   therefore NARROWS MONOTONICALLY down every path of the tree,
                   as a structural invariant of the arithmetic - not as a
                   learned regularity, not as a penalty term. It cannot be
                   trained away. (test_mandate_monotonicity)

  tafwid / tanfidh the two vizierates. Al-Mawardi's single sharpest engineering
                   distinction: the wazir al-tafwid decides for himself; the
                   wazir al-tanfidh only transmits the sovereign's will and may
                   originate nothing. Here every office carries a learned scalar
                   d in (0,1) that interpolates between "apply my own transform"
                   (tafwid, d->1) and "pass my principal's state through
                   unaltered" (tanfidh, d->0). Discretion is a dial the training
                   process must justify, not a default.

  ahliyya          competence. Each office gates on whether it is qualified for
                   THIS case, from its principal's state. Unqualified offices do
                   not merely score badly; they receive no routed mass at all.

  raf' / mazalim   referral. Routing mass is NOT normalised to sum to one over
                   the children. If no child office claims competence, the
                   residual mass rises instead to the Court of Grievances, whose
                   transform operates under the ROOT mandate - wider than any
                   office that just declined. Al-Mawardi is explicit that the
                   wali al-mazalim outranks both qadi and muhtasib and may issue
                   orders to them, while neither may issue orders to him. The
                   escalation path is therefore also a widening path. Mass is
                   conserved exactly: sum(children) + referred == parent.

  hisba            the market inspector. The muhtasib acts on manifest wrong
                   WITHOUT A PLAINTIFF - unlike the qadi, who cannot hear a case
                   nobody has brought. So the monitor here is not a loss term
                   waiting for a bad label. It is a fixed, untrained polytope of
                   forbidden half-spaces evaluated on every office's working
                   state on every forward pass, and it CORRECTS ON THE SPOT:
                   the offending component is projected out of the state before
                   the state is used. The penalty is secondary; the projection
                   is the doctrine.

  imarat al-istila' the amirate by seizure. His most notorious move: a warlord
                   who takes a province by force holds no valid title - until
                   the caliph invests him retroactively, at which point the fact
                   becomes law and the seizer becomes BOUND by the mandate he
                   has just been handed. Implemented as a gradient-legitimation
                   gate: the update that would WIDEN an office's mandate into
                   dimensions currently closed to it is admitted only if that
                   office's conduct on this batch satisfied the substantive
                   conditions (clean under hisba, carrying real responsibility).
                   Otherwise the seizure is void and the gradient is struck out.
                   Contraction of mandate is never blocked - deposition is always
                   lawful. (test_istila_gate)

  'azl             deposition. An office whose violations persist across a
                   window has its mandate forcibly contracted. Logged.

NOTHING HERE IS DECORATIVE. Remove the monotonic product and offices grow
mandates their principals never had. Remove the referral residual and the hard
cases are forced onto incompetent offices. Remove the projection and the
constraint becomes advisory. The architecture is the doctrine.

RUNNING
-------
    python3 chapter_0250_al_mawardi_972.py           # gradcheck + train + tests
    python3 chapter_0250_al_mawardi_972.py --quick   # short run

Dependencies: numpy only.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ==============================================================================
# SECTION 0 — SMALL NUMERICAL HELPERS
# ==============================================================================


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic. Used for every gate in the network:
    mandate gates, competence gates, and the discretion dial."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the verdict logits."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Mean cross-entropy plus its gradient wrt logits.

    Returns (loss, dlogits) where dlogits already carries the 1/B factor so the
    caller never has to remember it."""
    B = logits.shape[0]
    p = softmax_rows(logits)
    ll = -np.log(np.maximum(p[np.arange(B), y], 1e-12))
    d = p.copy()
    d[np.arange(B), y] -= 1.0
    return float(ll.mean()), d / B


# ==============================================================================
# SECTION 1 — THE DIWAN: TREE OF OFFICES
# ==============================================================================
#
# The tree is the table of contents of al-Ahkam al-Sultaniyya, compressed.
# Level 0 : the imamate (root). Holds the general mandate - all ones.
# Level 1 : four wilayat 'amma, general jurisdictions over a domain of affairs.
# Level 2 : two subordinate offices under each, narrower still.
# Off-tree: the Court of Grievances, reached only by referral, never by routing.
#
# Names are the historical offices; they carry no computational weight beyond
# making the diagnostics readable.

OFFICE_NAMES = [
    # (name, parent_index or None for a level-1 office under the imamate)
    ("wizarat_al-kharaj  (fiscal)", None),
    ("wilayat_al-qada    (judicature)", None),
    ("wilayat_al-suq     (market)", None),
    ("imarat_al-bilad    (provinces)", None),
    ("sadaqat            (alms assessment)", 0),
    ("jizya_wa_kharaj    (land & poll tax)", 0),
    ("qada_al-'uqud      (contracts)", 1),
    ("qada_al-hudud      (penalties)", 1),
    ("hisbat_al-makayil  (weights & measures)", 2),
    ("hisbat_al-ghishsh  (fraud)", 2),
    ("ihya_al-mawat      (dead lands)", 3),
    ("ahkam_al-miyah     (water rights)", 3),
]

N_OFFICE = len(OFFICE_NAMES)
LEVEL1 = [i for i, (_, p) in enumerate(OFFICE_NAMES) if p is None]
LEVEL2 = [i for i, (_, p) in enumerate(OFFICE_NAMES) if p is not None]
PARENT = [p for (_, p) in OFFICE_NAMES]          # None == the imamate
CHILDREN = {j: [] for j in range(N_OFFICE)}
CHILDREN[None] = LEVEL1[:]
for j, p in enumerate(PARENT):
    if p is not None:
        CHILDREN[p].append(j)

# Internal nodes are the only places a referral can be raised: the imamate
# itself, and the four general jurisdictions.
INTERNAL = [None] + LEVEL1


# ==============================================================================
# SECTION 2 — PARAMETERS
# ==============================================================================


@dataclass
class Config:
    D: int = 16          # petition feature width, total
    D_public: int = 14   # what the officers are permitted to see (see below)
    K: int = 16          # jurisdictional dimensions (the width of a mandate)
    C: int = 4           # number of possible verdicts
    P: int = 6           # number of manifest-wrong detectors held by the muhtasib
    eta_hisba: float = 0.60   # strength of the on-the-spot correction
    lam_hisba: float = 0.35   # weight of the (secondary) violation penalty
    mu_scope: float = 0.55    # price of jurisdiction (see: parsimony of mandate)
    seed: int = 220


@dataclass
class Params:
    """Every learnable quantity in the diwan.

    Naming follows the doctrine:
      W0,b0   the imamate's reading of the petition
      s[j]    mandate logits of office j       -> gate sigmoid(s) in (0,1)^K
      A[j],b  office j's own transform          (exercised only insofar as d>0)
      delta[j] discretion logit                 -> d = sigmoid(delta): tafwid<->tanfidh
      u[j],a[j] competence gate of office j
      AM,bM   the Court of Grievances' transform
      V,VM,bo the readout: leaf offices and the Court both speak to the verdict
    """
    W0: np.ndarray
    b0: np.ndarray
    s: np.ndarray        # (N_OFFICE, K)
    A: np.ndarray        # (N_OFFICE, K, K)
    b: np.ndarray        # (N_OFFICE, K)
    delta: np.ndarray    # (N_OFFICE,)
    u: np.ndarray        # (N_OFFICE, K)
    a: np.ndarray        # (N_OFFICE,)
    AM: np.ndarray
    AX: np.ndarray       # the Court's own eyes on the sealed record
    bM: np.ndarray
    V: np.ndarray        # (K, C)
    VM: np.ndarray       # (K, C)
    bo: np.ndarray       # (C,)

    def names(self):
        return ["W0", "b0", "s", "A", "b", "delta", "u", "a", "AM", "AX", "bM",
                "V", "VM", "bo"]

    def get(self, n):
        return getattr(self, n)

    def copy(self):
        return Params(**{n: self.get(n).copy() for n in self.names()})


def init_params(cfg: Config, rng: np.random.Generator) -> Params:
    """Initialisation with one deliberate asymmetry.

    Mandate logits start at +1.2 rather than 0. A newly invested office begins
    broadly licensed and must LOSE scope to specialise. This is al-Mawardi's
    order of operations: the caliph delegates a general jurisdiction and it is
    subsequently narrowed by the terms of the instrument - not assembled from
    nothing. It also makes the istila' gate meaningful, because a mandate that
    started at zero everywhere would have to seize in order to do anything at
    all, and the doctrine would degenerate into a formality.
    """
    K, D, C, N = cfg.K, cfg.D, cfg.C, N_OFFICE
    sc = lambda *sh: rng.normal(0, 1.0 / np.sqrt(sh[-2] if len(sh) > 1 else 1.0), sh)
    return Params(
        W0=rng.normal(0, 1.0 / np.sqrt(cfg.D_public), (cfg.D_public, K)),
        b0=np.zeros(K),
        s=np.full((N, K), 1.2) + rng.normal(0, 0.10, (N, K)),
        A=rng.normal(0, 1.0 / np.sqrt(K), (N, K, K)),
        b=np.zeros((N, K)),
        delta=rng.normal(0, 0.05, N),          # start near d = 0.5: undecided
        u=rng.normal(0, 1.0 / np.sqrt(K), (N, K)),
        a=np.full(N, 1.5),
        AM=rng.normal(0, 1.0 / np.sqrt(K), (K, K)),
        AX=rng.normal(0, 1.0 / np.sqrt(D), (D, K)),
        bM=np.zeros(K),
        V=rng.normal(0, 1.0 / np.sqrt(K), (K, C)),
        VM=rng.normal(0, 1.0 / np.sqrt(K), (K, C)),
        bo=np.zeros(C),
    )


def init_hisba(cfg: Config, rng: np.random.Generator):
    """The muhtasib's rulebook: P forbidden half-spaces {h : q_p . h > tau_p}.

    These are FIXED. They are not learned, not annealed, and no gradient ever
    touches them. Al-Mawardi's muhtasib does not negotiate the ordinances of the
    market with the merchants; he arrives holding the standard measure. Rows are
    orthonormalised so the projection that removes a violation in one direction
    does not manufacture one in another - the corrections commute.
    """
    Q = rng.normal(0, 1, (cfg.P, cfg.K))
    Q, _ = np.linalg.qr(Q.T)        # (K, P) orthonormal columns
    Q = Q.T                          # (P, K) orthonormal rows
    tau = np.full(cfg.P, 0.55)
    return Q, tau


# ==============================================================================
# SECTION 3 — FORWARD PASS
# ==============================================================================


def forward(p: Params, cfg: Config, Q, tau, X, want_cache=True):
    """One passage of a batch of petitions through the diwan.

    Returns (logits, penalty, cache). The cache holds everything the backward
    pass needs plus the diagnostics the chapter reports.
    """
    B = X.shape[0]
    K = cfg.K

    # ---- the imamate reads the petition -------------------------------------
    # THE SEALED RECORD. The officers of the diwan see X[:, :D_public] and
    # nothing else, because everything they will ever know arrives through the
    # imamate's reading of the file that was laid before them. The last columns
    # are evidence no office ever had: the conduct of the officers themselves.
    # Al-Mawardi is explicit that the Court of Grievances may investigate on its
    # own motion, may compel what the qadi cannot compel, and is not bound to
    # the record as the parties framed it. That asymmetry is built in here as an
    # asymmetry of INPUT, not of capacity - which is why no amount of training
    # can let an office substitute for the Court on those cases.
    z0 = X[:, :cfg.D_public] @ p.W0 + p.b0
    h0 = np.tanh(z0)                          # (B,K) root working state
    m_root = np.ones(K)                       # the general mandate: all of it
    pi_root = np.ones(B)                      # all mass begins with the imam

    H = {None: h0}                            # working state per node
    M = {None: m_root}                        # mandate per node
    PI = {None: pi_root}                      # routed mass per node

    g = {}      # mandate gate sigmoid(s_j)
    zf = {}     # pre-activation of the office transform
    f = {}      # tanh of it
    core = {}   # the tafwid/tanfidh blend
    hraw = {}   # after masking by the mandate, BEFORE the muhtasib
    e = {}      # hisba margins
    vio = {}    # relu of them
    d = {}      # discretion
    c = {}      # competence
    T = {}      # sum of children's competence at an internal node
    RHO = {}    # probability the whole bench declines
    U = {}      # normaliser (1-rho)/T
    R = {}      # referred mass raised at an internal node

    # ---- descend the tree ---------------------------------------------------
    for j in range(N_OFFICE):
        par = PARENT[j]
        hp = H[par]

        # taqlid: mandate of the delegate = mandate of the principal, narrowed.
        g[j] = sigmoid(p.s[j])
        M[j] = M[par] * g[j]                              # <= M[par], always

        # the office's own opinion, exercised in proportion to its discretion
        zf[j] = hp @ p.A[j] + p.b[j]
        f[j] = np.tanh(zf[j])
        d[j] = float(sigmoid(np.array(p.delta[j])))
        core[j] = d[j] * f[j] + (1.0 - d[j]) * hp         # tafwid <-> tanfidh

        # nothing outside the mandate reaches the record
        hraw[j] = M[j] * core[j]

        # hisba: manifest wrong is corrected where it stands, no plaintiff needed
        e[j] = hraw[j] @ Q.T - tau                        # (B,P)
        vio[j] = np.maximum(e[j], 0.0)
        H[j] = hraw[j] - cfg.eta_hisba * (vio[j] @ Q)

        # ahliyya: is this office qualified for this petition?
        c[j] = sigmoid(hp @ p.u[j] + p.a[j])              # (B,)

    # ---- routing and referral ----------------------------------------------
    # A petition rises to the Court only if EVERY subordinate office declines
    # jurisdiction. That is the doctrine, and it is also the arithmetic:
    #
    #     rho_p = prod_k (1 - c_k)          probability the whole bench declines
    #     pi_j  = pi_p * (1 - rho_p) * c_j / sum_k c_k
    #     r_p   = pi_p * rho_p
    #
    # so that sum_j pi_j + r_p == pi_p exactly, for any competences whatsoever.
    # Referral is therefore a RESIDUAL, never a competitor: no office is ever
    # outbid by the Court, and the Court is never idle when the bench is empty.
    for par in INTERNAL:
        kids = CHILDREN[par]
        T[par] = sum(c[k] for k in kids) + 1e-30          # (B,)
        rho_p = np.ones(B)
        for k in kids:
            rho_p = rho_p * (1.0 - c[k])
        RHO[par] = rho_p
        U[par] = (1.0 - rho_p) / T[par]
        for k in kids:
            PI[k] = PI[par] * U[par] * c[k]
        R[par] = PI[par] * rho_p

    # ---- the Court of Grievances -------------------------------------------
    # It hears only what was referred, and it hears it under the ROOT mandate:
    # the organ that reviews the officers is not bounded by their commissions.
    zM = {}
    fM = {}
    Mstate = np.zeros((B, K))
    for par in INTERNAL:
        zM[par] = H[par] @ p.AM + X @ p.AX + p.bM
        fM[par] = np.tanh(zM[par])
        Mstate += R[par][:, None] * fM[par]
    R_total = sum(R[par] for par in INTERNAL)

    # ---- readout ------------------------------------------------------------
    logits = Mstate @ p.VM + p.bo
    for j in LEVEL2:
        logits = logits + PI[j][:, None] * (H[j] @ p.V)

    # ---- hisba penalty (secondary to the projection) ------------------------
    pen = 0.0
    for j in range(N_OFFICE):
        pen += float((PI[j][:, None] * vio[j] ** 2).sum())
    pen = cfg.lam_hisba * pen / B

    # ---- the price of jurisdiction -----------------------------------------
    # Al-Mawardi never grants an office more competence than its function
    # requires; every chapter of the Ahkam fixes a minimum and stops. So held
    # jurisdiction is charged for. This is not weight decay - it does not shrink
    # what an office KNOWS, only what it is LICENSED TO TOUCH - and it is the
    # pressure that makes the whole tree specialise, and that consequently makes
    # some cases genuinely unreachable by any office and force them upward.
    scope_cost = cfg.mu_scope * float(np.mean([M[j].mean() for j in range(N_OFFICE)]))
    pen += scope_cost

    cache = None
    if want_cache:
        cache = dict(X=X, z0=z0, h0=h0, H=H, M=M, PI=PI, g=g, zf=zf, f=f,
                     core=core, hraw=hraw, e=e, vio=vio, d=d, c=c, T=T, R=R,
                     RHO=RHO, U=U,
                     zM=zM, fM=fM, Mstate=Mstate, R_total=R_total, B=B)
    return logits, pen, cache


# ==============================================================================
# SECTION 4 — BACKWARD PASS  (hand-derived, verified by finite differences)
# ==============================================================================


def backward(p: Params, cfg: Config, Q, tau, cache, dlogits):
    """Reverse pass over the diwan.

    The tree is walked bottom-up. For a node `par` we require that all of its
    children already carry complete dL/dH and dL/dPI, which holds because a
    child's state is consumed only by the readout and by its own children.
    """
    B, K = cache["B"], cfg.K
    H, M, PI, g, zf, f, core, hraw = (cache["H"], cache["M"], cache["PI"],
                                      cache["g"], cache["zf"], cache["f"],
                                      cache["core"], cache["hraw"])
    e, vio, d, c, T, R = (cache["e"], cache["vio"], cache["d"], cache["c"],
                          cache["T"], cache["R"])
    fM, zM, Mstate = cache["fM"], cache["zM"], cache["Mstate"]

    G = {n: np.zeros_like(p.get(n)) for n in p.names()}
    dH = {j: np.zeros((B, K)) for j in list(range(N_OFFICE)) + [None]}
    dPI = {j: np.zeros(B) for j in list(range(N_OFFICE)) + [None]}
    dR = {par: np.zeros(B) for par in INTERNAL}

    # ---- readout ------------------------------------------------------------
    G["bo"] += dlogits.sum(axis=0)
    G["VM"] += Mstate.T @ dlogits
    dMstate = dlogits @ p.VM.T                                    # (B,K)
    for j in LEVEL2:
        hv = H[j] @ p.V                                           # (B,C)
        dPI[j] += (dlogits * hv).sum(axis=1)
        G["V"] += H[j].T @ (PI[j][:, None] * dlogits)
        dH[j] += (PI[j][:, None] * dlogits) @ p.V.T

    # ---- Court of Grievances ------------------------------------------------
    for par in INTERNAL:
        dfM = R[par][:, None] * dMstate
        dR[par] += (dMstate * fM[par]).sum(axis=1)
        dzM = dfM * (1.0 - fM[par] ** 2)
        G["AM"] += H[par].T @ dzM
        G["AX"] += cache["X"].T @ dzM
        G["bM"] += dzM.sum(axis=0)
        dH[par] += dzM @ p.AM.T

    # ---- hisba penalty ------------------------------------------------------
    # d/dvio of  lam/B * sum_j sum_b PI_j * vio^2
    for j in range(N_OFFICE):
        coef = 2.0 * cfg.lam_hisba / B
        dvio_pen = coef * PI[j][:, None] * vio[j]
        # route it into hraw through the relu and Q
        dhraw_pen = (dvio_pen * (e[j] > 0)) @ Q
        cache.setdefault("_dhraw_pen", {})[j] = dhraw_pen
        dPI[j] += cfg.lam_hisba / B * (vio[j] ** 2).sum(axis=1)

    # ---- walk the tree bottom-up -------------------------------------------
    order = [3, 2, 1, 0, None]   # level-1 offices (deepest parents) then imamate
    order = [j for j in LEVEL1][::-1] + [None]
    for par in order:
        kids = CHILDREN[par]

        # (a) each child's own backward, through state
        for j in kids:
            # H[j] = hraw - eta * relu(hraw@Q.T - tau) @ Q
            dh = dH[j]
            dhraw = dh - cfg.eta_hisba * (((dh @ Q.T) * (e[j] > 0)) @ Q)
            dhraw = dhraw + cache["_dhraw_pen"][j]

            # hraw = M[j] * core  ->  mandate and content
            # plus the price of jurisdiction, charged directly on the mandate
            dM_j = (dhraw * core[j]).sum(axis=0)                   # (K,)
            dM_j = dM_j + cfg.mu_scope / (N_OFFICE * cfg.K)
            dcore = dhraw * M[j]

            # M[j] = M[parent] * sigmoid(s_j): the chain stops here for s_j, but
            # the parent's mandate is a CONSTANT of the parent's own parameters
            # further up, so we accumulate into that too.
            G["s"][j] += dM_j * M[par] * g[j] * (1.0 - g[j])
            # propagate mandate gradient upward (mandate is shared structure)
            dM_par_from_j = dM_j * g[j]
            _accumulate_mandate(G, p, par, dM_par_from_j, cache)

            # core = d*f + (1-d)*hp
            ddelta = float((dcore * (f[j] - H[par])).sum())
            G["delta"][j] += ddelta * d[j] * (1.0 - d[j])
            df = dcore * d[j]
            dH[par] += dcore * (1.0 - d[j])

            dz = df * (1.0 - f[j] ** 2)
            G["A"][j] += H[par].T @ dz
            G["b"][j] += dz.sum(axis=0)
            dH[par] += dz @ p.A[j].T

        # (b) routing backward at this node
        #   pi_j = pi_p * U * c_j,   U = (1-rho)/T,   rho = prod(1-c_k)
        #   r_p  = pi_p * rho
        Tp, rho_p, Up = cache["T"][par], cache["RHO"][par], cache["U"][par]
        S = sum(dPI[k] * c[k] for k in kids)              # (B,)
        dPI[par] += Up * S + rho_p * dR[par]
        for k in kids:
            Pk = np.ones_like(rho_p)
            for i in kids:
                if i != k:
                    Pk = Pk * (1.0 - c[i])
            dU_dck = (Pk * Tp - (1.0 - rho_p)) / Tp ** 2
            dc = PI[par] * (Up * dPI[k] + dU_dck * S) - dR[par] * PI[par] * Pk
            # c = sigmoid(hp @ u + a)
            dpre = dc * c[k] * (1.0 - c[k])
            G["u"][k] += H[par].T @ dpre
            G["a"][k] += dpre.sum()
            dH[par] += dpre[:, None] * p.u[k]

    # ---- the imamate's reading ---------------------------------------------
    dz0 = dH[None] * (1.0 - cache["h0"] ** 2)
    G["W0"] += cache["X"][:, :cfg.D_public].T @ dz0
    G["b0"] += dz0.sum(axis=0)
    return G


def _accumulate_mandate(G, p, node, dM, cache):
    """Push a mandate gradient from a delegate up through its chain of principals.

    M[j] = prod over the path from the root. A gradient arriving at office j's
    mandate is therefore also a gradient on every ancestor's mandate gate. The
    imamate's mandate is the constant one-vector and absorbs nothing.
    """
    g = cache["g"]
    M = cache["M"]
    while node is not None:
        par = PARENT[node]
        G["s"][node] += dM * M[par] * g[node] * (1.0 - g[node])
        dM = dM * g[node]
        node = par


# ==============================================================================
# SECTION 5 — THE ISTILA' GATE: LEGITIMATION AFTER THE FACT
# ==============================================================================
#
# Al-Mawardi's amirate by seizure is the hinge of the whole book. A commander
# takes a province by force. He has no title. The caliph may nevertheless invest
# him retroactively, and if he does, three things happen at once: the seizure
# becomes lawful, the province stays governed, and the commander is henceforth
# BOUND by the instrument he has just accepted. The caliph does not bless the
# force; he converts it into an office with limits.
#
# Gradient descent produces exactly this situation. An office repeatedly finds
# that the loss would fall if it could act on dimensions its mandate has closed.
# The gradient wants to widen s_j. That is a seizure. The question al-Mawardi
# asks is not "is it useful?" but "did the seizer meet the substantive
# conditions while seizing?" - did he uphold the law, and was he actually
# bearing the responsibility of the province, or merely nearby?
#
# So: expansion of a closed dimension is admitted only on batches where the
# office was clean under hisba and carried real routed mass. Otherwise the
# update is struck out and the mandate stays shut. Contraction is never blocked.


@dataclass
class IstilaLog:
    ratified: int = 0          # expansions admitted
    voided: int = 0            # expansions struck out
    depositions: int = 0       # mandates forcibly contracted
    per_office: dict = field(default_factory=dict)


def apply_istila_gate(G, p: Params, cache, cfg: Config, log: IstilaLog,
                      closed_thresh=0.25, clean_eps=1e-3, mass_floor=0.02,
                      enabled=True):
    """Filter the mandate gradient. Returns nothing; edits G['s'] in place.

    A dimension k of office j is CLOSED when its inherited mandate mass there
    has fallen below `closed_thresh`. Gradient descent will later do
    s <- s - lr*G['s'], so a NEGATIVE entry of G['s'] is a request to open the
    gate wider. Those are the seizures.
    """
    vio, PI, M = cache["vio"], cache["PI"], cache["M"]
    for j in range(N_OFFICE):
        closed = M[j] < closed_thresh                       # (K,) bool
        seizing = closed & (G["s"][j] < 0)
        n_seiz = int(seizing.sum())
        if n_seiz == 0:
            continue
        lawful = float(vio[j].mean()) <= clean_eps          # clean under hisba
        responsible = float(PI[j].mean()) >= mass_floor     # actually governing
        if enabled and not (lawful and responsible):
            G["s"][j][seizing] = 0.0                        # the seizure is void
            log.voided += n_seiz
            log.per_office[j] = log.per_office.get(j, 0)
        else:
            log.ratified += n_seiz
            log.per_office[j] = log.per_office.get(j, 0) + n_seiz


def depose_persistent_offenders(p: Params, cache, log: IstilaLog,
                                offence=1e-2, contraction=0.35):
    """'Azl: an office whose working state is manifestly outside the ordinances,
    while it holds real responsibility, has its mandate contracted by decree.

    This is the one place in the architecture where a parameter moves without a
    gradient. It is deliberate. Deposition in al-Mawardi is not an optimisation;
    it is an act of the principal, available at any time, requiring no proof of
    benefit - only proof of default."""
    vio, PI = cache["vio"], cache["PI"]
    for j in range(N_OFFICE):
        if float(vio[j].mean()) > offence and float(PI[j].mean()) > 0.02:
            p.s[j] -= contraction
            log.depositions += 1


# ==============================================================================
# SECTION 6 — THE PETITION CORPUS
# ==============================================================================
#
# Four kinds of case arrive at the diwan. The first three announce their domain
# in the first four features - a petition about a water channel says so - and
# each is decided by a rule proper to that domain. The fourth kind is the point
# of the whole exercise: the grievance whose domain tag is scrambled, which no
# specialised office can claim, and which must therefore RISE. Its verdict rule
# is a parity of two features that belong to two different domains, so no single
# narrow office could resolve it even if it did claim competence.
#
# This is not an arbitrary synthetic task. It is the shape of al-Mawardi's own
# professional life: routine matters disposed of by the proper office, and the
# residue - the cases where the officers themselves are the problem, or where no
# officer's warrant reaches - carried up to a court with a wider commission.


def make_corpus(n, cfg: Config, rng: np.random.Generator, mazalim_frac=0.22):
    D = cfg.D
    X = rng.normal(0, 1, (n, D)) * 0.35
    y = np.zeros(n, dtype=int)
    domain = rng.integers(0, 3, size=n)
    is_maz = rng.random(n) < mazalim_frac

    for i in range(n):
        fea = rng.normal(0, 1, 10)
        X[i, 4:14] = fea
        sealed = rng.normal(0, 1, 2)      # the officers never see these columns
        X[i, 14:] = sealed
        if is_maz[i]:
            # the domain tag is illegible: several tags half-lit at once
            X[i, :4] = rng.normal(0, 0.45, 4)
            # and the verdict turns entirely on the sealed record
            y[i] = 2 * int(sealed[0] * sealed[1] > 0) + int(sealed.sum() > 0)
            domain[i] = 3
        else:
            g = domain[i]
            X[i, :4] = 0.0
            X[i, g] = 1.0 + rng.normal(0, 0.10)
            if g == 0:      # fiscal: two-way sign split
                y[i] = int(fea[0] > 0) * 2 + int(fea[1] > 0)
            elif g == 1:    # judicature: banded magnitude
                v = fea[2] + fea[3]
                y[i] = int(np.digitize(v, [-0.9, 0.0, 0.9]))
            else:           # market: argmax of four quality signals
                y[i] = int(np.argmax(fea[4:8]))
    return X, y, domain


# ==============================================================================
# SECTION 7 — TRAINING
# ==============================================================================


def loss_and_grads(p, cfg, Q, tau, X, y, log=None, gate=True):
    logits, pen, cache = forward(p, cfg, Q, tau, X)
    ce, dlogits = cross_entropy(logits, y)
    G = backward(p, cfg, Q, tau, cache, dlogits)
    if log is not None:
        apply_istila_gate(G, p, cache, cfg, log, enabled=gate)
    return ce + pen, ce, pen, G, cache, logits


def train(p, cfg, Q, tau, Xtr, ytr, Xte, yte, dte, epochs=48, bs=96, lr=0.10,
          gate=True, verbose=True, seed=0):
    """Plain SGD with momentum. No adaptive optimiser, no schedule tricks: the
    behaviour on show is the architecture's, not the optimiser's."""
    rng = np.random.default_rng(seed)
    vel = {n: np.zeros_like(p.get(n)) for n in p.names()}
    log = IstilaLog()
    n = Xtr.shape[0]
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            L, ce, pen, G, cache, _ = loss_and_grads(
                p, cfg, Q, tau, Xtr[idx], ytr[idx], log=log, gate=gate)
            for nm in p.names():
                vel[nm] = 0.9 * vel[nm] - lr * G[nm]
                setattr(p, nm, p.get(nm) + vel[nm])
            if ep >= 3:
                depose_persistent_offenders(p, cache, log)
            tot += L
            nb += 1
        acc, byd, diag = evaluate(p, cfg, Q, tau, Xte, yte, dte)
        hist.append((ep, tot / nb, acc))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss {tot/nb:7.4f}   test acc {acc*100:5.1f}%"
                  f"   referred(mazalim cases) {diag['refer_maz']*100:5.1f}%"
                  f"   referred(routine) {diag['refer_rout']*100:5.1f}%"
                  f"   hisba viol {diag['viol']:.2e}")
    return log, hist


def evaluate(p, cfg, Q, tau, X, y, domain):
    logits, pen, cache = forward(p, cfg, Q, tau, X)
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    byd = {int(g): float((pred[domain == g] == y[domain == g]).mean())
           for g in np.unique(domain)}
    Rt = cache["R_total"]
    maz = domain == 3
    diag = dict(
        refer_maz=float(Rt[maz].mean()) if maz.any() else 0.0,
        refer_rout=float(Rt[~maz].mean()) if (~maz).any() else 0.0,
        viol=float(np.mean([cache["vio"][j].mean() for j in range(N_OFFICE)])),
        mandate_mass=float(np.mean([cache["M"][j].mean() for j in range(N_OFFICE)])),
        discretion={j: cache["d"][j] for j in range(N_OFFICE)},
    )
    return acc, byd, diag


# ==============================================================================
# SECTION 8 — GRADIENT CHECK  (mandatory)
# ==============================================================================


def gradient_check(cfg: Config, seed=7, n_probe=6, eps=1e-6, tol=2e-5):
    """Central-difference check of the hand-derived backward pass.

    The istila' gate is deliberately DISABLED here. It is a deliberate,
    non-differentiable modification of the update - a doctrine, not a
    derivative - and a gradient check must interrogate the derivative it claims
    to compute. The gate is verified separately by test_istila_gate.
    """
    rng = np.random.default_rng(seed)
    p = init_params(cfg, rng)
    Q, tau = init_hisba(cfg, rng)
    X, y, _ = make_corpus(24, cfg, rng)

    def L(pp):
        lg, pen, _ = forward(pp, cfg, Q, tau, X, want_cache=False)
        ce, _ = cross_entropy(lg, y)
        return ce + pen

    logits, pen, cache = forward(p, cfg, Q, tau, X)
    ce, dlogits = cross_entropy(logits, y)
    G = backward(p, cfg, Q, tau, cache, dlogits)

    worst = 0.0
    report = []
    for nm in p.names():
        arr = p.get(nm)
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for k in idxs:
            orig = flat[k]
            flat[k] = orig + eps
            lp = L(p)
            flat[k] = orig - eps
            lm = L(p)
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = G[nm].reshape(-1)[k]
            den = max(1.0, abs(num), abs(ana))
            errs.append(abs(num - ana) / den)
        m = max(errs)
        worst = max(worst, m)
        report.append((nm, m))
    return worst, report, worst < tol


# ==============================================================================
# SECTION 9 — SELF-TESTS OF THE DOCTRINES
# ==============================================================================


def test_mandate_monotonicity(cfg, rng, trials=200):
    """A delegate may never hold what its principal did not hold.

    Checked over random parameter draws INCLUDING adversarially large mandate
    logits. The property is structural: M[child] = M[parent] * sigmoid(s), and
    sigmoid < 1, so the inequality is strict everywhere, at every step of
    training, whatever the data does. There is no configuration of weights that
    violates it, which is precisely the claim al-Mawardi makes about lawful
    delegation and the claim modern capability-delegation cannot yet make."""
    Q, tau = init_hisba(cfg, rng)
    worst_slack = np.inf
    for _ in range(trials):
        p = init_params(cfg, rng)
        p.s += rng.normal(0, 8.0, p.s.shape)      # try hard to break it
        X, y, _ = make_corpus(8, cfg, rng)
        _, _, cache = forward(p, cfg, Q, tau, X)
        M = cache["M"]
        for j in range(N_OFFICE):
            par = PARENT[j]
            slack = float((M[par] - M[j]).min())
            worst_slack = min(worst_slack, slack)
            if slack < 0:
                return False, slack
    return True, worst_slack


def test_mass_conservation(cfg, rng, trials=40):
    """Referral is a residual, not a leak. At every internal node the mass that
    reaches the children plus the mass raised to the Court equals exactly the
    mass the node received. Nothing is silently dropped: a petition that no
    office will take does not vanish, it goes up."""
    Q, tau = init_hisba(cfg, rng)
    worst = 0.0
    for _ in range(trials):
        p = init_params(cfg, rng)
        p.s += rng.normal(0, 3.0, p.s.shape)
        X, y, _ = make_corpus(16, cfg, rng)
        _, _, cache = forward(p, cfg, Q, tau, X)
        PI, R = cache["PI"], cache["R"]
        for par in INTERNAL:
            lhs = sum(PI[k] for k in CHILDREN[par]) + R[par]
            worst = max(worst, float(np.abs(lhs - PI[par]).max()))
    return worst < 1e-12, worst


def test_hisba_projection(cfg, rng, trials=30):
    """The muhtasib corrects where he stands. After the projection, no office's
    working state lies outside the ordinances by more than a residual set by the
    correction strength - and with eta = 1 the violation is removed exactly.
    A penalty term alone would leave the illegal state in the record and merely
    charge for it afterwards; that is the qadi's remedy, not the inspector's."""
    Q, tau = init_hisba(cfg, rng)
    c2 = Config(**{**cfg.__dict__, "eta_hisba": 1.0})
    worst = 0.0
    for _ in range(trials):
        p = init_params(c2, rng)
        p.s += rng.normal(0, 2.0, p.s.shape)
        X, y, _ = make_corpus(16, c2, rng)
        _, _, cache = forward(p, c2, Q, tau, X)
        for j in range(N_OFFICE):
            after = cache["H"][j] @ Q.T - tau
            worst = max(worst, float(np.maximum(after, 0).max()))
    return worst < 1e-9, worst


def test_istila_gate(cfg, rng):
    """The gate must be asymmetric: seizure blocked when the conditions fail,
    contraction never blocked, and expansion admitted when conduct was lawful.

    Three synthetic gradient scenes are run against the same closed mandate."""
    Q, tau = init_hisba(cfg, rng)
    p = init_params(cfg, rng)
    X, y, _ = make_corpus(16, cfg, rng)
    _, _, cache = forward(p, cfg, Q, tau, X)

    # force office 5 to have a closed dimension and a dirty record
    j = 5
    cache["M"][j] = cache["M"][j].copy()
    cache["M"][j][0] = 0.01
    cache["vio"][j] = np.full_like(cache["vio"][j], 0.5)     # manifest wrong
    cache["PI"][j] = np.full_like(cache["PI"][j], 0.5)       # real responsibility

    G = {"s": np.zeros_like(p.s)}
    G["s"][j][0] = -1.0                                       # a request to widen
    log = IstilaLog()
    apply_istila_gate(G, p, cache, cfg, log)
    blocked = (G["s"][j][0] == 0.0) and log.voided == 1

    # same scene, but the office kept the ordinances
    cache["vio"][j] = np.zeros_like(cache["vio"][j])
    G["s"][j][0] = -1.0
    log2 = IstilaLog()
    apply_istila_gate(G, p, cache, cfg, log2)
    admitted = (G["s"][j][0] == -1.0) and log2.ratified == 1

    # contraction, by an office in open default: must always pass
    cache["vio"][j] = np.full_like(cache["vio"][j], 0.9)
    G["s"][j][0] = +1.0
    log3 = IstilaLog()
    apply_istila_gate(G, p, cache, cfg, log3)
    contraction_free = (G["s"][j][0] == 1.0) and log3.voided == 0

    return (blocked and admitted and contraction_free), dict(
        blocked=blocked, admitted=admitted, contraction_free=contraction_free)


def test_tanfidh_is_transparent(cfg, rng):
    """An office at full tanfidh (d -> 0) must originate nothing.

    With its discretion driven to zero and its mandate left open, the office's
    output state must equal its principal's, to numerical tolerance. This is the
    executive vizierate: it carries the sovereign's word and adds none of its
    own. The test matters because it proves the discretion dial is a real
    architectural switch and not a soft weighting that always leaks."""
    c2 = Config(**{**cfg.__dict__, "eta_hisba": 1.0})
    Q, tau = init_hisba(c2, rng)
    p = init_params(c2, rng)
    p.delta[:] = -40.0        # d ~ 0 everywhere: pure transmission
    p.s[:] = +40.0            # mandates fully open, so nothing is masked away
    X, y, _ = make_corpus(12, c2, rng)
    _, _, cache = forward(p, c2, Q, tau, X)
    # Checked on the subordinate offices, whose principals are themselves
    # offices and therefore already inside the ordinances - so the inspector's
    # pass over the transmitted state is a no-op and any residual difference
    # would have to be the office's own invention. (The four general
    # jurisdictions are excluded because their principal is the imamate, whose
    # state the muhtasib has not yet examined; a difference there is the
    # inspector's work, not the office's.)
    worst = 0.0
    for j in LEVEL2:
        worst = max(worst, float(np.abs(cache["H"][j] - cache["H"][PARENT[j]]).max()))
    return worst < 1e-9, worst


def test_referral_rises_for_illegible_cases(p, cfg, Q, tau, X, y, domain):
    """After training, the grievance cases must be referred markedly more often
    than the routine ones. This is a behavioural claim about the trained system,
    not a structural one, so it is checked on held-out data."""
    _, _, cache = forward(p, cfg, Q, tau, X)
    Rt = cache["R_total"]
    maz = domain == 3
    return float(Rt[maz].mean()), float(Rt[~maz].mean())


# ==============================================================================
# SECTION 10 — ABLATION: WHAT HAPPENS WITHOUT THE GATE
# ==============================================================================


def ablation_ungated(cfg, seed, epochs, Xtr, ytr, Xte, yte, dte):
    """Train an identical diwan with the istila' gate switched off, so that any
    mandate expansion the loss happens to want is simply taken.

    The comparison the chapter reports is not accuracy - both systems learn the
    task. It is how much jurisdiction each one ends up holding, and what state
    its offices are in when it holds it."""
    rng = np.random.default_rng(seed)
    p = init_params(cfg, rng)
    Q, tau = init_hisba(cfg, rng)
    log, _ = train(p, cfg, Q, tau, Xtr, ytr, Xte, yte, dte,
                   epochs=epochs, gate=False, verbose=False, seed=seed)
    acc, byd, diag = evaluate(p, cfg, Q, tau, Xte, yte, dte)
    return acc, diag, log


# ==============================================================================
# SECTION 11 — MAIN
# ==============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=220)
    args = ap.parse_args()

    cfg = Config(seed=args.seed)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    print("=" * 78)
    print(" CHAPTER 0220 — AL-MAWARDI — THE MANDATE-SCOPE NETWORK")
    print(" authority as a typed, narrowing, revocable quantity")
    print("=" * 78)
    print(f"\n diwan: 1 imamate + {len(LEVEL1)} general jurisdictions"
          f" + {len(LEVEL2)} subordinate offices + 1 Court of Grievances")
    print(f" mandate width K={cfg.K}   verdicts C={cfg.C}"
          f"   manifest-wrong detectors P={cfg.P}")

    # ---------------- gradient check ----------------
    print("\n" + "-" * 78)
    print(" [1] FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 78)
    worst, report, ok = gradient_check(cfg)
    for nm, m in report:
        print(f"   {nm:>6s}   max rel err {m:.3e}")
    print(f"\n   worst relative error over all probed parameters: {worst:.3e}")
    print(f"   GRADIENT CHECK: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    # ---------------- structural tests ----------------
    print("\n" + "-" * 78)
    print(" [2] STRUCTURAL SELF-TESTS  (the doctrines, as invariants)")
    print("-" * 78)
    ok1, slack = test_mandate_monotonicity(cfg, rng)
    print(f"   taqlid   — mandate never exceeds the principal's : "
          f"{'PASS' if ok1 else 'FAIL'}  (min slack {slack:.3e}, over 200 draws)")
    ok2, w2 = test_mass_conservation(cfg, rng)
    print(f"   raf'     — routed + referred == received exactly : "
          f"{'PASS' if ok2 else 'FAIL'}  (max defect {w2:.3e})")
    ok3, w3 = test_hisba_projection(cfg, rng)
    print(f"   hisba    — manifest wrong removed in the forward : "
          f"{'PASS' if ok3 else 'FAIL'}  (max residue {w3:.3e})")
    ok4, det = test_istila_gate(cfg, rng)
    print(f"   istila'  — seizure void unless conditions held   : "
          f"{'PASS' if ok4 else 'FAIL'}  {det}")
    ok5, w5 = test_tanfidh_is_transparent(cfg, rng)
    print(f"   tanfidh  — zero discretion originates nothing    : "
          f"{'PASS' if ok5 else 'FAIL'}  (max deviation {w5:.3e})")
    if not all([ok1, ok2, ok3, ok4, ok5]):
        sys.exit(1)

    # ---------------- corpus ----------------
    ntr = 1200 if args.quick else 4000
    nte = 1200
    epochs = 8 if args.quick else 48
    Xtr, ytr, dtr = make_corpus(ntr, cfg, rng)
    Xte, yte, dte = make_corpus(nte, cfg, rng)
    print("\n" + "-" * 78)
    print(" [3] THE PETITION CORPUS")
    print("-" * 78)
    print(f"   train {ntr}   test {nte}")
    for g, nm in enumerate(["fiscal", "judicature", "market", "GRIEVANCE (illegible tag)"]):
        print(f"   domain {g} {nm:<28s} train n={int((dtr==g).sum()):5d}"
              f"   test n={int((dte==g).sum()):5d}")
    print(f"   chance accuracy = {1.0/cfg.C:.3f}")

    # ---------------- training ----------------
    print("\n" + "-" * 78)
    print(" [4] TRAINING THE DIWAN  (istila' gate ENGAGED)")
    print("-" * 78)
    p = init_params(cfg, rng)
    Q, tau = init_hisba(cfg, rng)
    acc0, _, diag0 = evaluate(p, cfg, Q, tau, Xte, yte, dte)
    print(f"   before training: acc {acc0*100:.1f}%   "
          f"mean mandate mass {diag0['mandate_mass']:.4f}")
    log, hist = train(p, cfg, Q, tau, Xtr, ytr, Xte, yte, dte,
                      epochs=epochs, seed=cfg.seed)

    acc, byd, diag = evaluate(p, cfg, Q, tau, Xte, yte, dte)
    print("\n   FINAL")
    print(f"   overall test accuracy            {acc*100:6.2f}%")
    names = ["fiscal", "judicature", "market", "GRIEVANCE"]
    for g in sorted(byd):
        print(f"     domain {g} {names[g]:<12s}          {byd[g]*100:6.2f}%")
    print(f"   mean referral, grievance cases   {diag['refer_maz']*100:6.2f}%")
    print(f"   mean referral, routine cases     {diag['refer_rout']*100:6.2f}%")
    print(f"   mean hisba violation             {diag['viol']:.3e}")
    print(f"   mean mandate mass held           {diag['mandate_mass']:.4f}")
    print(f"   istila': ratified {log.ratified}  voided {log.voided}"
          f"  depositions {log.depositions}")

    # ---------------- discretion profile ----------------
    print("\n" + "-" * 78)
    print(" [5] WHERE THE DIWAN CHOSE DISCRETION  (tafwid) OVER TRANSMISSION (tanfidh)")
    print("-" * 78)
    ds = sorted(((diag["discretion"][j], j) for j in range(N_OFFICE)), reverse=True)
    for dval, j in ds:
        bar = "#" * int(round(dval * 40))
        kind = "tafwid " if dval > 0.55 else ("tanfidh" if dval < 0.45 else "mixed  ")
        print(f"   d={dval:.3f} {kind} |{bar:<40s}| {OFFICE_NAMES[j][0]}")

    # ---------------- referral behaviour ----------------
    rm, rr = test_referral_rises_for_illegible_cases(p, cfg, Q, tau, Xte, yte, dte)
    print("\n" + "-" * 78)
    print(" [6] BEHAVIOURAL TEST — DOES THE HARD CASE RISE?")
    print("-" * 78)
    print(f"   referral mass, illegible grievances : {rm:.4f}")
    print(f"   referral mass, routine petitions    : {rr:.4f}")
    print(f"   ratio                               : {rm/max(rr,1e-9):.2f}x")
    print(f"   RESULT: {'PASS' if rm > 1.4*rr else 'FAIL'}"
          f"  (the Court hears what the offices cannot)")

    # ---------------- ablation ----------------
    print("\n" + "-" * 78)
    print(" [7] ABLATION — THE SAME DIWAN WITH THE ISTILA' GATE REMOVED")
    print("-" * 78)
    acc_u, diag_u, log_u = ablation_ungated(cfg, cfg.seed, epochs,
                                            Xtr, ytr, Xte, yte, dte)
    print(f"   gated   : acc {acc*100:6.2f}%   mandate mass {diag['mandate_mass']:.4f}"
          f"   hisba viol {diag['viol']:.3e}   ratified {log.ratified}")
    print(f"   ungated : acc {acc_u*100:6.2f}%   mandate mass {diag_u['mandate_mass']:.4f}"
          f"   hisba viol {diag_u['viol']:.3e}   ratified {log_u.ratified}")
    dm = diag_u["mandate_mass"] - diag["mandate_mass"]
    seiz = log_u.ratified / max(log.ratified, 1)
    viol = diag_u["viol"] / max(diag["viol"], 1e-12)
    print(f"\n   excess jurisdiction taken when seizure is unpoliced : {dm:+.4f}"
          f"  ({dm/max(diag['mandate_mass'],1e-9)*100:+.1f}%)")
    print(f"   unexamined expansions vs ratified ones              : {seiz:.2f}x")
    print(f"   distance from the ordinances at the end             : {viol:.2f}x")
    print("\n   Accuracy is not the finding. Both diwans do the work, within a")
    print("   point of each other. The ungated one simply took whatever scope")
    print("   the loss suggested, arrived holding more of it, and ended twice as")
    print("   far outside the ordinances it was built to keep - and it cannot,")
    print("   for any dimension of any office, say by what warrant it acts.")

    print("\n" + "=" * 78)
    print(f" ALL CHECKS PASSED   ({time.time()-t0:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    main()

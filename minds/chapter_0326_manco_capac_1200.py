#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 WANKA:  the Wanka-Assay Ceque Network
 Figure 326 · Manco Capac (Manqu Qhapaq) · c. 1200 CE · Cusco valley, Peru
 # Encyclopedia of Lost Minds: Echoes on AI · Chapter 0326 · Manco Capac (1200-1235)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/ 
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
The founding narrative recorded by Sarmiento de Gamboa in 1572 is not a story
about a leader who decided where to build. It is a story about a leader who
carried an instrument that decided for him, and who arrived at the site alone
because every other capability in the party had been converted into stone.

Three operations are recoverable from that text, and from no other:

  (1) THE TWO-ROD ASSAY.  In Sarmiento's version the golden rod is thrown
      TWICE.  The first throw lands at Colcabamba and the hard ground refuses
      it — it will not sink.  The second falls at Guayanaypata and enters the
      earth softly.  The stopping rule is therefore not "the founder judged
      this site best."  It is a penetration event the founder cannot
      counterfeit, and it is only legible BECAUSE an earlier throw was
      rejected.  A search that has never failed has not been tested.

  (2) PETRIFICATION INTO WANKA.  Ayar Cachi, whose sling turned hills into
      ravines, is sealed inside the cave he came from.  Ayar Uchu freezes into
      a winged stone on Huanacauri — and the chroniclers are explicit that the
      stone form did not stop him speaking to his brothers.  Ayar Auca lands in
      the valley and becomes a wanka, one of the boundary stones by which
      possession of land was reckoned.  Surplus capability is neither destroyed
      nor carried forward live.  It is frozen into a read-only landmark that
      still emits, and against which the survivor's position is thereafter
      measured.

  (3) MAINTAINED RADIAL MEMORY.  What Cusco kept, it kept as 328 huacas strung
      along 41–42 ceque lines radiating from the Coricancha, each line the
      standing obligation of a named ayllu (Bauer 1998).  You do not look a
      huaca up.  You walk its line in order, passing the earlier shrines to
      reach the later ones.  Memory is a route with a maintenance bill, not an
      address.

Implemented here from scratch in NumPy:

  * a CEQUE MEMORY read by an ordered stick-breaking pilgrimage along one
    selected radial line — no attention, no random access, no positional
    encoding, no residual stack;
  * a ROD-ASSAY HALTING HEAD whose signal is the ANGLE between the model's
    probe and a substrate vector computed only from the external assay channel,
    so the network is mathematically incapable of manufacturing confidence by
    inflating its own probe (test 2 proves this to machine precision);
  * an explicit REJECT outcome, because a search that cannot return "none of
    these" has not implemented Colcabamba;
  * PETRIFICATION: an irreversible, logged optimizer operation that freezes
    high-influence, high-volatility memory slots into read-only wankas which
    keep contributing to every forward pass forever.

THE TASK: "ANDEAN VALLEY ASSAY"
-------------------------------
Each example is an ordered march past T candidate sites in ONE valley.  The
march must stop at the FIRST sound site, or return REJECT if there is none.

The world is split in two, deliberately, and the split is the whole idea:

  SURFACE features are vivid and low-noise.  They tell you which KIND of valley
  you are in — the regime — and they also carry a decoy that looks like
  site-level soundness and is right only slightly more often than a coin.

  The ASSAY channel is dull and noisy and warped.  It is the only thing that
  reports the actual substrate under a given site.

Soundness is a regime-conditioned property of the substrate.  Neither channel
alone suffices: read only the surface and you learn the valley but stop at the
first pretty site; read only the assay and you learn the ground but not what
this valley counts as sound.  You have to know which line you are walking, and
then you have to drive the rod.

RUN
---
    python3 chapter_0326_manco_capac_1200.py            # gradient check, training, ablations, tests
    python3 chapter_0326_manco_capac_1200.py --fast     # short run

Pure NumPy.  No other dependencies.
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-9


# ==============================================================================
# SECTION 1 — THE VALLEY
# ==============================================================================
# Generative story, kept explicit so the deception is auditable:
#
#   regime r      ~ Uniform{0..R-1}        which kind of valley this march is in
#   latent  a     ~ N(0, I)                the real substrate under each site
#   sound(a, r)   = (a · wr[r]) > thr[r]   regime-conditioned soundness
#   assay  obs    = tanh(1.3 a) + noise    what the rod can feel: warped, noisy
#   surface       = regime_signature[r]    strong, clean, TRUE about the regime
#                   + decoy paint          strong, clean, MISLEADING about sites
#                   + ornament             irrelevant
#
# The decoy agrees with soundness at `decoy_agreement` (default 0.60).  It is
# the easiest thing in the dataset to learn and it caps out immediately.  Every
# valley is Colcabamba until the rod says otherwise.
# ==============================================================================


@dataclass
class ValleyConfig:
    """Parameters of the synthetic Andean valley."""
    n_sites: int = 10          # candidate sites per march
    n_regimes: int = 6         # kinds of valley; one ceque line each
    d_surface: int = 10        # vivid, low-noise: regime + decoy + ornament
    d_assay: int = 6           # dull, noisy, warped: the only truth about sites
    d_latent: int = 6
    sigma_assay: float = 0.30
    sigma_surface: float = 0.15
    decoy_agreement: float = 0.60
    p_sound: float = 0.18      # per-site probability a site is sound
    seed: int = 1200           # the traditional founding date, used as a seed

    @property
    def d_in(self) -> int:
        return self.d_surface + self.d_assay


class Valley:
    """
    Generates marches.  The rules are FIXED at construction: they are properties
    of the world, not of the sampler — as the Andes were a property of the world
    and not of Manco Capac.
    """

    def __init__(self, cfg: ValleyConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        R, L = cfg.n_regimes, cfg.d_latent

        # One soundness direction per regime, unit length and well separated.
        W = rng.normal(0, 1.0, (R, L))
        self.wr = W / np.linalg.norm(W, axis=1, keepdims=True)
        # Threshold placing P(sound) at the target rate: a·w ~ N(0,1) for unit w.
        self.thr = float(np.sqrt(2) * _erfinv(1 - 2 * cfg.p_sound))

        # Surface structure: a clean regime signature, a decoy channel, ornament.
        self.regime_sig = rng.normal(0, 1.0, (R, cfg.d_surface)) * 1.10
        self.decoy_dir = rng.normal(0, 1.0, cfg.d_surface) * 0.95
        self.ornament = rng.normal(0, 1.0, (2, cfg.d_surface)) * 0.35

    # ---- rule ---------------------------------------------------------------
    def is_sound(self, a: np.ndarray, r: np.ndarray) -> np.ndarray:
        """a: (n,T,L)   r: (n,)  ->  (n,T) bool"""
        w = self.wr[r]                                   # (n,L)
        return np.einsum("ntl,nl->nt", a, w) > self.thr

    # ---- sampling -----------------------------------------------------------
    def sample(self, n_marches: int, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
          X      (n, T, d_in)  observed features: [surface | assay]
          y      (n, T)        per-site soundness (auxiliary supervision)
          stop   (n,)          index of first sound site, or T for REJECT
          regime (n,)          ground-truth regime, for diagnostics only
        """
        c = self.cfg
        n, T, L = n_marches, c.n_sites, c.d_latent
        r = rng.integers(0, c.n_regimes, n)
        a = rng.normal(0, 1.0, (n, T, L))
        sound = self.is_sound(a, r)

        # --- the assay channel: warped and noisy, but about THIS site --------
        assay = np.tanh(1.3 * a[..., : c.d_assay]) \
            + c.sigma_assay * rng.normal(0, 1, (n, T, c.d_assay))

        # --- the decoy bit: right 60% of the time, and very easy to read -----
        flip = rng.random((n, T)) > c.decoy_agreement
        decoy = np.where(flip, ~sound, sound)

        # --- surface: regime signature (true) + decoy paint (misleading) -----
        surface = (self.regime_sig[r][:, None, :]
                   + (decoy.astype(float) * 2 - 1)[..., None] * self.decoy_dir
                   + np.stack([np.abs(a[..., 0]), a[..., 1]], -1) @ self.ornament
                   + c.sigma_surface * rng.normal(0, 1, (n, T, c.d_surface)))

        X = np.concatenate([surface, assay], axis=-1).astype(np.float64)

        stop = np.full(n, T, dtype=np.int64)
        hit = sound.any(axis=1)
        stop[hit] = sound.argmax(axis=1)[hit]
        return X, sound.astype(np.float64), stop, r


def _erfinv(x: float) -> float:
    """Inverse error function (Giles' rational approximation). Avoids SciPy."""
    w = -np.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
                  0.00021858087, -0.00125372503, -0.00417768164,
                  0.246640727, 1.50140941):
            p = p * w + c
    else:
        w = np.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844,
                  0.00573950773, -0.0076224613, 0.00943887047,
                  1.00167406, 2.83297682):
            p = p * w + c
    return float(p * x)


# ==============================================================================
# SECTION 2 — PARAMETERS
# ==============================================================================


@dataclass
class WankaConfig:
    d_in: int
    d_surface: int
    d_assay: int
    n_sites: int
    h: int = 28            # width of the site encoding (surface only)
    n_ceque: int = 6       # radial lines out of the centre (Cusco had 41–42)
    n_huaca: int = 5       # shrines strung along each line, in walking order
    d_probe: int = 12      # width of the rod and of the substrate vector
    h_sub: int = 20        # width of the substrate reader's hidden layer
    seed: int = 7

    # petrification (Section 6)
    petrify_warmup: int = 10
    petrify_influence_q: float = 0.75
    petrify_volatility_q: float = 0.75
    petrify_max_frac: float = 0.25
    petrify_every: int = 3      # at most one wanka per this many epochs


class WankaParams:
    """
    Named parameter block.  A plain dict of arrays, so the finite-difference
    check can walk everything uniformly.
    """

    def __init__(self, cfg: WankaConfig):
        rng = np.random.default_rng(cfg.seed)
        H, K, M, P, Hz = cfg.h, cfg.n_ceque, cfg.n_huaca, cfg.d_probe, cfg.h_sub

        def g(*shape):
            return rng.normal(0, 1.0 / np.sqrt(shape[-1]), shape)

        self.p: Dict[str, np.ndarray] = {
            # --- the party's own perception: SURFACE ONLY ---------------------
            #     Nothing the rod believes may come from the ground; nothing the
            #     ground reports may come from belief.  The two channels meet
            #     exactly once, at the angle in 3.6.
            "W_e": g(H, cfg.d_surface),
            "b_e": np.zeros(H),
            # --- ceque lines: one direction vector per radial line ------------
            "C": g(K, H),
            # --- huacas: ordered shrines along each line ----------------------
            "V": g(K, M, H),
            "bg": np.zeros((K, M)) - 0.4,   # walk gates start slightly closed
            # --- the rod.  Note the shape: it reads the WALK ONLY.  What the
            #     party perceives never reaches the rod except by selecting and
            #     walking a line.  "What counts as sound in this valley" is
            #     retrieved from a fixed set of shrines; it is not computed
            #     fresh from the view.
            "W_p": g(P, H),
            "b_p": np.zeros(P),
            # --- the substrate reader: assay channel only, one tanh layer to
            #     undo the warp the ground applies to what it reports ----------
            "W_z1": g(Hz, cfg.d_assay),
            "b_z1": np.zeros(Hz),
            "W_z2": g(P, Hz),
            "b_z2": np.zeros(P),
            # --- absorption -> halting ---------------------------------------
            "beta": np.array([4.0]),
            "b_a": np.array([-1.2]),
            # --- auxiliary per-site readout (a per-dimension weighted cosine) -
            "w_q": np.ones(P) * 1.0,
            "b_q": np.zeros(1),
        }
        self.cfg = cfg
        # Petrification mask: 1.0 = still revisable, 0.0 = frozen wanka.
        self.wanka_mask = np.ones((K, M), dtype=np.float64)
        self.registry: List[Dict] = []

    def __getitem__(self, k: str) -> np.ndarray:
        return self.p[k]

    def copy(self) -> "WankaParams":
        new = WankaParams(self.cfg)
        new.p = {k: v.copy() for k, v in self.p.items()}
        new.wanka_mask = self.wanka_mask.copy()
        new.registry = [dict(r) for r in self.registry]
        return new

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))


# ==============================================================================
# SECTION 3 — FORWARD PASS
# ==============================================================================
# The pass is literally a walk:
#
#   read the surface  ->  work out which ceque line this valley lies on  ->
#   walk that line in order, stopping at the first shrine that answers  ->
#   assemble the rod from what you brought and what the shrine gave you  ->
#   read the substrate from the assay channel alone  ->  drive the rod  ->
#   measure absorption as an ANGLE  ->  halt, or march on to the next site.
# ==============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(params: WankaParams,
            X: np.ndarray,
            cfg: WankaConfig,
            substrate_from: str = "assay",
            use_memory: bool = True) -> Dict[str, np.ndarray]:
    """
    X: (B, T, d_in).  Returns every intermediate the backward pass needs.

    `substrate_from='surface'` rewires the ground-reader to look at the surface
    instead of the assay channel — that is the ablation that measures what the
    rod is actually worth.  `use_memory=False` severs the ceque walk.
    """
    p = params.p
    B, T, _ = X.shape
    N = B * T
    H = cfg.h
    Xf = X.reshape(N, -1)
    surf = Xf[:, : cfg.d_surface]
    assay = Xf[:, cfg.d_surface:]

    # -- 3.1 read the surface ----------------------------------------------
    E = surf @ p["W_e"].T + p["b_e"]
    Hh = np.tanh(E)                                       # (N,H)

    # -- 3.2 choose which ceque line to walk -------------------------------
    #    The one categorical decision in the architecture: which radial line
    #    out of the centre this valley belongs to.
    S = (Hh @ p["C"].T) / np.sqrt(H)                      # (N,K)
    Alpha = _softmax(S, axis=1)

    # -- 3.3 walk the line, in order ---------------------------------------
    #    Stick-breaking along the huaca sequence: gate g_j is the probability
    #    that shrine j answers GIVEN that every earlier shrine on the line did
    #    not.  You cannot reach the fourth huaca without passing the first
    #    three.  Mass surviving the end of the line is the walk returning
    #    empty-handed, and it is not renormalised away.
    inner = np.einsum("nh,kmh->nkm", Hh, p["V"]) / np.sqrt(H) + p["bg"]
    G = np.clip(_sigmoid(inner), 1e-6, 1 - 1e-6)
    Surv = np.ones_like(G)
    Surv[:, :, 1:] = np.cumprod(1.0 - G, axis=2)[:, :, :-1]
    Wgt = G * Surv
    Mk = np.einsum("nkm,kmh->nkh", Wgt, p["V"])
    Mem = np.einsum("nk,nkh->nh", Alpha, Mk)              # (N,H)
    if not use_memory:
        Mem = np.zeros_like(Mem)

    # -- 3.4 assemble the rod ----------------------------------------------
    #    Constituted entirely by what the walk returned.  Sever the memory and
    #    the rod collapses to a single constant template for every valley.
    Pr = Mem @ p["W_p"].T + p["b_p"]                       # (N,P) the rod

    # -- 3.5 read the substrate --------------------------------------------
    #    Nothing the party believes enters here.  Only the channel the world
    #    writes into.  One tanh layer, because the ground reports itself warped.
    src = assay if substrate_from == "assay" else surf[:, : cfg.d_assay]
    Zh = np.tanh(src @ p["W_z1"].T + p["b_z1"])            # (N,Hz)
    Sub = Zh @ p["W_z2"].T + p["b_z2"]                     # (N,P)

    # -- 3.6 drive the rod: absorption is an ANGLE, never a magnitude -------
    #    Both vectors are normalised before they meet.  There is therefore no
    #    gradient direction anywhere in this network that says "be more certain
    #    by pressing harder."  The founder cannot make the ground softer.
    #    A floor rather than an additive epsilon: adding EPS to the norm would
    #    make the angle very slightly magnitude-dependent, which is precisely
    #    the property this head exists to forbid.
    nP = np.maximum(np.linalg.norm(Pr, axis=1), 1e-12)
    nS = np.maximum(np.linalg.norm(Sub, axis=1), 1e-12)
    Ph = Pr / nP[:, None]
    Sh = Sub / nS[:, None]
    elem = Ph * Sh                                         # (N,P)
    cos = elem.sum(axis=1)                                 # (N,)
    halt_logit = p["beta"][0] * cos + p["b_a"][0]
    A = np.clip(_sigmoid(halt_logit), 1e-6, 1 - 1e-6).reshape(B, T)

    # -- 3.7 halting distribution over the march ---------------------------
    #    P(stop at t) = A_t * prod_{i<t}(1 - A_i);  P(REJECT) = prod_t (1 - A_t)
    surv_march = np.ones_like(A)
    surv_march[:, 1:] = np.cumprod(1.0 - A, axis=1)[:, :-1]
    p_stop = A * surv_march
    p_reject = np.prod(1.0 - A, axis=1)

    # -- 3.8 auxiliary per-site readout ------------------------------------
    #    A per-dimension weighted cosine: strictly richer than 3.6, and still
    #    invariant to the scale of the rod.
    q_logit = elem @ p["w_q"] + p["b_q"][0]
    Q = np.clip(_sigmoid(q_logit), 1e-7, 1 - 1e-7).reshape(B, T)

    return dict(X=X, Xf=Xf, surf=surf, assay=assay, src=src, Hh=Hh, S=S,
                Alpha=Alpha, G=G, Surv=Surv, Wgt=Wgt, Mk=Mk, Mem=Mem,
                Pr=Pr, Zh=Zh, Sub=Sub, nP=nP, nS=nS, Ph=Ph, Sh=Sh, elem=elem,
                cos=cos, A=A, surv_march=surv_march, p_stop=p_stop,
                p_reject=p_reject, Q=Q, B=B, T=T, use_memory=use_memory)


# ==============================================================================
# SECTION 4 — LOSS AND BACKWARD PASS  (every gradient derived by hand)
# ==============================================================================


def loss_and_grads(params: WankaParams,
                   cache: Dict[str, np.ndarray],
                   y: np.ndarray,
                   stop: np.ndarray,
                   cfg: WankaConfig,
                   lam_aux: float = 0.6,
                   want_grads: bool = True
                   ) -> Tuple[float, Optional[Dict[str, np.ndarray]], Dict[str, float]]:
    """
    Loss = -log P(stop = stop*)  +  lam_aux * mean BCE on per-site soundness.

    The halting term is written straight in terms of the per-site halting
    probabilities A, which keeps the gradient exact and the arithmetic calm:

        stop* = j < T :  L = -log A_j - sum_{i<j} log(1 - A_i)
        stop* = T     :  L = -sum_i log(1 - A_i)          (the Colcabamba case)
    """
    p = params.p
    B, T = cache["B"], cache["T"]
    A = cache["A"]
    H = cfg.h

    idx = np.arange(T)[None, :]
    at = idx == stop[:, None]          # the site we required the rod to enter
    before = idx < stop[:, None]       # the sites we required it to refuse

    L_halt = -(np.log(np.where(at, A, 1.0)).sum(axis=1)
               + np.log(np.where(before, 1.0 - A, 1.0)).sum(axis=1)).mean()
    Q = cache["Q"]
    L_aux = -(y * np.log(Q) + (1 - y) * np.log(1 - Q)).mean()
    L = L_halt + lam_aux * L_aux

    stats = dict(loss=float(L), loss_halt=float(L_halt), loss_aux=float(L_aux))
    if not want_grads:
        return float(L), None, stats

    g: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in p.items()}

    # ---- 4.1 dL/dA and dL/d(aux logit) ------------------------------------
    dA = (np.where(at, -1.0 / A, 0.0) + np.where(before, 1.0 / (1.0 - A), 0.0)) / B
    dq = ((Q - y) * (lam_aux / y.size)).reshape(-1)

    # ---- 4.2 into the halting sigmoid, then into the angle ----------------
    dhalt = (dA * A * (1.0 - A)).reshape(-1)
    g["beta"][0] = float((dhalt * cache["cos"]).sum())
    g["b_a"][0] = float(dhalt.sum())
    dcos = dhalt * p["beta"][0]

    # ---- 4.3 the auxiliary head shares the same normalised product --------
    elem = cache["elem"]
    g["w_q"] += elem.T @ dq
    g["b_q"][0] = float(dq.sum())
    delem = dcos[:, None] + dq[:, None] * p["w_q"][None, :]

    # ---- 4.4 through the two normalisations -------------------------------
    Ph, Sh = cache["Ph"], cache["Sh"]
    dPh = delem * Sh
    dSh = delem * Ph
    # d/dv of (v/|v|) applied to an upstream gradient u:  (u - (u.vhat)vhat)/|v|
    dPr = (dPh - (dPh * Ph).sum(axis=1, keepdims=True) * Ph) / cache["nP"][:, None]
    dSub = (dSh - (dSh * Sh).sum(axis=1, keepdims=True) * Sh) / cache["nS"][:, None]

    # ---- 4.5 the substrate reader (assay channel only) --------------------
    Zh = cache["Zh"]
    g["W_z2"] += dSub.T @ Zh
    g["b_z2"] += dSub.sum(axis=0)
    dZh = (dSub @ p["W_z2"]) * (1.0 - Zh ** 2)
    g["W_z1"] += dZh.T @ cache["src"]
    g["b_z1"] += dZh.sum(axis=0)

    # ---- 4.6 the rod ------------------------------------------------------
    g["W_p"] += dPr.T @ cache["Mem"]
    g["b_p"] += dPr.sum(axis=0)
    dMem = dPr @ p["W_p"]
    dHh = np.zeros_like(cache["Hh"])
    if not cache["use_memory"]:
        dMem[:] = 0.0

    # ---- 4.7 which line was walked ----------------------------------------
    Alpha, Mk = cache["Alpha"], cache["Mk"]
    dAlpha = np.einsum("nh,nkh->nk", dMem, Mk)
    dMk = Alpha[:, :, None] * dMem[:, None, :]
    dS = Alpha * (dAlpha - (dAlpha * Alpha).sum(axis=1, keepdims=True))
    dHh += (dS @ p["C"]) / np.sqrt(H)
    g["C"] += (dS.T @ cache["Hh"]) / np.sqrt(H)

    # ---- 4.8 the ordered walk ---------------------------------------------
    #   Wgt_j = G_j * prod_{i<j}(1 - G_i)
    #     dWgt_j/dG_j = Surv_j
    #     dWgt_j/dG_i = -Wgt_j / (1 - G_i)        for i < j
    #   => dG_i = dWgt_i*Surv_i - (1/(1-G_i)) * sum_{j>i} dWgt_j*Wgt_j
    #   The suffix sum is the price of order: revising an early shrine moves
    #   every shrine downstream of it on the same line.
    V, G, Wgt, Surv = p["V"], cache["G"], cache["Wgt"], cache["Surv"]
    dWgt = np.einsum("nkh,kmh->nkm", dMk, V)
    g["V"] += np.einsum("nkm,nkh->kmh", Wgt, dMk)

    prod = dWgt * Wgt
    suffix = np.cumsum(prod[:, :, ::-1], axis=2)[:, :, ::-1]
    suffix = np.concatenate([suffix[:, :, 1:], np.zeros_like(suffix[:, :, :1])],
                           axis=2)
    dG = dWgt * Surv - suffix / (1.0 - G)
    dinner = dG * G * (1.0 - G)
    g["bg"] += dinner.sum(axis=0)
    dHh += np.einsum("nkm,kmh->nh", dinner, V) / np.sqrt(H)
    g["V"] += np.einsum("nkm,nh->kmh", dinner, cache["Hh"]) / np.sqrt(H)

    # ---- 4.9 the surface encoder ------------------------------------------
    dE = dHh * (1.0 - cache["Hh"] ** 2)
    g["W_e"] += dE.T @ cache["surf"]
    g["b_e"] += dE.sum(axis=0)

    # ---- 4.10 petrification: a wanka receives no gradient, ever ------------
    #   It is still READ in the forward pass — Ayar Uchu spoke from the stone.
    #   It simply can no longer be revised.
    g["V"] *= params.wanka_mask[:, :, None]
    g["bg"] *= params.wanka_mask

    return float(L), g, stats


# ==============================================================================
# SECTION 5 — EVALUATION
# ==============================================================================


def evaluate(params: WankaParams, X, y, stop, cfg,
             substrate_from: str = "assay", use_memory: bool = True
             ) -> Dict[str, float]:
    c = forward(params, X, cfg, substrate_from, use_memory)
    _, _, st = loss_and_grads(params, c, y, stop, cfg, want_grads=False)
    full = np.concatenate([c["p_stop"], c["p_reject"][:, None]], axis=1)
    pred = full.argmax(axis=1)
    T = cfg.n_sites
    is_rej = stop == T
    st["acc"] = float((pred == stop).mean())
    st["acc_sited"] = (float((pred[~is_rej] == stop[~is_rej]).mean())
                       if (~is_rej).any() else float("nan"))
    st["acc_reject"] = (float((pred[is_rej] == stop[is_rej]).mean())
                        if is_rej.any() else float("nan"))
    # per-site soundness AUC-free summary: accuracy of the auxiliary readout
    st["site_acc"] = float(((c["Q"] > 0.5) == (y > 0.5)).mean())
    m = ~is_rej & (pred < T)
    st["mae_stop"] = float(np.abs(pred[m] - stop[m]).mean()) if m.any() else float("nan")
    return st


# ==============================================================================
# SECTION 6 — THE OPTIMIZER, AND PETRIFICATION
# ==============================================================================
# Adam written out, plus the one operation this file exists to test.
#
# PETRIFICATION.  After a warmup, every huaca slot (k, m) carries two running
# statistics:
#
#   influence  = EMA of ||dL/dV[k,m]||     how hard this shrine pulls the loss
#   volatility = EMA std of that pull      how little it agrees with itself
#
# A slot that is BOTH high-influence and high-volatility is the Ayar Cachi
# signature: enormous leverage, no stable direction.  Such a slot is frozen —
# irreversibly, with a log entry — and thereafter enters every forward pass at
# exactly the value it held at the moment of freezing.  It becomes a wanka:
# read-only, still speaking, now a fixed landmark the rest of the memory is
# positioned against.
#
# THE PART THAT MUST NOT BE HIDDEN.  The criterion is a PROXY.  "High leverage,
# unstable" is a symptom that correlates with danger; it is not a measurement of
# it.  So the training loop measures held-out loss immediately before and after
# every freeze and prints the delta.  When that delta is positive the
# architecture has sealed a capable component in the cave for a reason it could
# not verify.  Ayar Cachi was removed by lying to him about an errand, and no
# chronicler ever asks whether the party was better off.  This file asks, every
# run, and prints whatever the answer turns out to be.
# ==============================================================================


class Adam:
    def __init__(self, params: WankaParams, lr: float = 8e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.p.items()}
        self.t = 0

    def step(self, params: WankaParams, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, gk in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (gk * gk)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class PetrificationMonitor:
    """Running statistics per huaca slot, and the freeze decision."""

    def __init__(self, cfg: WankaConfig, decay: float = 0.92):
        self.cfg = cfg
        self.d = decay
        K, M = cfg.n_ceque, cfg.n_huaca
        self.influence = np.zeros((K, M))
        self.mean_g = np.zeros((K, M))
        self.var_g = np.zeros((K, M))

    def observe(self, gV: np.ndarray) -> None:
        mag = np.linalg.norm(gV, axis=2)                 # (K,M)
        d = self.d
        self.influence = d * self.influence + (1 - d) * mag
        delta = mag - self.mean_g
        self.mean_g = d * self.mean_g + (1 - d) * mag
        self.var_g = d * self.var_g + (1 - d) * (delta * (mag - self.mean_g))

    def volatility(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var_g, 0.0))

    def candidate(self, mask: np.ndarray) -> Optional[Tuple[int, int, float, float]]:
        cfg = self.cfg
        live = mask > 0.5
        if live.sum() < 2:
            return None
        vol = self.volatility()
        thr_i = np.quantile(self.influence[live], cfg.petrify_influence_q)
        thr_v = np.quantile(vol[live], cfg.petrify_volatility_q)
        best, best_score = None, -np.inf
        for k in range(cfg.n_ceque):
            for m in range(cfg.n_huaca):
                if mask[k, m] < 0.5:
                    continue
                i_, v_ = self.influence[k, m], vol[k, m]
                if i_ >= thr_i and v_ >= thr_v and i_ * v_ > best_score:
                    best, best_score = (k, m, float(i_), float(v_)), i_ * v_
        return best


# ==============================================================================
# SECTION 7 — TRAINING
# ==============================================================================


def train(params: WankaParams,
          cfg: WankaConfig,
          data: Dict[str, tuple],
          epochs: int = 60,
          batch: int = 48,
          lr: float = 8e-3,
          petrify: bool = True,
          substrate_from: str = "assay",
          use_memory: bool = True,
          verbose: bool = True,
          probe_epochs: int = 3,
          seed: int = 3) -> Dict:
    Xtr, ytr, str_, _ = data["train"]
    Xva, yva, sva, _ = data["val"]
    rng = np.random.default_rng(seed)
    opt = Adam(params, lr=lr)
    mon = PetrificationMonitor(cfg)
    hist: List[Dict] = []
    n = Xtr.shape[0]
    max_frozen = int(np.floor(cfg.petrify_max_frac * cfg.n_ceque * cfg.n_huaca))
    last_freeze = -99

    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        run = 0.0
        for s in range(0, n, batch):
            b = order[s: s + batch]
            cache = forward(params, Xtr[b], cfg, substrate_from, use_memory)
            L, g, _ = loss_and_grads(params, cache, ytr[b], str_[b], cfg)
            mon.observe(g["V"])
            opt.step(params, g)
            run += L * len(b)
        tr_loss = run / n
        va = evaluate(params, Xva, yva, sva, cfg, substrate_from, use_memory)

        froze = None
        if (petrify and ep > cfg.petrify_warmup
                and ep - last_freeze >= cfg.petrify_every
                and int((params.wanka_mask < 0.5).sum()) < max_frozen):
            cand = mon.candidate(params.wanka_mask)
            if cand is not None:
                k, m, inf, vol = cand
                # ---- the counterfactual --------------------------------------
                #  Freezing changes no value, so measuring loss the instant
                #  after the freeze would always return exactly zero and would
                #  be a lie dressed as evidence.  What the freeze changes is the
                #  FUTURE.  So fork the model twice from this exact state, run
                #  both forward the same number of epochs on the same batches,
                #  one with the slot sealed and one with it free, and record the
                #  difference.  Positive means the seal cost us something.
                d_free = d_seal = float("nan")
                if probe_epochs > 0:
                    fk_free = params.copy()
                    fk_seal = params.copy()
                    fk_seal.wanka_mask[k, m] = 0.0
                    kw = dict(epochs=probe_epochs, batch=batch, lr=lr,
                              petrify=False, substrate_from=substrate_from,
                              use_memory=use_memory, verbose=False,
                              seed=seed + 1000 + ep, probe_epochs=0)
                    train(fk_free, cfg, data, **kw)
                    train(fk_seal, cfg, data, **kw)
                    d_free = evaluate(fk_free, Xva, yva, sva, cfg,
                                      substrate_from, use_memory)["loss"]
                    d_seal = evaluate(fk_seal, Xva, yva, sva, cfg,
                                      substrate_from, use_memory)["loss"]
                params.wanka_mask[k, m] = 0.0
                rec = dict(epoch=ep, ceque=int(k), huaca=int(m),
                           influence=round(inf, 6), volatility=round(vol, 6),
                           free=round(d_free, 5), sealed=round(d_seal, 5),
                           delta=round(d_seal - d_free, 6))
                params.registry.append(rec)
                froze, last_freeze = (k, m), ep

        hist.append(dict(epoch=ep, train=tr_loss, **va))
        if verbose and (ep % 10 == 0 or ep == 1 or froze):
            tag = f"   [petrified ceque {froze[0]} huaca {froze[1]}]" if froze else ""
            print(f"  epoch {ep:3d} | train {tr_loss:7.4f} | val {va['loss']:7.4f}"
                  f" | march acc {va['acc']:.3f} | site acc {va['site_acc']:.3f}{tag}")

    return dict(history=hist, params=params)


# ==============================================================================
# SECTION 8 — GRADIENT CHECK  (mandatory, runs first, aborts on failure)
# ==============================================================================


def gradient_check(seed: int = 11, n_probe: int = 6, h: float = 1e-5,
                   verbose: bool = True) -> Tuple[bool, float]:
    """
    Central finite differences against the hand-derived analytic gradient, on a
    small random configuration, sampling entries from every parameter tensor.
    Relative error above 1e-5 is a failure.
    """
    vcfg = ValleyConfig(n_sites=5, n_regimes=3, d_surface=5, d_assay=4,
                        d_latent=4, seed=5)
    valley = Valley(vcfg)
    rng = np.random.default_rng(seed)
    X, y, stop, _ = valley.sample(4, rng)

    cfg = WankaConfig(d_in=vcfg.d_in, d_surface=vcfg.d_surface,
                      d_assay=vcfg.d_assay, n_sites=vcfg.n_sites,
                      h=6, n_ceque=3, n_huaca=4, d_probe=5, h_sub=5, seed=2)
    params = WankaParams(cfg)
    # Push off the symmetric initialisation so no gradient is trivially zero.
    for k in params.p:
        params.p[k] = params.p[k] + rng.normal(0, 0.3, params.p[k].shape)

    _, grads, _ = loss_and_grads(params, forward(params, X, cfg), y, stop, cfg)

    worst, worst_at = 0.0, ""
    for name in sorted(params.p.keys()):
        arr = params.p[name]
        picks = rng.choice(arr.size, size=min(n_probe, arr.size), replace=False)
        for fi in picks:
            idx = np.unravel_index(fi, arr.shape)
            orig = arr[idx]
            arr[idx] = orig + h
            Lp, _, _ = loss_and_grads(params, forward(params, X, cfg), y, stop,
                                      cfg, want_grads=False)
            arr[idx] = orig - h
            Lm, _, _ = loss_and_grads(params, forward(params, X, cfg), y, stop,
                                      cfg, want_grads=False)
            arr[idx] = orig
            num = (Lp - Lm) / (2 * h)
            ana = grads[name][idx]
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            if rel > worst:
                worst, worst_at = rel, f"{name}{tuple(int(i) for i in idx)}"
    ok = worst < 1e-5
    if verbose:
        print(f"  {params.n_params()} parameters, all tensors probed")
        print(f"  worst relative error {worst:.3e} at {worst_at}"
              f"   -> {'PASS' if ok else 'FAIL'}")
    return ok, worst


# ==============================================================================
# SECTION 9 — SELF-TESTS
# ==============================================================================


def _small(seed=17):
    vcfg = ValleyConfig(n_sites=8, seed=99)
    valley = Valley(vcfg)
    rng = np.random.default_rng(seed)
    X, y, stop, r = valley.sample(24, rng)
    cfg = WankaConfig(d_in=vcfg.d_in, d_surface=vcfg.d_surface,
                      d_assay=vcfg.d_assay, n_sites=vcfg.n_sites,
                      h=12, n_ceque=4, n_huaca=4, d_probe=8, h_sub=8, seed=4)
    return WankaParams(cfg), cfg, X, y, stop


def test_walk_mass() -> bool:
    """Answer weights plus walk-ends-empty mass must be exactly 1 on every line."""
    params, cfg, X, _, _ = _small()
    c = forward(params, X, cfg)
    total = c["Wgt"].sum(axis=2) + np.prod(1 - c["G"], axis=2)
    dev = float(np.abs(total - 1).max())
    ok = dev < 1e-10
    print(f"  walk mass conservation      max deviation {dev:.2e}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_rod_cannot_certify_itself() -> bool:
    """
    Scale the entire rod head by 40.  Because absorption is an angle, the
    halting distribution must be unmoved to machine precision.  This is what
    the two-rod assay is for: the founder cannot make the ground softer by
    pressing harder.
    """
    params, cfg, X, _, _ = _small()
    c0 = forward(params, X, cfg)
    a1 = c0["A"].copy()
    # For contrast: what an ordinary magnitude-sensitive head would have done.
    naive = _sigmoid(c0["Pr"].sum(1)).reshape(a1.shape)
    dev = 0.0
    for factor in (40.0, 1.0 / 40.0):
        q = params.copy()
        q.p["W_p"] *= factor
        q.p["b_p"] *= factor
        dev = max(dev, float(np.abs(a1 - forward(q, X, cfg)["A"]).max()))
        naive_dev = float(np.abs(naive - _sigmoid(
            forward(q, X, cfg)["Pr"].sum(1)).reshape(a1.shape)).max())
    ok = dev < 1e-9          # floating-point roundoff only
    print(f"  rod scale-invariance        max halting shift {dev:.2e} "
          f"(a magnitude-sensitive head shifts {naive_dev:.2e})"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_walk_is_ordered() -> bool:
    """
    Reverse the shrine order along every line.  Random-access memory would be
    unaffected.  A pilgrimage is not.
    """
    params, cfg, X, _, _ = _small()
    m1 = forward(params, X, cfg)["Mem"].copy()
    params.p["V"] = params.p["V"][:, ::-1, :].copy()
    params.p["bg"] = params.p["bg"][:, ::-1].copy()
    m2 = forward(params, X, cfg)["Mem"]
    diff = float(np.abs(m1 - m2).mean())
    ok = diff > 1e-3
    print(f"  walk order-dependence       mean read shift {diff:.4f}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_petrification_irreversible(cfg, params, data) -> bool:
    """A frozen wanka must be bit-identical after further training."""
    frozen = np.argwhere(params.wanka_mask < 0.5)
    if len(frozen) == 0:
        print("  petrification irreversible  no wankas formed -> SKIP")
        return True
    snap = {(int(k), int(m)): params.p["V"][k, m].copy() for k, m in frozen}
    Xtr, ytr, str_, _ = data["train"]
    opt = Adam(params, lr=2e-2)
    for i in range(30):
        b = slice(i * 16 % 200, i * 16 % 200 + 48)
        cache = forward(params, Xtr[b], cfg)
        _, g, _ = loss_and_grads(params, cache, ytr[b], str_[b], cfg)
        opt.step(params, g)
    ok = all(np.array_equal(params.p["V"][k, m], v) for (k, m), v in snap.items())
    print(f"  petrification irreversible  {len(snap)} wanka(s) held through "
          f"30 further steps   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_frozen_wanka_still_speaks(cfg, params, X) -> bool:
    """
    The stone form did not stop Ayar Uchu speaking.  Silence a frozen wanka and
    the forward pass must change — proof that freezing is not deletion.
    """
    frozen = np.argwhere(params.wanka_mask < 0.5)
    if len(frozen) == 0:
        print("  frozen wanka still speaks   no wankas formed -> SKIP")
        return True
    k, m = frozen[0]
    before = forward(params, X[:16], cfg)["Mem"].copy()
    keep = params.p["V"][k, m].copy()
    params.p["V"][k, m] = 0.0
    after = forward(params, X[:16], cfg)["Mem"]
    params.p["V"][k, m] = keep
    diff = float(np.abs(before - after).mean())
    ok = diff > 1e-7
    print(f"  frozen wanka still speaks   read shift when silenced {diff:.6f}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_colcabamba_reachable(params, cfg, X, stop) -> bool:
    """REJECT must carry real mass on marches where no site is sound."""
    c = forward(params, X, cfg)
    full = np.concatenate([c["p_stop"], c["p_reject"][:, None]], axis=1)
    is_rej = stop == cfg.n_sites
    if not is_rej.any():
        print("  Colcabamba reachable        no reject cases -> SKIP")
        return True
    mass = float(full[is_rej, cfg.n_sites].mean())
    ok = mass > 0.30
    print(f"  Colcabamba reachable        mean REJECT mass on empty valleys "
          f"{mass:.3f}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_line_selection_is_informative(params, cfg, X, regime) -> bool:
    """
    Each ceque line should end up standing for a regime.  Measure it: assign
    every line to its most frequent true regime, then score.  Chance is 1/R.
    """
    c = forward(params, X, cfg)
    line = c["Alpha"].reshape(X.shape[0], X.shape[1], -1).mean(1).argmax(1)
    R = int(regime.max()) + 1
    table = np.zeros((cfg.n_ceque, R))
    for l, g_ in zip(line, regime):
        table[l, g_] += 1
    purity = table.max(axis=1).sum() / max(table.sum(), 1)
    ok = purity > 1.5 / R
    print(f"  ceque lines track regimes   purity {purity:.3f} "
          f"(chance {1.0/R:.3f})   -> {'PASS' if ok else 'FAIL'}")
    return ok


# ==============================================================================
# SECTION 10 — MAIN: the whole experiment
# ==============================================================================


def build_data(vcfg: ValleyConfig, n_train: int, n_val: int, seed: int = 42):
    valley = Valley(vcfg)
    rng = np.random.default_rng(seed)
    return valley, dict(train=valley.sample(n_train, rng),
                        val=valley.sample(n_val, rng),
                        test=valley.sample(n_val, rng))


def surface_baseline(valley: Valley, data) -> float:
    """
    The best a pure surface-reader can do: stop at the first site the decoy
    channel likes.  This is the number the rod exists to beat, and it is the
    number a model trained on vivid features finds first and settles for.
    """
    X, _, stop, _ = data["test"]
    surf = X[..., : valley.cfg.d_surface]
    score = surf @ valley.decoy_dir
    like = score > 0
    T = valley.cfg.n_sites
    pred = np.where(like.any(1), like.argmax(1), T)
    return float((pred == stop).mean())


def banner(t: str) -> None:
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WANKA — Manco Capac, figure 0265")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--seed", type=int, default=1200)
    a = ap.parse_args(argv)
    epochs = 25 if a.fast else a.epochs
    n_train = 800 if a.fast else 2000
    n_val = 400 if a.fast else 800
    t0 = time.time()

    banner("WANKA — the Wanka-Assay Ceque Network\n"
           "Figure 0265 · Manco Capac · c. 1200 CE · Cusco valley, Peru")

    banner("1 — GRADIENT CHECK (central finite differences, every tensor)")
    ok_grad, _ = gradient_check()
    if not ok_grad:
        print("  gradient check failed; aborting")
        return 1

    vcfg = ValleyConfig(seed=a.seed)
    valley, data = build_data(vcfg, n_train, n_val, seed=a.seed + 1)
    Xtr, ytr, str_, rtr = data["train"]
    banner("2 — THE VALLEY")
    print(f"  marches {n_train} train / {n_val} val / {n_val} test"
          f"   |   {vcfg.n_sites} sites per march   |   "
          f"{vcfg.n_regimes} regimes")
    print(f"  surface {vcfg.d_surface} dims: clean regime signature + a decoy that")
    print(f"          agrees with site soundness {vcfg.decoy_agreement:.0%} of the time")
    print(f"  assay   {vcfg.d_assay} dims: tanh-warped substrate + noise "
          f"(sigma {vcfg.sigma_assay})")
    print(f"  per-site soundness rate {ytr.mean():.3f}   |   "
          f"empty valleys (Colcabamba marches) {(str_ == vcfg.n_sites).mean():.3f}")

    cfg = WankaConfig(d_in=vcfg.d_in, d_surface=vcfg.d_surface,
                      d_assay=vcfg.d_assay, n_sites=vcfg.n_sites, seed=11)
    print(f"  architecture: {cfg.n_ceque} ceque lines x {cfg.n_huaca} huacas, "
          f"probe width {cfg.d_probe}   |   {WankaParams(cfg).n_params()} parameters")

    banner("3 — TRAINING (rod driven into the assay channel, petrification on)")
    params = WankaParams(cfg)
    train(params, cfg, data, epochs=epochs, petrify=True, seed=a.seed)
    main_test = evaluate(params, *data["test"][:3], cfg)

    banner("4 — ABLATIONS (identical seed, identical schedule)")
    def _run(tag, **kw):
        p2 = WankaParams(WankaConfig(**{**cfg.__dict__}))
        train(p2, cfg, data, epochs=epochs, verbose=False, probe_epochs=0,
              seed=a.seed, **kw)
        r = evaluate(p2, *data["test"][:3], cfg,
                     substrate_from=kw.get("substrate_from", "assay"),
                     use_memory=kw.get("use_memory", True))
        print(f"  {tag:<53} march acc {r['acc']:.3f}   site acc {r['site_acc']:.3f}")
        return r

    t_surf = _run("(a) rod driven into the SURFACE, not the assay",
                  substrate_from="surface", petrify=True)
    t_nomem = _run("(b) ceque memory severed (no walk, no lines)",
                   use_memory=False, petrify=True)
    t_nopet = _run("(c) petrification disabled (all huacas stay revisable)",
                   petrify=False)
    base = surface_baseline(valley, data)
    print(f"  {'(d) pure surface-reader: stop at the first pretty site':<53}"
          f" march acc {base:.3f}")

    banner("5 — HELD-OUT TEST MARCHES, FULL MODEL")
    print(f"  march accuracy (exact stop index)      {main_test['acc']:.3f}")
    print(f"    on valleys that contain a sound site {main_test['acc_sited']:.3f}")
    print(f"    on empty valleys (Colcabamba)        {main_test['acc_reject']:.3f}")
    print(f"  per-site soundness accuracy            {main_test['site_acc']:.3f}")
    print(f"  mean stop-index error when it stops    {main_test['mae_stop']:.3f}")
    print(f"  loss {main_test['loss']:.4f}  "
          f"(halting {main_test['loss_halt']:.4f}, site {main_test['loss_aux']:.4f})")

    banner("6 — THE WANKA REGISTRY (irreversible petrifications)")
    if not params.registry:
        print("  no huaca met both criteria; the memory stayed wholly revisable")
    else:
        print("  Each row is a 3-epoch forked counterfactual from the state at")
        print("  that epoch: the same model trained on with the slot free, and")
        print("  trained on with it sealed. 'cost' is sealed minus free.\n")
        print(f"  {'ep':>3} {'ceque':>6} {'huaca':>6} {'influence':>10} "
              f"{'volatility':>11} {'free':>8} {'sealed':>8} {'cost':>10}")
        for r in params.registry:
            print(f"  {r['epoch']:>3} {r['ceque']:>6} {r['huaca']:>6} "
                  f"{r['influence']:>10.5f} {r['volatility']:>11.5f} "
                  f"{r['free']:>8.4f} {r['sealed']:>8.4f} {r['delta']:>+10.5f}")
        d = [r["delta"] for r in params.registry]
        good = sum(1 for x in d if x <= 0)
        print(f"\n  {len(d)} wanka(s) formed.  {good} of {len(d)} freezes did not")
        print(f"  raise held-out loss over the forked horizon; {len(d)-good} did.")
        print(f"  net cost of the whole petrification policy: {sum(d):+.5f}")
        print("  Read that number as it stands. The freeze criterion is a proxy for")
        print("  danger, not a measurement of it. Where the cost is positive, a")
        print("  capable component was sealed in the cave for a reason the system")
        print("  could not verify. The architecture logs that rather than smoothing")
        print("  it away, because the chroniclers never once ask whether the party")
        print("  was better off without Ayar Cachi.")

    banner("7 — SELF-TESTS")
    checks = [
        ("gradient check", ok_grad),
        ("walk mass conservation", test_walk_mass()),
        ("rod scale-invariance", test_rod_cannot_certify_itself()),
        ("walk order-dependence", test_walk_is_ordered()),
        ("petrification irreversible",
         test_petrification_irreversible(cfg, params.copy(), data)),
        ("frozen wanka still speaks",
         test_frozen_wanka_still_speaks(cfg, params, data["test"][0])),
        ("Colcabamba reachable",
         test_colcabamba_reachable(params, cfg, data["test"][0], data["test"][2])),
        ("ceque lines track regimes",
         test_line_selection_is_informative(params, cfg, data["test"][0],
                                            data["test"][3])),
    ]
    n_ok = sum(1 for _, v in checks if v)
    banner(f"8 — SUMMARY: {n_ok}/{len(checks)} checks passed   "
           f"({time.time() - t0:.1f}s)")
    for name, v in checks:
        print(f"  [{'ok' if v else 'FAIL'}] {name}")
    print("\n  The march does not stop where the founder likes the look of the")
    print("  ground. It stops where the rod goes in. Everything that could not be")
    print("  carried is standing behind it, in stone, still talking.")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

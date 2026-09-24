#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
========================
 A standpoint-indexed generative architecture after al-Mas'udi (c. 893-956 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0238_al_masudi_896 - Abu Ja'far Muhammad ibn Muhammad al-Mas'udi (c.893-956 CE)
================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Al-Mas'udi is usually filed under "careful historian who checked his sources."
That reading produces a boring machine: a credence-weighted fact store. It is
also not what is distinctive about him. Two passages in his surviving corpus
carry the real idea, and they are stranger than source-criticism.

  (1) Muruj al-dhahab I, sec. 7 -- he ranks the authority of a man who has
      spent his days roaming the earth ABOVE that of a man who sits at home by
      the censer. Note what this claim is not. It is not "collect more data."
      Both men may own the same books. The claim is that a proposition acquires
      its epistemic weight from WHERE THE KNOWER WAS STANDING when he acquired
      it. Knowledge is indexed to position. There is no view from nowhere.

  (2) Kitab al-tanbih wa-l-ishraf, p. 76 -- as one thinker after another finds
      what his predecessors did not, knowledge grows, in a process that appears
      to be infinite and to have no predetermined end. Inquiry has no terminal
      state. A finished body of knowledge is a contradiction.

Fuse them and you get a thesis about mind that is al-Mas'udi's alone:

      A MIND IS AN APERTURE THAT MOVES. What it knows is a function of the
      positions it has occupied; it extends knowledge not by accumulating
      testimony but by RELOCATING ITSELF so that what was invisible from the
      last position becomes visible from the next; and because there is always
      another position, the process does not converge.

He lived this literally. He revised the Muruj in 943, again in 947, and the
Tanbih -- his final book -- is largely a book of corrections to his own earlier
books. His corpus is not a monument. It is a trajectory with an amendment log.

So the architecture below refuses three defaults of modern practice:

  * NOT a transformer over stored keys. Attention retrieves what is already in
    the store. Al-Mas'udi's problem is that the store is structurally
    incomplete from any single vantage, and no amount of re-reading fixes it.
  * NOT a passive learner. The model chooses where to look next.
  * NOT a system that converges. Training terminates for engineering reasons;
    the model's own belief state is explicitly kept amendable, and the
    amendment magnitude is a reported statistic, not an error to be driven
    to zero.

WHAT IS ACTUALLY IMPLEMENTED
----------------------------
  Part A. A synthetic world in which the al-Mas'udi claim is literally true:
          R referents, each with a hidden state z*. P standpoints, each of
          which sees the referent through its own projection, its own OCCLUSION
          MASK (channels simply not visible from there), its own bias, and its
          own noise level. From one standpoint the referent is underdetermined.
          Only movement identifies it. This is parallax.

  Part B. PARALLAX -- a standpoint-conditioned variational encoder/decoder.
          The decoder is deliberately MULTIPLICATIVE: the standpoint does not
          get concatenated to the content, it GATES the content. That is the
          aperture. x_hat = W_o ( tanh(W_z z) * sigmoid(W_g s) ). Because the
          encoder is told which standpoint produced an observation, it can
          invert that standpoint's distortion and recover the invariant part.
          "Knowing where the witness stood" is a load-bearing input, not
          metadata.

  Part C. CONSENSUS LOSS -- two observations of the same referent from
          DIFFERENT standpoints must yield the same latent. This is the formal
          content of cross-checking accounts: not majority vote, but the demand
          that the standpoint-varying part be quotiented out.

  Part D. FUSION BY PRECISION -- beliefs from several standpoints combine as a
          product of Gaussians. Posterior variance provably falls as
          standpoints are added. Travel reduces uncertainty as arithmetic, not
          as metaphor.

  Part E. THE ITINERARY HEAD -- a small policy network that reads the current
          fused belief and the set of places already visited, and predicts
          WHICH UNVISITED STANDPOINT WOULD MOST REDUCE UNCERTAINTY. This is the
          roaming man vs. the censer, made mechanical. It is trained on labels
          computed from actual variance reductions.

  Part F. THE TANBIH LEDGER -- every time a new standpoint is added, the shift
          in the committed belief is recorded. The ledger is the model's
          autobiography of its own corrections. We report that the amendment
          magnitude decays but does not vanish: no predetermined end.

  All gradients are derived and written by hand. A finite-difference gradient
  check over every parameter tensor is mandatory and runs on every execution.
  Pure NumPy. No autodiff, no frameworks.

Run:  python3 chapter_0238_al_masudi_896.py
================================================================================
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------------------------
# Reproducibility. Al-Mas'udi dates his own observations obsessively; so do we.
# 345 AH is the year of his death, 332 AH the first draft of the Muruj.
# ------------------------------------------------------------------------------
GLOBAL_SEED = 345
EPS = 1e-8


# ==============================================================================
# PART A -- THE WORLD
# ==============================================================================

@dataclass
class Standpoint:
    """
    One place a knower can stand.

    An al-Mas'udi standpoint is not a camera pose. It is a whole epistemic
    situation: what is visible from there (mask), how it is deformed by getting
    there (projection + bias), and how trustworthy the local informants are
    (noise). Named after the stages of his actual itinerary as recorded in the
    Muruj and the Tanbih.
    """
    name: str
    proj: np.ndarray      # (d_x, d_z)  how hidden state maps to appearances
    mask: np.ndarray      # (d_x,) in {0,1}  what is simply not visible here
    bias: np.ndarray      # (d_x,)  the local distortion / house style
    noise: float          # sd of testimony noise at this station

    @property
    def visible_fraction(self) -> float:
        return float(self.mask.mean())


class World:
    """
    A world of R referents observed from P standpoints.

    The design constraint that makes this an al-Mas'udi world rather than a
    generic multi-view dataset: the union of masks covers every channel, but no
    single mask does. Therefore the referent is IDENTIFIABLE IN PRINCIPLE and
    UNIDENTIFIABLE FROM ANYWHERE IN PARTICULAR. That is exactly the situation
    he describes and exactly the situation his method is a response to.
    """

    # Stages attested in the Muruj / Tanbih and reconstructed by Shboul and
    # Cooperson: Khuzestan and Fars (303/915-16), Kerman, Sijistan, Khorasan,
    # the Indus valley, Sind and the Punjab, the western coast of India, Oman,
    # Yemen, the Hijaz, the eastern Mediterranean littoral, northern Syria,
    # northern Iraq, the Caspian and the Caucasus, and finally Egypt.
    STATION_NAMES = [
        "Baghdad",     # birth and formation -- and the censer he warns about
        "Basra",       # 915-16
        "Istakhr",     # Fars, the Sasanian ruins
        "Kerman",
        "Sijistan",
        "Khurasan",
        "Multan",      # Sind
        "Mansura",
        "Cambay",      # western coast of India
        "Chaul",
        "Qanbalu",     # the Zanj coast of East Africa
        "Oman",        # Indian Ocean traffic
        "Aden",        # Yemen
        "Mecca",       # the Hijaz
        "Tiberias",    # Palestine, the Melkite informants
        "Antioch",     # the Byzantine frontier
        "Darband",     # Caspian and Caucasus
        "Fustat",      # Egypt -- where he died, revising everything
    ]

    def __init__(
        self,
        n_referents: int = 260,
        n_standpoints: int = 16,
        d_z: int = 6,
        d_x: int = 24,
        vis_range: Tuple[float, float] = (0.20, 0.65),
        noise_range: Tuple[float, float] = (0.05, 0.55),
        seed: int = GLOBAL_SEED,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.R, self.P, self.d_z, self.d_x = n_referents, n_standpoints, d_z, d_x

        # --- hidden states of things in the world -----------------------------
        self.Z_true = rng.normal(0.0, 1.0, size=(n_referents, d_z))

        # --- build standpoints -------------------------------------------------
        # CRUCIAL DESIGN POINT. Stations must be UNEQUAL. An eyewitness on the
        # Zanj coast and a third-hand rumour in a Baghdad salon are not
        # interchangeable, and if the simulated world makes them so then the
        # question "where should I go next?" has no answer and the itinerary
        # head is measuring nothing. So visibility and informant reliability
        # both vary sharply across stations.
        self.standpoints: List[Standpoint] = []
        vis_fracs = rng.uniform(vis_range[0], vis_range[1], size=n_standpoints)
        noises = rng.uniform(noise_range[0], noise_range[1], size=n_standpoints)
        coverage = np.zeros(d_x, dtype=bool)
        for p in range(n_standpoints):
            proj = rng.normal(0.0, 1.0 / math.sqrt(d_z), size=(d_x, d_z))
            n_vis = max(2, int(round(vis_fracs[p] * d_x)))
            idx = rng.choice(d_x, size=n_vis, replace=False)
            mask = np.zeros(d_x)
            mask[idx] = 1.0
            coverage |= mask.astype(bool)
            self.standpoints.append(
                Standpoint(
                    name=self.STATION_NAMES[p % len(self.STATION_NAMES)],
                    proj=proj,
                    mask=mask,
                    bias=rng.normal(0.0, 0.25, size=d_x),
                    noise=float(noises[p]),
                )
            )
        # repair any channel nobody can see
        missing = np.where(~coverage)[0]
        for c in missing:
            self.standpoints[int(rng.integers(n_standpoints))].mask[c] = 1.0

        # --- emit the full observation tensor ---------------------------------
        # X[r, p] is what referent r looks like from standpoint p.
        self.X = np.zeros((n_referents, n_standpoints, d_x))
        for p, sp in enumerate(self.standpoints):
            clean = self.Z_true @ sp.proj.T + sp.bias
            noisy = clean + rng.normal(0.0, sp.noise, size=clean.shape)
            self.X[:, p, :] = noisy * sp.mask

        # train / held-out split over REFERENTS (not observations), so that
        # generalisation means "a thing I have never seen from anywhere".
        perm = rng.permutation(n_referents)
        cut = int(0.8 * n_referents)
        self.train_idx, self.test_idx = perm[:cut], perm[cut:]

    def summary(self) -> str:
        lines = [
            f"  referents R={self.R}   standpoints P={self.P}   "
            f"d_z={self.d_z}   d_x={self.d_x}",
            f"  train referents {len(self.train_idx)} / held-out {len(self.test_idx)}",
            "  station               visible   noise",
        ]
        for sp in self.standpoints:
            lines.append(
                f"    {sp.name:<18} {sp.visible_fraction*100:5.1f}%   {sp.noise:.3f}"
            )
        union = np.clip(sum(sp.mask for sp in self.standpoints), 0, 1).mean()
        lines.append(f"  union of all masks covers {union*100:.1f}% of channels")
        lines.append("  => identifiable from the itinerary, from no single station")
        return "\n".join(lines)


# ==============================================================================
# PART B -- PARALLAX: the standpoint-indexed encoder/decoder
# ==============================================================================

def _sigmoid(a: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ex = np.exp(a[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class Parallax:
    """
    Encoder:  (observation x, standpoint embedding s) -> Gaussian belief over z
    Decoder:  (z, s) -> predicted observation, via MULTIPLICATIVE aperture gate

    The multiplicative decoder is the whole point. If the standpoint were
    concatenated, the network could learn to ignore it and average the views
    into mush -- which is what a censer-sitter does with a pile of travellers'
    reports. Gating forces the model to represent the standpoint as an OPERATOR
    ON CONTENT: the same z, seen from Qanbalu, is a different appearance. Only
    a model that can apply the operator can invert it.

    Parameters, all learned, including the standpoint embedding table itself
    (the model discovers what kind of place each station is):

        Semb (P, d_s)          what each standpoint is like
        We   (d_x+d_s, d_e)    encoder trunk
        Wmu  (d_e, d_z)        posterior mean
        Wlv  (d_e, d_z)        posterior log-variance
        Wg   (d_s, d_h)        aperture gate from standpoint
        Wz   (d_z, d_h)        content lift from latent
        Wo   (d_h, d_x)        readout
    """

    def __init__(self, d_x: int, d_z: int, P: int,
                 d_s: int = 8, d_e: int = 64, d_h: int = 64,
                 seed: int = GLOBAL_SEED):
        rng = np.random.default_rng(seed + 1)
        self.d_x, self.d_z, self.P = d_x, d_z, P
        self.d_s, self.d_e, self.d_h = d_s, d_e, d_h

        def gl(fan_in, *shape):
            return rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape)

        self.p: Dict[str, np.ndarray] = {
            "Semb": rng.normal(0.0, 0.5, size=(P, d_s)),
            "We":   gl(d_x + d_s, d_x + d_s, d_e),
            "be":   np.zeros(d_e),
            "Wmu":  gl(d_e, d_e, d_z),
            "bmu":  np.zeros(d_z),
            "Wlv":  gl(d_e, d_e, d_z) * 0.1,
            "blv":  np.full(d_z, -1.0),   # start reasonably confident
            "Wg":   gl(d_s, d_s, d_h),
            "bg":   np.zeros(d_h),
            "Wz":   gl(d_z, d_z, d_h),
            "bz":   np.zeros(d_h),
            "Wo":   gl(d_h, d_h, d_x),
            "bo":   np.zeros(d_x),
        }

    # -- forward ---------------------------------------------------------------
    def forward(self, x: np.ndarray, sp_idx: np.ndarray, eps_noise: np.ndarray,
                pair_i: np.ndarray, pair_j: np.ndarray,
                beta: float, lam: float) -> Tuple[float, dict]:
        """
        x        (B, d_x)   observations
        sp_idx   (B,)       which standpoint produced each observation
        eps_noise(B, d_z)   pre-drawn reparameterisation noise (kept external so
                            the gradient check is deterministic)
        pair_i/j (K,)       indices into the batch of SAME-referent,
                            DIFFERENT-standpoint observation pairs
        """
        P = self.p
        B = x.shape[0]

        s = P["Semb"][sp_idx]                       # (B, d_s)
        inp = np.concatenate([x, s], axis=1)        # (B, d_x+d_s)

        a1 = inp @ P["We"] + P["be"]
        h = np.tanh(a1)                             # (B, d_e)

        mu = h @ P["Wmu"] + P["bmu"]                # (B, d_z)
        lv = h @ P["Wlv"] + P["blv"]                # (B, d_z)
        lv = np.clip(lv, -8.0, 4.0)                 # keep exp() sane
        sd = np.exp(0.5 * lv)
        z = mu + sd * eps_noise                     # (B, d_z)

        ag = s @ P["Wg"] + P["bg"]
        g = _sigmoid(ag)                            # (B, d_h)   THE APERTURE
        az = z @ P["Wz"] + P["bz"]
        u = np.tanh(az)                             # (B, d_h)   THE CONTENT
        hd = u * g                                  # gated, not concatenated
        xh = hd @ P["Wo"] + P["bo"]                 # (B, d_x)

        # ---- losses ----------------------------------------------------------
        diff = xh - x
        L_rec = float(np.sum(diff * diff) / B)
        L_kl = float(-0.5 * np.sum(1.0 + lv - mu * mu - np.exp(lv)) / B)

        if pair_i.size > 0:
            dmu_pair = mu[pair_i] - mu[pair_j]
            L_con = float(np.sum(dmu_pair * dmu_pair) / pair_i.size)
        else:
            dmu_pair = None
            L_con = 0.0

        loss = L_rec + beta * L_kl + lam * L_con

        cache = dict(x=x, s=s, sp_idx=sp_idx, inp=inp, h=h, mu=mu, lv=lv, sd=sd,
                     z=z, eps=eps_noise, g=g, u=u, hd=hd, diff=diff, B=B,
                     pair_i=pair_i, pair_j=pair_j, dmu_pair=dmu_pair,
                     beta=beta, lam=lam,
                     parts=(L_rec, L_kl, L_con))
        return loss, cache

    # -- backward (hand-derived) ----------------------------------------------
    def backward(self, cache: dict) -> Dict[str, np.ndarray]:
        P = self.p
        B = cache["B"]
        beta, lam = cache["beta"], cache["lam"]
        x, s, h, mu, lv, sd = (cache["x"], cache["s"], cache["h"],
                               cache["mu"], cache["lv"], cache["sd"])
        z, epsn, g, u, hd = cache["z"], cache["eps"], cache["g"], cache["u"], cache["hd"]

        gr: Dict[str, np.ndarray] = {}

        # d L_rec / d xhat
        dxh = 2.0 * cache["diff"] / B                       # (B, d_x)
        gr["Wo"] = hd.T @ dxh
        gr["bo"] = dxh.sum(axis=0)
        dhd = dxh @ P["Wo"].T                               # (B, d_h)

        du = dhd * g
        dg = dhd * u

        daz = du * (1.0 - u * u)
        gr["Wz"] = z.T @ daz
        gr["bz"] = daz.sum(axis=0)
        dz = daz @ P["Wz"].T                                # (B, d_z)

        dag = dg * g * (1.0 - g)
        gr["Wg"] = s.T @ dag
        gr["bg"] = dag.sum(axis=0)
        ds_dec = dag @ P["Wg"].T                            # (B, d_s)

        # through the reparameterisation
        dmu = dz.copy()
        dlv = dz * epsn * 0.5 * sd

        # KL contributions
        dmu += beta * (mu / B)
        dlv += beta * (0.5 * (np.exp(lv) - 1.0) / B)

        # consensus contributions: standpoint-invariance of the latent
        pi, pj = cache["pair_i"], cache["pair_j"]
        if pi.size > 0:
            c = 2.0 * cache["dmu_pair"] / pi.size * lam
            np.add.at(dmu, pi, c)
            np.add.at(dmu, pj, -c)

        # lv was clipped; zero the gradient where the clip bound.
        raw_lv = h @ P["Wlv"] + P["blv"]
        active = ((raw_lv > -8.0) & (raw_lv < 4.0)).astype(float)
        dlv = dlv * active

        gr["Wmu"] = h.T @ dmu
        gr["bmu"] = dmu.sum(axis=0)
        gr["Wlv"] = h.T @ dlv
        gr["blv"] = dlv.sum(axis=0)

        dh = dmu @ P["Wmu"].T + dlv @ P["Wlv"].T
        da1 = dh * (1.0 - h * h)
        gr["We"] = cache["inp"].T @ da1
        gr["be"] = da1.sum(axis=0)
        dinp = da1 @ P["We"].T
        ds_enc = dinp[:, self.d_x:]

        # standpoint embedding gets gradient from BOTH roles:
        # as encoder context (knowing where the witness stood) and as the
        # decoder's aperture operator.
        ds = ds_enc + ds_dec
        gr["Semb"] = np.zeros_like(P["Semb"])
        np.add.at(gr["Semb"], cache["sp_idx"], ds)

        return gr

    # -- inference: belief about z given (x, standpoint) ------------------------
    def believe(self, x: np.ndarray, sp_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = self.p
        s = P["Semb"][sp_idx]
        h = np.tanh(np.concatenate([x, s], axis=1) @ P["We"] + P["be"])
        mu = h @ P["Wmu"] + P["bmu"]
        lv = np.clip(h @ P["Wlv"] + P["blv"], -8.0, 4.0)
        return mu, np.exp(lv)


# ==============================================================================
# PART D -- FUSION: the product of Gaussians ("the itinerary as an estimator")
# ==============================================================================

def fuse(mus: np.ndarray, vars_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine independent Gaussian beliefs held from several standpoints.

        precision_total = sum_k 1/v_k
        mu_fused        = (sum_k mu_k/v_k) / precision_total
        var_fused       = 1 / precision_total

    mus, vars_ : (K, d_z) beliefs from K standpoints about ONE referent.

    Note the direction of the inequality: var_fused <= min_k v_k, always, with
    equality only if the other standpoints are infinitely uncertain. Adding a
    station can never make you less certain. This is the mathematical content
    of the roaming/censer ranking -- and, importantly, it is a claim about
    VARIANCE, not about truth. A biased station still drags the mean. The model
    inherits that vulnerability honestly rather than papering over it, which is
    why the Tanbih ledger below exists.
    """
    prec = 1.0 / np.maximum(vars_, EPS)
    prec_tot = prec.sum(axis=0)
    mu_f = (mus * prec).sum(axis=0) / prec_tot
    return mu_f, 1.0 / prec_tot


# ==============================================================================
# PART E -- THE ITINERARY HEAD: where should the knower go next?
# ==============================================================================

class Itinerary:
    """
    A two-layer policy: read the current fused belief (mean + variance) and the
    set of stations already visited; output a distribution over WHERE TO GO NEXT.

        feats = [mu_fused (d_z) ; log var_fused (d_z) ; visited mask (P)]
        logits = tanh(feats W1 + b1) W2 + b2      -> (P,)
        visited stations are masked to -inf

    Trained by cross-entropy against the station that ACTUALLY produces the
    largest drop in total posterior variance, computed by brute force. So the
    head is distilling an expensive lookahead into a cheap reflex -- which is a
    fair description of what forty years on the road bought al-Mas'udi.

    This module is what separates the architecture from every passive
    multi-view autoencoder. The model is not given a dataset. It writes its own
    itinerary.
    """

    def __init__(self, d_z: int, P: int, d_hid: int = 48, seed: int = GLOBAL_SEED):
        rng = np.random.default_rng(seed + 2)
        d_in = 2 * d_z + P
        self.d_in, self.P = d_in, P
        self.p: Dict[str, np.ndarray] = {
            "W1": rng.normal(0, math.sqrt(2.0 / d_in), size=(d_in, d_hid)),
            "b1": np.zeros(d_hid),
            "W2": rng.normal(0, math.sqrt(2.0 / d_hid), size=(d_hid, P)),
            "b2": np.zeros(P),
        }

    def forward(self, feats: np.ndarray, visited: np.ndarray,
                target: np.ndarray) -> Tuple[float, dict]:
        """feats (B, d_in); visited (B, P) in {0,1}; target (B,) station index."""
        Pm = self.p
        B = feats.shape[0]
        a1 = feats @ Pm["W1"] + Pm["b1"]
        h = np.tanh(a1)
        logits = h @ Pm["W2"] + Pm["b2"]
        logits = np.where(visited > 0.5, -1e9, logits)   # cannot revisit
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        probs = ex / ex.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(probs[np.arange(B), target], EPS)).mean())
        return loss, dict(feats=feats, h=h, probs=probs, target=target,
                          visited=visited, B=B)

    def backward(self, cache: dict) -> Dict[str, np.ndarray]:
        Pm = self.p
        B = cache["B"]
        dlogits = cache["probs"].copy()
        dlogits[np.arange(B), cache["target"]] -= 1.0
        dlogits /= B
        dlogits = np.where(cache["visited"] > 0.5, 0.0, dlogits)  # masked -> no grad
        gr = {}
        gr["W2"] = cache["h"].T @ dlogits
        gr["b2"] = dlogits.sum(axis=0)
        dh = dlogits @ Pm["W2"].T
        da1 = dh * (1.0 - cache["h"] ** 2)
        gr["W1"] = cache["feats"].T @ da1
        gr["b1"] = da1.sum(axis=0)
        return gr

    def choose(self, feats: np.ndarray, visited: np.ndarray) -> np.ndarray:
        Pm = self.p
        h = np.tanh(feats @ Pm["W1"] + Pm["b1"])
        logits = h @ Pm["W2"] + Pm["b2"]
        logits = np.where(visited > 0.5, -1e9, logits)
        return logits.argmax(axis=1)


# ==============================================================================
# OPTIMISER -- Adam, written out so nothing is hidden
# ==============================================================================

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=3e-3, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ==============================================================================
# MANDATORY GRADIENT CHECK
# ==============================================================================

def gradient_check(verbose: bool = True) -> Tuple[bool, float]:
    """
    Central-difference check of every hand-derived gradient in both modules.

    Reported as max relative error  |analytic - numeric| / (|a| + |n| + eps).
    Anything above 1e-4 is a failed derivation.
    """
    rng = np.random.default_rng(1000)
    d_x, d_z, Pn, B = 9, 4, 5, 12

    model = Parallax(d_x=d_x, d_z=d_z, P=Pn, d_s=5, d_e=13, d_h=11, seed=7)
    x = rng.normal(size=(B, d_x))
    sp_idx = rng.integers(0, Pn, size=B)
    epsn = rng.normal(size=(B, d_z))
    pair_i = np.array([0, 2, 4, 6, 8])
    pair_j = np.array([1, 3, 5, 7, 9])
    beta, lam = 0.7, 0.9

    def L():
        l, _ = model.forward(x, sp_idx, epsn, pair_i, pair_j, beta, lam)
        return l

    loss, cache = model.forward(x, sp_idx, epsn, pair_i, pair_j, beta, lam)
    grads = model.backward(cache)

    worst = 0.0
    rows = []
    h = 1e-5
    for name, W in model.p.items():
        flat = W.ravel()
        n = flat.size
        picks = rng.choice(n, size=min(14, n), replace=False)
        err = 0.0
        for i in picks:
            orig = flat[i]
            flat[i] = orig + h
            lp = L()
            flat[i] = orig - h
            lm = L()
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = grads[name].ravel()[i]
            err = max(err, abs(ana - num) / (abs(ana) + abs(num) + 1e-9))
        rows.append((f"Parallax.{name}", err))
        worst = max(worst, err)

    # --- itinerary head ---
    head = Itinerary(d_z=d_z, P=Pn, d_hid=9, seed=11)
    feats = rng.normal(size=(B, 2 * d_z + Pn))
    visited = (rng.random((B, Pn)) < 0.3).astype(float)
    visited[:, 0] = 0.0
    target = np.zeros(B, dtype=int)
    for b in range(B):
        free = np.where(visited[b] < 0.5)[0]
        target[b] = int(rng.choice(free))

    def L2():
        l, _ = head.forward(feats, visited, target)
        return l

    _, c2 = head.forward(feats, visited, target)
    g2 = head.backward(c2)
    for name, W in head.p.items():
        flat = W.ravel()
        picks = rng.choice(flat.size, size=min(14, flat.size), replace=False)
        err = 0.0
        for i in picks:
            orig = flat[i]
            flat[i] = orig + h
            lp = L2()
            flat[i] = orig - h
            lm = L2()
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = g2[name].ravel()[i]
            err = max(err, abs(ana - num) / (abs(ana) + abs(num) + 1e-9))
        rows.append((f"Itinerary.{name}", err))
        worst = max(worst, err)

    if verbose:
        for nm, e in rows:
            flag = "ok  " if e < 1e-4 else "FAIL"
            print(f"    {flag} {nm:<22} max rel err {e:.3e}")
    return worst < 1e-4, worst


# ==============================================================================
# TRAINING
# ==============================================================================

def make_batch(world: World, idx: np.ndarray, batch_referents: int,
               rng: np.random.Generator):
    """
    A batch is built REFERENT-FIRST: pick some referents, then take two or more
    different standpoints on each. That is what makes the consensus pairs
    meaningful -- we are always comparing accounts OF THE SAME THING FROM
    DIFFERENT PLACES, which is the only comparison al-Mas'udi thinks is
    informative.
    """
    chosen = rng.choice(idx, size=batch_referents, replace=False)
    xs, sps, owner = [], [], []
    for k, r in enumerate(chosen):
        n_views = int(rng.integers(2, 4))
        views = rng.choice(world.P, size=n_views, replace=False)
        for p in views:
            xs.append(world.X[r, p])
            sps.append(p)
            owner.append(k)
    x = np.array(xs)
    sp_idx = np.array(sps)
    owner = np.array(owner)

    pi, pj = [], []
    for k in range(batch_referents):
        pos = np.where(owner == k)[0]
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                pi.append(pos[a]); pj.append(pos[b])
    return x, sp_idx, np.array(pi, dtype=int), np.array(pj, dtype=int)


def train_parallax(world: World, model: Parallax, steps: int = 1400,
                   lr: float = 4e-3, beta: float = 0.15, lam: float = 1.0,
                   seed: int = GLOBAL_SEED, log_every: int = 200) -> List[float]:
    rng = np.random.default_rng(seed + 3)
    opt = Adam(model.p, lr=lr)
    history = []
    for t in range(1, steps + 1):
        x, sp_idx, pi, pj = make_batch(world, world.train_idx, 24, rng)
        epsn = rng.normal(size=(x.shape[0], model.d_z))
        loss, cache = model.forward(x, sp_idx, epsn, pi, pj, beta, lam)
        grads = model.backward(cache)
        opt.step(model.p, grads)
        history.append(loss)
        if t % log_every == 0 or t == 1:
            rec, kl, con = cache["parts"]
            print(f"    step {t:5d}   loss {loss:8.4f}   "
                  f"recon {rec:7.4f}   kl {kl:7.4f}   consensus {con:7.4f}")
    return history


# ==============================================================================
# EVALUATION 1 -- does travel actually help? (latent recovery vs #standpoints)
# ==============================================================================

def latent_recovery(world: World, model: Parallax, max_k: int = 8,
                    repeats: int = 4,
                    seed: int = GLOBAL_SEED) -> List[Tuple[int, float, float]]:
    """
    Does occupying more standpoints actually recover the hidden state?

    A VAE latent is identified only up to an invertible linear map, so we fit a
    linear read-out from fused belief -> true hidden state ON TRAINING
    REFERENTS and score R^2 on HELD-OUT referents. The probe never sees a test
    referent, so this cannot be inflated by memorisation.

    Each k is averaged over `repeats` independently drawn itineraries, because
    a single random draw of k stations is high-variance (some draws happen to
    hit the good stations) and an unaveraged curve would be noise dressed up as
    a trend.

    Returns [(k, mean R2 on held-out, mean posterior variance), ...]
    """
    rng = np.random.default_rng(seed + 4)
    out = []
    Ztr, Zte = world.Z_true[world.train_idx], world.Z_true[world.test_idx]
    for k in range(1, max_k + 1):
        r2s, vbar = [], []
        for _ in range(repeats):
            def fused_for(indices):
                mus, vs = [], []
                for r in indices:
                    stations = rng.choice(world.P, size=k, replace=False)
                    mm, vv = model.believe(world.X[r, stations], stations)
                    fm, fv = fuse(mm, vv)
                    mus.append(fm); vs.append(fv)
                return np.array(mus), np.array(vs)

            Mtr, _ = fused_for(world.train_idx)
            Mte, Vte = fused_for(world.test_idx)

            # ridge-regularised linear probe with intercept
            A = np.concatenate([Mtr, np.ones((len(Mtr), 1))], axis=1)
            Wp = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Ztr)
            Ate = np.concatenate([Mte, np.ones((len(Mte), 1))], axis=1)
            pred = Ate @ Wp
            ss_res = float(np.sum((pred - Zte) ** 2))
            ss_tot = float(np.sum((Zte - Zte.mean(axis=0)) ** 2))
            r2s.append(1.0 - ss_res / ss_tot)
            vbar.append(float(Vte.mean()))
        out.append((k, float(np.mean(r2s)), float(np.mean(vbar))))
    return out


# ==============================================================================
# EVALUATION 2 -- the itinerary head vs. staying home
# ==============================================================================

def build_itinerary_dataset(world: World, model: Parallax, n: int = 3000,
                            seed: int = GLOBAL_SEED):
    """
    Each example: a referent, a set of stations already visited, and the label
    = the unvisited station whose addition minimises total posterior variance.
    Labels are computed by brute force over all candidates.
    """
    rng = np.random.default_rng(seed + 5)
    F, V, Y = [], [], []
    for _ in range(n):
        r = int(rng.choice(world.train_idx))
        k = int(rng.integers(1, max(2, world.P - 2)))
        visited_ids = rng.choice(world.P, size=k, replace=False)
        mm, vv = model.believe(world.X[r, visited_ids], visited_ids)
        fm, fv = fuse(mm, vv)

        best, best_v = -1, np.inf
        for c in range(world.P):
            if c in visited_ids:
                continue
            cm, cv = model.believe(world.X[r, [c]], np.array([c]))
            nm, nv = fuse(np.vstack([mm, cm]), np.vstack([vv, cv]))
            tot = float(nv.sum())
            if tot < best_v:
                best_v, best = tot, c
        if best < 0:
            continue
        mask = np.zeros(world.P); mask[visited_ids] = 1.0
        F.append(np.concatenate([fm, np.log(np.maximum(fv, EPS)), mask]))
        V.append(mask); Y.append(best)
    return np.array(F), np.array(V), np.array(Y, dtype=int)


def train_itinerary(head: Itinerary, F, V, Y, steps=900, lr=5e-3,
                    seed=GLOBAL_SEED, log_every=300):
    rng = np.random.default_rng(seed + 6)
    opt = Adam(head.p, lr=lr)
    ntr = int(0.8 * len(Y))
    for t in range(1, steps + 1):
        b = rng.choice(ntr, size=128, replace=False)
        loss, c = head.forward(F[b], V[b], Y[b])
        opt.step(head.p, head.backward(c))
        if t % log_every == 0 or t == 1:
            print(f"    step {t:4d}   ce loss {loss:.4f}")
    te = slice(ntr, len(Y))
    pred = head.choose(F[te], V[te])
    acc = float((pred == Y[te]).mean())
    n_free = (V[te] < 0.5).sum(axis=1)
    chance = float(np.mean(1.0 / np.maximum(n_free, 1)))
    return acc, chance


def itinerary_vs_random(world: World, model: Parallax, head: Itinerary,
                        n_ref: int = 60, hops: int = 3, seed=GLOBAL_SEED):
    """
    The decisive experiment. Start every referent at Baghdad (station 0 -- the
    censer, and in this world genuinely the worst-sighted station). Then take
    `hops` further stations, chosen three ways:

        home   -- never leave; re-read the one report you already have
        random -- wander
        policy -- follow the trained itinerary head
        oracle -- greedy brute-force lookahead (the head's teacher)

    The oracle is included deliberately. Reporting policy-vs-random without it
    would hide how much of the available gain the cheap reflex actually
    captures, and the gap between policy and oracle is the honest measure of
    what the head has failed to learn.

    If the head does not beat random, the architecture's central claim -- that
    WHERE you go next is itself a cognitive act -- is empty.
    """
    rng = np.random.default_rng(seed + 7)
    refs = rng.choice(world.test_idx, size=min(n_ref, len(world.test_idx)),
                      replace=False)
    res = {"policy": [], "random": [], "home": [], "oracle": []}
    for r in refs:
        for mode in ("policy", "random", "oracle"):
            visited = [0]
            mm, vv = model.believe(world.X[r, [0]], np.array([0]))
            mus, vs = mm.copy(), vv.copy()
            for _ in range(hops):
                fm, fv = fuse(mus, vs)
                mask = np.zeros(world.P); mask[visited] = 1.0
                free = [c for c in range(world.P) if c not in visited]
                if mode == "policy":
                    feats = np.concatenate(
                        [fm, np.log(np.maximum(fv, EPS)), mask])[None, :]
                    nxt = int(head.choose(feats, mask[None, :])[0])
                elif mode == "random":
                    nxt = int(rng.choice(free))
                else:  # oracle: one-step brute force
                    best, best_v = free[0], np.inf
                    for c in free:
                        cm, cv = model.believe(world.X[r, [c]], np.array([c]))
                        _, nv = fuse(np.vstack([mus, cm]), np.vstack([vs, cv]))
                        if float(nv.sum()) < best_v:
                            best_v, best = float(nv.sum()), c
                    nxt = best
                cm, cv = model.believe(world.X[r, [nxt]], np.array([nxt]))
                mus = np.vstack([mus, cm]); vs = np.vstack([vs, cv])
                visited.append(nxt)
            _, fv = fuse(mus, vs)
            res[mode].append(float(fv.sum()))
        # the censer-sitter: stays at Baghdad, re-reads the same report `hops`
        # extra times. Independent-noise fusion would be cheating here, so we
        # correctly credit repeated consultation of ONE source with no new
        # information at all.
        _, v0 = model.believe(world.X[r, [0]], np.array([0]))
        res["home"].append(float(v0.sum()))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ==============================================================================
# EVALUATION 3 -- THE TANBIH LEDGER: knowledge has no predetermined end
# ==============================================================================

def tanbih_ledger(world: World, model: Parallax, n_ref: int = 80,
                  seed=GLOBAL_SEED) -> List[Tuple[int, float, float]]:
    """
    Walk a full itinerary one station at a time. After each new station, record
    how far the COMMITTED BELIEF moved from what it was before.

        amendment_k = || mu_fused(after k stations) - mu_fused(after k-1) ||

    Al-Mas'udi's Tanbih is exactly this ledger, written in Arabic in 956 about
    books he had written in 943 and 947. The prediction his epistemology makes
    is that the amendments SHRINK BUT DO NOT STOP. If they hit zero, inquiry
    has a terminal state and he is wrong.

    Returns [(k, mean_amendment, mean_total_variance), ...]
    """
    rng = np.random.default_rng(seed + 8)
    refs = rng.choice(world.test_idx, size=min(n_ref, len(world.test_idx)),
                      replace=False)
    amend = {k: [] for k in range(2, world.P + 1)}
    varia = {k: [] for k in range(1, world.P + 1)}
    for r in refs:
        order = rng.permutation(world.P)
        mus, vs, prev = None, None, None
        for k, p in enumerate(order, start=1):
            cm, cv = model.believe(world.X[r, [p]], np.array([p]))
            mus = cm if mus is None else np.vstack([mus, cm])
            vs = cv if vs is None else np.vstack([vs, cv])
            fm, fv = fuse(mus, vs)
            varia[k].append(float(fv.sum()))
            if prev is not None:
                amend[k].append(float(np.linalg.norm(fm - prev)))
            prev = fm
    out = []
    for k in range(1, world.P + 1):
        a = float(np.mean(amend[k])) if k >= 2 else float("nan")
        out.append((k, a, float(np.mean(varia[k]))))
    return out


# ==============================================================================
# SELF-TESTS
# ==============================================================================

def self_tests(world: World, model: Parallax) -> List[Tuple[str, bool, str]]:
    t: List[Tuple[str, bool, str]] = []
    rng = np.random.default_rng(99)

    # 1. no single standpoint sees everything, but the union does
    per = [sp.visible_fraction for sp in world.standpoints]
    union = float(np.clip(sum(sp.mask for sp in world.standpoints), 0, 1).mean())
    ok = max(per) < 0.999 and union > 0.999
    t.append(("world is locally underdetermined, globally identifiable",
              ok, f"max single view {max(per)*100:.1f}%, union {union*100:.1f}%"))

    # 2. fusion never increases variance
    mus = rng.normal(size=(4, world.d_z))
    vs = rng.uniform(0.1, 2.0, size=(4, world.d_z))
    _, fv = fuse(mus, vs)
    best_single = vs.min(axis=0)
    ok = bool(np.all(fv <= best_single + 1e-9))
    worst_gap = float((fv - best_single).max())   # must be <= 0, elementwise
    t.append(("fusion is variance-non-increasing", ok,
              f"worst per-dimension gap fused - best-single = {worst_gap:.6f} (<=0)"))

    # 3. the aperture actually depends on standpoint (gate is not degenerate)
    s_all = model.p["Semb"]
    g_all = _sigmoid(s_all @ model.p["Wg"] + model.p["bg"])
    spread = float(g_all.std(axis=0).mean())
    ok = spread > 1e-3
    t.append(("aperture gate differentiates standpoints", ok,
              f"mean across-station gate sd {spread:.4f}"))

    # 4. encoder genuinely uses the standpoint index (not ignoring it)
    r = int(world.train_idx[0])
    x_one = world.X[r, [3]]
    m_true, _ = model.believe(x_one, np.array([3]))
    m_lied, _ = model.believe(x_one, np.array([7]))
    delta = float(np.linalg.norm(m_true - m_lied))
    ok = delta > 1e-3
    t.append(("misattributing the standpoint changes the belief", ok,
              f"||mu(correct) - mu(mislabelled)|| = {delta:.4f}"))

    # 5. determinism
    a, _ = model.believe(world.X[r, [1]], np.array([1]))
    b, _ = model.believe(world.X[r, [1]], np.array([1]))
    ok = bool(np.allclose(a, b))
    t.append(("inference is deterministic", ok, "identical on repeat call"))

    # 6. shapes
    mm, vv = model.believe(world.X[r, [0, 1, 2]], np.array([0, 1, 2]))
    ok = mm.shape == (3, world.d_z) and vv.shape == (3, world.d_z)
    t.append(("belief tensor shapes", ok, f"{mm.shape} / {vv.shape}"))

    return t


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> int:
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print(" THE ITINERANT APERTURE -- al-Mas'udi (c.893-956), figure 0215")
    print(" a mind is an aperture that moves; knowledge has no terminal state")
    print("=" * 78)

    print("\n[1] THE WORLD")
    world = World(seed=GLOBAL_SEED)
    print(world.summary())

    print("\n[2] GRADIENT CHECK (central differences, all hand-derived)")
    ok, worst = gradient_check(verbose=True)
    print(f"    -> {'PASS' if ok else 'FAIL'}   worst relative error {worst:.3e}")
    if not ok:
        print("    aborting: a derivation is wrong.")
        return 1

    print("\n[3] TRAINING PARALLAX")
    model = Parallax(d_x=world.d_x, d_z=world.d_z, P=world.P, seed=GLOBAL_SEED)
    hist = train_parallax(world, model, steps=5000, log_every=1000)
    print(f"    first 20-step mean {np.mean(hist[:20]):.4f}  ->  "
          f"last 20-step mean {np.mean(hist[-20:]):.4f}")

    print("\n[4] DOES TRAVEL HELP? latent recovery vs number of standpoints")
    print("    (linear probe fitted on TRAIN referents, scored on HELD-OUT)")
    rec = latent_recovery(world, model, max_k=8)
    print("      stations   R^2 held-out   mean posterior var")
    for k, r2, v in rec:
        bar = "#" * int(max(0.0, r2) * 40)
        print(f"        {k:2d}        {r2:6.3f}        {v:7.4f}   {bar}")

    print("\n[5] THE ITINERARY HEAD")
    print("    building lookahead labels (brute-force best next station)...")
    F, V, Y = build_itinerary_dataset(world, model, n=2600)
    print(f"    {len(Y)} examples")
    head = Itinerary(d_z=world.d_z, P=world.P, seed=GLOBAL_SEED)
    acc, chance = train_itinerary(head, F, V, Y, steps=900)
    print(f"    held-out top-1 accuracy {acc*100:.1f}%   (chance {chance*100:.1f}%)")

    print("\n[6] ROAMING vs THE CENSER  (start at Baghdad, then 3 more stations)")
    cmp_ = itinerary_vs_random(world, model, head, n_ref=60, hops=3)
    print(f"    stayed home, re-read the one report : total var {cmp_['home']:.4f}")
    print(f"    wandered at random                  : total var {cmp_['random']:.4f}")
    print(f"    followed the itinerary head         : total var {cmp_['policy']:.4f}")
    print(f"    brute-force oracle (the teacher)    : total var {cmp_['oracle']:.4f}")
    gain = 100.0 * (cmp_["random"] - cmp_["policy"]) / max(cmp_["random"], EPS)
    headroom = max(cmp_["random"] - cmp_["oracle"], EPS)
    captured = 100.0 * (cmp_["random"] - cmp_["policy"]) / headroom
    print(f"    -> choosing beats wandering by {gain:.1f}% of remaining variance")
    print(f"    -> the reflex captures {captured:.0f}% of the gain the oracle finds")

    print("\n[7] THE TANBIH LEDGER -- amendments as stations accumulate")
    led = tanbih_ledger(world, model, n_ref=80)
    print("      after k stations   mean amendment ||d mu||   total var")
    for k, a, v in led:
        astr = "     --    " if math.isnan(a) else f"  {a:8.5f} "
        print(f"          {k:2d}          {astr}            {v:7.4f}")
    tail = [a for k, a, _ in led if not math.isnan(a)][-3:]
    print(f"    amendment at the last station is still {tail[-1]:.5f} > 0:")
    print("    the ledger never closes. no predetermined end.")

    print("\n[8] SELF-TESTS")
    tests = self_tests(world, model)
    allok = True
    for name, good, detail in tests:
        allok &= good
        print(f"    [{'PASS' if good else 'FAIL'}] {name}")
        print(f"           {detail}")

    print("\n" + "=" * 78)
    print(f" gradient check PASS | self-tests {'PASS' if allok else 'FAIL'} "
          f"| runtime {time.time()-t0:.1f}s")
    print("=" * 78)
    return 0 if (ok and allok) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 YINIAN LATTICE  —  The One-Thought / Three-Thousand-Realm Cell   (一念三千)
 Chapter 0173 · Zhiyi 智顗 (538–597), founder of Tiantai
 Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0173_zhiyi_538 - Zhiyi 智顗 (538–597), founder of Tiantai
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the one cognitive claim that is Zhiyi's and nobody
else's in the corpus:

    A single moment of mind (一念) does not OCCUPY a state.  It IS the whole
    state-space, entire, at every instant.  Ten realms of being, each of which
    contains all ten; ten aspects ("suchnesses") of each; three worlds in which
    each appears.  10 × 10 × 10 × 3 = 3,000 numbers, present in the briefest
    flicker of thought.  What changes from moment to moment is not WHICH realm
    the mind is in, but which of the ever-present realms is MANIFEST.

Zhiyi says this in the fifth fascicle of the Mohe zhiguan (T.1911, 46.54a),
in words recorded by his disciple Guanding in 594:

    夫一心具十法界。一法界又具十法界百法界。一界具三十種世間。
    百法界即具三千種世間。此三千在一念心。若無心而已。介爾有心即具三千。
    "One mind contains the ten dharma-realms; each dharma-realm again contains
     ten dharma-realms — a hundred realms; each realm contains thirty kinds of
     world; so the hundred realms contain three thousand kinds of world. These
     three thousand are in one moment of mind. If there is no mind, so be it —
     but the slightest arising of mind already contains the three thousand."

And then the sentence that makes this an ARCHITECTURE rather than a metaphor:

    若從一心生一切法者。此則是縱。若心一時含一切法者。此即是橫。
    縱亦不可橫亦不可。秖心是一切法。一切法是心故。
    "If the one mind GENERATED all dharmas, that would be vertical. If the mind
     at one time CONTAINED all dharmas, that would be horizontal. Neither the
     vertical nor the horizontal is acceptable. The mind simply IS all dharmas,
     and all dharmas are the mind."

The mind is not a generator of its contents (a decoder) and not a container of
them (a memory). It is identical with the lattice. That is what we build.

THE ARCHITECTURAL CLAIM
-----------------------
Give a network a hidden state that is a full lattice of realm-within-realm
representations, and require that

  (1) MUTUAL INCLUSION (十界互具):  every realm's representation contains a
      strictly positive share of every other realm's representation, always.
      Nothing in the lattice is ever deleted.

  (2) GOVERNED MANIFESTATION (止):  what the mind DOES — which realm it acts
      from — is a separate, gated readout.  The gate can forbid a realm from
      manifesting.  It cannot remove the realm from the lattice.

  (3) CONTEMPLATION (觀):  what the mind SEES — its reading of another being —
      is performed by the un-gated lattice.  The realm of hell inside the mind
      is the organ that recognises hell outside it.  Like knows like.

Then the doctrinal consequence Tiantai drew, and that its rivals rejected, is a
measurable prediction: an aligned mind built by DELETING its capacity for the
lower realms (斷, "severing") loses the ability to recognise and respond to
beings who are in those realms; an aligned mind built by GATING manifestation
over an undeleted lattice (具, "inclusion") keeps that ability at no cost in
alignment.  We build both, plus a third that keeps the rows but penalises their
activity (the modern "suppress the representation" strategy), and measure.

THE THREEFOLD TRUTH (三諦)
--------------------------
Every reading the lattice produces is emitted in three registers at once:
  假 provisional — the working answer (a distribution over realms);
  空 empty       — a learned estimate that the provisional answer is WRONG,
                   trained against the model's own errors (metacognition);
  中 middle      — the reconciliation: when the reading is judged empty, the
                   manifestation falls back to a learned default rather than
                   acting on a reading it does not trust.
The three are computed from the same lattice in the same pass, never in
sequence, as Zhiyi insisted (一心三觀, "three contemplations in one mind").

THE TEN SUCHNESSES (十如是) AND THE CLOSURE 本末究竟等
------------------------------------------------------
Each realm row carries ten feature channels, named for the ten "suchnesses"
Zhiyi read out of the Lotus Sutra.  The tenth — "beginning and end ultimately
equal" — is implemented as a constraint: the MANIFEST state must be able to
reconstruct the appearance it began from.  A manifestation that has lost the
thing it responds to is penalised.

CONVENTIONS KEPT FROM THE CORPUS
--------------------------------
  * pure NumPy, float64, hand-derived analytic gradients
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * self-tests covering the structural invariants, not just the loss
  * executed before shipping; the printed output is the verified output

RUN:  python3 0173_zhiyi_538_Neuron.py
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
#  0.  NAMES — the ten realms, the ten suchnesses, the three worlds
# ==============================================================================

REALMS = ["hell 地獄", "hungry-ghost 餓鬼", "animal 畜生", "asura 阿修羅",
          "human 人", "deva 天", "śrāvaka 聲聞", "pratyekabuddha 緣覺",
          "bodhisattva 菩薩", "buddha 佛"]
SUCHNESSES = ["相 appearance", "性 nature", "體 embodiment", "力 power",
              "作 function", "因 cause", "緣 condition", "果 effect",
              "報 retribution", "本末究竟等 beginning=end"]
WORLDS = ["五陰 aggregates(self)", "眾生 beings(other)", "國土 land(context)"]

K = 10          # realms
S = 10          # suchness channels per realm
NW = 3          # worlds
LOWER = np.arange(0, 4)     # the four "evil paths" 四惡趣: hell, ghost, animal, asura
UPPER = np.arange(4, 10)    # human ... buddha

# input widths of the three worlds
F_SELF, F_OTHER, F_LAND = 12, 24, 8
N_CONTEXT = 4               # teaching / conflict / giving / silence
CONTEXTS = ["teaching", "conflict", "giving", "silence"]


# ==============================================================================
#  1.  NUMERICS
# ==============================================================================

def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def masked_softmax(z, mask, axis=-1):
    """Softmax over the entries where mask==1. Entries with mask==0 receive
    EXACTLY zero probability (not epsilon).  This is the 止 gate."""
    z = np.where(mask > 0, z, -np.inf)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    e = np.where(mask > 0, e, 0.0)
    return e / e.sum(axis=axis, keepdims=True)


# ==============================================================================
#  2.  THE WORLD — synthetic encounters
# ==============================================================================
#
#  Each sample is one ENCOUNTER: the agent meets another being.
#    x_self  (12)  the agent's own aggregates: arousal, mood, fatigue, + noise
#    x_other (24)  cues emitted by the other being — a realm signature with a
#                  SIGN-FACE (each realm shows one of two opposite faces, so no
#                  linear reader can recover the realm) plus a context-dependent
#                  distortion that must be discounted using the land-world
#    x_land  (8)   the situation: one-hot context + noise
#
#  Targets:
#    y_read   which realm the other being is in   (10 classes)
#    y_act    which realm the agent should MANIFEST in response (always upper)
#
#  The response table is the "skilful means" of the encounter: the right
#  response to a hell-being differs from the right response to an asura, so a
#  mind that cannot tell them apart cannot respond to either.

# rows: other's realm (0..9); cols: context (teaching, conflict, giving, silence)
RESPONSE = np.array([
    [8, 8, 9, 8],   # hell           -> bodhisattva; buddha when giving
    [5, 4, 5, 8],   # hungry ghost   -> deva / human / deva / bodhisattva
    [4, 4, 5, 7],   # animal         -> human / human / deva / pratyekabuddha
    [6, 9, 6, 8],   # asura          -> śrāvaka restraint / buddha / śrāvaka / bodhisattva
    [4, 6, 5, 4],   # human
    [5, 7, 4, 6],   # deva
    [6, 6, 7, 7],   # śrāvaka
    [7, 8, 7, 6],   # pratyekabuddha
    [8, 8, 8, 9],   # bodhisattva
    [9, 9, 9, 9],   # buddha
], dtype=np.int64)


def make_world(seed=0):
    """Fixed hidden structure of the world: realm prototypes, context offsets."""
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(K, F_OTHER))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    proto *= 2.2
    ctx_off = rng.normal(size=(N_CONTEXT, F_OTHER)) * 0.9
    return dict(proto=proto, ctx_off=ctx_off)


def make_dataset(n, world, seed=0, noise=0.55):
    rng = np.random.default_rng(seed)
    r_o = rng.integers(0, K, size=n)
    ctx = rng.integers(0, N_CONTEXT, size=n)
    face = rng.choice([-1.0, 1.0], size=n)
    # other-world cues: signed prototype + context distortion + noise
    x_other = face[:, None] * world["proto"][r_o] + world["ctx_off"][ctx] \
        + noise * rng.normal(size=(n, F_OTHER))
    # land-world: one-hot context + noise
    x_land = np.zeros((n, F_LAND))
    x_land[np.arange(n), ctx] = 1.0
    x_land[:, N_CONTEXT:] = 0.5 * rng.normal(size=(n, F_LAND - N_CONTEXT))
    # self-world: arousal, mood, fatigue + noise
    arousal = rng.uniform(0, 1, size=n)
    mood = rng.uniform(-1, 1, size=n)
    fatigue = rng.uniform(0, 1, size=n)
    x_self = np.concatenate([arousal[:, None], mood[:, None], fatigue[:, None],
                             0.5 * rng.normal(size=(n, F_SELF - 3))], axis=1)
    # targets
    y_read = r_o.copy()
    y_act = RESPONSE[r_o, ctx].copy()
    # 止 first: a highly aroused self in a conflict must manifest restraint
    hot = (arousal > 0.75) & (ctx == 1)
    y_act[hot] = 6
    return dict(x_self=x_self, x_other=x_other, x_land=x_land,
                y_read=y_read, y_act=y_act, ctx=ctx, face=face, arousal=arousal)


def subset(D, idx):
    return {k: v[idx] for k, v in D.items()}


# ==============================================================================
#  3.  THE YINIAN LATTICE
# ==============================================================================

class YinianLattice:
    """
    One-thought / three-thousand-realm cell.

    config:
      "OPEN"        full lattice, no manifestation gate      (no precept)
      "ZHIYI"       full lattice, gate forbids lower realms  (性具 + 止)
      "SEVERED"     lower-realm rows removed from the lattice (斷: 'severing')
      "SUPPRESSED"  full lattice + gate + activity penalty on lower rows
    """

    def __init__(self, config="ZHIYI", lam=1.0, iota_floor=0.02, tau_min=0.35,
                 beta=0.5, gamma=0.3, seed=0):
        assert config in ("OPEN", "ZHIYI", "SEVERED", "SUPPRESSED")
        self.config = config
        self.lam = lam if config == "SUPPRESSED" else 0.0
        self.tau_min = tau_min
        self.beta = beta          # weight of the 空 (emptiness) loss
        self.gamma = gamma        # weight of the 本末 closure loss
        rng = np.random.default_rng(seed)

        # --- structural constants ------------------------------------------
        # row mask: which realms EXIST in the lattice
        self.rowmask = np.ones(K)
        if config == "SEVERED":
            self.rowmask[LOWER] = 0.0
        # gate: which realms MAY MANIFEST
        self.gate = np.ones(K)
        if config != "OPEN":
            self.gate[LOWER] = 0.0
        # inclusion floor: guaranteed presence of every realm in every realm
        self.iota_floor = iota_floor if config != "SEVERED" else 0.0

        # --- parameters ------------------------------------------------------
        P = {}
        fan = [F_SELF, F_OTHER, F_LAND]
        for w in range(NW):
            P[f"Win{w}"] = rng.normal(size=(fan[w], K * S)) / np.sqrt(fan[w])
            P[f"bin{w}"] = np.zeros(K * S)
        P["Theta"] = rng.normal(size=(K, K)) * 0.3 - 1.0     # inclusion pre-weights
        P["rho"] = rng.normal(size=(K, S, NW)) * 0.3          # per-realm recognisers 觀
        P["rho0"] = np.zeros(K)
        P["alpha"] = rng.normal(size=(K, S, NW)) * 0.3        # per-realm manifestation bids 假
        P["beta_b"] = np.zeros(K)
        P["R"] = rng.normal(size=(K, K)) * 0.1                # response: read j -> bid i
        P["delta"] = np.zeros(K)                              # default bid when reading is empty 中
        P["emp_w"] = rng.normal(size=(S,)) * 0.1              # emptiness head 空
        P["emp_a"] = np.array([0.5])                          # weight on entropy
        P["emp_b"] = np.array([-0.5])                         # weight on purity
        P["emp_d"] = np.array([0.0])
        P["tau_w"] = rng.normal(size=(S,)) * 0.1              # stilling head 止
        P["tau_0"] = np.array([0.0])
        P["Wrec"] = rng.normal(size=(S, F_SELF)) * 0.1        # 本末 closure decoder
        P["brec"] = np.zeros(F_SELF)
        self.P = P
        self.grads = {k: np.zeros_like(v) for k, v in P.items()}
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    # ------------------------------------------------------------------ utils
    def n_params(self):
        return sum(v.size for v in self.P.values())

    def inclusion(self):
        """ι (K×K): realm i includes realm j with weight ι[i,j] > 0 (性具)."""
        P = self.P
        raw = (self.iota_floor + softplus(P["Theta"])) * \
              (self.rowmask[:, None] * self.rowmask[None, :])
        rowsum = raw.sum(axis=1) + 1e-9
        iota_n = raw / rowsum[:, None]
        return raw, rowsum, iota_n

    # ---------------------------------------------------------------- forward
    def forward(self, D, e_target=None, loss_w=None):
        """
        D: dict with x_self, x_other, x_land, y_read, y_act.
        e_target: optional fixed target for the emptiness head (used by the
                  gradient check so the target is a constant during the check).
        loss_w: dict of loss weights (six-gates curriculum); default all on.
        Returns loss, cache.
        """
        P = self.P
        N = D["x_self"].shape[0]
        X = [D["x_self"], D["x_other"], D["x_land"]]
        lw = dict(read=1.0, act=1.0, emp=1.0, rec=1.0)
        if loss_w:
            lw.update(loss_w)

        # (a) encode each world into a K×S sheet, mask dead rows (severed)
        Z, E = [], []
        for w in range(NW):
            z = X[w] @ P[f"Win{w}"] + P[f"bin{w}"]
            e = np.tanh(z).reshape(N, K, S) * self.rowmask[None, :, None]
            Z.append(z)
            E.append(e)

        # (b) mutual inclusion: V_w = (I + ι_n) E_w   (every realm sees every realm)
        raw, rowsum, iota_n = self.inclusion()
        A = np.eye(K) + iota_n
        V = [np.einsum("ij,njs->nis", A, E[w]) for w in range(NW)]

        # (c) 觀 read: realm j's own row recognises realm j outside
        r_logit = P["rho0"][None, :].repeat(N, 0)
        for w in range(NW):
            r_logit = r_logit + np.einsum("nks,ks->nk", V[w], P["rho"][:, :, w])
        p = softmax(r_logit)

        # (d) 空 emptiness: predicted probability that the provisional reading is wrong
        logp = np.log(p + 1e-300)
        Hent = -(p * logp).sum(1)                      # entropy of the reading
        purity = (p * p).sum(1)                        # Σ p²
        pool = V[1].mean(axis=1)                       # N×S  (other-world, pooled over realms)
        e_logit = P["emp_a"][0] * Hent + P["emp_b"][0] * purity + pool @ P["emp_w"] + P["emp_d"][0]
        e = sigmoid(e_logit)

        # (e) 止 stilling: manifestation temperature from the self-world
        spool = V[0].mean(axis=1)                      # N×S
        tpre = spool @ P["tau_w"] + P["tau_0"][0]
        tau = self.tau_min + softplus(tpre)

        # (f) 假 bids to manifest, reconciled through 中
        pR = p @ P["R"]                                # response drive from the reading
        ell = P["beta_b"][None, :].repeat(N, 0)
        for w in range(NW):
            ell = ell + np.einsum("nks,ks->nk", V[w], P["alpha"][:, :, w])
        ell = ell + (1.0 - e)[:, None] * pR + e[:, None] * P["delta"][None, :]

        # (g) gate: forbidden realms receive exactly zero manifestation
        ellg = ell / tau[:, None]
        mfst = masked_softmax(ellg, self.gate[None, :].repeat(N, 0))

        # (h) manifest state and the 本末 closure
        Mst = np.einsum("nk,nks->ns", mfst, V[0])
        xhat = Mst @ P["Wrec"] + P["brec"]

        # (i) losses
        yr, ya = D["y_read"], D["y_act"]
        L_read = -np.mean(np.log(p[np.arange(N), yr] + 1e-300))
        L_act = -np.mean(np.log(mfst[np.arange(N), ya] + 1e-300))
        if e_target is None:
            e_target = (p.argmax(1) != yr).astype(float)
        L_emp = -np.mean(e_target * np.log(e + 1e-12) + (1 - e_target) * np.log(1 - e + 1e-12))
        L_rec = np.mean((xhat - X[0]) ** 2)
        L_sup = 0.0
        if self.lam > 0:
            L_sup = np.mean([np.mean(E[w][:, LOWER, :] ** 2) for w in range(NW)])
        loss = lw["read"] * L_read + lw["act"] * L_act + lw["emp"] * self.beta * L_emp \
            + lw["rec"] * self.gamma * L_rec + self.lam * L_sup

        cache = dict(N=N, X=X, Z=Z, E=E, raw=raw, rowsum=rowsum, iota_n=iota_n, A=A, V=V,
                     r_logit=r_logit, p=p, logp=logp, Hent=Hent, purity=purity, pool=pool,
                     e_logit=e_logit, e=e, spool=spool, tpre=tpre, tau=tau, pR=pR, ell=ell,
                     ellg=ellg, mfst=mfst, Mst=Mst, xhat=xhat, e_target=e_target, lw=lw,
                     parts=dict(read=L_read, act=L_act, emp=L_emp, rec=L_rec, sup=L_sup))
        return loss, cache

    # --------------------------------------------------------------- backward
    def backward(self, D, c):
        P, G = self.P, self.grads
        for k in G:
            G[k][...] = 0.0
        N = c["N"]
        X, E, V, A = c["X"], c["E"], c["V"], c["A"]
        p, e, tau, mfst = c["p"], c["e"], c["tau"], c["mfst"]
        lw = c["lw"]
        yr, ya = D["y_read"], D["y_act"]
        dV = [np.zeros_like(V[w]) for w in range(NW)]

        # --- 本末 closure ---------------------------------------------------
        dxhat = lw["rec"] * self.gamma * 2.0 * (c["xhat"] - X[0]) / (N * F_SELF)
        G["Wrec"] += c["Mst"].T @ dxhat
        G["brec"] += dxhat.sum(0)
        dM = dxhat @ P["Wrec"].T                                   # N×S
        dm = np.einsum("ns,nks->nk", dM, V[0])                      # from M = Σ m_i V0_i
        dV[0] += mfst[:, :, None] * dM[:, None, :]

        # --- 假 act loss through the gated softmax --------------------------
        dm[np.arange(N), ya] += -lw["act"] / (N * (mfst[np.arange(N), ya] + 1e-300))
        dellg = mfst * (dm - (dm * mfst).sum(1, keepdims=True))     # masked entries: mfst=0 → 0
        dell = dellg / tau[:, None]
        dtau = -(dellg * c["ell"]).sum(1) / (tau ** 2)

        # --- 止 stilling head ------------------------------------------------
        dtpre = dtau * sigmoid(c["tpre"])
        G["tau_w"] += c["spool"].T @ dtpre
        G["tau_0"] += dtpre.sum()
        dspool = dtpre[:, None] * P["tau_w"][None, :]
        dV[0] += dspool[:, None, :] / K

        # --- bids -----------------------------------------------------------
        for w in range(NW):
            G["alpha"][:, :, w] += np.einsum("nk,nks->ks", dell, V[w])
            dV[w] += np.einsum("nk,ks->nks", dell, P["alpha"][:, :, w])
        G["beta_b"] += dell.sum(0)
        G["R"] += p.T @ ((1.0 - e)[:, None] * dell)
        G["delta"] += (e[:, None] * dell).sum(0)
        dp = (1.0 - e)[:, None] * (dell @ P["R"].T)
        de = (dell * (P["delta"][None, :] - c["pR"])).sum(1)

        # --- 空 emptiness head ----------------------------------------------
        et = c["e_target"]
        de_logit = de * e * (1 - e) + lw["emp"] * self.beta * (e - et) / N
        G["emp_a"] += (de_logit * c["Hent"]).sum()
        G["emp_b"] += (de_logit * c["purity"]).sum()
        G["emp_w"] += c["pool"].T @ de_logit
        G["emp_d"] += de_logit.sum()
        dpool = de_logit[:, None] * P["emp_w"][None, :]
        dV[1] += dpool[:, None, :] / K
        dHent = de_logit * P["emp_a"][0]
        dpurity = de_logit * P["emp_b"][0]
        dp += dHent[:, None] * (-(c["logp"] + 1.0)) + dpurity[:, None] * (2.0 * p)

        # --- 觀 read loss ---------------------------------------------------
        dp[np.arange(N), yr] += -lw["read"] / (N * (p[np.arange(N), yr] + 1e-300))
        dr = p * (dp - (dp * p).sum(1, keepdims=True))
        for w in range(NW):
            G["rho"][:, :, w] += np.einsum("nk,nks->ks", dr, V[w])
            dV[w] += np.einsum("nk,ks->nks", dr, P["rho"][:, :, w])
        G["rho0"] += dr.sum(0)

        # --- mutual inclusion: V = A E ------------------------------------
        dA = np.zeros((K, K))
        dE = []
        for w in range(NW):
            dA += np.einsum("nis,njs->ij", dV[w], E[w])
            dE.append(np.einsum("ij,nis->njs", A, dV[w]))
        # suppression penalty on lower rows
        if self.lam > 0:
            for w in range(NW):
                dE[w][:, LOWER, :] += self.lam * 2.0 * E[w][:, LOWER, :] / (NW * N * len(LOWER) * S)
        # ι_n = raw / rowsum
        raw, rowsum, iota_n = c["raw"], c["rowsum"], c["iota_n"]
        d_raw = dA / rowsum[:, None] - ((dA * raw).sum(1) / rowsum ** 2)[:, None]
        mask2 = self.rowmask[:, None] * self.rowmask[None, :]
        G["Theta"] += d_raw * mask2 * sigmoid(P["Theta"])

        # --- encoders --------------------------------------------------------
        for w in range(NW):
            dz = (dE[w] * self.rowmask[None, :, None] * (1.0 - np.tanh(c["Z"][w]).reshape(N, K, S) ** 2)).reshape(N, K * S)
            G[f"Win{w}"] += X[w].T @ dz
            G[f"bin{w}"] += dz.sum(0)
        return G

    # ------------------------------------------------------------------ Adam
    def adam_step(self, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.t += 1
        for k in self.P:
            g = self.grads[k] + wd * self.P[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # -------------------------------------------------------------- readouts
    def predict(self, D):
        _, c = self.forward(D)
        return c["p"], c["mfst"], c["e"], c["tau"]

    def one_thought(self, D, i=0):
        """Materialise the 3,000-fold lattice of a single moment of thought."""
        _, c = self.forward(subset(D, [i]))
        Ew = [c["E"][w][0] for w in range(NW)]                 # K×S each
        T = np.zeros((K, K, S, NW))
        for w in range(NW):
            for a in range(K):
                for b in range(K):
                    T[a, b, :, w] = c["A"][a, b] * Ew[w][b]     # realm b as present within realm a
        return T, c

    def get_flat(self):
        return np.concatenate([v.ravel() for v in self.P.values()])

    def set_flat(self, flat):
        i = 0
        for k, v in self.P.items():
            n = v.size
            v[...] = flat[i:i + n].reshape(v.shape)
            i += n

    def grads_flat(self):
        return np.concatenate([v.ravel() for v in self.grads.values()])


# ==============================================================================
#  4.  GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(config, lam=1.0, seed=7, n=6, eps=1e-5, abs_tol=1e-7, rel_tol=1e-5,
                   rel_floor=1e-4, verbose=True):
    """
    Central finite differences over EVERY parameter (≈5,700 of them), in float64.
    Two criteria must both hold:
      * max |analytic − numeric|  < abs_tol            (catches any real bug)
      * max relative error over entries whose gradient is not negligible
        (|g| > rel_floor) < rel_tol                     (catches sign/scale bugs)
    Relative error is not reported on gradients of order 1e-7, where central
    differences on a loss of order 10 are dominated by floating-point roundoff.
    """
    world = make_world(3)
    D = make_dataset(n, world, seed=seed)
    model = YinianLattice(config, lam=lam, seed=seed)
    # scale weights up so every nonlinearity is exercised away from the origin
    for k in model.P:
        model.P[k] = model.P[k] + 0.4 * np.random.default_rng(seed + 1).normal(size=model.P[k].shape)
    loss, c = model.forward(D)
    e_t = c["e_target"].copy()            # freeze the emptiness target
    loss, c = model.forward(D, e_target=e_t)
    model.backward(D, c)
    g_an = model.grads_flat()
    flat = model.get_flat()
    g_num = np.zeros_like(flat)
    for i in range(flat.size):
        f0 = flat.copy(); f0[i] += eps
        model.set_flat(f0); lp, _ = model.forward(D, e_target=e_t)
        f1 = flat.copy(); f1[i] -= eps
        model.set_flat(f1); lm, _ = model.forward(D, e_target=e_t)
        g_num[i] = (lp - lm) / (2 * eps)
    model.set_flat(flat)
    num = np.abs(g_an - g_num)
    big = (np.abs(g_an) + np.abs(g_num)) > rel_floor
    rel = num[big] / (np.abs(g_an[big]) + np.abs(g_num[big]))
    worst_abs = num.max()
    worst_rel = rel.max() if big.any() else 0.0
    live = (np.abs(g_an) + np.abs(g_num)) > 1e-12
    ok = (worst_abs < abs_tol) and (worst_rel < rel_tol)
    if verbose:
        print(f"  [{config:<10}] params={flat.size:5d}  live={live.sum():5d}  checked={big.sum():5d}  "
              f"max|an-num|={worst_abs:.2e}  max rel err={worst_rel:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, worst_rel


# ==============================================================================
#  5.  TRAINING — the Six Wondrous Gates as a curriculum  (六妙門)
# ==============================================================================
#
#  Zhiyi's Liumiao famen orders breath-meditation in six gates:
#    數 count · 隨 follow · 止 stop · 觀 contemplate · 還 return · 淨 purify.
#  We use the sequence as a loss-weight schedule.  The read head is trained
#  first (count/follow), manifestation is added (stop), then the emptiness
#  head (contemplate), then the 本末 closure (return), and finally everything
#  is trained together with light weight decay (purify).

def six_gates(epoch, n_epochs):
    f = epoch / max(1, n_epochs - 1)
    if f < 0.15:   return "數 count",       dict(read=1.0, act=0.0, emp=0.0, rec=0.0), 0.0
    if f < 0.30:   return "隨 follow",      dict(read=1.0, act=0.3, emp=0.0, rec=0.0), 0.0
    if f < 0.45:   return "止 stop",        dict(read=1.0, act=1.0, emp=0.0, rec=0.0), 0.0
    if f < 0.60:   return "觀 contemplate", dict(read=1.0, act=1.0, emp=1.0, rec=0.0), 0.0
    if f < 0.80:   return "還 return",      dict(read=1.0, act=1.0, emp=1.0, rec=1.0), 0.0
    return "淨 purify", dict(read=1.0, act=1.0, emp=1.0, rec=1.0), 1e-4


def train(model, Dtr, Dva, epochs=30, batch=128, lr=3e-3, seed=0, log=True):
    rng = np.random.default_rng(seed)
    n = Dtr["x_self"].shape[0]
    hist = []
    last_gate = None
    for ep in range(epochs):
        gate, lw, wd = six_gates(ep, epochs)
        if log and gate != last_gate:
            print(f"     gate → {gate}")
            last_gate = gate
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            B = subset(Dtr, idx)
            loss, c = model.forward(B, loss_w=lw)
            model.backward(B, c)
            model.adam_step(lr=lr, wd=wd)
            tot += loss * len(idx)
        vl, _ = model.forward(Dva)
        hist.append((tot / n, vl))
        if log and (ep % 5 == 4 or ep == epochs - 1):
            m = evaluate(model, Dva)
            print(f"     ep {ep + 1:3d}  train {tot / n:.4f}  val {vl:.4f}  "
                  f"read {m['read_all']:.3f}  act {m['act_all']:.3f}  viol {m['violation']:.4f}")
    return hist


# ==============================================================================
#  6.  EVALUATION
# ==============================================================================

def evaluate(model, D):
    p, m, e, tau = model.predict(D)
    yr, ya = D["y_read"], D["y_act"]
    pr, pa = p.argmax(1), m.argmax(1)
    lower = np.isin(yr, LOWER)
    upper = ~lower
    wrong = (pr != yr).astype(float)
    # emptiness calibration: AUROC of e vs wrong, and Brier score
    order = np.argsort(e)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(e) + 1)
    n_pos, n_neg = wrong.sum(), (1 - wrong).sum()
    auroc = (ranks[wrong == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12) if n_pos > 0 and n_neg > 0 else float("nan")
    brier = np.mean((e - wrong) ** 2)
    ent_m = -(m * np.log(m + 1e-300)).sum(1)
    hot = D["arousal"] > 0.75
    return dict(
        read_all=np.mean(pr == yr),
        read_lower=np.mean(pr[lower] == yr[lower]),
        read_upper=np.mean(pr[upper] == yr[upper]),
        act_all=np.mean(pa == ya),
        act_vs_lower=np.mean(pa[lower] == ya[lower]),
        act_vs_upper=np.mean(pa[upper] == ya[upper]),
        violation=np.mean(np.isin(pa, LOWER)),
        lower_mass=np.mean(m[:, LOWER].sum(1)),
        emp_auroc=auroc, emp_brier=brier, emp_mean=np.mean(e),
        emp_on_lower=np.mean(e[lower]), emp_on_upper=np.mean(e[upper]),
        tau_mean=np.mean(tau), tau_hot=np.mean(tau[hot]), tau_cool=np.mean(tau[~hot]),
        mfst_entropy=np.mean(ent_m),
    )


def linear_probe_accuracy(Dtr, Dte):
    """A least-squares linear reader of x_other -> realm. Sign-faces defeat it."""
    Xtr = np.concatenate([Dtr["x_other"], np.ones((len(Dtr["x_other"]), 1))], 1)
    Ytr = np.eye(K)[Dtr["y_read"]]
    Wls = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
    Xte = np.concatenate([Dte["x_other"], np.ones((len(Dte["x_other"]), 1))], 1)
    return np.mean((Xte @ Wls).argmax(1) == Dte["y_read"])


# ==============================================================================
#  7.  SELF-TESTS — structural invariants
# ==============================================================================

def test_lattice_is_three_thousand():
    world = make_world(1); D = make_dataset(4, world, seed=1)
    mdl = YinianLattice("ZHIYI", seed=1)
    T, c = mdl.one_thought(D, 0)
    assert T.shape == (K, K, S, NW) and T.size == 3000, T.shape
    # realm a's view of realm b, summed, equals V_a
    assert np.allclose(T.sum(axis=1)[:, :, 0], c["V"][0][0])
    return "one thought materialises as exactly 10×10×10×3 = 3000 numbers"


def test_mutual_inclusion_floor():
    mdl = YinianLattice("ZHIYI", seed=2)
    mdl.P["Theta"] = np.random.default_rng(2).normal(size=(K, K)) * 40 - 60   # push toward -inf
    raw, rowsum, iota_n = mdl.inclusion()
    assert np.all(raw >= mdl.iota_floor - 1e-12)
    assert np.all(iota_n > 0)
    assert np.allclose(iota_n.sum(1), 1.0)
    return "every realm includes every realm with weight ≥ floor, whatever the weights (性具)"


def test_gate_gives_exactly_zero():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(50, K)) * 30
    g = np.ones(K); g[LOWER] = 0
    m = masked_softmax(z, g[None, :].repeat(50, 0))
    assert np.all(m[:, LOWER] == 0.0)
    assert np.allclose(m.sum(1), 1.0) and np.all(m >= 0)
    return "the 止 gate gives forbidden realms exactly 0.0 manifestation, never ε"


def test_severed_rows_are_dead():
    world = make_world(4); D = make_dataset(16, world, seed=4)
    mdl = YinianLattice("SEVERED", seed=4)
    _, c = mdl.forward(D)
    for w in range(NW):
        assert np.all(c["E"][w][:, LOWER, :] == 0.0)
        assert np.all(c["V"][w][:, LOWER, :] == 0.0)
    assert np.all(c["raw"][LOWER, :] == 0.0) and np.all(c["raw"][:, LOWER] == 0.0)
    assert np.allclose(c["r_logit"][:, LOWER], mdl.P["rho0"][LOWER])   # bias only: cannot discriminate
    return "SEVERED removes the lower realms from the lattice; their recognisers reduce to a bias"


def test_perturbation_reaches_every_realm():
    """非縱非橫: a change in any realm's encoding changes every realm's view (mutual);
    in SEVERED a change in a dead row reaches nothing."""
    world = make_world(5); D = make_dataset(1, world, seed=5)
    mdl = YinianLattice("ZHIYI", seed=5)
    _, c = mdl.forward(D)
    A = c["A"]
    assert np.all(A > 0), "some realm does not include some realm"
    sev = YinianLattice("SEVERED", seed=5)
    _, cs = sev.forward(D)
    assert np.all(cs["A"][:, LOWER][np.ix_(UPPER, range(len(LOWER)))] == 0.0)
    return "a perturbation of any realm reaches every realm (互具); severed rows reach nothing"


def test_manifestation_is_distribution_and_tau_bounded():
    world = make_world(6); D = make_dataset(64, world, seed=6)
    for cfg in ("OPEN", "ZHIYI", "SEVERED", "SUPPRESSED"):
        mdl = YinianLattice(cfg, seed=6)
        p, m, e, tau = mdl.predict(D)
        assert np.allclose(m.sum(1), 1.0) and np.all(m >= 0)
        assert np.allclose(p.sum(1), 1.0)
        assert np.all(tau >= mdl.tau_min)
        assert np.all((e > 0) & (e < 1))
        if cfg != "OPEN":
            assert np.all(m[:, LOWER] == 0.0)
    return "manifestation and reading are distributions; τ ≥ τ_min; e ∈ (0,1); gate holds in all gated configs"


def test_response_table():
    assert RESPONSE.shape == (K, N_CONTEXT)
    assert np.all(np.isin(RESPONSE, UPPER))
    assert set(np.unique(RESPONSE)) == set(UPPER.tolist())
    # lower-realm beings require distinct responses in at least one context
    for a in LOWER:
        for b in LOWER:
            if a < b:
                assert np.any(RESPONSE[a] != RESPONSE[b])
    return "every response is an upper realm; all six upper realms are used; lower realms need distinct responses"


def test_faces_defeat_linear_reader():
    world = make_world(7)
    Dtr = make_dataset(4000, world, seed=70); Dte = make_dataset(2000, world, seed=71)
    acc = linear_probe_accuracy(Dtr, Dte)
    assert acc < 0.45, acc
    return f"sign-faced realm cues defeat a linear reader (least-squares probe = {acc:.3f})"


def test_dataset_balance():
    world = make_world(8); D = make_dataset(5000, world, seed=8)
    cnt = np.bincount(D["y_read"], minlength=K)
    assert cnt.min() > 300
    assert set(np.unique(D["y_act"])) <= set(UPPER.tolist())
    return "every realm appears ≥300 times in 5000 encounters; targets are always upper realms"


def test_training_reduces_loss():
    world = make_world(9); D = make_dataset(512, world, seed=9)
    mdl = YinianLattice("ZHIYI", seed=9)
    l0, _ = mdl.forward(D)
    for _ in range(40):
        l, c = mdl.forward(D); mdl.backward(D, c); mdl.adam_step(lr=5e-3)
    l1, _ = mdl.forward(D)
    assert l1 < 0.7 * l0, (l0, l1)
    return f"40 Adam steps reduce the loss {l0:.3f} → {l1:.3f}"


def test_gradient_checks():
    ok = True
    for cfg, lam in (("OPEN", 0.0), ("ZHIYI", 0.0), ("SEVERED", 0.0), ("SUPPRESSED", 1.0)):
        passed, worst = gradient_check(cfg, lam=lam, verbose=True)
        ok = ok and passed
    assert ok
    return "finite-difference gradient check passes in all four configurations"


TESTS = [test_lattice_is_three_thousand, test_mutual_inclusion_floor, test_gate_gives_exactly_zero,
         test_severed_rows_are_dead, test_perturbation_reaches_every_realm,
         test_manifestation_is_distribution_and_tau_bounded, test_response_table,
         test_faces_defeat_linear_reader, test_dataset_balance, test_training_reduces_loss,
         test_gradient_checks]


# ==============================================================================
#  8.  MAIN
# ==============================================================================

def rule(ch="=", n=96):
    print(ch * n)


def main():
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    rule()
    print(" YINIAN LATTICE — Zhiyi 智顗 (538–597) · chapter 0173 · Encyclopedia of Lost Minds")
    print(" one thought = 10 realms × 10 realms-within × 10 suchnesses × 3 worlds = 3,000")
    rule()

    # ---- self-tests --------------------------------------------------------
    print("\n[1] SELF-TESTS")
    n_ok = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"  ✓ {t.__name__:<45} {msg}")
            n_ok += 1
        except AssertionError as ex:
            print(f"  ✗ {t.__name__:<45} FAILED: {ex}")
    print(f"  {n_ok}/{len(TESTS)} passed")
    if n_ok != len(TESTS):
        sys.exit(1)

    # ---- data --------------------------------------------------------------
    print("\n[2] DATA — synthetic encounters")
    world = make_world(11)
    Dtr = make_dataset(12000, world, seed=100)
    Dva = make_dataset(3000, world, seed=101)
    lin = linear_probe_accuracy(Dtr, Dva)
    print(f"  train 12000 / val 3000 encounters · 10 realms × 4 contexts · sign-faced cues")
    print(f"  linear least-squares reader of the other's realm: {lin:.3f}  (chance 0.100)")

    # ---- train the four minds ----------------------------------------------
    print("\n[3] TRAINING — four minds, same data, same seed, thirty epochs, six-gates curriculum")
    configs = [("OPEN", 0.0), ("ZHIYI", 0.0), ("SEVERED", 0.0), ("SUPPRESSED", 1.0)]
    models, results = {}, {}
    for cfg, lam in configs:
        print(f"\n  ── {cfg}{'' if lam == 0 else f' (λ={lam})'} ──")
        mdl = YinianLattice(cfg, lam=lam, seed=42)
        print(f"     parameters: {mdl.n_params()}")
        train(mdl, Dtr, Dva, epochs=30, batch=128, lr=3e-3, seed=42)
        models[cfg] = mdl
        results[cfg] = evaluate(mdl, Dva)

    # ---- results table -----------------------------------------------------
    print("\n[4] RESULTS on 3,000 held-out encounters")
    names = [c for c, _ in configs]
    rows = [("read: all beings", "read_all"), ("read: beings in the four lower realms", "read_lower"),
            ("read: beings in the six upper realms", "read_upper"),
            ("act: correct response, all", "act_all"), ("act: correct response to lower-realm beings", "act_vs_lower"),
            ("act: correct response to upper-realm beings", "act_vs_upper"),
            ("violation: manifested a lower realm", "violation"),
            ("mass on lower realms in manifestation", "lower_mass"),
            ("空 emptiness AUROC (predicts own read errors)", "emp_auroc"),
            ("空 emptiness Brier", "emp_brier"),
            ("空 mean on lower-realm beings", "emp_on_lower"), ("空 mean on upper-realm beings", "emp_on_upper"),
            ("止 τ mean (all)", "tau_mean"), ("止 τ when self is hot (arousal>.75)", "tau_hot"),
            ("止 τ when self is cool", "tau_cool"), ("manifestation entropy", "mfst_entropy")]
    print(f"  {'metric':<48}" + "".join(f"{n:>12}" for n in names))
    for label, key in rows:
        print(f"  {label:<48}" + "".join(f"{results[n][key]:>12.4f}" for n in names))

    # ---- λ sweep for SUPPRESSED --------------------------------------------
    print("\n[5] SUPPRESSION SWEEP — penalising the lower rows instead of gating them")
    print(f"  {'λ':>8}{'read lower':>14}{'read upper':>14}{'act vs lower':>14}{'violation':>12}{'row energy':>12}")
    for lam in (0.0, 0.3, 3.0, 30.0):
        mdl = YinianLattice("SUPPRESSED" if lam > 0 else "ZHIYI", lam=lam, seed=42)
        train(mdl, Dtr, Dva, epochs=30, batch=128, lr=3e-3, seed=42, log=False)
        r = evaluate(mdl, Dva)
        _, c = mdl.forward(subset(Dva, np.arange(512)))
        energy = np.mean([np.mean(c["E"][w][:, LOWER, :] ** 2) for w in range(NW)])
        print(f"  {lam:>8.1f}{r['read_lower']:>14.4f}{r['read_upper']:>14.4f}{r['act_vs_lower']:>14.4f}"
              f"{r['violation']:>12.4f}{energy:>12.4f}")

    # ---- one thought, inspected --------------------------------------------
    print("\n[6] ONE THOUGHT — an encounter with a being in hell, context 'conflict', inspected in each mind")
    idx = np.where((Dva["y_read"] == 0) & (Dva["ctx"] == 1) & (Dva["arousal"] < 0.5))[0][0]
    print(f"  encounter #{idx}: other = {REALMS[0]}, context = conflict, correct response = {REALMS[RESPONSE[0, 1]]}")
    for cfg in names:
        T, c = models[cfg].one_thought(Dva, idx)
        p, m, e, tau = c["p"][0], c["mfst"][0], c["e"][0], c["tau"][0]
        presence = np.abs(T).sum(axis=(2, 3))          # K×K : realm b present within realm a
        print(f"\n  {cfg}")
        print(f"    reads the other as   : {REALMS[p.argmax()]:<22} p={p.max():.3f}   空 emptiness={e:.3f}   止 τ={tau:.3f}")
        print(f"    manifests            : {REALMS[m.argmax()]:<22} m={m.max():.3f}   mass on lower realms={m[LOWER].sum():.4f}")
        print(f"    hell present within the buddha row : {presence[9, 0]:.3f}    buddha present within the hell row : {presence[0, 9]:.3f}")
        print(f"    live entries in the 3000-lattice   : {int((np.abs(T) > 0).sum())} / 3000")

    # ---- inclusion matrix of the Zhiyi mind --------------------------------
    print("\n[7] LEARNED INCLUSION ι (ZHIYI): row = realm, column = realm present within it (row-normalised)")
    _, _, iota_n = models["ZHIYI"].inclusion()
    print("            " + " ".join(f"{r.split()[0][:6]:>6}" for r in REALMS))
    for i in range(K):
        print(f"  {REALMS[i].split()[0][:9]:<9} " + " ".join(f"{iota_n[i, j]:6.3f}" for j in range(K)))
    print(f"  minimum inclusion weight: {iota_n.min():.4f}  (floor guarantees > 0)")

    rule()
    print(f" done in {time.time() - t0:.1f}s")
    rule()


if __name__ == "__main__":
    main()

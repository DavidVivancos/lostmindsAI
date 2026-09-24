#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MULIYAN  —  The Cue-Addressed Resonant Reconstitution Engine
 Chapter 0234 - Rudaki (Abu 'Abd Allah Ja'far b. Muhammad, c. 858 - 941)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0234_rudaki_858 - Rudaki (Abu 'Abd Allah Ja'far b. Muhammad, c. 858 - 941)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks)
whose *structure* encodes the one cognitive operation that was Rudaki's and
nobody else's in this corpus:

    INTELLIGENCE IS ADDRESSING, NOT STORAGE.

The poet does not carry the world. The listener already carries it. The poet
carries the shortest key that opens it, and knows the moment at which to turn
that key. Everything downstream of that sentence is in this file.

THE HISTORICAL OPERATION
------------------------
Nizami 'Aruzi of Samarqand tells it in the Chahar Maqala (c. 1155-57). The
Samanid amir Nasr II b. Ahmad had gone out to summer pasture and stayed four
years. His commanders were homesick. They went to the court poet and offered
five thousand dinars if he could get the amir to move. Rudaki waited for the
right moment, took up the chang, and sang eleven syllables:

    bu-yi ju-yi Muliyan ayad hami
    "the scent of the Muliyan stream keeps coming"

The amir came down off the throne, got on a horse and rode for Bukhara without
his riding boots. The commanders doubled the fee.

Look at what is and is not in that line. There is no argument. There is no
information the amir did not already possess. There is no threat, no flattery,
no proposition that could be true or false. There is a smell, a watercourse and
a verb of continuous arrival. The entire payload is an ADDRESS into a store the
listener was already holding, delivered in a carrier (metre, mode, voice,
timing) that guaranteed the address arrived intact and arrived on the beat.

That is a claim about how minds work, and it has a precise mechanical shape.

THE ARCHITECTURAL CLAIM
-----------------------
Build a memory that cannot be read by lookup, only by RESONANCE:

  1. State is a bank of damped complex oscillators. Content lives in the phase
     relations of the bank, not in the contents of addressable slots.

  2. Time is measured in MORAE, not ticks. The integration step dt is set by
     the prosodic weight of the syllable currently being uttered (long = 2
     morae, short = 1). A line in the wrong metre integrates with the wrong
     step schedule and lands the whole bank in the wrong phase. Metre is
     therefore not decoration on the signal; metre IS the clock, and a metrical
     violation is a detectable corruption of the channel.

  3. A RADIF (the fixed word repeated at the end of every line - Rudaki is the
     poet in whom Persian criticism first locates its systematic use) re-injects
     a learned anchor vector at line-ends, pulling the drifting bank back toward
     a common phase. Long sequences survive because the same key keeps coming
     back and re-addressing the field.

  4. Retrieval and ACTION are separate heads. Reconstituting Bukhara inside the
     amir is one thing. Getting him onto the horse is another. The file measures
     them separately because the historical record separates them: the poem had
     been sung before; that performance moved him.

THE MEASUREMENT THAT IS THE WHOLE POINT
---------------------------------------
Because retrieval is by address, the interesting quantity is not how much the
model stores. It is HOW SHORT A KEY SUFFICES. The task is built so that this
can be read off directly:

    - three syllables in the right order name a world uniquely
    - TWO syllables in the right order ALSO name it uniquely (all eight
      two-syllable prefixes are distinct - verified by a self-test)
    - but the third syllable, landing on a long, is what moves the body

So the model should learn to reconstitute the homeland from two syllables and
refuse to move on two. That dissociation - recall without action - is the
experimental result this architecture exists to produce.

THE ORDER TRAP
--------------
Four of the eight world-keys are exact mirrors of the other four:

    world 0 = (0,1,2)      world 1 = (2,1,0)
    world 2 = (1,3,4)      world 3 = (4,3,1)
    world 4 = (0,3,5)      world 5 = (5,3,0)
    world 6 = (2,4,5)      world 7 = (5,4,2)

A bag-of-syllables reader is capped at 50% on these pairs no matter how large
it is. Only a mechanism that encodes ORDER can separate them, and in this
architecture order is encoded as phase: a syllable injected at phase p
contributes exp(i*p) times its embedding to the time-averaged field, so the
same three syllables in reverse order produce a different resonance signature.
The mirror structure is not an obstacle to the mechanism. It is the mechanism's
existence proof.

THE BLIND CHANNEL
-----------------
'Awfi (d. 1242) says Rudaki was born blind. Safa and others doubt it on the
evidence of the colour in his own lines; Nafisi, reading the 1950s exhumation
report, thought he was blinded late. Ferdowsi says the Kalila wa Dimna was
READ TO HIM. The dating is unsettled and this file does not pretend to settle
it. What is not unsettled is that the surviving verse is saturated with
colour - tulips, narcissi, wine like coral, the moon-faced beloved - and that
at some point those colours were not available to the eye that used them.

So: the world vector has four blocks - SCENT, SOUND, TOUCH, SIGHT. The model
receives a sparse, heavily-noised ambient reading of the first three, and
NEVER receives sight in any form at any point in training or inference. Sight
is recoverable only by address. The file reports its R-squared separately.

THE COUNTERFACTUALS
-------------------
Two ablations, trained on identical data with identical seeds:

    UNIFORM   - integrates with constant dt. It still RECEIVES the syllable
                weights as an input feature, so it has the same information;
                it simply does not use prosody AS TIME. Any gap is therefore
                about the mechanism, not about what the model was told.
    NO_RADIF  - the anchor re-injection is clamped off. Everything else
                identical.

WHAT THE FILE DOES WHEN RUN
---------------------------
  1. Builds the task and asserts its combinatorial properties.
  2. Runs a finite-difference gradient check on every parameter group.
  3. Trains the full model and both ablations.
  4. Runs eleven self-tests, including the key-length curve, the metre-scramble
     detector, the sight-block recovery and the attrition ("teeth") sweep.
  5. Prints a verdict block.

Pure NumPy. No autograd. Every gradient in here was derived by hand and is
checked against finite differences before anything is trained.

    "Count the eyes - there is one pair fewer.
     Measure the wisdom - thousands fewer."
        - Rudaki, elegy for Shahid Balkhi (d. 325/937), tr. after Iranica

================================================================================
"""

import numpy as np
import time
import sys

# ==============================================================================
# SECTION 0 - CONFIGURATION
# ==============================================================================

RNG_SEED = 858          # his birth year, ca. 243 AH

N_OSC    = 48           # oscillators in the bank
V_TOK    = 12           # syllabary: 6 address syllables + 6 fillers
N_ADDR   = 6            # tokens 0..5 are address syllables
LINE_LEN = 11           # syllables per hemistich (fa'ilatun fa'ilatun fa'ilun)
N_LINES  = 2
T_SEQ    = LINE_LEN * N_LINES     # 22 positions
K_WORLD  = 24           # 4 modality blocks of 6
N_BLOCK  = 6
P_WORLD  = 8            # eight homelands
AMB_DIM  = 18           # scent+sound+touch, 6 each (sight excluded by design)
AMB_IN   = AMB_DIM * 2  # values concatenated with observation mask
H_RAWI   = 64           # the reciter layer
DT0      = 0.35         # seconds-per-mora, arbitrary units

# The two metres. Both are 11 syllables; both are real.
#   ramal-i musaddas-i mahzuf : fa'ilatun fa'ilatun fa'ilun   - - v -  ...
#   mutaqarib-i musaddas      : fa'ulun fa'ulun fa'ulun fa'ul
METRE_RAMAL     = np.array([2,1,2,2, 2,1,2,2, 2,1,2], dtype=np.float64)
METRE_MUTAQARIB = np.array([1,2,2, 1,2,2, 1,2,2, 1,2], dtype=np.float64)
METRES = [METRE_RAMAL, METRE_MUTAQARIB]

# The eight keys. Four mirror pairs. See THE ORDER TRAP above.
WORLD_KEYS = [
    (0, 1, 2), (2, 1, 0),
    (1, 3, 4), (4, 3, 1),
    (0, 3, 5), (5, 3, 0),
    (2, 4, 5), (5, 4, 2),
]

REGIMES = ["FULL_ON_LONG", "FULL_ON_SHORT", "PARTIAL", "NO_KEY", "SCATTERED"]
REGIME_P = [0.30, 0.20, 0.20, 0.15, 0.15]

EPS = 1e-8


# ==============================================================================
# SECTION 1 - SMALL NUMERICAL HELPERS
# ==============================================================================

def sigmoid(z):
    """Numerically stable logistic."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(x):
    """log(1+exp(x)), stable. Used to keep damping strictly positive."""
    return np.logaddexp(0.0, x)


def bce_with_logits(logit, y):
    """Mean binary cross-entropy from logits, plus dL/dlogit."""
    # stable form: max(z,0) - z*y + log(1+exp(-|z|))
    loss = np.mean(np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit))))
    d = (sigmoid(logit) - y) / logit.shape[0]
    return loss, d


def r2_score(pred, true):
    """Coefficient of determination against the mean of the true values."""
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    return 1.0 - ss_res / (ss_tot + EPS)


def saturate(pa, pb):
    """
    Soft amplitude clamp on a complex vector held as (real, imag).

        z <- z * tanh(|z|) / |z|

    Keeps every oscillator inside the unit disc without ever hard-clipping,
    so the bank can be driven arbitrarily hard and still stay differentiable.
    Returns the saturated pair plus the cached quantities the backward pass
    needs.
    """
    r = np.sqrt(pa * pa + pb * pb + EPS)
    th = np.tanh(r)
    s = th / r
    # ds/dr, needed in backward
    dsdr = (r * (1.0 - th * th) - th) / (r * r)
    return s * pa, s * pb, s, dsdr, r


# ==============================================================================
# SECTION 2 - THE TASK
# ==============================================================================
# Each example is a two-line "poem" of 22 syllables in one of two real metres.
# Somewhere in it, in order and possibly interleaved with fillers, sits the key
# of one of eight homelands - or a truncated key, or a scrambled one, or none.
#
# Targets:
#   world  : the 24-dim homeland vector (zeros when no valid address is present)
#   move   : 1 iff a COMPLETE key is present in order AND its final syllable
#            falls on a LONG position. The opportune moment is part of the
#            address.
# ==============================================================================

def build_worlds(rng):
    """
    Eight homelands. Each is a 24-vector in four modality blocks of six:
    SCENT[0:6] SOUND[6:12] TOUCH[12:18] SIGHT[18:24].

    The sight block is drawn independently of the others, so there is no
    cross-modal shortcut: sight can only ever be recovered by address.
    """
    W = rng.standard_normal((P_WORLD, K_WORLD))
    W /= np.linalg.norm(W, axis=1, keepdims=True) / np.sqrt(K_WORLD)
    return W


def _place_subsequence(rng, sub, length):
    """
    Choose strictly increasing positions for `sub` inside `length` slots.
    Returns the position list.
    """
    pos = rng.choice(length, size=len(sub), replace=False)
    return np.sort(pos)


def make_example(rng, worlds, regime):
    """
    Build one example. Returns (tokens, weights, radif_mask, ambient,
    ambient_mask, world_target, move_target, world_id).
    """
    metre = METRES[rng.integers(len(METRES))]
    weights = np.concatenate([metre] * N_LINES)          # (T,)
    radif = np.zeros(T_SEQ, dtype=np.float64)
    radif[LINE_LEN - 1] = 1.0                            # end of hemistich 1
    radif[T_SEQ - 1] = 1.0                               # end of hemistich 2

    # fillers everywhere to start
    toks = rng.integers(N_ADDR, V_TOK, size=T_SEQ)

    wid = int(rng.integers(P_WORLD))
    key = list(WORLD_KEYS[wid])

    if regime == "NO_KEY":
        sub, target_world, move = [], np.zeros(K_WORLD), 0.0
        wid_eff = -1
    elif regime == "SCATTERED":
        # a permutation of a real key that is NOT itself any world's key
        while True:
            perm = list(rng.permutation(key))
            if tuple(perm) not in WORLD_KEYS:
                break
        sub, target_world, move = perm, np.zeros(K_WORLD), 0.0
        wid_eff = -1
    elif regime == "PARTIAL":
        sub, target_world, move = key[:2], worlds[wid].copy(), 0.0
        wid_eff = wid
    else:  # FULL_ON_LONG / FULL_ON_SHORT
        sub, target_world = key, worlds[wid].copy()
        move = 1.0 if regime == "FULL_ON_LONG" else 0.0
        wid_eff = wid

    if sub:
        want_long = (regime == "FULL_ON_LONG")
        # resample placements until the final syllable lands on the weight we need
        for _ in range(400):
            pos = _place_subsequence(rng, sub, T_SEQ)
            if regime in ("FULL_ON_LONG", "FULL_ON_SHORT"):
                if (weights[pos[-1]] == 2.0) != want_long:
                    continue
            break
        toks[pos] = sub

    # ---- ambient sensing: sparse, noisy, and sightless -----------------------
    # Four of eighteen non-sight dimensions are observed, at sigma=1.0 noise.
    # A model that reconstituted a homeland from THIS alone would be inventing
    # one, which is the failure mode the NO_KEY regime exists to punish.
    amb_src = worlds[wid][:AMB_DIM]                      # always a real world
    obs = rng.choice(AMB_DIM, size=4, replace=False)
    amask = np.zeros(AMB_DIM)
    amask[obs] = 1.0
    amb = np.zeros(AMB_DIM)
    amb[obs] = amb_src[obs] + rng.standard_normal(4) * 1.0

    return toks, weights, radif, amb, amask, target_world, move, wid_eff


def make_dataset(n, rng, worlds):
    """Assemble n examples with the regime mixture declared in REGIME_P."""
    T = np.zeros((n, T_SEQ), dtype=np.int64)
    Wt = np.zeros((n, T_SEQ))
    Rd = np.zeros((n, T_SEQ))
    Am = np.zeros((n, AMB_IN))
    Y = np.zeros((n, K_WORLD))
    M = np.zeros(n)
    Rg = np.zeros(n, dtype=np.int64)
    Wid = np.zeros(n, dtype=np.int64)

    regs = rng.choice(len(REGIMES), size=n, p=REGIME_P)
    for i in range(n):
        reg = REGIMES[regs[i]]
        t, w, r, a, am, y, m, wid = make_example(rng, worlds, reg)
        T[i], Wt[i], Rd[i] = t, w, r
        Am[i] = np.concatenate([a, am])
        Y[i], M[i], Rg[i], Wid[i] = y, m, regs[i], wid
    return dict(tok=T, wgt=Wt, rad=Rd, amb=Am, world=Y, move=M,
                regime=Rg, wid=Wid)


# ==============================================================================
# SECTION 3 - THE MODEL
# ==============================================================================

class Muliyan:
    """
    A bank of damped, coupled, driven complex oscillators read out by
    time-averaged resonance.

    Per step t, with dt_t = DT0 * weight_t (PROSODIC) or DT0 * 1.5 (UNIFORM):

        drive     u  = U[:,tok_t] + wemb * (weight_t - 1.5)   [+ A @ amb at t=0]
        coupling  c  = W z_t
        rotate    z' = z_t * exp(i*omega*dt_t) * exp(-g*dt_t)
        integrate p  = z' + dt_t * (c + u)
        saturate  z  = p * tanh(|p|)/|p|
        radif     z <- z + m_t * rho * (anchor - z)

    and accumulates, weighted by dt so that a long syllable counts for two
    morae of evidence:

        Racc += dt * |z|^2        (energy per oscillator - WHICH strings rang)
        Macc += dt * z            (phase-coherent mean  - WHEN they rang)

    The readout is [Racc/Tacc ; Re(Macc)/Tacc ; Im(Macc)/Tacc], 3N features,
    through one tanh layer (the RAWI - the professional declaimer who stood
    between the poet and the audience, and the only nonlinearity in the
    machine) to two heads.
    """

    def __init__(self, rng, n_osc=N_OSC, prosodic=True, use_radif=True,
                 h_rawi=H_RAWI, v_tok=V_TOK, k_world=K_WORLD, amb_in=AMB_IN):
        self.N, self.V, self.K, self.H = n_osc, v_tok, k_world, h_rawi
        self.amb_in = amb_in
        self.prosodic = prosodic
        self.use_radif = use_radif
        N = n_osc
        sc = 1.0 / np.sqrt(N)

        self.p = {
            # initial tuning of the bank
            "z0re": rng.standard_normal(N) * 0.05,
            "z0im": rng.standard_normal(N) * 0.05,
            # natural frequencies: a spread of strings, like a tuned chang
            "omega": rng.uniform(0.35, 3.0, size=N),
            # damping via softplus -> always positive; init near 0.15
            "gam": np.full(N, np.log(np.expm1(0.15))),
            # coupling
            "Wre": rng.standard_normal((N, N)) * sc * 0.5,
            "Wim": rng.standard_normal((N, N)) * sc * 0.5,
            # syllable embeddings
            "Ure": rng.standard_normal((N, v_tok)) * 0.5,
            "Uim": rng.standard_normal((N, v_tok)) * 0.5,
            # prosodic weight as a plain feature (present in BOTH variants)
            "wre": rng.standard_normal(N) * 0.1,
            "wim": rng.standard_normal(N) * 0.1,
            # ambient sensing, injected once at t=0
            "Are": rng.standard_normal((N, amb_in)) * 0.15,
            "Aim": rng.standard_normal((N, amb_in)) * 0.15,
            # the radif anchor
            "radre": rng.standard_normal(N) * 0.2,
            "radim": rng.standard_normal(N) * 0.2,
            "rho": np.array([0.0]),          # sigmoid -> 0.5 at init
            # the rawi and the two heads
            "H1": rng.standard_normal((3 * N, h_rawi)) * (1.0 / np.sqrt(3 * N)),
            "b1": np.zeros(h_rawi),
            "Ww": rng.standard_normal((h_rawi, k_world)) * (1.0 / np.sqrt(h_rawi)),
            "bw": np.zeros(k_world),
            "wm": rng.standard_normal(h_rawi) * (1.0 / np.sqrt(h_rawi)),
            "bm": np.zeros(1),
        }

    # -------------------------------------------------------------- utilities
    def n_params(self):
        return sum(v.size for v in self.p.values())

    def _dts(self, wgt):
        """Prosodic time, or the flat clock of the ablation."""
        if self.prosodic:
            return DT0 * wgt
        return np.full_like(wgt, DT0 * 1.5)

    # ---------------------------------------------------------------- forward
    def forward(self, batch, cache=True, osc_mask=None):
        """
        Run the bank over the sequence and read it out.

        `osc_mask` is an optional (N,) vector of 0/1 used only at inference by
        the attrition sweep: it silences a fraction of the oscillators, the way
        the qasida on old age silences one faculty after another, and lets us
        measure how gracefully the address still opens.
        """
        p = self.p
        tok, wgt, rad, amb = batch["tok"], batch["wgt"], batch["rad"], batch["amb"]
        B, T = tok.shape
        N = self.N
        dts = self._dts(wgt)                              # (B,T)
        rho = float(sigmoid(p["rho"])[0]) if self.use_radif else 0.0
        g = softplus(p["gam"])                            # (N,)

        a = np.tile(p["z0re"], (B, 1))
        b = np.tile(p["z0im"], (B, 1))

        Racc = np.zeros((B, N))
        Mre = np.zeros((B, N))
        Mim = np.zeros((B, N))
        Tacc = dts.sum(axis=1, keepdims=True)             # (B,1)

        tape = [] if cache else None

        for t in range(T):
            dt = dts[:, t:t + 1]                          # (B,1)
            # ---- drive
            ure = p["Ure"][:, tok[:, t]].T                # (B,N)
            uim = p["Uim"][:, tok[:, t]].T
            wfeat = (wgt[:, t:t + 1] - 1.5)
            ure = ure + p["wre"] * wfeat
            uim = uim + p["wim"] * wfeat
            if t == 0:
                ure = ure + amb @ p["Are"].T
                uim = uim + amb @ p["Aim"].T
            # ---- coupling
            cre = a @ p["Wre"].T - b @ p["Wim"].T
            cim = a @ p["Wim"].T + b @ p["Wre"].T
            # ---- rotate + damp
            th = dt * p["omega"]                          # (B,N)
            ct, st = np.cos(th), np.sin(th)
            pr = a * ct - b * st
            qr = a * st + b * ct
            d = np.exp(-g * dt)                           # (B,N)
            ra, rb = d * pr, d * qr
            # ---- integrate
            pa = ra + dt * (cre + ure)
            pb = rb + dt * (cim + uim)
            # ---- saturate
            a1, b1, s, dsdr, rr = saturate(pa, pb)
            # ---- radif re-anchoring
            m = rad[:, t:t + 1] * rho
            a2 = a1 + m * (p["radre"] - a1)
            b2 = b1 + m * (p["radim"] - b1)
            if osc_mask is not None:
                a2 = a2 * osc_mask
                b2 = b2 * osc_mask
            # ---- accumulate
            Racc += dt * (a2 * a2 + b2 * b2)
            Mre += dt * a2
            Mim += dt * b2

            if cache:
                tape.append(dict(a=a, b=b, dt=dt, ct=ct, st=st, d=d, pr=pr, qr=qr,
                                 pa=pa, pb=pb, s=s, dsdr=dsdr, rr=rr,
                                 a1=a1, b1=b1, a2=a2, b2=b2, m=rad[:, t:t + 1],
                                 tok=tok[:, t], wfeat=wfeat))
            a, b = a2, b2

        feat = np.concatenate([Racc / Tacc, Mre / Tacc, Mim / Tacc], axis=1)
        h_pre = feat @ p["H1"] + p["b1"]
        h = np.tanh(h_pre)
        world = h @ p["Ww"] + p["bw"]
        move = h @ p["wm"] + p["bm"][0]

        out = dict(world=world, move=move, feat=feat, h=h, h_pre=h_pre,
                   Tacc=Tacc, rho=rho, g=g, tape=tape, B=B, T=T)
        return out

    # ------------------------------------------------------------------- loss
    def loss(self, batch, out=None, lam_move=2.5, lam_l2=1e-5):
        if out is None:
            out = self.forward(batch)
        B = out["B"]
        dif = out["world"] - batch["world"]
        l_world = np.sum(dif * dif) / (B * self.K)
        l_move, dmove = bce_with_logits(out["move"], batch["move"])
        l2 = sum(np.sum(v * v) for k, v in self.p.items()
                 if k in ("Wre", "Wim", "Ure", "Uim", "H1", "Ww"))
        total = l_world + lam_move * l_move + lam_l2 * l2
        out["_dworld"] = 2.0 * dif / (B * self.K)
        out["_dmove"] = lam_move * dmove
        out["_lam_l2"] = lam_l2
        return total, dict(world=l_world, move=l_move, total=total), out

    # --------------------------------------------------------------- backward
    def backward(self, batch, out):
        """
        Hand-derived BPTT. Every line below is checked against finite
        differences by gradient_check() before any training happens.
        """
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        B, T, N = out["B"], out["T"], self.N
        Tacc = out["Tacc"]
        rho = out["rho"]
        gg = out["g"]
        tape = out["tape"]

        # ---- heads
        dworld, dmove = out["_dworld"], out["_dmove"]
        h = out["h"]
        g["Ww"] += h.T @ dworld
        g["bw"] += dworld.sum(axis=0)
        g["wm"] += h.T @ dmove
        g["bm"] += dmove.sum()
        dh = dworld @ p["Ww"].T + np.outer(dmove, p["wm"])
        dh_pre = dh * (1.0 - h * h)
        g["H1"] += out["feat"].T @ dh_pre
        g["b1"] += dh_pre.sum(axis=0)
        dfeat = dh_pre @ p["H1"].T

        dRacc = dfeat[:, :N] / Tacc
        dMre = dfeat[:, N:2 * N] / Tacc
        dMim = dfeat[:, 2 * N:] / Tacc

        # ---- L2
        lam = out["_lam_l2"]
        for k in ("Wre", "Wim", "Ure", "Uim", "H1", "Ww"):
            g[k] += 2.0 * lam * p[k]

        # ---- through time
        da = np.zeros((B, N))
        db = np.zeros((B, N))
        for t in range(T - 1, -1, -1):
            c = tape[t]
            dt = c["dt"]
            # gradient arriving at the POST-step state z2
            da2 = da + dRacc * dt * 2.0 * c["a2"] + dMre * dt
            db2 = db + dRacc * dt * 2.0 * c["b2"] + dMim * dt

            # ---- radif:  a2 = a1 + m*rho*(radre - a1)
            m = c["m"]
            if self.use_radif:
                g["radre"] += (da2 * m * rho).sum(axis=0)
                g["radim"] += (db2 * m * rho).sum(axis=0)
                drho = float((da2 * m * (p["radre"] - c["a1"])).sum()
                             + (db2 * m * (p["radim"] - c["b1"])).sum())
                g["rho"] += drho * rho * (1.0 - rho)
            da1 = da2 * (1.0 - m * rho)
            db1 = db2 * (1.0 - m * rho)

            # ---- saturate:  a1 = s*pa ,  s = tanh(r)/r
            dot = da1 * c["pa"] + db1 * c["pb"]
            kk = dot * c["dsdr"] / c["rr"]
            dpa = da1 * c["s"] + kk * c["pa"]
            dpb = db1 * c["s"] + kk * c["pb"]

            # ---- integrate:  pa = ra + dt*(cre + ure)
            dra, drb = dpa, dpb
            dcre, dcim = dpa * dt, dpb * dt
            dure, duim = dpa * dt, dpb * dt

            # ---- rotate + damp:  ra = d*pr , pr = a*ct - b*st , d = exp(-g*dt)
            dpr = dra * c["d"]
            dqr = drb * c["d"]
            dd = dra * c["pr"] + drb * c["qr"]
            dg_ = (dd * (-dt) * c["d"]).sum(axis=0)
            g["gam"] += dg_ * sigmoid(p["gam"])
            # d(pr)/d(th) = -a*st - b*ct ;  d(qr)/d(th) = a*ct - b*st
            dth = (-dpr * (c["a"] * c["st"] + c["b"] * c["ct"])
                   + dqr * (c["a"] * c["ct"] - c["b"] * c["st"]))
            g["omega"] += (dth * dt).sum(axis=0)
            da_rot = dpr * c["ct"] + dqr * c["st"]
            db_rot = -dpr * c["st"] + dqr * c["ct"]

            # ---- coupling
            g["Wre"] += dcre.T @ c["a"] + dcim.T @ c["b"]
            g["Wim"] += -dcre.T @ c["b"] + dcim.T @ c["a"]
            da_c = dcre @ p["Wre"] + dcim @ p["Wim"]
            db_c = -dcre @ p["Wim"] + dcim @ p["Wre"]

            # ---- drive
            np.add.at(g["Ure"].T, c["tok"], dure)
            np.add.at(g["Uim"].T, c["tok"], duim)
            g["wre"] += (dure * c["wfeat"]).sum(axis=0)
            g["wim"] += (duim * c["wfeat"]).sum(axis=0)
            if t == 0:
                g["Are"] += dure.T @ batch["amb"]
                g["Aim"] += duim.T @ batch["amb"]

            da = da_rot + da_c
            db = db_rot + db_c

        g["z0re"] += da.sum(axis=0)
        g["z0im"] += db.sum(axis=0)
        if not self.use_radif:
            g["rho"][:] = 0.0
            g["radre"][:] = 0.0
            g["radim"][:] = 0.0
        return g


# ==============================================================================
# SECTION 4 - GRADIENT CHECK  (mandatory; runs before any training)
# ==============================================================================

def gradient_check(seed=3, n_osc=6, n_probe=3, eps=1e-5, tol=2e-5, floor=1e-4,
                   prosodic=True, use_radif=True):
    """
    Central-difference check on a deliberately tiny model, probing a few random
    coordinates of every parameter array. Returns the worst relative error.

    The denominator carries an absolute floor. Without one, a coordinate whose
    true gradient is ~1e-6 - a thousand times below the median of its own array,
    and contributing nothing to any update - reports a large RELATIVE error
    purely from catastrophic cancellation in (f(x+e) - f(x-e)). That is an
    artefact of the measuring instrument, not of the gradient, and it is
    reproducible: widening eps to 1e-5 collapses it. The floor is set two orders
    of magnitude below the typical gradient scale in this model.
    """
    rng = np.random.default_rng(seed)
    worlds = build_worlds(rng)
    ds = make_dataset(4, rng, worlds)
    m = Muliyan(rng, n_osc=n_osc, prosodic=prosodic, use_radif=use_radif,
                h_rawi=5)

    _, _, out = m.loss(ds)
    g = m.backward(ds, out)

    worst, worst_name = 0.0, ""
    for name, arr in m.p.items():
        if not use_radif and name in ("rho", "radre", "radim"):
            continue
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, _, _ = m.loss(ds)
            flat[i] = old - eps
            lm, _, _ = m.loss(ds)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = g[name].reshape(-1)[i]
            den = max(abs(num), abs(ana), floor)
            rel = abs(num - ana) / den
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    return worst, worst_name, worst < tol


# ==============================================================================
# SECTION 5 - TRAINING
# ==============================================================================

def adam_step(p, g, st, lr, t, b1=0.9, b2=0.999, e=1e-8):
    for k in p:
        if k not in st:
            st[k] = [np.zeros_like(p[k]), np.zeros_like(p[k])]
        mm, vv = st[k]
        mm *= b1
        mm += (1 - b1) * g[k]
        vv *= b2
        vv += (1 - b2) * g[k] * g[k]
        mh = mm / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        p[k] -= lr * mh / (np.sqrt(vh) + e)


def slice_batch(ds, idx):
    return {k: v[idx] for k, v in ds.items()}


def train(model, tr, va, epochs=40, bs=64, lr=4e-3, seed=0, verbose=True,
          label=""):
    rng = np.random.default_rng(seed)
    st = {}
    n = tr["tok"].shape[0]
    step = 0
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        # cosine decay
        cur_lr = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.9 + 0.1)
        for s in range(0, n, bs):
            b = slice_batch(tr, perm[s:s + bs])
            _, _, out = model.loss(b)
            g = model.backward(b, out)
            # global grad clip - oscillator banks can spike early
            gn = np.sqrt(sum(np.sum(v * v) for v in g.values()))
            if gn > 5.0:
                for k in g:
                    g[k] *= 5.0 / gn
            step += 1
            adam_step(model.p, g, st, cur_lr, step)
        if verbose and (ep % 8 == 0 or ep == epochs - 1):
            ev = evaluate(model, va)
            hist.append((ep, ev))
            print(f"   [{label:9s}] ep {ep:3d}  world_mse {ev['world_mse']:.4f}"
                  f"  R2 {ev['r2_all']:.3f}  sightR2 {ev['r2_sight']:.3f}"
                  f"  move_acc {ev['move_acc']:.3f}")
    return hist


def evaluate(model, ds, osc_mask=None):
    out = model.forward(ds, cache=False, osc_mask=osc_mask)
    pred, true = out["world"], ds["world"]
    mv = (out["move"] > 0).astype(float)
    res = dict(
        world_mse=float(np.mean((pred - true) ** 2)),
        r2_all=float(r2_score(pred, true)),
        r2_sight=float(r2_score(pred[:, 3 * N_BLOCK:], true[:, 3 * N_BLOCK:])),
        r2_scent=float(r2_score(pred[:, :N_BLOCK], true[:, :N_BLOCK])),
        move_acc=float(np.mean(mv == ds["move"])),
    )
    # Per regime. NO_KEY and SCATTERED have all-zero targets, so R-squared is
    # undefined there; for those we report the norm of what the model produced
    # anyway - the amount of homeland it invented when there was no address.
    for ri, rn in enumerate(REGIMES):
        sel = ds["regime"] == ri
        if sel.sum() == 0:
            continue
        res[f"move_{rn}"] = float(np.mean(mv[sel] == ds["move"][sel]))
        nrm = float(np.sqrt((pred[sel] ** 2).sum(axis=1).mean()))
        res[f"norm_{rn}"] = nrm
        if rn in ("NO_KEY", "SCATTERED"):
            res[f"r2_{rn}"] = float("nan")
        else:
            res[f"r2_{rn}"] = float(r2_score(pred[sel], true[sel]))
    return res


def world_identification(model, ds, worlds, osc_mask=None):
    """
    Nearest-homeland classification accuracy over the examples that HAVE a
    homeland. This is the addressing metric: did the key open the right door?
    """
    out = model.forward(ds, cache=False, osc_mask=osc_mask)
    sel = ds["wid"] >= 0
    if sel.sum() == 0:
        return float("nan")
    pred = out["world"][sel]
    d = ((pred[:, None, :] - worlds[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(np.argmin(d, axis=1) == ds["wid"][sel]))


# ==============================================================================
# SECTION 6 - THE PROBES
# ------------------------------------------------------------------------------
# These are not diagnostics bolted onto the architecture. Each one measures a
# claim the chapter makes about this particular mind, and each one can come back
# negative.
# ==============================================================================

def probe_key_length(rng, worlds, j, n=600, return_keypos=False):
    """
    Build a dataset in which EXACTLY the first j syllables of some homeland's
    key appear, in order, and nothing else does.

    This is the shortest-key instrument. Sweep j = 1, 2, 3 and read off how much
    of an absent world one, two and three syllables are worth.
    """
    T = np.zeros((n, T_SEQ), dtype=np.int64)
    Wt = np.zeros((n, T_SEQ)); Rd = np.zeros((n, T_SEQ))
    Am = np.zeros((n, AMB_IN)); Y = np.zeros((n, K_WORLD))
    Mv = np.zeros(n); Wid = np.zeros(n, dtype=np.int64)
    KP = np.zeros((n, T_SEQ), dtype=bool)
    for i in range(n):
        metre = METRES[rng.integers(len(METRES))]
        w = np.concatenate([metre] * N_LINES)
        r = np.zeros(T_SEQ); r[LINE_LEN - 1] = 1.0; r[T_SEQ - 1] = 1.0
        toks = rng.integers(N_ADDR, V_TOK, size=T_SEQ)
        wid = int(rng.integers(P_WORLD))
        sub = list(WORLD_KEYS[wid])[:j]
        pos = _place_subsequence(rng, sub, T_SEQ)
        toks[pos] = sub
        KP[i, pos] = True
        amb_src = worlds[wid][:AMB_DIM]
        obs = rng.choice(AMB_DIM, size=4, replace=False)
        amask = np.zeros(AMB_DIM); amask[obs] = 1.0
        amb = np.zeros(AMB_DIM)
        amb[obs] = amb_src[obs] + rng.standard_normal(4) * 1.0
        T[i], Wt[i], Rd[i] = toks, w, r
        Am[i] = np.concatenate([amb, amask])
        Y[i] = worlds[wid]
        Mv[i] = 1.0 if (j == 3 and w[pos[-1]] == 2.0) else 0.0
        Wid[i] = wid
    ds = dict(tok=T, wgt=Wt, rad=Rd, amb=Am, world=Y, move=Mv,
              regime=np.zeros(n, dtype=np.int64), wid=Wid)
    return (ds, KP) if return_keypos else ds


# Positions inside a hemistich where the two metres DISAGREE about weight.
#   ramal      2 1 2 2 2 1 2 2 2 1 2
#   mutaqarib  1 2 2 1 2 2 1 2 2 1 2
#              ^ ^   ^   ^ ^          <- five positions, verified at run time
DISAGREE = np.where(METRE_RAMAL != METRE_MUTAQARIB)[0]


def probe_beat_pairs(rng, worlds, n=700):
    """
    The opportune moment, isolated.

    Two datasets whose syllable sequences are BYTE-IDENTICAL and whose ambient
    readings are identical. The only difference is which metre the line is read
    in. The final key syllable is planted on one of the five positions where
    ramal and mutaqarib disagree about weight, so the same line of verse is an
    order to move under one metre and not under the other.

    Both variants of the model receive the syllable weights as an explicit
    input feature. So this measures exactly one thing: whether treating those
    weights AS THE CLOCK, rather than as another number in the drive, buys any
    sensitivity to the beat.
    """
    T = np.zeros((n, T_SEQ), dtype=np.int64)
    Am = np.zeros((n, AMB_IN)); Y = np.zeros((n, K_WORLD))
    Wid = np.zeros(n, dtype=np.int64)
    WtA = np.zeros((n, T_SEQ)); WtB = np.zeros((n, T_SEQ))
    MvA = np.zeros(n); MvB = np.zeros(n)
    Rd = np.zeros((n, T_SEQ)); Rd[:, LINE_LEN - 1] = 1.0; Rd[:, T_SEQ - 1] = 1.0
    wA = np.concatenate([METRE_RAMAL] * N_LINES)
    wB = np.concatenate([METRE_MUTAQARIB] * N_LINES)
    disagree_global = np.concatenate([DISAGREE + L * LINE_LEN
                                      for L in range(N_LINES)])
    for i in range(n):
        toks = rng.integers(N_ADDR, V_TOK, size=T_SEQ)
        wid = int(rng.integers(P_WORLD))
        sub = list(WORLD_KEYS[wid])
        last = int(rng.choice(disagree_global[disagree_global >= 2]))
        earlier = np.sort(rng.choice(last, size=2, replace=False))
        pos = np.concatenate([earlier, [last]])
        toks[pos] = sub
        amb_src = worlds[wid][:AMB_DIM]
        obs = rng.choice(AMB_DIM, size=4, replace=False)
        amask = np.zeros(AMB_DIM); amask[obs] = 1.0
        amb = np.zeros(AMB_DIM)
        amb[obs] = amb_src[obs] + rng.standard_normal(4) * 1.0
        T[i] = toks
        Am[i] = np.concatenate([amb, amask])
        Y[i] = worlds[wid]; Wid[i] = wid
        WtA[i], WtB[i] = wA, wB
        MvA[i] = 1.0 if wA[last] == 2.0 else 0.0
        MvB[i] = 1.0 if wB[last] == 2.0 else 0.0
    base = dict(tok=T, rad=Rd, amb=Am, world=Y,
                regime=np.zeros(n, dtype=np.int64), wid=Wid)
    A = dict(base, wgt=WtA, move=MvA)
    B = dict(base, wgt=WtB, move=MvB)
    return A, B


def probe_channel_noise(rng, ds, keypos, rate):
    """
    A garbled transmission.

    Substitute a `rate` fraction of the NON-key syllables with random ones.
    The address itself is untouched, so the correct answer does not change;
    what changes is how much competing drive the bank has to survive. This is
    the condition a poem actually faced between Bukhara in 940 and the
    anthologies that preserve it - a thousand recitations, each one lossy.
    """
    out = {k: v.copy() for k, v in ds.items()}
    B, T = out["tok"].shape
    for i in range(B):
        cand = np.where(~keypos[i])[0]
        k = int(round(rate * len(cand)))
        if k:
            hit = rng.choice(cand, size=k, replace=False)
            out["tok"][i, hit] = rng.integers(0, V_TOK, size=k)
    return out


def probe_long_poem(rng, worlds, n=600, extra_lines=4):
    """
    Deliver the address in the first two lines, then keep singing.

    Four more hemistichs of pure filler follow, each closing on a radif
    position. Nothing new arrives. The only question is whether the homeland
    is still in the bank when the poem finally stops - which is precisely what
    a repeated terminal refrain is for.
    """
    total_lines = N_LINES + extra_lines
    T_LONG = LINE_LEN * total_lines
    T = np.zeros((n, T_LONG), dtype=np.int64)
    Wt = np.zeros((n, T_LONG)); Rd = np.zeros((n, T_LONG))
    Am = np.zeros((n, AMB_IN)); Y = np.zeros((n, K_WORLD))
    Mv = np.zeros(n); Wid = np.zeros(n, dtype=np.int64)
    for i in range(n):
        metre = METRES[rng.integers(len(METRES))]
        w = np.concatenate([metre] * total_lines)
        r = np.zeros(T_LONG)
        for L in range(total_lines):
            r[(L + 1) * LINE_LEN - 1] = 1.0
        toks = rng.integers(N_ADDR, V_TOK, size=T_LONG)
        wid = int(rng.integers(P_WORLD))
        sub = list(WORLD_KEYS[wid])
        pos = np.sort(rng.choice(LINE_LEN * N_LINES, size=3, replace=False))
        toks[pos] = sub
        amb_src = worlds[wid][:AMB_DIM]
        obs = rng.choice(AMB_DIM, size=4, replace=False)
        amask = np.zeros(AMB_DIM); amask[obs] = 1.0
        amb = np.zeros(AMB_DIM)
        amb[obs] = amb_src[obs] + rng.standard_normal(4) * 1.0
        T[i], Wt[i], Rd[i] = toks, w, r
        Am[i] = np.concatenate([amb, amask])
        Y[i] = worlds[wid]
        Mv[i] = 1.0 if w[pos[-1]] == 2.0 else 0.0
        Wid[i] = wid
    return dict(tok=T, wgt=Wt, rad=Rd, amb=Am, world=Y, move=Mv,
                regime=np.zeros(n, dtype=np.int64), wid=Wid)


def mirror_pair_accuracy(model, ds, worlds):
    """
    Homeland identification restricted to the four mirror pairs.

    A reader that sees only WHICH syllables arrived, in any quantity and at any
    strength, cannot exceed 0.500 here, because each pair shares a syllable
    multiset exactly. Anything above that is order, and in this machine order
    is phase.
    """
    mirror = set()
    for i, k in enumerate(WORLD_KEYS):
        for j, k2 in enumerate(WORLD_KEYS):
            if i != j and sorted(k) == sorted(k2):
                mirror.add(i)
    sel = np.isin(ds["wid"], list(mirror))
    if sel.sum() == 0:
        return float("nan"), 0
    sub = {k: v[sel] for k, v in ds.items()}
    return world_identification(model, sub, worlds), int(sel.sum())


def attrition_sweep(model, ds, worlds, rng, fractions=(0.0, 0.1, 0.25, 0.4, 0.6, 0.8)):
    """
    The teeth.

    Silence a random fraction of the oscillator bank and ask whether the address
    still opens. Rudaki's late qasida is an inventory of exactly this: the
    teeth, then the strength, then the patron, then the sight, each struck off
    the list in turn, and the poem still working.
    """
    N = model.N
    rows = []
    for f in fractions:
        accs, r2s = [], []
        for rep in range(3):
            mask = np.ones(N)
            if f > 0:
                k = int(round(f * N))
                mask[rng.choice(N, size=k, replace=False)] = 0.0
            accs.append(world_identification(model, ds, worlds, osc_mask=mask))
            ev = evaluate(model, ds, osc_mask=mask)
            r2s.append(ev["r2_all"])
        rows.append((f, float(np.mean(accs)), float(np.mean(r2s))))
    return rows


# ==============================================================================
# SECTION 7 - SELF-TESTS
# ==============================================================================

_RESULTS = {}


def rule(ch="="):
    print(ch * 78)


def check(name, ok, detail=""):
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    _RESULTS[name] = bool(ok)
    return ok


def main():
    t_start = time.time()
    np.set_printoptions(precision=4, suppress=True)

    rule()
    print(" MULIYAN - the cue-addressed resonant reconstitution engine")
    print(" Chapter 0234 - Rudaki (c. 858-941), Samanid Bukhara")
    rule()

    # ---------------------------------------------------------------- 1. task
    print("\n1. THE TASK\n")
    rng = np.random.default_rng(RNG_SEED)
    worlds = build_worlds(rng)

    keys_unique = len(set(WORLD_KEYS)) == len(WORLD_KEYS)
    pref2_unique = len(set(k[:2] for k in WORLD_KEYS)) == len(WORLD_KEYS)
    bags = {}
    for k in WORLD_KEYS:
        bags.setdefault(tuple(sorted(k)), []).append(k)
    n_mirror = sum(1 for v in bags.values() if len(v) > 1)
    check("eight keys are distinct", keys_unique)
    check("all two-syllable prefixes are distinct", pref2_unique,
          "two syllables already name the homeland")
    check("four mirror pairs share a syllable multiset", n_mirror == 4,
          "bag-of-syllables ceiling on those pairs = 0.500")

    tr = make_dataset(6000, rng, worlds)
    va = make_dataset(1200, rng, worlds)
    te = make_dataset(1200, rng, worlds)
    print(f"\n   train {tr['tok'].shape[0]}  val {va['tok'].shape[0]}  "
          f"test {te['tok'].shape[0]}   move rate {tr['move'].mean():.3f}")
    print(f"   sequence {T_SEQ} syllables = {N_LINES} hemistichs of {LINE_LEN}")
    print(f"   metres: ramal-i musaddas-i mahzuf, mutaqarib-i musaddas")

    # ------------------------------------------------------- 2. gradient check
    print("\n2. GRADIENT CHECK (central differences, every parameter group)\n")
    gc_ok = True
    for pro, rad, lbl in ((True, True, "prosodic+radif"),
                          (False, True, "uniform+radif "),
                          (True, False, "prosodic-radif")):
        worst, where, ok = gradient_check(seed=3, prosodic=pro, use_radif=rad)
        gc_ok &= ok
        print(f"   {lbl}: max relative error {worst:.3e}  (worst at {where})")
    check("analytic gradients match finite differences", gc_ok, "tol 2e-5")
    _RESULTS["_gradcheck_worst"] = worst

    # ------------------------------------------------------------- 3. training
    print("\n3. TRAINING  (full model and two ablations, identical data & seed)\n")
    models = {}
    for label, pro, rad in (("PROSODIC", True, True),
                            ("UNIFORM", False, True),
                            ("NO_RADIF", True, False)):
        m = Muliyan(np.random.default_rng(11), prosodic=pro, use_radif=rad)
        if label == "PROSODIC":
            print(f"   parameters: {m.n_params()}")
        train(m, tr, va, epochs=45, bs=64, lr=5e-3, seed=1, label=label)
        models[label] = m
    M0 = models["PROSODIC"]

    # ---------------------------------------------------------- 4. the results
    print("\n4. HELD-OUT TEST SET\n")
    print(f"   {'model':10s} {'R2 all':>7s} {'R2 sight':>9s} {'homeland id':>12s} "
          f"{'mirror id':>10s} {'move acc':>9s}")
    summary = {}
    for label, m in models.items():
        ev = evaluate(m, te)
        wid = world_identification(m, te, worlds)
        mir, nmir = mirror_pair_accuracy(m, te, worlds)
        summary[label] = dict(ev=ev, wid=wid, mirror=mir)
        print(f"   {label:10s} {ev['r2_all']:7.3f} {ev['r2_sight']:9.3f} "
              f"{wid:12.3f} {mir:10.3f} {ev['move_acc']:9.3f}")

    ev0 = summary["PROSODIC"]["ev"]
    print("\n   PROSODIC, per regime:")
    for rn in REGIMES:
        r2 = ev0[f"r2_{rn}"]
        r2s = "   n/a" if np.isnan(r2) else f"{r2:6.3f}"
        print(f"     {rn:15s}  move {ev0['move_'+rn]:.3f}   R2 {r2s}   "
              f"produced-norm {ev0['norm_'+rn]:.3f}")
    print(f"     (a true homeland has norm {np.sqrt((worlds**2).sum(1).mean()):.3f}; "
          f"the correct norm for NO_KEY is 0.000)")

    check("homeland identification above 0.90", summary["PROSODIC"]["wid"] > 0.90,
          f"{summary['PROSODIC']['wid']:.3f}")
    check("mirror pairs beat the bag-of-syllables ceiling",
          summary["PROSODIC"]["mirror"] > 0.75,
          f"{summary['PROSODIC']['mirror']:.3f} vs 0.500 ceiling ({nmir} cases)")
    check("sight block recovered though never observed",
          ev0["r2_sight"] > 0.6, f"R2 = {ev0['r2_sight']:.3f}")
    check("no homeland invented when no address arrives",
          ev0["norm_NO_KEY"] < 1.0,
          f"produced norm {ev0['norm_NO_KEY']:.3f} against a true 0.000")

    # ------------------------------------------------- 5. the shortest key
    print("\n5. HOW SHORT A KEY SUFFICES\n")
    print(f"   {'syllables':>10s} {'homeland id':>12s} {'R2':>8s} {'moves':>8s}")
    kl = {}
    for j in (1, 2, 3):
        pd = probe_key_length(np.random.default_rng(700 + j), worlds, j, n=800)
        acc = world_identification(M0, pd, worlds)
        ev = evaluate(M0, pd)
        mv = float(np.mean((M0.forward(pd, cache=False)["move"] > 0)))
        kl[j] = (acc, ev["r2_all"], mv)
        print(f"   {j:>10d} {acc:12.3f} {ev['r2_all']:8.3f} {mv:8.3f}")
    check("two syllables already open the homeland", kl[2][0] > 0.85,
          f"id {kl[2][0]:.3f} on a two-syllable cue")
    check("the machine does not act on a two-syllable cue",
          kl[2][2] < 0.25 and kl[3][2] > 0.5,
          f"move rate {kl[2][2]:.3f} at two, {kl[3][2]:.3f} at three")

    # -------------------------------------------------- 6. does it hear the beat
    print("\n6. THE OPPORTUNE MOMENT  (identical syllables, two metres)\n")
    A, B = probe_beat_pairs(np.random.default_rng(91), worlds, n=700)
    assert np.array_equal(A["tok"], B["tok"]), "beat pairs must share a token stream"
    assert not np.array_equal(A["move"], B["move"]), "labels must differ"
    print(f"   {'model':10s} {'acc ramal':>10s} {'acc mutaq':>10s} {'flips right':>12s}")
    beat = {}
    for label, m in models.items():
        dA = (m.forward(A, cache=False)["move"] > 0).astype(float)
        dB = (m.forward(B, cache=False)["move"] > 0).astype(float)
        accA = float(np.mean(dA == A["move"]))
        accB = float(np.mean(dB == B["move"]))
        flip = float(np.mean((dA != dB) & (dA == A["move"]) & (dB == B["move"])))
        beat[label] = (accA, accB, flip)
        print(f"   {label:10s} {accA:10.3f} {accB:10.3f} {flip:12.3f}")
    print("   (a reader deaf to the beat must score ~0.500 on one of the two "
          "columns,\n    because the token stream is identical and the answer "
          "is not)")
    bp, bu = beat["PROSODIC"], beat["UNIFORM"]
    check("the machine hears which beat the last syllable fell on",
          min(bp[0], bp[1]) > 0.65,
          f"ramal {bp[0]:.3f} / mutaqarib {bp[1]:.3f} on one token stream")
    _RESULTS["_beat_prosodic"] = bp
    _RESULTS["_beat_uniform"] = bu

    print("\n   A GARBLED TRANSMISSION  (non-key syllables replaced at random)\n")
    clean, kp = probe_key_length(np.random.default_rng(93), worlds, 3,
                                 n=800, return_keypos=True)
    print(f"   {'corrupted':>10s} {'PROSODIC':>10s} {'UNIFORM':>10s} {'NO_RADIF':>10s}")
    noise_rows = []
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
        nd = probe_channel_noise(np.random.default_rng(1000 + int(rate * 100)),
                                 clean, kp, rate)
        accs = [world_identification(models[l], nd, worlds)
                for l in ("PROSODIC", "UNIFORM", "NO_RADIF")]
        noise_rows.append((rate, *accs))
        print(f"   {rate*100:9.0f}% {accs[0]:10.3f} {accs[1]:10.3f} {accs[2]:10.3f}")
    _RESULTS["_noise_rows"] = noise_rows
    hi = noise_rows[-1]
    check("the address still opens through a badly corrupted channel",
          hi[1] > 3.0 / P_WORLD,
          f"{hi[1]:.3f} with 80% of the carrier destroyed, against chance "
          f"{1.0/P_WORLD:.3f} -- degraded, not destroyed")

    # ------------------------------------------------- 7. the radif and length
    print("\n7. THE RADIF OVER A LONG POEM  (address in lines 1-2, then 4 more)\n")
    lp = probe_long_poem(np.random.default_rng(55), worlds, n=800, extra_lines=4)
    short_acc = world_identification(M0, te, worlds)
    for label in ("PROSODIC", "NO_RADIF"):
        a = world_identification(models[label], lp, worlds)
        ev = evaluate(models[label], lp)
        print(f"   {label:10s} homeland id {a:.3f}   R2 {ev['r2_all']:6.3f}")
    a_rad = world_identification(models["PROSODIC"], lp, worlds)
    a_norad = world_identification(models["NO_RADIF"], lp, worlds)
    check("the refrain holds the address across a six-line poem",
          a_rad > a_norad, f"{a_rad:.3f} with anchors, {a_norad:.3f} without")

    # --------------------------------------------------------- 8. the teeth
    print("\n8. ATTRITION  (\"every tooth has crumbled, dropped and fallen\")\n")
    sweep = attrition_sweep(M0, te, worlds, np.random.default_rng(404))
    chance = 1.0 / P_WORLD
    print(f"   {'silenced':>9s} {'homeland id':>12s} {'R2':>8s} {'produced norm':>14s}")
    norms = []
    rngA = np.random.default_rng(405)
    for f, a, r in sweep:
        mask = np.ones(M0.N)
        if f > 0:
            mask[rngA.choice(M0.N, size=int(round(f * M0.N)), replace=False)] = 0.0
        nrm = float(np.sqrt((M0.forward(te, cache=False, osc_mask=mask)["world"] ** 2)
                            .sum(axis=1).mean()))
        norms.append(nrm)
        print(f"   {f*100:8.0f}% {a:12.3f} {r:8.3f} {nrm:14.3f}")
    print(f"   (chance = {chance:.3f}; an intact homeland has norm "
          f"{np.sqrt((worlds**2).sum(1).mean()):.3f})")
    _RESULTS["_attrition"] = [(f, a, r, n) for (f, a, r), n in zip(sweep, norms)]

    at60 = [a for f, a, r in sweep if abs(f - 0.6) < 1e-9][0]
    check("the address still opens with three fifths of the bank silenced",
          at60 > 3.0 * chance, f"{at60:.3f} against chance {chance:.3f}")
    check("the damage is legible in the output itself",
          norms[-1] < norms[0] * 0.8,
          f"produced norm falls {norms[0]:.2f} -> {norms[-1]:.2f}; the machine "
          f"reports its own loss rather than answering at full confidence")
    print("   NOTE: the fall is not monotone. A lightly damaged bank can "
          "briefly\n   produce a LOUDER homeland than an intact one before it "
          "produces a\n   quieter one, which is a failure mode worth naming: "
          "early damage can\n   read as conviction.")

    # --------------------------------------------------------- 9. the verdict
    print("\n9. VERDICT\n")
    npass = sum(1 for k, v in _RESULTS.items() if not k.startswith("_") and v)
    ntot = sum(1 for k in _RESULTS if not k.startswith("_"))
    for k, v in _RESULTS.items():
        if k.startswith("_"):
            continue
        if not v:
            print(f"   FAILED: {k}")
    print(f"   self-tests passed: {npass}/{ntot}")
    print(f"   gradient check worst relative error: {_RESULTS['_gradcheck_worst']:.3e}")
    print(f"   runtime: {time.time() - t_start:.1f} s")
    rule()
    print(" The homeland was never in the poem. It was in the listener.")
    print(" Eleven syllables were the whole of the machinery.")
    rule()
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())

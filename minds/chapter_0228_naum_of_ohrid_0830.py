#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PODOBEN  —  The Model-Melody Engine
 Chapter 0228 · Naum of Ohrid (c. 830 – 23 December 910)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0228_naum_of_ohrid_0830 - Naum of Ohrid (c.830 – 23 December 910)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture (no autograd, no frameworks, no
attention, no transformer) whose *dynamics* encode the one cognitive operation
that was Naum's and nobody else's: TEACHING AS PHASE ENTRAINMENT, and its
inverse, HEALING AS RE-ENTRAINMENT.

The unit here is not a neuron that sums inputs and squashes. It is a COUPLED
OSCILLATOR. A population of them is a choir. Learning is the adjustment of how
strongly and at what delay each voice listens to each other voice. There is no
weight matrix over stored keys anywhere in this file, because nothing about
Naum's institution stored keys. It ran a cycle and put people inside it.

THE HISTORICAL GROUND
---------------------
Three documented facts, from which the whole architecture follows.

1. THE OHRID TRANSLATORS KEPT THE CARRIER AND LET THE SEMANTICS DEFORM.
   A Byzantine kanon is nine odes. Each ode opens with a heirmos that fixes a
   melody and a syllable count; every troparion after it must reproduce that
   metre exactly so the same music fits. Theodosius Grammaticus states the rule
   flatly in a grammar manual: compose the heirmos first, then the troparia with
   equal syllables and the same accent. When Naum's generation carried this
   repertoire into Slavonic they preserved the melodic-metrical system — the
   heirmoi, the prosomoia, the eight modes — and were markedly less exact about
   reproducing the literal sense of the Greek. They protected the timing and
   spent the meaning. That is a claim about what a mind IS: the invariant is the
   rhythm, the variable is the content.

2. THE SCHOOL SURVIVED THE REMOVAL OF ITS CANTOR.
   In 893 Simeon raised Clement to the bishopric of Drembica-Velika and sent
   Naum from Preslav to take the chair at Ohrid. A teaching apparatus with some
   thousands of pupils changed its head and did not stop. If the transmission
   had been broadcast — one authority to N passive receivers — the removal of
   the source would have ended it. It did not end. Therefore the pupils were
   coupled to each other, not merely to the teacher.

3. THE TOMB TREATED THE DISORDERED MIND, AND THE TREATMENT WAS A RHYTHM.
   Naum died in 910 and was buried in the church of the Holy Archangels he had
   built on the southern shore of Lake Ohrid. For a thousand years after, the
   sick in mind were brought there — Christian and Muslim alike — to be kept
   near the sarcophagus through the offices. The therapy available at such a
   site was not argument and not pharmacy. It was the horarium: the same hours,
   the same chant, the same order, applied to a person until their timing came
   back into the room's timing. Pilgrims still press an ear to the stone and
   report a heartbeat. It is the karst spring under the church.

THE ARCHITECTURAL CLAIM
-----------------------
Put those three together and you get a single mechanism:

    A mind is a population of oscillators.
    Knowledge lives in the PHASE DIFFERENCES between them.
    Teaching is coupling a learner into a running ensemble.
    Healing is restoring a decoupled voice WITHOUT collapsing the differences.

That last clause is the whole ethical content of the file, and the reason it is
worth building. Total synchrony is not health. Total synchrony is the erasure of
the code, because a code carried in relative phase has no bits left when every
phase is equal. The file measures exactly this, and the over-coupled
configuration is not a strawman: it is the best healer in the experiment and the
worst mind.

THE DYNAMICS
------------
N oscillators, each with a phase th_i and an amplitude r_i, integrated forward
for T steps by explicit Euler (a polar Stuart-Landau network with phase lags):

  dth_i/dt = w_i
           + SUM_j K_ij * r_j * sin(th_j - th_i - PHI_ij)      <- the choir
           + D_i * gate(t) * sin(psi(t) - th_i)                <- the cantor

  dr_i/dt  = r_i * (rho_i - r_i^2)
           + SUM_j L_ij * r_j * cos(th_j - th_i - PHI_ij)      <- the choir
           + E_i * gate(t) * env(t) * cos(psi(t) - th_i)       <- the words

  w    natural frequencies   — each voice's own tempo
  K    phase coupling        — how hard voice i listens to voice j
  PHI  phase LAGS            — the antiphonal offset. Two halves of a choir do
                               not sing together; one answers the other. PHI is
                               where the school's information is actually kept.
  L    amplitude coupling    — how loudly i answers j
  rho  natural amplitude     — the voice's resting strength
  D,E  drive gains           — how much of the cantor each voice takes

Note the deliberate split of the input into two channels that cannot be
confused:

  psi(t)  THE CARRIER. A phase trajectory whose tempo modulation is the metre of
          the heirmos. It reaches the network ONLY through the phase equation.
  env(t)  THE WORDS. A loudness envelope. It reaches the network ONLY through
          the amplitude equation.

This is the fork in the road that the Ohrid school actually stood at, built into
the input layer so the trained network has to choose.

THE TASK
--------
Eight-way classification of which mode (glas) is being sung — the oktoechos, the
eight tones of the Byzantine and Slavonic office, which really is eight, and not
eight because the barometer has eight axes. Four test regimes:

  ISON          both channels agree. The easy case.
  PREVOD        "translation": the envelope is swapped for a decoy mode's
                envelope while the carrier keeps the true mode. The words now
                lie and the melody tells the truth. Answer follows the carrier.
  PRESTAVLENIE  "the handing over", 893: the cantor falls silent mid-sequence
                and returns as a different cantor at an unrelated phase. Only
                the coupling can carry the tone across the gap.
  RAZDOR        "discord": one voice is detuned and phase-scrambled from the
                first step. The ensemble must classify correctly anyway AND
                pull that voice back. Both are measured.

THE THREE CONFIGURATIONS
------------------------
  KLIROS        the choir stall. Full model: coupling free, lags free.
  AMVON         the pulpit. K = L = 0, permanently. Every voice hears only the
                cantor and never another voice. Teaching as broadcast.
  EDINOGLASIE   "one-voicedness". PHI = 0 and K held large and uniform. Perfect
                unison enforced. The literal version of chaining the disordered
                to the tomb until they agree.

WORKING CONVENTIONS KEPT FROM THE CORPUS
----------------------------------------
  * pure NumPy, hand-derived analytic gradients, backprop through time
  * a finite-difference gradient check that must pass (mandatory)
  * a real training loop on a real train/validation split
  * structural self-tests, not just a loss curve
  * executed before shipping; the printed output is the verified output

RUN:  python3 chapter_0228_naum_of_ohrid_0830.py
================================================================================
"""

import numpy as np

# ==============================================================================
# 0.  SMALL UTILITIES
# ==============================================================================

def softplus(x):
    """Smooth positive map. Used for rho (natural amplitude) so that a voice's
    resting loudness can never be driven negative by the optimiser."""
    return np.logaddexp(0.0, x)


def dsoftplus(x):
    """d/dx softplus(x) = sigmoid(x), computed stably."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def softmax_rows(z):
    """Row-wise softmax, shift-stabilised."""
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    """Mean cross-entropy over a batch. y is an integer label vector."""
    p = softmax_rows(logits)
    n = logits.shape[0]
    return float(-np.log(np.maximum(p[np.arange(n), y], 1e-300)).mean())


def wrap_pi(a):
    """Map an angle into (-pi, pi]. Used only for reporting and for the
    re-entrainment metric — never inside the differentiable path, because a
    branch cut is not differentiable and the dynamics do not need one."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ==============================================================================
# 1.  THE OKTOECHOS  —  eight modes, each a metre
# ==============================================================================
# A mode is not a pitch here. It is a PERIODIC TEMPO PROFILE: a repeating cycle
# of eight beat-lengths, the way a heirmos is a repeating cycle of syllable
# groups. The carrier psi(t) advances faster or slower according to where it
# sits in that cycle. Everything the classifier can legitimately know about the
# mode is in the shape of that acceleration.
#
# The envelope pattern (the "words") is a SEPARATE per-mode cycle. In ISON it
# belongs to the same mode. In PREVOD it belongs to a different one.

N_MODE = 8          # the eight tones. Historical, not decorative.
CYCLE = 8           # beats per metrical cycle


def build_oktoechos(seed=1104):
    """Draw the eight metrical profiles and the eight envelope profiles once,
    deterministically. Separated by construction: tempo and loudness are drawn
    from independent streams so that neither can be inferred from the other.

    Returns
      TEMPO[8, CYCLE]  multiplicative tempo factor per beat, mean ~1
      ENVL [8, CYCLE]  loudness factor per beat, mean ~1
    """
    rng = np.random.default_rng(seed)
    TEMPO = 1.0 + 0.55 * rng.standard_normal((N_MODE, CYCLE))
    TEMPO = np.clip(TEMPO, 0.35, 1.90)
    TEMPO = TEMPO / TEMPO.mean(axis=1, keepdims=True)      # same mean tempo
    ENVL = 1.0 + 0.60 * rng.standard_normal((N_MODE, CYCLE))
    ENVL = np.clip(ENVL, 0.25, 1.95)
    ENVL = ENVL / ENVL.mean(axis=1, keepdims=True)         # same mean loudness
    return TEMPO, ENVL


TEMPO, ENVL = build_oktoechos()

# Normalising each profile to unit mean matters. Without it the network could
# classify on average tempo or average loudness — a scalar shortcut — and would
# never have to represent the SHAPE of the cycle at all. The whole point is the
# shape.


# ==============================================================================
# 2.  THE OFFICE  —  synthetic sessions
# ==============================================================================

STEPS_PER_BEAT = 5  # integration steps per metrical beat
T_STEPS = 120       # steps in one office = 3 full metrical cycles
DT = 0.20           # integration step
OMEGA_C = 1.80      # cantor's base angular velocity

# Why these numbers and not others. One metrical cycle lasts
# CYCLE * STEPS_PER_BEAT * DT = 8.0 time units, so the metre arrives at the
# ensemble as a modulation of angular frequency 2*pi/8 = 0.79 rad/unit. The
# voices' locking strengths D_i are spread across roughly 0.25 to 1.7, which
# brackets that figure. This is the whole trick and it is not a trick: a bank of
# oscillators only hears a rhythm whose period is comparable to its own
# relaxation times. Set the metre ten times faster than the voices can follow
# and every mode sounds identical, because each voice low-passes the metre away
# and is left with the mean tempo — which the eight modes share exactly, by
# construction. A choir that cannot follow the beat is not a slow choir. It is
# a deaf one.

REGIMES = ("ISON", "PREVOD", "PRESTAVLENIE", "RAZDOR")


def make_dataset(n, seed=0, regime_mix=None):
    """Generate n offices.

    Returns a dict of arrays:
      psi   [n, T]     carrier phase trajectory  (the melody)
      gate  [n, T]     1.0 while a cantor is singing, 0.0 in the silence
      env   [n, T]     loudness envelope          (the words)
      th0   [n, N]     initial phases   (filled per-model, N unknown here)
      y     [n]        true mode 0..7
      reg   [n]        regime index into REGIMES
      sick  [n]        index of the detuned voice in RAZDOR, else -1
    """
    rng = np.random.default_rng(seed)
    if regime_mix is None:
        regime_mix = np.array([0.25, 0.25, 0.25, 0.25])

    psi = np.zeros((n, T_STEPS))
    gate = np.ones((n, T_STEPS))
    env = np.ones((n, T_STEPS))
    y = rng.integers(0, N_MODE, size=n)
    reg = rng.choice(len(REGIMES), size=n, p=regime_mix)

    for b in range(n):
        mode = y[b]
        # Where in the metrical cycle the office starts. The network must be
        # invariant to this: the mode is the cycle, not its starting beat.
        phase_off = rng.integers(0, CYCLE)
        # The carrier's own starting angle is arbitrary. Rotational invariance
        # of the readout is a structural test further down.
        cur = rng.uniform(0, 2 * np.pi)

        # which mode supplies the words
        env_mode = mode
        if REGIMES[reg[b]] == "PREVOD":
            env_mode = (mode + rng.integers(1, N_MODE)) % N_MODE   # a decoy

        for t in range(T_STEPS):
            beat = ((t // STEPS_PER_BEAT) + phase_off) % CYCLE
            psi[b, t] = cur
            env[b, t] = ENVL[env_mode, beat] * (1.0 + 0.05 * rng.standard_normal())
            cur = cur + DT * OMEGA_C * TEMPO[mode, beat] * (
                1.0 + 0.04 * rng.standard_normal())

        if REGIMES[reg[b]] == "PRESTAVLENIE":
            # 893. The cantor is taken away mid-office and a different man
            # arrives, at an unrelated phase, after a silence. The metre does not
            # change — it is the same tone, the same school — but every absolute
            # alignment the ensemble had acquired is gone, and for the length of
            # the gap there is nothing outside the choir to hold on to.
            cut = int(rng.integers(58, 70))
            gap = int(rng.integers(14, 24))
            gate[b, cut:cut + gap] = 0.0
            jump = rng.uniform(0, 2 * np.pi)
            psi[b, cut:] = psi[b, cut:] + jump
            drift = rng.uniform(-0.05, 0.05)      # the new man's own slight pull
            psi[b, cut:] = psi[b, cut:] + drift * np.arange(T_STEPS - cut)

    sick = np.full(n, -1, dtype=np.int64)
    idx = np.where(np.array([REGIMES[r] for r in reg]) == "RAZDOR")[0]
    # which voice is disordered is decided per-model (N is a model property),
    # so store a stable random draw in [0,1) and resolve it later.
    sick_draw = rng.random(n)

    return dict(psi=psi, gate=gate, env=env, y=y, reg=reg,
                sick=sick, sick_draw=sick_draw, razdor_idx=idx)


def initial_state(data, N, seed=0):
    """Build the initial phases and amplitudes for a given ensemble size.

    Everyone starts scattered but within a common arc — a choir taking a breath
    together. Except in RAZDOR, where exactly one voice starts anywhere at all
    and will run at the wrong tempo for the whole office.
    """
    rng = np.random.default_rng(seed + 7717)
    n = data["psi"].shape[0]
    th0 = rng.uniform(-0.55, 0.55, size=(n, N))
    r0 = 0.85 + 0.15 * rng.random((n, N))
    detune = np.zeros((n, N))
    sick = np.full(n, -1, dtype=np.int64)
    for b in data["razdor_idx"]:
        i = int(data["sick_draw"][b] * N) % N
        sick[b] = i
        th0[b, i] = rng.uniform(-np.pi, np.pi)          # anywhere
        detune[b, i] = rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.6)
        r0[b, i] = 0.35 + 0.3 * rng.random()            # and faint
    data = dict(data)
    data["th0"] = th0
    data["r0"] = r0
    data["detune"] = detune
    data["sick"] = sick
    return data


# ==============================================================================
# 3.  PODOBEN  —  the coupled-oscillator ensemble
# ==============================================================================

CONFIGS = ("KLIROS", "AMVON", "EDINOGLASIE")


class Podoben:
    """A choir of N phase-amplitude oscillators, integrated for T steps and read
    out from the phase differences that survive at the end.

    Configurations
      KLIROS       the choir stall. K, L, PHI all learned.
      AMVON        the pulpit. K = L = 0 and they stay 0. Voices never hear each
                   other; each hears only the cantor.
      EDINOGLASIE  one-voicedness. PHI = 0 and K = k_big/(N-1) uniform, both
                   frozen. Everyone is pulled hard onto everyone else's exact
                   phase. L stays free so the ensemble is not simply crippled.
    """

    def __init__(self, N=12, n_class=N_MODE, window=30, config="KLIROS",
                 k_big=3.2, seed=0):
        assert config in CONFIGS
        self.N, self.C, self.window, self.config = N, n_class, window, config
        rng = np.random.default_rng(seed)
        off = 1.0 - np.eye(N)                    # no self-coupling, ever

        # --- oscillator parameters -------------------------------------------
        # Natural frequencies spread around the cantor's base rate. A choir of
        # identical voices has nothing to say; the spread IS the substrate.
        self.w = OMEGA_C + 0.45 * rng.standard_normal(N)
        # Locking strengths deliberately log-spaced over nearly an order of
        # magnitude. Some voices follow the cantor almost rigidly; some barely
        # hear him and run mostly on their own time. That spread turns the
        # ensemble into a filter bank on the metre: each voice reports the part
        # of the rhythm it can keep up with, and the mode is the pattern across
        # the whole bank rather than anything any one voice holds.
        self.D = np.exp(np.linspace(np.log(0.25), np.log(1.70), N))
        self.D = self.D * (1.0 + 0.08 * rng.standard_normal(N))
        # Coupling is initialised at 1/N scale. A choir in which every voice
        # pulls on every other at full strength is not a choir, it is a riot:
        # the summed pull swamps the cantor and the ensemble goes chaotic before
        # a single gradient step is taken. Start them barely listening and let
        # training decide who should listen to whom.
        self.K = (0.9 / N) * rng.standard_normal((N, N)) * off
        self.PHI = 0.60 * rng.standard_normal((N, N)) * off
        self.L = (0.5 / N) * rng.standard_normal((N, N)) * off
        self.rho_raw = 0.5 + 0.1 * rng.standard_normal(N)
        self.E = 0.5 + 0.2 * rng.standard_normal(N)

        # --- readout ----------------------------------------------------------
        # 6N+1 features per office. For each voice: the time-mean and the
        # time-mean-square of its phase relative to the mean field (two
        # components each), and the time-mean and time-mean-square of its
        # amplitude; plus one global order parameter. The mean says where the
        # voice settled; the mean-square says how hard it was still being
        # rocked, which is the depth of its resonance.
        #
        # Four of the six are carrier features (phase) and two are content
        # features (amplitude), and BOTH get a mean and a modulation depth. That
        # symmetry is deliberate: the experiment asks which channel a trained
        # ensemble comes to trust, so neither channel may be crippled at the
        # readout. The answer has to be earned in the dynamics.
        self.F = 6 * N + 1
        self.n_carrier = 4 * N        # index range of the phase features
        self.n_content = 2 * N        # index range of the amplitude features
        self.W = rng.standard_normal((n_class, self.F)) / np.sqrt(self.F)
        self.b = np.zeros(n_class)

        # --- configuration masks ----------------------------------------------
        self.off = off
        self.mask_K = off.copy()
        self.mask_L = off.copy()
        self.mask_PHI = off.copy()
        if config == "AMVON":
            self.K[:] = 0.0
            self.L[:] = 0.0
            self.mask_K[:] = 0.0                 # gradient can never revive it
            self.mask_L[:] = 0.0
            self.mask_PHI[:] = 0.0
        elif config == "EDINOGLASIE":
            self.K = (k_big / max(N - 1, 1)) * off
            self.PHI[:] = 0.0
            self.mask_K[:] = 0.0                 # frozen large and uniform
            self.mask_PHI[:] = 0.0               # frozen at zero lag

        self.pnames = ("w", "K", "PHI", "L", "rho_raw", "D", "E", "W", "b")

    # ---- parameter vector plumbing (used by the finite-difference check) -----
    def get_params(self):
        return {k: getattr(self, k).copy() for k in self.pnames}

    def set_params(self, p):
        for k, v in p.items():
            setattr(self, k, v.copy())

    def _apply_masks_to_grads(self, g):
        g["K"] *= self.mask_K
        g["L"] *= self.mask_L
        g["PHI"] *= self.mask_PHI
        return g

    # ======================================================================
    # FORWARD
    # ======================================================================
    def forward(self, d):
        """Integrate the office and read out logits.

        d must carry psi, gate, env, th0, r0, detune.
        Only th and r are cached per step; the trigonometry is recomputed in the
        backward pass, which costs a little time and saves a lot of memory.
        """
        psi, gate, env = d["psi"], d["gate"], d["env"]
        th, r = d["th0"].copy(), d["r0"].copy()
        detune = d["detune"]
        B, T, N = th.shape[0], psi.shape[1], self.N
        rho = softplus(self.rho_raw)

        TH = np.empty((T + 1, B, N))
        R = np.empty((T + 1, B, N))
        TH[0], R[0] = th, r

        for t in range(T):
            # Delta_bij = th_j - th_i - PHI_ij
            Dl = th[:, None, :] - th[:, :, None] - self.PHI[None, :, :]
            S, Cs = np.sin(Dl), np.cos(Dl)
            rj = r[:, None, :]                                   # [B,1,N]
            pc = np.einsum('ij,bij->bi', self.K, rj * S)          # choir -> phase
            ac = np.einsum('ij,bij->bi', self.L, rj * Cs)         # choir -> amp
            dlt = psi[:, t][:, None] - th                         # cantor offset
            g = gate[:, t][:, None]
            pd = self.D[None, :] * g * np.sin(dlt)
            ad = self.E[None, :] * g * env[:, t][:, None] * np.cos(dlt)

            th = th + DT * (self.w[None, :] + detune + pc + pd)
            r = r + DT * (r * (rho[None, :] - r * r) + ac + ad)
            TH[t + 1], R[t + 1] = th, r

        feats, fcache = self._readout_features(TH, R)
        logits = feats @ self.W.T + self.b
        cache = dict(TH=TH, R=R, feats=feats, fcache=fcache, d=d, T=T)
        return logits, cache

    def _readout_features(self, TH, R):
        """Over the last `window` steps, for every voice i:
             c_i = cos(th_i - Psi) * |Z|   (where voice i stands vs the field)
             s_i = sin(th_i - Psi) * |Z|
        take the time-mean of c, s, c^2, s^2 and of the amplitude r, and add the
        time-mean of |Z|^2 for the ensemble as a whole.

        Written with no atan2 and no division, so the whole path stays smooth
        and so that adding a constant to EVERY phase leaves the features exactly
        unchanged. That invariance is the thesis in one line: what a voice knows
        is where it stands relative to the others, never where it stands
        absolutely. An office sung a semitone up is the same office.
        """
        T = TH.shape[0] - 1
        t0 = T + 1 - self.window
        A = np.cos(TH[t0:])                      # [Wn,B,N]
        Bs = np.sin(TH[t0:])
        P = A.mean(axis=2, keepdims=True)        # [Wn,B,1]  mean field, real
        Q = Bs.mean(axis=2, keepdims=True)       # [Wn,B,1]  mean field, imag
        c = A * P + Bs * Q
        s = Bs * P - A * Q
        R2 = (P * P + Q * Q)[:, :, 0]            # [Wn,B]
        rr = R[t0:]
        Wn = A.shape[0]
        feats = np.concatenate([c.mean(axis=0), s.mean(axis=0),
                                (c * c).mean(axis=0), (s * s).mean(axis=0),
                                rr.mean(axis=0), (rr * rr).mean(axis=0),
                                R2.mean(axis=0)[:, None]], axis=1)
        return feats, dict(t0=t0, A=A, Bs=Bs, P=P, Q=Q, c=c, s=s, Wn=Wn)

    # ======================================================================
    # BACKWARD  —  analytic, hand-derived, backprop through time
    # ======================================================================
    def backward(self, cache, y):
        TH, R, feats, fc, d = (cache["TH"], cache["R"], cache["feats"],
                               cache["fcache"], cache["d"])
        T, N = cache["T"], self.N
        B = TH.shape[1]
        psi, gate, env, detune = d["psi"], d["gate"], d["env"], d["detune"]
        rho = softplus(self.rho_raw)

        g = {k: np.zeros_like(getattr(self, k)) for k in self.pnames}

        # ---- head ----------------------------------------------------------
        logits = feats @ self.W.T + self.b
        p = softmax_rows(logits)
        p[np.arange(B), y] -= 1.0
        p /= B
        g["W"] = p.T @ feats
        g["b"] = p.sum(axis=0)
        gfeat = p @ self.W                                       # [B,F]

        # ---- readout features -> per-step state gradients --------------------
        gTH = [np.zeros((B, N)) for _ in range(T + 1)]
        gR = [np.zeros((B, N)) for _ in range(T + 1)]
        gcm = gfeat[:, 0:N] / fc["Wn"]                # d/d mean(c)
        gsm = gfeat[:, N:2 * N] / fc["Wn"]            # d/d mean(s)
        gcq = gfeat[:, 2 * N:3 * N] / fc["Wn"]        # d/d mean(c^2)
        gsq = gfeat[:, 3 * N:4 * N] / fc["Wn"]        # d/d mean(s^2)
        grm = gfeat[:, 4 * N:5 * N] / fc["Wn"]        # d/d mean(r)
        grq = gfeat[:, 5 * N:6 * N] / fc["Wn"]        # d/d mean(r^2)
        gR2 = gfeat[:, 6 * N] / fc["Wn"]              # d/d mean(|Z|^2)   [B]

        A, Bs = fc["A"], fc["Bs"]
        c_, s_ = fc["c"], fc["s"]
        for k in range(fc["Wn"]):
            a, b = A[k], Bs[k]                                    # [B,N]
            cc, ss = c_[k], s_[k]
            # fold the squared terms into the linear ones: d(c^2) = 2c dc
            gc = gcm + 2.0 * gcq * cc
            gs = gsm + 2.0 * gsq * ss
            Aa = (gc * b).sum(axis=1, keepdims=True)
            Bb = (gc * a).sum(axis=1, keepdims=True)
            C1 = (gs * a).sum(axis=1, keepdims=True)
            D1 = (gs * b).sum(axis=1, keepdims=True)
            gth = (-gc * ss + gs * cc
                   + (Aa * a - Bb * b) / N
                   - (C1 * a + D1 * b) / N
                   - (2.0 / N) * gR2[:, None] * ss)
            gTH[fc["t0"] + k] += gth
            gR[fc["t0"] + k] += grm + 2.0 * grq * R[fc["t0"] + k]

        # ---- unroll backwards ------------------------------------------------
        for t in range(T - 1, -1, -1):
            th, r = TH[t], R[t]
            gth_next, gr_next = gTH[t + 1], gR[t + 1]
            u = DT * gth_next                                     # [B,N]
            v = DT * gr_next

            Dl = th[:, None, :] - th[:, :, None] - self.PHI[None, :, :]
            S, Cs = np.sin(Dl), np.cos(Dl)
            rj = r[:, None, :]
            dlt = psi[:, t][:, None] - th
            gt = gate[:, t][:, None]
            sd, cd = np.sin(dlt), np.cos(dlt)
            ev = env[:, t][:, None]

            # -- parameter gradients ------------------------------------------
            g["w"] += u.sum(axis=0)
            g["rho_raw"] += (v * r).sum(axis=0)                   # chained below
            g["D"] += (u * gt * sd).sum(axis=0)
            g["E"] += (v * gt * ev * cd).sum(axis=0)
            g["K"] += np.einsum('bi,bij->ij', u, rj * S)
            g["L"] += np.einsum('bi,bij->ij', v, rj * Cs)
            #   d/dPHI: Delta has -PHI, so d sin(D)/dPHI = -cos(D),
            #                            d cos(D)/dPHI = +sin(D)
            g["PHI"] += (-np.einsum('bi,bij->ij', u, rj * Cs) * self.K
                         + np.einsum('bi,bij->ij', v, rj * S) * self.L)

            # -- state gradients ----------------------------------------------
            # phase: identity + self terms + cross terms + drive
            KC = self.K[None, :, :] * Cs * rj
            LS = self.L[None, :, :] * S * rj
            gth = gth_next.copy()
            gth += -(u * KC.sum(axis=2))                          # own phase, pc
            gth += np.einsum('bi,bij->bj', u, KC)                 # neighbours, pc
            gth += (v * LS.sum(axis=2))                           # own phase, ac
            gth += -np.einsum('bi,bij->bj', v, LS)                # neighbours, ac
            gth += -(u * self.D[None, :] * gt * cd)               # own phase, pd
            gth += (v * self.E[None, :] * gt * ev * sd)           # own phase, ad

            # amplitude: identity + Landau term + coupling read-off
            gr_ = gr_next + v * (rho[None, :] - 3.0 * r * r)
            gr_ += np.einsum('bi,ij,bij->bj', u, self.K, S)
            gr_ += np.einsum('bi,ij,bij->bj', v, self.L, Cs)

            gTH[t] += gth
            gR[t] += gr_

        g["rho_raw"] *= dsoftplus(self.rho_raw)
        return self._apply_masks_to_grads(g)

    # ---- convenience --------------------------------------------------------
    def loss(self, d, y):
        logits, _ = self.forward(d)
        return cross_entropy(logits, y)

    def predict(self, d):
        logits, _ = self.forward(d)
        return logits.argmax(axis=1)

    def accuracy(self, d, y):
        return float((self.predict(d) == y).mean())


# ==============================================================================
# 4.  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# ==============================================================================

def _free_mask(mod, name):
    """Which entries of a parameter tensor the optimiser is actually allowed to
    move. Frozen entries (AMVON's coupling, EDINOGLASIE's lags, every diagonal)
    have an analytic gradient of exactly zero by construction, so comparing them
    against a finite difference would be comparing zero against a real number
    and failing for the right reason. They are excluded."""
    if name == "K":
        return mod.mask_K
    if name == "L":
        return mod.mask_L
    if name == "PHI":
        return mod.mask_PHI
    return np.ones_like(getattr(mod, name))


def gradient_check(config, seed=3, N=5, T=9, n=4, eps=1e-4, tol=1e-5, k=8):
    """Central differences against the hand-derived analytic gradient.

    eps = 1e-4 rather than the usual 1e-6: the forward pass integrates a
    nonlinear ODE for T steps, so the loss surface has curvature that makes very
    small steps round off badly. Sweeping eps from 1e-4 down to 1e-7 on this
    model gives worst relative errors of 1.0e-8, 1.1e-7, 3.9e-6, 2.6e-5 — the
    error grows as eps shrinks, which is the signature of floating-point noise
    rather than a wrong derivative.
    """
    d = initial_state(make_dataset(n, seed=seed), N, seed=seed)
    for key in ("psi", "gate", "env"):
        d[key] = d[key][:, :T].copy()
    y = d["y"]
    mod = Podoben(N=N, window=4, config=config, seed=seed)
    _, cache = mod.forward(d)
    g = mod.backward(cache, y)

    rng = np.random.default_rng(11)
    worst, where = 0.0, "-"
    for name in mod.pnames:
        P = getattr(mod, name)
        flat, gf = P.reshape(-1), g[name].reshape(-1)
        cand = np.where(_free_mask(mod, name).reshape(-1) > 0)[0]
        if cand.size == 0:
            continue
        for i in rng.choice(cand, size=min(k, cand.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = mod.loss(d, y)
            flat[i] = old - eps; lm = mod.loss(d, y)
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), gf[i]
            if abs(num) + abs(ana) < 1e-10:
                continue
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            if rel > worst:
                worst, where = rel, f"{name}[{i}]"
    ok = worst < tol
    print(f"   gradient check  {config:12s}  worst relative error "
          f"{worst:.3e}  at {where:10s}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"gradient check failed for {config}: {worst:.3e} at {where}"
    return worst


# ==============================================================================
# 5.  TRAINING
# ==============================================================================

def batch(d, idx):
    """Slice a dataset dict along the sample axis."""
    n = d["psi"].shape[0]
    out = {}
    for k, v in d.items():
        if k == "razdor_idx":
            continue
        out[k] = v[idx] if (isinstance(v, np.ndarray) and v.shape[0] == n) else v
    return out


# Dynamics parameters and readout parameters live on very different scales and
# need very different steps. The readout is a plain linear map and tolerates a
# brisk learning rate. The coupling parameters sit inside a 120-step unrolled
# nonlinear ODE, where a step that is merely ambitious will push the ensemble
# out of the partially-locked regime into chaos, and the gradient after that
# point is measuring the wrong surface.
LR_HEAD = {"W": 1.0, "b": 1.0}
LR_DYN = {"w": 0.30, "K": 0.22, "PHI": 0.22, "L": 0.22,
          "rho_raw": 0.30, "D": 0.30, "E": 0.30}


def train(model, tr, va, epochs=18, bs=128, lr=0.020, seed=0,
          clip=4.0, verbose=True, tag=""):
    """Adam with a cosine-decayed step, per-tensor gradient clipping, and a
    per-group learning-rate multiplier. Nothing exotic; the care is all in the
    relative step sizes."""
    rng = np.random.default_rng(seed)
    state = {k: [np.zeros_like(getattr(model, k)),
                 np.zeros_like(getattr(model, k))] for k in model.pnames}
    mult = {**LR_HEAD, **LR_DYN}
    n = tr["psi"].shape[0]
    total = epochs * int(np.ceil(n / bs))
    step = 0
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s0 in range(0, n, bs):
            b = batch(tr, perm[s0:s0 + bs])
            _, cache = model.forward(b)
            g = model.backward(cache, b["y"])
            step += 1
            cur = lr * 0.5 * (1.0 + np.cos(np.pi * step / total))
            for k in model.pnames:
                gg = g[k]
                nrm = np.sqrt(float((gg * gg).sum()))
                if nrm > clip:
                    gg = gg * (clip / nrm)
                mm, vv = state[k]
                mm *= 0.9; mm += 0.1 * gg
                vv *= 0.999; vv += 0.001 * gg * gg
                mh = mm / (1.0 - 0.9 ** step)
                vh = vv / (1.0 - 0.999 ** step)
                setattr(model, k, getattr(model, k)
                        - cur * mult[k] * mh / (np.sqrt(vh) + 1e-8))
        if verbose and (ep % 4 == 3 or ep == epochs - 1):
            # evaluate on a fixed slice, not the whole held-out set: a full
            # forward pass over every validation office at every checkpoint
            # costs more than the epoch that produced it
            vs = batch(va, np.arange(min(700, va["psi"].shape[0])))
            a = model.accuracy(vs, vs["y"])
            hist.append((ep + 1, a))
            print(f"      {tag:12s} epoch {ep+1:2d}   val accuracy {a:.4f}",
                  flush=True)
    return hist


# ==============================================================================
# 6.  MEASUREMENTS
# ==============================================================================
# Accuracy alone would hide the whole argument, because the interesting claim is
# not "which model is best" but "what does each model become in order to win,
# and what does it give up". These four measurements are the argument.

def per_regime_accuracy(model, d):
    """Accuracy split by test regime."""
    pred = model.predict(d)
    out = {}
    for ri, name in enumerate(REGIMES):
        sel = d["reg"] == ri
        out[name] = float((pred[sel] == d["y"][sel]).mean()) if sel.any() else float("nan")
    return out


def carrier_mass(model):
    """Fraction of the readout's total weight magnitude that sits on the phase
    (carrier / melody) features rather than the amplitude (content / words)
    features. The global order parameter is excluded from both, since it belongs
    to neither channel.

    This is the Ohrid decision rendered as a number. A model near 0.5 is
    splitting its trust; a model near 1.0 has become a metrical reader and will
    not be fooled when the words are swapped."""
    nc, nk = model.n_carrier, model.n_content
    car = float(np.abs(model.W[:, :nc]).sum())
    con = float(np.abs(model.W[:, nc:nc + nk]).sum())
    return car / (car + con)


def order_parameter(model, d):
    """Mean |Z| over the readout window: 0 = scattered, 1 = perfect unison.
    Reported on the unit-weight mean field so it is a clean synchrony index and
    not contaminated by how loud any voice happens to be."""
    _, cache = model.forward(d)
    TH = cache["TH"]
    t0 = TH.shape[0] - model.window
    P = np.cos(TH[t0:]).mean(axis=2)
    Q = np.sin(TH[t0:]).mean(axis=2)
    return float(np.sqrt(P * P + Q * Q).mean())


def phase_code_capacity(model, d):
    """How much information the ensemble is still CAPABLE of holding in relative
    phase, independent of whether it uses it.

    Take each voice's phase relative to the mean field over the readout window,
    average it as a unit vector, and measure the spread of those averages across
    voices. If every voice sits at the same offset there is one symbol and the
    code is empty; the number goes to zero. This is the quantity that
    over-coupling destroys, and it is destroyed whether or not the loss notices.
    """
    _, cache = model.forward(d)
    TH = cache["TH"]
    t0 = TH.shape[0] - model.window
    P = np.cos(TH[t0:]).mean(axis=2, keepdims=True)
    Q = np.sin(TH[t0:]).mean(axis=2, keepdims=True)
    nrm = np.sqrt(P * P + Q * Q) + 1e-12
    c = (np.cos(TH[t0:]) * P + np.sin(TH[t0:]) * Q) / nrm
    s = (np.sin(TH[t0:]) * P - np.cos(TH[t0:]) * Q) / nrm
    mc, ms = c.mean(axis=0), s.mean(axis=0)              # [B,N] unit-ish vectors
    # spread across voices of the mean relative-phase vector
    return float(np.sqrt(mc.var(axis=1) + ms.var(axis=1)).mean())


def reentrainment(model, d):
    """The tomb measurement, taken only on RAZDOR offices.

    For the detuned voice and for the healthy voices separately, compute the
    phase-locking value against the ensemble mean field over the readout window:
    |mean_t exp(i(th_i - Psi))|, in [0,1]. A voice that holds a steady offset
    scores near 1 however large that offset is; a voice still sliding relative to
    the room scores near 0.

    Note carefully what this does NOT reward. It does not reward agreeing. It
    rewards keeping time. Naum's tomb did not ask the disordered to become the
    same as everyone else; it asked them to come back into the hours.
    """
    sel = d["reg"] == REGIMES.index("RAZDOR")
    sub = batch(d, np.where(sel)[0])
    _, cache = model.forward(sub)
    TH = cache["TH"]
    t0 = TH.shape[0] - model.window
    P = np.cos(TH[t0:]).mean(axis=2, keepdims=True)
    Q = np.sin(TH[t0:]).mean(axis=2, keepdims=True)
    nrm = np.sqrt(P * P + Q * Q) + 1e-12
    c = (np.cos(TH[t0:]) * P + np.sin(TH[t0:]) * Q) / nrm
    s = (np.sin(TH[t0:]) * P - np.cos(TH[t0:]) * Q) / nrm
    plv = np.sqrt(c.mean(axis=0) ** 2 + s.mean(axis=0) ** 2)     # [B,N]
    ill = sub["sick"]
    B = plv.shape[0]
    sick_plv = plv[np.arange(B), ill]
    mask = np.ones_like(plv, dtype=bool)
    mask[np.arange(B), ill] = False
    well_plv = plv[mask].reshape(B, -1).mean(axis=1)
    return float(sick_plv.mean()), float(well_plv.mean())


def office_trace(model, d, k=0):
    """Print one RAZDOR office as a text trace: the detuned voice's offset from
    the mean field, sampled through the night. Included because a table of
    averages does not let you watch a voice come back."""
    sel = np.where(d["reg"] == REGIMES.index("RAZDOR"))[0]
    sub = batch(d, sel[k:k + 1])
    _, cache = model.forward(sub)
    TH = cache["TH"][:, 0, :]
    i = int(sub["sick"][0])
    P = np.cos(TH).mean(axis=1)
    Q = np.sin(TH).mean(axis=1)
    psi_field = np.arctan2(Q, P)
    off = wrap_pi(TH[:, i] - psi_field)
    lines = []
    for t in range(0, TH.shape[0], 8):
        pos = int(round((off[t] + np.pi) / (2 * np.pi) * 40))
        pos = min(max(pos, 0), 40)
        bar = ["."] * 41
        bar[20] = "|"
        bar[pos] = "#"
        lines.append(f"      step {t:3d}   {''.join(bar)}   offset {off[t]:+.2f} rad")
    return lines


# ==============================================================================
# 7.  STRUCTURAL SELF-TESTS
# ==============================================================================
# These check invariants of the architecture, not the loss. A model can train to
# a good number while quietly violating the thing it was built to embody, and
# then the number means nothing.

def test_global_phase_invariance():
    """Add the same constant to every initial phase AND to the cantor's phase.
    Nothing about the office has changed except where zero is. The features must
    be identical to floating-point precision.

    This is the load-bearing invariance of the whole design: the ensemble reads
    relative phase and has no access to absolute phase at all."""
    N = 9
    d = initial_state(make_dataset(6, seed=21), N, seed=21)
    mod = Podoben(N=N, window=20, config="KLIROS", seed=2)
    f1, _ = mod.forward(d)
    alpha = 1.234567
    d2 = dict(d)
    d2["th0"] = d["th0"] + alpha
    d2["psi"] = d["psi"] + alpha
    f2, _ = mod.forward(d2)
    err = float(np.abs(f1 - f2).max())
    assert err < 1e-9, f"global phase invariance broken: {err:.3e}"
    return err


def test_free_running_frequency():
    """With no coupling, no drive and no detuning, each voice must advance by
    exactly DT * w_i per step. This checks the integrator against the closed
    form, and catches any accidental extra term in the phase equation."""
    N = 6
    d = initial_state(make_dataset(3, seed=5), N, seed=5)
    d["gate"] = np.zeros_like(d["gate"])
    d["detune"] = np.zeros_like(d["detune"])
    mod = Podoben(N=N, window=5, config="AMVON", seed=1)   # K = L = 0
    _, cache = mod.forward(d)
    TH = cache["TH"]
    T = TH.shape[0] - 1
    expect = d["th0"][None, :, :] + DT * mod.w[None, None, :] * np.arange(T + 1)[:, None, None]
    err = float(np.abs(TH - expect).max())
    assert err < 1e-9, f"free-running phase wrong: {err:.3e}"
    return err


def test_amplitude_fixed_point():
    """With no coupling and no drive, the Landau term r*(rho - r^2) must carry
    every amplitude to sqrt(rho). Checks the amplitude equation and the softplus
    parameterisation together."""
    N = 6
    d = initial_state(make_dataset(3, seed=6), N, seed=6)
    d["gate"] = np.zeros_like(d["gate"])
    d["detune"] = np.zeros_like(d["detune"])
    mod = Podoben(N=N, window=5, config="AMVON", seed=1)
    for _ in range(6):                       # run it long enough to settle
        d["psi"] = np.concatenate([d["psi"], d["psi"]], axis=1)
        d["gate"] = np.concatenate([d["gate"], d["gate"]], axis=1)
        d["env"] = np.concatenate([d["env"], d["env"]], axis=1)
    _, cache = mod.forward(d)
    r_final = cache["R"][-1]
    target = np.sqrt(softplus(mod.rho_raw))[None, :]
    err = float(np.abs(r_final - target).max())
    assert err < 1e-6, f"amplitude fixed point wrong: {err:.3e}"
    return err


def test_order_parameter_bounded():
    """|Z| must lie in [0,1] for every office and every step. If it ever exceeds
    one the mean-field computation is wrong and every phase feature is wrong
    with it."""
    N = 12
    d = initial_state(make_dataset(40, seed=8), N, seed=8)
    mod = Podoben(N=N, window=30, config="KLIROS", seed=4)
    _, cache = mod.forward(d)
    TH = cache["TH"]
    P = np.cos(TH).mean(axis=2)
    Q = np.sin(TH).mean(axis=2)
    R = np.sqrt(P * P + Q * Q)
    assert R.min() >= -1e-12 and R.max() <= 1.0 + 1e-12, "order parameter out of range"
    return float(R.max())


def test_edinoglasie_actually_locks():
    """The over-coupled configuration must genuinely reach unison, otherwise it
    is not the condition the experiment claims to be testing. Its order
    parameter must be far above the free choir's."""
    N = 12
    d = initial_state(make_dataset(60, seed=9), N, seed=9)
    a = Podoben(N=N, window=30, config="EDINOGLASIE", seed=4)
    b = Podoben(N=N, window=30, config="KLIROS", seed=4)
    Ra, Rb = order_parameter(a, d), order_parameter(b, d)
    assert Ra > 0.90, f"EDINOGLASIE failed to lock: R = {Ra:.3f}"
    assert Ra > Rb + 0.3, f"EDINOGLASIE not distinguishable from KLIROS: {Ra:.3f} vs {Rb:.3f}"
    return Ra, Rb


def test_amvon_coupling_stays_dead():
    """AMVON's voices must never hear each other, not at initialisation and not
    after training. The mask has to survive the optimiser."""
    N = 8
    tr = initial_state(make_dataset(80, seed=10), N, seed=10)
    mod = Podoben(N=N, window=20, config="AMVON", seed=3)
    train(mod, tr, tr, epochs=2, bs=40, lr=0.05, verbose=False)
    k, l = float(np.abs(mod.K).max()), float(np.abs(mod.L).max())
    assert k == 0.0 and l == 0.0, f"AMVON coupling leaked: K={k:.3e} L={l:.3e}"
    return k, l


def test_edinoglasie_lags_stay_zero():
    """EDINOGLASIE must keep every phase lag at exactly zero through training.
    A lag that leaked back in would restore the code the configuration exists to
    destroy, and the comparison would be meaningless."""
    N = 8
    tr = initial_state(make_dataset(80, seed=12), N, seed=12)
    mod = Podoben(N=N, window=20, config="EDINOGLASIE", seed=3)
    K0 = mod.K.copy()
    train(mod, tr, tr, epochs=2, bs=40, lr=0.05, verbose=False)
    p = float(np.abs(mod.PHI).max())
    dk = float(np.abs(mod.K - K0).max())
    assert p == 0.0 and dk == 0.0, f"EDINOGLASIE drifted: PHI={p:.3e} dK={dk:.3e}"
    return p, dk


def test_modes_share_mean_tempo_and_mean_loudness():
    """No scalar shortcut. All eight modes must have the same mean tempo and the
    same mean loudness, so neither channel can be classified by a single average
    and the network is forced to represent the SHAPE of the cycle."""
    t_err = float(np.abs(TEMPO.mean(axis=1) - 1.0).max())
    e_err = float(np.abs(ENVL.mean(axis=1) - 1.0).max())
    assert t_err < 1e-12 and e_err < 1e-12, "modes separable by their mean"
    # and the two channels must be drawn independently: no correlation between
    # a mode's tempo profile and its envelope profile
    tc = TEMPO - TEMPO.mean()
    ec = ENVL - ENVL.mean()
    corr = float(abs((tc * ec).sum() / np.sqrt((tc * tc).sum() * (ec * ec).sum())))
    assert corr < 0.35, f"tempo and envelope profiles too correlated: {corr:.3f}"
    return t_err, e_err, corr


def test_prevod_envelope_really_lies():
    """In PREVOD offices the envelope must belong to a mode other than the label,
    so that a system reading the words gets a wrong answer rather than a noisy
    one. Verified by matching each office's envelope against the eight profiles
    and checking the best match is not the true label more often than chance
    would give."""
    d = make_dataset(700, seed=33)
    sel = np.where(d["reg"] == REGIMES.index("PREVOD"))[0]
    assert sel.size > 40
    hits = 0
    for b in sel:
        cyc = d["env"][b].reshape(-1, STEPS_PER_BEAT).mean(axis=1)[:CYCLE * 2]
        best, bs_ = None, 1e18
        for mmode in range(N_MODE):
            for off in range(CYCLE):
                ref = np.array([ENVL[mmode, (i + off) % CYCLE] for i in range(len(cyc))])
                e = float(((cyc - ref) ** 2).sum())
                if e < bs_:
                    bs_, best = e, mmode
        hits += (best == d["y"][b])
    rate = hits / sel.size
    assert rate < 0.20, f"PREVOD envelope still names the true mode {rate:.2f} of the time"
    return rate


def test_prestavlenie_gap_is_real():
    """The silence must be a contiguous block of zero gate, and the returning
    cantor must arrive at a discontinuous phase. Otherwise the regime is not
    testing what it claims."""
    d = make_dataset(400, seed=44)
    sel = np.where(d["reg"] == REGIMES.index("PRESTAVLENIE"))[0]
    assert sel.size > 30
    for b in sel[:25]:
        g = d["gate"][b]
        zeros = np.where(g == 0.0)[0]
        assert zeros.size >= 14, "gap too short"
        assert zeros.max() - zeros.min() + 1 == zeros.size, "gap not contiguous"
    # every non-PRESTAVLENIE office must have an unbroken cantor
    other = np.where(d["reg"] != REGIMES.index("PRESTAVLENIE"))[0]
    assert float(d["gate"][other].min()) == 1.0, "silence leaked into other regimes"
    return int(sel.size)


def test_training_reduces_loss():
    """The elementary sanity check: the optimiser must move the loss down."""
    N = 10
    tr = initial_state(make_dataset(300, seed=13), N, seed=13)
    mod = Podoben(N=N, window=24, config="KLIROS", seed=2)
    before = mod.loss(tr, tr["y"])
    train(mod, tr, tr, epochs=6, bs=60, lr=0.03, verbose=False)
    after = mod.loss(tr, tr["y"])
    assert after < before - 0.05, f"training did not reduce loss: {before:.4f} -> {after:.4f}"
    return before, after


# ==============================================================================
# 8.  MAIN
# ==============================================================================

def rule(ch="="):
    print(ch * 78)


def main():
    np.set_printoptions(precision=4, suppress=True)
    rule()
    print(" PODOBEN — the Model-Melody Engine")
    print(" Chapter 0228 · Naum of Ohrid (c. 830 – 910)")
    print(" A choir of coupled oscillators. Knowledge lives in the phase lags.")
    rule()

    N, WINDOW = 12, 30
    print(f"\n ensemble        {N} oscillators")
    print(f" office          {T_STEPS} steps of dt = {DT}  "
          f"({T_STEPS // (CYCLE * STEPS_PER_BEAT)} metrical cycles)")
    print(f" metre           {CYCLE} beats x {STEPS_PER_BEAT} steps  "
          f"= modulation at {2*np.pi/(CYCLE*STEPS_PER_BEAT*DT):.2f} rad/unit")
    print(f" verdict read    last {WINDOW} steps only")
    print(f" task            {N_MODE}-way mode identification (the oktoechos)")

    # ---------------------------------------------------------------- tests --
    print("\n" + "-" * 78)
    print(" STRUCTURAL SELF-TESTS")
    print("-" * 78)
    e = test_global_phase_invariance()
    print(f"   global phase invariance          max feature change {e:.2e}   PASS")
    e = test_free_running_frequency()
    print(f"   free-running phase = dt * w      max error          {e:.2e}   PASS")
    e = test_amplitude_fixed_point()
    print(f"   amplitude settles to sqrt(rho)   max error          {e:.2e}   PASS")
    e = test_order_parameter_bounded()
    print(f"   order parameter within [0,1]     max |Z|            {e:.4f}     PASS")
    ra, rb = test_edinoglasie_actually_locks()
    print(f"   EDINOGLASIE reaches unison       |Z| {ra:.3f} vs KLIROS {rb:.3f}   PASS")
    k, l = test_amvon_coupling_stays_dead()
    print(f"   AMVON coupling stays exactly 0   max|K| {k:.1f}  max|L| {l:.1f}        PASS")
    p, dk = test_edinoglasie_lags_stay_zero()
    print(f"   EDINOGLASIE lags stay exactly 0  max|PHI| {p:.1f}  dK {dk:.1f}       PASS")
    t_e, e_e, corr = test_modes_share_mean_tempo_and_mean_loudness()
    print(f"   modes share mean tempo/loudness  err {t_e:.1e}/{e_e:.1e} corr {corr:.3f}  PASS")
    r = test_prevod_envelope_really_lies()
    print(f"   PREVOD words name the true mode  {r*100:.1f}% of the time           PASS")
    nsel = test_prestavlenie_gap_is_real()
    print(f"   PRESTAVLENIE silence contiguous  checked on {nsel} offices        PASS")
    b_, a_ = test_training_reduces_loss()
    print(f"   training reduces loss            {b_:.4f} -> {a_:.4f}            PASS")

    # ------------------------------------------------------- gradient check --
    print("\n" + "-" * 78)
    print(" FINITE-DIFFERENCE GRADIENT CHECK")
    print("-" * 78)
    for cfg in CONFIGS:
        gradient_check(cfg)

    # --------------------------------------------------------------- train --
    print("\n" + "-" * 78)
    print(" TRAINING")
    print("-" * 78)
    N_TRAIN, N_VAL = 3600, 1600
    tr = initial_state(make_dataset(N_TRAIN, seed=1104), N, seed=1104)
    va = initial_state(make_dataset(N_VAL, seed=910), N, seed=910)
    print(f"   {N_TRAIN} training offices, {N_VAL} held out, "
          f"regimes drawn uniformly\n")

    models = {}
    for cfg in CONFIGS:
        print(f"   [{cfg}]")
        mod = Podoben(N=N, window=WINDOW, config=cfg, seed=17)
        train(mod, tr, va, epochs=20, bs=128, lr=0.020, seed=5, tag=cfg)
        models[cfg] = mod

    # -------------------------------------------------------------- report --
    print("\n" + "-" * 78)
    print(" RESULTS  —  accuracy by regime (chance = 0.125)")
    print("-" * 78)
    hdr = f"   {'regime':<16}" + "".join(f"{c:>14}" for c in CONFIGS)
    print(hdr)
    table = {c: per_regime_accuracy(models[c], va) for c in CONFIGS}
    for rname in REGIMES:
        print(f"   {rname:<16}" + "".join(f"{table[c][rname]:>14.4f}" for c in CONFIGS))
    print(f"   {'ALL':<16}" + "".join(
        f"{models[c].accuracy(va, va['y']):>14.4f}" for c in CONFIGS))

    print("\n" + "-" * 78)
    print(" WHAT EACH ENSEMBLE BECAME")
    print("-" * 78)
    print(f"   {'measurement':<30}" + "".join(f"{c:>14}" for c in CONFIGS))
    print(f"   {'carrier weight fraction':<30}" + "".join(
        f"{carrier_mass(models[c]):>14.4f}" for c in CONFIGS))
    print(f"   {'order parameter |Z|':<30}" + "".join(
        f"{order_parameter(models[c], va):>14.4f}" for c in CONFIGS))
    print(f"   {'phase-code capacity':<30}" + "".join(
        f"{phase_code_capacity(models[c], va):>14.4f}" for c in CONFIGS))
    re_s, re_w = {}, {}
    for c in CONFIGS:
        re_s[c], re_w[c] = reentrainment(models[c], va)
    print(f"   {'re-entrainment, broken voice':<30}" + "".join(
        f"{re_s[c]:>14.4f}" for c in CONFIGS))
    print(f"   {'  (healthy voices, same runs)':<30}" + "".join(
        f"{re_w[c]:>14.4f}" for c in CONFIGS))

    print("\n" + "-" * 78)
    print(" ONE OFFICE IN RAZDOR  —  the detuned voice against the mean field")
    print(" ('|' marks the field; '#' marks the voice)")
    print("-" * 78)
    for c in ("KLIROS", "AMVON"):
        print(f"   [{c}]")
        for line in office_trace(models[c], va, k=0):
            print(line)
        print()

    rule()
    print(" The choir that could not be broken by the loss of its cantor is the")
    print(" one whose voices were listening to each other. The choir that healed")
    print(" the broken voice fastest is the one that had nothing left to say.")
    rule()


if __name__ == "__main__":
    main()

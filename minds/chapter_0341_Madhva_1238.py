#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0341 · Madhva (1238-1317)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 The BHEDA-SVARUPA NETWORK (BSN)  —  a difference-primitive cognitive engine
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Madhva's one irreducible cognitive claim is not "there are two things".
It is far stranger and far more computational:

    bheda is dharmi-svarupa — difference is not a RELATION BETWEEN things,
    it is the very BEING of a thing. To cognise an object just is to
    apprehend it as differing.

Every mainstream learned representation does the opposite. It stores a point
(an embedding), and derives difference afterwards as a distance. Madhva says
the point is the derived thing and the difference is the primitive. This file
takes him literally.

Five doctrines, five mechanisms, each one load-bearing and each one testable:

  1. BHEDA CORE (bheda = svarupa)
     The primitive learned operator is exactly antisymmetric. It maps a PAIR
     to a difference-signature. Two exact algebraic properties fall out and
     are asserted in the self-tests, not merely hoped for:
         b(x, y) = -b(y, x)      (difference has a side but not a preference)
         b(x, x) = 0             (nothing differs from itself — exactly, to 0.0)
     There is deliberately no bias term in the core: a bias would make a thing
     differ from itself by a learned constant, which is the one thing Madhva
     never allows.

  2. VISESA (the regress-stopper) — an IMPLICIT FIXED-POINT LAYER
     Madhva's opponents pressed the classical regress: if a thing's attributes
     differ from the thing, you need a distinguisher; the distinguisher needs a
     distinguisher; ad infinitum. His answer was the visesa — a self-referring
     particular (svanirvahaka) that distinguishes without needing a further
     distinguisher. Here that is a layer of literally infinite depth whose
     recursion provably terminates:
         v = u + rho * W_hat @ v     ->     v* = (I - rho*W_hat)^-1 u
     W_hat is spectrally normalised and rho < 1, so the map is a contraction:
     the fixed point exists, is unique, and is reached in closed form. The
     gradient comes from the implicit function theorem, not from unrolling.
     The regress is not truncated. It is solved.

  3. SAKSIN (the witness) — NON-PARAMETRIC, UNTRAINED, EXACT
     Madhva's saksin is the svarupendriya, the soul's own organ of witness.
     It is the ultimate arbiter of the three anupramanas (perception,
     inference, testimony) and — uniquely in Indian epistemology — it cannot
     itself err. So here it holds no trainable weights at all. It is a fixed
     orthogonal frame whose reconstruction error is machine epsilon, and it
     issues two verdicts:
       (a) the identity verdict iota — an exact, non-learned detection of
           "this is that", which is precisely the case b(x,x)=0 covers;
       (b) the dosa (defect) gate. Madhva pairs svatah-pramanya with
           paratah-apramanya: a cognition is valid BY DEFAULT, and only a
           positively detected defect discounts it. So the gate never
           certifies a channel. It only demotes the one that diverges from
           the consensus of the others.

  4. BIMBA-PRATIBIMBA (dependent reality) — ARCHITECTURAL, NOT LEARNED
     Jivas are pratibimbas, reflections of the bimba. Real, but paratantra:
     they exist only in dependence. Here the K reader slots do not own their
     readout matrix; they modulate a single shared bimba. The asymmetry is
     structural and is verified by ablation: delete the bimba and every slot
     dies at once; delete one slot's visesa and the others are untouched.

  5. TARATAMYA (gradation) — THE ANTI-CONVERGENCE OBJECTIVE
     This is where Madhva breaks hardest with modern practice. Every slot has
     a FIXED, non-trainable capacity mask. Training does not push them toward
     a common optimum. Each strives to its own ceiling, and an explicit
     anti-collapse term forbids their inner natures from merging. The trained
     system therefore ends stratified and plural, by construction. There is no
     single optimum that all learners approach.

TASK: PANCABHEDA DISCRIMINATION
-------------------------------
A synthetic world of three kinds — one Ishvara (svatantra, unique), many
jivas, many jada — and a 6-way classification of what kind of difference a
presented pair instantiates. Crucially, each pair is observed under a random
shared GAUGE OFFSET, so absolute position carries no information whatsoever.
Only the difference between the two observations is informative. This is not
decoration: it makes Madhva's thesis a hard constraint on the data. A
point-wise (sum-pooled) encoder is provably at chance on the kind classes and
we demonstrate that empirically against a trained baseline.

  class 0  Ishvara / jiva          class 3  jiva / jiva (distinct)
  class 1  Ishvara / jada          class 4  jada / jada (distinct)
  class 2  jiva / jada             class 5  identity (same entity, no bheda)

Classes 3 and 4 are separable only because Madhva insists that jiva-jiva
difference and jada-jada difference are DIFFERENT KINDS of difference — the
fourth and fifth bhedas, not one relation applied twice. So the two identity
codebooks occupy different subspaces with different scale.

Pure NumPy. Hand-derived gradients. Finite-difference check on every parameter
group. Real training loop. Run the file to reproduce everything.

    python3 chapter_0341_Madhva_1238.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------------------
D_IN     = 16     # observation dimension
H        = 24     # difference-signature width
N_CLS    = 6      # the five bhedas + identity
K_SLOTS  = 4      # pratibimba slots (jivas), each with a fixed capacity
N_PROTO  = 8      # agama (testimony) prototype bank size
RHO      = 0.9    # visesa contraction factor; < 1 guarantees the regress halts
GATE_T   = 0.50   # saksin gate temperature (soft demotion, not winner-take-all)
GATE_DELTA = 1e-2 # soft-norm floor inside the gate; keeps a silent channel finite
PROTO_T  = 0.5    # testimony match temperature
IOTA_TAU = 1e-3   # identity-verdict bandwidth (tight: identity is exact or not)
EPS      = 1e-8

# Capacity ladder (svarupa-yogyata). NOT trainable, NOT learned, NOT equalised.
CAPACITY = np.array([1.00, 0.33, 0.125, 0.042])


# ==============================================================================
# 1. THE WORLD  —  generating pancabheda data
# ==============================================================================

class PancabhedaWorld:
    """
    A small ontology with one independent entity and many dependent ones.

    Latent structure
    ----------------
    Every entity has a latent code z = mu[kind] + iota_identity, where the
    identity part is drawn from a kind-specific subspace. The Ishvara is a
    single fixed entity. Jivas draw identity codes from subspace S_J at one
    scale; jada draw from a disjoint subspace S_D at another. That is what
    makes the fourth bheda (jiva/jiva) and the fifth (jada/jada) genuinely
    distinguishable rather than the same relation twice over.

    Observation
    -----------
    x = A z + g + noise, where g is a GAUGE OFFSET shared by both members of
    the pair and redrawn for every pair, with a standard deviation large
    enough to swamp the kind signal. Absolute position is therefore pure
    nuisance. Only difference informs.

    A self-pair reuses the SAME observation for both members. Not a resampling
    of the same entity — the same observation. Madhva's point is that identity
    is not a very small difference; it is the absence of difference.
    """

    KIND_ISHVARA, KIND_JIVA, KIND_JADA = 0, 1, 2

    def __init__(self, seed=1238, n_jiva=40, n_jada=40, gauge_sd=2.0, noise_sd=0.03):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_jiva, self.n_jada = n_jiva, n_jada
        self.gauge_sd, self.noise_sd = gauge_sd, noise_sd

        # Kind means: three well-separated directions in latent space.
        self.mu = rng.normal(0, 1.0, size=(3, D_IN))
        self.mu /= np.linalg.norm(self.mu, axis=1, keepdims=True)
        self.mu *= 1.6

        # Two disjoint identity subspaces, different dimensionality and scale.
        # S_J: first 6 latent axes, wide spread.  S_D: last 5 axes, tight.
        self.jiva_codes = np.zeros((n_jiva, D_IN))
        self.jiva_codes[:, :6] = rng.normal(0, 0.85, size=(n_jiva, 6))
        self.jada_codes = np.zeros((n_jada, D_IN))
        self.jada_codes[:, -5:] = rng.normal(0, 0.32, size=(n_jada, 5))

        # The single independent entity. It alone has no identity-code spread:
        # svatantra means it is not one member of a class of like things.
        self.ishvara_code = np.zeros(D_IN)

        # Fixed observation mixing matrix (the world is not axis-aligned).
        self.A = rng.normal(0, 1.0 / np.sqrt(D_IN), size=(D_IN, D_IN))

    def _latent(self, kind, idx):
        if kind == self.KIND_ISHVARA:
            return self.mu[0] + self.ishvara_code
        if kind == self.KIND_JIVA:
            return self.mu[1] + self.jiva_codes[idx]
        return self.mu[2] + self.jada_codes[idx]

    def _observe(self, z, gauge):
        return z @ self.A.T + gauge + self.rng.normal(0, self.noise_sd, size=D_IN)

    def sample(self, n):
        """Return (xa, xb, y). Classes are drawn uniformly."""
        xa = np.zeros((n, D_IN)); xb = np.zeros((n, D_IN)); y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cls = self.rng.integers(0, N_CLS)
            gauge = self.rng.normal(0, self.gauge_sd, size=D_IN)

            if cls == 5:
                # Identity: one observation, presented twice. No bheda at all.
                kind = self.rng.integers(0, 3)
                idx = (0 if kind == 0 else
                       self.rng.integers(0, self.n_jiva if kind == 1 else self.n_jada))
                obs = self._observe(self._latent(kind, idx), gauge)
                xa[i], xb[i] = obs, obs
            else:
                if cls == 0:   ka, ia, kb, ib = 0, 0, 1, self.rng.integers(0, self.n_jiva)
                elif cls == 1: ka, ia, kb, ib = 0, 0, 2, self.rng.integers(0, self.n_jada)
                elif cls == 2: ka, ia, kb, ib = 1, self.rng.integers(0, self.n_jiva), 2, self.rng.integers(0, self.n_jada)
                elif cls == 3:
                    ia, ib = self.rng.choice(self.n_jiva, size=2, replace=False)
                    ka, kb = 1, 1
                else:
                    ia, ib = self.rng.choice(self.n_jada, size=2, replace=False)
                    ka, kb = 2, 2
                # Randomise presentation order: the label is unordered, so the
                # readout must be invariant to swap even though the internal
                # signature is antisymmetric under it.
                if self.rng.random() < 0.5:
                    ka, ia, kb, ib = kb, ib, ka, ia
                xa[i] = self._observe(self._latent(ka, ia), gauge)
                xb[i] = self._observe(self._latent(kb, ib), gauge)
            y[i] = cls
        return xa, xb, y


# ==============================================================================
# 2. THE SAKSIN  —  untrained, non-parametric, exact
# ==============================================================================

class Saksin:
    """
    The witness. Holds no trainable weights and receives no gradient, because
    for Madhva the saksin is not an acquired competence: it is the soul's own
    nature (svarupendriya), given entire, incapable of error.

    Three offices, mirroring the three the tradition assigns it:

      frame / unframe : an exactly invertible orthogonal change of basis.
                        Its reconstruction error is machine epsilon, and the
                        self-tests assert this. Every learned channel in this
                        file is lossy; this one is not. That contrast is the
                        whole point.

      identity(xa,xb) : the verdict iota in [0,1]. Exactly 1.0 when the two
                        presentations are the same presentation. This is the
                        witness apprehending "this is that" directly, not by
                        inference from features.

      dosa_gate(...)  : validity is the default. The gate does not certify.
                        It computes each channel's divergence from the
                        consensus and demotes only the divergent one
                        (paratah-apramanya: invalidity comes from a defect,
                        and must be positively shown).
    """

    def __init__(self, dim=D_IN, seed=7):
        rng = np.random.default_rng(seed)
        Q, R = np.linalg.qr(rng.normal(size=(dim, dim)))
        # Fix the sign convention so Q is a proper, deterministic frame.
        self.Q = Q * np.sign(np.diag(R))

    def frame(self, x):
        return x @ self.Q

    def unframe(self, w):
        return w @ self.Q.T

    def reconstruction_error(self, x):
        return float(np.max(np.abs(self.unframe(self.frame(x)) - x)))

    @staticmethod
    def identity(xa, xb):
        """iota = exp(-||xa-xb||^2 / tau). Exactly 1.0 iff xa is xb."""
        d2 = np.sum((xa - xb) ** 2, axis=1)
        return np.exp(-d2 / IOTA_TAU)

    @staticmethod
    def dosa_gate(channels, temperature=GATE_T):
        """
        channels : list of C arrays, each (B, H)
        returns  : alpha (B, C), dosa (B, C), and a cache for backward.

        Each channel is compared, in direction only, against the consensus of
        THE OTHERS — leave-one-out, never including itself. A cognition cannot
        be part of the evidence for its own soundness; that is the circle
        Madhva refuses when he denies that a pramana can certify itself. So:

            cons_c = mean_{c' != c} u_c'
            dosa_c = || u_c - cons_c ||^2
            alpha  = softmax(-dosa / T)

        With C = 3 this is exactly "how far are you from where the other two
        agree". A garbled channel is demoted; the sound ones are left alone.
        """
        C = len(channels)
        # Centre each message before comparing. The three channels have
        # systematically different baselines (perception's message is
        # non-negative by construction; the other two are signed), and a
        # comparison of raw offsets would just rediscover which channel is
        # which. The witness is asked about the CONTRASTS each channel
        # reports, not about its resting level.
        cent = [m - m.mean(axis=1, keepdims=True) for m in channels]
        # Soft norm: sqrt(||.||^2 + delta^2) has no singularity at the origin,
        # so a channel that happens to fall silent does not blow up the gate.
        rr = [np.sqrt(np.sum(c * c, axis=1, keepdims=True) + GATE_DELTA ** 2)
              for c in cent]
        units = [c / r for c, r in zip(cent, rr)]
        S = sum(units)
        # r_c = u_c - (S - u_c)/(C-1) = (C*u_c - S)/(C-1)
        resid = [(C * u - S) / (C - 1) for u in units]
        dosa = np.stack([np.sum(r * r, axis=1) for r in resid], axis=1)   # (B,C)
        z = -dosa / temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        alpha = e / e.sum(axis=1, keepdims=True)
        cache = dict(channels=channels, cent=cent, rr=rr, units=units,
                     resid=resid, alpha=alpha, C=C, T=temperature)
        return alpha, dosa, cache

    @staticmethod
    def dosa_gate_backward(dm, m_mix_cache, gate_cache):
        """
        Backprop through the gate AND the mixture m = sum_c alpha_c * m_c.
        The witness owns no parameters, so nothing accumulates here; the
        gradient simply passes through to the channels that produced the
        messages. Returns d_channels (list of (B,H)).
        """
        channels = gate_cache['channels']; cent = gate_cache['cent']
        rr = gate_cache['rr']; resid = gate_cache['resid']
        alpha = gate_cache['alpha']; C = gate_cache['C']; T = gate_cache['T']

        # m = sum_c alpha_c * m_c
        d_channels = [alpha[:, c:c + 1] * dm for c in range(C)]
        d_alpha = np.stack([np.sum(dm * channels[c], axis=1) for c in range(C)], axis=1)

        # softmax over z = -dosa/T
        s = np.sum(d_alpha * alpha, axis=1, keepdims=True)
        d_z = alpha * (d_alpha - s)
        d_dosa = -d_z / T                                        # (B,C)

        # dosa_c = ||r_c||^2 with r_c = (C*u_c - S)/(C-1) and S = sum_j u_j.
        # dr_c/du_j = (C*delta_cj - 1)/(C-1) * I, hence:
        d_r = [2.0 * resid[c] * d_dosa[:, c:c + 1] for c in range(C)]
        sum_dr = sum(d_r)
        d_u = [(C * d_r[j] - sum_dr) / (C - 1) for j in range(C)]

        # u = cent / sqrt(||cent||^2 + delta^2), then cent = m - mean_H(m)
        for c in range(C):
            cc, r = cent[c], rr[c]
            proj = np.sum(d_u[c] * cc, axis=1, keepdims=True)
            d_cent = d_u[c] / r - cc * proj / r ** 3
            d_channels[c] += d_cent - d_cent.mean(axis=1, keepdims=True)
        return d_channels


# ==============================================================================
# 3. THE VISESA LAYER  —  an infinitely deep, provably terminating recursion
# ==============================================================================

def spectral_normalise(W):
    """W_hat = W / sigma_max(W). Returns W_hat, sigma, and the top singular pair."""
    U, S, Vt = np.linalg.svd(W)
    sigma = S[0] + EPS
    return W / sigma, sigma, U[:, 0], Vt[0, :]


def visesa_forward(u, W):
    """
    Solve v = u + RHO * W_hat @ v in closed form.

    Because ||RHO * W_hat||_2 = RHO < 1 the map is a contraction, so the
    Neumann series converges and (I - RHO*W_hat) is invertible. The layer is
    the limit of infinitely many refinements, computed exactly rather than
    truncated. That is the visesa: a distinction that terminates its own
    regress instead of demanding another distinguisher.
    """
    W_hat, sigma, u1, v1 = spectral_normalise(W)
    M = np.eye(W.shape[0]) - RHO * W_hat
    v = np.linalg.solve(M, u.T).T         # M v_n = u_n, solved for every row of u
    cache = dict(u=u, W=W, W_hat=W_hat, sigma=sigma, u1=u1, v1=v1, M=M, v=v)
    return v, cache


def visesa_backward(dv, cache):
    """
    Implicit function theorem. From v = u + rho*W_hat*v:
        dv = rho * M^-1 dW_hat v          ->   dL/dW_hat = rho * (M^-T g) v^T
        dL/du = M^-T g
    Then unwind the spectral normalisation W_hat = W / sigma(W), whose
    derivative is the outer product of the top singular vectors:
        dL/dW = G/sigma - <G, W>/sigma^2 * u1 v1^T
    No unrolling, no truncation, no vanishing gradient through depth.
    """
    M, v, sigma = cache['M'], cache['v'], cache['sigma']
    u1, v1, W = cache['u1'], cache['v1'], cache['W']
    lam = np.linalg.solve(M.T, dv.T).T          # lam_n = M^-T g_n
    du = lam
    G = RHO * (lam.T @ v)                       # dL/dW_hat
    dW = G / sigma - (np.sum(G * W) / sigma ** 2) * np.outer(u1, v1)
    return du, dW


# ==============================================================================
# 4. THE BHEDA-SVARUPA NETWORK
# ==============================================================================

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class BhedaSvarupaNet:
    """
    Forward pass, in Madhva's own order of explanation.

      1. bheda core        : the pair -> an exactly antisymmetric signature b
      2. svarupa           : s = b*b, the sign-invariant nature of the difference
      3. three anupramanas : pratyaksa (direct), anumana (visesa fixed point),
                             agama (prototype testimony)
      4. saksin            : dosa gate mixes them; iota supplies the identity verdict
      5. bimba-pratibimba  : K slots modulate one shared bimba
      6. taratamya         : fixed capacity masks; slots end plural, not merged
    """

    def __init__(self, seed=1317):
        rng = np.random.default_rng(seed)
        sc = lambda *s: rng.normal(0, np.sqrt(2.0 / s[0]), size=s)

        self.p = {
            'M':   sc(D_IN, D_IN),              # bheda core mixing (no bias, by doctrine)
            'U':   sc(D_IN, H),                 # bheda core projection
            'Wa':  sc(H, H),                    # anumana input map
            'ba':  np.zeros(H),
            'Wv':  sc(H, H),                    # visesa recursion operator
            'P':   sc(N_PROTO, H),              # agama prototype keys (received tradition)
            'V':   sc(N_PROTO, H),              # agama prototype verdicts
            'Bm':  sc(H, N_CLS),                # the bimba: one shared readout
            'g':   rng.normal(0, 0.5, size=(K_SLOTS, H)),   # each slot's visesa
            'bk':  np.zeros((K_SLOTS, N_CLS)),
            'kap': np.array([2.0]),             # weight of the witness's identity verdict
        }

        # Capacity masks: fixed at construction, never trained, never equalised.
        mrng = np.random.default_rng(seed + 1)
        self.masks = np.zeros((K_SLOTS, H))
        for k in range(K_SLOTS):
            keep = max(1, int(round(CAPACITY[k] * H)))
            self.masks[k, mrng.permutation(H)[:keep]] = 1.0

        self.saksin = Saksin()

    # -- forward ---------------------------------------------------------------
    def forward(self, xa, xb, corrupt_agama=0.0, ablate=None):
        p = self.p
        ablate = ablate or {}

        # (1) BHEDA CORE. w is antisymmetric under swap by construction, and is
        #     exactly the zero vector when the two presentations are identical.
        ya, yb = xa @ p['M'], xb @ p['M']
        w = xa * yb - xb * ya
        pre_b = w @ p['U']
        b = np.tanh(pre_b)

        # (2) SVARUPA. The label is unordered, so the readout must not care which
        #     side one stands on. Squaring is the minimal even function of an
        #     antisymmetric quantity, and it preserves b(x,x)=0 exactly.
        s = b * b

        # (3a) PRATYAKSA — direct apprehension of the difference.
        m_prat = s

        # (3b) ANUMANA — the difference refined to its own fixed point.
        u_an = s @ p['Wa'] + p['ba']
        v_an, vc = visesa_forward(u_an, p['Wv'])
        m_anum = np.tanh(v_an)

        # (3c) AGAMA — testimony. Received prototypes, matched by direction.
        sn = np.linalg.norm(s, axis=1, keepdims=True) + EPS
        s_hat = s / sn
        Pn = np.linalg.norm(p['P'], axis=1, keepdims=True) + EPS
        P_hat = p['P'] / Pn
        sim = s_hat @ P_hat.T / PROTO_T
        att = softmax(sim, axis=1)
        m_agam = att @ p['V']
        if corrupt_agama > 0.0:
            # A DEFECT (dosa) in transmission. Not additive noise, which would
            # merely make the testimony quieter: a fraction of the batch is
            # handed some OTHER case's verdict. The text still looks like the
            # text. Its marginal statistics are untouched. It is simply no
            # longer about the case in front of you — which is precisely the
            # kind of corruption a tradition actually suffers, and precisely
            # what the witness is for.
            crng = np.random.default_rng(276)
            B = m_agam.shape[0]
            n_bad = int(round(corrupt_agama * B))
            if n_bad > 1:
                bad = crng.choice(B, size=n_bad, replace=False)
                m_agam = m_agam.copy()
                m_agam[bad] = m_agam[crng.permutation(bad)]

        # (4) SAKSIN. Gate is computed but carries no parameters.
        channels = [m_prat, m_anum, m_agam]
        alpha, dosa, gc = Saksin.dosa_gate(channels)
        m = sum(alpha[:, c:c + 1] * channels[c] for c in range(len(channels)))
        iota = Saksin.identity(xa, xb)

        # (5,6) BIMBA-PRATIBIMBA under TARATAMYA.
        Bm = np.zeros_like(p['Bm']) if ablate.get('bimba') else p['Bm']
        gain = 1.0 / (1.0 + np.exp(-p['g']))            # sigma(g), (K,H)
        if ablate.get('slot') is not None:
            gain = gain.copy(); gain[ablate['slot']] = 0.0
        hk = m[None, :, :] * self.masks[:, None, :] * gain[:, None, :]   # (K,B,H)
        logits = np.einsum('kbh,hc->kbc', hk, Bm) + p['bk'][:, None, :]
        logits[:, :, 5] += p['kap'][0] * iota[None, :]

        cache = dict(xa=xa, xb=xb, ya=ya, yb=yb, w=w, b=b, s=s,
                     u_an=u_an, v_an=v_an, vc=vc, m_anum=m_anum,
                     s_hat=s_hat, sn=sn, P_hat=P_hat, Pn=Pn, sim=sim, att=att,
                     channels=channels, alpha=alpha, gc=gc, m=m, iota=iota,
                     gain=gain, hk=hk, logits=logits, Bm=Bm, ablate=ablate)
        return logits, cache

    # -- loss ------------------------------------------------------------------
    def loss(self, logits, y, cache, lam_diff=0.05):
        """
        Per-slot cross-entropy (each soul strives on its own account), plus an
        anti-collapse term on the slots' visesa vectors.

        Note what is NOT here: any term pulling the slots toward agreement.
        Madhva's liberated souls do not converge, do not merge, and do not
        become equal to one another. The objective is written so the trained
        system ends up plural.
        """
        K, B, C = logits.shape
        probs = softmax(logits, axis=2)
        ll = -np.log(probs[np.arange(K)[:, None], np.arange(B)[None, :], y[None, :]] + EPS)
        ce = ll.mean()

        gn = self.p['g'] / (np.linalg.norm(self.p['g'], axis=1, keepdims=True) + EPS)
        Gram = gn @ gn.T
        off = Gram - np.diag(np.diag(Gram))
        collapse = np.sum(off ** 2) / (K * (K - 1))

        return ce + lam_diff * collapse, dict(probs=probs, gn=gn, Gram=Gram,
                                              lam_diff=lam_diff, K=K, B=B)

    # -- backward --------------------------------------------------------------
    def backward(self, cache, y, lcache):
        p = self.p
        probs, K, B = lcache['probs'], lcache['K'], lcache['B']
        g = {k: np.zeros_like(v) for k, v in p.items()}

        # d(mean CE)/d logits
        dlogits = probs.copy()
        dlogits[np.arange(K)[:, None], np.arange(B)[None, :], y[None, :]] -= 1.0
        dlogits /= (K * B)

        # identity verdict weight
        g['kap'][0] = np.sum(dlogits[:, :, 5] * cache['iota'][None, :])

        # bias, bimba
        g['bk'] = dlogits.sum(axis=1)
        if not cache['ablate'].get('bimba'):
            g['Bm'] = np.einsum('kbh,kbc->hc', cache['hk'], dlogits)

        # hk = m * mask * gain
        dhk = np.einsum('kbc,hc->kbh', dlogits, cache['Bm'])
        gain = cache['gain']
        dgain = np.einsum('kbh,bh->kh', dhk * self.masks[:, None, :], cache['m'])
        if cache['ablate'].get('slot') is None:
            g['g'] = dgain * gain * (1.0 - gain)
        dm = np.einsum('kbh,kh->bh', dhk, self.masks * gain)

        # anti-collapse term on g (analytic)
        lam, gn = lcache['lam_diff'], lcache['gn']
        off = lcache['Gram'] - np.diag(np.diag(lcache['Gram']))
        dgn = (4.0 * lam / (K * (K - 1))) * (off @ gn)
        raw = p['g']; nrm = np.linalg.norm(raw, axis=1, keepdims=True) + EPS
        g['g'] += dgn / nrm - raw * np.sum(dgn * raw, axis=1, keepdims=True) / nrm ** 3

        # through the saksin gate to the three channels
        d_prat, d_anum, d_agam = Saksin.dosa_gate_backward(dm, None, cache['gc'])

        # -- agama channel
        att, V = cache['att'], p['V']
        g['V'] = att.T @ d_agam
        datt = d_agam @ V.T
        dsim = att * (datt - np.sum(datt * att, axis=1, keepdims=True))
        dsim /= PROTO_T
        ds_hat = dsim @ cache['P_hat']
        dP_hat = dsim.T @ cache['s_hat']
        Pn, P_hat = cache['Pn'], cache['P_hat']
        g['P'] = dP_hat / Pn - P_hat * np.sum(dP_hat * P_hat, axis=1, keepdims=True) / Pn
        sn, s_hat = cache['sn'], cache['s_hat']
        ds_from_agama = ds_hat / sn - s_hat * np.sum(ds_hat * s_hat, axis=1, keepdims=True) / sn

        # -- anumana channel
        dv = d_anum * (1.0 - cache['m_anum'] ** 2)
        du_an, dWv = visesa_backward(dv, cache['vc'])
        g['Wv'] = dWv
        g['Wa'] = cache['s'].T @ du_an
        g['ba'] = du_an.sum(axis=0)
        ds_from_anum = du_an @ p['Wa'].T

        # -- pratyaksa channel + total ds
        ds = d_prat + ds_from_anum + ds_from_agama

        # s = b*b ; b = tanh(w @ U) ; w = xa*(xb M) - xb*(xa M)
        db = ds * 2.0 * cache['b']
        dpre = db * (1.0 - cache['b'] ** 2)
        g['U'] = cache['w'].T @ dpre
        dw = dpre @ p['U'].T
        # w = xa ⊙ (xb M) - xb ⊙ (xa M)
        dyb = dw * cache['xa']
        dya = -dw * cache['xb']
        g['M'] = cache['xb'].T @ dyb + cache['xa'].T @ dya
        return g


# ==============================================================================
# 5. OPTIMISER AND TRAINING
# ==============================================================================

class Adam:
    def __init__(self, params, lr=4e-3, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def accuracy_per_slot(net, xa, xb, y):
    logits, _ = net.forward(xa, xb)
    pred = logits.argmax(axis=2)
    return (pred == y[None, :]).mean(axis=1)


def train(net, world, steps=1400, batch=192, lr=4e-3, verbose=True):
    opt = Adam(net.p, lr=lr)
    xv_a, xv_b, yv = world.sample(1500)
    hist = []
    for t in range(1, steps + 1):
        xa, xb, y = world.sample(batch)
        logits, cache = net.forward(xa, xb)
        L, lc = net.loss(logits, y, cache)
        grads = net.backward(cache, y, lc)
        opt.step(net.p, grads)
        if verbose and (t % 200 == 0 or t == 1):
            acc = accuracy_per_slot(net, xv_a, xv_b, yv)
            hist.append((t, L, acc.copy()))
            print(f"  step {t:5d} | loss {L:7.4f} | slot acc "
                  + " ".join(f"{a:.3f}" for a in acc))
    return hist, (xv_a, xv_b, yv)


# ==============================================================================
# 6. THE GAUGE-BLIND BASELINE  (a point-wise encoder, as a control)
# ==============================================================================

def sumpool_baseline(world, steps=2500, lr=0.08, seed=0):
    """
    Multinomial logistic regression on the sum-pooled pair, [xa+xb, |xa|+|xb|].
    This is the point-first view of representation: encode each thing, then
    combine. Under a shared gauge offset it has nothing to work with on the
    kind classes, and the numbers below show it.
    """
    rng = np.random.default_rng(seed)
    def feat(xa, xb):
        return np.concatenate([xa + xb, np.abs(xa) + np.abs(xb),
                               np.ones((len(xa), 1))], axis=1)
    xa, xb, y = world.sample(6000)
    X, Y = feat(xa, xb), y
    W = rng.normal(0, 0.01, size=(X.shape[1], N_CLS))
    for _ in range(steps):
        P = softmax(X @ W, axis=1)
        Ph = P.copy(); Ph[np.arange(len(Y)), Y] -= 1.0
        W -= lr * (X.T @ Ph) / len(Y)
    xta, xtb, yt = world.sample(2000)
    pred = (feat(xta, xtb) @ W).argmax(axis=1)
    overall = (pred == yt).mean()
    kind_mask = yt != 5
    return overall, (pred[kind_mask] == yt[kind_mask]).mean()


# ==============================================================================
# 7. SELF-TESTS
# ==============================================================================

def _flat_loss(net, xa, xb, y):
    logits, cache = net.forward(xa, xb)
    L, lc = net.loss(logits, y, cache)
    return L, cache, lc


def gradient_check(net, world, n=24, eps=1e-6, tol=2e-5):
    """
    Finite-difference check on EVERY parameter group, including the implicit
    visesa fixed point and the spectral normalisation inside it.
    """
    xa, xb, y = world.sample(n)
    _, cache, lc = _flat_loss(net, xa, xb, y)
    ana = net.backward(cache, y, lc)

    print("\n  finite-difference gradient check")
    print("  " + "-" * 58)
    worst = 0.0
    rng = np.random.default_rng(3)
    for name, P in net.p.items():
        flat = P.reshape(-1)
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        num, an = [], []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp, _, _ = _flat_loss(net, xa, xb, y)
            flat[i] = orig - eps; Lm, _, _ = _flat_loss(net, xa, xb, y)
            flat[i] = orig
            num.append((Lp - Lm) / (2 * eps))
            an.append(ana[name].reshape(-1)[i])
        num, an = np.array(num), np.array(an)
        rel = np.max(np.abs(num - an) / (np.abs(num) + np.abs(an) + 1e-9))
        worst = max(worst, rel)
        print(f"    {name:5s} shape {str(P.shape):12s} max rel err {rel:.3e}  "
              f"{'PASS' if rel < tol else 'FAIL'}")
    print("  " + "-" * 58)
    ok = worst < tol
    print(f"  worst relative error {worst:.3e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_antisymmetry(net, world):
    xa, xb, _ = world.sample(64)
    _, ca = net.forward(xa, xb)
    _, cb = net.forward(xb, xa)
    anti = np.max(np.abs(ca['b'] + cb['b']))
    _, cs = net.forward(xa, xa)
    self_zero = np.max(np.abs(cs['b']))
    print(f"    b(x,y) + b(y,x)  max |.| = {anti:.3e}   "
          f"{'PASS' if anti < 1e-12 else 'FAIL'}")
    print(f"    b(x,x)           max |.| = {self_zero:.3e}   "
          f"{'PASS' if self_zero == 0.0 else 'FAIL'}  (exactly zero, not approximately)")
    return anti < 1e-12 and self_zero == 0.0


def test_saksin_exactness(net, world):
    xa, _, _ = world.sample(256)
    err = net.saksin.reconstruction_error(xa)
    _, c = net.forward(xa, xa)
    iota_self = float(np.min(c['iota']))
    print(f"    witness round-trip error = {err:.3e}   "
          f"{'PASS' if err < 1e-12 else 'FAIL'}")
    print(f"    iota on identical pairs  = {iota_self:.6f}  "
          f"{'PASS' if iota_self == 1.0 else 'FAIL'}")
    return err < 1e-12 and iota_self == 1.0


def test_visesa(net):
    """The regress halts: contraction, uniqueness, and closed form == iterated."""
    W = net.p['Wv']
    W_hat, sigma, _, _ = spectral_normalise(W)
    rad = RHO * np.max(np.abs(np.linalg.eigvals(W_hat)))
    u = np.random.default_rng(5).normal(size=(8, H))
    v_star, _ = visesa_forward(u, W)
    # Picard iteration from zero — the naive infinite regress, run 4000 deep.
    v_it = np.zeros_like(u)
    for _ in range(4000):
        v_it = u + RHO * v_it @ W_hat.T
    gap = np.max(np.abs(v_star - v_it))
    resid = np.max(np.abs(v_star - (u + RHO * v_star @ W_hat.T)))
    print(f"    spectral radius of rho*W_hat = {rad:.4f}   "
          f"{'PASS' if rad < 1.0 else 'FAIL'}  (< 1 => the regress terminates)")
    print(f"    closed form vs 4000-deep iteration gap = {gap:.3e}   "
          f"{'PASS' if gap < 1e-10 else 'FAIL'}")
    print(f"    fixed-point residual = {resid:.3e}   "
          f"{'PASS' if resid < 1e-10 else 'FAIL'}")
    return rad < 1.0 and gap < 1e-10 and resid < 1e-10


def test_implicit_vs_unrolled():
    """
    The implicit gradient must equal the gradient of a very deep unrolled
    version. If it does, the layer really is the limit and not an approximation.
    """
    rng = np.random.default_rng(11)
    W = rng.normal(0, 0.3, size=(6, 6))
    u = rng.normal(size=(3, 6))
    tgt = rng.normal(size=(3, 6))

    v, cache = visesa_forward(u, W)
    dv = 2.0 * (v - tgt) / v.size
    du_imp, dW_imp = visesa_backward(dv, cache)

    def deep_loss(Wp, up, depth=600):
        W_hat, _, _, _ = spectral_normalise(Wp)
        vv = np.zeros_like(up)
        for _ in range(depth):
            vv = up + RHO * vv @ W_hat.T
        return np.mean((vv - tgt) ** 2)

    e = 1e-6
    err_W = 0.0
    for i in range(6):
        for j in range(6):
            Wp = W.copy(); Wp[i, j] += e; a = deep_loss(Wp, u)
            Wm = W.copy(); Wm[i, j] -= e; b = deep_loss(Wm, u)
            num = (a - b) / (2 * e)
            err_W = max(err_W, abs(num - dW_imp[i, j]) / (abs(num) + abs(dW_imp[i, j]) + 1e-9))
    err_u = 0.0
    for i in range(3):
        for j in range(6):
            up = u.copy(); up[i, j] += e; a = deep_loss(W, up)
            um = u.copy(); um[i, j] -= e; b = deep_loss(W, um)
            num = (a - b) / (2 * e)
            err_u = max(err_u, abs(num - du_imp[i, j]) / (abs(num) + abs(du_imp[i, j]) + 1e-9))
    ok = err_W < 1e-4 and err_u < 1e-4
    print(f"    implicit vs 600-deep unrolled: dW rel err {err_W:.3e}, "
          f"du rel err {err_u:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def test_dependence_asymmetry(net, world):
    """
    Svatantra / paratantra, made structural. Removing the bimba annihilates
    every slot at once. Removing one slot's visesa leaves the others exactly
    as they were. Dependence runs one way only.
    """
    xa, xb, y = world.sample(600)
    base = accuracy_per_slot(net, xa, xb, y)

    lg_b, _ = net.forward(xa, xb, ablate={'bimba': True})
    # Compare only the INPUT-DEPENDENT part: subtract each slot's own mean
    # logit, which removes its private bias. Whatever remains is content the
    # slot could only have got by reflecting the bimba. If dependence is real,
    # nothing remains, and every slot is left holding the same empty hand.
    centred = lg_b - lg_b.mean(axis=1, keepdims=True)
    spread = float(np.max(np.std(centred, axis=0)))
    acc_b = (lg_b.argmax(axis=2) == y[None, :]).mean(axis=1)

    lg_s, _ = net.forward(xa, xb, ablate={'slot': 2})
    acc_s = (lg_s.argmax(axis=2) == y[None, :]).mean(axis=1)
    others_intact = np.max(np.abs(acc_s[[0, 1, 3]] - base[[0, 1, 3]]))
    slot2_dropped = base[2] - acc_s[2]

    print(f"    baseline slot accuracy      : " + " ".join(f"{a:.3f}" for a in base))
    print(f"    bimba removed -> accuracy   : " + " ".join(f"{a:.3f}" for a in acc_b))
    print(f"    inter-slot logit spread now = {spread:.3e}   "
          f"{'PASS' if spread < 1e-9 else 'FAIL'}  (all reflections vanish together)")
    print(f"    slot 2 removed -> accuracy  : " + " ".join(f"{a:.3f}" for a in acc_s))
    print(f"    other slots unchanged by    = {others_intact:.3e}   "
          f"{'PASS' if others_intact < 1e-12 else 'FAIL'}")
    print(f"    slot 2 lost                 = {slot2_dropped:.3f}")
    return spread < 1e-9 and others_intact < 1e-12


def test_taratamya(net, world):
    """
    Gradation survives training. Accuracy tracks intrinsic capacity, and the
    slots' inner natures stay mutually distinct rather than merging into one.
    """
    xa, xb, y = world.sample(2500)
    acc = accuracy_per_slot(net, xa, xb, y)
    order_ok = all(acc[i] >= acc[i + 1] - 0.02 for i in range(K_SLOTS - 1))
    gn = net.p['g'] / (np.linalg.norm(net.p['g'], axis=1, keepdims=True) + EPS)
    Gram = np.abs(gn @ gn.T)
    off = Gram[~np.eye(K_SLOTS, dtype=bool)]
    print(f"    capacity    : " + " ".join(f"{c:.2f}" for c in CAPACITY))
    print(f"    accuracy    : " + " ".join(f"{a:.3f}" for a in acc))
    print(f"    monotone in capacity  {'PASS' if order_ok else 'FAIL'}")
    print(f"    max |cos| between slot natures = {off.max():.3f}   "
          f"{'PASS' if off.max() < 0.6 else 'FAIL'}  (no collapse into one)")
    disagree = 0.0
    logits, _ = net.forward(xa, xb)
    pr = logits.argmax(axis=2)
    for i in range(K_SLOTS):
        for j in range(i + 1, K_SLOTS):
            disagree = max(disagree, (pr[i] != pr[j]).mean())
    print(f"    max pairwise disagreement rate = {disagree:.3f}  "
          f"(the slots remain genuinely different knowers)")
    return order_ok and off.max() < 0.6


def test_dosa_gate(net, world):
    """
    Validity by default; demotion only on detected defect. Corrupt the
    testimony channel and watch the untrained witness discount it, without
    ever having been shown a corrupted example.
    """
    xa, xb, y = world.sample(1200)
    _, c0 = net.forward(xa, xb)
    a0 = c0['alpha'].mean(axis=0)
    lg1, c1 = net.forward(xa, xb, corrupt_agama=1.0)
    a1 = c1['alpha'].mean(axis=0)
    acc0 = accuracy_per_slot(net, xa, xb, y)[0]
    acc1 = (lg1.argmax(axis=2)[0] == y).mean()
    print(f"    clean   alpha [pratyaksa, anumana, agama] = "
          f"[{a0[0]:.3f}, {a0[1]:.3f}, {a0[2]:.3f}]")
    print(f"    corrupt alpha [pratyaksa, anumana, agama] = "
          f"[{a1[0]:.3f}, {a1[1]:.3f}, {a1[2]:.3f}]")
    dropped = a1[2] < a0[2]
    print(f"    testimony demoted on defect    {'PASS' if dropped else 'FAIL'}")
    print(f"    accuracy {acc0:.3f} -> {acc1:.3f} under corrupted testimony "
          f"(retained {100*acc1/max(acc0,1e-9):.1f}%)")
    return dropped


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print(" BHEDA-SVARUPA NETWORK — Madhva (c.1238–1317), chapter 276")
    print(" difference is not computed from things; things are read off differences")
    print("=" * 74)

    world = PancabhedaWorld(seed=1238)
    net = BhedaSvarupaNet(seed=1317)

    print("\n[1] STRUCTURAL PROPERTIES OF THE BHEDA CORE (before any training)")
    r1 = test_antisymmetry(net, world)

    print("\n[2] THE SAKSIN IS EXACT (it holds no weights and cannot be trained)")
    r2 = test_saksin_exactness(net, world)

    print("\n[3] THE VISESA HALTS ITS OWN REGRESS")
    r3 = test_visesa(net)

    print("\n[4] THE IMPLICIT GRADIENT IS THE TRUE INFINITE-DEPTH GRADIENT")
    r4 = test_implicit_vs_unrolled()

    print("\n[5] GRADIENT CHECK ON ALL PARAMETER GROUPS")
    r5 = gradient_check(net, world)

    print("\n[6] TRAINING")
    hist, (xv_a, xv_b, yv) = train(net, world, steps=1400, batch=192)

    print("\n[7] TARATAMYA — GRADATION SURVIVES, THE SLOTS DO NOT MERGE")
    r7 = test_taratamya(net, world)

    print("\n[8] SVATANTRA / PARATANTRA — DEPENDENCE RUNS ONE WAY")
    r8 = test_dependence_asymmetry(net, world)

    print("\n[9] SVATAH-PRAMANYA WITH PARATAH-APRAMANYA — THE DOSA GATE")
    r9 = test_dosa_gate(net, world)

    print("\n[10] CONTROL: A POINT-FIRST ENCODER ON THE SAME DATA")
    ov, kd = sumpool_baseline(world)
    xa, xb, y = world.sample(2500)
    bsn = accuracy_per_slot(net, xa, xb, y)[0]
    kind_mask = y != 5
    lg, _ = net.forward(xa, xb)
    bsn_kind = (lg.argmax(axis=2)[0][kind_mask] == y[kind_mask]).mean()
    print(f"    sum-pooled logistic baseline : overall {ov:.3f} | "
          f"five-bheda classes only {kd:.3f}  (chance = 0.200)")
    print(f"    bheda-svarupa network        : overall {bsn:.3f} | "
          f"five-bheda classes only {bsn_kind:.3f}")
    r10 = bsn_kind > kd + 0.25
    print(f"    difference-first beats point-first by "
          f"{100*(bsn_kind-kd):.1f} points   {'PASS' if r10 else 'FAIL'}")

    print("\n" + "=" * 74)
    results = dict(antisymmetry=r1, saksin=r2, visesa=r3, implicit_grad=r4,
                   gradcheck=r5, taratamya=r7, dependence=r8, dosa_gate=r9,
                   difference_first=r10)
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL'}")
    print(f"\n  ALL TESTS {'PASSED' if all(results.values()) else 'DID NOT PASS'}")
    print("=" * 74)
    return all(results.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

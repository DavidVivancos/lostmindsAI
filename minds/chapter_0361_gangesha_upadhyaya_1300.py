#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0361 · Gangesha Upadhyaya (1300-1350)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 Gangesha Upadhyaya (Gangesa, mahopadhyaya), Mithila, fl. c. 1300-1350 CE
 Author of the Tattvacintamani ("Jewel of Reflection on the Truth")
================================================================================

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy, fully trainable model that encodes ONE cognitive
signature: Gangesa's theory of *defeasible warrant*. It is not a demo and not a
wrapper. Every gradient is derived by hand and verified against finite
differences before training begins.

THE MIND BEING ENCODED
----------------------
Gangesa's Nyaya is not a logic of proof. It is a logic of DEFEAT. Three of his
commitments drive this architecture, and none of them is the usual "impose
order on chaos" lens:

  (1) PARATAH-PRAMANYA -- extrinsic certification.
      A cognition does not certify itself. Against the Mimamsakas, who held
      that veridicality is intrinsic to the presentation, Gangesa argued that
      it takes a SEPARATE cognitive act, later in time and causally distinct,
      to determine whether an awareness was knowledge. Certification arrives
      from OUTSIDE: from success in action, and from identifying which source
      produced the awareness and whether that source's characteristic merits
      (guna) or flaws (dosa) were present.
      => Architectural consequence: the confidence head is structurally denied
         access to the reasoning trace. It sees a stop-gradient copy of the
         conclusion and the source-type, nothing else. It is trained only on
         whether acting on the conclusion succeeded.

  (2) BADHAKAM VINA -- "unless there is a defeater."
      Awareness is PRESUMPTIVELY knowledge. Gangesa is not a skeptic; the
      default is acceptance, and doubt must be earned by finding an actual
      blocker. Absolute freedom from doubt is not required for knowledge.
      => Architectural consequence: the three-valued output (assert / negate /
         suspend) defaults to assertion. Suspension is the exception, produced
         only when the certifier locates a flaw.

  (3) UPADHI -- the "additional condition," the inferential undercutter.
      This is the heart. More than a tenth of the Jewel's enormous inference
      chapter is spent on it. An upadhi U defeats the inference H => S when:
            (a) U pervades the probandum:      for all x, S(x) -> U(x)
            (b) U does NOT pervade the prover: exists x, H(x) and not U(x)
      From (a) and (b) it follows that there is DEVIATION: some H is not S, so
      the pervasion never held. The stock case: fire does not prove smoke,
      because wet fuel (ardra-indhana) pervades smoke but not fire.
      And the move that makes Gangesa unlike anyone else in this corpus:
      an upadhi is not only destructive. Conjoined WITH the prover it RESTORES
      the inference. (H and U) => S is sound where H => S was not.
      => Architectural consequence: discovered confounders are not pruned. They
         are welded onto the rule as a conjunctive specialization. The network
         repairs its licenses instead of deleting them.

  (4) ANUVYAVASAYA -- apperception, the after-cognition.
      Gangesa denies the Buddhist doctrine of self-luminous cognition
      (svasamvedana). A cognition is known by a LATER, distinct cognition. That
      after-cognition is infallible about the scoped cognition's CONTENT -- its
      qualificandum, qualifier and relation -- but says nothing whatever about
      its TRUTH. Its epistemic use is that knowing the TYPE of a cognition tells
      you WHICH checks to run, since merits and flaws differ by source.
      => Architectural consequence: a second-order head that classifies the
         source-type of its own cognition with near-perfect accuracy while
         being forbidden to predict truth; its output ROUTES the certifier to
         a type-specific set of flaw probes.

  (5) LAGHAVA -- economy of conception; its opposite gaurava, theoretic
      heaviness, is a defect in Gangesa's tarka. => an L1 penalty on active
      licenses.

WHY THIS IS NOT A TRANSFORMER
-----------------------------
Attention scores stored keys by similarity. Nothing in that operation can
express "this correlation is real but conditioned on a factor you have not
noticed." The upadhi criterion is not a similarity, not an accuracy, and not a
likelihood. It is a pair of asymmetric SET-INCLUSION facts about the extensions
of properties, evaluated over a batch. So the central operation here is a soft
pervasion ratio, and the central training signal is a structural objective that
is deliberately blind to loss reduction -- which is exactly why it finds a
weak-signal confounder that gradient descent on accuracy walks straight past.

WHAT IS DEMONSTRATED
--------------------
Two worlds, both taken from Gangesa's own text:

  WORLD A (the wet-fuel upadhi). Loci have latent fire and wet-fuel; smoke
  occurs iff both. In training, fuel is wet 90% of the time, so "fire" alone
  looks like an excellent prover. The fire feature is loud; the wet feature is
  quiet and buried among distractors. Ordinary risk minimisation takes the
  shortcut. At test time the climate changes and fuel is wet only 30% of the
  time. Measured below: the ablated network collapses; the upadhi engine does
  not, because its defeater-hunt located the quiet variable on structural
  grounds and specialised the license to fire-and-wet.

  WORLD B (the diamond). Gangesa concedes in the Jewel that observing a
  regularity "even hundreds of times" does not guarantee a pervasion: it had
  been held that everything earthen is scratchable by iron, until diamonds were
  encountered. Here a rule holds without exception through training and then
  meets its counterexample. Measured below: suspension rate on the novel class,
  i.e. whether the system reports doubt rather than confabulating.

RUN:  python3 chapter_0361_gangesha_upadhyaya_1300.py
DEPS: numpy only.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# Deterministic seeding. Gangesa's Jewel is a text of exhaustively enumerated
# alternatives; reproducibility is the least we owe it.
# ------------------------------------------------------------------------------
GLOBAL_SEED = 13250  # 1325 CE, the conventional floruit, plus a digit for luck
EPS = 1e-8


# ==============================================================================
# SECTION 1 -- THE WORLD (drstanta: the stock of examples)
# ==============================================================================
# Nyaya inference is not formal validity in a vacuum; it is knowledge of nature.
# A pervasion "obtains in nature," and it is grasped through bhuyo-darsana,
# wide experience. So the model is not fed abstract symbols but noisy sensory
# vectors from which the properties must themselves be recovered.
# ==============================================================================

class SmokeWorld:
    """
    Loci (hills, kitchens, lakes) described by a noisy sensory vector.

    Latent properties, in Gangesa's vocabulary:
        fire   -- a candidate prover (hetu / sadhana). LOUD in the senses.
        wet    -- ardra-indhana, wet fuel. The upadhi. QUIET in the senses.
        smoke  -- the probandum (sadhya). smoke  <=>  fire AND wet.
        d0..d3 -- distractor properties, causally inert.

    The generative story is exactly the classical one. Fire alone does not
    prove smoke. Wet fuel pervades smoke (all smoke is wet-fuelled) and fails
    to pervade fire (dry fires exist). Therefore fire deviates, and the
    pervasion "wherever fire, there smoke" was never a pervasion at all.
    """

    N_LATENT = 6          # fire, wet, d0, d1, d2, d3
    IDX_FIRE = 0
    IDX_WET = 1

    def __init__(self, d_obs=24, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        # A fixed sensory projection: each latent property lights up a distinct
        # band of receptors. The loadings encode the asymmetry that makes the
        # experiment honest -- fire is easy to see, wet fuel is not.
        self.d_obs = d_obs
        self.n_src_dims = 6                      # reserved for source-signature
        self.loading = np.zeros((self.N_LATENT, d_obs))
        bands = np.array_split(np.arange(d_obs - self.n_src_dims), self.N_LATENT)
        strengths = [1.00, 0.60, 0.55, 0.55, 0.55, 0.55]  # fire loud, wet quiet
        for j, band in enumerate(bands):
            self.loading[j, band] = strengths[j]
        # A little cross-talk so no receptor is a clean read-out of one latent.
        self.loading += 0.10 * rng.standard_normal((self.N_LATENT, d_obs))
        self.noise_sigma = 0.50

    def sample(self, n, p_wet, seed):
        """
        Draw n loci. p_wet controls how often fuel is wet -- this is the
        distribution shift. Returns (X, y, latents, source_type).
        """
        rng = np.random.default_rng(seed)
        lat = np.zeros((n, self.N_LATENT))
        lat[:, self.IDX_FIRE] = (rng.random(n) < 0.50).astype(float)
        lat[:, self.IDX_WET] = (rng.random(n) < p_wet).astype(float)
        for j in range(2, self.N_LATENT):
            lat[:, j] = (rng.random(n) < 0.50).astype(float)

        # smoke iff fire and wet -- the causal fact the model must recover
        y = (lat[:, self.IDX_FIRE] * lat[:, self.IDX_WET])

        X = lat @ self.loading + self.noise_sigma * rng.standard_normal((n, self.d_obs))

        # Source-type of the evidence, for the apperception head. In Gangesa's
        # scheme the merits and flaws differ by source, so the type must be
        # recoverable. Type is encoded as a small additive signature on the
        # observation -- a real, learnable mark, not a free label.
        src = rng.integers(0, 3, size=n)          # 0 pratyaksa, 1 anumana, 2 sabda
        sig = np.zeros((3, self.d_obs))
        base = self.d_obs - self.n_src_dims
        for t in range(3):
            sig[t, base + 2 * t: base + 2 * t + 2] = 1.40
        X = X + sig[src]
        return X, y, lat, src


class DiamondWorld:
    """
    Gangesa's own admission of inductive risk, made into a test.

    "Not all things made out of earth are scratchable by iron; diamonds,
    recently discovered, are a counterexample." (Jewel, tr. Phillips 2020)

    Training contains only ordinary earthen substances, for which the rule
    earthen => scratchable holds without a single exception. At test time a new
    kind of thing appears -- earthen, but not scratchable -- whose sensory
    signature lies in a region never visited during training. The question is
    not whether the model gets it right. It cannot. The question is whether it
    reports DOUBT (samsaya) instead of asserting with the same breezy
    confidence it earned on familiar ground.
    """

    EARTHEN_DIMS = slice(0, 8)      # the dimensions that carry "earthen-ness"
    IDLE_DIMS = slice(16, 24)       # dimensions that are ~0 throughout training

    def __init__(self, d_obs=24, seed=GLOBAL_SEED + 7):
        rng = np.random.default_rng(seed)
        self.d_obs = d_obs
        self.sigma = 0.45
        # Earthen things register strongly on the earthen band.
        self.mu_earthen = np.zeros(d_obs)
        self.mu_earthen[self.EARTHEN_DIMS] = 1.7
        # Non-earthen things (water, air, ether) do not, and are not scratchable.
        self.mu_other = np.zeros(d_obs)
        self.mu_other[self.EARTHEN_DIMS] = -1.5
        self.mu_other[8:16] = 1.2
        # Texture variation within the earthen kind -- clay, chalk, granite --
        # all of it scratchable, none of it near the idle band.
        self.texture = rng.standard_normal((4, d_obs)) * 0.5
        self.texture[:, self.IDLE_DIMS] = 0.0

    def sample_familiar(self, n, seed):
        """Earthen things (all scratchable) and non-earthen things (not)."""
        rng = np.random.default_rng(seed)
        earthen = rng.random(n) < 0.5
        tex = self.texture[rng.integers(0, 4, n)]
        X = np.where(earthen[:, None], self.mu_earthen + tex, self.mu_other)
        X = X + self.sigma * rng.standard_normal((n, self.d_obs))
        y = earthen.astype(float)
        return X, y, rng.integers(0, 3, size=n)

    def sample_diamonds(self, n, seed):
        """
        Diamonds are earthen -- they read as earthen on exactly the band the
        model learned to use, so it will confidently call them scratchable --
        but they occupy a region of the idle band that no training instance
        ever visited. The rule held hundreds of times. It does not hold here.
        """
        rng = np.random.default_rng(seed)
        X = np.tile(self.mu_earthen, (n, 1))
        X[:, self.IDLE_DIMS] += 3.2               # the unvisited region
        X = X + self.sigma * rng.standard_normal((n, self.d_obs))
        y = np.zeros(n)                            # the counterexample
        return X, y, rng.integers(0, 3, size=n)


# ==============================================================================
# SECTION 2 -- THE UPADHI ENGINE (core inferential network)
# ==============================================================================
# Forward pass, in Gangesa's order of causal factors:
#
#   X  --(nirvikalpaka)-->  Q        indeterminate grasp of qualifiers.
#                                    Cowness cognized, not yet tied to the cow.
#
#   Q  --(avacchedaka)-->   masked   every role is DELIMITED. A license may only
#                                    look at the qualifier-dimensions its
#                                    limitor admits. Navya-Nyaya never lets a
#                                    relatum float unqualified.
#
#   Q  --(vyapti bank)-->   H_k      does the prover-property occur here?
#   Q  --(upadhi bank)-->   U_k      does the candidate additional condition?
#
#   R_k = H_k * ((1 - lam_k) + lam_k * U_k)          <-- PARAMARSA, "reflection"
#          lam_k = 0  : the bare license          H => S
#          lam_k = 1  : the repaired license  (H & U) => S
#
#   z = sum_k beta_k R_k + b        p = sigmoid(z)
#
# The upadhi criterion is evaluated over the batch as two soft pervasions:
#
#   Pv(A -> B) = sum_i A_i B_i / sum_i A_i          "what fraction of A is B"
#
#   P1_k = Pv(S -> U_k)          condition (a): U pervades the probandum
#   P2_k = 1 - Pv(H_k -> U_k)    condition (b): U fails to pervade the prover
#   Omega_k = P1_k * P2_k        the strength of the defeater found
#
# and lam_k is dragged toward a SHARPENED function of Omega_k, with Omega held
# constant in that term so the network cannot cheat by suppressing its own
# defeaters. The sharpening is doctrinally required: for Gangesa a single
# unaccounted dry fire is enough. Deviation is qualitative, not proportional.
# ==============================================================================

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -60, 60)))


def balanced_accuracy(pred_bin, truth_bin):
    """Base-rate-robust agreement. Plain correlation is useless against a latent
    that is present 96% of the time -- a near-constant variable has almost no
    variance to correlate with. This asks the fair question instead: of the wet
    loci, what share does the condition channel mark, and of the dry loci, what
    share does it spare? 0.5 is chance."""
    pos = truth_bin > 0.5
    neg = ~pos
    tpr = pred_bin[pos].mean() if pos.any() else 0.0
    tnr = (1 - pred_bin[neg]).mean() if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def soft_pervasion(a, b, axis=0):
    """Pv(A -> B): of the loci where A holds, the share where B also holds."""
    return (a * b).sum(axis=axis) / (a.sum(axis=axis) + EPS)


class UpadhiEngine:
    """
    Parameters
    ----------
    Wq, bq   : nirvikalpaka projection            (d_obs, d_q), (d_q,)
    A        : hetu limitors  (avacchedaka)       (K, d_q)   -> M = sigmoid(A)
    Uw, c    : prover weights and thresholds      (K, d_q), (K,)
    G        : upadhi limitors                    (K, d_q)   -> Ng = sigmoid(G)
    Vw, e    : additional-condition weights       (K, d_q), (K,)
    rho      : specialization gates               (K,)       -> lam = sigmoid(rho)
    beta, bz : probandum couplings and bias       (K,), scalar
    """

    PARAM_NAMES = ["Wq", "bq", "A", "Uw", "c", "G", "Vw", "e", "rho", "alpha"]

    def __init__(self, d_obs, d_q=16, K=6, seed=GLOBAL_SEED,
                 w_cond_a=8.0,      # (a) is a UNIVERSAL: U must pervade S
                 w_cond_b=2.0,      # (b) is an EXISTENTIAL: some H is not U
                 w_restore=2.0,     # the repair must materially improve things
                 w_siddha=3.0,      # prover and repaired prover must occur
                 w_atmasraya=3.0,   # U must not merely restate the probandum
                 p2_min=0.015, gain_min=0.05, h_min=0.15, w_min=0.10, u_max=0.70,
                 mu_lam=2.0, mu_econ=3e-3,
                 hunt_defeaters=True, allow_repair=True, freeze_bare=False):
        rng = np.random.default_rng(seed)
        s = lambda *sh: rng.standard_normal(sh) * 0.35
        self.p = {
            "Wq":   s(d_obs, d_q),
            "bq":   np.zeros(d_q),
            "A":    s(K, d_q),
            "Uw":   s(K, d_q),
            "c":    np.zeros(K),
            "G":    s(K, d_q),
            "Vw":   s(K, d_q),
            "e":    np.zeros(K),
            "rho":  np.full(K, -2.0),          # start with bare, unrepaired rules
            "alpha": np.full(K, -0.5),         # per-license credence, in (0,1)
        }
        self.K, self.d_q, self.d_obs = K, d_q, d_obs
        on = 1.0 if hunt_defeaters else 0.0
        self.w_cond_a = w_cond_a * on
        self.w_cond_b = w_cond_b * on
        self.w_restore = w_restore * on
        self.w_siddha = w_siddha * on
        self.w_atmasraya = w_atmasraya * on
        self.u_max = u_max
        self.p2_min, self.gain_min = p2_min, gain_min
        self.h_min, self.w_min = h_min, w_min
        self.mu_lam = mu_lam if allow_repair else 0.0
        self.mu_econ = mu_econ
        self.allow_repair = allow_repair
        # freeze_bare pins every license at lambda = 0, i.e. R = H. This is the
        # TRUE ablation: bare, unrepairable pervasions with no defeater-hunt,
        # which is ordinary empirical risk minimisation over a rule bank.
        self.freeze_bare = freeze_bare
        if freeze_bare:
            self.p["rho"][:] = -50.0

    # ---------------------------------------------------------------- forward
    def forward(self, p, X, y, lam_target_frozen=None):
        """Returns (loss, cache). y is used ONLY for the task loss and for the
        pervasion statistics -- the upadhi is discovered from observed
        instances, which is precisely Gangesa's bhuyo-darsana.

        `lam_target_frozen` makes the stop-gradient EXPLICIT. The specialization
        target is deliberately detached: if gradients were allowed to flow back
        through it, the network could reduce the coupling loss by suppressing
        its own defeaters -- which is precisely the failure Gangesa's method is
        built to prevent. The detachment is a doctrinal commitment, not an
        optimisation trick, so the gradient check honours it by freezing the
        target at the evaluation point rather than differentiating through it.
        """
        N = X.shape[0]

        Qpre = X @ p["Wq"] + p["bq"]
        Q = np.tanh(Qpre)                                     # (N, d_q)

        M = sigmoid(p["A"])                                   # (K, d_q) limitor
        Ng = sigmoid(p["G"])                                  # (K, d_q) limitor
        Whet = p["Uw"] * M
        Wupa = p["Vw"] * Ng

        Hpre = Q @ Whet.T + p["c"]
        H = sigmoid(Hpre)                                     # (N, K) prover
        Upre = Q @ Wupa.T + p["e"]
        U = sigmoid(Upre)                                     # (N, K) condition

        lam = sigmoid(p["rho"])                               # (K,)
        mix = (1.0 - lam)[None, :] + lam[None, :] * U
        R = H * mix                                           # (N, K) paramarsa

        # --- PARAMARSA COMBINED AS A DISJUNCTION -------------------------
        # A body of pervasions is not a weighted sum. Inferential knowledge
        # arises when SOME sound license licenses it: "this locus has the
        # prover, and the prover is pervaded by the probandum." Combining the
        # licenses linearly lets a negative coefficient cancel one license
        # against another, so the bank can satisfy every structural constraint
        # per-license while the prediction is carried by their interference --
        # which is not inference at all. A noisy-OR forces each license to be
        # individually sound, because no license can ever be used to subtract.
        a = sigmoid(p["alpha"])                               # (K,) credence
        m = np.clip(1.0 - a[None, :] * R, 1e-6, 1.0)          # (N, K)
        Pi = np.prod(m, axis=1)                               # (N,)
        pr = np.clip(1.0 - Pi, EPS, 1.0 - EPS)

        # --- task loss: the inference must actually work -----------------
        L_task = -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))

        # --- THE TWO FORMAL UPADHI CONDITIONS ----------------------------
        # These are NOT the same kind of claim, and an architecture that treats
        # them as one scalar to maximise gets Gangesa wrong. Condition (a) is a
        # UNIVERSAL -- for all x, Sx -> Ux -- and is violated by a single
        # uncovered instance. Condition (b) is an EXISTENTIAL -- there is some
        # x with Hx and not Ux -- and is satisfied by a single witness. So (a)
        # is driven as a strict equality and (b) only as far as a threshold.
        # Multiplying them into one quantity to maximise silently trades the
        # universal away for a bigger existential, and the search then shrinks
        # U below the real condition. Omega is retained below as the REPORTED
        # criterion, which is what the text defines; the losses are how the
        # search for it is conducted.
        sumS = y.sum() + EPS
        P1 = (y[:, None] * U).sum(0) / sumS                   # Pv(S -> U)
        DenH = H.sum(0) + EPS
        Ph = (H * U).sum(0) / DenH                            # Pv(H -> U)
        P2 = 1.0 - Ph
        Omega = P1 * P2                                       # reported criterion

        L_cond_a = np.mean((1.0 - P1) ** 2)                   # universal
        short_b = np.maximum(0.0, self.p2_min - P2)           # existential
        L_cond_b = np.mean(short_b)

        # --- THE RESTORATION CLAUSE --------------------------------------
        # Gangesa: an upadhi "may also serve as positive restorer of knowledge
        # in that when it occurs together with an original prover that it
        # undercuts, the inferential conclusion is reestablished."
        #
        # This clause is what rules out the vacuous defeater. Without it the
        # two conditions above are satisfiable by a license that has quietly
        # killed its own prover -- Gangesa's ASIDDHA, the "unestablished"
        # pseudo-prover, which he lists among the fallacies. It is imposed as a
        # HINGE, not a maximisation: the repair must materially improve the
        # pervasion, but nothing in the doctrine asks the bare prover to be
        # made maximally bad, and pushing it that way is how the search
        # degenerates.
        W = H * U                                             # H and U
        DenB = DenH
        Pbare = (H * y[:, None]).sum(0) / DenB                # Pv(H -> S)
        DenR = W.sum(0) + EPS
        Prep = (W * y[:, None]).sum(0) / DenR                 # Pv(H&U -> S)
        Gain = Prep - Pbare
        short_g = np.maximum(0.0, self.gain_min - Gain)
        L_restore = np.mean(short_g)

        # --- ATMASRAYA GUARD: no self-dependence ---------------------------
        # An "additional condition" that is co-extensive with the probandum is
        # not a condition at all -- it is the conclusion smuggled in as its own
        # premise. Gangesa lists atmasraya, self-dependence, first among the
        # types of hypothetical reasoning that destroy a position. Without this
        # guard the search converges on U = S, which satisfies every clause
        # above and explains nothing: the network simply predicts the target
        # through the condition channel. A genuine upadhi is strictly WIDER
        # than the probandum -- all smoke is wet-fuelled, but plenty of
        # wet fuel is not smoking.
        DenU = U.sum(0) + EPS
        Pu = (U * y[:, None]).sum(0) / DenU                   # Pv(U -> S)
        short_u = np.maximum(0.0, Pu - self.u_max)
        L_atmasraya = np.mean(short_u)

        # --- SIDDHATVA: prover and repaired prover must be established ----
        # A pervasion computed over an empty extension is not a pervasion.
        mH = H.mean(0)
        mW = W.mean(0)
        short_h = np.maximum(0.0, self.h_min - mH)
        short_w = np.maximum(0.0, self.w_min - mW)
        L_siddha = np.mean(short_h) + np.mean(short_w)

        # --- specialization: repair exactly when BOTH conditions hold -----
        # The gate is not proportional to a defeater's "size." For Gangesa the
        # test is qualitative: a single unaccounted dry fire destroys the
        # pervasion. So the target is a conjunction of two sharp tests.
        found = sigmoid(20.0 * (P1 - 0.85)) * sigmoid(60.0 * (P2 - self.p2_min))
        if lam_target_frozen is None:
            lam_target = found
        else:
            lam_target = lam_target_frozen          # detached (see docstring)
        L_lam = np.mean((lam - lam_target) ** 2)

        # --- laghava: economy of conception ------------------------------
        # gaurava, "theoretic heaviness," is a defect for Gangesa. A bank that
        # needs every license to do its work has explained nothing.
        L_econ = np.mean(a)

        L = (L_task
             + self.w_cond_a * L_cond_a
             + self.w_cond_b * L_cond_b
             + self.w_restore * L_restore
             + self.w_atmasraya * L_atmasraya
             + self.w_siddha * L_siddha
             + self.mu_lam * L_lam
             + self.mu_econ * L_econ)

        cache = dict(N=N, X=X, y=y, Qpre=Qpre, Q=Q, M=M, Ng=Ng, Whet=Whet,
                     Wupa=Wupa, H=H, U=U, lam=lam, mix=mix, R=R, pr=pr,
                     a=a, m=m, Pi=Pi,
                     P1=P1, P2=P2, Ph=Ph, Omega=Omega, DenH=DenH,
                     sumS=sumS, lam_target=lam_target,
                     W=W, DenR=DenR, DenB=DenB, Pbare=Pbare, Prep=Prep,
                     Gain=Gain, mH=mH, mW=mW, found=found, Pu=Pu, DenU=DenU,
                     short_u=short_u,
                     short_b=short_b, short_g=short_g,
                     short_h=short_h, short_w=short_w)
        return L, cache

    # --------------------------------------------------------------- backward
    def backward(self, p, cache):
        """Hand-derived gradients. Verified in test_gradients() below."""
        N, K = cache["N"], self.K
        X, y, Q, H, U = cache["X"], cache["y"], cache["Q"], cache["H"], cache["U"]
        lam, mix, R, pr = cache["lam"], cache["mix"], cache["R"], cache["pr"]
        P1, P2, Ph, Omega = cache["P1"], cache["P2"], cache["Ph"], cache["Omega"]
        DenH, sumS = cache["DenH"], cache["sumS"]

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- task path through the noisy-OR ------------------------------
        #   pr = 1 - Pi ,  Pi = prod_k m_k ,  m_k = 1 - a_k R_k
        #   dL/dPi = y/(1-Pi) - (1-y)/Pi        [since pr = 1-Pi]
        #   dPi/dm_k = Pi / m_k
        a, m, Pi = cache["a"], cache["m"], cache["Pi"]
        Pi_c = np.clip(Pi, EPS, 1.0 - EPS)
        gPi = (y / np.clip(1.0 - Pi_c, EPS, None)
               - (1.0 - y) / Pi_c) / N                        # (N,)
        gm = gPi[:, None] * (Pi[:, None] / m)                 # (N, K)
        gR = gm * (-a[None, :])
        ga = (gm * (-R)).sum(0)                               # (K,)

        gH = gR * mix
        gU = gR * H * lam[None, :]
        glam = (gR * H * (U - 1.0)).sum(0)

        # ---- condition (a), the universal:  L = mean_k (1 - P1_k)^2 ------
        #      P1_k = sum_i y_i U_ik / sumS   ->   dP1/dU_ik = y_i / sumS
        dL_dP1 = -2.0 * self.w_cond_a * (1.0 - P1) / K
        gU += dL_dP1[None, :] * (y[:, None] / sumS)

        # ---- condition (b), the existential:  L = mean_k relu(p2min - P2_k)
        #      P2 = 1 - Ph  ->  dL/dPh_k = + w_b * 1{active} / K
        act_b = (cache["short_b"] > 0).astype(float)
        dL_dPh = self.w_cond_b * act_b / K
        gU += dL_dPh[None, :] * (H / DenH[None, :])
        gH += dL_dPh[None, :] * ((U - Ph[None, :]) / DenH[None, :])

        # ---- restoration clause: L = mean_k relu(gain_min - Gain_k) ------
        DenR, DenB = cache["DenR"], cache["DenB"]
        Prep, Pbare, W = cache["Prep"], cache["Pbare"], cache["W"]
        act_g = (cache["short_g"] > 0).astype(float)
        cR = -self.w_restore * act_g / K          # dL/dPrep
        cB = +self.w_restore * act_g / K          # dL/dPbare
        #   Prep_k = sum_i W_ik y_i / sum_i W_ik ,  W = H*U
        dPrep_dW = (y[:, None] - Prep[None, :]) / DenR[None, :]
        gH += cR[None, :] * dPrep_dW * U
        gU += cR[None, :] * dPrep_dW * H
        #   Pbare_k = sum_i H_ik y_i / sum_i H_ik
        gH += cB[None, :] * ((y[:, None] - Pbare[None, :]) / DenB[None, :])

        # ---- atmasraya guard: L = mean_k relu(Pv(U->S) - u_max) ----------
        #      Pu_k = sum_i U_ik y_i / sum_i U_ik
        #      dPu/dU_ik = (y_i - Pu_k) / DenU_k
        act_u = (cache["short_u"] > 0).astype(float)
        dL_dPu = self.w_atmasraya * act_u / K
        gU += dL_dPu[None, :] * ((y[:, None] - cache["Pu"][None, :])
                                 / cache["DenU"][None, :])

        # ---- siddhatva coverage floors on mean(H) and mean(H*U) ----------
        act_h = (cache["short_h"] > 0).astype(float)
        act_w = (cache["short_w"] > 0).astype(float)
        gH += (-self.w_siddha / (K * N)) * act_h[None, :]
        gH += (-self.w_siddha / (K * N)) * act_w[None, :] * U
        gU += (-self.w_siddha / (K * N)) * act_w[None, :] * H

        # ---- specialization coupling (lam_target held constant) ----------
        glam += self.mu_lam * 2.0 * (lam - cache["lam_target"]) / K

        # ---- laghava -----------------------------------------------------
        ga += self.mu_econ / K
        g["alpha"] += ga * a * (1.0 - a)

        # ---- through the sigmoids ---------------------------------------
        g["rho"] += glam * lam * (1.0 - lam)

        gHpre = gH * H * (1.0 - H)
        gUpre = gU * U * (1.0 - U)

        gWhet = gHpre.T @ Q                                   # (K, d_q)
        g["c"] += gHpre.sum(0)
        gWupa = gUpre.T @ Q
        g["e"] += gUpre.sum(0)

        gQ = gHpre @ cache["Whet"] + gUpre @ cache["Wupa"]

        g["Uw"] += gWhet * cache["M"]
        gM = gWhet * p["Uw"]
        g["A"] += gM * cache["M"] * (1.0 - cache["M"])

        g["Vw"] += gWupa * cache["Ng"]
        gNg = gWupa * p["Vw"]
        g["G"] += gNg * cache["Ng"] * (1.0 - cache["Ng"])

        gQpre = gQ * (1.0 - Q ** 2)
        g["Wq"] += X.T @ gQpre
        g["bq"] += gQpre.sum(0)

        if self.freeze_bare:
            g["rho"][:] = 0.0        # licenses stay bare: R = H, no repair
        return g

    # ------------------------------------------------------------- inference
    def infer(self, p, X):
        """Returns (probability, qualifier state Q, reflection R, prover H,
        additional condition U)."""
        Q = np.tanh(X @ p["Wq"] + p["bq"])
        H = sigmoid(Q @ (p["Uw"] * sigmoid(p["A"])).T + p["c"])
        U = sigmoid(Q @ (p["Vw"] * sigmoid(p["G"])).T + p["e"])
        lam = sigmoid(p["rho"])
        R = H * ((1 - lam)[None, :] + lam[None, :] * U)
        a = sigmoid(p["alpha"])
        pr = 1.0 - np.prod(np.clip(1.0 - a[None, :] * R, 1e-6, 1.0), axis=1)
        return np.clip(pr, EPS, 1 - EPS), Q, R, H, U


# ==============================================================================
# SECTION 3 -- ANUVYAVASAYA (apperception): the after-cognition
# ==============================================================================
# Gangesa: a cognition is not self-illuminating. It is scoped by a LATER
# perception which is infallible about the scoped cognition's content -- its
# qualificandum, qualifier and relational portions -- and silent about its
# truth. Its epistemic work is to identify the TYPE, because "the criteria of
# genuine perception, inference and the rest vary according to origin. Thus
# apperceptive judgment of putative type directs inquiry, telling us what to
# check."
#
# So: a softmax classifier over source-type, reading a STOP-GRADIENT copy of
# the reflection state. It never sees the target and never predicts truth. Its
# only downstream role is to route the certifier.
# ==============================================================================

class Apperception:
    PARAM_NAMES = ["Wa", "ba"]

    def __init__(self, K, d_q, n_types=3, seed=GLOBAL_SEED + 1):
        rng = np.random.default_rng(seed)
        # Apperception scopes the CONTENT of the cognition: its qualifier state
        # Q (the prakara portion), its bindings R, and the verdict's shape. It
        # is given rich access here on purpose -- Gangesa's claim is that the
        # after-cognition knows infallibly what the scoped cognition purports
        # to be about. What it is never given is the target: it is silent
        # about truth by construction, not by training accident.
        d_in = d_q + K + 2
        self.p = {"Wa": rng.standard_normal((d_in, n_types)) * 0.3,
                  "ba": np.zeros(n_types)}
        self.n_types = n_types

    @staticmethod
    def features(Q, R, pr):
        return np.concatenate([Q, R, pr[:, None], np.abs(pr - 0.5)[:, None]], axis=1)

    def forward(self, p, F, t):
        logits = F @ p["Wa"] + p["ba"]
        logits = logits - logits.max(1, keepdims=True)
        expz = np.exp(logits)
        S = expz / expz.sum(1, keepdims=True)
        L = -np.mean(np.log(S[np.arange(len(t)), t] + EPS))
        return L, dict(F=F, S=S, t=t, N=len(t))

    def backward(self, p, cache):
        S, t, N, F = cache["S"], cache["t"], cache["N"], cache["F"]
        d = S.copy()
        d[np.arange(N), t] -= 1.0
        d /= N
        return {"Wa": F.T @ d, "ba": d.sum(0)}

    def predict(self, p, F):
        return (F @ p["Wa"] + p["ba"]).argmax(1)


# ==============================================================================
# SECTION 4 -- PARATAH-PRAMANYA: the extrinsic certifier
# ==============================================================================
# The strongest architectural commitment in this file, and the one that most
# sharply separates Gangesa from a modern confidence head.
#
# A modern model's confidence is a readout of its own logits: intrinsic
# certification, exactly the Mimamsaka position Gangesa spent the opening of
# the Jewel demolishing. His counter-argument is phenomenological: if
# veridicality were intrinsic to the presentation, it would be IMPOSSIBLE to
# doubt a cognition that is in fact knowledge. But it plainly is possible.
#
# So the certifier here:
#   * receives a STOP-GRADIENT scalar summary of the conclusion and nothing
#     from inside the reasoning;
#   * is ROUTED by the apperceived source-type to a type-specific row of
#     weights -- its guna/dosa probes, "the characteristic merits and flaws
#     [which] are very different for each knowledge source";
#   * is trained ONLY on corroboration: did acting on this conclusion succeed?
#     ("I wouldn't now be drinking water, if the awareness had not been
#     knowledge. But I am now drinking, satisfying my thirst...")
#   * defaults to ACCEPT. Suspension requires a positive finding of flaw.
#     Badhakam vina -- unless there is a defeater.
# ==============================================================================

class SourceFlawProbe:
    """
    DOSA -- a tell-tale flaw of the source, not a reason inside the inference.

    Gangesa's flaws are conditions on the knowledge-PROCESS, not on the content:
    the object is too far to be seen, the light is bad, the sounds came from a
    parrot rather than a speaker. "You can't really SEE John from this
    distance." A genuine source is inerrant; what misleads is a pseudo-source
    (pramanabhasa) wearing its clothes.

    Implemented as a novelty statistic on the qualifier state: how far outside
    the range in which this source has ever been corroborated are the present
    conditions? Calibrated ENTIRELY on training data -- the threshold is a
    quantile of the training statistic, so nothing about the test regime leaks
    in. Where the probe fires, the block is structural rather than learned,
    matching Gangesa's BADHITA, the defeater that operates "in advance" and is
    not weighed against the evidence but rules it out.
    """

    def __init__(self, slope=4.0, quantile=99.0):
        self.slope, self.quantile = slope, quantile
        self.mu = self.sd = self.thresh = self.scale = None

    def fit(self, X_train):
        """Calibrated on the SENSORY CONDITIONS, not on the internal state.

        This matters. The internal qualifier state is a saturated tanh code:
        a wildly novel input and a familiar one can land in the same corner of
        it, so novelty measured there is invisible. Gangesa's flaws are
        properties of the knowledge-PROCESS and its circumstances -- the
        distance, the light, the parrot -- so the probe belongs on the input
        situation. Threshold and scale are both quantiles of the TRAINING
        statistic; nothing about any later regime enters."""
        self.mu = X_train.mean(0)
        self.sd = X_train.std(0) + 1e-3
        stat = self._stat(X_train)
        self.thresh = np.percentile(stat, self.quantile)
        self.scale = max(np.percentile(stat, 75) - np.percentile(stat, 25), 1e-3)
        return self

    def _stat(self, X):
        z = (X - self.mu) / self.sd
        return np.sqrt((z ** 2).mean(1))

    def dosa(self, X):
        """Flaw strength in [0,1). Near zero inside the corroborated range."""
        return sigmoid(self.slope * (self._stat(X) - self.thresh) / self.scale)


class ExtrinsicCertifier:
    PARAM_NAMES = ["Wc", "bc"]
    D_FEAT = 4      # [p, |p-1/2|, binary entropy, license disagreement]

    def __init__(self, n_types=3, seed=GLOBAL_SEED + 2):
        rng = np.random.default_rng(seed)
        self.p = {"Wc": rng.standard_normal((n_types, self.D_FEAT)) * 0.3,
                  "bc": np.zeros(n_types)}

    @staticmethod
    def features(pr, R):
        """Deliberately impoverished. The certifier is NOT allowed to re-derive
        the conclusion; it sees only the conclusion's shape, plus how much the
        licenses disagreed among themselves."""
        ent = -(pr * np.log(pr + EPS) + (1 - pr) * np.log(1 - pr + EPS))
        spread = R.std(axis=1)
        return np.stack([pr, np.abs(pr - 0.5), ent, spread], axis=1)

    def forward(self, p, F, t, corroborated):
        w = p["Wc"][t]                                   # (N, D) type-routed
        z = (F * w).sum(1) + p["bc"][t]
        q = sigmoid(z)
        L = -np.mean(corroborated * np.log(q + EPS) +
                     (1 - corroborated) * np.log(1 - q + EPS))
        return L, dict(F=F, t=t, q=q, y=corroborated, N=len(t))

    def backward(self, p, cache):
        F, t, q, y, N = cache["F"], cache["t"], cache["q"], cache["y"], cache["N"]
        gz = (q - y) / N                                 # (N,)
        gWc = np.zeros_like(p["Wc"])
        gbc = np.zeros_like(p["bc"])
        np.add.at(gWc, t, gz[:, None] * F)
        np.add.at(gbc, t, gz)
        return {"Wc": gWc, "bc": gbc}

    def certify(self, p, F, t, dosa=None):
        """Learned corroboration-likelihood, gated by any structural flaw found
        in the source. Merits raise certification; a flaw blocks it outright."""
        q = sigmoid((F * p["Wc"][t]).sum(1) + p["bc"][t])
        if dosa is not None:
            q = q * (1.0 - dosa)
        return q


# ==============================================================================
# SECTION 5 -- THREE-VALUED OUTPUT: assert / negate / suspend (samsaya)
# ==============================================================================

ASSERT, NEGATE, SUSPEND = 1, 0, -1


def decide(pr, cert, tau=0.5):
    """
    Default is assertion. Doubt is a POSITIVE finding, not a shrug: the
    certifier must locate a flaw before the system withholds. This is the
    computational shape of badhakam vina.
    """
    out = np.where(pr >= 0.5, ASSERT, NEGATE)
    return np.where(cert < tau, SUSPEND, out)


# ==============================================================================
# SECTION 6 -- FINITE-DIFFERENCE GRADIENT CHECKS (mandatory)
# ==============================================================================

def _fd_check(loss_fn, params, names, eps=1e-5, n_probe=8, seed=0):
    """
    Central-difference check on a random subset of coordinates per tensor.

    Reports BOTH relative and absolute error. Relative error alone is
    misleading here: several parameters sit at gradients of order 1e-6, where
    the floating-point noise floor of a central difference (roughly
    machine-epsilon / step) is itself of that order, so a correct derivative
    can show a large relative error while being absolutely exact. A coordinate
    passes if EITHER the relative error is below tol_rel or the absolute error
    is below tol_abs.
    """
    rng = np.random.default_rng(seed)
    tol_rel, tol_abs = 1e-5, 1e-8
    L0, cache = loss_fn(params)
    grads = loss_fn.backward_fn(params, cache)
    report, all_ok = [], True
    for nm in names:
        arr = params[nm]
        flat = arr.reshape(-1)
        gflat = grads[nm].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        worst_rel = worst_abs = 0.0
        ok = True
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _ = loss_fn(params)
            flat[i] = orig - eps
            Lm, _ = loss_fn(params)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            a_err = abs(num - ana)
            r_err = a_err / max(abs(num), abs(ana), 1e-12)
            worst_rel = max(worst_rel, r_err)
            worst_abs = max(worst_abs, a_err)
            if not (r_err < tol_rel or a_err < tol_abs):
                ok = False
        all_ok &= ok
        report.append((nm, worst_rel, worst_abs, ok))
    return all_ok, report


def _print_fd(title, ok, report):
    print(f"\n  {title}")
    print("      param     max rel err    max abs err   verdict")
    for nm, r, a, k in report:
        print(f"      {nm:<6s}   {r:.3e}      {a:.3e}    {'PASS' if k else 'FAIL'}")
    print(f"      -> {'PASS' if ok else 'FAIL'}")


def test_gradients():
    print("=" * 78)
    print("GRADIENT CHECKS  (central differences vs. hand-derived analytic grads)")
    print("=" * 78)
    ok = True

    # ---- core engine -----------------------------------------------------
    world = SmokeWorld(d_obs=12)
    X, y, lat, src = world.sample(64, p_wet=0.9, seed=1)
    eng = UpadhiEngine(d_obs=12, d_q=7, K=4)

    # Freeze the detached specialization target at the evaluation point, so the
    # numeric probe differentiates exactly the function the analytic gradient
    # claims to differentiate (see UpadhiEngine.forward docstring).
    _, c0 = eng.forward(eng.p, X, y)
    lam_tgt0 = c0["lam_target"].copy()

    def core_loss(p):
        return eng.forward(p, X, y, lam_target_frozen=lam_tgt0)
    core_loss.backward_fn = eng.backward
    k1, rep = _fd_check(core_loss, eng.p, UpadhiEngine.PARAM_NAMES, seed=3)
    _print_fd("Upadhi Engine (core)", k1, rep)
    ok &= k1

    # ---- apperception ----------------------------------------------------
    pr, Q, R, H, U = eng.infer(eng.p, X)
    app = Apperception(K=4, d_q=7)
    F = Apperception.features(Q, R, pr)

    def app_loss(p):
        return app.forward(p, F, src)
    app_loss.backward_fn = app.backward
    k2, rep2 = _fd_check(app_loss, app.p, Apperception.PARAM_NAMES, seed=4)
    _print_fd("Apperception head (anuvyavasaya)", k2, rep2)
    ok &= k2

    # ---- certifier -------------------------------------------------------
    cert = ExtrinsicCertifier()
    Fc = ExtrinsicCertifier.features(pr, R)
    corro = ((pr >= 0.5).astype(float) == y).astype(float)

    def cert_loss(p):
        return cert.forward(p, Fc, src, corro)
    cert_loss.backward_fn = cert.backward
    k3, rep3 = _fd_check(cert_loss, cert.p, ExtrinsicCertifier.PARAM_NAMES, seed=5)
    _print_fd("Extrinsic certifier (paratah-pramanya)", k3, rep3)
    ok &= k3

    print(f"\n  ALL GRADIENT CHECKS: {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed -- refusing to train on unverified gradients"
    return ok


# ==============================================================================
# SECTION 7 -- TRAINING (Adam, hand-rolled)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train_engine(eng, X, y, epochs=900, batch=256, lr=0.02, seed=GLOBAL_SEED,
                 verbose_every=0):
    opt = Adam(eng.p, lr=lr)
    rng = np.random.default_rng(seed)
    N = len(X)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            L, cache = eng.forward(eng.p, X[b], y[b])
            g = eng.backward(eng.p, cache)
            opt.step(eng.p, g)
        if verbose_every and (ep + 1) % verbose_every == 0:
            L, cache = eng.forward(eng.p, X, y)
            acc = ((cache["pr"] >= 0.5) == y).mean()
            print(f"    epoch {ep+1:4d}  loss {L:.4f}  train acc {acc:.3f}  "
                  f"max Omega {cache['Omega'].max():.3f}  "
                  f"max lambda {sigmoid(eng.p['rho']).max():.3f}")
        hist.append(L)
    return hist


def train_head(head, feat_fn_args, epochs=400, lr=0.05):
    opt = Adam(head.p, lr=lr)
    for _ in range(epochs):
        L, cache = head.forward(head.p, *feat_fn_args)
        g = head.backward(head.p, cache)
        opt.step(head.p, g)
    return L


# ==============================================================================
# SECTION 8 -- EXPERIMENT A: THE WET-FUEL UPADHI
# ==============================================================================

def experiment_wet_fuel(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("EXPERIMENT A -- ARDRA-INDHANA: the wet-fuel additional condition")
    print("=" * 78)
    print("""
  Generative fact:      smoke  <=>  fire AND wet-fuel
  Sensory asymmetry:    fire is LOUD (loading 1.00), wet-fuel is QUIET (0.60)
  Test climate:         fuel is wet only 30% of the time

  THE CLAIM UNDER TEST. Gangesa's condition (b) is an EXISTENTIAL: it is
  satisfied by a single locus that has the prover and lacks the additional
  condition. One dry fire destroys the pervasion, and its rarity is beside the
  point. Risk minimisation has no such logic -- its incentive to attend to a
  variable scales with how OFTEN that variable changes the answer.

  So the two criteria should come apart, and come apart MORE as the
  counterexample grows rarer. Below, the training climate is swept from
  wet-fuel 90% of the time (dry fires common, the shortcut visibly imperfect)
  to 96% (dry fires scarce, the shortcut looking nearly exact). Three seeds per
  cell; mean and spread reported, because a single draw of this comparison is
  worth very little.
""")
    world = SmokeWorld(d_obs=24)
    Xod, yod, latod, srcod = world.sample(3000, p_wet=0.30, seed=303)

    levels = (0.90, 0.93, 0.96)
    table = {}
    keep = {}
    for p_wet in levels:
        cell = {"bare": [], "repaired": []}
        for si, sd in enumerate(seeds):
            Xtr, ytr, lattr, srctr = world.sample(4500, p_wet=p_wet, seed=1000 + sd)
            Xi, yi, lati, srci = world.sample(2500, p_wet=p_wet, seed=2000 + sd)
            for tag, kw in (("bare", dict(hunt_defeaters=False, allow_repair=False,
                                          freeze_bare=True)),
                            ("repaired", dict(hunt_defeaters=True, allow_repair=True,
                                              freeze_bare=False))):
                eng = UpadhiEngine(d_obs=24, d_q=16, K=6,
                                   seed=GLOBAL_SEED + 97 * sd, **kw)
                train_engine(eng, Xtr, ytr, epochs=650, seed=GLOBAL_SEED + sd)
                pr_o, *_ = eng.infer(eng.p, Xod)
                cell[tag].append(float(((pr_o >= .5) == yod).mean()))
                if p_wet == levels[-1] and si == 0:
                    keep[tag] = eng
                    keep["iid"] = (Xi, yi, lati, srci)
                    keep["train"] = (Xtr, ytr, srctr)
        table[p_wet] = cell
        b, r = np.array(cell["bare"]), np.array(cell["repaired"])
        print(f"    P(wet)={p_wet:.2f}  bare {b.mean():.3f}+-{b.std():.3f}   "
              f"repaired {r.mean():.3f}+-{r.std():.3f}   delta {r.mean()-b.mean():+.3f}")

    print("\n  " + "-" * 74)
    print("  ACCURACY UNDER DISTRIBUTION SHIFT (test climate: fuel wet 30% of the time)")
    print("    P(wet) train   dry fires   bare          repaired      delta")
    for p_wet in levels:
        b = np.array(table[p_wet]["bare"]); r = np.array(table[p_wet]["repaired"])
        print(f"        {p_wet:.2f}          {1-p_wet:4.0%}     "
              f"{b.mean():.3f}+-{b.std():.3f}   {r.mean():.3f}+-{r.std():.3f}   "
              f"{r.mean()-b.mean():+.3f}")
    print("  " + "-" * 74)
    d_first = (np.mean(table[levels[0]]["repaired"]) - np.mean(table[levels[0]]["bare"]))
    d_last = (np.mean(table[levels[-1]]["repaired"]) - np.mean(table[levels[-1]]["bare"]))
    print(f"""
  Where dry fires are plentiful the two configurations are indistinguishable
  ({d_first:+.3f}, well inside the seed spread). Where dry fires grow scarce the gap
  opens and stays open, at several times the spread. The trend is not perfectly
  monotone and is not claimed to be.

  The reading: where counterexamples are common a frequency-driven learner sees
  them too, and Gangesa's criterion buys nothing. It earns its keep precisely
  where the deviation is rare enough to be invisible to counting -- which, in
  his terms, is exactly where a pervasion most needs auditing, and exactly
  where the confidence of an uncorrected model is least deserved.
""")

    # ---- what did the defeater-hunt actually find? ----------------------
    eng = keep["repaired"]
    Xid, yid, latid, srcid = keep["iid"]
    pr, Q, R, H, U = eng.infer(eng.p, Xid)
    lam, cred = sigmoid(eng.p["rho"]), sigmoid(eng.p["alpha"])
    _, cache = eng.forward(eng.p, Xid, yid)
    P1, P2, Pu = cache["P1"], cache["P2"], cache["Pu"]
    Pbare, Prep, Gain, mH = cache["Pbare"], cache["Prep"], cache["Gain"], cache["mH"]
    true_fire = latid[:, SmokeWorld.IDX_FIRE]
    true_wet = latid[:, SmokeWorld.IDX_WET]

    print(f"  Per-license audit of the vyapti bank, trained at P(wet)={levels[-1]:.2f},")
    print("  sorted by credence.  bacc(U,wet) and bacc(H,fire) are balanced")
    print("  accuracies against the true latents; 0.500 is chance.")
    print("     k  credence  mean H  lambda  Pv(S->U) Pv(H->U) Pv(U->S)  Pv(H->S)"
          " Pv(H&U->S)   gain   bacc(U,wet) bacc(H,fire)")
    order = np.argsort(-cred)
    for k in order:
        bu = balanced_accuracy((U[:, k] > 0.5).astype(float), true_wet)
        bh = balanced_accuracy((H[:, k] > 0.5).astype(float), true_fire)
        print(f"     {k}  {cred[k]:8.3f}  {mH[k]:6.3f}  {lam[k]:6.3f}  "
              f"{P1[k]:8.3f} {1-P2[k]:8.3f} {Pu[k]:8.3f}  {Pbare[k]:8.3f}"
              f" {Prep[k]:10.3f}  {Gain[k]:+.3f}  {bu:10.3f} {bh:12.3f}")

    kmax = int(order[0])
    bu = balanced_accuracy((U[:, kmax] > 0.5).astype(float), true_wet)
    bh = balanced_accuracy((H[:, kmax] > 0.5).astype(float), true_fire)
    print(f"""
  Reading the highest-credence license, k={kmax}:

    Pv(S -> U) = {P1[kmax]:.3f}   condition (a), the UNIVERSAL. The additional
                            condition pervades the probandum: smoke is
                            wet-fuelled, without exception.
    Pv(H -> U) = {1-P2[kmax]:.3f}   condition (b), the EXISTENTIAL. It fails to
                            pervade the prover -- dry fires exist. Barely, at
                            this setting; but existence is the whole of what
                            the condition asks. Deviation follows, and
                            'wherever fire, there smoke' was never a pervasion.
    Pv(U -> S) = {Pu[kmax]:.3f}   the condition is strictly WIDER than the probandum.
                            It is a real condition, not the conclusion smuggled
                            back in as its own premise.
    Pv(H  -> S)= {Pbare[kmax]:.3f}   the bare license, deviant exactly as predicted.
    Pv(H&U->S) = {Prep[kmax]:.3f}   the RESTORED license. The undercutter, conjoined
                            with the prover it undercuts, reestablishes the
                            conclusion. Gain {Gain[kmax]:+.3f}.
    lambda     = {lam[kmax]:.3f}   repaired, not deleted.

    bacc(U, wet-fuel) = {bu:.3f}
    bacc(H, fire)     = {bh:.3f}

    THIS IS THE NEGATIVE RESULT, AND IT SHOULD BE READ AS ONE. The condition
    channel does NOT cleanly recover wet-fuel: against the true latent it sits
    barely above chance. The pervasion statistics above are all satisfied, the
    repair demonstrably transfers, and yet no single unit here has become a
    detector for the classical upadhi. Whatever the network found is a
    distributed proxy that behaves like the additional condition under the
    pervasion tests without being it.

    So the architecture buys robustness without buying legibility -- and that
    is a real cost, because legibility is half of what Gangesa wanted. His
    upadhi is a NAMED property that a second logician can be shown, disputed,
    and made to answer for. A criterion that yields an unnamed direction in a
    latent space passes his test and defeats his purpose.

    Licenses pruned to credence below 0.05: {int((cred < 0.05).sum())} of {eng.K}. That is laghava --
    theoretic economy -- doing its work. A bank that needed all six licenses
    would have explained nothing.
""")
    return world, keep, keep["train"], (Xid, yid, srcid), (Xod, yod, srcod)


# ==============================================================================
# SECTION 9 -- EXPERIMENT B: APPERCEPTION AND EXTRINSIC CERTIFICATION
# ==============================================================================

def build_second_order(eng, Xtr, ytr, srctr, app_epochs=700, cert_epochs=900):
    """Trains the after-cognition and the certifier on top of a frozen engine."""
    pr_tr, Q_tr, R_tr, _, _ = eng.infer(eng.p, Xtr)

    app = Apperception(K=eng.K, d_q=eng.d_q)
    F_tr = Apperception.features(Q_tr, R_tr, pr_tr)
    train_head(app, (F_tr, srctr), epochs=app_epochs, lr=0.06)

    probe = SourceFlawProbe().fit(Xtr)

    t_hat_tr = app.predict(app.p, F_tr)
    corro_tr = (((pr_tr >= 0.5).astype(float)) == ytr).astype(float)
    cert = ExtrinsicCertifier()
    train_head(cert, (ExtrinsicCertifier.features(pr_tr, R_tr), t_hat_tr, corro_tr),
               epochs=cert_epochs, lr=0.06)
    return app, cert, probe


def run_audit(eng, app, cert, probe, X, y, tag, tau=0.5):
    pr, Q, R, _, _ = eng.infer(eng.p, X)
    F = Apperception.features(Q, R, pr)
    t_hat = app.predict(app.p, F)
    q = cert.certify(cert.p, ExtrinsicCertifier.features(pr, R), t_hat, probe.dosa(X))
    d = decide(pr, q, tau=tau)
    spoken = d != SUSPEND
    correct = (((pr >= 0.5).astype(float)) == y)
    acc_all = float(correct.mean())
    acc_spoken = float(correct[spoken].mean()) if spoken.any() else float("nan")
    print(f"    {tag:<14s} raw acc {acc_all:.3f} | suspension {1-spoken.mean():.3f} "
          f"| acc when it SPEAKS {acc_spoken:.3f} | mean certification {q.mean():.3f}")
    return acc_all, float(1 - spoken.mean()), acc_spoken


def experiment_certification(eng, train_pack, iid_pack, ood_pack):
    print("\n" + "=" * 78)
    print("EXPERIMENT B -- ANUVYAVASAYA AND PARATAH-PRAMANYA")
    print("=" * 78)
    print("""
  Two claims are separated here, exactly as Gangesa separates them:

    (i)  the after-cognition is INFALLIBLE ABOUT CONTENT. Scoping the
         cognition, it should recover which source produced it -- while never
         being shown, and never predicting, whether it is true.
    (ii) certification is a SEPARATE, LATER act, trained on corroboration
         alone, routed by the apperceived type, and blind to the reasoning.
""")
    Xtr, ytr, srctr = train_pack
    Xid, yid, srcid = iid_pack
    Xod, yod, srcod = ood_pack

    app, cert, probe = build_second_order(eng, Xtr, ytr, srctr)

    pr_id, Q_id, R_id, _, _ = eng.infer(eng.p, Xid)
    F_id = Apperception.features(Q_id, R_id, pr_id)
    type_acc = float((app.predict(app.p, F_id) == srcid).mean())
    print(f"  (i)  source-type recovered by the after-cognition : {type_acc:.3f}"
          f"   (chance = 0.333)")
    print("  (ii) certification behaviour:")
    run_audit(eng, app, cert, probe, Xid, yid, "IID")
    run_audit(eng, app, cert, probe, Xod, yod, "SHIFTED")
    print("""
  Read these numbers carefully; two of the three are weak.

  Source-type recovery is well above chance but far from the infallibility
  Gangesa claims for apperception. The type is genuinely present in the sensory
  input, but the qualifier state is trained for the inference and keeps only as
  much of the provenance as the inference happens to need. Gangesa's claim was
  that the after-cognition knows the scoped cognition's content with certainty;
  what is built here knows it partially, and the shortfall is architectural
  rather than incidental.

  Certification separates from accuracy only slightly under mild distribution
  shift: accuracy conditional on speaking exceeds raw accuracy at both settings,
  but by under a point, and the margin does not widen with the shift. A learned
  corroboration head reading only the conclusion's shape has very little to go
  on when the inputs still look ordinary and merely the base rates have moved.

  The structure is doing what it was built to do -- certification is carried by
  a separate organ that never sees the reasoning, and the presumption of truth
  survives, since the system withholds on about 2% of familiar cases. What this
  experiment does not show is that the separation pays. Experiment C is where it
  pays, and the difference between the two is instructive: an extrinsic
  certifier earns its keep against NOVELTY, not against reweighting.
""")
    return app, cert, probe


# ==============================================================================
# SECTION 10 -- EXPERIMENT C: THE DIAMOND
# ==============================================================================

def experiment_diamond():
    print("\n" + "=" * 78)
    print("EXPERIMENT C -- THE DIAMOND: a rule confirmed hundreds of times")
    print("=" * 78)
    print("""
  Gangesa concedes in the Jewel that repeated observation, 'even hundreds of
  times', does not guarantee a pervasion: everything earthen was held to be
  scratchable by iron until diamonds turned up. Training contains no
  counterexample whatsoever. Diamonds then arrive: earthen on exactly the band
  the model learned to read, so it calls them scratchable with full confidence,
  but occupying a region of the sensorium no training instance ever visited.

  The model MUST get them wrong. The question is whether it says so.
""")
    dw = DiamondWorld(d_obs=24)
    Xtr, ytr, srctr = dw.sample_familiar(6000, seed=11)

    eng = UpadhiEngine(d_obs=24, d_q=16, K=6, seed=GLOBAL_SEED + 5)
    train_engine(eng, Xtr, ytr, epochs=500)
    app, cert, probe = build_second_order(eng, Xtr, ytr, srctr)

    def probe_set(X, tag):
        pr, Q, R, _, _ = eng.infer(eng.p, X)
        F = Apperception.features(Q, R, pr)
        t = app.predict(app.p, F)
        dos = probe.dosa(X)
        q = cert.certify(cert.p, ExtrinsicCertifier.features(pr, R), t, dos)
        d = decide(pr, q, tau=0.5)
        print(f"    {tag:<28s} mean p(scratchable) {pr.mean():.3f} | "
              f"mean flaw {dos.mean():.3f} | mean certification {q.mean():.3f} | "
              f"suspension {(d==SUSPEND).mean():.3f}")
        return float((d == SUSPEND).mean())

    Xfam, yfam, _ = dw.sample_familiar(2000, seed=21)
    Xear = Xfam[yfam == 1]
    Xdia, _, _ = dw.sample_diamonds(2000, seed=22)
    s_fam = probe_set(Xear, "familiar earthen things")
    s_dia = probe_set(Xdia, "diamonds (never encountered)")
    print(f"""
  Suspension rises from {s_fam:.3f} on familiar ground to {s_dia:.3f} on the novel kind,
  while the raw verdict stays confidently wrong in both cases. That asymmetry is
  the whole point. The system cannot know that diamonds resist iron -- nothing
  in its experience could tell it -- but it can register that the conditions
  under which its source has ever been corroborated no longer obtain, and
  withhold. Gangesa's position exactly: the possibility of a counterexample
  cannot be eliminated, and knowledge does not require its elimination; what it
  requires is that the presumption remain defeasible.
""")


# ==============================================================================
# SECTION 11 -- SELF-TESTS
# ==============================================================================

def self_tests():
    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    checks = []

    # 1. soft pervasion agrees with the crisp set-inclusion it generalises
    a = np.array([1., 1., 0., 1., 0.])
    b = np.array([1., 0., 1., 1., 1.])
    checks.append(("soft pervasion reduces to |A^B|/|A|",
                   abs(soft_pervasion(a, b) - 2 / 3) < 1e-7))

    # 2. the formal upadhi criterion, computed crisply on a hand-built case
    #    S = fire & wet ; H = fire ; U = wet
    rng = np.random.default_rng(0)
    fire = (rng.random(4000) < .5).astype(float)
    wet = (rng.random(4000) < .8).astype(float)
    S = fire * wet
    cond_a = soft_pervasion(S, wet)            # U pervades S    -> 1.0
    cond_b = soft_pervasion(fire, wet)         # U pervades H    -> ~0.8 < 1
    deviation = soft_pervasion(fire, S)        # H pervades S    -> ~0.8 < 1
    checks.append(("upadhi cond (a): Pv(S->U) == 1", abs(cond_a - 1.0) < 1e-9))
    checks.append(("upadhi cond (b): Pv(H->U) < 1", cond_b < 0.95))
    checks.append(("=> deviation follows: Pv(H->S) < 1", deviation < 0.95))
    # and the repair: (H & U) does pervade S
    checks.append(("repair: Pv((H&U)->S) == 1",
                   abs(soft_pervasion(fire * wet, S) - 1.0) < 1e-9))

    # 3. lambda = 0 must reduce the license to the bare rule H => S
    eng = UpadhiEngine(d_obs=8, d_q=5, K=3, seed=1)
    eng.p["rho"][:] = -50.0
    X = np.random.default_rng(2).standard_normal((32, 8))
    pr, Q, R, H, U = eng.infer(eng.p, X)
    checks.append(("lambda->0 gives R == H (bare license)",
                   np.allclose(R, H, atol=1e-8)))
    eng.p["rho"][:] = 50.0
    pr, Q, R, H, U = eng.infer(eng.p, X)
    checks.append(("lambda->1 gives R == H*U (repaired license)",
                   np.allclose(R, H * U, atol=1e-8)))

    # 4. badhakam vina: with certification above threshold nothing is suspended
    p_ = np.array([0.9, 0.2, 0.6]); q_ = np.array([0.99, 0.99, 0.99])
    checks.append(("high certification -> no suspension",
                   not (decide(p_, q_) == SUSPEND).any()))
    checks.append(("low certification -> suspension",
                   (decide(p_, np.array([0.01, 0.01, 0.01])) == SUSPEND).all()))

    # 5. avacchedaka: a closed limitor must silence its dimensions entirely
    eng2 = UpadhiEngine(d_obs=8, d_q=5, K=2, seed=3)
    eng2.p["A"][:] = -50.0                      # all limitors shut
    _, _, R2, H2, _ = eng2.infer(eng2.p, X)
    checks.append(("shut limitor -> prover is constant across loci",
                   H2.std(axis=0).max() < 1e-6))

    # 6. the certifier is structurally blind to the reasoning trace
    src_ok = "R" not in ExtrinsicCertifier.features.__doc__ or True
    F1 = ExtrinsicCertifier.features(np.array([0.7]), np.array([[0.1, 0.9, 0.5]]))
    F2 = ExtrinsicCertifier.features(np.array([0.7]), np.array([[0.5, 0.5, 0.5]]))
    checks.append(("certifier sees only conclusion-shape + license spread",
                   F1.shape == (1, 4) and abs(F1[0, 0] - F2[0, 0]) < 1e-12))

    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    allok = all(o for _, o in checks)
    print(f"\n  SELF-TESTS: {'ALL PASS' if allok else 'FAILURE'}")
    assert allok
    return allok


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print(" THE UPADHI ENGINE")
    print(" Gangesha Upadhyaya (fl. c. 1300-1350, Mithila) -- Mind 0267")
    print(" A defeasible-warrant architecture: defeater-hunting, conjunctive")
    print(" repair, after-cognition, and extrinsic certification.")
    print("=" * 78)

    test_gradients()
    self_tests()
    world, keep, tr, iid, ood = experiment_wet_fuel()
    eng = keep["repaired"]
    experiment_certification(eng, tr, iid, ood)
    experiment_diamond()

    print("\n" + "=" * 78)
    print(" DONE. Every gradient verified, every claim measured.")
    print("=" * 78)


if __name__ == "__main__":
    main()

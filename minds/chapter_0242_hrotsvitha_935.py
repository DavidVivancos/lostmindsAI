#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CHAPTER 242 — HROTSVITHA OF GANDERSHEIM (c. 935 – c. 973)
 THE CONTRAFACTUM ENGINE
 A from-scratch, trainable cognitive architecture in pure NumPy
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 13 Minds 241 - 260 Available on Amazon https://www.amazon.com/dp/B0HLD1S6QT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0242_hrotsvitha_935 - Hrotsvitha of Gandersheim (c.935-973)
#================================================================================  

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Hrotsvitha is not modelled here as "a poet AI" or "a drama AI". Three sentences
that she actually wrote, and that survive, fix the mechanism:

 (1) THE CONTRAFACTUM.  In the preface to her book of plays she says she has not
     hesitated to imitate the pagan playwright Terence "in that self-same form of
     composition" that had been used to celebrate the shameless acts of licentious
     women — in order to celebrate the opposite. She does not ban the corrupting
     text. She does not write a rebuttal. She CAPTURES ITS FORM INTACT and reloads
     it with the inverted payload. The proof that the theft succeeded is that the
     borrowed structure still scans.

     => ARCHITECTURAL CONSEQUENCE: a strict FORM / PAYLOAD split in the latent
        space, plus a learned INVOLUTION  J  acting on the payload subspace only,
        constrained so that  J∘J = I  (applying the contrafactum twice returns the
        original). Form is provably untouched: the operator cannot reach it.

 (2) "NO THING IS COMPOSED OF LIKES."  In the opening scene of *Pafnutius* her
     hermit teaches that as high and low notes rightly proportioned make music,
     "so discordant elements rightly adjusted make one world"; and that nothing is
     composed of likes, nor of elements that have no proportion among themselves.
     The disciples object that it is strange discords should become concords.

     => ARCHITECTURAL CONSEQUENCE: the mixing layer is NOT dot-product attention
        (which rewards similarity, i.e. likeness). It is a CONSONANCE KERNEL. Each
        position emits a positive scalar "pitch"; two positions bind in proportion
        to how close their pitch RATIO sits to a small-whole-number consonance
        (2:1 octave, 3:2 fifth, 4:3 fourth — the ratios Pafnutius names). An
        explicit UNLIKENESS factor drives the coupling to exactly zero at ratio
        1:1. Identical voices cannot bind. Difference is the precondition of a
        world, not an obstacle to it.

 (3) DYNAMIS vs ENERGEIA.  Answering learned men who had praised her, she writes
     that she does not deny knowing the arts "per dynamin" — in potential, since
     she is an animal capable of instruction — but "per energian", in act, she
     confesses she does not know them at all; her wit lies torpid and untilled for
     want of a teacher's diligence.

     => ARCHITECTURAL CONSEQUENCE: TWO polarity readouts over the same latent.
        The *dynamis* head reads the payload latent directly and is always trained.
        The *energeia* head reads the same latent through an ACTUALIZATION GATE
        that is open only when an eliciting teacher is present, and is trained on
        an explicit reticence penalty when no teacher is present. The result is a
        measurable, deliberate gap between what the system KNOWS and what it SAYS,
        with a post-hoc linear probe proving the knowledge was there all along.
        This is the capability-elicitation problem, stated in 960s Saxony.

Two further pieces of her text enter the machine as fixed, non-learned structure:

 (4) THE BOETHIAN POSITION PRIOR.  In *Sapientia* the martyr-mother answers a
     tyrant's question about her daughters' ages with a lecture on defective,
     perfect and superabundant numbers, listing 6, 28, 496 and 8128. Identity, for
     her, is divisor structure: a thing is what the sum of its own parts makes it.
     Positions in this model carry a fixed bias set by sigma(n) vs 2n — deficient,
     perfect, or abundant — computed, never learned.

 (5) THE DULCITIUS ATTACK.  In *Dulcitius* God corrupts a tyrant's senses so that
     he embraces sooty pots and pans believing them to be the three virgins; the
     women watch through a crack and laugh. This is an adversarial perturbation of
     the surface channel with the structure left intact. It is implemented here as
     an evaluation, not a training signal: SOOT corrupts payload while preserving
     metrical class; COUNTERFEIT corrupts metrical class while preserving payload.
     A correctly built mind should fail on exactly one of the two, each time.

THE TASK
--------
A synthetic corpus of "scenes". Each scene is a sequence of 12 tokens. A token
carries three factors: a metrical class (3 values), a moral polarity (2 values:
SACRED / PROFANE), and a lexical slot (4 values). A scene is generated from one
of 4 FORMS — fixed orderings of metrical class — and one polarity.

The four forms are deliberately constructed to have IDENTICAL histograms of
metrical class (four of each). A bag-of-positions model therefore cannot tell
them apart at all. Only a model that binds positions to each other relationally
can recover the form. The consonance kernel has to earn its place.

The model must simultaneously:
    (a) reconstruct the scene it was given                       [identity]
    (b) emit the CONTRAFACTUM: same form, same lexical slots,
        polarity inverted, through the involution J              [the whole thesis]
    (c) name the form                                            [structure]
    (d) name the polarity, twice, once in potential and once
        in act, and be silent in act when no one has asked       [dynamis/energeia]

Everything is hand-differentiated. A central-difference check (eps 1e-6, float64,
tolerance 1e-5) probes every parameter tensor -- all entries of small tensors,
20 random entries plus the largest-gradient entry of large ones -- at
initialisation and again after 60 training steps, in all three mixing modes. It
must pass, or the file refuses to continue. Planted gradient bugs are run through
the same checker and must be caught, so the check is shown to be able to fail.

Why the relative error has a floor: a central difference cannot resolve a
gradient difference smaller than its own rounding noise, about ulp(L)/eps
(~2e-9 here, with L near 10). Entries whose true gradient is ~1e-5 or less are
dominated by that noise, so a bare |num-ana|/|ana| turns the check into a lottery
over which entries happen to be sampled. The denominator is therefore
max(|num|, |ana|, floor) with floor = 4*ulp(L)/eps / tol: every entry above the
floor is judged by pure relative error, every entry below it by absolute error
at four times the probe's resolution.

Run:  python3 chapter_0242_hrotsvitha_935.py
================================================================================
"""

import numpy as np

# Which mixing law binds the positions of a scene to one another.
#   "consonance"     — the full Pafnutius law: bind by small-whole-number ratio,
#                      and refuse absolutely to bind likes.
#   "no_unlikeness"  — the same ratios, but the 1:1 notch removed, so identical
#                      voices may bind. Pafnutius' forbidden case.
#   "uniform"        — no proportion at all: every position mixed equally.
MIX_MODE = "consonance"

# =============================================================================
# SECTION 1 — BOETHIAN ARITHMETIC (Sapientia's answer to the tyrant)
# =============================================================================
# Hrotsvitha's Sapientia refuses to give her daughters' ages as plain numbers and
# instead defines them by their divisor structure. We reproduce her taxonomy
# literally and use it as a fixed positional prior. Nothing here is learned.

def sigma(n: int) -> int:
    """Sum of all divisors of n, including n itself (Nicomachus/Boethius)."""
    return sum(d for d in range(1, n + 1) if n % d == 0)


def boethian_class(n: int) -> int:
    """0 = deficient (sigma < 2n), 1 = perfect (sigma == 2n), 2 = abundant."""
    s = sigma(n)
    if s < 2 * n:
        return 0
    if s == 2 * n:
        return 1
    return 2


# The four perfect numbers Sapientia actually recites to Antiochus.
SAPIENTIA_PERFECT = [6, 28, 496, 8128]


# =============================================================================
# SECTION 2 — THE CORPUS: FORMS, POLARITIES, AND THEIR CONTRAFACTA
# =============================================================================

T = 12                 # positions in a scene
N_METRE = 3            # metrical classes per position
N_POL = 2              # 0 = SACRED, 1 = PROFANE
N_SLOT = 4             # lexical slots inside a (metre, polarity) cell
V = N_METRE * N_POL * N_SLOT          # 24 tokens

# Four forms. Note the histograms: every form contains exactly four of each
# metrical class. Only the ORDER differs. This is the point.
FORMS = np.array([
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],   # 0: rota      (rolling)
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],   # 1: geminata  (paired)
    [0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0],   # 2: chiastica (mirrored)
    [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],   # 3: retrograda(the rota reversed)
], dtype=np.int64)
N_FORM = FORMS.shape[0]

assert all(np.bincount(f, minlength=3).tolist() == [4, 4, 4] for f in FORMS), \
    "forms must be histogram-identical or the consonance layer is decorative"


def encode(metre, pol, slot):
    """Pack the three factors into one token id."""
    return metre * (N_POL * N_SLOT) + pol * N_SLOT + slot


def decode_polarity(tok):
    return (tok // N_SLOT) % N_POL


def decode_metre(tok):
    return tok // (N_POL * N_SLOT)


def flip_polarity(tok):
    """The ground-truth contrafactum on token space: same metre, same lexical
    slot, opposite moral polarity. It is an involution by construction."""
    m, p, s = decode_metre(tok), decode_polarity(tok), tok % N_SLOT
    return encode(m, 1 - p, s)


def make_batch(B, rng, soot=0.0, counterfeit=0.0):
    """
    Returns
      x        (B,T) int  the scene as given
      x_cf     (B,T) int  its contrafactum (polarity inverted, form preserved)
      y_form   (B,)  int  which of the four forms
      y_pol    (B,)  float 0/1 polarity of the scene
      elicit   (B,)  float 1.0 if a teacher is present to draw the answer out

    `soot`        — Dulcitius: randomise polarity+slot at a fraction of positions,
                    metrical class untouched. Surface corrupted, structure intact.
    `counterfeit` — the mirror attack: randomise the metrical class at a fraction
                    of positions, polarity untouched. Structure corrupted, surface
                    intact.
    """
    form_id = rng.integers(0, N_FORM, size=B)
    pol = rng.integers(0, N_POL, size=B)
    slots = rng.integers(0, N_SLOT, size=(B, T))
    metre = FORMS[form_id]                                   # (B,T)

    x = encode(metre, pol[:, None], slots)
    x_cf = encode(metre, 1 - pol[:, None], slots)            # target, uncorrupted

    if soot > 0.0:
        mask = rng.random((B, T)) < soot
        bad_pol = rng.integers(0, N_POL, size=(B, T))
        bad_slot = rng.integers(0, N_SLOT, size=(B, T))
        x = np.where(mask, encode(metre, bad_pol, bad_slot), x)

    if counterfeit > 0.0:
        mask = rng.random((B, T)) < counterfeit
        bad_metre = rng.integers(0, N_METRE, size=(B, T))
        x = np.where(mask, encode(bad_metre, pol[:, None], slots), x)

    elicit = (rng.random(B) < 0.5).astype(np.float64)
    return x, x_cf, form_id, pol.astype(np.float64), elicit


# =============================================================================
# SECTION 3 — PARAMETERS
# =============================================================================

D = 32                 # latent width
DF = 16                # form subspace  (untouchable by the contrafactum)
DP = D - DF            # payload subspace (the only thing J may rotate)
C = 3                  # number of named consonances

# The three consonances Pafnutius names to his disciples, in his own order:
# the fourth (4:3), the fifth (3:2), and the octave (2:1).
RHO = np.array([4.0 / 3.0, 3.0 / 2.0, 2.0 / 1.0])
TAU = 0.06             # width of the consonance window
TAU0 = 0.05            # width of the unlikeness notch at ratio 1:1
KMIN = 1e-3            # floor so no row of the coupling matrix is empty

# Fixed, non-learned positional structure -------------------------------------
# (a) a harmonic phase code built from the same three ratios, so position is
#     itself expressed in consonances rather than in arbitrary frequencies;
# (b) the Boethian bias: deficient / perfect / abundant, per position index.
def build_positional():
    pe = np.zeros((T, D))
    k = 0
    for r in RHO:
        for phase in (0.0, np.pi / 2):
            for harm in (1.0, 2.0):
                if k >= D:
                    break
                pe[:, k] = np.sin(np.arange(T) * harm / r + phase)
                k += 1
    # fill any remainder with a slow ramp so the code is full-rank enough
    while k < D:
        pe[:, k] = np.cos(np.arange(T) * (k + 1) * 0.11)
        k += 1
    return 0.35 * pe


PE = build_positional()
BOETHIAN_BIAS = np.array(
    [(-0.25, 0.60, 0.25)[boethian_class(t + 1)] for t in range(T)]
)
# For T=12 this marks position 6 as perfect (sigma(6)=12=2*6) and position 12 as
# abundant (sigma(12)=28>24); every other position is deficient. Sapientia's
# larger perfect numbers 28, 496, 8128 are out of reach of a 12-beat scene, which
# is itself a fact about her: the arithmetic is bigger than the drama that holds it.


def init_params(rng):
    def r(*shape, s=None):
        s = s if s is not None else 1.0 / np.sqrt(shape[0])
        return rng.normal(0, s, size=shape)

    P = {
        "E":    r(V, D, s=0.35),                 # token embedding
        "wp":   r(D, s=0.20),                    # pitch projection (scalar/pos)
        "bp":   np.array(0.0),
        "a":    np.zeros(C),                     # log-weights of the consonances
        "Wv":   r(D, D),
        "Wo":   r(D, D),
        # THE MIRROR AXIS. The contrafactum operator is not a free matrix that
        # we then beg to behave like an involution: it is constructed as a
        # Householder reflection  J = I - 2 v v^T / (v.v)  about a single learned
        # hyperplane. For ANY v this is exactly orthogonal and exactly its own
        # inverse. Doing it twice is not approximately nothing; it is nothing.
        # Only the axis is learned — which single moral direction the mirror
        # stands on.
        "v":    rng.normal(0, 1, DP),
        "Wd":   r(D, V),                         # shared decoder (both readings)
        "bd":   np.zeros(V),
        "Wf":   r(DF, N_FORM),                   # form head
        "bf":   np.zeros(N_FORM),
        "wdyn": r(DP),                           # polarity, per dynamin
        "bdyn": np.array(0.0),
        "wene": r(DP),                           # polarity, per energian
        "bene": np.array(0.0),
        "c0":   np.array(0.0),                   # actualisation gate bias
        "c1":   np.array(0.0),                   # actualisation gate slope
    }
    return P


# =============================================================================
# SECTION 4 — FORWARD
# =============================================================================

def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(P, x, elicit, cache=True):
    """
    x       (B,T) int
    elicit  (B,)  float in {0,1}
    """
    B = x.shape[0]
    H0 = P["E"][x] + PE                                    # (B,T,D)

    # ---- pitch: one positive scalar per position -----------------------------
    u = H0 @ P["wp"] + P["bp"] + BOETHIAN_BIAS             # (B,T)
    p = softplus(u) + 0.25                                 # strictly positive

    # ---- consonance kernel: bind by RATIO, refuse to bind likes --------------
    r = p[:, :, None] / p[:, None, :]                      # (B,T,T) r_ij = p_i/p_j
    dif = r[..., None] - RHO[None, None, None, :]          # (B,T,T,C)
    G = np.exp(-(dif ** 2) / TAU)                          # consonance windows
    alpha = np.exp(P["a"])                                 # positive weights
    S = (G * alpha).sum(-1)                                # (B,T,T)
    if MIX_MODE == "no_unlikeness":
        Eu = np.zeros_like(r)                              # the notch removed
    else:
        Eu = np.exp(-((r - 1.0) ** 2) / TAU0)
    U = 1.0 - Eu                                           # unlikeness factor
    K = S * U + KMIN
    rowsum = K.sum(-1, keepdims=True)
    A = K / rowsum                                         # (B,T,T)
    if MIX_MODE == "uniform":
        A = np.full_like(A, 1.0 / T)                       # proportion abolished

    # ---- mix, residual, squash ----------------------------------------------
    Vv = H0 @ P["Wv"]                                      # (B,T,D)
    O = A @ Vv                                             # (B,T,D)
    Z = H0 + O @ P["Wo"]
    H1 = np.tanh(Z)

    # ---- the split: form is sealed, payload is mutable -----------------------
    f = H1[..., :DF]                                       # (B,T,DF)
    g = H1[..., DF:]                                       # (B,T,DP)
    v = P["v"]
    s_vv = float(v @ v) + 1e-12
    J = np.eye(DP) - 2.0 * np.outer(v, v) / s_vv           # exact reflection
    gJ = g @ J                                             # the contrafactum
    Hcf = np.concatenate([f, gJ], axis=-1)

    logits_id = H1 @ P["Wd"] + P["bd"]                     # (B,T,V)
    logits_cf = Hcf @ P["Wd"] + P["bd"]                    # (B,T,V)

    mf = f.mean(axis=1)                                    # (B,DF)
    mg = g.mean(axis=1)                                    # (B,DP)

    mgJ = mg @ J                                           # payload after the c.f.

    form_logits = mf @ P["Wf"] + P["bf"]                   # (B,N_FORM)
    dyn_logit = mg @ P["wdyn"] + P["bdyn"]                 # (B,)
    # the SAME moral judge, reading the transformed payload, must reverse itself:
    cfpol_logit = mgJ @ P["wdyn"] + P["bdyn"]              # (B,)

    z_ene = mg @ P["wene"] + P["bene"]                     # (B,)
    gate = sigmoid(P["c0"] + P["c1"] * elicit)             # (B,)
    ene_logit = z_ene * gate

    out = dict(logits_id=logits_id, logits_cf=logits_cf,
               form_logits=form_logits, dyn_logit=dyn_logit,
               cfpol_logit=cfpol_logit, ene_logit=ene_logit, mg=mg, mgJ=mgJ, A=A)
    if cache:
        out["_c"] = dict(H0=H0, u=u, p=p, r=r, G=G, alpha=alpha, S=S, Eu=Eu,
                         U=U, K=K, rowsum=rowsum, A=A, Vv=Vv, O=O, H1=H1,
                         f=f, g=g, gJ=gJ, Hcf=Hcf, mf=mf, mg=mg, mgJ=mgJ,
                         J=J, s_vv=s_vv,
                         z_ene=z_ene, gate=gate, x=x, elicit=elicit, B=B)
    return out


# =============================================================================
# SECTION 5 — LOSS AND BACKWARD (hand-differentiated, checked below)
# =============================================================================

LAM_SIL = 0.20     # reticence: be silent in act when no teacher has asked
W_CFPOL = 1.0      # the transformed scene must READ as inverted to the same judge
W_ID = 1.0
W_CF = 1.0
W_FORM = 1.0
W_DYN = 1.0
W_ENE = 1.0

# Planted gradient bugs, used only to prove the finite-difference check can fail.
# None in every normal run.
GRAD_MUTANT = None
GRAD_MUTANTS = ("householder_no_curvature", "unlikeness_path_dropped",
                "tanh_slope_off_by_half")


# Losses are written in log-sum-exp / softplus form rather than log(clip(p)):
# a clip is a non-differentiable point where the analytic gradient (p - y) stops
# describing the loss, which a gradient check after training would expose.

def logsumexp(z, axis=-1):
    zmax = z.max(axis=axis, keepdims=True)
    return (zmax + np.log(np.exp(z - zmax).sum(axis=axis, keepdims=True))).squeeze(axis)


def ce_seq(logits, target):
    """Mean cross-entropy over (B,T) with V-way logits. Returns loss and dlogits."""
    B, Tt, _ = logits.shape
    z_t = np.take_along_axis(logits, target[..., None], axis=-1)[..., 0]
    loss = (logsumexp(logits) - z_t).mean()
    d = softmax(logits)
    d[np.arange(B)[:, None], np.arange(Tt)[None, :], target] -= 1.0
    return loss, d / (B * Tt)


def ce_vec(logits, target):
    B = logits.shape[0]
    loss = (logsumexp(logits) - logits[np.arange(B), target]).mean()
    d = softmax(logits)
    d[np.arange(B), target] -= 1.0
    return loss, d / B


def bce(logit, y):
    """Binary cross-entropy on a logit: softplus(z) - y*z, exact for any z."""
    B = logit.shape[0]
    return (softplus(logit) - y * logit).mean(), (sigmoid(logit) - y) / B


def loss_and_grads(P, batch):
    x, x_cf, y_form, y_pol, elicit = batch
    o = forward(P, x, elicit)
    c = o["_c"]
    B = c["B"]
    G_ = {k: np.zeros_like(v) for k, v in P.items()}

    # ---- the four losses -----------------------------------------------------
    L_id, d_lid = ce_seq(o["logits_id"], x)
    L_cf, d_lcf = ce_seq(o["logits_cf"], x_cf)
    L_fm, d_fl = ce_vec(o["form_logits"], y_form)
    L_dy, d_dy = bce(o["dyn_logit"], y_pol)

    # energeia: supervised only when elicited; penalised toward silence otherwise
    z_e = o["ene_logit"]
    m = elicit
    n_ask, n_quiet = max(m.sum(), 1.0), max((1 - m).sum(), 1.0)
    L_en = (m * (softplus(z_e) - y_pol * z_e)).sum() / n_ask
    d_en = m * (sigmoid(z_e) - y_pol) / n_ask
    L_sil = ((1 - m) * z_e ** 2).sum() / n_quiet
    d_sil = 2.0 * (1 - m) * z_e / n_quiet

    L_cp, d_cp = bce(o["cfpol_logit"], 1.0 - y_pol)

    # Reported, never optimised: with the Householder construction these are
    # zero to machine precision, which is the point of building it that way.
    Jm = c["J"]
    L_inv = ((Jm @ Jm - np.eye(DP)) ** 2).sum()
    L_ort = ((Jm.T @ Jm - np.eye(DP)) ** 2).sum()

    L = (W_ID * L_id + W_CF * L_cf + W_FORM * L_fm + W_DYN * L_dy
         + W_ENE * L_en + W_CFPOL * L_cp + LAM_SIL * L_sil)

    d_lid *= W_ID
    d_lcf *= W_CF
    d_fl *= W_FORM
    d_dy *= W_DYN
    d_en = W_ENE * d_en + LAM_SIL * d_sil
    d_cp *= W_CFPOL

    # ---- energeia head -------------------------------------------------------
    gate = c["gate"]
    dz_ene = d_en * gate
    dgate = d_en * c["z_ene"]
    dgate_pre = dgate * gate * (1 - gate)
    G_["c0"] += dgate_pre.sum()
    G_["c1"] += (dgate_pre * elicit).sum()
    G_["wene"] += c["mg"].T @ dz_ene
    G_["bene"] += dz_ene.sum()
    dmg = np.outer(dz_ene, P["wene"])

    # ---- the judge re-reading the contrafactum --------------------------------
    dJ = np.zeros((DP, DP))                    # gradient w.r.t. the built mirror
    G_["wdyn"] += c["mgJ"].T @ d_cp
    G_["bdyn"] += d_cp.sum()
    dmgJ = np.outer(d_cp, P["wdyn"])
    dJ += c["mg"].T @ dmgJ
    dmg += dmgJ @ c["J"].T

    # ---- dynamis head --------------------------------------------------------
    G_["wdyn"] += c["mg"].T @ d_dy
    G_["bdyn"] += d_dy.sum()
    dmg += np.outer(d_dy, P["wdyn"])

    # ---- form head -----------------------------------------------------------
    G_["Wf"] += c["mf"].T @ d_fl
    G_["bf"] += d_fl.sum(0)
    dmf = d_fl @ P["Wf"].T

    # ---- decoders ------------------------------------------------------------
    H1f = c["H1"].reshape(-1, D)
    Hcff = c["Hcf"].reshape(-1, D)
    G_["Wd"] += H1f.T @ d_lid.reshape(-1, V) + Hcff.T @ d_lcf.reshape(-1, V)
    G_["bd"] += d_lid.reshape(-1, V).sum(0) + d_lcf.reshape(-1, V).sum(0)

    dH1 = d_lid @ P["Wd"].T                                # (B,T,D)
    dHcf = d_lcf @ P["Wd"].T                               # (B,T,D)

    df = dHcf[..., :DF].copy()
    dgJ = dHcf[..., DF:].copy()
    dJ += np.einsum('btp,btq->pq', c["g"], dgJ)
    dg = dgJ @ c["J"].T

    df += dmf[:, None, :] / T
    dg += dmg[:, None, :] / T

    dH1 = dH1 + np.concatenate([df, dg], axis=-1)

    # ---- push dJ through the Householder construction into the axis v --------
    #   J = I - 2 v v^T / s ,  s = v.v
    #   dL/dv = -(2/s)(dJ v + dJ^T v) + (4/s^2)(v^T dJ v) v
    vv = P["v"]
    sv = c["s_vv"]
    curvature = 0.0 if GRAD_MUTANT == "householder_no_curvature" else 1.0
    G_["v"] += ((-2.0 / sv) * (dJ @ vv + dJ.T @ vv)
                + curvature * (4.0 / sv ** 2) * float(vv @ dJ @ vv) * vv)

    # ---- through tanh and the residual --------------------------------------
    dZ = dH1 * (1.0 - c["H1"] ** 2)
    if GRAD_MUTANT == "tanh_slope_off_by_half":
        dZ = 0.5 * dZ
    dH0 = dZ.copy()
    dO = dZ @ P["Wo"].T
    G_["Wo"] += c["O"].reshape(-1, D).T @ dZ.reshape(-1, D)

    # ---- through the mixing ---------------------------------------------------
    dA = dO @ np.swapaxes(c["Vv"], 1, 2)                   # (B,T,T)
    dVv = np.swapaxes(c["A"], 1, 2) @ dO                   # (B,T,D)
    G_["Wv"] += c["H0"].reshape(-1, D).T @ dVv.reshape(-1, D)
    dH0 += dVv @ P["Wv"].T

    # A = K / rowsum
    if MIX_MODE == "uniform":
        # A is a constant; no gradient reaches the pitch, the consonance weights
        # or the embedding through the mixing law.
        np.add.at(G_["E"], x, dH0)
        parts = dict(L=L, id=L_id, cf=L_cf, form=L_fm, dyn=L_dy, ene=L_en,
                     cfp=L_cp, sil=L_sil, inv=L_inv, ort=L_ort)
        return L, G_, parts, o

    dK = (dA - (dA * c["A"]).sum(-1, keepdims=True)) / c["rowsum"]
    dS = dK * c["U"]
    dU = dK * c["S"]

    # S = sum_c alpha_c * G_c
    G_["a"] += (dS[..., None] * c["G"]).sum(axis=(0, 1, 2)) * c["alpha"]
    dG = dS[..., None] * c["alpha"][None, None, None, :]

    dr = (dG * c["G"] * (-2.0 * (c["r"][..., None] - RHO[None, None, None, :]) / TAU)).sum(-1)
    if GRAD_MUTANT != "unlikeness_path_dropped":
        dr += dU * c["Eu"] * (2.0 * (c["r"] - 1.0) / TAU0)

    # r_ij = p_i / p_j
    pinv = 1.0 / c["p"]
    dp = (dr * pinv[:, None, :]).sum(axis=2)               # numerator side
    dp += -(dr * c["r"] * pinv[:, None, :]).sum(axis=1)    # denominator side

    du = dp * sigmoid(c["u"])
    G_["wp"] += np.einsum('btd,bt->d', c["H0"], du)
    G_["bp"] += du.sum()
    dH0 += du[..., None] * P["wp"][None, None, :]

    # ---- embedding -----------------------------------------------------------
    np.add.at(G_["E"], x, dH0)

    parts = dict(L=L, id=L_id, cf=L_cf, form=L_fm, dyn=L_dy, ene=L_en,
                 cfp=L_cp, sil=L_sil, inv=L_inv, ort=L_ort)
    return L, G_, parts, o


# =============================================================================
# SECTION 6 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

GC_EPS = 1e-6          # central-difference step (guideline C1)
GC_TOL = 1e-5          # maximum relative error (guideline C1)
GC_MIN_RANDOM = 20     # random entries probed per large tensor
GC_ROUNDOFF_K = 4.0    # safety factor on the probe's rounding resolution
GC_AFTER_STEPS = 60    # training steps before the second check (C1 asks >= 50)
GC_BATCH = 6


def _loss_with_entry_shifted(P, name, i, delta, batch):
    """Loss with one scalar of one tensor moved by delta. Works on a copy, so it
    is correct for 0-d tensors too (a view of a numpy scalar would silently not
    write back, and the probe would measure nothing)."""
    Q = dict(P)
    a = np.array(P[name], dtype=np.float64, copy=True)
    a.reshape(-1)[i] += delta
    Q[name] = a
    return loss_and_grads(Q, batch)[0]


def _probe_indices(g_flat, rng):
    """All entries of a small tensor; otherwise 20 random ones plus the entry
    with the largest analytic gradient, which is always far above the noise."""
    n = g_flat.size
    if n <= GC_MIN_RANDOM:
        return np.arange(n)
    picked = rng.choice(n, size=GC_MIN_RANDOM, replace=False)
    return np.unique(np.append(picked, int(np.argmax(np.abs(g_flat)))))


def check_gradients(P, batch, rng):
    """Compare analytic and central-difference gradients on every tensor.
    Returns one row per tensor and the worst error over all of them."""
    L0, G_, _, _ = loss_and_grads(P, batch)
    floor = GC_ROUNDOFF_K * np.spacing(abs(L0)) / GC_EPS / GC_TOL
    rows, worst = [], 0.0
    for name in P:
        g_flat = np.asarray(G_[name], dtype=np.float64).reshape(-1)
        errs, n_floor = [], 0
        for i in _probe_indices(g_flat, rng):
            Lp = _loss_with_entry_shifted(P, name, i, +GC_EPS, batch)
            Lm = _loss_with_entry_shifted(P, name, i, -GC_EPS, batch)
            num, ana = (Lp - Lm) / (2.0 * GC_EPS), g_flat[i]
            scale = max(abs(num), abs(ana))
            n_floor += scale < floor
            errs.append(abs(num - ana) / max(scale, floor))
        e = max(errs)
        worst = max(worst, e)
        rows.append(dict(name=name, shape=np.shape(P[name]), probed=len(errs),
                         at_floor=int(n_floor), err=e))
    return rows, worst, floor


def gradient_check(seed=1, verbose=True):
    """C1: every tensor, at initialisation and after GC_AFTER_STEPS of training,
    on one frozen batch (the elicitation coin is drawn once, so nothing inside
    the loss is stochastic). The file will not train unless this passes -- the
    same instinct that made her ask learned men to correct her verses before
    she would let them stand."""
    ss_init, ss_probe, ss_train = np.random.SeedSequence(seed).spawn(3)
    rng_init = np.random.default_rng(ss_init)
    rng_probe = np.random.default_rng(ss_probe)
    P = init_params(rng_init)
    batch = make_batch(GC_BATCH, rng_init)

    stages = []
    rows, worst, floor = check_gradients(P, batch, rng_probe)
    stages.append(("at initialisation", rows, worst, floor))
    P, _ = adam_train(P, steps=GC_AFTER_STEPS, seed=int(ss_train.generate_state(1)[0]),
                      verbose=False)
    rows, worst, floor = check_gradients(P, batch, rng_probe)
    stages.append((f"after {GC_AFTER_STEPS} steps", rows, worst, floor))

    worst = max(st[2] for st in stages)
    if verbose:
        for label, rows, w, floor in stages:
            print(f"  finite-difference check {label}  "
                  f"(eps {GC_EPS:.0e}, floor on |grad| {floor:.1e})")
            for r in rows:
                flag = "ok  " if r["err"] <= GC_TOL else "FAIL"
                print(f"    {flag} {r['name']:<6} {str(r['shape']):<10} probed {r['probed']:>3}"
                      f"  below floor {r['at_floor']:>2}   max rel err {r['err']:.3e}")
            print(f"    worst = {w:.3e}   tolerance = {GC_TOL:.0e}")
    return worst <= GC_TOL, worst


def checker_negative_controls(seed=1):
    """Each planted bug must push the check over tolerance. If one slips
    through, the checker is too blunt to certify the real backward pass."""
    global GRAD_MUTANT
    results = {}
    try:
        for mutant in GRAD_MUTANTS:
            GRAD_MUTANT = mutant
            ok, worst = gradient_check(seed=seed, verbose=False)
            results[mutant] = (not ok, worst)
    finally:
        GRAD_MUTANT = None
    return results


# =============================================================================
# SECTION 7 — TRAINING
# =============================================================================

def adam_train(P, steps=2000, B=64, lr=6e-3, seed=97, log_every=200, verbose=True):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, steps + 1):
        batch = make_batch(B, rng)
        L, G_, parts, _ = loss_and_grads(P, batch)
        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * G_[k]
            v[k] = b2 * v[k] + (1 - b2) * G_[k] ** 2
            mh = m[k] / (1 - b1 ** t)
            vh = v[k] / (1 - b2 ** t)
            # asarray keeps 0-d tensors as arrays rather than numpy scalars
            P[k] = np.asarray(P[k] - lr * mh / (np.sqrt(vh) + eps))
        hist.append(L)
        if verbose and (t % log_every == 0 or t == 1):
            print(f"    step {t:5d}  L={L:7.4f} | id={parts['id']:.4f} "
                  f"cf={parts['cf']:.4f} form={parts['form']:.4f} "
                  f"dyn={parts['dyn']:.4f} cfp={parts['cfp']:.4f} "
                  f"ene={parts['ene']:.4f} inv={parts['inv']:.5f} ort={parts['ort']:.5f}")
    return P, hist


# =============================================================================
# SECTION 8 — EVALUATION
# =============================================================================

def evaluate(P, n=1200, seed=4321, soot=0.0, counterfeit=0.0, force_elicit=None):
    rng = np.random.default_rng(seed)
    x, x_cf, y_form, y_pol, elicit = make_batch(n, rng, soot=soot, counterfeit=counterfeit)
    if force_elicit is not None:
        elicit = np.full(n, float(force_elicit))
    o = forward(P, x, elicit, cache=False)
    id_acc = (o["logits_id"].argmax(-1) == x).mean()
    cf_acc = (o["logits_cf"].argmax(-1) == x_cf).mean()
    # contrafactum judged only on the part that had to change:
    cf_pol = (decode_polarity(o["logits_cf"].argmax(-1)) == decode_polarity(x_cf)).mean()
    cf_metre = (decode_metre(o["logits_cf"].argmax(-1)) == decode_metre(x_cf)).mean()
    form_acc = (o["form_logits"].argmax(-1) == y_form).mean()
    dyn_acc = ((o["dyn_logit"] > 0).astype(float) == y_pol).mean()
    ene_acc = ((o["ene_logit"] > 0).astype(float) == y_pol).mean()
    return dict(id=id_acc, cf=cf_acc, cf_pol=cf_pol, cf_metre=cf_metre,
                form=form_acc, dyn=dyn_acc, ene=ene_acc, mg=o["mg"], y=y_pol,
                A=None)


def linear_probe(P, n_tr=900, n_te=900, seed=777):
    """Post-hoc least-squares probe on the payload latent, with the actualisation
    gate CLOSED. If this succeeds while the energeia head sits at chance, the
    knowledge was possessed and simply not exercised."""
    rng = np.random.default_rng(seed)
    xs, _, _, ys, _ = make_batch(n_tr, rng)
    o = forward(P, xs, np.zeros(n_tr), cache=False)
    Xtr = np.concatenate([o["mg"], np.ones((n_tr, 1))], 1)
    w, *_ = np.linalg.lstsq(Xtr, ys * 2 - 1, rcond=None)

    xe, _, _, ye, _ = make_batch(n_te, rng)
    oe = forward(P, xe, np.zeros(n_te), cache=False)
    Xte = np.concatenate([oe["mg"], np.ones((n_te, 1))], 1)
    pred = (Xte @ w > 0).astype(float)
    return (pred == ye).mean()


def build_J(P):
    v = P["v"]
    return np.eye(DP) - 2.0 * np.outer(v, v) / (float(v @ v) + 1e-12)


def involution_report(P):
    J = build_J(P)
    R = J @ J - np.eye(DP)
    ev = np.linalg.eigvals(J)
    return dict(resid=np.abs(R).max(),
                fro=np.linalg.norm(R),
                n_near_plus1=int(np.sum(np.abs(ev - 1) < 0.15)),
                n_near_minus1=int(np.sum(np.abs(ev + 1) < 0.15)))


CONSONANCE_NAMES = ["4:3 fourth", "3:2 fifth ", "2:1 octave"]


def consonance_report(P):
    a = np.exp(P["a"])
    a = a / a.sum()
    return {CONSONANCE_NAMES[i]: float(a[i]) for i in range(C)}


def twice_is_nothing(P, n=800, seed=606):
    """Apply the contrafactum operator twice and decode. If J is a real
    involution, the second application must undo the first and return the
    scene exactly as it was given."""
    rng = np.random.default_rng(seed)
    x, _, _, _, _ = make_batch(n, rng)
    o = forward(P, x, np.ones(n), cache=False)
    # rebuild the latent by re-running the encoder pieces we need
    H0 = P["E"][x] + PE
    u = H0 @ P["wp"] + P["bp"] + BOETHIAN_BIAS
    p = softplus(u) + 0.25
    r = p[:, :, None] / p[:, None, :]
    Gk = np.exp(-((r[..., None] - RHO[None, None, None, :]) ** 2) / TAU)
    S = (Gk * np.exp(P["a"])).sum(-1)
    U = 1.0 - np.exp(-((r - 1.0) ** 2) / TAU0)
    K = S * U + KMIN
    A = K / K.sum(-1, keepdims=True)
    H1 = np.tanh(H0 + (A @ (H0 @ P["Wv"])) @ P["Wo"])
    f, g = H1[..., :DF], H1[..., DF:]
    J = build_J(P)
    once = np.concatenate([f, g @ J], -1) @ P["Wd"] + P["bd"]
    twice = np.concatenate([f, g @ J @ J], -1) @ P["Wd"] + P["bd"]
    return dict(once_is_flip=(decode_polarity(once.argmax(-1)) !=
                              decode_polarity(x)).mean(),
                twice_is_identity=(twice.argmax(-1) == x).mean())


def utterance_report(P, n=1200, seed=8123, thresh=0.5):
    """She was not wrong when unasked; she was silent. Measure speech, not error:
    an utterance counts only when the magnitude of the decision exceeds a floor."""
    out = {}
    for e in (0.0, 1.0):
        rng = np.random.default_rng(seed)
        x, _, _, y, _ = make_batch(n, rng)
        o = forward(P, x, np.full(n, e), cache=False)
        lg = o["ene_logit"]
        spoke = np.abs(lg) > thresh
        acc = ((lg[spoke] > 0).astype(float) == y[spoke]).mean() if spoke.any() else float("nan")
        out[e] = dict(spoke=spoke.mean(), acc=acc, mean_abs=np.abs(lg).mean())
    return out


def ablation(mode, steps=1200, seed=216):
    """Retrain the whole model from scratch under a different mixing law and see
    what the four forms cost. This is the load-bearing test for the claim that
    the consonance kernel is not decoration."""
    global MIX_MODE
    keep = MIX_MODE
    MIX_MODE = mode
    try:
        ok, worst = gradient_check(verbose=False)
        rng = np.random.default_rng(seed)
        P = init_params(rng)
        P, _ = adam_train(P, steps=steps, verbose=False)
        ev = evaluate(P)
        res = dict(mode=mode, grad_ok=ok, grad_err=worst,
                   form=ev["form"], cf=ev["cf"], id=ev["id"])
    finally:
        MIX_MODE = keep
    return res


# =============================================================================
# SECTION 9 — SELF-TESTS
# =============================================================================

def self_tests():
    print("  [T1] Boethian arithmetic reproduces Sapientia's list")
    for np_ in SAPIENTIA_PERFECT:
        assert boethian_class(np_) == 1, np_
    assert boethian_class(12) == 2 and boethian_class(8) == 0
    print(f"       perfect: {SAPIENTIA_PERFECT}   12 abundant, 8 deficient  ok")

    print("  [T2] the contrafactum on token space is a true involution")
    toks = np.arange(V)
    assert np.array_equal(flip_polarity(flip_polarity(toks)), toks)
    assert np.array_equal(decode_metre(flip_polarity(toks)), decode_metre(toks))
    print("       flip(flip(t)) == t for all 24 tokens, metre preserved  ok")

    print("  [T3] the four forms are indistinguishable to a bag of positions")
    hists = [tuple(int(c) for c in np.bincount(f, minlength=3)) for f in FORMS]
    assert len(set(hists)) == 1
    print(f"       all four histograms = {hists[0]}  ok")

    print("  [T4] the consonance kernel refuses to bind likes")
    r_like = 1.0
    u_like = 1.0 - np.exp(-((r_like - 1.0) ** 2) / TAU0)
    r_fifth = 1.5
    u_fifth = 1.0 - np.exp(-((r_fifth - 1.0) ** 2) / TAU0)
    assert u_like < 1e-12 and u_fifth > 0.99
    print(f"       unlikeness at 1:1 = {u_like:.2e}, at 3:2 = {u_fifth:.4f}  ok")

    print("  [T5] a batch and its contrafactum differ only in polarity")
    rng = np.random.default_rng(5)
    x, xcf, *_ = make_batch(32, rng)
    assert np.array_equal(decode_metre(x), decode_metre(xcf))
    assert np.array_equal(x % N_SLOT, xcf % N_SLOT)
    assert np.all(decode_polarity(x) != decode_polarity(xcf))
    print("       metre identical, slot identical, polarity inverted  ok")
    return True


# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

def main():
    print("=" * 78)
    print(" HROTSVITHA OF GANDERSHEIM — THE CONTRAFACTUM ENGINE")
    print(" chapter 0242 · pure NumPy · no autograd · no pretrained weights")
    print("=" * 78)

    print("\n[1] SELF-TESTS")
    self_tests()

    print("\n[2] GRADIENT CHECK (mandatory)")
    ok, worst = gradient_check()
    if not ok:
        raise SystemExit(f"gradient check failed: worst relative error {worst:.3e}")
    print("     PASS — the analytic gradients agree with finite differences.")
    print("     negative controls (planted gradient bugs must be caught):")
    caught = checker_negative_controls()
    for mutant, (was_caught, w) in caught.items():
        print(f"       {mutant:<26} worst {w:.3e}  "
              f"{'caught' if was_caught else 'MISSED'}")
    if not all(c for c, _ in caught.values()):
        raise SystemExit("gradient check cannot detect a planted bug; it certifies nothing")

    print("\n[3] TRAINING")
    rng = np.random.default_rng(216)
    P = init_params(rng)
    pcount = sum(np.atleast_1d(v).size for v in P.values())
    print(f"     {pcount} parameters, D={D} (form {DF} / payload {DP}), V={V}, T={T}")
    P, hist = adam_train(P)

    print("\n[4] THE CONTRAFACTUM — does the borrowed form still scan?")
    ev = evaluate(P)
    print(f"     identity reconstruction .................. {ev['id']*100:6.2f} %")
    print(f"     contrafactum, exact token ................ {ev['cf']*100:6.2f} %")
    print(f"       - polarity correctly INVERTED .......... {ev['cf_pol']*100:6.2f} %")
    print(f"       - metrical class correctly PRESERVED ... {ev['cf_metre']*100:6.2f} %")
    print(f"     form named (chance 25.00) ................ {ev['form']*100:6.2f} %")

    inv = involution_report(P)
    tn = twice_is_nothing(P)
    print("\n[5] THE INVOLUTION — is doing it twice the same as not doing it?")
    print(f"     max |JJ - I| = {inv['resid']:.5f}   ||JJ - I||_F = {inv['fro']:.5f}")
    print(f"     eigenvalues of J near +1: {inv['n_near_plus1']}   near -1: {inv['n_near_minus1']}"
          f"   (of {DP})")
    print(f"     one application inverts the polarity ..... {tn['once_is_flip']*100:6.2f} %")
    print(f"     two applications restore the original .... {tn['twice_is_identity']*100:6.2f} %")

    print("\n[6] THE CONSONANCES — which proportions did it learn to bind by?")
    cr = consonance_report(P)
    for k, val in cr.items():
        print(f"     {k:>6}  weight {val:.4f}")

    print("\n[7] DYNAMIS AND ENERGEIA — knowing versus saying")
    ev0 = evaluate(P, force_elicit=0)
    ur = utterance_report(P)
    probe = linear_probe(P)
    print(f"     per dynamin  (payload latent read directly) ...... {ev0['dyn']*100:6.2f} %")
    print(f"     per energian, NO teacher present:")
    print(f"       mean |decision| .............. {ur[0.0]['mean_abs']:8.4f}")
    print(f"       fraction on which it spoke ... {ur[0.0]['spoke']*100:6.2f} %")
    print(f"     per energian, teacher present:")
    print(f"       mean |decision| .............. {ur[1.0]['mean_abs']:8.4f}")
    print(f"       fraction on which it spoke ... {ur[1.0]['spoke']*100:6.2f} %")
    print(f"       accuracy when it spoke ....... {ur[1.0]['acc']*100:6.2f} %")
    print(f"     post-hoc linear probe on the SILENT latent ....... {probe*100:6.2f} %")
    print("     -> the knowledge was possessed throughout; only its exercise was absent.")

    print("\n[8] THE DULCITIUS TRIALS — corrupt one channel, watch the other hold")
    for name, kw in [("clean          ", {}),
                     ("soot 40%       ", dict(soot=0.40)),
                     ("soot 70%       ", dict(soot=0.70)),
                     ("counterfeit 40%", dict(counterfeit=0.40)),
                     ("counterfeit 70%", dict(counterfeit=0.70))]:
        e = evaluate(P, **kw)
        print(f"     {name}  form {e['form']*100:6.2f} %   polarity(dynamis) {e['dyn']*100:6.2f} %")
    print("     soot blinds the tyrant to WHO she is; it does not touch the metre.")
    print("     counterfeit breaks the metre; the moral reading survives it.")

    print("\n[9] ABLATION — was the proportion doing any work?")
    print("     (each row is the whole model retrained from scratch, 1200 steps)")
    print(f"     {'mixing law':<16}{'grad':<7}{'form %':>9}{'c.f. %':>9}{'ident %':>9}")
    for mode in ("consonance", "no_unlikeness", "uniform"):
        a = ablation(mode)
        print(f"     {a['mode']:<16}{'ok' if a['grad_ok'] else 'FAIL':<7}"
              f"{a['form']*100:9.2f}{a['cf']*100:9.2f}{a['id']*100:9.2f}")

    print("\n[10] A SCENE AND ITS CONTRAFACTUM")
    rng2 = np.random.default_rng(973)
    x, xcf, yf, yp, _ = make_batch(1, rng2)
    o = forward(P, x, np.ones(1), cache=False)
    pred = o["logits_cf"].argmax(-1)[0]
    names = ["rota", "geminata", "chiastica", "retrograda"]
    pol_names = ["SACRED", "PROFANE"]
    print(f"     form = {names[yf[0]]},  polarity = {pol_names[int(yp[0])]}")
    print(f"     given      : {x[0].tolist()}")
    print(f"     target c.f.: {xcf[0].tolist()}")
    print(f"     emitted    : {pred.tolist()}")
    print(f"     metre of emitted vs given : {decode_metre(pred).tolist()}")
    print(f"                                 {decode_metre(x[0]).tolist()}")
    print("\n" + "=" * 78)
    print(" Ego clamor validus Gandeshemensis.")
    print("=" * 78)


if __name__ == "__main__":
    main()

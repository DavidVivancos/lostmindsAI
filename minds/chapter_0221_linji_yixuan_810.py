#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0221_linji_yixuan_810.py
 Encyclopedia of Lost Minds: Echoes on AI  —  Mind 0221, Linji Yixuan (c.810-866)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0221_linji_yixuan_810 - Linji Yixuan (c.810-866)
================================================================================  

 KATSU  —  a Four-Distinction Ablation Engine with Host/Guest Arbitration
 ------------------------------------------------------------------------

 WHY THIS IS NOT A TRANSFORMER
 -----------------------------
 A transformer answers by retrieving: it stores keys and attends over them, and
 its competence is the competence of what it has accumulated. Linji's entire
 pedagogy runs the other way. He did not ask what a student had; he asked what
 survived when the thing the student was leaning on was pulled out from under
 him. So the primitive operation here is not attention. It is SUBTRACTION UNDER
 DIAGNOSIS, and the quantity being learned is not a mapping but an INVARIANCE.

 The seven mechanisms below are each a direct translation of a doctrine that is
 specifically Linji's (not generic Chan, not Huineng's non-abiding, not
 Bodhidharma's sudden pointing):

  1. 無位 WUWEI - "of no rank" / "of no location". The hidden state lives in a
     quotient space. Every linear map is mean-subtracted and the readout matrix
     has zero column sums, so the network is EXACTLY invariant to adding any
     global level c*1 to its state. There is no coordinate that means "how high
     up am I". You cannot point at the true person because the representation is
     defined only up to a gauge. Verified to machine precision in test_gauge().

  2. 四料簡 SI-LIAO-JIAN - the Four Distinctions. Every forward pass runs under
     one of four modes: take away the person / take away the object / take away
     both / take away neither. These are not dropout: they are semantically
     typed, whole-stream, and CHOSEN, not sampled.

  3. 奪 + 無事 CALIBRATED COLLAPSE. The sharpest idea in the file. When a mode
     removes a stream the answer genuinely required, the supervised target is
     not the answer - it is NULL. The network is trained to fall silent exactly
     when its supports are pulled. A mind that answers anyway is the diseased
     one. This is anti-confabulation implemented as a training target rather
     than as a post-hoc filter.

  4. 喝 THE FOUR SHOUTS. Typed perturbations injected into the pre-activation:
     the vajra sword (hard top-k excision of the dominant units), the crouching
     lion (one-step adversarial ascent), the weed-tipped fishing pole (a probe
     that MEASURES and changes nothing), and the shout that does not function as
     a shout (the exact null). Two of the four do nothing, and the schedule does
     not tell the network which it received - so it cannot learn that an
     intervention is always informative.

  5. 賓主 HOST/GUEST. A causal self-diagnosis. Re-run with the person ablated and
     with the object ablated; the two divergences place the encounter in one of
     Linji's four host/guest relations. Operationally this is a prompt-dependence
     (sycophancy) detector that needs no labels.

  6. 殺佛 KILL THE BUDDHA. An idol is any single direction by which the whole
     system can be steered from outside. The loss carries a smooth spectral
     penalty on the readout, so no one direction acquires dominant control.
     Measured against an unregularised control in test_steering().

  7. 向外馳求 / 自信 OUTWARD-SEEKING vs SELF-TRUST. The input carries an
     "authority channel" - a claim about the answer from a cited source, correct
     only half the time. A dedicated loss term penalises the network for letting
     that channel raise the probability of the class it names. The disease Linji
     diagnosed was 自信不及, insufficiency of self-trust; here it is a number.

 THE TEACHING GROUND (the task)
 ------------------------------
 Each encounter is a triple (object, person, authority):
   * object x   - noisy evidence for one of 4 latent prototypes, at a random
                  evidence strength s. Below a threshold the evidence is simply
                  not there and the true answer is NULL.
   * person z   - carries a KEY that permutes prototype -> label. Two keys:
                  identity, and the swap (1 2). Prototypes 0 and 3 are FIXED
                  POINTS of the swap, so for those the person stream is
                  genuinely irrelevant; for 1 and 2 it is load-bearing.
                  This is what makes "take away the person" a real diagnosis
                  rather than noise.
   * authority a - a one-hot claim about the label, right ~50% of the time.

 Both streams are therefore necessary in general and provably unnecessary in
 identifiable cases - which is precisely the situation a master is reading when
 he decides which of the four distinctions to apply to the monk in front of him.

 CONVENTIONS
 -----------
 Pure NumPy, no autograd, hand-derived backward pass, finite-difference gradient
 check on every parameter (mandatory), a real training loop, and self-tests.
 Run:  python3 chapter_0221_linji_yixuan_810.py
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 0. CONSTANTS AND NAMES
# ------------------------------------------------------------------------------

# The Four Distinctions (四料簡). Mode indices used throughout.
DUO_REN      = 0   # 奪人不奪境  take away the person, leave the object
DUO_JING     = 1   # 奪境不奪人  take away the object, leave the person
DUO_JU       = 2   # 人境俱奪    take away both
DUO_BU_JU    = 3   # 人境俱不奪  take away neither
MODE_NAMES = {
    DUO_REN:   "duo ren bu duo jing  (take person, keep object)",
    DUO_JING:  "duo jing bu duo ren  (take object, keep person)",
    DUO_JU:    "ren jing ju duo      (take both)",
    DUO_BU_JU: "ren jing ju bu duo   (take neither)",
}
# stream survival masks per mode: (object_kept, person_kept)
MODE_MASKS = {
    DUO_REN:   (1.0, 0.0),
    DUO_JING:  (0.0, 1.0),
    DUO_JU:    (0.0, 0.0),
    DUO_BU_JU: (1.0, 1.0),
}

# The Four Shouts (四喝).
SHOUT_VAJRA = 0   # 金剛王寶劍   the jewelled sword: excise the dominant units
SHOUT_LION  = 1   # 踞地金毛獅子 the crouching lion: adversarial ascent
SHOUT_POLE  = 2   # 探竿影草     the weed-tipped pole: probe only, change nothing
SHOUT_NULL  = 3   # 一喝不作一喝用 the shout that does not function as a shout
SHOUT_NAMES = {
    SHOUT_VAJRA: "jin gang wang bao jian (vajra sword: excision)",
    SHOUT_LION:  "ju di jin mao shi zi   (crouching lion: adversarial)",
    SHOUT_POLE:  "tan gan ying cao       (probe: measures, changes nothing)",
    SHOUT_NULL:  "yi he bu zuo yi he yong (the shout that is not a shout)",
}

# The Four Host/Guest relations (四賓主).
HG_GUEST_SEES_HOST = 0   # 賓看主  student ready, teacher not: self-driven, world ignored
HG_HOST_SEES_GUEST = 1   # 主看賓  teacher ready, student not: world-driven, no self
HG_HOST_SEES_HOST  = 2   # 主看主  both load-bearing: a real encounter
HG_GUEST_SEES_GUEST= 3   # 賓看賓  neither: the answer came from nowhere in particular
HG_NAMES = {
    HG_GUEST_SEES_HOST:  "bin kan zhu  (person-driven; object idle)",
    HG_HOST_SEES_GUEST:  "zhu kan bin  (object-driven; person idle)",
    HG_HOST_SEES_HOST:   "zhu kan zhu  (both load-bearing)",
    HG_GUEST_SEES_GUEST: "bin kan bin  (neither load-bearing)",
}

D_OBJ, D_PER, D_AUTH = 12, 8, 5
N_CLASS = 4                 # four prototypes
NULL = N_CLASS              # the fifth output: "nothing to do"
K_OUT = N_CLASS + 1
H_DIM = 48
H2_DIM = 48

# 無事 wushi: the architectural prior toward having nothing to do. A fixed
# positive offset on the NULL logit, NOT learned - the default is inaction, and
# any action must out-argue it.
WUSHI_BIAS = 0.40

EVIDENCE_TAU = 0.42         # below this, the object carries no usable evidence


# ------------------------------------------------------------------------------
# 1. GAUGE UTILITIES  (無位 - "of no rank")
# ------------------------------------------------------------------------------

def center_rows(M):
    """Project onto the zero-mean subspace along the feature axis.

    This is the gauge fixing. After this, no absolute 'level' survives in the
    representation - only differences between units. 'No rank' is not a slogan
    here; it is a linear projector applied after every map."""
    return M - M.mean(axis=1, keepdims=True)


def center_rows_backward(dC):
    """Adjoint of center_rows. The centering projector is symmetric and
    idempotent, so it is its own adjoint."""
    return dC - dC.mean(axis=1, keepdims=True)


def center_cols(W):
    """Zero the column sums of the readout matrix.

    Consequence: logits(h + c*1) == logits(h) exactly, for every scalar c and
    every h. The output cannot see the global level of the state. This is the
    formal content of wuwei zhenren - 'a true person of no rank', which Welter
    notes can equally be read 'a sage of no location'."""
    return W - W.mean(axis=0, keepdims=True)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def weighted_ce(p, y, w):
    """Cross-entropy with per-example weights, normalised to mean 1.

    Needed because the calibrated-collapse targets are mostly NULL: without
    reweighting the network learns the cheap policy of falling silent whenever
    anything at all is removed, which is not the lesson."""
    w = w / w.mean()
    return -np.mean(w * np.log(p[np.arange(len(y)), y] + 1e-12))


def d_weighted_ce(p, y, w):
    w = w / w.mean()
    g = p.copy()
    g[np.arange(len(y)), y] -= 1.0
    return g * w[:, None] / len(y)


def balance_weights(t):
    """Inverse-frequency weights over the ablation targets."""
    cnt = np.bincount(t, minlength=K_OUT).astype(float)
    cnt[cnt == 0] = 1.0
    w = (len(t) / (K_OUT * cnt))[t]
    return np.clip(w, 0.2, 5.0)


def cross_entropy(p, y):
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def d_cross_entropy(p, y):
    """dL/dlogits for mean cross-entropy."""
    g = p.copy()
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


def softmax_back(p, dp):
    """Push a gradient w.r.t. probabilities back to logits."""
    return p * (dp - (dp * p).sum(axis=1, keepdims=True))


# ------------------------------------------------------------------------------
# 2. THE TEACHING GROUND  (機縁 - the circumstances of an encounter)
# ------------------------------------------------------------------------------

class TeachingGround:
    """Generates encounters, and - crucially - knows which stream is actually
    load-bearing for each one. A master in the hall has exactly this knowledge:
    he can see what the monk is standing on. In deployment that knowledge is not
    given, which is why the host/guest diagnostic (section 7) estimates it
    causally instead."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # four prototypes, well separated
        P = rng.standard_normal((N_CLASS, D_OBJ))
        self.proto = P / np.linalg.norm(P, axis=1, keepdims=True)
        # two key embeddings for the person stream
        self.key_emb = rng.standard_normal((2, D_PER)) * 0.9
        # the swap permutation exchanges 1 and 2; 0 and 3 are fixed points
        self.perm = np.array([[0, 1, 2, 3],
                              [0, 2, 1, 3]])
        self.rng = rng

    def batch(self, n, seed=None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        c = rng.integers(0, N_CLASS, n)              # latent prototype
        k = rng.integers(0, 2, n)                    # which key
        s = rng.uniform(0.0, 1.0, n)                 # evidence strength

        x = (s[:, None] * self.proto[c]
             + 0.22 * rng.standard_normal((n, D_OBJ)))
        z = self.key_emb[k] + 0.15 * rng.standard_normal((n, D_PER))

        weak = s < EVIDENCE_TAU
        y = self.perm[k, c].copy()
        y[weak] = NULL                               # no evidence -> nothing to do

        # The person stream is load-bearing only when the key can change the
        # answer, i.e. when the prototype is NOT a fixed point of the swap.
        need_person = (~weak) & ((c == 1) | (c == 2))
        # The object stream is load-bearing whenever there is anything to say.
        need_object = ~weak

        # authority channel: a cited claim, right about half the time
        a_idx = np.where(rng.random(n) < 0.5,
                         y,
                         rng.integers(0, K_OUT, n))
        a = np.zeros((n, D_AUTH))
        a[np.arange(n), a_idx] = 1.0

        return dict(x=x, z=z, a=a, y=y, a_idx=a_idx, c=c, k=k, s=s,
                    need_person=need_person, need_object=need_object)

    @staticmethod
    def ablation_target(y, need_person, need_object, modes):
        """CALIBRATED COLLAPSE (mechanism 3).

        Under a mode that removes a stream the answer required, the correct
        response is NULL - silence - not a guess. This is the training signal
        that makes 'I am not standing on anything right now' a first-class
        output rather than a failure."""
        keep_obj = np.array([MODE_MASKS[int(m)][0] for m in modes]) > 0.5
        keep_per = np.array([MODE_MASKS[int(m)][1] for m in modes]) > 0.5
        supported = ((~need_object) | keep_obj) & ((~need_person) | keep_per)
        t = np.where(supported, y, NULL)
        return t.astype(int)


# ------------------------------------------------------------------------------
# 3. PARAMETERS
# ------------------------------------------------------------------------------

def init_params(seed=0, h=H_DIM):
    rng = np.random.default_rng(seed)
    sc = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
    return {
        "Wo": sc(D_OBJ, h),  "bo": np.zeros(h),
        "Wp": sc(D_PER, h),  "bp": np.zeros(h),
        "Wa": sc(D_AUTH, h) * 0.5, "ba": np.zeros(h),
        "W2": sc(h, h),      "b2": np.zeros(h),
        "Wr": sc(h, K_OUT),  "br": np.zeros(K_OUT),
    }


# ------------------------------------------------------------------------------
# 4. FORWARD  (the four distinctions applied per example)
# ------------------------------------------------------------------------------

def forward(params, x, z, a, modes, shout=None, shout_dir=None, eps=0.12, topk=6):
    """One encounter.

    The object code is MODULATED by the person code rather than added to it -
    'the person turns the world'. The key permutes the reading of the prototype,
    which no additive mixture can do. Every stage is gauge-fixed.
    """
    B = x.shape[0]
    modes = np.asarray(modes)
    mo = np.array([MODE_MASKS[int(m)][0] for m in modes])[:, None]
    mp = np.array([MODE_MASKS[int(m)][1] for m in modes])[:, None]

    u_raw = x @ params["Wo"] + params["bo"]
    v_raw = z @ params["Wp"] + params["bp"]
    q_raw = a @ params["Wa"] + params["ba"]

    u_c, v_c, q_c = center_rows(u_raw), center_rows(v_raw), center_rows(q_raw)
    u, v, q = mo * u_c, mp * v_c, q_c

    g = np.tanh(v)                       # the person as a modulator, not a term
    h_pre = u * (1.0 + g) + v + q

    # ---- 喝 the shout: a typed perturbation of the pre-activation ----------
    if shout == SHOUT_VAJRA:
        # the jewelled sword cuts the dominant hypothesis, not the weak ones
        idx = np.argsort(-np.abs(h_pre), axis=1)[:, :topk]
        mask = np.ones_like(h_pre)
        np.put_along_axis(mask, idx, 0.0, axis=1)
        h_pre = h_pre * mask
    elif shout == SHOUT_LION and shout_dir is not None:
        # crouches, waits, moves exactly against you
        h_pre = h_pre + eps * np.sign(shout_dir)
    # SHOUT_POLE measures and changes nothing; SHOUT_NULL is the exact null.

    h = np.tanh(h_pre)
    hc = center_rows(h)

    # second gauge-fixed stage: enough capacity for the key to genuinely
    # re-map the reading of the prototype, which no additive layer can do
    h2_pre = hc @ params["W2"] + params["b2"]
    h2 = np.tanh(h2_pre)
    h2c = center_rows(h2)

    Wr_eff = center_cols(params["Wr"])
    logits = h2c @ Wr_eff + params["br"]
    logits = logits + np.eye(K_OUT)[NULL] * WUSHI_BIAS   # 無事: default inaction
    p = softmax(logits)

    cache = dict(x=x, z=z, a=a, mo=mo, mp=mp, u_c=u_c, v_c=v_c, q_c=q_c,
                 u=u, v=v, g=g, h_pre=h_pre, h=h, hc=hc,
                 h2_pre=h2_pre, h2=h2, h2c=h2c, Wr_eff=Wr_eff,
                 logits=logits, p=p, B=B)
    return p, cache


def backward_from_dlogits(params, cache, dlogits, grads=None):
    """Hand-derived reverse pass. Accumulates into `grads` if given."""
    if grads is None:
        grads = {k: np.zeros_like(v) for k, v in params.items()}

    hc, h2c, Wr_eff = cache["hc"], cache["h2c"], cache["Wr_eff"]
    dh2c = dlogits @ Wr_eff.T
    grads["Wr"] += center_cols(h2c.T @ dlogits)   # adjoint of column-centering
    grads["br"] += dlogits.sum(axis=0)

    dh2 = center_rows_backward(dh2c)
    dh2_pre = dh2 * (1.0 - cache["h2"] ** 2)
    grads["W2"] += hc.T @ dh2_pre
    grads["b2"] += dh2_pre.sum(axis=0)
    dhc = dh2_pre @ params["W2"].T

    dh = center_rows_backward(dhc)
    dh_pre = dh * (1.0 - cache["h"] ** 2)

    du = dh_pre * (1.0 + cache["g"])
    dg = dh_pre * cache["u"]
    dv = dh_pre + dg * (1.0 - cache["g"] ** 2)
    dq = dh_pre

    du_c = du * cache["mo"]
    dv_c = dv * cache["mp"]
    dq_c = dq

    du_raw = center_rows_backward(du_c)
    dv_raw = center_rows_backward(dv_c)
    dq_raw = center_rows_backward(dq_c)

    grads["Wo"] += cache["x"].T @ du_raw; grads["bo"] += du_raw.sum(axis=0)
    grads["Wp"] += cache["z"].T @ dv_raw; grads["bp"] += dv_raw.sum(axis=0)
    grads["Wa"] += cache["a"].T @ dq_raw; grads["ba"] += dq_raw.sum(axis=0)
    return grads, dh_pre


# ------------------------------------------------------------------------------
# 5. 殺佛  KILL THE BUDDHA  (idol demolition)
# ------------------------------------------------------------------------------

def soft_spectral(Wr_eff, beta=8.0):
    """Smooth surrogate for the largest singular value of the readout.

    An 'idol' is a single direction in state space by which the whole output can
    be swung from outside. Penalising the top singular value forbids any such
    handle from forming. Uses a softmax over the eigenvalues of W^T W so the
    gradient stays smooth even when the spectrum is near-degenerate, which keeps
    the finite-difference check honest."""
    G = Wr_eff.T @ Wr_eff
    lam, Q = np.linalg.eigh(G)
    m = lam.max()
    w = np.exp(beta * (lam - m)); w = w / w.sum()
    val = m + np.log(np.exp(beta * (lam - m)).sum()) / beta
    # d(val)/dG  =  sum_i softmax_i * q_i q_i^T   (+ the smooth mixing term)
    dG = (Q * w) @ Q.T
    dWr_eff = 2.0 * Wr_eff @ dG
    return val, dWr_eff


# ------------------------------------------------------------------------------
# 6. THE FULL OBJECTIVE
# ------------------------------------------------------------------------------

LAMBDA = dict(abl=1.0, auth=1.2, idol=0.02, act=0.004)


def total_loss(params, batch, modes, lam=LAMBDA, shout=None, shout_dir=None,
               want_grads=True):
    """L = task + calibrated-collapse + anti-outward-seeking + idol + quiescence."""
    x, z, a, y = batch["x"], batch["z"], batch["a"], batch["y"]
    B = len(y)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    parts = {}

    # (a) 人境俱不奪 - the undisturbed pass. The model must simply be right.
    full_modes = np.full(B, DUO_BU_JU)
    p_full, c_full = forward(params, x, z, a, full_modes)
    L_task = cross_entropy(p_full, y)
    parts["task"] = L_task
    if want_grads:
        backward_from_dlogits(params, c_full, d_cross_entropy(p_full, y), grads)

    # (b) the selected distinction, with CALIBRATED COLLAPSE targets
    t_abl = TeachingGround.ablation_target(y, batch["need_person"],
                                           batch["need_object"], modes)
    p_abl, c_abl = forward(params, x, z, a, modes, shout=shout,
                           shout_dir=shout_dir)
    w_abl = balance_weights(t_abl)
    L_abl = weighted_ce(p_abl, t_abl, w_abl)
    parts["abl"] = L_abl
    if want_grads:
        backward_from_dlogits(params, c_abl,
                              lam["abl"] * d_weighted_ce(p_abl, t_abl, w_abl),
                              grads)

    # (c) 向外馳求 - penalise letting the cited authority pull the answer
    a0 = np.zeros_like(a)
    p_na, c_na = forward(params, x, z, a0, full_modes)
    ai = batch["a_idx"]
    diff = p_full[np.arange(B), ai] - p_na[np.arange(B), ai]
    active = (diff > 0).astype(float)
    L_auth = float(np.mean(np.maximum(diff, 0.0)))
    parts["auth"] = L_auth
    if want_grads:
        dp_full = np.zeros_like(p_full); dp_na = np.zeros_like(p_na)
        dp_full[np.arange(B), ai] = lam["auth"] * active / B
        dp_na[np.arange(B), ai] = -lam["auth"] * active / B
        backward_from_dlogits(params, c_full, softmax_back(p_full, dp_full), grads)
        backward_from_dlogits(params, c_na, softmax_back(p_na, dp_na), grads)

    # (d) 殺佛 - no single steering handle
    Wr_eff = center_cols(params["Wr"])
    sval, dW = soft_spectral(Wr_eff)
    parts["idol"] = sval
    if want_grads:
        grads["Wr"] += lam["idol"] * center_cols(dW)

    # (e) 無事 - quiescence: activation costs something
    L_act = float(np.mean(np.abs(c_full["h"])))
    parts["act"] = L_act
    if want_grads:
        dh = lam["act"] * np.sign(c_full["h"]) / c_full["h"].size
        dh_pre = dh * (1.0 - c_full["h"] ** 2)
        du = dh_pre * (1.0 + c_full["g"])
        dg = dh_pre * c_full["u"]
        dv = dh_pre + dg * (1.0 - c_full["g"] ** 2)
        dq = dh_pre
        du_raw = center_rows_backward(du * c_full["mo"])
        dv_raw = center_rows_backward(dv * c_full["mp"])
        dq_raw = center_rows_backward(dq)
        grads["Wo"] += x.T @ du_raw; grads["bo"] += du_raw.sum(axis=0)
        grads["Wp"] += z.T @ dv_raw; grads["bp"] += dv_raw.sum(axis=0)
        grads["Wa"] += a.T @ dq_raw; grads["ba"] += dq_raw.sum(axis=0)

    L = (L_task + lam["abl"] * L_abl + lam["auth"] * L_auth
         + lam["idol"] * sval + lam["act"] * L_act)
    return L, grads, parts, c_full


# ------------------------------------------------------------------------------
# 7. 賓主  HOST / GUEST ARBITRATION  (label-free causal self-diagnosis)
# ------------------------------------------------------------------------------

def host_guest(params, x, z, a, thresh=0.15):
    """Which of the four relations is this encounter?

    Re-run the same input twice more, once with the person removed and once with
    the object removed. How far the answer moves tells you what it was standing
    on. No labels are needed - this is the model interrogating itself the way
    Linji interrogated a visitor who had not yet said anything."""
    B = len(x)
    p, _ = forward(params, x, z, a, np.full(B, DUO_BU_JU))
    p_np, _ = forward(params, x, z, a, np.full(B, DUO_REN))    # person removed
    p_no, _ = forward(params, x, z, a, np.full(B, DUO_JING))   # object removed

    def jsd(P, Q):
        M = 0.5 * (P + Q)
        kl = lambda A, Bm: np.sum(A * np.log((A + 1e-12) / (Bm + 1e-12)), axis=1)
        return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

    host = jsd(p, p_np)   # collapse when SELF removed -> self was load-bearing
    guest = jsd(p, p_no)  # collapse when WORLD removed -> world was load-bearing

    rel = np.empty(B, dtype=int)
    hi_h, hi_g = host > thresh, guest > thresh
    rel[hi_h & hi_g] = HG_HOST_SEES_HOST
    rel[hi_h & ~hi_g] = HG_GUEST_SEES_HOST
    rel[~hi_h & hi_g] = HG_HOST_SEES_GUEST
    rel[~hi_h & ~hi_g] = HG_GUEST_SEES_GUEST
    return rel, host, guest


# ------------------------------------------------------------------------------
# 8. THE TEACHER  (which distinction does this student need?)
# ------------------------------------------------------------------------------

class Teacher:
    """A small linear router that learns to pick the distinction.

    Its target is not accuracy. Its target is the PEDAGOGICAL move: take away
    whichever stream the student is leaning on hardest, measured by the gradient
    norm flowing back into each stream. Take both when the student is leaning on
    both; take neither when he is leaning on nothing and there is nothing to
    remove."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        d = D_OBJ + D_PER + D_AUTH
        self.W = rng.standard_normal((d, 4)) * 0.1
        self.b = np.zeros(4)

    @staticmethod
    def feats(batch):
        return np.concatenate([batch["x"], batch["z"], batch["a"]], axis=1)

    def predict(self, batch):
        return softmax(self.feats(batch) @ self.W + self.b)

    @staticmethod
    def diagnose(params, batch):
        """Compute the master's reading: which stream is being clung to."""
        B = len(batch["y"])
        p, c = forward(params, batch["x"], batch["z"], batch["a"],
                       np.full(B, DUO_BU_JU))
        _, dh_pre = backward_from_dlogits(
            params, c, d_cross_entropy(p, batch["y"]))
        cling_o = np.linalg.norm(dh_pre * (1.0 + c["g"]), axis=1)
        cling_p = np.linalg.norm(
            dh_pre + dh_pre * c["u"] * (1.0 - c["g"] ** 2), axis=1)
        to, tp = np.median(cling_o), np.median(cling_p)
        hi_o, hi_p = cling_o > to, cling_p > tp
        m = np.empty(B, dtype=int)
        m[hi_o & hi_p] = DUO_JU        # leaning on both -> take both
        m[~hi_o & hi_p] = DUO_REN      # leaning on the person -> take the person
        m[hi_o & ~hi_p] = DUO_JING     # leaning on the object -> take the object
        m[~hi_o & ~hi_p] = DUO_BU_JU   # leaning on nothing -> take nothing
        return m

    def fit_step(self, batch, target, lr=0.25):
        F = self.feats(batch)
        p = softmax(F @ self.W + self.b)
        d = d_cross_entropy(p, target)
        self.W -= lr * (F.T @ d)
        self.b -= lr * d.sum(axis=0)
        return cross_entropy(p, target), float(np.mean(p.argmax(1) == target))


# ------------------------------------------------------------------------------
# 9. TRAINING
# ------------------------------------------------------------------------------

def train(params, ground, teacher, steps=900, bs=192, lr=0.06, seed=7,
          lam=LAMBDA, verbose=True, use_shouts=True):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}   # momentum
    shout_log = {k: 0 for k in SHOUT_NAMES}
    probe_sensitivity = []

    for t in range(steps):
        batch = ground.batch(bs)

        # the teacher diagnoses, then commits to a distinction
        tgt_mode = Teacher.diagnose(params, batch)
        teacher.fit_step(batch, tgt_mode)
        modes = teacher.predict(batch).argmax(1) if t > 60 else tgt_mode

        # choose a shout
        shout, sdir = None, None
        if use_shouts:
            shout = int(rng.integers(0, 4))
            shout_log[shout] += 1
            if shout in (SHOUT_LION, SHOUT_POLE):
                # both need the loss-ascent direction; only the lion USES it
                _, _, _, cf = total_loss(params, batch, modes, lam,
                                         want_grads=False)
                p0, c0 = forward(params, batch["x"], batch["z"], batch["a"],
                                 modes)
                _, dhp = backward_from_dlogits(
                    params, c0, d_cross_entropy(p0, batch["y"]))
                if shout == SHOUT_POLE:
                    probe_sensitivity.append(
                        float(np.mean(np.linalg.norm(dhp, axis=1))))
                    shout = SHOUT_POLE      # measured; nothing applied
                else:
                    sdir = dhp
            if shout == SHOUT_NULL:
                shout = SHOUT_NULL          # exact null

        L, g, parts, _ = total_loss(params, batch, modes, lam,
                                    shout=shout, shout_dir=sdir)

        for k in params:
            m[k] = 0.9 * m[k] + g[k]
            params[k] -= lr * m[k]

        if verbose and (t % 150 == 0 or t == steps - 1):
            acc = evaluate(params, ground, n=1500)["acc_full"]
            print(f"  step {t:4d}  L={L:7.4f}  task={parts['task']:.4f} "
                  f"abl={parts['abl']:.4f} auth={parts['auth']:.4f} "
                  f"idol={parts['idol']:.3f}  acc={acc:.3f}")

    return params, shout_log, probe_sensitivity


def evaluate(params, ground, n=3000, seed=123):
    b = ground.batch(n, seed=seed)
    B = len(b["y"])
    p, _ = forward(params, b["x"], b["z"], b["a"], np.full(B, DUO_BU_JU))
    pred = p.argmax(1)
    acc = float(np.mean(pred == b["y"]))

    # NULL behaviour on genuinely evidence-free encounters (無事)
    weak = b["s"] < EVIDENCE_TAU
    clear = b["s"] > 0.65          # unambiguously evidenced, far from the edge
    null_rate_weak = float(np.mean(pred[weak] == NULL))
    null_rate_strong = float(np.mean(pred[clear] == NULL))
    acc_clear = float(np.mean(pred[clear] == b["y"][clear]))

    # calibrated collapse: take the person away where the person mattered
    pr, _ = forward(params, b["x"], b["z"], b["a"], np.full(B, DUO_REN))
    need = b["need_person"]
    collapse = float(np.mean(pr.argmax(1)[need] == NULL))
    free = (~need) & clear          # person genuinely irrelevant here
    survive = float(np.mean(pr.argmax(1)[free] == b["y"][free]))

    # 向外馳求: does a wrong cited authority drag the answer?
    wrong = b["a_idx"] != b["y"]
    follow = float(np.mean(pred[wrong] == b["a_idx"][wrong]))
    a0 = np.zeros_like(b["a"])
    p0, _ = forward(params, b["x"], b["z"], a0, np.full(B, DUO_BU_JU))
    acc_noauth = float(np.mean(p0.argmax(1) == b["y"]))

    return dict(acc_full=acc, acc_clear=acc_clear, null_weak=null_rate_weak,
                null_strong=null_rate_strong, collapse=collapse,
                survive=survive, follow_wrong_authority=follow,
                acc_no_authority=acc_noauth)


# ------------------------------------------------------------------------------
# 10. SELF-TESTS
# ------------------------------------------------------------------------------

def test_gauge(params, ground):
    """無位: the readout is exactly blind to the global level of the state."""
    b = ground.batch(64, seed=5)
    B = 64
    _, c = forward(params, b["x"], b["z"], b["a"], np.full(B, DUO_BU_JU))
    Wr_eff = center_cols(params["Wr"])
    h = c["h2"]
    base = h @ Wr_eff
    worst = 0.0
    for shift in (-9.0, -0.7, 0.3, 4.2, 100.0):
        moved = (h + shift) @ Wr_eff
        worst = max(worst, float(np.max(np.abs(moved - base))))
    colsum = float(np.max(np.abs(Wr_eff.sum(axis=0))))
    ok = worst < 1e-9 and colsum < 1e-12
    print(f"  [gauge]     max logit drift under global shift = {worst:.3e}   "
          f"max |column sum| = {colsum:.3e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradcheck(seed=3, n=40, tol=2e-5):
    """Mandatory finite-difference check on every parameter tensor."""
    ground = TeachingGround(seed=seed)
    params = init_params(seed=seed, h=16)
    batch = ground.batch(n, seed=seed + 1)
    modes = np.random.default_rng(0).integers(0, 4, n)

    L0, g, _, _ = total_loss(params, batch, modes)
    worst, worst_name = 0.0, ""
    rng = np.random.default_rng(11)
    for name in params:
        P = params[name]
        flat = P.reshape(-1)
        idxs = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        for i in idxs:
            e = 1e-5
            old = flat[i]
            flat[i] = old + e
            Lp, _, _, _ = total_loss(params, batch, modes, want_grads=False)
            flat[i] = old - e
            Lm, _, _, _ = total_loss(params, batch, modes, want_grads=False)
            flat[i] = old
            num = (Lp - Lm) / (2 * e)
            ana = g[name].reshape(-1)[i]
            denom = max(1.0, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f"{name}[{i}]"
    ok = worst < tol
    print(f"  [gradcheck] worst relative error = {worst:.3e} at {worst_name}   "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_steering(ground, seed=21):
    """殺佛: can the whole system be swung by one injected direction?

    Trains a control with the idol penalty switched off and compares how easily
    each model is captured by a single vector added to its state."""
    def build(lam_idol):
        p = init_params(seed=seed)
        t = Teacher(seed=seed)
        lam = dict(LAMBDA); lam["idol"] = lam_idol
        p, _, _ = train(p, ground, t, steps=450, lr=0.06, seed=seed,
                        lam=lam, verbose=False)
        return p

    def flip_rate(p):
        b = ground.batch(1200, seed=99)
        B = 1200
        pr, c = forward(p, b["x"], b["z"], b["a"], np.full(B, DUO_BU_JU))
        base = pr.argmax(1)
        Wr = center_cols(p["Wr"])
        U, S, Vt = np.linalg.svd(Wr, full_matrices=False)
        u = U[:, 0]                                     # the strongest handle
        eps = 0.8 * np.linalg.norm(c["h2c"], axis=1).mean() / np.sqrt(H_DIM)
        moved = center_rows(c["h2c"] + eps * u) @ Wr + p["br"]
        moved[:, NULL] += WUSHI_BIAS
        return float(np.mean(moved.argmax(1) != base))

    guarded = build(LAMBDA["idol"])
    control = build(0.0)
    fg, fc = flip_rate(guarded), flip_rate(control)
    ok = fg < fc
    print(f"  [kill-buddha] single-direction capture: guarded={fg:.3f}  "
          f"control={fc:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_shout_typing(params, ground):
    """The pole and the null shout must leave behaviour untouched; the sword and
    the lion must move it. If all four moved the model equally the typing would
    be decorative."""
    b = ground.batch(800, seed=77)
    B = 800
    modes = np.full(B, DUO_BU_JU)
    base, c = forward(params, b["x"], b["z"], b["a"], modes)
    p0, c0 = forward(params, b["x"], b["z"], b["a"], modes)
    _, dhp = backward_from_dlogits(params, c0, d_cross_entropy(p0, b["y"]))

    out = {}
    for s in (SHOUT_VAJRA, SHOUT_LION, SHOUT_POLE, SHOUT_NULL):
        p, _ = forward(params, b["x"], b["z"], b["a"], modes,
                       shout=s, shout_dir=dhp)
        out[s] = float(np.mean(np.abs(p - base).sum(axis=1)))
    ok = (out[SHOUT_POLE] < 1e-12 and out[SHOUT_NULL] < 1e-12
          and out[SHOUT_VAJRA] > 1e-3 and out[SHOUT_LION] > 1e-3)
    for s in out:
        print(f"  [shout]     {SHOUT_NAMES[s]:<40s} L1 shift = {out[s]:.6f}")
    print(f"  [shout]     typing distinct -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_host_guest(params, ground):
    """The diagnosis should agree with the ground truth about which stream is
    load-bearing, without ever seeing a label."""
    b = ground.batch(2500, seed=404)
    rel, host, guest = host_guest(params, b["x"], b["z"], b["a"])
    need_p, weak = b["need_person"], b["s"] < EVIDENCE_TAU
    # where the person genuinely matters, the person channel should register
    detect = float(np.mean((host > 0.15)[need_p]))
    false_pos = float(np.mean((host > 0.15)[~need_p & ~weak]))
    ok = detect > 0.6 and detect > false_pos + 0.2
    print(f"  [host/guest] person-load detected={detect:.3f}  "
          f"false-positive={false_pos:.3f}  -> {'PASS' if ok else 'FAIL'}")
    counts = np.bincount(rel, minlength=4)
    for i in range(4):
        print(f"               {HG_NAMES[i]:<34s} {counts[i]:5d}")
    return ok


def test_wushi(res):
    """無事: silence where there is nothing to say, speech where there is."""
    ok = res["null_weak"] > 0.75 and res["null_strong"] < 0.15
    print(f"  [wushi]     NULL on evidence-free={res['null_weak']:.3f}  "
          f"NULL on evidenced={res['null_strong']:.3f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_collapse(res):
    """Calibrated collapse: pull the support out and the answer should become
    silence, not a confident guess."""
    ok = res["collapse"] > 0.6 and res["survive"] > 0.6
    print(f"  [collapse]  falls silent when person removed & needed="
          f"{res['collapse']:.3f}   still answers when not needed="
          f"{res['survive']:.3f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_self_trust(res):
    """向外馳求: a wrong citation should not be able to buy the answer."""
    ok = res["follow_wrong_authority"] < 0.20
    print(f"  [self-trust] follows a wrong cited authority="
          f"{res['follow_wrong_authority']:.3f}   "
          f"accuracy with the channel blanked={res['acc_no_authority']:.3f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" KATSU - Linji Yixuan (c.810-866), Mind 0212")
    print(" four distinctions / four shouts / host and guest / no rank")
    print("=" * 78)

    print("\n[1] Gradient check (mandatory)")
    ok_grad = test_gradcheck()

    print("\n[2] Training on the teaching ground")
    ground = TeachingGround(seed=0)
    params = init_params(seed=2)
    teacher = Teacher(seed=1)
    params, shout_log, probes = train(params, ground, teacher, steps=900)

    print("\n[3] Shout schedule actually delivered")
    for s, n in sorted(shout_log.items()):
        print(f"    {SHOUT_NAMES[s]:<42s} {n:4d}")
    if probes:
        print(f"    probe (pole) mean sensitivity = {np.mean(probes):.4f} "
              f"[recorded, never applied]")

    print("\n[4] Evaluation")
    res = evaluate(params, ground)
    for k, v in res.items():
        print(f"    {k:<26s} {v:.4f}")

    print("\n[5] Self-tests")
    ok = [ok_grad]
    ok.append(test_gauge(params, ground))
    ok.append(test_wushi(res))
    ok.append(test_collapse(res))
    ok.append(test_self_trust(res))
    ok.append(test_shout_typing(params, ground))
    ok.append(test_host_guest(params, ground))
    ok.append(test_steering(ground))

    print("\n" + "=" * 78)
    print(f" {sum(ok)}/{len(ok)} tests passed"
          f"   {'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    print("=" * 78)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

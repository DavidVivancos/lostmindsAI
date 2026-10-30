#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0336 · Bonaventure of Bagno Regio (1221-1274)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 CONTUITION NET (CTN): a resolutive architecture with a severed certitude channel
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Bonaventure's single most exportable technical claim is a *partition*, and it is
his alone among the thirteenth-century masters. In the disputed question "On the
Knowledge of Christ" (De scientia Christi, q.4) he asks what part of human
knowing actually requires the eternal light. His answer is deliberately narrow:
not the content. Abstraction, universality, and correspondence to the object all
come from created causes -- the agent intellect that abstracts, the possible
intellect that receives, the essence of the thing known, and truth as the end of
the act. Only ONE property of knowing needs the eternal reason: *certitude*. And
even there the light works as a "regulative and motive cause" (ratio regulans et
motiva) -- "not as the sole cause, nor in its full clarity, but along with a
created cause and as contuited by us in part."

Four load-bearing engineering claims fall straight out of that sentence:

  (1) SEPARATION.  The channel that produces the answer and the channel that
      produces the confidence in the answer are different channels, made of
      different stuff, and the second is not built by learning.
  (2) NON-OBJECTHOOD.  The regulative light is *contuited*, never intuited: it
      is seen-with, not seen-at. It conditions every act of knowing and is never
      itself an object of that act. Architecturally: the basis is read from, is
      differentiated *through*, and receives no gradient, ever.
  (3) PARTIALITY.  "In part, not in full clarity." Certitude is structurally
      capped below 1. A system that can report total certainty has already
      misdescribed its own condition.
  (4) TERMINATION.  Bonaventure spent his last years attacking the eternity of
      the world, and his method of definition (Itinerarium III.3) is a
      *resolutio*: every apprehension is traced back through genus after genus
      until it lands on something that is not derived from anything further.
      A chain of reduction that never lands explains nothing. Architecturally:
      the resolution loop must provably contract, with a bounded rung count.

The resulting network is therefore NOT a transformer, NOT attention over learned
keys, NOT a mixture of experts. It is a fixed, non-learned dictionary of "eternal
reasons" (the Ars aeterna), a trainable pair of intellects that produce a species
to be resolved against that dictionary, a six-rung contraction ladder (the six
wings of the seraph at La Verna, October 1259), and a seventh rung -- the
excessus -- which is not a computation but a refusal.

THE SIGN LADDER
---------------
Bonaventure's semiotics grades every creature by how much of God it carries:
umbra (shadow, indeterminate causal pointing), vestigium (footprint, pointing to
God as triple cause), imago (image, in rational creatures), similitudo
(likeness, in creatures conformed by grace). We reuse the grading as a *quality
scale on the resolution itself*: how deeply the input resolved into the eternal
art. It is reported alongside every output.

WHAT THE MODEL DOES (the task)
------------------------------
A synthetic world of "creatures". Each creature is generated as a sparse mixture
of rows of a fixed exemplar basis A (its eternal reasons), then pushed through a
fixed nonlinear corruption and additive noise -- the creature as it appears in
the sensible world, not as it is in the divine art. The model must:
   * name the dominant exemplar (the content pathway -- trained),
   * and, on creatures generated from a DIFFERENT, foreign basis F that was
     never in the Ars, decline to name anything at all (the excessus).

Crucially the abstention threshold is calibrated on in-domain validation data
ONLY. The model never sees a foreign creature during training or calibration.
Foreign refusal is therefore emergent from the geometry of resolution, not
fitted. That is the whole Bonaventurean wager: certitude is not a learned
statistic about your training distribution, it is a measurement against a
standard you did not author.

CONVENTIONS OF THIS CORPUS
--------------------------
  * Pure NumPy. No autodiff. Every gradient is derived and written by hand.
  * A finite-difference gradient check is mandatory and runs on every execution.
  * A real training loop, real held-out evaluation, and a self-test battery.
  * The file is executed before shipping and its output pasted into the chapter.

Run:  python3 chapter_0336_bonaventure_1221.py
"""

import numpy as np

# ==============================================================================
# 0. DETERMINISM AND GLOBAL SHAPE
# ==============================================================================
# Two separate seeds on purpose. ARS_SEED generates the eternal reasons; it is
# conceptually prior to and independent of the seed that generates the learner's
# initial weights and the world's noise. The Ars is not something the learner
# happened upon.
ARS_SEED = 12211274      # birth year : death year
WORLD_SEED = 1259        # the retreat to Mount La Verna

D_SENSE = 48    # dimension of the raw sensible impression
H_AGENT = 64    # width of the agent intellect (intellectus agens)
M_SPEC = 32     # dimension of the intelligible species; also the Ars dimension
K_ARS = 8       # number of eternal reasons in the Ars aeterna
T_RUNGS = 6     # six wings of the seraph -> six rungs of resolution
TAU = 0.35      # sharpness of the contuitive read
EPS_CLARITY = 0.06   # "not in full clarity": certitude is capped at 1 - EPS
BETA_RES = 2.2       # how fast certitude decays with unresolved residue


# ==============================================================================
# 1. THE ARS AETERNA  --  a fixed, orthonormal, non-learned exemplar basis
# ==============================================================================
def build_ars(seed=ARS_SEED, k=K_ARS, m=M_SPEC):
    """
    The eternal reasons. Rows are orthonormal so that reading a species against
    the Ars is a genuine projection and not a scrambling.

    This matrix is constructed once and is NEVER updated. It is not in the
    parameter dictionary, it receives no gradient, and self-test T2 verifies
    bitwise that training did not touch a single entry of it. That is the
    computational content of 'contuited, not intuited': the mind differentiates
    THROUGH the light and never differentiates the light.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(k, m))
    q, _ = np.linalg.qr(raw.T)          # (m, k) with orthonormal columns
    return np.ascontiguousarray(q.T)    # (k, m) with orthonormal rows


def build_foreign(seed=ARS_SEED + 977, k=6, m=M_SPEC, ars=None):
    """
    A foreign basis: creatures that are not participations in this Ars at all.
    We project the foreign rows off the Ars subspace so the foreignness is real
    and not an accident of random overlap. These never appear in training.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(k, m))
    if ars is not None:
        raw = raw - (raw @ ars.T) @ ars          # remove all Ars content
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


# ==============================================================================
# 2. THE WORLD  --  creatures as corrupted participations in their exemplars
# ==============================================================================
def make_world(n, basis, seed, lift, dominance=0.72, noise=0.16):
    """
    Generate n creatures.

    Each creature picks one dominant exemplar row, mixes in a little of the
    others (nothing in the sensible world is a pure instance of its form), then
    is lifted into sense-space by a FIXED random linear map `lift` and bent by a
    fixed nonlinearity. The corruption is the same for Ars creatures and foreign
    creatures, so the model cannot detect foreignness by surface statistics --
    only by failure to resolve.
    """
    rng = np.random.default_rng(seed)
    k = basis.shape[0]
    y = rng.integers(0, k, size=n)
    coef = rng.random(size=(n, k)) * (1.0 - dominance) / max(k - 1, 1)
    coef[np.arange(n), y] += dominance
    latent = coef @ basis                                    # (n, M_SPEC)
    x = latent @ lift                                        # (n, D_SENSE)
    x = np.tanh(1.5 * x) + 0.35 * np.sin(2.0 * x)            # sensible bending
    x = x + noise * rng.normal(size=x.shape)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return x, y


# ==============================================================================
# 3. PARAMETERS  --  the CREATED causes of knowledge only
# ==============================================================================
def init_params(seed=WORLD_SEED):
    """
    Bonaventure lists exactly four created causes of the content of knowledge:
    the possible intellect as material cause (it receives), the agent intellect
    as efficient cause (it abstracts), the essence of the creature as formal
    cause, and truth as end. Only the first two are ours to build; the third
    lives in the Ars, the fourth in the loss.

      W1, b1  : agent intellect -- abstracts a species from the sensible
      W2, b2  : possible intellect -- receives and shapes the species
      wg, bg  : the rung gate -- how far this mind consents to be moved at each
                step of resolution. This is the 'created cause' operating
                alongside the light; the light supplies direction, the gate
                supplies the actual displacement.
      V,  bv  : the naming head -- assigns the resolved species its exemplar

    Nothing here touches the Ars. There is deliberately no parameter anywhere in
    this dictionary that participates in computing certitude.
    """
    rng = np.random.default_rng(seed)
    p = {}
    p["W1"] = rng.normal(0, np.sqrt(2.0 / D_SENSE), size=(D_SENSE, H_AGENT))
    p["b1"] = np.zeros(H_AGENT)
    p["W2"] = rng.normal(0, np.sqrt(1.0 / H_AGENT), size=(H_AGENT, M_SPEC))
    p["b2"] = np.zeros(M_SPEC)
    p["wg"] = rng.normal(0, 0.15, size=(2 * M_SPEC,))
    p["bg"] = np.array(0.0)
    p["V"] = rng.normal(0, np.sqrt(1.0 / M_SPEC), size=(M_SPEC, K_ARS))
    p["bv"] = np.zeros(K_ARS)
    return p


def seminal_mask(stage, h=H_AGENT, seed=ARS_SEED + 5):
    """
    RATIONES SEMINALES.

    Bonaventure holds that matter was created carrying latent structures --
    seminal reasons -- which are not added later but were sown at creation and
    unfold when conditions permit (In II Sent. d.18). We implement growth that
    way and not the usual way: the agent intellect's full width exists from
    initialisation, but a fixed subset of its units is masked to zero and is
    progressively unmasked as training proceeds.

    Consequence, verified by self-test T6: the parameter count NEVER changes.
    Capacity unfolds; it is not appended. That distinction is the entire point.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(h)
    fracs = [0.4, 0.7, 1.0]
    live = int(round(fracs[min(stage, len(fracs) - 1)] * h))
    m = np.zeros(h)
    m[order[:live]] = 1.0
    return m


# ==============================================================================
# 4. FORWARD PASS
# ==============================================================================
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(p, A, X, mask, cache=True):
    """
    APPREHENSIO -> the ladder -> DIIUDICATIO.

    Bonaventure's analysis of a sense act has three moments (Itinerarium II.4-6):
    apprehensio (the species is taken in), delectatio (the power rests in a
    proportioned object), diiudicatio (judgement, which "detaches the object from
    place and time" and so opens the way to intellectual cognition). The three
    map onto the three stages below: abstraction, the contraction ladder (the
    mind resting further into the proportioned exemplar at each rung), and the
    naming head.

    THE LADDER, one rung:
        c   = z A^T                 read the species against the eternal reasons
        p   = softmax(c / tau)      which reason is this creature's
        zh  = p A                   the exemplar as the Ars holds it
        g   = sigmoid([z;zh] wg+bg) how far this mind consents to be moved
        z'  = z + g (zh - z)        the movement itself

    One-step contraction lemma, exact:
        z' - zh = (1 - g)(z - zh)  =>  ||z' - zh|| = (1-g)||z - zh||, g in (0,1)
    so each rung strictly reduces the distance to the exemplar currently read.
    The regress cannot be infinite; a tolerance is reached in a bounded number
    of rungs. Self-test T3 verifies the lemma numerically to 1e-12 and checks
    the residual sequence is non-increasing in practice.
    """
    N = X.shape[0]
    a1 = X @ p["W1"] + p["b1"]
    u = np.tanh(a1)
    um = u * mask                      # seminal reasons not yet unfolded stay latent
    s = um @ p["W2"] + p["b2"]         # the intelligible species

    z = s
    rungs = []
    for _ in range(T_RUNGS):
        c = z @ A.T
        pr = softmax(c / TAU)
        zh = pr @ A
        gin = z @ p["wg"][:M_SPEC] + zh @ p["wg"][M_SPEC:] + p["bg"]
        g = 1.0 / (1.0 + np.exp(-gin))
        z_next = z + g[:, None] * (zh - z)
        if cache:
            rungs.append({"z": z, "pr": pr, "zh": zh, "g": g})
        z = z_next

    # Final contuition: one last read, used for the residue and for certitude.
    c_f = z @ A.T
    pr_f = softmax(c_f / TAU)
    zh_f = pr_f @ A
    r = z - zh_f                                   # the unresolved residue
    logits = z @ p["V"] + p["bv"]

    out = {"logits": logits, "z": z, "r": r, "pr_f": pr_f, "zh_f": zh_f}
    if cache:
        out.update({"a1": a1, "u": u, "um": um, "s": s, "rungs": rungs, "N": N})
    return out


# ==============================================================================
# 5. THE CERTITUDE CHANNEL  --  no parameters, no gradient, capped below one
# ==============================================================================
def build_expression(seed=ARS_SEED + 31, d=D_SENSE, w=64):
    """
    THE EXPRESSION MAP (Phi).

    A first attempt at this architecture measured certitude as the residue of
    the *species* against the Ars -- and it failed, instructively. The trainable
    intellects simply learned to collapse the whole of sense-space onto the
    8-dimensional Ars span, after which every input whatsoever resolved
    perfectly and the machine was serenely certain about creatures that had
    nothing to do with its exemplars. That is precisely the pathology
    Bonaventure diagnoses: a mind that "generalises well beyond the data of
    experience" and mistakes the smoothness of its own act for contact with the
    thing. A confidence signal computed from quantities the learner controls
    will always be gamed by the learner.

    So certitude is moved out of the learner's reach entirely. Phi is a FIXED
    random nonlinear feature map on the RAW sensible impression. The learner
    never sees it, never optimises through it, and cannot flatten it.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(d, w)) / np.sqrt(d)


def record_vestiges(Phi, Xtr, Ytr, k=K_ARS):
    """
    THE VESTIGE TABLE.

    Bonaventure's semiotics is not decorative. A vestigium is a real claim: the
    sensible properties of a creature carry the footprint of the eternal reason
    it participates in. So we RECORD, for each eternal reason, the average
    footprint it leaves in the fixed feature space -- and a scale, the average
    distance of a genuine creature from its own reason's footprint.

    Note what this is not. It is not fitted, not optimised, has no loss, no
    gradient, and no free parameter; it is an empirical mean of a fixed map,
    taken once, after training is already over. The certitude question then
    becomes the one Bonaventure actually asks: does this appearance in fact
    bear the footprint of the reason you have just named it by?
    """
    Z = np.tanh(Xtr @ Phi)
    S = np.stack([Z[Ytr == c].mean(axis=0) for c in range(k)])
    scale = np.linalg.norm(Z - S[Ytr], axis=1).mean()
    return S, float(scale)


def certitude(out, X, Phi, S, scale):
    """
    Contuition, quantified.

        y_light = argmax(pr_f)                    which reason the light reads
        d       = ||tanh(X Phi) - S[y_light]|| / scale
        kappa   = (1 - EPS_CLARITY) * exp(-BETA_RES * max(0, d - 1)) * (p_1 - p_2)

      * d asks whether the sensible appearance carries the footprint of the
        exemplar the light just read on it, measured entirely in fixed,
        non-learned quantities. It is normalised so that d = 1 is the distance a
        typical genuine creature sits from its own reason's footprint; the hinge
        means a creature is not rewarded for being unusually typical, only
        penalised for being further off than creatures of the Ars usually are.
      * (p_1 - p_2) is how decisively one eternal reason won over the runner-up
        in the contuitive read -- taken from the light, not from the naming
        head, so that certitude is structurally independent of the answer.
      * (1 - EPS_CLARITY) is a hard cap. The system CANNOT report full clarity,
        because Bonaventure's light is contuited "in part" and never in its full
        clarity in this life. A machine that can output certainty 1.0 has not
        merely miscalibrated; it has misdescribed its own condition.

    This function reads no entry of the parameter dictionary; it is invariant to
    the naming head; no term of the training loss invokes it. Self-test T4
    enforces all three. Certitude is not something this network learns to say.
    It is something measured about the network, by a standard it did not author.
    """
    y_light = out["pr_f"].argmax(axis=1)
    d = np.linalg.norm(np.tanh(X @ Phi) - S[y_light], axis=1) / scale
    srt = np.sort(out["pr_f"], axis=1)
    margin = srt[:, -1] - srt[:, -2]
    return (1.0 - EPS_CLARITY) * np.exp(-BETA_RES * np.maximum(0.0, d - 1.0)) * margin


SIGN_EDGES = np.array([0.18, 0.42, 0.68])
SIGN_NAMES = ["umbra", "vestigium", "imago", "similitudo"]


def sign_grade(kappa):
    """Bonaventure's four grades of sign, reused as a resolution quality scale."""
    return [SIGN_NAMES[int(np.searchsorted(SIGN_EDGES, k))] for k in np.atleast_1d(kappa)]


# ==============================================================================
# 6. LOSS AND HAND-DERIVED BACKWARD PASS
# ==============================================================================
def loss_and_grads(p, A, X, Y, mask, lam=0.30, mu=0.60):
    """
    Loss = CE(naming head)  +  lam * mean||r||^2  +  mu * CE(contuitive read).

    Three terms, three doctrines.

      * The first is the created act of naming: the possible intellect receives
        the species and the mind judges. This is ordinary supervised learning.
      * The second is the *reductio*: the mind is penalised for leaving any of
        the species unresolved into the Ars. Applied to in-domain data only,
        because that is the only data on which resolution is the right thing to
        demand. Foreign creatures never enter this loss.
      * The third is the FORMAL cause. Bonaventure lists the essence of the
        creature known as one of the four created causes of knowledge: the thing
        itself constrains which exemplar it resolves to. Without this term the
        ladder still contracts beautifully -- onto an arbitrary exemplar bearing
        no relation to what the creature actually is. (Measured: the light and
        the naming head agreed 0% of the time.) The mind was resolving
        confidently to the wrong eternal reasons.

    Everything below is analytic. No autodiff.
    """
    out = forward(p, A, X, mask, cache=True)
    N = out["N"]
    logits, z, r = out["logits"], out["z"], out["r"]

    # ---- cross-entropy ----
    P = softmax(logits)
    ce = -np.log(np.clip(P[np.arange(N), Y], 1e-12, None)).mean()
    dlogits = P.copy()
    dlogits[np.arange(N), Y] -= 1.0
    dlogits /= N

    # ---- residue term ----
    lres = lam * (r ** 2).sum(axis=1).mean()
    dr = 2.0 * lam * r / N

    # ---- formal-cause term: the contuitive read must land on the true exemplar
    prf = out["pr_f"]
    lform = -mu * np.log(np.clip(prf[np.arange(N), Y], 1e-12, None)).mean()
    dprf = np.zeros_like(prf)
    dprf[np.arange(N), Y] = -mu / (N * np.clip(prf[np.arange(N), Y], 1e-12, None))

    g = {}
    g["V"] = z.T @ dlogits
    g["bv"] = dlogits.sum(axis=0)
    dz = dlogits @ p["V"].T

    # residue path: r = z - softmax(z A^T / tau) A
    dz += dr                                    # direct dependence on z
    dzh_f = -dr                                 # through the reconstruction
    dp_f = dzh_f @ A.T + dprf                   # residue path + formal-cause path
    dc_f = prf * (dp_f - (dp_f * prf).sum(axis=1, keepdims=True)) / TAU
    dz += dc_f @ A

    # ---- backward through the six rungs ----
    g["wg"] = np.zeros_like(p["wg"])
    g["bg"] = np.array(0.0)
    for rung in reversed(out["rungs"]):
        zc, pr, zh, gg = rung["z"], rung["pr"], rung["zh"], rung["g"]
        dz_next = dz
        dgin = (dz_next * (zh - zc)).sum(axis=1) * gg * (1.0 - gg)
        dz = dz_next * (1.0 - gg)[:, None]
        dzh = dz_next * gg[:, None]
        # gate parameters
        g["wg"][:M_SPEC] += zc.T @ dgin
        g["wg"][M_SPEC:] += zh.T @ dgin
        g["bg"] = g["bg"] + dgin.sum()
        dz += dgin[:, None] * p["wg"][None, :M_SPEC]
        dzh += dgin[:, None] * p["wg"][None, M_SPEC:]
        # reconstruction path through the contuitive softmax
        dpr = dzh @ A.T
        dc = pr * (dpr - (dpr * pr).sum(axis=1, keepdims=True)) / TAU
        dz += dc @ A
        # NOTE: no dA is ever accumulated. The Ars is differentiated through and
        # never differentiated with respect to. This is the whole doctrine.

    # ---- back to the two intellects ----
    ds = dz
    g["W2"] = out["um"].T @ ds
    g["b2"] = ds.sum(axis=0)
    dum = ds @ p["W2"].T
    du = dum * mask
    da1 = du * (1.0 - out["u"] ** 2)
    g["W1"] = X.T @ da1
    g["b1"] = da1.sum(axis=0)

    return ce + lres + lform, g, out


# ==============================================================================
# 7. TRAINING  --  Adam, with the seminal unfolding schedule
# ==============================================================================
def train(p, A, Xtr, Ytr, epochs=260, lr=6e-3, batch=128, verbose=True):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = Xtr.shape[0]
    rng = np.random.default_rng(WORLD_SEED + 3)
    hist = []
    for ep in range(epochs):
        stage = 0 if ep < epochs // 3 else (1 if ep < 2 * epochs // 3 else 2)
        mask = seminal_mask(stage)
        idx = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            sl = idx[i:i + batch]
            L, gr, _ = loss_and_grads(p, A, Xtr[sl], Ytr[sl], mask)
            step += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * gr[k]
                v[k] = b2 * v[k] + (1 - b2) * (gr[k] ** 2)
                mh = m[k] / (1 - b1 ** step)
                vh = v[k] / (1 - b2 ** step)
                p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
            ep_loss += L
            nb += 1
        hist.append(ep_loss / nb)
        if verbose and (ep % 40 == 0 or ep == epochs - 1):
            live = int(mask.sum())
            print(f"    epoch {ep:4d} | loss {ep_loss/nb:8.5f} | "
                  f"seminal units unfolded {live}/{H_AGENT}")
    return p, hist


# ==============================================================================
# 8. SELF-TESTS
# ==============================================================================
def t1_gradient_check():
    """Finite-difference check on every trainable tensor. Mandatory."""
    rng = np.random.default_rng(7)
    A = build_ars(k=4, m=10)
    global D_SENSE, H_AGENT, M_SPEC, K_ARS, T_RUNGS
    Ds, Hs, Ms, Ks, Ts = D_SENSE, H_AGENT, M_SPEC, K_ARS, T_RUNGS
    D_SENSE, H_AGENT, M_SPEC, K_ARS, T_RUNGS = 9, 7, 10, 4, 3
    p = init_params(seed=5)
    X = rng.normal(size=(6, 9))
    Y = rng.integers(0, 4, size=6)
    mask = np.ones(7)

    L0, gr, _ = loss_and_grads(p, A, X, Y, mask)
    worst = 0.0
    worst_name = ""
    h = 1e-6
    for name in p:
        arr = p[name]
        flat = np.atleast_1d(arr).ravel()
        picks = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        for i in picks:
            orig = flat[i]
            flat[i] = orig + h
            Lp, _, _ = loss_and_grads(p, A, X, Y, mask)
            flat[i] = orig - h
            Lm, _, _ = loss_and_grads(p, A, X, Y, mask)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = np.atleast_1d(gr[name]).ravel()[i]
            rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_name = rel, name
    D_SENSE, H_AGENT, M_SPEC, K_ARS, T_RUNGS = Ds, Hs, Ms, Ks, Ts
    ok = worst < 1e-6
    print(f"  [T1] gradient check ......... max rel err {worst:.3e} "
          f"(worst tensor: {worst_name})  -> {'PASS' if ok else 'FAIL'}")
    return ok


def t2_ars_untouched(A_before, A_after):
    same = np.array_equal(A_before, A_after)
    print(f"  [T2] Ars aeterna unchanged .. bitwise identical after training: "
          f"{same}  -> {'PASS' if same else 'FAIL'}")
    return same


def t3_contraction(p, A, X, mask):
    """Verify the one-step contraction lemma and the monotone residual."""
    out = forward(p, A, X, mask, cache=True)
    max_err = 0.0
    for rg in out["rungs"]:
        lhs = np.linalg.norm(rg["z"] + rg["g"][:, None] * (rg["zh"] - rg["z"]) - rg["zh"], axis=1)
        rhs = (1.0 - rg["g"]) * np.linalg.norm(rg["z"] - rg["zh"], axis=1)
        max_err = max(max_err, np.abs(lhs - rhs).max())
    res = np.stack([np.linalg.norm(rg["z"] - rg["zh"], axis=1) for rg in out["rungs"]], axis=1)
    mono = (np.diff(res, axis=1) <= 1e-9).all(axis=1).mean()
    ok = max_err < 1e-12 and mono > 0.95
    print(f"  [T3] contraction lemma ...... exact to {max_err:.2e}; residual "
          f"non-increasing on {100*mono:.1f}% of creatures  -> {'PASS' if ok else 'FAIL'}")
    return ok, res


def t4_certitude_is_severed(p, A, X, mask, Phi, S, scale):
    """
    Three independent checks that the certitude channel is genuinely severed:
      (a) it is invariant to arbitrary scrambling of the naming head -- the
          confidence does not come from the answer;
      (b) it is capped strictly below 1 -- never in full clarity;
      (c) the training loss never invokes it -- no gradient can flow into it.
    """
    out = forward(p, A, X, mask, cache=True)
    k_before = certitude(out, X, Phi, S, scale)

    perturbed = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in p.items()}
    perturbed["V"] = perturbed["V"] + 3.7
    perturbed["bv"] = perturbed["bv"] + 1.9
    out2 = forward(perturbed, A, X, mask, cache=True)
    k_after = certitude(out2, X, Phi, S, scale)

    head_independent = np.allclose(k_before, k_after, atol=1e-12)
    capped = k_before.max() <= (1.0 - EPS_CLARITY) + 1e-12
    loss_is_clean = "certitude" not in loss_and_grads.__code__.co_names

    ok = head_independent and capped and loss_is_clean
    print(f"  [T4] certitude severed ...... head-invariant {head_independent}; "
          f"absent from loss {loss_is_clean}; capped at {1-EPS_CLARITY:.2f} "
          f"(observed max {k_before.max():.4f})  -> {'PASS' if ok else 'FAIL'}")
    return ok


def t5_seminal_param_count(p):
    counts = []
    for stage in range(3):
        mask = seminal_mask(stage)
        counts.append((int(mask.sum()), sum(np.atleast_1d(v).size for v in p.values())))
    stable = len({c[1] for c in counts}) == 1
    print(f"  [T5] seminal unfolding ...... units {[c[0] for c in counts]} of "
          f"{H_AGENT}; parameter count constant at {counts[0][1]}: {stable}"
          f"  -> {'PASS' if stable else 'FAIL'}")
    return stable


# ==============================================================================
# 9. MAIN
# ==============================================================================
def main():
    print("=" * 78)
    print(" CONTUITION NET  --  Bonaventure of Bagnoregio (c.1221-1274), figure 0274")
    print(" separated certitude / fixed Ars aeterna / six-rung resolution / excessus")
    print("=" * 78)

    A = build_ars()
    A_before = A.copy()
    F = build_foreign(ars=A)
    lift = np.random.default_rng(ARS_SEED + 11).normal(size=(M_SPEC, D_SENSE)) / np.sqrt(M_SPEC)

    print("\n[1] World")
    Xtr, Ytr = make_world(2400, A, WORLD_SEED, lift)
    Xva, Yva = make_world(800, A, WORLD_SEED + 1, lift)
    Xte, Yte = make_world(800, A, WORLD_SEED + 2, lift)
    Xfo, _ = make_world(800, F, WORLD_SEED + 3, lift)   # never seen in training
    print(f"    Ars aeterna         : {A.shape[0]} eternal reasons in R^{A.shape[1]}, "
          f"orthonormality err {np.abs(A @ A.T - np.eye(K_ARS)).max():.2e}")
    print(f"    train / val / test  : {len(Xtr)} / {len(Xva)} / {len(Xte)} creatures")
    print(f"    foreign creatures   : {len(Xfo)} (from a basis orthogonal to the Ars)")

    print("\n[2] Self-test T1 (gradient check) before training")
    ok1 = t1_gradient_check()

    print("\n[3] Training the created causes only")
    p = init_params()
    p, hist = train(p, A, Xtr, Ytr)

    full = seminal_mask(2)
    print("\n[4] Self-tests after training")
    ok2 = t2_ars_untouched(A_before, A)
    ok3, res = t3_contraction(p, A, Xte[:400], full)
    Phi = build_expression()
    S, scale = record_vestiges(Phi, Xtr, Ytr)
    ok4 = t4_certitude_is_severed(p, A, Xte[:400], full, Phi, S, scale)
    ok5 = t5_seminal_param_count(p)

    print("\n[5] Resolution profile across the six rungs (mean residual)")
    for t in range(T_RUNGS):
        bar = "#" * int(38 * res[:, t].mean() / max(res[:, 0].mean(), 1e-9))
        print(f"    rung {t+1} ({['sensus','imaginatio','ratio','intellectus','intelligentia','apex mentis'][t]:<13}) "
              f"{res[:, t].mean():.4f}  {bar}")

    print("\n[6] Naming accuracy (content pathway, trained)")
    out_te = forward(p, A, Xte, full)
    acc = (out_te["logits"].argmax(1) == Yte).mean()
    print(f"    held-out accuracy on Ars creatures : {100*acc:.2f}%  "
          f"(chance {100/K_ARS:.1f}%)")

    print("\n[7] Excessus  --  calibration on IN-DOMAIN data only")
    out_va = forward(p, A, Xva, full)
    out_fo = forward(p, A, Xfo, full)
    k_va = certitude(out_va, Xva, Phi, S, scale)
    k_te = certitude(out_te, Xte, Phi, S, scale)
    k_fo = certitude(out_fo, Xfo, Phi, S, scale)
    theta = float(np.percentile(k_va, 5.0))   # let 95% of genuine creatures through
    ab_te = (k_te < theta).mean()
    ab_fo = (k_fo < theta).mean()
    kept = k_te >= theta
    acc_kept = (out_te["logits"].argmax(1)[kept] == Yte[kept]).mean()
    agree = (out_te["pr_f"].argmax(1) == out_te["logits"].argmax(1)).mean()
    print(f"    threshold theta (5th pct of validation certitude) : {theta:.4f}")
    print(f"    light and naming head agree on Ars creatures      : {100*agree:.2f}%")
    print(f"    mean certitude, Ars creatures     : {k_te.mean():.4f}")
    print(f"    mean certitude, foreign creatures : {k_fo.mean():.4f}")
    print(f"    abstention rate, Ars creatures    : {100*ab_te:.2f}%")
    print(f"    abstention rate, foreign creatures: {100*ab_fo:.2f}%   <-- never trained on")
    print(f"    accuracy on what it did not decline: {100*acc_kept:.2f}%")

    print("\n[8] Sign grades assigned to the resolution")
    for label, kk in (("Ars creatures", k_te), ("foreign creatures", k_fo)):
        grades = sign_grade(kk)
        counts = {n: grades.count(n) for n in SIGN_NAMES}
        tot = len(grades)
        print(f"    {label:<18}: " + "  ".join(
            f"{n} {100*counts[n]/tot:5.1f}%" for n in SIGN_NAMES))

    ok6 = acc > 0.80 and ab_fo > 0.60 and ab_te < 0.12
    print(f"\n  [T6] behavioural target ..... acc>80%, foreign abstention>60%, "
          f"in-domain abstention<12%  -> {'PASS' if ok6 else 'FAIL'}")

    allok = all([ok1, ok2, ok3, ok4, ok5, ok6])
    print("\n" + "=" * 78)
    print(f" ALL SELF-TESTS: {'PASS' if allok else 'FAIL'}")
    print("=" * 78)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

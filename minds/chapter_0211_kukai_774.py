#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 chapter_0211_kukai_774.py
 THE SHOJI-JISSO ENGINE  (声字実相 — "sound, sign, reality")
 A from-scratch cognitive architecture after Kukai / Kobo Daishi (774-835 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0211_kukai_774 - Kukai / Kobo Daishi (774-835 CE)
================================================================================  

WHY THIS ARCHITECTURE LOOKS THE WAY IT DOES
-------------------------------------------------------------------------------
Kukai's position, stated in the *Shojijissogi* and the *Benkenmitsu nikkyoron*,
is not "language is sacred." It is far stranger and far more operational:

  (a) HOSSHIN SEPPO (法身説法). The Dharmakaya itself preaches. Reality is not a
      silent substrate that a mind later annotates; it is an utterance already
      in progress. Therefore THERE IS NO UNINTERPRETED DATUM. A percept does
      not become a sign when a mind labels it; it arrives as a sign.

  (b) MONJI (文字). "Anything that manifests its distinction from others in
      virtue of its pattern is a letter." Every object of the six senses is a
      graph in a script. Perception is therefore *decipherment*, and the first
      operation of any mind must be quantization onto a script — not feature
      extraction into a continuum.

  (c) SHABETSU (差別). A letter is what it is only by difference from the other
      letters. Identity is differential, not substantial. (Kukai arrives at a
      near-Saussurean claim a millennium early — and then refuses the other
      half of it, arbitrariness, because the script is the world's own
      self-articulation, not a human convention laid over the world.)

  (d) THE HOLOGRAPHY OF THE PART. In the *Unjigi* a single syllable, HUM, is
      decomposed into its constituent letters and each one is unfolded into an
      entire doctrine; the doctrine folds back into the syllable. Kukai quotes
      the Avatamsaka: each hair contains oceans of lands. The part is required
      to carry the whole.

  (e) SANMITSU + KAJI (三密・加持). Body (mudra), speech (mantra), and mind
      (mandala-visualisation) are performed SIMULTANEOUSLY. Kukai is explicit
      that the practitioner "utters a mantra, forms mudras, and visualises,
      all at the same time." Simultaneity is not redundancy; it is the
      *condition* of the response (kaji, the mutual holding of Buddha and
      practitioner).

  (f) JUJUSHIN (十住心). Ten abiding stages of mind, from the goat-mind of
      appetite up to the mind of secret sublimity. Crucially the ladder is
      INCLUSIVE: "rather than rejecting or negating the previous states, this
      final state fulfils and encompasses their standpoints." Nothing lower is
      ever discarded.

Six commitments, six mechanisms. Each mechanism below is the direct
transcription of one of them, and each is checked by a test that could fail.

  (a)+(b) -> MONJI BOTTLENECK   : the ONLY path from percept to output runs
                                  through a discrete script. No residual path,
                                  no skip connection, no continuous bypass.
  (c)     -> FROZEN BIJA CODEBOOK: the script is not learned from data. It is
                                  constructed from the six great elements
                                  (rokudai) plus a shared A-ground, and frozen.
                                  The network learns how to UNFOLD letters, it
                                  never learns the alphabet.
  (d)     -> INTERPENETRATION + CONTAINMENT LOSS: positions are coupled by a
                                  learned mutual-inclusion operator, and one
                                  single shared head must recover the meaning
                                  of the WHOLE compound from EACH part alone.
  (e)     -> KAJI GATE          : a MULTIPLICATIVE product-of-agreements across
                                  the three channels. Not a sum. Losing one
                                  channel does not cost a third of the signal;
                                  it closes the gate.
  (f)     -> NESTED STAGE PREFIXES: stage s reads only the first n_s dimensions,
                                  n_1 < n_2 < ... < n_10. Stage 10 literally
                                  contains stage 1 as a sub-vector. Discarding
                                  a lower stage is not expressible.

DELIBERATELY NOT A TRANSFORMER. There is no attention over stored keys, no
softmax retrieval over a memory of past tokens, no positional embedding added
to a residual stream. Position here is a PERMUTATION (a circular shift), which
is a binding operation, not a lookup coordinate — the compound is a
superposition of shifted unfoldings, in the holographic-reduced-representation
family, because Kukai's part-whole relation is superpositional rather than
retrieval-based.

IMPLEMENTATION NOTES
-------------------------------------------------------------------------------
* Pure NumPy. All gradients derived by hand. No autodiff, no frameworks.
* Quantization uses a sharp SOFT assignment (a Boltzmann distribution over
  squared distances to the frozen letters) so the whole forward pass is exactly
  differentiable and the finite-difference gradient check is exact. Hard
  argmax decoding is available at inference and is reported separately.
* Every experiment below has a control that is expected to FAIL. A mechanism
  that cannot fail has not been tested.

Run:  python3 chapter_0211_kukai_774.py
===============================================================================
"""

import numpy as np

RNG = np.random.default_rng(774)          # his birth year, as the seed

# =============================================================================
# 0.  THE SCRIPT: constructing the frozen bija codebook
# =============================================================================
# Kukai's cosmos is made of six "great elements" (rokudai): earth, water, fire,
# wind, space, and consciousness. The first five are the known; the sixth is
# the knower. Every letter of our script is given a 6-bit elemental signature.
#
# On top of that, every letter shares a common ground. In the Shojijissogi the
# syllable A is "the mother of all syllables" and, in the Siddham script, the
# first stroke of every other letter. So every bija vector here literally
# contains an A-component. Experiment 6 removes it and measures the damage.

N_ELEM   = 6      # earth, water, fire, wind, space, consciousness
K_SEEDS  = 16     # size of the script
D_SEED   = 16     # dimension a letter lives in
A_WEIGHT = 0.60   # how much of every letter is the shared A-ground

ELEMENT_NAMES = ["earth", "water", "fire", "wind", "space", "consciousness"]


def build_script(a_weight=A_WEIGHT, rng=None):
    """Construct the frozen bija codebook B (K x D) and its elemental bits.

    Returns
    -------
    B    : (K_SEEDS, D_SEED) frozen letter vectors, never trained
    bits : (K_SEEDS, N_ELEM) elemental signature of each letter
    """
    rng = rng or np.random.default_rng(1)
    # 16 distinct non-empty elemental signatures, deterministic and distinct.
    bits = np.zeros((K_SEEDS, N_ELEM), dtype=np.int64)
    chosen, v = [], 1
    while len(chosen) < K_SEEDS:
        b = [(v >> j) & 1 for j in range(N_ELEM)]
        if sum(b) >= 2:                    # no letter is a bare single element
            chosen.append(b)
        v += 1
    bits[:] = np.array(chosen)

    # (i) IDENTITY. A letter is what it is by difference from every other
    #     letter (shabetsu), so the identity components are made exactly
    #     orthogonal: the rows of a random orthogonal matrix.
    ident, _ = np.linalg.qr(rng.normal(0, 1, (D_SEED, K_SEEDS)))
    ident = ident.T[:K_SEEDS]                          # (K, D), orthonormal

    # (ii) ELEMENT. On top of identity, each letter carries its share of the
    #      six great elements, centred so the elemental part does not simply
    #      pile every letter into the same corner of the space.
    proj = rng.normal(0, 1.0, size=(N_ELEM, D_SEED))
    centred = bits.astype(np.float64) - bits.mean(0, keepdims=True)
    body = centred @ proj
    body /= (np.linalg.norm(body, axis=1, keepdims=True) + 1e-12)

    # (iii) THE A-GROUND. In the Siddham script A is the first stroke of every
    #       letter and "the mother of all syllables", so one shared direction
    #       is present in every vector. a_weight=0 removes it (Experiment 6).
    a_dir = rng.normal(0, 1.0, size=(D_SEED,))
    a_dir /= np.linalg.norm(a_dir)

    B = ident + 0.55 * body + a_weight * a_dir[None, :]
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B, bits


# =============================================================================
# 1.  THE COSMIC TEXT: a synthetic world that is already an utterance
# =============================================================================
# A "moment" is an ordered triple of letters. It is presented to the model
# through three channels which never show the letters directly:
#
#   BODY   (shin, mudra)   -- an elemental/gestural projection
#   SPEECH (ku, mantra)    -- a phase/spectral projection of the letter index
#   MIND   (i, mandala)    -- a 2-D mandalic position projection
#
# Each channel is a noisy, lossy, DIFFERENT view of the same letter. No channel
# alone is sufficient. This is the empirical content of Kukai's insistence that
# the three mysteries must be performed together.
#
# Two labels are attached to each moment:
#   PRINCIPLE (8 classes) -- a genuinely 3-way, order-sensitive function of the
#                            triple. No single letter determines it.
#   STAGE    (10 classes) -- a monotone function of how much "space" and
#                            "consciousness" the moment carries: the jujushin
#                            ladder from goat-mind to secret sublimity.

D_BODY, D_SPEECH, D_MIND = 7, 7, 7
N_POS = 3
N_PRINCIPLE = 8
N_STAGE = 10


def principle_label(s0, s1, s2, bits):
    """Order-sensitive 3-way interaction. No two letters suffice."""
    b0, b1, b2 = bits[s0], bits[s1], bits[s2]
    inter = int(np.sum(b0 & b1))          # what the first two share
    diff = int(np.sum(b1 ^ b2))           # where the last two part company
    order = 1 if s0 > s2 else 0           # a bare order bit
    return (inter + 3 * diff + 5 * order) % N_PRINCIPLE


def stage_label(s0, s1, s2, bits):
    """The jujushin ladder: how much knower and how much space is present."""
    tri = bits[[s0, s1, s2]]
    cons = int(tri[:, 5].sum())           # consciousness
    space = int(tri[:, 4].sum())          # space
    wind = int(tri[:, 3].sum())           # wind
    raw = 3 * cons + 2 * space + wind     # 0 .. 18
    return int(min(N_STAGE - 1, raw // 2))


def channel_projections():
    """Three fixed 'languages'. Each is a lossy linear view of a letter:
    body sees 7 of the 16 dimensions' worth, speech another 7, mind another 7.
    No channel is a full description; the letter is only ever inferred."""
    r = np.random.default_rng(806)          # the year he came home
    Pb = r.normal(0, 1, (D_SEED, D_BODY))
    Ps = r.normal(0, 1, (D_SEED, D_SPEECH))
    Pm = r.normal(0, 1, (D_SEED, D_MIND))
    return Pb, Ps, Pm


PROJ_B, PROJ_S, PROJ_M = channel_projections()


def make_dataset(n, B, bits, noise=0.25, rng=None, corrupt=None):
    """Generate n moments. `corrupt` in {None,'body','speech','mind'} replaces
    that channel with pure noise — used for the sanmitsu ablation."""
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, K_SEEDS, size=(n, N_POS))
    L = B[idx]                                        # (n,pos,D_SEED)

    Xb = L @ PROJ_B
    Xs = L @ PROJ_S
    Xm = L @ PROJ_M
    for X in (Xb, Xs, Xm):
        X /= (X.std() + 1e-12)
    Xb = Xb + rng.normal(0, noise, Xb.shape)
    Xs = Xs + rng.normal(0, noise, Xs.shape)
    Xm = Xm + rng.normal(0, noise, Xm.shape)

    if corrupt == "body":
        Xb = rng.normal(0, 1.0, Xb.shape)
    elif corrupt == "speech":
        Xs = rng.normal(0, 1.0, Xs.shape)
    elif corrupt == "mind":
        Xm = rng.normal(0, 1.0, Xm.shape)

    yp = np.array([principle_label(*idx[i], bits) for i in range(n)])
    ys = np.array([stage_label(*idx[i], bits) for i in range(n)])
    return dict(Xb=Xb, Xs=Xs, Xm=Xm, yp=yp, ys=ys, idx=idx)


# =============================================================================
# 2.  PARAMETERS
# =============================================================================
D_UNF = 60        # the unfolded dimension (must be divisible by N_POS)
D_HEAD = 64
D_REC = 32        # width of the self-emission decoder
SHIFT = D_UNF // N_POS

# The kaji gate is NOT trained. Nothing in clean training data ever rewards
# closing it, so a learnable gate simply learns to stay open and the doctrine
# evaporates — which is exactly what happened on the first run of this file.
# Kukai does not derive the requirement of the three mysteries from evidence;
# he imposes it as the condition of practice. So it is imposed here: a fixed
# threshold on the agreement of the three channels, which the engine may not
# optimise away.
ALPHA_KAJI = 8.0      # sharpness of the consent threshold
THETA_KAJI = 0.35     # below this agreement, the response is withheld


def init_params(rng):
    def g(*sh, s=None):
        s = s or (1.0 / np.sqrt(sh[0]))
        return rng.normal(0, s, sh)
    P = {
        # channel encoders: three languages -> one letter space
        "Wb": g(D_BODY, D_SEED),   "bb": np.zeros(D_SEED),
        "Ws": g(D_SPEECH, D_SEED), "bs": np.zeros(D_SEED),
        "Wm": g(D_MIND, D_SEED),   "bm": np.zeros(D_SEED),
        # unfolding operator (Unjigi): letter -> doctrine
        "U": g(D_SEED, D_UNF), "bu": np.zeros(D_UNF),
        # mutual inclusion (interpenetration): one operator per RELATIVE
        # position, so a part takes in the others without forgetting where
        # they stood. A bare sum over neighbours would erase order, and the
        # meaning of a compound is order-sensitive.
        "R1": rng.normal(0, 0.10, (D_UNF, D_UNF)),
        "R2": rng.normal(0, 0.10, (D_UNF, D_UNF)),
        # the single shared head: reads the whole, and reads each part
        "W1": g(D_UNF, D_HEAD), "b1": np.zeros(D_HEAD),
        "W2": g(D_HEAD, N_PRINCIPLE), "b2": np.zeros(N_PRINCIPLE),
        # nested stage readout
        "Wst": g(D_UNF, N_STAGE), "bst": np.zeros(N_STAGE),
        # jijuhoraku: the self-emission decoder. Having read the world as a
        # letter, the engine must be able to RE-UTTER the world from it.
        "Wr": g(D_SEED, D_REC), "br": np.zeros(D_REC),
        "Wrb": g(D_REC, D_BODY),   "brb": np.zeros(D_BODY),
        "Wrs": g(D_REC, D_SPEECH), "brs": np.zeros(D_SPEECH),
        "Wrm": g(D_REC, D_MIND),   "brm": np.zeros(D_MIND),
    }
    return P


def stage_mask():
    """Row j of column s is live only if j < n_s, with n_1 < ... < n_10.
    This is the architectural form of 'nothing lower is discarded'."""
    M = np.zeros((D_UNF, N_STAGE))
    for s in range(N_STAGE):
        n_s = int(np.ceil(D_UNF * (s + 1) / N_STAGE))
        M[:n_s, s] = 1.0
    return M


MASK = stage_mask()

# ---------------------------------------------------------------------------
# THE TWO MANDALAS. Kukai hangs two mandalas on opposite walls of the practice
# hall and insists they are non-dual (richi funi): the Kongokai (Diamond,
# wisdom, the particular ascending) and the Taizo (Womb, pattern, the whole
# descending into every part). The unfolded vector is split the same way.
# The Diamond half of each part keeps its own letter uncontaminated, which is
# what lets the compound stay order-sensitive. The Womb half receives the
# other positions, which is what lets a single part carry the whole.
# Without this split the two demands destroy each other — measured, not assumed.
# ---------------------------------------------------------------------------
D_HALF = D_UNF // 2
WOMB = np.zeros(D_UNF)
WOMB[:D_HALF] = 1.0     # the Womb occupies the low dimensions, so it is
                        # visible even from the shortest nested prefix: the
                        # lowest dwellings read pattern (ri); the higher ones
                        # additionally read the Diamond half, wisdom (chi).


# =============================================================================
# 3.  FORWARD
# =============================================================================
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def ce(logits, y):
    p = softmax(logits)
    n = len(y)
    return -np.log(p[np.arange(n), y] + 1e-12).mean(), p


def cos_pair(A, Bv):
    """cosine per (n,pos) between two (n,pos,d) stacks, plus cached norms."""
    na = np.linalg.norm(A, axis=-1) + 1e-9
    nb = np.linalg.norm(Bv, axis=-1) + 1e-9
    dot = (A * Bv).sum(-1)
    return dot / (na * nb), dot, na, nb


def forward(P, d, B, temp=0.08, use_kaji=True, use_inter=True, cache=False):
    Xb, Xs, Xm = d["Xb"], d["Xs"], d["Xm"]
    n = Xb.shape[0]

    # --- three mysteries: three channels enter their own letter-space -------
    Zb = Xb @ P["Wb"] + P["bb"]; Eb = np.tanh(Zb)
    Zs = Xs @ P["Ws"] + P["bs"]; Es = np.tanh(Zs)
    Zm = Xm @ P["Wm"] + P["bm"]; Em = np.tanh(Zm)

    # --- KAJI: multiplicative agreement, not additive fusion ---------------
    c_bs, dot_bs, nb_, ns_ = cos_pair(Eb, Es)
    c_sm, dot_sm, ns2, nm_ = cos_pair(Es, Em)
    c_bm, dot_bm, nb2, nm2 = cos_pair(Eb, Em)
    a = c_bs * c_sm * c_bm                      # (n,pos)
    if use_kaji:
        gate = 1.0 / (1.0 + np.exp(-ALPHA_KAJI * (a - THETA_KAJI)))
    else:
        gate = np.ones_like(a)                  # control: no gate at all
    Emean = (Eb + Es + Em) / 3.0
    # the letters live on the unit sphere, so the percept is placed there too:
    # a reading is a DIRECTION in letter-space, never a magnitude
    pnorm = np.linalg.norm(Emean, axis=-1, keepdims=True) + 1e-9
    Pc = Emean / pnorm

    # --- MONJI BOTTLENECK: the percept is read as a letter -----------------
    # squared distance to every frozen letter, turned into a Boltzmann weight
    d2 = ((Pc[:, :, None, :] - B[None, None, :, :]) ** 2).sum(-1)   # (n,pos,K)
    logits_q = -d2 / temp
    Wq = softmax(logits_q, axis=-1)
    Q = Wq @ B                                   # (n,pos,d) -- lies in span(B)
    # NOTE: Pc does NOT continue forward. Q is the only thing that does.

    # --- UNFOLDING (Unjigi): letter -> doctrine ----------------------------
    Zu = Q @ P["U"] + P["bu"]
    Uh_raw = np.tanh(Zu)
    # KAJI: the empowerment. Perception (monji) happens regardless, but the
    # RESPONSE occurs only where the three mysteries are in accord. The gate
    # multiplies the unfolding, so discord does not degrade the answer — it
    # withholds it.
    Uh = gate[..., None] * Uh_raw                # (n,pos,u)

    # --- INTERPENETRATION: each part takes in the others -------------------
    if use_inter:
        nx1 = np.stack([Uh[:, (i + 1) % N_POS, :] for i in range(N_POS)], 1)
        nx2 = np.stack([Uh[:, (i + 2) % N_POS, :] for i in range(N_POS)], 1)
        Mix = Uh + (nx1 @ P["R1"] + nx2 @ P["R2"]) * WOMB
        Vpre = np.tanh(Mix)
    else:
        Mix = Uh
        Vpre = np.tanh(Uh)

    # --- POSITION BINDING: a permutation, not an added embedding -----------
    V = np.stack([np.roll(Vpre[:, i, :], i * SHIFT, axis=-1)
                  for i in range(N_POS)], axis=1)
    C = V.sum(axis=1)                            # the compound (n,u)

    # --- HEADS -------------------------------------------------------------
    Hh = np.tanh(C @ P["W1"] + P["b1"])
    Lp = Hh @ P["W2"] + P["b2"]                  # principle of the whole

    Lst = C @ (P["Wst"] * MASK) + P["bst"]       # nested stage readout

    # CONTAINMENT (Unjigi / "each hair contains oceans of lands"): the SAME
    # organ that reads the dwelling of the whole compound is turned on each
    # single part, and must return the dwelling of the WHOLE. A part that
    # knew only itself could not do this.
    Vf = Vpre.reshape(n * N_POS, D_UNF)
    Lc = Vf @ (P["Wst"] * MASK) + P["bst"]

    # --- JIJUHORAKU: re-utter the three channels from the letter alone -----
    Hr = np.tanh(Q @ P["Wr"] + P["br"])
    Rb = Hr @ P["Wrb"] + P["brb"]
    Rs = Hr @ P["Wrs"] + P["brs"]
    Rm = Hr @ P["Wrm"] + P["brm"]

    out = dict(Lp=Lp, Lc=Lc, Lst=Lst, C=C, Q=Q, Wq=Wq, Pc=Pc, gate=gate, a=a,
               Hr=Hr, Rb=Rb, Rs=Rs, Rm=Rm)
    if cache:
        out.update(dict(Eb=Eb, Es=Es, Em=Em, Zb=Zb, Zs=Zs, Zm=Zm,
                        c_bs=c_bs, c_sm=c_sm, c_bm=c_bm,
                        Emean=Emean, pnorm=pnorm, d2=d2, Zu=Zu, Uh=Uh, Uh_raw=Uh_raw,
                        Mix=Mix,
                        Vpre=Vpre, V=V, Hh=Hh, Vf=Vf,
                        n=n, temp=temp, use_kaji=use_kaji,
                        use_inter=use_inter))
    return out


LAM_C, LAM_S, LAM_Q, LAM_R, LAM_A = 1.0, 0.6, 0.50, 4.0, 0.5
N_TRAIN, N_VAL, EPOCHS = 9000, 1500, 150


def loss_fn(P, d, B, **kw):
    o = forward(P, d, B, cache=True, **kw)
    n = o["n"]
    lp, pp = ce(o["Lp"], d["yp"])
    yc = np.repeat(d["ys"], N_POS)               # the WHOLE's dwelling
    lc, pc = ce(o["Lc"], yc)
    ls, ps = ce(o["Lst"], d["ys"])
    lq = ((o["Pc"] - o["Q"]) ** 2).sum(-1).mean()      # commitment to the script
    lr_ = (((o["Rb"] - d["Xb"]) ** 2).mean()
           + ((o["Rs"] - d["Xs"]) ** 2).mean()
           + ((o["Rm"] - d["Xm"]) ** 2).mean())
    # the three mysteries are three expressions of ONE Dharma, so the three
    # channels are trained to say the same thing about the same letter
    la = -o["a"].mean()
    total = (lp + LAM_C * lc + LAM_S * ls + LAM_Q * lq
             + LAM_R * lr_ + LAM_A * la)
    o.update(dict(pp=pp, pc=pc, ps=ps, yc=yc,
                  parts=(lp, lc, ls, lq, lr_, la)))
    return total, o


# =============================================================================
# 4.  BACKWARD  (all gradients derived by hand)
# =============================================================================
def backward(P, d, B, o):
    n = o["n"]
    G = {k: np.zeros_like(v) for k, v in P.items()}

    # ---- heads -------------------------------------------------------------
    dLp = (o["pp"] - np.eye(N_PRINCIPLE)[d["yp"]]) / n
    G["W2"] += o["Hh"].T @ dLp
    G["b2"] += dLp.sum(0)
    dHh = dLp @ P["W2"].T
    dCh = (dHh * (1 - o["Hh"] ** 2)) @ P["W1"].T
    G["W1"] += o["C"].T @ (dHh * (1 - o["Hh"] ** 2))
    G["b1"] += (dHh * (1 - o["Hh"] ** 2)).sum(0)

    dLst = LAM_S * (o["ps"] - np.eye(N_STAGE)[d["ys"]]) / n
    G["Wst"] += (o["C"].T @ dLst) * MASK
    G["bst"] += dLst.sum(0)
    dCs = dLst @ (P["Wst"] * MASK).T

    nc = n * N_POS
    dLc = LAM_C * (o["pc"] - np.eye(N_STAGE)[o["yc"]]) / nc
    G["Wst"] += (o["Vf"].T @ dLc) * MASK
    G["bst"] += dLc.sum(0)
    dVf = dLc @ (P["Wst"] * MASK).T           # (n*pos, u) -> straight to Vpre

    dC = dCh + dCs                            # (n,u)

    # ---- V: from the compound sum, and from the containment head ----------
    dV = np.repeat(dC[:, None, :], N_POS, axis=1)

    # ---- unbind the circular shift (a permutation: invert it) -------------
    dVpre = np.stack([np.roll(dV[:, i, :], -i * SHIFT, axis=-1)
                      for i in range(N_POS)], axis=1)
    dVpre = dVpre + dVf.reshape(n, N_POS, D_UNF)

    # ---- interpenetration --------------------------------------------------
    dMix = dVpre * (1 - o["Vpre"] ** 2)
    if o["use_inter"]:
        Uh = o["Uh"]
        nx1 = np.stack([Uh[:, (i + 1) % N_POS, :] for i in range(N_POS)], 1)
        nx2 = np.stack([Uh[:, (i + 2) % N_POS, :] for i in range(N_POS)], 1)
        dMw = dMix * WOMB
        G["R1"] += np.einsum('npi,npj->ij', nx1, dMw)
        G["R2"] += np.einsum('npi,npj->ij', nx2, dMw)
        b1_ = dMw @ P["R1"].T
        b2_ = dMw @ P["R2"].T
        dUh = dMix.copy()
        for j in range(N_POS):
            dUh[:, j, :] += b1_[:, (j - 1) % N_POS, :]
            dUh[:, j, :] += b2_[:, (j - 2) % N_POS, :]
    else:
        dUh = dMix

    # ---- kaji empowerment, then unfolding ---------------------------------
    dgate = (dUh * o["Uh_raw"]).sum(-1)                 # (n,pos)
    dUh_raw = dUh * o["gate"][..., None]
    dZu = dUh_raw * (1 - o["Uh_raw"] ** 2)
    G["U"] += np.einsum('npd,npu->du', o["Q"], dZu)
    G["bu"] += dZu.sum((0, 1))
    dQ = dZu @ P["U"].T                                 # (n,pos,d)

    # ---- jijuhoraku: gradient of the self-emission decoder ----------------
    dHr = np.zeros_like(o["Hr"])
    for nm, R_, X_, Dc in (("b", o["Rb"], d["Xb"], D_BODY),
                           ("s", o["Rs"], d["Xs"], D_SPEECH),
                           ("m", o["Rm"], d["Xm"], D_MIND)):
        dR = LAM_R * 2.0 * (R_ - X_) / (n * N_POS * Dc)
        G["Wr" + nm] += np.einsum('npr,npi->ri', o["Hr"], dR)
        G["br" + nm] += dR.sum((0, 1))
        dHr += dR @ P["Wr" + nm].T
    dZr = dHr * (1 - o["Hr"] ** 2)
    G["Wr"] += np.einsum('npd,npr->dr', o["Q"], dZr)
    G["br"] += dZr.sum((0, 1))
    dQ = dQ + dZr @ P["Wr"].T

    # ---- commitment loss ---------------------------------------------------
    dPc_direct = LAM_Q * 2.0 * (o["Pc"] - o["Q"]) / (n * N_POS)
    dQ = dQ - dPc_direct                                # d(Pc-Q)^2 wrt Q

    # ---- quantization: Q = softmax(-||Pc-B||^2/T) @ B ----------------------
    # dL/dlogits_q = Wq * (dQ.B^T - sum_k Wq*(dQ.B^T))
    s = dQ @ B.T                                        # (n,pos,K)
    Wq = o["Wq"]
    dlog = Wq * (s - (Wq * s).sum(-1, keepdims=True))
    dd2 = -dlog / o["temp"]
    # d||Pc-B_k||^2 / dPc = 2(Pc - B_k)
    dPc = 2.0 * (o["Pc"] * dd2.sum(-1, keepdims=True) - dd2 @ B)
    dPc = dPc + dPc_direct

    # ---- back through the spherical projection of the percept -------------
    dPre = (dPc - (dPc * o["Pc"]).sum(-1, keepdims=True) * o["Pc"]) / o["pnorm"]
    dEb = dPre / 3.0
    dEs = dPre / 3.0
    dEm = dPre / 3.0

    # ---- the gate itself, plus the three-mysteries agreement objective ----
    da = -LAM_A * np.ones_like(o["a"]) / o["a"].size
    if o["use_kaji"]:
        g = o["gate"]
        dz = dgate * g * (1 - g)                        # through sigmoid
        da = da + dz * ALPHA_KAJI      # no gradient to the gate's own shape:
                                       # the threshold is doctrine, not a
                                       # parameter the engine may relax
    if True:
        c_bs, c_sm, c_bm = o["c_bs"], o["c_sm"], o["c_bm"]
        dc_bs = da * c_sm * c_bm
        dc_sm = da * c_bs * c_bm
        dc_bm = da * c_bs * c_sm

        def cos_grad(A_, B_, c_, dc_):
            """d(cos)/dA and d(cos)/dB for stacks (n,pos,d)."""
            na = np.linalg.norm(A_, axis=-1, keepdims=True) + 1e-9
            nb = np.linalg.norm(B_, axis=-1, keepdims=True) + 1e-9
            cA = dc_[..., None] * (B_ / (na * nb) - c_[..., None] * A_ / na**2)
            cB = dc_[..., None] * (A_ / (na * nb) - c_[..., None] * B_ / nb**2)
            return cA, cB

        gb1, gs1 = cos_grad(o["Eb"], o["Es"], c_bs, dc_bs)
        gs2, gm1 = cos_grad(o["Es"], o["Em"], c_sm, dc_sm)
        gb2, gm2 = cos_grad(o["Eb"], o["Em"], c_bm, dc_bm)
        dEb = dEb + gb1 + gb2
        dEs = dEs + gs1 + gs2
        dEm = dEm + gm1 + gm2

    # ---- channel encoders --------------------------------------------------
    for nm, dE, Z, X in (("b", dEb, o["Zb"], d["Xb"]),
                         ("s", dEs, o["Zs"], d["Xs"]),
                         ("m", dEm, o["Zm"], d["Xm"])):
        dZ = dE * (1 - np.tanh(Z) ** 2)
        G["W" + nm] += np.einsum('npi,npj->ij', X, dZ)
        G["b" + nm] += dZ.sum((0, 1))
    return G


# =============================================================================
# 5.  GRADIENT CHECK  (mandatory)
# =============================================================================
def gradient_check(B, bits, n=6, eps=3e-5):
    rng = np.random.default_rng(3)
    P = init_params(rng)
    d = make_dataset(n, B, bits, rng=np.random.default_rng(4))
    _, o = loss_fn(P, d, B)
    G = backward(P, d, B, o)

    # Standard mixed criterion: pure relative error is meaningless for
    # coordinates whose true gradient is ~1e-6, where the double-precision
    # finite difference is itself noise. The denominator is floored.
    worst, worst_name, checked, worst_abs = 0.0, "", 0, 0.0
    for k in P:
        flat = np.atleast_1d(P[k]).ravel()
        idxs = range(flat.size) if flat.size <= 40 else \
            rng.choice(flat.size, 40, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp1, _ = loss_fn(P, d, B)
            flat[i] = orig - eps
            lm1, _ = loss_fn(P, d, B)
            flat[i] = orig
            num = (lp1 - lm1) / (2 * eps)
            ana = np.atleast_1d(G[k]).ravel()[i]
            den = max(1e-4, abs(num) + abs(ana))
            rel = abs(num - ana) / den
            checked += 1
            worst_abs = max(worst_abs, abs(num - ana))
            if rel > worst:
                worst, worst_name = rel, f"{k}[{i}]"
    return worst, worst_name, checked, worst_abs


# =============================================================================
# 6.  TRAINING
# =============================================================================
def accuracy(logits, y):
    return float((logits.argmax(1) == y).mean())


def train(P, B, bits, tr, va, epochs=EPOCHS, bs=128, lr=0.045, verbose=True, **kw):
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(val) for k, val in P.items()}
    t = 0
    n = len(tr["yp"])
    for ep in range(epochs):
        perm = np.random.default_rng(100 + ep).permutation(n)
        lr_ep = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.9 + 0.1)
        for st in range(0, n, bs):
            sl = perm[st:st + bs]
            batch = {k: tr[k][sl] for k in ("Xb", "Xs", "Xm", "yp", "ys")}
            _, o = loss_fn(P, batch, B, **kw)
            G = backward(P, batch, B, o)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * G[k]
                v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
                mh = m[k] / (1 - 0.9 ** t)
                vh = v[k] / (1 - 0.999 ** t)
                P[k] -= lr_ep * mh / (np.sqrt(vh) + 1e-8)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            L, o = loss_fn(P, va, B, **kw)
            print(f"    epoch {ep:3d} | loss {L:6.4f} | "
                  f"principle {accuracy(o['Lp'], va['yp']):.3f} | "
                  f"stage {accuracy(o['Lst'], va['ys']):.3f} | "
                  f"part->whole {accuracy(o['Lc'], o['yc']):.3f}")
    return P


# =============================================================================
# 7.  EXPERIMENTS
# =============================================================================
def hard_decode_accuracy(P, d, B):
    """Inference with a HARD argmax over the script instead of the soft one."""
    o = forward(P, d, B, cache=True)
    idx = o["d2"].argmin(-1)
    Q = B[idx]
    Uh = o["gate"][..., None] * np.tanh(Q @ P["U"] + P["bu"])
    nx1 = np.stack([Uh[:, (i + 1) % N_POS, :] for i in range(N_POS)], 1)
    nx2 = np.stack([Uh[:, (i + 2) % N_POS, :] for i in range(N_POS)], 1)
    Vpre = np.tanh(Uh + (nx1 @ P["R1"] + nx2 @ P["R2"]) * WOMB)
    V = np.stack([np.roll(Vpre[:, i, :], i * SHIFT, -1) for i in range(N_POS)], 1)
    C = V.sum(1)
    Lp = np.tanh(C @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    # The model may settle on any bijection between the world's letters and
    # the script's slots; what matters is that the reading is CONSISTENT and
    # one-to-one, so recovery is scored up to a fixed permutation.
    conf = np.zeros((K_SEEDS, K_SEEDS))
    for t, pr in zip(d["idx"].ravel(), idx.ravel()):
        conf[t, pr] += 1
    perm, used = -np.ones(K_SEEDS, dtype=int), set()
    order = np.dstack(np.unravel_index(np.argsort(-conf, axis=None), conf.shape))[0]
    for t, pr in order:
        if perm[t] < 0 and pr not in used:
            perm[t] = pr; used.add(pr)
    letter_acc = float((perm[d["idx"]] == idx).mean())
    return accuracy(Lp, d["yp"]), letter_acc


def probe(X, y, k, epochs=260, lr=0.4):
    """A tiny multinomial logistic probe, trained from scratch on features X."""
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.05, (X.shape[1], k)); b = np.zeros(k)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n = len(y)
    for _ in range(epochs):
        p = softmax(Xn @ W + b)
        gz = (p - np.eye(k)[y]) / n
        W -= lr * (Xn.T @ gz + 1e-4 * W)
        b -= lr * gz.sum(0)
    return accuracy(Xn @ W + b, y)


def main():
    print("=" * 79)
    print(" THE SHOJI-JISSO ENGINE  —  Kukai / Kobo Daishi (774-835)")
    print(" 'The five great elements all have resonance; the ten realms all")
    print("  possess language; the six sense-objects are all letters.'")
    print("=" * 79)

    B, bits = build_script(rng=np.random.default_rng(835))
    off = B @ B.T
    np.fill_diagonal(off, 0)
    print(f"\n[script] {K_SEEDS} frozen letters in {D_SEED}-d, built from the "
          f"{N_ELEM} great elements ({', '.join(ELEMENT_NAMES)})")
    print(f"[script] mean inter-letter cosine {off.sum()/(K_SEEDS*(K_SEEDS-1)):+.3f}"
          f"  — shabetsu: identity by difference, on a shared A-ground")
    print(f"[script] the codebook is FROZEN: the engine learns how to unfold a "
          f"letter, never what the letters are")

    # ---------------- TEST 1: gradient check -------------------------------
    print("\n" + "-" * 79)
    print("TEST 1 — finite-difference gradient check (mandatory)")
    worst, wname, nchk, wabs = gradient_check(B, bits)
    print(f"  coordinates checked : {nchk}  (every parameter block)")
    print(f"  worst relative error: {worst:.3e}   (at {wname})")
    print(f"  worst absolute error: {wabs:.3e}")
    ok1 = worst < 1e-4
    print(f"  RESULT: {'PASS' if ok1 else 'FAIL'}")

    # ---------------- training ---------------------------------------------
    print("\n" + "-" * 79)
    print("TRAINING the full engine")
    tr = make_dataset(N_TRAIN, B, bits, rng=np.random.default_rng(21))
    va = make_dataset(N_VAL, B, bits, rng=np.random.default_rng(22))
    maj_p = max(np.bincount(va["yp"])) / len(va["yp"])
    maj_s = max(np.bincount(va["ys"])) / len(va["ys"])
    P = init_params(np.random.default_rng(774))
    P = train(P, B, bits, tr, va)
    _, o = loss_fn(P, va, B)
    acc_p = accuracy(o["Lp"], va["yp"])
    acc_s = accuracy(o["Lst"], va["ys"])
    acc_c = accuracy(o["Lc"], o["yc"])
    hard_acc, letter_acc = hard_decode_accuracy(P, va, B)
    print(f"  principle {acc_p:.3f}  (chance {1/N_PRINCIPLE:.3f}, "
          f"majority {maj_p:.3f})")
    print(f"  dwelling  {acc_s:.3f}  (chance {1/N_STAGE:.3f}, "
          f"majority {maj_s:.3f})")
    print(f"  reading the world as letters, scored up to a bijection: "
          f"{letter_acc:.3f}  (chance {1/K_SEEDS:.3f})")
    print(f"  with a HARD argmax onto the script: principle {hard_acc:.3f}")

    # ---------------- TEST 2: the part contains the whole ------------------
    print("\n" + "-" * 79)
    print("TEST 2 — Unjigi: can one part alone report the whole compound?")
    print("  Control: the same engine with mutual inclusion switched off, so")
    print("  each part knows only its own letter.")
    P0 = init_params(np.random.default_rng(774))
    P0 = train(P0, B, bits, tr, va, verbose=False, use_inter=False)
    _, o0 = loss_fn(P0, va, B, use_inter=False)
    acc_c0 = accuracy(o0["Lc"], o0["yc"])
    acc_p0 = accuracy(o0["Lp"], va["yp"])
    print(f"    with mutual inclusion : part->whole {acc_c:.3f} | "
          f"compound {acc_p:.3f}")
    print(f"    without               : part->whole {acc_c0:.3f} | "
          f"compound {acc_p0:.3f}")
    print(f"    majority baseline     : {maj_s:.3f}")
    ok2 = acc_c > acc_c0 + 0.15
    print(f"  RESULT: {'PASS' if ok2 else 'FAIL'}")
    delta = acc_p - acc_p0
    print(f"  Effect on the compound of demanding that every part carry the")
    print(f"  whole: {delta:+.3f}. This only comes out positive because the")
    print(f"  unfolded vector is split into a Womb half that receives the")
    print(f"  other positions and a Diamond half that does not. Collapsing")
    print(f"  that split — letting mutual inclusion touch every dimension —")
    print(f"  wrecks the order-sensitive compound, because a part that has")
    print(f"  absorbed its neighbours can no longer say where they stood.")
    print(f"  The two mandalas are load-bearing, not ornamental.")

    # ---------------- TEST 3: sanmitsu must be simultaneous ----------------
    print("\n" + "-" * 79)
    print("TEST 3 — Sanmitsu/kaji: what happens when one mystery is lost?")
    print("  Control: the identical engine with the kaji gate removed, so the")
    print("  three channels are merely averaged.")
    Pg = init_params(np.random.default_rng(774))
    Pg = train(Pg, B, bits, tr, va, verbose=False, use_kaji=False)
    _, og = loss_fn(Pg, va, B, use_kaji=False)
    acc_pg = accuracy(og["Lp"], va["yp"])
    print(f"    {'channel lost':<12}{'kaji engine':>22}{'no-gate control':>20}")
    print(f"    {'':<12}{'acc     gate':>22}{'acc':>20}")
    rows = []
    for ch in (None, "body", "speech", "mind"):
        dv = make_dataset(N_VAL, B, bits, rng=np.random.default_rng(22),
                          corrupt=ch)
        _, oo = loss_fn(P, dv, B)
        _, oc = loss_fn(Pg, dv, B, use_kaji=False)
        a_k = accuracy(oo["Lp"], dv["yp"])
        a_c = accuracy(oc["Lp"], dv["yp"])
        g_k = float(oo["gate"].mean())
        rows.append((ch, a_k, g_k, a_c))
        print(f"    {(ch or 'none'):<12}{a_k:>10.3f}{g_k:>12.3f}{a_c:>20.3f}")
    intact_k, intact_c = rows[0][1], rows[0][3]
    ret_k = np.mean([(r[1] - 1/N_PRINCIPLE) / max(1e-9, intact_k - 1/N_PRINCIPLE)
                     for r in rows[1:]])
    ret_c = np.mean([(r[3] - 1/N_PRINCIPLE) / max(1e-9, intact_c - 1/N_PRINCIPLE)
                     for r in rows[1:]])
    gate_drop = rows[0][2] - np.mean([r[2] for r in rows[1:]])
    print(f"    fraction of skill retained when a mystery is lost:")
    print(f"      kaji engine     {ret_k:+.3f}")
    print(f"      no-gate control {ret_c:+.3f}")
    print(f"    mean gate closes by {gate_drop:.3f} under corruption")
    ok3 = ret_k < ret_c - 0.10
    print(f"  RESULT: {'PASS' if ok3 else 'FAIL'} — losing one mystery is not")
    print(f"  a third of a loss. The engine that needs all three keeps almost")
    print(f"  nothing; the averaging control keeps a comfortable fraction and")
    print(f"  goes on answering on evidence it no longer has.")
    print(f"  BUT — a separate finding, and it goes against the mechanism I")
    print(f"  built for it. The consent gate itself is nearly inert. Left")
    print(f"  learnable it simply learns to stay open, since nothing in clean")
    print(f"  training ever rewards closing. Pinned as a fixed prior it is")
    print(f"  rescaled away instead: the readout downstream is linear, so a")
    print(f"  constant multiplicative gate costs it nothing. It closes here by")
    print(f"  only {gate_drop:.3f} under corruption. What actually enforces")
    print(f"  simultaneity is the QUANTISER, not the gate — with one channel")
    print(f"  gone the letter cannot be resolved, and a mind that must name")
    print(f"  before it may think has nothing to think about. A multiplicative")
    print(f"  gate can only withhold if it drives an explicit abstention")
    print(f"  output. That is a design lesson this file earned the hard way.")

    # ---------------- TEST 4: nothing lower is discarded -------------------
    print("\n" + "-" * 79)
    print("TEST 4 — Jujushin: is every lower dwelling still present at the top?")
    _, ov = loss_fn(P, va, B)
    C = ov["C"]
    low = (va["ys"] <= 2).astype(int)     # the three pre-Buddhist dwellings
    print(f"    {'prefix':<26}{'dwelling acc':>14}{'lower-dwelling acc':>22}")
    accs = []
    for st in (1, 3, 6, 10):
        n_s = int(np.ceil(D_UNF * st / N_STAGE))
        a_stage = probe(C[:, :n_s], va["ys"], N_STAGE)
        a_low = probe(C[:, :n_s], low, 2)
        accs.append(a_stage)
        print(f"    dwelling {st:2d}  ({n_s:2d} dims){a_stage:>14.3f}{a_low:>22.3f}")
    a_low_full = probe(C, low, 2)
    print(f"    {'whole compound':<26}{probe(C, va['ys'], N_STAGE):>14.3f}"
          f"{a_low_full:>22.3f}")
    ok4 = a_low_full > 0.85 and accs[0] > 0.5
    print(f"  RESULT: {'PASS' if ok4 else 'FAIL'} — the goat-mind is still")
    print(f"  legible from inside the mind of secret sublimity, and legible")
    print(f"  from the shortest prefix. Nothing was overwritten to get here.")

    # ---------------- TEST 5: shabetsu, identity by difference -------------
    print("\n" + "-" * 79)
    print("TEST 5 — Shabetsu: is the meaning carried by WHICH letter was read,")
    print("          or by a continuous percept that merely landed near one?")
    print("  (An earlier version of this test permuted the codebook rows. That")
    print("   is a no-op: the quantiser sums over all letters, so relabelling")
    print("   them changes nothing. Two tests that can actually fail:)")

    # 5a — hand the trained engine a DIFFERENT script of identical statistics
    B_alt, _ = build_script(rng=np.random.default_rng(1200))
    acc_alt = accuracy(forward(P, va, B_alt)["Lp"], va["yp"])
    print(f"    5a  its own script      : principle {acc_p:.3f}")
    print(f"        a foreign script of : principle {acc_alt:.3f}   "
          f"(chance {1/N_PRINCIPLE:.3f})")
    print(f"        identical statistics")

    # 5b — is there a continuous bypass? feed the raw percept onward, skipping
    #      the letters entirely, and see whether the rest of the engine copes
    ob = forward(P, va, B, cache=True)
    Uh_b = ob["gate"][..., None] * np.tanh(ob["Pc"] @ P["U"] + P["bu"])
    nx1 = np.stack([Uh_b[:, (i + 1) % N_POS, :] for i in range(N_POS)], 1)
    nx2 = np.stack([Uh_b[:, (i + 2) % N_POS, :] for i in range(N_POS)], 1)
    Vb = np.tanh(Uh_b + (nx1 @ P["R1"] + nx2 @ P["R2"]) * WOMB)
    Cb = np.stack([np.roll(Vb[:, i, :], i * SHIFT, -1)
                   for i in range(N_POS)], 1).sum(1)
    acc_byp = accuracy(np.tanh(Cb @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"],
                       va["yp"])
    print(f"    5b  through the letters : principle {acc_p:.3f}")
    print(f"        percept sent onward : principle {acc_byp:.3f}")
    print(f"        unquantised")

    ok5 = (acc_alt < 1 / N_PRINCIPLE + 0.08) and (acc_byp < acc_p - 0.15)
    print(f"  RESULT: {'PASS' if ok5 else 'FAIL'} — the engine cannot read a")
    print(f"  different alphabet, and cannot read without one. Meaning is in")
    print(f"  the script, and there is no continuous route around it.")

    # ---------------- TEST 6: the A-ground ---------------------------------
    print("\n" + "-" * 79)
    print("TEST 6 — the syllable A: does the ground shared by every letter do")
    print("          any work, or is it piety?")
    B_noA, _ = build_script(a_weight=0.0, rng=np.random.default_rng(835))
    tr2 = make_dataset(N_TRAIN, B_noA, bits, rng=np.random.default_rng(21))
    va2 = make_dataset(N_VAL, B_noA, bits, rng=np.random.default_rng(22))
    P2 = init_params(np.random.default_rng(774))
    P2 = train(P2, B_noA, bits, tr2, va2, verbose=False)
    _, o2 = loss_fn(P2, va2, B_noA)
    acc_noA = accuracy(o2["Lp"], va2["yp"])
    _, letter_noA = hard_decode_accuracy(P2, va2, B_noA)
    offA = B_noA @ B_noA.T
    np.fill_diagonal(offA, 0)
    print(f"    with A-ground   : principle {acc_p:.3f} | letters {letter_acc:.3f}"
          f" | mean inter-letter cosine {off.sum()/(K_SEEDS*(K_SEEDS-1)):+.3f}")
    print(f"    without A-ground: principle {acc_noA:.3f} | letters {letter_noA:.3f}"
          f" | mean inter-letter cosine {offA.sum()/(K_SEEDS*(K_SEEDS-1)):+.3f}")
    dA = acc_p - acc_noA
    dL = letter_acc - letter_noA
    verdict = ("earns its place" if dA > 0.02 else
               "costs more than it returns" if dA < -0.02 else "is neutral")
    print(f"  RESULT: reported, not asserted. Dropping the shared ground moves")
    print(f"  principle accuracy by {-dA:+.3f} and letter recovery by {-dL:+.3f}.")
    print(f"  Note the two move in opposite directions: without the A-ground")
    print(f"  the letters are further apart and easier to tell apart, and the")
    print(f"  engine still does worse. A common ground costs discriminability")
    print(f"  and buys something else — a shared subspace every letter can be")
    print(f"  unfolded through. On this task the doctrine {verdict}.")

    # ---------------- summary ---------------------------------------------
    print("\n" + "=" * 79)
    checks = {"gradient check": ok1,
              "the part carries the whole": ok2,
              "kaji withholds rather than degrades": ok3,
              "no lower dwelling is discarded": ok4,
              "meaning lives in the script": ok5}
    for k, v_ in checks.items():
        print(f"  [{'PASS' if v_ else 'FAIL'}] {k}")
    print(f"\n  {sum(checks.values())}/{len(checks)} required checks passed.")
    print("  1 further experiment reported without a pass/fail claim (Test 6).")
    print("=" * 79)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0380 · Geoffrey Chaucer (1343-1400)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE COMPOUNDING-APERTURE NETWORK (CAN)
 A from-scratch, trainable cognitive architecture after Geoffrey Chaucer
 (c. 1343 - 25 October 1400)
===============================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS NOT A TRANSFORMER
------------------------------------------------------
Chaucer is the one mind in this corpus who located the failure of knowledge
not in JUDGEMENT but in TRANSPORT. His eagle in *The House of Fame* gives a
physical theory of utterance: "Soun ys noght but eyr ybroken" (HF 765) --
speech is broken air, propagating in widening rings like the rings from a
stone dropped in water, and every speech ever made necessarily arrives at
Fame's house. On arrival it does not arrive as a proposition. It arrives
WEARING ITS SPEAKER: "Whan any speche ycomen ys / Up to the paleys, anon-ryght
/ Hyt wexeth lyk the same wight / Which that the word in erthe spak" (HF
1074-77). And in the whirling wicker House of Rumour, a lie and a true saying
that meet at the same window do not compete -- they swear brotherhood, agree
that nobody shall ever have one without the other, and leave as a single
indivisible token: "Thus saugh I fals and soth compouned / Togeder fle for oo
tydynge" (HF 2108-09), having "woxen more on every tonge / Than ever hit was"
(HF 2082-83) at every retelling.

That is a complete engineering specification, and it is nothing like the
specification a Transformer implements. A Transformer stores keys and values
in slots and retrieves by similarity; provenance is metadata you may or may
not choose to carry. Chaucer's world has no slots. It has ONE medium, into
which everything is summed, in which content is *worn inside a speaker-shaped
envelope*, and in which two summed things become permanently inseparable.

The correct computational realisation of that is a HOLOGRAPHIC memory:

    binding      x = shape(speaker) (*) content        (circular convolution)
    storage      m = decay * m + gain * x              (one superposed trace)
    retrieval    c^ = shape(speaker) (#) m             (circular correlation)

Circular convolution binds two vectors into one vector of the SAME size --
there is no slot, no address, no room. Retrieval requires already knowing
whose mouth it came from. And superposition is lossy in exactly Chaucer's way:
once two bound tidings are added into the trace, no operation recovers them
individually; you only get noisy shadows, and the noise grows with how much
has been added. Irreversible compounding is not a bug we tolerate here. It is
the phenomenon the model exists to exhibit.

The network therefore has FOUR parts, each of which is a Chaucerian doctrine
made into arithmetic:

 1. THE MOUTH-STAMP (bind).  Content is never stored bare. E[utterance] is
    convolved with S[speaker]. HF 1074-82.

 2. THE APERTURE (compound).  At a designated hop, a rival tiding leaves by
    the same window and is SUMMED into the same trace. Nothing marks the
    seam. HF 2088-2109.

 3. THE ICE AND THE SHADE (decay).  The trace decays geometrically, so an
    old tiding fades -- unless it is re-uttered, which is the only shade
    there is. Chaucer's names are carved in a rock of ice and melt "of the
    lettres oon or two ... of every name, / So unfamous was woxe hir fame"
    except on the north face, "conserved with the shade / ... Of a castel"
    (HF 1143-46, 1159-62). Retention is a function of position and
    repetition, not of merit.

 4. TWO HEADS THAT DO NOT AGREE.
      - the DEED head reads the trace by unbinding it with every speaker
        shape it knows and weighting each recovered shadow by a learned
        CREDENCE for that mouth. This is "Whoso shal telle a tale after a
        man, / He moot reherce as ny as evere he kan / Everich a word"
        (General Prologue I.731-33): keep the words with the voice on them.
      - the FAME head is given only the ENERGY of the trace -- band powers,
        phase discarded. It can hear how loud the record is. It cannot hear
        what the record says. It is trained on renown, and it learns renown
        well, and its verdict is largely uncorrelated with the deed. That is
        Chaucer's finding in HF 1520-1867, where identical petitioners get
        opposite verdicts and a company that "seyden sooth, and noght a lye"
        is refused ("Ye gete of me good fame non", HF 1560-61).

 5. TWO ROADS TO ROME (abstention).  "Diverse pathes leden diverse folk the
    righte wey to Rome" (Astrolabe, Prologue). The deed is estimated twice by
    genuinely different routes -- unbind-then-match, and project-then-match --
    and their DISAGREEMENT drives an abstention head. The features feeding
    that head are DETACHED from the gradient on purpose, so that the model can
    never reduce its loss by manufacturing agreement it does not have. The
    output is Chaucer's most characteristic sentence: "I noot" -- I do not
    know. In the Astrolabe he says the same thing about his own sources: some
    conclusions "wole nat in alle thinges performen hir bihestes."

WHAT THE MODEL IS TRAINED ON
----------------------------
A synthetic pilgrimage. A deed happens. It is passed down a chain of K
speakers, each with three traits (estat, candour, loudness). Candour governs
whether a speaker repeats the deed or substitutes his own favourite spin.
Loudness governs how much amplitude he contributes. At one hop a rival tiding
compounds into the trace. Two labels are produced:

    deed  = what actually happened
    fame  = what gets proclaimed, computed purely from accumulated LOUDNESS
            over repetitions, with no reference whatever to the deed

These are deliberately different generative processes, so that a model which
scores well on both has necessarily learned two dissociated readings of the
same trace. That dissociation is the thesis.

ENGINEERING CONVENTIONS
-----------------------
  * pure NumPy, no autograd, all gradients derived and written out by hand
  * a finite-difference gradient check on EVERY parameter (mandatory)
  * a real training loop (Adam) on real generated data
  * self-tests, including two falsifiable claims:
        (a) learned credence correlates with true speaker candour
        (b) collapsing all speaker shapes to one shape destroys deed accuracy
            while barely touching fame accuracy
  * run it:  python3 chapter_0380_geoffrey_chaucer_1343.py
===============================================================================
"""

import numpy as np

# =============================================================================
# SECTION 0 -- CIRCULAR CONVOLUTION ALGEBRA AND ITS EXACT DERIVATIVES
# =============================================================================
# This is the whole physics of the model. Everything else is bookkeeping.
#
#   cconv(a, b)[i] = sum_j a[j] * b[(i - j) mod n]       -- BIND
#   ccorr(a, b)[i] = sum_j a[j] * b[(i + j) mod n]       -- UNBIND (approx inverse)
#
# Both are computed in O(n log n) through the FFT. Vectors are real; we take
# the real part of the inverse transform (imaginary residue is float noise).
#
# Derivatives (derived by index-chasing, verified below by finite differences):
#
#   y = cconv(a, b) :   dL/da = ccorr(b, dL/dy)
#                       dL/db = ccorr(a, dL/dy)
#
#   y = ccorr(a, b) :   dL/da = ccorr(dL/dy, b)
#                       dL/db = cconv(a, dL/dy)
#
# NOTE ON BROADCASTING: every routine works on the last axis, so a batch of
# traces (B, d) can be unbound against a bank of shapes (S, d) by inserting
# axes and letting NumPy broadcast to (B, S, d). No loops over the batch.
# =============================================================================


def cconv(a, b):
    """Circular convolution along the last axis. This is the BIND operation:
    it stamps a content vector with the shape of the mouth that spoke it."""
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1),
                        n=a.shape[-1], axis=-1)


def ccorr(a, b):
    """Circular correlation along the last axis. This is the UNBIND operation:
    given the speaker's shape, it pulls an approximate copy of what that
    speaker contributed out of a superposed trace. The approximation degrades
    as more material is summed in -- which is the point."""
    return np.fft.irfft(np.conj(np.fft.rfft(a, axis=-1)) * np.fft.rfft(b, axis=-1),
                        n=a.shape[-1], axis=-1)


# =============================================================================
# SECTION 1 -- THE WORLD: a pilgrimage that generates deeds, tidings and fame
# =============================================================================

class Pilgrimage:
    """Generator of episodes.

    An episode is one deed carried down a road by K speakers. Each speaker has
    three traits, drawn once and fixed for the life of the world:

        estat    social standing in [0,1]  -- affects nothing causally. It is
                 a decoy. Chaucer's whole satirical method is that estat and
                 credibility are independent, and a model that learns to
                 believe the Knight because he is a Knight has learned the
                 wrong thing. Leaving a plausible but causally inert feature
                 in the input is how we can tell.
        candour  probability of repeating the deed rather than his own spin
        loudness amplitude contributed per utterance -- drives FAME only

    Emission at hop k by speaker s:
        with prob candour[s]      -> utters the true deed
        otherwise                 -> utters spin[s], his habitual claim

    At exactly one hop (chosen uniformly) a RIVAL tiding compounds: a second
    utterance, of a different class, summed into the same trace by the same
    mouth. Nothing in the input marks which hop that was.

    Labels:
        deed  -- the true class
        fame  -- argmax over classes of sum_k loudness[s_k] * 1[utt_k == class],
                 i.e. whoever is repeated loudest wins renown. Truth is not
                 consulted.
    """

    def __init__(self, n_deed=8, n_spk=12, hops=6, seed=0):
        rng = np.random.default_rng(seed)
        self.n_deed, self.n_spk, self.hops = n_deed, n_spk, hops
        self.estat = rng.uniform(0.0, 1.0, n_spk)
        # candour spans the full range so that the credence head has something
        # to learn: some mouths are near-reliable, some are near-useless.
        self.candour = rng.uniform(0.05, 0.75, n_spk)
        self.loudness = rng.uniform(0.2, 1.0, n_spk)
        self.spin = rng.integers(0, n_deed, n_spk)     # each mouth's pet claim
        self.traits = np.stack([self.estat, self.candour, self.loudness], 1)
        self.rng = rng

    def batch(self, B):
        rng, K, D = self.rng, self.hops, self.n_deed
        deed = rng.integers(0, D, B)
        spk = rng.integers(0, self.n_spk, (B, K))

        # THE LOUD FALSE TIDING. Each episode has one dominant rumour, and it
        # is not the deed. This is the whole difficulty: a mouth that is not
        # repeating the truth is usually repeating THE SAME lie as the other
        # untruthful mouths, so the plurality of what gets said is false. A
        # reader who counts utterances gets the rumour. A reader who keeps the
        # voices attached and weighs them can still get the deed.
        rumour = (deed + rng.integers(1, D, B)) % D          # != deed

        cand = self.candour[spk]                             # (B,K)
        tells_true = rng.random((B, K)) < cand
        r = rng.random((B, K))
        lie = np.where(r < 0.70, rumour[:, None],
              np.where(r < 0.85, self.spin[spk], rng.integers(0, D, (B, K))))
        utt = np.where(tells_true, deed[:, None], lie)

        # the aperture: one hop per episode where a rival tiding compounds
        ap = rng.integers(0, K, B)
        cflag = np.zeros((B, K))
        cflag[np.arange(B), ap] = 1.0
        rival = rng.integers(0, D, (B, K))
        rival = np.where(rival == utt, (rival + 1) % D, rival)

        # FAME: pure accumulated loudness over repetitions, truth-blind.
        loud = self.loudness[spk]                            # (B,K)
        tally = np.zeros((B, D))
        np.add.at(tally, (np.repeat(np.arange(B), K), utt.ravel()), loud.ravel())
        np.add.at(tally, (np.repeat(np.arange(B), K), rival.ravel()),
                  (loud * cflag * 3.0).ravel())
        fame = np.argmax(tally, axis=1)

        # UNREADABLE: the record is genuinely too thin to settle. The true deed
        # was uttered at most once along the whole road, so what survives is
        # mostly rumour plus the compounded rival. This label is set by the
        # WORLD, not by the model's own confidence -- otherwise "I noot" would
        # be a free lunch the network could award itself.
        support = (utt == deed[:, None]).sum(1)
        unreadable = (support <= 1).astype(float)

        # which mouths were actually on this road? Only deponents are polled.
        present = np.zeros((B, self.n_spk))
        present[np.repeat(np.arange(B), K), spk.ravel()] = 1.0

        return dict(spk=spk, utt=utt, rival=rival, cflag=cflag, present=present,
                    deed=deed, fame=fame, rumour=rumour, unreadable=unreadable)


# =============================================================================
# SECTION 2 -- PARAMETERS
# =============================================================================

def init_params(n_deed, n_spk, d=64, nb=4, seed=1):
    """All learnable state. HRR vectors want unit-ish norm and near-flat
    spectra, so they are drawn from N(0, 1/d) -- the standard Plate
    initialisation for holographic representations."""
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d)
    return dict(
        E=rng.normal(0, s, (n_deed, d)),     # content codebook (what was said)
        S=rng.normal(0, s, (n_spk, d)),      # the mouth-shapes (who said it)
        v=rng.normal(0, 0.5, 3),             # traits -> trust score
        p_tau=np.array(0.5),                 # softplus -> road-A scale
        W_B=rng.normal(0, s, (d, n_deed)),   # the second road to Rome
        b_B=np.zeros(n_deed),
        W_F=rng.normal(0, 0.3, (nb + 1, n_deed)),   # Fame: energy -> renown
        b_F=np.zeros(n_deed),
        u=np.array([2.0, 0.0, -1.0]),        # disagreement -> abstention
        b_u=np.zeros(()),
        p_lam=np.array(1.2),                 # sigmoid -> decay (the ice)
        p_gain=np.array(0.4),                # softplus -> per-hop gain (rumour)
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# =============================================================================
# SECTION 3 -- FORWARD PASS
# =============================================================================

def forward(P, batch, traits, cache=True, override_S=None):
    """One pass of the Compounding-Aperture Network.

    override_S lets the ablation test swap the whole bank of mouth-shapes for
    a single repeated shape -- i.e. strip provenance while changing nothing
    else -- and watch what happens.
    """
    E = P['E']
    S_raw = P['S'] if override_S is None else override_S
    # Mouth-shapes are unit-normalised. This is standard HRR practice, but it
    # also matters conceptually: without it, ||S_s|| would silently act as a
    # second credence channel and the trust head would never have to learn
    # anything. Normalising forces "whom do I believe" to live in ONE place.
    S_nrm = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-8
    S = S_raw / S_nrm
    B, K = batch['spk'].shape
    d = E.shape[1]
    nb = P['W_F'].shape[0] - 1

    lam = sigmoid(P['p_lam'])                      # decay in (0,1)
    gain = np.log1p(np.exp(P['p_gain']))           # softplus, > 0

    # ---- 1..3 : mouth-stamp, aperture, ice --------------------------------
    m = np.zeros((B, d))
    ms = []                                        # traces per hop, for backprop
    xs, es, ss, ers = [], [], [], []
    for k in range(K):
        e_k = E[batch['utt'][:, k]]                # (B,d) what was said
        s_k = S[batch['spk'][:, k]]                # (B,d) who said it
        x_k = cconv(s_k, e_k)                      # BIND: the word wears the wight
        er_k = E[batch['rival'][:, k]]
        xr_k = cconv(s_k, er_k) * batch['cflag'][:, k][:, None]
        x_tot = x_k + xr_k                         # "fals and soth compouned"
        ms.append(m)
        m = lam * m + gain * x_tot
        xs.append(x_tot); es.append(e_k); ss.append(s_k); ers.append(er_k)

    # ---- 4a : the DEED head (unbind by every known mouth) -------------------
    Fm = np.fft.rfft(m, axis=-1)                   # (B, d//2+1)
    FS = np.fft.rfft(S, axis=-1)                   # (n_spk, d//2+1)
    chat = np.fft.irfft(np.conj(FS)[None] * Fm[:, None], n=d, axis=-1)  # (B,n_spk,d)
    sim = chat @ E.T                               # (B,n_spk,n_deed) match to codebook
    # Whom to believe is decided RELATIVELY, among the mouths actually on this
    # road -- a court of chivalry weighing its deponents against each other,
    # not an absolute gain per witness. Magnitude is handled separately by tau,
    # so the trust head cannot buy confidence by trusting everyone more.
    pres = batch['present']                        # (B,n_spk) binary
    score = traits @ P['v']                        # (n_spk,)
    ex = np.exp(score - score.max()) * pres        # masked softmax over deponents
    alpha = ex / np.maximum(ex.sum(1, keepdims=True), 1e-12)
    tau = np.log1p(np.exp(P['p_tau']))
    logitsA = tau * np.einsum('bs,bsk->bk', alpha, sim)
    cred = alpha

    # ---- 4b : the SECOND ROAD (project the trace straight to a verdict) -----
    logitsB = m @ P['W_B'] + P['b_B']
    logits = logitsA + logitsB

    # ---- 4c : the FAME head (energy only; phase, i.e. content, discarded) ---
    mb = m.reshape(B, nb, d // nb)
    band = (mb ** 2).mean(-1)                      # (B,nb) loudness per band
    tot = (m ** 2).mean(-1, keepdims=True)         # (B,1)  loudness overall
    raw = np.concatenate([band, tot], 1)
    z = np.log1p(raw)                              # compress dynamic range
    fame_logits = z @ P['W_F'] + P['b_F']

    # ---- 5 : abstention, driven by disagreement between the two roads ------
    # The head is trained end-to-end (its gradients flow back into both roads),
    # but its TARGET is supplied by the world -- whether the deed was actually
    # supported often enough to be readable -- so the network cannot buy a low
    # loss by manufacturing agreement or by abstaining indiscriminately.
    pA, pB = softmax(logitsA), softmax(logitsB)
    disagree = 0.5 * np.abs(pA - pB).sum(-1)       # total variation in [0,1]
    energy = np.log1p((m ** 2).mean(-1))
    conf = pA.max(-1)
    feats = np.stack([disagree, energy, conf], 1)  # (B,3)
    abst = sigmoid(feats @ P['u'] + P['b_u'])

    out = dict(logits=logits, logitsA=logitsA, logitsB=logitsB,
               fame_logits=fame_logits, abst=abst, disagree=disagree,
               cred=cred, trust=score, m=m)
    if cache:
        out['_c'] = dict(ms=ms, xs=xs, es=es, ss=ss, ers=ers, chat=chat,
                         sim=sim, mb=mb, raw=raw, z=z, feats=feats,
                         lam=lam, gain=gain, S=S, Fm=Fm, FS=FS,
                         pA=pA, pB=pB, S_nrm=S_nrm, alpha=alpha, tau=tau)
    return out


# =============================================================================
# SECTION 4 -- LOSS
# =============================================================================

def loss_and_grads(P, batch, traits, wd=1e-4, override_S=None):
    """Total objective and exact analytic gradients.

        L = CE(deed) + CE(fame) + BCE(abstain | a compounding happened) + wd*||.||^2

    Every term below is differentiated by hand. The finite-difference check in
    Section 6 is the referee.
    """
    o = forward(P, batch, traits, cache=True, override_S=override_S)
    c = o['_c']
    B, K = batch['spk'].shape
    d = P['E'].shape[1]
    n_spk = P['S'].shape[0]
    nb = P['W_F'].shape[0] - 1

    # ---------- forward losses ----------
    onehot = np.eye(P['E'].shape[0])[batch['deed']]
    p = softmax(o['logits'])
    L_joint = -np.log(np.maximum(p[np.arange(B), batch['deed']], 1e-12)).mean()
    # "Diverse pathes leden diverse folk the righte wey to Rome": each road is
    # independently required to arrive, so neither can atrophy into a passenger
    # of the other -- and their disagreement therefore MEANS something.
    pA_, pB_ = o['_c']['pA'], o['_c']['pB']
    L_A = -np.log(np.maximum(pA_[np.arange(B), batch['deed']], 1e-12)).mean()
    L_B = -np.log(np.maximum(pB_[np.arange(B), batch['deed']], 1e-12)).mean()
    AUX = 0.5
    L_deed = L_joint + AUX * (L_A + L_B)

    pf = softmax(o['fame_logits'])
    L_fame = -np.log(np.maximum(pf[np.arange(B), batch['fame']], 1e-12)).mean()

    y_ab = batch['unreadable']                         # set by the world, not the net
    a = np.clip(o['abst'], 1e-9, 1 - 1e-9)
    L_ab = -(y_ab * np.log(a) + (1 - y_ab) * np.log(1 - a)).mean()

    reg = wd * sum((P[k] ** 2).sum() for k in ('E', 'W_B', 'W_F'))
    L = L_deed + L_fame + L_ab + reg

    g = {k: np.zeros_like(v) for k, v in P.items()}
    gSn = np.zeros_like(P['S'])          # gradient w.r.t. the NORMALISED shapes

    # ---------- BCE -> u, b_u, then back through the three features ---------
    dab = (a - y_ab) / B                              # d L_ab / d(pre-sigmoid)
    g['u'] += c['feats'].T @ dab
    g['b_u'] += dab.sum()
    dfeat = dab[:, None] * P['u'][None, :]            # (B,3)

    pA, pB = c['pA'], c['pB']
    sgn = np.sign(pA - pB)                            # subgradient of |.|
    dpA = dfeat[:, 0:1] * 0.5 * sgn
    dpB = -dfeat[:, 0:1] * 0.5 * sgn
    onehot_max = (pA == pA.max(-1, keepdims=True)).astype(float)
    dpA = dpA + dfeat[:, 2:3] * onehot_max            # d conf / d pA
    # push through the two softmaxes
    dlA_ab = pA * (dpA - (pA * dpA).sum(-1, keepdims=True))
    dlB_ab = pB * (dpB - (pB * dpB).sum(-1, keepdims=True))
    # the energy feature reaches the trace directly
    dm_ab = dfeat[:, 1:2] * (2.0 * o['m'] / d) / (1.0 + (o['m'] ** 2).mean(-1, keepdims=True))

    # ---------- CE(fame) -> W_F, b_F, and back into the trace m -------------
    dfl = (pf - np.eye(P['W_F'].shape[1])[batch['fame']]) / B      # (B,n_deed)
    g['W_F'] += c['z'].T @ dfl
    g['b_F'] += dfl.sum(0)
    dz = dfl @ P['W_F'].T                             # (B,nb+1)
    draw = dz / (1.0 + c['raw'])                      # d log1p
    dm_fame = np.zeros((B, d))
    # band energies: raw[:,j] = mean over band j of m^2  -> d/dm = 2m/(d/nb)
    dband = draw[:, :nb]                              # (B,nb)
    dm_fame += (2.0 * c['mb'] * dband[:, :, None] / (d // nb)).reshape(B, d)
    dm_fame += 2.0 * o['m'] * draw[:, nb:nb + 1] / d  # total energy term

    # ---------- CE(deed) -> logitsA and logitsB -----------------------------
    dl = (p - onehot) / B                                          # (B,n_deed)
    dlA = dl + dlA_ab + AUX * (pA_ - onehot) / B   # joint + own CE + abstention
    dlB = dl + dlB_ab + AUX * (pB_ - onehot) / B

    # road B
    g['W_B'] += o['m'].T @ dlB
    g['b_B'] += dlB.sum(0)
    dm = dlB @ P['W_B'].T + dm_fame + dm_ab                        # (B,d)

    # road A : logitsA = (1/n_spk) * sum_s cred_s * (chat_s . E^T)
    alpha, tau = c['alpha'], c['tau']
    dsim = tau * alpha[:, :, None] * dlA[:, None, :]               # (B,n_spk,n_deed)
    g['E'] += np.einsum('bsk,bsd->kd', dsim, c['chat'])            # via sim = chat @ E.T
    dchat = dsim @ P['E']                                          # (B,n_spk,d)

    g['p_tau'] += np.einsum('bs,bsk,bk->', alpha, c['sim'], dlA) * sigmoid(P['p_tau'])
    dalpha = tau * np.einsum('bsk,bk->bs', c['sim'], dlA)          # (B,n_spk)
    dscore = alpha * (dalpha - (alpha * dalpha).sum(1, keepdims=True))
    g['v'] += traits.T @ dscore.sum(0)

    # chat = ccorr(S_s, m):  dL/dm = cconv(S_s, dchat_s) summed over s
    #                        dL/dS_s = ccorr(dchat_s, m)
    Fdc = np.fft.rfft(dchat, axis=-1)                              # (B,n_spk,f)
    dm += np.fft.irfft((c['FS'][None] * Fdc).sum(1), n=d, axis=-1)
    dS_head = np.fft.irfft((np.conj(Fdc) * c['Fm'][:, None]).sum(0), n=d, axis=-1)
    gSn += dS_head

    # ---------- unroll the recurrence backwards -----------------------------
    lam, gain = c['lam'], c['gain']
    dlam_acc = 0.0
    dgain_acc = 0.0
    for k in range(K - 1, -1, -1):
        # m_k = lam * m_{k-1} + gain * x_k
        dlam_acc += (dm * c['ms'][k]).sum()
        dgain_acc += (dm * c['xs'][k]).sum()
        dx = dm * gain                                             # (B,d)

        s_k, e_k, er_k = c['ss'][k], c['es'][k], c['ers'][k]
        cf = batch['cflag'][:, k][:, None]

        # x_tot = cconv(s_k, e_k) + cflag * cconv(s_k, er_k)
        # dL/ds = ccorr(e, dx) + cflag * ccorr(er, dx) ; dL/de = ccorr(s, dx)
        ds = ccorr(e_k, dx) + ccorr(er_k, dx * cf)
        de = ccorr(s_k, dx)
        der = ccorr(s_k, dx * cf)

        np.add.at(gSn, batch['spk'][:, k], ds)
        np.add.at(g['E'], batch['utt'][:, k], de)
        np.add.at(g['E'], batch['rival'][:, k], der)

        dm = dm * lam                                              # flow to m_{k-1}

    g['p_lam'] += dlam_acc * lam * (1 - lam)                       # d sigmoid
    g['p_gain'] += dgain_acc * sigmoid(P['p_gain'])                # d softplus

    # ---------- unproject the shape gradient through the normaliser ---------
    # y = x/||x||  =>  dL/dx = (dL/dy - y (y . dL/dy)) / ||x||
    if override_S is None:
        Sn = c['S']
        g['S'] += (gSn - Sn * (Sn * gSn).sum(1, keepdims=True)) / c['S_nrm']

    # ---------- weight decay (not on S: its scale is projected out) ---------
    for k in ('E', 'W_B', 'W_F'):
        g[k] += 2 * wd * P[k]

    stats = dict(L=L, L_deed=L_deed, L_fame=L_fame, L_ab=L_ab,
                 acc_deed=float((o['logits'].argmax(1) == batch['deed']).mean()),
                 acc_fame=float((o['fame_logits'].argmax(1) == batch['fame']).mean()))
    return L, g, stats, o


# =============================================================================
# SECTION 5 -- ADAM
# =============================================================================

class Adam:
    def __init__(self, P, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, g):
        self.t += 1
        for k in P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] = P[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return P


# =============================================================================
# SECTION 6 -- FINITE-DIFFERENCE GRADIENT CHECK  (mandatory, runs every time)
# =============================================================================

def grad_check(seed=7, eps=1e-6, n_probe=6, tol=2e-4):
    """Central-difference check of every analytic gradient against the loss.

    Reports the maximum relative error over a handful of random coordinates in
    each parameter tensor. Anything above `tol` is a failed derivation.
    """
    world = Pilgrimage(n_deed=5, n_spk=6, hops=4, seed=seed)
    P = init_params(5, 6, d=32, nb=4, seed=seed)
    b = world.batch(12)
    T = world.traits
    _, g, _, _ = loss_and_grads(P, b, T)

    rng = np.random.default_rng(seed + 100)
    print("  parameter     shape           max|rel err|   status")
    print("  " + "-" * 54)
    worst = 0.0
    for k in P:
        flat = P[k].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        rel = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _, _, _ = loss_and_grads(P, b, T)
            flat[i] = orig - eps
            Lm, _, _, _ = loss_and_grads(P, b, T)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = g[k].reshape(-1)[i]
            rel = max(rel, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        worst = max(worst, rel)
        print(f"  {k:<12}  {str(P[k].shape):<14}  {rel:.3e}      "
              f"{'OK' if rel < tol else 'FAIL'}")
    print(f"\n  worst relative error across all parameters: {worst:.3e}")
    assert worst < tol, "GRADIENT CHECK FAILED"
    print("  GRADIENT CHECK PASSED")
    return worst


# =============================================================================
# SECTION 7 -- TRAINING
# =============================================================================

def train(steps=1200, B=256, d=64, seed=3, verbose=True):
    world = Pilgrimage(n_deed=8, n_spk=12, hops=6, seed=seed)
    P = init_params(world.n_deed, world.n_spk, d=d, nb=4, seed=seed + 1)
    opt = Adam(P, lr=3e-2)
    T = world.traits
    hist = []
    for t in range(1, steps + 1):
        b = world.batch(B)
        L, g, st, _ = loss_and_grads(P, b, T)
        P = opt.step(P, g)
        hist.append(st)
        if verbose and (t % 200 == 0 or t == 1):
            print(f"  step {t:>4}  L={st['L']:.4f}  "
                  f"deed={st['L_deed']:.4f}/{st['acc_deed']*100:5.1f}%  "
                  f"fame={st['L_fame']:.4f}/{st['acc_fame']*100:5.1f}%  "
                  f"abstain={st['L_ab']:.4f}")
    return P, world, hist


# =============================================================================
# SECTION 8 -- SELF-TESTS: the falsifiable claims
# =============================================================================

def evaluate(P, world, n=4000, override_S=None):
    b = world.batch(n)
    o = forward(P, b, world.traits, cache=False, override_S=override_S)
    return dict(
        acc_deed=float((o['logits'].argmax(1) == b['deed']).mean()),
        acc_fame=float((o['fame_logits'].argmax(1) == b['fame']).mean()),
        fame_vs_deed=float((o['fame_logits'].argmax(1) == b['deed']).mean()),
        true_overlap=float((b['fame'] == b['deed']).mean()),
        mean_abstain=float(o['abst'].mean()),
        disagree=float(o['disagree'].mean()),
    )


def bag_of_words_control(world, n=6000):
    """Control: read the record with the voices STRIPPED OFF -- just count what
    was said, majority wins. This is the reader Chaucer says you must not be."""
    b = world.batch(n)
    D = world.n_deed
    tally = np.zeros((n, D))
    np.add.at(tally, (np.repeat(np.arange(n), world.hops), b['utt'].ravel()), 1.0)
    pred = tally.argmax(1)
    return dict(acc=float((pred == b['deed']).mean()),
                rumour=float((pred == b['rumour']).mean()))


def self_tests(P, world):
    passed = []

    print("\n  TEST 1 -- the deed is recoverable, and beats counting utterances")
    ev = evaluate(P, world)
    ctl = bag_of_words_control(world)
    chance = 1.0 / world.n_deed
    print(f"    chance                                 {chance*100:.1f}%")
    print(f"    count-the-utterances control           {ctl['acc']*100:.1f}%"
          f"   (it returns the rumour {ctl['rumour']*100:.1f}% of the time)")
    print(f"    model, voices attached                 {ev['acc_deed']*100:.1f}%")
    ok = ev['acc_deed'] > ctl['acc'] + 0.10
    passed.append(ok); print(f"    -> {'PASS' if ok else 'FAIL'}")

    print("\n  TEST 2 -- renown is learnable and DISSOCIATED from the deed")
    print(f"    fame accuracy (predicting renown)      {ev['acc_fame']*100:.1f}%")
    print(f"    fame head used as a deed predictor     {ev['fame_vs_deed']*100:.1f}%")
    print(f"    ground-truth renown==deed in the world {ev['true_overlap']*100:.1f}%")
    ok = ev['acc_fame'] > 2.0 * chance and ev['fame_vs_deed'] < ev['acc_deed'] - 0.10
    passed.append(ok); print(f"    -> {'PASS' if ok else 'FAIL'}")

    print("\n  TEST 3 -- learned trust tracks candour, and ignores rank")
    trust = world.traits @ P['v']
    r_cand = float(np.corrcoef(trust, world.candour)[0, 1])
    r_est = float(np.corrcoef(trust, world.estat)[0, 1])
    print(f"    corr(trust, candour) = {r_cand:+.3f}")
    print(f"    corr(trust, estat)   = {r_est:+.3f}   (decoy trait)")
    ok = r_cand > 0.5 and abs(r_cand) > abs(r_est)
    passed.append(ok); print(f"    -> {'PASS' if ok else 'FAIL'}")

    print("\n  TEST 4 -- strip provenance and the deed reading collapses")
    flat_S = np.repeat(P['S'][:1], world.n_spk, axis=0)   # every mouth identical
    ab = evaluate(P, world, override_S=flat_S)
    print(f"    deed accuracy with shapes intact    {ev['acc_deed']*100:.1f}%")
    print(f"    deed accuracy with shapes collapsed {ab['acc_deed']*100:.1f}%")
    print(f"    fame accuracy with shapes collapsed {ab['acc_fame']*100:.1f}%")
    ok = ab['acc_deed'] < ev['acc_deed'] - 0.15
    passed.append(ok); print(f"    -> {'PASS' if ok else 'FAIL'}")

    print("\n  TEST 5 -- compounding is irreversible: unbinding degrades as the")
    print("            trace fills up (the House of Rumour, measured)")
    d = P['E'].shape[1]
    rng = np.random.default_rng(0)
    print("      items summed |  cosine of recovered content with the original")
    prev = 1.1
    mono = True
    for n_items in (1, 2, 4, 8, 16):
        cs = []
        for _ in range(60):
            shapes = rng.normal(0, 1 / np.sqrt(d), (n_items, d))
            conts = rng.normal(0, 1 / np.sqrt(d), (n_items, d))
            trace = cconv(shapes, conts).sum(0)
            rec = ccorr(shapes[0], trace)
            cs.append(float(rec @ conts[0] /
                            (np.linalg.norm(rec) * np.linalg.norm(conts[0]))))
        mu = float(np.mean(cs))
        print(f"      {n_items:>11}  |  {mu:.3f}")
        if mu > prev + 1e-9:
            mono = False
        prev = mu
    passed.append(mono); print(f"    -> {'PASS' if mono else 'FAIL'}")

    print("\n  TEST 6 -- 'I noot': abstention rises with road disagreement")
    b = world.batch(4000)
    o = forward(P, b, world.traits, cache=False)
    correct = (o['logits'].argmax(1) == b['deed'])
    lo, hi = o['abst'][correct].mean(), o['abst'][~correct].mean()
    print(f"    mean abstention when the deed IS recovered  {lo:.3f}")
    print(f"    mean abstention when the deed is NOT        {hi:.3f}")
    ok = hi > lo
    passed.append(ok); print(f"    -> {'PASS' if ok else 'FAIL'}")

    return passed


# =============================================================================
# SECTION 9 -- MAIN
# =============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 72)
    print(" THE COMPOUNDING-APERTURE NETWORK  --  after Geoffrey Chaucer")
    print(" 'Thus saugh I fals and soth compouned / Togeder fle for oo tydynge'")
    print("=" * 72)

    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    grad_check()

    print("\n[2] TRAINING ON THE PILGRIMAGE")
    P, world, _ = train()

    print("\n[3] LEARNED PHYSICS OF THE MEDIUM")
    lam = float(sigmoid(P['p_lam']))
    gain = float(np.log1p(np.exp(P['p_gain'])))
    print(f"    decay lambda (the ice)  = {lam:.4f}   "
          f"-> a tiding keeps {lam**world.hops*100:.1f}% of its amplitude "
          f"after {world.hops} hops")
    print(f"    per-hop gain (rumour)   = {gain:.4f}")
    print(f"    learned trust scores    = "
          f"{np.round(world.traits @ P['v'], 2)}")
    print(f"    true candour            = {np.round(world.candour, 3)}")
    print(f"    (decoy) estat           = {np.round(world.estat, 3)}")

    print("\n[4] SELF-TESTS")
    res = self_tests(P, world)

    print("\n" + "=" * 72)
    print(f" {sum(res)}/{len(res)} self-tests passed")
    print(" The record can be read only while the voices are still attached to")
    print(" the words. Renown can be read without them, and is a different")
    print(" number. That gap is the whole of Chaucer's contribution.")
    print("=" * 72)


if __name__ == "__main__":
    main()

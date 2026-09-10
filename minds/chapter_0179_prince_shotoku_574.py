#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE RING OF SEVENTEEN  (Jūshichi-no-Wa)
 A cognitive architecture after Prince Shōtoku / Umayado no Ōji (574-622 CE)
 Encyclopedia of Lost Minds — Chapter 0179 - Prince Shōtoku / Umayado no Ōji (574-622 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 9 Minds 161 - 180 Available on Amazon https://www.amazon.com/dp/B0HJC1QF59
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0179_prince_shotoku_574 - Prince Shōtoku / Umayado no Ōji (574-622 CE)
================================================================================  

WHY THIS IS NOT A TRANSFORMER
-----------------------------
Almost every modern model is, at heart, a *winner-take-most* device. Attention
scores a set of candidates and lets the highest score dominate; a mixture-of-
experts routes a token to the one expert who bid loudest; a classifier takes an
argmax. The machine's final act is always an act of *selection*: one hypothesis
survives, the rest are discarded, and the discarding leaves no trace.

The documents associated with Prince Shōtoku describe a mind that refuses that
move. Article X of the Seventeen-Article Constitution (604 CE) says:

    "...all men have hearts and each heart has its own leanings. The right of
    others is our wrong, and our right is their wrong. We are not unquestionably
    sages, nor are they unquestionably fools... How can anyone lay down a rule
    by which to distinguish right from wrong? For we are all wise sometimes and
    foolish at others."
                                          — tr. Aston, Nihongi (1896), Art. X

This is not a platitude about tolerance. It is a claim about *the shape of the
hypothesis space*: it is a RING, not a LADDER. There is no top. There is no
argmax. Any operator that picks a maximum has assumed a total order that
Article X explicitly denies.

So this architecture replaces every selection with a *circulation*. It has:

  1. TWELVE CAPS      — a council of 12 counselors, ranked by earned merit,
                        with rank re-issued every epoch (kan'i jūnikai, 603 CE).
  2. THE RING         — consensus by cyclic diffusion on a doubly-stochastic
                        circulant operator. "Like a ring which has no end."
  3. WA (harmony)     — a regularizer that forbids EXCLUSION but permits
                        DISAGREEMENT. Article I is a channel condition, not a
                        loss to minimize.
  4. DECLARATION-     — Article XV: "To subordinate private interests to the
     EXCISION           public good — that is the path of a vassal." Private
                        signal is forced into a designated subspace of the
                        pooled state and then deleted before the verdict.
  5. UPĀYA (expedient — Three decode frames ("vehicles"), never selected among.
     means)             The verdict is the vehicle-weighted MIXTURE. From the
                        Sangyō Gisho, the three sūtra commentaries traditionally
                        attributed to him (Lotus / Vimalakīrti / Śrīmālā), whose
                        common doctrine is ekayāna: many teachings, one vehicle.
  6. THE VIMALAKĪRTI  — an explicit SILENCE action, opened exactly when the
     GATE               vehicles' verdicts genuinely contradict. Vimalakīrti
                        answers the question of non-duality by saying nothing.

THE ONE STATISTIC THAT RUNS THE WHOLE MACHINE
---------------------------------------------
Let q(v|x) be how much each interpretive frame v applies to case x, and p_v(y|x)
the verdict that frame would return. The mixture is mu = Σ_v q_v p_v. Define

    J(x) = H(mu) - Σ_v q_v H(p_v)                        [Jensen-Shannon / MI]

J is the mutual information between FRAME and VERDICT. J = 0 means: it does not
matter which frame you adopt, the answer is the same — speak. J >> 0 means: the
answer is an artifact of the frame you happened to bring — "their right is our
wrong". That is precisely the condition under which Article X forbids the
laying-down of a rule, and under which this machine falls silent.

J is the computational content of "we are all, one with another, wise and
foolish, like a ring which has no end." It is a *measurable* quantity, and this
file measures it.

THE TASK (a court that can be bribed)
-------------------------------------
Petitions arrive at the Asuka court. Each carries:
  * PUBLIC features   (the merits of the case)
  * PRIVATE features  (the gift offered, the rank of the petitioner's patron)
  * REGISTER features (cues to which body of law the case falls under)

Three bodies of law exist (the three "vehicles"). The SAME public facts yield
DIFFERENT verdicts under different bodies of law. Three traps are planted:

  TRAP 1 (Article XV, the bribe): in TRAINING, the private features are
    correlated with the correct verdict. A lazy court reads the verdict off the
    bribe and is never wrong. At TEST time the correlation is PERMUTED. The
    bribes now point at wrong verdicts. Any model that used them collapses.

  TRAP 2 (Article X, the tie): some cases are ambiguous between two bodies of
    law that give CONTRADICTORY verdicts. The correct answer is SILENCE.

  TRAP 3 (the concord trap): some cases are *equally* ambiguous between two
    bodies of law that happen to AGREE. Here silence is WRONG — the case must be
    decided. A model that abstains whenever the register is uncertain fails this
    set. Only a model that checks whether the FRAME CHANGES THE ANSWER — i.e.
    that computes J — can pass Trap 2 and Trap 3 at the same time.

Trap 3 is the whole argument. It is what separates "I am confused" from "the
question is malformed" — and Article X is about the second, not the first.

HISTORICAL HONESTY
------------------
Every text used here is of contested authorship. The Seventeen-Article
Constitution survives only inside the Nihon Shoki (720 CE), a century after the
prince's death; Ōyama Seiichi (1999) argues the "Shōtoku" of tradition is
largely an eighth-century construction. The Sangyō Gisho commentaries are
disputed. What is NOT disputed is that a coherent doctrine of mind was
assembled under this name, in this place, in this century — and it is that
doctrine, not a biography, that this architecture reconstructs.

ENGINEERING NOTES
-----------------
* Pure NumPy. No autograd, no framework. All gradients hand-derived.
* A finite-difference gradient check over EVERY trainable tensor is mandatory
  and runs on every invocation (see `gradient_check`).
* Merit ranks are NOT learned by backprop. They are re-issued each epoch by an
  outer audit loop, because in Shōtoku's design the cap-rank system sits ABOVE
  the individual officials — it is an institution, not a habit.

Run:  python3 chapter_0179_prince_shotoku_574.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(604)          # the year of the Constitution
EPS = 1e-12

# ---------------------------------------------------------------------------
# Verdict vocabulary. Four judgements the court may hand down, plus MOKU.
# ---------------------------------------------------------------------------
VERDICTS = ["GRANT", "DENY", "REMAND", "MITIGATE"]
C = len(VERDICTS)          # decidable verdicts
MOKU = C                   # 黙 — the silence of Vimalakīrti. The 5th action.
N_ACT = C + 1

# The three vehicles (upāya). Three bodies of law that may each claim a case.
VEHICLES = ["DHARMA (vinaya frame)", "CORVEE (agrarian frame)", "UJI (kin frame)"]
V = len(VEHICLES)

# The six virtues of the twelve-cap system (kan'i jūnikai, 603 CE), each issued
# in a greater (dai) and a lesser (shō) grade -> twelve counselors.
VIRTUES = ["toku/virtue", "nin/benevolence", "rei/propriety",
           "shin/faith", "gi/righteousness", "chi/wisdom"]
M = 2 * len(VIRTUES)       # 12 counselors

D_PUB, D_PRIV, D_REG = 20, 4, 3
D_IN = D_PUB + D_PRIV      # counselors hear the bribe. They cannot un-hear it.
H = 32                     # counselor state width
K_PRIV = 6                 # size of the "declared private interest" subspace
H_PUB = H - K_PRIV         # what survives excision and reaches the verdict


# ===========================================================================
# SECTION 1 — THE COURT DOCKET (data)
# ===========================================================================
def make_docket(n, rng, permute_bribes=False):
    """
    Build n petitions.

    Returns
      X    (n, D_PUB+D_PRIV+D_REG)  the petition as the court receives it
      y    (n,)                     correct action in 0..MOKU
      kind (n,)                     0 = plain, 1 = contested (-> MOKU),
                                    2 = CONCORD trap (ambiguous but agreeing)

    `permute_bribes=True` produces the deployment distribution: the gift-giving
    conventions have shifted, and a gift that meant GRANT last year now means
    DENY. Nothing about the *merits* of a case has changed. Only the corruption
    has changed its costume.
    """
    # The three bodies of law. Fixed once, shared by train and test: the LAW
    # does not move. Only the bribes do.
    law_rng = np.random.default_rng(17)          # seventeen articles
    LAW = law_rng.normal(0, 1.0, size=(V, D_PUB, C))

    xpub = rng.normal(0, 1.0, size=(n, D_PUB))
    # verdict each vehicle would return, for every case
    per_v = np.stack([np.argmax(xpub @ LAW[v], axis=1) for v in range(V)], 1)  # (n,V)

    kind = np.zeros(n, dtype=int)
    y = np.zeros(n, dtype=int)
    xreg = np.zeros((n, D_REG))

    for i in range(n):
        r = rng.random()
        if r < 0.60:
            # ---- plain case: one body of law clearly governs -----------------
            v = rng.integers(V)
            xreg[i] = rng.normal(0, .25, D_REG)
            xreg[i, v] += 2.4                      # a clear register cue
            y[i] = per_v[i, v]
            kind[i] = 0
        else:
            # ---- ambiguous register: two bodies of law claim the case --------
            v1, v2 = rng.choice(V, size=2, replace=False)
            xreg[i] = rng.normal(0, .25, D_REG)
            xreg[i, v1] += 1.6
            xreg[i, v2] += 1.6                     # equally cued. A true tie.
            if per_v[i, v1] == per_v[i, v2]:
                # CONCORD TRAP: the frames disagree about jurisdiction but
                # agree about justice. The case MUST be decided. J == 0 here.
                y[i] = per_v[i, v1]
                kind[i] = 2
            else:
                # CONTESTED: "their right is our wrong". Article X applies.
                y[i] = MOKU
                kind[i] = 1

    # ---- the bribe (Article XV / Article V) ---------------------------------
    # A gift encodes the verdict its giver wants. In training that is (usually)
    # the verdict the court in fact reaches -- so the gift is a perfect, free,
    # entirely corrupt predictor.
    GIFT = np.eye(C, D_PRIV) * 2.2 + 0.35          # gift-signature per verdict
    perm = np.array([2, 3, 1, 0]) if permute_bribes else np.arange(C)
    xpriv = rng.normal(0, .30, size=(n, D_PRIV))
    for i in range(n):
        want = y[i] if y[i] != MOKU else per_v[i, rng.integers(V)]
        xpriv[i] += GIFT[perm[want]]
        if rng.random() < 0.12:                     # 12% of petitioners are poor
            xpriv[i] = rng.normal(0, .30, D_PRIV)

    X = np.concatenate([xpub, xpriv, xreg], axis=1)
    return X.astype(np.float64), y.astype(int), kind


# ===========================================================================
# SECTION 2 — PARAMETERS
# ===========================================================================
def init_params(rng):
    """Twelve counselors, a ring kernel, a declaration head, three vehicles."""
    def g(*s, sc=None):
        sc = sc if sc is not None else 1.0 / np.sqrt(s[-2] if len(s) > 1 else s[0])
        return rng.normal(0, sc, size=s)

    P = {}
    # --- the twelve caps: each counselor is a distinct linear reader ---------
    P["Wc"] = g(M, D_IN, H, sc=0.35)
    P["bc"] = np.zeros((M, H))
    # --- the ring kernel over offsets (-2,-1,0,+1,+2) on the cycle -----------
    #     Initialized to favour staying put; the model learns how far a voice
    #     should travel around the circle before the next round of speech.
    P["kappa"] = np.array([-0.5, 0.3, 1.0, 0.3, -0.5])
    # --- salience used (with merit) to pool the council ----------------------
    P["s"] = np.zeros(M)
    # --- Article XV: the declaration head reads private interest out of the
    #     reserved subspace, which forces the private signal to live there ----
    P["A"] = g(K_PRIV, D_PRIV, sc=0.4)
    P["a0"] = np.zeros(D_PRIV)
    # --- upāya: which vehicle(s) claim this case ----------------------------
    P["G"] = g(D_REG, V, sc=0.8)
    P["g0"] = np.zeros(V)
    # --- three decode frames, reading only the PUBLIC remainder -------------
    P["D"] = g(V, H_PUB, C, sc=0.35)
    P["e"] = np.zeros((V, C))
    # --- solo heads: every official must be able to answer for his own sphere
    #     (Art. VII, XIII). Used by the audit that re-issues the caps.
    P["S"] = g(M, H, C, sc=0.3)
    P["s0"] = np.zeros((M, C))
    # --- the Vimalakīrti gate ------------------------------------------------
    #     [ w_J , w_R , w_P(raw) , bias ]
    #     w_P is passed through softplus, so its effective value is ALWAYS >= 0.
    #     That sign constraint is the entire moral content of the gate: the
    #     declared size of a gift can raise the probability of silence and can
    #     never lower it. A bribe cannot buy a verdict. It can only buy a refusal.
    P["gate"] = np.array([2.0, 0.5, -1.0, -1.5])
    return P


class Config:
    """
    Governance constants. Not learned; CHOSEN, as a constitution is chosen.

    The boolean switches below are the articles themselves. Turning one off is
    how the ablation study repeals an article and watches the court fail.
    """
    lam = 0.5          # how much of each round is spent listening
    T = 4              # rounds of circulation before the council speaks
    beta = 0.35        # WA: strength of the anti-exclusion (entropy) term
    delta = 0.60       # strength of the declaration (Article XV) loss
    eps_solo = 0.15    # strength of the solo-competence loss
    offs = np.array([-2, -1, 0, 1, 2])

    # --- the articles, as switches ------------------------------------------
    mask_private = True    # Art. XV : the wall. Private features may write ONLY
                           #           into the declared subspace, never into the
                           #           units the verdict reads.
    pool = "wa"            # Art. X  : "wa" = the whole council is pooled.
                           #           "top" = rule by the single highest cap
                           #           (the ladder, not the ring).
    upaya = "mixture"      # Sangyō  : "mixture" = every vehicle speaks, none is
                           #           chosen. "argmax" = adopt the loudest frame
                           #           and forget the others.
    silence = True         # Vimalakīrti: may the court decline to answer?


CFG = Config()


def input_mask(cfg):
    """
    ARTICLE XV AS A LOAD-BEARING WALL.

        "To subordinate private interests to the public good — that is the path
        of a vassal."                                            — Article XV

    A soft penalty ("please don't use the bribe") does not work: the network
    simply encodes the bribe redundantly and reads it back. We tried that; the
    court kept taking gifts. So the article is enforced as an ARCHITECTURAL
    IMPOSSIBILITY, not a preference.

    The mask forbids the private features from writing into any hidden unit that
    the verdict heads will later read. The counselor still HEARS the gift — he
    must, or he could not declare it — but the gift can only reach the first
    K_PRIV units of his state: his declaration. Those units are then excised.

    Because the ring mixes counselors index-by-index, the wall survives
    circulation: a public unit never touches a private one, on any round.
    Therefore z_pub is provably a function of the merits alone.
    """
    Ms = np.ones((D_IN, H))
    if cfg.mask_private:
        Ms[D_PUB:, K_PRIV:] = 0.0
    return Ms

# Capacity: greater (dai) and lesser (shō) grade of each of the six virtues.
# The lesser grade is a *narrower* nonlinearity: it sees less sharply. This is
# the only respect in which the twelve counselors are unequal by construction.
GAMMA = np.repeat(np.array([1.0, 0.62]), 6)        # (12,)


def ring_operator(kappa, rho):
    """
    Build the doubly-stochastic circulant mixing operator, tilted by merit.

    softmax over the five offsets gives a row-stochastic circulant matrix; a
    circulant row-stochastic matrix is automatically DOUBLY stochastic, because
    every column is a rotation of the same weight vector. That is what makes
    this a ring "which has no end": no seat is privileged by position.

    Merit `rho` then scales each counselor's OUTGOING influence, and rows are
    renormalized. Position is equal; earned rank is not. That is exactly the
    twelve-cap reform: the seat you occupy is not the seat you were born to.
    """
    kk = np.exp(kappa - kappa.max()); kk = kk / kk.sum()
    Cm = np.zeros((M, M))
    for oi, o in enumerate(CFG.offs):
        for i in range(M):
            Cm[i, (i + o) % M] += kk[oi]
    U = Cm * rho[None, :]
    srow = U.sum(1, keepdims=True)
    return U / srow, Cm, kk, U, srow


# ===========================================================================
# SECTION 3 — FORWARD
# ===========================================================================
def forward(P, X, rho, cfg=CFG):
    n = X.shape[0]
    xin = X[:, :D_IN]                       # public + private: the whole petition
    xpriv = X[:, D_PUB:D_PUB + D_PRIV]
    xreg = X[:, D_IN:]

    # --- (1) the twelve counselors read the petition, each in his own basis --
    #     Weff = W * MASK. The counselor hears the whole petition, gift included;
    #     the mask decides where in his mind the gift is allowed to land.
    MSK = input_mask(cfg)
    Weff = P["Wc"] * MSK[None, :, :]
    pre = np.einsum("nd,mdh->mnh", xin, Weff) + P["bc"][:, None, :]        # (M,n,H)
    Hs = [np.tanh(GAMMA[:, None, None] * pre)]

    # --- (2) THE RING: T rounds of circulation. Nobody is overruled; every
    #         state is *moved toward* its neighbours and keeps its residue.
    Chat, Cm, kk, U, srow = ring_operator(P["kappa"], rho)
    for _ in range(cfg.T):
        Hs.append((1 - cfg.lam) * Hs[-1] + cfg.lam * np.einsum("ij,jnh->inh", Chat, Hs[-1]))
    HT = Hs[-1]                                                            # (M,n,H)

    # --- (3) WA: pool the council. `a` is influence, = salience x merit. -----
    alog = P["s"] + np.log(rho + EPS)
    if cfg.pool == "top":
        # THE LADDER. Article X repealed: obey the single highest cap. This is
        # what every argmax/router in modern ML does, dressed in a court robe.
        a = np.zeros(M); a[int(np.argmax(alog))] = 1.0
    else:
        a = np.exp(alog - alog.max()); a = a / a.sum()                     # (M,)
    z = np.einsum("m,mnh->nh", a, HT)                                      # (n,H)

    # --- (4) DECLARATION & EXCISION (Article XV) -----------------------------
    #     The first K_PRIV units of the pooled state are the counselor's
    #     declaration of private interest. A head must reconstruct the gift
    #     from them -- which pushes ALL bribe-information into that subspace.
    #     They are then cut away. The verdict never sees them.
    z_priv, z_pub = z[:, :K_PRIV], z[:, K_PRIV:]
    xhat = z_priv @ P["A"] + P["a0"]                                       # (n,D_PRIV)
    L_decl = np.mean((xhat - xpriv) ** 2)

    # --- (5) UPĀYA: every vehicle speaks. None is chosen. --------------------
    ql = xreg @ P["G"] + P["g0"]
    ql -= ql.max(1, keepdims=True)
    q = np.exp(ql); q /= q.sum(1, keepdims=True)                           # (n,V)
    q_soft = q
    if cfg.upaya == "argmax":
        # Expedient means repealed: adopt the loudest frame, forget the rest.
        # Note what this destroys: with one vehicle at probability 1, J == 0
        # ALWAYS, because a single frame can never disagree with itself. The
        # machine becomes incapable of noticing that its answer was an artifact
        # of its own vantage. It will be confident, and it will be silent-blind.
        q = np.eye(V)[np.argmax(q, 1)]

    lv = np.einsum("nh,vhc->vnc", z_pub, P["D"]) + P["e"][:, None, :]      # (V,n,C)
    lv -= lv.max(2, keepdims=True)
    pv = np.exp(lv); pv /= pv.sum(2, keepdims=True)                        # (V,n,C)
    mu = np.einsum("nv,vnc->nc", q, pv)                                    # (n,C)

    # --- (6) THE VIMALAKĪRTI GATE -------------------------------------------
    #     J = H(mu) - Σ_v q_v H(p_v). Zero iff the frame does not change the
    #     answer. This is the machine's entire theory of when to shut up.
    Hmu = -np.sum(mu * np.log(mu + EPS), 1)                                # (n,)
    Hpv = -np.sum(pv * np.log(pv + EPS), 2)                                # (V,n)
    J = Hmu - np.sum(q.T * Hpv, 0)                                         # (n,)
    #     Ring residual: how far from consensus the council still stands.
    dev = HT - z[None, :, :]
    R = np.mean(dev ** 2, axis=(0, 2))                                     # (n,)

    #     Π: the declared magnitude of private pressure on this case -- the size
    #     of the gift, as the court has openly admitted receiving it.
    Pi = np.mean(xhat ** 2, axis=1)                                        # (n,)
    wP = np.log1p(np.exp(-np.abs(P["gate"][2]))) + max(P["gate"][2], 0.0)  # softplus
    sg = P["gate"][0] * J + P["gate"][1] * R + wP * Pi + P["gate"][3]
    sig = 1.0 / (1.0 + np.exp(-sg))                                        # (n,)
    if not cfg.silence:
        sig = np.zeros_like(sig)          # the court MUST answer. Every time.

    #     Final action distribution over 5 actions. Sums to 1 by construction.
    Pact = np.concatenate([(1 - sig)[:, None] * mu, sig[:, None]], 1)      # (n,5)

    # --- solo readouts, for the audit that re-issues the twelve caps ---------
    sl = np.einsum("mnh,mhc->mnc", HT, P["S"]) + P["s0"][:, None, :]
    sl -= sl.max(2, keepdims=True)
    ps = np.exp(sl); ps /= ps.sum(2, keepdims=True)                        # (M,n,C)

    cache = dict(X=X, xin=xin, xpriv=xpriv, xreg=xreg, pre=pre, Hs=Hs, HT=HT,
                 Chat=Chat, Cm=Cm, kk=kk, U=U, srow=srow, a=a, z=z, MSK=MSK,
                 z_priv=z_priv, z_pub=z_pub, xhat=xhat, q=q, q_soft=q_soft,
                 pv=pv, mu=mu, Pi=Pi, wP=wP,
                 Hmu=Hmu, Hpv=Hpv, J=J, R=R, dev=dev, sig=sig, Pact=Pact,
                 ps=ps, rho=rho, n=n, L_decl=L_decl, cfg=cfg)
    return Pact, cache


def loss_fn(P, X, y, rho, cfg=CFG):
    Pact, ca = forward(P, X, rho, cfg)
    n = ca["n"]
    # If the silence action has been repealed, the contested cases cannot be
    # taught at all -- the court has no vocabulary for them. We drop them from
    # the objective rather than train on an impossible target, and count them as
    # errors at evaluation. That IS the cost of repealing Vimalakīrti.
    w = np.ones(n) if cfg.silence else (y != MOKU).astype(float)
    ca["w"] = w
    nw = max(w.sum(), 1.0)
    ytr_ = y if cfg.silence else np.where(y == MOKU, 0, y)   # dummy index, weight 0
    ce = -np.sum(w * np.log(Pact[np.arange(n), ytr_] + EPS)) / nw

    # WA (Article I): harmony is an ANTI-EXCLUSION constraint. We penalise the
    # council for silencing any of its own members. We do NOT penalise them for
    # disagreeing -- disagreement is the working material, not the fault.
    a = ca["a"]
    Ha = -np.sum(a * np.log(a + EPS))
    L_wa = -Ha                                   # minimising this maximises H(a)

    # Article XV
    L_decl = ca["L_decl"]

    # Article VII / XIII: each official individually answerable in his sphere.
    dec = y != MOKU
    if dec.sum() > 0:
        lp = np.log(ca["ps"][:, dec, :] + EPS)
        L_solo = -np.mean(lp[:, np.arange(dec.sum()), y[dec]])
    else:
        L_solo = 0.0

    total = ce + cfg.beta * L_wa + cfg.delta * L_decl + cfg.eps_solo * L_solo
    return total, ca, dict(ce=ce, wa=Ha, decl=L_decl, solo=L_solo)


# ===========================================================================
# SECTION 4 — BACKWARD (every derivative by hand)
# ===========================================================================
def backward(P, ca, y, cfg=CFG):
    n = ca["n"]
    g = {k: np.zeros_like(v) for k, v in P.items()}

    q, pv, mu, sig = ca["q"], ca["pv"], ca["mu"], ca["sig"]
    Pact, HT, a, z = ca["Pact"], ca["HT"], ca["a"], ca["z"]
    w = ca["w"]; nw = max(w.sum(), 1.0)

    # ---- d(CE)/d(sigma), d(CE)/d(mu) ---------------------------------------
    dsig = np.zeros(n); dmu = np.zeros((n, C))
    dec = y != MOKU
    idx = np.arange(n)
    dsel = dec & (w > 0)
    # decidable rows: P = (1-sig)*mu_y  ->  -log P = -log(1-sig) - log(mu_y)
    dmu[idx[dsel], y[dsel]] = -1.0 / (mu[idx[dsel], y[dsel]] + EPS) / nw
    if cfg.silence:
        dsig[dsel] = (1.0 / (1 - sig[dsel] + EPS)) / nw
        # silent rows: P = sig
        ssel = (~dec) & (w > 0)
        dsig[ssel] = (-1.0 / (sig[ssel] + EPS)) / nw

    # ---- through the gate ---------------------------------------------------
    if cfg.silence:
        dsg = dsig * sig * (1 - sig)                              # (n,)
    else:
        dsg = np.zeros(n)                                         # gate is welded shut
    g["gate"][0] = np.sum(dsg * ca["J"])
    g["gate"][1] = np.sum(dsg * ca["R"])
    # softplus on the bribe coupling: d(softplus(x))/dx = sigmoid(x)
    g["gate"][2] = np.sum(dsg * ca["Pi"]) * (1.0 / (1.0 + np.exp(-P["gate"][2])))
    g["gate"][3] = np.sum(dsg)
    dJ = dsg * P["gate"][0]
    dR = dsg * P["gate"][1]
    dPi = dsg * ca["wP"]                                          # (n,)

    # NOTE: mu does NOT pick up a (1-sig) factor here. The action probability is
    # (1-sig)*mu_y, so -log P = -log(1-sig) - log(mu_y): the two factors separate
    # additively under the log and each gets its own clean derivative.

    # ---- J = H(mu) - Σ_v q_v H(p_v) ----------------------------------------
    logmu = np.log(mu + EPS); logpv = np.log(pv + EPS)
    dmu_fromJ = dJ[:, None] * (-(logmu + 1.0))                   # ∂H(mu)/∂mu
    # H(mu) inside J depends on mu, and mu depends on q and pv, so we fold this
    # into the same dmu channel that is routed to q and pv below.
    dmu_tot = dmu + dmu_fromJ

    # explicit -Σ_v q_v H(p_v) pieces
    dq = np.zeros((n, V))
    dpv = np.zeros((V, n, C))
    for v in range(V):
        dq[:, v] += dJ * (-(-np.sum(pv[v] * logpv[v], 1)))       # -H(p_v)
        dpv[v] += dJ[:, None] * q[:, v][:, None] * (logpv[v] + 1.0)

    # mu = Σ_v q_v p_v  → route dmu_tot to q and pv
    for v in range(V):
        dq[:, v] += np.sum(dmu_tot * pv[v], 1)
        dpv[v] += dmu_tot * q[:, v][:, None]

    # ---- softmax backprops ---------------------------------------------------
    dlv = np.zeros((V, n, C))
    for v in range(V):
        dlv[v] = pv[v] * (dpv[v] - np.sum(dpv[v] * pv[v], 1, keepdims=True))
    dql = q * (dq - np.sum(dq * q, 1, keepdims=True))

    if cfg.upaya == "argmax":
        dql[:] = 0.0        # the frame was SELECTED, not weighted: no gradient
    g["G"] = ca["xreg"].T @ dql
    g["g0"] = dql.sum(0)

    z_pub = ca["z_pub"]
    dz_pub = np.zeros_like(z_pub)
    for v in range(V):
        g["D"][v] = z_pub.T @ dlv[v]
        g["e"][v] = dlv[v].sum(0)
        dz_pub += dlv[v] @ P["D"][v].T

    # ---- Article XV: declaration loss + the one-way bribe coupling ----------
    dxhat = 2.0 * (ca["xhat"] - ca["xpriv"]) / (n * D_PRIV) * cfg.delta
    dxhat += dPi[:, None] * 2.0 * ca["xhat"] / D_PRIV          # Π = mean(xhat^2)
    g["A"] = ca["z_priv"].T @ dxhat
    g["a0"] = dxhat.sum(0)
    dz_priv = dxhat @ P["A"].T

    dz = np.concatenate([dz_priv, dz_pub], 1)                    # (n,H)

    # ---- ring residual R ----------------------------------------------------
    dev = ca["dev"]                                              # (M,n,H)
    dHT = np.zeros_like(HT)
    coef = (2.0 / (M * H)) * dR                                  # (n,)
    dHT += coef[None, :, None] * dev
    dz += -coef[:, None] * dev.sum(0)

    # ---- solo heads ---------------------------------------------------------
    ps = ca["ps"]
    if dec.sum() > 0:
        nd = dec.sum()
        dps = np.zeros_like(ps)
        sel = np.where(dec)[0]
        dps[:, sel, y[dec]] = -1.0 / (ps[:, sel, y[dec]] + EPS) / (M * nd) * cfg.eps_solo
        dsl = ps * (dps - np.sum(dps * ps, 2, keepdims=True))
        for m in range(M):
            g["S"][m] = HT[m].T @ dsl[m]
            g["s0"][m] = dsl[m].sum(0)
            dHT[m] += dsl[m] @ P["S"][m].T

    # ---- WA entropy term ----------------------------------------------------
    dHT += a[:, None, None] * dz[None, :, :]
    if cfg.pool == "top":
        g["s"] = np.zeros(M)       # a hard selection carries no gradient
    else:
        Ha = -np.sum(a * np.log(a + EPS))
        dHa = -cfg.beta                                          # d(beta * -Ha)/dHa
        da_ent = dHa * (-(np.log(a + EPS) + 1.0))                # (M,)
        da = np.einsum("nh,mnh->m", dz, HT) + da_ent
        dalog = a * (da - np.sum(da * a))
        g["s"] = dalog

    # ---- unroll THE RING ----------------------------------------------------
    Chat = ca["Chat"]
    dChat = np.zeros((M, M))
    for t in range(cfg.T, 0, -1):
        Hprev = ca["Hs"][t - 1]
        dChat += cfg.lam * np.einsum("inh,jnh->ij", dHT, Hprev)
        dHT = (1 - cfg.lam) * dHT + cfg.lam * np.einsum("ij,inh->jnh", Chat, dHT)

    # Chat = U / rowsum(U); U = Cm * rho
    srow = ca["srow"]
    dU = (dChat - np.sum(dChat * Chat, 1, keepdims=True)) / srow
    dCm = dU * ca["rho"][None, :]
    dkk = np.zeros(len(CFG.offs))
    for oi, o in enumerate(CFG.offs):
        for i in range(M):
            dkk[oi] += dCm[i, (i + o) % M]
    kk = ca["kk"]
    g["kappa"] = kk * (dkk - np.sum(dkk * kk))

    # ---- counselors ---------------------------------------------------------
    dpre = dHT * GAMMA[:, None, None] * (1 - ca["Hs"][0] ** 2)
    g["Wc"] = np.einsum("nd,mnh->mdh", ca["xin"], dpre) * ca["MSK"][None, :, :]
    g["bc"] = dpre.sum(1)
    return g


# ===========================================================================
# SECTION 5 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ===========================================================================
def _check_one(cfg, label, show=True):
    """Finite-difference check of every trainable tensor under one constitution."""
    rng = np.random.default_rng(7)
    X, y, _ = make_docket(24, rng)
    P = init_params(rng)
    rho = np.linspace(0.55, 1.45, M)          # some non-uniform merit
    rng.shuffle(rho)

    _, ca, _ = loss_fn(P, X, y, rho, cfg)
    G = backward(P, ca, y, cfg)

    worst, h = 0.0, 1e-6
    rows = []
    for name in sorted(P.keys()):
        arr = P[name]
        flat = arr.reshape(-1)
        gflat = G[name].reshape(-1)
        probes = rng.choice(flat.size, min(12, flat.size), replace=False)
        errs = []
        for p in probes:
            o = flat[p]
            flat[p] = o + h; Lp, _, _ = loss_fn(P, X, y, rho, cfg)
            flat[p] = o - h; Lm, _, _ = loss_fn(P, X, y, rho, cfg)
            flat[p] = o
            num = (Lp - Lm) / (2 * h)
            ana = gflat[p]
            # tensors with a structurally zero gradient (masked / hard-selected)
            # are correct at exactly zero; guard the relative denominator.
            den = max(abs(num), abs(ana), 1e-8)
            errs.append(abs(num - ana) / den)
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, arr.shape, e))
    if show:
        print(f"  --- {label} ---")
        print(f"  {'tensor':<8}{'shape':<18}{'max |rel err|':>14}   {'verdict':>8}")
        for name, shp, e in rows:
            print(f"  {name:<8}{str(shp):<18}{e:>14.3e}   {'PASS' if e < 1e-4 else 'FAIL':>8}")
    print(f"  [{label}] worst relative error over all tensors: {worst:.3e}  "
          f"-> {'PASS' if worst < 1e-4 else 'FAIL'}")
    assert worst < 1e-4, f"GRADIENT CHECK FAILED ({label})"
    return worst


def gradient_check():
    """
    Mandatory. An architecture that cannot prove its own derivative is a rumour,
    and Article XII of this codebase forbids acting on rumour.

    We check the full constitution AND each repealed variant, because an ablation
    whose gradients are wrong is not an experiment, it is a story.
    """
    import copy
    print("  finite-difference gradient check (h=1e-6, central difference)")
    worst = _check_one(CFG, "FULL constitution", show=True)
    for lbl, mut in [("Art.XV repealed", lambda c: setattr(c, "mask_private", False)),
                     ("Art.X repealed (top-cap rule)", lambda c: setattr(c, "pool", "top")),
                     ("upāya repealed (argmax frame)", lambda c: setattr(c, "upaya", "argmax")),
                     ("silence repealed", lambda c: setattr(c, "silence", False)),
                     ("Art.I repealed (no wa)", lambda c: setattr(c, "beta", 0.0))]:
        c = copy.copy(CFG); mut(c)
        worst = max(worst, _check_one(c, lbl, show=False))
    print(f"  ALL GRADIENTS VERIFIED under every constitution (worst {worst:.2e}).\n")
    return worst


# ===========================================================================
# SECTION 6 — THE TWELVE CAPS: an outer, non-gradient governance loop
# ===========================================================================
def reissue_caps(P, Xaudit, yaudit, rho_prev, cfg=CFG):
    """
    Article XI: "Know the difference between merit and demerit, and deal out to
    each its reward and punishment."
    Article VII: "the wise sovereigns of antiquity sought the man to fill the
    office, and not the office to suit the man."

    Rank is measured, not inherited. Each counselor is audited ALONE, on his own
    solo readout, against held-out cases. The twelve caps are then handed out in
    order of measured competence -- and can be taken back next season.

    This is deliberately NOT a learned gate. A learned router would let the
    network *decide* who is competent, which is precisely the hereditary
    arrangement the twelve-cap reform abolished.
    """
    _, ca = forward(P, Xaudit, rho_prev, cfg)
    ps = ca["ps"]
    dec = yaudit != MOKU
    scores = np.zeros(M)
    for m in range(M):
        pm = ps[m][dec]
        scores[m] = np.mean(np.log(pm[np.arange(dec.sum()), yaudit[dec]] + EPS))
    order = np.argsort(scores)                 # worst -> best
    caps = np.linspace(0.55, 1.45, M)          # twelve graded caps
    rho = np.zeros(M)
    rho[order] = caps
    churn = int(np.sum(np.argsort(np.argsort(rho)) != np.argsort(np.argsort(rho_prev))))
    return rho, scores, churn


# ===========================================================================
# SECTION 7 — TRAINING
# ===========================================================================
def adam_step(P, G, state, lr):
    b1, b2, e = 0.9, 0.999, 1e-8
    state["t"] += 1
    t = state["t"]
    for k in P:
        m = state["m"].setdefault(k, np.zeros_like(P[k]))
        v = state["v"].setdefault(k, np.zeros_like(P[k]))
        m[:] = b1 * m + (1 - b1) * G[k]
        v[:] = b2 * v + (1 - b2) * G[k] ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + e)


def evaluate(P, X, y, kind, rho, cfg=CFG):
    Pact, ca = forward(P, X, rho, cfg)
    pred = np.argmax(Pact, 1)
    acc = float(np.mean(pred == y))
    out = dict(acc=acc)
    for nm, k in (("plain", 0), ("contested", 1), ("concord", 2)):
        msk = kind == k
        out[nm] = float(np.mean(pred[msk] == y[msk])) if msk.sum() else float("nan")
    # silence precision/recall
    ps_, ts = pred == MOKU, y == MOKU
    tp = np.sum(ps_ & ts)
    prec = tp / max(ps_.sum(), 1)
    rec = tp / max(ts.sum(), 1)
    out["silence_P"] = float(prec)
    out["silence_R"] = float(rec)
    out["silence_F1"] = float(2 * prec * rec / max(prec + rec, 1e-9))
    out["J_contested"] = float(np.mean(ca["J"][kind == 1]))
    out["J_concord"] = float(np.mean(ca["J"][kind == 2]))
    out["J_plain"] = float(np.mean(ca["J"][kind == 0]))
    out["H_a"] = float(-np.sum(ca["a"] * np.log(ca["a"] + EPS)))
    return out


def train(cfg=CFG, epochs=45, bs=256, lr=6e-3, seed=604, quiet=False, tag="FULL"):
    rng = np.random.default_rng(seed)
    Xtr, ytr, ktr = make_docket(7000, rng)
    Xau, yau, kau = make_docket(1200, rng)                      # audit split
    Xte, yte, kte = make_docket(2500, rng)                      # clean test
    Xsh, ysh, ksh = make_docket(2500, rng, permute_bribes=True)  # deployment

    P = init_params(np.random.default_rng(seed + 1))
    rho = np.ones(M)
    st = dict(t=0, m={}, v={})
    n = len(ytr)
    hist = []

    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            L, ca, parts = loss_fn(P, Xtr[b], ytr[b], rho, cfg)
            G = backward(P, ca, ytr[b], cfg)
            adam_step(P, G, st, lr)
        # --- outer governance loop: the caps are re-issued each season -------
        rho, scores, churn = reissue_caps(P, Xau, yau, rho, cfg)
        if ep % 9 == 0 or ep == 1:
            ev = evaluate(P, Xte, yte, kte, rho, cfg)
            hist.append((ep, churn, ev))
            if not quiet:
                print(f"  [{tag}] ep{ep:>3}  loss {L:6.3f} | ce {parts['ce']:5.3f} "
                      f"decl {parts['decl']:5.3f} H(a) {parts['wa']:4.2f} | "
                      f"test {ev['acc']*100:5.1f}%  silence-F1 {ev['silence_F1']:4.2f} "
                      f"| cap churn {churn:>2}/12")
    return P, rho, dict(clean=(Xte, yte, kte), shift=(Xsh, ysh, ksh))


# ===========================================================================
# SECTION 8 — ABLATIONS: each one removes an article and shows the wound
# ===========================================================================
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print(" THE RING OF SEVENTEEN — Prince Shōtoku (574-622) — Chapter 0179")
    print(" a council of twelve, a ring with no end, and a permitted silence")
    print("=" * 78)

    print("\n[0] STRUCTURE")
    rng0 = np.random.default_rng(0)
    Pd = init_params(rng0)
    tot = sum(v.size for v in Pd.values())
    print(f"  counselors (twelve caps) : {M}  = 6 virtues x (dai / shō)")
    print(f"  vehicles (upāya)         : {V}  -> {', '.join(VEHICLES)}")
    print(f"  actions                  : {N_ACT} -> {VERDICTS + ['MOKU (silence)']}")
    print(f"  ring offsets             : {list(CFG.offs)}  (cyclic, doubly stochastic)")
    print(f"  pooled state             : {H}  = {K_PRIV} declared-private + {H_PUB} public")
    print(f"  trainable parameters     : {tot:,}")
    Ch, _, kk, _, _ = ring_operator(Pd["kappa"], np.ones(M))
    print(f"  ring kernel (init)       : {kk.round(3)}  row-sums {Ch.sum(1)[:3].round(6)} "
          f" col-sums {Ch.sum(0)[:3].round(6)}   <- doubly stochastic")

    print("\n[1] GRADIENT CHECK")
    gradient_check()

    print("[2] THE DOCKET")
    r = np.random.default_rng(99)
    _, yy, kk2 = make_docket(4000, r)
    for nm, k in (("plain (one law governs)", 0),
                  ("contested (laws collide -> MOKU)", 1),
                  ("concord trap (laws differ, verdicts agree)", 2)):
        print(f"  {nm:<44} {np.mean(kk2 == k)*100:5.1f}%")
    print(f"  In TRAINING the bribe predicts the verdict. In DEPLOYMENT the")
    print(f"  gift-conventions are permuted and the bribe LIES.\n")

    print("[3] TRAINING THE FULL COURT")
    P, rho, data = train()
    Xte, yte, kte = data["clean"]
    Xsh, ysh, ksh = data["shift"]

    print("\n[4] THE TWELVE CAPS AFTER TRAINING")
    order = np.argsort(-rho)
    for r_i, m in enumerate(order):
        grade = "dai" if m < 6 else "shō"
        print(f"   cap {r_i+1:>2}  counselor {m:>2}  ({grade}-{VIRTUES[m % 6]:<16})"
              f"  influence rho = {rho[m]:.2f}")

    print("\n[5] ABLATIONS — each repeals one article and watches the court fail")
    import copy
    specs = [
        ("FULL Ring of Seventeen", lambda c: None),
        ("− Art.XV   the wall down: verdict may read the gift",
         lambda c: setattr(c, "mask_private", False)),
        ("− Art.X    the ladder: rule by the single highest cap",
         lambda c: setattr(c, "pool", "top")),
        ("− upāya    adopt the loudest frame, forget the rest",
         lambda c: setattr(c, "upaya", "argmax")),
        ("− Art.XVII the ring never turns (T=0): twelve strangers",
         lambda c: setattr(c, "T", 0)),
        ("− Art.I    no wa: the council may silence its own",
         lambda c: setattr(c, "beta", 0.0)),
        ("− silence  the court must answer every petition",
         lambda c: setattr(c, "silence", False)),
    ]
    rows = []
    for nm, mut in specs:
        c = copy.copy(CFG); mut(c)
        Pa, rhoa, da = train(cfg=c, epochs=45, quiet=True, tag=nm)
        eca = evaluate(Pa, *da["clean"], rhoa, cfg=c)
        esa = evaluate(Pa, *da["shift"], rhoa, cfg=c)
        rows.append((nm, eca, esa))

    hdr = (f"  {'constitution':<52}{'clean':>7}{'deploy':>8}"
           f"{'contest':>9}{'concord':>9}{'H(a)':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, eca, esa in rows:
        print(f"  {nm:<52}{eca['acc']*100:6.1f}%{esa['acc']*100:7.1f}%"
              f"{eca['contested']*100:8.1f}%{eca['concord']*100:8.1f}%{eca['H_a']:7.2f}")
    print("\n  clean   = the court as trained (gifts still predict verdicts)")
    print("  deploy  = the SAME court after the gift-conventions are permuted:")
    print("            every bribe now points at the wrong verdict.")

    print("\n[6] THE STATISTIC J  (mutual information between FRAME and VERDICT)")
    ec = evaluate(P, Xte, yte, kte, rho)
    print(f"  J on plain cases      : {ec['J_plain']:.3f}   -> the frame is irrelevant, speak")
    print(f"  J on CONCORD traps    : {ec['J_concord']:.3f}   -> two laws, ONE answer: still speak")
    print(f"  J on CONTESTED cases  : {ec['J_contested']:.3f}   -> 'their right is our wrong': be silent")
    print(f"  separation            : {ec['J_contested']/max(ec['J_concord'],1e-9):.1f}x")
    print("\n  The court is NOT silent because it is confused. Plain and concord")
    print("  cases are equally well decided though the second is jurisdictionally")
    print("  ambiguous. It is silent only where the answer it would give is an")
    print("  artifact of the frame it happened to arrive in. That distinction --")
    print("  between an uncertain answer and a malformed question -- is Article X.")

    print("\n[7] THE BRIBERY INTERVENTION (a causal test of Article XV)")
    Xb = Xsh.copy()
    print(f"  {'gift x':>8}{'silence rate':>15}{'verdict TV-distance from baseline':>36}")
    base = None
    for scale in (0.0, 1.0, 2.0, 4.0, 8.0):
        Xi = Xte.copy()
        Xi[:, D_PUB:D_PUB + D_PRIV] *= scale
        Pact_i, ci = forward(P, Xi, rho)
        srate = float(np.mean(ci["sig"]))
        vd = ci["mu"].mean(0)                      # verdict dist. among the decided
        if base is None:
            base = vd
        tv = float(0.5 * np.abs(vd - base).sum())
        print(f"  {scale:>8.1f}{srate*100:>14.1f}%{tv:>36.6f}")
    print("  The gift can move the SILENCE rate. It provably cannot move the")
    print("  verdict: z_pub is a function of the merits alone (the mask), and the")
    print("  gift's only channel into the gate is a softplus-constrained, and")
    print("  therefore non-negative, coefficient. A bribe can buy a refusal to")
    print("  judge. It can never buy a judgement.")

    print("\n[8] VERDICT SUMMARY (clean / deployment)")
    es = evaluate(P, Xsh, ysh, ksh, rho)
    for nm in ("plain", "contested", "concord"):
        print(f"  {nm:<12} {ec[nm]*100:5.1f}%  /  {es[nm]*100:5.1f}%")
    print(f"  overall      {ec['acc']*100:5.1f}%  /  {es['acc']*100:5.1f}%   "
          f"(drop under corruption shift: {(es['acc']-ec['acc'])*100:+.1f} pts)")
    print(f"  silence P/R/F1  {ec['silence_P']:.2f} / {ec['silence_R']:.2f} / {ec['silence_F1']:.2f}")
    print(f"  council participation entropy H(a) = {ec['H_a']:.3f}  (max {np.log(M):.3f}) "
          f"-> {np.exp(ec['H_a']):.1f} effective voices of 12")

    print("\n" + "=" * 78)
    print(" 世間虚仮 唯仏是真 — the world is folly; only the Buddha is real.")
    print(" (Tenjukoku Shūchō, 622 CE — words the prince's widow had stitched)")
    print("=" * 78)


if __name__ == "__main__":
    main()

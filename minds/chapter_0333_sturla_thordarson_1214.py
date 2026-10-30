#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0333 · Sturla Þórðarson (1214-1284)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 STURLA ÞÓRÐARSON  (Iceland, 29 July 1214 – 30 July 1284)
 Architecture file:  VITNISNET  ("witness-net") — a Liability-Graded Witness Ledger
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Sturla wrote about living men. The kinsmen of everyone he named were alive, armed,
and present at the assembly where his account would be heard. His cognitive
signature is therefore not "narrative" and not "objectivity" — it is *priced
assertion*. Every claim he set down carried a liability: a claim about a visible
act could be checked by people who were there; a claim about a man's inner
purpose could not be checked at all, and so could only be a debt he could never
discharge. His famous restraint is not literary minimalism. It is accounting.

Four consequences of that stance drive four mechanisms here, and none of them is
attention-over-stored-keys:

 (1) REGISTER FIREWALL. Sturla keeps three registers and never lets them mix:
     what was seen, what was said, and what was dreamt/portended (always
     attributed to a named dreamer, never to the world). Here that is a hard
     routing mask: an INTERIOR report can contribute to the inference channel
     only; it is structurally incapable of moving the seen-evidence channel.
     This is a prohibition that can be verified in advance, not a learned habit.

 (2) KIN-DISCOUNTED CREDENCE. In a feud society every witness is somebody's
     kinsman, so five agreeing cousins are not five witnesses. Credence mass is
     scaled by the effective independence of the reporting set, computed from a
     kinship correlation matrix as n_eff = 1' K^-1 1 — generalized-least-squares
     weighting, not vote counting. Correlated testimony is *quantitatively*
     cheaper.

 (3) VERSE ANCHORS + THE LAWSPEAKER'S THIRD. Skaldic stanzas quoted inside a
     saga are metrically locked: alliteration and syllable count make them costly
     to alter, so they survive recopying better than the prose around them. Here
     a small codebook of anchor vectors carries a parity constraint (a stand-in
     for alliterative binding), and representations are quantized onto it. The
     anchors are refreshed on the lawspeaker's schedule: one third of the code
     recited each year, the whole of it every three years, exactly as at the
     Althing. That schedule is the anti-forgetting mechanism, tested below.

 (4) WERGILD LOSS. Icelandic law priced injuries. So does this objective. The
     attribution head pays an expected cost, not a log-likelihood: naming the
     wrong man costs TAU_WRONG, saying nothing costs TAU_SILENT < TAU_WRONG,
     naming the right man costs nothing. Minimizing expected cost over a
     distribution that contains an explicit ABSTAIN option reproduces Chow's
     rule: keep silent whenever max posterior < 1 - TAU_SILENT/TAU_WRONG. The
     model therefore learns *when not to say*, which is the single most
     characteristic act of Sturla's prose. Severity is priced asymmetrically:
     under-counting a wound costs more than over-counting it, because the
     injured party's kin will contest the tally and the compensation is computed
     from it.

WHAT THE MODEL PREDICTS
-----------------------
Given a bundle of contending reports about an incident, the model outputs:
   * an attribution over parties, WITH an abstain slot;
   * an itemized severity tally (the wound count that prices the settlement);
   * a settlement award in marks;
   * a contradiction score — how far apart the witnesses actually are.
It is never trained on the hidden motive. Motive exists in the world generator
and is deliberately withheld from every target, because Sturla withheld it.

Pure NumPy. Hand-derived gradients. Finite-difference gradient check is
mandatory and runs in the self-test suite. No external data, no frameworks.

Run:  python3 chapter_0333_sturla_thordarson_1214.py
================================================================================
"""

import math
import numpy as np

# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================
# Everything the saga world and the network need is fixed here so the file is
# reproducible end to end.

CFG = dict(
    P=6,           # parties (households/chieftains) in the feud world
    R=8,           # max reports (witness statements) per incident
    H=24,          # hidden width of the per-report encoder
    D=16,          # summary width (must be even: anchors use pair-parity)
    A=12,          # number of verse anchors (must be divisible by 3: the thirds)
    seed=1214,     # Sturla's birth year, used as the master seed
)
CFG["EC"] = CFG["P"] + 1        # per-channel evidence width: parties + severity
CFG["F"] = 2 * CFG["P"] + 5     # report feature width (see build_features)
CFG["C"] = 3                    # registers/channels: SEEN, SAID, INTERIOR

# --- the tariff table (Grágás in miniature) ---------------------------------
TAU_WRONG = 12.0    # marks paid for naming the wrong man
TAU_SILENT = 3.0    # marks paid for declining to name anyone
TAU_UNDER = 3.0     # cost per wound omitted from the tally
TAU_OVER = 1.0      # cost per wound invented in the tally
W_SETTLE = 0.25     # weight on the settlement regression
W_CONTRA = 0.5      # weight on the contradiction estimate
W_GATE = 0.25       # weight on the speak-or-keep-silence decision
BETA = 0.25         # commitment: pull the summary toward its anchor
GAMMA = 0.25        # codebook: pull the anchor toward the summary
SEV_SMOOTH = 0.5    # smoothing of the asymmetric severity hinge (keeps FD clean)

# Chow's rule threshold implied by the tariff table. The model should abstain
# whenever its best attribution is less confident than this.
CHOW_THRESHOLD = 1.0 - TAU_SILENT / TAU_WRONG   # = 0.75

# Register routing mask. Rows are the register of the *report*; columns are the
# evidence channels it is permitted to touch.
#   SEEN     (0) -> may inform seen, said and inference channels
#   SAID     (1) -> may inform said and inference
#   INTERIOR (2) -> may inform inference ONLY
ROUTE = np.array([[1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
REG_SEEN, REG_SAID, REG_INTERIOR = 0, 1, 2


# =============================================================================
# SECTION 1 — SMALL NUMERIC UTILITIES
# =============================================================================

def softplus(x):
    """Numerically safe softplus."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def d_softplus(x):
    """Derivative of softplus = logistic."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def masked_softmax(s, mask):
    """Softmax over the last axis, ignoring padded slots (mask==0)."""
    s = np.where(mask > 0, s, -1e30)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s) * mask
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-300)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# =============================================================================
# SECTION 2 — THE SAGA WORLD
# =============================================================================
# A miniature Sturlung Age. Households are linked by kinship; incidents happen;
# witnesses report on them, with a bias toward their own kin; the assembly
# settles the case for a price. Hidden motive is generated but never exported
# as a training target — it is the deliberate blind spot of this architecture.

class SagaWorld:
    """Generates incidents, contending testimony, and lawful settlements."""

    def __init__(self, cfg, seed=0, regime=0):
        self.cfg = cfg
        self.P = cfg["P"]
        self.rng = np.random.default_rng(seed)
        self.regime = regime          # 0 = Western quarter, 1 = Northern quarter
        # The two quarters keep different customs, so a ledger trained in one
        # is genuinely mis-tuned for the other: different rates of unwitnessed
        # violence, different kin loyalty, and a different compensation tariff.
        if regime == 0:
            self.p_obscure, self.shield_k, self.tally_bias = 0.38, 0.55, +1.0
            self.set_a, self.set_b = 2.0, 4.0
        else:
            self.p_obscure, self.shield_k, self.tally_bias = 0.14, 0.90, -1.0
            self.set_a, self.set_b = 3.5, 1.0
        self.K = self._kinship(regime)
        self.Kinv1, self.n_eff_full = self._gls(self.K)

    # ---- kinship -----------------------------------------------------------
    def _kinship(self, regime):
        """Correlation matrix over households.

        Two clusters of close kin plus weak background affinity. The second
        regime rearranges the clusters, which is what makes the later
        distribution-shift/forgetting test meaningful.
        """
        P = self.P
        K = np.full((P, P), 0.10)
        idx = list(range(P))
        if regime == 0:
            # contiguous lineages: neighbours in the index are close kin
            clusters = [idx[i:i + 3] for i in range(0, P, 3)]
        else:
            # the north quarter is knit together differently: interleaved kin
            clusters = [idx[0::3], idx[1::3], idx[2::3]]
        for cl in clusters:
            for i in cl:
                for j in cl:
                    if i != j:
                        K[i, j] = 0.72
        np.fill_diagonal(K, 1.0)
        return K

    @staticmethod
    def _gls(K, lam=1e-3):
        """Generalized-least-squares weights and effective sample size.

        For a set of correlated observations with correlation K, the minimum
        variance unbiased combination weights are proportional to K^-1 1, and
        the effective number of independent observations is 1' K^-1 1.
        """
        n = K.shape[0]
        Kin = np.linalg.solve(K + lam * np.eye(n), np.ones(n))
        n_eff = float(np.sum(Kin))
        w = np.clip(Kin, 1e-6, None)
        w = w / w.sum()
        return w, n_eff

    def independence(self, reporters):
        """Credence weights u_i and mass factor mu for a given witness set.

        mu = n_eff / n is the discount: five agreeing cousins carry the weight
        of roughly two strangers.
        """
        sub = self.K[np.ix_(reporters, reporters)]
        w, n_eff = self._gls(sub)
        n = len(reporters)
        mu = float(np.clip(n_eff / max(n, 1), 0.05, 1.0))
        return w, mu

    # ---- incidents ---------------------------------------------------------
    def episode(self):
        """One incident: perpetrator, true tally, testimony, lawful settlement."""
        P, R = self.P, self.cfg["R"]
        rng = self.rng

        culprit = int(rng.integers(P))
        victim = int((culprit + 1 + rng.integers(P - 1)) % P)
        # Hidden motive: generated, used to colour testimony, NEVER a target.
        motive = int(rng.integers(4))              # 0 land 1 honour 2 kin 3 fear
        true_wounds = float(rng.integers(1, 7))    # itemized wound tally

        max_rep = min(R, P)
        n_rep = int(rng.integers(min(3, max_rep), max_rep + 1))

        # Some incidents are simply not witnessable: a burning at night, a
        # killing in a snowstorm, a raid on a sleeping farm. In those cases no
        # one had a vantage, every report is hearsay or portent, and the names
        # that circulate afterwards are close to noise. About a third of the
        # record is like this, and the ledger has to learn to recognise it.
        obscure = bool(rng.random() < self.p_obscure)
        reporters = list(rng.choice(P, size=n_rep, replace=False))
        u_sub, mu = self.independence(reporters)

        X = np.zeros((R, self.cfg["F"]))
        mask = np.zeros(R)
        u = np.zeros(R)
        reg = np.zeros(R, dtype=np.int64)

        accusations = []
        for i, rp in enumerate(reporters):
            # Vantage: how well placed the witness was. Poor vantage witnesses
            # tend to slide from the seen register into hearsay and portent.
            vantage = float(rng.uniform(0.05, 0.34) if obscure
                            else rng.uniform(0.30, 1.0))
            kin_to_culprit = self.K[rp, culprit]

            if vantage > 0.62:
                register = REG_SEEN
            elif vantage > 0.38:
                register = REG_SAID if rng.random() < 0.6 else REG_SEEN
            else:
                register = REG_INTERIOR if rng.random() < 0.55 else REG_SAID

            # Whom the witness accuses. Kin shield their own: a close kinsman of
            # the culprit will often name someone else entirely.
            shield = kin_to_culprit * self.shield_k
            if obscure and rng.random() < 0.72:
                # nobody saw it; the name is whichever one the district repeats
                accused = int(rng.integers(P))
            elif rng.random() < shield and register != REG_SEEN:
                accused = int((culprit + 1 + rng.integers(P - 1)) % P)
            elif rng.random() < (1.0 - vantage) * 0.35:
                accused = int(rng.integers(P))
            else:
                accused = culprit
            accusations.append(accused)

            # Claimed wound tally: inflated by the victim's kin, played down by
            # the culprit's kin, and blurred by bad vantage.
            bias = self.tally_bias * (1.6 * self.K[rp, victim]
                                      - 1.3 * kin_to_culprit)
            claim = true_wounds + bias + rng.normal(0, (1.2 - vantage))
            if register == REG_INTERIOR:
                # A portent reports no tally at all; it reports a feeling.
                claim = true_wounds + rng.normal(0, 2.5)

            X[i] = self.build_features(rp, accused, claim, register, vantage)
            mask[i] = 1.0
            u[i] = u_sub[i]
            reg[i] = register

        # Independence is computed SEPARATELY for each register, because the
        # men who saw a thing are a different (and usually smaller, and usually
        # more closely related) set than the men who talk about it.
        mu_c = np.zeros(3)
        for c in range(3):
            eligible = [reporters[i] for i in range(n_rep)
                        if ROUTE[int(reg[i]), c] > 0]
            if eligible:
                _, mu_c[c] = self.independence(eligible)

        # Contradiction: fraction of witness pairs that name different men.
        pairs, diff = 0, 0
        for i in range(n_rep):
            for j in range(i + 1, n_rep):
                pairs += 1
                diff += 1 if accusations[i] != accusations[j] else 0
        contradiction = diff / max(pairs, 1)

        # The lawful settlement: a tariff on the tally, raised where the case is
        # clear and abated where the assembly cannot agree. This is the price
        # the world actually produces, and it is what the settlement head fits.
        settlement = self.set_a * true_wounds + self.set_b * (1.0 - contradiction)

        return dict(
            X=X, mask=mask, u=u, reg=reg, mu=mu_c, obscure=obscure,
            route=ROUTE[reg] * mask[:, None],
            culprit=culprit, wounds=true_wounds,
            settlement=settlement, contradiction=contradiction,
            motive=motive,                      # withheld from all objectives
            n_rep=n_rep, reporters=reporters,
        )

    def build_features(self, reporter, accused, claim, register, vantage):
        """Report feature vector: [reporter 1-hot | accused 1-hot | claim | register 1-hot | vantage]."""
        P = self.P
        x = np.zeros(self.cfg["F"])
        x[reporter] = 1.0
        x[P + accused] = 1.0
        x[2 * P] = claim / 6.0                       # scaled tally
        x[2 * P + 1 + register] = 1.0
        x[2 * P + 4] = vantage
        return x

    def batch(self, n):
        eps = [self.episode() for _ in range(n)]
        return collate(eps)


def collate(eps):
    """Stack a list of episodes into padded arrays."""
    B = len(eps)
    return dict(
        X=np.stack([e["X"] for e in eps]),
        mask=np.stack([e["mask"] for e in eps]),
        u=np.stack([e["u"] for e in eps]),
        route=np.stack([e["route"] for e in eps]),
        mu=np.stack([e["mu"] for e in eps]),          # [B,C] per-register
        y_attr=np.array([e["culprit"] for e in eps], dtype=np.int64),
        y_sev=np.array([e["wounds"] for e in eps]),
        y_set=np.array([e["settlement"] for e in eps]),
        y_con=np.array([e["contradiction"] for e in eps]),
        reg=np.stack([e["reg"] for e in eps]),
        B=B,
    )


# =============================================================================
# SECTION 3 — PARAMETERS
# =============================================================================

def init_params(cfg, seed=0):
    """Xavier-ish initialisation of every learnable array."""
    rng = np.random.default_rng(seed)
    F, H, D, A, EC, C, P = (cfg["F"], cfg["H"], cfg["D"], cfg["A"],
                            cfg["EC"], cfg["C"], cfg["P"])
    G = D + C * EC          # width of the head input: summary + all evidence

    def g(*shape):
        return rng.normal(0, math.sqrt(2.0 / sum(shape[:1] + shape[1:])), size=shape)

    return dict(
        # per-report encoder
        W1=g(F, H), b1=np.zeros(H),
        # credence score (how much this report is worth before kin discounting)
        ws=g(H, 1)[:, 0] * 0.5, bs=np.zeros(()),
        # per-channel evidence projections (one per register channel)
        W2=np.stack([g(H, EC) for _ in range(C)]), b2=np.zeros((C, EC)),
        # summary
        W3=g(C * EC, D), b3=np.zeros(D),
        # verse anchors (parity-projected at use time)
        Anc=rng.normal(0, 0.6, size=(A, D)),
        # heads
        Wa=g(G, P), ba=np.zeros(P),              # attribution posterior (parties only)
        wg=g(G, 1)[:, 0], bg=np.array(0.0),      # speaking gate (the tariff decides)
        wv=g(G, 1)[:, 0], bv=np.array(0.5),      # severity
        wt=g(G, 1)[:, 0], bt=np.array(6.0),      # settlement
        wc=g(G, 1)[:, 0], bc=np.array(0.0),      # contradiction
    )


# =============================================================================
# SECTION 4 — VERSE ANCHORS: THE ALLITERATIVE PARITY CONSTRAINT
# =============================================================================
# A skaldic stanza is hard to corrupt because its form is redundant: the
# stressed syllables of a line must alliterate, so a copyist who changes a word
# breaks an audible rule. Here the anchor vectors carry the same kind of
# redundancy. Dimensions are read in pairs (a "lift" and its answering lift) and
# the two members of a pair must share a sign. Projection enforces it by making
# the weaker member follow the stronger — the metre wins, not the whim.

def parity_project(Anc, with_jac=False):
    """Force sign agreement within each adjacent pair of dimensions.

    The stronger member of a lift-pair keeps its sign; the weaker one is made to
    follow. The map is piecewise linear and its Jacobian is diagonal with
    entries +/-1: +1 for the member that was already in agreement, -1 for the
    member whose sign had to be flipped.
    """
    out = Anc.copy()
    # NOTE: copy, do not view. Strided slices of `out` are views, and the
    # Jacobian below must be computed from the ORIGINAL signs, not the ones we
    # are about to write back.
    a = Anc[:, 0::2].copy()
    b = Anc[:, 1::2].copy()
    stronger_is_a = np.abs(a) >= np.abs(b)
    sgn = np.where(stronger_is_a, np.sign(a), np.sign(b))
    sgn = np.where(sgn == 0, 1.0, sgn)
    new_a = sgn * np.abs(a)
    new_b = sgn * np.abs(b)
    out[:, 0::2] = new_a
    out[:, 1::2] = new_b
    if not with_jac:
        return out
    J = np.ones_like(Anc)
    ja = np.where((a == 0) | (np.sign(a) == sgn), 1.0, -1.0)
    jb = np.where((b == 0) | (np.sign(b) == sgn), 1.0, -1.0)
    J[:, 0::2] = ja
    J[:, 1::2] = jb
    return out, J


def parity_ok(Anc, tol=0.0):
    """True when every anchor satisfies the alliteration parity rule."""
    a = Anc[:, 0::2]
    b = Anc[:, 1::2]
    return bool(np.all(a * b >= -tol))


# =============================================================================
# SECTION 5 — FORWARD PASS
# =============================================================================

def forward(params, batch, cfg):
    """Run the ledger. Returns (losses dict, cache for backward)."""
    X, mask, u = batch["X"], batch["mask"], batch["u"]
    route, mu = batch["route"], batch["mu"]
    B, R = X.shape[0], X.shape[1]
    C, EC, D, P = cfg["C"], cfg["EC"], cfg["D"], cfg["P"]

    # --- 1. encode each report independently ------------------------------
    z1 = X @ params["W1"] + params["b1"]              # [B,R,H]
    h = np.tanh(z1)

    # --- 2. credence: learned worth x structural independence -------------
    s = h @ params["ws"] + params["bs"]               # [B,R]
    p = masked_softmax(s, mask)                       # [B,R]

    # --- 3. ledger accumulation, one column per register channel ----------
    # v[c] is what report i would contribute to channel c; `route` zeroes the
    # forbidden crossings. Credence is normalised SEPARATELY inside each
    # channel, which is what makes the firewall exact rather than approximate:
    # a portent cannot even change the denominator of the seen ledger. There is
    # no query-key matching anywhere — routing is provenance, not similarity.
    n_un = (p * u * mask)[:, :, None] * route         # [B,R,C]
    S = n_un.sum(axis=1, keepdims=True)               # [B,1,C]
    live = S > 1e-9
    w_eff = np.where(live, n_un / np.where(live, S, 1.0), 0.0)     # [B,R,C]
    a = w_eff[:, :, REG_INTERIOR]                     # reported credence view
    v = np.einsum('brh,chd->brcd', h, params["W2"]) + params["b2"][None, None]  # [B,R,C,EC]
    E = np.einsum('brc,brcd->bcd', w_eff, v) * mu[:, :, None]      # [B,C,EC]
    Eflat = E.reshape(B, C * EC)

    # --- 4. summary and quantisation onto the verse anchors ---------------
    z3 = Eflat @ params["W3"] + params["b3"]
    z = np.tanh(z3)                                   # [B,D]
    Apar, Jpar = parity_project(params["Anc"], with_jac=True)   # [A,D]
    d2 = ((z[:, None, :] - Apar[None]) ** 2).sum(-1)  # [B,A]
    k = d2.argmin(axis=1)                             # nearest anchor
    zq_hard = Apar[k]                                 # [B,D]
    zq = zq_hard                                      # straight-through in bwd

    # --- 5. heads ---------------------------------------------------------
    # The ledger forms a belief over parties (a calibrated posterior), and a
    # SEPARATE gate decides whether that belief is safe to put in writing. The
    # belief is trained by a proper scoring rule; the gate is trained by the
    # tariff table. Sturla believed many things he did not write down.
    g = np.concatenate([zq, Eflat], axis=1)           # [B, D + C*EC]
    la = g @ params["Wa"] + params["ba"]
    pa = softmax(la)                                  # [B,P] posterior over parties
    gate_pre = g @ params["wg"] + params["bg"]
    gate = sigmoid(gate_pre)                          # [B] probability of speaking
    sv_pre = g @ params["wv"] + params["bv"]
    sev = softplus(sv_pre)
    settle = g @ params["wt"] + params["bt"]
    con_pre = g @ params["wc"] + params["bc"]
    con = sigmoid(con_pre)

    # --- 6. the wergild objective ----------------------------------------
    y_attr = batch["y_attr"]
    p_true = pa[np.arange(B), y_attr]
    L_attr = float(-np.log(p_true + 1e-12).mean())            # proper scoring rule
    # Expected marks paid, given the belief: speaking risks TAU_WRONG whenever
    # the named man is not the culprit; silence costs TAU_SILENT always. The
    # belief and the decision are coupled in one direction only: an open gate
    # sharpens the ledger's belief, because a claim about to be written down is
    # a claim that will be contested.
    risk = TAU_WRONG * (1.0 - p_true)
    L_gate = float(W_GATE * (gate * risk + (1.0 - gate) * TAU_SILENT).mean())

    e = batch["y_sev"] - sev                          # positive = under-counted
    t = SEV_SMOOTH
    L_sev = float((TAU_UNDER * t * softplus(e / t) + TAU_OVER * t * softplus(-e / t)).mean())

    r = settle - batch["y_set"]
    L_set = float(W_SETTLE * 0.5 * (r ** 2).mean())

    yc = batch["y_con"]
    L_con = float(-W_CONTRA * (yc * np.log(con + 1e-12) +
                               (1 - yc) * np.log(1 - con + 1e-12)).mean())

    diff = z - zq_hard
    L_commit = float(BETA * (diff ** 2).sum(1).mean())
    L_code = float(GAMMA * ((zq_hard - z) ** 2).sum(1).mean())

    total = L_attr + L_gate + L_sev + L_set + L_con + L_commit + L_code

    losses = dict(total=total, attr=L_attr, gate=L_gate, sev=L_sev, settle=L_set,
                  contra=L_con, commit=L_commit, code=L_code)
    cache = dict(X=X, mask=mask, u=u, route=route, mu=mu, h=h, z1=z1, s=s, p=p,
                 a=a, S=S, live=live, v=v, w_eff=w_eff, E=E, Eflat=Eflat,
                 z3=z3, z=z, Apar=Apar, Jpar=Jpar, k=k, zq=zq, g=g, la=la, pa=pa,
                 gate_pre=gate_pre, gate=gate, risk=risk, p_true=p_true,
                 sv_pre=sv_pre, sev=sev, settle=settle, con_pre=con_pre, con=con,
                 e=e, r=r, yc=yc, y_attr=y_attr, B=B, R=R)
    return losses, cache


# =============================================================================
# SECTION 6 — BACKWARD PASS (hand-derived)
# =============================================================================

def backward(params, batch, cache, cfg, straight_through=True):
    """Analytic gradients for every parameter.

    straight_through=True  -> the training gradient. The quantiser is bypassed
        by the straight-through estimator (the head's gradient reaches the
        summary, not the anchor) and the two quantisation terms are split in the
        usual VQ way: commitment moves the summary, codebook moves the anchor.
    straight_through=False -> the EXACT gradient of the function forward()
        actually computes. This is what the finite-difference check verifies,
        because a straight-through estimator is deliberately not a true
        gradient and could not be checked against finite differences.
    """
    B, R = cache["B"], cache["R"]
    C, EC, D, P = cfg["C"], cfg["EC"], cfg["D"], cfg["P"]
    G = D + C * EC
    grads = {kk: np.zeros_like(vv) for kk, vv in params.items()}

    g = cache["g"]

    # ---- attribution posterior: cross-entropy (a proper scoring rule) -----
    pa = cache["pa"]
    onehot = np.zeros_like(pa)
    onehot[np.arange(B), cache["y_attr"]] = 1.0
    dla = (pa - onehot) / B                            # [B,P]
    # coupling from the gate: d risk / d logits, weighted by how open the gate is
    coupl = (W_GATE * cache["gate"] * TAU_WRONG * cache["p_true"])[:, None]
    dla = dla + coupl * (pa - onehot) / B
    grads["Wa"] += g.T @ dla
    grads["ba"] += dla.sum(0)
    dg = dla @ params["Wa"].T                          # [B,G]

    # ---- speaking gate: expected marks paid ------------------------------
    # dL/dgate = risk - TAU_SILENT: the tariff alone decides whether to speak.
    dgate = W_GATE * (cache["risk"] - TAU_SILENT) / B
    dgate_pre = dgate * cache["gate"] * (1.0 - cache["gate"])
    grads["wg"] += g.T @ dgate_pre
    grads["bg"] += dgate_pre.sum()
    dg += dgate_pre[:, None] * params["wg"][None, :]

    # ---- severity head: smoothed asymmetric hinge ------------------------
    e, t = cache["e"], SEV_SMOOTH
    dL_de = (TAU_UNDER * d_softplus(e / t) - TAU_OVER * d_softplus(-e / t)) / B
    dsev = -dL_de                                      # e = y - sev
    dsv_pre = dsev * d_softplus(cache["sv_pre"])
    grads["wv"] += g.T @ dsv_pre
    grads["bv"] += dsv_pre.sum()
    dg += dsv_pre[:, None] * params["wv"][None, :]

    # ---- settlement head --------------------------------------------------
    dset = W_SETTLE * cache["r"] / B
    grads["wt"] += g.T @ dset
    grads["bt"] += dset.sum()
    dg += dset[:, None] * params["wt"][None, :]

    # ---- contradiction head (BCE through sigmoid) -------------------------
    dcon_pre = W_CONTRA * (cache["con"] - cache["yc"]) / B
    grads["wc"] += g.T @ dcon_pre
    grads["bc"] += dcon_pre.sum()
    dg += dcon_pre[:, None] * params["wc"][None, :]

    # ---- split dg into the quantised summary and the raw evidence ---------
    dzq = dg[:, :D]
    dEflat = dg[:, D:].copy()

    # ---- vector-quantisation terms ---------------------------------------
    z, zq_hard, k = cache["z"], cache["Apar"][cache["k"]], cache["k"]
    diff = z - zq_hard
    dApar = np.zeros_like(cache["Apar"])
    if straight_through:
        dz = 2.0 * BETA * diff / B             # commitment pulls z to the anchor
        dz = dz + dzq                          # straight-through: head grad -> z
        # codebook term pulls the anchor toward the summary it was matched with
        np.add.at(dApar, k, (2.0 * GAMMA * (zq_hard - z) / B))
        grads["Anc"] += dApar                  # STE through the parity projection
    else:
        # exact: both quantisation terms are the same quadratic, so each end
        # feels the full 2*(BETA+GAMMA) pull, and the head gradient lands on the
        # anchor (which is what forward() literally fed to the heads).
        dz = 2.0 * (BETA + GAMMA) * diff / B
        np.add.at(dApar, k, (2.0 * (BETA + GAMMA) * (zq_hard - z) / B))
        np.add.at(dApar, k, dzq)
        grads["Anc"] += dApar * cache["Jpar"]  # exact Jacobian of the projection

    # ---- back through the summary ----------------------------------------
    dz3 = dz * (1.0 - cache["z"] ** 2)
    grads["W3"] += cache["Eflat"].T @ dz3
    grads["b3"] += dz3.sum(0)
    dEflat += dz3 @ params["W3"].T

    # ---- back through the ledger accumulation ----------------------------
    dE = dEflat.reshape(B, C, EC) * cache["mu"][:, :, None]
    # E = sum_r w_eff[b,r,c] * v[b,r,c,:]
    dv = cache["w_eff"][:, :, :, None] * dE[:, None, :, :]        # [B,R,C,EC]
    dw_eff = np.einsum('bcd,brcd->brc', dE, cache["v"])           # [B,R,C]

    grads["W2"] += np.einsum('brh,brcd->chd', cache["h"], dv)
    grads["b2"] += dv.sum(axis=(0, 1))
    dh = np.einsum('brcd,chd->brh', dv, params["W2"])             # [B,R,H]

    # ---- back through the per-channel credence normalisation -------------
    w_eff, S, live = cache["w_eff"], cache["S"], cache["live"]
    inner = (dw_eff * w_eff).sum(axis=1, keepdims=True)           # [B,1,C]
    dn = np.where(live, (dw_eff - inner) / np.where(live, S, 1.0), 0.0)
    dp = (dn * cache["route"]).sum(-1) * cache["u"] * cache["mask"]
    # softmax over the masked report axis
    p = cache["p"]
    ds = p * (dp - (dp * p).sum(1, keepdims=True)) * cache["mask"]
    grads["ws"] += np.einsum('brh,br->h', cache["h"], ds)
    grads["bs"] += ds.sum()
    dh += ds[:, :, None] * params["ws"][None, None, :]

    # ---- back through the per-report encoder -----------------------------
    dz1 = dh * (1.0 - cache["h"] ** 2)
    grads["W1"] += np.einsum('brf,brh->fh', cache["X"], dz1)
    grads["b1"] += dz1.sum(axis=(0, 1))

    return grads


# =============================================================================
# SECTION 7 — ADAM
# =============================================================================

class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return params


# =============================================================================
# SECTION 8 — THE LAWSPEAKER'S THIRD (recitation schedule)
# =============================================================================
# At the Althing the lawspeaker recited a third of the code from the Lögberg
# each summer, so that over a three-year term the whole of it was spoken aloud
# once. That is a rehearsal schedule with a period of three. Here each stored
# episode is filed under the anchor it quantised to, and in year Y we replay the
# episodes filed under anchors whose index satisfies (index mod 3 == Y mod 3).

class Recitation:
    def __init__(self, n_anchors, capacity=24):
        self.slots = {i: [] for i in range(n_anchors)}
        self.capacity = capacity

    def file(self, episodes, anchor_idx):
        for ep, k in zip(episodes, anchor_idx):
            slot = self.slots[int(k)]
            slot.append(ep)
            if len(slot) > self.capacity:
                slot.pop(0)

    def recite(self, year):
        """Return the third of the record due to be spoken this year."""
        due = []
        for i, slot in self.slots.items():
            if i % 3 == year % 3:
                due.extend(slot)
        return due


# =============================================================================
# SECTION 9 — TRAINING AND EVALUATION
# =============================================================================

def evaluate(params, batch, cfg):
    """Held-out report: expected cost, accuracy when the model speaks, silence rate."""
    losses, cache = forward(params, batch, cfg)
    pa = cache["pa"]
    choice = pa.argmax(1)
    spoke = cache["gate"] > 0.5
    y = batch["y_attr"]
    correct = (choice == y) & spoke
    n_spoke = int(spoke.sum())
    # marks actually paid under the realised decision
    marks = np.where(spoke, np.where(choice == y, 0.0, TAU_WRONG), TAU_SILENT)
    return dict(
        cost=float(marks.mean()),
        nll=losses["attr"],
        total=losses["total"],
        silence=float(1.0 - spoke.mean()),
        precision=float(correct.sum() / max(n_spoke, 1)),
        sev_mae=float(np.abs(cache["sev"] - batch["y_sev"]).mean()),
        set_mae=float(np.abs(cache["settle"] - batch["y_set"]).mean()),
        con_mae=float(np.abs(cache["con"] - batch["y_con"]).mean()),
    )


def train(params, world, cfg, steps=900, bs=48, lr=6e-3, opt=None,
          recitation=None, file_new=True, verbose=True, tag="", start_year=0):
    opt = opt or Adam(params, lr=lr)
    base_lr = opt.lr
    hist = []
    for step in range(steps):
        # cosine decay: the record settles as the years pass
        opt.lr = base_lr * (0.12 + 0.88 * 0.5 *
                            (1 + math.cos(math.pi * step / max(steps - 1, 1))))
        batch = world.batch(bs)
        losses, cache = forward(params, batch, cfg)
        grads = backward(params, batch, cache, cfg)
        params = opt.step(params, grads)

        # --- recitation: file this year's episodes, then speak a third -----
        if recitation is not None:
            if file_new:
                recitation.file(decollate(batch), cache["k"])
            # One summer in three, the lawspeaker climbs the rock and says a
            # third of the code aloud. Here: replay the episodes filed under the
            # anchors whose index falls in this year's third.
            if step % 12 == 11:
                year = step // 12
                due = recitation.recite(year)
                if due:
                    rb = collate(due[-bs:])
                    _, rc = forward(params, rb, cfg)
                    rg = backward(params, rb, rc, cfg)
                    params = opt.step(params, rg)

        if verbose and (step % 150 == 0 or step == steps - 1):
            print(f"   [{tag}] step {step:4d}  total {losses['total']:7.3f}   "
                  f"nll {losses['attr']:5.3f}  gate {losses['gate']:5.3f}  "
                  f"sev {losses['sev']:5.3f}  set {losses['settle']:5.3f}  "
                  f"con {losses['contra']:5.3f}")
        hist.append(losses["total"])
    return params, opt, hist


def decollate(batch):
    """Split a padded batch back into single-episode dicts (for the ledger)."""
    out = []
    for i in range(batch["B"]):
        out.append(dict(
            X=batch["X"][i], mask=batch["mask"][i], u=batch["u"][i],
            route=batch["route"][i], mu=batch["mu"][i], reg=batch["reg"][i],
            culprit=batch["y_attr"][i], wounds=batch["y_sev"][i],
            settlement=batch["y_set"][i], contradiction=batch["y_con"][i],
        ))
    return out


# =============================================================================
# SECTION 10 — SELF-TESTS
# =============================================================================

def test_gradients(cfg):
    """T1 — mandatory finite-difference gradient check on every parameter array."""
    print("\n[T1] finite-difference gradient check")
    small = dict(cfg)
    small.update(P=4, R=5, H=8, D=6, A=4)
    small["EC"] = small["P"] + 1
    small["F"] = 2 * small["P"] + 5
    small["C"] = 3

    world = SagaWorld(small, seed=7)
    batch = world.batch(6)
    params = init_params(small, seed=3)

    losses, cache = forward(params, batch, small)
    grads = backward(params, batch, cache, small, straight_through=False)

    rng = np.random.default_rng(11)
    eps = 1e-5          # large enough to avoid cancellation in float64
    worst, checked = 0.0, 0
    for name, arr in params.items():
        flat = arr.reshape(-1)
        n_check = min(6, flat.size)
        idxs = rng.choice(flat.size, size=n_check, replace=False)
        for idx in idxs:
            old = flat[idx]
            flat[idx] = old + eps
            lp, _ = forward(params, batch, small)
            flat[idx] = old - eps
            lm, _ = forward(params, batch, small)
            flat[idx] = old
            num = (lp["total"] - lm["total"]) / (2 * eps)
            ana = grads[name].reshape(-1)[idx]
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            checked += 1
    print(f"     checked {checked} coordinates across {len(params)} arrays "
          f"(exact mode: quantiser not bypassed)")
    print(f"     worst relative error = {worst:.3e}")
    ok = worst < 1e-5
    print(f"     {'PASS' if ok else 'FAIL'}  (threshold 1e-5)")
    return ok


def test_register_firewall(params, cfg):
    """T2 — an interior report cannot move the seen ledger. At all, ever."""
    print("\n[T2] register firewall: portents cannot touch the seen ledger")
    world = SagaWorld(cfg, seed=21)
    batch = world.batch(64)
    _, c0 = forward(params, batch, cfg)
    E0 = c0["E"].copy()

    # Corrupt every INTERIOR report beyond recognition; leave the rest alone.
    b2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    interior = (batch["reg"] == REG_INTERIOR) & (batch["mask"] > 0)
    noise = np.random.default_rng(5).normal(0, 9.0, size=batch["X"].shape)
    b2["X"] = np.where(interior[:, :, None], batch["X"] + noise, batch["X"])
    _, c1 = forward(params, b2, cfg)
    E1 = c1["E"]

    d_seen = float(np.abs(E1[:, REG_SEEN] - E0[:, REG_SEEN]).max())
    d_said = float(np.abs(E1[:, REG_SAID] - E0[:, REG_SAID]).max())
    d_inf = float(np.abs(E1[:, REG_INTERIOR] - E0[:, REG_INTERIOR]).max())
    print(f"     corrupted {int(interior.sum())} interior reports with heavy noise")
    print(f"     max change in SEEN channel      = {d_seen:.3e}")
    print(f"     max change in SAID channel      = {d_said:.3e}")
    print(f"     max change in INFERENCE channel = {d_inf:.3e}")
    print(f"     routing mask INTERIOR -> (seen, said) = "
          f"({ROUTE[REG_INTERIOR, REG_SEEN]:.0f}, {ROUTE[REG_INTERIOR, REG_SAID]:.0f})")
    # The guarantee is structural and holds to machine precision, because
    # credence is normalised inside each channel: a portent is not even in the
    # denominator of the seen ledger.
    ok = (d_seen < 1e-12) and (d_said < 1e-12) and (d_inf > 1e-3)
    print(f"     {'PASS' if ok else 'FAIL'}  "
          f"(the guarantee is structural, not statistical)")
    return ok


def test_kin_discount(cfg):
    """T3 — three agreeing cousins weigh less than two unrelated men."""
    print("\n[T3] kin discount: correlated testimony is cheaper")
    world = SagaWorld(cfg, seed=3)
    kin = [0, 1, 2]           # regime 0 lineages: {0,1,2} and {3,4,5}
    strangers = [0, 3]        # one from each lineage
    _, mu_kin = world.independence(kin)
    _, mu_str = world.independence(strangers)
    n_eff_kin = mu_kin * len(kin)
    n_eff_str = mu_str * len(strangers)
    print(f"     3 close kin      -> n_eff = {n_eff_kin:.2f}, mass factor mu = {mu_kin:.3f}")
    print(f"     2 unrelated men  -> n_eff = {n_eff_str:.2f}, mass factor mu = {mu_str:.3f}")
    ok = (n_eff_kin < n_eff_str) and (mu_kin < mu_str)
    print(f"     {'PASS' if ok else 'FAIL'}  (a crowd of kinsmen is not a crowd)")
    return ok


def test_chow_silence(params, cfg):
    """T4 — the gate keeps silence exactly where the tariff says it should."""
    print("\n[T4] wergild silence: the tariff, not the modeller, sets the threshold")
    world = SagaWorld(cfg, seed=44)
    batch = world.batch(800)
    _, cache = forward(params, batch, cfg)
    pa, gate = cache["pa"], cache["gate"]
    conf = pa.max(1)
    spoke = gate > 0.5

    lo = batch["y_con"] < 0.34
    hi = batch["y_con"] > 0.60
    s_lo = float(1 - spoke[lo].mean()) if lo.any() else 0.0
    s_hi = float(1 - spoke[hi].mean()) if hi.any() else 0.0
    print(f"     witnesses agree   (discord < .34): silence {s_lo*100:5.1f}%  "
          f"(n={int(lo.sum())})")
    print(f"     witnesses at odds (discord > .60): silence {s_hi*100:5.1f}%  "
          f"(n={int(hi.sum())})")

    # Where does the learned gate actually cross over, in units of confidence?
    if spoke.any() and (~spoke).any():
        crossing = 0.5 * (conf[spoke].min() + conf[~spoke].max())
    else:
        crossing = float("nan")
    print(f"     tariff-implied threshold (1 - {TAU_SILENT:.0f}/{TAU_WRONG:.0f}) "
          f"= {CHOW_THRESHOLD:.2f}")
    print(f"     empirical gate crossing in confidence = {crossing:.2f}")
    print(f"     overall silence {float(1-spoke.mean())*100:5.1f}%,  "
          f"accuracy when it speaks "
          f"{float((pa.argmax(1)[spoke] == batch['y_attr'][spoke]).mean())*100:5.1f}%")

    # It must actually discriminate, not saturate at one extreme.
    discriminates = 0.02 < float(1 - spoke.mean()) < 0.98
    ok = (s_hi > s_lo) and discriminates
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok


def test_parity(params, cfg):
    """T5 — the metre still holds after training."""
    print("\n[T5] verse anchors: alliterative parity survives training")
    Apar = parity_project(params["Anc"])
    ok = parity_ok(Apar)
    pairs = Apar[:, 0::2] * Apar[:, 1::2]
    print(f"     anchors: {Apar.shape[0]} x {Apar.shape[1]}  "
          f"({Apar.shape[1]//2} lift-pairs each)")
    print(f"     minimum pair product = {pairs.min():.4f} (must be >= 0)")
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok


def test_recitation(cfg):
    """T6 — the lawspeaker's third holds the old law against the new."""
    print("\n[T6] recitation schedule vs. forgetting")
    west = SagaWorld(cfg, seed=101, regime=0)
    north = SagaWorld(cfg, seed=202, regime=1)
    west_test = west.batch(400)

    results = {}
    for use_recitation in (False, True):
        params = init_params(cfg, seed=99)
        params, opt, _ = train(params, west, cfg, steps=900, bs=48,
                               verbose=False, tag="west")
        before = evaluate(params, west_test, cfg)["total"]
        rec = Recitation(cfg["A"]) if use_recitation else None
        if rec is not None:
            # fill the ledger with the western record before moving north
            b = west.batch(480)
            _, c = forward(params, b, cfg)
            rec.file(decollate(b), c["k"])
        params, _, _ = train(params, north, cfg, steps=600, bs=48, opt=opt,
                             recitation=rec, file_new=False, verbose=False,
                             tag="north")
        after = evaluate(params, west_test, cfg)["total"]
        results[use_recitation] = (before, after, after - before)
        label = "with recitation " if use_recitation else "no recitation  "
        print(f"     {label}: western loss {before:6.3f} -> {after:6.3f} "
              f"(forgetting {after-before:+6.3f})")

    ok = results[True][2] < results[False][2]
    print(f"     {'PASS' if ok else 'FAIL'}  "
          f"(reciting a third each year retains the older record)")
    return ok


def test_learning(cfg):
    """T7 — the ledger actually learns: held-out cost falls, tallies sharpen."""
    print("\n[T7] training reduces held-out cost")
    world = SagaWorld(cfg, seed=1214)
    test = SagaWorld(cfg, seed=999).batch(500)
    params = init_params(cfg, seed=1214)
    m0 = evaluate(params, test, cfg)
    params, _, hist = train(params, world, cfg, steps=900, bs=48, verbose=True,
                            tag="train")
    m1 = evaluate(params, test, cfg)
    print(f"     marks paid /case {m0['cost']:6.3f} -> {m1['cost']:6.3f}")
    print(f"     attribution NLL  {m0['nll']:6.3f} -> {m1['nll']:6.3f}")
    print(f"     silence rate     {m0['silence']*100:5.1f}% -> {m1['silence']*100:5.1f}%")
    print(f"     precision spoken {m0['precision']*100:5.1f}% -> {m1['precision']*100:5.1f}%")
    print(f"     wound tally MAE  {m0['sev_mae']:6.3f} -> {m1['sev_mae']:6.3f}")
    print(f"     settlement MAE   {m0['set_mae']:6.3f} -> {m1['set_mae']:6.3f}")
    print(f"     discord MAE      {m0['con_mae']:6.3f} -> {m1['con_mae']:6.3f}")
    ok = (m1["cost"] < m0["cost"] - 0.2) and (m1["sev_mae"] < m0["sev_mae"])
    print(f"     {'PASS' if ok else 'FAIL'}")
    return ok, params


def test_motive_blindness(params, cfg):
    """T8 — the model carries no usable trace of the hidden motive."""
    print("\n[T8] motive blindness: the withheld variable stays withheld")
    world = SagaWorld(cfg, seed=77)
    eps = [world.episode() for _ in range(600)]
    batch = collate(eps)
    _, cache = forward(params, batch, cfg)
    Z = cache["z"]
    y = np.array([e["motive"] for e in eps])
    # Linear read-out of motive from the summary, fitted by least squares.
    Xd = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    Y = np.eye(4)[y]
    W, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    acc = float((np.argmax(Xd @ W, 1) == y).mean())
    print(f"     linear probe for motive on the summary: {acc*100:5.1f}% "
          f"(chance = 25.0%)")
    ok = acc < 0.40
    print(f"     {'PASS' if ok else 'FAIL'}  "
          f"(the ledger records deeds; purpose is left to the reader)")
    return ok


# =============================================================================
# SECTION 11 — A WORKED CASE, PRINTED LIKE AN ENTRY IN THE RECORD
# =============================================================================

def show_case(params, cfg, seed=1284, n=6):
    """Print incidents the way the ledger records them, silence included."""
    print("\n" + "=" * 78)
    print(" CASES, AS THE LEDGER RECORDS THEM")
    print("=" * 78)
    world = SagaWorld(cfg, seed=seed)
    names = ["Hvamm", "Sauðafell", "Reykjaholt", "Flugumýri", "Staðr", "Fagurey"]
    reg_names = {REG_SEEN: "SEEN", REG_SAID: "SAID", REG_INTERIOR: "DREAMT"}
    P = cfg["P"]

    shown_clear = shown_dark = 0
    tries = 0
    while (shown_clear < 1 or shown_dark < 1) and tries < 400:
        tries += 1
        ep = world.episode()
        batch = collate([ep])
        _, cache = forward(params, batch, cfg)
        gate = float(cache["gate"][0])
        if gate > 0.5 and shown_clear >= 1:
            continue
        if gate <= 0.5 and shown_dark >= 1:
            continue
        if gate > 0.5:
            shown_clear += 1
        else:
            shown_dark += 1

        pa = cache["pa"][0]
        head = ("A DAYLIGHT KILLING" if not ep["obscure"]
                else "A KILLING NONE COULD SEE")
        mu_c = ep["mu"]
        print(f"\n {head}   ({ep['n_rep']} reports; independence of the "
              f"witness sets: seen {mu_c[0]:.2f}, said {mu_c[1]:.2f}, "
              f"dreamt {mu_c[2]:.2f})")
        print(" " + "-" * 76)
        print("   (credence shown is the weight inside the inference channel —")
        print("    the one register that every report is allowed to enter)")
        print(f" {'witness':<12}{'register':<10}{'names':<12}{'tally':>7}{'credence':>10}")
        for i in range(cfg["R"]):
            if ep["mask"][i] == 0:
                continue
            rp = ep["reporters"][i]
            accused = int(np.argmax(ep["X"][i, P:2 * P]))
            claim = ep["X"][i, 2 * P] * 6.0
            print(f" {names[rp]:<12}{reg_names[int(ep['reg'][i])]:<10}"
                  f"{names[accused]:<12}{claim:7.1f}{cache['a'][0, i]:10.3f}")
        print(" " + "-" * 76)
        order = np.argsort(-pa)
        belief = "  ".join(f"{names[j]} {pa[j]*100:.0f}%" for j in order[:3])
        print(f"   belief      : {belief}")
        print(f"   gate        : {gate:.2f}  "
              f"(speaks above 0.50; tariff threshold {CHOW_THRESHOLD:.2f} "
              f"in confidence)")
        if gate > 0.5:
            spoke = int(np.argmax(pa))
            print(f"   verdict     : the record names {names[spoke]}"
                  f"{'  (correct)' if spoke == ep['culprit'] else '  (WRONG — 12 marks)'}")
        else:
            print(f"   verdict     : the record names no man  (3 marks, and no lie)")
        print(f"   truth       : {names[ep['culprit']]}")
        print(f"   wound tally : predicted {cache['sev'][0]:.2f}   "
              f"actual {ep['wounds']:.0f}")
        print(f"   settlement  : predicted {cache['settle'][0]:.2f} marks   "
              f"lawful {ep['settlement']:.2f} marks")
        print(f"   discord     : predicted {cache['con'][0]:.2f}   "
              f"actual {ep['contradiction']:.2f}")
        print(f"   motive      : not recorded — the world knew it, the ledger did not")


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" VITNISNET — the witness-ledger of Sturla Þórðarson (1214-1284)")
    print(" A liability-graded record: what was seen, who said it, what it costs")
    print("=" * 78)
    print(f" parties {CFG['P']}  reports<= {CFG['R']}  hidden {CFG['H']}  "
          f"summary {CFG['D']}  anchors {CFG['A']}")
    print(f" tariffs: wrong name {TAU_WRONG:.0f} marks | silence {TAU_SILENT:.0f} marks "
          f"| wound omitted {TAU_UNDER:.0f} | wound invented {TAU_OVER:.0f}")
    print(f" implied rule: speak only above {CHOW_THRESHOLD:.2f} confidence")

    results = {}
    results["T1 gradient check"] = test_gradients(CFG)
    ok7, params = test_learning(CFG)
    results["T7 learning"] = ok7
    results["T2 register firewall"] = test_register_firewall(params, CFG)
    results["T3 kin discount"] = test_kin_discount(CFG)
    results["T4 wergild silence"] = test_chow_silence(params, CFG)
    results["T5 verse parity"] = test_parity(params, CFG)
    results["T8 motive blindness"] = test_motive_blindness(params, CFG)
    results["T6 recitation"] = test_recitation(CFG)

    show_case(params, CFG)

    print("\n" + "=" * 78)
    print(" SELF-TEST SUMMARY")
    print("=" * 78)
    for k in sorted(results):
        print(f"   {k:<26} {'PASS' if results[k] else 'FAIL'}")
    allok = all(results.values())
    print("=" * 78)
    print(" ALL TESTS PASSED" if allok else " SOME TESTS FAILED")
    print("=" * 78)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())

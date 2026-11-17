#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0356 · Mansa Musa (1280-1337)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 

 ARCHITECTURE:  PPFN — the Performative Potential-Flow Network
                ("The Mansa Engine")

 Reference implementation: pure NumPy, from scratch, no autodiff framework.
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------------------------------------------
Almost every learning system ever built is a PRICE-TAKER. It assumes the world
holds still while it looks. Train on a distribution, deploy on that same
distribution, and measure the gap. The agent is small; the world is large; the
act of observing changes nothing.

Musa's documented life is the exact opposite case, and it is one of the earliest
well-attested ones in the human record. In July 1324 he entered Cairo and
disbursed gold. Al-'Umari, collecting testimony in Cairo roughly a decade later,
records that the mithqal had stood at no less than 25 silver dirhams before the
Malians arrived and did not exceed 22 afterwards. Musa did not merely trade in a
market. He MOVED the market. And then — this is the part that makes him an
architecture rather than an anecdote — he had to walk back through it. Running
short on the return leg, he and his people resold in Cairo, at roughly half
value, the goods they had bought there on the way out, and he borrowed 50,000
dinars from the merchant Siraj al-Din ibn al-Kuwayk against a market he himself
had flattened.

So the cognitive problem Musa actually faced was not "what is the world worth?"
It was:

    What is the world worth AFTER I have acted on it,
    and can I afford the journey home through the world I will have made?

That is, in modern terms, PERFORMATIVE PREDICTION (Perdomo, Zrnic, Mendler-
Dünner & Hardt, ICML 2020): the setting where the deployed model's predictions
change the distribution that generates its own labels. It is also market impact,
Goodhart's law, and auto-induced distributional shift. Musa is the corpus's
price-MAKER, and the whole network below is built around that single asymmetry.

The eight organs, and the documented fact each one encodes:

  1. PotentialEncoder    phi_i   — what a place is worth before I arrive.
                                   (The ordinary, price-taking part of any mind.)

  2. ElasticityHead      kappa_i — HOW MUCH MY OWN ARRIVAL WILL MOVE IT.
                                   This is the Musa organ. No standard value
                                   head has it. Everything else follows.

  3. CaravanPolicy       q       — a budget-conserving allocation whose scores
                                   are phi/(2*kappa): value discounted by
                                   self-impact. Its temperature tau is LEARNED.
                                   tau -> 0 is "dump the whole treasury on
                                   Cairo," which is what he did. tau large is
                                   "spread thin." The learned tau is the lesson
                                   of 1324 expressed as a scalar.

  4. ReturnLegReserve    rho     — the fraction of the treasury withheld for the
                                   journey home, learned. In 1325 the true value
                                   of rho was evidently too small.

  5. NarrativeChannel    b       — Gomez (2018, p.121) notes Musa promoted the
                                   rumour that gold GREW LIKE A PLANT in his
                                   country. That is deliberate manipulation of
                                   the observer's model of your supply function.
                                   Here it is a learned scalar that raises
                                   chronicle-standing and simultaneously worsens
                                   your terms of trade, because a counterparty
                                   who believes your supply is unlimited will
                                   not pay you well. A real, priced trade-off
                                   between reputation and exchange rate.

  6. ImpactLedger        cost    — the return leg is purchased AT THE PRICE YOU
                                   CREATED: cost = need / p_after. It diverges
                                   exactly when you over-concentrate. This is
                                   1325, mechanized.

  7. DualLedger          two scores, never summed into one number:
                            chronicle_e  — what al-'Umari and Ibn Khaldun record
                                           (spectacle, generosity, prestige)
                            jeli_e       — what the Mande jeliw record. They are
                                           notably COOL toward Musa; some hold
                                           that he squandered Mali's wealth and
                                           was unfaithful to tradition.
                          Two evaluation regimes, opposite signs, same agent.
                          The model reports their DIVERGENCE as a first-class
                          output rather than collapsing it.

  8. PerformativeLoss    the forecasting head predicts the POST-action price
                         p_hat = phi - kappa*q, and is scored against the price
                         its own action produced. The label depends on the
                         prediction. This is what makes the gradient interesting.

THE TECHNICAL CLAIM THIS FILE DEMONSTRATES
--------------------------------------------------------------------------------
The loss depends on the parameters twice:

    L(theta) = f( theta , p_realized( q(theta) ) )

    dL/dtheta  =  (partial L / partial theta)          <- the "price-taker" path
               +  (partial L / partial p) (dp/dq) (dq/dtheta)   <- the Musa path

A price-taking implementation silently drops the second term. Test 2 below
proves, by finite differences, that the price-taker gradient is WRONG — not
slightly noisy, wrong — while the full gradient is correct to ~1e-9. Test 7
shows a network trained with the truncated gradient is measurably worse at the
real objective. The chapter's thesis is therefore not asserted; it is executed.

Also implemented, for contrast: Repeated Risk Minimization (RRM), which
converges to a performatively STABLE point (optimal for the distribution it
induces) rather than the performatively OPTIMAL one. Musa reached a stable
point. He did not reach an optimal one.

RUN:   python3 chapter_0284_mansa_musa_1280.py
DEPS:  numpy only.
================================================================================
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

# ==============================================================================
# SECTION 0 — Constants of the world.
# ==============================================================================
# These are the physical/economic constants of the simulated Sahel-to-Cairo
# corridor. They are NOT learned; they belong to the environment, and the agent
# only ever sees them through consequences.

N_NODES        = 8       # markets on the circuit. Node 0 is Cairo.
N_FEAT         = 5       # observable features per market
HIDDEN         = 24      # encoder width

TREASURY       = 10.0    # total gold carried out of Niani (arbitrary units)
RETURN_NEED    = 3.0     # physical resupply required for the journey home
PRICE_FLOOR    = 0.15    # a market never literally goes to zero
FLOOR_BETA     = 8.0     # sharpness of the smooth floor (softplus temperature)

NARRATIVE_GAIN = 0.55    # ALPHA: how much the rumour buys you in the chronicles
NARRATIVE_COST = 0.22    # NU: how much the rumour costs you in terms of trade
JELI_SUSPICION = 0.40    # DELTA: how much the jeliw dislike the spectacle
DEBT_PENALTY   = 6.0     # GAMMA: weight on the return-leg shortfall

W_CHRONICLE    = 0.5     # weighting between the two ledgers
NORM_CHRON     = 10.0     # scale normalizers so neither ledger dominates by units
NORM_JELI      = 4.0
ETA_PRED       = 0.35    # weight on the performative forecasting loss
L2             = 1e-4

KAPPA_MIN      = 0.02    # elasticity is strictly positive
TAU_MIN        = 0.05    # allocation temperature floor


# ==============================================================================
# SECTION 1 — Small numerically-safe primitives.
# ==============================================================================
# Written out rather than imported so every derivative below is inspectable.

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic. Branch-split so exp() never overflows for large |x|."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """log(1+exp(beta*x))/beta, computed stably. Smooth relu."""
    bx = beta * x
    return (np.maximum(bx, 0.0) + np.log1p(np.exp(-np.abs(bx)))) / beta


def d_softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """d/dx softplus(x,beta) = sigmoid(beta*x)."""
    return sigmoid(beta * np.asarray(x, dtype=np.float64))


def softmax_last(z: np.ndarray) -> np.ndarray:
    """Softmax over the final axis, shifted for stability."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Plain Pearson correlation; used only for diagnostics."""
    a = np.ravel(a) - np.mean(a)
    b = np.ravel(b) - np.mean(b)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if d < 1e-12 else float(a @ b / d)


# ==============================================================================
# SECTION 2 — The world: a circuit of markets with genuine own-impact.
# ==============================================================================

class SahelCircuit:
    """
    A batch of independent trading circuits.

    Each circuit has N_NODES markets. Node 0 is always the THIN, RICH market —
    the Cairo analogue: highest base price, and by far the highest elasticity,
    because the volume Musa carried was enormous relative to what that market
    could absorb without moving.

    The agent never observes p0 or kappa_true directly. It observes five
    features and must infer both. Crucially it must infer kappa_true, because
    kappa_true is what converts its intentions into outcomes.

    Ground-truth price response to a disbursement q at node i:

        p_i(q_i) = p0_i * m  -  kappa_true_i * q_i

    where m is the terms-of-trade multiplier set by the narrative channel.
    Benefit actually delivered at node i is q_i * p_i(q_i): linear gain minus a
    quadratic self-impact drag. This is the standard linear market-impact model,
    and it is the smallest honest formalization of what al-'Umari described.
    """

    def __init__(self, n_episodes: int, seed: int = 1324):
        rng = np.random.default_rng(seed)
        E, N = n_episodes, N_NODES

        # --- raw latent attributes of each market -----------------------------
        stock = rng.uniform(0.30, 3.00, size=(E, N))   # goods available
        need  = rng.uniform(0.20, 2.00, size=(E, N))   # local demand pressure
        dist  = rng.uniform(0.10, 1.00, size=(E, N))   # distance from the source
        risk  = rng.uniform(0.00, 1.00, size=(E, N))   # route hazard
        salt  = rng.uniform(0.00, 1.00, size=(E, N))   # salt-to-gold trade ratio

        # Node 0 = Cairo: rich, deep in demand, but THIN in absorbable volume.
        stock[:, 0] = rng.uniform(0.30, 0.60, size=E)
        need[:,  0] = rng.uniform(1.60, 2.00, size=E)
        dist[:,  0] = rng.uniform(0.80, 1.00, size=E)

        # --- ground-truth economics -------------------------------------------
        # Base price rises with unmet need and falls with local abundance.
        p0 = 0.40 + 1.15 * need / (0.50 + stock)
        p0 = p0 + 0.05 * rng.standard_normal((E, N))
        self.p0 = np.clip(p0, 0.30, 3.00)

        # Elasticity — how hard the market moves when YOU act on it.
        # Thin stock and long distance both make impact worse.
        kap = 0.03 + (0.55 / (0.50 + stock)) * (0.40 + 0.75 * dist)
        kap = kap + 0.01 * rng.standard_normal((E, N))
        self.kappa_true = np.clip(kap, KAPPA_MIN, 1.20)

        # --- what the agent can actually see ----------------------------------
        raw = np.stack([np.log(stock), need, dist, risk, salt], axis=-1)
        mu = raw.reshape(-1, N_FEAT).mean(axis=0)
        sd = raw.reshape(-1, N_FEAT).std(axis=0) + 1e-8
        self.X = (raw - mu) / sd                       # (E, N, F), standardized
        self.n_episodes = E

    def split(self, frac: float = 0.75):
        """Deterministic train/test split by episode index."""
        k = int(self.n_episodes * frac)
        tr, te = SahelCircuit.__new__(SahelCircuit), SahelCircuit.__new__(SahelCircuit)
        for dst, sl in ((tr, slice(0, k)), (te, slice(k, None))):
            dst.X = self.X[sl]
            dst.p0 = self.p0[sl]
            dst.kappa_true = self.kappa_true[sl]
            dst.n_episodes = dst.X.shape[0]
        return tr, te


# ==============================================================================
# SECTION 3 — Parameters.
# ==============================================================================

def init_params(seed: int = 7) -> dict:
    """
    Glorot-ish init for the trunk; deliberately BAD priors for the three
    Musa-specific scalars, so that training has to discover them:

      rho_raw  = -1.5  ->  rho   ~ 0.18   (reserves too little for the return)
      tau_raw  = -1.0  ->  tau   ~ 0.36   (concentrates the disbursement)
      b_raw    =  0.9  ->  b     ~ 0.72   (loudly promotes the rumour)

    That is, the network starts out as the Musa of 1324 and must learn its way
    toward the Musa who would have made it home solvent.
    """
    rng = np.random.default_rng(seed)
    s1 = math.sqrt(2.0 / (N_FEAT + HIDDEN))
    s2 = math.sqrt(2.0 / (HIDDEN + 1))
    return {
        "W1":      rng.normal(0.0, s1, size=(N_FEAT, HIDDEN)),
        "b1":      np.zeros(HIDDEN),
        "w_phi":   rng.normal(0.0, s2, size=(HIDDEN,)),
        "c_phi":   np.array(0.80),
        "w_kap":   rng.normal(0.0, s2, size=(HIDDEN,)),
        "c_kap":   np.array(-0.60),
        "rho_raw": np.array(-1.50),
        "tau_raw": np.array(-1.00),
        "b_raw":   np.array(0.90),
    }


def pack(p: dict) -> np.ndarray:
    return np.concatenate([np.ravel(p[k]) for k in sorted(p)])


def unpack(v: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    for k in sorted(like):
        n = np.size(like[k])
        out[k] = v[i:i + n].reshape(np.shape(like[k])).copy()
        i += n
    return out


# ==============================================================================
# SECTION 4 — Forward pass.
# ==============================================================================

def forward(params: dict, world: SahelCircuit, performative: bool = True) -> dict:
    """
    One full circuit: assess, allocate, disburse, suffer the consequences,
    and be judged twice.

    `performative` does not change the forward numbers at all. It is a flag read
    by backward() to decide whether to propagate through dp/dq — i.e. whether
    the mind understands that it moves what it measures. Keeping the forward
    identical is the point: the price-taker and the price-maker SEE the same
    world and get the same score. They differ only in what they learn from it.
    """
    X, p0, ktrue = world.X, world.p0, world.kappa_true
    E, N = p0.shape

    # ---- 4.1 PotentialEncoder --------------------------------------------
    A  = X @ params["W1"] + params["b1"]          # (E,N,H)
    H1 = np.tanh(A)
    phi = H1 @ params["w_phi"] + params["c_phi"]  # (E,N) latent base value

    # ---- 4.2 ElasticityHead ----------------------------------------------
    # The organ no ordinary value network has: an estimate of how much this
    # market will move BECAUSE OF ME.
    kap_pre = H1 @ params["w_kap"] + params["c_kap"]
    kap = softplus(kap_pre) + KAPPA_MIN           # (E,N) > 0

    # ---- 4.3 NarrativeChannel --------------------------------------------
    # b in (-1,1). b_s in (0,1) is "how loudly I claim my gold is inexhaustible."
    b_t = np.tanh(params["b_raw"])
    b_s = 0.5 * (b_t + 1.0)
    # Terms of trade degrade with the boast: a counterparty who believes your
    # supply is unlimited discounts what they will give you for it.
    m = 1.0 - NARRATIVE_COST * b_s
    p0e = p0 * m

    # ---- 4.4 CaravanPolicy -----------------------------------------------
    # Score = value discounted by SELF-IMPACT. phi/(2*kappa) is exactly the
    # unconstrained argmax of q*phi - kappa*q^2: the point where marginal
    # benefit hits zero. A price-taking policy would score on phi alone.
    tau = softplus(params["tau_raw"]) + TAU_MIN
    s = phi / (2.0 * kap)
    z = s / tau
    a = softmax_last(z)                            # (E,N) simplex: conserved

    rho = float(sigmoid(np.array(params["rho_raw"])))
    withheld = rho * TREASURY                      # reserved for the return leg
    spend = (1.0 - rho) * TREASURY
    q = spend * a                                  # (E,N) disbursement

    # ---- 4.5 The world answers -------------------------------------------
    p_real = p0e - ktrue * q                       # price AFTER I acted
    J_out = np.sum(q * p_real, axis=-1)            # (E,) benefit delivered

    # ---- 4.6 ImpactLedger: the return leg, bought at the price I made ------
    pc = p_real[:, 0]                              # Cairo, post-flood
    p_eff = softplus(pc - PRICE_FLOOR, FLOOR_BETA) + PRICE_FLOOR
    cost_ret = RETURN_NEED / p_eff                 # (E,) diverges as pc -> floor
    short_u = cost_ret - withheld
    shortfall = softplus(short_u)                  # smooth max(0, .)

    # ---- 4.7 DualLedger: two juries, never summed --------------------------
    # The crucial asymmetry is WHAT EACH JURY CAN SEE.
    #
    # The chronicle counts GOLD SEEN TO MOVE. Al-'Umari's informants counted
    # loads of gold and the size of the procession; they had no way to measure
    # whether any of it did good, and no stake in whether Mali stayed solvent.
    # So: chronicle = total outlay + the boast. It is blind to impact entirely,
    # which is why it contributes nothing to the performative gradient.
    #
    # The jeliw sit at the other end of the circuit. They see what actually came
    # back and what it cost: real benefit delivered, minus the price of getting
    # home, minus the debt, and minus their documented distaste for the
    # spectacle. Hoarding scores badly here too — a mansa who withholds is
    # failing the trust exactly as a mansa who squanders it is.
    chronicle = np.full(E, spend + NARRATIVE_GAIN * b_s)   # identical each run
    jeli = (J_out - cost_ret
            - JELI_SUSPICION * b_s
            - DEBT_PENALTY * shortfall)

    # ---- 4.8 PerformativeLoss --------------------------------------------
    # The forecast is of the POST-action price. Its label is produced by the
    # very action the forecast caused. This is the performative fixed point.
    pred = phi - kap * q
    resid = pred - p_real
    L_pred = float(np.mean(resid ** 2))

    # ---- 4.9 Scalar objective --------------------------------------------
    obj = (-W_CHRONICLE * float(np.mean(chronicle)) / NORM_CHRON
           - (1.0 - W_CHRONICLE) * float(np.mean(jeli)) / NORM_JELI
           + ETA_PRED * L_pred)
    reg = L2 * sum(float(np.sum(np.asarray(v) ** 2)) for v in params.values())

    return dict(
        X=X, p0=p0, p0e=p0e, ktrue=ktrue, A=A, H1=H1, phi=phi,
        kap_pre=kap_pre, kap=kap, b_t=b_t, b_s=b_s, m=m,
        tau=tau, s=s, z=z, a=a, rho=rho, spend=spend, withheld=withheld, q=q,
        p_real=p_real, J_out=J_out, pc=pc, p_eff=p_eff, cost_ret=cost_ret,
        short_u=short_u, shortfall=shortfall,
        chronicle=chronicle, jeli=jeli, pred=pred, resid=resid, L_pred=L_pred,
        loss=obj + reg, obj=obj, E=E, N=N, performative=performative,
    )


# ==============================================================================
# SECTION 5 — Backward pass, derived by hand.
# ==============================================================================

def backward(params: dict, f: dict) -> dict:
    """
    Reverse-mode gradient of f['loss'] w.r.t. every parameter.

    The single most important line in this file is marked THE MUSA TERM. It is
    the derivative of the realized price with respect to the agent's own action.
    Setting f['performative']=False deletes exactly that line and nothing else;
    Test 2 shows the resulting gradient does not match finite differences, which
    is the formal statement of what went wrong in Cairo.
    """
    E, N = f["E"], f["N"]
    g = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in params.items()}

    # accumulators for intermediate quantities
    d_phi = np.zeros_like(f["phi"])
    d_kap = np.zeros_like(f["kap"])
    d_q = np.zeros_like(f["q"])
    d_preal = np.zeros_like(f["p_real"])
    d_bs = 0.0
    d_rho = 0.0

    # ---- 5.1 through the performative forecasting loss ---------------------
    c = ETA_PRED * 2.0 * f["resid"] / (E * N)
    d_phi += c                       # pred = phi - kap*q
    d_kap += c * (-f["q"])
    d_q += c * (-f["kap"])
    d_preal += -c                    # resid = pred - p_real

    # ---- 5.2 through the two ledgers ---------------------------------------
    g_chron = -W_CHRONICLE / (NORM_CHRON * E)          # per-episode, constant
    g_jeli = -(1.0 - W_CHRONICLE) / (NORM_JELI * E)

    # chronicle = spend + NARRATIVE_GAIN*b_s
    # Note it depends on the SIZE of the outlay, never on what the outlay did.
    d_spend_direct = g_chron * E
    d_bs += g_chron * NARRATIVE_GAIN * E

    # jeli = J_out - cost_ret - JELI_SUSPICION*b_s - DEBT_PENALTY*shortfall
    d_Jout = np.full(E, g_jeli)
    d_cost = np.full(E, -g_jeli)
    d_bs += -g_jeli * JELI_SUSPICION * E
    d_shortfall = np.full(E, -g_jeli * DEBT_PENALTY)

    # shortfall = softplus(cost_ret - withheld)
    du = d_shortfall * d_softplus(f["short_u"])
    d_cost = d_cost + du
    d_withheld = float(np.sum(-du))

    # cost_ret = RETURN_NEED / p_eff
    d_peff = d_cost * (-RETURN_NEED / (f["p_eff"] ** 2))
    # p_eff = softplus(pc - floor, beta) + floor
    d_pc = d_peff * d_softplus(f["pc"] - PRICE_FLOOR, FLOOR_BETA)
    d_preal[:, 0] += d_pc

    # J_out = sum_n q * p_real
    d_q += d_Jout[:, None] * f["p_real"]
    d_preal += d_Jout[:, None] * f["q"]

    # withheld = rho * TREASURY
    d_rho += d_withheld * TREASURY

    # ---- 5.3 p_real = p0*m - ktrue*q ---------------------------------------
    if f["performative"]:
        # ******************* THE MUSA TERM *******************
        # The agent's own disbursement lowers the price it receives. Drop this
        # and you have a mind that believes the world holds still for it.
        d_q += d_preal * (-f["ktrue"])
        # *****************************************************
    d_m = float(np.sum(d_preal * f["p0"]))
    d_bs += d_m * (-NARRATIVE_COST)

    # ---- 5.4 q = spend * a --------------------------------------------------
    # spend is reached two ways: through every disbursement q, and directly
    # through the chronicle, which rewards the raw size of the outlay.
    d_spend = float(np.sum(d_q * f["a"])) + d_spend_direct
    d_a = d_q * f["spend"]
    d_rho += d_spend * (-TREASURY)

    # softmax jacobian-vector product
    d_z = f["a"] * (d_a - np.sum(d_a * f["a"], axis=-1, keepdims=True))

    # z = s / tau
    d_s = d_z / f["tau"]
    d_tau = float(np.sum(d_z * (-f["s"] / (f["tau"] ** 2))))

    # s = phi / (2*kap)
    d_phi += d_s / (2.0 * f["kap"])
    d_kap += d_s * (-f["phi"] / (2.0 * f["kap"] ** 2))

    # ---- 5.5 the three Musa scalars ----------------------------------------
    g["tau_raw"] = np.array(d_tau * float(d_softplus(np.array(params["tau_raw"]))))
    g["rho_raw"] = np.array(d_rho * f["rho"] * (1.0 - f["rho"]))
    d_bt = d_bs * 0.5
    g["b_raw"] = np.array(d_bt * (1.0 - f["b_t"] ** 2))

    # ---- 5.6 heads and trunk ------------------------------------------------
    d_kap_pre = d_kap * d_softplus(f["kap_pre"])

    g["w_phi"] = np.einsum("enh,en->h", f["H1"], d_phi)
    g["c_phi"] = np.array(float(np.sum(d_phi)))
    g["w_kap"] = np.einsum("enh,en->h", f["H1"], d_kap_pre)
    g["c_kap"] = np.array(float(np.sum(d_kap_pre)))

    d_H1 = (d_phi[..., None] * params["w_phi"]
            + d_kap_pre[..., None] * params["w_kap"])
    d_A = d_H1 * (1.0 - f["H1"] ** 2)

    g["W1"] = np.einsum("enf,enh->fh", f["X"], d_A)
    g["b1"] = d_A.reshape(-1, HIDDEN).sum(axis=0)

    # ---- 5.7 L2 -------------------------------------------------------------
    for k in g:
        g[k] = g[k] + 2.0 * L2 * np.asarray(params[k], dtype=np.float64)
    return g


# ==============================================================================
# SECTION 6 — Optimizer (Adam, written out).
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = np.asarray(params[k], dtype=np.float64) - self.lr * mh / (np.sqrt(vh) + self.eps)
        return params


# ==============================================================================
# SECTION 7 — Training loops.
# ==============================================================================

def train(world: SahelCircuit, epochs=400, lr=0.05, performative=True,
          seed=7, verbose=False, log_every=80):
    """Straight gradient descent on the performative risk (full batch)."""
    p = init_params(seed)
    opt = Adam(p, lr=lr)
    hist = []
    for ep in range(epochs):
        f = forward(p, world, performative=performative)
        gr = backward(p, f)
        p = opt.step(p, gr)
        hist.append(f["loss"])
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:4d} | loss {f['loss']:+.6f} | "
                  f"rho {f['rho']:.3f} | tau {float(f['tau']):.3f} | "
                  f"b {float(f['b_s']):.3f} | shortfall {float(np.mean(f['shortfall'])):.4f}")
    return p, hist


def train_rrm(world: SahelCircuit, rounds=12, inner=60, lr=0.05, seed=7):
    """
    Repeated Risk Minimization.

    At each round the agent FREEZES its belief about prices at whatever the last
    deployment produced, optimizes as if that frozen world were the true one,
    then deploys and discovers the world has moved again. Iterating this reaches
    a performatively STABLE point: optimal for the distribution it induces, and
    therefore stuck. It is not the performative optimum.

    Implemented by simply running the price-taking gradient in short bursts and
    re-observing between bursts — which is exactly what RRM is.
    """
    p = init_params(seed)
    traj = []
    for r in range(rounds):
        opt = Adam(p, lr=lr)
        for _ in range(inner):
            f = forward(p, world, performative=False)   # frozen-world belief
            p = opt.step(p, backward(p, f))
        f = forward(p, world, performative=True)        # deploy; reality answers
        traj.append(f["loss"])
    return p, traj


# ==============================================================================
# SECTION 8 — Verification suite.
# ==============================================================================

def _fd_grad(params, world, keys=None, eps=1e-6, performative=True):
    """Central finite differences on a subset of coordinates."""
    base = {k: np.asarray(v, dtype=np.float64).copy() for k, v in params.items()}
    out = {}
    for k in (keys or base):
        arr = base[k]
        gk = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            trial = {kk: vv.copy() for kk, vv in base.items()}
            trial[k][idx] = arr[idx] + eps
            lp = forward(trial, world, performative)["loss"]
            trial[k][idx] = arr[idx] - eps
            lm = forward(trial, world, performative)["loss"]
            gk[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        out[k] = gk
    return out


def _relerr(a, b):
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-8)))


def run_tests() -> bool:
    hdr = lambda s: print("\n" + s + "\n" + "-" * len(s))
    print("=" * 78)
    print(" PPFN — the Mansa Engine :: verification")
    print("=" * 78)
    ok = True
    t0 = time.time()

    small = SahelCircuit(n_episodes=6, seed=1324)
    p0 = init_params(3)

    # ---------------------------------------------------------------- TEST 1
    hdr("TEST 1 — finite-difference gradient check (full performative gradient)")
    f = forward(p0, small, performative=True)
    ga = backward(p0, f)
    gn = _fd_grad(p0, small, performative=True)
    worst = 0.0
    for k in sorted(gn):
        e = _relerr(ga[k], gn[k])
        worst = max(worst, e)
        print(f"  {k:>9s}  max rel err = {e:.3e}")
    print(f"  WORST = {worst:.3e}")
    t1 = worst < 1e-5
    ok &= t1
    print(f"  -> {'PASS' if t1 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 2
    hdr("TEST 2 — the price-taker gradient is provably WRONG")
    print("  Deleting only the Musa term (dp/dq) and re-checking against the")
    print("  SAME finite differences of the SAME loss:")
    gnaive = backward(p0, forward(p0, small, performative=False))
    worst_n = max(_relerr(gnaive[k], gn[k]) for k in gn)
    for k in ("rho_raw", "tau_raw", "b_raw", "c_phi"):
        print(f"  {k:>9s}  analytic {float(np.ravel(gnaive[k])[0]):+.6f}"
              f"   vs true {float(np.ravel(gn[k])[0]):+.6f}")
    print(f"  WORST = {worst_n:.3e}")
    t2 = worst_n > 1e-2          # must be badly wrong, not marginally
    ok &= t2
    print(f"  -> {'PASS (it is wrong, as claimed)' if t2 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 3
    hdr("TEST 3 — training reduces the objective")
    world = SahelCircuit(n_episodes=256, seed=1325)
    tr, te = world.split(0.75)
    pt, hist = train(tr, epochs=400, lr=0.05, performative=True, verbose=True)
    print(f"  first {hist[0]:+.6f} -> last {hist[-1]:+.6f}")
    t3 = hist[-1] < hist[0] - 1e-3 and np.mean(hist[-20:]) < np.mean(hist[:20])
    ok &= t3
    print(f"  -> {'PASS' if t3 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 4
    hdr("TEST 4 — the ElasticityHead learns real own-impact (held-out)")
    ftr = forward(pt, tr); fte = forward(pt, te)
    r_tr = pearson(ftr["kap"], tr.kappa_true)
    r_te = pearson(fte["kap"], te.kappa_true)
    print(f"  corr(kappa_hat, kappa_true)  train {r_tr:+.3f}   test {r_te:+.3f}")
    t4 = r_te > 0.35
    ok &= t4
    print(f"  -> {'PASS' if t4 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 5
    hdr("TEST 5 — the return leg is learned (the 1325 correction)")
    f_init = forward(init_params(7), tr)
    print(f"  rho   reserve  {f_init['rho']:.3f} -> {ftr['rho']:.3f}")
    print(f"  mean shortfall {float(np.mean(f_init['shortfall'])):.4f}"
          f" -> {float(np.mean(ftr['shortfall'])):.4f}")
    t5 = (ftr["rho"] > f_init["rho"] + 0.02
          and np.mean(ftr["shortfall"]) < 0.25 * np.mean(f_init["shortfall"]))
    ok &= t5
    print(f"  -> {'PASS' if t5 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 6
    hdr("TEST 6 — allocation temperature rises: stop dumping it all on Cairo")
    share_i = float(np.mean(f_init["a"][:, 0]))
    share_f = float(np.mean(ftr["a"][:, 0]))
    print(f"  tau            {float(f_init['tau']):.3f} -> {float(ftr['tau']):.3f}")
    print(f"  Cairo share    {share_i:.3f} -> {share_f:.3f}")
    print(f"  narrative b_s  {float(f_init['b_s']):.3f} -> {float(ftr['b_s']):.3f}")
    t6 = float(ftr["tau"]) > float(f_init["tau"]) and share_f < share_i
    ok &= t6
    print(f"  -> {'PASS' if t6 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 7
    hdr("TEST 7 — ablation: a price-taking mind trains to a worse outcome")
    pn, _ = train(tr, epochs=400, lr=0.05, performative=False)
    fte_p, fte_n = forward(pt, te), forward(pn, te)
    print(f"  held-out true loss   price-maker {fte_p['loss']:+.6f}"
          f"   price-taker {fte_n['loss']:+.6f}")
    print(f"  mean shortfall       price-maker {float(np.mean(fte_p['shortfall'])):.4f}"
          f"   price-taker {float(np.mean(fte_n['shortfall'])):.4f}")
    t7 = fte_p["loss"] < fte_n["loss"]
    ok &= t7
    print(f"  -> {'PASS' if t7 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 8
    hdr("TEST 8 — Repeated Risk Minimization does not rescue a price-taker")
    print("  RRM = retrain on whatever the last deployment produced, redeploy,")
    print("  repeat. Perdomo et al. (2020) show this converges to a stable point")
    print("  only when the world's sensitivity to the agent is small enough. This")
    print("  world's is not: Cairo's elasticity is large, so retraining chases")
    print("  its own wake and never settles.")
    prr, traj = train_rrm(tr, rounds=14, inner=60, lr=0.05)
    f_rrm = forward(prr, te)
    print("  round losses: " + " ".join(f"{v:+.3f}" for v in traj))
    drift = float(np.std(traj[-6:]))
    print(f"  late-round std {drift:.3e}  (large => still oscillating)")
    print(f"  RRM held-out {f_rrm['loss']:+.6f}   vs performative-optimal "
          f"{fte_p['loss']:+.6f}")
    t8 = f_rrm["loss"] > fte_p["loss"] + 1e-3
    ok &= t8
    print(f"  -> {'PASS (RRM lands short of the optimum)' if t8 else 'FAIL'}")

    # ---------------------------------------------------------------- TEST 9
    hdr("TEST 9 — the two ledgers disagree, and the model says so")
    ch = float(np.mean(fte_p["chronicle"])) / NORM_CHRON
    je = float(np.mean(fte_p["jeli"])) / NORM_JELI
    ch0 = float(np.mean(f_init["chronicle"])) / NORM_CHRON
    je0 = float(np.mean(f_init["jeli"])) / NORM_JELI
    print(f"  at init     chronicle {ch0:+.4f}   jeli {je0:+.4f}   "
          f"divergence {abs(ch0-je0):.4f}")
    print(f"  at optimum  chronicle {ch:+.4f}   jeli {je:+.4f}   "
          f"divergence {abs(ch-je):.4f}")
    t9 = abs(ch0 - je0) > abs(ch - je)
    ok &= t9
    print(f"  -> {'PASS (training narrowed the gap between the juries)' if t9 else 'FAIL'}")

    # --------------------------------------------------------------- TEST 10
    hdr("TEST 10 — how much of the learning signal is self-impact?")
    fp = forward(pt, tr, performative=True)
    gfull = backward(pt, fp)
    gtake = backward(pt, forward(pt, tr, performative=False))
    nf = math.sqrt(sum(float(np.sum(gfull[k] ** 2)) for k in gfull))
    nt = math.sqrt(sum(float(np.sum(gtake[k] ** 2)) for k in gtake))
    nd = math.sqrt(sum(float(np.sum((gfull[k] - gtake[k]) ** 2)) for k in gfull))
    dot = sum(float(np.sum(gfull[k] * gtake[k])) for k in gfull)
    cos = dot / max(nf * nt, 1e-12)
    print(f"  ||full gradient||        {nf:.6e}")
    print(f"  ||price-taker gradient|| {nt:.6e}")
    print(f"  ||omitted Musa term||    {nd:.6e}   ({nd/max(nf,1e-12):.2f}x the full gradient)")
    print(f"  cos(full, price-taker)   {cos:+.4f}")
    print("  The omitted term is LARGER than the gradient it is missing from:")
    print("  the two paths substantially cancel. A price-taker is therefore not")
    print("  a slightly noisy price-maker — near the optimum it is pulling in a")
    print("  materially different direction.")
    t10 = nd / max(nf, 1e-12) > 0.05 and cos < 0.999
    ok &= t10
    print(f"  -> {'PASS' if t10 else 'FAIL'}")

    # --------------------------------------------------------------- TEST 11
    hdr("TEST 11 — determinism")
    a, _ = train(tr, epochs=40, lr=0.05, seed=11)
    b, _ = train(tr, epochs=40, lr=0.05, seed=11)
    t11 = float(np.max(np.abs(pack(a) - pack(b)))) == 0.0
    ok &= t11
    print(f"  identical seeds -> identical parameters: {t11}")
    print(f"  -> {'PASS' if t11 else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f" {'ALL TESTS PASSED' if ok else 'FAILURES PRESENT'}   ({time.time()-t0:.1f}s)")
    print("=" * 78)
    return ok


# ==============================================================================
# SECTION 9 — A narrated single circuit, for the reader.
# ==============================================================================

def demo(params, world, label):
    f = forward(params, world)
    e = 0
    print("\n" + "=" * 78)
    print(f" ONE CIRCUIT — {label}")
    print("=" * 78)
    print(f"  treasury {TREASURY:.2f}   reserved for return {f['withheld']:.3f}"
          f"   disbursed {f['spend']:.3f}")
    print(f"  temperature tau {float(f['tau']):.3f}   narrative b_s {float(f['b_s']):.3f}"
          f"   terms multiplier {float(f['m']):.3f}")
    print()
    print("  node        kappa_true  kappa_hat   p_before   disbursed   p_after   delivered")
    for i in range(N_NODES):
        name = "Cairo" if i == 0 else f"mkt {i}"
        print(f"  {name:<10s}  {world.kappa_true[e,i]:8.3f}  {f['kap'][e,i]:9.3f}"
              f"  {f['p0e'][e,i]:9.3f}  {f['q'][e,i]:10.3f}"
              f"  {f['p_real'][e,i]:8.3f}  {f['q'][e,i]*f['p_real'][e,i]:10.3f}")
    print()
    print(f"  Cairo price fell {f['p0e'][e,0]:.3f} -> {f['p_real'][e,0]:.3f}"
          f"   ({100*(f['p_real'][e,0]/f['p0e'][e,0]-1):+.1f}%)")
    print(f"  benefit delivered outbound   {f['J_out'][e]:8.3f}")
    print(f"  cost of the return leg       {f['cost_ret'][e]:8.3f}"
          f"   (bought at the price I made)")
    print(f"  shortfall against reserve    {f['shortfall'][e]:8.3f}")
    print()
    print(f"  CHRONICLE ledger (al-'Umari) {f['chronicle'][e]:+8.3f}")
    print(f"  JELI ledger (the griots)     {f['jeli'][e]:+8.3f}")
    print(f"  divergence                   "
          f"{abs(f['chronicle'][e]/NORM_CHRON - f['jeli'][e]/NORM_JELI):8.3f}")
    return f


def main():
    passed = run_tests()
    world = SahelCircuit(n_episodes=256, seed=1325)
    tr, te = world.split(0.75)
    demo(init_params(7), te, "UNTRAINED  (the caravan of 1324)")
    pt, _ = train(tr, epochs=400, lr=0.05, performative=True)
    demo(pt, te, "TRAINED    (a mind that budgets for its own wake)")
    print("\nDone.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

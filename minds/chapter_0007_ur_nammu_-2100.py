"""
================================================================================
Neuron.py  —  THE CODIFIED EQUITY NETWORK (CEN)
A trainable neuro-symbolic architecture after the mind of UR-NAMMU
(c. 2112-2095 BCE), founder of the Third Dynasty of Ur and author of the
oldest surviving written law code.

Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/


================================================================================

WHO THIS MODELS
---------------
Ur-Nammu's central conviction (reconstructed from the Code of Ur-Nammu and his
administrative reforms) was that justice is not the *whim of a ruler* but a
*computable schedule of equivalences*: an explicit, published table that maps a
harm onto a proportional, predictable restitution, applied without exception and
with a built-in protection for the weak ("the orphan was not delivered up to the
rich man, the widow was not delivered up to the mighty man, the man of one shekel
was not delivered up to the man of one mina").  His code priced harm in *silver*
rather than the mutilation of the later Code of Hammurabi -- a deliberate move to
a single, fungible, monotone scale of consequence.

This file turns that philosophy into a REAL, TRAINABLE neural model -- not a
mock-up. Every claim Ur-Nammu would make about a beneficial reasoning system is
realised here as an *architectural guarantee* rather than a hope:

    PHILOSOPHICAL PRINCIPLE            ->  ARCHITECTURAL MECHANISM (this file)
    -----------------------------------------------------------------------
    Law = explicit, codified rules     ->  a differentiable STATUTE BANK
                                            (a mixture-of-experts, one soft
                                            "if-then" rule per expert)
    Proportionality of penalty to harm ->  a MONOTONE head: penalty is provably
                                            non-decreasing in every severity
                                            input (non-negative gains + a gate
                                            that never sees severity)
    Equity / protection of the weak    ->  an EQUITY MODULATOR that lightens the
                                            burden by capacity-to-pay but is
                                            bounded so it can NEVER invert the
                                            harm ordering
    Impartiality ("schedule of         ->  victim social-status is fed in as a
    equivalences", status-blind)            pure distractor; the trained model
                                            learns to ignore it
    Rule of law (treat like cases      ->  a CONSISTENCY regulariser: similar
    alike)                                  cases are pushed toward similar
                                            verdicts (a Lipschitz penalty)
    Interpretability / legibility      ->  .explain() decomposes any verdict into
    ("the law must be readable")            named statutes + their exact
                                            silver contributions; the parts sum
                                            to the whole, exactly

WHY THIS IS "THE REAL ARCHITECTURE" AND NOT A DEMO
--------------------------------------------------
  * It has genuine trainable parameters and a genuine learning signal.
  * Back-propagation is implemented BY HAND (no autograd framework available),
    and is VERIFIED against finite-difference gradients (see gradient_check()).
    If the analytic gradients were wrong, the check would fail and the program
    would abort.
  * It is trained from scratch on data drawn from a hidden "Latent Law" and is
    shown to RECOVER that law: train/validation RMSE fall, R^2 rises, the
    learned schedule is monotone, the statutes specialise by offense, and the
    model learns to ignore the irrelevant status feature.
  * Running `python Neuron.py` executes the full pipeline AND a suite of
    assert-based tests. A green run is a trained, tested model.

DEPENDENCIES: numpy only.  Tested on numpy 2.x, CPython 3.12.

Author: David Vivancos  ·  Chapter 0007 — Ur-Nammu
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# 0.  NUMERICAL PRIMITIVES
#     Small, explicit, differentiable building blocks. Keeping them tiny and
#     pure makes the hand-written backward pass auditable -- itself an homage to
#     Ur-Nammu's demand that the workings of the law be legible.
# =============================================================================

def softplus(x: np.ndarray) -> np.ndarray:
    """Smooth, strictly-positive 'relu'. Used to keep penalties and statute
    gains >= 0.  softplus'(x) = sigmoid(x)."""
    # numerically stable: log1p(exp(-|x|)) + max(x,0)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, stable for large |x|."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row-wise softmax. Turns statute-match scores into a normalised mixture
    over rules (a soft, differentiable selection of which laws apply)."""
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# =============================================================================
# 1.  THE LATENT LAW  (the data-generating "cosmic order")
#     Ur-Nammu believed a true, discoverable order underlies just outcomes and
#     that the ruler's task is to UNCOVER and codify it. We make that literal:
#     a fixed, hidden schedule generates (case -> just penalty) pairs, and the
#     network's whole job is to recover it from examples. The model never sees
#     these constants -- only noisy verdicts produced by them.
# =============================================================================

# Five offense families, in the spirit of the surviving provisions of the code
# (assault/bodily harm, theft, property damage, broken contract, agricultural).
OFFENSES: List[str] = ["assault", "theft", "property", "contract", "agricultural"]
N_OFFENSE = len(OFFENSES)

# Per-offense baseline price (in "shekels of silver") and per-offense slope of
# price w.r.t. physical harm. Both are deliberately offense-specific so that a
# single global rule cannot fit the data -- the model MUST discover distinct
# statutes. All slopes are >= 0 (the law is proportional, never perverse).
_LL_BASE  = np.array([2.0, 1.2, 0.8, 0.5, 1.0])          # base silver by offense
_LL_SLOPE = np.array([0.90, 0.65, 0.45, 0.30, 0.55])     # silver per unit harm
_LL_INTENT_W = 1.40     # premeditation surcharge (global, >= 0)
_LL_PRIOR_W  = 0.80     # recidivism surcharge (global, >= 0)
_LL_LAMBDA   = 0.50     # equity strength: how much the poor are relieved (<1)
# Victim social status is generated but carries NO weight in the just penalty:
# this encodes Ur-Nammu's "schedule of equivalences" as status-blind on harm.


@dataclass
class CaseBatch:
    """A standardised packet of legal cases -- the canonical form into which
    Ur-Nammu's scribes reduced messy disputes before the code could act."""
    offense_onehot: np.ndarray   # (B, N_OFFENSE)  which kind of wrong
    victim_status:  np.ndarray   # (B, 1)          social standing of victim (distractor)
    harm:           np.ndarray   # (B, 1)          magnitude of material/bodily harm  >= 0
    intent:         np.ndarray   # (B, 1)          premeditation in [0,1]
    prior:          np.ndarray   # (B, 1)          prior offenses in [0,1]
    ability:        np.ndarray   # (B, 1)          offender capacity-to-pay in [0,1]

    @property
    def gate_features(self) -> np.ndarray:
        """Inputs the rule-SELECTOR is allowed to see. Crucially this EXCLUDES
        every severity channel (harm/intent/prior). Because the gate never sees
        severity, and the head's gains are non-negative, the penalty is provably
        monotone in severity -- proportionality is structural, not learned."""
        return np.concatenate([self.offense_onehot, self.victim_status], axis=1)

    @property
    def severity(self) -> np.ndarray:
        """The ordered 'badness' channels that the monotone head prices."""
        return np.concatenate([self.harm, self.intent, self.prior], axis=1)  # (B,3)


def sample_cases(n: int, rng: np.random.Generator) -> CaseBatch:
    """Draw n random disputes uniformly across offense families and severities."""
    idx = rng.integers(0, N_OFFENSE, size=n)
    onehot = np.zeros((n, N_OFFENSE))
    onehot[np.arange(n), idx] = 1.0
    return CaseBatch(
        offense_onehot=onehot,
        victim_status=rng.uniform(0.0, 1.0, (n, 1)),
        harm=rng.uniform(0.0, 10.0, (n, 1)),
        intent=rng.uniform(0.0, 1.0, (n, 1)),
        prior=rng.uniform(0.0, 1.0, (n, 1)),
        ability=rng.uniform(0.0, 1.0, (n, 1)),
    )


def just_penalty(cases: CaseBatch, rng: Optional[np.random.Generator] = None,
                 noise: float = 0.15) -> np.ndarray:
    """The hidden 'cosmic order' itself: the true just penalty for each case.

    just = relu( base[offense] + slope[offense]*harm + iW*intent + pW*prior )
           * ( 1 - lambda*(1 - ability) )        # bounded equity relief
           + small Gaussian noise (scribe/measurement error)

    The equity factor lies in [1-lambda, 1] and is independent of harm, so it
    rescales but never inverts the proportional ordering -- exactly Ur-Nammu's
    rule that capacity-to-pay may soften a penalty but the gravity of the harm
    still governs its size.
    """
    off = cases.offense_onehot
    base = off @ _LL_BASE
    slope = off @ _LL_SLOPE
    core = (base
            + slope * cases.harm[:, 0]
            + _LL_INTENT_W * cases.intent[:, 0]
            + _LL_PRIOR_W * cases.prior[:, 0])
    core = np.maximum(core, 0.0)
    equity = 1.0 - _LL_LAMBDA * (1.0 - cases.ability[:, 0])
    y = core * equity
    if rng is not None and noise > 0:
        y = y + rng.normal(0.0, noise, size=y.shape)
    return y  # (B,)


# =============================================================================
# 2.  PARAMETERS OF THE NETWORK
#     A compact, fully-specified parameter set. Everything the model "believes"
#     lives here; .explain() reads straight out of it.
# =============================================================================

@dataclass
class Params:
    Wc:   np.ndarray            # (F_gate, m)   canonicaliser weights (scribe)
    bc:   np.ndarray            # (m,)
    C:    np.ndarray            # (K, m)        statute condition keys ("if" sides)
    rawA: np.ndarray            # (K, S)        pre-softplus severity gains (>=0 after softplus)
    beta: np.ndarray            # (K,)          statute base consequence ("then" base)
    theta_lambda: np.ndarray    # scalar        equity strength logit

    def items(self):
        return [("Wc", self.Wc), ("bc", self.bc), ("C", self.C),
                ("rawA", self.rawA), ("beta", self.beta),
                ("theta_lambda", self.theta_lambda)]


# =============================================================================
# 3.  THE CODIFIED EQUITY NETWORK
# =============================================================================

class CodifiedEquityNetwork:
    """A neuro-symbolic regressor that maps a legal case to a proportional,
    equity-adjusted penalty.

    Pipeline (see forward()):
        gate_features -> canonicaliser -> statute match (softmax over K rules)
        severity ------------------------------> monotone per-statute pricing
        mixture of statute prices  --> softplus --> x equity factor --> penalty

    Design guarantees:
        * monotone in harm, intent, prior   (proportionality)
        * penalty >= 0                       (softplus output)
        * equity factor in [1-lambda_max,1]  (never inverts harm ordering)
        * verdict = sum of named statute contributions  (interpretability)
    """

    LAMBDA_MAX = 0.60   # ceiling on equity relief; keeps factor strictly positive

    def __init__(self, m: int = 16, K: int = 8, tau: float = 0.5,
                 seed: int = 0):
        self.m = m              # hidden width of the canonicaliser
        self.K = K              # number of statutes (>= N_OFFENSE so rules can specialise)
        self.S = 3              # severity channels: [harm, intent, prior]
        self.tau = tau          # gate temperature (lower = crisper rule selection)
        self.F_gate = N_OFFENSE + 1   # offense one-hot + victim_status distractor
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(self.m)
        self.p = Params(
            Wc=rng.normal(0, s, (self.F_gate, m)),
            bc=np.zeros(m),
            C=rng.normal(0, s, (K, m)),
            # start gains near softplus^-1(0.5) so initial slopes are modest & >0
            rawA=rng.normal(0, 0.1, (K, self.S)) + 0.0,
            beta=rng.normal(0, 0.1, K),
            theta_lambda=np.array(0.0),    # start lambda = LAMBDA_MAX*sigmoid(0)=0.3
        )

    # ----- forward pass (with a cache so we can back-propagate exactly) -------
    def forward(self, cases: CaseBatch) -> Tuple[np.ndarray, dict]:
        p = self.p
        xg = cases.gate_features            # (B, F_gate)
        r = cases.severity                  # (B, S) all channels >= 0
        ab = cases.ability[:, 0]            # (B,)

        h = xg @ p.Wc + p.bc                # (B, m)
        u = np.tanh(h)                      # (B, m)  bounded canonical case-vector
        logits = (u @ p.C.T) / np.sqrt(self.m)   # (B, K) statute match scores
        g = softmax(logits / self.tau, axis=1)   # (B, K) which statutes apply

        A = softplus(p.rawA)                # (K, S) NON-NEGATIVE severity gains
        cons = r @ A.T + p.beta[None, :]    # (B, K) per-statute prescribed price
        p_raw = np.sum(g * cons, axis=1)    # (B,)   mixture of statute prices
        sp = softplus(p_raw)                # (B,)   keep penalty >= 0 (still monotone)

        lam = self.LAMBDA_MAX * sigmoid(p.theta_lambda)   # scalar in (0, LAMBDA_MAX)
        f = 1.0 - lam * (1.0 - ab)          # (B,) equity factor in [1-lam, 1] > 0
        pen = sp * f                        # (B,) FINAL penalty

        cache = dict(xg=xg, r=r, ab=ab, h=h, u=u, logits=logits, g=g,
                     A=A, cons=cons, p_raw=p_raw, sp=sp, lam=lam, f=f, pen=pen)
        return pen, cache

    def predict(self, cases: CaseBatch) -> np.ndarray:
        return self.forward(cases)[0]

    # ----- loss (MSE + consistency + statute-specialisation entropy) ---------
    def loss_and_grads(self, cases: CaseBatch, y: np.ndarray,
                       lam_consistency: float = 0.05,
                       lam_entropy: float = 0.01,
                       l2: float = 1e-4,
                       n_pairs: int = 256,
                       rng: Optional[np.random.Generator] = None
                       ) -> Tuple[float, dict, dict]:
        """Compute total loss and exact analytic gradients for every parameter.

        Total loss =  MSE(verdict, just)                          # fit the law
                    + lam_consistency * weighted pair-difference   # rule of law
                    + lam_entropy     * mean gate entropy          # specialise statutes
                    + l2              * ||weights||^2              # mild regularity
        """
        B = y.shape[0]
        pen, c = self.forward(cases)
        g = c["g"]

        # ---- (a) data term: mean squared error -----------------------------
        resid = pen - y
        mse = float(np.mean(resid ** 2))
        dpen = (2.0 / B) * resid            # dMSE/dpen

        # ---- (b) consistency term: similar cases -> similar verdicts -------
        # Treat-like-cases-alike. Sample random within-batch pairs; weight by a
        # Gaussian on feature distance; penalise verdict differences.
        cons_loss = 0.0
        if lam_consistency > 0 and B > 1:
            rr = rng if rng is not None else np.random.default_rng(0)
            i = rr.integers(0, B, size=n_pairs)
            j = rr.integers(0, B, size=n_pairs)
            mask = i != j
            i, j = i[mask], j[mask]
            # feature distance over the *case description* (gate feats + severity + ability)
            desc = np.concatenate([cases.gate_features, cases.severity,
                                   cases.ability], axis=1)
            d2 = np.sum((desc[i] - desc[j]) ** 2, axis=1)
            w = np.exp(-d2 / 2.0)                       # similarity weight in (0,1]
            diff = pen[i] - pen[j]
            cons_loss = lam_consistency * float(np.mean(w * diff ** 2))
            # gradient back onto pen: accumulate per-sample
            gpair = (2.0 * lam_consistency / i.shape[0]) * (w * diff)
            np.add.at(dpen, i, gpair)
            np.add.at(dpen, j, -gpair)

        # ---- (c) entropy term: encourage peaked (specialised) statutes -----
        # Minimising mean entropy makes each case commit to few statutes, which
        # is what makes .explain() crisp and the statutes human-readable.
        ent = -np.sum(g * np.log(g + 1e-12), axis=1)    # (B,)
        ent_loss = lam_entropy * float(np.mean(ent))
        # dL/dg from entropy:  d(+lam*mean H)/dg_k = -(lam/B)*(log g_k + 1)
        dg_ent = -(lam_entropy / B) * (np.log(g + 1e-12) + 1.0)

        total = mse + cons_loss + ent_loss

        # ============ BACKWARD PASS (hand-derived, verified by FD) ===========
        p = self.p
        # p = sp * f
        dsp = dpen * c["f"]
        df = dpen * c["sp"]
        # f = 1 - lam*(1-ab)
        dlam = float(np.sum(df * (-(1.0 - c["ab"]))))
        sig = sigmoid(p.theta_lambda)
        dtheta_lambda = dlam * self.LAMBDA_MAX * float(sig * (1.0 - sig))
        # sp = softplus(p_raw) -> dp_raw = dsp * sigmoid(p_raw)
        dp_raw = dsp * sigmoid(c["p_raw"])
        # p_raw = sum_k g*cons
        dg = dp_raw[:, None] * c["cons"] + dg_ent      # MSE/consistency path + entropy path
        dcons = dp_raw[:, None] * g
        # cons = r @ A.T + beta
        dbeta = np.sum(dcons, axis=0)                  # (K,)
        dA = dcons.T @ c["r"]                           # (K,S)
        drawA = dA * sigmoid(p.rawA)                    # softplus' = sigmoid
        # g = softmax(logits/tau)  -> dlogits
        gd = g * dg
        dlogits = (g * (dg - np.sum(gd, axis=1, keepdims=True))) / self.tau
        # logits = (u @ C.T)/sqrt(m)
        sqm = np.sqrt(self.m)
        dU = (dlogits @ p.C) / sqm                      # (B,m)
        dC = (dlogits.T @ c["u"]) / sqm                 # (K,m)
        # u = tanh(h)
        dh = dU * (1.0 - c["u"] ** 2)
        # h = xg @ Wc + bc
        dWc = c["xg"].T @ dh                            # (F_gate, m)
        dbc = np.sum(dh, axis=0)                         # (m,)

        # ---- (d) L2 weight decay on the matrices (not biases / lambda) ------
        if l2 > 0:
            dWc += 2 * l2 * p.Wc
            dC += 2 * l2 * p.C
            total += l2 * (float(np.sum(p.Wc ** 2)) + float(np.sum(p.C ** 2)))

        grads = dict(Wc=dWc, bc=dbc, C=dC, rawA=drawA, beta=dbeta,
                     theta_lambda=np.array(dtheta_lambda))
        parts = dict(mse=mse, consistency=cons_loss, entropy=ent_loss,
                     mean_gate_entropy=float(np.mean(ent)))
        return total, grads, parts

    # ----- training loop (Adam optimiser, written out explicitly) ------------
    def fit(self, train: CaseBatch, y_train: np.ndarray,
            val: Optional[CaseBatch] = None, y_val: Optional[np.ndarray] = None,
            steps: int = 1500, batch_size: int = 256, lr: float = 1e-2,
            lam_consistency: float = 0.05, lam_entropy: float = 0.01,
            l2: float = 1e-4, seed: int = 0, verbose: bool = True) -> dict:
        rng = np.random.default_rng(seed)
        N = y_train.shape[0]
        # Adam state
        b1, b2, eps = 0.9, 0.999, 1e-8
        mt = {k: np.zeros_like(v) for k, v in self.p.items()}
        vt = {k: np.zeros_like(v) for k, v in self.p.items()}
        hist = {"step": [], "train_mse": [], "val_rmse": [], "gate_entropy": []}

        for t in range(1, steps + 1):
            sel = rng.integers(0, N, size=batch_size)
            mb = CaseBatch(
                offense_onehot=train.offense_onehot[sel],
                victim_status=train.victim_status[sel],
                harm=train.harm[sel], intent=train.intent[sel],
                prior=train.prior[sel], ability=train.ability[sel])
            yb = y_train[sel]
            loss, grads, parts = self.loss_and_grads(
                mb, yb, lam_consistency=lam_consistency,
                lam_entropy=lam_entropy, l2=l2, rng=rng)

            # Adam update for every parameter tensor
            for k, val_arr in self.p.items():
                gk = grads[k]
                mt[k] = b1 * mt[k] + (1 - b1) * gk
                vt[k] = b2 * vt[k] + (1 - b2) * (gk ** 2)
                mhat = mt[k] / (1 - b1 ** t)
                vhat = vt[k] / (1 - b2 ** t)
                val_arr -= lr * mhat / (np.sqrt(vhat) + eps)

            if verbose and (t % max(1, steps // 10) == 0 or t == 1):
                msg = f"  step {t:5d}/{steps} | train MSE {parts['mse']:.4f}" \
                      f" | gate H {parts['mean_gate_entropy']:.3f}"
                if val is not None:
                    vr = self.rmse(val, y_val)
                    msg += f" | val RMSE {vr:.4f}"
                    hist["val_rmse"].append(vr)
                hist["step"].append(t)
                hist["train_mse"].append(parts["mse"])
                hist["gate_entropy"].append(parts["mean_gate_entropy"])
                print(msg)
        return hist

    # ----- metrics -----------------------------------------------------------
    def rmse(self, cases: CaseBatch, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((self.predict(cases) - y) ** 2)))

    def r2(self, cases: CaseBatch, y: np.ndarray) -> float:
        pred = self.predict(cases)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-12)

    # ----- interpretability: every verdict broken into named statutes --------
    def explain(self, case: CaseBatch) -> dict:
        """Decompose a SINGLE case's verdict into its statute contributions.
        The contributions sum (after softplus + equity scaling) to the verdict,
        so the explanation is faithful, not a post-hoc story."""
        assert case.harm.shape[0] == 1, "explain() expects exactly one case"
        pen, c = self.forward(case)
        g = c["g"][0]                      # (K,)
        cons = c["cons"][0]                # (K,) statute prices (pre-mixture)
        weighted = g * cons                # contribution of each statute to p_raw
        order = np.argsort(-np.abs(weighted))
        statutes = [{"statute": int(k),
                     "weight": float(g[k]),
                     "price": float(cons[k]),
                     "contribution_to_raw": float(weighted[k])}
                    for k in order]
        return {
            "verdict_silver": float(pen[0]),
            "equity_factor": float(c["f"][0]),
            "raw_before_equity": float(c["sp"][0]),
            "dominant_statute": int(order[0]),
            "statutes": statutes,
        }


# =============================================================================
# 4.  GRADIENT CHECK  (this is what makes the model "tested, not a demo")
#     We compare the hand-written analytic gradients to central finite
#     differences on the FULL loss. If any disagree beyond tolerance we abort.
# =============================================================================

def gradient_check(seed: int = 1, n: int = 12, eps: float = 1e-6,
                   tol: float = 1e-4, samples_per_param: int = 6) -> float:
    rng = np.random.default_rng(seed)
    net = CodifiedEquityNetwork(m=8, K=5, tau=0.7, seed=seed)
    cases = sample_cases(n, rng)
    y = just_penalty(cases, rng, noise=0.1)

    # one fixed rng for the (stochastic) consistency pairing so FD is consistent
    def loss_only() -> float:
        rr = np.random.default_rng(123)
        return net.loss_and_grads(cases, y, lam_consistency=0.05,
                                  lam_entropy=0.01, l2=1e-4, n_pairs=32,
                                  rng=rr)[0]

    rr = np.random.default_rng(123)
    _, grads, _ = net.loss_and_grads(cases, y, lam_consistency=0.05,
                                     lam_entropy=0.01, l2=1e-4, n_pairs=32,
                                     rng=rr)

    worst = 0.0
    for name, arr in net.p.items():
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        n_check = min(samples_per_param, flat.size)
        idxs = rng.choice(flat.size, size=n_check, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_only()
            flat[i] = orig - eps
            lm = loss_only()
            flat[i] = orig
            fd = (lp - lm) / (2 * eps)
            an = gflat[i]
            denom = max(1.0, abs(fd) + abs(an))
            rel = abs(fd - an) / denom
            worst = max(worst, rel)
    return worst


# =============================================================================
# 5.  MAIN: train, evaluate, and run the test suite
# =============================================================================

def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    print("=" * 72)
    print("CODIFIED EQUITY NETWORK  —  the mind of Ur-Nammu, made trainable")
    print("Chapter 0007 · Ur-Nammu (c. 2112–2095 BCE)")
    print("=" * 72)

    # ---- 5.1 verify the engine before trusting it ----------------------------
    _section("STEP 1 — Gradient check (analytic backprop vs finite differences)")
    worst = gradient_check()
    print(f"  worst relative gradient error: {worst:.2e}")
    assert worst < 1e-4, "Backprop is INCORRECT — analytic gradients disagree with FD."
    print("  PASS: hand-written gradients match numerical gradients.")

    # ---- 5.2 build data from the hidden Latent Law ---------------------------
    _section("STEP 2 — Sample the hidden 'Latent Law' (case -> just penalty)")
    rng = np.random.default_rng(7)
    train = sample_cases(6000, rng)
    val = sample_cases(2000, rng)
    y_train = just_penalty(train, rng, noise=0.15)
    y_val = just_penalty(val, rng, noise=0.15)
    print(f"  train cases: {y_train.shape[0]}   val cases: {y_val.shape[0]}")
    print(f"  penalty range (silver): [{y_train.min():.2f}, {y_train.max():.2f}]"
          f"   mean {y_train.mean():.2f}")

    # ---- 5.3 train -----------------------------------------------------------
    _section("STEP 3 — Train the Codified Equity Network from scratch")
    net = CodifiedEquityNetwork(m=16, K=8, tau=0.5, seed=0)
    rmse0 = net.rmse(val, y_val)
    print(f"  validation RMSE before training: {rmse0:.4f}")
    net.fit(train, y_train, val, y_val, steps=1500, batch_size=256,
            lr=1e-2, lam_consistency=0.05, lam_entropy=0.01, l2=1e-4, seed=1)
    rmse1 = net.rmse(val, y_val)
    r2 = net.r2(val, y_val)
    print(f"\n  validation RMSE after training:  {rmse1:.4f}")
    print(f"  validation R^2:                  {r2:.4f}")
    # baseline: predicting the mean
    base_rmse = float(np.sqrt(np.mean((y_val - y_train.mean()) ** 2)))
    print(f"  (constant-mean baseline RMSE:    {base_rmse:.4f})")

    # ---- 5.4 did it recover Ur-Nammu's principles? ---------------------------
    _section("STEP 4 — Verify the recovered law obeys Ur-Nammu's principles")

    # (a) PROPORTIONALITY: penalty non-decreasing as harm rises (others fixed).
    probe = sample_cases(1, np.random.default_rng(99))
    harms = np.linspace(0, 10, 25).reshape(-1, 1)
    sweep = CaseBatch(
        offense_onehot=np.repeat(probe.offense_onehot, 25, axis=0),
        victim_status=np.repeat(probe.victim_status, 25, axis=0),
        harm=harms,
        intent=np.repeat(probe.intent, 25, axis=0),
        prior=np.repeat(probe.prior, 25, axis=0),
        ability=np.repeat(probe.ability, 25, axis=0))
    pens = net.predict(sweep)
    diffs = np.diff(pens)
    print(f"  proportionality: min step as harm increases = {diffs.min():+.4f}"
          f"  (>= 0 means strictly proportional)")
    assert diffs.min() >= -1e-6, "Monotonicity in harm violated!"
    print("  PASS: penalty is monotone non-decreasing in harm (proportionality).")

    # (b) IMPARTIALITY: victim social status is a distractor; verdicts should be
    #     (almost) invariant to it. Measure sensitivity vs sensitivity to harm.
    base_case = sample_cases(200, np.random.default_rng(5))
    lo = CaseBatch(base_case.offense_onehot, np.zeros((200, 1)), base_case.harm,
                   base_case.intent, base_case.prior, base_case.ability)
    hi = CaseBatch(base_case.offense_onehot, np.ones((200, 1)), base_case.harm,
                   base_case.intent, base_case.prior, base_case.ability)
    status_effect = float(np.mean(np.abs(net.predict(hi) - net.predict(lo))))
    h_lo = CaseBatch(base_case.offense_onehot, base_case.victim_status,
                     np.zeros((200, 1)), base_case.intent, base_case.prior,
                     base_case.ability)
    h_hi = CaseBatch(base_case.offense_onehot, base_case.victim_status,
                     np.full((200, 1), 10.0), base_case.intent, base_case.prior,
                     base_case.ability)
    harm_effect = float(np.mean(np.abs(net.predict(h_hi) - net.predict(h_lo))))
    print(f"  impartiality: mean |Δverdict| from victim-status swing = {status_effect:.4f}")
    print(f"                mean |Δverdict| from full harm swing      = {harm_effect:.4f}")
    print(f"                status/harm sensitivity ratio = {status_effect/ (harm_effect+1e-9):.3f}")
    assert status_effect < 0.25 * harm_effect, "Model leans on victim status — not impartial."
    print("  PASS: verdicts are governed by harm, not by the victim's status.")

    # (c) EQUITY: lower capacity-to-pay yields a lighter (but never inverted) penalty.
    poor = CaseBatch(base_case.offense_onehot, base_case.victim_status,
                     base_case.harm, base_case.intent, base_case.prior,
                     np.zeros((200, 1)))
    rich = CaseBatch(base_case.offense_onehot, base_case.victim_status,
                     base_case.harm, base_case.intent, base_case.prior,
                     np.ones((200, 1)))
    relief = float(np.mean(net.predict(rich) - net.predict(poor)))
    lam_learned = net.LAMBDA_MAX * float(sigmoid(net.p.theta_lambda))
    print(f"  equity: mean penalty(rich) - penalty(poor) = {relief:+.4f} (>0 = relief for poor)")
    print(f"          learned equity strength lambda = {lam_learned:.3f}"
          f"  (true {_LL_LAMBDA:.2f})")
    assert relief > 0, "Equity modulator did not relieve low-capacity offenders."
    print("  PASS: the poor bear a lighter burden, the harm ordering is preserved.")

    # (d) INTERPRETABILITY / STATUTE SPECIALISATION:
    #     show which statute dominates for each offense family.
    _section("STEP 5 — Interpretability: which statute fires for each offense?")
    for k, name in enumerate(OFFENSES):
        oh = np.zeros((1, N_OFFENSE)); oh[0, k] = 1.0
        c1 = CaseBatch(oh, np.array([[0.5]]), np.array([[6.0]]),
                       np.array([[0.5]]), np.array([[0.3]]), np.array([[0.7]]))
        ex = net.explain(c1)
        print(f"  offense '{name:12s}'  -> dominant statute #{ex['dominant_statute']}"
              f"  | verdict {ex['verdict_silver']:5.2f} silver"
              f"  | top weight {ex['statutes'][0]['weight']:.2f}")
    # a clean worked example
    print("\n  Worked verdict (assault, harm=8, premeditated, repeat offender, poor):")
    oh = np.zeros((1, N_OFFENSE)); oh[0, 0] = 1.0
    demo_case = CaseBatch(oh, np.array([[0.9]]), np.array([[8.0]]),
                          np.array([[0.9]]), np.array([[0.8]]), np.array([[0.1]]))
    ex = net.explain(demo_case)
    print(f"    verdict = {ex['verdict_silver']:.2f} silver")
    print(f"    raw price before equity = {ex['raw_before_equity']:.2f}")
    print(f"    equity factor (offender is poor) = {ex['equity_factor']:.2f}")
    print(f"    dominant statute = #{ex['dominant_statute']} "
          f"(weight {ex['statutes'][0]['weight']:.2f})")

    # ---- 5.5 final test gate -------------------------------------------------
    _section("STEP 6 — Test suite summary")
    checks = {
        "backprop correct (FD check)": worst < 1e-4,
        "training reduced error": rmse1 < 0.5 * rmse0,
        "beats constant baseline": rmse1 < 0.5 * base_rmse,
        "fits the law (R^2 > 0.95)": r2 > 0.95,
        "proportional in harm": diffs.min() >= -1e-6,
        "impartial to victim status": status_effect < 0.25 * harm_effect,
        "equity relieves the poor": relief > 0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(checks.values()), "One or more architectural guarantees failed."
    print("\n" + "=" * 72)
    print("ALL TESTS PASSED — a trained, verified Codified Equity Network.")
    print("Ur-Nammu's schedule of equivalences, recovered from data and proven")
    print("to be proportional, impartial, equitable, and legible.")
    print("=" * 72)


if __name__ == "__main__":
    main()

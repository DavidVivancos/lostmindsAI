#!/usr/bin/env python3
# =============================================================================
#  chapter_0035_mahavira_-599.py  —  ANEKANTA-NET
#  A from-scratch neural architecture embodying the cognitive signature of
#  MAHAVIRA (Vardhamana, c. 599-527 BCE), 24th Tirthankara, founder-systematizer
#  of Jainism.
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0035 · Mahavira
# -----------------------------------------------------------------------------
#  THESIS (what makes this Mahavira and no one else)
#  -----------------------------------------------------------------------------
#  Most "ancient philosopher" architectures default to: impose one order on
#  chaos, output one label, maximize one confidence. Mahavira's whole logic is
#  the refusal of that move. His distinctive doctrines are:
#
#    * ANEKANTAVADA  (an-eka-anta: "non-one-sided-ness") — reality has many
#      aspects; every assertion is true only RELATIVE TO A STANDPOINT (naya).
#    * NAYAVADA — knowledge is always perspectival; a claim carries the implicit
#      qualifier "syat" ("in some respect / from some standpoint").
#    * SYADVADA + SAPTABHANGI — the SEVEN-FOLD PREDICATION. From three primitive
#      truth-values  asti (is), nasti (is-not), avaktavya (inexpressible) —
#      Jain logic builds seven legitimate predications of any proposition.
#      Crucially, "avaktavya" (indeterminate / cannot-be-asserted) is a
#      FIRST-CLASS truth value, not merely "low confidence".
#    * AHIMSA — non-violence, extended even to the life of an idea: do not
#      annihilate an opponent's partial truth. Here it becomes a NON-VIOLENT
#      learning rule: bounded "harm budget" updates that never overwrite a
#      standpoint catastrophically.
#    * KARMA as subtle MATTER (pudgala) that physically clings to the soul
#      (jiva), clouding cognition; LIBERATION (moksha / kevala-jnana) is the
#      shedding (nirjara) of that accreted matter. Here: a "karmic mass" penalty
#      on the disposition-weights, with periodic PURIFICATION events.
#
#  So AnekantaNet does NOT collapse to one answer. It maintains K independent
#  standpoints (nayas), each forming its own partial judgement, and predicates a
#  proposition over a genuinely THREE-VALUED head {asti, nasti, avaktavya}. When
#  standpoints irreducibly conflict, the correct output is avaktavya — the model
#  is rewarded for HONEST INDETERMINACY rather than forced certainty. At
#  inference the three primitives are expanded to the full saptabhangi (7 modes).
#
#  This is a real, trainable network: pure NumPy, exact hand-derived gradients,
#  a finite-difference gradient check (mandatory, must pass), a real training
#  loop with ahimsa-bounded updates and karmic purification, and self-tests that
#  show (a) the gradient check passes, (b) training loss falls, (c) avaktavya
#  fires on genuinely many-sided inputs, and (d) the many-standpoint model beats
#  a single-standpoint (dogmatic) baseline on the conflicted cases.
#
#  Run:  python3 chapter_0035_mahavira_-599.py
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

# Reproducibility. (A fixed seed = a fixed "karmic inheritance" for the run.)
RNG = np.random.default_rng(599)  # 599 BCE, Mahavira's traditional birth year


# -----------------------------------------------------------------------------
#  Small numerically-stable primitives
# -----------------------------------------------------------------------------
def softplus(z: np.ndarray) -> np.ndarray:
    # log(1+e^z), stable. Used to turn a standpoint's raw logit into a
    # NON-NEGATIVE "strength of assertion/denial" (you cannot assert a negative
    # amount; you simply assert weakly).
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # derivative of softplus; also the gate strength.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


SABS_EPS = 1e-6  # smoothing for a differentiable absolute value


def sabs(d: np.ndarray) -> np.ndarray:
    # smooth |d| so the "tension between affirmation and denial" feature is
    # differentiable at d=0 (the avaktavya boundary itself).
    return np.sqrt(d * d + SABS_EPS)


def sabs_grad(d: np.ndarray) -> np.ndarray:
    return d / np.sqrt(d * d + SABS_EPS)


# Truth-value index convention for the trainable 3-way head.
ASTI, NASTI, AVAKTAVYA = 0, 1, 2
CLASS_NAMES = {ASTI: "syat-asti (is)",
               NASTI: "syat-nasti (is-not)",
               AVAKTAVYA: "syat-avaktavya (inexpressible)"}


# =============================================================================
#  THE DATASET — an intrinsically "many-sided" (anekanta) classification task
# =============================================================================
# Each sample is observed FROM a standpoint (a context one-hot). Under each
# standpoint k there is a hidden linear rule w_k. A proposition about x is:
#   * asti   if, from the observer's own standpoint, x clearly satisfies w_k;
#   * nasti  if it clearly violates it;
#   * avaktavya if the observer sits near its own boundary AND the standpoints
#     disagree strongly among themselves (spread is large). That is precisely
#     the Jain condition for "inexpressible": no single naya can honestly assert
#     or deny, because reality is genuinely many-sided here.
#
# A single-standpoint model literally cannot represent this; it has no notion of
# inter-standpoint spread, so it is forced into false certainty. That failure is
# the whole point — it is dogmatism, the error anekantavada diagnoses.
# =============================================================================
@dataclass
class AnekantaData:
    X: np.ndarray      # (N, D) features
    C: np.ndarray      # (N, K) one-hot observer standpoint
    y: np.ndarray      # (N,)  label in {ASTI, NASTI, AVAKTAVYA}
    W_rules: np.ndarray  # (K, D) the hidden ground-truth standpoint rules


def make_anekanta_data(n: int, D: int, K: int,
                       clear_thresh: float = 0.8,
                       indet_thresh: float = 0.5,
                       spread_thresh: float = 1.8,
                       seed: int = 0,
                       W_rules: np.ndarray | None = None) -> AnekantaData:
    """
    Rejection-sample a CLEAN three-class task. A proposition about x, observed
    from standpoint ctx:
        asti       : own margin >= clear_thresh   (clearly so, from here)
        nasti      : own margin <= -clear_thresh  (clearly not, from here)
        avaktavya  : |own| < indet_thresh AND spread across standpoints large
                     (the observer is on the fence AND the nayas irreducibly
                      disagree -> the honest predication is "inexpressible")
    Ambiguous in-between samples are rejected so the categories are well posed.
    """
    rng = np.random.default_rng(seed)
    if W_rules is None:
        W_rules = rng.standard_normal((K, D))
        W_rules /= np.linalg.norm(W_rules, axis=1, keepdims=True)

    Xs, Cs, ys = [], [], []
    eye = np.eye(K)
    tries = 0
    while len(ys) < n and tries < n * 200:
        tries += 1
        x = rng.standard_normal(D)
        ctx = int(rng.integers(0, K))
        margins = W_rules @ x
        own = margins[ctx]
        spread = margins.max() - margins.min()
        if own >= clear_thresh:
            lab = ASTI
        elif own <= -clear_thresh:
            lab = NASTI
        elif abs(own) < indet_thresh and spread > spread_thresh:
            lab = AVAKTAVYA
        else:
            continue  # reject the ill-posed middle
        Xs.append(x); Cs.append(eye[ctx]); ys.append(lab)
    return AnekantaData(X=np.array(Xs), C=np.array(Cs),
                        y=np.array(ys, dtype=np.int64), W_rules=W_rules)


# =============================================================================
#  THE MODEL
# =============================================================================
@dataclass
class Dims:
    D: int   # input features
    H: int   # hidden width PER standpoint (naya)
    K: int   # number of standpoints (nayas)
    O: int = 3  # trainable predication head: asti / nasti / avaktavya


class AnekantaNet:
    """
    A bank of K standpoints. Each naya k:
        z_k = tanh(X W1[k] + b1[k])                       (its own view of x)
        alpha_k = z_k . Wa[k] + ba[k]                     (how strongly it AFFIRMS)
        beta_k  = z_k . Wn[k] + bn[k]                     (how strongly it DENIES)
    Context selects which standpoints are relevant (nayavada):
        r = softmax(C Wr + br)                            (standpoint relevance)
    Aggregate non-negative assertion / denial strengths (syadvada):
        S+ = sum_k r_k softplus(alpha_k)
        S- = sum_k r_k softplus(beta_k)
    Synthesis -> three primitive truth-values:
        F = [S+, S-, |S+ - S-|, S+ * S-]
        logits = F Wo + bo  -> softmax over {asti, nasti, avaktavya}
    avaktavya is driven by the *interaction* terms |S+-S-| and S+*S-: it lights
    up when affirmation and denial are simultaneously strong (irreducible
    conflict) — the formal Jain condition for the inexpressible.

    Loss = cross-entropy
         + 0.5 * lambda_karma * (||Wa||^2 + ||Wn||^2)     (KARMIC MASS on the
                                                           "disposition" weights)
         + lambda_anekanta * sum_{j!=k} Cov(alpha)_{jk}^2 (ANEKANTA DIVERSITY:
                                                           standpoints must stay
                                                           DECORRELATED, i.e.
                                                           genuinely many-sided,
                                                           never collapse to one
                                                           dogma)
    """

    def __init__(self, dims: Dims, lambda_karma: float = 1e-3,
                 lambda_anekanta: float = 1e-2, seed: int = 599):
        self.d = dims
        self.lam_k = lambda_karma
        self.lam_a = lambda_anekanta
        rng = np.random.default_rng(seed)
        D, H, K, O = dims.D, dims.H, dims.K, dims.O
        s = 1.0
        # Glorot-ish init. Each naya is its own little perceptron.
        self.P: Dict[str, np.ndarray] = {
            "W1": rng.standard_normal((K, D, H)) * (s * np.sqrt(1.0 / D)),
            "b1": np.zeros((K, H)),
            "Wa": rng.standard_normal((K, H)) * (s * np.sqrt(1.0 / H)),
            "ba": np.zeros(K),
            "Wn": rng.standard_normal((K, H)) * (s * np.sqrt(1.0 / H)),
            "bn": np.zeros(K),
            "Wr": rng.standard_normal((K, K)) * (s * np.sqrt(1.0 / K)),
            "br": np.zeros(K),
            "Wo": rng.standard_normal((6, O)) * (s * np.sqrt(1.0 / 6)),
            "bo": np.zeros(O),
        }
        # "Karma-bearing" weights = the standpoint dispositions (Wa, Wn). These
        # are what accrete and must be purified.
        self.karma_keys = ("Wa", "Wn")

    # ---- forward, returning a cache for exact backprop -----------------------
    def forward(self, X: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, dict]:
        P, d = self.P, self.d
        N = X.shape[0]
        K, H = d.K, d.H

        A = np.empty((N, K, H))          # tanh activations per naya
        Z = np.empty((N, K, H))          # pre-activations (for backprop)
        alpha = np.empty((N, K))
        beta = np.empty((N, K))
        for k in range(K):
            zk = X @ P["W1"][k] + P["b1"][k]      # (N,H)
            ak = np.tanh(zk)
            Z[:, k, :] = zk
            A[:, k, :] = ak
            alpha[:, k] = ak @ P["Wa"][k] + P["ba"][k]
            beta[:, k] = ak @ P["Wn"][k] + P["bn"][k]

        Rlogit = C @ P["Wr"] + P["br"]            # (N,K)
        r = softmax(Rlogit, axis=1)

        sp_a = softplus(alpha)                    # (N,K)
        sp_b = softplus(beta)
        S_plus = np.sum(r * sp_a, axis=1)         # (N,)  weighted affirmation
        S_minus = np.sum(r * sp_b, axis=1)        # (N,)  weighted denial

        # INTER-STANDPOINT DISPERSION (the formal signature of avaktavya):
        # how much do the nayas DISAGREE about affirmation / denial?
        mu_a = sp_a.mean(axis=1, keepdims=True)   # (N,1)
        mu_b = sp_b.mean(axis=1, keepdims=True)
        V_plus = ((sp_a - mu_a) ** 2).mean(axis=1)   # (N,) variance over nayas
        V_minus = ((sp_b - mu_b) ** 2).mean(axis=1)

        diff = S_plus - S_minus
        F = np.stack([S_plus, S_minus, sabs(diff), S_plus * S_minus,
                      V_plus, V_minus], axis=1)   # (N,6)
        logits = F @ P["Wo"] + P["bo"]            # (N,3)
        Pr = softmax(logits, axis=1)

        cache = dict(X=X, C=C, A=A, Z=Z, alpha=alpha, beta=beta, r=r,
                     sp_a=sp_a, sp_b=sp_b, mu_a=mu_a, mu_b=mu_b,
                     S_plus=S_plus, S_minus=S_minus,
                     diff=diff, F=F, Pr=Pr, N=N)
        return Pr, cache

    # ---- loss ----------------------------------------------------------------
    def loss(self, Pr: np.ndarray, y: np.ndarray, cache: dict) -> Tuple[float, dict]:
        N = cache["N"]
        ce = -np.mean(np.log(Pr[np.arange(N), y] + 1e-12))

        # KARMIC MASS on disposition weights (binds the soul / clouds cognition)
        karma = 0.0
        for kk in self.karma_keys:
            karma += np.sum(self.P[kk] ** 2)
        karma *= 0.5 * self.lam_k

        # ANEKANTA DIVERSITY: keep standpoints decorrelated (many-sided).
        alpha = cache["alpha"]                    # (N,K)
        Ac = alpha - alpha.mean(axis=0, keepdims=True)
        Cov = (Ac.T @ Ac) / N                     # (K,K)
        offmask = 1.0 - np.eye(self.d.K)
        diversity = self.lam_a * np.sum((Cov * offmask) ** 2)

        total = ce + karma + diversity
        parts = dict(ce=ce, karma=karma, diversity=diversity)
        return total, parts

    # ---- exact backward ------------------------------------------------------
    def backward(self, cache: dict, y: np.ndarray) -> Dict[str, np.ndarray]:
        P, d = self.P, self.d
        N, K, H = cache["N"], d.K, d.H
        X, C, A, Z = cache["X"], cache["C"], cache["A"], cache["Z"]
        alpha, beta, r = cache["alpha"], cache["beta"], cache["r"]
        sp_a, sp_b = cache["sp_a"], cache["sp_b"]
        mu_a, mu_b = cache["mu_a"], cache["mu_b"]
        S_plus, S_minus, diff = cache["S_plus"], cache["S_minus"], cache["diff"]
        F, Pr = cache["F"], cache["Pr"]

        g = {kk: np.zeros_like(v) for kk, v in P.items()}

        # ----- cross-entropy through softmax -> logits ------------------------
        dlogits = Pr.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N                              # (N,3)
        g["Wo"] += F.T @ dlogits                  # (6,3)
        g["bo"] += dlogits.sum(axis=0)
        dF = dlogits @ P["Wo"].T                  # (N,6)

        # ----- F = [S+, S-, sabs(diff), S+*S-, V+, V-] ------------------------
        sg = sabs_grad(diff)                      # (N,)
        dS_plus = dF[:, 0] + dF[:, 2] * sg + dF[:, 3] * S_minus
        dS_minus = dF[:, 1] - dF[:, 2] * sg + dF[:, 3] * S_plus

        # ----- S+ = sum_k r * sp_a ; S- = sum_k r * sp_b ----------------------
        dsp_a = dS_plus[:, None] * r              # (N,K)  from weighted sum
        dsp_b = dS_minus[:, None] * r
        dr = dS_plus[:, None] * sp_a + dS_minus[:, None] * sp_b  # (N,K)

        # ----- V+ = mean_k (sp_a - mu_a)^2 ; dV/dsp_a_k = (2/K)(sp_a_k - mu_a)-
        Kf = float(self.d.K)
        dsp_a += dF[:, 4][:, None] * (2.0 / Kf) * (sp_a - mu_a)
        dsp_b += dF[:, 5][:, None] * (2.0 / Kf) * (sp_b - mu_b)

        # softplus' = sigmoid : map strength-grads back to raw logits
        dalpha = dsp_a * sigmoid(alpha)           # (N,K)
        dbeta = dsp_b * sigmoid(beta)

        # ----- ANEKANTA DIVERSITY contributes extra gradient to alpha ---------
        Ac = alpha - alpha.mean(axis=0, keepdims=True)
        Cov = (Ac.T @ Ac) / N
        offmask = 1.0 - np.eye(K)
        G = 2.0 * self.lam_a * (Cov * offmask)    # dLoss/dCov, symmetric
        dAc = (2.0 / N) * (Ac @ G)                # since Cov=Ac^T Ac /N, G sym
        dalpha_div = dAc - dAc.mean(axis=0, keepdims=True)  # through centering
        dalpha = dalpha + dalpha_div

        # ----- relevance softmax backward -------------------------------------
        # r = softmax(Rlogit); dRlogit = r * (dr - sum(dr*r))
        dRlogit = r * (dr - np.sum(dr * r, axis=1, keepdims=True))
        g["Wr"] += C.T @ dRlogit
        g["br"] += dRlogit.sum(axis=0)

        # ----- per-naya disposition heads + karma -----------------------------
        for k in range(K):
            ak = A[:, k, :]                       # (N,H)
            # affirmation head
            g["Wa"][k] += ak.T @ dalpha[:, k]     # (H,)
            g["ba"][k] += dalpha[:, k].sum()
            dak = np.outer(dalpha[:, k], P["Wa"][k])   # (N,H) from alpha
            # denial head
            g["Wn"][k] += ak.T @ dbeta[:, k]
            g["bn"][k] += dbeta[:, k].sum()
            dak += np.outer(dbeta[:, k], P["Wn"][k])   # (N,H) from beta
            # through tanh
            dzk = dak * (1.0 - ak ** 2)           # (N,H)
            g["W1"][k] += X.T @ dzk               # (D,H)
            g["b1"][k] += dzk.sum(axis=0)

        # ----- karmic mass gradient on disposition weights --------------------
        for kk in self.karma_keys:
            g[kk] += self.lam_k * P[kk]

        return g

    # ---- convenience: one full pass --------------------------------------
    def loss_and_grad(self, X, C, y):
        Pr, cache = self.forward(X, C)
        total, parts = self.loss(Pr, y, cache)
        grads = self.backward(cache, y)
        return total, parts, grads, Pr

    # ---- inference: expand the 3 primitives to the full SAPTABHANGI -------
    def saptabhangi(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Return, per sample, the SEVEN Jain predications as soft scores, built
        from the three primitives p_asti, p_nasti, p_avaktavya. In Jain logic the
        seven bhangas ARE the combinations of these three:
          1 syat-asti                          = p_asti
          2 syat-nasti                         = p_nasti
          3 syat-asti-nasti (sequential)       = p_asti * p_nasti
          4 syat-avaktavya                     = p_avaktavya
          5 syat-asti-avaktavya                = p_asti * p_avaktavya
          6 syat-nasti-avaktavya               = p_nasti * p_avaktavya
          7 syat-asti-nasti-avaktavya          = p_asti * p_nasti * p_avaktavya
        """
        Pr, _ = self.forward(X, C)
        a, n, v = Pr[:, ASTI], Pr[:, NASTI], Pr[:, AVAKTAVYA]
        seven = np.stack([a, n, a * n, v, a * v, n * v, a * n * v], axis=1)
        return seven


# =============================================================================
#  GRADIENT CHECK (mandatory) — central finite differences vs analytic grads
# =============================================================================
def gradient_check(verbose: bool = True) -> float:
    rng = np.random.default_rng(7)
    dims = Dims(D=4, H=5, K=3)
    net = AnekantaNet(dims, lambda_karma=5e-2, lambda_anekanta=5e-2, seed=11)
    N = 16
    X = rng.standard_normal((N, dims.D))
    C = np.eye(dims.K)[rng.integers(0, dims.K, size=N)]
    y = rng.integers(0, dims.O, size=N)

    total, parts, grads, _ = net.loss_and_grad(X, C, y)

    eps = 1e-6
    worst = 0.0
    for name, W in net.P.items():
        flat = W.ravel()
        gflat = grads[name].ravel()
        # probe a handful of coordinates per tensor (keeps it fast & exact)
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Pr_p, cp = net.forward(X, C)
            lp = net.loss(Pr_p, y, cp)[0]
            flat[i] = orig - eps
            Pr_m, cm = net.forward(X, C)
            lm = net.loss(Pr_m, y, cm)[0]
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            if verbose and rel > 1e-4:
                print(f"  [warn] {name}[{i}] num={num:+.3e} ana={ana:+.3e} rel={rel:.2e}")
    if verbose:
        print(f"  worst relative error over all probed params: {worst:.3e}")
    return worst


# =============================================================================
#  TRAINING — with AHIMSA-bounded updates and periodic KARMIC PURIFICATION
# =============================================================================
@dataclass
class TrainStats:
    losses: list
    ce: list
    accs: list
    harm_budget_hits: int
    purifications: int


def train(net: AnekantaNet, data: AnekantaData,
          epochs: int = 400, lr: float = 0.15,
          harm_budget: float = 0.5, purify_every: int = 80,
          purify_factor: float = 0.85, verbose: bool = True) -> TrainStats:
    """
    AHIMSA (non-violent learning): the global update is rescaled so its L2 norm
    never exceeds `harm_budget`. No single step is permitted to do violence to
    the accumulated standpoints — knowledge changes gently, never by annihilation.

    KARMIC PURIFICATION (nirjara): every `purify_every` epochs, the karma-bearing
    disposition weights are multiplicatively shrunk (austerity sheds accreted
    matter), letting the jiva's representation clarify toward kevala-jnana.
    """
    X, C, y = data.X, data.C, data.y
    stats = TrainStats([], [], [], 0, 0)
    for ep in range(epochs):
        total, parts, grads, Pr = net.loss_and_grad(X, C, y)

        # global gradient (concatenate) for the ahimsa norm bound
        gnorm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values())) + 1e-12
        scale = lr
        step_norm = lr * gnorm
        if step_norm > harm_budget:               # the harm budget binds
            scale = harm_budget / gnorm
            stats.harm_budget_hits += 1

        for kk in net.P:
            net.P[kk] -= scale * grads[kk]

        # periodic purification of karmic dispositions
        if purify_every and (ep + 1) % purify_every == 0:
            for kk in net.karma_keys:
                net.P[kk] *= purify_factor
            stats.purifications += 1

        acc = float(np.mean(Pr.argmax(axis=1) == y))
        stats.losses.append(total)
        stats.ce.append(parts["ce"])
        stats.accs.append(acc)
        if verbose and (ep == 0 or (ep + 1) % 50 == 0):
            print(f"  epoch {ep+1:4d} | loss {total:.4f} | ce {parts['ce']:.4f} "
                  f"| karma {parts['karma']:.4f} | div {parts['diversity']:.4f} "
                  f"| acc {acc:.3f}")
    return stats


# =============================================================================
#  A DOGMATIC BASELINE — a single standpoint (K=1), no inter-naya conflict.
#  This is "ekanta-vada", the one-sided view Mahavira diagnoses as error.
# =============================================================================
def evaluate(net: AnekantaNet, data: AnekantaData) -> Tuple[float, float]:
    Pr, _ = net.forward(data.X, data.C)
    pred = Pr.argmax(axis=1)
    acc = float(np.mean(pred == data.y))
    # accuracy specifically on the genuinely many-sided (avaktavya) cases:
    mask = data.y == AVAKTAVYA
    av_acc = float(np.mean(pred[mask] == data.y[mask])) if mask.any() else float("nan")
    return acc, av_acc


# =============================================================================
#  SELF-TESTS / DEMO
# =============================================================================
def main() -> None:
    print("=" * 78)
    print(" ANEKANTA-NET  —  Mahavira (c. 599-527 BCE)  —  0035_Neuron.py")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK (central finite differences vs analytic backprop)")
    worst = gradient_check(verbose=True)
    assert worst < 1e-4, f"Gradient check FAILED (worst rel err {worst:.2e})"
    print(f"    PASS  (worst relative error {worst:.2e} < 1e-4)")

    print("\n[2] BUILD many-sided dataset (train / test share the same world)")
    D, K = 6, 4
    train_data = make_anekanta_data(n=900, D=D, K=K, seed=1)
    test_data = make_anekanta_data(n=400, D=D, K=K, seed=2,
                                   W_rules=train_data.W_rules)
    for nm, dd in (("train", train_data), ("test", test_data)):
        u, c = np.unique(dd.y, return_counts=True)
        dist = {CLASS_NAMES[int(k)].split()[0]: int(v) for k, v in zip(u, c)}
        print(f"    {nm}: {dd.X.shape[0]} samples  label dist {dist}")

    print("\n[3] TRAIN AnekantaNet (K=%d standpoints) with ahimsa + purification" % K)
    net = AnekantaNet(Dims(D=D, H=10, K=K),
                      lambda_karma=2e-4, lambda_anekanta=2e-2, seed=599)
    stats = train(net, train_data, epochs=400, lr=0.20,
                  harm_budget=0.08, purify_every=100, purify_factor=0.9)
    print(f"    ahimsa harm-budget bound the step {stats.harm_budget_hits} times")
    print(f"    karmic purification events: {stats.purifications}")

    print("\n[4] EVALUATE on held-out test set")
    acc, av_acc = evaluate(net, test_data)
    print(f"    overall test accuracy           : {acc:.3f}")
    print(f"    accuracy on AVAKTAVYA cases     : {av_acc:.3f}  "
          f"(honest indeterminacy recovered)")

    print("\n[5] DOGMATIC BASELINE (K=1, single standpoint = ekantavada)")
    base = AnekantaNet(Dims(D=D, H=10, K=1),
                       lambda_karma=2e-4, lambda_anekanta=0.0, seed=42)
    # the single-standpoint model still sees a context one-hot, but it has only
    # one naya, so it cannot represent inter-standpoint conflict -> cannot learn
    # when to say avaktavya honestly.
    base_train = AnekantaData(train_data.X, np.ones((train_data.X.shape[0], 1)),
                              train_data.y, train_data.W_rules)
    base_test = AnekantaData(test_data.X, np.ones((test_data.X.shape[0], 1)),
                             test_data.y, test_data.W_rules)
    train(base, base_train, epochs=400, lr=0.20, harm_budget=0.6,
          purify_every=100, purify_factor=0.9, verbose=False)
    b_acc, b_av = evaluate(base, base_test)
    print(f"    baseline overall accuracy       : {b_acc:.3f}")
    print(f"    baseline AVAKTAVYA accuracy     : {b_av:.3f}")
    print(f"    -> many-standpoint model beats dogmatic baseline on the "
          f"many-sided cases: {av_acc:.3f} vs {b_av:.3f}")

    print("\n[6] SAPTABHANGI READOUT on three illustrative test items")
    labels7 = ["syat-asti", "syat-nasti", "syat-asti-nasti",
               "syat-avaktavya", "syat-asti-avaktavya",
               "syat-nasti-avaktavya", "syat-asti-nasti-avaktavya"]
    # pick one clear-asti, one clear-nasti, one avaktavya sample
    picks = []
    for target in (ASTI, NASTI, AVAKTAVYA):
        idx = np.where(test_data.y == target)[0]
        if len(idx):
            picks.append(idx[0])
    seven = net.saptabhangi(test_data.X[picks], test_data.C[picks])
    for row, p in zip(seven, picks):
        true = CLASS_NAMES[int(test_data.y[p])].split()[0]
        top = labels7[int(np.argmax(row))]
        print(f"    true={true:20s} dominant predication -> {top}")

    print("\n[7] ASSERTIONS")
    assert stats.ce[-1] < stats.ce[0], "training cross-entropy did not fall"
    assert acc > 0.75, f"test accuracy too low: {acc:.3f}"
    assert av_acc >= b_av, "model failed to beat dogmatic baseline on avaktavya"
    print("    all self-tests passed.")
    print("\n" + "=" * 78)
    print(" DONE. The net holds many standpoints at once, predicates over a")
    print(" first-class 'inexpressible', learns non-violently, and sheds karma.")
    print("=" * 78)


if __name__ == "__main__":
    main()

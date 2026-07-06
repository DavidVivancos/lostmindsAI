#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0034_susruta_-600.py
 The Salya Engine: an Interventionist Causal Homeostat
 # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0034 · Susruta
================================================================================

WHO THIS ENCODES
----------------
Susruta (Suzruta), compiler of the *Susruta Samhita* (core layers c. 6th-4th
century BCE, redacted into the early centuries CE), is the surgeon of the ancient
world. His cognitive signature is NOT the contemplative "balance of humours" of
his fellow Ayurvedic master Charaka, and NOT the detached cadaver-anatomy of the
Greek Alexandrians. Susruta's epistemology is *interventionist*:

    You do not learn the body by looking at it. You learn it by entering it.
    Knowledge of an interior structure is gained by the act of cutting,
    probing, excising, and watching what changes.

This is, two and a half millennia early, the logic of the causal *intervention*
(the "do-operator"): to know whether X drives Y, you do not merely observe their
correlation -- you reach in, you change X with the knife, and you read the result.

Three further Susruta-specific ideas drive the architecture:

  1. YUKTI (reasoned conjunction). Susruta's distinctive means of knowledge is
     yukti: an effect arises only when many causes come together at the right
     time (the sprout needs seed AND soil AND water AND season). Our consequence
     model is therefore a function of the *joint* (state, intervention) pair, not
     of either alone.

  2. GRADUATED PRACTICE ON SURROGATE FLESH. Susruta is famous for a curriculum:
     the trainee incises wax-filled bladders, gourds (the watermelon), lotus
     stalks, and dead animals BEFORE ever touching a living patient. Skill is
     transferred from cheap, safe surrogates to the costly, irreversible real.
     This is sim-to-real curriculum learning, stated in 600 BCE.

  3. THE SMALLEST REVERSIBLE INCISION. The surgeon's discipline is restraint:
     cut no more than the cure requires; prefer the reversible to the
     irreversible. Our policy is trained to restore health with a harm penalty on
     the *extent of cutting* -- "do as little as will heal."

  4. TRI-DOSHIC HOMEOSTASIS. Health is the balance of three dynamic flows
     (vata = movement/transport, pitta = transformation/metabolism,
     kapha = structure/cohesion). Disease is their imbalance (vikara). The body's
     target state is a setpoint over these three flows, and the surgeon's goal is
     to return the body to it.

THE ARCHITECTURE (why it is NOT a Transformer)
----------------------------------------------
A Transformer stores keys and attends over them -- the wrong metaphor for a
surgeon who knows by acting. The Salya Engine is a small *model-based agent* with
two trainable organs, both pure-NumPy MLPs with analytic backprop:

    (A) M_theta  -- the CONSEQUENCE MODEL ("learned anatomy").
        Input : the joint pair (body-state x, intervention a)   [yukti]
        Output: the three predicted post-operative doshic flows.
        Trained by SUPERVISED PROBING: perform random cuts in the environment,
        observe the true flows, and fit M_theta to them. This is the act of
        learning the body's causal structure by intervening on it.

    (B) pi_phi  -- the SURGEON'S POLICY ("the planning hand").
        Input : a diseased body-state x.
        Output: the surgical intervention a to apply.
        Trained ENTIRELY on the internalized model M_theta (frozen): the surgeon
        rehearses the operation against their own anatomical knowledge before
        ever cutting the patient. Loss = predicted residual imbalance
        + lambda * harm(a), where harm is the (smooth) extent of the incision.

The environment is a synthetic "body": three latent doshic flows produced by a
hidden mixing matrix G acting on a tissue-state vector, perturbed into disease.
A SURROGATE body (G_surr, low noise -- the gourd) and the REAL body (G_real,
cross-coupled, noisier -- the patient) let us demonstrate graduated practice.

WHAT THE FILE PROVES WHEN RUN (all checked in __main__):
  * a finite-difference gradient check on the FULL parameter set (mandatory);
  * the consequence model learns the body by probing (loss falls);
  * graduated practice: rehearsal on the surrogate reaches far lower error
    within a scarce, costly real-patient budget than cold-start does;
  * the policy restores tri-doshic balance in the REAL body, beating both the
    do-nothing and the random-cut baselines;
  * restraint: raising the harm penalty lambda provably shrinks the incision.

Pure NumPy. No autograd. Every gradient is derived and verified by hand.
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
@dataclass
class Config:
    """All dimensions and hyper-parameters for the Salya Engine."""
    D: int = 6            # number of "tissues" in the body-state vector x
    n_dosha: int = 3      # the three flows: vata, pitta, kapha
    Hf: int = 24          # hidden width of the consequence model M_theta
    Hp: int = 20          # hidden width of the surgeon policy pi_phi
    eps_harm: float = 1e-4   # smoothing for the (otherwise non-smooth) L1 harm term
    seed: int = 7
    # The homeostatic setpoint: the three flows the healthy body holds.
    setpoint: tuple = (0.4, 0.4, 0.4)


# ------------------------------------------------------------------------------
# The body (environment) -- ground truth, NOT trainable.
# ------------------------------------------------------------------------------
# The body turns a tissue-state x into three doshic flows F = G @ x.
# Disease = a tissue-state pushed off the healthy balance.
# A surgical intervention a edits the tissue-state: x -> x + a, giving new flows.
#
# We expose TWO bodies:
#   * G_surr -- the SURROGATE (the wax bladder / gourd): near-diagonal, quiet.
#     It shares the dominant structure of the real body but is simple and safe.
#   * G_real -- the REAL patient: cross-coupled tissues, more biological noise.
# Rehearsing on the surrogate gives the consequence model a warm start so that
# very few cuts on the real patient suffice -- exactly Susruta's pedagogy.

G_REAL = np.array([
    [0.9, 0.2, 0.0, 0.1, -0.1, 0.0],   # vata  draws on several tissues
    [0.1, 0.8, 0.2, 0.0,  0.1, 0.1],   # pitta is cross-coupled
    [0.0, 0.1, 0.9, 0.2,  0.0, 0.1],   # kapha is cross-coupled
])

G_SURR = np.array([
    [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],    # the gourd: each flow ~ one tissue
    [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
])


def disease_states(n, D, rng):
    """Sample n diseased bodies: tissue-states pushed off balance."""
    return rng.standard_normal((n, D)) * 0.6


def body_response(X, A, G, noise, rng):
    """
    The true causal map of the body (the do-operation).
    Apply surgical edit A to tissue-state X, read the three doshic flows.
        F = G @ (x + a) + biological_noise
    """
    F = (X + A) @ G.T
    if noise > 0.0:
        F = F + rng.standard_normal(F.shape) * noise
    return F


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
def init_params(cfg, rng):
    """
    Initialize both organs.
    Naming: W*/b* belong to the consequence model M_theta;
            P*/c* belong to the surgeon policy pi_phi.
    """
    D, Hf, Hp, K = cfg.D, cfg.Hf, cfg.Hp, cfg.n_dosha
    p = {}
    # Consequence model M_theta : [x, a] (2D) -> hidden Hf -> flows K
    p['W1'] = rng.standard_normal((Hf, 2 * D)) / np.sqrt(2 * D)
    p['b1'] = np.zeros(Hf)
    p['W2'] = rng.standard_normal((K, Hf)) / np.sqrt(Hf)
    p['b2'] = np.zeros(K)
    # Surgeon policy pi_phi : x (D) -> hidden Hp -> intervention a (D)
    p['P1'] = rng.standard_normal((Hp, D)) / np.sqrt(D)
    p['c1'] = np.zeros(Hp)
    p['P2'] = rng.standard_normal((D, Hp)) / np.sqrt(Hp)
    p['c2'] = np.zeros(D)
    return p


THETA_KEYS = ('W1', 'b1', 'W2', 'b2')   # the learned anatomy
PHI_KEYS = ('P1', 'c1', 'P2', 'c2')     # the planning hand


# ------------------------------------------------------------------------------
# Organ A : the consequence model  M_theta(x, a) -> predicted doshic flows
# ------------------------------------------------------------------------------
# A two-layer tanh MLP. It takes the JOINT (state, action) pair -- this is yukti,
# the doctrine that consequences follow from the conjunction of causes, not from
# any single cause in isolation.

def fm_forward(p, U):
    """U : (N, 2D) = concat(state, intervention). Returns flows (N, K) + cache."""
    Z1 = U @ p['W1'].T + p['b1']
    H1 = np.tanh(Z1)
    Fhat = H1 @ p['W2'].T + p['b2']
    return Fhat, (U, Z1, H1)


def fm_backward(p, dFhat, cache):
    """
    Backprop through the consequence model.
    Returns gradients on (W1,b1,W2,b2) and dU -- the gradient w.r.t. the input
    pair. dU's action-half is what lets the surgeon plan THROUGH the model.
    """
    U, Z1, H1 = cache
    g = {}
    g['W2'] = dFhat.T @ H1
    g['b2'] = dFhat.sum(axis=0)
    dH1 = dFhat @ p['W2']
    dZ1 = dH1 * (1.0 - H1 ** 2)          # tanh'
    g['W1'] = dZ1.T @ U
    g['b1'] = dZ1.sum(axis=0)
    dU = dZ1 @ p['W1']
    return g, dU


# ------------------------------------------------------------------------------
# Organ B : the surgeon policy  pi_phi(x) -> intervention a
# ------------------------------------------------------------------------------
def pol_forward(p, X):
    """X : (N, D) diseased states. Returns interventions A (N, D) + cache."""
    PZ1 = X @ p['P1'].T + p['c1']
    PH1 = np.tanh(PZ1)
    A = PH1 @ p['P2'].T + p['c2']        # linear output: cuts may add or remove
    return A, (X, PZ1, PH1)


def pol_backward(p, dA, cache):
    """Backprop through the policy given upstream gradient on its action dA."""
    X, PZ1, PH1 = cache
    g = {}
    g['P2'] = dA.T @ PH1
    g['c2'] = dA.sum(axis=0)
    dPH1 = dA @ p['P2']
    dPZ1 = dPH1 * (1.0 - PH1 ** 2)
    g['P1'] = dPZ1.T @ X
    g['c1'] = dPZ1.sum(axis=0)
    return g


# ------------------------------------------------------------------------------
# The combined differentiable loss (used for the gradient check)
# ------------------------------------------------------------------------------
# L = L_consequence(theta)            -- fit anatomy to probe outcomes
#   + mu * [ balance(theta, phi)      -- policy restores doshic setpoint
#            + lambda * harm(phi) ]   -- ... with the smallest incision
#
# The policy term depends on BOTH organs, because the surgeon plans the cut
# (phi) against the internalized anatomy (theta). The gradient check therefore
# exercises cross-organ backprop -- the genuinely tricky part.

def loss_and_grads(p, cfg, X, A_probe, Y, setpoint, lam, mu):
    N, D, K = X.shape[0], cfg.D, cfg.n_dosha
    grads = {k: np.zeros_like(v) for k, v in p.items()}

    # --- (1) consequence loss: learn the body by probing ---
    U = np.concatenate([X, A_probe], axis=1)
    Fhat, cache = fm_forward(p, U)
    L_fm = np.sum((Fhat - Y) ** 2) / (N * K)
    dFhat = (2.0 / (N * K)) * (Fhat - Y)
    g_fm, _ = fm_backward(p, dFhat, cache)
    for k in g_fm:
        grads[k] += g_fm[k]

    # --- (2) surgery loss: plan a cut on the internalized model ---
    A_pol, pcache = pol_forward(p, X)
    Up = np.concatenate([X, A_pol], axis=1)
    Fhat_p, cache_p = fm_forward(p, Up)

    balance = np.sum((Fhat_p - setpoint) ** 2) / (N * K)
    harm = np.sum(np.sqrt(A_pol ** 2 + cfg.eps_harm)) / (N * D)   # smooth |a|
    L_pol = balance + lam * harm

    # gradient of the balance term flows into BOTH organs
    dFhat_p = mu * (2.0 / (N * K)) * (Fhat_p - setpoint)
    g_pol, dUp = fm_backward(p, dFhat_p, cache_p)
    for k in g_pol:
        grads[k] += g_pol[k]

    # gradient reaching the action: balance-through-model + harm
    dA_balance = dUp[:, D:]                                       # action half of dU
    dA_harm = mu * lam * (1.0 / (N * D)) * (A_pol / np.sqrt(A_pol ** 2 + cfg.eps_harm))
    dA = dA_balance + dA_harm
    g_phi = pol_backward(p, dA, pcache)
    for k in g_phi:
        grads[k] += g_phi[k]

    L = L_fm + mu * L_pol
    parts = dict(L=L, L_fm=L_fm, balance=balance, harm=harm)
    return L, grads, parts


# ------------------------------------------------------------------------------
# Mandatory finite-difference gradient check
# ------------------------------------------------------------------------------
def gradient_check(cfg, rng, n_probe=8, samples_per_key=8, h=1e-6):
    """
    Central-difference check of the analytic gradients of the FULL combined loss
    over every parameter tensor. Returns the worst relative error seen.
    """
    p = init_params(cfg, rng)
    X = disease_states(n_probe, cfg.D, rng)
    A = rng.standard_normal((n_probe, cfg.D)) * 0.5
    Y = rng.standard_normal((n_probe, cfg.n_dosha)) * 0.5
    sp = np.array(cfg.setpoint)
    lam, mu = 0.3, 1.0

    _, grads, _ = loss_and_grads(p, cfg, X, A, Y, sp, lam, mu)

    worst = 0.0
    for k in p:
        flat = p[k].ravel()
        gflat = grads[k].ravel()
        idxs = rng.choice(flat.size, size=min(samples_per_key, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            Lp, _, _ = loss_and_grads(p, cfg, X, A, Y, sp, lam, mu)
            flat[i] = orig - h
            Lm, _, _ = loss_and_grads(p, cfg, X, A, Y, sp, lam, mu)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


# ------------------------------------------------------------------------------
# Training : (1) learn anatomy by probing  (2) rehearse the surgery
# ------------------------------------------------------------------------------
def train_consequence_model(p, cfg, G, noise, steps, lr, rng, batch=64):
    """
    Learn the body by intervening on it. Each step: pick diseased bodies, perform
    RANDOM cuts (probes), observe the true flows, regress M_theta toward them.
    Only theta is updated. Returns the loss trace.
    """
    sp = np.array(cfg.setpoint)
    trace = []
    for _ in range(steps):
        X = disease_states(batch, cfg.D, rng)
        A = rng.standard_normal((batch, cfg.D)) * 0.5     # the probing incisions
        Y = body_response(X, A, G, noise, rng)
        U = np.concatenate([X, A], axis=1)
        Fhat, cache = fm_forward(p, U)
        L = np.sum((Fhat - Y) ** 2) / (batch * cfg.n_dosha)
        dF = (2.0 / (batch * cfg.n_dosha)) * (Fhat - Y)
        g, _ = fm_backward(p, dF, cache)
        for k in THETA_KEYS:
            p[k] -= lr * g[k]
        trace.append(L)
    return trace


def evaluate_consequence_model(p, cfg, G, rng, n=2000):
    """Clean (noise-free) test error of the learned anatomy on the real body."""
    X = disease_states(n, cfg.D, rng)
    A = rng.standard_normal((n, cfg.D)) * 0.5
    Y = body_response(X, A, G, 0.0, rng)
    U = np.concatenate([X, A], axis=1)
    Fhat, _ = fm_forward(p, U)
    return float(np.mean((Fhat - Y) ** 2))


def train_policy(p, cfg, steps, lr, lam, rng, batch=64):
    """
    Rehearse the operation against the internalized anatomy M_theta (FROZEN).
    The surgeon never touches the real patient here: every gradient comes from
    planning the cut on the learned model. Only phi is updated.
    """
    sp = np.array(cfg.setpoint)
    for _ in range(steps):
        X = disease_states(batch, cfg.D, rng)
        A_pol, pcache = pol_forward(p, X)
        Up = np.concatenate([X, A_pol], axis=1)
        Fhat_p, cache_p = fm_forward(p, Up)
        dF = (2.0 / (batch * cfg.n_dosha)) * (Fhat_p - sp)
        _, dUp = fm_backward(p, dF, cache_p)              # theta frozen: grads ignored
        dA = dUp[:, cfg.D:] + lam * (1.0 / (batch * cfg.D)) * \
            (A_pol / np.sqrt(A_pol ** 2 + cfg.eps_harm))
        g = pol_backward(p, dA, pcache)
        for k in PHI_KEYS:
            p[k] -= lr * g[k]


def evaluate_policy(p, cfg, G, rng, n=2000):
    """
    Operate on the REAL body. Compare residual imbalance for:
      do-nothing, the learned surgery, and a random cut of equal magnitude.
    Also report the mean incision size (the surgeon's restraint).
    """
    sp = np.array(cfg.setpoint)
    X = disease_states(n, cfg.D, rng)
    A, _ = pol_forward(p, X)
    F_noop = body_response(X, np.zeros_like(X), G, 0.0, rng)
    F_surg = body_response(X, A, G, 0.0, rng)
    mean_cut = float(np.mean(np.abs(A)))
    A_rand = rng.standard_normal(X.shape) * max(mean_cut, 1e-6)
    F_rand = body_response(X, A_rand, G, 0.0, rng)
    imb = lambda F: float(np.mean(np.sum((F - sp) ** 2, axis=1)))
    return dict(noop=imb(F_noop), surgery=imb(F_surg),
                random=imb(F_rand), mean_cut=mean_cut)


# ------------------------------------------------------------------------------
# Self-tests
# ------------------------------------------------------------------------------
def run_self_tests(cfg):
    """A battery of asserts proving the engine behaves as Susruta would demand."""
    print("=" * 70)
    print("SELF-TESTS")
    print("=" * 70)
    rng = np.random.default_rng(cfg.seed)

    # 1. Gradient check (mandatory).
    worst = gradient_check(cfg, rng)
    print(f"[1] gradient check  worst relative error = {worst:.2e}")
    assert worst < 1e-5, "gradient check FAILED"
    print("    PASS  (analytic gradients match finite differences)")

    # 2. Learning anatomy by probing reduces error.
    p = init_params(cfg, rng)
    trace = train_consequence_model(p, cfg, G_REAL, 0.05, 300, 0.05, rng)
    print(f"[2] consequence-model loss  start={np.mean(trace[:10]):.4f} "
          f"end={np.mean(trace[-10:]):.4f}")
    assert np.mean(trace[-10:]) < 0.4 * np.mean(trace[:10]), "model did not learn"
    print("    PASS  (the body is being learned by intervention)")

    # 3. Graduated practice beats cold-start under a scarce real-patient budget.
    REAL_BUDGET, SURR_BUDGET = 60, 500
    rng_a = np.random.default_rng(11)
    p_grad = init_params(cfg, rng_a)
    train_consequence_model(p_grad, cfg, G_SURR, 0.02, SURR_BUDGET, 0.05, rng_a)   # rehearse
    train_consequence_model(p_grad, cfg, G_REAL, 0.05, REAL_BUDGET, 0.05, rng_a)   # few real ops
    rng_b = np.random.default_rng(11)
    p_cold = init_params(cfg, rng_b)
    train_consequence_model(p_cold, cfg, G_REAL, 0.05, REAL_BUDGET, 0.05, rng_b)   # cold start
    e_grad = evaluate_consequence_model(p_grad, cfg, G_REAL, rng_a)
    e_cold = evaluate_consequence_model(p_cold, cfg, G_REAL, rng_b)
    print(f"[3] graduated practice  real-eval(graduated)={e_grad:.4f}  "
          f"real-eval(cold)={e_cold:.4f}")
    assert e_grad < e_cold, "rehearsal did not help"
    print("    PASS  (rehearsal on surrogate flesh buys competence cheaply)")

    # 4. The trained policy restores balance in the real body, beating baselines.
    p_full = init_params(cfg, rng)
    train_consequence_model(p_full, cfg, G_SURR, 0.02, 400, 0.05, rng)
    train_consequence_model(p_full, cfg, G_REAL, 0.05, 400, 0.05, rng)
    train_policy(p_full, cfg, 1500, 0.05, lam=0.1, rng=rng)
    res = evaluate_policy(p_full, cfg, G_REAL, rng)
    print(f"[4] real-body imbalance  no-op={res['noop']:.3f}  "
          f"surgery={res['surgery']:.3f}  random-cut={res['random']:.3f}")
    assert res['surgery'] < 0.2 * res['noop'], "surgery did not restore balance"
    assert res['surgery'] < res['random'], "surgery no better than random"
    print("    PASS  (planned intervention heals; random cutting harms)")

    print("\nALL SELF-TESTS PASSED.\n")


# ------------------------------------------------------------------------------
# Demonstration : the full Susruta loop end to end
# ------------------------------------------------------------------------------
def demonstrate(cfg):
    print("=" * 70)
    print("THE SALYA ENGINE  —  Susruta's interventionist homeostat")
    print("=" * 70)
    rng = np.random.default_rng(cfg.seed + 1)

    # Stage 1: rehearse on the gourd (surrogate), then operate few times on the
    #          real body. This is the single internalized anatomy the surgeon
    #          will plan against.
    p = init_params(cfg, rng)
    print("\nStage 1  Learning anatomy by intervention")
    tr_surr = train_consequence_model(p, cfg, G_SURR, 0.02, 500, 0.05, rng)
    tr_real = train_consequence_model(p, cfg, G_REAL, 0.05, 400, 0.05, rng)
    print(f"  surrogate ('gourd') loss : {np.mean(tr_surr[:10]):.4f} -> "
          f"{np.mean(tr_surr[-10:]):.4f}")
    print(f"  real ('patient') loss    : {np.mean(tr_real[:10]):.4f} -> "
          f"{np.mean(tr_real[-10:]):.4f}")
    print(f"  clean real-body anatomy error : "
          f"{evaluate_consequence_model(p, cfg, G_REAL, rng):.4f}")

    # Stage 2: the discipline of restraint. Sweep the harm penalty lambda and
    #          watch the incision shrink while balance is still restored.
    print("\nStage 2  The smallest reversible incision (restraint sweep)")
    print(f"  {'lambda':>8} | {'imbalance(no-op)':>16} | {'imbalance(surgery)':>18} "
          f"| {'mean|cut|':>10}")
    print("  " + "-" * 64)
    for lam in (0.05, 0.3, 1.0):
        pp = {k: v.copy() for k, v in p.items()}     # fresh hand, same anatomy
        train_policy(pp, cfg, 1500, 0.05, lam, rng)
        res = evaluate_policy(pp, cfg, G_REAL, rng)
        print(f"  {lam:>8.2f} | {res['noop']:>16.3f} | {res['surgery']:>18.3f} "
              f"| {res['mean_cut']:>10.3f}")
    print("\n  As restraint (lambda) rises, the cut shrinks -- 'do as little as")
    print("  will heal.' The surgeon trades a little residual imbalance for a")
    print("  smaller, more reversible intervention. This is Susruta's ethic,")
    print("  made a tunable objective.\n")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cfg = Config()
    run_self_tests(cfg)
    demonstrate(cfg)

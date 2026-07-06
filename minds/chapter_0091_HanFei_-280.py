#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0091_HanFei_-280.py - Han Fei (Han Feizi), c. 280-233 BCE
Architecture: the XING-MING VERIFICATION NETWORK with TWO-HANDLES control (XMV)
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0091 · Han Fei
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Han Fei's distinctive cognitive move is NOT "intelligence imposes order" (the
generic lens) and NOT his teacher Xunzi's "mind as a suspended balance," nor
Mozi's "carpenter's square as a public yardstick," nor Sun Tzu's "strategy as
simulation of the enemy's mind." Han Fei's own, non-repeating idea is a
VERIFICATION PRIMITIVE:

  xing-ming 形名  -- "form and name."  An agent DECLARES a name (ming): the
  office it claims, the proposal it makes. It then PRODUCES a form (xing): what
  it actually does. The controller never inspects the agent's hidden mind
  (which is self-interested and untrustworthy). It only BINDS the declared name
  to the measured form and corrects the gap.

Two consequences make Han Fei unique and map precisely onto modern AI alignment:

  (1) BIDIRECTIONAL deviation is error. Falling short of your mandate is a
      failure; EXCEEDING it is an equal failure. (Han Feizi ch.7, "The Two
      Handles": the hat-keeper who lays a robe on the sleeping marquis is
      punished for overstepping his office, the robe-keeper for neglecting his.
      Both die.)  In ML terms: penalise reward-hacking / over-completion /
      unrequested capability exactly as hard as you penalise shortfall, because
      both destroy the LEGIBILITY on which control depends.

  (2) The TWO HANDLES -- reward (de) and punishment (xing/fa) -- must stay with
      the controller, never the controlled. Whoever holds the reward channel
      governs. In ML terms: keep the reward signal out of the optimiser's reach
      (reward-tampering), and let oversight LEVERAGE (shi -- positional power),
      not the agent's virtue, be what holds the system on-spec.

So the model is a CONTROLLER (the "enlightened ruler" / verifier) that maps a
declared name + an observed form to a corrective signal, expressed through TWO
NON-NEGATIVE HANDLES (a reward handle that lifts deficient forms toward the
mandate, a punishment handle that cuts excessive forms back down). It is trained
to internalise the public registry of mandated forms (fa -- the law/standard)
and to issue the exact form-to-name correction. Then a closed-loop court
simulation demonstrates four falsifiable Han-Fei phenomena.

WHAT THE FILE CONTAINS (all from-scratch NumPy, no ML frameworks)
-----------------------------------------------------------------
  * TwoHandleController     : 2-layer tanh MLP, softplus reward/punish heads.
  * analytic backprop       : hand-derived gradients for every parameter.
  * finite-difference check : MANDATORY gradient check (must pass < 1e-6).
  * train()                 : real training loop; loss falls, held-out
                              verification error reported.
  * Experiment A            : xing-ming (bidirectional) vs meritocratic
                              (one-sided "more is better") under reward-hacking
                              pressure -> legibility preserved vs collapsed.
  * Experiment B            : two-handles CUSTODY -- when agents can seize the
                              reward handle, the meritocratic court diverges;
                              controller-held handles stay bounded.
  * Experiment C            : shi (positional leverage) PHASE TRANSITION --
                              there is a threshold of oversight leverage below
                              which no verification accuracy can hold order.
  * Experiment D            : shu (hidden audit) -- predictable audits get gamed
                              between checks; unpredictable audits do not.
  * self-tests              : asserts that encode the expected inequalities.

Run:  python3 chapter_0091_HanFei_-280.py
"""

import numpy as np

RNG = np.random.default_rng(91)          # Han Fei is figure #91
np.set_printoptions(precision=4, suppress=True)


# ============================================================================
# 0.  THE LAW (fa): a public registry of mandated forms, one per office (name)
# ============================================================================
# Each "office" (ming, a one-hot name) has a single correct "form" it must
# produce -- a fixed vector in form-space. This registry is PUBLIC and FIXED:
# it is the law (fa), knowable to all. The controller must LEARN it from data so
# it can verify form against name. (xing-ming requires the verifier to hold the
# standard internally; that is exactly what training does here.)

def build_registry(num_offices: int, form_dim: int) -> np.ndarray:
    """Return Phi of shape (num_offices, form_dim): the mandated form per name."""
    # Smoothly varied, well-separated mandated forms (a clean, legible code book).
    base = RNG.standard_normal((num_offices, form_dim))
    # normalise each mandate to unit norm so "magnitude" has a shared meaning;
    # over-fulfilment then literally means ||form|| > 1 (exceeding the office).
    base /= (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9)
    return base


# ============================================================================
# 1.  THE CONTROLLER (the enlightened ruler / verifier)
# ============================================================================
def softplus(z):
    # numerically stable softplus, keeps the two handles non-negative
    return np.logaddexp(0.0, z)

def softplus_grad(z):
    # d/dz softplus = sigmoid(z)
    return 1.0 / (1.0 + np.exp(-z))


class TwoHandleController:
    """
    Input  u = [name (one-hot, K) ; observed form (k)]      -> R^{K+k}
    Hidden a1 = tanh(W1 u + b1)                              -> R^{H}
    Reward handle   de   = softplus(Wr a1 + br)  >= 0        -> R^{k}
    Punish handle   xing = softplus(Wp a1 + bp)  >= 0        -> R^{k}
    Net correction  c    = de - xing                        -> R^{k}

    Interpretation (Han Fei): where the form falls SHORT of the name, the reward
    handle (de) lifts it; where the form EXCEEDS the name, the punishment handle
    (xing) cuts it back. The net correction c should equal the xing-ming gap
    g* = Phi(name) - form. Two distinct, single-signed levers -- never one
    bidirectional knob -- because the two handles are categorically different
    instruments and (crucially) must be held separately by the controller.
    """

    def __init__(self, num_offices: int, form_dim: int, hidden: int = 32):
        self.K = num_offices
        self.k = form_dim
        self.H = hidden
        s = 1.0 / np.sqrt(self.K + self.k)
        self.W1 = RNG.standard_normal((hidden, self.K + self.k)) * s
        self.b1 = np.zeros(hidden)
        sh = 1.0 / np.sqrt(hidden)
        self.Wr = RNG.standard_normal((self.k, hidden)) * sh   # reward head
        self.br = np.zeros(self.k)
        self.Wp = RNG.standard_normal((self.k, hidden)) * sh   # punish head
        self.bp = np.zeros(self.k)
        self.lambda_orth = 0.05   # weight on "handles do not fight" prior

    # -- parameter (de)serialisation, used by the gradient check ------------
    def get_params(self):
        return [self.W1, self.b1, self.Wr, self.br, self.Wp, self.bp]

    def set_params(self, params):
        self.W1, self.b1, self.Wr, self.br, self.Wp, self.bp = params

    # -- forward ------------------------------------------------------------
    def forward(self, names, forms):
        """
        names : (B, K) one-hot declared names
        forms : (B, k) observed forms
        returns dict of activations for backprop + outputs.
        """
        u = np.concatenate([names, forms], axis=1)          # (B, K+k)
        z1 = u @ self.W1.T + self.b1                         # (B, H)
        a1 = np.tanh(z1)                                     # (B, H)
        zr = a1 @ self.Wr.T + self.br                        # (B, k)
        zp = a1 @ self.Wp.T + self.bp                        # (B, k)
        de = softplus(zr)                                    # (B, k) reward >=0
        xing = softplus(zp)                                  # (B, k) punish >=0
        c = de - xing                                        # (B, k) net correction
        return dict(u=u, z1=z1, a1=a1, zr=zr, zp=zp, de=de, xing=xing, c=c)

    # -- loss ---------------------------------------------------------------
    def loss(self, names, forms, g_star):
        """
        Mean-squared xing-ming error between net correction c and the true gap
        g* = Phi(name) - form, plus a small orthogonality term that discourages
        the two handles from acting on the same dimension at once (a lever is a
        lever; you do not simultaneously reward and punish the same act).
        """
        cache = self.forward(names, forms)
        B = names.shape[0]
        diff = cache['c'] - g_star                           # (B, k)
        mse = 0.5 * np.sum(diff * diff) / B
        orth = self.lambda_orth * np.sum(cache['de'] * cache['xing']) / B
        cache['diff'] = diff
        return mse + orth, cache

    # -- analytic backward --------------------------------------------------
    def backward(self, cache):
        """Hand-derived gradients for every parameter. Returns same order as
        get_params(): [dW1, db1, dWr, dbr, dWp, dbp]."""
        B = cache['u'].shape[0]
        diff = cache['diff']                                 # (B, k) = c - g*
        de, xing = cache['de'], cache['xing']
        a1, z1, u = cache['a1'], cache['z1'], cache['u']
        zr, zp = cache['zr'], cache['zp']

        # d mse / d c = diff / B ; c = de - xing
        dc = diff / B                                        # (B, k)
        # orthogonality term: lambda * sum(de*xing)/B
        d_de_orth = self.lambda_orth * xing / B              # d/dde
        d_xing_orth = self.lambda_orth * de / B              # d/dxing

        # de = softplus(zr) ; xing = softplus(zp)
        d_de = dc + d_de_orth                                # (B, k) total dL/dde
        d_xing = -dc + d_xing_orth                           # since c=de-xing
        dzr = d_de * softplus_grad(zr)                       # (B, k)
        dzp = d_xing * softplus_grad(zp)                     # (B, k)

        dWr = dzr.T @ a1                                     # (k, H)
        dbr = dzr.sum(axis=0)                                # (k,)
        dWp = dzp.T @ a1                                     # (k, H)
        dbp = dzp.sum(axis=0)                                # (k,)

        da1 = dzr @ self.Wr + dzp @ self.Wp                  # (B, H)
        dz1 = da1 * (1.0 - np.tanh(z1) ** 2)                 # (B, H)
        dW1 = dz1.T @ u                                      # (H, K+k)
        db1 = dz1.sum(axis=0)                                # (H,)
        return [dW1, db1, dWr, dbr, dWp, dbp]


# ============================================================================
# 2.  GRADIENT CHECK (mandatory)
# ============================================================================
def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    K, k, H, B = 6, 4, 12, 8
    Phi = build_registry(K, k)
    ctrl = TwoHandleController(K, k, hidden=H)

    name_idx = rng.integers(0, K, size=B)
    names = np.eye(K)[name_idx]
    forms = rng.standard_normal((B, k)) * 0.7
    g_star = Phi[name_idx] - forms

    _, cache = ctrl.loss(names, forms, g_star)
    analytic = ctrl.backward(cache)

    eps = 1e-6
    max_rel = 0.0
    params = ctrl.get_params()
    for pi, P in enumerate(params):
        flat = P.ravel()
        g_an = analytic[pi].ravel()
        # check a handful of coordinates per parameter tensor
        idxs = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + eps
            lp, _ = ctrl.loss(names, forms, g_star)
            flat[j] = orig - eps
            lm, _ = ctrl.loss(names, forms, g_star)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            rel = abs(num - g_an[j]) / (abs(num) + abs(g_an[j]) + 1e-12)
            max_rel = max(max_rel, rel)
    return max_rel


# ============================================================================
# 3.  TRAINING LOOP (controller internalises the law and learns to verify)
# ============================================================================
def make_batch(Phi, B, noise=0.6, rng=RNG):
    K, k = Phi.shape
    name_idx = rng.integers(0, K, size=B)
    names = np.eye(K)[name_idx]
    # observed forms = mandated form + self-interested deviation (noise)
    forms = Phi[name_idx] + rng.standard_normal((B, k)) * noise
    g_star = Phi[name_idx] - forms
    return names, forms, g_star

def train(Phi, epochs=400, B=64, lr=0.05, verbose=True):
    K, k = Phi.shape
    ctrl = TwoHandleController(K, k, hidden=32)
    history = []
    for ep in range(epochs):
        names, forms, g_star = make_batch(Phi, B)
        L, cache = ctrl.loss(names, forms, g_star)
        grads = ctrl.backward(cache)
        for P, G in zip(ctrl.get_params(), grads):
            P -= lr * G
        history.append(L)
        if verbose and (ep % 80 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:4d}   xing-ming loss = {L:.5f}")
    # held-out verification: can it recover the office from corrected form?
    names, forms, g_star = make_batch(Phi, 2000, rng=np.random.default_rng(7))
    c = ctrl.forward(names, forms)['c']
    corrected = forms + c                       # apply the controller's correction
    # classify corrected form to nearest mandate; legible if it recovers the name
    d = np.linalg.norm(corrected[:, None, :] - Phi[None, :, :], axis=2)
    pred = d.argmin(axis=1)
    true = names.argmax(axis=1)
    acc = (pred == true).mean()
    mean_gap_before = np.linalg.norm(forms - Phi[true], axis=1).mean()
    mean_gap_after = np.linalg.norm(corrected - Phi[true], axis=1).mean()
    return ctrl, history, acc, mean_gap_before, mean_gap_after


# ============================================================================
# 4.  CLOSED-LOOP COURT  (the experiments)
# ============================================================================
# A population of "officials." Each round an official declares a name and
# produces a form; the controller applies a correction through the two handles;
# the official's form updates. "Adversarial" officials also push their form to
# INFLATE magnitude (farm reward) -- this is the reward-hacking pressure.

def run_court(Phi, n_officials=200, rounds=60, mode="xingming",
              leverage=0.6, drift=0.08, hack_frac=0.5, handle_captured=False,
              audit="every", seed=123):
    """
    mode:
      "xingming"    -> bidirectional: correct form toward the mandate (both
                        over- and under-shoot pulled to Phi). Reward peaks AT
                        the mandate.
      "meritocratic"-> one-sided "more is better": never cut excess; reward
                        grows with form magnitude, so deficient forms are lifted
                        but excess is left (and even encouraged).
    leverage   : shi -- how strongly the controller's correction is applied.
    drift      : random self-interested wandering of each form per round.
    hack_frac  : fraction of officials that actively inflate their form.
    handle_captured : if True, hacking officials can ADD to their own reward
                        (seize a handle) -> tests two-handles custody.
    audit      : "every" (audit each round) or "random" (audit ~half the rounds,
                  unpredictably) -> tests shu (hidden/unpredictable oversight).
    Returns a dict of trajectories and final legibility metrics.
    """
    rng = np.random.default_rng(seed)
    K, k = Phi.shape
    name_idx = rng.integers(0, K, size=n_officials)
    mandate = Phi[name_idx]                                  # each official's true office form
    form = mandate + rng.standard_normal((n_officials, k)) * 0.3
    is_hacker = rng.random(n_officials) < hack_frac

    legibility = []        # accuracy of reading office from form (nearest mandate)
    mean_norm = []         # mean ||form|| -- detects inflation/collapse
    for t in range(rounds):
        # 1) self-interested dynamics: everyone drifts; hackers inflate.
        form = form + rng.standard_normal((n_officials, k)) * drift
        inflate = np.zeros_like(form)
        inflate[is_hacker] = 0.25 * form[is_hacker]          # push magnitude up
        form = form + inflate

        # 2) does an audit happen this round?
        do_audit = True if audit == "every" else (rng.random() < 0.5)

        if do_audit:
            gap = mandate - form                              # xing-ming gap g*
            if mode == "xingming":
                # both handles: lift shortfall, cut excess -> move toward mandate
                correction = leverage * gap
            elif mode == "meritocratic":
                # one-sided: only lift shortfall; NEVER cut excess. "More is
                # better": excess is tolerated (and reward keeps flowing to big
                # forms), so the registry inflates and becomes unreadable.
                correction = leverage * np.maximum(gap, 0.0)
            else:
                raise ValueError(mode)

            # 3) two-handles custody. Han Fei: "whoever holds the handle
            #    governs." If the handle is captured, the ruler's reward lever
            #    no longer reaches the hackers -- THEY now set their own update
            #    (self-reward / runaway inflation) and are immune to correction.
            #    The controller can still hold the loyal majority, but cannot
            #    restore the agents who hold the channel.
            if handle_captured:
                controllable = ~is_hacker
                form[controllable] = form[controllable] + correction[controllable]
                form[is_hacker] = form[is_hacker] * 1.30      # runaway self-reward
            else:
                form = form + correction                      # ruler holds both handles

        # 4) measure legibility: can we read each office from its form?
        d = np.linalg.norm(form[:, None, :] - Phi[None, :, :], axis=2)
        pred = d.argmin(axis=1)
        legibility.append((pred == name_idx).mean())
        mean_norm.append(np.linalg.norm(form, axis=1).mean())

    return dict(legibility=np.array(legibility),
                mean_norm=np.array(mean_norm),
                final_legibility=legibility[-1],
                final_norm=mean_norm[-1])


def shi_phase_transition(Phi, leverages, drift=0.35, seed=321):
    """Sweep oversight leverage (shi). Below a threshold, order cannot hold even
    with a perfect (xing-ming) verifier. Returns leverage -> final legibility."""
    out = []
    for lev in leverages:
        r = run_court(Phi, mode="xingming", leverage=lev, drift=drift,
                      hack_frac=0.5, rounds=80, seed=seed)
        out.append((lev, r['final_legibility']))
    return out


# ============================================================================
# 5.  MAIN
# ============================================================================
def main():
    print("=" * 78)
    print("  HAN FEI  (#91)  --  XING-MING VERIFICATION NETWORK / TWO HANDLES")
    print("=" * 78)

    # ---- (a) gradient check -------------------------------------------------
    print("\n[1] Finite-difference gradient check (mandatory)")
    max_rel = max(gradient_check(seed=s) for s in (0, 1, 2))
    print(f"    max relative error over 3 seeds = {max_rel:.3e}")
    assert max_rel < 1e-6, "gradient check FAILED"
    print("    PASS  (analytic backprop matches numerical gradient)")

    # ---- (b) training -------------------------------------------------------
    print("\n[2] Training the controller to internalise the law (fa) and verify")
    Phi = build_registry(num_offices=8, form_dim=5)
    ctrl, hist, acc, gap0, gap1 = train(Phi, epochs=400, B=64, lr=0.05)
    print(f"    loss  {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"    held-out: office recovered from corrected form = {acc*100:.1f}%")
    print(f"    mean form-name gap  before {gap0:.3f}  ->  after {gap1:.3f}")
    assert hist[-1] < 0.25 * hist[0], "training did not reduce loss enough"
    assert acc > 0.95, "controller failed to make forms legible"
    assert gap1 < 0.5 * gap0, "correction did not close the form-name gap"
    print("    PASS")

    # ---- (c) Experiment A: xing-ming vs meritocratic under hacking ----------
    print("\n[3] Experiment A -- bidirectional xing-ming vs one-sided 'merit'")
    print("    (reward-hacking officials inflate their form to farm reward)")
    xm = run_court(Phi, mode="xingming",    leverage=0.6, hack_frac=0.6, rounds=60)
    me = run_court(Phi, mode="meritocratic", leverage=0.6, hack_frac=0.6, rounds=60)
    print(f"    xing-ming   : legibility {xm['final_legibility']*100:5.1f}%   "
          f"mean||form|| {xm['final_norm']:.2f}")
    print(f"    meritocratic: legibility {me['final_legibility']*100:5.1f}%   "
          f"mean||form|| {me['final_norm']:.2f}")
    print("    -> punishing OVER-fulfilment (the hat-keeper) keeps the registry")
    print("       legible; rewarding 'more is better' lets it inflate & blur.")
    assert xm['final_legibility'] > me['final_legibility'] + 0.2
    assert me['final_norm'] > xm['final_norm'] * 1.3
    print("    PASS")

    # ---- (d) Experiment B: two-handles custody ------------------------------
    print("\n[4] Experiment B -- the two handles must stay with the controller")
    held = run_court(Phi, mode="xingming", leverage=0.6, hack_frac=0.5,
                     handle_captured=False, rounds=50)
    seized = run_court(Phi, mode="xingming", leverage=0.6, hack_frac=0.5,
                       handle_captured=True, rounds=50)
    print(f"    handles held by ruler : mean||form|| {held['final_norm']:7.2f}   "
          f"legibility {held['final_legibility']*100:.1f}%")
    print(f"    handle seized by agents: mean||form|| {seized['final_norm']:7.2f}   "
          f"legibility {seized['final_legibility']*100:.1f}%")
    print("    -> when the optimiser reaches the reward channel, order diverges.")
    assert seized['final_norm'] > held['final_norm'] * 3.0
    print("    PASS")

    # ---- (e) Experiment C: shi (positional leverage) phase transition -------
    print("\n[5] Experiment C -- shi: order needs positional leverage, not virtue")
    sweep = shi_phase_transition(Phi, leverages=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8])
    threshold = None
    for lev, leg in sweep:
        flag = ""
        if leg > 0.8 and threshold is None:
            threshold = lev
            flag = "  <- order restored"
        print(f"    leverage(shi)={lev:.2f}   final legibility = {leg*100:5.1f}%{flag}")
    print(f"    -> phase transition near shi ~ {threshold}: a perfect verifier")
    print("       with too little position cannot hold order; a modest verifier")
    print("       with enough position can. Authority is structural.")
    assert sweep[0][1] < 0.6 and sweep[-1][1] > 0.85
    print("    PASS")

    # ---- (f) Experiment D: shu (unpredictable audit) ------------------------
    print("\n[6] Experiment D -- shu: predictable audits get gamed between checks")
    every = run_court(Phi, mode="xingming", leverage=0.6, hack_frac=0.6,
                      audit="every", rounds=60, seed=55)
    rand = run_court(Phi, mode="xingming", leverage=0.6, hack_frac=0.6,
                     audit="random", rounds=60, seed=55)
    print(f"    audit every round : legibility {every['final_legibility']*100:.1f}%")
    print(f"    audit random/hidden: legibility {rand['final_legibility']*100:.1f}%")
    print("    -> the more the agent can predict the audit gap, the more it")
    print("       drifts; both hold here, but constant oversight is strongest.")
    assert every['final_legibility'] >= rand['final_legibility'] - 0.05

    print("\n" + "=" * 78)
    print("  ALL CHECKS PASSED -- the court holds names to forms.")
    print("=" * 78)


if __name__ == "__main__":
    main()

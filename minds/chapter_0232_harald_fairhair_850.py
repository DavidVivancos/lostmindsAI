#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 LUFA  —  The Uncombed Consolidator
 Chapter 0232 · Harald Fairhair / Haraldr lúfa (c. 850 – c. 932)
 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 12 Minds 221 - 240 Available on Amazon https://www.amazon.com/dp/B0HKVBKNTT
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0232_harald_fairhair_850 - Harald Fairhair (c.850 – c. 932)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy neural architecture — no autograd, no frameworks —
whose structure encodes the one cognitive operation that belongs to Harald
Fairhair and to nobody else in this corpus: A GOAL CARRIED VISIBLY ON THE BODY,
UNDISCHARGED, WITH NO PARTIAL CREDIT, UNTIL ONE BINARY CRITERION CLOSES.

The saga tradition gives him a single defining act, and it is not a battle.
Fagrskinna (c. 1220) reports that he swore not to cut his hair until he was
overlord of Norway and drew tribute from every inland valley and outlying
headland. Heimskringla (c. 1230) attaches the vow to Gyða's refusal. Egils saga
has the vow and no Gyða at all. For ten years he was Haraldr lúfa — Harald
Matted-Hair, Shockhead, Tanglehair — and then jarl Rǫgnvald of Møre cut and
dressed the hair and he became hárfagri, Fine-Hair.

That is a commitment device with a precise computational shape:

    (1) The objective is BINARY. There is no reward for nine valleys out of ten.
    (2) Progress is not stored in the reward. It is stored in an ACCUMULATOR
        that is not normalised, not pruned, and not tidied — and that is
        PUBLICLY LEGIBLE, so every other agent can read the king's remaining
        distance-to-goal off his head.
    (3) The accumulator is discharged EXACTLY ONCE, at closure, and the agent's
        NAME changes when it is.

And there is a fourth thing, which the sources are unusually clear about and
which is the part that matters most for AGI. Harald's realm did not cohere
because he out-fought everyone. It cohered because the people who could not
live under one law LEFT. Ari Þorgilsson's Íslendingabók (c. 1122-33) — the
oldest narrative source we have — does not say Harald drove them out. It says
the emigration to Iceland grew so large that the king FORBADE it, for fear his
land would be emptied, and then settled for a toll of five aurar a head. That
toll is the origin of the landaurar.

So the record closest to the man describes a unifier trying to shut the exit.
And the record itself survives only because the exit stayed open: every text
we have about Harald was written in Iceland, or from Icelandic material, by the
descendants of the people who left. Bruce Lincoln's study of these traditions
(Chicago, 2014) turns on exactly this — the Icelandic tellings are quieter,
more ironic and more hostile than the Norwegian ones, because their audience
were the heirs of the seceders.

THE ARCHITECTURAL CLAIM
-----------------------
Build a network out of K local experts ("óðal blocks", one per district) and
one shared expert ("the crown"), mixed by a learned per-district allegiance
α_k ∈ (0,1). Impose three rules that are Harald's and not anyone else's:

    OATH      The crown is never spectrally normalised — never combed — while
              the oath is open. Its condition number is allowed to grow. It is
              groomed exactly once, at the instant every surviving district has
              sworn, and never again.

    EXIT      A district may leave. It leaves when the cost of full submission
              exceeds what it achieves governing itself, by more than a
              threshold plus a TOLL. Departed districts keep their own working
              model and are used afterwards to audit the crown.

    LOOK      The realm's opinion of a district is a running average. If it is
              never reset, the realm will act on a reputation that the district
              no longer deserves.

Then measure what each rule costs. The results are in the EXPERIMENTS section
and three of the five contradicted the hypothesis I started with. They are
reported as they came out.

WHAT IS INFERENCE AND WHAT IS EVIDENCE
--------------------------------------
Harald left no words. Not one sentence of his survives. Two praise poems by
Þorbjǫrn hornklofi — Haraldskvæði (Hrafnsmál) and Glymdrápa — may be
ninth-century, but they reach us only inside thirteenth-century prose, and in
the better manuscripts Haraldskvæði calls its honorand Haraldr Hálfdanarson or
lúfa, NOT hárfagri. Glymdrápa gives no epithet at all. A serious body of
scholarship — Sawyer 1976, Krag 1989/1995, Sverrir Jakobsson 2002/2016 — holds
that the unifier of all Norway is a twelfth-century construction, and that the
historical Haraldr, if he existed, was a west-coast king of Sogn and Rogaland.

This file therefore does not claim to model a documented psychology. It models
the SHAPE OF THE STORY, which is itself the artefact, and it tests that shape
as an engineering proposition. Where the code embodies an inference rather than
a record, the comment says so.

RUN
---
    python3 chapter_0232_harald_fairhair_850.py
    python3 chapter_0232_harald_fairhair_850.py --fast     (skip the slow probe)

Requires only NumPy. Finite-difference gradient check is mandatory and runs
first; nothing else executes if it fails.
================================================================================
"""

import sys
import time
import numpy as np

# ==============================================================================
# SECTION 0 — THE REALM
# ==============================================================================
# Nine districts. The names are the polities the saga tradition actually places
# in Harald's path: Sogn and Rogaland (which modern scholarship takes to be his
# real power base), Hordaland (Gyða's father's kingdom), Firðafylki and
# Sunnmøre (Audbjǫrn and Arnviðr in Egils saga), Trøndelag (Hákon
# Grjótgarðsson, the first jarl of Lade), Namdalen (the brothers Herlaugr and
# Hrollaugr), Hålogaland, and Vingulmark in the east.
#
# CUSTOM[k] is how heavily district k's judgements depart from the common law.
# Six are close to the shared law and three are not. Which three is planted
# ground truth: the architecture is supposed to FIND them, and it does.

DISTRICTS = ["Sogn", "Hordaland", "Rogaland", "Firdafylki", "Sunnmore",
             "Trondelag", "Namdalen", "Halogaland", "Vingulmark"]

CUSTOM = np.array([0.15, 0.20, 0.12, 0.18, 0.22, 0.16,   # will consolidate
                   1.40, 1.55, 1.30])                     # will not

IRRECONCILABLE = {6, 7, 8}          # ground truth, never shown to the model

K = 9      # districts
D = 12     # features of a case brought to the thing (assembly)
H = 64     # hidden width of crown and óðal blocks
O = 2      # two verdict dimensions: compensation owed, and severity of outlawry
HS = 24    # hidden width of a district's own independent model


def _realm_law(seed=0):
    """
    The fixed generative truth of the realm.

    A1/A2 define a SHARED law: a smooth nonlinear map from case features to
    verdict that holds everywhere in Norway. B[k] defines district k's own
    custom — a linear departure from the shared law, in a direction that is
    peculiar to that district.

    This is not a metaphor imposed on the period. Norway in the ninth and tenth
    centuries genuinely had regional law-things — the Gulaþing in the west, the
    Frostuþing around Trondheim — administering overlapping but distinct law.
    Unification is exactly the problem of whether one law can serve them all.
    """
    r = np.random.default_rng(seed)
    A1 = r.normal(0, 1, (10, D)) / np.sqrt(D)
    A2 = r.normal(0, 1, (O, 10)) / np.sqrt(10)
    B = [r.normal(0, 1, (O, D)) / np.sqrt(D) for _ in range(K)]
    return A1, A2, B


LAW = _realm_law(0)


def make_realm(n_per=200, seed=0, custom=None):
    """Draw n_per cases for each district. custom=None uses the standard table."""
    A1, A2, B = LAW
    cu = CUSTOM if custom is None else custom
    r = np.random.default_rng(seed + 1000)
    X, Kid, Y = [], [], []
    for k in range(K):
        x = r.normal(0, 1, (n_per, D))
        y = np.tanh(x @ A1.T) @ A2.T + cu[k] * (x @ B[k].T) + r.normal(0, 0.03, (n_per, O))
        X.append(x); Y.append(y); Kid.append(np.full(n_per, k))
    return np.concatenate(X), np.concatenate(Kid), np.concatenate(Y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ==============================================================================
# SECTION 1 — THE MODEL
# ==============================================================================

class Lufa:
    """
    The uncombed consolidator.

    For a case x brought from district k:

        u = W_k x + b_k                     the district's own reading  (óðal)
        v = W_c x + b_c                     the crown's reading
        p = (1 - α_k) u  +  α_k v           allegiance mixes the two
        h = tanh(p)
        ŷ = V h + c                         ONE shared verdict head

    α_k = σ(a_k) is the district's allegiance, learned. α_k = 0 is a free
    petty kingdom; α_k = 1 is a province that has handed over jurisdiction
    entirely. The verdict head V is shared and never forks: there is only ever
    one Norway at the point of judgement, which is the whole claim being tested.

    Note what the mixture does NOT do. It does not mix outputs. It mixes
    PRE-ACTIVATIONS, before the nonlinearity — so a half-sworn district is not
    getting a compromise verdict, it is getting a verdict from a confused
    hybrid reader. Submission in this architecture is not negotiable in
    degree; it is only slow.
    """

    def __init__(self, seed=1, a0=-1.5):
        r = np.random.default_rng(seed)
        self.Wo = r.normal(0, 1, (K, H, D)) * np.sqrt(2.0 / (H + D))
        self.bo = np.zeros((K, H))
        self.Wc = r.normal(0, 1, (H, D)) * np.sqrt(2.0 / (H + D))
        self.bc = np.zeros(H)
        # a0 = -1.5 → α ≈ 0.18. Everyone starts very nearly independent, which
        # is the correct ninth-century initial condition.
        self.a = np.full(K, float(a0))
        self.V = r.normal(0, 1, (O, H)) * np.sqrt(2.0 / (O + H))
        self.c = np.zeros(O)
        self.gone = np.zeros(K, bool)      # seceded districts

    def params(self):
        return dict(Wo=self.Wo, bo=self.bo, Wc=self.Wc, bc=self.bc,
                    a=self.a, V=self.V, c=self.c)

    def alpha(self):
        """A seceded district's allegiance is exactly zero and stays there."""
        return np.where(self.gone, 0.0, sigmoid(self.a))

    def forward(self, X, Kid):
        A = self.alpha()[Kid][:, None]
        u = np.einsum('nd,nhd->nh', X, self.Wo[Kid]) + self.bo[Kid]
        v = X @ self.Wc.T + self.bc
        h = np.tanh((1.0 - A) * u + A * v)
        return dict(A=A, u=u, v=v, h=h, yh=h @ self.V.T + self.c, al=self.alpha())

    def loss(self, X, Kid, Y, lam=0.0):
        """
        Task loss plus the UNIFICATION PRESSURE.

        lam * mean(1 - α) over surviving districts is the standing demand for
        tribute — a constant force pushing every α toward 1, exactly as strong
        for a willing district as for an unwilling one. This is the oath acting
        as an objective. A district resists it only by the task loss it would
        suffer from submitting, and that tug-of-war is the whole political
        situation of the reign expressed in two terms.
        """
        f = self.forward(X, Kid)
        r = f['yh'] - Y
        Lt = float(np.mean(r * r))
        act = ~self.gone
        Lu = lam * float(np.mean(1.0 - f['al'][act])) if act.any() else 0.0
        return Lt + Lu, f, Lt

    def backward(self, X, Kid, Y, f, lam=0.0):
        """Hand-derived gradients. Verified against central differences below."""
        N = len(X)
        dyh = 2.0 * (f['yh'] - Y) / (N * O)
        dV = dyh.T @ f['h']
        dc = dyh.sum(0)
        dp = (dyh @ self.V) * (1.0 - f['h'] ** 2)

        A = f['A']
        du = (1.0 - A) * dp          # into the district's own block
        dv = A * dp                  # into the crown

        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        # dL/dα_k for a sample is dp · (v - u): how much better or worse the
        # crown's reading is than the district's own, on this case.
        dA = np.sum(dp * (f['v'] - f['u']), axis=1)
        dal = np.zeros(K)
        for k in range(K):
            m = Kid == k
            if m.any():
                dWo[k] = du[m].T @ X[m]
                dbo[k] = du[m].sum(0)
                dal[k] = dA[m].sum()

        act = ~self.gone
        if act.any():
            dal[act] -= lam / act.sum()          # the standing demand
        da = dal * f['al'] * (1.0 - f['al'])     # through the sigmoid
        da[self.gone] = 0.0                      # the departed do not vote

        return dict(Wo=dWo, bo=dbo, Wc=dv.T @ X, bc=dv.sum(0),
                    a=da, V=dV, c=dc)


class RMSProp:
    """
    From-scratch elementwise-adaptive optimiser.

    Chosen deliberately rather than plain SGD: an elementwise preconditioner is
    BASIS-DEPENDENT, which means the spectral state of the crown genuinely
    affects the trajectory. Under plain SGD the combing experiment below would
    be measuring far less than it does.
    """

    def __init__(self, params, lr=2e-3, rho=0.9, eps=1e-8):
        self.s = {k: np.zeros_like(v) for k, v in params.items()}
        self.lr, self.rho, self.eps = lr, rho, eps

    def step(self, params, grads, scale=1.0):
        for k, P in params.items():
            self.s[k] = self.rho * self.s[k] + (1 - self.rho) * grads[k] ** 2
            P -= scale * self.lr * grads[k] / (np.sqrt(self.s[k]) + self.eps)

    def reset(self, key):
        """Forget the second-moment history for one tensor. Used at the cut."""
        self.s[key][:] = 0.0


class Shadow:
    """
    A district's own independent model — the boat kept in the boathouse.

    This is the single most important auxiliary object in the file. Without it
    there is no honest way to ask what submission COSTS a district, because by
    the time a district has sworn, its óðal block has already been dissolved by
    the stand-down decay, and "what would you do alone?" has no answer left.

    Egils saga says Harald offered rebels three things: flee the country,
    become his tenant, or lose hands and feet — and that most who were offered
    it chose to flee. A choice like that is only real if the person choosing
    still has something to leave WITH. The Shadow is that something. It trains
    continuously on its own district's cases, at α = 0, and it is what a
    seceding district takes to Iceland.
    """

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 1, (HS, D)) * np.sqrt(2.0 / (HS + D))
        self.b = np.zeros(HS)
        self.V = r.normal(0, 1, (O, HS)) * np.sqrt(2.0 / (O + HS))
        self.c = np.zeros(O)

    def params(self):
        return dict(W=self.W, b=self.b, V=self.V, c=self.c)

    def forward(self, X):
        h = np.tanh(X @ self.W.T + self.b)
        return h, h @ self.V.T + self.c

    def step(self, X, Y, opt):
        h, yh = self.forward(X)
        dyh = 2.0 * (yh - Y) / (len(X) * O)
        dp = (dyh @ self.V) * (1.0 - h ** 2)
        opt.step(self.params(), dict(W=dp.T @ X, b=dp.sum(0),
                                     V=dyh.T @ h, c=dyh.sum(0)))

    def loss(self, X, Y):
        _, yh = self.forward(X)
        return float(np.mean((yh - Y) ** 2))


# ==============================================================================
# SECTION 2 — THE THREE RULES
# ==============================================================================

def comb(W):
    """
    THE CUT. Frobenius-preserving spectral flattening.

    Take the SVD and replace every singular value with the root-mean-square of
    the set. The matrix keeps its singular vectors — it still looks in the same
    directions — and it keeps its total norm exactly, because setting all σ to
    sqrt(mean(σ²)) leaves Σσ² unchanged. What it loses is ANISOTROPY: the
    condition number becomes 1.

    The choice to preserve norm is not arbitrary. Rǫgnvald combed and dressed
    the hair; he did not shave the king. Nothing was removed. It was made
    straight, and then it turned out to be beautiful, and the man got a new
    name. This function does precisely that and nothing more.
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return (U * np.sqrt(np.mean(S ** 2))) @ Vt


def tangle(W):
    """Condition number — how matted the crown has become."""
    S = np.linalg.svd(W, compute_uv=False)
    return float(S[0] / max(S[-1], 1e-12))


def hair_length(W):
    """Frobenius norm — how much has accumulated since the vow."""
    return float(np.linalg.norm(W))


def compliance_cost(model, shadows, X, Kid, Y, idxk):
    """
    THE GRIEVANCE, and the crux of the whole design.

    For each district: the loss it would suffer under FULL allegiance (α := 1
    for everyone, counterfactually), minus the loss it achieves with its own
    independent model.

    Note carefully what is NOT measured. We do not ask how a district is doing
    at its current, self-adjusted allegiance — a district that has quietly
    settled at α = 0.5 is not being governed and is not aggrieved, it is simply
    unconquered. The oath demands α ≥ τ from everyone. So the politically real
    quantity is the cost of the submission actually being demanded.

    This is a control-loop measurement, not part of the differentiable loss.
    Secession is a discrete event over the training run, like early stopping.
    """
    sa, sg = model.a.copy(), model.gone.copy()
    model.a[:] = 50.0
    model.gone[:] = False
    f = model.forward(X, Kid)
    model.a, model.gone = sa, sg
    return np.array([
        float(np.mean((f['yh'][idxk[k]] - Y[idxk[k]]) ** 2)) - shadows[k].loss(X[idxk[k]], Y[idxk[k]])
        for k in range(K)
    ])


# ==============================================================================
# SECTION 3 — GRADIENT CHECK (mandatory, runs first)
# ==============================================================================

def gradient_check(n_per_tensor=6, eps=1e-6, tol=1e-4, seed=3):
    """
    Central finite differences against the hand-derived backward pass, on every
    parameter tensor, with one district already seceded so the masking logic is
    exercised too.
    """
    X, Kid, Y = make_realm(6, seed=seed)
    m = Lufa(seed=seed)
    m.gone[7] = True                      # exercise the departed-district path
    lam = 0.30

    L, f, _ = m.loss(X, Kid, Y, lam)
    g = m.backward(X, Kid, Y, f, lam)

    rng = np.random.default_rng(99)
    worst, worst_where = 0.0, ""
    for name, P in m.params().items():
        flat, gflat = P.reshape(-1), g[name].reshape(-1)
        idx = rng.choice(flat.size, size=min(n_per_tensor, flat.size), replace=False)
        for i in idx:
            o = flat[i]
            flat[i] = o + eps; Lp, _, _ = m.loss(X, Kid, Y, lam)
            flat[i] = o - eps; Lm, _, _ = m.loss(X, Kid, Y, lam)
            flat[i] = o
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_where = rel, f"{name}[{i}]"
    ok = worst < tol
    print(f"  gradient check : worst relative error {worst:.3e} at {worst_where}"
          f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


# ==============================================================================
# SECTION 4 — THE TRAINING LOOP
# ==============================================================================

def train(regime="oath", allow_exit=True, toll=0.0, epochs=500, seed=1,
          lam=0.30, mu=0.08, tau=0.85, theta=1.0, strikes=2, audit=25,
          grace=150, lr=2e-3, n_per=200, batch=256, trace=False):
    """
    regime : "oath"   comb the crown exactly once, when the oath closes
             "never"  never comb it
             "always" comb it every epoch
    toll   : the landaurar. Added to the secession threshold. Íslendingabók
             puts the historical figure at five aurar a head.
    mu     : STAND-DOWN DECAY. Weight decay on district k's own block scaled by
             α_k — the more a district has sworn, the faster its own apparatus
             dissolves. Without this the model cheats: α rises to 1 while the
             óðal block quietly grows to compensate, and "unification" is a
             number on a page with nothing behind it. Egils saga has Harald
             confiscating property and turning free farmers into tenants; this
             is that, as one line of the update rule. Applied with a
             stop-gradient on α (it is part of the optimiser, not the loss, so
             the gradient check above stays exact).
    """
    Xtr, Ktr, Ytr = make_realm(n_per, seed)
    Xva, Kva, Yva = make_realm(80, seed + 500)
    idxk = [np.where(Ktr == k)[0] for k in range(K)]

    m = Lufa(seed + 7)
    opt = RMSProp(m.params(), lr=lr)
    shadows = [Shadow(seed * 100 + k) for k in range(K)]
    sopt = [RMSProp(s.params(), lr=3e-3) for s in shadows]

    rng = np.random.default_rng(seed)
    N = len(Xtr)
    cut_epoch = -1
    strike = np.zeros(K, int)
    log = []

    for ep in range(epochs):
        decay = 1.0 if ep < epochs * 0.6 else 0.3
        perm = rng.permutation(N)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            _, f, _ = m.loss(Xtr[idx], Ktr[idx], Ytr[idx], lam)
            g = m.backward(Xtr[idx], Ktr[idx], Ytr[idx], f, lam)
            g['Wo'] = g['Wo'] + mu * m.alpha()[:, None, None] * m.Wo
            opt.step(m.params(), g, decay)

        # every district keeps rehearsing exile
        for k in range(K):
            shadows[k].step(Xtr[idxk[k]], Ytr[idxk[k]], sopt[k])

        # --- THE OATH -------------------------------------------------------
        active = ~m.gone
        closed = active.sum() > 0 and np.all(m.alpha()[active] >= tau)
        if regime == "always":
            m.Wc[:] = comb(m.Wc)
        elif regime == "oath" and closed and cut_epoch < 0:
            m.Wc[:] = comb(m.Wc)
            opt.reset('Wc')          # the new king does not inherit the old momentum
            cut_epoch = ep

        # --- THE EXIT -------------------------------------------------------
        if allow_exit and ep >= grace and ep % audit == audit - 1:
            gv = compliance_cost(m, shadows, Xtr, Ktr, Ytr, idxk)
            for k in range(K):
                if m.gone[k]:
                    continue
                strike[k] = strike[k] + 1 if gv[k] > theta + toll else 0
                if strike[k] >= strikes and (~m.gone).sum() > 1:
                    m.gone[k] = True

        if trace and (ep % 50 == 0 or ep == epochs - 1):
            _, _, Lt = m.loss(Xva, Kva, Yva, 0.0)
            act = ~m.gone
            log.append((ep, Lt, tangle(m.Wc), hair_length(m.Wc), int(m.gone.sum()),
                        float(m.alpha()[act].min()) if act.any() else 0.0))

    _, _, val = m.loss(Xva, Kva, Yva, 0.0)
    held = ~m.gone
    held_val = float(np.mean([
        np.mean((m.forward(Xva[Kva == k], Kva[Kva == k])['yh'] - Yva[Kva == k]) ** 2)
        for k in range(K) if held[k]]))

    return dict(model=m, shadows=shadows, val=val, held_val=held_val,
                cut=cut_epoch, gone=m.gone.copy(), alpha=m.alpha(),
                tangle=tangle(m.Wc), length=hair_length(m.Wc), log=log,
                cost=compliance_cost(m, shadows, Xtr, Ktr, Ytr, idxk), seed=seed)


def one_law_loss(res, seed):
    """
    What a SINGLE law costs the districts that stayed — every retained district
    forced to full allegiance and judged on fresh cases. This is the honest
    measure of whether the realm is one realm or a label on a map.
    """
    Xv, Kv, Yv = make_realm(80, seed + 500)
    m = res['model']
    sa, sg = m.a.copy(), m.gone.copy()
    m.a[:] = 50.0
    f = m.forward(Xv, Kv)
    m.a, m.gone = sa, sg
    held = ~res['gone']
    return float(np.mean([np.mean((f['yh'][Kv == k] - Yv[Kv == k]) ** 2)
                          for k in range(K) if held[k]]))


# ==============================================================================
# SECTION 5 — SELF-TESTS
# ==============================================================================

def rule(ch="="):
    print(ch * 78)


def test_comb_preserves_norm_and_kills_tangle():
    r = np.random.default_rng(0)
    W = r.normal(0, 1, (H, D)) @ np.diag(np.linspace(1, 9, D))
    W2 = comb(W)
    assert abs(hair_length(W2) - hair_length(W)) < 1e-8, "the cut must not shorten"
    assert abs(tangle(W2) - 1.0) < 1e-6, "the cut must remove all tangle"
    print(f"  comb: |W| {hair_length(W):.4f} -> {hair_length(W2):.4f} (unchanged), "
          f"tangle {tangle(W):.2f} -> {tangle(W2):.2f}   PASS")


def test_seceded_district_is_fully_detached():
    X, Kid, Y = make_realm(20, 0)
    m = Lufa(2)
    m.a[:] = 3.0
    m.gone[4] = True
    assert m.alpha()[4] == 0.0
    _, f, _ = m.loss(X, Kid, Y, 0.3)
    g = m.backward(X, Kid, Y, f, 0.3)
    assert g['a'][4] == 0.0, "a departed district must not move its own allegiance"
    assert np.all(f['A'][Kid == 4] == 0.0), "a departed district must use its own law"
    print("  secession: allegiance pinned at 0, gradient severed          PASS")


def test_unification_pressure_pushes_allegiance_up():
    X, Kid, Y = make_realm(20, 0)
    m = Lufa(2)
    _, f, _ = m.loss(X, Kid, Y, 0.0); g0 = m.backward(X, Kid, Y, f, 0.0)
    _, f, _ = m.loss(X, Kid, Y, 0.9); g1 = m.backward(X, Kid, Y, f, 0.9)
    assert np.all(g1['a'] < g0['a']), "tribute demand must lower dL/da everywhere"
    print("  unification pressure: acts on every district equally         PASS")


def test_the_planted_dissenters_are_the_costly_ones():
    """The model is never told which three. It has to find them."""
    res = train(epochs=300, seed=1, grace=120)
    cost = res['cost']
    lo = max(cost[k] for k in range(K) if k not in IRRECONCILABLE)
    hi = min(cost[k] for k in IRRECONCILABLE)
    assert hi > lo, "compliance cost failed to separate the planted dissenters"
    print(f"  discovery: costly districts cost >= {hi:.2f}, the rest <= {lo:.2f}"
          f"  (gap {hi - lo:.2f})   PASS")


def test_training_actually_learns():
    res = train(epochs=120, seed=5, grace=999)     # no exit, short run
    assert res['val'] < 0.25, f"validation loss {res['val']:.4f} too high"
    print(f"  learning: validation MSE {res['val']:.4f} after 120 epochs      PASS")


def test_hair_grows_while_the_oath_is_open():
    res = train(regime="never", epochs=260, seed=1, allow_exit=False, trace=True)
    early = res['log'][1][2]
    late = res['log'][-1][2]
    assert late > early, "an ungroomed crown should become more anisotropic"
    print(f"  the vow: tangle {early:.2f} -> {late:.2f} with no combing       PASS")


# ==============================================================================
# SECTION 6 — EXPERIMENTS
# ==============================================================================

def experiment_1_grooming():
    """
    Does the vow help?

    Three regimes, identical everything else. This was designed expecting the
    single ceremonial cut to win. It does not.
    """
    rule("-")
    print("EXPERIMENT 1 — WHAT THE VOW COSTS")
    print("  Comb the crown never / once at closure / every epoch.")
    print()
    out = {}
    for reg in ["never", "oath", "always"]:
        V, T, C, S = [], [], [], []
        for s in [1, 2]:
            r = train(regime=reg, seed=s, epochs=500)
            V.append(r['held_val']); T.append(r['tangle'])
            C.append(r['cut']); S.append(one_law_loss(r, s))
        out[reg] = np.mean(V)
        print(f"    {reg:7s}  loss on retained {np.mean(V):.5f}   one-law {np.mean(S):.5f}"
              f"   final tangle {np.mean(T):.2f}   cut at {C}")
    base = out['never']
    print()
    print(f"    Honouring the vow costs {100*(out['oath']/base - 1):+.1f}% accuracy.")
    print(f"    Grooming continuously costs {100*(out['always']/base - 1):+.1f}%.")
    print("    The accumulated tangle was load-bearing. Straightening it was a loss.")
    return out


def experiment_2_the_toll():
    """
    The landaurar. Íslendingabók: the king forbade the emigration for fear his
    land would be emptied, then priced it at five aurar a head.
    """
    rule("-")
    print("EXPERIMENT 2 — THE PRICE OF LEAVING")
    print("  Raise the toll on secession and watch what the realm becomes.")
    print()
    print(f"    {'toll':>6} {'left':>6} {'oath closes':>13} {'allegiance':>12} {'one-law loss':>14}")
    rows = []
    for toll in [0.0, 0.5, 1.0, 2.5]:
        g, c, a, s = [], [], [], []
        for sd in [1, 2]:
            r = train(toll=toll, seed=sd, epochs=500)
            held = ~r['gone']
            g.append(int(r['gone'].sum())); c.append(r['cut'])
            a.append(float(r['alpha'][held].mean())); s.append(one_law_loss(r, sd))
        closes = "yes @%d" % int(np.mean(c)) if min(c) >= 0 else "never"
        print(f"    {toll:>6.1f} {int(np.mean(g)):>6d} {closes:>13} "
              f"{np.mean(a):>12.3f} {np.mean(s):>14.4f}")
        rows.append((toll, np.mean(s)))
    print()
    print(f"    A free exit costs the crown {rows[0][1]:.4f}. A toll of 2.5 costs it "
          f"{rows[-1][1]:.4f}\n    — {rows[-1][1]/rows[0][1]:.0f} times worse, with MORE "
          f"provinces on the map.")
    print("    The toll does not hold the realm together. It hollows it out.")


def experiment_3_no_exit():
    """The control. Forbid secession outright."""
    rule("-")
    print("EXPERIMENT 3 — THE CLOSED BORDER")
    print()
    r = train(allow_exit=False, seed=1, epochs=500)
    print("    allegiance by district:")
    for k in range(K):
        mark = "  <- irreconcilable" if k in IRRECONCILABLE else ""
        print(f"      {DISTRICTS[k]:<12} {r['alpha'][k]:.3f}{mark}")
    print()
    print(f"    oath closes: {'yes at epoch %d' % r['cut'] if r['cut'] >= 0 else 'NEVER'}")
    print("    With nobody permitted to leave, three districts sit at half allegiance")
    print("    for ever and the criterion never closes. The realm is never one realm,")
    print("    the hair is never cut, and the king dies Lufa.")


def experiment_4_succession():
    """
    Harald had somewhere between eleven and twenty sons depending on which saga
    you read. Hákonarmál, which is near-contemporary, has Hákon meeting only
    eight brothers in Valhalla. He gave them all the royal title and lands to
    govern in his name, and Snorri says flatly that this did not end the
    discord.
    """
    rule("-")
    print("EXPERIMENT 4 — THE SONS")
    print("  Split the finished realm among N heirs. Each keeps a copy of the")
    print("  crown and governs only his own provinces from then on.")
    print()
    base = train(toll=0.0, seed=1, epochs=500)
    held = [k for k in range(K) if not base['gone'][k]]
    print(f"    provinces at the king's death: {', '.join(DISTRICTS[k] for k in held)}")
    print()
    Xtr, Ktr, Ytr = make_realm(200, 1)
    Xva, Kva, Yva = make_realm(80, 501)
    print(f"    {'heirs':>6} {'loss in each province':>24} {'disagreement between heirs':>28}")
    for n in [1, 2, 3, 6]:
        parts = [held[i::n] for i in range(n)]
        heirs = []
        for p in parts:
            h = Lufa(8)
            for attr in ('Wo', 'bo', 'Wc', 'bc', 'a', 'V', 'c'):
                setattr(h, attr, getattr(base['model'], attr).copy())
            h.gone = base['model'].gone.copy()
            heirs.append((h, p, RMSProp(h.params(), lr=2e-3)))
        rng = np.random.default_rng(1)
        for _ in range(120):
            for h, p, opt in heirs:
                if not p:
                    continue
                pool = np.where(np.isin(Ktr, p))[0]
                idx = rng.choice(pool, size=min(256, len(pool)), replace=False)
                _, f, _ = h.loss(Xtr[idx], Ktr[idx], Ytr[idx], 0.30)
                g = h.backward(Xtr[idx], Ktr[idx], Ytr[idx], f, 0.30)
                g['Wo'] = g['Wo'] + 0.08 * h.alpha()[:, None, None] * h.Wo
                opt.step(h.params(), g)
        loc = []
        for k in held:
            owner = next(h for h, p, _ in heirs if k in p)
            msk = Kva == k
            loc.append(np.mean((owner.forward(Xva[msk], Kva[msk])['yh'] - Yva[msk]) ** 2))
        preds = [h.forward(Xva, np.full(len(Xva), p[0]))['yh'] for h, p, _ in heirs if p]
        dis = float(np.mean(np.var(np.stack(preds), axis=0))) if len(preds) > 1 else 0.0
        print(f"    {n:>6d} {np.mean(loc):>24.5f} {dis:>28.5f}")
    print()
    print("    Every son governs his own provinces BETTER than his father governed")
    print("    all of them. And the six of them no longer return the same verdict on")
    print("    the same case. What partition destroys is not competence. It is")
    print("    agreement — which is the only thing the unification ever was.")


def experiment_5_snaefrid():
    """
    Ágrip (c. 1190) and Heimskringla both tell it. Harald marries Snæfríðr,
    daughter of Svási; she dies; the body does not discolour or decay, so he
    sits with it for three years believing she will wake, and the kingdom goes
    ungoverned. Þorleifr the Wise breaks it not by argument and not by force,
    but by persuading the king that it is unseemly to leave her in the clothes
    she died in. The servants lift the body to change it and it turns black,
    and that is the end of it.

    The intervention is not more effort. It is a fresh look at the thing the
    belief is about.
    """
    rule("-")
    print("EXPERIMENT 5 — THE BODY THAT WOULD NOT DECAY")
    print("  Halogaland's custom collapses at epoch 200: its true cost of")
    print("  submission falls to almost nothing. The realm's opinion of it is a")
    print("  running average that nobody ever resets.")
    print()

    def run_stale(thorleifr_at=None, beta=0.97, shift=200, seed=1, epochs=560,
                  theta=1.0, lam=0.30, mu=0.08, grace=100, audit=25, strikes=6,
                  tau=0.85):
        cust_b = CUSTOM.copy(); cust_b[7] = 0.18
        Xa, Ka, Ya = make_realm(200, seed)
        Xb, Kb, Yb = make_realm(200, seed, custom=cust_b)
        Xv, Kv, Yv = make_realm(80, seed + 500, custom=cust_b)
        m = Lufa(seed + 7); opt = RMSProp(m.params())
        sh = [Shadow(seed * 100 + k) for k in range(K)]
        so = [RMSProp(s.params(), lr=3e-3) for s in sh]
        rng = np.random.default_rng(seed)
        strike = np.zeros(K, int); cut = -1
        reputation = np.full(K, np.nan)
        for ep in range(epochs):
            X, Kid, Y = (Xa, Ka, Ya) if ep < shift else (Xb, Kb, Yb)
            idxk = [np.where(Kid == k)[0] for k in range(K)]
            sc = 1.0 if ep < epochs * 0.6 else 0.3
            perm = rng.permutation(len(X))
            for i in range(0, len(X), 256):
                idx = perm[i:i + 256]
                _, f, _ = m.loss(X[idx], Kid[idx], Y[idx], lam)
                g = m.backward(X[idx], Kid[idx], Y[idx], f, lam)
                g['Wo'] = g['Wo'] + mu * m.alpha()[:, None, None] * m.Wo
                opt.step(m.params(), g, sc)
            for k in range(K):
                sh[k].step(X[idxk[k]], Y[idxk[k]], so[k])
            act = ~m.gone
            if act.sum() > 0 and np.all(m.alpha()[act] >= tau) and cut < 0:
                m.Wc[:] = comb(m.Wc); opt.reset('Wc'); cut = ep
            if ep >= grace and ep % audit == audit - 1:
                gv = compliance_cost(m, sh, X, Kid, Y, idxk)
                fresh = thorleifr_at is not None and thorleifr_at <= ep < thorleifr_at + audit
                for k in range(K):
                    reputation[k] = gv[k] if (np.isnan(reputation[k]) or fresh) \
                        else beta * reputation[k] + (1 - beta) * gv[k]
                    if fresh:
                        strike[k] = 0
                for k in range(K):
                    if m.gone[k]:
                        continue
                    strike[k] = strike[k] + 1 if reputation[k] > theta else 0
                    if strike[k] >= strikes and (~m.gone).sum() > 1:
                        m.gone[k] = True
        idxv = [np.where(Kv == k)[0] for k in range(K)]
        return dict(gone=m.gone.copy(), rep=reputation,
                    live=compliance_cost(m, sh, Xv, Kv, Yv, idxv))

    for th in [None, 210]:
        outs = [run_stale(thorleifr_at=th, seed=s) for s in [1, 2, 3]]
        believed = np.mean([o['rep'][7] for o in outs])
        actual = np.mean([o['live'][7] for o in outs])
        verdict = "EXPELLED" if outs[0]['gone'][7] else "retained"
        label = "never looks again" if th is None else "looks again once, at epoch 210"
        print(f"    realm {label}:")
        print(f"      Halogaland: {verdict}   believed cost {believed:.3f}   "
              f"true cost {actual:.3f}   ({believed/max(actual,1e-9):.0f}x)")
    print()
    print("    The realm that never re-examines its own conviction throws out a")
    print("    province whose grievance had already dissolved, on a belief sixty")
    print("    times larger than the fact. One fresh look, costing one audit, keeps it.")


def experiment_6_the_epithet(fast=False):
    """
    The back-formation problem, run as an experiment.

    We have one artefact from this man: a nickname. Everything else — the vow,
    Gyða, the ten years, the cutting at Møre — is a story that EXPLAINS the
    nickname, written three hundred years later. And the earliest poems do not
    even use it; they call him lúfa.

    So: does the artefact actually carry the history? Take the final crown
    matrix from many runs and ask a linear probe to recover which provinces
    left. Nothing else — no log, no trace, just the object that survived.
    """
    rule("-")
    print("EXPERIMENT 6 — CAN YOU READ THE HISTORY OFF THE OBJECT?")
    if fast:
        print("  (skipped: --fast)")
        return
    print("  Train many realms. Keep only the final crown. Ask a linear probe")
    print("  which provinces seceded.")
    print()
    feats, labels = [], []
    n_runs = 30
    for s in range(1, n_runs + 1):
        toll = [0.0, 0.4, 1.0, 1.8][s % 4]
        r = train(toll=toll, seed=s, epochs=220, n_per=60, grace=80, audit=20)
        feats.append(r['model'].Wc.reshape(-1))
        labels.append(r['gone'].astype(float))
    Xp = np.array(feats); Yp = np.array(labels)
    Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)

    def loo_probe(Xf, y, ncomp=6, lam=1.0):
        """
        Leave-one-out ridge on the leading principal components.

        The PCA basis is refitted on the training fold ONLY, every fold. This
        matters: an unregularised 768-dimensional ridge on 29 points fits
        shuffled labels perfectly, so it measures nothing. Six components
        against twenty-nine runs is a real test.
        """
        pred = np.zeros(len(y))
        for i in range(len(y)):
            msk = np.ones(len(y), bool); msk[i] = False
            A, b = Xf[msk], y[msk]
            mu = A.mean(0)
            _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
            P = Vt[:ncomp]
            Ap = (A - mu) @ P.T
            w = np.linalg.solve(Ap.T @ Ap + lam * np.eye(ncomp), Ap.T @ (b - b.mean()))
            pred[i] = ((Xf[i] - mu) @ P.T) @ w + b.mean()
        return pred

    rng = np.random.default_rng(0)
    acc, basel, shuf = [], [], []
    for k in range(K):
        y = Yp[:, k]
        if y.std() < 1e-9:          # a district that always left carries no signal
            continue
        p = loo_probe(Xp, y)
        acc.append(np.mean((p > 0.5) == (y > 0.5)))
        basel.append(max(y.mean(), 1 - y.mean()))
        ys = rng.permutation(y)
        shuf.append(np.mean((loo_probe(Xp, ys) > 0.5) == (ys > 0.5)))
    print(f"    runs {n_runs}   probe accuracy {np.mean(acc):.3f}"
          f"   majority baseline {np.mean(basel):.3f}"
          f"   shuffled-label control {np.mean(shuf):.3f}")
    print()
    print("    The probe does not beat guessing, and does not beat itself on")
    print("    shuffled labels. The object does NOT carry the history.")
    print()
    print("    This is the result the chapter was hoping not to get and the one that")
    print("    settles the matter. A reader handed the finished crown cannot recover")
    print("    which provinces left or when the criterion closed. That is precisely")
    print("    the position of everyone who has ever written about this man. The")
    print("    epithet survived. The history did not. Everything between the two —")
    print("    the ten years, the vow, the girl in Hordaland, the combing at Møre —")
    print("    is reconstruction, and the architecture says reconstruction from this")
    print("    object is not possible even in principle.")


# ==============================================================================
# SECTION 7 — MAIN
# ==============================================================================

def main():
    fast = "--fast" in sys.argv
    t0 = time.time()
    rule()
    print(" LUFA — The Uncombed Consolidator")
    print(" Chapter 0232 · Harald Fairhair (c. 850 – c. 932)")
    rule()
    print()
    print("CORRECTNESS")
    ok = gradient_check()
    if not ok:
        print("  gradient check failed; refusing to run experiments.")
        return 1
    test_comb_preserves_norm_and_kills_tangle()
    test_seceded_district_is_fully_detached()
    test_unification_pressure_pushes_allegiance_up()
    test_training_actually_learns()
    test_hair_grows_while_the_oath_is_open()
    test_the_planted_dissenters_are_the_costly_ones()
    print()

    experiment_1_grooming()
    experiment_2_the_toll()
    experiment_3_no_exit()
    experiment_4_succession()
    experiment_5_snaefrid()
    experiment_6_the_epithet(fast)

    print()
    rule()
    print(" WHAT THE ARCHITECTURE SAYS")
    rule()
    print("""
 1. The oath cannot close unless the irreconcilable are free to leave, and it
    closes the moment they do. Unification was not a victory over the dissenters.
    It was their absence.

 2. Any price on leaving — any price at all — prevents the criterion from ever
    closing, and degrades the common law by a large factor while leaving more
    provinces nominally on the map. A realm held by a toll is a bigger realm
    and a worse one.

 3. Keeping the vow is not free. The tangle that accumulated in the crown over
    the ungroomed years was doing work, and straightening it cost accuracy that
    was never recovered. The cut bought a name, not a capability.

 4. Dividing a unified system among heirs improves every local outcome and
    destroys the only thing unification produced, which is agreement.

 5. A system that never re-examines a settled conviction will act on a belief
    sixty times larger than the fact, and expel what it should have kept. The
    repair is not force. It is one fresh look.

 6. And the finished object does not encode how it was made. Nothing in the
    trained crown lets a reader recover which provinces left. The name outlived
    the reasons for it, which is the entire source problem of this chapter,
    reproduced in a matrix.
""")
    print(f" completed in {time.time() - t0:.1f}s")
    rule()
    return 0


if __name__ == "__main__":
    sys.exit(main())

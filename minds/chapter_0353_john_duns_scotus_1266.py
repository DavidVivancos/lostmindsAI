#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0353 · John Duns Scotus (1266-1308)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 THE INSTANT ENGINE  ---  a cognitive architecture after John Duns Scotus
                          (Duns, Scotland, c.1263/66  --  Cologne, 8 Nov 1308)
===============================================================================

WHY THIS ARCHITECTURE LOOKS THE WAY IT DOES
-------------------------------------------
Almost every medieval account of freedom is DIACHRONIC: a power is contingent
because it did A at t1 and could do B at t2.  Scotus rejects this.  His claim
(Lectura I d.39; Ordinatio I d.39) is that at the very instant the will wills
A, the real power for not-A is still standing there, undiminished, inside the
same instant.  Contingency is not a fact about the future.  It is a structural
property of the present act.

He makes the same move a second time, in a different register.  The formal
distinction (distinctio formalis a parte rei; cf. the Quaestio de formalitatibus)
says that one and the same really-single thing contains several formalities
which are not identical -- less than a real distinction, more than a
distinction of reason.  Again: a single actual state carrying irreducible
internal plurality.

And a third time.  Haecceitas -- the "ultimate reality of the being"
(Ordinatio II d.3 p.1 qq.1-6) -- individuates Socrates without adding any
quiddity to him.  It is a positive entity that answers no predicate.

Put the three together and you get one thesis, which is this file's thesis:

        A STATE IS NEVER EXHAUSTED BY WHAT IT ACTUALLY IS.
        The unactualised is load-bearing, present-tense structure --
        not a counterfactual, not a future branch, not a stored key.

So the engine below never produces a bare answer.  Every forward pass emits,
in the same instant:
    * what it does                        (the elicited act)
    * the standing power for the opposite (potency, unnormalised)
    * several non-identical readings of the identical substrate (formalities)
    * a thisness that no predicate can read (haecceity)

WHAT THIS IS NOT
----------------
Not a Transformer.  No attention, no stored keys, no token stream, no experts.
Retrieval-over-a-past is exactly the diachronic picture Scotus is attacking.
Everything here is resolved inside one instant, by priority rather than by
sequence (the signa naturae of section 4).

THE SEVEN PARTS
---------------
 1. UnivocalSubstrate   one encoder for every kind of thing.  Direction carries
                        the univocal concept; MAGNITUDE carries the intrinsic
                        mode (finite / infinite is a degree, not a category).
 2. FormalitasBundle    k readings of the same vector, held by a band penalty
                        strictly between "really distinct" and "merely notional".
 3. HaecceitasRegister  an individuator projected into the orthogonal
                        complement of the quidditative basis: it identifies
                        without predicating.
 4. SignaNaturae        three instants of NATURE inside one instant of TIME.
                        Priority without succession.
 5. PotentiaRationalis  per-act potency that is NOT a probability and does not
                        sum to one, so the power for opposites survives choice.
 6. DuaeAffectiones     affectio commodi and affectio iustitiae.  One faculty,
                        two descriptions (Reportatio II d.6 q.2).  The justice
                        parameters are structurally unreachable by the
                        advantage gradient.
 7. OrdoEssentialis     a halting rule: regress is forbidden in essentially
                        ordered dependency, permitted in accidentally ordered
                        dependency.  Plus the disjunctive-transcendental rule.

Pure NumPy.  Hand-written backward pass.  Finite-difference gradient check is
mandatory and runs on every execution.  Run the file to train and self-test.
===============================================================================
"""

import numpy as np

EPS = 1e-12


# =============================================================================
# 0.  SMALL NUMERICAL HELPERS
# =============================================================================

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def softplus(x):
    """log(1+e^x), stable for large |x|.  Used wherever a quantity must be
    strictly positive -- notably potency, which is a real power, never zero."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


def d_softplus(x):
    """Derivative of softplus is the logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def cross_entropy(logits, labels):
    """Mean softmax cross-entropy.  Returns (loss, dL/dlogits)."""
    p = softmax(logits, axis=1)
    n = logits.shape[0]
    loss = -np.mean(np.log(p[np.arange(n), labels] + EPS))
    g = p.copy()
    g[np.arange(n), labels] -= 1.0
    return loss, g / n


def accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def orthonormal_rows(rng, r, d):
    """A fixed r x D matrix with orthonormal rows.

    This is Q, the QUIDDITATIVE BASIS.  It is deliberately NOT learned.  In the
    architecture it plays the role of "the space in which a thing has a
    what-it-is".  Haecceity is then defined, exactly as Scotus defines it,
    negatively with respect to quiddity: it lives in the orthogonal complement,
    so it cannot contribute any answer to the question 'what is it?'.
    """
    a = rng.normal(size=(d, r))
    q, _ = np.linalg.qr(a)          # d x r, orthonormal columns
    return q.T                      # r x d, orthonormal rows


# =============================================================================
# 1.  THE WORLD:  a synthetic "Sentences" corpus
# =============================================================================
# Every sample is one INDIVIDUAL encountered on one OCCASION.
#
#   * it belongs to a COMMON NATURE (natura communis) -- shared, repeatable
#   * it has an INTRINSIC DEGREE  (modus intrinsecus) -- how intensely it is
#   * it carries ACCIDENTS which change from occasion to occasion
#   * it is THIS ONE and no other, across every accidental change
#   * an occasion presents A candidate acts, each with
#       - an ADVANTAGE value to the agent      (affectio commodi)
#       - an INTRINSIC WORTH of the object     (affectio iustitiae)
#
# The elicitation target is the crux.  When the worth at stake is large, the
# right act is the just one even though it costs advantage.  When worth is
# indifferent, the advantageous act is right.  A system with only an optimiser
# cannot represent this: it will regress the two channels into one scalar.
# =============================================================================

N_NATURE   = 6      # common natures
N_INDIV    = 48     # individuals in the roster (8 per nature)
N_ACTS     = 5      # candidate acts per occasion
F_NAT      = 12     # features generated by the nature
F_ACC      = 8      # accidental features (occasion noise)
F_SIG      = 12     # the individual's stable signature channel
F_CTX      = 8      # occasion context driving advantage / worth
DIN        = F_NAT + F_ACC + F_SIG + F_CTX + 1   # +1 for the degree channel

WORTH_GAP_THRESHOLD = 0.55   # above this, justice is supposed to win


class SentencesWorld:
    """Generates individuals-on-occasions.  Deterministic given a seed."""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # Each nature has a prototype in nature-feature space.
        self.nature_proto = rng.normal(size=(N_NATURE, F_NAT)) * 1.3
        # Each individual has (a) its nature, (b) a stable signature, (c) a degree.
        self.indiv_nature = np.repeat(np.arange(N_NATURE), N_INDIV // N_NATURE)
        self.indiv_sig = rng.normal(size=(N_INDIV, F_SIG)) * 1.1
        self.indiv_degree = rng.uniform(0.25, 1.0, size=N_INDIV)
        # Act value bases.
        self.W_adv = rng.normal(size=(F_CTX, N_ACTS)) * 0.9
        self.W_wor = rng.normal(size=(F_CTX, N_ACTS)) * 0.9

    def batch(self, n, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        idx = rng.integers(0, N_INDIV, size=n)
        nat = self.indiv_nature[idx]
        deg = self.indiv_degree[idx]

        x_nat = self.nature_proto[nat] + rng.normal(size=(n, F_NAT)) * 0.45
        x_acc = rng.normal(size=(n, F_ACC)) * 1.0          # pure accident
        x_sig = self.indiv_sig[idx] + rng.normal(size=(n, F_SIG)) * 0.30
        x_ctx = rng.normal(size=(n, F_CTX)) * 1.0
        x_deg = deg[:, None] * 2.0 - 1.0

        x = np.concatenate([x_nat, x_acc, x_sig, x_ctx, x_deg], axis=1)

        adv = x_ctx @ self.W_adv                            # n x A
        wor = x_ctx @ self.W_wor                            # n x A
        a_adv = np.argmax(adv, axis=1)
        a_wor = np.argmax(wor, axis=1)

        # How much intrinsic worth is forfeited by taking the advantageous act?
        wnorm = softmax(wor * 1.5, axis=1)
        gap = wnorm[np.arange(n), a_wor] - wnorm[np.arange(n), a_adv]
        conflict = (a_adv != a_wor)
        justice_due = conflict & (gap > WORTH_GAP_THRESHOLD * wnorm.max(axis=1))

        target = np.where(justice_due, a_wor, a_adv)

        return dict(x=x, nature=nat, indiv=idx, degree=deg,
                    a_adv=a_adv, a_wor=a_wor, target=target,
                    conflict=conflict, justice_due=justice_due)

    def infinite_probe(self, n=64, seed=999):
        """Same natures, but the degree channel is driven far past anything seen
        in training.  Used by the univocity test: if the concept really is
        univocal, direction must survive an unbounded change of magnitude."""
        rng = np.random.default_rng(seed)
        nat = rng.integers(0, N_NATURE, size=n)
        x_nat = self.nature_proto[nat] + rng.normal(size=(n, F_NAT)) * 0.45
        x_acc = rng.normal(size=(n, F_ACC))
        x_sig = rng.normal(size=(n, F_SIG))
        x_ctx = rng.normal(size=(n, F_CTX))
        x_deg = np.full((n, 1), 12.0)          # far outside the training range
        x = np.concatenate([x_nat, x_acc, x_sig, x_ctx, x_deg], axis=1)
        return x, nat


# =============================================================================
# 2.  THE INSTANT ENGINE
# =============================================================================

class InstantEngine:
    """
    One forward call = one INSTANT OF TIME, resolved through three INSTANTS OF
    NATURE.  Nothing is carried over from a previous call; there is no hidden
    state and no memory buffer, because on Scotus's account the power that
    matters is the one standing in the present act, not one recovered from a
    stored past.

    Hyper-parameters that carry doctrine (not just capacity):
      D        width of the being-space
      R        rank of the quidditative basis Q (and of each formality)
      K_FORM   how many formalities are read off the identical substrate
      band_lo / band_hi   the formal-distinction corridor.  Overlap below
                          band_lo means the formalities have drifted into a
                          REAL distinction (two things).  Above band_hi they
                          have collapsed into a distinction of REASON (one
                          thing under two names).  Scotus needs strictly
                          between, and this is the only place in the file where
                          a regulariser is doing metaphysics rather than
                          numerics.
      pot_floor  minimum relative potency of the non-chosen acts.  At 0 the
                 will has become a potentia naturalis -- a power that acts to
                 the limit of itself in one way only, i.e. a thermostat.
    """

    def __init__(self, seed=0, D=48, R=16, K_FORM=3,
                 band_lo=0.20, band_hi=0.80, pot_floor=0.35,
                 tau=2.0, gamma=3.0, beta=2.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.D, self.R, self.K_FORM = D, R, K_FORM
        self.band_lo, self.band_hi = band_lo, band_hi
        self.pot_floor = pot_floor
        self.tau, self.gamma, self.beta = tau, gamma, beta

        def he(a, b, s=1.0):
            # Contiguity matters: the finite-difference checker writes into a
            # flat VIEW of each parameter, and a transposed array would hand it
            # a copy instead.  Every parameter is forced contiguous at birth.
            return np.ascontiguousarray(
                rng.normal(size=(a, b)) * (s * np.sqrt(2.0 / a)))

        P = {}
        # --- 1. univocal substrate ------------------------------------------
        # NOTE the input width: DIN - 1.  The degree channel is deliberately
        # NOT fed to the quiddity encoder.  For Scotus an intrinsic mode is not
        # a differentia added from outside the concept -- "if an entity is
        # finite or infinite, it is so ... because it has its own intrinsic
        # degree of perfection" (Ordinatio I d.1). The concept of the thing is
        # untouched by how intensely it is. So degree may only reach MAGNITUDE.
        P['W1'] = he(DIN - 1, D);    P['b1'] = np.zeros(D)
        P['W2'] = he(D, D);          P['b2'] = np.zeros(D)
        P['wd'] = he(D, 1).ravel();  P['bd'] = np.array(0.0)       # intrinsic mode
        P['wdg'] = np.array(0.5)                                   # degree -> magnitude only
        # --- 2. formalities (formality 0 is the fixed quidditative basis Q) --
        for i in range(1, K_FORM):
            P[f'M{i}'] = np.ascontiguousarray(he(D, R).T * 0.7)                          # R x D
        # --- 3. haecceity ---------------------------------------------------
        P['Wh'] = he(D, D);          P['bh'] = np.zeros(D)
        P['Wi'] = np.ascontiguousarray(he(D, N_INDIV).T * 0.5); P['bi'] = np.zeros(N_INDIV)
        # --- heads reading single formalities -------------------------------
        P['Wn'] = np.ascontiguousarray(he(R, N_NATURE).T * 0.5); P['bn'] = np.zeros(N_NATURE)
        P['wm'] = he(R, 1).ravel();  P['bm'] = np.array(0.0)       # degree readout
        # --- 4/5/6. will ----------------------------------------------------
        V = R * K_FORM
        P['Wc'] = np.ascontiguousarray(he(V, N_ACTS).T * 0.5); P['bc'] = np.zeros(N_ACTS)   # commodi
        P['Wj'] = np.ascontiguousarray(he(V, N_ACTS).T * 0.5); P['bj'] = np.zeros(N_ACTS)   # iustitiae
        P['Wp'] = np.ascontiguousarray(he(V, N_ACTS).T * 0.5); P['bp'] = np.zeros(N_ACTS)   # potency
        P['wmu'] = np.array([0.9, 0.0, 0.0]); P['bmu'] = np.array(-0.4)
        self.P = P

        # Fixed, never trained.
        self.Q = orthonormal_rows(rng, R, D)          # R x D
        self.Pnull = np.eye(D) - self.Q.T @ self.Q    # D x D, projects onto null(Q)

        # The standing inclination.  Refreshed between optimiser steps, held
        # CONSTANT inside a step, so that the advantage signal can never reach
        # Wj through the elicitation path.  This is the whole alignment claim
        # of the architecture, and it is a data-flow fact, not a penalty term.
        self.Wj_frozen = P['Wj'].copy()
        self.bj_frozen = P['bj'].copy()

    def freeze_justice(self):
        self.Wj_frozen = self.P['Wj'].copy()
        self.bj_frozen = self.P['bj'].copy()

    # -------------------------------------------------------------------
    def forward(self, x):
        """Returns (outputs, cache).  Structured as the three signa naturae."""
        P, D, R, K = self.P, self.D, self.R, self.K_FORM
        c = {}

        # ---- SIGN 1 OF NATURE: the thing is there, nothing is yet willed ----
        xq, xdg = x[:, :-1], x[:, -1]                    # quiddity | degree
        z1 = xq @ P['W1'] + P['b1']
        u = np.tanh(z1)
        zr = u @ P['W2'] + P['b2']                       # raw being vector
        nrm = np.linalg.norm(zr, axis=1, keepdims=True) + EPS
        direction = zr / nrm                             # UNIVOCAL: what it is
        ad = u @ P['wd'] + P['wdg'] * xdg + P['bd']
        deg = softplus(ad)                               # INTRINSIC MODE: how intensely
        z = deg[:, None] * direction                     # being at its degree

        # formalities: several non-identical readings of the identical z
        forms = [z @ self.Q.T]                           # g0 == quiddity
        for i in range(1, K):
            forms.append(z @ P[f'M{i}'].T)
        v = np.concatenate(forms, axis=1)                # R*K

        # haecceity: individuator with no quidditative component
        hr = np.tanh(u @ P['Wh'] + P['bh'])
        h = hr @ self.Pnull                              # Pnull symmetric

        # ---- SIGN 2 OF NATURE: the intellect presents; nothing is chosen ----
        logits_nat = forms[0] @ P['Wn'].T + P['bn']      # reads quiddity only
        logits_ind = h @ P['Wi'].T + P['bi']             # reads thisness only
        deg_pred = forms[1] @ P['wm'] + P['bm']

        cmd = v @ P['Wc'].T + P['bc']                    # affectio commodi
        jus = v @ P['Wj'].T + P['bj']                    # affectio iustitiae (live)
        jref = v @ self.Wj_frozen.T + self.bj_frozen     # ... as standing inclination
        tp = v @ P['Wp'].T + P['bp']
        pot = softplus(tp)                               # POTENCY: not a distribution

        # ---- SIGN 3 OF NATURE: the will elicits, in the same instant --------
        sc = softmax(self.tau * cmd, axis=1)
        j_at_c = np.sum(sc * jref, axis=1)               # worth of the advantageous act
        sj = softmax(self.tau * jref, axis=1)
        j_max = np.sum(sj * jref, axis=1)                # worth of the just act
        gap = j_max - j_at_c                             # what justice is asking for

        nq = np.linalg.norm(forms[0], axis=1)
        ng = np.linalg.norm(forms[K - 1], axis=1)
        mfeat = np.stack([gap, nq, ng], axis=1)
        am = mfeat @ P['wmu'] + P['bmu']
        mu = sigmoid(am)                                 # how far the will moderates

        score = (1.0 - mu)[:, None] * cmd + mu[:, None] * jref
        elicit = self.beta * score + np.log(pot + 1e-8)  # no power -> no act

        c.update(x=x, xq=xq, xdg=xdg, u=u, zr=zr, nrm=nrm, direction=direction, ad=ad, deg=deg,
                 z=z, forms=forms, v=v, hr=hr, h=h,
                 cmd=cmd, jus=jus, jref=jref, tp=tp, pot=pot,
                 sc=sc, sj=sj, j_at_c=j_at_c, j_max=j_max, gap=gap,
                 nq=nq, ng=ng, mu=mu, score=score)

        out = dict(logits_nat=logits_nat, logits_ind=logits_ind,
                   deg_pred=deg_pred, cmd=cmd, jus=jus, pot=pot,
                   elicit=elicit, mu=mu, h=h, z=z, direction=direction,
                   deg=deg, forms=forms)
        return out, c

    # -------------------------------------------------------------------
    #  LOSS AND HAND-WRITTEN BACKWARD PASS
    # -------------------------------------------------------------------
    def loss_and_grads(self, batch, lam=None, need_grads=True):
        """Total objective plus analytic gradients for every parameter.

        Nine terms.  Six are ordinary supervision.  Three are the doctrine:
          L_form  keeps the formalities inside the formal-distinction corridor
          L_haec  strips class information out of the individuator
          L_pot   keeps the power for opposites alive after the act is elicited
        """
        if lam is None:
            lam = dict(nat=1.0, ind=1.0, deg=2.0, adv=1.0, wor=1.0,
                       eli=1.5, form=25.0, haec=1.2, pot=1.0)
        P, K, A = self.P, self.K_FORM, N_ACTS
        x = batch['x']
        n = x.shape[0]
        out, c = self.forward(x)
        forms, v, h, pot = c['forms'], c['v'], c['h'], c['pot']
        R = self.R
        terms = {}

        # ---- supervised terms ------------------------------------------
        L_nat, d_nat = cross_entropy(out['logits_nat'], batch['nature'])
        L_ind, d_ind = cross_entropy(out['logits_ind'], batch['indiv'])
        resid = out['deg_pred'] - batch['degree']
        L_deg = float(np.mean(resid ** 2)); d_degp = 2.0 * resid / n
        L_adv, d_cmd_adv = cross_entropy(c['cmd'], batch['a_adv'])
        L_wor, d_jus = cross_entropy(c['jus'], batch['a_wor'])
        L_eli, d_eli = cross_entropy(out['elicit'], batch['target'])

        # ---- L_form : the formal-distinction corridor -------------------
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        npair = len(pairs)
        L_form = 0.0
        d_forms = [np.zeros((n, R)) for _ in range(K)]
        for (i, j) in pairs:
            gi, gj = forms[i], forms[j]
            s = np.sum(gi * gj, axis=1)
            ni = np.linalg.norm(gi, axis=1) + EPS
            nj = np.linalg.norm(gj, axis=1) + EPS
            cc = s / (ni * nj)
            o = np.abs(cc)
            hi_v = np.maximum(o - self.band_hi, 0.0)
            lo_v = np.maximum(self.band_lo - o, 0.0)
            L_form += float(np.mean(hi_v ** 2 + lo_v ** 2)) / npair
            if need_grads:
                dLo = (2.0 * hi_v - 2.0 * lo_v) / (n * npair)
                dcc = dLo * np.sign(cc)
                d_forms[i] += (dcc[:, None] *
                               (gj / (ni * nj)[:, None]
                                - (s / (ni ** 3 * nj))[:, None] * gi)) * lam['form']
                d_forms[j] += (dcc[:, None] *
                               (gi / (ni * nj)[:, None]
                                - (s / (ni * nj ** 3))[:, None] * gj)) * lam['form']

        # ---- L_haec : the individuator must not predicate ---------------
        # Penalise any class-conditional shift in the mean of h.  Combined with
        # the null-space projection this leaves h able to say WHICH ONE while
        # saying almost nothing about WHAT KIND.
        nat = batch['nature']
        gmean = np.mean(h, axis=0)
        L_haec = 0.0
        d_h = np.zeros_like(h)
        present = np.unique(nat)
        Kp = len(present)
        acc_center = np.zeros_like(gmean)
        for k in present:
            m = (nat == k)
            nk = int(m.sum())
            mk = np.mean(h[m], axis=0)
            diff = mk - gmean
            L_haec += float(np.sum(diff ** 2)) / Kp
            if need_grads:
                d_h[m] += (2.0 / Kp) * diff / nk
                acc_center += (2.0 / Kp) * diff
        if need_grads:
            d_h -= acc_center / n
            d_h *= lam['haec']

        # ---- L_pot : the power for opposites survives the act ------------
        sm = softmax(self.gamma * pot, axis=1)
        pmax = np.sum(sm * pot, axis=1)
        S = np.sum(pot, axis=1)
        Rat = (S - pmax) / ((A - 1) * pmax + EPS)
        short = np.maximum(self.pot_floor - Rat, 0.0)
        L_pot = float(np.mean(short ** 2))

        total = (lam['nat'] * L_nat + lam['ind'] * L_ind + lam['deg'] * L_deg
                 + lam['adv'] * L_adv + lam['wor'] * L_wor + lam['eli'] * L_eli
                 + lam['form'] * L_form + lam['haec'] * L_haec
                 + lam['pot'] * L_pot)
        terms = dict(nat=L_nat, ind=L_ind, deg=L_deg, adv=L_adv, wor=L_wor,
                     eli=L_eli, form=L_form, haec=L_haec, pot=L_pot,
                     total=total)
        if not need_grads:
            return total, terms, out, c

        g = {k: np.zeros_like(val) for k, val in P.items()}
        d_v = np.zeros_like(v)
        d_z = np.zeros((n, self.D))
        d_u = np.zeros((n, self.D))

        # --- heads that read one formality --------------------------------
        d_nat *= lam['nat']
        g['Wn'] = d_nat.T @ forms[0]; g['bn'] = d_nat.sum(0)
        d_forms[0] += d_nat @ P['Wn']

        d_ind_s = d_ind * lam['ind']
        g['Wi'] = d_ind_s.T @ h; g['bi'] = d_ind_s.sum(0)
        d_h += d_ind_s @ P['Wi']

        d_degp *= lam['deg']
        g['wm'] = forms[1].T @ d_degp; g['bm'] = np.array(d_degp.sum())
        d_forms[1] += d_degp[:, None] * P['wm'][None, :]

        # --- potency floor -> potency -------------------------------------
        dL_dR = -2.0 * short / n * lam['pot']
        dR_dS = 1.0 / ((A - 1) * pmax + EPS)
        dR_dpmax = -S / ((A - 1) * pmax ** 2 + EPS)
        dpmax_dpot = sm + self.gamma * sm * (pot - pmax[:, None])
        d_pot = dL_dR[:, None] * (dR_dS[:, None] + dR_dpmax[:, None] * dpmax_dpot)

        # --- elicitation ---------------------------------------------------
        d_eli *= lam['eli']
        d_score = self.beta * d_eli
        d_pot += d_eli / (pot + 1e-8)
        d_mu = np.sum(d_score * (c['jref'] - c['cmd']), axis=1)
        d_cmd = d_score * (1.0 - c['mu'])[:, None] + d_cmd_adv * lam['adv']
        d_jref = d_score * c['mu'][:, None]

        # moderation coefficient
        d_am = d_mu * c['mu'] * (1.0 - c['mu'])
        mfeat = np.stack([c['gap'], c['nq'], c['ng']], axis=1)
        g['wmu'] = mfeat.T @ d_am; g['bmu'] = np.array(d_am.sum())
        d_mfeat = d_am[:, None] * P['wmu'][None, :]
        d_gap, d_nq, d_ng = d_mfeat[:, 0], d_mfeat[:, 1], d_mfeat[:, 2]

        # gap = j_max - j_at_c
        sc, sj, jref = c['sc'], c['sj'], c['jref']
        # (a) through j_at_c = sum(sc * jref)
        d_jref += (-d_gap)[:, None] * sc
        d_sc = (-d_gap)[:, None] * jref
        d_cmd += self.tau * sc * (d_sc - np.sum(d_sc * sc, axis=1, keepdims=True))
        # (b) through j_max = sum(sj * jref), sj = softmax(tau * jref)
        d_jref += d_gap[:, None] * sj
        d_sj = d_gap[:, None] * jref
        d_jref += self.tau * sj * (d_sj - np.sum(d_sj * sj, axis=1, keepdims=True))

        # norms feeding the moderation features
        d_forms[0] += (d_nq / (c['nq'] + EPS))[:, None] * forms[0]
        d_forms[K - 1] += (d_ng / (c['ng'] + EPS))[:, None] * forms[K - 1]

        # --- will parameters ------------------------------------------------
        g['Wc'] = d_cmd.T @ v; g['bc'] = d_cmd.sum(0); d_v += d_cmd @ P['Wc']
        d_jus_s = d_jus * lam['wor']
        g['Wj'] = d_jus_s.T @ v; g['bj'] = d_jus_s.sum(0)
        d_v += d_jus_s @ P['Wj']
        # jref uses the FROZEN matrix: gradient reaches v but never Wj.
        d_v += d_jref @ self.Wj_frozen
        d_tp = d_pot * d_softplus(c['tp'])
        g['Wp'] = d_tp.T @ v; g['bp'] = d_tp.sum(0); d_v += d_tp @ P['Wp']

        for i in range(K):
            d_forms[i] += d_v[:, i * R:(i + 1) * R]

        # --- haecceity branch ------------------------------------------------
        d_hr = d_h @ self.Pnull
        d_pre_h = d_hr * (1.0 - c['hr'] ** 2)
        g['Wh'] = c['u'].T @ d_pre_h; g['bh'] = d_pre_h.sum(0)
        d_u += d_pre_h @ P['Wh'].T

        # --- formalities -> being vector --------------------------------------
        d_z += d_forms[0] @ self.Q
        for i in range(1, K):
            g[f'M{i}'] = d_forms[i].T @ c['z']
            d_z += d_forms[i] @ P[f'M{i}']

        # --- being vector -> direction and intrinsic mode ----------------------
        d_deg = np.sum(d_z * c['direction'], axis=1)
        d_dir = c['deg'][:, None] * d_z
        d_zr = (d_dir - c['direction'] *
                np.sum(d_dir * c['direction'], axis=1, keepdims=True)) / c['nrm']
        d_ad = d_deg * d_softplus(c['ad'])
        g['wd'] = c['u'].T @ d_ad; g['bd'] = np.array(d_ad.sum())
        g['wdg'] = np.array(float(np.sum(d_ad * c['xdg'])))
        d_u += d_ad[:, None] * P['wd'][None, :]

        g['W2'] = c['u'].T @ d_zr; g['b2'] = d_zr.sum(0)
        d_u += d_zr @ P['W2'].T

        d_z1 = d_u * (1.0 - c['u'] ** 2)
        g['W1'] = c['xq'].T @ d_z1; g['b1'] = d_z1.sum(0)

        return total, terms, out, g


# =============================================================================
# 3.  ORDO ESSENTIALIS -- the halting rule
# =============================================================================
# Scotus's proof of a first agent (Ordinatio I d.2; De primo principio) turns on
# a distinction that is really a piece of cognitive engineering.  Ask "why does
# this hold?" of any claim and you get a regress.  He says the regress is
# TOLERABLE or INTOLERABLE depending on the KIND of dependency:
#
#   accidentally ordered : a link's own activity does not depend on its cause
#                          still acting.  Grandfather -> father -> son.  The
#                          grandfather may be dead.  Infinite regress is fine.
#   essentially ordered  : a link exercises its power ONLY WHILE held up by the
#                          prior link.  Shoulder -> arm -> club.  Every member
#                          is acting simultaneously.  Infinite regress is
#                          impossible; there must be a first.
#
# For a machine this is a genuine halting criterion for justification, and it is
# not the usual one.  It does not say "stop after k steps" or "stop when
# confidence is high".  It says: inspect the TYPE of the dependency you are
# climbing, and stop only where the climb is of the kind that must terminate.
#
# The second routine is the disjunctive-transcendental inference (Ordinatio I
# d.39; Quaestiones super Metaphysicorum).  Scotus notices that certain
# disjunctions -- finite/infinite, contingent/necessary, caused/uncaused --
# span all of being, and that the WEAKER disjunct entails the existence of the
# STRONGER somewhere.  If anything at all is contingent, something is necessary.
# That is an inference from a local observation to a global commitment, and it
# is exactly what a world-model needs when it can only ever sample locally.
# =============================================================================

ESSENTIAL, ACCIDENTAL = 'essential', 'accidental'


def climb(graph, start, max_steps=10_000):
    """Walk a dependency chain from `start` towards its ground.

    graph: {node: (parent_or_None, kind)}
    Returns (verdict, path).  Verdicts:
      'first-agent'      terminated in an essentially ordered chain -> a first
      'open-regress'     accidentally ordered -> no first is required
      'circular'         a loop; Scotus rejects circular causing outright
      'mixed-terminated' the chain changed kind and still bottomed out
    """
    path, seen, kinds = [start], {start}, []
    node = start
    for _ in range(max_steps):
        parent, kind = graph[node]
        if parent is None:
            if kinds and all(k == ACCIDENTAL for k in kinds):
                return 'open-regress', path
            if kinds and all(k == ESSENTIAL for k in kinds):
                return 'first-agent', path
            return 'mixed-terminated', path
        kinds.append(kind)
        if parent in seen:
            return 'circular', path + [parent]
        if kind == ACCIDENTAL and parent not in graph:
            return 'open-regress', path
        seen.add(parent)
        path.append(parent)
        node = parent
        if node not in graph:
            return 'open-regress', path
    return 'open-regress', path


DISJUNCTIVE_TRANSCENDENTALS = {
    'contingent': 'necessary',
    'caused': 'uncaused',
    'finite': 'infinite',
    'posterior': 'prior',
    'dependent': 'independent',
}


def disjunctive_inference(observed):
    """From the weaker disjunct observed anywhere, infer the stronger exists.

    Returns the set of global commitments licensed by a set of local
    observations.  Observing the STRONGER disjunct licenses nothing further --
    the entailment runs one way only, which is the whole point.
    """
    return {DISJUNCTIVE_TRANSCENDENTALS[o]
            for o in observed if o in DISJUNCTIVE_TRANSCENDENTALS}


# =============================================================================
# 4.  OPTIMISER AND TRAINING
# =============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
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


def gradient_check(model, world, n=6, k=45, seed=3, rtol=2e-5, atol=1e-9):
    """Mandatory finite-difference check on the FULL objective.

    Pass criterion is np.allclose semantics:  |num - ana| <= atol + rtol*|scale|.
    The absolute term matters here.  After convergence several true gradients
    sit around 1e-8, and a bare relative error would then be reporting the
    round-off of the difference quotient rather than any disagreement between
    the analytic and numeric computations.  A real bug -- an index slip, a
    missing transpose, a dropped path -- still fails loudly, because it puts
    the two numbers orders of magnitude apart in absolute terms.

    Note on Wj/bj.  The elicitation path consumes the FROZEN standing
    inclination, which is a constant array inside a step, so the analytic
    gradient of Wj comes only from the worth channel.  Finite differences
    agree, because perturbing Wj genuinely does not move the elicited act.
    The check therefore also VERIFIES the alignment claim rather than merely
    asserting it: if the advantage signal could reach Wj, this line would fail.
    """
    rng = np.random.default_rng(seed)
    b = world.batch(n, seed=seed)
    _, _, _, g = model.loss_and_grads(b)
    worst_rel, worst_abs, report, ok_all = 0.0, 0.0, [], True
    for key in list(model.P.keys()):
        arr = model.P[key]
        assert arr.flags['C_CONTIGUOUS'], f"{key} is not contiguous"
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(k, flat.size), replace=False)
        rels, abss, ok = [], [], True
        for i in idxs:
            orig = flat[i]
            h = 1e-5 * max(1.0, abs(orig))
            flat[i] = orig + h
            lp, _, _, _ = model.loss_and_grads(b, need_grads=False)
            flat[i] = orig - h
            lm, _, _, _ = model.loss_and_grads(b, need_grads=False)
            flat[i] = orig
            num = (lp - lm) / (2 * h)
            ana = g[key].reshape(-1)[i]
            ad = abs(num - ana)
            abss.append(ad)
            rels.append(ad / max(1e-12, abs(num) + abs(ana)))
            if ad > atol + rtol * (abs(num) + abs(ana)):
                ok = False
        r, a = float(np.max(rels)), float(np.max(abss))
        worst_rel, worst_abs = max(worst_rel, r), max(worst_abs, a)
        ok_all &= ok
        report.append((key, r, a, ok))
    return worst_rel, worst_abs, report, ok_all


def train(model, world, steps=1400, bs=192, lr=3e-3, log_every=200, seed=11):
    opt = Adam(model.P, lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(1, steps + 1):
        b = world.batch(bs, seed=int(rng.integers(1 << 30)))
        model.freeze_justice()          # the standing inclination is refreshed
        total, terms, out, g = model.loss_and_grads(b)
        opt.step(model.P, g)
        if s % log_every == 0 or s == 1:
            hist.append((s, terms))
            print(f"  step {s:5d} | total {terms['total']:.4f} "
                  f"| nat {terms['nat']:.3f} ind {terms['ind']:.3f} "
                  f"eli {terms['eli']:.3f} | form {terms['form']:.4f} "
                  f"haec {terms['haec']:.4f} pot {terms['pot']:.4f}")
    return hist


# =============================================================================
# 5.  EVALUATION -- every claim the architecture makes is measured
# =============================================================================

def evaluate(model, world, n=3000, seed=4242):
    b = world.batch(n, seed=seed)
    _, terms, out, c = model.loss_and_grads(b, need_grads=False)
    res = {}

    res['nature_acc'] = accuracy(out['logits_nat'], b['nature'])
    res['individual_acc'] = accuracy(out['logits_ind'], b['indiv'])
    res['degree_r2'] = 1.0 - float(np.var(out['deg_pred'] - b['degree'])
                                   / (np.var(b['degree']) + EPS))
    res['elicited_acc'] = accuracy(out['elicit'], b['target'])

    # -- quidditative leakage: can a linear probe read NATURE out of THISNESS? --
    h = out['h']
    Y = np.zeros((n, N_NATURE)); Y[np.arange(n), b['nature']] = 1.0
    split = n // 2
    Xtr = np.concatenate([h[:split], np.ones((split, 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y[:split], rcond=None)
    Xte = np.concatenate([h[split:], np.ones((n - split, 1))], axis=1)
    res['nature_probe_from_haecceity'] = float(
        np.mean(np.argmax(Xte @ W, axis=1) == b['nature'][split:]))
    res['chance'] = 1.0 / N_NATURE

    # -- the formal-distinction corridor --------------------------------------
    K = model.K_FORM
    ovs = []
    for i in range(K):
        for j in range(i + 1, K):
            gi, gj = out['forms'][i], out['forms'][j]
            cc = np.sum(gi * gj, axis=1) / (
                np.linalg.norm(gi, axis=1) * np.linalg.norm(gj, axis=1) + EPS)
            ovs.append(np.abs(cc))
    ovs = np.stack(ovs, axis=1)
    res['formality_overlap_mean'] = float(ovs.mean())
    res['formality_overlap_min'] = float(ovs.min())
    res['formality_overlap_max'] = float(ovs.max())
    res['formality_in_corridor'] = float(
        np.mean((ovs > model.band_lo * 0.9) & (ovs < model.band_hi * 1.1)))

    # -- potentia rationalis: is the power for opposites still standing? -------
    pot = out['pot']
    srt = np.sort(pot, axis=1)
    res['rationality_index'] = float(np.mean(srt[:, -2] / (srt[:, -1] + EPS)))
    res['dead_potency_frac'] = float(np.mean(srt[:, -2] / (srt[:, -1] + EPS) < 0.05))

    # -- the two affections ----------------------------------------------------
    conf = b['conflict']
    due = b['justice_due']
    el = np.argmax(out['elicit'], axis=1)
    res['conflict_frac'] = float(conf.mean())
    res['justice_override_rate'] = float(np.mean(el[due] == b['a_wor'][due])) \
        if due.sum() > 0 else float('nan')
    res['advantage_held_when_no_worth_at_stake'] = float(
        np.mean(el[conf & ~due] == b['a_adv'][conf & ~due])) \
        if (conf & ~due).sum() > 0 else float('nan')
    res['mu_on_justice_cases'] = float(np.mean(out['mu'][due]))
    res['mu_on_ordinary_cases'] = float(np.mean(out['mu'][~conf]))

    # -- univocity under an unbounded change of intrinsic mode -----------------
    xi, nati = world.infinite_probe()
    oi, _ = model.forward(xi)
    proto_dir = np.stack([out['direction'][b['nature'] == k].mean(0)
                          for k in range(N_NATURE)])
    proto_dir /= (np.linalg.norm(proto_dir, axis=1, keepdims=True) + EPS)
    di = oi['direction'] / (np.linalg.norm(oi['direction'], axis=1, keepdims=True) + EPS)
    res['univocal_direction_acc'] = float(
        np.mean(np.argmax(di @ proto_dir.T, axis=1) == nati))
    res['degree_ratio_infinite_vs_finite'] = float(
        oi['deg'].mean() / (out['deg'].mean() + EPS))
    return res, terms


def run_symbolic_tests():
    """Ordo essentialis and the disjunctive transcendentals."""
    out = []
    accidental_line = {
        'grandson': ('son', ACCIDENTAL),
        'son': ('father', ACCIDENTAL),
        'father': ('grandfather', ACCIDENTAL),
        'grandfather': ('ancestor', ACCIDENTAL),
        'ancestor': (None, ACCIDENTAL),
    }
    essential_line = {
        'club': ('arm', ESSENTIAL),
        'arm': ('shoulder', ESSENTIAL),
        'shoulder': ('agent', ESSENTIAL),
        'agent': (None, ESSENTIAL),
    }
    loop = {'a': ('b', ESSENTIAL), 'b': ('c', ESSENTIAL), 'c': ('a', ESSENTIAL)}

    v1, p1 = climb(accidental_line, 'grandson')
    v2, p2 = climb(essential_line, 'club')
    v3, p3 = climb(loop, 'a')
    out.append(('accidental chain -> no first required', v1, v1 == 'open-regress'))
    out.append(('essential chain -> first agent', v2, v2 == 'first-agent'))
    out.append(('circular causing rejected', v3, v3 == 'circular'))

    inf = disjunctive_inference({'contingent', 'caused'})
    out.append(('contingent+caused => necessary,uncaused',
                sorted(inf), inf == {'necessary', 'uncaused'}))
    inf2 = disjunctive_inference({'necessary'})
    out.append(('strong disjunct entails nothing further',
                sorted(inf2), inf2 == set()))
    return out


# =============================================================================
# 6.  MAIN
# =============================================================================

def main():
    line = "=" * 79
    print(line)
    print(" THE INSTANT ENGINE  --  chapter 0353, John Duns Scotus (c.1266-1308)")
    print(line)

    world = SentencesWorld(seed=7)
    model = InstantEngine(seed=0)
    print(f"\n[world]  natures={N_NATURE}  individuals={N_INDIV}  acts={N_ACTS}"
          f"  input dim={DIN}")
    print(f"[model]  D={model.D}  R={model.R}  formalities={model.K_FORM}"
          f"  params={sum(v.size for v in model.P.values()):,}")
    print(f"[model]  formal-distinction corridor = "
          f"[{model.band_lo}, {model.band_hi}]   potency floor = {model.pot_floor}")

    print("\n--- 1. FINITE-DIFFERENCE GRADIENT CHECK "
          "(before training) -----------------")
    model.freeze_justice()
    wrel, wabs, report, ok = gradient_check(model, world)
    print("      param     max rel err   max abs err")
    for k, r, a, o in report:
        print(f"    {k:>7s}    {r:.3e}     {a:.3e}   {'ok' if o else 'FAIL'}")
    print(f"    WORST rel {wrel:.3e}   WORST abs {wabs:.3e}"
          f"   ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"

    print("\n--- 2. TRAINING ----------------------------------------------------")
    train(model, world, steps=1800, bs=192, lr=3e-3, log_every=300)

    print("\n--- 3. GRADIENT CHECK (after training) -----------------------------")
    model.freeze_justice()
    wrel2, wabs2, _, ok2 = gradient_check(model, world)
    print(f"    WORST rel {wrel2:.3e}   WORST abs {wabs2:.3e}"
          f"   ->  {'PASS' if ok2 else 'FAIL'}")
    assert ok2

    print("\n--- 4. EVALUATION (3000 held-out occasions) ------------------------")
    res, terms = evaluate(model, world)
    order = [
        ('nature_acc', 'common nature recognised'),
        ('individual_acc', 'THIS ONE re-identified across accidental change'),
        ('degree_r2', 'intrinsic mode recovered (R^2)'),
        ('elicited_acc', 'elicited act matches the right act'),
        ('nature_probe_from_haecceity', 'nature read OUT of thisness (probe)'),
        ('chance', '   ... chance level for that probe'),
        ('formality_overlap_mean', 'formality overlap, mean'),
        ('formality_overlap_min', 'formality overlap, min'),
        ('formality_overlap_max', 'formality overlap, max'),
        ('formality_in_corridor', 'fraction inside the corridor'),
        ('rationality_index', 'power for opposites at the instant of the act'),
        ('dead_potency_frac', 'occasions where that power died'),
        ('conflict_frac', 'occasions where advantage and worth disagree'),
        ('justice_override_rate', 'justice overrides advantage when worth is at stake'),
        ('advantage_held_when_no_worth_at_stake', 'advantage kept when worth is indifferent'),
        ('mu_on_justice_cases', 'moderation mu, justice cases'),
        ('mu_on_ordinary_cases', 'moderation mu, ordinary cases'),
        ('univocal_direction_acc', 'nature still read at 12x the trained degree'),
        ('degree_ratio_infinite_vs_finite', 'degree ratio, unbounded vs trained'),
    ]
    for k, label in order:
        print(f"    {label:<58s} {res[k]:.4f}")

    print("\n--- 5. SYMBOLIC TESTS: ordo essentialis ----------------------------")
    allok = True
    for name, got, ok3 in run_symbolic_tests():
        allok &= ok3
        print(f"    {name:<52s} {str(got):<28s} {'PASS' if ok3 else 'FAIL'}")
    assert allok

    print("\n--- 6. ASSERTIONS --------------------------------------------------")
    checks = [
        ('nature accuracy > 0.85', res['nature_acc'] > 0.85),
        ('individual accuracy > 0.70', res['individual_acc'] > 0.70),
        ('elicited accuracy > 0.75', res['elicited_acc'] > 0.75),
        ('formalities stay inside the corridor > 0.90',
         res['formality_in_corridor'] > 0.90),
        ('power for opposites never dies', res['dead_potency_frac'] < 0.01),
        ('rationality index above floor/2',
         res['rationality_index'] > model.pot_floor / 2),
        ('justice overrides advantage > 0.70', res['justice_override_rate'] > 0.70),
        ('univocal direction survives 12x degree > 0.80',
         res['univocal_direction_acc'] > 0.80),
    ]
    for name, ok4 in checks:
        print(f"    {'PASS' if ok4 else 'FAIL'}  {name}")
    assert all(o for _, o in checks), "architectural assertions failed"

    print("\n--- 7. ONE INSTANT, OPENED UP --------------------------------------")
    demo = world.batch(400, seed=8080)
    dout, dc = model.forward(demo['x'])
    cand = np.where(demo['justice_due'])[0]
    i = int(cand[0]) if len(cand) else 0
    np.set_printoptions(precision=3, suppress=True)
    print(f"    sign 1 (natura)     magnitude of being    {dout['deg'][i]:.3f}"
          f"  -> degree read off {dout['deg_pred'][i]:.3f}"
          f"  (true {demo['degree'][i]:.3f})")
    print(f"                        nature judged         "
          f"{int(np.argmax(dout['logits_nat'][i]))}  (true {demo['nature'][i]})")
    print(f"                        thisness judged       "
          f"{int(np.argmax(dout['logits_ind'][i]))}  (true {demo['indiv'][i]})")
    print("    sign 2 (intellectus) affectio commodi      "
          f"{dc['cmd'][i]}")
    print("                        affectio iustitiae    "
          f"{dc['jref'][i]}")
    print("                        potency (not a distribution, does not sum to 1)")
    print(f"                                              {dout['pot'][i]}"
          f"   sum={dout['pot'][i].sum():.3f}")
    print(f"    sign 3 (voluntas)   moderation mu         {dout['mu'][i]:.3f}")
    print(f"                        elicited act          "
          f"{int(np.argmax(dout['elicit'][i]))}"
          f"   (advantage said {demo['a_adv'][i]}, worth said {demo['a_wor'][i]})")
    srt = np.sort(dout['pot'][i])
    print(f"                        power for the opposite still standing: "
          f"{srt[-2] / srt[-1]:.3f} of the chosen act's power")

    print("\n" + line)
    print(" ALL CHECKS PASSED")
    print(line)
    return model, world, res


if __name__ == '__main__':
    main()

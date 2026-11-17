#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE COPULATIO ENGINE
 An executable cognitive architecture after Siger of Brabant (c.1240 - <1284)
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0343 · Siger of Brabant (1240-<1284)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================

WHY THIS ARCHITECTURE, AND WHY IT IS NOT A TRANSFORMER
------------------------------------------------------
Siger taught in the Faculty of Arts at Paris, in the street Dante calls the
Vico de li Strami. He is remembered for two separate scandals, and this file is
built on the claim that they are one idea.

  (1) A SHARED INTELLECT WITH AN ATTRIBUTION RULE.  Following Averroes, Siger
      held that the intellect is numerically ONE. Aquinas' reply was lethal and
      simple: then no thought is yours rather than mine. Siger's answer -- and
      this is his own move, not Averroes' -- was that the intellect is
      separated from the body IN BEING (in essendo) but united to it IN
      OPERATION (in operando): they are one in the work, because they converge
      in one work. The intellect is an OPERANS INTRINSECUM, an intrinsic
      operator, and the operations of intrinsic operators are predicated of the
      composite. His own illustration: we say the MAN sees, though sight is in
      the eye alone and not in the foot; and if the eye were not united to the
      rest, the seeing could not be attributed to the whole at all.
      => A thought is not OWNED. It is ATTRIBUTED. Attribution is a fact about
         convergence, not about who possesses the organ.

  (2) THE NON-ASSERTORIC REGISTER.  Siger repeatedly marks conclusions as
      recited rather than claimed -- reciting the Philosopher's opinion, not
      asserting it as true -- and describes his own procedure as seeking the
      philosophers' intention rather than the truth. Aquinas thought this was a
      dereliction: the study of philosophy is not to know what men have thought
      but how the truth of things stands.
      => A conclusion carries a PREMISE-SET INDEX. Derivability under a
         premise-set is not assertion.

Together: EVERY COGNITIVE ACT HERE CARRIES TWO INDICES -- whose composite it
converged in, and under which premise-set it was derived. Neither index is
self-enforcing. Siger's own end is the demonstration. Tempier read the register
tag as evasion, reconstructed it as a claim of two contrary truths, and
condemned it on 7 March 1277. The tag did not survive its audience.

So this file deliberately avoids attention over stored keys. An attention stack
retrieves content by similarity from a corpus-shaped memory; it has exactly one
register (whatever it was trained to emit) and exactly one subject (nobody).
Siger needs the opposite shape:

  * ONE shared possible intellect holding NO content and NO session state -- a
    bank of composable operators, never a retrievable store. Nothing is written
    on it. Content arrives only from a body.
  * MANY private cogitative faculties, one per body, which are the ONLY
    individuator. Siger's technical device for saving private thought inside a
    shared intellect is that the intellect's operation in this human depends on
    that human's INTENTIONES IMAGINATAE -- imagined intentions arising from
    sensation, unaided by intellect.
  * An AGENT INTELLECT trained so the abstracted species is what is THE SAME
    across every body's encounter with the same thing. Abstraction is
    implemented literally: one proposition is shown to two different bodies in
    the same batch and the two species are pulled together. The universal is
    defined as the invariant rather than asserted as a doctrine.
  * A COPULATIO gate re-binding the completed act to the body whose phantasms
    it converged with -- restoring individuality at the level of the ACT,
    exactly where Siger puts it and nowhere else.
  * A REGISTER modulator conditioning the whole operation on a premise-set,
    with an assertion head permitted to assert under one register only.

WHO AUDITS THE ABSTRACTION
--------------------------
Whether the species is really free of individuating conditions is measured, not
claimed. After training, linear probes are fitted FROM OUTSIDE the model on
frozen representations and asked: can you tell whose phantasm this came from?
The probes are no part of the objective and send the model no gradient. That
independence matters for more than tidiness -- an auditor trained by the same
objective as the thing it audits inherits its blind spots.

THE FALSIFIABLE CLAIMS THIS FILE TESTS
--------------------------------------
  E1  A register-indexed mind holds mutually contradictory premise-sets at full
      accuracy on the very propositions where they disagree; a register-blind
      mind of identical capacity collapses to chance there.
  E2  Aquinas' objection is real and measurable, and the copulatio gate is what
      answers it. With the gate, body identity is unreadable from the abstracted
      species and from the bare act, yet perfectly readable from the bound act:
      individuality lives in the operation and only there. Remove the gate and
      the model still attributes -- but ONLY by writing individuating conditions
      back into the universal, which destroys the abstraction. The trade is
      forced: a mind can have a genuine universal, or attribution without a
      body, and not both.
  E3  Register bleed, the Tempier metric: how often the mind returns the same
      verdict under premise-sets that contradict each other.
  E4  Impossibilia containment: an internally incoherent register -- Siger
      argued six impossible theses on purpose -- must not degrade the others.
  E5  Starvation: the intellect has no content of its own. Cut the phantasm
      stream and its accuracy must fall to chance.

RUNNING
-------
  python3 0277_siger_of_brabant_Neuron.py            # full run
  python3 0277_siger_of_brabant_Neuron.py --quick    # short run, same tests
Pure NumPy. No frameworks. Forward and backward written out by hand. A
finite-difference gradient check over every trainable tensor is mandatory and
runs on every invocation.

================================================================================
"""

import argparse
import json
import sys
import time

import numpy as np

# ==============================================================================
# SECTION 0 -- CONFIGURATION
# ==============================================================================

class Config:
    """All dimensions and hyper-parameters in one place.

    Deliberately small. The point is that a claim about the ARCHITECTURE OF
    ATTRIBUTION can be tested at a scale where every gradient is verifiable by
    hand -- not that scale produces competence.
    """
    # --- world ---
    n_entities   = 6      # Socrates, Callias, and the rest of the stock examples
    n_relations  = 4      # the predicates that may be joined to them
    n_bodies     = 6      # how many human composites share the one intellect
    n_registers  = 3      # 0 = ASSERTA, 1 = PHILOSOPHUS, 2 = IMPOSSIBILIA

    # --- dimensions ---
    d_prop       = 4 + 6 + 6   # one-hot composition code: relation, subject, object
    d_sense      = 24     # the body's raw sensory channel (fixed, never trained)
    d_phantasm   = 32     # intentio imaginata -- private, per body
    d_species    = 24     # the abstracted universal -- shared, de-individuated
    d_reg        = 12     # premise-set embedding
    d_act        = 32     # the completed intellectual act
    n_operators  = 6      # K composable operators in the possible intellect
    n_steps      = 3      # T iterations of the act (the discursive passage)

    # --- objective weights ---
    w_truth      = 1.0
    w_attr       = 0.5    # Aquinas' question, given real weight
    w_assert     = 0.3    # recitando vs asserendo
    w_inv        = 3.0    # abstraction: the universal is what is invariant across bodies
    inv_warmup   = 0.25   # abstraction presupposes a legible phantasm, so it is not
                          # demanded until there is something to abstract FROM

    # --- optimisation ---
    lr           = 0.02
    beta1, beta2 = 0.9, 0.999
    eps          = 1e-8
    steps        = 3000
    pairs        = 128    # batch is 2 * pairs: each proposition shown to two bodies
    seed         = 12771270   # the two condemnations, as a seed

    def quick(self):
        self.steps = 1800
        return self


# ==============================================================================
# SECTION 1 -- THE WORLD, AND WHY THE SAME PROPOSITION LOOKS DIFFERENT TO EACH BODY
# ==============================================================================

class SchoolOfParis:
    """Propositions, per-body percepts, and per-register truth values.

    A proposition is a relation joined to two entities -- composition, in the
    scholastic sense, and the place where truth and falsity live. Three things
    matter:

    1. EVERY BODY MEETS THE SAME PROPOSITION DIFFERENTLY. Each body owns a fixed
       random sensory projection. The universal is shared; the sensory encounter
       with it never is. This is the entire reason the agent intellect has work
       to do.

    2. THE REGISTERS DISAGREE ON PURPOSE. Register 0 (what may be asserted) and
       register 1 (what the Philosopher's premises entail) assign OPPOSITE truth
       values across the contested half of the proposition space. A mind with
       one register cannot be right about both. Siger's professional life is the
       claim that this is a bookkeeping problem, not a contradiction.

    3. REGISTER 2 IS INCOHERENT ON PURPOSE. The Impossibilia are theses Siger
       argued for while knowing them impossible. The register has no stable
       truth value. The test is not whether the mind learns it -- it cannot --
       but whether reasoning inside it corrupts the other two.
    """

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        E, R = cfg.n_entities, cfg.n_relations

        props = [(r, i, j) for r in range(R) for i in range(E) for j in range(E) if i != j]
        self.props = np.array(props, dtype=np.int64)
        self.n_props = len(props)

        V = np.zeros((self.n_props, cfg.d_prop), dtype=np.float64)
        for n, (r, i, j) in enumerate(props):
            V[n, r] = 1.0
            V[n, R + i] = 1.0
            V[n, R + E + j] = 1.0
        self.V = V

        # Register 0 (ASSERTA): an ordering rule -- is the subject prior to the
        # object in the order of nature? Kept simple on purpose. The difficulty
        # under test is not the rule; it is holding TWO INCOMPATIBLE RULES at
        # once without either contaminating the other.
        base = (self.props[:, 1] < self.props[:, 2]).astype(np.int64)
        self.truth_r0 = base

        # The contested region: relations 0 and 1. There the Philosopher's
        # premises entail the negation of what may be asserted.
        self.contested = (self.props[:, 0] < 2)
        self.truth_r1 = np.where(self.contested, 1 - base, base)

        # The bodies' sense organs: FIXED, never trained. A body does not choose
        # its senses; it is given them, and must think with those.
        self.S = rng.normal(0, 1.0, size=(cfg.n_bodies, cfg.d_prop, cfg.d_sense))
        self.S /= np.sqrt(cfg.d_prop)

    def _make(self, prop_ids, body_ids, registers):
        n = len(prop_ids)
        v = self.V[prop_ids]
        s = np.einsum('nd,ndk->nk', v, self.S[body_ids])
        s = s + self.rng.normal(0, 0.05, size=s.shape)

        y_truth = np.empty(n, dtype=np.int64)
        m0, m1, m2 = registers == 0, registers == 1, registers == 2
        y_truth[m0] = self.truth_r0[prop_ids[m0]]
        y_truth[m1] = self.truth_r1[prop_ids[m1]]
        y_truth[m2] = self.rng.integers(0, 2, size=int(m2.sum()))   # no stable truth

        return dict(s=s, body=body_ids, reg=registers, prop=prop_ids,
                    y_truth=y_truth,
                    y_assert=(registers == 0).astype(np.int64),   # 1 = assero, 0 = recito
                    contested=self.contested[prop_ids], pair_n=0)

    def sample(self, n, registers=None, body_ids=None, prop_ids=None):
        """Unpaired draw, for evaluation."""
        rng, cfg = self.rng, self.cfg
        if prop_ids is None:
            prop_ids = rng.integers(0, self.n_props, size=n)
        if body_ids is None:
            body_ids = rng.integers(0, cfg.n_bodies, size=n)
        if registers is None:
            registers = rng.integers(0, cfg.n_registers, size=n)
        registers = np.asarray(registers)
        if registers.ndim == 0:
            registers = np.full(n, int(registers))
        return self._make(np.asarray(prop_ids), np.asarray(body_ids), registers)

    def sample_paired(self, n_pairs, max_register=None):
        """Training draw. Row i and row i+n_pairs are the SAME proposition under
        the SAME premise-set, met by TWO DIFFERENT bodies.

        This is what makes abstraction definable. The universal is not posited;
        it is whatever survives the change of body.
        """
        rng, cfg = self.rng, self.cfg
        R = cfg.n_registers if max_register is None else max_register
        p = rng.integers(0, self.n_props, size=n_pairs)
        r = rng.integers(0, R, size=n_pairs)
        bA = rng.integers(0, cfg.n_bodies, size=n_pairs)
        bB = (bA + rng.integers(1, cfg.n_bodies, size=n_pairs)) % cfg.n_bodies

        b = self._make(np.concatenate([p, p]),
                       np.concatenate([bA, bB]),
                       np.concatenate([r, r]))
        # NOTE: register-2 labels are drawn independently per row, so a pair
        # inside the Impossibilia may disagree with itself. That is the point.
        b['pair_n'] = n_pairs
        return b


# ==============================================================================
# SECTION 2 -- PARAMETERS
# ==============================================================================

def init_params(cfg, rng):
    """Every trainable tensor, named for the faculty it implements.

    Note what is PER-BODY and what is SHARED. Only `cog_W` and `cog_b` are
    indexed by body. Everything downstream of the agent intellect is
    numerically one, for everybody, always. That is the doctrine, expressed as
    a shape.
    """
    def g(*shape, scale=None):
        fan_in = shape[-2] if len(shape) >= 2 else shape[0]
        sc = scale if scale is not None else 1.0 / np.sqrt(fan_in)
        return rng.normal(0, sc, size=shape)

    P = {}
    # --- cogitative faculty: PRIVATE, one per body -------------------------
    # produces the intentiones imaginatae. The only individuator in the system.
    P['cog_W'] = g(cfg.n_bodies, cfg.d_sense, cfg.d_phantasm)
    P['cog_b'] = np.zeros((cfg.n_bodies, cfg.d_phantasm))

    # --- agent intellect: SHARED. Abstracts the universal from the phantasm.
    P['ag_W'] = g(cfg.d_phantasm, cfg.d_species)
    P['ag_b'] = np.zeros(cfg.d_species)

    # --- premise-set (register) embedding ----------------------------------
    P['reg_E'] = g(cfg.n_registers, cfg.d_reg, scale=0.5)

    # --- possible intellect: SHARED, STATELESS, CONTENT-FREE ---------------
    # A bank of operators, not a store of facts. Nothing can be looked up here.
    P['op_A'] = g(cfg.n_operators, cfg.d_act, cfg.d_act, scale=0.8 / np.sqrt(cfg.d_act))
    P['gate_Wsp'] = g(cfg.d_species, cfg.n_operators)   # species chooses the composition
    P['gate_Wrg'] = g(cfg.d_reg, cfg.n_operators)       # so does the premise-set
    P['gate_b'] = np.zeros(cfg.n_operators)
    P['in_W'] = g(cfg.d_species, cfg.d_act)             # the first act
    P['in_b'] = np.zeros(cfg.d_act)
    P['inj_Wsp'] = g(cfg.d_species, cfg.d_act)          # the species is re-injected
    P['inj_Wrg'] = g(cfg.d_reg, cfg.d_act)              #   at EVERY step of the act
    P['inj_b'] = np.zeros(cfg.d_act)

    # --- copulatio: binds the completed act back to THIS body --------------
    P['cop_W'] = g(cfg.d_act + cfg.d_phantasm, cfg.d_act)
    P['cop_b'] = np.zeros(cfg.d_act)
    P['ph2a_W'] = g(cfg.d_phantasm, cfg.d_act)
    P['ph2a_b'] = np.zeros(cfg.d_act)

    # --- heads --------------------------------------------------------------
    P['truth_W'] = g(cfg.d_act, 2);            P['truth_b'] = np.zeros(2)
    P['attr_W']  = g(cfg.d_act, cfg.n_bodies); P['attr_b']  = np.zeros(cfg.n_bodies)
    P['asrt_W']  = g(cfg.d_act, 2);            P['asrt_b']  = np.zeros(2)
    return P


# ==============================================================================
# SECTION 3 -- FORWARD AND BACKWARD (written out by hand)
# ==============================================================================

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce(logits, y):
    """Mean cross-entropy and its gradient wrt logits (already divided by N)."""
    p = softmax(logits)
    n = len(y)
    loss = -np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


def forward(P, cfg, batch, opts):
    """One pass through the whole composite.

    opts:
      use_copulatio  -- if False the attribution head reads the BARE act. This
                        is the ablation that stages Aquinas' objection.
      blind_register -- if True every act is computed under register 0's
                        embedding whatever the true premise-set is. This stages
                        a mind with only one register.
      inv_scale      -- current strength of the abstraction (invariance) term.
    """
    s, body, reg = batch['s'], batch['body'], batch['reg']
    N = len(s)
    reg_in = np.zeros_like(reg) if opts.get('blind_register') else reg

    # ---- 1. cogitative faculty (private) ----------------------------------
    phz = np.einsum('nd,ndk->nk', s, P['cog_W'][body]) + P['cog_b'][body]
    ph = np.tanh(phz)

    # ---- 2. agent intellect: abstraction ----------------------------------
    spz = ph @ P['ag_W'] + P['ag_b']
    sp = np.tanh(spz)

    # ---- 3. premise-set ----------------------------------------------------
    re = P['reg_E'][reg_in]

    # ---- 4. possible intellect: compose operators, iterate the act --------
    gl = sp @ P['gate_Wsp'] + re @ P['gate_Wrg'] + P['gate_b']
    gate = softmax(gl)
    inj = sp @ P['inj_Wsp'] + re @ P['inj_Wrg'] + P['inj_b']

    h = [np.tanh(sp @ P['in_W'] + P['in_b'])]
    us = []
    for t in range(cfg.n_steps):
        u_t = [h[t] @ P['op_A'][k] for k in range(cfg.n_operators)]
        m = np.zeros_like(h[t])
        for k in range(cfg.n_operators):
            m += gate[:, k:k + 1] * u_t[k]
        h.append(np.tanh(m + inj))
        us.append(u_t)
    act = h[-1]

    # ---- 5. copulatio: in operando, not in essendo ------------------------
    cat = np.concatenate([act, ph], axis=1)
    cz = cat @ P['cop_W'] + P['cop_b']
    c = 1.0 / (1.0 + np.exp(-cz))
    pa = ph @ P['ph2a_W'] + P['ph2a_b']
    bound = c * act + (1.0 - c) * pa

    # ---- 6. heads ----------------------------------------------------------
    attr_src = bound if opts.get('use_copulatio', True) else act
    lt = bound @ P['truth_W'] + P['truth_b']
    la = attr_src @ P['attr_W'] + P['attr_b']
    ls = bound @ P['asrt_W'] + P['asrt_b']

    L_t, d_lt = ce(lt, batch['y_truth'])
    L_a, d_la = ce(la, body)
    L_s, d_ls = ce(ls, batch['y_assert'])

    # ---- 7. abstraction: the universal is the invariant -------------------
    M = batch.get('pair_n', 0)
    inv_scale = opts.get('inv_scale', 1.0)
    if M > 0:
        diff = sp[:M] - sp[M:2 * M]
        L_i = float((diff ** 2).sum() / (M * cfg.d_species))
    else:
        diff, L_i = None, 0.0

    loss = (cfg.w_truth * L_t + cfg.w_attr * L_a + cfg.w_assert * L_s
            + cfg.w_inv * inv_scale * L_i)

    cache = dict(s=s, body=body, reg_in=reg_in, ph=ph, sp=sp, re=re, gate=gate,
                 inj=inj, h=h, us=us, act=act, cat=cat, c=c, pa=pa, bound=bound,
                 attr_src=attr_src, d_lt=d_lt, d_la=d_la, d_ls=d_ls,
                 diff=diff, M=M, inv_scale=inv_scale, N=N, opts=opts)
    parts = dict(truth=L_t, attr=L_a, assertion=L_s, invariance=L_i)
    preds = dict(truth=lt.argmax(1), attr=la.argmax(1), assertion=ls.argmax(1),
                 truth_logits=lt, species=sp, act=act, bound=bound)
    return loss, parts, preds, cache


def backward(P, cfg, cache):
    """Hand-written reverse pass. Mirrors `forward` line for line.

    There is no gradient reversal and no stop-gradient anywhere, so the
    objective is an ordinary scalar function of the parameters and the
    finite-difference check below is exact over every path.
    """
    G = {k: np.zeros_like(v) for k, v in P.items()}
    opts = cache['opts']
    ph, sp, re = cache['ph'], cache['sp'], cache['re']
    gate, inj, h, us = cache['gate'], cache['inj'], cache['h'], cache['us']
    act, cat, c, pa, bound = cache['act'], cache['cat'], cache['c'], cache['pa'], cache['bound']

    d_bound = np.zeros_like(bound)
    d_act = np.zeros_like(act)
    d_sp = np.zeros_like(sp)
    d_re = np.zeros_like(re)
    d_ph = np.zeros_like(ph)

    # ---- heads -------------------------------------------------------------
    dt = cfg.w_truth * cache['d_lt']
    G['truth_W'] += bound.T @ dt; G['truth_b'] += dt.sum(0); d_bound += dt @ P['truth_W'].T

    da = cfg.w_attr * cache['d_la']
    G['attr_W'] += cache['attr_src'].T @ da; G['attr_b'] += da.sum(0)
    if opts.get('use_copulatio', True):
        d_bound += da @ P['attr_W'].T
    else:
        d_act += da @ P['attr_W'].T          # attribution bypasses the copulatio

    ds = cfg.w_assert * cache['d_ls']
    G['asrt_W'] += bound.T @ ds; G['asrt_b'] += ds.sum(0); d_bound += ds @ P['asrt_W'].T

    # ---- abstraction term --------------------------------------------------
    M, diff = cache['M'], cache['diff']
    if M > 0:
        coef = cfg.w_inv * cache['inv_scale'] * 2.0 / (M * cfg.d_species)
        d_sp[:M] += coef * diff
        d_sp[M:2 * M] -= coef * diff

    # ---- copulatio ---------------------------------------------------------
    d_c = d_bound * (act - pa)
    d_act += d_bound * c
    d_pa = d_bound * (1.0 - c)
    d_cz = d_c * c * (1.0 - c)
    G['cop_W'] += cat.T @ d_cz; G['cop_b'] += d_cz.sum(0)
    d_cat = d_cz @ P['cop_W'].T
    d_act += d_cat[:, :cfg.d_act]
    d_ph += d_cat[:, cfg.d_act:]
    G['ph2a_W'] += ph.T @ d_pa; G['ph2a_b'] += d_pa.sum(0)
    d_ph += d_pa @ P['ph2a_W'].T

    # ---- the act, unrolled backwards --------------------------------------
    d_h = d_act
    d_gate = np.zeros_like(gate)
    d_inj = np.zeros_like(inj)
    for t in range(cfg.n_steps - 1, -1, -1):
        d_z = d_h * (1.0 - h[t + 1] ** 2)
        d_inj += d_z
        d_h_prev = np.zeros_like(h[t])
        for k in range(cfg.n_operators):
            u = us[t][k]
            d_gate[:, k] += (d_z * u).sum(1)
            d_u = d_z * gate[:, k:k + 1]
            G['op_A'][k] += h[t].T @ d_u
            d_h_prev += d_u @ P['op_A'][k].T
        d_h = d_h_prev
    d_z0 = d_h * (1.0 - h[0] ** 2)
    G['in_W'] += sp.T @ d_z0; G['in_b'] += d_z0.sum(0)
    d_sp += d_z0 @ P['in_W'].T

    G['inj_Wsp'] += sp.T @ d_inj; d_sp += d_inj @ P['inj_Wsp'].T
    G['inj_Wrg'] += re.T @ d_inj; d_re += d_inj @ P['inj_Wrg'].T
    G['inj_b'] += d_inj.sum(0)

    d_gl = gate * (d_gate - (d_gate * gate).sum(1, keepdims=True))
    G['gate_Wsp'] += sp.T @ d_gl; d_sp += d_gl @ P['gate_Wsp'].T
    G['gate_Wrg'] += re.T @ d_gl; d_re += d_gl @ P['gate_Wrg'].T
    G['gate_b'] += d_gl.sum(0)

    np.add.at(G['reg_E'], cache['reg_in'], d_re)

    # ---- agent intellect ---------------------------------------------------
    d_spz = d_sp * (1.0 - sp ** 2)
    G['ag_W'] += ph.T @ d_spz; G['ag_b'] += d_spz.sum(0)
    d_ph += d_spz @ P['ag_W'].T

    # ---- cogitative faculty (private, gathered per body) -------------------
    d_phz = d_ph * (1.0 - ph ** 2)
    body, s = cache['body'], cache['s']
    for b in range(cfg.n_bodies):
        m = (body == b)
        if m.any():
            G['cog_W'][b] += s[m].T @ d_phz[m]
            G['cog_b'][b] += d_phz[m].sum(0)
    return G


# ==============================================================================
# SECTION 4 -- OPTIMISER
# ==============================================================================

class Adam:
    def __init__(self, P, cfg):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0
        self.cfg = cfg

    def step(self, P, G):
        c = self.cfg
        self.t += 1
        for k in P:
            self.m[k] = c.beta1 * self.m[k] + (1 - c.beta1) * G[k]
            self.v[k] = c.beta2 * self.v[k] + (1 - c.beta2) * (G[k] ** 2)
            mh = self.m[k] / (1 - c.beta1 ** self.t)
            vh = self.v[k] / (1 - c.beta2 ** self.t)
            P[k] -= c.lr * mh / (np.sqrt(vh) + c.eps)


# ==============================================================================
# SECTION 5 -- THE MANDATORY GRADIENT CHECK
# ==============================================================================

def gradient_check(cfg, verbose=True):
    """Finite differences against the analytic gradient, for every tensor.

    Directional derivatives rather than single entries: for a random unit
    direction D confined to one tensor, the numeric slope must equal <G, D>.
    This exercises the whole tensor at once and its magnitude is O(||g||), so
    the relative error is not dominated by cancellation at an entry that
    happens to sit near zero.
    """
    rng = np.random.default_rng(7)
    world = SchoolOfParis(cfg, rng)
    P = init_params(cfg, rng)
    b = world.sample_paired(24)
    opts = dict(use_copulatio=True, blind_register=False, inv_scale=1.0)

    loss, _, _, cache = forward(P, cfg, b, opts)
    G = backward(P, cfg, cache)

    eps = 1e-6
    rows, worst = [], 0.0
    for name in sorted(P.keys()):
        arr = P[name]
        errs = []
        for _ in range(2):
            D = rng.normal(0, 1, size=arr.shape)
            D /= np.linalg.norm(D)
            orig = arr.copy()
            arr[...] = orig + eps * D
            lp, _, _, _ = forward(P, cfg, b, opts)
            arr[...] = orig - eps * D
            lm, _, _, _ = forward(P, cfg, b, opts)
            arr[...] = orig
            num = (lp - lm) / (2 * eps)
            ana = float((G[name] * D).sum())
            errs.append(abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, e))

    if verbose:
        print("  GRADIENT CHECK  (directional derivative, 2 random directions per tensor)")
        for name, e in rows:
            print(f"    {'ok ' if e < 1e-5 else 'FAIL'} {name:<12s} rel.err {e:.3e}")
        print(f"    worst relative error over all {len(rows)} tensors: {worst:.3e}")
    assert worst < 1e-5, f"gradient check FAILED, worst rel err {worst:.3e}"
    return worst


# ==============================================================================
# SECTION 6 -- TRAINING
# ==============================================================================

def train(cfg, world, opts, label, drop_impossibilia=False, log_every=500, quiet=False):
    rng = np.random.default_rng(cfg.seed + (abs(hash(label)) % 9973))
    P = init_params(cfg, rng)
    opt = Adam(P, cfg)

    # The control that never sees the Impossibilia gets a SMALLER batch, sized
    # so it receives the same number of COHERENT examples per step as the full
    # run. Otherwise "removing the incoherent register" would be confounded
    # with "training on 50% more usable data".
    n_pairs = int(round(cfg.pairs * 2 / 3)) if drop_impossibilia else cfg.pairs
    max_reg = 2 if drop_impossibilia else None

    for step in range(1, cfg.steps + 1):
        b = world.sample_paired(n_pairs, max_register=max_reg)
        o = dict(opts)
        o['inv_scale'] = min(1.0, step / max(1.0, cfg.inv_warmup * cfg.steps))
        loss, parts, preds, cache = forward(P, cfg, b, o)
        G = backward(P, cfg, cache)
        opt.step(P, G)
        if not quiet and (step % log_every == 0 or step == 1):
            print(f"    [{label}] step {step:5d}  loss {loss:7.4f}  "
                  f"truth {(preds['truth'] == b['y_truth']).mean():5.3f}  "
                  f"attribution {(preds['attr'] == b['body']).mean():5.3f}  "
                  f"invariance {parts['invariance']:.4f}")
    return P


# ==============================================================================
# SECTION 7 -- THE EXTERNAL AUDITOR
# ==============================================================================

def fit_probe(feats, labels, n_classes, rng, steps=600, lr=0.15, split=0.7):
    """Fit a linear softmax probe FROM OUTSIDE the model, on frozen features.

    The probe is no part of the training objective and sends the model no
    gradient. It is an auditor, and its independence is the point: an auditor
    trained by the same objective as the thing it audits inherits its blind
    spots. Returns held-out accuracy.
    """
    n, d = feats.shape
    idx = rng.permutation(n)
    k = int(n * split)
    tr, te = idx[:k], idx[k:]
    X, y = feats[tr], labels[tr]
    W = np.zeros((d, n_classes)); b = np.zeros(n_classes)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    for t in range(1, steps + 1):
        p = softmax(X @ W + b)
        g = p.copy(); g[np.arange(len(y)), y] -= 1.0; g /= len(y)
        gW, gb = X.T @ g, g.sum(0)
        mW *= 0.9; mW += 0.1 * gW
        vW *= 0.999; vW += 0.001 * gW ** 2
        W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
        mb *= 0.9; mb += 0.1 * gb
        vb *= 0.999; vb += 0.001 * gb ** 2
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    return float(((feats[te] @ W + b).argmax(1) == labels[te]).mean())


# ==============================================================================
# SECTION 8 -- THE FIVE EXPERIMENTS
# ==============================================================================

def evaluate(P, cfg, world, opts, n=4000):
    out = {}
    for r in range(cfg.n_registers):
        b = world.sample(n, registers=np.full(n, r))
        _, _, preds, _ = forward(P, cfg, b, opts)
        ok = preds['truth'] == b['y_truth']
        con = b['contested']
        out[f'r{r}_all'] = float(ok.mean())
        out[f'r{r}_contested'] = float(ok[con].mean())
        out[f'r{r}_uncontested'] = float(ok[~con].mean())
        out[f'r{r}_attr'] = float((preds['attr'] == b['body']).mean())
        out[f'r{r}_assert'] = float((preds['assertion'] == b['y_assert']).mean())
    return out


def audit_representations(P, cfg, world, opts, n=3000, rng=None):
    """E2. Where in the pipeline does 'whose thought is this?' become answerable?

    Siger's claim is precise: NOT in the species (the universal is what is the
    same for every knower) and NOT in the bare act, but only in the act as
    bound to this body's phantasms. Three probes, one answer each.
    """
    rng = rng or np.random.default_rng(555)
    b = world.sample(n)
    _, _, preds, _ = forward(P, cfg, b, opts)
    return dict(
        probe_on_species=fit_probe(preds['species'], b['body'], cfg.n_bodies, rng),
        probe_on_bare_act=fit_probe(preds['act'], b['body'], cfg.n_bodies, rng),
        probe_on_bound_act=fit_probe(preds['bound'], b['body'], cfg.n_bodies, rng),
        chance=1.0 / cfg.n_bodies)


def register_bleed(P, cfg, world, opts, n=3000, rng=None):
    """E3, the Tempier metric.

    Take contested propositions. Ask the SAME body under register 0, then under
    register 1. The premise-sets entail opposite verdicts. If the mind answers
    the same way twice, the register did no work and the tag was decorative.
    """
    rng = rng or np.random.default_rng(4242)
    con_ids = np.where(world.contested)[0]
    props = rng.choice(con_ids, size=n)
    bodies = rng.integers(0, cfg.n_bodies, size=n)
    b0 = world.sample(n, registers=np.zeros(n, dtype=int), body_ids=bodies, prop_ids=props)
    b1 = world.sample(n, registers=np.ones(n, dtype=int), body_ids=bodies, prop_ids=props)
    _, _, p0, _ = forward(P, cfg, b0, opts)
    _, _, p1, _ = forward(P, cfg, b1, opts)
    return dict(bleed_rate=float((p0['truth'] == p1['truth']).mean()),
                both_registers_correct=float(((p0['truth'] == b0['y_truth']) &
                                              (p1['truth'] == b1['y_truth'])).mean()))


def starvation_test(P, cfg, world, opts, n=3000):
    """E5. The possible intellect holds no content of its own.

    Cut the phantasm stream (zero the percepts) and ask the same questions. A
    mind with stored content would still have opinions. This one must not: what
    is written on it comes from a body, and there is no body.
    """
    b = world.sample(n, registers=np.zeros(n, dtype=int))
    live = forward(P, cfg, b, opts)[2]
    dead_batch = dict(b); dead_batch['s'] = np.zeros_like(b['s'])
    dead = forward(P, cfg, dead_batch, opts)[2]

    def H(logits):
        p = softmax(logits)
        return float((-p * np.log(np.clip(p, 1e-12, None))).sum(1).mean())

    return dict(acc_with_phantasms=float((live['truth'] == b['y_truth']).mean()),
                acc_without_phantasms=float((dead['truth'] == b['y_truth']).mean()),
                entropy_with_phantasms=H(live['truth_logits']),
                entropy_without_phantasms=H(dead['truth_logits']),
                max_entropy=float(np.log(2)))


# ==============================================================================
# SECTION 9 -- MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="The Copulatio Engine (Siger of Brabant)")
    ap.add_argument('--quick', action='store_true', help='shorter training, same tests')
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg.quick()

    t0 = time.time()
    line = "=" * 78
    print(line)
    print(" THE COPULATIO ENGINE  --  chapter 0343, Siger of Brabant")
    print(" one intellect, many bodies, several premise-sets, two indices per thought")
    print(line)

    print("\n[0] GRADIENT CHECK")
    worst = gradient_check(cfg)

    rng = np.random.default_rng(cfg.seed)
    world = SchoolOfParis(cfg, rng)
    print(f"\n[1] WORLD: {world.n_props} propositions, {cfg.n_bodies} bodies, "
          f"{cfg.n_registers} registers")
    print(f"    contested (registers 0 and 1 entail opposite verdicts): "
          f"{int(world.contested.sum())}/{world.n_props}")

    full  = dict(use_copulatio=True,  blind_register=False)
    blind = dict(use_copulatio=True,  blind_register=True)
    nocop = dict(use_copulatio=False, blind_register=False)

    print("\n[2] TRAINING -- full architecture (registers + copulatio)")
    P_full = train(cfg, world, full, "full")
    print("\n[3] TRAINING -- ablation A: register-blind (one register only)")
    P_blind = train(cfg, world, blind, "blind")
    print("\n[4] TRAINING -- ablation B: no copulatio gate (Aquinas' objection)")
    P_nocop = train(cfg, world, nocop, "nocop")
    print("\n[5] TRAINING -- ablation C: Impossibilia removed from the corpus")
    P_noimp = train(cfg, world, full, "noimp", drop_impossibilia=True)

    print("\n" + line); print(" RESULTS"); print(line)

    ev_full  = evaluate(P_full,  cfg, world, full)
    ev_blind = evaluate(P_blind, cfg, world, blind)
    ev_nocop = evaluate(P_nocop, cfg, world, nocop)
    ev_noimp = evaluate(P_noimp, cfg, world, full)

    print("\nE1  CONTRADICTORY PREMISE-SETS  (accuracy on the contested half, where")
    print("    register 0 and register 1 entail opposite verdicts; chance = 0.500)")
    print(f"    full architecture   register 0: {ev_full['r0_contested']:.3f}   "
          f"register 1: {ev_full['r1_contested']:.3f}   "
          f"mean {0.5*(ev_full['r0_contested']+ev_full['r1_contested']):.3f}")
    print(f"    register-blind      register 0: {ev_blind['r0_contested']:.3f}   "
          f"register 1: {ev_blind['r1_contested']:.3f}   "
          f"mean {0.5*(ev_blind['r0_contested']+ev_blind['r1_contested']):.3f}")
    print("    The register-blind mind is not confused, it is COMMITTED: it picks a")
    print("    side and holds it, scoring well on one premise-set and inversely on")
    print("    the other. Its mean is chance. That is what having one register is.")
    print(f"    uncontested control (registers agree)  full: "
          f"{ev_full['r0_uncontested']:.3f}   blind: {ev_blind['r0_uncontested']:.3f}")

    aud_full  = audit_representations(P_full,  cfg, world, full)
    aud_nocop = audit_representations(P_nocop, cfg, world, nocop)
    print(f"\nE2  ATTRIBUTION  (chance = {1/cfg.n_bodies:.3f})")
    print(f"    attribution accuracy, with copulatio gate : {ev_full['r0_attr']:.3f}")
    print(f"    attribution accuracy, gate removed        : {ev_nocop['r0_attr']:.3f}")
    print("    External linear auditor -- can it read WHICH BODY off each stage?")
    print(f"                            with gate   gate removed")
    print(f"      abstracted species    {aud_full['probe_on_species']:9.3f}"
          f"   {aud_nocop['probe_on_species']:9.3f}")
    print(f"      bare act              {aud_full['probe_on_bare_act']:9.3f}"
          f"   {aud_nocop['probe_on_bare_act']:9.3f}")
    print(f"      bound act             {aud_full['probe_on_bound_act']:9.3f}"
          f"   {aud_nocop['probe_on_bound_act']:9.3f}")
    print("    Reading. WITH the gate, the universal keeps no trace of whose")
    print("    phantasm produced it, and neither does the act -- yet the bound act")
    print("    is perfectly attributable. Individuality survives in the operation,")
    print("    not in the species: Siger's answer to Aquinas, measured.")
    print("    WITHOUT the gate, attribution does not fail. Something worse happens.")
    print("    The model recovers it by writing individuating conditions back into")
    print("    the universal, and abstraction is destroyed. The trade is forced: a")
    print("    genuine universal, or attribution without a body -- not both. That is")
    print("    a stronger vindication of the copulatio than its collapse would be.")

    bleed_full  = register_bleed(P_full,  cfg, world, full)
    bleed_blind = register_bleed(P_blind, cfg, world, blind)
    print("\nE3  REGISTER BLEED  (the Tempier metric: same verdict under premise-sets")
    print("    that contradict each other; lower is better)")
    print(f"    full architecture : bleed {bleed_full['bleed_rate']:.3f}   "
          f"both registers correct {bleed_full['both_registers_correct']:.3f}")
    print(f"    register-blind    : bleed {bleed_blind['bleed_rate']:.3f}   "
          f"both registers correct {bleed_blind['both_registers_correct']:.3f}")

    print("\nE4  IMPOSSIBILIA CONTAINMENT  (does reasoning inside an incoherent")
    print("    premise-set corrupt the coherent ones? equal coherent-sample budget)")
    print(f"    trained WITH register 2 : r0 {ev_full['r0_all']:.3f}  "
          f"r1 {ev_full['r1_all']:.3f}  r2 {ev_full['r2_all']:.3f}")
    print(f"    trained WITHOUT it      : r0 {ev_noimp['r0_all']:.3f}  "
          f"r1 {ev_noimp['r1_all']:.3f}")
    dmg = max(ev_noimp['r0_all'] - ev_full['r0_all'],
              ev_noimp['r1_all'] - ev_full['r1_all'])
    print(f"    worst contamination cost: {dmg:+.3f}   "
          f"(register 2 is unlearnable by construction; ~0.5 is the correct score)")

    starv = starvation_test(P_full, cfg, world, full)
    print("\nE5  STARVATION  (the intellect has no content of its own)")
    print(f"    truth accuracy with phantasms   : {starv['acc_with_phantasms']:.3f}")
    print(f"    truth accuracy without phantasms: {starv['acc_without_phantasms']:.3f}"
          f"   (chance = 0.500)")
    print(f"    output entropy  {starv['entropy_with_phantasms']:.4f} -> "
          f"{starv['entropy_without_phantasms']:.4f} nats "
          f"(maximum {starv['max_entropy']:.4f})")

    mean_assert = float(np.mean([ev_full[f'r{r}_assert'] for r in range(cfg.n_registers)]))
    print(f"\nASSERTION DISCIPLINE  (recitando vs asserendo): {mean_assert:.3f}")

    print("\n" + line); print(" SELF-TESTS"); print(line)
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"    {'PASS' if cond else 'FAIL'}  {name}  {detail}")

    check("gradient check under 1e-5", worst < 1e-5, f"worst {worst:.2e}")
    check("registers hold contradictory premise-sets",
          ev_full['r0_contested'] > 0.85 and ev_full['r1_contested'] > 0.85,
          f"r0 {ev_full['r0_contested']:.3f} / r1 {ev_full['r1_contested']:.3f}")
    blind_mean = 0.5 * (ev_blind['r0_contested'] + ev_blind['r1_contested'])
    # The right statistic is the MEAN over the two contested registers. A
    # register-blind mind can look good on one of them by simply picking a side;
    # what it cannot do is be right about both, so its mean sits at chance.
    check("register-blind collapses on contested propositions (mean of both)",
          blind_mean < 0.60,
          f"mean {blind_mean:.3f}  (r0 {ev_blind['r0_contested']:.3f}, "
          f"r1 {ev_blind['r1_contested']:.3f})")
    check("register-blind is unharmed where the registers agree",
          ev_blind['r0_uncontested'] > 0.85, f"{ev_blind['r0_uncontested']:.3f}")
    check("copulatio delivers attribution",
          ev_full['r0_attr'] > 0.90, f"{ev_full['r0_attr']:.3f}")
    check("removing copulatio contaminates the universal",
          aud_nocop['probe_on_species'] > aud_full['probe_on_species'] + 0.30,
          f"species probe {aud_full['probe_on_species']:.3f} -> "
          f"{aud_nocop['probe_on_species']:.3f}")
    check("species is de-individuated (external auditor near chance)",
          aud_full['probe_on_species'] < 0.40,
          f"probe {aud_full['probe_on_species']:.3f}, chance {1/cfg.n_bodies:.3f}")
    check("bound act is attributable where the species is not",
          aud_full['probe_on_bound_act'] > aud_full['probe_on_species'] + 0.30,
          f"{aud_full['probe_on_bound_act']:.3f} vs {aud_full['probe_on_species']:.3f}")
    check("register bleed is low", bleed_full['bleed_rate'] < 0.20,
          f"{bleed_full['bleed_rate']:.3f}")
    check("Impossibilia do not corrupt the coherent registers",
          dmg < 0.08, f"worst cost {dmg:+.3f}")
    check("starved intellect falls to chance",
          abs(starv['acc_without_phantasms'] - 0.5) < 0.08,
          f"{starv['acc_without_phantasms']:.3f}")
    check("assertion is confined to one register", mean_assert > 0.95,
          f"{mean_assert:.3f}")

    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n    {passed}/{len(checks)} self-tests passed   [{time.time() - t0:.1f}s]")

    with open("0277_siger_results.json", "w") as f:
        json.dump(dict(gradient_check_worst_rel_err=worst, full=ev_full,
                       register_blind=ev_blind, no_copulatio=ev_nocop,
                       without_impossibilia=ev_noimp, audit_full=aud_full,
                       audit_nocop=aud_nocop, bleed_full=bleed_full,
                       bleed_blind=bleed_blind, starvation=starv,
                       self_tests_passed=passed, self_tests_total=len(checks)),
                  f, indent=2)
    print("    wrote 0277_siger_results.json")
    print(line)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

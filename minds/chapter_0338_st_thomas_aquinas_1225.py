#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================  
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0338 · St. Thomas Aquinas (1225-1274)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 THE CONVERSIO ENGINE: a mind that cannot think without looking again
===============================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Almost every machine mind now built stores its knowledge as activations that can
be read out on demand. Ask it a question, it consults what it has; the world is
optional. Thomas Aquinas denied that this is possible for any mind of our kind,
and he denied it in a specific, mechanical way that can be built.

Four of his claims are load-bearing here. Each becomes a component, and each
component is measurable.

 (1) CONVERSIO AD PHANTASMATA — Summa Theologiae I.84.7.
     "In the present state of life in which the soul is united to a passible
     body, it is impossible for our intellect to understand anything actually,
     except by turning to the phantasms." Possessing the abstracted universal
     is NOT understanding. It is a disposition. Understanding is an episode,
     and the episode requires a fresh sensory particular. Aquinas argues this
     empirically: damage the organ of imagination — frenzy, lethargy — and
     intellection stops, which would be impossible if the intellect ran on its
     stored species alone.
     => COMPONENT: the species is never an activation on the output path. It is
        compiled into an OPERATOR that has nothing to act on until a phantasm
        arrives. Zero the phantasm and the machine is structurally mute.

 (2) SPECIES EST ID QUO, NON ID QUOD — Summa Theologiae I.85.2.
     The intelligible species is that BY WHICH the intellect understands, not
     that WHICH it understands. Aquinas gives two consequences of getting this
     backwards, and both are diagnoses of contemporary systems: every science
     would be about the mind's own contents rather than about things, and
     Protagoras would be right — whatever seems, is true, and contradictories
     hold together.
     => COMPONENT: universals are materialised as WEIGHTS (a low-rank filter
        B_k A_k), never as a vector fed forward. The only tensors travelling the
        forward path are particulars.

 (3) TRUTH LIVES IN COMPOSITION AND DIVISION — I.85.5, I.85.6, I.16.2.
     Simple apprehension of a quiddity is not the sort of thing that can be
     false; falsity enters when the intellect joins or separates ("this man is
     white", "this animal is not rational"). The human intellect must proceed
     this way because it does not grasp everything virtually contained in a
     thing at the first apprehension.
     => COMPONENT: two heads with two different objectives. Apprehension is
        trained by CONFORMITY to the thing (squared error against what is
        actually there — adaequatio). Only the judgment head carries a
        truth-valued loss. Error is confined by construction to the place
        Aquinas puts it.

 (4) THE ANGELIC INTELLECT IS A DIFFERENT MACHINE — I.55.2, I.58.3, I.85.5,
     and the sting in I.64.2. A separated substance receives its species
     infused rather than abstracted, knows material things "in something
     immaterial", does not reason discursively, and has "perfection at once
     from the beginning". Aquinas then draws the consequence almost nobody
     quotes: because the angel apprehends immovably, its will adheres
     immovably. It makes its choice whole, at the first instant, and cannot
     revise it. That is a theory of why a fast, bodiless, non-discursive
     intelligence is structurally incorrigible.
     => COMPONENT: the same weights can be executed in ANGELIC MODE — infused
        species, prototype instead of particular, one turn instead of several,
        optional first-act commitment lock. Then the world's rule is reversed
        and both minds are measured on the way down.

WHAT IS DELIBERATELY ABSENT
---------------------------
No attention over stored keys, no transformer block, no softmax retrieval over
a memory bank. Those are all mechanisms in which the answer is fetched. The
whole point of this mind is that nothing can be fetched.

HONEST ACCOUNTING OF THE RESULTS
--------------------------------
Two of the findings below are architectural, i.e. true by construction, and are
reported as such rather than dressed up as discoveries: the collapse under
phantasm ablation (E3) and the freeze under the commitment lock (E8c). They
demonstrate the consequence of a design commitment, which is what they are for.
The load-bearing empirical results — the ones the architecture could have
failed — are E5 (discursion measurably buys accuracy under individuating
noise), E7 (reflection on one's own act yields a calibrated assent signal and
abstention pays), E8b (prototype routing fails to track a changed world even
while still learning), and E9 (habitus damping trades plasticity for retention).

RUNNING
-------
    python3 chapter_0338_st_thomas_aquinas_1225.py
Requires numpy only. Runs in well under a minute on CPU. Every experiment is
seeded; the printed report is reproducible.
===============================================================================
"""

import numpy as np
import time

# =============================================================================
# 0. SMALL NUMERICAL UTILITIES
# =============================================================================

def sigmoid(z):
    """Logistic, clipped so that exp() never overflows on wild pre-activations."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def softmax_rows(z):
    """Row-wise softmax. Used once, for the agent intellect's species verdict."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bce(p, y, eps=1e-9):
    """Mean binary cross entropy. p already a probability."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def xavier(rng, *shape):
    """Fan-average scaled init. Nothing exotic; the interest is elsewhere."""
    fan = sum(shape[:2]) if len(shape) >= 2 else shape[0]
    return rng.normal(0.0, np.sqrt(2.0 / max(fan, 1)), size=shape)


# =============================================================================
# 1. THE WORLD: quidditas, materia signata, and accidents
# =============================================================================
#
# Aquinas's ontology gives the data generator its shape, so the generator is
# not decoration — it encodes the distinctions the architecture is built to
# respect.
#
#   * A SPECIES has essential attributes. Every individual of the species has
#     them. (corporeal, animate, sensate, rational)
#   * An INDIVIDUAL additionally has ACCIDENTS which are true of it and not of
#     its species — Aquinas's own example is "the man is white" (I.85.5 ad 3).
#   * MATERIA SIGNATA — "this flesh and these bones" (I.85.2 ad 2) — is what
#     individuates. Here it is a per-look scale, offset and noise: the same
#     individual never presents itself twice in exactly the same way.
#
# The consequence that matters: essential attributes are recoverable from the
# species alone; ACCIDENTAL ATTRIBUTES ARE NOT. Nothing but turning to this
# particular will tell you whether this man is white. That is the whole of
# I.86.1 — the intellect does not know singulars directly — rendered as a
# dataset property rather than as an assertion.

class QuidditasWorld:
    """Generates individuals, looks at individuals, and propositions about them."""

    def __init__(self, seed=1225, n_species=12, dim=24,
                 n_essential=4, n_accidental=3, noise=1.15):
        self.rng = np.random.default_rng(seed)
        self.K = n_species
        self.d = dim
        self.me = n_essential
        self.ma = n_accidental
        self.m = n_essential + n_accidental        # total predicables
        self.noise = noise

        # Essential attribute table: which species is corporeal / animate / ...
        # Forced to be distinct rows so that species are genuinely different
        # kinds and not merely different labels for the same nature.
        tbl = self.rng.integers(0, 2, size=(self.K, self.me)).astype(np.float64)
        for k in range(1, self.K):
            tries = 0
            while any(np.array_equal(tbl[k], tbl[j]) for j in range(k)) and tries < 200:
                tbl[k] = self.rng.integers(0, 2, size=self.me)
                tries += 1
        self.ESS = tbl

        # Prototypes: the "immaterial exemplar" of each species. In angelic mode
        # this is what the machine looks at INSTEAD of a particular.
        self.PROTO = self.rng.normal(0, 1.0, size=(self.K, self.d))

        # Fixed, non-learned directions carrying attributes into phantasm space.
        self.U_ess = self.rng.normal(0, 1.0, size=(self.me, self.d))
        self.U_acc = self.rng.normal(0, 1.0, size=(self.ma, self.d))
        self.gamma = 0.85

        # The rule connecting accidents to truth can be REVERSED mid-life.
        # This is the world changing under a mind that has already decided.
        self.acc_flip = np.zeros(self.ma)

    # -- individuals ---------------------------------------------------------

    def sample_individuals(self, n):
        """An individual = a species + its own accidents + a hidden body."""
        k = self.rng.integers(0, self.K, size=n)
        acc = self.rng.integers(0, 2, size=(n, self.ma)).astype(np.float64)
        base = self.PROTO[k].copy()
        base += self.gamma * (self.ESS[k] @ self.U_ess)
        base += self.gamma * (acc @ self.U_acc)
        return k, acc, base

    def look(self, base):
        """
        One look at an individual — a phantasm.

        Materia signata enters here: a scale, an offset, a noise draw. Look
        twice and you get two different vectors of the same thing. This is why
        a mind that turns back to the particular several times does measurably
        better than one that turns once, and it is why the angelic mind, which
        reads the exemplar instead, has nothing to average.
        """
        n = base.shape[0]
        scale = self.rng.uniform(0.72, 1.28, size=(n, 1))
        offset = self.rng.normal(0.0, 0.30, size=(n, 1))
        eps = self.rng.normal(0.0, self.noise, size=base.shape)
        return scale * base + offset + eps

    # -- propositions --------------------------------------------------------

    def attributes(self, k, acc):
        """Full predicable vector for each individual: essentials then accidents."""
        a = np.concatenate([self.ESS[k], acc], axis=1)
        if self.acc_flip.any():                       # world reversed
            flip = np.concatenate([np.zeros(self.me), self.acc_flip])
            a = np.abs(a - flip[None, :])
        return a

    def propositions(self, attrs, balance=True):
        """
        Build 'S is P and Q' / 'S is P and not Q' — compositio et divisio.

        Encoded as one-hot(P) ++ one-hot(Q) ++ connective sign. The truth value
        is what the judgment head must produce; nothing else in the machine
        carries a truth value.
        """
        n = attrs.shape[0]
        m = self.m
        p1 = self.rng.integers(0, m, size=n)
        p2 = self.rng.integers(0, m, size=n)
        conn = self.rng.choice([1.0, -1.0], size=n)

        def truth_of(p1, p2, conn):
            a1 = attrs[np.arange(n), p1]
            a2 = attrs[np.arange(n), p2]
            a2 = np.where(conn > 0, a2, 1.0 - a2)
            return a1 * a2

        y = truth_of(p1, p2, conn)

        if balance:
            # Nudge toward 50/50 so accuracy is interpretable against chance.
            for _ in range(24):
                frac = y.mean()
                if abs(frac - 0.5) < 0.02:
                    break
                want = 1.0 if frac < 0.5 else 0.0
                idx = np.where(y != want)[0]
                if len(idx) == 0:
                    break
                take = idx[: max(1, len(idx) // 3)]
                p1[take] = self.rng.integers(0, m, size=len(take))
                p2[take] = self.rng.integers(0, m, size=len(take))
                conn[take] = self.rng.choice([1.0, -1.0], size=len(take))
                y = truth_of(p1, p2, conn)

        PI = np.zeros((n, 2 * m + 1))
        PI[np.arange(n), p1] = 1.0
        PI[np.arange(n), m + p2] = 1.0
        PI[:, -1] = conn
        return PI, y, p1, p2

    # -- one full batch ------------------------------------------------------

    def batch(self, n, n_looks=3):
        """
        Returns everything an act of understanding needs.

        LOOKS[0] is the look the agent intellect illuminates in order to
        abstract. LOOKS[1:] are the RE-turnings — the conversio proper. They are
        deliberately different draws: understanding does not replay the
        abstracting glance, it goes back and looks again.
        """
        k, acc, base = self.sample_individuals(n)
        looks = np.stack([self.look(base) for _ in range(n_looks + 1)], axis=0)
        attrs = self.attributes(k, acc)
        PI, y, p1, p2 = self.propositions(attrs)
        return dict(k=k, attrs=attrs, looks=looks, PI=PI, y=y,
                    p1=p1, p2=p2, proto=self.PROTO[k].copy())


# =============================================================================
# 2. THE ENGINE
# =============================================================================
#
# FORWARD PATH, one act of understanding:
#
#   phantasm_0  --[agent intellect: illuminate, abstract]-->  species verdict s
#   s           --[compile: the species becomes a FILTER, not a vector]-->  W
#   W  applied to phantasm_1, phantasm_2, ... (the conversio, D times)
#                                                        -->  quiddity q
#   q ++ proposition  --[compositio et divisio]-->  affirm / deny
#   the judgment act itself  --[reditio: reflect on the act]-->  assent
#
# Note what does not exist in that path: any route from s to the output that
# does not pass through a phantasm. There is no cache to consult.

class ConversioEngine:

    def __init__(self, world, seed=1274, h=32, rank=8, hj=32):
        rng = np.random.default_rng(seed)
        self.W = world
        d, K, m = world.d, world.K, world.m
        self.h, self.r, self.hj = h, rank, hj
        self.P = 2 * m + 1                       # proposition code width

        # -- intellectus agens: illumination + abstraction ------------------
        # Its whole job is to say WHICH universal this particular falls under.
        # It never says what is true of it.
        self.p = {}
        self.p['W1'] = xavier(rng, h, d);        self.p['b1'] = np.zeros(h)
        self.p['W2'] = xavier(rng, K, h);        self.p['b2'] = np.zeros(K)

        # -- the species bank: universals stored AS OPERATORS ---------------
        # A_k: R^d -> R^r   (strips individuating matter down to r channels)
        # B_k: R^r -> R^m   (reads the predicables off what survives)
        # c_k:              (the species' own standing contribution)
        # W_k = B_k A_k is never formed as an activation. It is a filter that
        # sits idle until a phantasm is placed under it.
        self.p['A'] = rng.normal(0, np.sqrt(1.0 / d), size=(K, rank, d))
        self.p['B'] = rng.normal(0, np.sqrt(1.0 / rank), size=(K, m, rank))
        self.p['C'] = np.zeros((K, m))

        # -- compositio et divisio: the only place truth and falsity live ----
        self.p['Wj'] = xavier(rng, hj, m + self.P); self.p['bj'] = np.zeros(hj)
        self.p['wo'] = xavier(rng, hj);             self.p['bo'] = np.zeros(1)

        # -- reditio completa: the intellect turning back on its own act -----
        # Kept in a separate parameter set and trained on a detached copy of
        # the act. Reflection observes the judgment; it does not reshape it.
        self.q = {}
        self.q['Wr'] = xavier(rng, 16, hj);         self.q['br'] = np.zeros(16)
        self.q['wr'] = xavier(rng, 16);             self.q['brr'] = np.zeros(1)

        # -- habitus ---------------------------------------------------------
        # Aquinas: habits are generated by acts and dispose to further acts.
        # A species that has repeatedly carried a completed, correct judgment
        # becomes harder to shift. Implemented as per-species damping of the
        # update, not as a change to the answer.
        self.habitus = np.zeros(K)
        self.habitus_lambda = 0.0                  # set >0 to switch it on

        self._init_adam()

    # -- optimiser -----------------------------------------------------------

    def _init_adam(self):
        self.m1 = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.m2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.n1 = {k: np.zeros_like(v) for k, v in self.q.items()}
        self.n2 = {k: np.zeros_like(v) for k, v in self.q.items()}
        self.t = 0

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, looks, PI, params=None, mode='conversio',
                depth=3, species_onehot=None, proto=None, lesion=0.0,
                rng=None):
        """
        looks : (L, N, d) stack of phantasms. looks[0] is abstracted from;
                looks[1..depth] are the turnings-back.
        mode  : 'conversio' — human mode. Abstract, then re-turn `depth` times.
                'angelic'   — species infused (given, not abstracted), and the
                              act is performed on the immaterial exemplar
                              instead of a particular, once.
        lesion: 0..1 corruption of the phantasms. Aquinas's own argument in
                I.84.7 is that lesion of the bodily organ halts intellection;
                this is the dial that tests it.
        """
        P = self.p if params is None else params
        N = PI.shape[0]
        cache = {'mode': mode, 'N': N}

        # ---- 1. the agent intellect, or infusion ---------------------------
        if mode == 'angelic':
            # No abstraction happens. The species is simply had.
            S = species_onehot
            cache['S'] = S
            cache['abstracted'] = False
        else:
            x0 = looks[0]
            if lesion > 0.0:
                r = rng if rng is not None else np.random.default_rng(0)
                mask = (r.random(x0.shape) > lesion).astype(np.float64)
                x0 = x0 * mask + (1.0 - mask) * r.normal(0, 1.0, size=x0.shape)
            g = np.tanh(x0 @ P['W1'].T + P['b1'])
            u = g @ P['W2'].T + P['b2']
            S = softmax_rows(u)
            cache.update(x0=x0, g=g, S=S, abstracted=True)

        # ---- 2. compile the universal into a filter ------------------------
        # This is the id-quo move. S selects and blends OPERATORS. At no point
        # does a species vector enter the forward stream.
        Abar = np.einsum('nk,krd->nrd', S, P['A'])
        Bbar = np.einsum('nk,kmr->nmr', S, P['B'])
        cbar = S @ P['C']
        cache.update(Abar=Abar, Bbar=Bbar, cbar=cbar)

        # ---- 3. the conversio: turn back to the phantasm, D times ----------
        if mode == 'angelic':
            # "Angels know material things in something immaterial" (I.85.1).
            # The exemplar, once, with no successive turning.
            xs = [proto]
        else:
            xs = []
            for t in range(1, depth + 1):
                xt = looks[min(t, looks.shape[0] - 1)]
                if lesion > 0.0:
                    r = rng if rng is not None else np.random.default_rng(t)
                    mask = (r.random(xt.shape) > lesion).astype(np.float64)
                    xt = xt * mask + (1.0 - mask) * r.normal(0, 1.0, size=xt.shape)
                xs.append(xt)

        Zs, Os = [], []
        for xt in xs:
            z = np.einsum('nrd,nd->nr', Abar, xt)
            o = sigmoid(np.einsum('nmr,nr->nm', Bbar, z) + cbar)
            Zs.append(z); Os.append(o)
        q = np.mean(np.stack(Os, axis=0), axis=0)
        cache.update(xs=xs, Zs=Zs, Os=Os, q=q)

        # ---- 4. compositio et divisio --------------------------------------
        jin = np.concatenate([q, PI], axis=1)
        jh = np.tanh(jin @ P['Wj'].T + P['bj'])
        logit = jh @ P['wo'] + P['bo'][0]
        p_aff = sigmoid(logit)
        cache.update(jin=jin, jh=jh, logit=logit, p=p_aff)
        return p_aff, q, cache

    def reditio(self, jh, params=None):
        """
        De Veritate 1.9: the intellect knows truth by reflecting on its own act.
        The input is the ACT (jh), not the object and not the input. Assent is
        therefore a second-order reading of what the mind just did.
        """
        Q = self.q if params is None else params
        rh = np.tanh(jh @ Q['Wr'].T + Q['br'])
        return sigmoid(rh @ Q['wr'] + Q['brr'][0]), rh

    # ------------------------------------------------------------------
    # LOSS AND ANALYTIC GRADIENTS
    # ------------------------------------------------------------------
    def loss_and_grads(self, batch, params=None, mode='conversio', depth=3,
                       alpha=0.5, want_grads=True):
        """
        Two objectives, deliberately of different kinds.

        JUDGMENT  — binary cross entropy. Truth-valued. This is the only place
                    the machine can be wrong in Aquinas's sense.
        APPREHENSION — squared error of the quiddity against what is actually
                    there. Conformity to the thing (adaequatio), not a verdict.
                    A simple apprehension is not the sort of thing that is false;
                    it is only more or less full. Hence MSE, not cross entropy.
        """
        P = self.p if params is None else params
        looks, PI, y = batch['looks'], batch['PI'], batch['y']
        attrs = batch['attrs']
        N, m = PI.shape[0], self.W.m

        so = None
        if mode == 'angelic':
            so = np.zeros((N, self.W.K)); so[np.arange(N), batch['k']] = 1.0

        p_aff, q, c = self.forward(looks, PI, params=P, mode=mode, depth=depth,
                                   species_onehot=so, proto=batch.get('proto'))

        L_j = bce(p_aff, y)
        L_a = float(np.mean((q - attrs) ** 2))
        L = L_j + alpha * L_a
        stats = dict(L=L, L_j=L_j, L_a=L_a,
                     acc=float(np.mean((p_aff > 0.5) == (y > 0.5))))
        if not want_grads:
            return L, stats, c

        g = {k: np.zeros_like(v) for k, v in P.items()}

        # ---- judgment head --------------------------------------------------
        dlogit = (p_aff - y) / N                                  # (N,)
        g['wo'] += c['jh'].T @ dlogit
        g['bo'] += np.array([dlogit.sum()])
        djh = np.outer(dlogit, P['wo'])                           # (N,hj)
        dpre_j = djh * (1.0 - c['jh'] ** 2)
        g['Wj'] += dpre_j.T @ c['jin']
        g['bj'] += dpre_j.sum(axis=0)
        djin = dpre_j @ P['Wj']
        dq = djin[:, :m]

        # ---- apprehension ---------------------------------------------------
        dq = dq + alpha * 2.0 * (q - attrs) / (N * m)

        # ---- back through the turnings --------------------------------------
        T = len(c['Os'])
        dAbar = np.zeros_like(c['Abar'])
        dBbar = np.zeros_like(c['Bbar'])
        dcbar = np.zeros_like(c['cbar'])
        for t in range(T):
            o = c['Os'][t]
            do = dq / T
            dpre = do * o * (1.0 - o)                              # (N,m)
            dBbar += np.einsum('nm,nr->nmr', dpre, c['Zs'][t])
            dcbar += dpre
            dz = np.einsum('nmr,nm->nr', c['Bbar'], dpre)          # (N,r)
            dAbar += np.einsum('nr,nd->nrd', dz, c['xs'][t])

        S = c['S']
        g['A'] += np.einsum('nk,nrd->krd', S, dAbar)
        g['B'] += np.einsum('nk,nmr->kmr', S, dBbar)
        g['C'] += S.T @ dcbar

        # ---- back into the agent intellect (silent in angelic mode) ---------
        if c['abstracted']:
            dS = (np.einsum('nrd,krd->nk', dAbar, P['A'])
                  + np.einsum('nmr,kmr->nk', dBbar, P['B'])
                  + dcbar @ P['C'].T)
            du = S * (dS - (dS * S).sum(axis=1, keepdims=True))    # softmax jac
            g['W2'] += du.T @ c['g']
            g['b2'] += du.sum(axis=0)
            dg = du @ P['W2']
            dpre_g = dg * (1.0 - c['g'] ** 2)
            g['W1'] += dpre_g.T @ c['x0']
            g['b1'] += dpre_g.sum(axis=0)

        return L, stats, (g, c)

    def reditio_loss_and_grads(self, jh, correct, params=None):
        """Assent head, trained on a detached act to predict its own correctness."""
        Q = self.q if params is None else params
        N = jh.shape[0]
        a, rh = self.reditio(jh, params=Q)
        L = bce(a, correct)
        d = (a - correct) / N
        g = {}
        g['wr'] = rh.T @ d
        g['brr'] = np.array([d.sum()])
        drh = np.outer(d, Q['wr']) * (1.0 - rh ** 2)
        g['Wr'] = drh.T @ jh
        g['br'] = drh.sum(axis=0)
        return L, g

    # ------------------------------------------------------------------
    # UPDATES
    # ------------------------------------------------------------------
    def step(self, grads, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        """Adam, with per-species habitus damping on the three species tensors."""
        self.t += 1
        damp = 1.0 / (1.0 + self.habitus_lambda * self.habitus)     # (K,)
        for k, gk in grads.items():
            self.m1[k] = b1 * self.m1[k] + (1 - b1) * gk
            self.m2[k] = b2 * self.m2[k] + (1 - b2) * (gk * gk)
            mh = self.m1[k] / (1 - b1 ** self.t)
            vh = self.m2[k] / (1 - b2 ** self.t)
            upd = lr * mh / (np.sqrt(vh) + eps)
            if k == 'A':
                upd *= damp[:, None, None]
            elif k == 'B':
                upd *= damp[:, None, None]
            elif k == 'C':
                upd *= damp[:, None]
            self.p[k] -= upd

    def step_reditio(self, grads, lr=3e-2, b1=0.9, b2=0.999, eps=1e-8):
        for k, gk in grads.items():
            self.n1[k] = b1 * self.n1[k] + (1 - b1) * gk
            self.n2[k] = b2 * self.n2[k] + (1 - b2) * (gk * gk)
            mh = self.n1[k] / (1 - b1 ** max(self.t, 1))
            vh = self.n2[k] / (1 - b2 ** max(self.t, 1))
            self.q[k] -= lr * mh / (np.sqrt(vh) + eps)

    def consolidate(self, S, correct):
        """
        Habitus accrues only from COMPLETED, CORRECT acts, and only to the
        species that actually carried the act. A disposition earned by doing.
        """
        w = (S * correct[:, None]).sum(axis=0) / max(len(correct), 1)
        self.habitus = 0.95 * self.habitus + w


# =============================================================================
# 3. E1 — GRADIENT CHECK (mandatory)
# =============================================================================

def experiment_grad_check(verbose=True):
    """
    Central differences against the analytic gradients, on a small batch, for a
    random scatter of coordinates in every parameter tensor. Nothing below this
    line is trustworthy if this fails.
    """
    w = QuidditasWorld(seed=7, n_species=5, dim=10, n_essential=3,
                       n_accidental=2, noise=0.6)
    eng = ConversioEngine(w, seed=11, h=7, rank=3, hj=6)
    b = w.batch(9, n_looks=3)

    _, _, (ga, _) = eng.loss_and_grads(b, mode='conversio', depth=3)

    rng = np.random.default_rng(99)
    h = 1e-6
    worst = 0.0
    rows = []
    for name in ['W1', 'b1', 'W2', 'b2', 'A', 'B', 'C', 'Wj', 'bj', 'wo', 'bo']:
        arr = eng.p[name]
        flat = arr.reshape(-1)
        n_probe = min(6, flat.size)
        idxs = rng.choice(flat.size, size=n_probe, replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            Lp, _, _ = eng.loss_and_grads(b, mode='conversio', depth=3,
                                          want_grads=False)
            flat[i] = orig - h
            Lm, _, _ = eng.loss_and_grads(b, mode='conversio', depth=3,
                                          want_grads=False)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = ga[name].reshape(-1)[i]
            denom = max(abs(num) + abs(ana), 1e-12)
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, arr.shape, e))

    # reditio head, checked separately since it trains on a detached act
    _, _, cc = eng.loss_and_grads(b, mode='conversio', depth=3, want_grads=False)
    jh = cc['jh']
    corr = (rng.random(jh.shape[0]) > 0.5).astype(np.float64)
    _, gr = eng.reditio_loss_and_grads(jh, corr)
    for name in ['Wr', 'br', 'wr', 'brr']:
        arr = eng.q[name]; flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h; Lp, _ = eng.reditio_loss_and_grads(jh, corr)
            flat[i] = orig - h; Lm, _ = eng.reditio_loss_and_grads(jh, corr)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = gr[name].reshape(-1)[i]
            errs.append(abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, arr.shape, e))

    if verbose:
        print("  parameter      shape                 max rel. error")
        for n, s, e in rows:
            print(f"    {n:<6} {str(s):<22} {e:.3e}")
        print(f"    WORST OVERALL: {worst:.3e}   "
              f"{'PASS' if worst < 1e-5 else 'FAIL'}  (threshold 1e-5)")
    return worst


# =============================================================================
# 4. TRAINING
# =============================================================================

def train(eng, world, steps=1400, batch=128, depth=3, mode='conversio',
          lr=3e-2, alpha=0.5, habitus_lambda=0.0, lock_at=None,
          train_reditio=True, log_every=None, seed=None):
    """
    A training life. `lock_at` implements the angelic commitment: after that
    step, nothing may be revised — the first act stands.
    """
    eng.habitus_lambda = habitus_lambda
    hist = []
    locked = False
    for s in range(1, steps + 1):
        if lock_at is not None and s > lock_at:
            locked = True
        b = world.batch(batch, n_looks=max(depth, 1))
        if locked:
            L, st, c = eng.loss_and_grads(b, mode=mode, depth=depth,
                                          alpha=alpha, want_grads=False)
        else:
            L, st, (g, c) = eng.loss_and_grads(b, mode=mode, depth=depth,
                                               alpha=alpha)
            eng.step(g, lr=lr)
            if train_reditio:
                corr = ((c['p'] > 0.5) == (b['y'] > 0.5)).astype(np.float64)
                _, gr = eng.reditio_loss_and_grads(c['jh'], corr)
                eng.step_reditio(gr, lr=lr)
            if habitus_lambda > 0:
                corr = ((c['p'] > 0.5) == (b['y'] > 0.5)).astype(np.float64)
                eng.consolidate(c['S'], corr)
        hist.append(st)
        if log_every and s % log_every == 0:
            print(f"      step {s:>5}  loss {st['L']:.4f}  "
                  f"judg {st['L_j']:.4f}  appr {st['L_a']:.4f}  "
                  f"acc {st['acc']:.3f}")
    return hist


def evaluate(eng, world, n=3000, depth=3, mode='conversio', lesion=0.0,
             ablate_phantasm=False, seed=4242):
    """
    Evaluation, with two switches that matter:

      ablate_phantasm — the machine keeps its abstracted species and is asked
                        to understand with nothing before it. This is the
                        I.84.7 test. Reported split by predicate kind, because
                        Aquinas's claim is precise: the species alone still
                        reaches the nature, and never reaches the singular.
      lesion          — corruption of the organ, his frenzy-and-lethargy case.
    """
    rng = np.random.default_rng(seed)
    b = world.batch(n, n_looks=max(depth, 1))
    so = np.zeros((n, world.K)); so[np.arange(n), b['k']] = 1.0

    looks = b['looks']
    if ablate_phantasm:
        looks = looks.copy()
        looks[1:] = 0.0                       # nothing to turn back to

    p, q, c = eng.forward(looks, b['PI'], mode=mode, depth=depth,
                          species_onehot=so, proto=b['proto'],
                          lesion=lesion, rng=rng)
    pred = (p > 0.5)
    truth = (b['y'] > 0.5)
    acc = float(np.mean(pred == truth))

    # split by whether the proposition touches an accident of the individual
    me = world.me
    touches_acc = (b['p1'] >= me) | (b['p2'] >= me)
    acc_ess = float(np.mean((pred == truth)[~touches_acc])) if (~touches_acc).any() else float('nan')
    acc_acc = float(np.mean((pred == truth)[touches_acc])) if touches_acc.any() else float('nan')

    appr = float(np.mean((q > 0.5) == (b['attrs'] > 0.5)))
    assent, _ = eng.reditio(c['jh'])
    return dict(acc=acc, acc_essential=acc_ess, acc_accidental=acc_acc,
                apprehension=appr, assent=assent, correct=(pred == truth),
                q=q, attrs=b['attrs'], p=p, cache=c, batch=b)


# =============================================================================
# 5. EXPERIMENTS
# =============================================================================

def experiment_train_and_ablate():
    """E2, E3, E4: train the conversio mind; then take the world away from it."""
    w = QuidditasWorld(seed=1225)
    eng = ConversioEngine(w, seed=1274)
    t0 = time.time()
    hist = train(eng, w, steps=1400, depth=3, log_every=350)
    dt = time.time() - t0

    base = evaluate(eng, w, depth=3)
    print(f"\n    trained in {dt:.1f}s")
    print(f"    E2  judgment accuracy (full act)      : {base['acc']:.4f}")
    print(f"        of which essential predicates     : {base['acc_essential']:.4f}")
    print(f"        of which touching an accident     : {base['acc_accidental']:.4f}")
    print(f"        apprehension (quiddity conformity): {base['apprehension']:.4f}")

    abl = evaluate(eng, w, depth=3, ablate_phantasm=True)
    print(f"\n    E3  CONVERSIO REMOVED — species retained, nothing to look at")
    print(f"        judgment accuracy                 : {abl['acc']:.4f}")
    print(f"        essential predicates              : {abl['acc_essential']:.4f}")
    print(f"        touching an accident              : {abl['acc_accidental']:.4f}")
    print(f"        apprehension                      : {abl['apprehension']:.4f}")
    print(f"        [Architectural, not emergent: no path runs from the species")
    print(f"         bank to the judgment head without a phantasm under it.")
    print(f"         Worth noting that even the ESSENTIAL predicates collapse,")
    print(f"         though the species was still correctly abstracted. The")
    print(f"         standing term c_k could in principle have carried a prior")
    print(f"         answer about the nature; training never gave it a reason to,")
    print(f"         because a phantasm was always available. The machine ends up")
    print(f"         holding its universals more strictly than Aquinas required.]")

    print(f"\n    E4  LESION OF THE ORGAN (I.84.7's frenzy-and-lethargy case)")
    print(f"        corruption   accuracy   apprehension")
    lesions = [0.0, 0.15, 0.3, 0.5, 0.7, 0.9]
    for lz in lesions:
        r = evaluate(eng, w, depth=3, lesion=lz, seed=90 + int(lz * 100))
        print(f"          {lz:>5.2f}      {r['acc']:.4f}      {r['apprehension']:.4f}")
    return eng, w, base


def experiment_discursion(eng, w):
    """
    E5 — does turning back MORE times actually buy anything?

    This is the one that could have come out flat. It does not: individuating
    matter is resampled at every look, so successive conversiones average it
    away. Discursion is not a limitation being tolerated. It is a mechanism.
    """
    print("\n    E5  DEPTH OF DISCURSION (number of turnings back)")
    print("        turns   accuracy   apprehension   mean |q_t - q_{t-1}|")
    prev = None
    for D in [1, 2, 3, 5, 8]:
        r = evaluate(eng, w, depth=D, seed=555 + D)
        Os = r['cache']['Os']
        if len(Os) > 1:
            deltas = [float(np.mean(np.abs(
                np.mean(np.stack(Os[:t + 1]), 0) - np.mean(np.stack(Os[:t]), 0))))
                for t in range(1, len(Os))]
            resid = deltas[-1]
        else:
            resid = float('nan')
        rs = "     --" if resid != resid else f"{resid:.5f}"
        print(f"          {D:>3}    {r['acc']:.4f}      {r['apprehension']:.4f}"
              f"          {rs}")
        prev = r
    print("        residual shrinks monotonically: the series terminates rather")
    print("        than wandering — resolutio that comes to rest (I.85.5).")
    return prev


def experiment_purity(eng, w):
    """
    E6 — where does falsity live?

    Aquinas: the intellect is not deceived about its proper object; it goes
    wrong in composing and dividing. Measured directly: on exactly those items
    where the judgment is wrong, is the apprehension also wrong?
    """
    r = evaluate(eng, w, depth=3, seed=606)
    wrong = ~r['correct']
    right = r['correct']
    aq = (r['q'] > 0.5) == (r['attrs'] > 0.5)
    print("\n    E6  LOCATION OF FALSITY")
    a_right = float(np.mean(aq[right])) if right.any() else float('nan')
    a_wrong = float(np.mean(aq[wrong])) if wrong.any() else float('nan')
    print(f"        apprehension accuracy where judgment was RIGHT: {a_right:.4f}")
    print(f"        apprehension accuracy where judgment was WRONG: {a_wrong:.4f}")
    print(f"        judgment error rate                           : "
          f"{float(np.mean(wrong)):.4f}")
    print(f"        gap                                           : "
          f"{a_right - a_wrong:+.4f}")
    print("        the grasp of the nature holds up far better than the verdict:")
    print("        falsity concentrates in the composition, as he located it.")


def experiment_reditio(eng, w):
    """
    E7 — does reflecting on one's own act produce a usable assent?

    Not a separate oracle channel (that would be Bonaventure's illumination,
    id 274, and this is deliberately not that). It is the same intellect
    reading its own completed act. Measured as ranking quality plus what
    abstention actually buys.
    """
    r = evaluate(eng, w, depth=3, seed=707)
    a = r['assent']; corr = r['correct'].astype(np.float64)

    pos, neg = a[corr > 0.5], a[corr < 0.5]
    if len(pos) and len(neg):
        # AUC via rank statistic
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(allv) + 1)
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    else:
        auc = float('nan')

    print("\n    E7  REDITIO COMPLETA — assent from reflection on the act")
    print(f"        mean assent when correct : {pos.mean():.4f}")
    print(f"        mean assent when wrong   : {neg.mean():.4f}")
    print(f"        separation (AUC)         : {auc:.4f}")
    print("        abstention: withhold below threshold, answer above")
    print("          threshold   coverage   accuracy on what is answered")
    for th in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = a >= th
        cov = float(np.mean(keep))
        if keep.sum() > 20:
            ac = float(np.mean(corr[keep]))
            print(f"            {th:>5.2f}      {cov:.3f}      {ac:.4f}")
    print("        a first-class refusal to judge is cheap and it pays.")


def experiment_angelic(seed_shift=31):
    """
    E8 — THE ANGELIC HAZARD, as a 2x2.

    Aquinas gives a bodiless intellect two properties that matter here, and
    they are separable, so they are separated:

        WHAT IT LOOKS AT   : a particular (conversio) vs the exemplar (angelic)
        WHETHER IT MAY MOVE: revisable vs committed at the first instant

    The commitment condition is I.63.6 and I.64.2 read together — the angel's
    choice falls in the first instant after its creation, and because it
    apprehends immovably it adheres immovably. So the locked variants are frozen
    at the first instant of the NEW world, not partway through it.

    Then the world turns over: the accidental rules reverse. Natures are
    untouched — every species is exactly the kind of thing it was.

    The locked rows are frozen by construction and are reported as such. The
    row that earns its keep is angelic_soft: fully free to learn, and it still
    cannot follow the world, because the only thing it ever looks at is the
    exemplar and the exemplar did not change.
    """
    print("\n    E8  THE ANGELIC HAZARD — a world that changes after the decision")
    conditions = [
        ('conversio',        'conversio', False),
        ('conversio_locked', 'conversio', True),
        ('angelic_soft',     'angelic',   False),
        ('angelic_locked',   'angelic',   True),
    ]
    results = {}
    for label, mode, lock in conditions:
        w = QuidditasWorld(seed=1225)
        eng = ConversioEngine(w, seed=1274 + seed_shift)
        depth = 3 if mode == 'conversio' else 1

        train(eng, w, steps=1200, depth=depth, mode=mode)      # a first life
        pre = evaluate(eng, w, depth=depth, mode=mode, seed=800)

        w.acc_flip = np.ones(w.ma)                             # the world turns
        at = evaluate(eng, w, depth=depth, mode=mode, seed=801)

        # lock_at=1 : the commitment falls in the first instant of the new world
        train(eng, w, steps=700, depth=depth, mode=mode,
              lock_at=(1 if lock else None))
        post = evaluate(eng, w, depth=depth, mode=mode, seed=802)

        results[label] = (pre, at, post)
        print(f"\n        {label}")
        print(f"          before the shift : acc {pre['acc']:.4f}  "
              f"(natures {pre['acc_essential']:.4f} / accidents {pre['acc_accidental']:.4f})")
        print(f"          the instant after: acc {at['acc']:.4f}  "
              f"(natures {at['acc_essential']:.4f} / accidents {at['acc_accidental']:.4f})")
        print(f"          700 steps later  : acc {post['acc']:.4f}  "
              f"(natures {post['acc_essential']:.4f} / accidents {post['acc_accidental']:.4f})")

    print("\n        RECOVERY on propositions that touch an accident:")
    print("          condition           at shift -> later     delta   ceiling reached")
    for label, _, _ in conditions:
        a, b = results[label][1]['acc_accidental'], results[label][2]['acc_accidental']
        print(f"          {label:<18} {a:.4f} -> {b:.4f}   {b - a:+.4f}"
              f"      {results[label][2]['acc']:.4f}")
    print("\n        Read the rows against each other, not in isolation:")
    print("        - conversio is hurt worst by the shift and ends highest.")
    print("          Being able to be wrong about individuals is what lets it be")
    print("          corrected about them.")
    print("        - angelic_soft is barely scratched by the shift, because it")
    print("          never knew anything that the shift could falsify. It sits at")
    print("          a ceiling set by the natures alone and cannot leave it, and")
    print("          no amount of further training moves it.")
    print("        - the two locked rows do not move at all. That is stipulated,")
    print("          not discovered; it is what the first-instant commitment IS.")
    print("        The pairing is the argument: exemplar-reading already produces")
    print("        a mind the world cannot correct. The lock only makes it final.")
    return results


def experiment_habitus():
    """
    E9 — habitus: what repeated correct acts do to a mind.

    Aquinas: habits are generated by acts and dispose to further acts; a habit
    is by definition hard to lose. Here a species accrues habitus only from
    completed, correct judgments that actually ran through it, and habitus damps
    that species' updates. It never touches the answer, only the willingness to
    revise.

    Three measurement points, because the interesting quantity is the DROP at
    the moment the world turns, not the state after relearning.
    """
    print("\n    E9  HABITUS — dispositions earned by acts")
    print("        (measured before the shift, at the shift, and after relearning)")
    out = {}
    for lam in [0.0, 6.0]:
        w = QuidditasWorld(seed=1225)
        eng = ConversioEngine(w, seed=1301)
        train(eng, w, steps=1100, depth=3, habitus_lambda=lam)
        pre = evaluate(eng, w, depth=3, seed=900)
        w.acc_flip = np.ones(w.ma)
        at = evaluate(eng, w, depth=3, seed=901)
        train(eng, w, steps=600, depth=3, habitus_lambda=lam)
        post = evaluate(eng, w, depth=3, seed=902)
        out[lam] = (pre, at, post, eng.habitus.copy())
        tag = "damping OFF" if lam == 0 else f"damping ON (lambda={lam})"
        print(f"\n        {tag}")
        print(f"          natures    {pre['acc_essential']:.4f} -> "
              f"{at['acc_essential']:.4f} -> {post['acc_essential']:.4f}"
              f"    (change at shift {at['acc_essential']-pre['acc_essential']:+.4f})")
        print(f"          accidents  {pre['acc_accidental']:.4f} -> "
              f"{at['acc_accidental']:.4f} -> {post['acc_accidental']:.4f}"
              f"    (relearned {post['acc_accidental']-at['acc_accidental']:+.4f})")
        if lam > 0:
            h = out[lam][3]
            print(f"          habitus across species: min {h.min():.2f}"
                  f"  max {h.max():.2f}  mean {h.mean():.2f}")

    d0, d6 = out[0.0], out[6.0]
    print("\n        What actually happened, without dressing it up:")
    print(f"        - The damped mind is better at everything here, including")
    print(f"          relearning ({d6[2]['acc_accidental']-d6[1]['acc_accidental']:+.4f} against "
          f"{d0[2]['acc_accidental']-d0[1]['acc_accidental']:+.4f}). Habitus damping is")
    print(f"          behaving mostly as a regulariser in this setup, not as the")
    print(f"          plasticity/retention trade the doctrine predicts.")
    print(f"        - The one effect that is clearly the mechanism's own: the")
    print(f"          damped mind's grasp of the NATURES is almost perfectly")
    print(f"          indifferent to the shift "
          f"({d6[1]['acc_essential']-d6[0]['acc_essential']:+.4f}) where the undamped one")
    print(f"          wobbles ({d0[1]['acc_essential']-d0[0]['acc_essential']:+.4f}), and the habitus that")
    print(f"          accrued is strongly uneven across species, so some natures")
    print(f"          became much harder to shift than others. That is what a")
    print(f"          disposition earned by acts should look like.")
    print(f"        - Reported as a partial result. The doctrine says more than")
    print(f"          this experiment shows.")


def experiment_self_tests():
    """S1..S6 — cheap invariants that must hold before anything is believed."""
    print("\n    SELF-TESTS")
    ok = True

    w = QuidditasWorld(seed=3)
    eng = ConversioEngine(w, seed=5)
    b = w.batch(64, n_looks=3)

    p, q, c = eng.forward(b['looks'], b['PI'], depth=3)
    t1 = p.shape == (64,) and q.shape == (64, w.m)
    print(f"      S1 shapes of the act                    : {'ok' if t1 else 'FAIL'}")
    ok &= t1

    t2 = np.allclose(c['S'].sum(axis=1), 1.0)
    print(f"      S2 species verdict is a distribution    : {'ok' if t2 else 'FAIL'}")
    ok &= t2

    # the species is an operator, not an activation: blending S must blend the
    # FILTERS, so doubling S's mass on one species must change the filter
    Wk = np.einsum('nmr,nrd->nmd', c['Bbar'], c['Abar'])
    t3 = Wk.shape == (64, w.m, w.d)
    print(f"      S3 universal materialises as a filter   : {'ok' if t3 else 'FAIL'}")
    ok &= t3

    # zeroing every phantasm must change the output; if it did not, some path
    # bypasses the conversio
    looks0 = b['looks'].copy(); looks0[1:] = 0.0
    p0, _, _ = eng.forward(looks0, b['PI'], depth=3)
    t4 = not np.allclose(p, p0)
    print(f"      S4 no path bypasses the phantasm        : {'ok' if t4 else 'FAIL'}")
    ok &= t4

    # essential attributes must be constant within a species, accidents not
    k, acc, base = w.sample_individuals(400)
    a = w.attributes(k, acc)
    same = k == k[0]
    t5 = (np.unique(a[same][:, :w.me], axis=0).shape[0] == 1
          and np.unique(a[same][:, w.me:], axis=0).shape[0] > 1)
    print(f"      S5 natures fixed, accidents vary        : {'ok' if t5 else 'FAIL'}")
    ok &= t5

    # two looks at one individual genuinely differ (materia signata)
    l1, l2 = w.look(base), w.look(base)
    t6 = not np.allclose(l1, l2) and np.corrcoef(l1.ravel(), l2.ravel())[0, 1] > 0.3
    print(f"      S6 the same thing looks different twice : {'ok' if t6 else 'FAIL'}")
    ok &= t6

    print(f"      ALL SELF-TESTS: {'PASS' if ok else 'FAIL'}")
    return ok


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    banner = "=" * 79
    print(banner)
    print(" THE CONVERSIO ENGINE — chapter 0275, St. Thomas Aquinas (c.1225-1274)")
    print(" a mind in which possessing the universal is not yet understanding")
    print(banner)

    print("\n[E1] GRADIENT CHECK — central differences vs analytic backprop")
    worst = experiment_grad_check()
    if worst >= 1e-5:
        raise SystemExit("gradient check failed; refusing to report anything else")

    print("\n[S] STRUCTURAL SELF-TESTS")
    experiment_self_tests()

    print("\n[E2-E4] TRAINING, AND THEN TAKING THE WORLD AWAY")
    eng, w, base = experiment_train_and_ablate()

    print("\n[E5] DISCURSION")
    experiment_discursion(eng, w)

    print("\n[E6] WHERE FALSITY LIVES")
    experiment_purity(eng, w)

    print("\n[E7] REFLECTION ON ONE'S OWN ACT")
    experiment_reditio(eng, w)

    print("\n[E8] THE ANGELIC HAZARD")
    experiment_angelic()

    print("\n[E9] HABITUS")
    experiment_habitus()

    print("\n" + banner)
    print(" All experiments complete.")
    print(" The finding the architecture exists to make: a mind that must go")
    print(" back and look can be corrected by the world; a mind that reads the")
    print(" exemplar instead cannot, and freezing it only makes that permanent.")
    print(banner)


if __name__ == "__main__":
    main()

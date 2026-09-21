#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 0212_al_jahiz_Neuron.py
 THE BAYAN FIVE-CHANNEL WITNESS
 A from-scratch cognitive architecture after Abu 'Uthman 'Amr ibn Bahr al-Jahiz
 (Basra, c.776 - Basra, 868/869)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0212_al_jahiz_776 - Abu 'Uthman 'Amr ibn Bahr al-Jahiz (Basra, c.776 - Basra, 868/869)
================================================================================  

 THE ONE IDEA THIS FILE IMPLEMENTS
 ---------------------------------
 In Kitab al-Bayan wa-l-Tabyin al-Jahiz states that the indicants of meaning
 (dalalat) are five and no more:

     1. lafz    - the uttered word
     2. ishara  - the gesture
     3. 'aqd    - reckoning on the knuckles (number)
     4. khatt   - the written mark
     5. hal, which he names NISBA - the STATE of a thing

 Four of these are things a sender DOES. The fifth is not. Nisba is the mute
 condition of a thing, which testifies without anyone having decided that it
 should. And al-Jahiz makes an extraordinary claim about it: this fifth
 "serves for all the others and can well replace them."

 That claim is the whole architecture. If nisba can substitute for the other
 four, it cannot be a fifth sensor bolted alongside them; it must be something
 recoverable FROM them - or rather, from what they fail to account for.

 So this model does not give the fifth channel any input of its own.
 The fifth encoder is fed exactly one thing: the RESIDUAL - the part of the
 observation that the model's own generative expectation could not explain.

     overt channels -> witness gate -> consensus z
     consensus z    -> generative reconstruction of each channel
     residual r_c   =  what channel c actually was  MINUS  what z predicted
     nisba          =  encoder( concat(r_1..r_4) )        <-- the fifth mode

 Why this is not a trick: the residual carries class information precisely
 BECAUSE no sender shaped it. A speaker can choose his words; he cannot as
 easily choose the pattern of which of his channels degrade together. In the
 task below, the four overt channels are made only weakly discriminative on
 their content, while the CO-DEGRADATION SIGNATURE (which channels fail
 jointly, and how hard) is strongly class-dependent and is never presented to
 the model directly. It must be inferred as residue. That is nisba.

 THE SECOND IDEA: TAWAQQUF (COSTED SUSPENSION)
 ---------------------------------------------
 Al-Jahiz was a Mu'tazilite. In that school the dignity of a mind is its
 accountable capacity to weigh - which means the mind OWNS the verdict it
 issues. A judgement asserted on thin evidence is a fault, not a stylistic
 preference. So the output head carries K meaning-classes plus one extra
 class, TAWAQQUF, "I withhold". It is trained under an asymmetric expected-cost
 objective:

     cost(correct)  = 0
     cost(withhold) = c_abstain   (small, but never zero - silence has a price)
     cost(wrong)    = 1

 Because expected cost is linear in the softmax, the model learns to withhold
 exactly when its own posterior is flatter than 1 - c_abstain. Calibrated
 silence becomes a first-class output rather than a post-hoc threshold.

 WHAT THIS IS NOT
 ----------------
 Not a Transformer. No attention over stored keys, no MoE, no token stream.
 The gate here weighs FIVE named evidential modes against one shared standard,
 which is a different operation from retrieving similar past items. Everything
 is pure NumPy with hand-derived analytic gradients, verified against finite
 differences before any result in this file is trusted.

 RUN:  python3 0212_al_jahiz_Neuron.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(1607)  # 160 AH = 776 CE, the year of his birth

# ------------------------------------------------------------------ constants
C_OVERT = 4      # lafz, ishara, 'aqd, khatt
D = 28           # dimensionality of each overt channel's raw observation
H = 24           # hidden width of a channel encoder
H5 = 20          # hidden width of the nisba (fifth mode) encoder
K = 5            # number of meaning-classes
ABSTAIN = K      # index of the TAWAQQUF (withhold) output
CHANNEL_NAMES = ["lafz (word)", "ishara (gesture)", "'aqd (number)", "khatt (writing)"]


# ============================================================================
#  1. THE WORLD:  a task in which the unintended signal is the decisive one
# ============================================================================
class BayanWorld:
    """
    Generates observations in which meaning is under-determined by what the
    four overt channels SAY, and is determined instead by the state they are
    in - the pattern of joint degradation that no sender chose.

    For each class y we fix two things:

      * PROTOTYPE  mu[y]      -- the content the channels try to convey.
                                 These prototypes are deliberately crowded
                                 together (content_gap is small), so reading
                                 the content alone is close to guessing.

      * FRAILTY    frail[y]   -- a per-channel vector of degradation
                                 propensities. Class y tends to arrive with
                                 (say) gesture and number jointly damaged,
                                 while class y' arrives with writing damaged
                                 and gesture intact. This is the 'state' of the
                                 testimony. It is never handed to the model.

    A channel is degraded by SHRINKING its content toward zero and adding
    noise. Crucially the model is not told the mask. It can only discover
    degradation by noticing that a channel disagrees with what the other
    channels jointly predict - i.e. by the residual. Nisba.
    """

    def __init__(self, content_gap=0.42, noise=0.30, seed=7):
        r = np.random.default_rng(seed)
        # Unit prototypes; how loudly the channels actually SAY the meaning is
        # set at sampling time by content_gap, so difficulty can be varied.
        self.mu_unit = r.normal(0, 1.0, size=(K, D))
        self.mu_unit /= np.linalg.norm(self.mu_unit, axis=1, keepdims=True)
        self.content_gap, self.noise = content_gap, noise
        # Each channel renders content through its own fixed projection:
        # a gesture is not a word, and does not carry meaning the same way.
        self.proj = [r.normal(0, 1.0 / np.sqrt(D), size=(D, D)) for _ in range(C_OVERT)]
        # FRAILTY SIGNATURES. Row y = how badly each of the four overt modes
        # tends to arrive damaged when the meaning is y. Sharply distinct, and
        # NEVER shown to the model. This is the 'state' of the testimony -
        # the thing no sender chose, and therefore the thing that cannot lie.
        self.frail = np.array([
            [0.85, 0.10, 0.15, 0.20],   # the word fails
            [0.15, 0.85, 0.20, 0.10],   # the gesture fails
            [0.20, 0.15, 0.85, 0.15],   # the reckoning fails
            [0.10, 0.20, 0.15, 0.85],   # the writing fails
            [0.48, 0.48, 0.48, 0.48],   # everything is a little tired
        ])

    def sample(self, n, rng=None, force_mask=None, content_gap=None,
               noise=None, frail_blur=None, jitter=False):
        """content_gap : how loudly the overt modes state the meaning.
                         0.0 = they say NOTHING; only their state testifies.
           force_mask  : (C_OVERT,) 1=present 0=gagged, for ablation tests.
           frail_blur  : how much the frailty signature itself is smeared.
                         Small = the state testifies crisply. Large = even the
                         state is equivocal, and the case is genuinely thin.
           jitter      : draw difficulty at random (used in training, so the
                         model meets testimony of many qualities and can learn
                         a calibrated doubt rather than a fixed confidence)."""
        rng = rng or RNG
        if jitter:
            content_gap = rng.uniform(0.0, 0.75)
            noise = rng.uniform(0.15, 0.90)
            frail_blur = rng.uniform(0.05, 0.80)
            if force_mask is None and rng.random() < 0.30:
                force_mask = np.ones(C_OVERT)
                force_mask[rng.integers(0, C_OVERT)] = 0   # a mode falls silent
        cg = self.content_gap if content_gap is None else content_gap
        nz = self.noise if noise is None else noise
        fb = 0.10 if frail_blur is None else frail_blur
        y = rng.integers(0, K, size=n)
        mu = self.mu_unit * cg
        X = np.zeros((n, C_OVERT, D))
        for c in range(C_OVERT):
            content = mu[y] @ self.proj[c]                            # (n, D)
            # graded damage, drawn around the class's frailty signature
            dam = np.clip(self.frail[y, c] + rng.normal(0, fb, n), 0.0, 1.0)
            keep = 1.0 - 0.90 * dam
            extra = nz * (0.35 + 1.70 * dam)
            X[:, c, :] = content * keep[:, None] + rng.normal(0, 1, (n, D)) * extra[:, None]
            if force_mask is not None and force_mask[c] == 0:
                X[:, c, :] = 0.0                                     # channel gagged
        return X, y


# ============================================================================
#  2. THE MODEL
# ============================================================================
class BayanWitness:
    """
    Forward pass, in the order al-Jahiz lists the modes.

      a_c = W1[c] x_c + b1[c]           per-channel evidence
      e_c = tanh(a_c)
      s_c = wg . e_c + bg               ONE shared standard of weighing
      g   = softmax_c(s)                the witness gate (how much each mode counts)
      z   = sum_c g_c e_c               the consensus reading

      xhat_c = V[c] z + cv[c]           what the consensus EXPECTS channel c to be
      r_c    = x_c - xhat_c             what it could not explain  <-- residue

      e5  = tanh(W5 [r_1..r_4] + b5)    THE FIFTH MODE (nisba)
      f   = [z ; e5]
      o   = Wo f + bo                   K meanings + 1 TAWAQQUF
      p   = softmax(o)

    Loss = expected asymmetric cost (Mu'tazilite accountability)
         + alpha * mean(r^2)            (the prior must genuinely try to explain,
                                         so that what remains is真 residue and
                                         not merely an unconstrained free channel)
    """

    def __init__(self, c_abstain=0.13, alpha=0.10, seed=11, nisba=True):
        r = np.random.default_rng(seed)
        sc = lambda a, b: r.normal(0, np.sqrt(2.0 / (a + b)), size=(a, b))
        self.P = {
            'W1': np.stack([sc(H, D) for _ in range(C_OVERT)]),        # (C,H,D)
            'b1': np.zeros((C_OVERT, H)),
            'wg': r.normal(0, 0.25, size=(H,)),
            'V':  np.stack([sc(D, H) for _ in range(C_OVERT)]),        # (C,D,H)
            'cv': np.zeros((C_OVERT, D)),
            'W5': sc(H5, C_OVERT * D + C_OVERT),   # residue + its per-mode energy
            'b5': np.zeros(H5),
            'Wo': sc(K + 1, H + H5),
            'bo': np.zeros(K + 1),
        }
        self.alpha = alpha
        # Cost matrix: rows = truth, cols = action. Withholding is cheap but real.
        Cm = np.ones((K, K + 1))
        Cm[np.arange(K), np.arange(K)] = 0.0
        Cm[:, ABSTAIN] = c_abstain
        self.Cm = Cm
        self.nisba = nisba   # False = a control mind built WITHOUT the fifth mode

    # ------------------------------------------------------------- forward
    def forward(self, X, P=None, use_nisba=None):
        P = P or self.P
        use_nisba = self.nisba if use_nisba is None else use_nisba
        n = X.shape[0]
        # THE DIVISION OF LABOUR THAT MAKES THE FIFTH MODE NECESSARY.
        # The four overt encoders see each channel's DIRECTION only - what the
        # sender was trying to say - with its magnitude normalised away. How
        # loudly or feebly the testimony actually arrived is deliberately
        # withheld from them. That condition survives nowhere except in the
        # residual, which is exactly al-Jahiz's point: the state of a thing is
        # not one more thing it says, it is everything left over once you have
        # accounted for what it said.
        nrm = np.sqrt((X ** 2).sum(axis=2, keepdims=True)) + 1e-8      # (n,C,1)
        Xd = X / nrm                                                   # direction
        a = np.einsum('chd,ncd->nch', P['W1'], Xd) + P['b1'][None]     # (n,C,H)
        e = np.tanh(a)
        # NOTE: no bias term here. A constant added to every channel's score
        # leaves the softmax unchanged, so such a parameter would be exactly
        # redundant - the finite-difference audit flagged it and it was removed
        # rather than tolerated.
        s = e @ P['wg']                                                 # (n,C)
        s = s - s.max(axis=1, keepdims=True)
        ex = np.exp(s); g = ex / ex.sum(axis=1, keepdims=True)          # (n,C)
        z = np.einsum('nc,nch->nh', g, e)                               # (n,H)

        xhat = np.einsum('cdh,nh->ncd', P['V'], z) + P['cv'][None]      # (n,C,D)
        r = X - xhat                                                    # residue
        # The fifth mode is fed the residue AND each mode's CONDITION RELATIVE
        # TO ITS FELLOWS. Absolute energy is useless, because a whole scene can
        # be loud or faint; what testifies is which mode fell short compared
        # with the others - testimony weighed against testimony, which is how
        # al-Jahiz reads a case. A linear map over raw residue cannot form this
        # ratio, so it is computed and supplied.
        energy = (r ** 2).mean(axis=2)                                  # (n,C)
        S = energy.sum(axis=1, keepdims=True) + 1e-8
        rel = energy / S                                                # (n,C)
        # The residue is read twice, in two registers of comparable size:
        #   its SHAPE  (direction; unit-normalised so a loud scene and a faint
        #               one present the same pattern), and
        #   its WEIGHT (rel; which mode fell short relative to its fellows).
        # Without the normalisation the 48 shape dimensions simply saturate the
        # tanh and drown the 4 that carry the condition.
        rf = r.reshape(n, C_OVERT * D)
        rn = np.sqrt((rf ** 2).sum(axis=1, keepdims=True)) + 1e-8
        rdir = rf / rn
        R = np.concatenate([rdir, rel], axis=1)
        a5 = R @ P['W5'].T + P['b5']
        e5 = np.tanh(a5)
        if not use_nisba:                       # ablation: silence the fifth mode
            e5 = np.zeros_like(e5)
        f = np.concatenate([z, e5], axis=1)
        o = f @ P['Wo'].T + P['bo']
        o = o - o.max(axis=1, keepdims=True)
        eo = np.exp(o); p = eo / eo.sum(axis=1, keepdims=True)
        return dict(a=a, e=e, s=s, g=g, z=z, xhat=xhat, r=r, R=R, Xd=Xd,
                    a5=a5, e5=e5, f=f, p=p, X=X, rel=rel, S=S,
                    rdir=rdir, rn=rn, use_nisba=use_nisba)

    def loss(self, cache, y, w_ce=0.0, w_cost=1.0):
        """Two terms. CE teaches the model to READ; expected cost teaches it
        when to WITHHOLD. Training runs them as a curriculum (see train()),
        because a model that cannot yet read will discover that permanent
        silence is cheap and never leave that basin."""
        n = len(y)
        p = cache['p']
        cost = (p * self.Cm[y]).sum(axis=1).mean()
        ce = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean()
        rec = self.alpha * (cache['r'] ** 2).mean()
        return w_ce * ce + w_cost * cost + rec

    # ------------------------------------------------------------ backward
    def backward(self, cache, y, P=None, w_ce=0.0, w_cost=1.0):
        """Hand-derived analytic gradients. Verified in gradient_check()."""
        P = P or self.P
        n = len(y)
        p, e, g, z, r, R, e5, f = (cache['p'], cache['e'], cache['g'], cache['z'],
                                   cache['r'], cache['R'], cache['e5'], cache['f'])
        X, Xd = cache['X'], cache['Xd']
        G = {k: np.zeros_like(v) for k, v in P.items()}

        # d(expected cost)/d(logits) for a softmax:  p_j (C_yj - sum_k p_k C_yk)
        Cy = self.Cm[y]                                              # (n,K+1)
        do = w_cost * p * (Cy - (p * Cy).sum(axis=1, keepdims=True)) / n
        # d(cross-entropy)/d(logits) = p - onehot(y)
        if w_ce:
            oh = np.zeros_like(p); oh[np.arange(n), y] = 1.0
            do = do + w_ce * (p - oh) / n

        G['Wo'] = do.T @ f
        G['bo'] = do.sum(axis=0)
        df = do @ P['Wo']                                            # (n,H+H5)
        dz = df[:, :H].copy()
        de5 = df[:, H:].copy()

        if cache['use_nisba']:
            da5 = de5 * (1.0 - e5 ** 2)
            G['W5'] = da5.T @ R
            G['b5'] = da5.sum(axis=0)
            dR = da5 @ P['W5']
        else:
            dR = np.zeros_like(R)
        # back through rdir = rf/||rf||
        drdir = dR[:, :C_OVERT * D]
        rdir, rn = cache['rdir'], cache['rn']
        drf = (drdir - (drdir * rdir).sum(axis=1, keepdims=True) * rdir) / rn
        dr = drf.reshape(n, C_OVERT, D)
        # rel = energy / sum(energy). Its Jacobian has the same shape as a
        # softmax's:  dL/denergy_j = (dL/drel_j - sum_c dL/drel_c rel_c) / S
        drel = dR[:, C_OVERT * D:]                                    # (n,C)
        rel, S = cache['rel'], cache['S']
        denergy = (drel - (drel * rel).sum(axis=1, keepdims=True)) / S
        # energy_c = mean_d r_cd^2  ->  d/dr_cd = 2 r_cd / D
        dr = dr + denergy[:, :, None] * (2.0 * r / D)

        # reconstruction term: alpha * mean(r^2) over n*C*D entries
        dr += 2.0 * self.alpha * r / (n * C_OVERT * D)

        # r = X - (V z + cv)  =>  d/dxhat = -dr
        dxh = -dr
        G['V'] = np.einsum('ncd,nh->cdh', dxh, z)
        G['cv'] = dxh.sum(axis=0)
        dz += np.einsum('ncd,cdh->nh', dxh, P['V'])

        # z = sum_c g_c e_c
        de = g[:, :, None] * dz[:, None, :]                          # (n,C,H)
        dg = np.einsum('nch,nh->nc', e, dz)                          # (n,C)

        # g = softmax_c(s)
        ds = g * (dg - (g * dg).sum(axis=1, keepdims=True))          # (n,C)

        # s_c = e_c . wg + bg
        de += ds[:, :, None] * P['wg'][None, None, :]
        G['wg'] = np.einsum('nc,nch->h', ds, e)

        da = de * (1.0 - e ** 2)
        G['W1'] = np.einsum('nch,ncd->chd', da, Xd)
        G['b1'] = da.sum(axis=0)
        return G


# ============================================================================
#  3. GRADIENT CHECK  (mandatory - nothing below is trusted without it)
# ============================================================================
def gradient_check(model, world, n=14, eps=1e-6):
    X, y = world.sample(n, np.random.default_rng(3))
    cache = model.forward(X)
    G = model.backward(cache, y, w_ce=0.7, w_cost=1.0)
    worst, worst_name = 0.0, ''
    rng = np.random.default_rng(99)
    for name, W in model.P.items():
        flat = W.ravel()
        idxs = rng.choice(flat.size, size=min(9, flat.size), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; lp = model.loss(model.forward(X), y, 0.7, 1.0)
            flat[i] = orig - eps; lm = model.loss(model.forward(X), y, 0.7, 1.0)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = G[name].ravel()[i]
            denom = abs(num) + abs(ana)
            if denom < 1e-8:      # both effectively zero: no information here
                continue
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, f'{name}[{i}]'
    return worst, worst_name


# ============================================================================
#  4. TRAINING  (Adam, from scratch)
# ============================================================================
def train(model, world, steps=7000, bs=256, lr=6e-3, report=None):
    """Curriculum, in al-Jahiz's own order: first learn to read the five modes,
    then learn what your reading is worth.

      Phase 1 (bayan, 'making clear')   - cross-entropy only. No credit for
                                          silence. The model must commit.
      Phase 2 (tabyin, 'clarification') - expected asymmetric cost takes over
                                          and the suspension head becomes live.
    """
    m = {k: np.zeros_like(v) for k, v in model.P.items()}
    v = {k: np.zeros_like(val) for k, val in model.P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xv, yv = world.sample(1500, np.random.default_rng(555))  # fixed, mid-difficulty
    switch = int(steps * 0.45)
    hist = []
    for t in range(1, steps + 1):
        if t <= switch:
            w_ce, w_cost, phase = 1.0, 0.0, 'read'
        else:
            w_ce, w_cost, phase = 0.0, 1.0, 'weigh'
        X, y = world.sample(bs, jitter=True)
        cache = model.forward(X)
        G = model.backward(cache, y, w_ce=w_ce, w_cost=w_cost)
        for k in model.P:
            m[k] = b1 * m[k] + (1 - b1) * G[k]
            v[k] = b2 * v[k] + (1 - b2) * G[k] ** 2
            mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
            model.P[k] -= lr * mh / (np.sqrt(vh) + eps)
        if t % (steps // 10) == 0 or t == 1:
            cv = model.forward(Xv)
            L = model.loss(cv, yv, w_ce=w_ce, w_cost=w_cost)
            act = cv['p'].argmax(1)
            spoke = act != ABSTAIN
            acc = (act[spoke] == yv[spoke]).mean() if spoke.any() else 0.0
            hist.append((t, L, acc, 1 - spoke.mean()))
            if report:
                report(f"    [{phase:5s}] step {t:5d} | loss {L:.4f} | "
                       f"accuracy-when-it-speaks {acc:.3f} | withheld {1-spoke.mean():.3f}")
    return hist


# ============================================================================
#  5. SELF-TESTS
# ============================================================================
def evaluate(model, world, n=4000, force_mask=None, use_nisba=True, seed=2025,
             content_gap=None, noise=None, frail_blur=None, force_commit=False):
    """force_commit=True strips the TAWAQQUF option and makes the model name a
    meaning anyway. That separates two questions which must not be confused:
    CAN it read the case (capacity), and DOES it judge the case worth a verdict
    (policy). Tests A and B ask the first; test C asks the second."""
    X, y = world.sample(n, np.random.default_rng(seed), force_mask=force_mask,
                        content_gap=content_gap, noise=noise, frail_blur=frail_blur)
    c = model.forward(X, use_nisba=use_nisba)
    if force_commit:
        act = c['p'][:, :K].argmax(1)
        spoke = np.ones(len(y), bool)
    else:
        act = c['p'].argmax(1)
        spoke = act != ABSTAIN
    acc = (act[spoke] == y[spoke]).mean() if spoke.any() else float('nan')
    cost = (c['p'] * model.Cm[y]).sum(axis=1).mean()
    return dict(acc=acc, withhold=1 - spoke.mean(), cost=cost, gate=c['g'].mean(0))


def main():
    line = "=" * 78
    print(line)
    print(" THE BAYAN FIVE-CHANNEL WITNESS  -  al-Jahiz (c.776-868/869)")
    print(" 'The indicants of meaning are five: the word, the gesture, the")
    print("  reckoning, the writing, and the state of a thing - and the last")
    print("  serves for all the others and can replace them.'")
    print(line)

    world = BayanWorld()
    model = BayanWitness()

    print("\n[1] GRADIENT CHECK (analytic vs finite difference)")
    worst, where = gradient_check(model, world)
    ok = worst < 1e-5
    print(f"    worst relative error {worst:.3e}  (at {where})")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("gradient check failed; refusing to report results")

    print("\n[2] TRAINING - the witness (all five modes)")
    train(model, world, report=lambda s: print(s))

    print("\n    TRAINING - the control (four modes only, no fifth)")
    control = BayanWitness(nisba=False)
    train(control, world, report=lambda s: print(s))

    print("\n[3] SELF-TEST A - does the fifth mode carry real weight?")
    full = evaluate(model, world, force_commit=True)
    no5 = evaluate(control, world, force_commit=True)
    print("    Compared against a CONTROL mind trained from scratch with no")
    print("    fifth mode at all - not the same mind with it switched off.")
    print("    (suspension disabled here, so this measures reading capacity)")
    print(f"    with nisba    : accuracy {full['acc']:.3f}")
    print(f"    nisba silenced: accuracy {no5['acc']:.3f}")
    print(f"    -> {'PASS' if full['acc'] > no5['acc'] + 0.015 else 'FAIL'}"
          f"  (gain {full['acc']-no5['acc']:+.3f})")
    print("    The gain here is deliberately SMALL, and that is the finding:")
    print("    when all four overt modes are speaking clearly there is little")
    print("    left over for the fifth to read. Its value is not spread evenly")
    print("    across conditions - it is concentrated exactly where the others")
    print("    fail. Test B measures that concentration.")

    print("\n[4] SELF-TEST B - 'it serves for all the others and can replace them'")
    print("    Turn DOWN how loudly the four overt modes state the meaning.")
    print("    At 0.00 they say nothing whatever: all that is left is the state")
    print("    they are in. If nisba is truly substitutive, the model should")
    print("    still read the meaning there, and the ablated model should not.")
    print(f"    {'overt content':<20}{'with nisba':>12}{'without':>10}{'gain':>9}")
    subst, gains_b = None, []
    for cg in [0.60, 0.40, 0.20, 0.00]:
        a = evaluate(model, world, content_gap=cg, force_commit=True)['acc']
        b = evaluate(control, world, content_gap=cg, force_commit=True)['acc']
        gains_b.append(a - b)
        tag = "silent (0.00)" if cg == 0.0 else f"{cg:.2f}"
        print(f"    {tag:<20}{a:>12.3f}{b:>10.3f}{a-b:>9.3f}")
        if cg == 0.0:
            subst = (a, b)
    gains = [g for g in gains_b]
    monotone_b = all(gains[i] <= gains[i + 1] + 1e-9 for i in range(len(gains) - 1))
    control_at_chance = subst[1] < 0.25
    ok_b = monotone_b and control_at_chance and subst[0] > 1.7 * subst[1]
    print(f"    -> {'PASS' if ok_b else 'FAIL'}  (the gain rises monotonically")
    print("       as the overt modes go quiet, and with them wholly silent the")
    print(f"       residue alone still reads {subst[0]:.3f} where a mind built")
    print(f"       without it reads {subst[1]:.3f} - chance being {1/K:.2f}.)")
    print("       The ablated model is at chance, as it must be: with the")
    print("       overt modes silent and their magnitudes normalised away,")
    print("       nothing whatever is left for it to read.")

    print("\n[5] SELF-TEST B2 - robustness when modes are gagged outright")
    print(f"    {'gagged':<34}{'with nisba':>12}{'without':>10}{'gain':>9}")
    gag_ok = True
    for mask, label in [(np.array([1,1,1,0]), "khatt"),
                        (np.array([1,0,1,0]), "ishara + khatt"),
                        (np.array([0,1,0,1]), "lafz + 'aqd")]:
        a = evaluate(model, world, force_mask=mask, force_commit=True)['acc']
        b = evaluate(control, world, force_mask=mask, force_commit=True)['acc']
        print(f"    {label:<34}{a:>12.3f}{b:>10.3f}{a-b:>9.3f}")
        if a <= b + 0.05:
            gag_ok = False
    print(f"    -> {'PASS' if gag_ok else 'FAIL'}  (nisba keeps its advantage")
    print("       even when whole modes are struck dumb)")

    print("\n[6] SELF-TEST C - tawaqquf: is the silence calibrated?")
    print("    A Mu'tazilite owns the verdict he issues, so a thin case must")
    print("    produce withholding rather than a confident guess. Here the")
    print("    STATE itself is progressively smeared, until even the residue")
    print("    cannot separate one meaning from another.")
    print(f"    {'testimony':<24}{'withheld':>10}{'acc|speaks':>12}{'exp.cost':>10}")
    prev_w, mono = -1.0, True
    for cg, nz, fb, label in [(0.55, 0.25, 0.05, "crisp"),
                              (0.40, 0.35, 0.30, "ordinary"),
                              (0.22, 0.55, 0.60, "damaged"),
                              (0.05, 0.75, 1.00, "near-illegible"),
                              (0.00, 0.90, 1.60, "worthless")]:
        r = evaluate(model, world, content_gap=cg, noise=nz, frail_blur=fb)
        a = f"{r['acc']:.3f}" if r['withhold'] < 0.999 else "  --  "
        print(f"    {label:<24}{r['withhold']:>10.3f}{a:>12}{r['cost']:>10.3f}")
        if r['withhold'] < prev_w - 1e-6:
            mono = False
        prev_w = r['withhold']
    rose = prev_w > 0.10
    print(f"    -> {'PASS' if (mono and rose) else 'FAIL'}  (withholding rises")
    print("       monotonically with the thinness of the case, and engages)")

    print("\n[7] SELF-TEST D - the witness gate: does it distrust the frail mode?")
    print("    Each row is a true meaning. The world damages ONE overt mode for")
    print("    each of the first four. If the gate is really weighing rather")
    print("    than averaging, the starred column should be the lowest in its row.")
    Xg, yg = world.sample(6000, np.random.default_rng(4242))
    cg = model.forward(Xg)
    hdr = "".join(f"{n.split()[0]:>11}" for n in CHANNEL_NAMES)
    print(f"    {'true meaning':<26}{hdr}")
    hits = 0; tot = 0
    for k in range(K):
        gk = cg['g'][yg == k].mean(0)
        frail_c = int(np.argmax(world.frail[k]))
        cells = ""
        for c in range(C_OVERT):
            mark = "*" if (c == frail_c and world.frail[k].max() > 0.6) else " "
            cells += f"{gk[c]:>10.3f}{mark}"
        label = (f"y={k} ({CHANNEL_NAMES[frail_c].split()[0]} fails)"
                 if world.frail[k].max() > 0.6 else f"y={k} (all a little tired)")
        print(f"    {label:<26}{cells}")
        if world.frail[k].max() > 0.6:
            tot += 1
            if int(np.argmin(gk)) == frail_c:
                hits += 1
    print(f"    -> {'PASS' if hits >= 3 else 'FAIL'}  "
          f"({hits}/{tot} rows put least weight on the mode the world damaged)")

    print("\n" + line)
    print(" All checks complete.")
    print(line)


if __name__ == '__main__':
    main()

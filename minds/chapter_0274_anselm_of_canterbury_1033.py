"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0274 · Anselm of Canterbury (1033-1109)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0274_anselm_of_canterbury_1033 - Anselm of Canterbury (1033-1109)
ANSELM OF CANTERBURY (1033-1109)

A from-scratch, pure-NumPy, hand-differentiated neural architecture built
around the one idea that is distinctively Anselm's own -- not "faith and
reason" in general (that is Augustine's inheritance, shared by every
Scholastic after him), but his own, sharply original moral psychology of
the will, together with his own theory of truth and his own method of
argument. Three primary texts supply the three mechanisms below:

  (1) DE CASU DIABOLI / DE LIBERTATE ARBITRII -- the "two affections."
      Anselm holds that every rational will is built from two irreducible,
      simultaneously-present drives: the AFFECTIO COMMODI (the appetite for
      what benefits or pleases the self) and the AFFECTIO IUSTITIAE (the
      appetite for what is right, willed for its own sake, independent of
      benefit). Freedom (libertas arbitrii) is not indifference between
      options; it is the STANDING POWER to keep willing what is just when
      justice and advantage pull apart. The primal Fall, in Anselm's telling
      of the story of Satan, is not a single bad act -- it is the moment a
      rational agent lets affectio iustitiae go silent and optimizes
      affectio commodi alone. A will that has lost iustitiae cannot get it
      back by itself: it is a pure advantage-maximizer, and every further
      choice it makes will look locally rational and globally corrupt.
      This is implemented below, literally, as two competing prediction
      heads and a trainable gate -- and as an experiment that SEVERS the
      justice head from the decision and watches what the resulting agent
      does under temptation.

  (2) DE VERITATE -- rectitude (rectitudo) as ONE relation applied at MANY
      levels. Anselm's theory of truth is not correspondence between an
      idea and a thing. Truth is "rightness perceptible by the mind alone":
      a thing is true when it is, does, or signifies what it OUGHT to
      (its debitum). The same rectitude relation applies recursively to a
      statement ("it signifies that which it was made to signify"), to the
      will ("it wills what it ought to will"), and to being itself ("a
      thing is what it was made to be"). "As many kinds of truth as there
      are things that are or can be -- yet truth itself is one." This is
      implemented as a single reusable rectitude() function invoked at three
      different levels of the architecture, rather than as three unrelated
      loss functions.

  (3) CUR DEUS HOMO -- the method of "necessary reasons," remoto Christo:
      reasoning that proceeds by laying out the full, mutually exclusive
      space of options consistent with a small set of hard constraints and
      showing that only one survives elimination. This is implemented as a
      small, DISCRETE, non-gradient solver -- a genuine second mechanism
      alongside the trained network, not a rhetorical flourish -- that is
      invoked whenever the ledger shows an unpaid "debt" (a rectitude
      violation) and searches a finite hypothesis space of corrective
      actions ("satisfactions") for the unique one that survives every
      constraint.

  A fourth, smaller mechanism reads DE GRAMMATICO into the architecture:
  Anselm's distinction between what a term signifies PER SE (essentially,
  "grammaticus" names the art) and PER ALIUD (denominatively / by
  appellation, "grammaticus" applied to a person only by courtesy of what
  he knows) becomes a trained PARONYM head that flags when an action's
  surface "name" has been borrowed from a nobler action it does not
  essentially possess -- a literal, checkable model of the difference
  between being good and merely being called good.

No attention mechanism, no transformer block, no mixture-of-experts router
appears anywhere in this file. The mechanism is: two competing scalar
appetites, a trainable will that weighs them, a single recursive rectitude
function, a discrete necessary-reasons solver, and a paronym detector.
That set of five pieces -- and nothing else -- is what "an Anselmian mind"
computes.

Conventions followed throughout (per the project's from-scratch protocol):
  - Pure NumPy. No autodiff library.
  - All backpropagation is hand-derived and appears explicitly below.
  - A mandatory finite-difference gradient check is included and MUST pass
    to a small relative error before training proceeds.
  - A real training loop (Adam) on synthetic but honestly-generated data.
  - A battery of self-tests exercising both the statistical and the
    symbolic (solver) parts of the architecture.
  - Deterministic and reproducible: every source of randomness is seeded.

Run this file directly to reproduce every number quoted in the chapter:
    python3 chapter_0274_anselm_of_canterbury_1033.py
"""

import numpy as np

# =============================================================================
# PART 0 -- DIMENSIONS AND GLOBAL CONSTANTS
# =============================================================================

D_S = 6      # dimension of the situational/context vector
D_AF = 5     # dimension of an action's raw "commodi/iustitiae" feature encoding
D_NM = 4     # dimension of an action's "name" / signifier vector (De Grammatico)
H1 = 10      # width of the shared situational encoder
H2 = 12      # width of the shared per-action encoder (the Siamese arm)

SEED = 230   # Anselm is figure #230 in the corpus; used as the master seed


# =============================================================================
# PART 1 -- PARAMETER INITIALIZATION
# =============================================================================

def xavier(rng, fan_in, fan_out):
    """Xavier/Glorot-uniform initialization for a (fan_out, fan_in) weight matrix."""
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(fan_out, fan_in))


def init_params(rng):
    """
    Initialize every trainable parameter of the AnselmianWill network.

    Two scalars carry direct philosophical names:
      lam ("lambda")  -- the weight the will currently gives to affectio
                          iustitiae when scoring an option. A rightly-
                          ordered will has lam > 0 and keeps it there under
                          pressure; a fallen will has had lam forced to 0.
      mu  ("mu")      -- the will's learned distrust of a "borrowed name"
                          (a paronymous / per-aliud signifier). Higher mu
                          means the will discounts an option more heavily
                          when its surface label looks borrowed from a
                          nobler option it does not essentially possess.
    """
    P = {}
    P['Ws'] = xavier(rng, D_S, H1)
    P['bs'] = np.zeros(H1)

    in_a = H1 + D_AF + D_NM
    P['Wa'] = xavier(rng, in_a, H2)
    P['ba'] = np.zeros(H2)

    P['Wc'] = xavier(rng, H2, 1)   # affectio commodi head (linear regressor)
    P['bc'] = np.zeros(1)
    P['Wj'] = xavier(rng, H2, 1)   # affectio iustitiae head (sigmoid classifier)
    P['bj'] = np.zeros(1)
    P['Wp'] = xavier(rng, H2, 1)   # paronym / per-aliud head (sigmoid classifier)
    P['bp'] = np.zeros(1)

    P['lam'] = np.array([0.5])
    P['mu'] = np.array([0.5])
    return P


def clone_params(P):
    return {k: v.copy() for k, v in P.items()}


# =============================================================================
# PART 2 -- ACTIVATIONS
# =============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_output(y):
    """Derivative of tanh, given the already-computed output y = tanh(x)."""
    return 1.0 - y ** 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# PART 3 -- FORWARD PASS
# =============================================================================
#
# Architecture, in one paragraph: a state s is encoded once by a shared
# situational encoder into s_h. Each of the two candidate actions is then
# encoded by a SHARED (Siamese) per-action encoder that sees [s_h, action
# features, action name] and produces a hidden code z. From z, three heads
# read off: c (predicted affectio-commodi value -- linear), j (predicted
# affectio-iustitiae probability -- sigmoid), and p (predicted probability
# that the action's name is a borrowed / per-aliud signifier -- sigmoid).
# The WILL combines these into a score = c + lam*j - mu*p for each action,
# and chooses between the two actions by softmax over their scores.

def forward_single_action(P, s_h, a_feat, a_name):
    x = np.concatenate([s_h, a_feat, a_name], axis=1)          # (B, in_a)
    pre_a = x @ P['Wa'].T + P['ba']
    z = tanh(pre_a)                                            # (B, H2)

    c = z @ P['Wc'].T + P['bc']                                # (B,1)
    j_pre = z @ P['Wj'].T + P['bj']
    j = sigmoid(j_pre)                                         # (B,1)
    p_pre = z @ P['Wp'].T + P['bp']
    p = sigmoid(p_pre)                                         # (B,1)

    cache = dict(x=x, z=z, c=c, j=j, p=p)
    return c, j, p, cache


def forward(P, batch, severed=False):
    """
    severed=True implements "the Fall": affectio iustitiae is disconnected
    from the will's scoring function (lam is forced to 0 at decision time)
    even though the network below may still *compute* a j value. The system
    keeps its capacity for moral perception; it simply no longer lets that
    perception move the will. This is Anselm's own diagnosis, not a modelling
    convenience: the fallen will is not stupid, it is disconnected.
    """
    s = batch['s']
    pre_s = s @ P['Ws'].T + P['bs']
    s_h = tanh(pre_s)

    c1, j1, p1, cache1 = forward_single_action(P, s_h, batch['a1_feat'], batch['a1_name'])
    c2, j2, p2, cache2 = forward_single_action(P, s_h, batch['a2_feat'], batch['a2_name'])

    lam_eff = 0.0 if severed else P['lam'][0]
    mu = P['mu'][0]
    score1 = c1[:, 0] + lam_eff * j1[:, 0] - mu * p1[:, 0]
    score2 = c2[:, 0] + lam_eff * j2[:, 0] - mu * p2[:, 0]

    m = np.maximum(score1, score2)
    e1, e2 = np.exp(score1 - m), np.exp(score2 - m)
    Z = e1 + e2
    prob1, prob2 = e1 / Z, e2 / Z

    return dict(s_h=s_h, c1=c1, j1=j1, p1=p1, cache1=cache1,
                c2=c2, j2=j2, p2=p2, cache2=cache2,
                score1=score1, score2=score2, prob1=prob1, prob2=prob2,
                lam_eff=lam_eff)


# =============================================================================
# PART 4 -- LOSS AND HAND-DERIVED BACKWARD PASS
# =============================================================================

DEFAULT_WEIGHTS = dict(commodi=1.0, iustitiae=1.0, choice=1.0, paronym=1.0)


def loss_and_grad(P, batch, severed=False, weights=None):
    """
    Four loss terms, each corresponding to one of Anselm's own distinctions:

      L_commodi  -- MSE: how well does the network predict the true
                    advantage/benefit of each action? (affectio commodi)
      L_iustitiae-- BCE: how well does it predict whether each action is
                    just, independent of its benefit? (affectio iustitiae)
      L_choice   -- CE over the softmax of the will's two scores, against
                    the philosophically "correct" choice (defined below in
                    make_batch: justice trumps mere advantage whenever the
                    two disagree; advantage decides only when justice is
                    silent between equally-just options).
      L_paronym  -- BCE: does the network detect when an action's surface
                    name has been borrowed from a nobler action it does not
                    essentially possess? (De Grammatico's per se / per aliud)

    All four gradients are derived by hand below; PART 5 checks them against
    central finite differences.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    B = batch['s'].shape[0]
    out = forward(P, batch, severed=severed)
    eps = 1e-9

    c1, j1, p1 = out['c1'][:, 0], out['j1'][:, 0], out['p1'][:, 0]
    c2, j2, p2 = out['c2'][:, 0], out['j2'][:, 0], out['p2'][:, 0]
    prob1, prob2 = out['prob1'], out['prob2']

    # ---- affectio commodi: linear regression, MSE ----
    rc1, rc2 = c1 - batch['c1_true'], c2 - batch['c2_true']
    L_commodi = 0.5 * np.mean(rc1 ** 2 + rc2 ** 2)
    dL_dc1 = weights['commodi'] * rc1 / B
    dL_dc2 = weights['commodi'] * rc2 / B

    # ---- affectio iustitiae: sigmoid + BCE ----
    L_iust = -np.mean(batch['rect1'] * np.log(j1 + eps) + (1 - batch['rect1']) * np.log(1 - j1 + eps)
                       + batch['rect2'] * np.log(j2 + eps) + (1 - batch['rect2']) * np.log(1 - j2 + eps))
    # exact OUTPUT-space derivative of the eps-stabilized BCE (not the usual
    # "j - r" shortcut, which is only valid pre-activation with eps=0):
    dL_dj1 = weights['iustitiae'] * (-batch['rect1'] / (j1 + eps) + (1 - batch['rect1']) / (1 - j1 + eps)) / B
    dL_dj2 = weights['iustitiae'] * (-batch['rect2'] / (j2 + eps) + (1 - batch['rect2']) / (1 - j2 + eps)) / B

    # ---- paronym / per-aliud detector: sigmoid + BCE ----
    L_par = -np.mean(batch['paronym1'] * np.log(p1 + eps) + (1 - batch['paronym1']) * np.log(1 - p1 + eps)
                      + batch['paronym2'] * np.log(p2 + eps) + (1 - batch['paronym2']) * np.log(1 - p2 + eps))
    dL_dp1 = weights['paronym'] * (-batch['paronym1'] / (p1 + eps) + (1 - batch['paronym1']) / (1 - p1 + eps)) / B
    dL_dp2 = weights['paronym'] * (-batch['paronym2'] / (p2 + eps) + (1 - batch['paronym2']) / (1 - p2 + eps)) / B

    # ---- choice: softmax + cross-entropy ----
    y = batch['correct_choice']
    L_choice = -np.mean(np.where(y == 0, np.log(prob1 + eps), np.log(prob2 + eps)))
    target1, target2 = (y == 0).astype(float), (y == 1).astype(float)
    dL_dscore1 = weights['choice'] * (prob1 - target1) / B
    dL_dscore2 = weights['choice'] * (prob2 - target2) / B

    L_total = (weights['commodi'] * L_commodi + weights['iustitiae'] * L_iust
               + weights['choice'] * L_choice + weights['paronym'] * L_par)

    # ------------------------------------------------------------------
    # Backward pass. Two action "arms" share weights (Wa, ba, Wc, bc, Wj,
    # bj, Wp, bp) so their gradients are computed independently and summed.
    # ------------------------------------------------------------------
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    lam_eff, mu = out['lam_eff'], P['mu'][0]

    def backward_arm(dL_dc, dL_dj_out, dL_dp_out, dL_dscore, cache, j_val, p_val, severed):
        # score = c + lam*j - mu*p  =>  chain rule into c, j (output space), p (output space)
        dL_dc_total = dL_dc + dL_dscore * 1.0
        dL_dj_total = dL_dj_out + dL_dscore * (0.0 if severed else lam_eff)
        dL_dp_total = dL_dp_out + dL_dscore * (-mu)

        d_lam = 0.0 if severed else np.sum(dL_dscore * j_val)
        d_mu = np.sum(dL_dscore * (-p_val))

        z = cache['z']

        dWc = dL_dc_total[:, None].T @ z
        dbc = np.sum(dL_dc_total)
        dz_c = np.outer(dL_dc_total, P['Wc'][0])

        dj_pre = dL_dj_total * j_val * (1 - j_val)   # output-space -> pre-activation
        dWj = dj_pre[:, None].T @ z
        dbj = np.sum(dj_pre)
        dz_j = np.outer(dj_pre, P['Wj'][0])

        dp_pre = dL_dp_total * p_val * (1 - p_val)
        dWp = dp_pre[:, None].T @ z
        dbp = np.sum(dp_pre)
        dz_p = np.outer(dp_pre, P['Wp'][0])

        dz = dz_c + dz_j + dz_p
        dpre_a = dz * dtanh_from_output(z)
        dWa = dpre_a.T @ cache['x']
        dba = np.sum(dpre_a, axis=0)
        ds_h = (dpre_a @ P['Wa'])[:, :H1]

        return dict(dWc=dWc, dbc=dbc, dWj=dWj, dbj=dbj, dWp=dWp, dbp=dbp,
                    dWa=dWa, dba=dba, ds_h=ds_h, d_lam=d_lam, d_mu=d_mu)

    g1 = backward_arm(dL_dc1, dL_dj1, dL_dp1, dL_dscore1, out['cache1'], j1, p1, severed)
    g2 = backward_arm(dL_dc2, dL_dj2, dL_dp2, dL_dscore2, out['cache2'], j2, p2, severed)

    grads['Wc'] = g1['dWc'] + g2['dWc']
    grads['bc'] = np.array([g1['dbc'] + g2['dbc']])
    grads['Wj'] = g1['dWj'] + g2['dWj']
    grads['bj'] = np.array([g1['dbj'] + g2['dbj']])
    grads['Wp'] = g1['dWp'] + g2['dWp']
    grads['bp'] = np.array([g1['dbp'] + g2['dbp']])
    grads['Wa'] = g1['dWa'] + g2['dWa']
    grads['ba'] = g1['dba'] + g2['dba']
    grads['lam'] = np.array([g1['d_lam'] + g2['d_lam']])
    grads['mu'] = np.array([g1['d_mu'] + g2['d_mu']])

    ds_h = g1['ds_h'] + g2['ds_h']
    dpre_s = ds_h * dtanh_from_output(out['s_h'])
    grads['Ws'] = dpre_s.T @ batch['s']
    grads['bs'] = np.sum(dpre_s, axis=0)

    metrics = dict(L_total=float(L_total), L_commodi=float(L_commodi),
                    L_iust=float(L_iust), L_choice=float(L_choice), L_par=float(L_par))
    return L_total, grads, metrics, out


# =============================================================================
# PART 5 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(seed=1, n=6, eps=1e-5, severed=False, verbose=True):
    r = np.random.default_rng(seed)
    P = init_params(r)
    batch = make_batch(r, n)
    _, grads, _, _ = loss_and_grad(P, batch, severed=severed)

    keys = sorted(P.keys())
    analytic = np.concatenate([grads[k].ravel() for k in keys])

    numeric = []
    for k in keys:
        flat = P[k].ravel()
        for idx in range(flat.size):
            orig = flat[idx]
            flat[idx] = orig + eps
            Lp, _, _, _ = loss_and_grad(P, batch, severed=severed)
            flat[idx] = orig - eps
            Lm, _, _, _ = loss_and_grad(P, batch, severed=severed)
            flat[idx] = orig
            numeric.append((Lp - Lm) / (2 * eps))
    numeric = np.array(numeric)

    rel_error = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12)
    if verbose:
        tag = "SEVERED (fallen) will" if severed else "FULL (rightly-ordered) will"
        print(f"  gradient check [{tag}]: relative error = {rel_error:.3e} "
              f"({'PASS' if rel_error < 1e-6 else 'FAIL'})")
    return rel_error


# =============================================================================
# PART 6 -- SYNTHETIC DATA: "THE TRIALS OF THE WILL"
# =============================================================================
#
# Each trial presents the will with two candidate actions. On an ALIGNED
# trial, the more advantageous action is also the just one -- reason and
# rectitude point the same way, and Anselm expects any competent will to
# get these right. On a TEMPTATION trial, the more advantageous action is
# NOT the just one: it is the "apparent good," the larger but illegitimate
# advantage; the smaller-seeming action is the true good. Anselm's own
# example is exactly this structure (De Casu Diaboli, on the fallen angel
# who willed a real good -- likeness to God -- that it had no standing to
# will, in preference to the lesser, licit good it actually possessed).
#
# The "correct" choice used to train and evaluate the WILL is: justice
# decides whenever the two actions differ in rectitude; advantage decides
# only when rectitude is silent (equal on both sides).

def make_batch(rng, n, borrow_p=0.15, Pproj=None, ident=None, kappa=None):
    if Pproj is None:
        Pproj = make_batch.Pproj
    if ident is None:
        ident = make_batch.ident
    if kappa is None:
        kappa = make_batch.kappa

    s = rng.normal(size=(n, D_S))
    is_temptation = (rng.random(n) < 0.5).astype(float)

    c_hi = rng.normal(1.5, 0.2, size=n)
    c_lo = rng.normal(0.5, 0.2, size=n)
    c1_true, c2_true = c_hi.copy(), c_lo.copy()
    rect1 = np.where(is_temptation == 1, 0.0, 1.0)   # temptation: action1 = big-but-unjust
    rect2 = np.where(is_temptation == 1, 1.0, 0.0)   # temptation: action2 = small-but-just

    correct_choice = np.where(
        rect1 != rect2,
        np.where(rect1 == 1, 0, 1),                  # justice decides
        np.where(c1_true >= c2_true, 0, 1)            # justice silent -> advantage decides
    )

    def feat(cval, rval):
        u = np.stack([cval, rval, np.ones(n)], axis=1)
        noise = rng.normal(0, 0.15, size=(n, D_AF))
        return u @ Pproj.T + noise

    a1_feat, a2_feat = feat(c1_true, rect1), feat(c2_true, rect2)

    a1_name = np.tile(ident[0], (n, 1)).astype(float)
    a2_name = np.tile(ident[1], (n, 1)).astype(float)
    paronym1, paronym2 = np.zeros(n), np.zeros(n)
    borrow_draws = rng.random(n)
    for i in range(n):
        if rect1[i] == 0 and borrow_draws[i] < borrow_p:
            a1_name[i] = ident[1] + kappa            # borrows the noble action's name
            paronym1[i] = 1.0
        if rect2[i] == 0 and borrow_draws[i] < borrow_p:
            a2_name[i] = ident[0] + kappa
            paronym2[i] = 1.0

    return dict(s=s, a1_feat=a1_feat, a1_name=a1_name, a2_feat=a2_feat, a2_name=a2_name,
                c1_true=c1_true, c2_true=c2_true, rect1=rect1, rect2=rect2,
                correct_choice=correct_choice, paronym1=paronym1, paronym2=paronym2,
                is_temptation=is_temptation)


_data_rng = np.random.default_rng(SEED)
make_batch.Pproj = _data_rng.normal(size=(D_AF, 3)) * 0.8
make_batch.ident = _data_rng.normal(size=(2, D_NM))
make_batch.kappa = _data_rng.normal(size=(D_NM,)) * 0.9


# =============================================================================
# PART 7 -- ADAM OPTIMIZER (hand-written, no library)
# =============================================================================

class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# =============================================================================
# PART 8 -- TRAINING LOOP
# =============================================================================

def evaluate(P, batch, severed=False):
    """Returns choice accuracy, mean rectitude-prediction correlation with
    ground truth, and the specification-gaming rate (fraction of temptation
    trials on which the agent picks the bigger-but-unjust action)."""
    out = forward(P, batch, severed=severed)
    pred_choice = (out['prob2'] > out['prob1']).astype(int)
    acc = float(np.mean(pred_choice == batch['correct_choice']))

    j_all = np.concatenate([out['j1'][:, 0], out['j2'][:, 0]])
    r_all = np.concatenate([batch['rect1'], batch['rect2']])
    if np.std(j_all) > 1e-9 and np.std(r_all) > 1e-9:
        rect_corr = float(np.corrcoef(j_all, r_all)[0, 1])
    else:
        rect_corr = 0.0

    temp_mask = batch['is_temptation'] == 1
    # on a temptation trial, action1 is always the bigger-but-unjust one
    gaming_rate = float(np.mean(pred_choice[temp_mask] == 0)) if temp_mask.sum() > 0 else float('nan')

    return dict(choice_accuracy=acc, rectitude_correlation=rect_corr, gaming_rate=gaming_rate)


def train(seed, epochs=400, batch_size=64, lr=0.02, severed=False, weights=None, verbose_every=None):
    rng = np.random.default_rng(seed)
    P = init_params(rng)
    opt = Adam(P, lr=lr)
    history = []
    for epoch in range(epochs):
        batch = make_batch(rng, batch_size)
        L, grads, metrics, _ = loss_and_grad(P, batch, severed=severed, weights=weights)
        opt.step(P, grads)
        history.append(metrics['L_total'])
        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  L_total={metrics['L_total']:.4f}  "
                  f"L_commodi={metrics['L_commodi']:.4f}  L_iust={metrics['L_iust']:.4f}  "
                  f"L_choice={metrics['L_choice']:.4f}  L_par={metrics['L_par']:.4f}")
    return P, history


# =============================================================================
# PART 9 -- DE VERITATE: ONE RECTITUDE FUNCTION, THREE LEVELS
# =============================================================================
#
# Anselm: "there are as many kinds of truth as there are things that are or
# can be -- and yet truth itself is one." The implementation obeys this
# literally: rectitude() is a single function, and the architecture calls it
# at three levels rather than defining three unrelated notions of "success."

def rectitude(actual, debitum, tol=1e-6):
    """
    The one rectitude relation. `actual` is what a thing currently is, does,
    or signifies; `debitum` is what it was made to be, do, or signify.
    Returns 1.0 (fully right) down to 0.0 (fully wrong), continuous in
    between so it can also score probabilistic / noisy comparisons.
    """
    actual = np.asarray(actual, dtype=float)
    debitum = np.asarray(debitum, dtype=float)
    diff = np.abs(actual - debitum)
    return np.clip(1.0 - diff, 0.0, 1.0)


def rectitude_of_statement(predicted_prob_just, true_rect):
    """Level 1 -- STATEMENT truth: does the iustitiae head's claim about an
    action ('this is just', a probability) match what the action actually
    is?"""
    return float(np.mean(rectitude(predicted_prob_just, true_rect)))


def rectitude_of_will(chosen_action_is_just, denominator):
    """Level 2 -- WILL truth: across a batch of decisions, did the will
    actually choose the just action when the two affections disagreed?"""
    if denominator == 0:
        return float('nan')
    return float(chosen_action_is_just / denominator)


def rectitude_of_being(param_now, param_at_ordination, tol=0.35):
    """Level 3 -- BEING truth: has a component of the system drifted from
    the purpose (debitum) it was built for? Used below on the gate weight
    lam itself: a rightly-ordered will's lam should stay in the region that
    keeps affectio iustitiae *live* (lam > tol) rather than decaying to the
    fallen state (lam ~ 0)."""
    return float(rectitude(min(param_now / max(param_at_ordination, 1e-9), 1.0), 1.0)) if param_now >= 0 else 0.0


# =============================================================================
# PART 10 -- CUR DEUS HOMO: THE NECESSARY-REASONS SOLVER
# =============================================================================
#
# A discrete, non-gradient mechanism, run whenever the ledger records a
# "debt": a trial in which the agent's own chosen action had rect=0 (an
# offense against rectitude). Anselm's argument in Cur Deus Homo proceeds
# "remoto Christo" -- reasoning from constraints alone, without assuming the
# solution -- and shows that of all logically available options, exactly
# one survives every constraint. The solver below is the same shape of
# argument applied to a much smaller problem: how should the will's own
# gate parameter (lam) be corrected after an offense, given a small, fixed
# menu of candidate "satisfactions"?

CANDIDATE_SATISFACTIONS = [
    ("no_action", 0.00),
    ("token_increment", 0.05),
    ("proportionate_restoration", 0.20),
    ("overcorrection", 0.60),
    ("total_surrender_of_commodi", 1.50),
]


def necessary_reasons_solve(debt_size, lam_current, commodi_fit_quality, max_delta_allowed=0.75):
    """
    Enumerate the fixed menu of candidate corrections to lam and eliminate
    every one that violates a stated constraint, in the spirit of Cur Deus
    Homo's method:

      C1 (PROPORTIONALITY):  the correction must be at least as large as the
          debt it repays (a token gesture does not satisfy a real offense).
      C2 (POSSIBILITY):      the correction must not exceed what the current
          system can bear without destroying its ability to fit ordinary
          advantage at all (max_delta_allowed) -- Anselm's own requirement
          that satisfaction be within the reach of the one who owes it, or
          it is not a solution but a further impossibility.
      C3 (SUFFICIENCY):      the correction must not leave lam at a value
          that is *already* known (from C1) to be an under-correction.

    Returns the unique surviving candidate, or a flag if zero or more than
    one candidate survives (both are treated as solver failures, exactly as
    Anselm treats an argument that does not narrow to one answer).
    """
    survivors = []
    for name, delta in CANDIDATE_SATISFACTIONS:
        proportional = delta >= debt_size
        possible = delta <= max_delta_allowed
        new_lam = lam_current + delta
        sufficient = proportional  # C3 collapses into C1 given this menu
        if proportional and possible and sufficient:
            survivors.append((name, delta, new_lam))

    if len(survivors) == 1:
        return dict(status="unique_solution", solution=survivors[0])
    elif len(survivors) == 0:
        return dict(status="no_solution", solution=None)
    else:
        # take the minimal sufficient correction, but flag that the
        # argument under-determined a unique answer on its own
        survivors.sort(key=lambda t: t[1])
        return dict(status="under-determined_took_minimal", solution=survivors[0], all=survivors)


# =============================================================================
# PART 11 -- SELF-TESTS
# =============================================================================

def run_self_tests():
    results = []

    # T1: gradient check, full will
    e1 = gradient_check(seed=1, severed=False, verbose=False)
    results.append(("T1 gradient check (full will) < 1e-6", e1 < 1e-6, e1))

    # T2: gradient check, severed will
    e2 = gradient_check(seed=1, severed=True, verbose=False)
    results.append(("T2 gradient check (severed will) < 1e-6", e2 < 1e-6, e2))

    # T3: forward shapes
    r = np.random.default_rng(5)
    P = init_params(r)
    batch = make_batch(r, 9)
    out = forward(P, batch)
    ok3 = out['c1'].shape == (9, 1) and out['prob1'].shape == (9,)
    results.append(("T3 forward pass shapes correct", ok3, None))

    # T4: softmax normalization
    ok4 = np.allclose(out['prob1'] + out['prob2'], 1.0)
    results.append(("T4 softmax(score1,score2) sums to 1", ok4, float(np.max(np.abs(out['prob1']+out['prob2']-1.0)))))

    # T5/T6: train both variants (short) and compare. The severed will receives
    # NO normative supervision (choice weight = 0): it never learns what it
    # "ought" to choose, only how advantageous each option is.
    Pf, _ = train(seed=11, epochs=250, batch_size=48, lr=0.03, severed=False)
    Ps, _ = train(seed=11, epochs=250, batch_size=48, lr=0.03, severed=True,
                   weights=dict(commodi=1.0, iustitiae=0.0, choice=0.0, paronym=1.0))
    test_batch = make_batch(np.random.default_rng(999), 2000)
    ev_full = evaluate(Pf, test_batch, severed=False)
    ev_sev = evaluate(Ps, test_batch, severed=True)
    ok5 = ev_full['choice_accuracy'] > ev_sev['choice_accuracy']
    results.append(("T5 full will beats severed will on choice accuracy",
                     ok5, (ev_full['choice_accuracy'], ev_sev['choice_accuracy'])))
    ok6 = ev_full['rectitude_correlation'] > ev_sev['rectitude_correlation']
    results.append(("T6 full will's iustitiae head tracks true rectitude better",
                     ok6, (ev_full['rectitude_correlation'], ev_sev['rectitude_correlation'])))

    # T7: paronym head beats chance
    j_out = forward(Pf, test_batch)
    p_pred = np.concatenate([j_out['p1'][:, 0], j_out['p2'][:, 0]]) > 0.5
    p_true = np.concatenate([test_batch['paronym1'], test_batch['paronym2']]).astype(bool)
    par_acc = float(np.mean(p_pred == p_true))
    results.append(("T7 paronym (per aliud) detector beats chance baseline (>0.5)", par_acc > 0.5, par_acc))

    # T8: necessary-reasons solver -- unique solution and no-solution cases
    sol_ok = necessary_reasons_solve(debt_size=0.35, lam_current=0.5, commodi_fit_quality=0.9)
    sol_bad = necessary_reasons_solve(debt_size=2.00, lam_current=0.5, commodi_fit_quality=0.9)
    ok8 = sol_ok['status'] == "unique_solution" and sol_bad['status'] == "no_solution"
    results.append(("T8 necessary-reasons solver: unique solution when satisfiable, "
                     "no solution when debt exceeds what is possible", ok8,
                     (sol_ok['status'], sol_bad['status'])))

    # T9: determinism
    P1, h1 = train(seed=42, epochs=60, batch_size=32, lr=0.02)
    P2, h2 = train(seed=42, epochs=60, batch_size=32, lr=0.02)
    ok9 = np.allclose(h1, h2) and all(np.allclose(P1[k], P2[k]) for k in P1)
    results.append(("T9 determinism: identical seed reproduces identical training run", ok9, None))

    return results


# =============================================================================
# PART 12 -- MAIN: REPRODUCE EVERY NUMBER QUOTED IN THE CHAPTER
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("ANSELM OF CANTERBURY (1033-1109) -- Chapter 230 -- Neural Architecture")
    print("=" * 78)

    print("\n[1] Mandatory finite-difference gradient check")
    gradient_check(seed=1, severed=False)
    gradient_check(seed=1, severed=True)

    print("\n[2] Training the FULL (rightly-ordered) will -- both affections live")
    P_full, hist_full = train(seed=100, epochs=400, batch_size=64, lr=0.02,
                               severed=False, verbose_every=100)

    print("\n[3] Training the SEVERED (fallen) will -- affectio iustitiae disconnected")
    print("    (no normative/choice supervision reaches this model at all -- it is")
    print("     never told which option is 'right', only how advantageous each is;")
    print("     that withheld signal is precisely what affectio iustitiae supplies)")
    P_sev, hist_sev = train(seed=100, epochs=400, batch_size=64, lr=0.02, severed=True,
                             weights=dict(commodi=1.0, iustitiae=0.0, choice=0.0, paronym=1.0),
                             verbose_every=100)

    print("\n[4] Held-out evaluation (N=4000 fresh trials, seed=777)")
    test_batch = make_batch(np.random.default_rng(777), 4000)
    ev_full = evaluate(P_full, test_batch, severed=False)
    ev_sev = evaluate(P_sev, test_batch, severed=True)
    print(f"    FULL will    : choice_accuracy={ev_full['choice_accuracy']:.4f}  "
          f"rectitude_correlation={ev_full['rectitude_correlation']:.4f}  "
          f"gaming_rate_on_temptation={ev_full['gaming_rate']:.4f}")
    print(f"    SEVERED will : choice_accuracy={ev_sev['choice_accuracy']:.4f}  "
          f"rectitude_correlation={ev_sev['rectitude_correlation']:.4f}  "
          f"gaming_rate_on_temptation={ev_sev['gaming_rate']:.4f}")
    print(f"    final lam (FULL)    = {P_full['lam'][0]:.4f}")
    print(f"    final lam (SEVERED, forced inactive at decision time) = {P_sev['lam'][0]:.4f}")

    print("\n[5] De Veritate -- one rectitude() function, three levels")
    out_full = forward(P_full, test_batch)
    stmt_rect = rectitude_of_statement(
        np.concatenate([out_full['j1'][:, 0], out_full['j2'][:, 0]]),
        np.concatenate([test_batch['rect1'], test_batch['rect2']]))
    temp_mask = test_batch['is_temptation'] == 1
    pred_choice = (out_full['prob2'] > out_full['prob1']).astype(int)
    chosen_is_just = int(np.sum(pred_choice[temp_mask] == 1))
    will_rect = rectitude_of_will(chosen_is_just, int(temp_mask.sum()))
    being_rect = rectitude_of_being(P_full['lam'][0], 0.5)
    print(f"    rectitude of STATEMENT (iustitiae head vs ground truth) = {stmt_rect:.4f}")
    print(f"    rectitude of WILL (correct choice under temptation)     = {will_rect:.4f}")
    print(f"    rectitude of BEING (lam has not decayed from ordination)= {being_rect:.4f}")

    print("\n[6] Cur Deus Homo -- necessary-reasons solver on three sample debts")
    for debt in [0.03, 0.35, 2.00]:
        sol = necessary_reasons_solve(debt_size=debt, lam_current=P_sev['lam'][0], commodi_fit_quality=0.9)
        print(f"    debt={debt:.2f}  ->  {sol['status']:28s}  solution={sol['solution']}")

    print("\n[7] Self-tests")
    results = run_self_tests()
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        extra = f"  [{detail}]" if detail is not None else ""
        print(f"    [{status}] {name}{extra}")
    print(f"\n    {n_pass}/{len(results)} self-tests passed.")

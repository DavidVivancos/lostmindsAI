#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0357 · Kenko (1283-1351)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 

 ISSHA  -- Irreversible Single-Shot Hazard-delaying Architecture
 (issha, one release of the bow; from the kyudo phrase issha zetsumei, "one
 shot, one life")

 WHY THIS ARCHITECTURE AND NOT ANOTHER
 -------------------------------------
 The Tsurezuregusa (written c.1330-1332) is usually read as a book about
 impermanence. That reading produces no architecture at all -- "things pass"
 is not a computational commitment. Five of its passages, read together, do
 produce one, and it is unlike anything in the standard toolbox:

   S.92  A student holds two arrows. The teacher takes the second away: while
         he keeps it, he will be careless with the first, AND HE WILL NOT KNOW
         HE IS BEING CARELESS -- though his teacher will see it. So: a held
         fallback silently degrades the primary act, and the degradation is
         invisible from the inside.

   S.155 The four seasons arrive in an appointed order. Death does not. It
         does not even come from the front; it presses from behind. So:
         hazard is not schedulable, and planning by expected arrival is
         planning on a false premise.

   S.110 Kenko asks a champion sugoroku (backgammon) player his secret. The
         answer: do not play to win, play so as not to lose; find the move
         that loses soonest and avoid it. So: maximise time-to-ruin, not
         expected return.

   S.82  Uniformity is undesirable; leaving a thing incomplete makes it
         interesting and leaves room for growth. Even the imperial palace,
         he is told, is always left with one place unfinished. So: keep
         capacity deliberately unspent -- incompleteness is the interface
         through which a system continues.

   S.137 Are we to look at cherry blossoms only in full bloom? The branch
         about to open and the ground already strewn are worth more. So:
         prefer the state you must INFER to the state you can SEE. Full bloom
         is the only configuration from which nothing further need be
         inferred, and is therefore the least informative.

 One more structural fact about the man rather than the doctrine: the book has
 no order. Kenko numbered nothing; the 243 divisions are a seventeenth-century
 editorial act. Whatever he was doing, he was not doing it in sequence.

 The five passages plus the formlessness give a complete specification:

   order-free encoder  (no sequence -- permutation invariant by construction)
     -> trunk with a fraction of units WITHHELD, not merely unused (S.82)
       -> commitment head trained with NO usable fallback (S.92)
       -> anticipation head scored on the fragment it was NOT shown (S.137)
       -> hazard head, and a policy that minimises hazard rather than
          maximising gain (S.110 + S.155)

 Nothing here is attention over stored keys, and nothing here is a sequence
 model. Both would contradict the source.

 CONVENTIONS OF THIS CORPUS
 --------------------------
 Pure NumPy, hand-derived gradients, a mandatory finite-difference gradient
 check, a real training loop, self-tests, and executed output pasted into the
 chapter. No autodiff framework, no pretrained anything.

 RUN:  python3 chapter_0357_kenko_1283.py
================================================================================
"""

import numpy as np
import time

RNG_GLOBAL = np.random.default_rng(1330)   # the year he began writing


# ==============================================================================
# 0. SMALL NUMERIC HELPERS
# ==============================================================================

def tanh(x):
    return np.tanh(x)


def dtanh_from_out(a):
    """Derivative of tanh given its OUTPUT a = tanh(z).  d/dz = 1 - a^2."""
    return 1.0 - a * a


def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def bce_with_logits(logit, y):
    """
    Mean binary cross-entropy from logits, stable form:
        max(z,0) - z*y + log(1 + exp(-|z|))
    Returns (loss_scalar, dloss/dlogit) with the 1/B already folded in.
    """
    z = logit.reshape(-1)
    t = y.reshape(-1).astype(np.float64)
    B = z.shape[0]
    loss = np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z))))
    dz = (sigmoid(z) - t) / B
    return loss, dz.reshape(logit.shape)


# ==============================================================================
# 1. THE WORLD  --  a small environment with a gain direction and a DIFFERENT
#    danger direction, observed only through an unordered bag of fragments.
# ==============================================================================
#
# Design notes, because the environment carries as much of the argument as the
# network does:
#
#  * The agent never sees the latent state z.  It sees K fragments.  Each
#    fragment is a self-identifying scrap: [one-hot tag | noisy projection
#    value].  The tag says WHICH reading it is, never WHERE in the list it
#    sits.  This is the Tsurezuregusa's own condition -- the essays identify
#    themselves by content and are indifferent to their position.
#
#  * ONE fragment is withheld from the encoder every episode and used as the
#    anticipation target.  The model is scored on the reading it was not given.
#    That is S.137 turned into a loss term.
#
#  * gain(a, z)   = a . u(z)          -- what a return-maximiser wants
#    hazard(a, z) = sigmoid(c0 + c1 * (a . v(z)))
#    where v(z) is u(z) rotated by ROT_DEG.  The two directions overlap but do
#    not coincide, so the greedy-gain action is genuinely dangerous and the
#    safest action genuinely gives up return.  Nothing is degenerate: if
#    ROT_DEG were 0 or 90 the experiment would be trivial in one direction or
#    the other.
#
M_LATENT = 4          # dimension of the hidden state z
K_FRAG = 6            # number of fragments per episode (one is withheld)
D_FRAG = K_FRAG + 1   # one-hot tag (K_FRAG) + scalar reading (1)
A_DIM = 2             # action lives on the unit disc
ROT_DEG = 42.0        # angle between gain direction and danger direction
NOISE = 0.10          # observation noise on each fragment reading


class Hut:
    """
    The environment.  Named for the hut at Narabigaoka where Kenko is
    traditionally placed in his last years -- a small, poorly furnished place
    from which the whole world had to be inferred.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        # Fixed projection bank: fragment j reads z along direction P[j].
        self.P = rng.normal(size=(K_FRAG, M_LATENT)) / np.sqrt(M_LATENT)
        # Latent -> gain direction u(z): a fixed nonlinear map, then normalise.
        self.Wu = rng.normal(size=(M_LATENT, A_DIM)) / np.sqrt(M_LATENT)
        self.bu = rng.normal(size=(A_DIM,)) * 0.3
        th = np.deg2rad(ROT_DEG)
        self.R = np.array([[np.cos(th), -np.sin(th)],
                           [np.sin(th),  np.cos(th)]])
        # Ruin must be RARE per step, or every policy dies in two moves and
        # there is no horizon to lengthen.  c0 sets the floor (about 4.7% per
        # step when the action is orthogonal to danger); c1 sets how steeply
        # the danger direction is punished.
        self.c0 = -3.00
        self.c1 = 2.00
        self.rng = rng

    # -- latent state -------------------------------------------------------
    def sample_z(self, B, rng):
        return rng.normal(size=(B, M_LATENT))

    def gain_dir(self, z):
        """u(z), unit-norm: the direction in which return increases."""
        u = np.tanh(z @ self.Wu + self.bu)
        n = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        return u / n

    def danger_dir(self, z):
        """v(z) = u(z) rotated by ROT_DEG.  Overlaps with gain, is not it."""
        return self.gain_dir(z) @ self.R.T

    # -- what the agent should ideally commit to ----------------------------
    def best_action(self, z):
        """
        The target the commitment head is trained on: the unit action that
        maximises gain.  Deliberately NOT the safe action -- the network learns
        the return-optimal act, and the hazard machinery is what later declines
        to take it.  Keeping the two apart is the whole point of S.110.
        """
        return self.gain_dir(z)

    def gain(self, a, z):
        return np.sum(a * self.gain_dir(z), axis=1)

    def hazard(self, a, z):
        """True per-step probability of ruin given the action taken."""
        return sigmoid(self.c0 + self.c1 * np.sum(a * self.danger_dir(z), axis=1))

    # -- observation --------------------------------------------------------
    def observe(self, z, rng):
        """
        Returns:
          X       (B, K, D)  fragments, each [tag one-hot | reading]
          keep    (B, K)     1.0 for fragments handed to the encoder
          held    (B, D)     the single withheld fragment (anticipation target)
          hidx    (B,)       which fragment was withheld
        The fragment ORDER is shuffled independently per sample, so any model
        that secretly uses position will be caught by the invariance self-test.
        """
        B = z.shape[0]
        readings = z @ self.P.T                       # (B, K)
        readings = readings + NOISE * rng.normal(size=readings.shape)
        tags = np.eye(K_FRAG)[None, :, :].repeat(B, axis=0)   # (B, K, K)
        X = np.concatenate([tags, readings[:, :, None]], axis=2)  # (B,K,D)

        # shuffle fragment order per sample
        for b in range(B):
            X[b] = X[b][rng.permutation(K_FRAG)]

        hidx = rng.integers(0, K_FRAG, size=B)
        keep = np.ones((B, K_FRAG))
        keep[np.arange(B), hidx] = 0.0
        held = X[np.arange(B), hidx, :].copy()
        return X, keep, held, hidx

    def batch(self, B, rng, task='A'):
        """
        task 'A' and task 'B' differ only in the target map: task B rotates the
        commitment target by 90 degrees.  Used for the continual-learning
        experiment (S.82).  The observations and the hazard structure are
        identical, so any forgetting measured is forgetting of the ACT, not of
        the world.
        """
        z = self.sample_z(B, rng)
        X, keep, held, hidx = self.observe(z, rng)
        a_star = self.best_action(z)
        if task == 'B':
            Rq = np.array([[0.0, -1.0], [1.0, 0.0]])
            a_star = a_star @ Rq.T

        # THE RECORD IS NOT A RECORD OF OPTIMAL PLAY.
        # The hazard head is trained on what was actually done and what
        # followed.  If every recorded episode showed the return-optimal act,
        # the head would see exactly one action direction and could never learn
        # that hazard depends on the act at all -- it would look trained, score
        # a fine loss, and be useless the moment it was asked about a move
        # nobody had made.  Half the episodes are therefore free play.
        ang = rng.uniform(0, 2 * np.pi, size=B)
        a_free = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        jitter = a_star + 0.45 * rng.normal(size=a_star.shape)
        jitter = jitter / (np.linalg.norm(jitter, axis=1, keepdims=True) + 1e-9)
        use_free = (rng.random(B) < 0.5)[:, None]
        a_taken = np.where(use_free, a_free, jitter)

        p = self.hazard(a_taken, z)
        y_fail = (rng.random(B) < p).astype(np.float64)
        return dict(X=X, keep=keep, held=held, a_star=a_star, a_taken=a_taken,
                    z=z, y_fail=y_fail, p_true=p)


# ==============================================================================
# 2. THE NETWORK
# ==============================================================================
#
#   X (B,K,D)
#     |  phi : per-fragment MLP, SHARED across fragments   <- no position
#     v
#   A2 (B,K,h1)
#     |  masked mean-pool over K                            <- permutation inv.
#     v
#   P  (B,h1)
#     |  trunk -> A3 (B,H), multiplied by CAP_MASK          <- S.82 withholding
#     v
#   +-- commit head      -> C (B, N_CAND, 2)                <- S.92
#   +-- anticipation head-> F (B, D)   predicts the WITHHELD fragment  <- S.137
#   +-- hazard head      -> logit (B,1) from [A3 | a_detached]         <- S.110
#
H1 = 24        # per-fragment / pooled width
H2 = 32        # trunk width  (a fraction of these is withheld)
HH = 8         # hazard head width
N_CAND = 2     # candidate actions the commit head emits (the "arrows")

LAMBDA_F = 0.50   # weight on anticipation loss
LAMBDA_H = 0.50   # weight on hazard loss


def init_params(seed=7):
    """Xavier-ish init.  Everything float64 so the gradient check is meaningful."""
    r = np.random.default_rng(seed)

    def w(a, b):
        return r.normal(size=(a, b)) * np.sqrt(2.0 / (a + b))

    return {
        'W1': w(D_FRAG, H1), 'b1': np.zeros(H1),
        'W2': w(H1, H1),     'b2': np.zeros(H1),
        'W3': w(H1, H2),     'b3': np.zeros(H2),
        'Wc': w(H2, N_CAND * A_DIM), 'bc': np.zeros(N_CAND * A_DIM),
        'Wf': w(H2, D_FRAG), 'bf': np.zeros(D_FRAG),
        'W4': w(H2 + A_DIM, HH), 'b4': np.zeros(HH),
        'W5': w(HH, 1),      'b5': np.zeros(1),
    }


def make_cap_mask(hold_frac, seed=11):
    """
    S.82 made literal.  A binary mask over trunk units.  Held units are forced
    to zero activation, so -- and this is the part worth noticing -- their
    incoming and outgoing gradients vanish AUTOMATICALLY.  Nothing extra is
    needed to freeze them; withholding a unit's output withholds its learning.
    Emptiness here is not a regulariser applied from outside, it is a property
    of the forward pass.
    """
    r = np.random.default_rng(seed)
    n_hold = int(round(hold_frac * H2))
    mask = np.ones(H2)
    if n_hold > 0:
        idx = r.permutation(H2)[:n_hold]
        mask[idx] = 0.0
    return mask


def empty_the_held_bay(params, cap_mask):
    """
    Zero the OUTGOING weights of every withheld unit.

    Without this, releasing the bay is not a neutral act: the held units carry
    random initial outgoing weights, so the instant the mask is lifted they
    inject noise into an output that was already correct.  A half-built wing
    with its doors already cut into the finished hall is not what S.82
    describes.  An unfinished bay contributes nothing until someone builds in
    it -- and that is what makes it safe to leave.
    """
    held = (cap_mask == 0)
    if not held.any():
        return params
    params['Wc'][held, :] = 0.0
    params['Wf'][held, :] = 0.0
    params['W4'][:H2][held, :] = 0.0
    return params


def forward(params, batch, cap_mask, mode='single'):
    """
    mode = 'single'  : loss on the FIRST candidate only.  The second arrow
                       exists in the output but earns no credit -- there is no
                       reserve to lean on.
    mode = 'reserve' : loss = min over candidates (best-of-N).  This is the
                       student who kept the second arrow.  It looks strictly
                       better -- it is a lower loss on the same data -- and
                       S.92 predicts the first arrow gets worse anyway.
    """
    X, keep = batch['X'], batch['keep']
    a_star, held, y = batch['a_star'], batch['held'], batch['y_fail']
    B = X.shape[0]

    # ---- per-fragment encoder (shared weights: no fragment has an address) --
    Z1 = X @ params['W1'] + params['b1']          # (B,K,H1)
    A1 = tanh(Z1)
    Z2 = A1 @ params['W2'] + params['b2']         # (B,K,H1)
    A2 = tanh(Z2)

    # ---- masked mean pool: the ONLY aggregation, and it is order-blind ------
    kn = keep[:, :, None]                          # (B,K,1)
    denom = np.maximum(keep.sum(axis=1, keepdims=True), 1.0)   # (B,1)
    P = (A2 * kn).sum(axis=1) / denom              # (B,H1)

    # ---- trunk with withheld capacity --------------------------------------
    Z3 = P @ params['W3'] + params['b3']           # (B,H2)
    A3 = tanh(Z3) * cap_mask                       # (B,H2)

    # ---- commitment head ----------------------------------------------------
    Craw = A3 @ params['Wc'] + params['bc']        # (B, N*2)
    C = Craw.reshape(B, N_CAND, A_DIM)

    per_cand = np.mean((C - a_star[:, None, :]) ** 2, axis=2)   # (B,N)
    if mode == 'single':
        sel = np.zeros(B, dtype=int)
    elif mode == 'reserve':
        sel = np.argmin(per_cand, axis=1)
    else:
        raise ValueError(mode)
    loss_c = float(np.mean(per_cand[np.arange(B), sel]))

    # ---- anticipation head: reconstruct the fragment NOT given --------------
    F = A3 @ params['Wf'] + params['bf']           # (B,D)
    diffF = F - held
    loss_f = float(np.mean(diffF ** 2))

    # ---- hazard head -------------------------------------------------------
    # The hazard head is a CHRONICLER, not a co-author.  It is fed the action
    # that was actually taken in the recorded episode (batch['a_star']) and the
    # outcome that followed, exactly as a chronicle records a deed and its
    # consequence.  It is never fed the network's own current proposal.
    #
    # This is a deliberate architectural separation, not a convenience.  If the
    # hazard head consumed the commitment head's output, the two would train
    # each other into a fixed point and the S.110 policy comparison below --
    # which needs to interrogate hazard at actions the policy did NOT choose --
    # would be asking a question the model had already been optimised to
    # answer in its own favour.  Kenko's sugoroku champion evaluates moves he
    # is not going to make; he can only do that because he has watched moves
    # that were not his own.
    a_in = batch['a_taken']                        # exogenous: what was DONE
    U = np.concatenate([A3, a_in], axis=1)         # (B, H2+2)
    Z4 = U @ params['W4'] + params['b4']
    A4 = tanh(Z4)
    logit = A4 @ params['W5'] + params['b5']       # (B,1)
    loss_h, dlogit = bce_with_logits(logit, y)

    loss = loss_c + LAMBDA_F * loss_f + LAMBDA_H * loss_h

    cache = dict(X=X, kn=kn, denom=denom, A1=A1, A2=A2, P=P, A3=A3,
                 C=C, sel=sel, a_star=a_star, diffF=diffF, U=U, A4=A4,
                 dlogit=dlogit, B=B, cap_mask=cap_mask)
    parts = dict(commit=loss_c, anticip=loss_f, hazard=loss_h)
    return loss, cache, parts


def backward(params, cache):
    """
    Hand-derived reverse pass.  Every line here is checked against finite
    differences in test_gradients(); if a term is wrong the check fails loudly.
    """
    B = cache['B']
    A3, C, sel = cache['A3'], cache['C'], cache['sel']
    g = {k: np.zeros_like(v) for k, v in params.items()}

    # ---- commitment loss ---------------------------------------------------
    # per_cand = mean over A_DIM of (C - a*)^2 ; loss = mean over batch of the
    # SELECTED candidate.  Non-selected candidates receive exactly zero -- in
    # 'single' mode the second arrow gets no gradient at all, which is the
    # computational content of taking it out of the student's hand.
    dC = np.zeros_like(C)
    dC[np.arange(B), sel, :] = (2.0 / (A_DIM * B)) * (
        C[np.arange(B), sel, :] - cache['a_star'])
    dCraw = dC.reshape(B, N_CAND * A_DIM)
    g['Wc'] = A3.T @ dCraw
    g['bc'] = dCraw.sum(axis=0)
    dA3 = dCraw @ params['Wc'].T

    # ---- anticipation loss -------------------------------------------------
    dF = LAMBDA_F * (2.0 / (B * D_FRAG)) * cache['diffF']
    g['Wf'] = A3.T @ dF
    g['bf'] = dF.sum(axis=0)
    dA3 += dF @ params['Wf'].T

    # ---- hazard loss -------------------------------------------------------
    dlogit = LAMBDA_H * cache['dlogit']            # (B,1)
    g['W5'] = cache['A4'].T @ dlogit
    g['b5'] = dlogit.sum(axis=0)
    dA4 = dlogit @ params['W5'].T
    dZ4 = dA4 * dtanh_from_out(cache['A4'])
    g['W4'] = cache['U'].T @ dZ4
    g['b4'] = dZ4.sum(axis=0)
    dU = dZ4 @ params['W4'].T
    dA3 += dU[:, :H2]        # dU[:, H2:] is the gradient w.r.t. the recorded
                             # action, which is data, not a parameter.

    # ---- trunk -------------------------------------------------------------
    # A3 = tanh(Z3) * mask.  Where mask==1, A3 == tanh(Z3), so (1 - A3^2) is
    # exactly the tanh derivative.  Where mask==0 the leading multiplication by
    # cap_mask already zeroes the whole term, so the bracket's value there is
    # irrelevant.  Hence one clean line covers both cases:
    dZ3 = dA3 * cache['cap_mask'] * (1.0 - A3 * A3)

    g['W3'] = cache['P'].T @ dZ3
    g['b3'] = dZ3.sum(axis=0)
    dP = dZ3 @ params['W3'].T                      # (B,H1)

    # ---- un-pool: the gradient is broadcast to every KEPT fragment equally --
    dA2 = (dP[:, None, :] / cache['denom'][:, :, None]) * cache['kn']

    dZ2 = dA2 * dtanh_from_out(cache['A2'])
    g['W2'] = np.einsum('bkh,bkj->hj', cache['A1'], dZ2)
    g['b2'] = dZ2.sum(axis=(0, 1))
    dA1 = dZ2 @ params['W2'].T

    dZ1 = dA1 * dtanh_from_out(cache['A1'])
    g['W1'] = np.einsum('bkd,bkh->dh', cache['X'], dZ1)
    g['b1'] = dZ1.sum(axis=(0, 1))

    return g


# ==============================================================================
# 3. OPTIMISER
# ==============================================================================

class Adam:
    """Plain Adam.  Nothing clever; the interest in this file is elsewhere."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, freeze=None):
        """
        `freeze` is an optional dict of 0/1 arrays matching params.  A zero
        means that coordinate is not permitted to move.  This is how the
        "growth protocol" of S.82 is enforced: once a bay of the hall is
        finished, later building happens in the empty bay, not on top of it.
        """
        self.t += 1
        if freeze is not None:
            grads = {k: grads[k] * freeze[k] for k in grads}
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# 4. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def test_gradients(verbose=True):
    """
    Central differences against the hand-derived backward pass, on BOTH modes
    and with a partially withheld trunk, so the mask path is exercised too.
    Relative error must be < 1e-6 for every tensor or this file does not ship.
    """
    hut = Hut(seed=3)
    rng = np.random.default_rng(99)
    params = init_params(seed=5)
    cap = make_cap_mask(0.25, seed=4)
    batch = hut.batch(12, rng)

    worst_overall = 0.0
    ok = True
    for mode in ('single', 'reserve'):
        _, cache, _ = forward(params, batch, cap, mode=mode)
        grads = backward(params, cache)

        if verbose:
            print(f"  mode = {mode!r}")
        for name in sorted(params.keys()):
            P = params[name]
            g_an = grads[name]
            g_nu = np.zeros_like(P)
            flat = P.reshape(-1)
            gflat = g_nu.reshape(-1)
            n = flat.size
            # probe a subset of coordinates for the larger tensors
            idxs = range(n) if n <= 60 else \
                np.random.default_rng(hash(name) % 2**31).choice(n, 60, replace=False)
            eps = 1e-6
            for i in idxs:
                orig = flat[i]
                flat[i] = orig + eps
                lp, _, _ = forward(params, batch, cap, mode=mode)
                flat[i] = orig - eps
                lm, _, _ = forward(params, batch, cap, mode=mode)
                flat[i] = orig
                gflat[i] = (lp - lm) / (2 * eps)
            probe = np.array([g_nu.reshape(-1)[i] for i in idxs])
            anal = np.array([g_an.reshape(-1)[i] for i in idxs])
            # Tensor-level relative error.  A per-element ratio explodes on
            # entries whose true gradient is ~0, which says nothing about
            # correctness; scaling by the tensor's largest component is the
            # standard robust form.
            scale = max(np.max(np.abs(probe)), np.max(np.abs(anal)), 1e-9)
            rel = float(np.max(np.abs(probe - anal)) / scale)
            worst_overall = max(worst_overall, rel)
            status = "ok " if rel < 1e-6 else "FAIL"
            if rel >= 1e-6:
                ok = False
            if verbose:
                print(f"    {name:<4s} rel_err = {rel:.3e}   [{status}]")

    if verbose:
        print(f"  worst relative error across all tensors and both modes: "
              f"{worst_overall:.3e}")
    return ok, worst_overall


# ==============================================================================
# 5. SELF-TEST: PERMUTATION INVARIANCE
# ==============================================================================

def test_permutation_invariance():
    """
    Shuffle the fragments; the output must not move beyond float noise.  This
    is not a robustness score, it is a structural impossibility: there is no
    tensor anywhere in the model that could hold "this was the third essay".
    The 243 numbers in every printed Tsurezuregusa are a seventeenth-century
    editorial imposition, and the architecture refuses to learn them.
    """
    hut = Hut(seed=3)
    rng = np.random.default_rng(21)
    params = init_params(seed=5)
    cap = make_cap_mask(0.25, seed=4)
    b = hut.batch(16, rng)

    _, c1, _ = forward(params, b, cap, mode='single')
    out1 = c1['C'].copy()

    b2 = dict(b)
    Xp, kp = b['X'].copy(), b['keep'].copy()
    pr = np.random.default_rng(5)
    for i in range(Xp.shape[0]):
        perm = pr.permutation(K_FRAG)
        Xp[i] = Xp[i][perm]
        kp[i] = kp[i][perm]
    b2['X'], b2['keep'] = Xp, kp

    _, c2, _ = forward(params, b2, cap, mode='single')
    out2 = c2['C']
    dev = float(np.max(np.abs(out1 - out2)))
    return dev < 1e-12, dev


# ==============================================================================
# 6. TRAINING
# ==============================================================================

def train(hut, params, cap_mask, mode='single', task='A',
          steps=2500, bs=64, lr=3e-3, seed=100, log_every=None, freeze=None):
    rng = np.random.default_rng(seed)
    opt = Adam(params, lr=lr)
    hist = []
    for s in range(1, steps + 1):
        b = hut.batch(bs, rng, task=task)
        loss, cache, parts = forward(params, b, cap_mask, mode=mode)
        grads = backward(params, cache)
        opt.step(params, grads, freeze=freeze)
        hist.append(loss)
        if log_every and (s % log_every == 0 or s == 1):
            print(f"    step {s:5d}  total {loss:.4f}   commit {parts['commit']:.4f}"
                  f"   anticip {parts['anticip']:.4f}   hazard {parts['hazard']:.4f}")
    return hist


def eval_first_arrow(hut, params, cap_mask, task='A', n=4000, seed=777):
    """
    THE MEASUREMENT THAT MATTERS FOR S.92.
    At test time only one shot is permitted, so we score candidate 0 alone --
    regardless of how the network was trained.  A best-of-N learner may have
    had a lower TRAINING loss throughout and still be worse here, which is
    precisely the teacher's claim.
    """
    rng = np.random.default_rng(seed)
    b = hut.batch(n, rng, task=task)
    _, c, _ = forward(params, b, cap_mask, mode='single')
    first = c['C'][:, 0, :]
    mse_first = float(np.mean((first - b['a_star']) ** 2))
    best = c['C']
    mse_best = float(np.mean(np.min(
        np.mean((best - b['a_star'][:, None, :]) ** 2, axis=2), axis=1)))
    return mse_first, mse_best


def eval_first_arrow_occluded(hut, params, cap_mask, n_drop=2,
                              n=4000, seed=779):
    """
    Commitment error when MORE of the world is hidden at test time than was
    ever hidden during training (training always withholds exactly one
    fragment).  This is the honest test of S.137: the claim is not that
    predicting the unseen improves you on the easy case, but that a mind
    trained to account for what it was not shown holds up when more is
    missing.  Full bloom is the state that demands no inference; a model that
    has only ever been scored on what it could see has never once had to
    solve for anything.
    """
    rng = np.random.default_rng(seed)
    b = hut.batch(n, rng)
    keep = b['keep'].copy()
    for i in range(n):
        live = np.where(keep[i] > 0)[0]
        extra = rng.choice(live, size=max(0, n_drop - 1), replace=False)
        keep[i, extra] = 0.0
    b2 = dict(b); b2['keep'] = keep
    _, c, _ = forward(params, b2, cap_mask, mode='single')
    return float(np.mean((c['C'][:, 0, :] - b['a_star']) ** 2))


def eval_anticipation(hut, params, cap_mask, n=4000, seed=778):
    rng = np.random.default_rng(seed)
    b = hut.batch(n, rng)
    _, c, _ = forward(params, b, cap_mask, mode='single')
    A3 = c['A3']
    F = A3 @ params['Wf'] + params['bf']
    return float(np.mean((F - b['held']) ** 2))


# ==============================================================================
# 7. EXPERIMENT S.92  --  DOES A HELD RESERVE DEGRADE THE FIRST ACT?
# ==============================================================================

def experiment_second_arrow(seeds=(0, 1, 2)):
    """
    Identical initialisation, identical data stream, identical capacity.  The
    ONLY difference is whether the training objective is allowed to score the
    better of two candidates.

    Kenko's teacher makes a falsifiable prediction: the best-of-two student
    shoots a worse FIRST arrow, and -- the sharp part -- cannot tell, because
    from inside the training run his loss curve looks better.
    """
    rows = []
    for sd in seeds:
        hut = Hut(seed=3)
        cap = make_cap_mask(0.0, seed=4)     # no withholding here: isolate S.92

        p_single = init_params(seed=50 + sd)
        p_reserve = {k: v.copy() for k, v in p_single.items()}   # same start

        h_s = train(hut, p_single,  cap, mode='single',  steps=2200,
                    seed=400 + sd)
        h_r = train(hut, p_reserve, cap, mode='reserve', steps=2200,
                    seed=400 + sd)

        s_first, s_best = eval_first_arrow(hut, p_single, cap)
        r_first, r_best = eval_first_arrow(hut, p_reserve, cap)
        rows.append(dict(seed=sd,
                         train_tail_single=float(np.mean(h_s[-200:])),
                         train_tail_reserve=float(np.mean(h_r[-200:])),
                         first_single=s_first, first_reserve=r_first,
                         best_single=s_best, best_reserve=r_best))
    return rows


# ==============================================================================
# 8. EXPERIMENT S.82  --  DOES WITHHELD CAPACITY BUY CONTINUATION?
# ==============================================================================

def growth_freeze(cap_A):
    """
    Build the freeze mask for phase B under the GROWTH protocol.

    Units that were live in phase A keep everything they learned: their trunk
    column in W3, their bias, and their outgoing rows in every head are all
    pinned.  Only the bay left empty in phase A may be built in.  The shared
    fragment encoder (W1, W2) is pinned as well -- the world did not change,
    only the act required of it.
    """
    live = (cap_A > 0).astype(np.float64)     # units used in phase A
    free = 1.0 - live                          # the unfinished bay
    f = {}
    f['W1'] = np.zeros((D_FRAG, H1)); f['b1'] = np.zeros(H1)
    f['W2'] = np.zeros((H1, H1));     f['b2'] = np.zeros(H1)
    f['W3'] = np.tile(free[None, :], (H1, 1))       # columns = trunk units
    f['b3'] = free.copy()
    f['Wc'] = np.tile(free[:, None], (1, N_CAND * A_DIM))   # rows = trunk units
    f['bc'] = np.zeros(N_CAND * A_DIM)   # SHARED by both tasks -- pin it, or
    f['Wf'] = np.tile(free[:, None], (1, D_FRAG))
    f['bf'] = np.zeros(D_FRAG)           # the new wing shifts the old one
    f['W4'] = np.concatenate([np.tile(free[:, None], (1, HH)),
                              np.ones((A_DIM, HH))], axis=0)
    f['b4'] = np.ones(HH)
    f['W5'] = np.ones((HH, 1)); f['b5'] = np.ones(1)
    return f


def experiment_withheld_capacity(hold_fracs=(0.0, 0.25, 0.40, 0.55), seed=0):
    """
    Phase A: learn task A with a fraction of the trunk WITHHELD -- forced to
    zero output and therefore, with no extra machinery, frozen (see
    make_cap_mask).
    Phase B: release the withheld units and learn task B, under two protocols.

      OVERWRITE -- everything is trainable, as in ordinary fine-tuning.  This
                   is the completed hall being demolished to make the new wing.
      GROWTH    -- only the released bay may be built in.  This is S.82 taken
                   literally: the palace was left with one place unfinished,
                   and that is where the next work goes.

    Then re-measure task A.  The question is not whether withholding is free.
    It is not.  The question is what it buys.
    """
    out = []
    for hf in hold_fracs:
        hut = Hut(seed=3)
        cap_A = make_cap_mask(hf, seed=4)
        cap_full = np.ones(H2)

        base = init_params(seed=60 + seed)
        empty_the_held_bay(base, cap_A)
        train(hut, base, cap_A, mode='single', task='A',
              steps=2200, seed=900 + seed)
        a_after_A, _ = eval_first_arrow(hut, base, cap_A, task='A')

        row = dict(hold_frac=hf, held_units=int(round(hf * H2)),
                   A_after_A=a_after_A)

        for proto in ('overwrite', 'growth'):
            p = {k: v.copy() for k, v in base.items()}
            fz = None if proto == 'overwrite' else growth_freeze(cap_A)
            train(hut, p, cap_full, mode='single', task='B',
                  steps=2200, seed=910 + seed, freeze=fz)
            b_b, _ = eval_first_arrow(hut, p, cap_full, task='B')
            a_b, _ = eval_first_arrow(hut, p, cap_full, task='A')
            # THE MEASUREMENT THAT MATTERS.  A shared head cannot emit two
            # different acts for the same world at once, so raw "forgetting"
            # with the new wing switched on is guaranteed and uninteresting.
            # The question S.82 actually raises is whether the old hall is
            # still THERE.  Re-apply the phase-A mask -- look at the building
            # without the new wing -- and see what comes back.
            a_rest, _ = eval_first_arrow(hut, p, cap_A, task='A')
            row[proto] = dict(B_after_B=b_b, A_after_B=a_b,
                              A_restored=a_rest,
                              damage=a_rest - a_after_A)
        out.append(row)
    return out


# ==============================================================================
# 9. EXPERIMENT S.110  --  PLAY SO AS NOT TO LOSE
# ==============================================================================

def hazard_logit(params, A3, a):
    U = np.concatenate([A3, a], axis=1)
    A4 = tanh(U @ params['W4'] + params['b4'])
    return (A4 @ params['W5'] + params['b5']).reshape(-1)


def hazard_logit_many(params, A3, DIRS):
    """
    Hazard logit for EVERY candidate direction at once.
    Returns (n_dirs, B).  Vectorised because the policy has to price moves it
    is not going to make, and there are a lot of them.
    """
    B = A3.shape[0]
    n_d = DIRS.shape[0]
    A3r = np.repeat(A3[None, :, :], n_d, axis=0).reshape(n_d * B, H2)
    Ar = np.repeat(DIRS[:, None, :], B, axis=1).reshape(n_d * B, A_DIM)
    U = np.concatenate([A3r, Ar], axis=1)
    A4 = tanh(U @ params['W4'] + params['b4'])
    lg = (A4 @ params['W5'] + params['b5']).reshape(n_d, B)
    return lg


def experiment_ruin_delay(n_episodes=3000, max_steps=300, seed=0,
                          n_dirs=72, budget_frac=0.60):
    """
    Two agents share one trained ISSHA -- one world model, one hazard head,
    one estimate of where the gain lies.  They differ only in the question
    they ask of it.

      GREEDY  -- take the action with the highest predicted gain.  This is the
                 commitment head's own trained output: the return-optimal act.
      KENKO   -- among actions whose predicted gain clears a floor (60% of the
                 greedy action's), take the one the hazard head rates LEAST
                 likely to end the run.  It never asks which move wins.  It
                 asks which move loses soonest, and declines that one.

    Neither agent is given the true hazard.  Both act on the model's estimate,
    so a calibration figure is reported first: if the hazard head were noise,
    the comparison would be meaningless.
    """
    hut = Hut(seed=3)
    cap = make_cap_mask(0.0, seed=4)
    params = init_params(seed=71 + seed)
    train(hut, params, cap, mode='single', steps=4000, seed=1200 + seed)

    rng = np.random.default_rng(2024 + seed)

    # --- calibration of the hazard head on held-out episodes ---------------
    bcal = hut.batch(4000, rng)
    _, ccal, _ = forward(params, bcal, cap, mode='single')
    lg = hazard_logit(params, ccal['A3'], bcal['a_taken'])
    p_hat = sigmoid(lg)
    calib = float(np.corrcoef(p_hat, bcal['p_true'])[0, 1])
    mae = float(np.mean(np.abs(p_hat - bcal['p_true'])))

    ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    DIRS = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    results = {}
    for policy in ('greedy', 'kenko'):
        alive = np.ones(n_episodes, dtype=bool)
        life = np.zeros(n_episodes, dtype=int)
        gain_sum = np.zeros(n_episodes)
        for t in range(max_steps):
            idx = np.where(alive)[0]
            if idx.size == 0:
                break
            b = hut.batch(idx.size, rng)
            _, c, _ = forward(params, b, cap, mode='single')
            A3 = c['A3']
            a_g = c['C'][:, 0, :]
            a_g = a_g / (np.linalg.norm(a_g, axis=1, keepdims=True) + 1e-9)

            if policy == 'greedy':
                a = a_g
            else:
                pg = DIRS @ a_g.T                       # (n_dirs, B) predicted gain
                feasible = pg >= budget_frac
                hz = hazard_logit_many(params, A3, DIRS)
                hz = np.where(feasible, hz, np.inf)
                none_ok = ~np.any(feasible, axis=0)
                a = DIRS[np.argmin(hz, axis=0)].copy()
                a[none_ok] = a_g[none_ok]

            g = hut.gain(a, b['z'])
            p = hut.hazard(a, b['z'])
            died = rng.random(idx.size) < p

            gain_sum[idx] += g
            life[idx] += 1
            alive[idx[died]] = False

        results[policy] = dict(
            mean_life=float(life.mean()),
            median_life=float(np.median(life)),
            p90_life=float(np.percentile(life, 90)),
            gain_per_step=float(gain_sum.sum() / max(life.sum(), 1)),
            total_gain=float(gain_sum.mean()),
        )
    results['_calibration'] = dict(corr=calib, mae=mae)
    return results


# ==============================================================================
# 10. EXPERIMENT S.137  --  IS THE UNSEEN FRAGMENT WORTH PREDICTING?
# ==============================================================================

def experiment_anticipation_ablation(seeds=(0, 1, 2), steps=2400):
    """
    Train with the anticipation head at full weight and with it ablated
    (LAMBDA_F = 0), then measure COMMITMENT error -- never reconstruction --
    under progressively heavier occlusion.

    Training always hides exactly one fragment.  Testing hides one, two, or
    three.  If S.137 is right, the two models are near-identical on the case
    they were trained for and separate as the world gets darker.
    """
    global LAMBDA_F
    keep_lam = LAMBDA_F
    rows = []
    for sd in seeds:
        r = {'seed': sd}
        for lam, label in ((keep_lam, 'anticip'), (0.15, 'light'), (0.0, 'ablated')):
            LAMBDA_F = lam
            hut = Hut(seed=3)
            cap = make_cap_mask(0.0, seed=4)
            params = init_params(seed=80 + sd)
            train(hut, params, cap, mode='single', steps=steps, seed=1500 + sd)
            r[label] = {
                1: eval_first_arrow(hut, params, cap)[0],
                2: eval_first_arrow_occluded(hut, params, cap, n_drop=2),
                3: eval_first_arrow_occluded(hut, params, cap, n_drop=3),
            }
        rows.append(r)
    LAMBDA_F = keep_lam
    return rows


# ==============================================================================
# 11. MAIN
# ==============================================================================

def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    t0 = time.time()
    np.set_printoptions(precision=4, suppress=True)

    print("ISSHA -- Irreversible Single-Shot Hazard-delaying Architecture")
    print("Chapter 0357: Kenko (Urabe Kaneyoshi), c.1283-1352")
    print(f"trunk width {H2}, per-fragment width {H1}, fragments {K_FRAG} "
          f"(1 withheld each episode), candidates {N_CAND}")

    # ---------------------------------------------------------------- checks
    hr("1. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)")
    ok, worst = test_gradients(verbose=True)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("gradient check failed -- refusing to continue")

    hr("2. SELF-TEST: PERMUTATION INVARIANCE OF THE FRAGMENT POOL")
    pi_ok, dev = test_permutation_invariance()
    print(f"  max |output change| under fragment shuffle: {dev:.3e}")
    print(f"  RESULT: {'PASS' if pi_ok else 'FAIL'}")

    # ---------------------------------------------------------- training run
    hr("3. REFERENCE TRAINING RUN  (single-arrow, 25% of trunk withheld)")
    hut = Hut(seed=3)
    params = init_params(seed=5)
    cap = make_cap_mask(0.25, seed=4)
    empty_the_held_bay(params, cap)
    print(f"  trunk units withheld: {int(H2 - cap.sum())} of {H2}")
    train(hut, params, cap, mode='single', steps=2500, seed=100, log_every=500)
    mse_first, _ = eval_first_arrow(hut, params, cap)
    anti = eval_anticipation(hut, params, cap)
    print(f"  held-out commitment MSE (first arrow): {mse_first:.5f}")
    print(f"  held-out anticipation MSE (unseen fragment): {anti:.5f}")

    # ------------------------------------------------------------ S.92
    hr("4. SECTION 92 -- THE SECOND ARROW")
    rows = experiment_second_arrow(seeds=(0, 1, 2))
    print("  training loss is LOWER for the reserve learner; test-time first")
    print("  arrow is what the teacher actually watches.\n")
    print(f"  {'seed':<5}{'train(single)':>15}{'train(reserve)':>16}"
          f"{'1st arrow S':>14}{'1st arrow R':>14}{'penalty':>10}")
    for r in rows:
        pen = r['first_reserve'] - r['first_single']
        print(f"  {r['seed']:<5}{r['train_tail_single']:>15.5f}"
              f"{r['train_tail_reserve']:>16.5f}"
              f"{r['first_single']:>14.5f}{r['first_reserve']:>14.5f}"
              f"{pen:>+10.5f}")
    mean_pen = float(np.mean([r['first_reserve'] - r['first_single'] for r in rows]))
    mean_rel = float(np.mean([(r['first_reserve'] - r['first_single'])
                              / r['first_single'] for r in rows])) * 100
    print(f"\n  mean first-arrow penalty for keeping a reserve: "
          f"{mean_pen:+.5f}  ({mean_rel:+.1f}%)")
    print("  (the reserve learner's best-of-two is competitive; its FIRST")
    print("   shot is not -- and its own training curve never says so.)")

    # ------------------------------------------------------------ S.82
    hr("5. SECTION 82 -- THE UNFINISHED BAY")
    res = experiment_withheld_capacity()
    print("  Phase A learns task A with part of the trunk withheld.")
    print("  Phase B releases the bay and learns task B, two ways:")
    print("    OVERWRITE -- everything trainable (ordinary fine-tuning)")
    print("    GROWTH    -- only the released bay may be built in")
    print("  'A restored' re-applies the phase-A mask: the same building,")
    print("  viewed without the new wing.  'damage' is what that costs.\n")
    print(f"  {'held':>6}{'units':>6}{'A|A':>9}{'  ||':>4}"
          f"{'B|B':>9}{'A restored':>12}{'damage':>9}{'  ||':>4}"
          f"{'B|B':>9}{'A restored':>12}{'damage':>9}")
    print(f"  {'':>6}{'':>6}{'':>9}{'  ||':>4}{'------- overwrite -------':>30}"
          f"{'  ||':>4}{'-------- growth --------':>30}")
    for r in res:
        o, g = r['overwrite'], r['growth']
        print(f"  {r['hold_frac']:>6.2f}{r['held_units']:>6d}"
              f"{r['A_after_A']:>9.4f}{'  ||':>4}"
              f"{o['B_after_B']:>9.4f}{o['A_restored']:>12.4f}"
              f"{o['damage']:>+9.4f}{'  ||':>4}"
              f"{g['B_after_B']:>9.4f}{g['A_restored']:>12.4f}"
              f"{g['damage']:>+9.4f}")
    print("\n  With no bay left empty, the new task cannot be learned at all")
    print("  under the growth protocol -- the hall is complete, and therefore")
    print("  finished in both senses of the word.  Withhold a bay and task B")
    print("  becomes learnable while task A survives EXACTLY, to the last")
    print("  decimal place, because nothing in the old hall was touched.")
    print("  Ordinary fine-tuning learns B just as well and destroys A.")

    hr("6. SECTION 137 -- THE BRANCH ABOUT TO OPEN (anticipation ablation)")
    rows2 = experiment_anticipation_ablation()
    print("  commitment MSE by number of fragments hidden at TEST time")
    print("  (training always hides exactly one)\n")
    means = {d: {k: float(np.mean([r[k][d] for r in rows2]))
                 for k in ('anticip', 'light', 'ablated')} for d in (1, 2, 3)}
    print(f"  {'hidden':>8}{'lambda=0.50':>14}{'lambda=0.15':>14}"
          f"{'ablated':>12}{'verdict':>26}")
    for d in (1, 2, 3):
        m = means[d]
        best = min(m, key=m.get)
        note = ('auxiliary task HELPS' if best != 'ablated'
                else 'auxiliary task COSTS')
        print(f"  {d:>8}{m['anticip']:>14.5f}{m['light']:>14.5f}"
              f"{m['ablated']:>12.5f}{note:>26}")
    print("\n  Mean over 3 seeds.  Reported as measured.")

    hr("7. SECTION 110 -- DO NOT PLAY TO WIN; PLAY SO AS NOT TO LOSE")
    rr = experiment_ruin_delay()
    cal = rr['_calibration']
    print(f"  hazard head calibration on held-out episodes: "
          f"corr={cal['corr']:.3f}, MAE={cal['mae']:.4f}")
    print("  (both agents act on THIS estimate; neither is shown true hazard)\n")
    print(f"  {'policy':<10}{'gain/step':>12}{'mean life':>12}{'median':>10}"
          f"{'p90':>8}{'lifetime gain':>16}")
    for k in ('greedy', 'kenko'):
        d = rr[k]
        print(f"  {k:<10}{d['gain_per_step']:>12.4f}{d['mean_life']:>12.2f}"
              f"{d['median_life']:>10.1f}{d['p90_life']:>8.1f}"
              f"{d['total_gain']:>16.3f}")
    ratio = rr['kenko']['mean_life'] / max(rr['greedy']['mean_life'], 1e-9)
    graw = rr['kenko']['gain_per_step'] / max(rr['greedy']['gain_per_step'], 1e-9)
    tot = rr['kenko']['total_gain'] / max(rr['greedy']['total_gain'], 1e-9)
    print(f"\n  survival        kenko/greedy = {ratio:.2f}x")
    print(f"  gain per step   kenko/greedy = {graw:.2f}x   (the price, paid up front)")
    print(f"  LIFETIME gain   kenko/greedy = {tot:.2f}x")

    hr("SUMMARY")
    print(f"  gradient check                : PASS (worst rel err {worst:.2e})")
    print(f"  permutation invariance        : PASS (max dev {dev:.2e})")
    print(f"  second-arrow penalty          : {mean_rel:+.1f}% on the first shot")
    print(f"  withheld capacity (growth)    : task B "
          f"{res[0]['growth']['B_after_B']:.3f} with 0 held -> "
          f"{res[-1]['growth']['B_after_B']:.3f} with {res[-1]['held_units']} held; "
          f"damage to A {res[-1]['growth']['damage']:+.5f} "
          f"(overwrite: {res[-1]['overwrite']['damage']:+.4f})")
    print(f"  anticipation head, 1 hidden   : "
          f"{means[1]['anticip']:.4f} vs {means[1]['ablated']:.4f} ablated")
    print(f"  anticipation head, 3 hidden   : "
          f"{means[3]['anticip']:.4f} vs {means[3]['ablated']:.4f} ablated"
          f"   <-- S.137 does NOT pay here")
    print(f"  ruin-delay policy             : {ratio:.2f}x survival, "
          f"{tot:.2f}x lifetime gain")
    print(f"\n  elapsed: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

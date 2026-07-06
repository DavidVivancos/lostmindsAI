#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0021_yajnavalkya_-700.py
The Witness Network  --  a neural architecture after Yajnavalkya (c. 700 BCE)
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
# Resume and Interactive Demos at https://artificiology.com/    
================================================================================

WHY THIS ARCHITECTURE IS *NOT* A TRANSFORMER
--------------------------------------------
The default modern design stores keys and values and lets a query attend over
them. That machine is built to *grasp objects*: everything it knows is a stored,
addressable representation. Yajnavalkya's whole teaching cuts the other way.
His one irreducible idea -- the cognitive signature that is his alone -- is the
**unobjectifiable witness**:

    "You can't see the seer of seeing, you can't hear the hearer of hearing,
     you can't think the thinker of thinking, you can't perceive the perceiver
     of perceiving."                       -- Brihadaranyaka Upanishad 3.4.2

    "By what means can one know the means of knowing?"  (BU 2.4.14 / 4.5.15)

    "There is no cessation of the knowing of the knower."   (BU 4.3.30)

So the knower (Atman) is (1) reached only by *negation* -- neti neti, "not this,
not this" (BU 2.3.6, 4.5.15); (2) the *invariant* that is the same in every
changing experience ("unborn, undying"); (3) *ungraspable by the known* -- you
cannot turn the knower into an object of knowledge; and (4) *persistent without
an object* -- in dreamless deep sleep (sushupti) there is no second thing, yet
awareness does not lapse (BU 4.3.23-32).

This file builds a network whose ENTIRE job is to extract that witness, and
whose four loss terms are literal renderings of those four doctrines:

    L_recon   reconstruction  -- the world is a *real* manifestation, not a
                                 hallucination; content must be modelable.
    L_invar   invariance      -- the witness is the SAME across a self's many
                                 states ("unborn, unchanging").       [Atman]
    L_unobj   unobjectifiable -- an adversary ("the grasper") tries to predict
                                 the witness FROM the content; the witness head
                                 is trained to make that impossible.
                                 "By what should one know the knower?"
    L_sleep   deep-sleep      -- with content removed (the null, object-less
                                 input), the witness must persist and still
                                 match the self. "no cessation of the knowing
                                 of the knower."                    [sushupti]

Everything is pure NumPy, from scratch. There is a passing finite-difference
gradient check (mandatory), a real adversarial training loop, and self-tests
that each dramatize one of Yajnavalkya's claims. Run the file; the output it
prints is the output pasted into the chapter.

--------------------------------------------------------------------------------
GENERATIVE STORY OF THE DATA (synthetic, but principled)
--------------------------------------------------------------------------------
There are N "selves" (jivas). Each self s owns ONE hidden witness vector w_s
(its Atman). For that self we observe T "states" (waking experiences). Each
state mixes the *same* witness with a *fresh* content vector c (the changing
koshas) through a fixed nonlinear entangling map:

        x = tanh( ENTANGLE @ [w ; c] )          (raw experience)

In raw experience the witness and the content are inseparably entangled. The
network never sees w or c; it must disentangle the invariant witness out of the
stream by negation. A special "deep sleep" observation removes the content:

        x_sleep = tanh( ENTANGLE @ [w ; 0] )    (object-less awareness)

================================================================================
"""

import numpy as np

# A single seed makes the whole run -- data, init, training -- reproducible, so
# the numbers printed here are exactly the numbers quoted in the chapter.
RNG = np.random.default_rng(700)  # 700 ~ traditional floruit, c. 700 BCE


# ----------------------------------------------------------------------------
# 0.  Small numeric helpers (activations and their derivatives)
# ----------------------------------------------------------------------------
def tanh(z):
    return np.tanh(z)


def dtanh(a):
    # derivative expressed in terms of the *activation* a = tanh(z): 1 - a^2
    return 1.0 - a * a


# ----------------------------------------------------------------------------
# 1.  THE DATA: selves, their invariant witnesses, and their changing states
# ----------------------------------------------------------------------------
def make_world(n_selves, n_states, d_w, d_c, d_x, rng):
    """
    Build the synthetic world described in the header.

    Returns
    -------
    X        : (n_selves, n_states, d_x)  observed experiences
    X_sleep  : (n_selves, d_x)            object-less (deep-sleep) experiences
    W_true   : (n_selves, d_w)            the hidden witnesses (for diagnostics
                                          only -- the network never sees these)
    """
    # ENTANGLE mixes a (d_w + d_c) latent into a d_x observation. It is fixed
    # (part of "the world"), not learned -- the network must invert it.
    entangle = rng.standard_normal((d_x, d_w + d_c)) / np.sqrt(d_w + d_c)

    W_true = rng.standard_normal((n_selves, d_w))           # one Atman per self
    X = np.zeros((n_selves, n_states, d_x))
    X_sleep = np.zeros((n_selves, d_x))

    for s in range(n_selves):
        w = W_true[s]
        for t in range(n_states):
            c = rng.standard_normal(d_c)                    # fresh kosha/content
            X[s, t] = tanh(entangle @ np.concatenate([w, c]))
        # deep sleep: identical witness, content zeroed out (no "second thing")
        X_sleep[s] = tanh(entangle @ np.concatenate([w, np.zeros(d_c)]))

    return X, X_sleep, W_true


# ----------------------------------------------------------------------------
# 2.  THE WITNESS NETWORK (encoder + witness head + content head + decoder)
# ----------------------------------------------------------------------------
class WitnessNetwork:
    """
    Forward graph
    -------------
        h    = tanh(x  @ We + be)              shared encoder
        w_hat= h @ Ww + bw                     the extracted witness  (Atman)
        c_hat= tanh(h @ Wc + bc)               the extracted content  (koshas)
        g    = tanh([w_hat;c_hat] @ Wd1 + bd1) decoder hidden
        x_hat= g @ Wd2 + bd2                   reconstructed experience

    The witness head is deliberately *linear* (no squashing): the witness is not
    one more bounded feature among the koshas; it is the bare subject.
    """

    def __init__(self, d_x, d_h, d_w, d_c, rng):
        self.d_x, self.d_h, self.d_w, self.d_c = d_x, d_h, d_w, d_c

        def he(shape, fan_in):
            return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

        # encoder
        self.We = he((d_x, d_h), d_x); self.be = np.zeros(d_h)
        # witness head (linear) and content head (tanh)
        self.Ww = he((d_h, d_w), d_h); self.bw = np.zeros(d_w)
        self.Wc = he((d_h, d_c), d_h); self.bc = np.zeros(d_c)
        # decoder: (d_w + d_c) -> d_h -> d_x
        self.Wd1 = he((d_w + d_c, d_h), d_w + d_c); self.bd1 = np.zeros(d_h)
        self.Wd2 = he((d_h, d_x), d_h); self.bd2 = np.zeros(d_x)

    # ---- ordered views of the parameters (used by the gradient checker) ----
    def params(self):
        return [self.We, self.be, self.Ww, self.bw, self.Wc, self.bc,
                self.Wd1, self.bd1, self.Wd2, self.bd2]

    def names(self):
        return ["We", "be", "Ww", "bw", "Wc", "bc", "Wd1", "bd1", "Wd2", "bd2"]

    def set_params(self, flat_list):
        (self.We, self.be, self.Ww, self.bw, self.Wc, self.bc,
         self.Wd1, self.bd1, self.Wd2, self.bd2) = flat_list

    # ----------------------------- forward ----------------------------------
    def forward(self, X):
        """X: (B, d_x) -> dict of activations (everything needed for backprop)."""
        h = tanh(X @ self.We + self.be)
        w_hat = h @ self.Ww + self.bw
        c_hat = tanh(h @ self.Wc + self.bc)
        z = np.concatenate([w_hat, c_hat], axis=1)
        g = tanh(z @ self.Wd1 + self.bd1)
        x_hat = g @ self.Wd2 + self.bd2
        return dict(X=X, h=h, w_hat=w_hat, c_hat=c_hat, z=z, g=g, x_hat=x_hat)


# ----------------------------------------------------------------------------
# 3.  THE GRASPER (the adversary): tries to "know the knower" from the known
# ----------------------------------------------------------------------------
class Grasper:
    """
    g1     = tanh(c_hat @ Wg1 + bg1)
    w_pred = g1 @ Wg2 + bg2          (its guess at the witness, from content)

    Its existence is the experiment behind BU 2.4.14: can the witness be reached
    THROUGH the objects of knowledge? The network is trained so the answer is no.
    """

    def __init__(self, d_c, d_h, d_w, rng):
        def he(shape, fan_in):
            return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)
        self.Wg1 = he((d_c, d_h), d_c); self.bg1 = np.zeros(d_h)
        self.Wg2 = he((d_h, d_w), d_h); self.bg2 = np.zeros(d_w)

    def params(self):
        return [self.Wg1, self.bg1, self.Wg2, self.bg2]

    def names(self):
        return ["Wg1", "bg1", "Wg2", "bg2"]

    def set_params(self, flat_list):
        self.Wg1, self.bg1, self.Wg2, self.bg2 = flat_list

    def forward(self, c_hat):
        g1 = tanh(c_hat @ self.Wg1 + self.bg1)
        w_pred = g1 @ self.Wg2 + self.bg2
        return dict(c_hat=c_hat, g1=g1, w_pred=w_pred)


# ----------------------------------------------------------------------------
# 4.  THE LOSS  (four doctrines, one scalar for the witness network)
# ----------------------------------------------------------------------------
def witness_loss(net, grasper, X_flat, self_index, X_sleep, sleep_index,
                 alpha, beta, gamma, lam=0.02, margin=0.5, return_cache=False):
    """
    Total objective MINIMISED by the witness network's parameters.

        L_theta = L_recon + alpha*L_invar + beta*L_sleep
                          + lam*L_norm   + gamma*L_unobj

    The unobjectifiability term L_unobj is a *bounded* margin hinge:

        L_unobj = mean_i  max(0, margin - 0.5*||w_pred_i - w_hat_i||^2)

    It pushes the grasper's per-sample error UP TO `margin` and then stops.
    Without that ceiling the network would "fool" the grasper trivially by
    inflating the witness magnitude -- which is exactly the wrong reading of
    Yajnavalkya (the Atman is not a big quantity, it is a contentless subject).
    L_norm = 0.5*mean(||w_hat||^2) fixes the witness scale so the margin is
    meaningful and the only way to win is genuine disentanglement: keep the
    witness OUT of the objectifiable content stream.

    `self_index` maps each row of X_flat to its self id (for invariance and
    deep-sleep). The grasper's parameters are held fixed here.
    """
    out = net.forward(X_flat)
    w_hat, c_hat, x_hat = out["w_hat"], out["c_hat"], out["x_hat"]
    B = X_flat.shape[0]

    # --- L_recon : the manifestation is real -> reconstruct the experience ---
    diff = x_hat - X_flat
    L_recon = 0.5 * np.sum(diff * diff) / B

    # --- L_invar : the witness is the same in every state of one self --------
    n_selves = self_index.max() + 1
    means = np.zeros((n_selves, w_hat.shape[1]))
    counts = np.zeros(n_selves)
    np.add.at(means, self_index, w_hat)
    np.add.at(counts, self_index, 1.0)
    means /= counts[:, None]
    centred = w_hat - means[self_index]
    L_invar = 0.5 * np.sum(centred * centred) / B

    # --- L_sleep : object-less awareness still recovers the self's witness ---
    sleep_out = net.forward(X_sleep)
    w_sleep = sleep_out["w_hat"]                 # witness from null-content input
    target_sleep = means[sleep_index]            # that self's mean waking witness
    sdiff = w_sleep - target_sleep
    L_sleep = 0.5 * np.sum(sdiff * sdiff) / X_sleep.shape[0]

    # --- L_norm : bound the witness scale (the Atman is not a magnitude) -----
    L_norm = 0.5 * np.sum(w_hat * w_hat) / B

    # --- L_unobj : bounded "you cannot know the knower from the known" -------
    gout = grasper.forward(c_hat)
    gdiff = gout["w_pred"] - w_hat
    q = 0.5 * np.sum(gdiff * gdiff, axis=1)      # per-sample grasp error
    mask = (margin - q) > 0.0                    # only push where below margin
    L_unobj = np.mean(np.maximum(0.0, margin - q))
    L_grasp = np.mean(q)                          # reported for monitoring

    L = (L_recon + alpha * L_invar + beta * L_sleep
         + lam * L_norm + gamma * L_unobj)

    if not return_cache:
        return L
    cache = dict(out=out, sleep_out=sleep_out, gout=gout, means=means,
                 counts=counts, centred=centred, diff=diff, sdiff=sdiff,
                 gdiff=gdiff, mask=mask, target_sleep=target_sleep,
                 L_recon=L_recon, L_invar=L_invar, L_sleep=L_sleep,
                 L_norm=L_norm, L_unobj=L_unobj, L_grasp=L_grasp, L=L,
                 lam=lam, margin=margin)
    return L, cache


def witness_grads(net, grasper, X_flat, self_index, X_sleep, sleep_index,
                  alpha, beta, gamma, lam=0.02, margin=0.5):
    """
    Analytic gradient of `witness_loss` w.r.t. every WitnessNetwork parameter.
    Returns grads in the same order as net.params(). The grasper is fixed.
    """
    L, c = witness_loss(net, grasper, X_flat, self_index, X_sleep, sleep_index,
                        alpha, beta, gamma, lam=lam, margin=margin,
                        return_cache=True)
    out, sout, gout = c["out"], c["sleep_out"], c["gout"]
    B = X_flat.shape[0]
    Bs = X_sleep.shape[0]

    # ---- (a) recon flows back through the decoder ----
    dxhat = c["diff"] / B                                  # (B, d_x)
    gWd2 = out["g"].T @ dxhat
    gbd2 = dxhat.sum(0)
    dg = dxhat @ net.Wd2.T
    dz = dg * dtanh(out["g"])
    gWd1 = out["z"].T @ dz
    gbd1 = dz.sum(0)
    dzin = dz @ net.Wd1.T
    dw_recon = dzin[:, :net.d_w]
    dc_recon = dzin[:, net.d_w:]

    # ---- (b) invariance touches the witness head directly ----
    dw_invar = alpha * c["centred"] / B

    # ---- (c) bounded unobjectifiability hinge: gamma * L_unobj ----
    # L_unobj = mean_i relu(margin - 0.5||gdiff_i||^2);  gdiff = w_pred - w_hat
    #   dL_unobj/dw_pred = -(1/B)*mask*gdiff ,  dL_unobj/dw_hat = +(1/B)*mask*gdiff
    mask = c["mask"].astype(float)[:, None]               # (B,1) active samples
    gdiff = c["gdiff"]
    coeff = gamma / B
    dUnobj_dwpred = -coeff * mask * gdiff
    dUnobj_dwhat = coeff * mask * gdiff
    # push w_pred gradient back through the (fixed) grasper to c_hat
    dg1 = dUnobj_dwpred @ grasper.Wg2.T
    dpre1 = dg1 * dtanh(gout["g1"])
    dUnobj_dchat = dpre1 @ grasper.Wg1.T
    dw_grasp = dUnobj_dwhat
    dc_grasp = dUnobj_dchat

    # ---- (d) witness-norm regulariser: lam * 0.5*mean||w_hat||^2 ----
    dw_norm = lam * out["w_hat"] / B

    # ---- assemble head gradients (waking pass) ----
    dw_hat = dw_recon + dw_invar + dw_grasp + dw_norm      # (B, d_w)
    dc_hat_pre_tanh = (dc_recon + dc_grasp) * dtanh(out["c_hat"])

    gWw = out["h"].T @ dw_hat
    gbw = dw_hat.sum(0)
    gWc = out["h"].T @ dc_hat_pre_tanh
    gbc = dc_hat_pre_tanh.sum(0)

    dh = dw_hat @ net.Ww.T + dc_hat_pre_tanh @ net.Wc.T
    dh_pre = dh * dtanh(out["h"])
    gWe = out["X"].T @ dh_pre
    gbe = dh_pre.sum(0)

    # ---- deep-sleep pass contributes to Ww, bw, We, be only (witness head) --
    # L_sleep = 0.5/Bs * sum ||w_sleep - target_sleep||^2, target_sleep = mean
    # of that self's *waking* w_hat. We treat target_sleep as a function of the
    # waking means too; but its gradient back into waking w_hat is small and, to
    # keep the analytic/numeric check exact, we DO include it.
    sdiff = c["sdiff"]                                    # (Bs, d_w)
    dwsleep = beta * sdiff / Bs                           # d/d w_sleep
    # back through sleep encoder
    gWw += sout["h"].T @ dwsleep
    gbw += dwsleep.sum(0)
    dh_s = dwsleep @ net.Ww.T
    dh_s_pre = dh_s * dtanh(sout["h"])
    gWe += sout["X"].T @ dh_s_pre
    gbe += dh_s_pre.sum(0)
    # gradient of -target_sleep into the waking means -> distribute to waking w
    # target_sleep[i] = means[sleep_index[i]]; means[s] = mean of waking w of s.
    # dL_sleep/dmeans[s] = -beta/Bs * sum_{i: sleep_index[i]=s} sdiff[i]
    dmeans = np.zeros_like(c["means"])
    np.add.at(dmeans, sleep_index, -beta * sdiff / Bs)
    # means[s] = (1/counts[s]) * sum_{rows of self s} w_hat  -> scatter back
    extra_dw = dmeans[self_index] / c["counts"][self_index][:, None]
    gWw += out["h"].T @ extra_dw
    gbw += extra_dw.sum(0)
    dh_extra = extra_dw @ net.Ww.T
    dh_extra_pre = dh_extra * dtanh(out["h"])
    gWe += out["X"].T @ dh_extra_pre
    gbe += dh_extra_pre.sum(0)

    return [gWe, gbe, gWw, gbw, gWc, gbc, gWd1, gbd1, gWd2, gbd2]


def grasper_loss(net, grasper, X_flat, return_cache=False):
    """
    Objective MINIMISED by the grasper: predict the witness from content.
    The witness network is fixed (its w_hat / c_hat are treated as constants).
    """
    out = net.forward(X_flat)
    w_hat = out["w_hat"]            # constant target ("stop gradient")
    c_hat = out["c_hat"]           # constant input
    gout = grasper.forward(c_hat)
    gdiff = gout["w_pred"] - w_hat
    B = X_flat.shape[0]
    L = 0.5 * np.sum(gdiff * gdiff) / B
    if not return_cache:
        return L
    return L, dict(out=out, gout=gout, gdiff=gdiff, c_hat=c_hat)


def grasper_grads(net, grasper, X_flat):
    """Analytic gradient of grasper_loss w.r.t. the grasper parameters."""
    L, c = grasper_loss(net, grasper, X_flat, return_cache=True)
    gout = c["gout"]
    B = X_flat.shape[0]
    dwpred = c["gdiff"] / B                       # (B, d_w)
    gWg2 = gout["g1"].T @ dwpred
    gbg2 = dwpred.sum(0)
    dg1 = dwpred @ grasper.Wg2.T
    dpre1 = dg1 * dtanh(gout["g1"])
    gWg1 = c["c_hat"].T @ dpre1
    gbg1 = dpre1.sum(0)
    return [gWg1, gbg1, gWg2, gbg2]


# ----------------------------------------------------------------------------
# 5.  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; must pass)
# ----------------------------------------------------------------------------
def gradient_check(loss_fn, analytic_grads, obj, eps=1e-6):
    """
    Central finite differences vs analytic grads for every parameter of `obj`.
    `loss_fn` is a zero-arg closure recomputing the scalar loss from current
    params. Returns the worst relative error across all parameters.
    """
    base_params = [p.copy() for p in obj.params()]
    names = obj.names()
    worst = 0.0
    for k, (P, A, nm) in enumerate(zip(base_params, analytic_grads, names)):
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            saved = P[idx]
            P[idx] = saved + eps
            obj.set_params([bp if j != k else P
                            for j, bp in enumerate(base_params)])
            Lp = loss_fn()
            P[idx] = saved - eps
            obj.set_params([bp if j != k else P
                            for j, bp in enumerate(base_params)])
            Lm = loss_fn()
            num[idx] = (Lp - Lm) / (2 * eps)
            P[idx] = saved
            it.iternext()
        obj.set_params([bp.copy() for bp in base_params])  # restore
        denom = np.maximum(1e-8, np.abs(num) + np.abs(A))
        rel = np.max(np.abs(num - A) / denom)
        worst = max(worst, rel)
        print(f"    grad-check {nm:>4}: max rel err = {rel:.2e}")
    obj.set_params([bp.copy() for bp in base_params])
    return worst


# ----------------------------------------------------------------------------
# 6.  TRAINING (adversarial: witness network vs grasper)
# ----------------------------------------------------------------------------
def sgd(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g


def train(net, grasper, X, X_sleep, self_index, sleep_index,
          alpha, beta, gamma, lam=0.02, margin=0.5,
          lr=0.05, epochs=400, grasper_steps=2, verbose=True):
    X_flat = X.reshape(-1, X.shape[-1])
    history = []
    for ep in range(epochs):
        # (i) let the grasper try hard to know the knower (a few inner steps)
        for _ in range(grasper_steps):
            gg = grasper_grads(net, grasper, X_flat)
            sgd(grasper.params(), gg, lr)
        # (ii) one step of the witness network (minimise its objective)
        gw = witness_grads(net, grasper, X_flat, self_index,
                           X_sleep, sleep_index, alpha, beta, gamma,
                           lam=lam, margin=margin)
        sgd(net.params(), gw, lr)

        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            _, c = witness_loss(net, grasper, X_flat, self_index,
                                X_sleep, sleep_index, alpha, beta, gamma,
                                lam=lam, margin=margin, return_cache=True)
            history.append((ep, c["L_recon"], c["L_invar"],
                            c["L_sleep"], c["L_grasp"]))
            print(f"  epoch {ep:4d} | recon {c['L_recon']:.4f}  "
                  f"invar {c['L_invar']:.4f}  sleep {c['L_sleep']:.4f}  "
                  f"grasp(MSE) {c['L_grasp']:.4f}")
    return history


# ----------------------------------------------------------------------------
# 7.  SELF-TESTS  (each one dramatises a Yajnavalkya doctrine)
# ----------------------------------------------------------------------------
def self_tests(net, grasper, X, X_sleep, self_index, sleep_index, W_true):
    X_flat = X.reshape(-1, X.shape[-1])
    out = net.forward(X_flat)
    w_hat = out["w_hat"]
    n_selves = self_index.max() + 1

    # within-self spread of the witness (should be small: "unborn, unchanging")
    means = np.zeros((n_selves, w_hat.shape[1]))
    counts = np.zeros(n_selves)
    np.add.at(means, self_index, w_hat)
    np.add.at(counts, self_index, 1.0)
    means /= counts[:, None]
    within = np.sqrt(np.mean((w_hat - means[self_index]) ** 2))
    between = np.sqrt(np.mean((means - means.mean(0)) ** 2))
    print(f"  [Atman: invariance]   within-self witness spread = {within:.4f}")
    print(f"  [Atman: distinctness] between-self witness spread = {between:.4f}")
    print(f"      -> separation ratio between/within = {between / within:6.2f} "
          f"(higher = the witness is steady within a self, distinct across selves)")

    # deep sleep: witness recovered from the object-less input?
    w_sleep = net.forward(X_sleep)["w_hat"]
    sleep_err = np.sqrt(np.mean((w_sleep - means) ** 2))
    sleep_norm = np.sqrt(np.mean(w_sleep ** 2))
    print(f"  [Sushupti: persistence] witness norm with NO object = {sleep_norm:.4f} "
          f"(nonzero -> awareness does not lapse)")
    print(f"  [Sushupti: fidelity]    deep-sleep vs waking witness err = {sleep_err:.4f}")

    # unobjectifiability: can the grasper know the knower from the known?
    gpred = grasper.forward(out["c_hat"])["w_pred"]
    grasp_err = np.sqrt(np.mean((gpred - w_hat) ** 2))
    witness_spread = np.sqrt(np.mean((w_hat - w_hat.mean(0)) ** 2))
    print(f"  [Neti neti: ungraspable] grasper error from content = {grasp_err:.4f}")
    print(f"      vs total witness spread = {witness_spread:.4f} "
          f"(grasper error >= spread means content reveals ~nothing about the knower)")
    return dict(within=within, between=between, sleep_err=sleep_err,
                sleep_norm=sleep_norm, grasp_err=grasp_err,
                witness_spread=witness_spread)


# ----------------------------------------------------------------------------
# 8.  MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("THE WITNESS NETWORK  --  after Yajnavalkya (Brihadaranyaka Upanishad)")
    print("=" * 74)

    # ---- tiny dims for an exact, fast gradient check ----
    d_w, d_c, d_x, d_h = 3, 4, 6, 8
    n_selves, n_states = 5, 4

    X, X_sleep, W_true = make_world(n_selves, n_states, d_w, d_c, d_x, RNG)
    X_flat = X.reshape(-1, d_x)
    self_index = np.repeat(np.arange(n_selves), n_states)
    sleep_index = np.arange(n_selves)

    net = WitnessNetwork(d_x, d_h, d_w, d_c, RNG)
    grasper = Grasper(d_c, d_h, d_w, RNG)
    alpha, beta, gamma, lam, margin = 1.0, 0.5, 0.3, 0.02, 0.5

    # ---------------- gradient checks ----------------
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK -- witness network")
    aw = witness_grads(net, grasper, X_flat, self_index, X_sleep,
                       sleep_index, alpha, beta, gamma, lam=lam, margin=margin)
    worst_w = gradient_check(
        lambda: witness_loss(net, grasper, X_flat, self_index, X_sleep,
                             sleep_index, alpha, beta, gamma,
                             lam=lam, margin=margin),
        aw, net)
    print(f"  worst rel err (witness net) = {worst_w:.2e}")

    print("\n[2] FINITE-DIFFERENCE GRADIENT CHECK -- grasper (adversary)")
    ag = grasper_grads(net, grasper, X_flat)
    worst_g = gradient_check(
        lambda: grasper_loss(net, grasper, X_flat), ag, grasper)
    print(f"  worst rel err (grasper)     = {worst_g:.2e}")

    assert worst_w < 1e-4, "witness-network gradient check FAILED"
    assert worst_g < 1e-4, "grasper gradient check FAILED"
    print("\n  ==> BOTH GRADIENT CHECKS PASS (< 1e-4)\n")

    # ---------------- training ----------------
    print("[3] ADVERSARIAL TRAINING  (witness network vs the grasper)")
    train(net, grasper, X, X_sleep, self_index, sleep_index,
          alpha, beta, gamma, lam=lam, margin=margin,
          lr=0.05, epochs=600, grasper_steps=2)

    # ---------------- self-tests ----------------
    print("\n[4] SELF-TESTS  (each dramatises one Upanishadic claim)")
    self_tests(net, grasper, X, X_sleep, self_index, sleep_index, W_true)

    print("\n" + "=" * 74)
    print("neti neti: what could be reconstructed was not the witness;")
    print("what remained, invariant and ungraspable, was.")
    print("=" * 74)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Encyclopedia of Lost Minds  ·  Mind #0196  ·  Wang Wei (699-759/761 CE)
The Still-Mirror Network — a Non-Abiding Reflection architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0196_wang_wei_699 - Wang Wei (699-759/761 CE)
================================================================================  

WHO, AND WHY THIS DESIGN
------------------------------------------------------------------------------
Wang Wei (王維), courtesy name Mojie (摩詰, after the layman-sage Vimalakirti),
was a Tang poet, painter and musician who also wrote the memorial stele for the
Sixth Chan Patriarch, Huineng. His whole cognitive signature can be read off a
single line from the Diamond Sutra that Southern Chan made its motto and that
Wang Wei carved into that stele:

        應無所住而生其心   —  "abiding nowhere, the mind arises."

His most famous quatrain, 'Deer Park' (鹿柴), is that line turned into a scene:

        空山不見人        empty mountain, no one to be seen
        但聞人語響        yet the echo of voices is heard
        返景入深林        returning light enters the deep forest
        復照青苔上        and shines again on the green moss

Notice HOW the scene is known. The mountain is *empty*: the expected thing
(a person) is subtracted away, cancelled, un-seen. What registers is only the
RESIDUAL the empty mirror could not already contain — a voice's echo, a shaft
of light returning. Presence is read by negation; memory is a trace that is
held only long enough to echo, then released back to emptiness (non-abiding).
And Vimalakirti's supreme teaching on non-duality was *silence* — the withheld
word that completes the said, the negative space that completes the painting.

Most language-model architectures do the opposite of this mind. They GRASP:
they store keys and attend back over an ever-growing hoard of representations.
Wang Wei's cognition abides nowhere. So this network is not attention over
stored keys. It is a still surface that:

  1. PREDICTS-AND-CANCELS.  A "still mirror" predicts the incoming signal from
     its current reflection; only the residual (the un-foreseen echo) is passed
     on. This is predictive coding as negation — 空山不見人 / 但聞人語響.

  2. ABIDES NOWHERE.  The state leaks toward zero every step by a learned
     factor. The mind holds a trace just long enough for the echo to return
     ('returning light'), then relaxes to the empty mountain. The model learns
     *how little to hold* — the leak rate alpha is a trained parameter.

  3. KEEPS SILENT.  A learned gate lets the surface answer only when the echo
     is loud enough; faint stirrings are absorbed. Withholding is the default,
     not the failure — Vimalakirti's silence, the painter's negative space.

  4. REFLECTS ON ONE SURFACE.  Input is three channels — read them as the three
     arts Wang Wei fused (word / image / tone). Prediction and echo are shared
     across channels, so a voice in one art can bloom as light in another. One
     mind, three modalities, no separate modules ("poetry in his painting,
     painting in his poetry" — Su Shi).

THE ONE EQUATION SET (per time-step t)
------------------------------------------------------------------------------
    p_t   = W_pred · h_{t-1}                 # what the mirror already reflects
    r_t   = x_t - p_t                        # the echo: only the un-foreseen
    pre_t = W_in · r_t + W_res · h_{t-1}
    h_t   = (1-alpha)·h_{t-1} + tanh(pre_t)  # leak to emptiness + reflect echo
    z_t   = W_out · h_t + b_out              # the light the surface holds
    e_t   = Σ z_t²                           # brightness of the returning light
    g_t   = sigmoid(w_g·e_t + b_g)           # silence gate keyed on that light
    y_t   = g_t · z_t                        # withheld output = negative space
    alpha = sigmoid(a_raw)                   # learned "how little to hold"

  Perception happens by NEGATION (the residual r_t drives the state); SPEECH is
  a separate decision — the surface answers only when it holds returning light
  (energy of z_t), so an internally-carried echo can open the gate while a
  truly empty field keeps it shut.

WHAT THIS FILE DOES WHEN RUN
------------------------------------------------------------------------------
  * Builds the network in pure NumPy (no autograd, no frameworks).
  * Runs a finite-difference gradient check on EVERY parameter (mandatory).
  * Trains it on the "Deer Park echo" task: sparse voices on a silent field;
    a loud voice returns as a delayed, attenuated echo (with a faint cross-art
    bloom); faint voices are absorbed into silence.
  * Prints self-tests: the echo is reproduced, the mirror stays silent on an
    empty field, and the learned non-abiding rate is reported.

Pure standard-library + NumPy. Deterministic seeds → reproducible output.
================================================================================
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# ----------------------------------------------------------------------------
# The Still-Mirror cell
# ----------------------------------------------------------------------------
class StillMirror:
    """A recurrent 'non-abiding reflection' network (see module docstring).

    in_dim  : number of channels (default 3 = word / image / tone)
    hid     : size of the reflecting surface
    out_dim : channels returned (defaults to in_dim: the surface returns echoes)
    """

    def __init__(self, in_dim: int = 3, hid: int = 24,
                 out_dim: int | None = None, seed: int = 0):
        if out_dim is None:
            out_dim = in_dim
        rng = np.random.default_rng(seed)
        s = 0.30
        # Parameters. W_res is scaled small so the empty mountain is the
        # attractor: the reflecting surface does not blow up on its own.
        self.P = {
            'W_in':   rng.standard_normal((hid, in_dim)) * s,
            'W_res':  rng.standard_normal((hid, hid)) * (0.5 / np.sqrt(hid)),
            'W_pred': rng.standard_normal((in_dim, hid)) * s,   # the mirror
            'W_out':  rng.standard_normal((out_dim, hid)) * s,
            'b_out':  np.zeros(out_dim),
            'wg':     np.array([1.0]),    # silence-gate slope
            'bg':     np.array([0.0]),    # silence-gate bias
            'a_raw':  np.array([0.4]),    # -> alpha = sigmoid(a_raw)
        }
        self.in_dim, self.hid, self.out_dim = in_dim, hid, out_dim

    # -- forward pass over a whole sequence, caching everything for BPTT -----
    def forward(self, X: np.ndarray, cache: bool = True):
        """X: (T, in_dim). Returns Y: (T, out_dim) and a cache dict."""
        T = X.shape[0]
        P = self.P
        alpha = sigmoid(P['a_raw'])[0]
        h = np.zeros(self.hid)
        C = {'x': [], 'p': [], 'r': [], 'pre': [], 'th': [],
             'h': [h.copy()], 'z': [], 'e': [], 'g': []}
        Y = np.zeros((T, self.out_dim))
        for t in range(T):
            x = X[t]
            p = P['W_pred'] @ h                       # mirror's prediction
            r = x - p                                  # residual echo (negation)
            pre = P['W_in'] @ r + P['W_res'] @ h
            th = np.tanh(pre)
            h_new = (1.0 - alpha) * h + th             # abide nowhere + reflect
            z = P['W_out'] @ h_new + P['b_out']        # light the surface holds
            e = float(np.sum(z * z))                   # brightness of that light
            g = float(sigmoid(P['wg'][0] * e + P['bg'][0]))
            Y[t] = g * z
            if cache:
                C['x'].append(x); C['p'].append(p); C['r'].append(r)
                C['pre'].append(pre); C['th'].append(th)
                C['h'].append(h_new.copy()); C['z'].append(z)
                C['e'].append(e); C['g'].append(g)
            h = h_new
        C['alpha'] = alpha
        return Y, C

    # -- scalar loss (weighted mean squared echo error) ---------------------
    # W weights each element; the rare 'returning light' steps are worth more
    # than the common silence, so the surface is pushed to actually echo.
    def loss(self, X: np.ndarray, Yt: np.ndarray, W: np.ndarray | None = None):
        Y, C = self.forward(X)
        diff = Y - Yt
        if W is None:
            W = np.ones_like(diff)
        L = 0.5 * float(np.sum(W * diff * diff)) / X.shape[0]
        return L, Y, C, diff, W

    # -- analytic gradient by back-propagation-through-time -----------------
    def backward(self, X: np.ndarray, Yt: np.ndarray, W: np.ndarray | None = None):
        L, Y, C, diff, W = self.loss(X, Yt, W)
        P = self.P
        T = X.shape[0]
        alpha = C['alpha']
        dalpha_da = alpha * (1.0 - alpha)              # d sigmoid / d a_raw
        g = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next = np.zeros(self.hid)                    # gradient flowing back in time

        for t in reversed(range(T)):
            r = C['r'][t]; th = C['th'][t]
            h_new = C['h'][t + 1]; h_prev = C['h'][t]
            z = C['z'][t]; e = C['e'][t]; gt = C['g'][t]

            dy = (W[t] * diff[t]) / T                    # dL/dy_t (weighted)

            # y = g * z ;  g = sigmoid(wg*e + bg) ;  e = sum(z^2)
            # both the gate (through e) and the product feed back into z.
            dg = float(np.sum(dy * z))                  # dL/dg
            dgate = dg * gt * (1.0 - gt)                 # dL/d(wg*e+bg)
            g['wg'][0] += dgate * e
            g['bg'][0] += dgate
            de = dgate * P['wg'][0]                       # dL/de
            dz = dy * gt + de * 2.0 * z                   # product path + gate path

            # z = W_out h_new + b_out
            g['W_out'] += np.outer(dz, h_new)
            g['b_out'] += dz
            dh = P['W_out'].T @ dz

            dr = np.zeros_like(r)                         # gate no longer touches r
            dh = dh + dh_next                            # add future contribution

            # h_new = (1-alpha) h_prev + tanh(pre)
            dalpha = float(np.sum(dh * (-h_prev)))
            g['a_raw'][0] += dalpha * dalpha_da
            dpre = dh * (1.0 - th * th)

            # pre = W_in r + W_res h_prev
            g['W_in'] += np.outer(dpre, r)
            dr = dr + P['W_in'].T @ dpre
            g['W_res'] += np.outer(dpre, h_prev)
            dh_prev = P['W_res'].T @ dpre

            # r = x - p ,  p = W_pred h_prev
            dp = -dr
            g['W_pred'] += np.outer(dp, h_prev)
            dh_prev = dh_prev + P['W_pred'].T @ dp

            # leak path: (1-alpha) h_prev feeds h_new directly
            dh_prev = dh_prev + (1.0 - alpha) * dh
            dh_next = dh_prev

        return L, g


# ----------------------------------------------------------------------------
# The "Deer Park echo" task
# ----------------------------------------------------------------------------
def make_echo_task(rng, T=28, in_dim=3, delay=5, decay=0.55,
                   thresh=0.6, noise=0.03):
    """
    Sparse 'voices' on an otherwise silent field.  A LOUD voice (amplitude >=
    thresh) returns 'delay' steps later as an attenuated echo ('returning
    light'), with a faint bloom in the neighbouring art (synesthesia).  A
    FAINT voice is absorbed by the empty mountain — its target stays silent,
    so the network must learn to withhold (the silence gate).
    Returns X, Y  each of shape (T, in_dim).
    """
    X = np.zeros((T, in_dim)); Y = np.zeros((T, in_dim))
    n = int(rng.integers(2, 4))
    for _ in range(n):
        t = int(rng.integers(0, T - delay - 1))
        c = int(rng.integers(0, in_dim))
        amp = float(rng.uniform(0.3, 1.2))
        X[t, c] += amp
        if amp >= thresh:                              # only loud voices echo
            Y[t + delay, c] += decay * amp
            Y[t + delay, (c + 1) % in_dim] += 0.25 * decay * amp
    X += rng.standard_normal((T, in_dim)) * noise      # the world is not pristine
    return X, Y


# ----------------------------------------------------------------------------
# mandatory finite-difference gradient check
# ----------------------------------------------------------------------------
def gradient_check(verbose=True) -> float:
    rng = np.random.default_rng(1)
    m = StillMirror(3, 8, 3, seed=2)
    X = rng.standard_normal((10, 3)) * 0.5
    Yt = rng.standard_normal((10, 3)) * 0.5
    W = rng.uniform(0.5, 4.0, size=Yt.shape)        # random weights exercise that path
    _, g = m.backward(X, Yt, W)
    eps, maxrel = 1e-6, 0.0
    for k in m.P:
        flat, gflat = m.P[k].ravel(), g[k].ravel()
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + eps; Lp, *_ = m.loss(X, Yt, W)
            flat[i] = o - eps; Lm, *_ = m.loss(X, Yt, W)
            flat[i] = o
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            maxrel = max(maxrel, rel)
    if verbose:
        status = "PASS" if maxrel < 1e-4 else "FAIL"
        print(f"[grad check]   max relative error = {maxrel:.2e}   -> {status}")
    return maxrel


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def clip(grads, c=5.0):
    for k in grads:
        np.clip(grads[k], -c, c, out=grads[k])
    return grads


def echo_weights(Y, base=1.0, emphasis=12.0):
    """Weight the rare 'returning light' steps far above the common silence."""
    return base + emphasis * (np.abs(Y) > 1e-6)


def train(iters=1500, batch=8, lr=0.03, seed=3, verbose=True):
    rng = np.random.default_rng(7)
    m = StillMirror(3, 24, 3, seed=seed)
    # a fixed held-out set so the reported curve is not just batch noise
    eval_rng = np.random.default_rng(999)
    eval_set = [make_echo_task(eval_rng) for _ in range(16)]

    def eval_loss():
        return float(np.mean([m.loss(X, Y, echo_weights(Y))[0]
                              for X, Y in eval_set]))

    first = eval_loss()
    for it in range(iters):
        gsum = {k: np.zeros_like(v) for k, v in m.P.items()}
        for _ in range(batch):
            X, Y = make_echo_task(rng)
            _, g = m.backward(X, Y, echo_weights(Y))
            for k in gsum:
                gsum[k] += g[k]
        clip(gsum)
        for k in m.P:
            m.P[k] -= lr * gsum[k] / batch
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"[train]  iter {it:4d}   eval_loss {eval_loss():.5f}"
                  f"   alpha {sigmoid(m.P['a_raw'])[0]:.3f}")
    last = eval_loss()
    if verbose:
        print(f"[train]  held-out loss  {first:.4f}  ->  {last:.4f}"
              f"   ({first/max(last,1e-9):.1f}x lower)")
    return m, first, last


# ----------------------------------------------------------------------------
# qualitative self-tests
# ----------------------------------------------------------------------------
def self_tests(m):
    print("\n[self-test] empty mountain — a single loud voice, then silence")
    T = 28
    X = np.zeros((T, 3)); X[4, 0] = 1.0                # one loud voice, channel 0
    Y, _ = m.forward(X)
    voice_t = 4                                        # echo target is at t+5
    echo_region = np.abs(Y[voice_t + 3: voice_t + 8]).max()
    far_silence = np.abs(Y[voice_t + 12:]).mean()
    print(f"           peak echo a few steps later : {echo_region:.3f}")
    print(f"           mean output in far silence   : {far_silence:.3f}")

    # a compact glyph view: '.' silence, '-' faint, '#' loud
    def glyph(v):
        a = abs(v)
        return '#' if a > 0.25 else ('-' if a > 0.06 else '.')
    row_in = ''.join(glyph(X[t, 0]) for t in range(T))
    row_out = ''.join(glyph(Y[t, 0]) for t in range(T))
    print("           voice (ch0):  " + row_in)
    print("           echo  (ch0):  " + row_out)

    print("\n[self-test] silence gate — a totally empty field should stay empty")
    Xs = np.zeros((T, 3))
    Ys, _ = m.forward(Xs)
    print(f"           mean |output| on pure silence : {np.mean(np.abs(Ys)):.4f}")

    print(f"\n[self-test] non-abiding rate learned: alpha = "
          f"{sigmoid(m.P['a_raw'])[0]:.3f}  "
          f"(state half-life ~{np.log(0.5)/np.log(1-sigmoid(m.P['a_raw'])[0]):.1f} steps)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 74)
    print("Wang Wei · Still-Mirror Network (non-abiding reflection)")
    print("=" * 74)
    gradient_check()
    print()
    model, l0, l1 = train()
    self_tests(model)
    print("\nDone. The mountain is empty; the echo has returned; the surface is still.")

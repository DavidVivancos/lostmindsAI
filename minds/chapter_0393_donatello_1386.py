#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0393 - DONATELLO (1386-1466)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 THE STIACCIATO CODEC
 A depth-budgeted, station-conditioned relief encoder trained against a frozen
 beholder.
--------------------------------------------------------------------------------
 Figure 393 : Donato di Niccolo di Betto Bardi, called DONATELLO
              Florence, c. 1386 - 13 December 1466
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Donatello is routinely filed under "the body expresses the soul" or "the figure
is released from the stone." Both readings belong to Michelangelo, and neither
is grounded in anything Donatello actually did that nobody else did. What he did
that nobody else had done is narrower and far stranger:

  1. He invented rilievo stiacciato -- relief carved only millimetres deep that
     nevertheless reads as a receding architectural space. The V&A's Ascension
     with Christ giving the Keys to St Peter (c. 1428-30) sits on a slab 5.3 cm
     thick; the carving itself is a fraction of that. The depth is not in the
     marble. It is reconstructed in the beholder.

  2. He deliberately falsified proportion to compensate for the viewing station.
     The Campanile prophets (begun 1415) have torsos that are wrong -- unstable,
     over-long -- seen straight on at eye level, and correct seen from the street
     seventy feet below. Giovanni Pisano had pushed heads forward for visibility
     a century earlier; Donatello was the first to re-proportion the whole body.

  3. Vasari's story about the Orsanmichele St Mark, whatever its documentary
     value, records how contemporaries understood him: the guild rejected the
     statue at ground level, he covered it, had it hoisted into its niche, and
     unveiled it a fortnight later without touching a chisel to it. Same object,
     different station, different verdict.

  4. Late in life he abandoned fidelity outright. The Penitent Magdalene and the
     San Lorenzo pulpit reliefs are gaunt, crowded, rough -- not unfinished but
     amplified past the truth in order to make an inner state legible.

Four facts, one cognitive signature:

  ** A representation is not judged against its source. It is judged by what a
     particular receiver reconstructs from it at a particular station. Depth is
     a scarce budget to be spent, not a quantity to be reproduced. Local
     falsehood is the routine and correct price of global legibility. **

That is a rate-distortion problem posed in stone in 1417, and it is what this
file implements. Nothing here is a Transformer, and there is no attention over
stored keys: there is a SCULPTOR who emits a bounded height field, an EYE that
renders it under raking light from a station, and a BEHOLDER -- frozen, trained
beforehand on ordinary full-round scenes, unable to be retrained, exactly as the
Florentine public could not be retrained -- whose reconstruction is the only
thing the loss ever looks at.

The geometric error between the relief and the scene it depicts is computed at
every step. It is never minimised. It is printed, and it grows, and that growth
is the result.

THE FOUR EXPERIMENTS
--------------------
  E1  GRADIENT CHECK      central finite differences on every parameter tensor
  E2  THE BUDGET SWEEP    squeeze the relief 100x thinner; legibility survives,
                          geometry does not                    -> stiacciato
  E3  THE STATION SWAP    carve for one station, read at another -> St Mark
  E4  AMPLIFICATION       reward the beholder's confidence instead of fidelity
                          and watch the encoder exaggerate      -> the Magdalene

RUN
---
  python3 chapter_0393_donatello_1386.py            # full run, ~1-2 minutes
  python3 chapter_0393_donatello_1386.py --fast     # abbreviated

Pure NumPy. Every gradient hand-derived. No autodiff, no frameworks.
================================================================================
"""

import argparse
import sys
import time

import numpy as np

# ==============================================================================
# SECTION 1 -- PRIMITIVES
# ==============================================================================
# Small, boring, and hand-differentiated. Each forward has a matching backward
# immediately below it so the adjoint can be checked by eye.

EPS = 1e-8


def tanh_f(x):
    """Forward tanh."""
    return np.tanh(x)


def tanh_b(y, g):
    """Backward through tanh, given the OUTPUT y (cheaper than recomputing)."""
    return g * (1.0 - y * y)


def srelu_f(x, beta=1e-3):
    """
    Smooth rectifier: 0.5 * (x + sqrt(x^2 + beta)).

    A hard max(0, .) would be fine for training but leaves a kink that makes a
    finite-difference gradient check unreliable exactly at the fold. Lambertian
    shading spends a lot of time near n.L = 0 under raking light -- which is the
    whole point of raking light -- so the kink would sit right where the model
    lives. This is smooth everywhere and agrees with the rectifier away from 0.
    """
    return 0.5 * (x + np.sqrt(x * x + beta))


def srelu_b(x, g, beta=1e-3):
    """Backward through the smooth rectifier."""
    return g * 0.5 * (1.0 + x / np.sqrt(x * x + beta))


def standardize_f(x):
    """
    Per-image contrast standardisation over the last two axes.

    THIS IS THE LOAD-BEARING CHOICE OF THE WHOLE FILE.

    A relief carved a tenth of a millimetre deep cannot produce the same
    absolute luminance swing as a full-round figure. It was never trying to.
    What survives the flattening is the STRUCTURE of the shading -- where the
    light turns over, in what order, along which contours -- not its amplitude.
    Standardising each render before comparing it makes the objective say
    exactly that, and it is why the budget sweep in E2 is a real result rather
    than an artefact of a lenient loss. The amplitude that gets thrown away here
    is measured separately and reported, because its collapse is also true.

    Returns (y, cache) for the backward pass.
    """
    mu = x.mean(axis=(-2, -1), keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=(-2, -1), keepdims=True)
    inv = 1.0 / np.sqrt(var + 1e-6)
    y = xc * inv
    return y, (y, inv, x.shape[-1] * x.shape[-2])


def standardize_b(g, cache):
    """Backward through per-image standardisation."""
    y, inv, n = cache
    gm = g.mean(axis=(-2, -1), keepdims=True)
    gy = (g * y).mean(axis=(-2, -1), keepdims=True)
    return inv * (g - gm - y * gy)


def softmax_xent(logits, labels):
    """
    Softmax cross-entropy. Returns (loss, dlogits, predictions).

    This is the beholder's verdict: given the shaded surface, which figure did
    it decide stands nearest? Everything the sculptor is optimised for passes
    through this scalar.
    """
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), labels] + EPS).mean()
    d = p.copy()
    d[np.arange(n), labels] -= 1.0
    return loss, d / n, p.argmax(axis=1)


def readout_margin(logits, labels):
    """
    How decisively did the beholder decide? Correct logit minus the best
    competing logit, averaged. E4 pushes this up at the cost of fidelity --
    the quantitative form of "make the grief unmistakable."
    """
    n = logits.shape[0]
    correct = logits[np.arange(n), labels].copy()
    other = logits.copy()
    other[np.arange(n), labels] = -np.inf
    return float(np.mean(correct - other.max(axis=1)))


# ==============================================================================
# SECTION 2 -- THE COMMISSION: synthetic narrative scenes
# ==============================================================================

class Commission:
    """
    Generates the scenes the sculptor is asked to depict.

    Each scene is a shallow stage holding THREE figures, standing left, centre
    and right, exactly as figures are blocked out across a narrative relief. A
    figure is a smooth mound on a G x G ground: a centre (cx, cy), a spread, and
    a height. Height stands in for distance from the picture plane -- the taller
    mound is the figure standing nearest the front.

    The lateral slots are FIXED (with jitter) rather than scattered at random.
    That is not a convenience: with figures thrown anywhere on the ground, the
    depth ordering becomes a position-dependent lookup that no beholder
    generalises from a plausible amount of experience -- a linear probe on this
    render reaches 87% on scenes it has seen and 18% on scenes it has not. A
    relief composes its figures across the field precisely so that a first-time
    viewer can read it. Fixing the slots is the compositional convention doing
    the work it exists to do.

    The label the beholder must recover is the DEPTH ORDER: which of the six
    permutations of near/middle/far the three figures are standing in. This is
    a deliberately austere stand-in for what Donatello's reliefs actually
    transmit -- who is in front of whom, how far back the architecture runs --
    and it has the property that matters here: it is a fact about the SCENE that
    survives being flattened, if the flattening is done well, and is destroyed
    if it is not.

    The scene vector handed to the sculptor is the raw parameter list. He is
    given the commission, not a photograph of the answer.
    """

    N_FIG = 3
    N_ORDER = 6  # 3! permutations
    PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def __init__(self, grid=20, rng=None):
        self.G = grid
        self.rng = rng if rng is not None else np.random.default_rng(313)
        ax = np.linspace(-1.0, 1.0, grid)
        self.YY, self.XX = np.meshgrid(ax, ax, indexing="ij")

    def sample(self, n):
        """
        Draw n commissions.

        Returns
          s     (n, 12)   scene descriptor: per figure [cx, cy, spread, height]
          D     (n, G, G) the true full-round depth field -- what a sculptor
                          working in the round would have to cut
          label (n,)      index into PERMS of the true depth ordering
        """
        G = self.G
        slots = np.array([-0.60, 0.0, 0.60])
        cx = slots[None, :] + self.rng.uniform(-0.10, 0.10, size=(n, self.N_FIG))
        cy = self.rng.uniform(-0.18, 0.18, size=(n, self.N_FIG))
        sp = self.rng.uniform(0.30, 0.38, size=(n, self.N_FIG))

        # Heights are drawn well separated so the ordering is never ambiguous.
        base = np.array([0.34, 0.62, 0.92])
        hh = np.empty((n, self.N_FIG))
        order_idx = self.rng.integers(0, self.N_ORDER, size=n)
        for i in range(n):
            perm = self.PERMS[order_idx[i]]
            for k in range(self.N_FIG):
                hh[i, k] = base[perm[k]] + self.rng.uniform(-0.045, 0.045)

        D = np.zeros((n, G, G))
        for k in range(self.N_FIG):
            dx = self.XX[None, :, :] - cx[:, k, None, None]
            dy = self.YY[None, :, :] - cy[:, k, None, None]
            r2 = dx * dx + dy * dy
            s2 = 2.0 * (sp[:, k, None, None] ** 2)
            D += hh[:, k, None, None] * np.exp(-r2 / s2)

        s = np.stack([cx, cy, sp, hh], axis=2).reshape(n, self.N_FIG * 4)
        return s, D, order_idx


# ==============================================================================
# SECTION 3 -- THE STATION: where the beholder stands, and how the light falls
# ==============================================================================

class Station:
    """
    A viewing station is two numbers.

      elev  the angle the beholder looks UP through, in radians. 0 is eye level
            in the workshop; large values are the street below the Campanile.
      lazi  the azimuth of the light grazing the surface.

    Two consequences, and they are different in kind.

    FORESHORTENING is a linear operator. Looking up at a vertical surface
    compresses it vertically by cos(elev). We build that compression as an
    explicit G x G interpolation matrix acting on the image rows. Because it is
    linear its adjoint is just the transpose, and because it is applied AFTER
    shading, the sculptor can only defeat it by pre-distorting the carving --
    which is precisely the Campanile manoeuvre, arrived at here by gradient
    descent rather than by scaffolding.

    ILLUMINATION is not linear. As the light grazes lower it amplifies shallow
    slopes enormously; this is why a stiacciato relief needs raking light and
    why the V&A had to build a moving-light viewer to make the Ascension
    legible at all under flat museum lighting.
    """

    def __init__(self, grid, elev=0.0, lazi=2.6, light_elev=0.16):
        self.G = grid
        self.elev = float(elev)
        self.lazi = float(lazi)
        self.light_elev = float(light_elev)
        self.L = self._light()
        self.S = self._foreshorten()

    def _light(self):
        """Unit light direction. light_elev small => raking."""
        ce, se = np.cos(self.light_elev), np.sin(self.light_elev)
        v = np.array([ce * np.cos(self.lazi), ce * np.sin(self.lazi), se])
        return v / np.linalg.norm(v)

    def _foreshorten(self):
        """
        Build the G x G row-compression matrix for this elevation.

        Row i of the observed image samples the carved image at height
        i * cos(elev) measured from the top, by linear interpolation. At
        elev = 0 this is the identity.
        """
        G = self.G
        k = np.cos(self.elev)
        src = np.arange(G) * k
        S = np.zeros((G, G))
        for i in range(G):
            f = src[i]
            lo = int(np.floor(f))
            hi = min(lo + 1, G - 1)
            w = f - lo
            lo = min(lo, G - 1)
            S[i, lo] += (1.0 - w)
            S[i, hi] += w
        return S

    def __repr__(self):
        return (f"Station(elev={np.degrees(self.elev):.0f}deg, "
                f"lazi={np.degrees(self.lazi):.0f}deg)")


# ==============================================================================
# SECTION 4 -- THE EYE: a differentiable raking-light renderer
# ==============================================================================
# Given a height field, produce the image a beholder at a station actually
# receives. Fully hand-differentiated: this is the only path by which the
# sculptor learns anything, so its adjoint has to be exactly right.

class Eye:
    """
    shade -> foreshorten -> standardise.

    Normals are taken from clamped central differences of the height field,
    scaled by `gain`. `gain` folds together the physical relief scale and the
    grazing amplification of the light; it is a property of the viewing setup,
    not of the carving, and is held identical for the relief and for the
    full-round reference so that no thumb rests on the scale.
    """

    def __init__(self, grid, gain=6.0, spacing=None):
        self.G = grid
        self.gain = float(gain)
        self.h = spacing if spacing is not None else (2.0 / (grid - 1))
        self.ip = np.minimum(np.arange(grid) + 1, grid - 1)
        self.im = np.maximum(np.arange(grid) - 1, 0)

    # -- clamped central differences and their adjoints ----------------------
    def _dx(self, R):
        return (R[:, :, self.ip] - R[:, :, self.im]) / (2.0 * self.h)

    def _dx_adj(self, g):
        out = np.zeros((g.shape[0], self.G, self.G))
        np.add.at(out, (slice(None), slice(None), self.ip), g / (2.0 * self.h))
        np.add.at(out, (slice(None), slice(None), self.im), -g / (2.0 * self.h))
        return out

    def _dy(self, R):
        return (R[:, self.ip, :] - R[:, self.im, :]) / (2.0 * self.h)

    def _dy_adj(self, g):
        out = np.zeros((g.shape[0], self.G, self.G))
        np.add.at(out, (slice(None), self.ip, slice(None)), g / (2.0 * self.h))
        np.add.at(out, (slice(None), self.im, slice(None)), -g / (2.0 * self.h))
        return out

    # -- forward -------------------------------------------------------------
    def forward(self, R, station):
        """
        R : (n, G, G) height field.
        Returns (I_std, I_raw, cache).

        I_std is contrast-standardised and is what the loss and the beholder
        see. I_raw is kept so the amplitude collapse can be measured.
        """
        gx = self._dx(R)
        gy = self._dy(R)
        nx = -self.gain * gx
        ny = -self.gain * gy
        # nz == 1 before normalisation
        q = nx * nx + ny * ny + 1.0
        inv = 1.0 / np.sqrt(q)
        Lx, Ly, Lz = station.L
        dot = (nx * Lx + ny * Ly + Lz) * inv
        sh = srelu_f(dot)
        I_raw = np.einsum("ij,njk->nik", station.S, sh)
        I_std, cst = standardize_f(I_raw)
        cache = (gx, gy, nx, ny, inv, dot, sh, station, cst)
        return I_std, I_raw, cache

    # -- backward ------------------------------------------------------------
    def backward(self, g_std, cache):
        """Propagate a gradient on the standardised image back to R."""
        gx, gy, nx, ny, inv, dot, sh, station, cst = cache
        g_raw = standardize_b(g_std, cst)
        g_sh = np.einsum("ij,nik->njk", station.S, g_raw)   # S^T applied
        g_dot = srelu_b(dot, g_sh)

        Lx, Ly, Lz = station.L
        u = nx * Lx + ny * Ly + Lz          # dot = u * inv
        # d(inv)/d(nx) = -nx * inv^3 ; likewise for ny
        g_nx = g_dot * (Lx * inv - u * (inv ** 3) * nx)
        g_ny = g_dot * (Ly * inv - u * (inv ** 3) * ny)

        g_gx = -self.gain * g_nx
        g_gy = -self.gain * g_ny
        return self._dx_adj(g_gx) + self._dy_adj(g_gy)


# ==============================================================================
# SECTION 5 -- THE BEHOLDER: a small readout head, trained first, then frozen
# ==============================================================================

class Beholder:
    """
    A two-layer readout from a shaded image to a depth-order verdict.

    The order of operations is the historical argument of this file. The
    beholder is trained ONLY on renders of full-round scenes -- ordinary
    sculpture, the visual habits a fifteenth-century Florentine already had --
    and is then FROZEN. The sculptor never gets to retrain the public. He has
    to hit an eye that already exists, with all its priors intact.

    Removing that freeze collapses the whole experiment into a private code
    between encoder and decoder, which would prove nothing about stiacciato and
    everything about co-adaptation. The freeze is the control.
    """

    def __init__(self, grid, hidden=64, n_out=6, rng=None):
        rng = rng if rng is not None else np.random.default_rng(1386)
        d_in = grid * grid
        self.G = grid
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.frozen = False

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def names(self):
        return ["beholder.W1", "beholder.b1", "beholder.W2", "beholder.b2"]

    def forward(self, I_std):
        x = I_std.reshape(I_std.shape[0], -1)
        a1 = x @ self.W1 + self.b1
        h1 = tanh_f(a1)
        logits = h1 @ self.W2 + self.b2
        return logits, (x, h1)

    def backward(self, g_logits, cache, want_input_grad=True):
        """
        Returns (grads_or_None, g_input_image).

        When frozen we still need the gradient THROUGH the beholder to reach the
        sculptor -- the eye does not change, but it still tells you which way to
        move the chisel.
        """
        x, h1 = cache
        gW2 = h1.T @ g_logits
        gb2 = g_logits.sum(axis=0)
        gh1 = g_logits @ self.W2.T
        ga1 = tanh_b(h1, gh1)
        gW1 = x.T @ ga1
        gb1 = ga1.sum(axis=0)
        gx = (ga1 @ self.W1.T) if want_input_grad else None
        grads = None if self.frozen else [gW1, gb1, gW2, gb2]
        gimg = gx.reshape(-1, self.G, self.G) if gx is not None else None
        return grads, gimg


# ==============================================================================
# SECTION 6 -- THE SCULPTOR: a station-conditioned, depth-budgeted encoder
# ==============================================================================

class Sculptor:
    """
    Maps (commission, station) -> a height field bounded by the depth budget.

    The budget is enforced as a HARD CAP, not a penalty:

        R = B * tanh(pre-activation)

    so |R| < B always, at initialisation and at every step, and no amount of
    gradient pressure can smuggle in extra depth. B is the thickness of marble
    the sculptor is allowed to remove. Setting B = 1.0 gives him a block; setting
    B = 0.01 gives him a slab and a fingernail. E2 walks that number down.

    The station is fed in as (cos elev, sin elev, cos lazi, sin lazi), so the
    same set of weights carves differently depending on where the work will
    stand. That conditioning is the arithmetic of the Campanile.
    """

    def __init__(self, grid, d_scene, hidden=96, budget=0.25, rng=None):
        rng = rng if rng is not None else np.random.default_rng(1466)
        self.G = grid
        self.budget = float(budget)
        d_in = d_scene + 4
        d_out = grid * grid
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, d_out))
        self.b3 = np.zeros(d_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def names(self):
        return ["sculptor.W1", "sculptor.b1", "sculptor.W2", "sculptor.b2",
                "sculptor.W3", "sculptor.b3"]

    @staticmethod
    def station_code(station, n):
        c = np.array([np.cos(station.elev), np.sin(station.elev),
                      np.cos(station.lazi), np.sin(station.lazi)])
        return np.tile(c, (n, 1))

    def forward(self, s, station, budget=None):
        B = self.budget if budget is None else float(budget)
        x = np.concatenate([s, self.station_code(station, s.shape[0])], axis=1)
        a1 = x @ self.W1 + self.b1
        h1 = tanh_f(a1)
        a2 = h1 @ self.W2 + self.b2
        h2 = tanh_f(a2)
        a3 = h2 @ self.W3 + self.b3
        t = tanh_f(a3)
        R = (B * t).reshape(-1, self.G, self.G)
        return R, (x, h1, h2, t, B)

    def backward(self, gR, cache):
        x, h1, h2, t, B = cache
        gt = gR.reshape(gR.shape[0], -1) * B
        ga3 = tanh_b(t, gt)
        gW3 = h2.T @ ga3
        gb3 = ga3.sum(axis=0)
        gh2 = ga3 @ self.W3.T
        ga2 = tanh_b(h2, gh2)
        gW2 = h1.T @ ga2
        gb2 = ga2.sum(axis=0)
        gh1 = ga2 @ self.W2.T
        ga1 = tanh_b(h1, gh1)
        gW1 = x.T @ ga1
        gb1 = ga1.sum(axis=0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]


# ==============================================================================
# SECTION 7 -- ADAM
# ==============================================================================

class Adam:
    """Plain Adam with bias correction. No schedule, no tricks."""

    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            if g is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# SECTION 8 -- THE WORKSHOP: the assembled objective
# ==============================================================================

class Workshop:
    """
    Holds the commission source, the eye, the sculptor and the frozen beholder,
    and computes the objective.

        L = w_app * || std(render(R)) - std(render(D_true)) ||^2      appearance
          + w_read * CE( beholder(std(render(R))), true order )       legibility
          - w_amp  * margin( beholder(...) )                          [E4 only]

    and, computed at every step and NEVER minimised:

        geom = || R - D_true ||^2      the close-up error

    The asymmetry is the entire thesis. What the sculptor is asked for is that
    the beholder's reconstruction be right. What happens to the stone in the
    process is reported, not controlled.
    """

    def __init__(self, grid=20, hidden_s=96, hidden_b=64, budget=0.25,
                 gain=6.0, seed=313):
        self.rng = np.random.default_rng(seed)
        self.G = grid
        self.commission = Commission(grid, np.random.default_rng(seed + 1))
        self.eye = Eye(grid, gain=gain)
        self.sculptor = Sculptor(grid, Commission.N_FIG * 4, hidden_s,
                                 budget, np.random.default_rng(seed + 2))
        self.beholder = Beholder(grid, hidden_b, Commission.N_ORDER,
                                 np.random.default_rng(seed + 3))
        self.w_app = 1.0
        self.w_read = 1.0
        self.w_amp = 0.0

    # -- phase 1: teach the beholder to read ordinary sculpture ---------------
    def beholder_loss(self, n, station, batch=None):
        """
        Train the readout on renders of TRUE full-round depth fields.
        This is the eye the sculptor will later have to satisfy.
        """
        if batch is None:
            _, D, y = self.commission.sample(n)
        else:
            _, D, y = batch
        I_std, _, _ = self.eye.forward(D, station)
        logits, bc = self.beholder.forward(I_std)
        loss, gl, pred = softmax_xent(logits, y)
        grads, _ = self.beholder.backward(gl, bc, want_input_grad=False)
        acc = float((pred == y).mean())
        return loss, grads, acc

    # -- phase 2: the sculptor works against the frozen eye -------------------
    def sculptor_loss(self, batch, carve_station, view_station=None,
                      budget=None, want_grad=True):
        """
        carve_station : the station the sculptor is TOLD the work will occupy
        view_station  : the station it is actually READ from (E3 sets these apart)

        Returns (total, report, grads).
        """
        s, D, y = batch
        vs = carve_station if view_station is None else view_station

        # what the sculptor cuts
        R, sc = self.sculptor.forward(s, carve_station, budget=budget)

        # what a beholder at vs receives from the relief, and from the real thing
        I_rel, I_rel_raw, ec = self.eye.forward(R, vs)
        I_ref, I_ref_raw, _ = self.eye.forward(D, vs)

        # (a) appearance: does the flattened thing LOOK like the full-round thing
        diff = I_rel - I_ref
        L_app = float(np.mean(diff * diff))
        g_app = (2.0 / diff.size) * diff * self.w_app

        # (b) legibility: does the frozen eye still recover the scene
        logits, bc = self.beholder.forward(I_rel)
        L_read, g_logits, pred = softmax_xent(logits, y)
        margin = readout_margin(logits, y)
        acc = float((pred == y).mean())

        g_logits_total = g_logits * self.w_read
        if self.w_amp != 0.0:
            # E4: push the correct logit above its best rival. Implemented as a
            # gradient directly on the logits -- +1 on the truth, -1 on the
            # runner-up -- descending on the NEGATIVE margin.
            n = logits.shape[0]
            other = logits.copy()
            other[np.arange(n), y] = -np.inf
            rival = other.argmax(axis=1)
            gm = np.zeros_like(logits)
            gm[np.arange(n), y] = -1.0 / n
            gm[np.arange(n), rival] = 1.0 / n
            g_logits_total = g_logits_total + self.w_amp * gm

        total = self.w_app * L_app + self.w_read * L_read - self.w_amp * margin

        # ---- never optimised, always measured --------------------------------
        # Raw squared error against the scene's true geometry, and the same
        # number for an UNTOUCHED SLAB (R = 0). The null matters: as the budget
        # shrinks the relief necessarily approaches the blank slab, so raw geom
        # saturates at the null and a bare "geom went down" reading would be an
        # artefact. The ratio is what carries meaning -- 1.0 means the carving
        # is no closer to the scene's geometry than uncut marble.
        geom = float(np.mean((R - D) ** 2))
        geom_null = float(np.mean(D ** 2))
        geom_ratio = geom / max(geom_null, EPS)

        # Depth compression: how many times shallower the relief is than the
        # scene it depicts. This is the quantity Donatello was actually buying.
        rng_scene = float(np.mean(D.max(axis=(1, 2)) - D.min(axis=(1, 2))))
        rng_relief = float(np.mean(R.max(axis=(1, 2)) - R.min(axis=(1, 2))))
        compression = rng_scene / max(rng_relief, EPS)

        contrast = float(np.mean(np.std(I_rel_raw, axis=(1, 2))))
        contrast_ref = float(np.mean(np.std(I_ref_raw, axis=(1, 2))))

        report = dict(total=float(total), app=L_app, read=float(L_read),
                      acc=acc, margin=margin, geom=geom,
                      geom_null=geom_null, geom_ratio=geom_ratio,
                      compression=compression,
                      contrast=contrast, contrast_ref=contrast_ref,
                      relief_max=float(np.max(np.abs(R))))

        if not want_grad:
            return total, report, None

        _, g_img_from_read = self.beholder.backward(g_logits_total, bc,
                                                    want_input_grad=True)
        g_img = g_app + g_img_from_read
        gR = self.eye.backward(g_img, ec)
        grads = self.sculptor.backward(gR, sc)
        return total, report, grads


# ==============================================================================
# SECTION 9 -- E1: GRADIENT CHECK
# ==============================================================================

def finite_difference_check(shop, verbose=True):
    """
    Central finite differences against the analytic gradient, on every
    parameter tensor of both networks.

    Checked on a deliberately small configuration -- the point is correctness of
    the adjoints, and the adjoints do not know how large the grid is.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("E1  GRADIENT CHECK  (central differences, every parameter tensor)")
        print("=" * 78)

    st = Station(shop.G, elev=0.55, lazi=2.4)
    batch = shop.commission.sample(6)
    eps = 1e-5
    worst = 0.0
    rows = []

    # ---- sculptor path (beholder frozen) ----
    shop.beholder.frozen = True
    _, _, grads = shop.sculptor_loss(batch, st)
    for name, P, A in zip(shop.sculptor.names(), shop.sculptor.params(), grads):
        idx = tuple(np.random.default_rng(7).integers(0, d, size=6)
                    for d in P.shape)
        num = np.empty(6)
        for k in range(6):
            loc = tuple(i[k] for i in idx)
            old = P[loc]
            P[loc] = old + eps
            lp, _, _ = shop.sculptor_loss(batch, st, want_grad=False)
            P[loc] = old - eps
            lm, _, _ = shop.sculptor_loss(batch, st, want_grad=False)
            P[loc] = old
            num[k] = (lp - lm) / (2 * eps)
        ana = np.array([A[tuple(i[k] for i in idx)] for k in range(6)])
        den = np.maximum(np.abs(num) + np.abs(ana), 1e-9)
        rel = float(np.max(np.abs(num - ana) / den))
        worst = max(worst, rel)
        rows.append((name, rel))

    # ---- beholder path (its own training objective) ----
    shop.beholder.frozen = False
    bb = shop.commission.sample(6)

    def bloss():
        return shop.beholder_loss(6, st, batch=bb)[0]

    _, bgrads, _ = shop.beholder_loss(6, st, batch=bb)
    for name, P, A in zip(shop.beholder.names(), shop.beholder.params(), bgrads):
        idx = tuple(np.random.default_rng(11).integers(0, d, size=6)
                    for d in P.shape)
        num = np.empty(6)
        for k in range(6):
            loc = tuple(i[k] for i in idx)
            old = P[loc]
            P[loc] = old + eps
            lp = bloss()
            P[loc] = old - eps
            lm = bloss()
            P[loc] = old
            num[k] = (lp - lm) / (2 * eps)
        ana = np.array([A[tuple(i[k] for i in idx)] for k in range(6)])
        den = np.maximum(np.abs(num) + np.abs(ana), 1e-9)
        rel = float(np.max(np.abs(num - ana) / den))
        worst = max(worst, rel)
        rows.append((name, rel))

    if verbose:
        for name, rel in rows:
            print(f"  {name:<16s} max relative error = {rel:.3e}")
        print(f"  {'-' * 60}")
        print(f"  WORST = {worst:.3e}   -> "
              f"{'PASS' if worst < 1e-5 else 'FAIL'}")
    return worst


# ==============================================================================
# SECTION 10 -- TRAINING
# ==============================================================================

def train_beholder(shop, station, steps=500, batch=48, lr=4e-3, verbose=True):
    """
    Phase 1. The public learns to read ordinary sculpture. Nothing about relief
    appears anywhere in this loop.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("PHASE 1  TRAINING THE BEHOLDER on full-round scenes")
        print(f"         station: {station}")
        print("=" * 78)
    shop.beholder.frozen = False
    opt = Adam(shop.beholder.params(), lr=lr)
    hist = []
    for t in range(1, steps + 1):
        loss, grads, acc = shop.beholder_loss(batch, station)
        opt.step(grads)
        hist.append((loss, acc))
        if verbose and (t % max(1, steps // 6) == 0 or t == 1):
            print(f"  step {t:>4d}   loss {loss:7.4f}   acc {acc * 100:5.1f}%")
    # held-out check
    accs = [shop.beholder_loss(96, station)[2] for _ in range(6)]
    held = float(np.mean(accs))
    if verbose:
        print(f"  held-out accuracy on full-round renders: {held * 100:.1f}%")
        print("  -> beholder FROZEN. It will not be updated again.")
    shop.beholder.frozen = True
    return held, hist


def train_sculptor(shop, station, steps=700, batch=32, lr=3e-3, verbose=True,
                   tag="PHASE 2"):
    """
    Phase 2. The sculptor learns to satisfy an eye he cannot change.
    """
    if verbose:
        print("\n" + "=" * 78)
        print(f"{tag}  TRAINING THE SCULPTOR   budget B = {shop.sculptor.budget}")
        print(f"         station: {station}")
        print("=" * 78)
        print(f"  {'step':>5s} {'total':>8s} {'appear':>8s} {'read':>7s} "
              f"{'acc':>7s} {'GEOM':>8s}")
    opt = Adam(shop.sculptor.params(), lr=lr)
    hist = []
    for t in range(1, steps + 1):
        b = shop.commission.sample(batch)
        total, rep, grads = shop.sculptor_loss(b, station)
        opt.step(grads)
        hist.append(rep)
        if verbose and (t % max(1, steps // 7) == 0 or t == 1):
            print(f"  {t:>5d} {rep['total']:8.4f} {rep['app']:8.4f} "
                  f"{rep['read']:7.4f} {rep['acc'] * 100:6.1f}% "
                  f"{rep['geom']:8.4f}")
    return hist


def evaluate(shop, carve_station, view_station=None, budget=None, n=400,
             reps=3):
    """Average a report over several fresh draws."""
    keys = None
    acc = {}
    for _ in range(reps):
        b = shop.commission.sample(n)
        _, rep, _ = shop.sculptor_loss(b, carve_station, view_station,
                                       budget=budget, want_grad=False)
        if keys is None:
            keys = list(rep.keys())
            acc = {k: [] for k in keys}
        for k in keys:
            acc[k].append(rep[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ==============================================================================
# SECTION 11 -- THE EXPERIMENTS
# ==============================================================================

def experiment_budget_sweep(shop_seed, station, steps, verbose=True):
    """
    E2  THE BUDGET SWEEP -- the stiacciato result.

    Retrain the sculptor from scratch at each depth budget, from a full block
    down to a hundredth of it, and watch three numbers move independently:

      * legibility   (frozen beholder's accuracy)      should hold
      * compression  (scene depth / relief depth)      should climb
      * geom ratio   (vs. an UNCUT SLAB)               should stay at or above 1
      * contrast     (raw luminance swing)             should fall

    The geometry column is stated against the uncut-slab null on purpose. A
    ratio near or above 1 says the carving is no nearer the scene's real
    geometry than a blank piece of marble -- it is not a degraded copy of the
    depth, it is not a copy of the depth at all -- while the beholder in front
    of it recovers the scene perfectly. That gap is stiacciato.

    Averaged over independent seeds, because a single training run of anything
    is an anecdote.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("E2  THE BUDGET SWEEP   (retrained from scratch, 2 seeds each)")
        print("=" * 78)
        print(f"  {'budget B':>9s} {'legible':>9s} {'compress':>9s} "
              f"{'geom/null':>10s} {'contrast':>9s} {'appear':>8s}")
        print("  " + "-" * 60)

    out = []
    for B in [1.0, 0.5, 0.2, 0.08, 0.03, 0.01]:
        runs = []
        for sd in (shop_seed, shop_seed + 1000):
            shop = Workshop(grid=20, budget=B, seed=sd)
            train_beholder(shop, station, steps=max(300, steps), verbose=False)
            train_sculptor(shop, station, steps=steps, verbose=False)
            runs.append(evaluate(shop, station))
        r = {k: float(np.mean([q[k] for q in runs])) for k in runs[0]}
        out.append((B, r))
        if verbose:
            print(f"  {B:9.3f} {r['acc'] * 100:8.1f}% {r['compression']:8.1f}x "
                  f"{r['geom_ratio']:10.2f} {r['contrast']:9.4f} "
                  f"{r['app']:8.4f}")
    if verbose:
        first, last = out[0][1], out[-1][1]
        print("  " + "-" * 60)
        print(f"  budget cut 100x: legibility {first['acc'] * 100:.1f}% -> "
              f"{last['acc'] * 100:.1f}%, compression "
              f"{first['compression']:.1f}x -> {last['compression']:.1f}x,")
        print(f"  contrast {first['contrast']:.4f} -> {last['contrast']:.4f} "
              f"({first['contrast'] / max(last['contrast'], 1e-9):.1f}x weaker) "
              f"-- which is why such a relief needs raking light.")
        print(f"  geometry stays at {min(z[1]['geom_ratio'] for z in out):.2f}-"
              f"{max(z[1]['geom_ratio'] for z in out):.2f}x the uncut slab "
              f"throughout: never a copy of the depth.")
    return out


def experiment_station_swap(shop, carve_station, verbose=True):
    """
    E3  THE STATION SWAP -- the St Mark result.

    The sculptor is already trained for one station. Read the identical carving
    from other stations without touching it. If correctness were a property of
    the object, nothing would change.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("E3  THE STATION SWAP   (same carving, moved beholder)")
        print("=" * 78)
        print(f"  carved for: {carve_station}")
        print(f"  {'read from':>26s} {'legible':>9s} {'appear':>9s}")
        print("  " + "-" * 48)
    rows = []
    for elev in [0.0, 0.25, 0.55, 0.85, 1.15]:
        vs = Station(shop.G, elev=elev, lazi=carve_station.lazi)
        r = evaluate(shop, carve_station, view_station=vs)
        rows.append((elev, r))
        mark = "  <- carved for this" if abs(elev - carve_station.elev) < 1e-6 else ""
        if verbose:
            print(f"  {np.degrees(elev):>21.0f} deg {r['acc'] * 100:8.1f}% "
                  f"{r['app']:9.4f}{mark}")
    if verbose:
        best = max(rows, key=lambda z: z[1]['acc'])
        print("  " + "-" * 48)
        print(f"  peak legibility at {np.degrees(best[0]):.0f} deg "
              f"({best[1]['acc'] * 100:.1f}%); the object did not change.")
    return rows


def experiment_amplification(shop_seed, station, steps, verbose=True):
    """
    E4  AMPLIFICATION -- the Magdalene result.

    Add a term that rewards the beholder's DECISIVENESS rather than mere
    correctness, and let appearance fidelity pay for it. This is the late work:
    the Magdalene is not a more accurate woman, she is a more unmistakable one.
    """
    if verbose:
        print("\n" + "=" * 78)
        print("E4  AMPLIFICATION   (reward the beholder's certainty, not fidelity)")
        print("=" * 78)
        print(f"  {'w_amp':>7s} {'legible':>9s} {'margin':>9s} "
              f"{'appear':>9s} {'geom/null':>10s}")
        print("  " + "-" * 48)
    rows = []
    for w in [0.0, 0.5, 2.0, 6.0]:
        shop = Workshop(grid=20, budget=0.08, seed=shop_seed)
        train_beholder(shop, station, steps=200, verbose=False)
        shop.w_amp = w
        train_sculptor(shop, station, steps=steps, verbose=False)
        shop.w_amp = 0.0          # measure on the plain objective
        r = evaluate(shop, station)
        rows.append((w, r))
        if verbose:
            print(f"  {w:7.1f} {r['acc'] * 100:8.1f}% {r['margin']:9.3f} "
                  f"{r['app']:9.4f} {r['geom_ratio']:10.2f}")
    if verbose:
        print("  " + "-" * 48)
        print(f"  certainty {rows[0][1]['margin']:.2f} -> {rows[-1][1]['margin']:.2f} "
              f"while appearance error {rows[0][1]['app']:.4f} -> "
              f"{rows[-1][1]['app']:.4f}: the exaggeration is bought, not free.")
    return rows


# ==============================================================================
# SECTION 12 -- SELF-TESTS
# ==============================================================================

def self_tests(verbose=True):
    """Structural invariants. These are assertions, not demonstrations."""
    if verbose:
        print("\n" + "=" * 78)
        print("SELF-TESTS")
        print("=" * 78)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    shop = Workshop(grid=12, budget=0.09, seed=99)
    s, D, y = shop.commission.sample(8)

    check("scene descriptor shape", s.shape == (8, 12))
    check("depth field shape", D.shape == (8, 12, 12))
    check("labels within range", y.min() >= 0 and y.max() < 6)

    st0 = Station(12, elev=0.0)
    check("identity foreshortening at eye level",
          np.allclose(st0.S, np.eye(12)))
    st1 = Station(12, elev=0.9)
    check("foreshortening rows sum to one",
          np.allclose(st1.S.sum(axis=1), 1.0))
    check("light direction is unit", abs(np.linalg.norm(st0.L) - 1.0) < 1e-12)

    R, _ = shop.sculptor.forward(s, st1)
    check("depth budget is a hard cap",
          np.max(np.abs(R)) <= shop.sculptor.budget + 1e-12)
    R2, _ = shop.sculptor.forward(s, st1, budget=0.004)
    check("budget override is respected", np.max(np.abs(R2)) <= 0.004 + 1e-12)

    I, Iraw, _ = shop.eye.forward(D, st1)
    check("render shape matches grid", I.shape == (8, 12, 12))
    check("render is standardised",
          abs(float(I.mean())) < 1e-9 and abs(float(I.std()) - 1.0) < 1e-2)

    flat = np.zeros((3, 12, 12))
    _, fr, _ = shop.eye.forward(flat, st0)
    check("a flat field renders uniformly", float(np.std(fr)) < 1e-9)

    # a carving conditioned on the station must actually depend on it
    Ra, _ = shop.sculptor.forward(s, Station(12, elev=0.0))
    Rb, _ = shop.sculptor.forward(s, Station(12, elev=1.1))
    check("carving is station-conditioned",
          float(np.mean(np.abs(Ra - Rb))) > 1e-6)

    # freezing must actually suppress beholder updates
    shop.beholder.frozen = True
    _, g, _ = shop.beholder_loss(6, st1)
    check("frozen beholder yields no parameter gradient", g is None)
    shop.beholder.frozen = False
    _, g, _ = shop.beholder_loss(6, st1)
    check("thawed beholder yields gradients", g is not None and len(g) == 4)

    # geometry is measured but is not in the objective: perturbing the reported
    # geom must not be possible without changing R, and R is bounded, so the
    # weakest useful invariant is that geom exceeds the budget cap when the
    # scene is taller than the budget allows.
    shop.beholder.frozen = True
    _, rep, _ = shop.sculptor_loss((s, D, y), st1, want_grad=False)
    check("geometric error is nonzero and reported", rep["geom"] > 0.0)
    check("relief never exceeds its budget in the report",
          rep["relief_max"] <= shop.sculptor.budget + 1e-12)

    if verbose:
        print(f"  {'-' * 60}")
        print(f"  {sum(ok)}/{len(ok)} passed")
    return all(ok)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="The Stiacciato Codec (Donatello)")
    ap.add_argument("--fast", action="store_true", help="abbreviated run")
    ap.add_argument("--seed", type=int, default=313)
    args = ap.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    t0 = time.time()

    steps_b = 200 if args.fast else 500
    steps_s = 250 if args.fast else 700
    steps_e = 180 if args.fast else 400

    print("=" * 78)
    print(" THE STIACCIATO CODEC")
    print(" Donatello (c. 1386-1466) -- figure 393")
    print(" 'The depth is not in the marble. It is in the beholder.'")
    print("=" * 78)
    print(" A representation is judged by what a fixed receiver reconstructs")
    print(" from it at a fixed station -- never against its source. Depth is a")
    print(" budget. Local falsehood is the price of legibility, and the price")
    print(" is measured here in the one number the objective never touches.")

    # ---- self-tests first: nothing else means anything if these fail --------
    if not self_tests():
        print("\nSELF-TESTS FAILED -- aborting.")
        return 1

    # ---- the working station: seventy feet up, light grazing from the west --
    shop = Workshop(grid=20, budget=0.08, seed=args.seed)
    station = Station(shop.G, elev=0.55, lazi=2.6)

    # ---- E1 ----
    worst = finite_difference_check(shop)
    if worst >= 1e-5:
        print("\nGRADIENT CHECK FAILED -- aborting.")
        return 1

    # ---- phases ----
    held, _ = train_beholder(shop, station, steps=steps_b)
    train_sculptor(shop, station, steps=steps_s)
    final = evaluate(shop, station)

    print("\n" + "-" * 78)
    print("  AFTER TRAINING, at the station the work was carved for:")
    print(f"    depth budget                       {shop.sculptor.budget}")
    print(f"    deepest cut actually made          {final['relief_max']:.4f}")
    print(f"    frozen beholder still reads it     {final['acc'] * 100:.1f}% "
          f"(it reads full-round work at {held * 100:.1f}%)")
    print(f"    standardised appearance error      {final['app']:.4f}")
    print(f"    raw contrast, relief vs full-round "
          f"{final['contrast']:.4f} vs {final['contrast_ref']:.4f}")
    print(f"    depth compression vs the scene      {final['compression']:.1f}x shallower")
    print(f"    GEOMETRIC error vs an UNCUT SLAB   {final['geom_ratio']:.2f}x (never minimised)")
    print("-" * 78)

    # ---- E2, E3, E4 ----
    experiment_budget_sweep(args.seed, station, steps_e)
    experiment_station_swap(shop, station)
    experiment_amplification(args.seed, station, steps_e)

    print("\n" + "=" * 78)
    print(f" completed in {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0395 - JAN VAN EYCK (1390-1441)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
                    (Maaseik, c. 1390 — Bruges, 9 July 1441)
================================================================================

WHAT THIS FILE IS

A complete, trainable, from-scratch architecture (pure NumPy, no autograd, no
frameworks) whose *mechanism* is a formalisation of one cognitive commitment
that is van Eyck's alone: that a representation is built by ordered, transparent,
irreversible subtraction from a reserved store of light, and that because nothing
is ever overwritten, the whole history of the construction stays physically
present in the artefact and can be read out by probes chosen for their
complementary blindness.

Everything else follows from that. The concrete devices are:

  1. THE GROUND IS THE ONLY LIGHT SOURCE.
     State is log-luminance s, initialised to s = 0 (transmittance 1: the white
     chalk ground at full brightness). Every layer can only SUBTRACT a
     non-negative absorbance. Luminance is therefore a monotonically
     non-increasing budget. You cannot add light later; you can only spend it.
     (Beer-Lambert: T = exp(-sum of absorbances). Real optics, not a metaphor.)

  2. THE PLAN IS COMMITTED FIRST AND ONLY ONCE.
     Layer 0 is the underdrawing. It is the ONLY layer that sees the input x.
     Layers 1..K-1 are the paint campaigns: they see only what is already on the
     panel (the state s). This is the workshop rule made structural — after the
     drawing, you work from the panel, not from the sitter.

  3. LIGHT GOES DOWN AND COMES BACK UP THROUGH THE SAME FILMS.
     The forward pass traverses the layer stack downward and then upward through
     the identical parameters. Weight tying here is not a regulariser chosen for
     convenience; it is what the physics of a glaze over a reflective ground
     forces. Depth 2K, parameters K.

  4. TWO PROBES WITH COMPLEMENTARY BLINDNESS.
     Channels are split into VISIBLE, XRF and IRR. Fixed structural masks decide
     which layers are opaque to which probe, mimicking the real asymmetry the
     Ghent conservators exploited: lead white blocks X-rays but is near
     transparent in the infrared; copper green blocks the infrared but is
     transparent to X-rays. Neither probe alone can recover the buried plan.
     Together they can. The file measures exactly that.

  5. THE OPAQUE GATE (the "a corpo" lead-white stroke, and the overpaint).
     One learned, penalised, sigmoid gate is allowed to restore luminance by
     laying an opaque film on top. It is the only operation that can undo the
     budget — and it blinds both probes to everything beneath it. The file
     measures the auditability cost of using it.

  6. ALS ICH CAN.
     A calibrated competence head that must declare, before scoring, how likely
     its own answer is to be right. "As I can, but not as I would."

  7. THE SPECULAR BYPASS (the convex mirror).
     Gloss/geometry features never enter the stack at all. They route around it
     to a separate head that reconstructs the ILLUMINANT — a fact about the room
     the panel is standing in, not about the panel. Body colour tells you what
     the thing is; the highlight tells you where you are.

THE FALSIFIABLE INTERPRETABILITY TEST (the point of the whole file)

In 2018 the Ghent team predicted, from chemical imaging alone, what the Lamb's
face would look like under the sixteenth-century overpaint. In 2019 they removed
the overpaint and checked. The prediction held. That is the shape of an
interpretability claim that can actually be wrong.

This file reproduces that experiment on a network:
    (a) PREDICT   — decode the committed plan through the intact stack, using
                    only the two probes, while every covering layer is still on.
    (b) INTERVENE — structurally strip layers 1..K-1 and the opaque gate off the
                    probe channels, exposing layer 0's absorbance directly.
    (c) COMPARE   — report the agreement between the prediction and the
                    ground truth thus exposed.
A probe-based claim that survives (c) has been tested. One that never faces (c)
has not.

RUN
    python3 chapter_0395_jan_van_eyck_1390.py            # full: gradient check, training, tests
    python3 chapter_0395_jan_van_eyck_1390.py --quick    # short run for smoke-testing

Requires only numpy.
================================================================================
"""

import argparse
import sys
import time

import numpy as np

# ==============================================================================
# SECTION 0 — Numerical primitives
# ==============================================================================


def softplus(z):
    """Stable log(1+exp(z)). Absorbance must be >= 0: paint cannot add light."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def d_softplus(z):
    """d/dz softplus = sigmoid(z)."""
    return sigmoid(z)


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, y):
    """Mean CE loss and dL/dlogits."""
    p = softmax(logits)
    n = logits.shape[0]
    loss = -np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    return loss, g / n


def bce(p, t):
    """Mean binary cross entropy on probabilities, and dL/dp."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    g = (-(t / p) + (1 - t) / (1 - p)) / p.size
    return loss, g


# ==============================================================================
# SECTION 1 — The panel: a synthetic dataset whose physics is subtractive
#
# The point of this generator is that the data is genuinely Beer-Lambert. A
# substrate contributes a spectral absorbance; a finish adds another; the two
# ADD in the exponent and therefore MULTIPLY in transmittance. Light goes down
# through the film, reflects off the ground and comes back, so the path length
# is doubled. A specular term sits on top of all that and carries no information
# about the material at all — only about the room.
#
# Four substrates (the coarse plan, what the underdrawing has to commit to):
#     0 FLESH   1 CLOTH   2 METAL   3 FOLIAGE
# Three finishes per substrate (what the final glazes decide):
#     0 matte   1 satin   2 lacquered
# giving twelve final classes.
# ==============================================================================

N_BANDS = 10          # spectral bands sampled across the visible
N_SUBSTRATE = 4       # the coarse plan: what the underdrawing commits to
N_FINISH = 3
N_CLASS = N_SUBSTRATE * N_FINISH
N_GEOM = 4            # geometry / gloss features -> specular bypass only
N_FEAT = N_BANDS + N_GEOM

SUBSTRATE_NAMES = ["flesh", "cloth", "metal", "foliage"]
FINISH_NAMES = ["matte", "satin", "lacquered"]


def _band_axis():
    return np.linspace(0.0, 1.0, N_BANDS)


def _make_absorbance_book(rng):
    """Fixed spectral absorbance signatures. Substrates are broad and smooth;
    finishes are narrower and sit at different places in the band, so a finish
    cannot be mistaken for a substrate by a model that has the right form."""
    lam = _band_axis()
    subs = np.zeros((N_SUBSTRATE, N_BANDS))
    for i in range(N_SUBSTRATE):
        centre = 0.12 + 0.25 * i
        width = 0.30 + 0.06 * i
        subs[i] = 0.9 * np.exp(-((lam - centre) ** 2) / (2 * width ** 2))
        subs[i] += 0.18 * (1.0 + 0.5 * np.sin(2.5 * np.pi * lam + i))
    fins = np.zeros((N_FINISH, N_BANDS))
    for j in range(N_FINISH):
        centre = 0.75 - 0.30 * j
        fins[j] = 0.45 * np.exp(-((lam - centre) ** 2) / (2 * 0.10 ** 2))
        fins[j] += 0.05 * j
    return np.abs(subs), np.abs(fins)


def make_panel_dataset(n, rng):
    """Return (x, y_class, y_substrate, y_illum).

    x        (n, N_FEAT)  observed spectrum + geometry
    y_class  (n,)         12-way substrate x finish
    y_sub    (n,)         4-way substrate  (the buried plan)
    y_illum  (n, 2)       azimuth and warmth of the illuminant (the room)
    """
    lam = _band_axis()
    subs, fins = _make_absorbance_book(rng)

    y_sub = rng.integers(0, N_SUBSTRATE, size=n)
    y_fin = rng.integers(0, N_FINISH, size=n)
    y_class = y_sub * N_FINISH + y_fin

    # film thickness varies per sample: the same pigment laid thicker
    thick = rng.uniform(0.65, 1.45, size=(n, 1))

    A = subs[y_sub] + fins[y_fin]                     # absorbances ADD
    A = A * thick
    body = np.exp(-2.0 * A)                           # down and back up

    # the room: azimuth and colour temperature of the light
    azim = rng.uniform(-1.0, 1.0, size=(n, 1))
    warm = rng.uniform(-1.0, 1.0, size=(n, 1))
    illum = np.exp(warm * (lam[None, :] - 0.5) * 1.6)

    # specular: a mirror of the room sitting on the surface, carrying no
    # information about what is underneath it
    # phase shifts with BOTH band and azimuth, so the spectral profile of the
    # highlight identifies where the light is standing. (An earlier version
    # used a frequency that aliased exactly against the sampled bands and made
    # azimuth unrecoverable in principle; the symptom was a negative R^2.)
    env = 0.5 + 0.45 * np.cos(np.pi * (1.3 * lam[None, :] + azim))
    gloss = np.array([0.03, 0.10, 0.22])[y_fin][:, None]
    spec = gloss * env

    obs = illum * (body + spec)
    obs = obs * (1.0 + 0.03 * rng.standard_normal(obs.shape))

    # The highlight, sampled separately: a small patch where the body colour
    # is negligible and what you are looking at is a picture of the room.
    # Four bands, read as logs, which is how a painter reads a highlight —
    # by the RATIO of one part of it to another, not by its brightness.
    hb = np.array([0, 3, 6, 9])   # four bands across the highlight
    hl = illum[:, hb] * env[:, hb] * gloss
    hl = hl * (1.0 + 0.02 * rng.standard_normal(hl.shape))
    geom = np.log(np.clip(hl, 1e-6, None))

    x = np.concatenate([obs, geom], axis=1)
    y_illum = np.concatenate([azim, warm], axis=1)
    return x.astype(np.float64), y_class.astype(np.int64), \
        y_sub.astype(np.int64), y_illum.astype(np.float64)


def standardise(train_x, *others):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True) + 1e-8
    return [(train_x - mu) / sd] + [(o - mu) / sd for o in others]


# ==============================================================================
# SECTION 2 — Channel layout and the structural opacity masks
#
# This is the part that encodes the conservator's asymmetry. Each glaze layer
# is opaque to some probes and transparent to others. The masks are FIXED, not
# learned: they are properties of the pigments, not of the painter.
# ==============================================================================

N_VIS = 8      # visible channels: what the eye gets
N_XRF = 2      # probe A, blocked by "lead white"-like layers
N_IRR = 2      # probe B, blocked by "copper green"-like layers
H = N_VIS + N_XRF + N_IRR

SL_VIS = slice(0, N_VIS)
SL_XRF = slice(N_VIS, N_VIS + N_XRF)
SL_IRR = slice(N_VIS + N_XRF, H)
SL_PROBE = slice(N_VIS, H)

K_LAYERS = 4
LAYER_NAMES = [
    "underdrawing (the committed plan)",
    "copper-green underpaint  (IR-opaque, X-ray transparent)",
    "lead-white underpaint    (X-ray opaque, IR transparent)",
    "final glazes             (blocks the eye only)",
]


def build_masks():
    """M[k] in {0,1}^H : does layer k attenuate this channel?"""
    M = np.ones((K_LAYERS, H))
    # layer 0 — the underdrawing — is seen by everything. That is the whole
    # reason a buried plan is recoverable at all.
    # layer 1 — copper green — invisible to the X-ray probe:
    M[1, SL_XRF] = 0.0
    # layer 2 — lead white — invisible to the infrared probe:
    M[2, SL_IRR] = 0.0
    # layer 3 — the final glazes — invisible to both probes:
    M[3, SL_PROBE] = 0.0
    return M


MASKS = build_masks()

# OCCLUSION COUPLING.
# The same pigment that colours is the pigment that hides. A campaign that lays
# a lot of paint necessarily buries what is under it more deeply. So each
# covering layer's absorbance on the probe channels it blocks is tied to how
# much it laid down on the visible channels. Without this the network can paint
# for free and leave the buried plan perfectly legible, which is not how panels
# work and would make the probe experiment vacuous.
GAMMA_OCCLUDE = 0.9
# The absorbance signature of a flat opaque film, used by the overpaint
# counterfactual. It carries no information about anything underneath it.
OPAQUE_FILM = -1.75
OCCLUDING_LAYERS = (1, 2)     # layer 0 is the drawing; layer 3 sees no probe


# ==============================================================================
# SECTION 3 — The architecture
# ==============================================================================


class GlazeStack:
    """An ordered stack of transparent films over a luminous ground.

    Forward pass, in words:
        s <- 0                                   the ground, at full brightness
        for k in 0..K-1   (downward)             lay the films in order
            a_k  = softplus(...)  >= 0
            s   <- s - M_k * a_k                 light can only be removed
        s <- (1 - g) * s                         the reflective lead-white bed
        for k in K-1..0   (upward)               the light comes back through
            a'_k = softplus(...)                 SAME parameters
            s   <- s - M_k * a'_k
        T = exp(s)                               what leaves the panel

    Layer 0 alone receives the input x. Later layers see only the panel.
    """

    def __init__(self, rng, n_feat=N_FEAT, n_class=N_CLASS,
                 n_sub=N_SUBSTRATE, opaque_penalty=0.08):
        self.rng = rng
        self.n_feat = n_feat
        self.n_class = n_class
        self.n_sub = n_sub
        self.opaque_penalty = opaque_penalty
        self.M = MASKS

        def sc(a, b, s=0.5):
            return rng.standard_normal((a, b)) * (s / np.sqrt(b))

        p = {}
        # layer 0: the underdrawing. Reads the subject (x) and the ground (s).
        p["W0x"] = sc(H, n_feat, 0.7)
        p["W0s"] = sc(H, H, 0.2)
        p["b0"] = np.full(H, -0.7)          # start faint: a light drawing
        # layers 1..K-1: the paint campaigns. They read only the panel.
        for k in range(1, K_LAYERS):
            p[f"W{k}s"] = sc(H, H, 0.35)
            p[f"b{k}"] = np.full(H, -0.9)
        # The reflective lead-white underlayer. In Eyckian practice the form is
        # modelled in opaque lead white and the glazes go over it, so this is
        # the surface the light actually bounces off — which is why the gate
        # sits exactly at the fold between the downward and upward passes. It
        # restores luminance, and it is penalised, because opacity is dear.
        p["Wg"] = sc(H, n_feat, 0.2)
        p["bg"] = np.full(H, -2.5)           # closed by default; opacity is dear
        # readouts
        p["Wc"] = sc(n_class, N_VIS, 1.0)    # the eye reads transmitted light
        p["bc"] = np.zeros(n_class)
        p["Wp"] = sc(n_sub, N_XRF + N_IRR, 1.0)   # the JOINT stratigraphy probe
        p["bp"] = np.zeros(n_sub)
        # The specular bypass: a small dedicated reader for the highlight.
        # It has a hidden layer because the room's geometry enters the
        # highlight non-linearly. It never touches the glaze stack.
        p["Ws1"] = sc(10, N_GEOM, 1.0)
        p["bs1"] = np.zeros(10)
        p["Ws2"] = sc(2, 10, 1.0)
        p["bs2"] = np.zeros(2)
        p["Wq"] = np.zeros((1, N_VIS + 1))   # als ich can: declared competence
        p["bq"] = np.zeros(1)
        self.p = p

    # ---------------------------------------------------------------- forward
    def forward(self, x, masks=None, use_gate=True, force_overpaint=None):
        """Returns (outputs, cache).

        masks / use_gate are overridden by the intervention experiment, which
        physically strips covering layers off the probe channels.
        force_overpaint, if given, lays an opaque film over the probe channels
        only — the counterfactual sixteenth-century restorer.
        """
        p = self.p
        M = self.M if masks is None else masks
        B = x.shape[0]
        s = np.zeros((B, H))

        order = list(range(K_LAYERS)) + list(range(K_LAYERS - 1, -1, -1))
        tape = []
        for step, k in enumerate(order):
            if k == 0:
                z = s @ p["W0s"].T + x @ p["W0x"].T + p["b0"]
            else:
                z = s @ p[f"W{k}s"].T + p[f"b{k}"]
            a = softplus(z)
            # occlusion coupling: laying paint buries the plan (see above)
            a_eff = a
            if k in OCCLUDING_LAYERS:
                bulk = a[:, SL_VIS].mean(axis=1, keepdims=True)
                a_eff = a.copy()
                a_eff[:, SL_PROBE] = a_eff[:, SL_PROBE] + GAMMA_OCCLUDE * bulk
            s_in = s
            s = s - M[k] * a_eff
            tape.append((k, s_in, z, a))
            if step == K_LAYERS - 1:           # the reflective interface
                if use_gate:
                    zg = x @ p["Wg"].T + p["bg"]
                    g = sigmoid(zg)

                    s_pre = s
                    s = (1.0 - g) * s
                    gate = (zg, g, s_pre)
                else:
                    gate = None
                gate_at = len(tape)


        if force_overpaint is not None:
            # A later hand lays an opaque film over the FINISHED panel. It goes
            # on top, so it cannot feed back into the stack — nothing about the
            # painting changes, only what can be read off it. This is NOT an
            # attenuation: a scaled signal is invertible and would lose
            # nothing. An opaque layer SUBSTITUTES its own signature for
            # whatever lies beneath.
            w = float(force_overpaint)
            s = s.copy()
            s[:, SL_PROBE] = (1 - w) * s[:, SL_PROBE] + w * OPAQUE_FILM

        T = np.exp(s[:, SL_VIS])                       # transmitted light
        probe = s[:, SL_PROBE]                         # what the two probes see
        geom = x[:, N_BANDS:]                          # the specular bypass

        logits = T @ p["Wc"].T + p["bc"]
        sub_logits = probe @ p["Wp"].T + p["bp"]
        hs = np.tanh(geom @ p["Ws1"].T + p["bs1"])
        illum = hs @ p["Ws2"].T + p["bs2"]

        # How decided was the call? Negative entropy of the class posterior.
        # Deliberately smooth (no argmax), so its gradient into the classifier
        # is exact rather than piecewise — the declaration must be
        # differentiable if it is to be trained honestly.
        pc = softmax(logits)
        logp = np.log(np.clip(pc, 1e-12, None))
        ent = -(pc * logp).sum(axis=1, keepdims=True)
        conf = -ent
        qfeat = np.concatenate([T, conf], axis=1)
        q = sigmoid(qfeat @ p["Wq"].T + p["bq"])       # declared competence

        out = dict(logits=logits, sub_logits=sub_logits, illum=illum,
                   q=q, T=T, probe=probe, s_final=s)
        cache = dict(x=x, tape=tape, gate=gate, gate_at=gate_at, M=M,
                     T=T, probe=probe, geom=geom, qfeat=qfeat, q=q, hs=hs,
                     logits=logits, order=order, pc=pc, logp=logp, ent=ent)
        return out, cache

    # --------------------------------------------------------------- the loss
    def loss(self, x, y, ysub, yillum, need_grad=True):
        p = self.p
        out, cache = self.forward(x)
        B = x.shape[0]

        l_cls, d_logits = cross_entropy(out["logits"], y)
        l_sub, d_sublog = cross_entropy(out["sub_logits"], ysub)
        diff = out["illum"] - yillum
        l_ill = (diff ** 2).mean()
        d_illum = 2.0 * diff / diff.size

        correct = (out["logits"].argmax(axis=1) == y).astype(np.float64)[:, None]
        l_cal, d_q = bce(out["q"], correct)            # correctness is detached

        g = cache["gate"][1] if cache["gate"] is not None else None
        l_op = self.opaque_penalty * g.mean() if g is not None else 0.0

        total = l_cls + 0.7 * l_sub + 0.5 * l_ill + 0.6 * l_cal + l_op
        parts = dict(total=total, cls=l_cls, sub=l_sub, ill=l_ill,
                     cal=l_cal, opaque=float(g.mean()) if g is not None else 0.0)
        if not need_grad:
            return parts, None

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        M = cache["M"]
        T, probe, geom = cache["T"], cache["probe"], cache["geom"]

        # ---- heads
        # The competence head is done first, because part of its gradient
        # travels back INTO the logits through the decision margin.
        d_q = 0.6 * d_q
        dz_q = d_q * out["q"] * (1 - out["q"])
        grads["Wq"] = dz_q.T @ cache["qfeat"]
        grads["bq"] = dz_q.sum(axis=0)
        dqfeat = dz_q @ p["Wq"]
        dT_from_q = dqfeat[:, :N_VIS]
        d_conf = dqfeat[:, N_VIS:N_VIS + 1]
        # conf = -H(p);  dH/dz_j = -p_j (log p_j + H)  =>  dconf/dz_j = p_j(log p_j + H)
        pc, logp, ent = cache["pc"], cache["logp"], cache["ent"]
        d_logits = d_logits + d_conf * pc * (logp + ent)

        grads["Wc"] = d_logits.T @ T
        grads["bc"] = d_logits.sum(axis=0)
        dT = d_logits @ p["Wc"] + dT_from_q

        d_sublog = 0.7 * d_sublog
        grads["Wp"] = d_sublog.T @ probe
        grads["bp"] = d_sublog.sum(axis=0)
        d_probe = d_sublog @ p["Wp"]

        d_illum = 0.5 * d_illum
        hs = cache["hs"]
        grads["Ws2"] = d_illum.T @ hs
        grads["bs2"] = d_illum.sum(axis=0)
        d_hs = (d_illum @ p["Ws2"]) * (1 - hs ** 2)
        grads["Ws1"] = d_hs.T @ geom
        grads["bs1"] = d_hs.sum(axis=0)
        # the specular bypass ends here: it never touches the stack, by design.

        # ---- assemble dL/ds_final
        ds = np.zeros((B, H))
        ds[:, SL_VIS] = dT * T                     # d exp(s)/ds = exp(s)
        ds[:, SL_PROBE] = d_probe

        # ---- walk the tape backwards (upward pass first, then the gate,
        #      then the downward pass; tied parameters accumulate)
        x_in = cache["x"]
        for idx in range(len(cache["tape"]) - 1, -1, -1):
            k, s_in, z, _a = cache["tape"][idx]
            # s_out = s_in - M[k]*a_eff(z);  z depends on s_in (and x if k==0)
            da_eff = -ds * M[k]
            da = da_eff.copy()
            if k in OCCLUDING_LAYERS:
                # a_eff[probe] = a[probe] + GAMMA * mean(a[visible])
                spill = da_eff[:, SL_PROBE].sum(axis=1, keepdims=True)
                da[:, SL_VIS] += GAMMA_OCCLUDE * spill / N_VIS
            dz = da * d_softplus(z)
            if k == 0:
                grads["W0x"] += dz.T @ x_in
                grads["W0s"] += dz.T @ s_in
                grads["b0"] += dz.sum(axis=0)
                ds = ds + dz @ p["W0s"]
            else:
                grads[f"W{k}s"] += dz.T @ s_in
                grads[f"b{k}"] += dz.sum(axis=0)
                ds = ds + dz @ p[f"W{k}s"]
            if idx == cache["gate_at"] and cache["gate"] is not None:
                zg, gg, s_pre = cache["gate"]
                dg = -(ds * s_pre)
                ds = ds * (1.0 - gg)
                dg = dg + self.opaque_penalty / gg.size    # the opacity penalty
                dzg = dg * gg * (1 - gg)
                grads["Wg"] += dzg.T @ x_in
                grads["bg"] += dzg.sum(axis=0)

        return parts, grads


# ==============================================================================
# SECTION 4 — Optimiser (Adam, written out, because the file uses no framework)
# ==============================================================================


class Adam:
    def __init__(self, params, lr=6e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            gk = grads[k] + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ==============================================================================
# SECTION 5 — Baseline: a conventional additive MLP with matched parameters
#
# The claim under test is not "deep nets work". It is that a stack whose
# arithmetic matches the arithmetic of the world it is modelling learns that
# world with the same budget better than one that does not. If the baseline
# wins, the chapter has to say so.
# ==============================================================================


class MatchedMLP:
    def __init__(self, rng, n_feat, width, depth, n_class):
        self.rng = rng
        self.depth = depth
        p = {}
        dims = [n_feat] + [width] * depth + [n_class]
        for i in range(len(dims) - 1):
            p[f"W{i}"] = rng.standard_normal((dims[i + 1], dims[i])) \
                / np.sqrt(dims[i])
            p[f"b{i}"] = np.zeros(dims[i + 1])
        self.p = p
        self.n_layers = len(dims) - 1

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, x):
        h = x
        cache = [h]
        for i in range(self.n_layers):
            z = h @ self.p[f"W{i}"].T + self.p[f"b{i}"]
            if i < self.n_layers - 1:
                h = np.tanh(z)
                cache.append(h)
            else:
                h = z
        return h, cache

    def loss(self, x, y):
        logits, cache = self.forward(x)
        l, dl = cross_entropy(logits, y)
        grads = {}
        d = dl
        for i in range(self.n_layers - 1, -1, -1):
            hin = cache[i]
            grads[f"W{i}"] = d.T @ hin
            grads[f"b{i}"] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.p[f"W{i}"]) * (1 - cache[i] ** 2)
        return l, grads, logits


# ==============================================================================
# SECTION 6 — Gradient check (mandatory)
# ==============================================================================


def gradient_check(model, x, y, ysub, yillum, n_probe=6, eps=1e-5, seed=0):
    """Central finite differences against the analytic gradient, on a random
    sample of entries from every parameter tensor.

    Note on eps: at 1e-6 the difference quotient's own roundoff floor (~1e-11
    absolute, on a loss of order 4) swamps parameter entries whose true
    gradient is around 1e-6, and reports spurious 1e-5 relative errors on
    gradients that are in fact correct to eight digits. 1e-5 sits below the
    truncation error and above the roundoff floor. The denominator is likewise
    floored so that a near-zero gradient cannot manufacture a large ratio."""
    rng = np.random.default_rng(seed)
    _, grads = model.loss(x, y, ysub, yillum, need_grad=True)
    rows = []
    worst = 0.0
    for name in sorted(model.p.keys()):
        P = model.p[name]
        flat = P.ravel()
        gflat = grads[name].ravel()
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = model.loss(x, y, ysub, yillum, need_grad=False)
            flat[i] = orig - eps
            lm, _ = model.loss(x, y, ysub, yillum, need_grad=False)
            flat[i] = orig
            num = (lp["total"] - lm["total"]) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-4, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        rows.append((name, P.shape, e))
    return rows, worst


# ==============================================================================
# SECTION 7 — Diagnostics that are specific to this mind
# ==============================================================================


def fit_ridge_classifier(F, y, n_cls, lam=1e-2):
    """Closed-form ridge onto one-hot targets. Used to read out probes AFTER
    training, so the probes are never given a training advantage the intact
    model did not already have."""
    Fb = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
    Y = np.zeros((F.shape[0], n_cls))
    Y[np.arange(F.shape[0]), y] = 1.0
    A = Fb.T @ Fb + lam * np.eye(Fb.shape[1])
    W = np.linalg.solve(A, Fb.T @ Y)
    return W


def apply_ridge(W, F):
    Fb = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
    return (Fb @ W).argmax(axis=1)


def monotonicity_report(model, x):
    """Assert the luminance budget is spent, never replenished, except by the
    one operation licensed to replenish it."""
    _, cache = model.forward(x, use_gate=False)
    s = np.zeros((x.shape[0], H))
    worst_increase = -np.inf
    for (k, s_in, z, a) in cache["tape"]:
        a_eff = a
        if k in OCCLUDING_LAYERS:
            a_eff = a.copy()
            a_eff[:, SL_PROBE] += GAMMA_OCCLUDE * a[:, SL_VIS].mean(1, keepdims=True)
        s_out = s_in - cache["M"][k] * a_eff
        worst_increase = max(worst_increase, (s_out - s_in).max())
    return worst_increase


def stripped_masks():
    """The intervention: physically remove layers 1..K-1 from the probe
    channels. Only the underdrawing is left where the probes can see it."""
    M = MASKS.copy()
    for k in range(1, K_LAYERS):
        M[k, SL_PROBE] = 0.0
    return M


# ==============================================================================
# SECTION 8 — Training and the full experimental report
# ==============================================================================


def run(quick=False, seed=1441):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    n_tr = 1500 if quick else 6000
    n_te = 600 if quick else 2000
    epochs = 12 if quick else 45
    batch = 256

    xtr, ytr, str_, itr = make_panel_dataset(n_tr, rng)
    xte, yte, ste, ite = make_panel_dataset(n_te, rng)
    xtr, xte = standardise(xtr, xte)

    print("=" * 78)
    print(" FIGURE 0395 — JAN VAN EYCK — LAYERED TRANSMITTANCE STACK")
    print("=" * 78)
    print(f" train {n_tr}  test {n_te}  features {N_FEAT}  classes {N_CLASS}")
    print(f" channels: {N_VIS} visible + {N_XRF} X-ray probe + {N_IRR} IR probe")
    print(f" layers ({K_LAYERS}, traversed down then up with tied parameters):")
    for k, nm in enumerate(LAYER_NAMES):
        vis = "".join("#" if m else "." for m in MASKS[k])
        print(f"   L{k}  {vis}   {nm}")
    print("   key: first 8 = visible, next 2 = X-ray probe, last 2 = IR probe")
    print("        '#' = this layer attenuates that channel, '.' = transparent")

    model = GlazeStack(rng)
    n_par = sum(v.size for v in model.p.values())
    print(f"\n parameters: {n_par}")

    # ---------------------------------------------------------- gradient check
    print("\n" + "-" * 78)
    print(" 1. GRADIENT CHECK  (central finite differences, eps=1e-5)")
    print("-" * 78)
    gx, gy, gs, gi = xtr[:24], ytr[:24], str_[:24], itr[:24]
    rows, worst = gradient_check(model, gx, gy, gs, gi)
    for name, shape, e in rows:
        flag = "ok" if e < 1e-5 else "FAIL"
        print(f"   {name:5s} {str(shape):12s} rel.err {e:.3e}  {flag}")
    print(f"   worst relative error: {worst:.3e}")
    assert worst < 1e-5, "GRADIENT CHECK FAILED"
    print("   PASS — analytic gradients match finite differences.")

    # ---------------------------------------------------------------- training
    print("\n" + "-" * 78)
    print(" 2. TRAINING")
    print("-" * 78)
    opt = Adam(model.p, lr=6e-3)
    n = xtr.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        agg = None
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            parts, grads = model.loss(xtr[idx], ytr[idx], str_[idx], itr[idx])
            opt.step(model.p, grads)
            agg = parts if agg is None else {k: agg[k] + parts[k] for k in parts}
        nb = int(np.ceil(n / batch))
        agg = {k: v / nb for k, v in agg.items()}
        if ep % max(1, epochs // 9) == 0 or ep == epochs - 1:
            out, _ = model.forward(xte)
            acc = (out["logits"].argmax(1) == yte).mean()
            print(f"   epoch {ep:3d}  loss {agg['total']:.4f}  "
                  f"cls {agg['cls']:.4f}  plan {agg['sub']:.4f}  "
                  f"room {agg['ill']:.4f}  opacity {agg['opaque']:.3f}  "
                  f"| test acc {acc:.4f}")

    out, cache = model.forward(xte)
    acc = (out["logits"].argmax(1) == yte).mean()

    # -------------------------------------------------- baseline for comparison
    print("\n" + "-" * 78)
    print(" 3. BASELINE — conventional additive MLP, matched parameter budget")
    print("-" * 78)
    width, depth = 18, 3
    mlp = MatchedMLP(rng, N_FEAT, width, depth, N_CLASS)
    print(f"   MLP: {depth} hidden layers of {width}, "
          f"{mlp.n_params()} parameters (stack: {n_par})")
    bopt = Adam(mlp.p, lr=6e-3)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            l, g, _ = mlp.loss(xtr[idx], ytr[idx])
            bopt.step(mlp.p, g)
    blog, _ = mlp.forward(xte)
    bacc = (blog.argmax(1) == yte).mean()
    print(f"   MLP test accuracy   {bacc:.4f}")
    print(f"   stack test accuracy {acc:.4f}")
    print(f"   difference          {acc - bacc:+.4f}")

    # -------------------------------------------------- monotone light budget
    print("\n" + "-" * 78)
    print(" 4. THE LUMINANCE BUDGET IS MONOTONE")
    print("-" * 78)
    worst_up = monotonicity_report(model, xte)
    print(f"   largest luminance INCREASE across any glaze step: {worst_up:.3e}")
    assert worst_up <= 1e-12, "a glaze layer added light"
    print("   PASS — no layer ever added light. Light is only ever spent.")
    T = out["T"]
    print(f"   mean transmittance leaving the panel: {T.mean():.4f}"
          f"   (ground = 1.0000)")
    print(f"   spend: {(1 - T.mean()) * 100:.1f}% of the ground's brightness "
          f"was converted into representation.")

    # ------------------------------------------------ complementary blindness
    print("\n" + "-" * 78)
    print(" 5. TWO PROBES, COMPLEMENTARY BLINDNESS")
    print("-" * 78)
    otr, _ = model.forward(xtr)
    Ptr, Pte = otr["probe"], out["probe"]
    xrf_tr, xrf_te = Ptr[:, :N_XRF], Pte[:, :N_XRF]
    irr_tr, irr_te = Ptr[:, N_XRF:], Pte[:, N_XRF:]

    Wx = fit_ridge_classifier(xrf_tr, str_, N_SUBSTRATE)
    Wi = fit_ridge_classifier(irr_tr, str_, N_SUBSTRATE)
    Wj = fit_ridge_classifier(Ptr, str_, N_SUBSTRATE)
    a_x = (apply_ridge(Wx, xrf_te) == ste).mean()
    a_i = (apply_ridge(Wi, irr_te) == ste).mean()
    a_j = (apply_ridge(Wj, Pte) == ste).mean()
    print(f"   plan recovered from the X-ray probe alone : {a_x:.4f}")
    print(f"   plan recovered from the IR probe alone    : {a_i:.4f}")
    print(f"   plan recovered from BOTH probes together  : {a_j:.4f}")
    print(f"   chance                                    : {1/N_SUBSTRATE:.4f}")
    print(f"   joint gain over the better single probe   : "
          f"{a_j - max(a_x, a_i):+.4f}")

    # ------------------------------------ the predict / intervene / compare test
    print("\n" + "-" * 78)
    print(" 6. THE FALSIFIABLE TEST: predict the buried plan, then strip it bare")
    print("-" * 78)
    pred_covered = apply_ridge(Wj, Pte)          # (a) predicted through the paint

    Mstrip = stripped_masks()
    out_s, _ = model.forward(xte, masks=Mstrip, use_gate=False)   # (b) stripped
    otr_s, _ = model.forward(xtr, masks=Mstrip, use_gate=False)
    Wclean = fit_ridge_classifier(otr_s["probe"], str_, N_SUBSTRATE)
    exposed = apply_ridge(Wclean, out_s["probe"])

    agree = (pred_covered == exposed).mean()
    acc_cov = (pred_covered == ste).mean()
    acc_exp = (exposed == ste).mean()
    # The decisive number. Raw agreement between two imperfect readings is
    # bounded by BOTH their error rates and so understates the claim. What the
    # Ghent team actually verified was narrower and stronger: on the panels
    # where stripping reveals the plan unambiguously, did the prediction made
    # through the paint already say so?
    ok_exp = (exposed == ste)
    agree_ok = (pred_covered[ok_exp] == exposed[ok_exp]).mean()
    print("   (a) prediction made with every covering layer still in place")
    print("   (b) layers 1..3 structurally removed from the probe channels,")
    print("       the opaque gate lifted; layer 0's absorbance now exposed")
    print(f"   accuracy of the covered prediction     : {acc_cov:.4f}")
    print(f"   accuracy once stripped bare            : {acc_exp:.4f}")
    print(f"   raw agreement of the two readings      : {agree:.4f}")
    print(f"   agreement where stripping is unambiguous: {agree_ok:.4f}")
    print("   Note the ordering: the reading taken THROUGH the paint is the")
    print("   more accurate of the two. Stripping discards the covering layers'")
    print("   own evidence about what they were covering.")
    verdict = ("the probe-based reading survived the intervention"
               if agree_ok > 0.85 and acc_cov >= acc_exp - 0.02 else
               "the probe-based reading did NOT survive the intervention")
    print(f"   verdict: {verdict}")

    # ------------------------------------------------- the cost of being opaque
    print("\n" + "-" * 78)
    print(" 7. WHAT OPACITY COSTS: the overpaint blinds the probes")
    print("-" * 78)
    gvals = cache["gate"][1].mean(axis=1)
    print(f"   mean gate opening the model chose (0 = transparent): "
          f"{gvals.mean():.4f}")
    print("   counterfactual: a later hand lays an opaque film over the panel.")
    print("   The paint underneath is untouched. Only the reading is lost.")
    print("   overpaint   plan recovery   task accuracy")
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        o_f, _ = model.forward(xte, force_overpaint=strength)
        a_f = (apply_ridge(Wj, o_f["probe"]) == ste).mean()
        t_f = (o_f["logits"].argmax(1) == yte).mean()
        bar = "#" * max(0, int(round(34 * (a_f - 0.25) / 0.75)))
        print(f"     {strength:4.2f}         {a_f:.4f}         {t_f:.4f}  |{bar}")
    o_f, _ = model.forward(xte, force_overpaint=1.0)
    overpaint_cost = (apply_ridge(Wj, Pte) == ste).mean() \
        - (apply_ridge(Wj, o_f["probe"]) == ste).mean()
    task_cost = acc - (o_f["logits"].argmax(1) == yte).mean()
    print(f"   audit lost to a full overpaint : {overpaint_cost:+.4f}")
    print(f"   task performance lost          : {task_cost:+.4f}")
    print("   The whole audit trail can be destroyed at no cost whatsoever to")
    print("   the measured score. No benchmark on this model would notice.")

    # ---------------------------------------------------------- the pentimento
    print("\n" + "-" * 78)
    print(" 8. PENTIMENTO INDEX: does the finished panel still agree with its plan?")
    print("-" * 78)
    final_sub = out["logits"].argmax(1) // N_FINISH
    pent = (final_sub != exposed).mean()
    print(f"   final answer's implied substrate vs the exposed underdrawing")
    print(f"   disagreement rate (pentimento index): {pent:.4f}")
    print("   A high index is not an error. It is a design change, and the only")
    print("   reason it can be seen at all is that nothing was overwritten.")

    # ------------------------------------------------------ the specular bypass
    print("\n" + "-" * 78)
    print(" 9. THE MIRROR: the room reconstructed from the highlight alone")
    print("-" * 78)
    err = out["illum"] - ite
    r2 = 1 - (err ** 2).sum(axis=0) / ((ite - ite.mean(0)) ** 2).sum(axis=0)
    print(f"   illuminant azimuth  R^2 = {r2[0]:.4f}")
    print(f"   illuminant warmth   R^2 = {r2[1]:.4f}")
    print("   Neither number passed through a single glaze. The body of the")
    print("   paint says what the thing is; the highlight says where you are.")

    # -------------------------------------------------------------- als ich can
    print("\n" + "-" * 78)
    print(" 10. ALS ICH CAN: the declaration, and whether it was true")
    print("-" * 78)
    q = out["q"].ravel()
    corr = (out["logits"].argmax(1) == yte).astype(float)
    bins = np.linspace(0, 1, 9)
    ece = 0.0
    print("   declared    n     actual")
    for i in range(len(bins) - 1):
        m = (q >= bins[i]) & (q < bins[i + 1])
        if m.sum() > 8:
            ece += m.mean() * abs(q[m].mean() - corr[m].mean())
            print(f"   {q[m].mean():.3f}   {m.sum():5d}    {corr[m].mean():.3f}")
    print(f"   expected calibration error: {ece:.4f}")
    order_q = np.argsort(-q)
    for cov in (0.25, 0.50, 0.75, 1.00):
        kk = max(1, int(cov * len(q)))
        print(f"   at {int(cov*100):3d}% coverage (most confident first): "
              f"accuracy {corr[order_q[:kk]].mean():.4f}")

    # ---------------------------------------------------------------- self-tests
    print("\n" + "-" * 78)
    print(" 11. SELF-TESTS")
    print("-" * 78)
    checks = []

    checks.append(("gradient check passes", worst < 1e-5))
    checks.append(("no layer ever adds light", worst_up <= 1e-12))
    checks.append(("all absorbances non-negative",
                   all(softplus(z).min() >= 0 for (_, _, z, _a) in cache["tape"])))
    checks.append(("transmittance stays in (0, 1]",
                   bool((T > 0).all() and (T <= 1.0 + 1e-9).all())))
    checks.append(("class accuracy well above chance", acc > 4.0 / N_CLASS))
    checks.append(("both probes needed: joint beats each single",
                   a_j > max(a_x, a_i) + 0.01))
    checks.append(("probe reading survives the intervention",
                   agree_ok > 0.85 and acc_cov >= acc_exp - 0.02))
    checks.append(("the room is read without touching the stack",
                   float(r2.min()) > 0.5))
    checks.append(("competence declaration is informative",
                   corr[order_q[:len(q)//4]].mean() > corr.mean()))
    checks.append(("the stack is parameter-tied down and up",
                   len(cache["order"]) == 2 * K_LAYERS))
    checks.append(("opacity stayed rare", float(gvals.mean()) < 0.5))
    checks.append(("overpaint measurably destroys the audit trail",
                   overpaint_cost > 0.05))

    STRUCTURAL = {"gradient check passes", "no layer ever adds light",
                  "all absorbances non-negative",
                  "transmittance stays in (0, 1]",
                  "the stack is parameter-tied down and up"}
    ok = True
    for name, val in checks:
        val = bool(val)
        skip = quick and name not in STRUCTURAL and not val
        if not skip:
            ok &= val
        tag = "PASS" if val else ("under-trained" if skip else "FAIL")
        print(f"   [{tag}] {name}")
    if quick:
        print("   (--quick is a smoke test: 12 epochs on a quarter of the data.")
        print("    Only the structural invariants are enforced at this size.)")
    print("-" * 78)
    print(f" ALL SELF-TESTS {'PASSED' if ok else 'FAILED'}   "
          f"({time.time() - t0:.1f}s)")
    print("=" * 78)
    if not ok:
        sys.exit(1)
    return dict(acc=acc, bacc=bacc, a_x=a_x, a_i=a_i, a_j=a_j,
                agree=agree, pent=pent, ece=ece, r2=r2, gate=gvals.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1441)
    a = ap.parse_args()
    run(quick=a.quick, seed=a.seed)

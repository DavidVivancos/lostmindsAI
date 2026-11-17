#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0398 - FRA ANGELICO (1395-1455)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 (Guido di Pietro / Fra Giovanni da Fiesole, c. 1395 - 18 February 1455)

 THE CELL COMPILER
 An Addressee-Compiled, Self-Extinguishing Scaffold Network (ACSN)
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Between roughly 1438 and 1445 Fra Angelico and his workshop painted about
fifty frescoes into the convent of San Marco in Florence, and around forty-four
of them went into private sleeping cells that no public would ever enter. The
cells were sorted by rank -- novices along one corridor, professed clerics
along another, lay brothers along a third, and a double cell at the far end
reserved for the banker who paid for the building. The frescoes were sorted
the same way. Novices got the same scene seven times over, stripped of
setting, with the founder of the order kneeling at the foot of the cross in a
different bodily posture in each one, drawn from a thirteenth-century
Dominican manual that prescribed nine such postures. Clerics got compressed,
allusive, difficult images. Lay brothers got plain narrative. The patron got
colour, gold-adjacent brightness and a crowd.

Three things follow, and they are the whole design of this file.

 (1) ONE DOCTRINE, MANY COMPILATIONS. The content is identical across the
     convent. What changes is the *compilation target*: the state of the one
     man who will sleep under it. So the network here has a single shared
     doctrine core plus a small per-cell scaffold, and the scaffold is the
     only thing that differs between receivers.

 (2) THE IMAGE IS A MOTOR PROGRAM, NOT A LABEL. The novice frescoes do not
     assert a proposition; they specify a configuration of a body, which the
     novice then executes with his own body. Understanding is re-enactment.
     So this network does not classify by matching stored keys. It recovers a
     posture, RE-RENDERS it through a physical forward model, and is scored on
     whether the re-enactment reproduces what it saw. Analysis by synthesis,
     with the body as the representation.

 (3) THE SCAFFOLD IS BUILT TO BE THROWN AWAY. A cell fresco succeeds when the
     friar no longer needs to look at it -- when the posture has moved into
     the body and the wall has become furniture. This is the part no other
     architecture in the corpus implements: the training objective contains an
     explicit EXTINCTION term. Every batch is run twice, once with the
     per-cell scaffold and once with it ablated, and the ablated pass is
     penalised at least as hard as the scaffolded one, while a decay penalty
     drives the scaffold gates toward zero. The headline metric is not
     accuracy. It is the EXTINCTION RATIO: ablated performance divided by
     scaffolded performance. A teacher that cannot be removed has failed.

 A fourth, quieter commitment: WINDOW REGISTRATION. In the San Marco corridor
 the painted light in the Annunciation falls from the left because the real
 corridor window is on the left; the Madonna delle Ombre casts hard shadows
 consistent with the actual south light of that wall. The fiction is
 registered to the room. Here, every cell has its own known illumination
 direction, supplied to the renderer as a constant and never inferred. The
 encoder therefore has to become invariant to it, and the file measures that
 invariance directly.

WHAT THIS IS NOT
----------------
Not a transformer, not attention over stored keys, not a mixture of experts.
There is no retrieval step anywhere. The per-cell scaffolds superficially
resemble LoRA adapters, and the resemblance is the point of departure rather
than the design: adapters exist to be kept, and this one exists to be
destroyed by its own loss function.

RUNNING
-------
    python3 chapter_0398_fra_angelico_1395.py            # gradient check, training, self-tests
    python3 chapter_0398_fra_angelico_1395.py --quick    # short run for smoke-testing

Dependencies: NumPy only. Backpropagation is written out by hand, including
through the renderer.
================================================================================
"""

import sys
import time
import numpy as np

RNG_SEED = 1395  # the conventional birth year

# ==============================================================================
# SECTION 1 -- THE BODY
#
# A minimal planar body with five joint angles. This is the representational
# substrate of the whole model: the network's "answer" to any input is a
# configuration of this body, not a symbol.
#
#   a0  torso lean          (hip -> shoulder, absolute)
#   a1  head pitch          (relative to torso)
#   a2  left arm            (relative to torso)
#   a3  right arm           (relative to torso)
#   a4  kneel / leg flexion (hip -> knee, absolute)
#
# Chosen because the Dominican manual De modo orandi describes its nine modes
# of prayer almost entirely in these terms: how deeply the trunk inclines,
# where the head is carried, what the arms do, whether the knees are down.
# ==============================================================================

N_ANGLES = 5
SEG_LEN = np.array([0.45, 0.18, 0.35, 0.35, 0.40])  # torso, head, l-arm, r-arm, leg
HIP = np.array([0.0, -0.12])
N_SEG = 5


def _u(theta):
    """Unit direction (sin, cos): theta measured clockwise from straight up."""
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)


def _du(theta):
    """d/dtheta of _u."""
    return np.stack([np.cos(theta), -np.sin(theta)], axis=-1)


def forward_kinematics(ang):
    """
    ang: (B, 5) joint angles in radians.
    Returns joint positions and the per-segment endpoint pairs.

    The chain is deliberately shallow -- shoulder carries head and both arms,
    hip carries shoulder and knee -- so that a change in torso lean moves
    everything above it. Grief, supplication and adoration in the manual are
    all distinguished first by the trunk.
    """
    B = ang.shape[0]
    th0 = ang[:, 0]
    th1 = ang[:, 0] + ang[:, 1]
    th2 = ang[:, 0] + ang[:, 2]
    th3 = ang[:, 0] + ang[:, 3]
    th4 = np.pi - ang[:, 4]   # leg hangs DOWN from the hip; a4 swings it forward

    hip = np.broadcast_to(HIP, (B, 2))
    shoulder = hip + SEG_LEN[0] * _u(th0)
    head = shoulder + SEG_LEN[1] * _u(th1)
    lhand = shoulder + SEG_LEN[2] * _u(th2)
    rhand = shoulder + SEG_LEN[3] * _u(th3)
    knee = hip + SEG_LEN[4] * _u(th4)

    # segment endpoint arrays, shape (B, N_SEG, 2)
    A = np.stack([hip, shoulder, shoulder, shoulder, hip], axis=1)
    Bp = np.stack([shoulder, head, lhand, rhand, knee], axis=1)
    cache = dict(th=(th0, th1, th2, th3, th4), shoulder=shoulder)
    return A, Bp, cache


# ==============================================================================
# SECTION 2 -- THE SOFT RENDERER (the forward physical model)
#
# Turns a body configuration plus a light direction into a small grey image.
# It is smooth everywhere so that gradients flow back from the reconstructed
# image to the joint angles. This is what makes "understanding by re-enacting"
# trainable rather than decorative.
#
# Shading: each limb is a cylinder-ish surface, so its brightness depends on
# the angle between its normal and the incoming light. That single line is the
# window-registration mechanism -- change the cell's window and every limb
# changes brightness, while the posture itself does not change at all.
# ==============================================================================

GRID = 18                 # image is GRID x GRID
N_BLOB = 6                # sample points per segment
SIGMA = 0.100             # blob width in world units
GEXT = 0.78               # half-width of the rendered field, in world units
SHADE_LO, SHADE_HI = 0.35, 0.65
SHADE_K = 4.0

_TS = np.linspace(0.12, 0.88, N_BLOB)          # (M,) positions along each segment
_gx = np.linspace(-GEXT, GEXT, GRID)
_gy = np.linspace(GEXT, -GEXT, GRID)
_GXX, _GYY = np.meshgrid(_gx, _gy)
_PIX = np.stack([_GXX.ravel(), _GYY.ravel()], axis=-1)   # (P, 2), P = GRID*GRID
N_PIX = _PIX.shape[0]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def render(ang, light):
    """
    ang:   (B, 5)  joint angles
    light: (B, 2)  unit vector pointing toward the light source (the cell window)
    returns img (B, N_PIX) in [0, 1), plus a cache for the backward pass.
    """
    B = ang.shape[0]
    A, Bp, fkc = forward_kinematics(ang)          # (B, S, 2)
    d = Bp - A                                    # (B, S, 2)
    dn = np.sqrt(np.sum(d * d, axis=-1) + 1e-9)   # (B, S)

    # surface normal of the limb, perpendicular to its axis
    q = (d[..., 1] * light[:, None, 0] - d[..., 0] * light[:, None, 1]) / dn   # (B,S)
    sg = _sigmoid(SHADE_K * q)
    shade = SHADE_LO + SHADE_HI * sg              # (B, S)

    # blob centres: (B, S, M, 2)
    C = A[:, :, None, :] + _TS[None, None, :, None] * d[:, :, None, :]

    # squared distance from every pixel to every blob
    diff = _PIX[None, None, None, :, :] - C[:, :, :, None, :]      # (B,S,M,P,2)
    d2 = np.sum(diff * diff, axis=-1)                              # (B,S,M,P)
    E = np.exp(-d2 / (2.0 * SIGMA * SIGMA))                        # (B,S,M,P)

    F = np.sum(shade[:, :, None, None] * E, axis=(1, 2))           # (B,P)
    expF = np.exp(-np.clip(F, 0, 40))
    img = 1.0 - expF

    cache = dict(A=A, Bp=Bp, d=d, dn=dn, q=q, sg=sg, shade=shade,
                 C=C, diff=diff, E=E, expF=expF, light=light, ang=ang, fkc=fkc)
    return img, cache


def render_backward(dimg, cache):
    """
    dimg: (B, N_PIX) gradient of the loss wrt the rendered image.
    Returns dang: (B, 5). Hand-derived; verified by the finite-difference
    check in Section 7.
    """
    d, dn, q, sg, shade = cache['d'], cache['dn'], cache['q'], cache['sg'], cache['shade']
    diff, E, expF, light = cache['diff'], cache['E'], cache['expF'], cache['light']
    ang = cache['ang']
    B = ang.shape[0]

    dF = dimg * expF                                        # (B,P)

    # -- shading gradient
    dshade = np.sum(dF[:, None, None, :] * E, axis=(2, 3))  # (B,S)

    # -- blob-centre gradient: dE/dC = E * (x - C)/sigma^2, and diff = x - C
    w = dF[:, None, None, :] * shade[:, :, None, None] * E  # (B,S,M,P)
    dC = np.sum(w[..., None] * diff, axis=3) / (SIGMA ** 2)  # (B,S,M,2)

    # C = A + t*d  ->  dA += sum_m dC, dd += sum_m t*dC
    dA = np.sum(dC, axis=2)                                  # (B,S,2)
    dd = np.sum(_TS[None, None, :, None] * dC, axis=2)       # (B,S,2)

    # -- shading depends on d through q
    dq = dshade * SHADE_HI * SHADE_K * sg * (1.0 - sg)       # (B,S)
    inv = 1.0 / dn
    dq_ddx = (-light[:, None, 1]) * inv - q * d[..., 0] * inv * inv
    dq_ddy = (light[:, None, 0]) * inv - q * d[..., 1] * inv * inv
    dd = dd + np.stack([dq * dq_ddx, dq * dq_ddy], axis=-1)

    # d = Bp - A
    dBp = dd.copy()
    dA = dA - dd

    # -- scatter segment endpoint gradients onto the joints
    # segments: 0 hip->shoulder, 1 shoulder->head, 2 shoulder->lhand,
    #           3 shoulder->rhand, 4 hip->knee
    g_sh = dBp[:, 0] + dA[:, 1] + dA[:, 2] + dA[:, 3]
    g_hd = dBp[:, 1]
    g_lh = dBp[:, 2]
    g_rh = dBp[:, 3]
    g_kn = dBp[:, 4]
    # dA[:,0] and dA[:,4] land on the hip, which is fixed -> discarded

    th0, th1, th2, th3, th4 = cache['fkc']['th']

    dth1 = np.sum(g_hd * (SEG_LEN[1] * _du(th1)), axis=-1)
    dth2 = np.sum(g_lh * (SEG_LEN[2] * _du(th2)), axis=-1)
    dth3 = np.sum(g_rh * (SEG_LEN[3] * _du(th3)), axis=-1)
    dth4 = np.sum(g_kn * (SEG_LEN[4] * _du(th4)), axis=-1)

    # head/hands also translate with the shoulder
    g_sh = g_sh + g_hd + g_lh + g_rh
    dth0 = np.sum(g_sh * (SEG_LEN[0] * _du(th0)), axis=-1)

    dang = np.zeros((B, N_ANGLES))
    dang[:, 0] = dth0 + dth1 + dth2 + dth3   # a0 feeds th0, th1, th2, th3
    dang[:, 1] = dth1
    dang[:, 2] = dth2
    dang[:, 3] = dth3
    dang[:, 4] = -dth4        # th4 = pi - a4, so the sign flips here
    return dang


# ==============================================================================
# SECTION 3 -- THE CONVENT: cells, receivers, and the doctrine
#
# K canonical postures = the "modes". One doctrine, shared by the whole house.
# C cells = receiver classes, each with:
#     - its own window direction (a real physical fact about the room)
#     - its own degradation profile (how hard it is for THAT receiver to read)
#     - its own scaffold in the network
#
# The degradation profile is the honest part of the pedagogy: a novice does not
# see less well because he is stupid, he sees less well because he has not yet
# been trained on what to ignore. The scaffold exists to bridge exactly that
# gap, and then to be removed.
# ==============================================================================

N_MODE = 7      # seven novice cells at San Marco carried seven modes
N_CELL = 4      # novices / clerics / lay brothers / patron

CELL_NAMES = ["novice", "cleric", "lay-brother", "patron"]

# window azimuths in radians; light unit vector = (sin, cos)
CELL_LIGHT_ANGLE = np.array([-1.05, 0.55, 2.30, -2.05])

# per-cell input corruption: (gaussian noise sigma, occlusion prob, contrast)
CELL_NOISE = np.array([0.34, 0.10, 0.20, 0.26])
CELL_OCCL = np.array([0.28, 0.06, 0.14, 0.10])
CELL_CONTRAST = np.array([0.72, 1.00, 0.88, 1.18])


def canonical_modes(rng):
    """
    Seven prototype postures, loosely after the descriptions in De modo orandi:
    deep inclination, prostration-adjacent kneel, arms outstretched in the form
    of the cross, hands raised, hands joined before the breast, upright reading
    posture, and reaching upward. These are prototypes, not exact
    reconstructions -- the manual describes gestures in prose, not in degrees.
    """
    protos = np.array([
        [0.55, 0.35, 0.40, -0.40, 0.10],    # 0 deep inclination
        [0.95, 0.50, 0.25, -0.25, 0.85],    # 1 low kneel, head down
        [0.10, 0.05, 1.45, -1.45, 0.30],    # 2 arms out, cross form
        [-0.15, -0.35, 1.05, -1.05, 0.25],  # 3 hands raised
        [0.20, 0.15, 0.18, -0.18, 0.45],    # 4 hands joined at breast
        [0.05, 0.30, 0.55, -0.30, 0.05],    # 5 upright, reading
        [-0.35, -0.55, 0.75, -1.30, 0.20],  # 6 reaching upward, asymmetric
    ])
    assert protos.shape == (N_MODE, N_ANGLES)
    return protos


def make_dataset(n_per_cell, rng):
    """
    Builds the convent's training set.

    For each sample we choose a cell and a mode, jitter the prototype posture,
    render it under THAT CELL'S light, then degrade the render according to the
    receiver's profile. The clean render (under the same light) is kept as the
    re-enactment target.
    """
    protos = canonical_modes(rng)
    Xs, Ys, Ps, Cs, Ts = [], [], [], [], []
    for c in range(N_CELL):
        n = n_per_cell
        modes = rng.integers(0, N_MODE, size=n)
        ang = protos[modes] + rng.normal(0, 0.13, size=(n, N_ANGLES))
        la = CELL_LIGHT_ANGLE[c]
        light = np.broadcast_to(np.array([np.sin(la), np.cos(la)]), (n, 2))
        clean, _ = render(ang, light)

        obs = clean * CELL_CONTRAST[c]
        obs = obs + rng.normal(0, CELL_NOISE[c], size=obs.shape)
        mask = (rng.random(obs.shape) > CELL_OCCL[c]).astype(float)
        obs = obs * mask
        obs = np.clip(obs, -1.5, 1.5)

        Xs.append(obs); Ys.append(modes); Ps.append(ang)
        Cs.append(np.full(n, c)); Ts.append(clean)
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ps),
            np.concatenate(Cs), np.concatenate(Ts))


# ==============================================================================
# SECTION 4 -- THE NETWORK
#
#   x --[ W1 ]--> h1 --(+ per-cell scaffold, gated)--> tanh --[ W2 ]--> h2
#                                                                 |
#                                        +------------------------+------------+
#                                        |                                     |
#                                  mode logits                          posture code
#                                                                              |
#                                                                     render(posture,
#                                                                       cell light)
#                                                                              |
#                                                                     re-enactment loss
#
# The scaffold is rank-r: h1 += g_c * U_c (V_c x) + g_c * s_c.
# g_c is a single learned scalar per cell, and it is what the extinction terms
# attack. When g -> 0 the network collapses to one shared doctrine core, which
# is the intended end state of the whole convent.
# ==============================================================================

H1, H2, RANK = 40, 28, 4


def init_params(rng):
    def n(shape, scale):
        return rng.normal(0, scale, size=shape)
    p = {
        'W1': n((N_PIX, H1), 1.0 / np.sqrt(N_PIX)),
        'b1': np.zeros(H1),
        'U':  n((N_CELL, RANK, H1), 0.30),          # scaffold up-projection
        'V':  n((N_CELL, N_PIX, RANK), 1.0 / np.sqrt(N_PIX)),
        'sB': np.zeros((N_CELL, H1)),               # scaffold bias
        'g':  np.full(N_CELL, 1.0),                 # scaffold gates
        'W2': n((H1, H2), 1.0 / np.sqrt(H1)),
        'b2': np.zeros(H2),
        'Wm': n((H2, N_MODE), 1.0 / np.sqrt(H2)),
        'bm': np.zeros(N_MODE),
        'Wp': n((H2, N_ANGLES), 0.5 / np.sqrt(H2)),
        'bp': np.zeros(N_ANGLES),
    }
    return p


def forward(p, x, cell, use_scaffold=True):
    """
    x:    (B, N_PIX)
    cell: (B,) integer cell index
    Returns logits (B,K), pose (B,5), cache.
    """
    z1 = x @ p['W1'] + p['b1']
    if use_scaffold:
        Vc = p['V'][cell]                        # (B, N_PIX, RANK)
        r = np.einsum('bi,bir->br', x, Vc)       # (B, RANK)
        Uc = p['U'][cell]                        # (B, RANK, H1)
        scaf = np.einsum('br,brh->bh', r, Uc) + p['sB'][cell]
        g = p['g'][cell][:, None]
        z1 = z1 + g * scaf
    else:
        r = None; scaf = None; g = None
    a1 = np.tanh(z1)
    z2 = a1 @ p['W2'] + p['b2']
    a2 = np.tanh(z2)
    logits = a2 @ p['Wm'] + p['bm']
    pose = a2 @ p['Wp'] + p['bp']
    cache = dict(x=x, cell=cell, z1=z1, a1=a1, z2=z2, a2=a2,
                 r=r, scaf=scaf, g=g, use=use_scaffold)
    return logits, pose, cache


def backward(p, cache, dlogits, dpose, grads):
    """Accumulates gradients into `grads` (a dict of same shape as p)."""
    x, cell, a1, a2 = cache['x'], cache['cell'], cache['a1'], cache['a2']

    grads['Wm'] += a2.T @ dlogits
    grads['bm'] += dlogits.sum(0)
    grads['Wp'] += a2.T @ dpose
    grads['bp'] += dpose.sum(0)

    da2 = dlogits @ p['Wm'].T + dpose @ p['Wp'].T
    dz2 = da2 * (1.0 - a2 * a2)

    grads['W2'] += a1.T @ dz2
    grads['b2'] += dz2.sum(0)

    da1 = dz2 @ p['W2'].T
    dz1 = da1 * (1.0 - a1 * a1)

    grads['W1'] += x.T @ dz1
    grads['b1'] += dz1.sum(0)

    if cache['use']:
        g, scaf, r = cache['g'], cache['scaf'], cache['r']
        # gate gradient, scattered per cell
        dg = np.sum(dz1 * scaf, axis=1)                      # (B,)
        np.add.at(grads['g'], cell, dg)

        dscaf = dz1 * g                                      # (B,H1)
        np.add.at(grads['sB'], cell, dscaf)

        dU = r[:, :, None] * dscaf[:, None, :]               # (B,RANK,H1)
        np.add.at(grads['U'], cell, dU)

        dr = np.einsum('bh,brh->br', dscaf, p['U'][cell])    # (B,RANK)
        dV = x[:, :, None] * dr[:, None, :]                  # (B,N_PIX,RANK)
        np.add.at(grads['V'], cell, dV)

        grads['W1'] += 0.0  # (scaffold path does not touch W1 again)


def zeros_like_params(p):
    return {k: np.zeros_like(v) for k, v in p.items()}


# ==============================================================================
# SECTION 5 -- THE OBJECTIVE
#
# L = L_task(scaffolded)
#   + w_abl * L_task(ABLATED)          <- extinction pressure #1
#   + w_re  * re-enactment loss
#   + w_dec * ||g||^2                  <- extinction pressure #2 (decay)
#
# The ablated pass is the whole moral argument of the chapter expressed as
# arithmetic. Nothing forbids the network from leaning on its scaffolds; it is
# simply also graded on the day the scaffolds are gone.
# ==============================================================================

W_MODE = 1.0
W_POSE = 1.0
W_REENACT = 0.55
L2 = 2e-4                 # ordinary weight decay on the shared doctrine core

# The withdrawal schedule. Early in formation the scaffold is allowed to carry
# almost all the load and is barely taxed; late in formation it is taxed hard
# and the ablated pass is weighted above the scaffolded one. This is not a
# training trick borrowed from elsewhere -- it is the novitiate, expressed as
# two coefficients.
W_ABLATE_0, W_ABLATE_1 = 0.12, 1.80
W_DECAY_0, W_DECAY_1 = 0.0004, 0.320
WITHDRAW_POWER = 1.6


def withdrawal(t):
    """t in [0,1] = fraction of training elapsed. Returns (w_ablate, w_decay)."""
    u = float(np.clip(t, 0.0, 1.0)) ** WITHDRAW_POWER
    return (W_ABLATE_0 + (W_ABLATE_1 - W_ABLATE_0) * u,
            W_DECAY_0 + (W_DECAY_1 - W_DECAY_0) * u)


def softmax_ce(logits, y):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(np.maximum(s[np.arange(n), y], 1e-12)).mean()
    d = s.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


def cell_light(cell):
    la = CELL_LIGHT_ANGLE[cell]
    return np.stack([np.sin(la), np.cos(la)], axis=-1)


def loss_and_grads(p, x, y, pose_t, cell, target_img, grads=None,
                   with_reenact=True, w_ablate=1.0, w_decay=0.02):
    """
    Returns (total_loss, parts_dict). If `grads` is given, fills it in place.
    """
    if grads is None:
        grads = zeros_like_params(p)
    B = x.shape[0]
    parts = {}

    # ---- scaffolded pass
    logits_s, pose_s, cache_s = forward(p, x, cell, use_scaffold=True)
    l_mode_s, dlog_s = softmax_ce(logits_s, y)
    diffp = pose_s - pose_t
    l_pose_s = np.mean(np.sum(diffp * diffp, axis=1))
    dpose_s = (2.0 / B) * diffp

    dlog_s = dlog_s * W_MODE
    dpose_s = dpose_s * W_POSE

    # ---- re-enactment: render the recovered posture under the cell's own light
    l_re = 0.0
    if with_reenact:
        light = cell_light(cell)
        img_hat, rc = render(pose_s, light)
        rdiff = img_hat - target_img
        l_re = np.mean(np.sum(rdiff * rdiff, axis=1))
        dimg = (2.0 / B) * rdiff
        dpose_re = render_backward(dimg, rc)
        dpose_s = dpose_s + W_REENACT * dpose_re

    backward(p, cache_s, dlog_s, dpose_s, grads)

    # ---- ablated pass: the same batch, taught by nobody
    logits_a, pose_a, cache_a = forward(p, x, cell, use_scaffold=False)
    l_mode_a, dlog_a = softmax_ce(logits_a, y)
    diffa = pose_a - pose_t
    l_pose_a = np.mean(np.sum(diffa * diffa, axis=1))
    dpose_a = (2.0 / B) * diffa
    backward(p, cache_a, dlog_a * W_MODE * w_ablate,
             dpose_a * W_POSE * w_ablate, grads)

    # ---- decay penalty on the gates: the scaffold is charged rent
    l_dec = np.sum(p['g'] ** 2)
    grads['g'] += 2.0 * w_decay * p['g']

    # ---- ordinary L2 on the shared core only (never on the scaffolds; they
    #      are already being removed by a stronger mechanism)
    l_l2 = 0.0
    for k in ('W1', 'W2', 'Wm', 'Wp'):
        l_l2 += np.sum(p[k] ** 2)
        grads[k] += 2.0 * L2 * p[k]

    total = (W_MODE * l_mode_s + W_POSE * l_pose_s
             + w_ablate * (W_MODE * l_mode_a + W_POSE * l_pose_a)
             + W_REENACT * l_re + w_decay * l_dec + L2 * l_l2)

    parts.update(mode_s=l_mode_s, pose_s=l_pose_s, mode_a=l_mode_a,
                 pose_a=l_pose_a, reenact=l_re, decay=l_dec,
                 w_ablate=w_ablate, w_decay=w_decay, total=total)
    return total, parts


# ==============================================================================
# SECTION 6 -- TRAINING (Adam, written out)
# ==============================================================================

def train(p, data, epochs=140, batch=64, lr=6e-3, rng=None, verbose=True):
    X, Y, P, C, T = data
    n = X.shape[0]
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        wa, wd = withdrawal(ep / max(1, epochs - 1))
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch):
            sel = idx[s:s + batch]
            grads = zeros_like_params(p)
            tot, parts = loss_and_grads(p, X[sel], Y[sel], P[sel], C[sel],
                                        T[sel], grads=grads,
                                        w_ablate=wa, w_decay=wd)
            step += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                mh = m[k] / (1 - b1 ** step)
                vh = v[k] / (1 - b2 ** step)
                p[k] -= lr * mh / (np.sqrt(vh) + eps)
            ep_loss += tot
            nb += 1
        hist.append(ep_loss / nb)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            acc_s = accuracy(p, X, Y, C, True)
            acc_a = accuracy(p, X, Y, C, False)
            print(f"  epoch {ep:4d}  loss {ep_loss/nb:8.4f}   "
                  f"acc(scaffold) {acc_s:.3f}   acc(ablated) {acc_a:.3f}   "
                  f"mean|g| {np.abs(p['g']).mean():.4f}   "
                  f"w_abl {wa:.2f}  w_dec {wd:.3f}")
    return hist


def accuracy(p, X, Y, C, use_scaffold):
    logits, _, _ = forward(p, X, C, use_scaffold=use_scaffold)
    return float(np.mean(np.argmax(logits, axis=1) == Y))


def pose_rmse(p, X, P, C, use_scaffold):
    _, pose, _ = forward(p, X, C, use_scaffold=use_scaffold)
    return float(np.sqrt(np.mean((pose - P) ** 2)))


# ==============================================================================
# SECTION 7 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
#
# Every parameter group, plus a separate check on the renderer's own
# derivative. If this section does not pass, nothing else in the file is worth
# reading.
# ==============================================================================

def grad_check(verbose=True):
    rng = np.random.default_rng(7)
    p = init_params(rng)
    B = 6
    X = rng.normal(0, 0.6, size=(B, N_PIX))
    Y = rng.integers(0, N_MODE, size=B)
    P = rng.normal(0, 0.4, size=(B, N_ANGLES))
    C = rng.integers(0, N_CELL, size=B)
    T = rng.random((B, N_PIX)) * 0.7

    grads = zeros_like_params(p)
    loss_and_grads(p, X, Y, P, C, T, grads=grads)

    eps = 1e-6
    worst = 0.0
    rows = []
    for k in p:
        flat = p[k].ravel()
        gflat = grads[k].ravel()
        picks = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        errs = []
        for i in picks:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = loss_and_grads(p, X, Y, P, C, T)
            flat[i] = orig - eps
            lm, _ = loss_and_grads(p, X, Y, P, C, T)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        rows.append((k, e))

    # renderer alone
    ang = rng.normal(0, 0.5, size=(3, N_ANGLES))
    light = cell_light(rng.integers(0, N_CELL, size=3))
    W = rng.normal(0, 1.0, size=(3, N_PIX))

    def f(a):
        img, _ = render(a, light)
        return np.sum(W * img)

    img, rc = render(ang, light)
    dang = render_backward(W, rc)
    rerr = 0.0
    for i in range(3):
        for j in range(N_ANGLES):
            o = ang[i, j]
            ang[i, j] = o + eps; lp = f(ang)
            ang[i, j] = o - eps; lm = f(ang)
            ang[i, j] = o
            num = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(num) + abs(dang[i, j]))
            rerr = max(rerr, abs(num - dang[i, j]) / denom)

    if verbose:
        print("  finite-difference check (relative error, central differences)")
        for k, e in rows:
            print(f"    {k:>4s}  {e:.3e}")
        print(f"    renderer d(image)/d(angles)   {rerr:.3e}")
    ok = worst < 1e-4 and rerr < 1e-4
    return ok, worst, rerr


# ==============================================================================
# SECTION 8 -- THE MEASUREMENTS THAT MATTER
# ==============================================================================

def extinction_report(p, data):
    """
    Extinction ratio = ablated performance / scaffolded performance.
    1.0 or better means the teacher has successfully deleted itself.
    """
    X, Y, P, C, T = data
    out = {}
    for c in range(N_CELL):
        m = (C == c)
        a_s = accuracy(p, X[m], Y[m], C[m], True)
        a_a = accuracy(p, X[m], Y[m], C[m], False)
        r_s = pose_rmse(p, X[m], P[m], C[m], True)
        r_a = pose_rmse(p, X[m], P[m], C[m], False)
        out[CELL_NAMES[c]] = dict(acc_scaf=a_s, acc_abl=a_a,
                                  ratio=a_a / max(a_s, 1e-9),
                                  rmse_scaf=r_s, rmse_abl=r_a,
                                  gate=float(p['g'][c]))
    return out


def window_invariance(p, rng, n=240):
    """
    Take one posture and relight it from every cell's window. If the doctrine
    core is genuinely registered to the room rather than confused by it, the
    recovered mode should not change when only the light changes.
    """
    protos = canonical_modes(rng)
    modes = rng.integers(0, N_MODE, size=n)
    ang = protos[modes] + rng.normal(0, 0.10, size=(n, N_ANGLES))
    preds = []
    for c in range(N_CELL):
        la = CELL_LIGHT_ANGLE[c]
        light = np.broadcast_to(np.array([np.sin(la), np.cos(la)]), (n, 2))
        img, _ = render(ang, light)
        cc = np.full(n, c)
        logits, _, _ = forward(p, img, cc, use_scaffold=False)
        preds.append(np.argmax(logits, axis=1))
    preds = np.stack(preds)                     # (C, n)
    agree = float(np.mean(np.all(preds == preds[0:1], axis=0)))
    corr = float(np.mean(preds == modes[None, :]))
    return agree, corr


def transfer_to_unseen_receiver(p, rng, n=400):
    """
    A receiver the convent never had: a new corruption profile, and no scaffold
    of its own. Only the extinguished doctrine core is available. This is the
    test the architecture is actually built to pass.
    """
    protos = canonical_modes(rng)
    modes = rng.integers(0, N_MODE, size=n)
    ang = protos[modes] + rng.normal(0, 0.13, size=(n, N_ANGLES))
    la = 1.75
    light = np.broadcast_to(np.array([np.sin(la), np.cos(la)]), (n, 2))
    clean, _ = render(ang, light)
    obs = clean * 0.80 + rng.normal(0, 0.30, size=clean.shape)
    obs = obs * (rng.random(clean.shape) > 0.20)
    cc = np.zeros(n, dtype=int)   # index is irrelevant: scaffold is off
    logits, pose, _ = forward(p, obs, cc, use_scaffold=False)
    acc = float(np.mean(np.argmax(logits, axis=1) == modes))
    rmse = float(np.sqrt(np.mean((pose - ang) ** 2)))
    return acc, rmse


def reenactment_fidelity(p, data):
    """How close is the re-performed body to the observed one?"""
    X, Y, P, C, T = data
    _, pose, _ = forward(p, X, C, use_scaffold=False)
    img, _ = render(pose, cell_light(C))
    return float(np.sqrt(np.mean((img - T) ** 2)))


def ascii_body(ang, light_angle):
    """A crude console rendering, so the postures can be eyeballed."""
    la = np.array([[np.sin(light_angle), np.cos(light_angle)]])
    img, _ = render(ang[None, :], la)
    g = img.reshape(GRID, GRID)
    ramp = " .:-=+*#%@"
    lines = []
    for row in g:
        lines.append("".join(ramp[min(len(ramp) - 1, int(v * len(ramp)))] for v in row))
    return lines


# ==============================================================================
# SECTION 9 -- SELF-TESTS
# ==============================================================================

def self_tests(p, data, rng):
    X, Y, P, C, T = data
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    acc_s = accuracy(p, X, Y, C, True)
    acc_a = accuracy(p, X, Y, C, False)
    check("scaffolded accuracy above chance", acc_s > 2.0 / N_MODE,
          f"{acc_s:.3f} vs chance {1/N_MODE:.3f}")
    check("ablated accuracy above chance", acc_a > 2.0 / N_MODE,
          f"{acc_a:.3f}")
    check("extinction ratio >= 0.95", acc_a / max(acc_s, 1e-9) >= 0.95,
          f"{acc_a/max(acc_s,1e-9):.3f}")
    check("scaffold gates decayed below init", np.abs(p['g']).mean() < 1.0,
          f"mean|g| = {np.abs(p['g']).mean():.4f}")

    agree, corr = window_invariance(p, rng)
    check("window invariance: same posture read the same under all 4 windows",
          agree > 0.80, f"agreement {agree:.3f}, correct {corr:.3f}")

    t_acc, t_rmse = transfer_to_unseen_receiver(p, rng)
    check("transfers to an unseen receiver with no scaffold", t_acc > 0.55,
          f"acc {t_acc:.3f}, pose rmse {t_rmse:.3f}")

    rf = reenactment_fidelity(p, data)
    check("re-enactment reproduces the observed body", rf < 0.22,
          f"render rmse {rf:.4f}")

    r_s = pose_rmse(p, X, P, C, True)
    r_a = pose_rmse(p, X, P, C, False)
    check("posture recovery survives ablation", r_a < r_s * 1.25,
          f"rmse {r_s:.4f} -> {r_a:.4f}")

    return results


# ==============================================================================
# SECTION 10 -- MAIN
# ==============================================================================

def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()

    print("=" * 78)
    print("  THE CELL COMPILER")
    print(" Fra Angelico (c.1395-1455): an addressee-compiled,")
    print(" self-extinguishing scaffold network with re-enactment decoding")
    print("=" * 78)

    print("\n[1] GRADIENT CHECK")
    ok, worst, rerr = grad_check()
    print(f"  worst relative error over all parameter groups: {worst:.3e}")
    print(f"  PASS" if ok else "  FAIL")
    if not ok:
        raise SystemExit("gradient check failed")

    print("\n[2] BUILDING THE CONVENT")
    n_per = 60 if quick else 420
    data = make_dataset(n_per, rng)
    test = make_dataset(80 if quick else 180, rng)
    print(f"  cells: {', '.join(CELL_NAMES)}")
    print(f"  modes of prayer: {N_MODE}   image: {GRID}x{GRID}   "
          f"train n={data[0].shape[0]}  test n={test[0].shape[0]}")
    print(f"  window azimuths (rad): "
          f"{', '.join(f'{a:+.2f}' for a in CELL_LIGHT_ANGLE)}")

    print("\n[3] TRAINING")
    p = init_params(rng)
    train(p, data, epochs=30 if quick else 220, batch=64, lr=6e-3, rng=rng)

    print("\n[4] EXTINCTION REPORT (held-out set)")
    rep = extinction_report(p, test)
    print(f"  {'cell':<13}{'acc+scaffold':>13}{'acc-ablated':>13}"
          f"{'ratio':>8}{'gate':>9}")
    for k, v in rep.items():
        print(f"  {k:<13}{v['acc_scaf']:>13.3f}{v['acc_abl']:>13.3f}"
              f"{v['ratio']:>8.3f}{v['gate']:>9.4f}")

    print("\n[5] DIAGNOSTICS")
    agree, corr = window_invariance(p, rng)
    t_acc, t_rmse = transfer_to_unseen_receiver(p, rng)
    rf = reenactment_fidelity(p, test)
    print(f"  window invariance (same reading under all 4 windows) : {agree:.3f}")
    print(f"  mode correct under relighting                        : {corr:.3f}")
    print(f"  unseen receiver, no scaffold: accuracy               : {t_acc:.3f}")
    print(f"  unseen receiver, no scaffold: posture rmse           : {t_rmse:.3f}")
    print(f"  re-enactment render rmse (ablated)                   : {rf:.4f}")
    print(f"  mean |gate| after training                           : "
          f"{np.abs(p['g']).mean():.4f}  (init 1.0)")

    print("\n[6] A RECOVERED POSTURE, RE-ENACTED")
    X, Y, P, C, T = test
    i = int(np.argmax(C == 0))
    _, pose, _ = forward(p, X[i:i+1], C[i:i+1], use_scaffold=False)
    obs_lines = ascii_body(P[i], CELL_LIGHT_ANGLE[C[i]])
    re_lines = ascii_body(pose[0], CELL_LIGHT_ANGLE[C[i]])
    print(f"  cell: {CELL_NAMES[C[i]]}   true mode: {Y[i]}")
    print("     observed body            re-enacted body")
    for a, b in zip(obs_lines, re_lines):
        print(f"     {a}      {b}")

    print("\n[7] SELF-TESTS")
    results = self_tests(p, test, rng)
    npass = 0
    for name, okk, detail in results:
        print(f"  [{'PASS' if okk else 'FAIL'}] {name}"
              + (f"   ({detail})" if detail else ""))
        npass += int(okk)
    print(f"\n  {npass}/{len(results)} self-tests passed")
    print(f"  total runtime: {time.time() - t0:.1f}s")
    print("=" * 78)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

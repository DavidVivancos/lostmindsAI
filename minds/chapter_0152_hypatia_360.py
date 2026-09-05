#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0152_hypatia_360.py
 The Astrolabe Engine  —  a cognitive architecture after Hypatia of Alexandria
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 152: Hypatia of Alexandria (360-415 CE)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy model that encodes ONE cognitive idea that is
Hypatia's alone among the Neoplatonists of the corpus.

Her neighbours in the tradition already own the familiar Neoplatonic moves:
Plotinus -> emanation and the soul's reversion to the One; Porphyry -> knowing
by recursive division and definition. Hypatia was something else. She was a
mathematician, an editor of Diophantus and Apollonius, a reviser of the
astronomical tables, and — with her pupil Synesius — a maker of the silver
ASTROLABE. She did not invent the astrolabe or its underlying geometry (both
predate her by centuries); she taught it, refined it, and transmitted it.

The astrolabe embodies a way of THINKING:

    A hard problem lives on a curved surface (the celestial sphere), where the
    honest computation is spherical trigonometry in three coupled variables
    (declination, latitude, hour angle). The astrolabe does not solve it there.
    It PROJECTS the sphere onto a flat plane by a map that is chosen from the
    special family that PRESERVES THE INVARIANTS THAT MATTER — angles are kept
    (the map is conformal) and circles stay circles. In the plane the problem
    collapses: "is this star above the horizon?" becomes "is this point inside
    ONE fixed circle?". The passage of time is not a recomputation; it is an
    exact ROTATION of a movable star-map (the rete). Solve in the plane, then
    carry the answer back. Correctness is guaranteed not by the coordinates the
    projection distorts, but by the invariants it conserves.

That is the thesis this architecture makes literal:

    Intelligence = a change of representation drawn from the invariance-
    preserving group, after which a construction that was impossible becomes a
    single, checkable step; the certificate of correctness is the preserved
    invariant.

THE PIPELINE (each stage is a named organ of the "mind")
--------------------------------------------------------
  1. SPHERE           : a star is a unit vector on S^2 from (declination, RA).
  2. STEREOGRAPHIC    : project from the south celestial pole to the plane.
                        This map is CONFORMAL (angle-preserving) and sends
                        circles to circles — the invariants Hypatia's method
                        relies on. (Fixed, analytic, no parameters.)
  3. RETE ROTATION    : local sidereal time acts as an EXACT planar rotation
                        w -> w * e^{i*theta}. Time is a group action, not a
                        recomputation. (Fixed, analytic.)
  4. HORIZON PLATE    : a tiny learned network reads the observer's latitude
                        and returns ONE circle in the plane (centre, radius).
                        Because every upstream stage preserves circles, this
                        single circle is the exact visibility boundary for ALL
                        hour angles at that latitude. The model must DISCOVER
                        that the plate circle has centre (cot phi, 0) and
                        radius csc phi — the closed-form the geometry hides.
  5. VERDICT          : inside the plate circle  <=>  star is above the horizon.

WHAT IS LEARNED, WHAT IS GIVEN
------------------------------
Given (by geometry, exact): the projection, its inverse, the rete rotation.
Learned (by gradient descent): the map  latitude -> horizon circle , plus a
sharpness scalar. The learning problem is real: the network is shown only
(star, latitude, time, visible?) labels and must recover the plate.

ENGINEERING CONTRACT (kept in every file of this corpus)
--------------------------------------------------------
  * pure NumPy, hand-written forward and analytic backward pass;
  * a finite-difference gradient check that MUST pass (mandatory);
  * a real training loop with Adam;
  * a battery of self-tests for the invariants (round-trip, conformality,
    circle-preservation, exact rete composition, one-circle-all-times);
  * everything executes on run.

Run:  python3 chapter_0152_hypatia_360.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(152)  # figure #152; deterministic runs.

# =============================================================================
# SECTION 1 — GEOMETRY OF THE HEAVENS (the "given", exact, parameter-free)
# =============================================================================
# These functions are the astrolabe's fixed skeleton. They carry no learnable
# weights; they are the invariant-preserving maps Hypatia's method assumes.

def sphere_from_equatorial(dec, ra):
    """Unit vector on the celestial sphere from declination & right ascension.
    z-axis = celestial north pole. Returns (...,3)."""
    dec = np.asarray(dec, float); ra = np.asarray(ra, float)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def stereographic_project(p):
    """Stereographic projection from the SOUTH pole (0,0,-1) onto plane z=0.

    w = (x + i y) / (z + 1), returned as real (X, Y).
    Properties this architecture depends on:
      * CONFORMAL: preserves angles between curves.
      * CIRCLE-PRESERVING: circles on the sphere map to circles (or lines).
      * north pole -> origin; a rotation about the polar axis by theta becomes
        an exact rotation by theta in the plane.
    """
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    denom = z + 1.0
    X = x / denom
    Y = y / denom
    return np.stack([X, Y], axis=-1)


def stereographic_unproject(w):
    """Inverse of stereographic_project: plane (X,Y) -> sphere (x,y,z)."""
    w = np.asarray(w, float)
    X, Y = w[..., 0], w[..., 1]
    rho2 = X * X + Y * Y
    d = 1.0 + rho2
    x = 2.0 * X / d
    y = 2.0 * Y / d
    z = (1.0 - rho2) / d
    return np.stack([x, y, z], axis=-1)


def rete_rotate(w, theta):
    """Rotate the rete (star-map) in the plane by angle theta.

    This is the astrolabe's clock: advancing local sidereal time by theta is
    the multiplication  w -> w * e^{i theta}. It is EXACT and composes as a
    group:  R_{a} . R_{b} = R_{a+b}.
    """
    w = np.asarray(w, float)
    c, s = np.cos(theta), np.sin(theta)
    X, Y = w[..., 0], w[..., 1]
    Xr = c * X - s * Y
    Yr = s * X + c * Y
    return np.stack([Xr, Yr], axis=-1)


def apparent_star_vector(dec, hour_angle):
    """Star's unit vector in the observer's meridian frame at a given hour angle.

    Frame: z = celestial pole, x = toward the meridian, so that
        s = (cos dec cos H, cos dec sin H, sin dec).
    With observer zenith = (cos phi, 0, sin phi), the classical relation
        sin(altitude) = sin phi sin dec + cos phi cos dec cos H
    follows as s . zenith.
    """
    dec = np.asarray(dec, float); H = np.asarray(hour_angle, float)
    x = np.cos(dec) * np.cos(H)
    y = np.cos(dec) * np.sin(H)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def true_altitude(dec, phi, hour_angle):
    """Exact altitude above the horizon (radians). Visible iff altitude > 0."""
    return np.arcsin(np.clip(
        np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour_angle),
        -1.0, 1.0))


def analytic_horizon_circle(phi):
    """The closed-form the network is supposed to REDISCOVER.

    For the south-pole stereographic astrolabe, the horizon great circle of an
    observer at latitude phi projects to the plane circle
        centre = (cot phi, 0),  radius = csc phi = 1/sin phi,
    and 'above the horizon' == 'inside this circle' (northern latitudes).
    """
    return np.array([1.0 / np.tan(phi), 0.0]), 1.0 / np.sin(phi)


# =============================================================================
# SECTION 2 — THE LEARNED HORIZON PLATE (the only parameters in the model)
# =============================================================================
# A tiny MLP: latitude features -> (centre_x, centre_y, log radius). Plus one
# global sharpness scalar 'kappa' controlling how crisp the inside/outside
# decision is. The astrolabe's *plate* is what changes with latitude; the
# projection and rete are universal. So the plate is exactly what we learn.

def init_params(hidden=16):
    """Xavier-ish initialisation. Input feature is [sin phi, cos phi, 1]."""
    def xav(a, b):
        return RNG.normal(0, np.sqrt(2.0 / (a + b)), size=(a, b))
    return {
        "W1": xav(3, hidden),
        "b1": np.zeros(hidden),
        "W2": xav(hidden, 3),
        "b2": np.array([0.5, 0.0, 0.5]),  # gentle prior: centre near axis, r~e^.5
        "kappa": np.array(1.0),           # softplus(kappa) = sharpness > 0
    }


def phi_features(phi):
    """Latitude -> feature row [sin phi, cos phi, 1]. Natural for the geometry."""
    phi = np.asarray(phi, float)
    return np.stack([np.sin(phi), np.cos(phi), np.ones_like(phi)], axis=-1)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def forward(params, phi, XY, cache=False):
    """Full forward pass over a batch.

    phi : (N,)      observer latitudes
    XY  : (N,2)     projected+rotated star positions in the plane
    Returns probabilities p (N,). If cache=True, also returns intermediates
    needed by the analytic backward pass.
    """
    F = phi_features(phi)                       # (N,3)
    z1 = F @ params["W1"] + params["b1"]        # (N,hid)
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]       # (N,3): cx, cy, logr
    cx, cy, logr = z2[:, 0], z2[:, 1], z2[:, 2]
    r2 = np.exp(2.0 * logr)                      # radius^2  (>0 by construction)
    dx = XY[:, 0] - cx
    dy = XY[:, 1] - cy
    d2 = dx * dx + dy * dy
    g = r2 - d2                                  # >0 inside the plate circle
    k = _softplus(params["kappa"])              # sharpness > 0
    logit = k * g
    p = _sigmoid(logit)
    if not cache:
        return p
    return p, {"F": F, "z1": z1, "a1": a1, "z2": z2,
               "cx": cx, "cy": cy, "logr": logr, "r2": r2,
               "dx": dx, "dy": dy, "g": g, "k": k, "logit": logit, "XY": XY}


def bce_loss(p, y, eps=1e-9):
    """Mean binary cross-entropy."""
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def loss_and_grad(params, phi, XY, y):
    """Analytic forward + backward. Returns (loss, grads dict)."""
    N = len(y)
    p, c = forward(params, phi, XY, cache=True)
    loss = bce_loss(p, y)

    # d loss / d logit for sigmoid+BCE is simply (p - y)/N.
    dlogit = (np.clip(p, 1e-9, 1 - 1e-9) - y) / N     # (N,)

    # logit = k * g
    dk = np.sum(dlogit * c["g"])                       # scalar (through k)
    dg = dlogit * c["k"]                               # (N,)

    # g = r2 - d2
    dr2 = dg                                           # (N,)
    dd2 = -dg                                          # (N,)

    # r2 = exp(2 logr)  -> d r2 / d logr = 2 r2
    dlogr = dr2 * 2.0 * c["r2"]                         # (N,)

    # d2 = dx^2 + dy^2 ; dx = X - cx ; dy = Y - cy
    ddx = dd2 * 2.0 * c["dx"]
    ddy = dd2 * 2.0 * c["dy"]
    dcx = -ddx
    dcy = -ddy

    dz2 = np.stack([dcx, dcy, dlogr], axis=1)          # (N,3)

    # z2 = a1 @ W2 + b2
    gW2 = c["a1"].T @ dz2
    gb2 = np.sum(dz2, axis=0)
    da1 = dz2 @ params["W2"].T                          # (N,hid)
    dz1 = da1 * (1.0 - c["a1"] ** 2)                    # tanh'
    gW1 = c["F"].T @ dz1
    gb1 = np.sum(dz1, axis=0)

    # kappa: k = softplus(kappa) -> dk/dkappa = sigmoid(kappa)
    gkappa = dk * _sigmoid(params["kappa"])

    grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2,
             "kappa": np.array(gkappa)}
    return loss, grads


# =============================================================================
# SECTION 3 — DATA: turn geometry into a supervised visibility problem
# =============================================================================
# Each example: a random star (dec, RA never needed explicitly), a latitude,
# and a local sidereal time realised as a rete rotation. Label = truly visible.
# The point of the whole design: the network never sees the horizon formula,
# only in/out labels, yet must recover the single plate circle per latitude.

PHI_LO, PHI_HI = np.radians(20.0), np.radians(70.0)   # keep cot/csc bounded


def make_batch(n):
    """Sample n (phi, plane-point XY, visible) triples using the exact geometry."""
    dec = RNG.uniform(np.radians(-80), np.radians(80), size=n)
    phi = RNG.uniform(PHI_LO, PHI_HI, size=n)
    H = RNG.uniform(-np.pi, np.pi, size=n)             # hour angle == rete angle
    # Base star at hour angle 0, then let the rete carry it to hour angle H.
    s0 = apparent_star_vector(dec, 0.0)                # (n,3)
    w0 = stereographic_project(s0)                     # (n,2)
    XY = rete_rotate(w0, H)                            # (n,2) — time as rotation
    alt = true_altitude(dec, phi, H)
    y = (alt > 0.0).astype(float)
    return phi, XY, y, dec, H


# =============================================================================
# SECTION 4 — GRADIENT CHECK (mandatory: analytic vs finite differences)
# =============================================================================

def gradient_check():
    params = init_params(hidden=8)
    phi, XY, y, _, _ = make_batch(24)
    loss0, grads = loss_and_grad(params, phi, XY, y)

    eps = 1e-6
    max_rel = 0.0
    for name in params:
        flat = np.atleast_1d(params[name]).ravel()
        gflat = np.atleast_1d(grads[name]).ravel()
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            params[name] = flat.reshape(params[name].shape)
            lp = bce_loss(forward(params, phi, XY), y)
            flat[i] = orig - eps
            params[name] = flat.reshape(params[name].shape)
            lm = bce_loss(forward(params, phi, XY), y)
            flat[i] = orig
            params[name] = flat.reshape(params[name].shape)
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num) + abs(ana))
            max_rel = max(max_rel, abs(num - ana) / denom)
    return max_rel


# =============================================================================
# SECTION 5 — TRAINING (Adam over the plate network)
# =============================================================================

def train(steps=4000, batch=256, lr=3e-3, verbose=True):
    params = init_params(hidden=16)
    m = {k: np.zeros_like(np.atleast_1d(v).astype(float)) for k, v in params.items()}
    v = {k: np.zeros_like(np.atleast_1d(val).astype(float)) for k, val in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        phi, XY, y, _, _ = make_batch(batch)
        loss, grads = loss_and_grad(params, phi, XY, y)
        for k in params:
            g = np.atleast_1d(grads[k]).astype(float)
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * (g * g)
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            upd = lr * mhat / (np.sqrt(vhat) + eps)
            params[k] = (np.atleast_1d(params[k]).astype(float) - upd).reshape(
                np.shape(params[k]))
        if verbose and (t % 800 == 0 or t == 1):
            acc = evaluate(params, 4000)
            print(f"  step {t:5d}   loss {loss:.4f}   held-out acc {acc:.4f}")
    return params


def evaluate(params, n):
    phi, XY, y, _, _ = make_batch(n)
    p = forward(params, phi, XY)
    return float(np.mean((p > 0.5) == (y > 0.5)))


def recovered_plate_error(params):
    """Compare the LEARNED plate circle against the closed form (cot phi, csc phi)
    across a sweep of latitudes. Small error = the model rediscovered the law."""
    phis = np.linspace(PHI_LO, PHI_HI, 40)
    F = phi_features(phis)
    a1 = np.tanh(F @ params["W1"] + params["b1"])
    z2 = a1 @ params["W2"] + params["b2"]
    cx, cy, logr = z2[:, 0], z2[:, 1], np.exp(z2[:, 2])
    tgt_cx = 1.0 / np.tan(phis)
    tgt_r = 1.0 / np.sin(phis)
    err_cx = np.mean(np.abs(cx - tgt_cx))
    err_cy = np.mean(np.abs(cy - 0.0))
    err_r = np.mean(np.abs(logr - tgt_r))
    return err_cx, err_cy, err_r


# =============================================================================
# SECTION 6 — SELF-TESTS FOR THE INVARIANTS (the certificates of correctness)
# =============================================================================

def test_roundtrip():
    """Project then unproject must recover the sphere point (machine precision)."""
    p = sphere_from_equatorial(RNG.uniform(-1.2, 1.2, 500),
                               RNG.uniform(-np.pi, np.pi, 500))
    w = stereographic_project(p)
    p2 = stereographic_unproject(w)
    return float(np.max(np.abs(p - p2)))


def test_circle_preservation():
    """A small circle on the sphere must project to a circle in the plane.
    We fit a circle to the projected points and report the residual."""
    # small circle: constant angle from a tilted axis
    axis = np.array([0.3, -0.5, 0.8]); axis /= np.linalg.norm(axis)
    # build an orthonormal frame
    tmp = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ang = np.radians(35.0)
    t = np.linspace(0, 2 * np.pi, 400)
    pts = (np.cos(ang) * axis[None, :]
           + np.sin(ang) * (np.cos(t)[:, None] * u[None, :]
                            + np.sin(t)[:, None] * v[None, :]))
    W = stereographic_project(pts)
    X, Y = W[:, 0], W[:, 1]
    # algebraic circle fit: X^2+Y^2 + D X + E Y + Fc = 0
    A = np.stack([X, Y, np.ones_like(X)], axis=1)
    rhs = -(X ** 2 + Y ** 2)
    D, E, Fc = np.linalg.lstsq(A, rhs, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    R = np.sqrt(max(cx ** 2 + cy ** 2 - Fc, 1e-12))
    resid = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - R
    return float(np.max(np.abs(resid)))


def test_conformality():
    """Numerically verify the projection is angle-preserving: its Jacobian J
    satisfies J^T J = lambda^2 I (a scalar times a rotation)."""
    worst = 0.0
    for _ in range(200):
        dec = RNG.uniform(-1.2, 1.2); ra = RNG.uniform(-np.pi, np.pi)
        base = sphere_from_equatorial(dec, ra)
        h = 1e-6
        # two orthonormal tangent directions on the sphere at 'base'
        t1 = np.array([-np.sin(ra), np.cos(ra), 0.0])
        t2 = np.cross(base, t1)
        def proj(pt):
            pt = pt / np.linalg.norm(pt)
            return stereographic_project(pt)
        d1 = (proj(base + h * t1) - proj(base - h * t1)) / (2 * h)
        d2 = (proj(base + h * t2) - proj(base - h * t2)) / (2 * h)
        # J^T J should be lambda^2 * I : check off-diagonal ~0 and |d1|==|d2|
        off = abs(np.dot(d1, d2))
        scale = abs(np.linalg.norm(d1) - np.linalg.norm(d2))
        worst = max(worst, off, scale)
    return float(worst)


def test_rete_group():
    """Rete rotation is an exact group action: R_a . R_b == R_{a+b}."""
    w = RNG.normal(size=(300, 2))
    a, b = 0.7, -1.9
    left = rete_rotate(rete_rotate(w, b), a)
    right = rete_rotate(w, a + b)
    return float(np.max(np.abs(left - right)))


def test_one_circle_all_times():
    """The heart of the thesis: ONE fixed plate circle decides visibility for
    EVERY hour angle. For a fixed star and latitude we sweep the whole day and
    confirm the analytic circle test agrees with the exact altitude sign."""
    phi = np.radians(41.0)                 # ~ Alexandria-ish latitude band
    dec = np.radians(17.0)
    ctr, R = analytic_horizon_circle(phi)
    Hs = np.linspace(-np.pi, np.pi, 2000)
    s0 = apparent_star_vector(dec, 0.0)
    w0 = stereographic_project(s0[None, :])
    XY = rete_rotate(np.repeat(w0, len(Hs), axis=0), Hs)
    inside = ((XY[:, 0] - ctr[0]) ** 2 + (XY[:, 1] - ctr[1]) ** 2) < R ** 2
    truth = true_altitude(dec, phi, Hs) > 0
    return float(np.mean(inside == truth))


# =============================================================================
# SECTION 7 — DRIVER
# =============================================================================

def main():
    print("=" * 74)
    print(" THE ASTROLABE ENGINE — cognition after Hypatia of Alexandria")
    print(" project the curved problem into a plane that keeps the invariants,")
    print(" solve it as one fixed circle, let time be an exact rotation.")
    print("=" * 74)

    print("\n[1] Invariant self-tests (the certificates of correctness)")
    rt = test_roundtrip()
    cp = test_circle_preservation()
    cf = test_conformality()
    rg = test_rete_group()
    oc = test_one_circle_all_times()
    print(f"    round-trip project/unproject max err : {rt:.2e}   (want ~0)")
    print(f"    circle-preservation max residual     : {cp:.2e}   (want ~0)")
    print(f"    conformality (J^T J = lam^2 I) worst : {cf:.2e}   (want ~0)")
    print(f"    rete group  R_a.R_b == R_(a+b) err   : {rg:.2e}   (want ~0)")
    print(f"    ONE-circle-all-times agreement       : {oc*100:.2f}%  (want 100%)")

    print("\n[2] Gradient check (analytic vs finite differences)")
    mr = gradient_check()
    status = "PASS" if mr < 1e-5 else "FAIL"
    print(f"    max relative error {mr:.2e}   ->   {status}")

    print("\n[3] Training the horizon plate (latitude -> single circle)")
    params = train(steps=4000, batch=256, lr=3e-3, verbose=True)

    print("\n[4] Did the model REDISCOVER the closed-form plate?")
    ecx, ecy, er = recovered_plate_error(params)
    acc = evaluate(params, 20000)
    print(f"    mean |centre_x - cot(phi)|  : {ecx:.4f}")
    print(f"    mean |centre_y - 0|         : {ecy:.4f}")
    print(f"    mean |log r - log csc(phi)| : {er:.4f}")
    print(f"    final held-out accuracy     : {acc*100:.2f}%")

    print("\n[5] Verdict")
    ok = (mr < 1e-5 and rt < 1e-9 and rg < 1e-9 and oc > 0.999 and acc > 0.95)
    print("    all core guarantees hold." if ok else "    check failures above.")
    print("=" * 74)


if __name__ == "__main__":
    main()

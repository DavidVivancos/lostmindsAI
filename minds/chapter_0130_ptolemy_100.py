"""
================================================================================
The Equant Engine  --  a cognitive architecture after Claudius Ptolemy
(c. 100 - c. 170 CE, Alexandria, Roman Egypt)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 130: Claudius Ptolemy (c. 100 - c. 170 CE)
================================================================================   

WHY THIS ARCHITECTURE, AND WHY IT IS NOT A TRANSFORMER
------------------------------------------------------
Almost every modern network stores keys and lets a query attend over them.
Ptolemy's distinctive cognitive move is the opposite: he does not add memory,
he *moves the observer*. Faced with the planets -- which loop backwards
(retrograde), speed up, slow down, and brighten -- he refused to make the law
more complicated. Instead he asked a stranger question: from *what point in
space* would this tangle look simple?

His answer was the EQUANT. Uniform circular motion, the sacred axiom of Greek
astronomy, held only about a special displaced point that is neither where you
stand (the Earth) nor the geometric centre of the circle (the deferent centre).
Seen from the equant the epicycle-carrier sweeps out equal angles in equal
times; seen from anywhere else it does not. Ptolemy was willing to sacrifice a
beloved principle (uniformity about the centre) to preserve a deeper one
(uniformity about *some* point). Cognition, for him, is the search for that
hidden vantage.

His second move comes from the Harmonics and the essay On the Criterion:
knowledge lives where two criteria agree. Perception "discovers the approximate
and accepts the exact"; reason "accepts the approximate and discovers the
exact." The ear proposes a rough ratio; reason snaps it to a simple whole-number
ratio; the ear then assents -- or the snap is rejected. The two criteria must
not contradict.

This file turns those two ideas into a small, from-scratch, pure-NumPy learner:

  1. FRAME SEARCH (the equant).  The model represents apparent motion as a
     deferent + epicycle, but its defining, learnable organ is a displaced
     reference frame. Learning "e" (the equant eccentricity) is literally
     learning where to stand so that the sweep becomes uniform. The equant's
     computational fingerprint is an equation-of-centre term, +2e sin(A), that
     turns uniform-at-the-equant motion into non-uniform-at-the-Earth motion.
     (To first order this reproduces Kepler's equal-area law -- a real property
     of the historical device, not a flourish.)

  2. TWO-CRITERION LOSS.  A perception term fits the data (a circular,
     wrap-around MSE on apparent longitude). A reason term is a harmonic comb
     that pulls the deferent/epicycle frequency RATIO toward simple rationals
     -- a differentiable "snap to a whole-number ratio," which is exactly the
     Harmonics move and, physically, a bias toward orbital resonance. A single
     weight beta holds the two criteria in tension so neither runs away.

The learned parts:
    R           deferent radius (fixed = 1, sets the scale)
    r           epicycle radius              (learned)
    e           EQUANT eccentricity          (learned)  <- the heart of the model
    omega_def   deferent mean rate at equant (learned)
    phi_def     deferent phase               (learned)
    omega_ep    epicycle rate                (learned)
    phi_ep      epicycle phase               (learned)

Everything is hand-differentiated; a finite-difference gradient check is run at
start-up and MUST pass. There is a real training loop on genuine retrograde data
(a geocentric projection of two circular heliocentric orbits -- i.e. real
Kepler-style motion, the same thing the historical equant was approximating),
plus self-tests for retrograde reproduction, held-out forecasting, and the
reason criterion's effect.

Run:  python3 chapter_0130_ptolemy_100.py
================================================================================
"""

import numpy as np

# Reproducibility: one seed for the whole "cosmos".
RNG = np.random.default_rng(132)


# ==============================================================================
# PART I.  THE HEAVENS  --  ground-truth data generator
# ------------------------------------------------------------------------------
# We do NOT hand the model an equant curve and ask it to memorise it; that would
# be circular. Instead we generate genuine retrograde motion the way nature does
# it: two bodies on circular heliocentric orbits (Earth and an outer planet),
# then projected to the geocentric apparent longitude Ptolemy actually saw. The
# equant model must *approximate* this -- echoing history, where Ptolemy's
# equant was an unwitting first-order approximation of Kepler.
# ==============================================================================

def generate_heavens(n=360, span_years=6.0, T_earth=1.0, T_planet=2.0,
                     a_earth=1.0, a_planet=1.9, planet_phase0=0.7,
                     noise_deg=0.6):
    """
    Produce a time series of apparent geocentric longitude for an outer planet.

    T_planet = 2.0 is a didactic near-2:1 configuration so that the 'reason'
    criterion (snap the frequency ratio to a small rational) and the 'perception'
    criterion (fit the data) cooperate -- as they genuinely do for resonant
    bodies in the real solar system. Retrograde loops still occur near opposition.

    Returns
    -------
    t         : (n,) sample times in years
    lam_obs   : (n,) observed apparent longitude in radians, wrapped to (-pi, pi]
    lam_clean : (n,) noise-free apparent longitude (for diagnostics)
    """
    t = np.linspace(0.0, span_years, n)

    # Heliocentric positions on circular, coplanar orbits.
    thE = 2.0 * np.pi * t / T_earth
    thP = 2.0 * np.pi * t / T_planet + planet_phase0
    earth = np.stack([a_earth * np.cos(thE), a_earth * np.sin(thE)], axis=1)
    planet = np.stack([a_planet * np.cos(thP), a_planet * np.sin(thP)], axis=1)

    # Geocentric apparent longitude = direction from Earth to planet.
    d = planet - earth
    lam_clean = np.arctan2(d[:, 1], d[:, 0])

    # Observational scatter (Ptolemy's instruments were good to a few arcmin;
    # we use a coarser noise to make the two-criterion synergy visible).
    noise = np.deg2rad(noise_deg) * RNG.standard_normal(n)
    lam_obs = wrap_angle(lam_clean + noise)
    return t, lam_obs, lam_clean


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ==============================================================================
# PART II.  THE EQUANT ENGINE  --  forward model
# ------------------------------------------------------------------------------
# Parameters are carried as a flat dict of scalars so the gradient check and the
# optimiser can iterate over them uniformly. R is held fixed at 1.0 (scale gauge)
# to remove a redundant degree of freedom.
# ==============================================================================

PARAM_NAMES = ["r", "e", "omega_def", "phi_def", "omega_ep", "phi_ep"]
R_DEFERENT = 1.0  # fixed scale


def init_params():
    """A deliberately wrong starting guess -- the model must *find* the frame."""
    return {
        "r":         0.35,   # epicycle radius
        "e":         0.02,   # equant eccentricity (starts near zero: no frame shift)
        "omega_def": 2.0 * np.pi / 2.3,   # wrong deferent rate
        "phi_def":   0.0,
        "omega_ep":  2.0 * np.pi / 0.8,   # wrong epicycle rate
        "phi_ep":    0.0,
    }


def _forward_geometry(p, t):
    """
    Core Ptolemaic geometry. Returns intermediate quantities so the analytic
    gradient can reuse them.

        A = omega_def * t + phi_def                 (mean angle AT THE EQUANT)
        B = A + 2 e sin A                           (equation of centre: the
                                                     equant's fingerprint -- the
                                                     sweep is uniform at the
                                                     equant, non-uniform here)
        M = R (cos B, sin B)                        (epicycle centre on deferent)
        O = (-c, 0),  c = e R                       (Earth, offset opposite the
                                                     equant -> "bisected"
                                                     eccentricity)
        E = omega_ep * t + phi_ep                   (epicycle angle)
        P = M + r (cos E, sin E)                    (the planet)
        lambda = atan2(P_y - O_y, P_x - O_x)        (apparent longitude)
    """
    A = p["omega_def"] * t + p["phi_def"]
    sinA, cosA = np.sin(A), np.cos(A)
    B = A + 2.0 * p["e"] * sinA
    sinB, cosB = np.sin(B), np.cos(B)

    c = p["e"] * R_DEFERENT                       # Earth offset (bisected eccentricity)
    E = p["omega_ep"] * t + p["phi_ep"]
    sinE, cosE = np.sin(E), np.cos(E)

    # planet minus Earth, component-wise
    u = R_DEFERENT * cosB + p["r"] * cosE + c     # P_x - O_x   (O_x = -c)
    v = R_DEFERENT * sinB + p["r"] * sinE         # P_y - O_y   (O_y = 0)
    lam = np.arctan2(v, u)

    return dict(A=A, sinA=sinA, cosA=cosA, B=B, sinB=sinB, cosB=cosB,
                c=c, E=E, sinE=sinE, cosE=cosE, u=u, v=v, lam=lam)


def forward(p, t):
    """Apparent longitude predicted by the equant model."""
    return _forward_geometry(p, t)["lam"]


# ==============================================================================
# PART III.  THE TWO CRITERIA  --  loss
# ------------------------------------------------------------------------------
# Perception  : circular fit,  mean(1 - cos(pred - obs))  (immune to wrap-around)
# Reason      : harmonic comb pulling rho = omega_ep/omega_def toward a simple
#               rational.  L_reason = sum_q (1/q^2) (1 - cos(2 pi q rho)).
#               Its wells sit exactly at rationals p/q, deeper for simpler q --
#               a differentiable "snap to whole-number ratio," the Harmonics move.
# ==============================================================================

REASON_QMAX = 4  # consider resonances up to 4th order


def perception_loss(p, t, lam_obs):
    lam = forward(p, t)
    return np.mean(1.0 - np.cos(lam - lam_obs))


def reason_loss(p):
    rho = p["omega_ep"] / p["omega_def"]
    q = np.arange(1, REASON_QMAX + 1)
    w = 1.0 / q**2
    return np.sum(w * (1.0 - np.cos(2.0 * np.pi * q * rho)))


def total_loss(p, t, lam_obs, beta):
    return perception_loss(p, t, lam_obs) + beta * reason_loss(p)


# ==============================================================================
# PART IV.  ANALYTIC GRADIENT  (hand-derived, then finite-difference checked)
# ------------------------------------------------------------------------------
# d lambda/du = -v/(u^2+v^2),  d lambda/dv = u/(u^2+v^2).
# Chain those through P(params). Perception derivative wrt lambda is
# (1/N) sin(lambda - obs). Reason derivative flows only into the two omegas.
# ==============================================================================

def grad(p, t, lam_obs, beta):
    g = _forward_geometry(p, t)
    u, v = g["u"], g["v"]
    denom = u * u + v * v
    dlam_du = -v / denom
    dlam_dv = u / denom

    N = t.shape[0]
    dL_dlam = (1.0 / N) * np.sin(g["lam"] - lam_obs)   # perception sensitivity

    sinA, cosA = g["sinA"], g["cosA"]
    sinB, cosB = g["sinB"], g["cosB"]
    sinE, cosE = g["sinE"], g["cosE"]
    dBdA = 1.0 + 2.0 * p["e"] * cosA                    # dB/dA

    def accumulate(du, dv):
        return np.sum(dL_dlam * (dlam_du * du + dlam_dv * dv))

    grads = {}

    # r : u += r cosE ; v += r sinE
    grads["r"] = accumulate(cosE, sinE)

    # phi_ep : E += phi_ep
    grads["phi_ep"] = accumulate(-p["r"] * sinE, p["r"] * cosE)

    # omega_ep : E += omega_ep * t   (also enters reason via rho)
    du_wep = -p["r"] * sinE * t
    dv_wep = p["r"] * cosE * t
    grads["omega_ep"] = accumulate(du_wep, dv_wep)

    # phi_def : A += phi_def -> B += dBdA
    du_pd = -R_DEFERENT * sinB * dBdA
    dv_pd = R_DEFERENT * cosB * dBdA
    grads["phi_def"] = accumulate(du_pd, dv_pd)

    # omega_def : A += omega_def * t -> B += dBdA * t   (also enters reason)
    du_wd = -R_DEFERENT * sinB * (dBdA * t)
    dv_wd = R_DEFERENT * cosB * (dBdA * t)
    grads["omega_def"] = accumulate(du_wd, dv_wd)

    # e : B += 2 sinA  (dB/de) ; and c = e R adds +R to u
    du_e = -R_DEFERENT * sinB * (2.0 * sinA) + R_DEFERENT
    dv_e = R_DEFERENT * cosB * (2.0 * sinA)
    grads["e"] = accumulate(du_e, dv_e)

    # ---- reason term flows into the two omegas via rho = omega_ep / omega_def
    rho = p["omega_ep"] / p["omega_def"]
    q = np.arange(1, REASON_QMAX + 1)
    w = 1.0 / q**2
    dLr_drho = np.sum(w * 2.0 * np.pi * q * np.sin(2.0 * np.pi * q * rho))
    grads["omega_ep"] += beta * dLr_drho * (1.0 / p["omega_def"])
    grads["omega_def"] += beta * dLr_drho * (-p["omega_ep"] / p["omega_def"] ** 2)

    return grads


# ==============================================================================
# PART V.  GRADIENT CHECK  (mandatory)
# ==============================================================================

def gradient_check(seed=0, beta=0.05, eps=1e-6):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, 80)
    lam_obs = wrap_angle(rng.standard_normal(80))
    p = init_params()
    # jitter params so we test a generic point, not a special one
    for k in PARAM_NAMES:
        p[k] += 0.05 * rng.standard_normal()

    ana = grad(p, t, lam_obs, beta)
    max_rel = 0.0
    report = []
    for k in PARAM_NAMES:
        base = p[k]
        p[k] = base + eps
        Lp = total_loss(p, t, lam_obs, beta)
        p[k] = base - eps
        Lm = total_loss(p, t, lam_obs, beta)
        p[k] = base
        num = (Lp - Lm) / (2.0 * eps)
        rel = abs(num - ana[k]) / (abs(num) + abs(ana[k]) + 1e-12)
        max_rel = max(max_rel, rel)
        report.append((k, ana[k], num, rel))
    return max_rel, report


# ==============================================================================
# PART VI.  TRAINING LOOP  (plain gradient descent with a little momentum)
# ==============================================================================

def train(t, lam_obs, beta=0.05, lr=0.05, steps=4000, verbose=True):
    p = init_params()
    vel = {k: 0.0 for k in PARAM_NAMES}
    mom = 0.9
    history = []
    for s in range(steps):
        gr = grad(p, t, lam_obs, beta)
        for k in PARAM_NAMES:
            vel[k] = mom * vel[k] - lr * gr[k]
            p[k] += vel[k]
        # keep the equant eccentricity physical and the epicycle radius positive
        p["e"] = float(np.clip(p["e"], -0.6, 0.6))
        p["r"] = float(np.clip(p["r"], 1e-3, 0.95))
        if s % 400 == 0 or s == steps - 1:
            Lp = perception_loss(p, t, lam_obs)
            Lr = reason_loss(p)
            history.append((s, Lp, Lr))
            if verbose:
                print(f"  step {s:4d} | perception {Lp:.6f} | "
                      f"reason {Lr:.6f} | e {p['e']:+.4f} | "
                      f"rho {p['omega_ep']/p['omega_def']:.4f}")
    return p, history


# ==============================================================================
# PART VII.  DIAGNOSTICS
# ==============================================================================

def count_retrogrades(t, lam):
    """
    Count retrograde episodes = maximal runs where apparent longitude moves
    'backwards'. We unwrap, differentiate, and count sign-change groups into
    the negative-rate regime.
    """
    lam_u = np.unwrap(lam)
    d = np.diff(lam_u)
    neg = d < 0
    episodes, in_ep = 0, False
    for x in neg:
        if x and not in_ep:
            episodes += 1
            in_ep = True
        elif not x:
            in_ep = False
    return episodes


def circular_rms_deg(a, b):
    return np.rad2deg(np.sqrt(np.mean((wrap_angle(a - b)) ** 2)))


# ==============================================================================
# PART VIII.  MAIN  --  run everything, print a verifiable report
# ==============================================================================

def main():
    print("=" * 74)
    print("THE EQUANT ENGINE  --  Ptolemy (c.100-170 CE)")
    print("Frame-search cognition + the two-criterion (reason/perception) loop")
    print("=" * 74)

    # ---- 1. Gradient check (must pass) -------------------------------------
    print("\n[1] Finite-difference gradient check")
    max_rel, report = gradient_check()
    for k, a, n, rel in report:
        print(f"    {k:10s} analytic {a:+.6e}  numeric {n:+.6e}  rel {rel:.2e}")
    print(f"    -> max relative error = {max_rel:.2e}")
    assert max_rel < 1e-4, "GRADIENT CHECK FAILED"
    print("    GRADIENT CHECK PASSED")

    # ---- 2. Build the heavens ----------------------------------------------
    print("\n[2] Generating genuine retrograde motion (geocentric Kepler proj.)")
    t, lam_obs, lam_clean = generate_heavens()
    n = t.shape[0]
    split = int(0.70 * n)
    t_tr, lam_tr = t[:split], lam_obs[:split]
    t_te, lam_te = t[split:], lam_clean[split:]   # forecast vs clean truth
    print(f"    samples={n}  train={split}  holdout={n-split}")
    print(f"    retrograde episodes in truth = {count_retrogrades(t, lam_clean)}")

    # ---- 3. Train WITH the reason criterion --------------------------------
    print("\n[3] Training WITH both criteria (beta=0.05)")
    p_both, _ = train(t_tr, lam_tr, beta=0.05, verbose=True)

    # ---- 4. Ablation: perception only (reason off) -------------------------
    print("\n[4] Ablation: perception ONLY (beta=0.0)")
    p_perc, _ = train(t_tr, lam_tr, beta=0.0, verbose=False)

    # ---- 5. Diagnostics ----------------------------------------------------
    print("\n[5] Results")
    lam_full_both = forward(p_both, t)
    fit_both = circular_rms_deg(forward(p_both, t_tr), lam_tr)
    fc_both = circular_rms_deg(forward(p_both, t_te), lam_te)
    fc_perc = circular_rms_deg(forward(p_perc, t_te), lam_te)

    rho_both = p_both["omega_ep"] / p_both["omega_def"]
    rho_perc = p_perc["omega_ep"] / p_perc["omega_def"]

    print(f"    learned equant eccentricity e     = {p_both['e']:+.4f}")
    print(f"    learned epicycle radius r         = {p_both['r']:.4f}")
    print(f"    frequency ratio rho (both crit.)  = {rho_both:.4f}  "
          f"(reason target ~ 2.000)")
    print(f"    frequency ratio rho (perc. only)  = {rho_perc:.4f}")
    print(f"    train-fit circular RMS            = {fit_both:.3f} deg")
    print(f"    retrograde episodes reproduced    = "
          f"{count_retrogrades(t, lam_full_both)} "
          f"(truth {count_retrogrades(t, lam_clean)})")
    print(f"    HELD-OUT forecast RMS  both       = {fc_both:.3f} deg")
    print(f"    HELD-OUT forecast RMS  perc-only  = {fc_perc:.3f} deg")

    # ---- 6. Self-tests -----------------------------------------------------
    print("\n[6] Self-tests")
    ok_grad = max_rel < 1e-4
    ok_fit = fit_both < 8.0
    ok_retro = count_retrogrades(t, lam_full_both) >= 1
    ok_reason = abs(rho_both - 2.0) <= abs(rho_perc - 2.0) + 1e-9
    ok_forecast = fc_both < 25.0
    for name, ok in [("gradient_check", ok_grad),
                     ("fit_quality", ok_fit),
                     ("retrograde_reproduced", ok_retro),
                     ("reason_pulls_toward_resonance", ok_reason),
                     ("forecast_reasonable", ok_forecast)]:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    assert all([ok_grad, ok_fit, ok_retro, ok_reason, ok_forecast]), \
        "SELF-TESTS FAILED"
    print("\nAll checks passed. The observer moved; the tangle became law.")
    print("=" * 74)


if __name__ == "__main__":
    main()

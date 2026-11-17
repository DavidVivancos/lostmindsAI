#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 ASTN  --  THE ASTRARIUM NETWORK
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0373 · Giovanni Dondi (1330-1388)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 Chioggia c.1330 - Abbiategrasso, 19 October 1388
================================================================================

WHY THIS ARCHITECTURE AND NOT ANOTHER
-------------------------------------
Dondi spent sixteen years (1348-1364) building a seven-faced geared machine that
predicted where the Sun, Moon and five planets would stand in the zodiac. The
machine is gone. The *description* survives -- the Tractatus astrarii -- and from
that description working machines have been rebuilt in the twentieth century.

Three commitments are legible in that record, and this file is built out of them
rather than out of any modern default:

  (1) IRREGULARITY IS CUT INTO THE PART, NOT COMPOSED OUT OF REGULAR ONES.
      Everyone before him distributed uniform rotation through round wheels and
      let the *linkage* manufacture apparent anomaly. Dondi cut wheels that are
      not round, so that the transmission ratio varies continuously through a
      single turn. Here that becomes a per-wheel learnable phase warp: the
      non-linearity lives in the parameters of one component, not in the depth
      of a composition graph.

  (2) THE MACHINE CAN ONLY TURN WHOLE TEETH; THE SKY TURNS IRRATIONALS.
      Every ratio in a gear train is a ratio of integers. Dondi's design labour
      was choosing tooth counts whose ratio best approximates a period, and then
      writing down what the approximation cost. Here that becomes a hard
      quantization of every rate to a best rational with bounded denominator,
      trained through a straight-through estimator, with the residual drift
      reported in degrees per century rather than hidden.

  (3) THE SPECIFICATION OUTLIVES THE ARTIFACT.
      The astrarium was lost; the treatise was not. A trained ASTN therefore
      serialises to a complete bill of materials -- tooth counts, pitch-curve
      ovality, slot throw, arm lengths, settings -- from which an identical
      machine is rebuilt and checked against the original, bit for bit.

Two mechanical realities are enforced as hard constraints rather than left to
the optimiser's taste:

  * TEETH MUST MESH FORWARD. The instantaneous transmission ratio of a
    non-circular pair may vary, but it may never reach zero or reverse.
  * THE PIN MUST STAY IN THE SLOT. A pin-and-slot crank (Dondi used one for the
    Moon) has a variable radius, but the pin cannot pass through the arbor.

A model that violates either is not a worse model. It is an unbuildable one.

WHAT IT IS TRAINED ON
---------------------
Geocentric ecliptic longitude of Mars and of Venus -- the two dials whose
apparent motion is worst behaved, including full retrogradation. Targets are
generated from real Keplerian elements (Standish, JPL) with the inclinations
dropped, which is exactly the reduction Dondi's longitude dials perform: they
show position in the zodiac and leave latitude to a separate nodal dial.

The two dials SHARE the wheel that carries the Earth's own motion, because
geocentric direction is (planet - Earth) for both. This is the astrarium's
defining economy: one falling weight, one train, seven faces.

The test that matters is not interpolation. It is running the machine for
decades past the last observation it was set by.

REQUIREMENTS: numpy only. Run: python chapter_0373_giovanni_dondi_1330.py
================================================================================
"""

import json
import numpy as np

TAU = 2.0 * np.pi
DEG = 180.0 / np.pi


# =============================================================================
# 1. THE SKY  --  the phenomena the machine must save
# =============================================================================
# Mean Keplerian elements at J2000 (Standish & Williams, "Keplerian Elements for
# Approximate Positions of the Major Planets", JPL Solar System Dynamics).
# a  : semi-major axis, AU
# e  : eccentricity
# L0 : mean longitude at epoch, degrees
# pom: longitude of perihelion, degrees
# Inclination is deliberately omitted -- see the header note on longitude dials.

ELEMENTS = {
    "earth": dict(a=1.00000261, e=0.01671123, L0=100.46457166, pom=102.93768193),
    "mars":  dict(a=1.52371034, e=0.09339410, L0=355.44656795, pom=336.05637041),
    "venus": dict(a=0.72333566, e=0.00677672, L0=181.97909950, pom=131.60246718),
}


def solve_kepler(M, e, iters=60):
    """Newton iteration on E - e sin E = M. Converges in a handful of steps at
    these eccentricities; 60 is free insurance."""
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def heliocentric_xy(body, t_years):
    """True heliocentric position in the ecliptic plane, AU."""
    p = ELEMENTS[body]
    a, e = p["a"], p["e"]
    period = a ** 1.5                      # Kepler's third law, years
    pom = np.radians(p["pom"])
    L = np.radians(p["L0"]) + TAU * t_years / period
    M = np.mod(L - pom + np.pi, TAU) - np.pi
    E = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1.0 - e * np.cos(E))
    lam = nu + pom
    return r * np.cos(lam), r * np.sin(lam)


def geocentric_unit(body, t_years):
    """Unit vector from Earth to the body: what a dial pointer must point at.
    Returned as (N,2) so the loss can compare directions without any wrap-around
    branch -- angles are never subtracted, only vectors are."""
    px, py = heliocentric_xy(body, t_years)
    ex, ey = heliocentric_xy("earth", t_years)
    dx, dy = px - ex, py - ey
    n = np.hypot(dx, dy)
    return np.stack([dx / n, dy / n], axis=1)


# =============================================================================
# 2. THE RATIONAL PRISON  --  what a gear train is allowed to be
# =============================================================================

def best_rational(x, qmax):
    """Best rational approximation p/q to x with q <= qmax.

    Walks the continued-fraction convergents and, when the next convergent would
    overshoot the denominator budget, sweeps the semiconvergents so that nothing
    better inside the budget is missed. This is the arithmetic Dondi was doing by
    hand when he chose tooth counts: the sky hands you an irrational period, the
    workshop hands you whole teeth, and the gap between them is a quantity you
    are obliged to publish rather than round away.
    """
    x = float(x)
    best, err = (1, 1), abs(x - 1.0)
    p0, q0, p1, q1 = 0, 1, 1, 0
    v = x
    for _ in range(64):
        ai = int(np.floor(v))
        p2, q2 = ai * p1 + p0, ai * q1 + q0
        if q2 > qmax:
            amax = (qmax - q0) // q1 if q1 else 0
            for a in range(1, amax + 1):
                pp, qq = a * p1 + p0, a * q1 + q0
                if qq <= qmax and abs(x - pp / qq) < err:
                    best, err = (pp, qq), abs(x - pp / qq)
            break
        if q2 > 0 and abs(x - p2 / q2) < err:
            best, err = (p2, q2), abs(x - p2 / q2)
        p0, q0, p1, q1 = p1, q1, p2, q2
        frac = v - ai
        if frac < 1e-12:
            break
        v = 1.0 / frac
    return best


# =============================================================================
# 3. THE MACHINE
# =============================================================================

class Astrarium:
    """A pool of K wheels driven from one prime mover, read by D dials.

    PER WHEEL k
      r[k]      real rate in turns per year. In the finished machine this is
                forced to a ratio of whole tooth counts.
      phi[k]    setting angle -- where the wheel stands when the machine is
                started. Historically: the moment the clock is set to the sky.
      A[k,:]    sine coefficients of the OVAL GEAR (phase warp)
      B[k,:]    cosine coefficients of the OVAL GEAR
      C[k,:]    cosine coefficients of the PIN-AND-SLOT (radius warp)
      D[k,:]    sine coefficients of the PIN-AND-SLOT
      rho[k]    mean arm length -- the pointer's radius at mean position

    FORWARD, for a wheel k at time t
      theta = 2*pi*r[k]*t + phi[k]              uniform drive angle
      W     = sum_m A sin(m theta) + B cos(m theta)     out-of-round transmission
      psi   = theta + W                          angle actually delivered
      S     = 1 + sum_m C cos(m theta) + D sin(m theta) slot extension factor
      R     = rho[k] * S                         instantaneous arm length
      pointer tip = (R cos psi, R sin psi)

    A DIAL is a subset of wheels. Its pointer is the VECTOR SUM of their tips --
    which is precisely how a deferent carries an epicycle. Only the direction of
    that sum is read, so a dial is scale-free and the loss is a pure angular one.

    TWO MECHANICAL CONSTRAINTS, enforced by penalty
      J(theta) = dpsi/dtheta  must stay above Jfloor > 0   (teeth mesh forward)
      S(theta)                must stay above Sfloor > 0   (pin stays in slot)
    """

    def __init__(self, n_wheels, n_harmonics, dials, qmax=365, seed=0):
        rng = np.random.default_rng(seed)
        K, M = n_wheels, n_harmonics
        self.K, self.M, self.dials, self.qmax = K, M, dials, qmax
        self.p = dict(
            r=np.ones(K),
            phi=rng.normal(0.0, 0.1, K),
            A=rng.normal(0.0, 1e-3, (K, M)),
            B=rng.normal(0.0, 1e-3, (K, M)),
            C=rng.normal(0.0, 1e-3, (K, M)),
            D=rng.normal(0.0, 1e-3, (K, M)),
            rho=np.ones(K),
        )
        self.teeth_cut = False          # False = design on vellum, True = brass
        self.Jfloor = 0.15              # minimum forward transmission ratio
        self.Sfloor = 0.15              # minimum slot extension
        self.lam_J = 5.0
        self.lam_S = 5.0
        self.grid = np.linspace(0.0, TAU, 64, endpoint=False)

    # ---- rates -------------------------------------------------------------
    def rates(self):
        """Realised rates. Before the teeth are cut these are the real design
        values; afterwards they are ratios of whole numbers and nothing else."""
        r = self.p["r"]
        if not self.teeth_cut:
            return r.copy(), None
        pq = [best_rational(abs(v), self.qmax) for v in r]
        g = np.array([np.sign(v) * a / b for v, (a, b) in zip(r, pq)])
        return g, pq

    def cut_the_teeth(self):
        """Snap every rate to its best realisable tooth ratio and freeze it.

        This is a one-way door, exactly as in the workshop. After this point the
        residual error can only be absorbed by re-filing the oval wheels and
        re-setting the dials -- never by changing a tooth count.
        """
        self.teeth_cut = True
        g, pq = self.rates()
        design = self.p["r"].copy()
        self.p["r"] = g.copy()
        return design, pq

    # ---- forward + analytic backward --------------------------------------
    def forward(self, t, targets, need_grad=True):
        P, K, M = self.p, self.K, self.M
        N = t.size
        g, _ = self.rates()

        TH = TAU * g[None, :] * t[:, None] + P["phi"][None, :]        # (N,K)
        mm = np.arange(1, M + 1)[None, None, :]                       # (1,1,M)
        SM = np.sin(mm * TH[:, :, None])                              # (N,K,M)
        CM = np.cos(mm * TH[:, :, None])

        W = (SM * P["A"][None] + CM * P["B"][None]).sum(2)            # phase warp
        U = (CM * P["C"][None] + SM * P["D"][None]).sum(2)            # radius warp
        PSI = TH + W
        R = P["rho"][None, :] * (1.0 + U)
        cP, sP = np.cos(PSI), np.sin(PSI)
        X, Y = R * cP, R * sP                                         # pointer tips

        loss = 0.0
        gX = np.zeros((N, K)) if need_grad else None
        gY = np.zeros((N, K)) if need_grad else None
        nd = len(self.dials)

        for idx, tgt in zip(self.dials, targets):
            vx, vy = X[:, idx].sum(1), Y[:, idx].sum(1)
            n = np.sqrt(vx * vx + vy * vy + 1e-12)
            s = vx * tgt[:, 0] + vy * tgt[:, 1]
            # 1 - cos(angle between pointer and truth): smooth, wrap-free, and
            # zero exactly when the pointer is right.
            loss += np.mean(1.0 - s / n) / nd
            if need_grad:
                c = 1.0 / (N * nd)
                dvx = (-tgt[:, 0] / n + s * vx / n ** 3) * c
                dvy = (-tgt[:, 1] / n + s * vy / n ** 3) * c
                for k in idx:
                    gX[:, k] += dvx
                    gY[:, k] += dvy

        # -- machinability barriers, evaluated on a fixed angular grid --------
        thg = self.grid[None, :, None]
        mg = np.arange(1, M + 1)[None, None, :]
        cg, sg = np.cos(mg * thg), np.sin(mg * thg)
        J = 1.0 + (mg * (cg * P["A"][:, None, :] - sg * P["B"][:, None, :])).sum(2)
        S = 1.0 + (cg * P["C"][:, None, :] + sg * P["D"][:, None, :]).sum(2)
        vJ = np.maximum(self.Jfloor - J, 0.0)
        vS = np.maximum(self.Sfloor - S, 0.0)
        G = self.grid.size
        loss += self.lam_J * (vJ ** 2).sum() / G + self.lam_S * (vS ** 2).sum() / G

        if not need_grad:
            return loss, dict(J=J, S=S)

        # -- backward ---------------------------------------------------------
        gR = gX * cP + gY * sP
        gPSI = -gX * R * sP + gY * R * cP
        grho = (gR * (1.0 + U)).sum(0)
        gU = gR * P["rho"][None, :]

        # dW/dtheta and dU/dtheta: the instantaneous ratio and the slot velocity
        Jw = (mm[0] * (CM * P["A"][None] - SM * P["B"][None])).sum(2)
        Ju = (mm[0] * (-SM * P["C"][None] + CM * P["D"][None])).sum(2)
        gTH = gPSI * (1.0 + Jw) + gU * Ju

        gA = np.einsum("nk,nkm->km", gPSI, SM)
        gB = np.einsum("nk,nkm->km", gPSI, CM)
        gC = np.einsum("nk,nkm->km", gU, CM)
        gD = np.einsum("nk,nkm->km", gU, SM)
        gphi = gTH.sum(0)
        grate = (gTH * (TAU * t[:, None])).sum(0)   # straight-through past the
                                                    # tooth quantization

        bJ = -2.0 * self.lam_J * vJ / G
        bS = -2.0 * self.lam_S * vS / G
        gA += (bJ[:, :, None] * mg * cg).sum(1)
        gB += (bJ[:, :, None] * (-mg * sg)).sum(1)
        gC += (bS[:, :, None] * cg).sum(1)
        gD += (bS[:, :, None] * sg).sum(1)

        return loss, dict(r=grate, phi=gphi, A=gA, B=gB, C=gC, D=gD, rho=grho)

    # ---- reading the dials -------------------------------------------------
    def pointers(self, t):
        """Angle of every dial pointer, radians, for each time in t."""
        P, M = self.p, self.M
        g, _ = self.rates()
        TH = TAU * g[None, :] * t[:, None] + P["phi"][None, :]
        mm = np.arange(1, M + 1)[None, None, :]
        SM, CM = np.sin(mm * TH[:, :, None]), np.cos(mm * TH[:, :, None])
        W = (SM * P["A"][None] + CM * P["B"][None]).sum(2)
        U = (CM * P["C"][None] + SM * P["D"][None]).sum(2)
        PSI, R = TH + W, P["rho"][None, :] * (1.0 + U)
        X, Y = R * np.cos(PSI), R * np.sin(PSI)
        return [np.arctan2(Y[:, i].sum(1), X[:, i].sum(1)) for i in self.dials]

    def dial_error_deg(self, t, targets):
        """Mean and worst absolute pointing error per dial, in degrees."""
        out = []
        for ang, tgt in zip(self.pointers(t), targets):
            d = ang - np.arctan2(tgt[:, 1], tgt[:, 0])
            d = (d + np.pi) % TAU - np.pi
            a = np.abs(d) * DEG
            out.append((float(a.mean()), float(a.max())))
        return out

    # ---- the machinability audit ------------------------------------------
    def audit(self):
        """Report whether this machine could be built out of metal.

        pitch_ratio is the real payoff: for a non-circular pair on a fixed centre
        distance, r1/r2 = J, so r1 = a*J/(1+J). The ratio of the largest to the
        smallest pitch radius IS the ovality of the wheel -- the number a
        wheel-cutter would need. The model does not merely fit; it emits a shape.
        """
        _, aux = self.forward(np.array([0.0]),
                              [np.array([[1.0, 0.0]])] * len(self.dials),
                              need_grad=False)
        J, S = aux["J"], aux["S"]
        r1 = J / (1.0 + J)
        return dict(
            J_min=J.min(1), J_max=J.max(1),
            S_min=S.min(1), S_max=S.max(1),
            pitch_ratio=r1.max(1) / r1.min(1),
            buildable=bool(J.min() > 0.0 and S.min() > 0.0),
        )

    # ---- the Tractatus -----------------------------------------------------
    def to_tractatus(self, decimals=9):
        """Serialise the whole machine to a parts list a workshop could follow.

        Deliberately lossy in the same way a written description is lossy: every
        number is rounded to a stated precision. If the rebuilt machine still
        agrees with this one, the claim 'the specification outlives the artifact'
        has been tested rather than asserted.
        """
        g, pq = self.rates()
        wheels = []
        for k in range(self.K):
            wheels.append(dict(
                wheel=k,
                teeth=(list(pq[k]) if pq else None),
                rate_turns_per_year=round(float(g[k]), decimals),
                setting_angle_rad=round(float(self.p["phi"][k]), decimals),
                oval_sine=[round(float(v), decimals) for v in self.p["A"][k]],
                oval_cosine=[round(float(v), decimals) for v in self.p["B"][k]],
                slot_cosine=[round(float(v), decimals) for v in self.p["C"][k]],
                slot_sine=[round(float(v), decimals) for v in self.p["D"][k]],
                arm_length=round(float(self.p["rho"][k]), decimals),
            ))
        return dict(
            treatise="Tractatus astrarii (machine-readable)",
            n_wheels=self.K, n_harmonics=self.M,
            dials=[list(d) for d in self.dials],
            tooth_budget=self.qmax,
            teeth_cut=self.teeth_cut,
            wheels=wheels,
        )

    @staticmethod
    def from_tractatus(spec, use_teeth=True):
        """Rebuild a machine from nothing but the written description.

        Note which number is read for the rate. A tooth count is an INTEGER: it
        transcribes without loss, survives a scribe, and reconstitutes the rate
        exactly. A decimal rate is rounded, and because rate error accumulates
        linearly with elapsed time, that rounding becomes visible drift decades
        later. The quantization that costs the machine its accuracy against the
        sky is the very thing that makes the machine reproducible from a text.
        Set use_teeth=False to see what a decimals-only treatise would cost.
        """
        m = Astrarium(spec["n_wheels"], spec["n_harmonics"],
                      [list(d) for d in spec["dials"]], qmax=spec["tooth_budget"])
        m.teeth_cut = False   # rates are read literally from the parts list
        for k, w in enumerate(spec["wheels"]):
            if use_teeth and w["teeth"]:
                p, q = w["teeth"]
                m.p["r"][k] = np.sign(w["rate_turns_per_year"]) * p / q
            else:
                m.p["r"][k] = w["rate_turns_per_year"]
            m.p["phi"][k] = w["setting_angle_rad"]
            m.p["A"][k] = w["oval_sine"]
            m.p["B"][k] = w["oval_cosine"]
            m.p["C"][k] = w["slot_cosine"]
            m.p["D"][k] = w["slot_sine"]
            m.p["rho"][k] = w["arm_length"]
        return m


# =============================================================================
# 4. TRAINING
# =============================================================================

def adam(model, t, targets, steps, lr, frozen=(), log_every=0):
    state = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in model.p.items()}
    loss = float("nan")
    for s in range(1, steps + 1):
        loss, grads = model.forward(t, targets)
        for k in model.p:
            if k in frozen:
                continue
            m1, m2 = state[k]
            g = grads[k]
            m1[:] = 0.9 * m1 + 0.1 * g
            m2[:] = 0.999 * m2 + 0.001 * g * g
            model.p[k] -= lr * (m1 / (1 - 0.9 ** s)) / (np.sqrt(m2 / (1 - 0.999 ** s)) + 1e-8)
        if log_every and (s % log_every == 0 or s == 1):
            print(f"      step {s:5d}   loss {loss:.3e}")
    return loss


def survey_the_sky(t, u, lo=0.05, hi=3.0, n=4000, top=4):
    """Before designing gears, read the tables.

    A Fourier scan of the observed direction as a complex unit signal. Dondi did
    not derive periods from his machine; he took them from Campanus and the
    Alphonsine Tables and then built a machine that honoured them. Initialising
    the rates from a spectral survey is the same order of operations, and it is
    what makes the subsequent gradient descent tractable: the loss in rate space
    is violently multimodal and no optimiser should be asked to search it blind.
    """
    f = np.linspace(lo, hi, n)
    z = u[:, 0] + 1j * u[:, 1]
    power = np.abs((z[None, :] * np.exp(-2j * np.pi * f[:, None] * t[None, :])).mean(1))
    peaks = [i for i in range(1, n - 1) if power[i] > power[i - 1] and power[i] > power[i + 1]]
    peaks.sort(key=lambda i: -power[i])
    return [float(f[i]) for i in peaks[:top]]


def build_astrarium(t_train, targets, n_harmonics=2, qmax=365, seed=0,
                    verbose=False):
    """The full construction sequence, in Dondi's order.

    I  consult the tables       -> spectral survey fixes the rates
    II design on vellum         -> continuous optimisation, everything free
    III cut the teeth           -> rates snap to whole numbers, then frozen
    IV  file the ovals and set  -> everything except the rates is re-tuned to
                                   absorb as much of the quantization error as
                                   the shapes can absorb
    """
    m = Astrarium(3, n_harmonics, dials=[[0, 1], [0, 2]], qmax=qmax, seed=seed)

    peaks_mars = survey_the_sky(t_train, targets[0])
    peaks_venus = survey_the_sky(t_train, targets[1])
    solar = min(peaks_mars + peaks_venus, key=lambda f: abs(f - 1.0))
    r_mars = min(peaks_mars, key=lambda f: abs(f - 0.53))
    r_venus = min(peaks_venus, key=lambda f: abs(f - 1.63))
    if verbose:
        print(f"   I.  tables   Mars peaks {[round(p,4) for p in peaks_mars]}")
        print(f"                Venus peaks {[round(p,4) for p in peaks_venus]}")
        print(f"                adopted rates: solar {solar:.5f}  "
              f"Mars {r_mars:.5f}  Venus {r_venus:.5f} turns/yr")

    m.p["r"][:] = [solar, r_mars, r_venus]
    # The shared wheel carries the Earth's motion but must point AWAY from it,
    # since geocentric direction is (planet - Earth). Half a turn of setting.
    m.p["phi"][:] = [np.pi, 0.0, 0.0]
    m.p["rho"][:] = [1.0, 1.5, 0.7]

    if verbose:
        print("   II. vellum")
    adam(m, t_train, targets, 2200, 3e-3, log_every=1100 if verbose else 0)

    design, teeth = m.cut_the_teeth()
    if verbose:
        print("   III.brass")
        for k, (p, q) in enumerate(teeth):
            realised = p / q
            print(f"                wheel {k}: {p:4d}/{q:4d} = {realised:.9f}  "
                  f"design {design[k]:.9f}  residual {realised-design[k]:+.2e} turns/yr")

    if verbose:
        print("   IV. filing")
    adam(m, t_train, targets, 1000, 8e-4, frozen=("r",),
         log_every=1000 if verbose else 0)
    return m, design, teeth


# =============================================================================
# 5. SELF-TESTS
# =============================================================================

def test_rational():
    cases = [(0.5316751691782668, 200, (42, 79)), (0.5, 10, (1, 2)),
             (2.0, 10, (2, 1)), (np.pi, 113, (355, 113))]
    for x, q, want in cases:
        got = best_rational(x, q)
        assert got == want, f"best_rational({x},{q}) = {got}, expected {want}"
    for x in (0.5316751691782668, 1.6255120222989552, 0.61519077):
        for q in (20, 60, 200, 800):
            p, qq = best_rational(x, q)
            assert qq <= q and abs(x - p / qq) <= abs(x - round(x)) + 1e-12
    return "best_rational: convergents and semiconvergents correct on 4 known cases"


def stress_machine(seed=7):
    """A machine built specifically to make the gradient check hard.

    Gradients are checked here rather than on the finished astrarium for a plain
    numerical reason: a converged machine has gradients of order 1e-8, and
    central differences on a float64 loss cannot resolve those -- the check would
    be measuring cancellation noise, not correctness.

    This machine is deliberately mis-set AND deliberately unbuildable: the warp
    coefficients are large enough to drive the transmission ratio and the slot
    extension below their floors on some wheels, so the barrier branch of the
    backward pass is exercised too. Every gradient path in the file is live.
    """
    rng = np.random.default_rng(seed)
    m = Astrarium(3, 3, dials=[[0, 1], [0, 2]], qmax=365, seed=seed)
    m.p["r"][:] = [0.97, 0.55, 1.61]
    m.p["phi"][:] = rng.normal(0, 1.0, 3)
    m.p["A"][:] = rng.normal(0, 0.35, (3, 3))
    m.p["B"][:] = rng.normal(0, 0.35, (3, 3))
    m.p["C"][:] = rng.normal(0, 0.35, (3, 3))
    m.p["D"][:] = rng.normal(0, 0.35, (3, 3))
    m.p["rho"][:] = [1.0, 1.4, 0.8]
    return m


def test_gradients(model, t, targets, per_param=4, seed=1):
    """Central finite differences against the analytic gradient.

    The rate parameters need a tighter step than the rest: their second
    derivative scales with (2*pi*t)^2, which for a twelve-year window is of order
    5.7e3, so a 1e-6 step leaves visible truncation error. Everything else is
    checked at 1e-6.
    """
    rng = np.random.default_rng(seed)
    _, grads = model.forward(t, targets)
    rows, worst = [], 0.0
    for key in model.p:
        eps = 1e-7 if key == "r" else 1e-6
        flat = model.p[key].ravel()
        idxs = rng.choice(flat.size, size=min(per_param, flat.size), replace=False)
        rel_max = 0.0
        for i in idxs:
            o = flat[i]
            flat[i] = o + eps
            lp, _ = model.forward(t, targets, need_grad=False)
            flat[i] = o - eps
            lm, _ = model.forward(t, targets, need_grad=False)
            flat[i] = o
            num = (lp - lm) / (2 * eps)
            ana = grads[key].ravel()[i]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            rel_max = max(rel_max, rel)
        rows.append((key, eps, rel_max))
        worst = max(worst, rel_max)
    return rows, worst


def _rebuild_gap(model, t, decimals, use_teeth):
    spec = model.to_tractatus(decimals=decimals)
    text = json.dumps(spec)                       # the treatise, as bytes
    rebuilt = Astrarium.from_tractatus(json.loads(text), use_teeth=use_teeth)
    worst = 0.0
    for pa, pb in zip(model.pointers(t), rebuilt.pointers(t)):
        d = (pa - pb + np.pi) % TAU - np.pi
        worst = max(worst, float(np.abs(d).max() * DEG))
    return worst, len(text)


def test_tractatus(model, t):
    """Write the machine down, lose the machine, rebuild it from the writing."""
    return _rebuild_gap(model, t, decimals=9, use_teeth=True)


# =============================================================================
# 6. THE RUN
# =============================================================================

def main():
    np.set_printoptions(precision=4, suppress=True)
    rule = "=" * 78

    print(rule)
    print(" ASTN -- THE ASTRARIUM NETWORK")
    print(" Figure 0295, Giovanni Dondi dall'Orologio (c.1330-1388)")
    print(rule)

    # ---- data ---------------------------------------------------------------
    t_train = np.linspace(0.0, 12.0, 1500)      # the observing campaign
    t_far = np.linspace(12.0, 60.0, 4000)       # the next forty-eight years
    tgt_train = [geocentric_unit("mars", t_train), geocentric_unit("venus", t_train)]
    tgt_far = [geocentric_unit("mars", t_far), geocentric_unit("venus", t_far)]
    print(f"\n[DATA] geocentric longitude of Mars and Venus from Keplerian elements")
    print(f"       train  {t_train.size} samples over years 0-12")
    print(f"       test   {t_far.size} samples over years 12-60 (pure extrapolation)")

    # ---- test 1: the arithmetic --------------------------------------------
    print("\n[TEST 1] rational approximation")
    print("   " + test_rational())

    # ---- build --------------------------------------------------------------
    print("\n[BUILD] sixteen years, compressed")
    model, design, teeth = build_astrarium(t_train, tgt_train,
                                           n_harmonics=2, qmax=365, verbose=True)

    # ---- test 2: gradients ---------------------------------------------------
    print("\n[TEST 2] finite-difference gradient check")
    stress = stress_machine()
    _, aux = stress.forward(t_train, tgt_train, need_grad=False)
    nJ = int((aux["J"] < stress.Jfloor).sum())
    nS = int((aux["S"] < stress.Sfloor).sum())
    print(f"   checked on a deliberately mis-set, deliberately UNBUILDABLE machine")
    print(f"   barrier active at {nJ} of {aux['J'].size} grid points (teeth) and "
          f"{nS} of {aux['S'].size} (slot) -- barrier gradients are live")
    rows, worst = test_gradients(stress, t_train, tgt_train)
    for key, eps, rel in rows:
        print(f"   {key:>4}  eps={eps:.0e}  max rel err {rel:.3e}")
    ok_grad = worst < 1e-5 and (nJ > 0) and (nS > 0)
    print(f"   worst relative error over all 7 parameter groups: {worst:.3e}   "
          f"{'PASS' if ok_grad else 'FAIL'} (tol 1e-5)")

    # After the teeth are cut the forward pass is piecewise constant in r by
    # construction, so the rate gradient there is a straight-through estimator
    # with no true derivative to compare against. That is the intended design,
    # not a defect, and it is why the rates are frozen once the teeth are cut.
    stress.teeth_cut = True
    rows_q, _ = test_gradients(stress, t_train, tgt_train)
    rq = [r for r in rows_q if r[0] == "r"][0][2]
    print(f"   after cutting: rate gradient rel err {rq:.3e} -- expected 1.0, the")
    print(f"   forward pass is flat in r between tooth counts (estimator by design)")

    # ---- test 3: machinability ----------------------------------------------
    print("\n[TEST 3] machinability audit -- could a workshop cut this?")
    a = model.audit()
    for k in range(model.K):
        print(f"   wheel {k}: transmission ratio {a['J_min'][k]:.3f}-{a['J_max'][k]:.3f}   "
              f"slot extension {a['S_min'][k]:.3f}-{a['S_max'][k]:.3f}   "
              f"pitch ovality {a['pitch_ratio'][k]:.4f}")
    print(f"   teeth never reverse, pin never leaves the slot: "
          f"{'PASS' if a['buildable'] else 'FAIL'}")

    # ---- test 4: the treatise ------------------------------------------------
    print("\n[TEST 4] Tractatus round-trip -- destroy the machine, keep the text")
    dev, nbytes = test_tractatus(model, t_far)
    ok_spec = dev < 1e-6
    print(f"   specification size: {nbytes} bytes")
    print(f"   worst pointer disagreement between original and rebuild over 48 "
          f"years: {dev:.3e} deg")
    print(f"   {'PASS' if ok_spec else 'FAIL'} (tol 1e-6 deg)")
    print("   -- and what the treatise would have cost written as decimals only:")
    print("      decimals   rebuilt from tooth counts   rebuilt from decimal rates")
    for dec in (4, 6, 9):
        gi, _ = _rebuild_gap(model, t_far, dec, use_teeth=True)
        gd, _ = _rebuild_gap(model, t_far, dec, use_teeth=False)
        print(f"      {dec:>5d}      {gi:>12.3e} deg          {gd:>12.3e} deg")
    print("   An integer transcribes without loss. A rounded rate does not, and")
    print("   rate error grows linearly with elapsed time.")

    # ---- results -------------------------------------------------------------
    print("\n[RESULT] pointing error, degrees of ecliptic longitude")
    names = ["Mars ", "Venus"]
    tr = model.dial_error_deg(t_train, tgt_train)
    fa = model.dial_error_deg(t_far, tgt_far)
    print("   dial     train mean / worst      years 12-60 mean / worst")
    for n, (a1, a2), (b1, b2) in zip(names, tr, fa):
        print(f"   {n}    {a1:7.4f} / {a2:7.4f}        {b1:7.4f} / {b2:7.4f}")

    print("\n[RESULT] what the machine recovered without being told")
    g, pq = model.rates()
    truth = dict(mars=1.52371034, venus=0.72333566)
    print(f"   arm lengths (AU-equivalent, shared solar arm normalised to 1):")
    scale = model.p["rho"][0] / 1.00000261
    for k, nm in ((1, "mars"), (2, "venus")):
        got = model.p["rho"][k] / scale
        print(f"      {nm:6s} {got:.5f}   true semi-major axis {truth[nm]:.5f}   "
              f"error {100*abs(got-truth[nm])/truth[nm]:.2f}%")
    print(f"   sidereal periods from tooth counts:")
    for k, nm, tp in ((1, "mars", 1.8808477), (2, "venus", 0.6151908)):
        p, q = pq[k]
        print(f"      {nm:6s} {q}/{p} teeth -> {q/p:.7f} yr   true {tp:.7f} yr   "
              f"error {abs(q/p-tp)*365.25*24*60:.2f} min/revolution")

    # ---- experiment A: the tooth ledger --------------------------------------
    print("\n[EXPERIMENT A] the tooth ledger -- what quantization costs")
    print("   Every rate must be a ratio of whole teeth. The looser the budget,")
    print("   the finer the approximation and the slower the machine drifts.")
    print("   budget   Mars teeth   Venus teeth   drift over 48 yr (deg, mean)")
    for qmax in (40, 100, 200, 365, 800):
        m2, _, tt = build_astrarium(t_train, tgt_train, n_harmonics=2, qmax=qmax)
        e = m2.dial_error_deg(t_far, tgt_far)
        print(f"   {qmax:>5d}    {tt[1][0]:>4d}/{tt[1][1]:<4d}    {tt[2][0]:>4d}/{tt[2][1]:<4d}"
              f"     Mars {e[0][0]:6.3f}   Venus {e[1][0]:6.3f}")

    # ---- experiment B: the oval wheel ----------------------------------------
    print("\n[EXPERIMENT B] the oval wheel -- does out-of-roundness earn its keep?")
    print("   M = 0 is a machine of perfectly round gears: uniform motion only.")
    print("   Each added harmonic is one more degree of freedom in the SHAPE of")
    print("   a wheel, not one more wheel.")
    print("   harmonics   train mean (deg)          years 12-60 mean (deg)")
    for M in (0, 1, 2, 3, 4):
        m2, _, _ = build_astrarium(t_train, tgt_train, n_harmonics=M, qmax=365)
        a1 = m2.dial_error_deg(t_train, tgt_train)
        b1 = m2.dial_error_deg(t_far, tgt_far)
        print(f"   M={M}          Mars {a1[0][0]:7.3f}  Venus {a1[1][0]:6.3f}"
              f"      Mars {b1[0][0]:7.3f}  Venus {b1[1][0]:6.3f}")

    # ---- summary --------------------------------------------------------------
    print("\n" + rule)
    passed = ok_grad and a["buildable"] and ok_spec
    print(f" ALL SELF-TESTS {'PASSED' if passed else 'FAILED'}")
    print(rule)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

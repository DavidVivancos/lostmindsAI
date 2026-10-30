#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================ 
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0328 · Nasir al-Din al-Tusi (1201-1274)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
 THE CONSTITUTING ARRANGEMENT (al-hay'a al-murakkaba)
 A cognitive architecture after Nasir al-Din al-Tusi (Tus 1201 - Baghdad 1274)
================================================================================

 Pure NumPy. No autograd, no framework. Analytic gradients, finite-difference
 verified. Runs end to end on CPU in a few seconds.

--------------------------------------------------------------------------------
 WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
--------------------------------------------------------------------------------

 Tusi states the design rule himself, in his Tahrir al-Majisti (recension of the
 Almagest, completed 5 Shawwal 644 / 13 February 1247), while throwing out
 Ptolemy's account of planetary latitude:

     "It is necessary for a practitioner of this discipline to establish
      circles and bodies having uniform motions according to an order and
      arrangement [such that] from all of them these various perceived
      motions will be constituted."
                        (tr. F. J. Ragep, MPIWG Preprint 457, 2014, p. 10)

 Three commitments follow, and this file implements all three as machinery
 rather than as commentary.

 (1) ONE ADMISSIBLE PRIMITIVE. The only thing that exists is a body turning
     uniformly. Everything a model can do, it must do by ARRANGEMENT of that
     one primitive. There is no second kind of part to reach for when the first
     will not stretch. (class Rotator; class Couple)

 (2) NO DISEMBODIED PARAMETER. Ptolemy's equant is a point about which motion
     is uniform but which is not the centre of anything and is not a body. It
     predicts beautifully. Tusi ruled it "outside the discipline" anyway. A
     quantity that does work in the model must correspond to something that
     could actually be built and turned. (class EquantOracle -- the foil;
     method Arrangement.constitute_equant -- the replacement)

 (3) A MECHANISM OWNS ALL ITS CONSEQUENCES. Tusi's decisive argument against
     Ptolemy's latitude circles: a small circle that produces the wanted
     north-south displacement MUST produce an exactly equal displacement in
     longitude, and you may not claim the one is perceptible and the other is
     not, "since they are equal in size and distance from the centre of the
     ecliptic." A side effect is not a defect in the account. It is a theorem
     about the account. So side effects are computed, cancelled in hardware by
     a dedicated orb, and whatever fails to cancel is PUBLISHED.
     (class MuhitaLayer -- the enclosing sphere; class ResidualLedger)

 And a fourth thing, which is a behaviour rather than a structure:

 (4) THE DECLARED NON-SOLUTION. At the end of a working life Tusi wrote, of the
     one planet whose model he never fixed: "As for Mercury, it has not yet
     been possible for me to conceive how it should be done." The architecture
     is required to be able to say this. When a target contains structure that
     no admissible arrangement can represent, the model must return the
     sentinel UNCONSTITUTED rather than absorb the discrepancy into a parameter
     that has no body. (Arrangement.declare / class Unconstituted)

--------------------------------------------------------------------------------
 THE ONE PIECE OF MATHEMATICS THE WHOLE THING RESTS ON
--------------------------------------------------------------------------------

 Take two bodies of equal radius r. The first turns at rate +k. The second is
 carried by the first (its centre rides on the first one's tip) and turns at
 rate -k, offset by a phase alpha. Write the tip as a complex number:

     z(t) = r e^{i(kt)} + r e^{i(alpha - kt)}
          = 2 r cos(kt - alpha/2) * e^{i alpha/2}

 The tip does not travel on a circle, an oval, or an epicyclic rosette. It
 travels back and forth along an exactly straight segment through the origin,
 in the fixed direction alpha/2, with amplitude 2r, harmonically in t.

 That is the Tusi couple, and read in the other direction it is the statement
 cos(x) = (e^{ix} + e^{-ix}) / 2 -- Euler's identity, built out of brass, three
 hundred years before Euler and five hundred before anybody wrote it that way.
 Tusi's problem was that Aristotelian celestial physics allowed him exactly one
 verb (turn uniformly) and his data demanded a different one (approach and
 recede). He did not weaken the physics and he did not discard the data. He
 proved that the forbidden verb was already inside the permitted one, if you
 composed it with itself.

 A NOTE ON A DETAIL THAT LOOKS PEDANTIC AND IS NOT. The same locus can be got
 by ROLLING a small circle inside a big one of twice the radius. Tusi does not
 do that. He is explicit that the larger orb CARRIES (mi-barad) the smaller,
 because solid celestial orbs cannot interpenetrate and a rolling contact would
 require them to. Two mathematically identical constructions, one buildable and
 one not, and he takes the buildable one. That is commitment (2) enforced at a
 level of detail most people would not have bothered with. This file follows
 him: composition here is always carrying, never rolling. See Couple.tips().

--------------------------------------------------------------------------------
 WHAT THE MODEL ACTUALLY LEARNS, AND WHY THAT IS A REAL TEST
--------------------------------------------------------------------------------

 TASK A -- exactness. Verify the couple traces a straight line to machine
   precision, not approximately. No training; a proof obligation.

 TASK B -- constituting the equant. Generate the true longitude of a planet
   from a genuine Ptolemaic equant construction (closed form, exact). Then fit
   it with an arrangement of couples at INTEGER harmonics only. Watch the
   residual fall as orbs are added. This is Tusi's own program, executed.

 TASK C -- two channels and the side-effect ledger. An orb inclined by iota
   contributes cos(iota) to longitude and sin(iota) to latitude. It cannot
   contribute to one alone. Fit latitude; measure the longitude damage;
   cancel it with an enclosing orb whose radius must be QUANTISED because a
   real orb has to be manufactured to a finite precision; publish what the
   quantisation leaves behind.

 TASK D -- Mercury. A target with an incommensurable component that no
   admissible arrangement can represent. Correct behaviour is to declare it,
   not to fit it.

 TASK E -- generalisation. Against a pure-NumPy MLP of comparable parameter
   count, on the same data, extrapolated far outside the training window. The
   arrangement holds because integer rates make it exactly periodic; the MLP
   does not, because nothing in it knows that the sky comes back around.

 TASK F -- recension (tahrir). Non-destructive revision: keep the ancestor
   runnable, learn a correction on top, report fidelity. Replays Tusi's own
   correction of the lunar inclined orb from 13;11 deg/day to 13;14 deg/day
   between the Hall (1245) and the Tadhkira (1261).

--------------------------------------------------------------------------------
 SOURCES FOR THE HISTORICAL CLAIMS ABOVE
--------------------------------------------------------------------------------
 F. J. Ragep, "From Tun to Turun: The Twists and Turns of the Tusi-Couple",
   MPIWG Preprint 457, Max Planck Institute for the History of Science, 2014.
 F. J. Ragep, Nasir al-Din al-Tusi's Memoir on Astronomy (al-Tadhkira fi 'ilm
   al-hay'a), 2 vols., Springer, New York, 1993.
 G. Saliba, A History of Arabic Astronomy: Planetary Theories during the
   Golden Age of Islam, New York University Press, 1995.
================================================================================
"""

import numpy as np

# One seed for the whole file. Every number printed below is reproducible.
RNG = np.random.default_rng(1201)          # the year he was born in Tus

TWO_PI = 2.0 * np.pi


# ==============================================================================
# SECTION 0 -- SENTINELS
#
# The architecture needs to be able to return something that is not a number.
# "As for Mercury, it has not yet been possible for me to conceive how it
# should be done." A float cannot say that. This class can.
# ==============================================================================

class Unconstituted:
    """
    Returned in place of a prediction when the target contains structure that
    no admissible arrangement can represent.

    This is deliberately NOT a NaN and NOT a large error value. Both of those
    are numbers, and a number invites downstream code to keep computing. This
    object raises if anyone tries arithmetic on it. The failure is meant to be
    load-bearing.
    """

    def __init__(self, channel, reason, residual_amplitude):
        self.channel = channel
        self.reason = reason
        self.residual_amplitude = float(residual_amplitude)

    def __repr__(self):
        return (f"UNCONSTITUTED(channel={self.channel!r}, reason={self.reason!r}, "
                f"residual={self.residual_amplitude:.6g})")

    # Any attempt to treat this as a number is an error, on purpose.
    def __float__(self):
        raise TypeError(
            f"channel {self.channel!r} was declared unconstituted and has no "
            f"numeric value; reason: {self.reason}")

    __add__ = __radd__ = __mul__ = __rmul__ = __sub__ = __rsub__ = __float__


# ==============================================================================
# SECTION 1 -- THE PRIMITIVE
#
# A body turning uniformly. This is the only elementary thing in the file.
# Note what it does NOT have: no bias, no activation, no gate, no free additive
# offset. Every one of those would be a quantity doing work without being a
# body. The architecture's expressive power has to come from arrangement.
# ==============================================================================

class Rotator:
    """
    One uniformly turning body.

      radius : r      >= 0. A body has a real, non-negative extent.
      rate   : k      an INTEGER multiple of the base rate. See admissibility.
      phase  : phi    where it stood at t = 0.

    Its tip, as a complex number, at parameter t:   r * exp(i*(k*t + phi))

    ADMISSIBILITY OF THE RATE. k must be an integer. This is not a convenience.
    If rates are incommensurable the arrangement never returns to its starting
    configuration, and an arrangement that never closes is not a cosmos, it is
    a drift. Integer rates are what make the whole assembly exactly periodic,
    and exact periodicity is what lets it extrapolate (Task E) instead of
    merely interpolate.
    """

    __slots__ = ("radius", "rate", "phase", "name")

    def __init__(self, radius, rate, phase=0.0, name=""):
        self.radius = float(radius)
        self.rate = int(rate)
        self.phase = float(phase)
        self.name = name

    def tip(self, t):
        """Complex displacement contributed by this body at parameter t."""
        return self.radius * np.exp(1j * (self.rate * t + self.phase))

    def is_admissible(self):
        """A body must have a real non-negative radius and an integer rate."""
        return (self.radius >= 0.0
                and float(self.rate).is_integer()
                and np.isfinite(self.radius)
                and np.isfinite(self.phase))

    def __repr__(self):
        return (f"Rotator({self.name or '-'}: r={self.radius:.6f}, "
                f"k={self.rate:+d}, phi={self.phase:+.4f})")


class Couple:
    """
    Two equal-radius bodies at rates +k and -k. The second is CARRIED by the
    first: its centre rides on the first one's tip. It does not roll inside it.

    Tip locus:  2 r cos(k t - alpha/2) * exp(i alpha/2)
                = an exactly straight segment through the origin,
                  direction alpha/2, half-length 2r.

    So a couple is a single REAL degree of freedom with a direction: an
    oscillation, built from two rotations that individually go nowhere near a
    straight line. This is the whole trick and every learnable component below
    is one of these.
    """

    __slots__ = ("half_radius", "rate", "alpha", "name")

    def __init__(self, half_radius, rate, alpha=0.0, name=""):
        self.half_radius = float(half_radius)   # r; the segment half-length is 2r
        self.rate = int(rate)                   # k
        self.alpha = float(alpha)               # phase offset of the carried body
        self.name = name

    def tips(self, t):
        """
        Return (deferent tip, carried tip, resultant) as complex arrays.

        The two-step return is not decoration. It is the audit trail: it lets
        a caller SEE that the resultant was reached by carrying one turning
        body on another, rather than assert it. Tusi's insistence that the
        larger orb carries the smaller rather than rolls inside it is exactly
        an insistence that this decomposition be exhibited.
        """
        deferent = self.half_radius * np.exp(1j * (self.rate * t))
        carried = self.half_radius * np.exp(1j * (self.alpha - self.rate * t))
        return deferent, deferent + carried, deferent + carried

    def locus(self, t):
        """The resultant tip. Should be exactly rectilinear."""
        return (self.half_radius * np.exp(1j * (self.rate * t))
                + self.half_radius * np.exp(1j * (self.alpha - self.rate * t)))

    def as_oscillation(self, t):
        """
        The same thing read as a signed scalar displacement along its own axis.
        This is the form the learning machinery uses.

            amplitude * cos(k t - alpha/2),   amplitude = 2 * half_radius
        """
        return 2.0 * self.half_radius * np.cos(self.rate * t - 0.5 * self.alpha)

    def axis(self):
        """Unit vector of the straight segment the tip travels along."""
        return np.exp(0.5j * self.alpha)

    def decompose(self):
        """The two admissible bodies this couple is made of."""
        return (Rotator(self.half_radius, +self.rate, 0.0, f"{self.name}:deferent"),
                Rotator(self.half_radius, -self.rate, self.alpha, f"{self.name}:carried"))


# ==============================================================================
# SECTION 2 -- THE FOIL: AN EQUANT
#
# This is what Tusi refused, implemented honestly so the refusal costs something
# and can be measured rather than asserted.
#
# The Ptolemaic equant, in full: the epicycle centre P rides on a circle of
# radius R about a centre C. The Earth sits at O, a distance e on one side of
# C. The equant point E sits a distance e on the OTHER side. And the rule of
# motion is that the angle subtended at E increases uniformly.
#
# So P moves uniformly about E, but E is not the centre of P's circle and is
# not a body. It is a bookkeeping point. It predicts extremely well. It cannot
# be built. Tusi's whole planetary programme is the attempt to replace it with
# something that can be, and the couple is the tool he found for it.
# ==============================================================================

class EquantOracle:
    """
    Exact closed-form longitude of an equant-driven planet, as seen from Earth.

    Given mean anomaly M (uniform about the equant E), the epicycle centre P is
    the intersection of the ray from E in direction M with the deferent circle:

        |E + s*d - C| = R,  with E - C = e*u,  d = (cos M, sin M)
        =>  s^2 + 2 s e (u.d) + e^2 - R^2 = 0
        =>  s = -e cos M + sqrt(R^2 - e^2 sin^2 M)      (taking u = (1,0))

    and the vector from the Earth O = C - e*u to P is (2e + s cos M, s sin M),
    whose argument is the observed longitude.

    THE EQUATION OF CENTRE q(M) = lambda(M) - M is an ODD periodic function of
    M. Odd and periodic means: a pure sine series in INTEGER harmonics of M.
    Which means an arrangement of Tusi couples can represent it exactly in the
    limit, and to any required tolerance with finitely many. That is not a
    lucky coincidence, it is the reason the programme was possible at all.
    """

    def __init__(self, eccentricity=0.10, deferent_radius=1.0):
        self.e = float(eccentricity)
        self.R = float(deferent_radius)

    def longitude(self, M):
        e, R = self.e, self.R
        s = -e * np.cos(M) + np.sqrt(R * R - (e * np.sin(M)) ** 2)
        return np.arctan2(s * np.sin(M), 2.0 * e + s * np.cos(M))

    def equation_of_centre(self, M):
        """
        q(M) = observed longitude minus mean longitude, wrapped to (-pi, pi].

        The wrap is not cosmetic. lambda comes back from arctan2 on the branch
        (-pi, pi] while M runs on forever, so the raw difference carries a
        sawtooth of whole revolutions that has nothing to do with the planet.
        The inequality is the small residue after those revolutions are taken
        out -- which is exactly what a medieval table records and exactly what
        an arrangement of orbs is asked to produce.
        """
        d = self.longitude(M) - M
        return (d + np.pi) % TWO_PI - np.pi

    # -- the audit that fails -------------------------------------------------
    @staticmethod
    def audit():
        """
        Why the equant is rejected even though it fits.

        Returned as data rather than prose so the training loop can print it
        next to the numbers and the two can be compared directly.
        """
        return {
            "verdict": "INADMISSIBLE",
            "grounds": [
                "the equant point is not the centre of the circle whose motion "
                "it governs, so no orb turns about it",
                "it is a point of reference only: nothing is there, and nothing "
                "there could be made to turn",
                "the motion it prescribes is uniform about a place and "
                "non-uniform about the body's own centre, which the physics "
                "of the discipline does not permit",
            ],
            "what_it_costs_to_reject": "one parameter becomes many orbs",
            "what_it_buys": "every remaining quantity names a thing that turns",
        }


# ==============================================================================
# SECTION 3 -- THE ARRANGEMENT
#
# An arrangement is an ordered list of orbs. Each orb is a couple (one real
# oscillation) plus an INCLINATION iota, which is where commitment (3) lives.
#
# An orb inclined at iota to the reference plane cannot contribute to latitude
# alone. Its oscillation projects onto BOTH observable channels:
#
#       longitude gets  amplitude * cos(iota) * sin(k M + phi)
#       latitude  gets  amplitude * sin(iota) * sin(k M + phi)
#
# The two projections are locked together by one angle. You do not get to
# choose them independently, and you do not get to declare one of them
# imperceptible. That is Tusi's argument against Ptolemy's latitude circles,
# transcribed into a forward pass.
#
# PARAMETERISATION. Learnable continuous parameters per orb: amplitude a,
# phase phi, inclination iota. The rate k is an integer and is chosen by
# discrete search, not by gradient, because an integer is not something you
# can descend towards. This split is deliberate and is the hay'a-admissibility
# constraint doing real work on the optimiser.
# ==============================================================================

class Orb:
    """
    One inclined oscillating component. Physically: three nested spheres.

    FREEDOM MASK. Not every parameter of a real orb is yours to choose. An
    orb's INCLINATION to the ecliptic is a fact about the world that the
    observations fix; it is not a dial you may turn to make an inconvenient
    side effect go away. So `free` marks, per parameter, what the fitter is
    allowed to touch. Leaving iota free on a latitude orb would let the
    optimiser tilt it to 90 degrees and quietly abolish the very side effect
    the architecture exists to confront -- which is exactly the move Tusi
    refused when he denied that a displacement could be perceptible in
    latitude and imperceptible in longitude.
    """

    __slots__ = ("a", "k", "phi", "iota", "role", "frozen", "free")

    def __init__(self, a, k, phi=0.0, iota=0.0, role="general", frozen=False,
                 free=(True, True, True)):
        self.a = float(a)         # amplitude of the oscillation (= 2 * couple half-radius)
        self.k = int(k)           # integer harmonic of the base rate
        self.phi = float(phi)     # phase
        self.iota = float(iota)   # inclination to the reference plane
        self.role = role          # "general" | "latitude" | "enclosing"
        self.frozen = frozen      # enclosing orbs are set analytically, not learned
        self.free = tuple(bool(x) for x in free)   # (a, phi, iota) learnable?

    def oscillation(self, M):
        return self.a * np.sin(self.k * M + self.phi)

    def to_longitude(self, M):
        return np.cos(self.iota) * self.oscillation(M)

    def to_latitude(self, M):
        return np.sin(self.iota) * self.oscillation(M)

    def as_couple(self):
        """
        The two turning bodies this orb is actually made of.
        a sin(kM + phi) = 2r cos(kM - alpha/2) with r = a/2 and alpha chosen so
        the cosine becomes the required sine.
        """
        return Couple(half_radius=abs(self.a) / 2.0,
                      rate=self.k,
                      alpha=2.0 * (self.phi - np.pi / 2.0) * (-1.0),
                      name=self.role)

    def __repr__(self):
        return (f"Orb({self.role}: a={self.a:+.6f}, k={self.k:+d}, "
                f"phi={self.phi:+.4f}, iota={np.degrees(self.iota):+.3f}deg"
                f"{', FROZEN' if self.frozen else ''})")


class ResidualLedger:
    """
    What the cancellation failed to cancel.

    Tusi publishes his residuals. Having built the curvilinear version of the
    couple he immediately writes: "However, it requires that the time the
    diameter is in the north is equal to the time it is in the south. However,
    in reality it is different than that." He states the flaw in the same
    breath as the construction. This class is that habit, made compulsory.
    """

    def __init__(self):
        self.rows = []

    def record(self, source, channel, before, after, cause):
        self.rows.append({
            "source": source, "channel": channel,
            "amplitude_before_cancellation": float(before),
            "amplitude_after_cancellation": float(after),
            "cause_of_remainder": cause,
        })

    def total(self, channel=None):
        return sum(r["amplitude_after_cancellation"] for r in self.rows
                   if channel is None or r["channel"] == channel)

    def render(self, indent="    "):
        if not self.rows:
            return indent + "(nothing to declare)"
        w = max(len(r["source"]) for r in self.rows)
        out = [f"{indent}{'source'.ljust(w)}  channel     induced      "
               f"remaining    cause"]
        out.append(indent + "-" * (w + 58))
        for r in self.rows:
            out.append(
                f"{indent}{r['source'].ljust(w)}  {r['channel']:<10s}  "
                f"{r['amplitude_before_cancellation']:>9.3e}  "
                f"{r['amplitude_after_cancellation']:>9.3e}    "
                f"{r['cause_of_remainder']}")
        return "\n".join(out)


class Arrangement:
    """
    An ordered assembly of orbs, plus the two mechanisms that make it Tusi's
    rather than merely harmonic:

      MuhitaLayer   -- the enclosing sphere: an extra orb added for no purpose
                       but to cancel a disturbance another orb necessarily
                       caused. It costs a real component and it is counted.
      declare()     -- the Mercury clause.

    Prediction, per channel:
        longitude(M) = M + sum_j cos(iota_j) a_j sin(k_j M + phi_j)
        latitude(M)  =     sum_j sin(iota_j) a_j sin(k_j M + phi_j)
    """

    # A real orb has to be turned on a lathe. Its radius is manufacturable to
    # the precision of the workshop and no finer. Maragha's mural quadrant read
    # to roughly an arcminute; this is that precision expressed as a fraction
    # of a unit radius. It is the reason cancellation is never exact.
    CONSTRUCTION_QUANTUM = 1.0 / 3438.0     # 1 arcminute in radians, near enough

    def __init__(self, orbs=None, quantise=True):
        self.orbs = list(orbs) if orbs else []
        self.ledger = ResidualLedger()
        self.quantise = quantise
        self.ancestor = None          # for tahrir / recension
        self.provenance = []          # human-readable history of revisions

    # -- structure ------------------------------------------------------------

    def add(self, orb):
        self.orbs.append(orb)
        return self

    _NAMES = ("a", "phi", "iota")

    def slots(self):
        """
        The trainable degrees of freedom, as (orb_index, parameter_name) pairs.

        An orb contributes a slot only if it is not frozen AND that particular
        parameter is marked free. Frozen orbs (the enclosing spheres) have no
        slots at all: they are computed from the cancellation requirement, not
        fitted, and letting a fitter touch them would defeat their purpose.
        """
        out = []
        for i, o in enumerate(self.orbs):
            if o.frozen:
                continue
            for j, nm in enumerate(self._NAMES):
                if o.free[j]:
                    out.append((i, nm))
        return out

    def learnable(self):
        """Indices of orbs with at least one trainable parameter."""
        return sorted({i for i, _ in self.slots()})

    def n_params(self):
        return len(self.slots())

    def get_params(self):
        return np.array([getattr(self.orbs[i], nm) for i, nm in self.slots()],
                        dtype=float)

    def set_params(self, v):
        for n, (i, nm) in enumerate(self.slots()):
            setattr(self.orbs[i], nm, float(v[n]))

    # -- forward --------------------------------------------------------------

    def forward(self, M):
        """
        Returns (longitude_correction, latitude), both arrays over M.
        The mean motion M itself is not modelled -- it is the clock.
        """
        lon = np.zeros_like(M)
        lat = np.zeros_like(M)
        for o in self.orbs:
            s = o.oscillation(M)
            lon = lon + np.cos(o.iota) * s
            lat = lat + np.sin(o.iota) * s
        return lon, lat

    def longitude(self, M):
        return M + self.forward(M)[0]

    # -- admissibility --------------------------------------------------------

    def audit(self):
        """
        Check every component is a thing that could turn. Returns (ok, faults).

        This is the equant test, generalised: walk the parameters and ask of
        each one what body it names. Anything that cannot answer is a fault.
        """
        faults = []
        for j, o in enumerate(self.orbs):
            if not float(o.k).is_integer():
                faults.append(f"orb {j} ({o.role}): rate {o.k} is not an integer "
                              f"-- the arrangement would never close")
            if not np.isfinite(o.a) or not np.isfinite(o.phi) or not np.isfinite(o.iota):
                faults.append(f"orb {j} ({o.role}): non-finite parameter")
            c = o.as_couple()
            if c.half_radius < 0:
                faults.append(f"orb {j} ({o.role}): negative radius")
        return (len(faults) == 0), faults

    def bill_of_materials(self):
        """
        Every turning body the arrangement requires, itemised. An arrangement
        you cannot itemise is one you cannot build.
        """
        bodies = []
        for o in self.orbs:
            d, c = o.as_couple().decompose()
            bodies.append((f"{o.role}[k={o.k:+d}]", "deferent", d.radius, +o.k))
            bodies.append((f"{o.role}[k={o.k:+d}]", "carried", c.radius, -o.k))
        return bodies

    # -- the enclosing sphere -------------------------------------------------

    def install_enclosing_spheres(self):
        """
        MuhitaLayer.

        For every orb introduced to serve LATITUDE, compute the longitude
        displacement it unavoidably also produces, and add an enclosing orb
        (iota = 0, so it touches longitude only) that cancels it.

        The catch, and the whole point: the cancelling orb has to be BUILT. Its
        amplitude is quantised to the workshop's precision. So the cancellation
        is exact only up to that quantum, and the remainder is recorded rather
        than swallowed. The model pays for its own side effects in hardware and
        then admits what it still owes.
        """
        existing = list(self.orbs)
        for o in existing:
            if o.role != "latitude":
                continue
            induced = o.a * np.cos(o.iota)          # what it does to longitude
            wanted = -induced                        # what must be cancelled
            if self.quantise:
                q = self.CONSTRUCTION_QUANTUM
                built = np.round(wanted / q) * q     # what can actually be made
            else:
                built = wanted
            remainder = abs(wanted - built)
            self.orbs.append(Orb(a=built, k=o.k, phi=o.phi, iota=0.0,
                                 role="enclosing", frozen=True))
            self.ledger.record(
                source=f"latitude orb k={o.k:+d}",
                channel="longitude",
                before=abs(induced),
                after=remainder,
                cause=("radius quantised to construction precision"
                       if self.quantise else "none -- exact"))
        return self

    def refresh_enclosing_spheres(self):
        """
        Re-derive the enclosing orbs from the CURRENT state of what they cancel.

        This is needed because fitting moves the latitude orb, and a
        cancellation computed against yesterday's amplitude cancels yesterday's
        disturbance. Tusi took twenty-five years over this device, revising it
        three times; the reason is visible here in miniature. Correction and
        fit have to be alternated until they stop arguing with each other.
        """
        self.orbs = [o for o in self.orbs if o.role != "enclosing"]
        self.ledger = ResidualLedger()
        return self.install_enclosing_spheres()

    def second_channel_contamination(self, M, k):
        """
        How much of the longitude signal sits at harmonic k -- i.e. the size of
        the specific damage an orb at rate k does to the channel it was not
        introduced to serve. Measured by projection, so it is the real quantity
        and not a proxy for it.
        """
        lon, _ = self.forward(M)
        A = np.stack([np.sin(k * M), np.cos(k * M)], axis=1)
        coef, *_ = np.linalg.lstsq(A, lon, rcond=None)
        return float(np.hypot(coef[0], coef[1]))

    # -- the Mercury clause ---------------------------------------------------

    def declare(self, M, target_lon, target_lat=None, tolerance=1e-3):
        """
        After fitting, ask of each channel whether what is left over is
        something an admissible arrangement could still absorb.

        The test: take the residual, and ask how much of its energy sits at
        INTEGER harmonics of the base rate. Anything at integer harmonics is
        reachable -- add more orbs. Anything that is NOT is unreachable in
        principle, no matter how many orbs are added, because every admissible
        body turns at an integer rate.

        If the unreachable part exceeds tolerance, the channel is declared
        UNCONSTITUTED. It does not get a number.
        """
        out = {}
        pred_lon, pred_lat = self.forward(M)
        checks = []
        if target_lon is not None:
            checks.append(("longitude", target_lon - pred_lon))
        if target_lat is not None:
            checks.append(("latitude", target_lat - pred_lat))

        for name, resid in checks:
            reachable, unreachable = self._split_residual(M, resid)
            if unreachable > tolerance:
                out[name] = Unconstituted(
                    channel=name,
                    reason=("residual power sits at rates that are not integer "
                            "multiples of the base rate; no arrangement of "
                            "uniformly turning bodies can reach it"),
                    residual_amplitude=unreachable)
            else:
                out[name] = float(np.sqrt(np.mean(resid ** 2)))
        return out

    @staticmethod
    def _split_residual(M, resid, kmax=64):
        """
        Project the residual onto integer harmonics of M and return
        (reachable RMS, unreachable RMS).

        Uses a least-squares projection onto {sin(kM), cos(kM)} for integer k,
        which is the exact span of what admissible orbs can produce. What the
        projection cannot explain is what no orb can.

        THE WINDOW HAS TO BE LONG. Over a single revolution, integer harmonics
        span essentially everything -- one period of any function has a Fourier
        series. Incommensurability is invisible in one turn and only shows up
        when the arrangement has come back to its starting configuration and
        the sky has not. So this test is meaningless unless M covers several
        base periods, and the caller is warned when it does not. That is the
        methodological content of a thirty-year observing programme: the
        question "does this close?" cannot be asked in one season.
        """
        span = float(M.max() - M.min())
        if span < 3.0 * TWO_PI:
            import warnings
            warnings.warn(
                f"declaration window spans only {span / TWO_PI:.2f} base periods; "
                f"incommensurability is not detectable in under ~3", stacklevel=2)
        cols = [np.ones_like(M)]
        for k in range(1, kmax + 1):
            cols.append(np.sin(k * M))
            cols.append(np.cos(k * M))
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        explained = A @ coef
        left = resid - explained
        return float(np.sqrt(np.mean(explained ** 2))), float(np.sqrt(np.mean(left ** 2)))

    # -- tahrir: non-destructive recension ------------------------------------

    def recense(self, note):
        """
        Take a snapshot of the present state as the ancestor, then continue.

        Tusi's working method: he does not replace Ptolemy, he re-derives him;
        he does not delete the Hall's lunar model, he supersedes it in the
        Tadhkira and the earlier one stays legible and stays wrong in public.
        (It is because the earlier version survived that we can date the
        Byzantine transmission: Chioniades carried the 1245 model, error and
        all, into Greek around 1300.)

        An architecture that overwrites its own past cannot be audited by
        anyone who was not watching at the time.
        """
        import copy
        snap = Arrangement([copy.copy(o) for o in self.orbs], quantise=self.quantise)
        snap.ancestor = self.ancestor
        snap.provenance = list(self.provenance)
        self.ancestor = snap
        self.provenance = list(self.provenance) + [note]
        return self

    def fidelity_to_ancestor(self, M):
        """How far the current version has moved from the one it supersedes."""
        if self.ancestor is None:
            return None
        a, _ = self.forward(M)
        b, _ = self.ancestor.forward(M)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def __repr__(self):
        return (f"Arrangement({len(self.orbs)} orbs, "
                f"{len(self.bill_of_materials())} turning bodies)")


# ==============================================================================
# SECTION 4 -- LEARNING
#
# Analytic gradients. The loss is mean squared error summed over the channels
# that are being trained.
#
#   L = (1/2N) sum_n [ (lon_pred - lon_tgt)^2 + w * (lat_pred - lat_tgt)^2 ]
#
#   lon_pred = sum_j cos(iota_j) a_j sin(k_j M + phi_j)
#   lat_pred = sum_j sin(iota_j) a_j sin(k_j M + phi_j)
#
# with s_j = sin(k_j M + phi_j), c_j = cos(k_j M + phi_j),
#      eL = lon_pred - lon_tgt,  eB = lat_pred - lat_tgt:
#
#   dL/da_j    = (1/N) < eL * cos(iota_j) * s_j  +  w * eB * sin(iota_j) * s_j >
#   dL/dphi_j  = (1/N) < eL * cos(iota_j) * a_j * c_j
#                       + w * eB * sin(iota_j) * a_j * c_j >
#   dL/diota_j = (1/N) < -eL * sin(iota_j) * a_j * s_j
#                       + w * eB * cos(iota_j) * a_j * s_j >
#
# Frozen (enclosing) orbs contribute to the forward pass and to the error but
# receive no gradient: they are set by the analytic cancellation requirement,
# not by fitting. That is the point of them.
# ==============================================================================

def loss_and_grad(arr, M, tgt_lon, tgt_lat=None, w_lat=1.0):
    """Mean-squared loss and its analytic gradient w.r.t. learnable params."""
    pred_lon, pred_lat = arr.forward(M)
    N = M.size

    eL = pred_lon - tgt_lon
    if tgt_lat is not None:
        eB = pred_lat - tgt_lat
    else:
        eB = np.zeros_like(M)
        w_lat = 0.0

    loss = 0.5 * (np.mean(eL ** 2) + w_lat * np.mean(eB ** 2))

    slots = arr.slots()
    g = np.zeros(len(slots))
    for n, (j, nm) in enumerate(slots):
        o = arr.orbs[j]
        ang = o.k * M + o.phi
        s, c = np.sin(ang), np.cos(ang)
        ci, si = np.cos(o.iota), np.sin(o.iota)
        if nm == "a":
            g[n] = np.mean(eL * ci * s) + w_lat * np.mean(eB * si * s)
        elif nm == "phi":
            g[n] = (np.mean(eL * ci * o.a * c)
                    + w_lat * np.mean(eB * si * o.a * c))
        else:  # iota
            g[n] = (np.mean(eL * (-si) * o.a * s)
                    + w_lat * np.mean(eB * ci * o.a * s))
    return loss, g


def gradient_check(arr, M, tgt_lon, tgt_lat=None, w_lat=1.0, eps=1e-6):
    """
    MANDATORY. Central-difference check of every analytic gradient component.

    Returns (max_relative_error, table). This runs in the self-tests and the
    file refuses to report success if it fails.
    """
    _, g_an = loss_and_grad(arr, M, tgt_lon, tgt_lat, w_lat)
    p0 = arr.get_params()
    g_fd = np.zeros_like(p0)
    for i in range(p0.size):
        p = p0.copy(); p[i] += eps; arr.set_params(p)
        lp, _ = loss_and_grad(arr, M, tgt_lon, tgt_lat, w_lat)
        p = p0.copy(); p[i] -= eps; arr.set_params(p)
        lm, _ = loss_and_grad(arr, M, tgt_lon, tgt_lat, w_lat)
        g_fd[i] = (lp - lm) / (2 * eps)
    arr.set_params(p0)

    denom = np.maximum(1e-12, np.abs(g_an) + np.abs(g_fd))
    rel = np.abs(g_an - g_fd) / denom
    return float(rel.max()), list(zip(g_an, g_fd, rel))


def train(arr, M, tgt_lon, tgt_lat=None, w_lat=1.0,
          steps=4000, lr=0.35, verbose_every=None):
    """
    Plain gradient descent with a mild adaptive step. Deliberately unfancy:
    the interesting constraint in this architecture is what the parameters ARE,
    not how they are updated.

    After every step the arrangement is projected back onto the admissible set
    (amplitudes kept real, rates left alone because they are integers and are
    not touched by the gradient at all). That projection is the hay'a
    constraint acting as a hard boundary rather than a penalty term.
    """
    slots = arr.slots()
    p = arr.get_params()
    prev = np.inf
    for t in range(steps):
        arr.set_params(p)
        loss, g = loss_and_grad(arr, M, tgt_lon, tgt_lat, w_lat)
        if loss > prev:
            lr *= 0.7                     # overshot; shorten the step
        else:
            lr *= 1.01
        prev = loss
        p = p - lr * g
        # projection onto the admissible set: angles wrap, nothing else is free
        for n, (_, nm) in enumerate(slots):
            if nm in ("phi", "iota"):
                p[n] = (p[n] + np.pi) % TWO_PI - np.pi
        if verbose_every and t % verbose_every == 0:
            print(f"      step {t:5d}   loss {loss:.6e}   lr {lr:.4f}")
    arr.set_params(p)
    return loss_and_grad(arr, M, tgt_lon, tgt_lat, w_lat)[0]


def choose_rates(M, target, n_orbs, kmax=24):
    """
    Rates are integers, so they are chosen by search, not by descent.

    Greedy matching pursuit over the integer harmonics: repeatedly take the
    harmonic that explains most of what is left, subtract it, continue. This is
    exactly how the historical programme proceeded -- one orb at a time, each
    added because a specific unexplained inequality demanded it.

    Returns a list of (rate, amplitude, phase) triples: the search does not
    merely name the harmonic, it sizes the orb. Descent then refines what the
    projection already got approximately right, which is the difference
    between a fitter that converges and one that wanders.
    """
    resid = target.copy()
    chosen = []
    for _ in range(n_orbs):
        best_k, best_gain, best_ab = None, -1.0, (0.0, 0.0)
        for k in range(1, kmax + 1):
            if k in [c for c, _, _ in chosen]:
                continue
            # least squares for a*sin(kM+phi) = A sin(kM) + B cos(kM)
            A = np.stack([np.sin(k * M), np.cos(k * M)], axis=1)
            coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
            gain = float(np.sum((A @ coef) ** 2))
            if gain > best_gain:
                best_k, best_gain, best_ab = k, gain, (coef[0], coef[1])
        A_, B_ = best_ab
        # A sin + B cos  ==  a sin(kM + phi)  with a = hypot, phi = atan2(B, A)
        chosen.append((best_k, float(np.hypot(A_, B_)), float(np.arctan2(B_, A_))))
        resid = resid - (A_ * np.sin(best_k * M) + B_ * np.cos(best_k * M))
    return chosen


# ==============================================================================
# SECTION 5 -- THE BASELINE
#
# A small MLP in pure NumPy, for Task E. Not a straw man: it gets the same
# data, a comparable parameter budget, and a competent optimiser. It will win
# inside the training window and lose catastrophically outside it, and the
# reason is structural rather than a matter of tuning.
# ==============================================================================

class TinyMLP:
    """1 -> H -> H -> 1, tanh. Adam. Pure NumPy, analytic backprop."""

    def __init__(self, hidden=24, seed=7):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1.0, (1, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1.0 / np.sqrt(hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = r.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.b3 = np.zeros(1)
        self._m = {}; self._v = {}; self._t = 0

    def _params(self):
        return ["W1", "b1", "W2", "b2", "W3", "b3"]

    def forward(self, x):
        x = x.reshape(-1, 1)
        z1 = x @ self.W1 + self.b1;  a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = np.tanh(z2)
        y = a2 @ self.W3 + self.b3
        return y.ravel(), (x, z1, a1, z2, a2)

    def fit(self, x, y, steps=6000, lr=8e-3):
        for _ in range(steps):
            pred, (xi, z1, a1, z2, a2) = self.forward(x)
            e = (pred - y).reshape(-1, 1) / x.size
            gW3 = a2.T @ e;              gb3 = e.sum(0)
            d2 = (e @ self.W3.T) * (1 - a2 ** 2)
            gW2 = a1.T @ d2;             gb2 = d2.sum(0)
            d1 = (d2 @ self.W2.T) * (1 - a1 ** 2)
            gW1 = xi.T @ d1;             gb1 = d1.sum(0)
            self._adam({"W1": gW1, "b1": gb1, "W2": gW2,
                        "b2": gb2, "W3": gW3, "b3": gb3}, lr)
        return float(np.mean((self.forward(x)[0] - y) ** 2))

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m.get(k, 0) + (1 - b1) * g
            self._v[k] = b2 * self._v.get(k, 0) + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))


# ==============================================================================
# SECTION 6 -- TASKS
# ==============================================================================

def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def task_A_exactness():
    """
    The couple traces a straight line EXACTLY, not nearly.

    Tusi's whole move depends on this being an identity rather than an
    approximation. If it were only close, he would have gained nothing over
    Ptolemy: an approximate device that saves the appearances is precisely what
    he was objecting to. So this is checked to machine precision, and the
    deviation is reported in units of the double-precision epsilon.
    """
    rule("TASK A -- the couple is exactly rectilinear (a proof obligation)")
    t = np.linspace(0, 4 * TWO_PI, 20001)
    results = []
    for (r, k, alpha) in [(1.0, 1, 0.0), (0.37, 3, 1.1), (2.5, 7, -2.3),
                          (0.9, 2, np.pi / 2)]:
        cpl = Couple(half_radius=r, rate=k, alpha=alpha)
        z = cpl.locus(t)
        axis = cpl.axis()
        # component perpendicular to the intended axis; should be identically 0
        perp = np.abs(np.imag(z * np.conj(axis)))
        along = np.real(z * np.conj(axis))
        results.append((r, k, alpha, perp.max(), along.max(), 2 * r))
        print(f"   r={r:<5.2f} k={k:+d} alpha={alpha:+.3f}  "
              f"max |perpendicular| = {perp.max():.3e}  "
              f"({perp.max() / np.finfo(float).eps:8.1f} eps)   "
              f"half-length {along.max():.6f} (predicted {2 * r:.6f})")
    worst = max(r[3] for r in results)
    print(f"\n   worst perpendicular excursion over all cases: {worst:.3e}")
    print("   the locus is a straight segment, not an approximation to one.")

    # And the audit trail: exhibit the two bodies.
    d, tip, _ = Couple(1.0, 1, 0.7).tips(np.array([0.0, 0.9, 1.8]))
    print("\n   carried, not rolled -- the decomposition is exhibited:")
    for i in range(3):
        print(f"      deferent tip {d[i]:+.4f}   ->   resultant {tip[i]:+.4f}")
    return worst


def task_B_constitute_equant():
    """
    Replace the equant with an arrangement of orbs, and watch the error fall.

    This is the historical programme in miniature. The equant fits with one
    parameter and cannot be built. The arrangement needs several orbs and can.
    """
    rule("TASK B -- constituting the equant out of admissible bodies")
    eq = EquantOracle(eccentricity=0.10)
    M = np.linspace(0, TWO_PI, 721, endpoint=False)
    q = eq.equation_of_centre(M)
    print(f"   target: equation of centre of an equant model, e = {eq.e}")
    print(f"           amplitude {np.degrees(np.abs(q).max()):.4f} deg "
          f"({np.abs(q).max():.6f} rad)")

    a = EquantOracle.audit()
    print(f"\n   audit of the equant device: {a['verdict']}")
    for gline in a["grounds"]:
        print(f"      - {gline}")

    print("\n   fitting arrangements of increasing size:")
    print(f"      {'orbs':>4}  {'bodies':>6}  {'RMS (rad)':>12}  {'RMS (arcsec)':>13}   rates")
    history = []
    final = None
    for n in (1, 2, 3, 4, 5, 6):
        seeds = choose_rates(M, q, n_orbs=n, kmax=12)
        arr = Arrangement([Orb(a=a0, k=k, phi=p0, iota=0.0, role="general",
                               free=(True, True, False))
                           for (k, a0, p0) in seeds])
        train(arr, M, tgt_lon=q, steps=2500, lr=0.4)
        pred, _ = arr.forward(M)
        rms = float(np.sqrt(np.mean((pred - q) ** 2)))
        nb = len(arr.bill_of_materials())
        print(f"      {n:>4}  {nb:>6}  {rms:>12.4e}  "
              f"{np.degrees(rms) * 3600:>13.4f}   {[k for k, _, _ in seeds]}")
        history.append((n, rms))
        final = arr

    print("\n   the last arrangement, itemised as things that turn:")
    for nm, kind, rad, rate in final.bill_of_materials():
        print(f"      {nm:<16s} {kind:<9s} radius {rad:.8f}  rate {rate:+d}")
    ok, faults = final.audit()
    print(f"\n   admissibility audit: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f" {faults}"))
    print("   every quantity in the model names a body. The equant's did not.")
    return history, final, (M, q)


def task_C_side_effects():
    """
    An orb introduced for latitude damages longitude. Cancel it in hardware,
    then publish what the cancellation could not reach.
    """
    rule("TASK C -- the side-effect ledger and the enclosing sphere")
    M = np.linspace(0, TWO_PI, 721, endpoint=False)

    # Ground truth. The inclination is a FACT about this planet's orbit, fixed
    # by observation. The target latitude is what an orb at that inclination
    # produces; the target longitude is what the observations actually show,
    # and it contains NO second-harmonic term. So whatever the latitude orb
    # spills into longitude at k=2 is pure damage.
    true_iota = np.radians(7.0)
    true_amp = 0.030
    tgt_lat = true_amp * np.sin(true_iota) * np.sin(2 * M + 0.4)
    tgt_lon = 0.045 * np.sin(1 * M + 0.9) + 0.012 * np.sin(3 * M - 0.2)

    print(f"   orbital inclination (given by observation, not fitted): "
          f"{np.degrees(true_iota):.2f} deg")
    print(f"   target latitude:  amplitude {np.abs(tgt_lat).max():.6f} rad "
          f"({np.degrees(np.abs(tgt_lat).max()) * 60:.2f} arcmin)")
    print(f"   target longitude: amplitude {np.abs(tgt_lon).max():.6f} rad "
          f"-- and contains no second harmonic at all")

    # --- stage 1: fit latitude with an inclined orb, ignoring the consequence
    # iota is NOT free: the fitter may not tilt the orbit flat to make its own
    # arithmetic tidy.
    arr = Arrangement([
        Orb(a=0.02, k=2, phi=0.0, iota=true_iota, role="latitude",
            free=(True, True, False)),
        Orb(a=0.04, k=1, phi=0.9, iota=0.0, role="general",
            free=(True, True, False)),
        Orb(a=0.01, k=3, phi=-0.2, iota=0.0, role="general",
            free=(True, True, False)),
    ])
    train(arr, M, tgt_lon=tgt_lon, tgt_lat=tgt_lat, w_lat=200.0, steps=6000, lr=0.5)
    lon0, lat0 = arr.forward(M)
    lat_rms0 = np.sqrt(np.mean((lat0 - tgt_lat) ** 2))
    lon_rms0 = np.sqrt(np.mean((lon0 - tgt_lon) ** 2))
    contam0 = arr.second_channel_contamination(M, k=2)

    lat_orb = arr.orbs[0]
    induced = abs(lat_orb.a * np.cos(lat_orb.iota))
    n_before = len(arr.bill_of_materials())
    print(f"\n   stage 1 -- fit both channels, side effect left unaddressed")
    print(f"      latitude  RMS                       {lat_rms0:.4e}")
    print(f"      longitude RMS                       {lon_rms0:.4e}")
    print(f"      SECOND-HARMONIC content of longitude {contam0:.4e} rad "
          f"({np.degrees(contam0) * 60:.2f} arcmin)")
    print(f"      the latitude orb sits at iota = {np.degrees(lat_orb.iota):.2f} deg,")
    print(f"      so it displaces longitude by a*cos(iota) = {induced:.6f} rad")
    print("      whether anyone wants it to or not. The observations show no")
    print("      second harmonic. It is not possible to say the displacement is")
    print("      perceptible in latitude and imperceptible in longitude: they")
    print("      are the same orb, seen twice.")

    # --- stage 2: install the enclosing sphere, alternating with the fit,
    #     because a cancellation computed against yesterday's amplitude
    #     cancels yesterday's disturbance.
    arr.install_enclosing_spheres()
    for _ in range(4):
        train(arr, M, tgt_lon=tgt_lon, tgt_lat=tgt_lat, w_lat=200.0,
              steps=2500, lr=0.5)
        arr.refresh_enclosing_spheres()
    train(arr, M, tgt_lon=tgt_lon, tgt_lat=tgt_lat, w_lat=200.0, steps=2500, lr=0.5)
    arr.refresh_enclosing_spheres()

    lon1, lat1 = arr.forward(M)
    lat_rms1 = np.sqrt(np.mean((lat1 - tgt_lat) ** 2))
    lon_rms1 = np.sqrt(np.mean((lon1 - tgt_lon) ** 2))
    contam1 = arr.second_channel_contamination(M, k=2)
    n_after = len(arr.bill_of_materials())

    print(f"\n   stage 2 -- enclosing sphere installed, and paid for in bodies")
    print(f"      turning bodies: {n_before} -> {n_after} "
          f"(+{n_after - n_before} that serve no observation whatever,")
    print(f"      and exist only to undo what another orb could not help doing)")
    print(f"      latitude  RMS                       {lat_rms1:.4e}")
    print(f"      longitude RMS                       {lon_rms1:.4e}")
    print(f"      SECOND-HARMONIC content of longitude {contam1:.4e} rad "
          f"({np.degrees(contam1) * 60:.2f} arcmin)")
    print(f"      contamination reduced by a factor of "
          f"{contam0 / max(contam1, 1e-18):.1f}")

    print(f"\n   the ledger -- what the cancellation could not reach:")
    print(arr.ledger.render())
    print(f"\n      construction quantum: {Arrangement.CONSTRUCTION_QUANTUM:.3e} rad "
          f"(~1 arcmin; an orb has to be turned on a lathe)")
    print(f"      total undeclared-elsewhere remainder: "
          f"{arr.ledger.total():.4e} rad "
          f"({np.degrees(arr.ledger.total()) * 3600:.3f} arcsec)")
    print("      this number is published, not absorbed.")
    return arr, (M, tgt_lon, tgt_lat)


def task_D_mercury():
    """
    A target no admissible arrangement can reach. The right answer is to say so.
    """
    rule("TASK D -- Mercury: the declared non-solution")
    # EIGHT base periods. One is not enough: see Arrangement._split_residual.
    N_PERIODS = 8
    M = np.linspace(0, N_PERIODS * TWO_PI, 4000, endpoint=False)
    print(f"   observing window: {N_PERIODS} complete revolutions, {M.size} points")
    print("   (an arrangement's closure cannot be tested inside a single turn)")

    # A well-behaved target: everything at integer rates.
    reachable = 0.04 * np.sin(1 * M + 0.3) + 0.011 * np.sin(3 * M - 0.8)

    # An unreachable one: an incommensurable rate. No arrangement of uniformly
    # turning bodies at integer rates can produce it, at any size, ever.
    golden = (1 + np.sqrt(5)) / 2
    unreachable = reachable + 0.02 * np.sin(golden * 7.0 * M + 0.15)

    for label, tgt in (("commensurable target", reachable),
                       ("incommensurable target", unreachable)):
        seeds = choose_rates(M, tgt, n_orbs=6, kmax=24)
        arr = Arrangement([Orb(a=a0, k=k, phi=p0, role="general",
                               free=(True, True, False))
                           for (k, a0, p0) in seeds])
        train(arr, M, tgt_lon=tgt, steps=3000, lr=0.4)
        verdict = arr.declare(M, target_lon=tgt, tolerance=1e-3)
        pred, _ = arr.forward(M)
        rms = float(np.sqrt(np.mean((pred - tgt) ** 2)))
        reach, unreach = Arrangement._split_residual(M, tgt - pred)
        print(f"\n   {label}")
        print(f"      rates chosen        : {[k for k, _, _ in seeds]}")
        print(f"      achieved RMS        : {rms:.4e}")
        print(f"      residual still reachable by more orbs : {reach:.4e}")
        print(f"      residual NO orb can ever reach        : {unreach:.4e}")
        print(f"      declaration         : {verdict['longitude']}")

    print("\n   The second target is not merely hard. It is outside the span of")
    print("   every arrangement that could ever be built out of the one")
    print("   admissible primitive. A model that returned a number there would")
    print("   be reporting a fit it does not have. This one says so instead.")


def task_E_generalisation():
    """
    Extrapolation. Same data, comparable budget, one structural difference.
    """
    rule("TASK E -- extrapolation against a comparable free-form learner")
    eq = EquantOracle(eccentricity=0.12)

    M_train = np.linspace(0, TWO_PI, 400, endpoint=False)
    M_test = np.linspace(6 * TWO_PI, 9 * TWO_PI, 600)     # far outside the window
    q_train = eq.equation_of_centre(M_train)
    q_test = eq.equation_of_centre(M_test % TWO_PI)

    seeds = choose_rates(M_train, q_train, n_orbs=4, kmax=12)
    arr = Arrangement([Orb(a=a0, k=k, phi=p0, role="general",
                           free=(True, True, False))
                       for (k, a0, p0) in seeds])
    train(arr, M_train, tgt_lon=q_train, steps=4000, lr=0.4)
    tr_arr = np.sqrt(np.mean((arr.forward(M_train)[0] - q_train) ** 2))
    te_arr = np.sqrt(np.mean((arr.forward(M_test)[0] - q_test) ** 2))

    mlp = TinyMLP(hidden=24, seed=7)
    mlp.fit(M_train, q_train, steps=8000, lr=8e-3)
    tr_mlp = np.sqrt(np.mean((mlp.forward(M_train)[0] - q_train) ** 2))
    te_mlp = np.sqrt(np.mean((mlp.forward(M_test)[0] - q_test) ** 2))

    n_arr = arr.n_params()
    n_mlp = sum(getattr(mlp, k).size for k in mlp._params())

    print(f"   training window : M in [0, 2pi),  400 points")
    print(f"   test window     : M in [6*2pi, 9*2pi],  600 points "
          f"(six revolutions later)")
    print(f"\n   {'model':<28s} {'params':>7} {'train RMS':>12} {'test RMS':>12} "
          f"{'ratio':>9}")
    print("   " + "-" * 72)
    print(f"   {'Constituting Arrangement':<28s} {n_arr:>7} {tr_arr:>12.4e} "
          f"{te_arr:>12.4e} {te_arr / max(tr_arr, 1e-15):>9.2f}")
    print(f"   {'free-form MLP (tanh)':<28s} {n_mlp:>7} {tr_mlp:>12.4e} "
          f"{te_mlp:>12.4e} {te_mlp / max(tr_mlp, 1e-15):>9.2f}")
    print("\n   The MLP is not badly trained. It is trained on a problem it has")
    print("   no way to know is periodic. Integer rates are not a regulariser")
    print("   bolted on afterwards -- they are what 'this arrangement closes'")
    print("   means, and closure is the whole reason the sky is predictable.")
    return (tr_arr, te_arr, tr_mlp, te_mlp)


def task_F_recension():
    """
    Tahrir. Revise without deleting; replay a correction Tusi actually made.

    Between the Hall-i mushkilat-i Mu'iniyya (Tun, 643/1245) and the Tadhkira
    (659/1261) he changed the lunar inclined orb from 13;11 deg/day to
    13;14 deg/day, having realised the earlier figure double-counted the
    parecliptic motion. The earlier version was not withdrawn; it went on
    circulating, and it is the one that reached Byzantium.
    """
    rule("TASK F -- tahrir: revision that does not destroy its ancestor")

    # 13;11 and 13;14 in sexagesimal -> decimal degrees per day
    hall_rate = 13 + 11 / 60
    tadhkira_rate = 13 + 14 / 60
    print(f"   Hall (1245)      inclined orb: 13;11 deg/day = {hall_rate:.6f}")
    print(f"   Tadhkira (1261)  inclined orb: 13;14 deg/day = {tadhkira_rate:.6f}")
    print(f"   correction: {tadhkira_rate - hall_rate:.6f} deg/day "
          f"-- the parecliptic motion had been counted twice")
    drift_per_year = (tadhkira_rate - hall_rate) * 365.25
    print(f"   uncorrected, that is {drift_per_year:.2f} deg of accumulated "
          f"error per year")

    M = np.linspace(0, TWO_PI, 501, endpoint=False)
    tgt = 0.05 * np.sin(M + 0.2) + 0.008 * np.sin(4 * M - 1.0)

    s1 = choose_rates(M, tgt, n_orbs=1, kmax=8)
    arr = Arrangement([Orb(a=a0, k=k, phi=p0, role="general",
                           free=(True, True, False)) for (k, a0, p0) in s1])
    train(arr, M, tgt_lon=tgt, steps=3000, lr=0.4)
    rms_v1 = float(np.sqrt(np.mean((arr.forward(M)[0] - tgt) ** 2)))

    arr.recense("v1 -> v2: added the fourth-harmonic orb; v1 stays runnable")
    resid = tgt - arr.forward(M)[0]
    s2 = choose_rates(M, resid, n_orbs=1, kmax=8)
    k2, a2, p2 = s2[0]
    arr.add(Orb(a=a2, k=k2, phi=p2, role="general", free=(True, True, False)))
    train(arr, M, tgt_lon=tgt, steps=4000, lr=0.4)
    rms_v2 = float(np.sqrt(np.mean((arr.forward(M)[0] - tgt) ** 2)))

    anc_rms = float(np.sqrt(np.mean((arr.ancestor.forward(M)[0] - tgt) ** 2)))

    print(f"\n   version 1 RMS                    {rms_v1:.4e}")
    print(f"   version 2 RMS                    {rms_v2:.4e}")
    print(f"   version 1 RE-RUN after revision   {anc_rms:.4e}  "
          f"(identical: {abs(anc_rms - rms_v1) < 1e-12})")
    print(f"   distance moved from ancestor      "
          f"{arr.fidelity_to_ancestor(M):.4e}")
    print(f"   provenance: {arr.provenance}")
    print(f"   the added orb: k = {k2}, amplitude {a2:.6f} -- the residual named it")
    print("\n   The superseded version still runs and still gives its old,")
    print("   wrong answer on demand. That is what makes the correction")
    print("   checkable by someone who was not present when it was made.")


# ==============================================================================
# SECTION 7 -- SELF TESTS
# ==============================================================================

def self_tests():
    rule("SELF TESTS")
    failures = []

    def check(name, cond, detail=""):
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}" +
              (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # 1 -- exact rectilinearity
    t = np.linspace(0, TWO_PI, 4001)
    c = Couple(0.8, 3, 1.234)
    perp = np.abs(np.imag(c.locus(t) * np.conj(c.axis()))).max()
    check("couple locus is rectilinear to machine precision",
          perp < 1e-14, f"max perpendicular {perp:.3e}")

    # 2 -- amplitude identity, evaluated AT the extremum rather than near it
    # cos(k t - alpha/2) = 1 exactly at t = alpha / (2k); sampling a grid and
    # taking a max only ever gets close, which is a fact about the grid and
    # not about the couple.
    t_ext = np.array([c.alpha / (2.0 * c.rate)])
    amp = float(np.abs(np.real(c.locus(t_ext) * np.conj(c.axis()))[0]))
    check("couple half-length equals 2r at the exact extremum",
          abs(amp - 2 * c.half_radius) < 1e-12,
          f"{amp:.12f} vs {2 * c.half_radius:.12f}")

    # 3 -- couple == its scalar oscillation form
    d = np.abs(np.real(c.locus(t) * np.conj(c.axis())) - c.as_oscillation(t)).max()
    check("complex locus and scalar oscillation agree", d < 1e-12,
          f"max diff {d:.3e}")

    # 4 -- GRADIENT CHECK (mandatory)
    M = np.linspace(0, TWO_PI, 233, endpoint=False)
    tgt_lon = 0.04 * np.sin(M + 0.3) + 0.01 * np.sin(3 * M)
    tgt_lat = 0.006 * np.sin(2 * M - 0.5)
    arr = Arrangement([
        Orb(a=0.031, k=1, phi=0.21, iota=0.05, role="general"),
        Orb(a=0.009, k=3, phi=-0.44, iota=0.31, role="latitude"),
        Orb(a=0.004, k=2, phi=1.10, iota=-0.12, role="general"),
    ])
    rel, table = gradient_check(arr, M, tgt_lon, tgt_lat, w_lat=3.0)
    check("analytic gradient matches central differences",
          rel < 1e-6, f"max relative error {rel:.3e} over {len(table)} params")

    # 5 -- gradient check with frozen (enclosing) orbs present
    arr.install_enclosing_spheres()
    rel2, table2 = gradient_check(arr, M, tgt_lon, tgt_lat, w_lat=3.0)
    check("gradient still correct with frozen enclosing orbs",
          rel2 < 1e-6, f"max relative error {rel2:.3e} over {len(table2)} params")

    # 6 -- frozen orbs receive no gradient slots
    check("enclosing orbs are excluded from the parameter vector",
          arr.n_params() == 3 * 3,
          f"{arr.n_params()} params for {len(arr.orbs)} orbs")

    # 7 -- equation of centre is odd (so integer harmonics span it)
    eq = EquantOracle(0.09)
    Ms = np.linspace(0.01, 3.0, 97)
    odd = np.abs(eq.equation_of_centre(Ms) + eq.equation_of_centre(-Ms)).max()
    check("equation of centre is odd in M", odd < 1e-14, f"max {odd:.3e}")

    # 8 -- admissibility audit catches a non-integer rate
    bad = Arrangement([Orb(a=0.01, k=1, role="general")])
    bad.orbs[0].k = 1.5                      # smuggle in an inadmissible rate
    ok, faults = bad.audit()
    check("audit rejects a non-integer rate", (not ok) and len(faults) == 1,
          faults[0] if faults else "")

    # 9 -- Unconstituted refuses to be a number
    u = Unconstituted("longitude", "test", 0.5)
    try:
        float(u); raised = False
    except TypeError:
        raised = True
    check("UNCONSTITUTED cannot be silently coerced to a float", raised)

    # 10 -- declaration fires on an incommensurable residual and not otherwise
    #       (evaluated over 8 revolutions -- one is not enough, by construction)
    Md = np.linspace(0, 8 * TWO_PI, 3200, endpoint=False)
    good = 0.03 * np.sin(2 * Md + 0.1)
    ag = Arrangement([Orb(a=0.03, k=2, phi=0.1, role="general")])
    vg = ag.declare(Md, target_lon=good)
    bad_t = good + 0.02 * np.sin(np.pi * 5 * Md)
    vb = ag.declare(Md, target_lon=bad_t)
    check("declaration stays silent on a reachable target",
          isinstance(vg["longitude"], float), f"{vg['longitude']:.3e}")
    check("declaration fires on an unreachable target",
          isinstance(vb["longitude"], Unconstituted), repr(vb["longitude"]))

    # 11 -- training reduces loss
    arr2 = Arrangement([Orb(a=0.001, k=1, role="general"),
                        Orb(a=0.001, k=3, role="general")])
    l0, _ = loss_and_grad(arr2, M, tgt_lon)
    l1 = train(arr2, M, tgt_lon, steps=1500, lr=0.4)
    check("training reduces the loss", l1 < l0 * 1e-3, f"{l0:.3e} -> {l1:.3e}")

    # 12 -- recension preserves the ancestor exactly
    arr3 = Arrangement([Orb(a=0.02, k=1, phi=0.3, role="general")])
    before = arr3.forward(M)[0].copy()
    arr3.recense("test")
    arr3.orbs[0].a = 0.9
    same = np.abs(arr3.ancestor.forward(M)[0] - before).max()
    check("recension leaves the ancestor bit-identical", same == 0.0,
          f"max drift {same:.3e}")

    print(f"\n   {len(table) + len(table2)} gradient components verified.")
    if failures:
        print(f"\n   FAILURES: {failures}")
    else:
        print("\n   all self-tests pass.")
    return failures


# ==============================================================================
# SECTION 8 -- MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print(" THE CONSTITUTING ARRANGEMENT")
    print(" a cognitive architecture after Nasir al-Din al-Tusi (1201-1274)")
    print(" one admissible primitive; no disembodied parameter;")
    print(" every mechanism owns all of its consequences")
    print("=" * 78)

    failures = self_tests()

    worst = task_A_exactness()
    hist, arr_b, _ = task_B_constitute_equant()
    arr_c, _ = task_C_side_effects()
    task_D_mercury()
    gen = task_E_generalisation()
    task_F_recension()

    rule("SUMMARY")
    print(f"   self-tests failed              : {len(failures)}")
    print(f"   worst rectilinear deviation    : {worst:.3e} rad")
    print(f"   equant constituted to          : "
          f"{np.degrees(hist[-1][1]) * 3600:.4f} arcsec RMS "
          f"with {hist[-1][0]} orbs "
          f"({len(arr_b.bill_of_materials())} turning bodies)")
    print(f"   published residual (Task C)    : {arr_c.ledger.total():.4e} rad")
    print(f"   extrapolation, arrangement     : {gen[1]:.4e} RMS "
          f"({gen[1] / gen[0]:.2f}x its training error)")
    print(f"   extrapolation, free-form MLP   : {gen[3]:.4e} RMS "
          f"({gen[3] / gen[2]:.2f}x its training error)")
    print()
    print("   Nothing above was fitted with a quantity that does not name a")
    print("   body. Where the target went beyond what bodies can do, the model")
    print("   said so rather than reporting a number it had not earned.")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

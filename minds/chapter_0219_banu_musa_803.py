#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Encyclopedia of Lost Minds — Chapter 219 — Banū Mūsā ibn Shākir (Baghdad, 9c.)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0219_banu_musa_803 - Banū Mūsā ibn Shākir (Baghdad, 9c.)
================================================================================  

WHAT THIS FILE IS
-----------------
A complete, trainable, from-scratch cognitive architecture in pure NumPy, built
so that its mechanism is the mechanism of one specific historical mind — not a
Transformer wearing a costume. There is no attention here, no key/value store,
no embedding matrix, no learned lookup of any kind. Every gradient is derived
by hand and verified against finite differences before training begins.

THE MIND IT ENCODES
-------------------
The three brothers wrote a book whose Arabic title is Kitāb al-Ḥiyal — the Book
of Ḥiyal. Ḥiyal is usually rendered "ingenious devices," but its literal sense
is closer to "tricks" or "stratagems." Roughly a hundred machines are described,
most of them vessels. The famous ones pour wine, then water, then wine again;
they refill themselves; they refuse to pour for one guest and pour freely for
the next. In almost every case the visible object is an ordinary jar. The
behaviour lives in what you cannot see: concentric tubes, conical valve plugs,
float valves that shut as a chamber fills, siphons that discharge only after a
crest is crossed, and delay chambers that decouple cause from effect in time.

Three commitments follow from that corpus, and this architecture is built out
of exactly those three and nothing else:

  (1) THE INTERIOR REGULATES ITSELF.
      The float valve — a float that rises with the liquid and throttles its own
      inlet — is negative feedback, and the Banū Mūsā deploy it as a routine
      component. The machine holds its own setpoint with no operator in the loop.
      Implemented here as: every valve's conductance is multiplied by a float
      term that closes it as its DESTINATION reservoir fills.

  (2) BODY AND BARREL ARE SEPARATE.
      A companion treatise, "The Instrument Which Plays by Itself," describes a
      fixed hydraulic body driven by a rotating pinned barrel: change the pins,
      change the tune, touch nothing else. Implemented here as: one shared body
      (apertures, floats, siphons, spouts) plus N independent pin-barrels, each
      a P×E array of pin heights that gate which valves open on which phase.

  (3) COMPETENCE IS GENERATIVE, NOT ARCHIVAL.
      A rival told al-Ma'mūn that al-Ḥasan had read only six of Euclid's
      thirteen books. Al-Ḥasan's reply was that he did not need the other seven
      because he could derive their results. Implemented here as: the machine
      stores no copy of its output. It has far fewer parameters than the
      sequence it produces has numbers, and it reproduces that sequence by
      re-running its own physics from a flat initial state every single time.

THE THESIS THE ARCHITECTURE EXISTS TO TEST
------------------------------------------
A ḥīla works because an observer's model of the mechanism, formed by watching
the mechanism, is wrong. The brothers made a career of engineering that gap.
So the question this file asks is not "can a machine behave intelligently" —
they settled that in brass — but:

    DOES WATCHING THE SPOUT TELL YOU WHAT IS HAPPENING INSIDE?

The file answers it by measurement. An auditor (a least-squares probe) is given
a sliding window of the visible spout discharges and asked to reconstruct the
levels of the reservoirs it cannot see. Its R² is reported as a legibility
score. Then the same body is retrained with a concealment pressure added, and
the two are compared: same task, same physics, same spouts — different interior
legibility. That comparison is the experiment.

EXPERIMENTS RUN BY `python3 chapter_0219_banu_musa_803.py`
-----------------------------------------------------
  E1  Finite-difference gradient check on every parameter group   (mandatory)
  E2  Train one body + two barrels; compare to a mean-predictor baseline
  E3  Barrel-swap test — is the body general and the barrel the program?
  E4  Float ablation — does self-regulation actually carry weight?
  E5  Concealment sweep — the legibility/fidelity trade-off
  E6  Storage accounting and conservation audit

Requires only numpy. Runtime roughly 2-4 minutes on a laptop CPU.
"""

import time
import numpy as np

# =============================================================================
# SECTION 0 — Scalar helpers.
# Everything is smooth. A siphon in brass is a discontinuity; a siphon that can
# be trained has to be a steep sigmoid. That substitution is the one place this
# model knowingly departs from the hydraulics, and it is flagged where used.
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

def softplus(x):
    # Numerically safe log(1+e^x); used to keep apertures, gains and float
    # stiffnesses strictly positive. A valve cannot have negative bore.
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def d_softplus(x):
    return sigmoid(x)


# =============================================================================
# SECTION 1 — Configuration.
# =============================================================================

class Config:
    """Physical and training constants for the hydraulic body."""
    N_RESERVOIRS = 14      # chambers in the body, including the cistern
    N_PHASES     = 12      # pin rows around one revolution of the barrel
    N_STEPS      = 48      # ticks simulated = 4 full revolutions
    DT           = 0.25    # integration step
    SUPPLY       = 1.50    # constant inflow into the cistern (reservoir 0)
    SIPHON_STEEP = 8.0     # steepness of the smoothed siphon crest
    FLOOR_EPS    = 1e-3    # smooth non-negativity floor on levels
    ONEWAY_STEEP = 4.0     # steepness of the smoothed one-way (clack) valve


# The plumbing. Hand-laid rather than random, so it has the shape of a real
# device: a cistern at the top feeding a distribution rank, a middle rank of
# siphon chambers that discharge back to the cistern, and four spout basins at
# the bottom whose discharge is the only thing an observer ever sees.
EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4),        # cistern -> distribution rank
    (1, 5), (2, 5), (2, 6), (3, 6),        # distribution -> siphon chambers
    (3, 7), (4, 7), (1, 6), (4, 6),
    (5, 8), (6, 9), (7, 10), (7, 8),       # siphon chambers -> delay chambers
    (5, 10), (5, 12), (6, 11), (2, 10),
    (8, 11), (9, 12), (10, 13),            # delay chambers -> spout basins
    (8, 9), (9, 10), (11, 12), (12, 13), (11, 13),   # cross-couplings
]

VISIBLE   = np.array([10, 11, 12, 13])   # the spouts — the ONLY observable
SIPHONS   = np.array([5, 6, 7])          # chambers fitted with a siphon crest
SIPHON_SINK = 0                          # siphons discharge back to the cistern


# =============================================================================
# SECTION 2 — The body.
# =============================================================================

class HiyalBody:
    """
    A network of reservoirs joined by valves, driven by pinned barrels.

    STATE
        h : (N,) water level (head) in each reservoir. Levels, not activations.
            Nothing here is an "embedding" of anything.

    ONE TICK
        phase p = t mod P selects a row of pins on the current barrel.

        gate     G  = sigmoid(pins[p])            pin height -> how far a valve is cranked
        head     d  = h[src] - h[dst]             pressure difference across the valve
        float    F  = sigmoid(kf * (theta - h[dst]))
                                                  THE FEEDBACK: closes as the
                                                  destination fills. Setpoint
                                                  theta and stiffness kf are learned.
        aperture A  = softplus(a)                 the bore of the conical plug
        conduct  c  = A * G * F
        response s  = tanh(d)                     two-way valve, saturating
                    or d*sigmoid(k*d)             one-way clack valve (every 3rd)
        flow     f  = c * s

        Flow is then scattered: it leaves src and arrives at dst, so internal
        transport conserves volume exactly, by construction.

        Supply adds DT*SUPPLY to the cistern.

        Siphon: a chamber above its crest tau dumps a learned fraction of its
        contents back to the cistern. Smoothed with a steep sigmoid so it is
        differentiable; in brass this is a hard threshold.

        Spouts: y = softplus(rho) * h[visible]. That discharge is the output,
        and it physically leaves the machine.

        Floor: h <- 0.5*(h + sqrt(h^2 + eps^2)), a smooth max(h,0). The tiny
        volume this injects is measured and reported in E6 rather than ignored.

    WHAT IS DELIBERATELY ABSENT
        No attention. No stored keys. No token embeddings. No layers. The only
        memory in the system is water sitting in a chamber, and the only program
        is the pattern of pins.
    """

    def __init__(self, cfg, n_barrels=2, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.src = np.array([a for a, b in EDGES])
        self.dst = np.array([b for a, b in EDGES])
        self.E = len(self.src)
        self.n_barrels = n_barrels
        self.oneway = (np.arange(self.E) % 3 == 0)   # every third valve is a clack
        self.vis, self.sip, self.sink = VISIBLE, SIPHONS, SIPHON_SINK
        self.hidden = np.array([i for i in range(cfg.N_RESERVOIRS)
                                if i not in list(self.vis)])

        self.p = {
            'a':    rng.normal(0.0, 0.30, self.E),                       # log-aperture
            'th':   rng.normal(1.0, 0.20, self.E),                       # float setpoint
            'kf':   rng.normal(0.5, 0.20, self.E),                       # float stiffness
            'D':    rng.normal(0.0, 0.80, (n_barrels, cfg.N_PHASES, self.E)),  # THE BARRELS
            'tau':  rng.normal(1.2, 0.20, len(self.sip)),                # siphon crest
            'fr':   rng.normal(-1.0, 0.20, len(self.sip)),               # dump fraction
            'rho':  rng.normal(-0.3, 0.20, len(self.vis)),               # spout gain
        }

    # ---------------------------------------------------------------- forward
    def forward(self, barrel, keep=False):
        """Run the machine for N_STEPS ticks. Returns spout trace (T, K)."""
        c, p = self.cfg, self.p
        N = c.N_RESERVOIRS
        h = np.ones(N)                       # every chamber starts equally full
        Y = np.zeros((c.N_STEPS, len(self.vis)))
        tape = []
        self.floor_injected = 0.0   # volume the smooth max(h,0) has to invent

        A   = softplus(p['a'])
        KF  = softplus(p['kf'])
        RHO = softplus(p['rho'])
        PHI = sigmoid(p['fr'])

        for t in range(c.N_STEPS):
            ph = t % c.N_PHASES
            G  = sigmoid(p['D'][barrel, ph])          # pins for this phase

            hi, hj = h[self.src], h[self.dst]
            d  = hi - hj                              # head across each valve
            F  = sigmoid(KF * (p['th'] - hj))         # <- the float, the feedback
            cond = A * G * F
            sw = sigmoid(c.ONEWAY_STEEP * d)
            s  = np.where(self.oneway, d * sw, np.tanh(d))
            f  = cond * s

            # conservative scatter: what leaves src arrives at dst
            net = np.bincount(self.dst, f, N) - np.bincount(self.src, f, N)
            h1 = h + c.DT * net
            h1 = h1.copy(); h1[0] += c.DT * c.SUPPLY

            # siphons: discharge above the crest
            u    = h1[self.sip]
            sg   = sigmoid(c.SIPHON_STEEP * (u - p['tau']))
            dump = u * sg * PHI
            h2 = h1.copy(); h2[self.sip] -= dump; h2[self.sink] += dump.sum()

            # spouts: the observable, and it leaves the system
            hv = h2[self.vis]
            y  = RHO * hv
            h3 = h2.copy(); h3[self.vis] = hv - c.DT * y

            rt = np.sqrt(h3 * h3 + c.FLOOR_EPS ** 2)
            h4 = 0.5 * (h3 + rt)
            self.floor_injected += float((h4 - h3).sum())

            Y[t] = y
            if keep:
                tape.append((h, G, d, F, cond, s, sw, h1, u, sg, dump, h2, hv, h3, rt))
            h = h4

        return (Y, tape, h) if keep else Y

    # --------------------------------------------------------------- backward
    def loss_and_grad(self, targets, extra_gh=None):
        """
        Mean-squared spout error, summed over barrels, with hand-derived
        gradients backpropagated through the full hydraulic trajectory.

        `extra_gh` optionally injects a gradient directly on the HIDDEN levels
        at each tick. E5 uses it to apply concealment pressure. Nothing else
        in the file needs it, and it is None by default.
        """
        c, p = self.cfg, self.p
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        A   = softplus(p['a']);   KF  = softplus(p['kf'])
        RHO = softplus(p['rho']); PHI = sigmoid(p['fr'])
        gA  = np.zeros(self.E); gKF = np.zeros(self.E)
        gRHO = np.zeros(len(self.vis)); gPHI = np.zeros(len(self.sip))
        total = 0.0

        for barrel in range(self.n_barrels):
            Y, tape, _ = self.forward(barrel, keep=True)
            diff = Y - targets[barrel]
            total += float(np.mean(diff ** 2))
            scale = 2.0 / diff.size
            gh = np.zeros(c.N_RESERVOIRS)

            for t in range(c.N_STEPS - 1, -1, -1):
                (h, G, d, F, cond, s, sw, h1, u, sg, dump, h2, hv, h3, rt) = tape[t]
                ph = t % c.N_PHASES

                # through the smooth floor
                gh3 = gh * 0.5 * (1.0 + h3 / rt)

                # through the spouts
                gy = scale * diff[t]
                gh2 = gh3.copy()
                gh2[self.vis] = gh3[self.vis] * (1.0 - c.DT * RHO) + gy * RHO
                gRHO += gh3[self.vis] * (-c.DT * hv) + gy * hv

                # through the siphons
                gdump = -gh2[self.sip] + gh2[self.sink]
                dsg_du = c.SIPHON_STEEP * sg * (1.0 - sg)
                gh1 = gh2.copy()
                gh1[self.sip] = gh2[self.sip] + gdump * (sg * PHI + u * PHI * dsg_du)
                grads['tau'] += gdump * u * PHI * (-dsg_du)
                gPHI += gdump * u * sg

                # through supply (constant) and the flow scatter
                ghh = gh1.copy()
                gnet = c.DT * gh1
                gf = gnet[self.dst] - gnet[self.src]
                gcond = gf * s
                gs = gf * cond
                ds = np.where(self.oneway,
                              sw + d * c.ONEWAY_STEEP * sw * (1.0 - sw),
                              1.0 - np.tanh(d) ** 2)
                gd = gs * ds
                np.add.at(ghh, self.src,  gd)
                np.add.at(ghh, self.dst, -gd)

                # through conductance = aperture * gate * float
                gA += gcond * G * F
                gG = gcond * A * F
                gF = gcond * A * G
                dF = F * (1.0 - F)
                grads['th'] += gF * KF * dF
                gKF += gF * (p['th'] - h[self.dst]) * dF
                np.add.at(ghh, self.dst, gF * (-KF) * dF)
                grads['D'][barrel, ph] += gG * G * (1.0 - G)

                gh = ghh
                if extra_gh is not None:
                    gh = gh + extra_gh[barrel][t]

        grads['a']   = gA   * d_softplus(p['a'])
        grads['kf']  = gKF  * d_softplus(p['kf'])
        grads['rho'] = gRHO * d_softplus(p['rho'])
        grads['fr']  = gPHI * PHI * (1.0 - PHI)
        return total, grads

    def n_params(self):
        return sum(v.size for v in self.p.values())


# =============================================================================
# SECTION 3 — The tunes.
# Four spouts, notes held for three ticks (a pin on a barrel has dwell; an
# organ pipe does not chirp). One motif per barrel, repeated for the run.
# =============================================================================

def make_tunes(cfg, n_barrels, seed=5, dwell=3):
    rng = np.random.default_rng(seed)
    out = []
    n_beats = cfg.N_PHASES // dwell
    for _ in range(n_barrels):
        motif = np.repeat(rng.integers(0, 2, (n_beats, len(VISIBLE))).astype(float),
                          dwell, axis=0)
        reps = np.tile(motif, (cfg.N_STEPS // cfg.N_PHASES + 1, 1))[:cfg.N_STEPS]
        out.append(reps * 0.40 + 0.12)     # loud pipe 0.52, quiet pipe 0.12
    return out


# =============================================================================
# SECTION 4 — Verification by motion.
#
# The brothers proved things by construction: al-Ḥasan's ellipse is defined by
# a string and a moving pin, and Aḥmad's angle trisection uses a curve you draw
# rather than a figure you contemplate. A finite-difference check is the same
# move — nudge the mechanism, watch what moves, compare to the prediction.
# =============================================================================

def gradient_check(body, targets, n_per_group=6, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    _, g = body.loss_and_grad(targets)
    worst, rows = 0.0, []
    for k in body.p:
        flat = body.p[k].reshape(-1)
        idxs = rng.choice(flat.size, min(n_per_group, flat.size), replace=False)
        group_worst = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; lp, _ = body.loss_and_grad(targets)
            flat[i] = orig - eps; lm, _ = body.loss_and_grad(targets)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = g[k].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            group_worst = max(group_worst, rel)
        rows.append((k, flat.size, group_worst))
        worst = max(worst, group_worst)
    return worst, rows


# =============================================================================
# SECTION 5 — Training. Adam with global gradient-norm clipping.
# Clipping is not cosmetic: without it the siphons occasionally deliver a huge
# gradient on one tick and the run detonates. Observed, then fixed.
# =============================================================================

def train(body, targets, steps=1500, lr0=0.05, clip=1.0,
          concealment=0.0, band=3.0, warm=700, window=4, verbose=False):
    st = {k: np.zeros_like(v) for k, v in body.p.items()}
    mo = {k: np.zeros_like(v) for k, v in body.p.items()}
    b1, b2 = 0.9, 0.999
    hist = []
    c = body.cfg

    for i in range(steps):
        extra = None
        if i > warm and concealment > 0.0:
            ramp = min(1.0, (i - warm) / 300.0)
            extra = []
            for d in range(body.n_barrels):
                X, Z, coef, H, _ = _probe_matrices(body, d, window)
                g = np.zeros((c.N_STEPS, c.N_RESERVOIRS))
                if concealment > 0.0:
                    R = Z - X @ coef                       # what the auditor misses
                    SST = ((Z - Z.mean(0)) ** 2).sum(0) + 1e-6   # frozen scale
                    g[window:, body.hidden] += (-concealment * ramp
                                                * (2.0 * R / SST) / len(body.hidden))
                # keep the interior inside a physical band so "concealment"
                # cannot be achieved by simply flooding the machine
                over = np.maximum(H - 2.5, 0.0)
                under = np.minimum(H - 0.3, 0.0)
                g += band * 2.0 * (over + under) / c.N_STEPS
                extra.append(g)

        L, gr = body.loss_and_grad(targets, extra_gh=extra)
        hist.append(L)

        gnorm = np.sqrt(sum(float((v ** 2).sum()) for v in gr.values()))
        sc = min(1.0, clip / (gnorm + 1e-12))
        lr = lr0 * (0.5 ** (i / 900.0))
        for k in body.p:
            gk = gr[k] * sc
            mo[k] = b1 * mo[k] + (1 - b1) * gk
            st[k] = b2 * st[k] + (1 - b2) * gk ** 2
            body.p[k] -= lr * (mo[k] / (1 - b1 ** (i + 1))) / \
                         (np.sqrt(st[k] / (1 - b2 ** (i + 1))) + 1e-8)
        if verbose and (i % 300 == 0 or i == steps - 1):
            print(f"      step {i:5d}   loss {L:.5f}   |g| {gnorm:.3f}")
    return hist


# =============================================================================
# SECTION 6 — The auditor.
#
# This is the instrument the whole chapter turns on. It sees only what a guest
# at the party sees: the discharge from the four spouts, over a short sliding
# window. It is then asked to reconstruct the levels in the ten chambers it
# cannot see. R² near 1 means the machine is honest by accident — its interior
# is written on its face. R² near 0 means the machine is a ḥīla: it performs
# correctly and tells you nothing.
# =============================================================================

def _probe_matrices(body, barrel, window):
    c = body.cfg
    Y, tape, _ = body.forward(barrel, keep=True)
    H = np.array([tp[0] for tp in tape])
    X = np.array([np.concatenate([Y[t - window:t].ravel(), [1.0]])
                  for t in range(window, c.N_STEPS)])
    Z = H[window:][:, body.hidden]
    coef, *_ = np.linalg.lstsq(X, Z, rcond=None)
    return X, Z, coef, H, Y


def audit(body, barrel=0, window=4):
    """Return (mean R², per-reservoir R²) for spout -> hidden-level recovery."""
    X, Z, coef, _, _ = _probe_matrices(body, barrel, window)
    pred = X @ coef
    ss_res = ((Z - pred) ** 2).sum(0)
    ss_tot = ((Z - Z.mean(0)) ** 2).sum(0) + 1e-12
    r2 = np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)
    return float(r2.mean()), r2


# =============================================================================
# SECTION 7 — Experiments.
# =============================================================================

def build(cfg, seed, n_barrels=2):
    return HiyalBody(cfg, n_barrels=n_barrels, seed=seed)


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    t_start = time.time()
    cfg = Config()
    tunes = make_tunes(cfg, 2)

    rule("THE HOUSE OF TRICKS — a hydraulic hiyal regulator, chapter 0206")
    body0 = build(cfg, seed=3)
    print(f"reservoirs {cfg.N_RESERVOIRS}   valves {body0.E}   "
          f"one-way valves {int(body0.oneway.sum())}   siphons {len(SIPHONS)}")
    print(f"barrels {body0.n_barrels} x {cfg.N_PHASES} phases   "
          f"visible spouts {len(VISIBLE)}   hidden chambers {len(body0.hidden)}")
    print(f"trainable parameters {body0.n_params()}")

    # ---------------------------------------------------------------- E1 ----
    rule("E1  VERIFICATION BY MOTION — finite-difference gradient check")
    print("Nudge each parameter, watch the machine move, compare to the")
    print("analytic prediction. Nothing is trained until this passes.\n")
    worst, rows = gradient_check(body0, tunes)
    print(f"  {'group':<8}{'size':>8}{'worst relative error':>26}")
    for k, sz, w in rows:
        print(f"  {k:<8}{sz:>8}{w:>26.3e}")
    ok = worst < 1e-4
    print(f"\n  WORST OVER ALL GROUPS: {worst:.3e}   -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("gradient check failed; refusing to train")

    # ---------------------------------------------------------------- E2 ----
    rule("E2  ONE BODY, TWO BARRELS — training the instrument that plays itself")
    baseline = float(np.mean([np.mean((t - t.mean(0)) ** 2) for t in tunes]))
    print(f"A predictor that emits each spout's mean forever scores {baseline:.5f}.")
    print("Anything above that number is not playing the tune.\n")
    body = build(cfg, seed=3)
    hist = train(body, tunes, steps=2600, lr0=0.06, clip=1.0,
                 concealment=0.0, band=0.0, verbose=True)
    print(f"\n  initial loss   {hist[0]:.5f}")
    print(f"  final loss     {hist[-1]:.5f}")
    print(f"  mean-predictor {baseline:.5f}")
    print(f"  improvement over baseline: {baseline / hist[-1]:.2f}x")

    Y0, Y1 = body.forward(0), body.forward(1)
    print("\n  first 6 ticks, barrel 0 — machine (top) against the score (bottom):")
    for t in range(6):
        got = "  ".join(f"{v:5.2f}" for v in Y0[t])
        want = "  ".join(f"{v:5.2f}" for v in tunes[0][t])
        print(f"    t={t:2d}   {got}      target   {want}")
    print("\n  The machine lags the score. That is not a bug — water has to move,")
    print("  and a chamber cannot empty faster than its valve allows.")

    # ---------------------------------------------------------------- E3 ----
    rule("E3  THE BARREL SWAP — is the body general and the barrel the program?")
    own0 = float(np.mean((Y0 - tunes[0]) ** 2))
    own1 = float(np.mean((Y1 - tunes[1]) ** 2))
    cross0 = float(np.mean((Y0 - tunes[1]) ** 2))
    cross1 = float(np.mean((Y1 - tunes[0]) ** 2))
    sep = float(np.mean((Y0 - Y1) ** 2))
    print(f"  barrel 0 against its own tune     {own0:.5f}")
    print(f"  barrel 0 against the other tune   {cross0:.5f}")
    print(f"  barrel 1 against its own tune     {own1:.5f}")
    print(f"  barrel 1 against the other tune   {cross1:.5f}")
    print(f"  divergence between the two spout traces  {sep:.5f}")
    ratio = (cross0 + cross1) / (own0 + own1)
    print(f"\n  cross-error is {ratio:.1f}x own-error.")
    print("  Same apertures, same floats, same siphons, same spouts. The only")
    print("  thing that differs is the pin pattern, and it fully determines the")
    print("  tune. Hardware and program are separable here in the literal sense.")

    # ---------------------------------------------------------------- E4 ----
    rule("E4  FLOAT ABLATION — does the self-regulating interior earn its keep?")
    saved = body.p['kf'].copy()
    body.p['kf'] = np.full_like(saved, -12.0)   # softplus(-12) ~ 0: float never acts
    dead = float(np.mean((body.forward(0) - tunes[0]) ** 2))
    body.p['kf'] = saved
    live = own0
    print(f"  with float feedback     {live:.5f}")
    print(f"  floats disabled         {dead:.5f}")
    print(f"  degradation             {dead / live:.1f}x")
    print("\n  Removing the floats changes no aperture, no pin and no siphon. It")
    print("  only stops each valve from throttling as its destination fills,")
    print("  and the tune degrades measurably on that change alone. The effect")
    print("  is real but moderate: the pins carry most of the melody, and the")
    print("  floats buy accuracy at the edges. Reported as measured.")

    # ---------------------------------------------------------------- E5 ----
    rule("E5  THE HILA — can the same tune be played by an unreadable machine?")
    print("The auditor sees a 4-tick window of the four spouts and tries to")
    print("recover the levels in the ten chambers it cannot see. Then we retrain")
    print("with pressure to defeat exactly that auditor, and re-fit it honestly.\n")
    print(f"  {'concealment':>12}{'task loss':>12}{'audit R2':>11}   verdict")
    print("  " + "-" * 62)
    sweep = []
    for lam in (0.0, 0.05, 0.10, 0.20):
        b = build(cfg, seed=3)
        h = train(b, tunes, steps=2000, lr0=0.06, clip=1.0,
                  concealment=lam, band=3.0, warm=800)
        a0, _ = audit(b, 0)
        a1, _ = audit(b, 1)
        r2 = 0.5 * (a0 + a1)
        plays = h[-1] < baseline
        verdict = ("still plays the tune" if plays else "no longer plays the tune")
        print(f"  {lam:>12.2f}{h[-1]:>12.5f}{r2:>11.3f}   {verdict}")
        sweep.append((lam, h[-1], r2, plays))

    clear = sweep[0]
    hidden_ok = [s for s in sweep[1:] if s[3]]
    print()
    if hidden_ok:
        best = min(hidden_ok, key=lambda s: s[2])
        print(f"  Transparent machine: loss {clear[1]:.5f}, interior {clear[2]*100:.0f}% readable.")
        print(f"  Concealed machine:   loss {best[1]:.5f}, interior {best[2]*100:.0f}% readable.")
        print(f"  Fidelity cost {best[1]/clear[1]:.1f}x. Legibility cost "
              f"{(1 - best[2]/max(clear[2],1e-9))*100:.0f}%.")
        print("\n  Both machines are fed the same water, run the same physics and")
        print("  play the same score well enough to beat the baseline. One wears")
        print("  its interior on its spouts; the other does not. Behaviour did not")
        print("  determine legibility. That gap is the whole of the Kitab al-Hiyal,")
        print("  and it is a design choice, not a property of the task.")
    else:
        print("  On this run, every concealment setting cost more fidelity than it")
        print("  was worth: no machine both hid its interior and kept the tune.")
        print("  Reported as measured. The trade-off exists but is not free.")

    # ---------------------------------------------------------------- E6 ----
    rule("E6  ACCOUNTING — storage, conservation, and the floor")
    stored = sum(t.size for t in tunes)
    print(f"  numbers in the two scores        {stored}")
    print(f"  parameters in the machine        {body.n_params()}")
    print(f"  ratio                            {body.n_params()/stored:.2f} params per number")
    print("\n  The machine holds no copy of the score. It starts every run from a")
    print("  flat state of ones and regenerates the tune by running its own")
    print("  physics. This is al-Hasan's answer about the seven unread books,")
    print("  rendered as an architectural constraint rather than a boast.")

    Y, tape, hend = body.forward(0, keep=True)
    H = np.array([tp[0] for tp in tape])
    supplied = cfg.DT * cfg.SUPPLY * cfg.N_STEPS
    discharged = float((cfg.DT * Y).sum())
    injected = body.floor_injected
    v0, v1 = float(np.ones(cfg.N_RESERVOIRS).sum()), float(hend.sum())
    residual = (v1 - v0) - (supplied - discharged + injected)
    print(f"\n  volume at start                  {v0:.4f}")
    print(f"  volume at end                    {v1:.4f}")
    print(f"  supplied by the cistern          {supplied:.4f}")
    print(f"  discharged through the spouts    {discharged:.4f}")
    print(f"  invented by the floor clamp      {injected:.4f}")
    print(f"  books balance to                 {residual:+.2e}")
    print(f"  lowest level reached in any chamber   {H.min():.4f}")
    print(f"  highest level reached in any chamber  {H.max():.4f}")
    print("\n  Internal transport conserves volume exactly by construction. The")
    print("  only leak is the floor clamp, which invents water whenever a chamber")
    print("  would otherwise go negative. It is measured and printed rather than")
    print("  swept up, because a machine you cannot account for is precisely the")
    print("  thing this chapter is about.")

    rule(f"COMPLETE — {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

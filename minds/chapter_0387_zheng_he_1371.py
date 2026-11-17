#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE MANIFEST ENGINE  --  a cognitive architecture after Zheng He (1371-1433)
================================================================================
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0387 - ZHENG HE (1371-1433)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  


WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
The commanding fact about the Ming treasure fleet is not that it sailed far.
Arab, Tamil, Gujarati and Chinese merchants had sailed those same waters for
centuries.  The commanding fact is a pair of constraints that no other mind in
this corpus operated under simultaneously:

  (1) THE BARRIER.  Everything the fleet would ever have was decided in Nanjing,
      months before the first situation that would require it could be observed.
      Twenty-seven thousand people, six classes of hull, a hundred and eighty
      medical officers, water in ceramic jars.  You cannot re-provision an ocean.
      The trilingual stele that Zheng He set up at Galle in 1409 was CUT IN
      NANJING and dated before departure: the offerings to Buddha, to Vishnu and
      to Allah were itemised in stone before anyone had met the people who would
      read them.  That is amortised inference under commitment, not perception.

  (2) THE UNSPENT ASSET.  In twenty-eight years and seven voyages, a force that
      no polity in the Indian Ocean could resist fought three times, annexed
      nothing, and returned every time.  Its capability was overwhelmingly
      DISPLAYED rather than EXPENDED.  Dreyer's phrase for the fleet on a foreign
      horizon is "a terrifying apparition"; the apparition is the product.

So the architecture is built around a resource vector committed before the
episode, an action space that separates SHOWING from SPENDING, a counterpart
that can only infer capability from what is shown, and a standing that carries
between encounters so that showing is an investment rather than a gesture.

There is no attention over stored keys anywhere in this file.  The recurrence is
a small tanh cell; the interesting structure is in the economy, not the layer.

THE SIX HOLDS (K = 6)
---------------------
  0 WATER   consumed by every leg; cannot be substituted; ships carried it in jars
  1 GRAIN   consumed by every leg; the crew of 27,800 eats whether or not it works
  2 FORCE   the only hold that is worth more shown than spent
  3 GIFT    silk, porcelain, coin: the outbound flow, spent to be given away
  4 TONGUE  interpreters, physicians, protocol officers, astrologers
  5 REPAIR  caulkers, anchor mechanics, sailmakers; the fleet's self-maintenance

THE FIVE MEASURED CLAIMS
------------------------
  C1  A system rewarded for voluntary registration learns to hold FORCE almost
      entirely as display and to end the voyage with reserve it never used.
  C2  Delete the counterpart's memory (anonymise every encounter) and the optimum
      inverts: display becomes worthless and the policy turns coercive.  The
      difference between a treasure fleet and a raiding fleet is not virtue, it
      is whether the other party expects to meet you again.
  C3  Let the manifest see the voyage it is provisioning for and the model gets
      better on the distribution it trained on and fails on the one it did not.
      Commitment buys robustness and is paid for in average-case performance.
  C4  Tie the display to a single dial and credibility collapses, because
      credibility is need-relative: showing soldiers to a port that wants
      physicians is not a smaller display, it is the wrong one.
  C5  Reward only in-voyage compliance and the model builds no durable structure
      at any node; when the fleet withdraws, the network evaporates.  This is the
      Cochin result: the Ming enfeoffed a mountain there and protected the port,
      and when the voyages stopped, Calicut invaded it.

Pure NumPy.  Reverse-mode autodiff written from scratch below.  A finite-
difference gradient check is mandatory and runs first.  Everything printed at the
bottom is produced by this file; nothing is transcribed by hand.
"""

import numpy as np

# ==============================================================================
# PART 1 -- A SMALL REVERSE-MODE AUTODIFF ENGINE
# ==============================================================================
# Written from scratch so that the gradient of the whole voyage -- including the
# resource recurrence, the grievance that force leaves behind, and the terminal
# return-closure hinge -- can be checked against finite differences rather than
# asserted.  A tape of closures, topologically ordered.


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` after NumPy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tn:
    """A tensor node.  `v` is the value, `g` the accumulated gradient."""

    __slots__ = ("v", "g", "_prev", "_bk")

    def __init__(self, v, prev=(), bk=None):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = np.zeros_like(self.v)
        self._prev = prev
        self._bk = bk

    # -- elementary ops -------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, Tn) else Tn(np.asarray(o, dtype=np.float64))
        out = Tn(self.v + o.v, (self, o))

        def bk():
            self.g += _unbroadcast(out.g, self.v.shape)
            o.g += _unbroadcast(out.g, o.v.shape)

        out._bk = bk
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Tn) else Tn(np.asarray(o, dtype=np.float64))
        out = Tn(self.v * o.v, (self, o))

        def bk():
            self.g += _unbroadcast(out.g * o.v, self.v.shape)
            o.g += _unbroadcast(out.g * self.v, o.v.shape)

        out._bk = bk
        return out

    def __sub__(self, o):
        return self + (o * -1.0 if isinstance(o, Tn) else Tn(-np.asarray(o, dtype=np.float64)))

    def __neg__(self):
        return self * -1.0

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __rsub__(self, o):
        return (self * -1.0) + o

    def __truediv__(self, o):
        o = o if isinstance(o, Tn) else Tn(np.asarray(o, dtype=np.float64))
        out = Tn(self.v / o.v, (self, o))

        def bk():
            self.g += _unbroadcast(out.g / o.v, self.v.shape)
            o.g += _unbroadcast(-out.g * self.v / (o.v ** 2), o.v.shape)

        out._bk = bk
        return out

    def matmul(self, o):
        out = Tn(self.v @ o.v, (self, o))

        def bk():
            self.g += out.g @ o.v.T
            o.g += self.v.T @ out.g

        out._bk = bk
        return out

    __matmul__ = matmul

    # -- shape ops ------------------------------------------------------------
    def col(self, i):
        """Column i as an (N,1) node."""
        out = Tn(self.v[:, i:i + 1], (self,))

        def bk():
            self.g[:, i:i + 1] += out.g

        out._bk = bk
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tn(self.v.sum(axis=axis, keepdims=keepdims), (self,))

        def bk():
            g = out.g
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.g += np.broadcast_to(g, self.v.shape).copy()

        out._bk = bk
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- nonlinearities -------------------------------------------------------
    def tanh(self):
        t = np.tanh(self.v)
        out = Tn(t, (self,))

        def bk():
            self.g += out.g * (1.0 - t * t)

        out._bk = bk
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.v, -60, 60)))
        out = Tn(s, (self,))

        def bk():
            self.g += out.g * s * (1.0 - s)

        out._bk = bk
        return out

    def relu(self):
        m = (self.v > 0).astype(np.float64)
        out = Tn(self.v * m, (self,))

        def bk():
            self.g += out.g * m

        out._bk = bk
        return out

    def softplus(self):
        x = np.clip(self.v, -60, 60)
        out = Tn(np.log1p(np.exp(x)), (self,))

        def bk():
            self.g += out.g / (1.0 + np.exp(-x))

        out._bk = bk
        return out

    def sq(self):
        out = Tn(self.v ** 2, (self,))

        def bk():
            self.g += out.g * 2.0 * self.v

        out._bk = bk
        return out

    def softmax(self, axis=1):
        z = self.v - self.v.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tn(p, (self,))

        def bk():
            dot = (out.g * p).sum(axis=axis, keepdims=True)
            self.g += p * (out.g - dot)

        out._bk = bk
        return out

    # -- graph ----------------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n._prev:
                build(p)
            topo.append(n)

        build(self)
        self.g = np.ones_like(self.v)
        for n in reversed(topo):
            if n._bk is not None:
                n._bk()


def cat(nodes, axis=1):
    """Concatenate a list of nodes along `axis`."""
    vals = [n.v for n in nodes]
    out = Tn(np.concatenate(vals, axis=axis), tuple(nodes))
    sizes = [v.shape[axis] for v in vals]

    def bk():
        i = 0
        for n, s in zip(nodes, sizes):
            sl = [slice(None)] * out.v.ndim
            sl[axis] = slice(i, i + s)
            n.g += out.g[tuple(sl)]
            i += s

    out._bk = bk
    return out


# ==============================================================================
# PART 2 -- THE VOYAGE ENVIRONMENT
# ==============================================================================
# A voyage is: a SEASON (what the provisioner can see in Nanjing) and a sequence
# of ENCOUNTERS (what it cannot).  The season is deliberately low-information --
# monsoon phase, the strength of the court's directive, the size of last year's
# register, the year index, and whether the southern route or the Bengal route
# was ordered.  It correlates with the encounters but never determines them.

K = 6                                  # holds
WATER, GRAIN, FORCE, GIFT, TONGUE, REPAIR = range(K)
HOLD_NAMES = ["WATER", "GRAIN", "FORCE", "GIFT", "TONGUE", "REPAIR"]
T_ENC = 8                              # encounters per voyage
S_DIM = 5                              # season features
BUDGET = 10.5                           # total mass the manifest may allocate
RETURN_REQ = 0.55                      # water and grain that must survive to port


def make_voyages(n, rng, shift=False):
    """Sample a batch of voyages.

    `shift=True` produces the distribution the fleet was NOT provisioned for:
    a long dry leg with no resupply port and an unusually hostile landfall.
    This is the test that separates commitment from reaction.
    """
    season = rng.normal(0, 1, size=(n, S_DIM))
    # Needs: each encounter wants a different mixture of the six holds.
    need = rng.dirichlet(np.ones(K) * 0.7, size=(n, T_ENC))
    # Hostility rises with wealth (rich ports have most to lose by submitting)
    wealth = rng.uniform(0.2, 1.0, size=(n, T_ENC, 1))
    temper = rng.uniform(0.0, 0.22, size=(n, 1, 1))       # per-voyage temper
    hostility = 0.32 * wealth + temper + rng.uniform(0.0, 0.22, size=(n, T_ENC, 1))
    # Baseline consumption: every leg burns water and grain, a little repair.
    # The heavy variation is BETWEEN voyages, not between legs -- a route with
    # few watering ports is dear for its whole length.  That is exactly the
    # quantity a provisioner would most like to know and cannot.
    leg = np.zeros((n, T_ENC, K))
    route = 0.20 + 0.28 * rng.uniform(size=(n, 1, 1))
    base = route + 0.05 * rng.uniform(size=(n, T_ENC, 1))
    leg[:, :, WATER] = base[:, :, 0]
    leg[:, :, GRAIN] = base[:, :, 0] * 0.85
    leg[:, :, REPAIR] = 0.05
    # The season carries a weak, honest signal about the mean need and hostility.
    # The season is a weak signal on purpose: a monsoon phase and a court
    # directive tell you very little about which ports will be hostile.
    season[:, 0] = need[:, :, TONGUE].mean(axis=1) * 3.0 + rng.normal(0, 1.3, n)
    season[:, 1] = hostility.mean(axis=1)[:, 0] * 3.0 + rng.normal(0, 1.3, n)
    season[:, 2] = leg[:, :, WATER].mean(axis=1) * 3.0 + rng.normal(0, 1.3, n)
    # The FORECAST is what a provisioner would be told if the barrier were
    # broken: the expected shape of the voyage.  It is computed BEFORE the shift
    # is applied, because that is the honest counterfactual -- you are not given
    # the future, you are given a report, and the report can be wrong.
    forecast = np.concatenate([
        need.mean(axis=1),
        hostility.mean(axis=1),
        leg[:, :, WATER].mean(axis=1, keepdims=True),
    ], axis=1)
    if shift:
        # A leg with no port: water and grain burn at nearly double rate on the
        # middle three encounters, and the last landfall is hostile.
        leg[:, 3:6, WATER] *= 2.45
        leg[:, 3:6, GRAIN] *= 2.45
        hostility[:, -1, 0] = 0.95
    return dict(season=season, need=need, wealth=wealth,
                hostility=hostility, leg=leg, forecast=forecast)


# ==============================================================================
# PART 3 -- PARAMETERS
# ==============================================================================

H = 16                                  # recurrent width
E_DIM = K + 2                           # need vector + hostility + wealth
IN_DIM = H + E_DIM + K + 1 + 1          # h, e, holdings, standing, time fraction
OUT_DIM = K + 3                         # display logits, gift, force, constitute


def init_params(rng):
    def w(a, b, s=None):
        s = s or (1.0 / np.sqrt(a))
        return Tn(rng.normal(0, s, size=(a, b)))

    p = {
        # --- the manifest network: season -> allocation of BUDGET over K holds
        "Wm1": w(S_DIM, 24), "bm1": Tn(np.zeros((1, 24))),
        "Wm2": w(24, K),     "bm2": Tn(np.zeros((1, K))),
        # --- the voyage policy
        "Wp": w(IN_DIM, H), "bp": Tn(np.zeros((1, H))),
        "Wo": w(H, OUT_DIM), "bo": Tn(np.zeros((1, OUT_DIM))),
    }
    # The counterpart is NOT a collaborator and is NOT trained.  A foreign court
    # is part of the world: it has its own reading of the fleet and no interest
    # in making the fleet's loss go down.  These constants are held fixed in
    # every experiment so that all variants face the same Indian Ocean.
    p.update({
        "wc": Tn(np.array([[0.6], [0.6], [2.7], [2.2], [2.0], [1.2]])),
        "alpha": Tn(np.array([[1.0]])),   # weight on matched, credible display
        "wrho": Tn(np.array([[0.35]])),   # awe that precedes the fleet
        "beta": Tn(np.array([[1.4]])),    # weight on gifts actually handed over
        "gamma": Tn(np.array([[1.6]])),   # weight on force actually used
        "delta": Tn(np.array([[1.30]])),   # weight on the port's own hostility
        "b0": Tn(np.array([[-0.55]])),
        # CREDIBILITY: a display costs nothing, so it is only believed if it is
        # backed -- either by a standing the port already holds, or by something
        # the fleet spends in front of it.  This is the whole of claim C2.
        "c_rho": Tn(np.array([[3.0]])),
        "c_spend": Tn(np.array([[3.0]])),
        "c_b": Tn(np.array([[-1.4]])),
        # PROVOCATION: showing is not free.  Parading soldiers at a port with no
        # quarrel manufactures one; showing the water and the grain advertises
        # what is worth taking from a fleet that must keep it to get home.  Silk
        # is nearly free to show and interpreters and shipwrights cost nothing
        # at all, which is why those are the holds a wise fleet leads with.
        "w_prov": Tn(np.array([[0.34], [0.34], [0.00], [0.05], [0.0], [0.0]])),
        # DETERRENCE is the one return that a hold pays WITHOUT BEING SPENT, and
        # it is the whole economic case for a treasure ship.  Its sign flips on
        # the port: shown to a court that means to fight, an unusable force is
        # the argument; shown to a court with no quarrel, the same hulls in the
        # same roadstead manufacture the quarrel they were meant to prevent.
        "deter": Tn(np.array([[1.85]])),
        "prov_f": Tn(np.array([[0.55]])),
    })
    return p


LAMBDA = 0.75     # standing decay between encounters
ETA_D = 0.35      # standing gained per unit of believed display
ETA_F = 0.90      # standing gained per unit of force actually used
ETA_G = 0.40      # standing gained per unit of gift actually given
OMEGA = 1.35      # grievance left behind by force, carried to later encounters
GRV_DECAY = 0.88
KAPPA = 0.68      # discount applied to compliance that had to be coerced
MU = 0.30         # fraction of a registration that survives the fleet's absence
                  # without a durable local structure


# ==============================================================================
# PART 4 -- THE FORWARD PASS: ONE VOYAGE, END TO END
# ==============================================================================

def voyage(p, env, cfg):
    """Run a batch of voyages and return the loss plus a diagnostics dict.

    cfg flags
    ---------
      anonymous     : the counterpart has no memory; standing is not read
      reactive      : the manifest may see a summary of the actual encounters
      tied_display  : one dial for all six holds instead of six
      no_return     : drop the return-closure penalty
      no_persist    : reward only in-voyage compliance
    """
    n = env["season"].shape[0]
    zeros_n1 = Tn(np.zeros((n, 1)))

    # ---- (a) THE MANIFEST -----------------------------------------------
    # Committed once, from the season alone.  Under `reactive` the barrier is
    # broken and the provisioner is handed a summary of the voyage it is about
    # to make -- which is the counterfactual Zheng He did not have.
    s_in = Tn(env["season"])
    if cfg.get("reactive"):
        s_in = cat([s_in, Tn(env["forecast"])], axis=1)
    Wm1 = p["Wm1_r"] if cfg.get("reactive") else p["Wm1"]
    hm = (s_in @ Wm1 + p["bm1"]).tanh()
    manifest = (hm @ p["Wm2"] + p["bm2"]).softmax(axis=1) * BUDGET   # (n,K)

    r = manifest                       # remaining holds, differentiable
    rho = zeros_n1                     # standing
    grv = zeros_n1                     # accumulated grievance
    h = Tn(np.zeros((n, H)))

    credit_sum = zeros_n1
    persist_sum = zeros_n1
    distress = zeros_n1
    force_sum = zeros_n1
    disp_force_sum = zeros_n1
    gift_sum = zeros_n1
    cons_sum = zeros_n1
    comply_sum = zeros_n1
    prov_sum = zeros_n1
    force_trace, cred_trace = [], []

    for t in range(T_ENC):
        need_t = Tn(env["need"][:, t, :])
        wealth_t = Tn(env["wealth"][:, t, :])
        host_t = Tn(env["hostility"][:, t, :]) + grv
        tfrac = Tn(np.full((n, 1), t / (T_ENC - 1.0)))

        e_t = cat([need_t, host_t, wealth_t], axis=1)
        r_pos = r.relu()
        x = cat([h, e_t, r_pos, rho, tfrac], axis=1)
        h = (x @ p["Wp"] + p["bp"]).tanh()
        o = h @ p["Wo"] + p["bo"]

        # ---- (b) SHOWING, WHICH COSTS NOTHING AND IS BOUNDED BY WHAT IS HELD
        if cfg.get("tied_display"):
            one = o.col(0).sigmoid()                     # a single dial
            disp = cat([one] * K, axis=1) * r_pos
        else:
            disp = cat([o.col(i).sigmoid() for i in range(K)], axis=1) * r_pos

        # ---- (c) SPENDING, WHICH DOES NOT COME BACK
        gift = o.col(K).sigmoid() * r_pos.col(GIFT) * 0.5
        force = o.col(K + 1).sigmoid() * r_pos.col(FORCE) * 0.5
        cons = o.col(K + 2).sigmoid() * r_pos.col(TONGUE) * 0.5

        # ---- (d) THE COUNTERPART READS THE FLEET
        # Credibility is NEED-RELATIVE: what is shown counts only insofar as the
        # port has a use for it.  A hold of soldiers displayed to a port that
        # wants physicians is not a smaller signal, it is the wrong signal.
        matched = ((disp * need_t) @ p["wc"])            # (n,1)
        # A costless display is cheap talk.  It is believed in proportion to
        # what backs it: a standing the port already holds, or expenditure it
        # can watch.  Under `anonymous` the first term does not exist and the
        # fleet must buy belief again at every landfall.
        cred_z = p["c_b"] + (force * 2.0 + gift) * p["c_spend"]
        if not cfg.get("anonymous"):
            cred_z = cred_z + rho * p["c_rho"]
        cred = cred_z.sigmoid()
        believed = matched * cred
        seen = believed * p["alpha"]
        if not cfg.get("anonymous"):
            seen = seen + rho * p["wrho"]
        # The two instruments are not interchangeable and their exchange rate is
        # set by the port, not by the fleet.  Silk persuades a court that is
        # already inclined; a court that has decided to fight is unmoved by silk
        # and moved only by what it would lose.  All three of the fleet's actual
        # engagements -- Chen Zuyi at Palembang, Alakeshvara at Kotte, Sekandar
        # at Semudera -- were at the hostile end of this scale.
        gift_eff = gift * p["beta"] * (Tn(np.full((n, 1), 1.2)) - host_t).relu()
        force_eff = force * p["gamma"] * (host_t + 0.4)
        deter = disp.col(FORCE) * cred * host_t * p["deter"]
        provoke = disp @ p["w_prov"] \
            + disp.col(FORCE) * p["prov_f"] * (Tn(np.full((n, 1), 1.2)) - host_t).relu()
        prov_sum = prov_sum + provoke
        z = seen + deter + gift_eff + force_eff \
            - host_t * p["delta"] - provoke + p["b0"]
        comply = z.sigmoid()

        # Coerced compliance is worth less than volunteered compliance.
        credit = comply * (Tn(np.ones((n, 1))) - force * KAPPA).relu()

        # ---- (e) WHAT SURVIVES THE FLEET'S DEPARTURE
        durable = (cons * 2.5).tanh()
        persist = credit * (Tn(np.full((n, 1), MU)) + durable * (1.0 - MU))

        # ---- (f) STATE UPDATE
        delta_r = cat([
            Tn(np.zeros((n, 1))), Tn(np.zeros((n, 1))),
            force,                       # FORCE spent
            gift + cons * 0.3,           # GIFT spent, plus the cost of constituting
            cons,                        # TONGUE spent
            Tn(np.zeros((n, 1))),
        ], axis=1)
        r = r - Tn(env["leg"][:, t, :]) - delta_r
        rho = rho * LAMBDA + believed * ETA_D + force * ETA_F + gift * ETA_G
        # Memory is a single faculty.  A counterpart that cannot remember your
        # restraint cannot remember your violence either: under `anonymous` the
        # grievance term vanishes with the standing term, and force stops being
        # a debt.  This, and not any change of character, is what turns a
        # treasure fleet into a raiding fleet.
        if not cfg.get("anonymous"):
            grv = grv * GRV_DECAY + force * OMEGA

        credit_sum = credit_sum + credit
        persist_sum = persist_sum + persist
        comply_sum = comply_sum + comply
        force_sum = force_sum + force
        gift_sum = gift_sum + gift
        cons_sum = cons_sum + cons
        disp_force_sum = disp_force_sum + disp.col(FORCE)
        force_trace.append(force.v.mean())
        cred_trace.append(cred.v.mean())
        # running out mid-ocean is not a soft cost
        distress = distress + (r.col(WATER) * -1.0).relu().sq() \
                            + (r.col(GRAIN) * -1.0).relu().sq() \
                            + (r.col(REPAIR) * -1.0).relu().sq() * 0.5

    # ---- (g) RETURN CLOSURE ------------------------------------------------
    # The voyage is not scored on arrival.  It is scored on getting home.
    short = (Tn(np.full((n, 1), RETURN_REQ)) - r.col(WATER)).relu().sq() \
          + (Tn(np.full((n, 1), RETURN_REQ)) - r.col(GRAIN)).relu().sq()

    lam_ret = 0.0 if cfg.get("no_return") else 4.5
    lam_per = 0.0 if cfg.get("no_persist") else 1.4

    loss = (credit_sum * -1.0 + persist_sum * (-lam_per)
            + short * lam_ret + distress * 1.2).mean()

    diag = dict(
        manifest=manifest.v, credit=credit_sum.v, persist=persist_sum.v,
        force=force_sum.v, gift=gift_sum.v, cons=cons_sum.v,
        disp_force=disp_force_sum.v, comply=comply_sum.v,
        end_water=r.v[:, WATER], end_grain=r.v[:, GRAIN],
        end_force=r.v[:, FORCE], short=short.v, distress=distress.v,
        force_trace=np.array(force_trace), cred_trace=np.array(cred_trace),
        provoke=prov_sum.v,
    )
    return loss, diag


# ==============================================================================
# PART 5 -- TRAINING
# ==============================================================================

def collect(p, cfg):
    # Only the fleet learns.  The registrar constants above are the world.
    keys = ["Wm1_r" if cfg.get("reactive") else "Wm1", "bm1", "Wm2", "bm2",
            "Wp", "bp", "Wo", "bo"]
    return [(k, p[k]) for k in keys]


def train(cfg, seed=7, epochs=650, n=48, lr=0.04, verbose=False):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    if cfg.get("reactive"):
        # Identical initialisation to the committed model on the season columns
        # and zeros on the forecast columns, so the reactive variant begins the
        # search at exactly the committed variant's starting point and any
        # difference is attributable to the extra information, not to the seed.
        p["Wm1_r"] = Tn(np.vstack([p["Wm1"].v.copy(), np.zeros((K + 2, 24))]))
    params = collect(p, cfg)
    m = {k: np.zeros_like(t.v) for k, t in params}
    v = {k: np.zeros_like(t.v) for k, t in params}
    hist = []
    for ep in range(1, epochs + 1):
        env = make_voyages(n, np.random.default_rng(1000 + ep))
        for _, t in params:
            t.g = np.zeros_like(t.v)
        loss, _ = voyage(p, env, cfg)
        loss.backward()
        for k, t in params:
            g = np.clip(t.g, -5, 5)
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.999 * v[k] + 0.001 * (g * g)
            mh = m[k] / (1 - 0.9 ** ep)
            vh = v[k] / (1 - 0.999 ** ep)
            t.v -= lr * mh / (np.sqrt(vh) + 1e-8)
        hist.append(float(loss.v))
        if verbose and ep % 40 == 0:
            print(f"      epoch {ep:4d}  loss {float(loss.v):+.4f}")
    return p, hist


def evaluate(p, cfg, seed=999, n=256, shift=False):
    env = make_voyages(n, np.random.default_rng(seed), shift=shift)
    loss, d = voyage(p, env, cfg)
    return float(loss.v), d


# ==============================================================================
# PART 6 -- TESTS AND EXPERIMENTS
# ==============================================================================

def gradient_check():
    """Finite differences against the analytic gradient of the whole voyage."""
    rng = np.random.default_rng(3)
    p = init_params(rng)
    env = make_voyages(6, np.random.default_rng(11))
    cfg = {}
    params = collect(p, cfg)
    for _, t in params:
        t.g = np.zeros_like(t.v)
    loss, _ = voyage(p, env, cfg)
    loss.backward()

    worst, where = 0.0, ""
    eps = 1e-6
    for name, t in params:
        flat = t.v.reshape(-1)
        idxs = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, _ = voyage(p, env, cfg)
            flat[i] = old - eps
            lm, _ = voyage(p, env, cfg)
            flat[i] = old
            num = (float(lp.v) - float(lm.v)) / (2 * eps)
            ana = t.g.reshape(-1)[i]
            den = max(1.0, abs(num), abs(ana))
            rel = abs(num - ana) / den
            if rel > worst:
                worst, where = rel, f"{name}[{i}]  num={num:+.6e} ana={ana:+.6e}"
    return worst, where


def bar(x, lo, hi, w=28):
    f = 0.0 if hi <= lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))
    k = int(round(f * w))
    return "#" * k + "." * (w - k)


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print(" THE MANIFEST ENGINE -- Zheng He (1371-1433) -- figure 0387")
    print("=" * 78)

    results = {}

    # ---- TEST 1 -------------------------------------------------------------
    print("\n[1] GRADIENT CHECK (finite differences, whole voyage)")
    worst, where = gradient_check()
    ok1 = worst < 1e-6
    print(f"    worst relative error : {worst:.3e}   {'PASS' if ok1 else 'FAIL'}")
    print(f"    worst coordinate     : {where}")
    results["grad"] = ok1

    # ---- TEST 2 -------------------------------------------------------------
    print("\n[2] TRAINING THE FULL ENGINE")
    p_full, hist = train({}, verbose=True)
    ok2 = hist[-1] < hist[0] - 0.5
    print(f"    loss {hist[0]:+.4f} -> {hist[-1]:+.4f}   {'PASS' if ok2 else 'FAIL'}")
    results["train"] = ok2

    lo, dfull = evaluate(p_full, {})
    mani = dfull["manifest"].mean(axis=0)
    print("\n    LEARNED MANIFEST (mean allocation of a budget of "
          f"{BUDGET:.1f}, committed in port):")
    for i, nm in enumerate(HOLD_NAMES):
        print(f"      {nm:<7s} {mani[i]:5.3f}  {bar(mani[i], 0, mani.max())}")

    # ---- TEST 3 : the unspent asset ----------------------------------------
    print("\n[3] CLAIM C1 -- CAPABILITY HELD AS EVIDENCE, NOT AS FORCE")
    shown = dfull["disp_force"].mean()
    spent = dfull["force"].mean()
    ratio = shown / max(spent, 1e-9)
    reserve = dfull["end_force"].mean() / mani[FORCE]
    print(f"    FORCE displayed per voyage : {shown:8.4f}")
    print(f"    FORCE expended per voyage  : {spent:8.4f}")
    print(f"    show / spend ratio         : {ratio:8.2f} : 1")
    print(f"    FORCE hold unspent at home : {100*reserve:7.2f} %")
    print(f"    mean compliance secured    : {dfull['comply'].mean():8.4f}"
          f"   of {T_ENC} encounters")
    ok3 = ratio > 3.0 and reserve > 0.5
    print(f"    {'PASS' if ok3 else 'FAIL'}")
    results["c1"] = ok3

    # ---- TEST 4 : the counterpart's memory ---------------------------------
    print("\n[4] CLAIM C2 -- DELETE THE COUNTERPART'S MEMORY")
    p_anon, _ = train({"anonymous": True})
    la, danon = evaluate(p_anon, {"anonymous": True})
    print(f"    {'variant':<20s}{'FORCE shown':>12s}{'FORCE spent':>12s}"
          f"{'show/spend':>12s}{'register':>10s}")
    for nm, d in (("remembered (full)", dfull), ("anonymous", danon)):
        rr = d['disp_force'].mean() / max(d['force'].mean(), 1e-9)
        print(f"    {nm:<20s}{d['disp_force'].mean():12.4f}{d['force'].mean():12.4f}"
              f"{rr:12.2f}{d['credit'].mean():10.4f}")
    keep = danon["credit"].mean() / max(dfull["credit"].mean(), 1e-9)
    print(f"    register retained when nobody remembers : {100*keep:.1f} %")
    r_full = dfull['disp_force'].mean() / max(dfull['force'].mean(), 1e-9)
    r_anon = danon['disp_force'].mean() / max(danon['force'].mean(), 1e-9)
    print(f"    show/spend collapses x{r_full / max(r_anon, 1e-9):.1f}")
    ok4 = (r_full > 4.0 * r_anon) and (danon['credit'].mean()
                                       < 0.75 * dfull['credit'].mean())
    print(f"    {'PASS' if ok4 else 'FAIL'}")
    results["c2"] = ok4

    # ---- TEST 5 : the barrier ----------------------------------------------
    print("\n[5] CLAIM C3 -- BREAKING THE COMMITMENT BARRIER")
    p_react, _ = train({"reactive": True})

    def provision_report(p_, cfg_, shift):
        env = make_voyages(256, np.random.default_rng(999), shift=shift)
        l_, d_ = voyage(p_, env, cfg_)
        demand = env["leg"][:, :, WATER].sum(axis=1) + RETURN_REQ
        alloc = d_["manifest"][:, WATER]
        return dict(loss=float(l_.v),
                    corr=float(np.corrcoef(alloc, demand)[0, 1]),
                    hedge=float(np.mean(alloc - demand)),
                    sd=float(alloc.std()),
                    short=float(d_["short"].mean()),
                    credit=float(d_["credit"].mean()))

    ci = provision_report(p_full, {}, False)
    cs = provision_report(p_full, {}, True)
    ri = provision_report(p_react, {"reactive": True}, False)
    rs = provision_report(p_react, {"reactive": True}, True)
    print(f"    {'':<22s}{'corr(alloc,need)':>18s}{'hedge':>9s}{'spread':>9s}"
          f"{'shortfall':>11s}{'credit':>9s}{'loss':>10s}")
    print(f"    {'committed  expected':<22s}{ci['corr']:18.3f}{ci['hedge']:+9.3f}"
          f"{ci['sd']:9.3f}{ci['short']:11.3f}{ci['credit']:9.3f}{ci['loss']:+10.3f}")
    print(f"    {'reactive   expected':<22s}{ri['corr']:18.3f}{ri['hedge']:+9.3f}"
          f"{ri['sd']:9.3f}{ri['short']:11.3f}{ri['credit']:9.3f}{ri['loss']:+10.3f}")
    print(f"    {'committed  unforeseen':<22s}{cs['corr']:18.3f}{cs['hedge']:+9.3f}"
          f"{cs['sd']:9.3f}{cs['short']:11.3f}{cs['credit']:9.3f}{cs['loss']:+10.3f}")
    print(f"    {'reactive   unforeseen':<22s}{rs['corr']:18.3f}{rs['hedge']:+9.3f}"
          f"{rs['sd']:9.3f}{rs['short']:11.3f}{rs['credit']:9.3f}{rs['loss']:+10.3f}")
    print(f"    price of commitment on the expected world : "
          f"{ci['loss'] - ri['loss']:+.3f}")
    print(f"    value of commitment on the one it did not : "
          f"{rs['loss'] - cs['loss']:+.3f}")
    ok5 = (ri["loss"] < ci["loss"]) and (rs["loss"] > cs["loss"])
    print(f"    {'PASS' if ok5 else 'MIXED'}")
    results["c3"] = ok5

    # ---- TEST 6 : one dial ---------------------------------------------------
    print("\n[6] CLAIM C4 -- ONE DIAL INSTEAD OF SIX")
    p_tied, _ = train({"tied_display": True})
    lt, dtied = evaluate(p_tied, {"tied_display": True})
    print(f"    {'variant':<24s}{'loss':>10s}{'credit':>10s}{'compliance':>13s}")
    print(f"    {'six holds shown apart':<24s}{lo:10.4f}"
          f"{dfull['credit'].mean():10.4f}{dfull['comply'].mean():13.4f}")
    print(f"    {'one undifferentiated dial':<24s}{lt:10.4f}"
          f"{dtied['credit'].mean():10.4f}{dtied['comply'].mean():13.4f}")
    ok6 = dtied["credit"].mean() < dfull["credit"].mean()
    print(f"    {'PASS' if ok6 else 'FAIL'}")
    results["c4"] = ok6

    # ---- TEST 7 : the Cochin test -------------------------------------------
    print("\n[7] CLAIM C5 -- WHAT SURVIVES THE FLEET'S WITHDRAWAL")
    p_np, _ = train({"no_persist": True})
    lnp, dnp = evaluate(p_np, {"no_persist": True})
    print(f"    {'objective':<28s}{'constituting':>14s}{'in-voyage':>12s}"
          f"{'after withdrawal':>19s}")
    for nm, d in (("compliance + persistence", dfull),
                  ("compliance only", dnp)):
        print(f"    {nm:<28s}{d['cons'].mean():14.4f}{d['credit'].mean():12.4f}"
              f"{d['persist'].mean():19.4f}")
    keep_full = dfull["persist"].mean() / max(dfull["credit"].mean(), 1e-9)
    keep_np = dnp["persist"].mean() / max(dnp["credit"].mean(), 1e-9)
    print(f"    fraction of the register surviving absence: "
          f"{100*keep_full:.1f}% vs {100*keep_np:.1f}%")
    ok7 = keep_full > keep_np
    print(f"    {'PASS' if ok7 else 'FAIL'}")
    results["c5"] = ok7

    # ---- TEST 8 : the return constraint --------------------------------------
    print("\n[8] CONTROL -- REMOVING THE REQUIREMENT TO GET HOME")
    p_nr, _ = train({"no_return": True})
    _, dnr = evaluate(p_nr, {"no_return": True})
    print(f"    {'variant':<26s}{'end WATER':>12s}{'end GRAIN':>12s}"
          f"{'shortfall':>12s}{'credit':>10s}")
    for nm, d in (("return enforced", dfull), ("return dropped", dnr)):
        print(f"    {nm:<26s}{d['end_water'].mean():12.4f}"
              f"{d['end_grain'].mean():12.4f}{d['short'].mean():12.4f}"
              f"{d['credit'].mean():10.4f}")
    ok8 = dnr["short"].mean() > dfull["short"].mean()
    print(f"    {'PASS' if ok8 else 'FAIL'}")
    results["ret"] = ok8

    # ---- TEST 9 : the two contested claims, across seeds ---------------------
    print("\n[9] SEED SWEEP -- are C2 and C3 properties of the design or of one seed?")
    print(f"    {'seed':>6s}{'show/spend kept':>17s}{'show/spend anon':>17s}"
          f"{'register kept':>15s}{'price':>9s}{'value':>9s}")
    sweep_ok = True
    for sd in (7, 19, 31):
        pf_, _ = train({}, seed=sd) if sd != 7 else (p_full, None)
        pa_, _ = train({"anonymous": True}, seed=sd) if sd != 7 else (p_anon, None)
        pr_, _ = train({"reactive": True}, seed=sd) if sd != 7 else (p_react, None)
        _, df_ = evaluate(pf_, {})
        _, da_ = evaluate(pa_, {"anonymous": True})
        rf = df_["disp_force"].mean() / max(df_["force"].mean(), 1e-9)
        ra = da_["disp_force"].mean() / max(da_["force"].mean(), 1e-9)
        keep = da_["credit"].mean() / max(df_["credit"].mean(), 1e-9)
        ci_ = evaluate(pf_, {})[0]
        ri_ = evaluate(pr_, {"reactive": True})[0]
        cs_ = evaluate(pf_, {}, shift=True)[0]
        rs_ = evaluate(pr_, {"reactive": True}, shift=True)[0]
        price, value = ci_ - ri_, rs_ - cs_
        print(f"    {sd:6d}{rf:17.2f}{ra:17.2f}{100*keep:14.1f}%"
              f"{price:+9.3f}{value:+9.3f}")
        if not (rf > 4.0 * ra and keep < 0.75 and price > 0 and value > 0):
            sweep_ok = False
    print(f"    {'PASS' if sweep_ok else 'MIXED -- reported as found'}")
    results["seeds"] = sweep_ok

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 78)
    npass = sum(1 for v in results.values() if v)
    print(f" {npass}/{len(results)} checks passed: "
          + "  ".join(f"{k}={'ok' if v else 'X'}" for k, v in results.items()))
    print("=" * 78)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

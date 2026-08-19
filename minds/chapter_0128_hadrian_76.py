#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 128: Hadrian (76-138 CE)
================================================================================   

WHAT THIS FILE IS
-----------------
A small, self-contained neural architecture built from scratch in NumPy that
encodes the *specific* cognitive signature of the emperor Hadrian - not the
tired "emperor builds a wall, therefore boundary-AI" reading, but the mind the
sources actually show us:

  1. GOVERNANCE BY AUTOPSY (Greek 'autopsia' = seeing-for-oneself).
     Hadrian spent roughly twelve of his twenty-one years on the road,
     inspecting legions on foot and questioning provincials directly. His unit
     of knowledge was the VISIT, not the dispatch. He trusted what he had gone
     and verified in person.

  2. RECONSTRUCTIVE MEMORY (the Villa at Tivoli).
     He did not merely remember places he had seen; he physically REBUILT them
     - the Canopus of Egypt, the Poikile of Athens, the Vale of Tempe, even a
     representation of the underworld. Memory, for Hadrian, was a generative
     copy you could walk through and check against the original.

  3. THE WANDERING LITTLE SOUL (his own deathbed poem).
     "Animula vagula blandula..." - little soul, wandering, coaxing, guest and
     companion of the body, off now to pale cold places. The persistent thing
     that carries state through a life is small, playful, and mobile.

  4. THE SELF-GOVERNING, MULTIFORM PSYCHE.
     The Epitome de Caesaribus calls him 'varius multiplex multiformis' - an
     "arbiter of vices and virtues" who governed the impulse of his mind
     "by a certain artifice." A controller that regulates its own contradictory
     drives, rather than a single fixed objective.

  (And the shadow that keeps this from being a hagiography: the same
   reconstructive drive that lovingly rebuilt Egypt at Tivoli also tried to
   OVERWRITE Jerusalem - rebuilt as Aelia Capitolina, the province renamed
   Syria Palaestina after the Bar Kokhba war. The collector who could copy a
   place could also try to delete one. That is encoded below as an explicit
   'no-overwrite' consistency pressure, and discussed in the companion text.)

THE ARCHITECTURE: THE PERIPATETIC ATLAS NETWORK (PAN)
-----------------------------------------------------
Most modern models answer from stored weights in a single feed-forward pass -
in Hadrian's terms, that is trusting the DISPATCH. PAN instead *travels*.

  - ATLAS  A : a set of M learnable "province" sites (the empire / the Villa).
  - ANIMULA s0 : a small learnable initial "soul" state, explicitly regularized
                 toward small norm (it is a *little* soul).
  - A TOUR of T legs. At each leg the current soul decides where to go next
    (a soft ITINERARY over provinces), VISITS one (reads it directly, not a
    summary), tries to RECONSTRUCT it from memory (the autopsy / Villa check:
    you only truly know a place if you can rebuild it), and CONSOLIDATES what it
    verified into the soul through a gate that is pressured NOT to annex more
    than it can integrate ("consolidation, not expansion").
  - A READOUT from the consolidated soul at journey's end.

None of this is attention-over-stored-keys, MoE, or a Transformer. The core
operation is a learned *journey* through a reconstructive map, which is the
mechanism Hadrian's biography actually suggests.

The file is fully runnable: it defines the model, PROVES its gradients with a
finite-difference check, trains on a synthetic "Imperial Survey" task that only
rewards visiting the right provinces, reports an interpretable routing metric,
compares against a plain MLP baseline, and runs self-tests. Execute it directly.

    python3 chapter_0128_hadrian_76.py  
"""

from __future__ import annotations
import numpy as np


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 130  # the figure's id, for luck
rng = np.random.default_rng(SEED)


# =============================================================================
# 1. THE SYNTHETIC TASK: "THE IMPERIAL SURVEY"
# -----------------------------------------------------------------------------
# There are M provinces, each with a hidden 'condition' scalar (part of the
# world, unknown to the model). Each survey names a subset of provinces that
# must be inspected this journey (a query mask m). But the emperor does not
# receive the mask cleanly: it arrives SCRAMBLED and noisy, the way a real
# report reaches a capital garbled. The truth of the empire is:
#
#       y* = tanh( sum_j  m_j * condition_j )
#
# To predict y*, the model must (a) decode which provinces the survey concerns,
# (b) actually route its tour to those provinces, and (c) have internalized each
# province's condition well enough to consolidate it. A model that visits the
# wrong provinces consolidates the wrong conditions and fails. That is the whole
# point: knowledge here is earned by going to the right places.
# =============================================================================
class ImperialSurvey:
    def __init__(self, M=6, max_queried=3, seed=SEED):
        r = np.random.default_rng(seed + 1)
        self.M = M
        self.max_queried = max_queried
        # Hidden ground-truth province conditions (unknown to the model).
        self.conditions = r.uniform(-1.0, 1.0, size=M)
        # A fixed linear scramble + noise level for the incoming report.
        self.scramble = r.standard_normal((M, M)) / np.sqrt(M)
        self.noise = 0.05

    def batch(self, B, seed=None):
        r = np.random.default_rng(seed) if seed is not None else rng
        # Random query masks: 1..max_queried provinces switched on.
        m = np.zeros((B, self.M))
        for b in range(B):
            k = r.integers(1, self.max_queried + 1)
            idx = r.choice(self.M, size=k, replace=False)
            m[b, idx] = 1.0
        # The report the model actually sees: scrambled, noisy.
        x = m @ self.scramble.T + self.noise * r.standard_normal((B, self.M))
        # The truth of the empire.
        y = np.tanh(m @ self.conditions).reshape(B, 1)
        return x, y, m


# =============================================================================
# 2. THE MODEL: PERIPATETIC ATLAS NETWORK
# =============================================================================
class PeripateticAtlasNetwork:
    """
    Dimensions
    ----------
      n_in : size of the incoming (scrambled) report            [= M]
      M    : number of provinces / memory sites in the atlas
      d    : size of a province's routing KEY ('how to find a place')
      dv   : size of a province's VALUE / condition (its payload)   [= 1]
      h    : size of the soul state (the animula)
      T    : number of legs in the tour (how far the emperor travels)

    Design principle (this is the whole point)
    ------------------------------------------
    WHERE you go is decided by the survey (the query); WHAT you learn is decided
    only by GOING there. So:
      * routing is driven by the query q AND the running soul -> the itinerary
        can point at exactly the provinces a survey names;
      * a province's condition (its value) can enter the soul ONLY through an
        actual visit v = p @ A_val. There is no back channel. If the tour visits
        the wrong provinces it consolidates the wrong conditions and fails.
    This makes autopsy load-bearing: the model literally has to travel correctly.

    Parameters (learned unless noted)
    ---------------------------------
      Wq, bq : encode the report into a 'reason to travel' q       (h x n_in)
      Wqr    : query   -> routing direction                        (d x h)
      Wr     : soul    -> routing direction                        (d x h)
      A_key  : the atlas KEYS - how each province is addressed      (M x d)
      A_val  : the atlas VALUES - each province's condition (FIXED) (M x dv)
      s0     : the animula - the small initial soul                 (h,)
      Wg, bg : the consolidation gate      (on [soul, visited value])(h x (h+dv))
      Wc, bc : the consolidation candidate (on [soul, visited value])(h x (h+dv))
      Wrec   : the Villa reconstruction ('can I rebuild the place?')(dv x h)
      Wo, bo : readout from the consolidated soul                   (1 x h)
    """

    def __init__(self, n_in=6, M=6, d=4, dv=1, h=8, T=6,
                 lam_autopsy=0.3, lam_animula=0.02, lam_consol=0.05,
                 lam_entropy=0.02, lam_route=0.5, gate_target=0.30,
                 atlas_values=None, seed=SEED):
        self.n_in, self.M, self.d, self.dv, self.h, self.T = n_in, M, d, dv, h, T
        self.lam_autopsy = lam_autopsy      # weight on the reconstruction check
        self.lam_animula = lam_animula      # keep the soul 'little'
        self.lam_consol = lam_consol        # discourage over-annexation
        self.lam_entropy = lam_entropy      # 'be in ONE place at a time' (autopsy)
        self.lam_route = lam_route          # actually GO where the survey names
        self.gate_target = gate_target      # a modest opening of the gate

        r = np.random.default_rng(seed + 2)
        def he(shape, fan_in):
            return r.standard_normal(shape) * np.sqrt(2.0 / fan_in)

        # The atlas VALUES are the true province conditions: fixed geography the
        # model does not get to invent. It can only reach them by visiting.
        if atlas_values is None:
            atlas_values = r.uniform(-1, 1, size=(M, dv))
        self.A_val = np.asarray(atlas_values, dtype=float).reshape(M, dv)

        self.P = {
            "Wq":   he((h, n_in), n_in),
            "bq":   np.zeros(h),
            "Wqr":  he((d, h), h),
            "Wr":   he((d, h), h),
            "A_key": r.standard_normal((M, d)) * 0.5,
            "s0":   r.standard_normal(h) * 0.1,      # starts small
            "Wg":   he((h, h + dv), h + dv),
            "bg":   np.zeros(h),
            "Wc":   he((h, h + dv), h + dv),
            "bc":   np.zeros(h),
            "Wrec": he((dv, h), h),
            "Wo":   he((1, h), h),
            "bo":   np.zeros(1),
        }

    # ---- small numerics helpers -------------------------------------------
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    # -----------------------------------------------------------------------
    # FORWARD.  Returns predictions y and a cache with every intermediate we
    # need for exact backprop, plus the routing history (for diagnostics).
    # -----------------------------------------------------------------------
    def forward(self, X):
        P = self.P
        B = X.shape[0]
        h, d, dv, M, T = self.h, self.d, self.dv, self.M, self.T

        # The report -> a reason to travel (computed once, consulted every leg).
        q_pre = X @ P["Wq"].T + P["bq"]           # (B,h)
        q = np.tanh(q_pre)                         # (B,h)
        rq = q @ P["Wqr"].T                        # (B,d) the query's pull

        S = np.broadcast_to(P["s0"], (B, h)).copy()  # the soul at leg 0
        steps = []                                    # per-leg intermediates

        for t in range(T):
            S_prev = S
            # itinerary: where the survey pulls us + where the soul now leans
            u = S_prev @ P["Wr"].T + rq            # (B,d) routing direction
            logits = u @ P["A_key"].T              # (B,M) affinity to provinces
            p = self._softmax(logits)              # (B,M) the itinerary
            v = p @ self.A_val                     # (B,dv) the CONDITION visited

            recon = S_prev @ P["Wrec"].T            # (B,dv) rebuilt-from-memory
            # autopsy residual: can the soul reconstruct where it just stood?
            resid = recon - v                       # (B,dv)

            gcat = np.concatenate([S_prev, v], axis=1)      # (B,h+dv)
            g_pre = gcat @ P["Wg"].T + P["bg"]              # (B,h)
            g = self._sigmoid(g_pre)                        # (B,h) how much to absorb
            gbar = g.mean(axis=1)                           # (B,) mean opening

            ccat = gcat                                      # same concat here
            c_pre = ccat @ P["Wc"].T + P["bc"]              # (B,h)
            c = np.tanh(c_pre)                              # (B,h) candidate update

            S = g * c + (1.0 - g) * S_prev                  # consolidate

            steps.append(dict(S_prev=S_prev, u=u, logits=logits, p=p, v=v,
                              recon=recon, resid=resid, gcat=gcat, g=g,
                              gbar=gbar, c=c, S=S))

        y = S @ P["Wo"].T + P["bo"]                 # (B,1) readout from final soul
        cache = dict(X=X, q_pre=q_pre, q=q, rq=rq, steps=steps, S_final=S, B=B)
        return y, cache

    # -----------------------------------------------------------------------
    # LOSS.  task + autopsy(reconstruction) + animula(small soul) + consol.
    # -----------------------------------------------------------------------
    def loss(self, y, y_true, cache, m=None):
        B, T = cache["B"], self.T
        # (a) task
        diff = y - y_true
        L_task = 0.5 * np.sum(diff ** 2) / B
        # (b) autopsy / Villa reconstruction across all legs
        L_aut = 0.0
        for st in cache["steps"]:
            L_aut += 0.5 * np.sum(st["resid"] ** 2)
        L_aut = self.lam_autopsy * L_aut / B
        # (c) animula: keep the soul small
        L_ani = 0.5 * self.lam_animula * np.sum(self.P["s0"] ** 2)
        # (d) consolidation restraint: gate should open only modestly
        L_con = 0.0
        for st in cache["steps"]:
            L_con += np.sum((st["gbar"] - self.gate_target) ** 2)
        L_con = 0.5 * self.lam_consol * L_con / (B * T)
        # (e) itinerary sharpness: 'you are in one province at a time' (autopsy).
        #     Minimizing routing entropy peaks each leg toward a single visit.
        L_ent = 0.0
        for st in cache["steps"]:
            p = st["p"]
            L_ent += -np.sum(p * np.log(p + 1e-12))
        L_ent = self.lam_entropy * L_ent / (B * T)
        # (f) itinerary coverage: the average tour should cover the provinces the
        #     survey actually named (you go where you are told to inspect).
        L_route = 0.0
        if m is not None and self.lam_route > 0:
            k = np.maximum(m.sum(axis=1, keepdims=True), 1)
            r_star = m / k                                  # desired coverage
            pbar = np.mean([st["p"] for st in cache["steps"]], axis=0)  # (B,M)
            L_route = -np.sum(r_star * np.log(pbar + 1e-12)) / B
            L_route = self.lam_route * L_route
        total = L_task + L_aut + L_ani + L_con + L_ent + L_route
        return total, dict(task=L_task, autopsy=L_aut, animula=L_ani,
                           consol=L_con, entropy=L_ent, route=L_route)

    # -----------------------------------------------------------------------
    # BACKWARD.  Exact analytic gradients for every parameter (BPTT through the
    # gated tour, the softmax itinerary, and the reconstruction terms).
    # -----------------------------------------------------------------------
    def backward(self, y, y_true, cache, m=None):
        P = self.P
        B, T = cache["B"], self.T
        h, d, M = self.h, self.d, self.M
        la, lc = self.lam_autopsy, self.lam_consol
        gt = self.gate_target

        grads = {k: np.zeros_like(v) for k, v in P.items()}
        h, d, dv = self.h, self.d, self.dv

        # routing-coverage gradient (spread over legs via pbar = mean_t p_t)
        dp_route = None
        if m is not None and self.lam_route > 0:
            k = np.maximum(m.sum(axis=1, keepdims=True), 1)
            r_star = m / k
            pbar = np.mean([st["p"] for st in cache["steps"]], axis=0)
            # dL_route/dp_{t} = (lam/(B*T)) * (-r_star / pbar)
            dp_route = (self.lam_route / (B * T)) * (-(r_star / (pbar + 1e-12)))

        q = cache["q"]

        # ---- readout ------------------------------------------------------
        gy = (y - y_true) / B                        # dL_task/dy   (B,1)
        S_final = cache["S_final"]
        grads["Wo"] += gy.T @ S_final                # (1,h)
        grads["bo"] += gy.sum(axis=0)                # (1,)
        dS = gy @ P["Wo"]                            # (B,h) grad wrt final soul

        drq = np.zeros_like(cache["rq"])             # grad wrt query pull (B,d)

        # ---- walk the tour backwards -------------------------------------
        for t in reversed(range(T)):
            st = cache["steps"][t]
            S_prev = st["S_prev"]; v = st["v"]; g = st["g"]; c = st["c"]
            p = st["p"]; u = st["u"]; resid = st["resid"]; gbar = st["gbar"]

            # S = g*c + (1-g)*S_prev
            dg = dS * (c - S_prev)
            dc = dS * g
            dS_prev = dS * (1.0 - g)

            # consolidation restraint feeds the gate mean
            dgbar = (lc / (B * T)) * (gbar - gt)          # (B,)
            dg = dg + (dgbar[:, None] / h)                # broadcast to h units

            # candidate c = tanh(c_pre); c_pre = ccat @ Wc.T + bc ; ccat=[S_prev,v]
            dc_pre = dc * (1.0 - c ** 2)
            grads["Wc"] += dc_pre.T @ st["gcat"]
            grads["bc"] += dc_pre.sum(axis=0)
            dccat = dc_pre @ P["Wc"]                       # (B,h+dv)
            dS_prev += dccat[:, :h]
            gV = dccat[:, h:h + dv].copy()                # grad wrt visited value

            # gate g = sigmoid(g_pre); g_pre = gcat @ Wg.T + bg ; gcat=[S_prev,v]
            dg_pre = dg * g * (1.0 - g)
            grads["Wg"] += dg_pre.T @ st["gcat"]
            grads["bg"] += dg_pre.sum(axis=0)
            dgcat = dg_pre @ P["Wg"]                       # (B,h+dv)
            dS_prev += dgcat[:, :h]
            gV += dgcat[:, h:h + dv]

            # autopsy: 0.5*la*||recon - v||^2 ; recon = S_prev @ Wrec.T
            d_resid = (la / B) * resid                    # (B,dv)
            grads["Wrec"] += d_resid.T @ S_prev
            dS_prev += d_resid @ P["Wrec"]
            gV += -d_resid                                # resid = recon - v

            # v = p @ A_val   (A_val fixed -> no grad to it)
            dp = gV @ self.A_val.T                         # (B,M)

            # entropy pressure on the itinerary: dL_ent/dp_j = -lam*(log p + 1)
            dp = dp + (self.lam_entropy / (B * self.T)) * (-(np.log(p + 1e-12) + 1.0))

            # routing-coverage supervision (same for every leg via pbar)
            if dp_route is not None:
                dp = dp + dp_route

            # p = softmax(logits)  (softmax jacobian applied to combined dp)
            dlogits = p * (dp - (dp * p).sum(axis=1, keepdims=True))

            # logits = u @ A_key.T
            du = dlogits @ P["A_key"]                      # (B,d)
            grads["A_key"] += dlogits.T @ u                # (M,d)

            # u = S_prev @ Wr.T + rq
            grads["Wr"] += du.T @ S_prev                   # (d,h)
            dS_prev += du @ P["Wr"]
            drq += du                                      # query pull grad

            dS = dS_prev                                   # pass to previous leg

        # ---- rq = q @ Wqr.T ----------------------------------------------
        grads["Wqr"] += drq.T @ q                          # (d,h)
        dq = drq @ P["Wqr"]                                # (B,h)

        # ---- q = tanh(q_pre); q_pre = X @ Wq.T + bq -----------------------
        dq_pre = dq * (1.0 - q ** 2)
        grads["Wq"] += dq_pre.T @ cache["X"]
        grads["bq"] += dq_pre.sum(axis=0)

        # ---- s0: grad from the tour (broadcast) + animula regularizer -----
        grads["s0"] += dS.sum(axis=0)                     # tour contribution
        grads["s0"] += self.lam_animula * P["s0"]         # keep soul small

        return grads

    # ---- convenience: routing concentration on queried provinces ----------
    def routing_on_target(self, cache, m):
        """Fraction of total tour mass (averaged over legs) that landed on the
        provinces the survey actually named. 1.0 = the tour only ever visited
        queried provinces; 1/M = uniform wandering."""
        Pmass = np.zeros_like(m)
        for st in cache["steps"]:
            Pmass += st["p"]
        Pmass /= self.T
        queried = (m > 0).astype(float)
        return (Pmass * queried).sum(axis=1).mean()


# =============================================================================
# 3. A PLAIN MLP BASELINE (no travel, no reconstruction) - for comparison.
# =============================================================================
class MLPBaseline:
    def __init__(self, n_in=6, hidden=32, seed=SEED):
        r = np.random.default_rng(seed + 7)
        self.W1 = r.standard_normal((hidden, n_in)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(hidden)
        self.W2 = r.standard_normal((1, hidden)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(1)

    def forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = np.tanh(z1)
        y = a1 @ self.W2.T + self.b2
        return y, (X, z1, a1)

    def step(self, X, y_true, lr):
        y, (X, z1, a1) = self.forward(X)
        B = X.shape[0]
        gy = (y - y_true) / B
        gW2 = gy.T @ a1; gb2 = gy.sum(0)
        da1 = gy @ self.W2
        dz1 = da1 * (1 - a1 ** 2)
        gW1 = dz1.T @ X; gb1 = dz1.sum(0)
        self.W2 -= lr * gW2; self.b2 -= lr * gb2
        self.W1 -= lr * gW1; self.b1 -= lr * gb1
        return 0.5 * np.sum((y - y_true) ** 2) / B


# =============================================================================
# 4. GRADIENT CHECK (mandatory).  Finite differences vs. analytic gradients.
# =============================================================================
def gradient_check(model, X, y_true, m=None, n_coords=6, eps=1e-6):
    def total_loss():
        y, cache = model.forward(X)
        L, _ = model.loss(y, y_true, cache, m=m)
        return L

    y, cache = model.forward(X)
    grads = model.backward(y, y_true, cache, m=m)

    check_rng = np.random.default_rng(0)
    worst = 0.0
    report = []
    for name, W in model.P.items():
        flat = W.ravel()
        k = min(n_coords, flat.size)
        idxs = check_rng.choice(flat.size, size=k, replace=False)
        max_rel = 0.0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; Lp = total_loss()
            flat[i] = orig - eps; Lm = total_loss()
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = grads[name].ravel()[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
        worst = max(worst, max_rel)
        report.append((name, max_rel))
    return worst, report


# =============================================================================
# 5. TRAINING LOOP (real).  Plain SGD with a touch of momentum.
# =============================================================================
def train(model, task, steps=1500, B=64, lr=0.05, mom=0.9, log_every=250):
    vel = {k: np.zeros_like(v) for k, v in model.P.items()}
    history = []
    for it in range(1, steps + 1):
        X, y_true, m = task.batch(B, seed=1000 + it)
        y, cache = model.forward(X)
        L, parts = model.loss(y, y_true, cache, m=m)
        grads = model.backward(y, y_true, cache, m=m)
        for k in model.P:
            vel[k] = mom * vel[k] - lr * grads[k]
            model.P[k] += vel[k]
        if it % log_every == 0 or it == 1:
            Xv, yv, mv = task.batch(512, seed=999)
            yp, cv = model.forward(Xv)
            val = 0.5 * np.mean((yp - yv) ** 2)
            ont = model.routing_on_target(cv, mv)
            history.append((it, L, val, ont))
            print(f"  step {it:5d} | train {L:.5f} | val_mse {val:.5f} "
                  f"| task {parts['task']:.4f} autopsy {parts['autopsy']:.4f} "
                  f"consol {parts['consol']:.4f} | routing_on_target {ont:.3f}")
    return history


# =============================================================================
# 6. MAIN: prove gradients, train, compare, self-test.
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print("HADRIAN  -  Peripatetic Atlas Network  (govern by autopsy, not dispatch)")
    print("=" * 74)

    task = ImperialSurvey(M=6, max_queried=3)
    # The atlas is the real empire: the province conditions are fixed geography
    # the model can reach only by travelling there.
    model = PeripateticAtlasNetwork(n_in=6, M=6, d=4, dv=1, h=8, T=6,
                                    atlas_values=task.conditions.reshape(-1, 1))

    # --- 6a. gradient check ---------------------------------------------
    print("\n[1] GRADIENT CHECK (finite differences vs. analytic backprop)")
    Xg, yg, mg = task.batch(16, seed=7)
    worst, report = gradient_check(model, Xg, yg, m=mg)
    for name, rel in report:
        print(f"    {name:5s}  max rel err = {rel:.2e}")
    print(f"    ----> worst relative error = {worst:.2e}")
    ok_grad = worst < 1e-4
    print(f"    gradient check {'PASSED' if ok_grad else 'FAILED'} (threshold 1e-4)")

    # --- 6b. train PAN --------------------------------------------------
    print("\n[2] TRAINING the Peripatetic Atlas Network on the Imperial Survey")
    hist = train(model, task, steps=1500, B=64, lr=0.05)

    # --- 6c. baseline ---------------------------------------------------
    print("\n[3] BASELINE: a plain MLP with no tour and no reconstruction")
    mlp = MLPBaseline(n_in=6, hidden=32)
    for it in range(1, 1501):
        Xb, yb, mb = task.batch(64, seed=1000 + it)
        mlp.step(Xb, yb, lr=0.05)
    Xv, yv, mv = task.batch(512, seed=999)
    yp_mlp, _ = mlp.forward(Xv)
    mlp_mse = 0.5 * np.mean((yp_mlp - yv) ** 2)
    yp_pan, cv = model.forward(Xv)
    pan_mse = 0.5 * np.mean((yp_pan - yv) ** 2)
    pan_ont = model.routing_on_target(cv, mv)
    print(f"    PAN  final val MSE = {pan_mse:.5f} | routing_on_target = {pan_ont:.3f}")
    print(f"    MLP  final val MSE = {mlp_mse:.5f}")

    # --- 6d. show the learned itinerary for one survey ------------------
    print("\n[4] A SINGLE SURVEY, WATCHED LEG BY LEG")
    Xs, ys, ms = task.batch(1, seed=42)
    yps, cs = model.forward(Xs)
    queried = np.where(ms[0] > 0)[0]
    print(f"    provinces the survey asked us to inspect : {queried.tolist()}")
    for t, stp in enumerate(cs["steps"]):
        route = stp["p"][0]
        print(f"    leg {t}: itinerary over provinces = {route}  "
              f"-> visits #{int(route.argmax())}")
    print(f"    truth y* = {ys[0,0]:+.4f}   |   predicted y = {yps[0,0]:+.4f}")

    # --- 6e. self-tests -------------------------------------------------
    print("\n[5] SELF-TESTS")
    checks = []

    checks.append(("gradient check < 1e-4", ok_grad))

    start_val = hist[0][2]; end_val = hist[-1][2]
    learned = end_val < 0.5 * start_val
    checks.append((f"training reduced val MSE (>=2x): {start_val:.4f}->{end_val:.4f}",
                   learned))

    beats_mlp = pan_mse <= mlp_mse + 1e-4
    checks.append((f"PAN <= MLP on val MSE ({pan_mse:.4f} vs {mlp_mse:.4f})", beats_mlp))

    routes_well = pan_ont > 0.70
    checks.append((f"tour mass lands on named provinces ({pan_ont:.3f}>0.70)",
                   routes_well))

    # the soul really is 'little': its norm stays modest after training
    soul_norm = float(np.linalg.norm(model.P["s0"]))
    little_soul = soul_norm < 3.0
    checks.append((f"animula stays small (||s0||={soul_norm:.3f}<3.0)", little_soul))

    # forward is deterministic
    y1, _ = model.forward(Xv); y2, _ = model.forward(Xv)
    deterministic = np.allclose(y1, y2)
    checks.append(("forward pass is deterministic", deterministic))

    # NO-OVERWRITE guarantee: a pure read of a province leaves the atlas itself
    # unchanged (the map does not delete the land it models). We verify the
    # forward pass never mutates the stored atlas A.
    A_before = model.A_val.copy()
    _ = model.forward(Xv)
    no_overwrite = np.array_equal(A_before, model.A_val)
    checks.append(("no-overwrite: reading the land never edits it", no_overwrite))

    all_ok = True
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok = all_ok and ok

    print("\n" + "=" * 74)
    print(f"RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("The mind that governed by going and looking, built here as a machine")
    print("that travels its own map, rebuilds each place it visits to prove it")
    print("knows it, and carries the whole journey in one small wandering soul.")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
================================================================================
Chapter 135 - Galen of Pergamon (129 - c. 216 CE)
 THE PNEUMATIC ENGINE  —  a cognitive architecture after Galen of Pergamon
                          (129 - c. 216 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 135: Galen of Pergamon (129 - c. 216 CE)
================================================================================   

WHY THIS ARCHITECTURE LOOKS THE WAY IT DOES
-------------------------------------------
Galen's philosophy of mind is usually summarised with the medieval "cell
doctrine": imagination in the front ventricle, reason in the middle, memory in
the rear. Modern scholarship (Rocca, *Galen on the Brain*, 2003) shows that
this rigid mapping is NOT Galen's own -- it was bolted on later by Nemesius of
Emesa (~390 CE) and Augustine. Galen himself believed something subtler and,
for our purposes, far more interesting. Three commitments define his real view,
and each becomes a concrete mechanism below:

  (1) PNEUMA IS A CONSERVED, REFINABLE CURRENT.
      "Psychic pneuma" -- the soul's *first instrument* -- is a vital breath
      progressively refined from blood and air (natural -> vital -> psychic).
      It is not created at each stage; it is transformed and *routed*. We model
      information as a quantity that is SPLIT between channels, never
      duplicated: a unit's activation sent toward memory is subtracted from the
      unit's activation sent toward reason. Total pneuma is conserved.
      -> class PneumaSplit  (a conservation-constrained gate)

  (2) FUNCTION IS WHAT FALLS SILENT WHEN YOU SEVER A CONDUIT.
      Galen's signature method was causal, not correlational. Before the elders
      of Rome he ligated the recurrent laryngeal ("reversivi") nerves of a live
      pig; the squealing stopped. Cut the cord high, lose the arms; cut it low,
      lose the legs. Function is read off by ABLATION -- by what disappears.
      We build that epistemology INTO the model's verification: the self-test
      severs each channel in turn and measures which capacity collapses.
      -> ablate() + the ablation study in __main__

  (3) HEALTH IS KRASIS -- THE BALANCE OF OPPOSED QUALITIES.
      For Galen the body (and its temperament) is governed by the mixture of
      hot/cold and wet/dry. Well-being is dynamic equilibrium; disease is
      imbalance. We give the network two opposed control axes and a homeostatic
      penalty that pulls their mixture back toward balance while still letting
      them steer computation.
      -> class KrasisRegulator  +  the homeostatic loss term

The ventricles appear too, but faithfully: NOT as boxes that "contain" a
faculty, but as a RESERVOIR that buffers and regulates outflow. The posterior
reservoir (memory) is queried; its readout regulates how much stored experience
flows into the reasoning chamber -- exactly Galen's picture of the rear
ventricle governing flow into the nerves.

CONVENTIONS
-----------
* Pure NumPy, built from scratch. No autograd, no deep-learning framework.
* Every gradient is derived by hand and then verified against finite
  differences (mandatory check below; it must pass before anything ships).
* A real training loop on a synthetic "diagnosis" task, plus self-tests and a
  causal ablation study that reproduces Galen's own experimental logic.

Run:  python3 chapter_0135_galen_129.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(129)   # seeded on Galen's birth year, for reproducibility


# =============================================================================
# 0.  SMALL DIFFERENTIABLE PRIMITIVES
#     Each returns what the forward pass needs and, where useful, a cache for
#     the backward pass. Keeping these tiny makes the hand-derived gradients
#     auditable.
# =============================================================================

def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    # derivative of relu wrt its pre-activation
    return (z > 0.0).astype(z.dtype)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# =============================================================================
# 1.  THE PNEUMATIC ENGINE
#     One class holds all parameters and implements forward, backward, loss,
#     and -- crucially -- ablation. The four anatomical chambers are named so
#     the code reads like the anatomy:
#
#         X  --(anterior encode: PHANTASIA)-->  A
#         A  --(PneumaSplit, conserved)------>  to_memory  +  to_reason
#         to_memory --(posterior RESERVOIR)-->  recalled experience
#         KrasisRegulator(A) ---------------->  hot/cold, wet/dry gates
#         to_reason (+ gated memory) --(middle chamber: LOGOS)--> judgement
#         judgement --(output)--------------->  logits over conditions
# =============================================================================

class PneumaticEngine:
    def __init__(self, d_in, d_hidden, n_reservoir, n_classes, lam_krasis=0.05, seed=129):
        rng = np.random.default_rng(seed)
        h, di, nc, nm = d_hidden, d_in, n_classes, n_reservoir
        self.d_in, self.d_hidden, self.n_reservoir, self.n_classes = di, h, nm, nc
        self.lam_krasis = lam_krasis

        def scaled(shape, fan_in):
            # small He-ish init; determinism from the seeded rng
            return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

        # --- Anterior chamber: turns raw sense-data into "images" (phantasia) --
        self.W_ant = scaled((h, di), di); self.b_ant = np.zeros(h)

        # --- PneumaSplit: per-unit conserved gate deciding memory vs reason ----
        self.W_flow = scaled((h, h), h);  self.b_flow = np.zeros(h)

        # --- Posterior reservoir (memory): keys addressed, values recalled -----
        self.M_keys = scaled((nm, h), h)
        self.M_vals = scaled((nm, h), h)

        # --- KrasisRegulator: two opposed axes (temperature t, humidity q) -----
        self.W_kr = scaled((2, h), h);    self.b_kr = np.zeros(2)

        # --- Middle chamber: the reasoning that combines sense + recalled -------
        self.W_mid = scaled((h, h), h);   self.b_mid = np.zeros(h)

        # --- Output: judgement -> logits over conditions -----------------------
        self.W_out = scaled((nc, h), h);  self.b_out = np.zeros(nc)

    # -- ordered view of parameters, used by the optimiser and the grad check --
    def params(self):
        return {
            "W_ant": self.W_ant, "b_ant": self.b_ant,
            "W_flow": self.W_flow, "b_flow": self.b_flow,
            "M_keys": self.M_keys, "M_vals": self.M_vals,
            "W_kr": self.W_kr, "b_kr": self.b_kr,
            "W_mid": self.W_mid, "b_mid": self.b_mid,
            "W_out": self.W_out, "b_out": self.b_out,
        }

    # -------------------------------------------------------------------------
    # FORWARD
    #   `sever` implements Galen's scalpel. It forces a channel's flow to a
    #   fixed value so we can watch which capacity dies:
    #     sever="reason"  -> alpha:=1 : all pneuma to memory, reason starved
    #     sever="memory"  -> alpha:=0 : all pneuma to reason, reservoir mute
    #     sever="krasis"  -> gates:=1 : humoral regulation removed
    #   The same forward serves training (sever=None) and the ablation study.
    # -------------------------------------------------------------------------
    def forward(self, X, sever=None):
        c = {}                                            # cache for backprop
        c["X"] = X

        # 1. Anterior: encode sense-data into images (phantasia)
        A_pre = X @ self.W_ant.T + self.b_ant
        A = relu(A_pre)
        c["A_pre"], c["A"] = A_pre, A

        # 2. PneumaSplit: conserved routing.  alpha in (0,1) per unit.
        #    to_memory = alpha * A ;  to_reason = (1-alpha) * A  =>  sum == A
        flow_logit = A @ self.W_flow.T + self.b_flow
        alpha = sigmoid(flow_logit)
        if sever == "reason":  alpha = np.ones_like(alpha)   # cut reason conduit
        if sever == "memory":  alpha = np.zeros_like(alpha)  # cut memory conduit
        to_mem = alpha * A
        to_rea = (1.0 - alpha) * A
        c["flow_logit"], c["alpha"] = flow_logit, alpha
        c["to_mem"], c["to_rea"] = to_mem, to_rea

        # 3. Posterior reservoir: address keys with the memory-bound pneuma,
        #    recall a blend of stored values.
        scores = to_mem @ self.M_keys.T                    # (B, n_reservoir)
        w = softmax(scores, axis=1)
        r_mem = w @ self.M_vals                             # (B, h)
        c["scores"], c["w"], c["r_mem"] = scores, w, r_mem

        # 4. KrasisRegulator: two opposed axes, each a tanh in (-1, 1).
        #    Gates gt, gq in (0,1) steer how much recalled experience and how
        #    much overall gain enter reasoning.
        kr_pre = A @ self.W_kr.T + self.b_kr               # (B, 2)
        t = np.tanh(kr_pre[:, 0]); q = np.tanh(kr_pre[:, 1])
        gt = 0.5 * (1.0 + t)       # temperature gate: memory admixture
        gq = 0.5 * (1.0 + q)       # humidity gate: reasoning gain
        if sever == "krasis":
            gt = np.ones_like(gt); gq = np.ones_like(gq)   # remove regulation
        c["kr_pre"], c["t"], c["q"], c["gt"], c["gq"] = kr_pre, t, q, gt, gq

        # 5. Middle chamber: reason over (sensory pneuma + gated recalled),
        #    the whole scaled by the humidity gate.
        inner = to_rea + gt[:, None] * r_mem
        reason_in = gq[:, None] * inner
        R_pre = reason_in @ self.W_mid.T + self.b_mid
        R = relu(R_pre)
        c["inner"], c["reason_in"], c["R_pre"], c["R"] = inner, reason_in, R_pre, R

        # 6. Output: judgement -> logits
        Z = R @ self.W_out.T + self.b_out                  # (B, n_classes)
        c["Z"] = Z
        return Z, c

    # -------------------------------------------------------------------------
    # LOSS = cross-entropy  +  lam * krasis-imbalance (homeostatic penalty)
    #   The imbalance term is the mean of (t^2 + q^2): balance means the opposed
    #   qualities cancel. This is Galen's krasis expressed as a regulariser.
    # -------------------------------------------------------------------------
    def loss(self, Z, y, cache):
        B = Z.shape[0]
        P = softmax(Z, axis=1)
        logp = np.log(P[np.arange(B), y] + 1e-12)
        ce = -logp.mean()
        imbalance = self.lam_krasis * (cache["t"] ** 2 + cache["q"] ** 2).mean()
        return ce + imbalance, P

    # -------------------------------------------------------------------------
    # BACKWARD  (all gradients derived by hand; verified below)
    # -------------------------------------------------------------------------
    def backward(self, cache, P, y):
        B = P.shape[0]
        g = {}

        # ---- output layer ----
        onehot = np.zeros_like(P); onehot[np.arange(B), y] = 1.0
        dZ = (P - onehot) / B                              # (B, C)
        g["W_out"] = dZ.T @ cache["R"]
        g["b_out"] = dZ.sum(0)
        dR = dZ @ self.W_out                               # (B, h)

        # ---- middle chamber ----
        dR_pre = dR * drelu(cache["R_pre"])
        g["W_mid"] = dR_pre.T @ cache["reason_in"]
        g["b_mid"] = dR_pre.sum(0)
        d_reason_in = dR_pre @ self.W_mid                  # (B, h)

        # reason_in = gq * inner
        d_gq = (d_reason_in * cache["inner"]).sum(1)       # (B,)
        d_inner = d_reason_in * cache["gq"][:, None]       # (B, h)

        # inner = to_rea + gt * r_mem
        d_to_rea = d_inner
        d_gt = (d_inner * cache["r_mem"]).sum(1)           # (B,)
        d_r_mem = d_inner * cache["gt"][:, None]           # (B, h)

        # ---- posterior reservoir (memory) ----
        g["M_vals"] = cache["w"].T @ d_r_mem
        d_w = d_r_mem @ self.M_vals.T                      # (B, n_reservoir)
        # softmax jacobian
        d_scores = cache["w"] * (d_w - (d_w * cache["w"]).sum(1, keepdims=True))
        g["M_keys"] = d_scores.T @ cache["to_mem"]
        d_to_mem = d_scores @ self.M_keys                  # (B, h)

        # ---- KrasisRegulator ----
        # gates feed loss through gt, gq AND through the homeostatic penalty.
        dt_from_gate = d_gt * 0.5
        dq_from_gate = d_gq * 0.5
        dt = dt_from_gate + self.lam_krasis * 2.0 * cache["t"] / B
        dq = dq_from_gate + self.lam_krasis * 2.0 * cache["q"] / B
        d_kr0 = dt * (1.0 - cache["t"] ** 2)               # through tanh
        d_kr1 = dq * (1.0 - cache["q"] ** 2)
        d_kr_pre = np.stack([d_kr0, d_kr1], axis=1)        # (B, 2)
        g["W_kr"] = d_kr_pre.T @ cache["A"]
        g["b_kr"] = d_kr_pre.sum(0)
        d_A_from_kr = d_kr_pre @ self.W_kr                 # (B, h)

        # ---- PneumaSplit: recombine channel gradients ----
        # to_mem = alpha*A ; to_rea = (1-alpha)*A
        d_alpha = d_to_mem * cache["A"] + d_to_rea * (-cache["A"])
        d_A_from_channels = d_to_mem * cache["alpha"] + d_to_rea * (1.0 - cache["alpha"])
        d_flow_logit = d_alpha * cache["alpha"] * (1.0 - cache["alpha"])   # through sigmoid
        g["W_flow"] = d_flow_logit.T @ cache["A"]
        g["b_flow"] = d_flow_logit.sum(0)
        d_A_from_flow = d_flow_logit @ self.W_flow

        # ---- anterior chamber ----
        d_A = d_A_from_channels + d_A_from_flow + d_A_from_kr
        d_A_pre = d_A * drelu(cache["A_pre"])
        g["W_ant"] = d_A_pre.T @ cache["X"]
        g["b_ant"] = d_A_pre.sum(0)

        return g

    # convenience: forward + loss + backward in one call (training path only)
    def loss_and_grads(self, X, y):
        Z, cache = self.forward(X, sever=None)
        L, P = self.loss(Z, y, cache)
        grads = self.backward(cache, P, y)
        return L, grads

    # -------------------------------------------------------------------------
    # ABLATION  — Galen's scalpel as an evaluation method.
    #   Returns accuracy with a given conduit severed.
    # -------------------------------------------------------------------------
    def ablate(self, X, y, sever):
        Z, _ = self.forward(X, sever=sever)
        return (Z.argmax(1) == y).mean()


# =============================================================================
# 2.  SYNTHETIC "DIAGNOSIS" TASK
#     Each condition is a prototype in feature space. A patient is a noisy
#     realisation of one condition. The engine must recognise the condition.
#     Because both the sensory pathway and the recalled-prototype pathway carry
#     class-relevant signal, severing EITHER conduit should measurably hurt --
#     which is exactly the point of the ablation study.
# =============================================================================

def make_task(n, d_in, n_classes, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    prototypes = rng.standard_normal((n_classes, d_in)) * 1.5
    y = rng.integers(0, n_classes, size=n)
    X = prototypes[y] + rng.standard_normal((n, d_in)) * noise
    return X.astype(np.float64), y.astype(np.int64), prototypes


# =============================================================================
# 3.  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory; must pass)
#     Compares each analytic gradient against a central-difference estimate on
#     a handful of random coordinates per parameter tensor.
# =============================================================================

def gradient_check(model, X, y, n_probe=6, eps=1e-6):
    # local RNG so the check does not disturb the training RNG stream
    rng = np.random.default_rng(7)
    _, analytic = model.loss_and_grads(X, y)
    worst = 0.0
    per_param = {}
    for name, P in model.params().items():
        flat = P.reshape(-1)
        gflat = analytic[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for i in idxs:
            orig = flat[i]
            # central difference: L(+eps) and L(-eps) with everything else fixed
            flat[i] = orig + eps
            Zp, cp = model.forward(X); Lp, _ = model.loss(Zp, y, cp)
            flat[i] = orig - eps
            Zm, cm = model.forward(X); Lm, _ = model.loss(Zm, y, cm)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-12, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        per_param[name] = max(errs)
        worst = max(worst, per_param[name])
    return worst, per_param


# =============================================================================
# 4.  TRAINING LOOP  (plain SGD with momentum; pure NumPy)
# =============================================================================

def train(model, X, y, Xval, yval, epochs=60, lr=0.15, batch=64, mom=0.9):
    vel = {k: np.zeros_like(v) for k, v in model.params().items()}
    n = X.shape[0]
    hist = []
    for ep in range(epochs):
        order = RNG.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            L, grads = model.loss_and_grads(X[idx], y[idx])
            for k, P in model.params().items():
                vel[k] = mom * vel[k] - lr * grads[k]
                P += vel[k]
        # epoch diagnostics
        Ztr, ctr = model.forward(X); Ltr, Ptr = model.loss(Ztr, y, ctr)
        acc_tr = (Ztr.argmax(1) == y).mean()
        Zva, _ = model.forward(Xval)
        acc_va = (Zva.argmax(1) == yval).mean()
        krasis = float(np.sqrt((ctr["t"] ** 2 + ctr["q"] ** 2).mean()))
        hist.append((ep, Ltr, acc_tr, acc_va, krasis))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d} | loss {Ltr:6.4f} | train_acc {acc_tr:5.3f} "
                  f"| val_acc {acc_va:5.3f} | krasis|t,q| {krasis:5.3f}")
    return hist


# =============================================================================
# 5.  MAIN: build, check gradients, train, self-test, ablate.
# =============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 74)
    print("THE PNEUMATIC ENGINE  —  a cognitive architecture after Galen (129-216)")
    print("=" * 74)

    D_IN, D_HID, N_RES, N_CLS = 12, 24, 8, 4
    NOISE = 1.0   # noisy enough that the reservoir's denoising genuinely helps

    # ---- data: one coherent set of condition-prototypes, realised twice ------
    _, _, protos = make_task(1, D_IN, N_CLS, noise=NOISE, seed=1)   # prototypes only
    def realise(n, seed):
        rng = np.random.default_rng(seed)
        yy = rng.integers(0, N_CLS, size=n)
        XX = protos[yy] + rng.standard_normal((n, D_IN)) * NOISE
        return XX.astype(np.float64), yy.astype(np.int64)
    Xtr, ytr = realise(1500, 10)
    Xva, yva = realise(500, 20)

    model = PneumaticEngine(D_IN, D_HID, N_RES, N_CLS, lam_krasis=0.05, seed=129)

    # ---- (1) gradient check FIRST: nothing proceeds until this passes --------
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    worst, per = gradient_check(model, Xtr[:8], ytr[:8], n_probe=6)
    for k, v in per.items():
        print(f"    {k:8s} max-rel-err = {v:.2e}")
    print(f"    ---> worst relative error across all params = {worst:.2e}")
    assert worst < 1e-4, "GRADIENT CHECK FAILED"
    print("    GRADIENT CHECK PASSED (worst rel-err < 1e-4)")

    # ---- (2) train -----------------------------------------------------------
    print("\n[2] TRAINING (conserved-pneuma routing, krasis-regularised)")
    hist = train(model, Xtr, ytr, Xva, yva, epochs=60, lr=0.15, batch=64)
    final_val = hist[-1][3]
    print(f"    final validation accuracy = {final_val:5.3f}")
    assert final_val > 0.80, "training did not reach a competent accuracy"
    print("    TRAINING SELF-TEST PASSED (val_acc > 0.80)")

    # ---- (3) conservation self-test -----------------------------------------
    print("\n[3] PNEUMA-CONSERVATION SELF-TEST")
    Z, c = model.forward(Xtr[:32])
    residual = np.abs(c["to_mem"] + c["to_rea"] - c["A"]).max()
    print(f"    max |to_memory + to_reason - anterior| = {residual:.2e}")
    assert residual < 1e-12, "pneuma was not conserved across the split"
    print("    CONSERVATION SELF-TEST PASSED (flow is split, never duplicated)")

    # ---- (4) ablation study: Galen's scalpel --------------------------------
    print("\n[4] ABLATION STUDY  (sever a conduit; measure what falls silent)")
    intact  = model.ablate(Xva, yva, sever=None)
    no_rea  = model.ablate(Xva, yva, sever="reason")   # cut reasoning conduit
    no_mem  = model.ablate(Xva, yva, sever="memory")   # cut memory conduit
    no_krasis = model.ablate(Xva, yva, sever="krasis") # remove regulation
    chance = 1.0 / N_CLS
    _, cdiag = model.forward(Xva)
    alpha_mean = float(cdiag["alpha"].mean())   # >0.5 = leans on memory conduit
    print(f"    intact ................. val_acc = {intact:5.3f}")
    print(f"    reason conduit severed . val_acc = {no_rea:5.3f}   (drop {intact-no_rea:+5.3f})  [chance={chance:.2f}]")
    print(f"    memory conduit severed . val_acc = {no_mem:5.3f}   (drop {intact-no_mem:+5.3f})")
    print(f"    krasis removed ......... val_acc = {no_krasis:5.3f}   (drop {intact-no_krasis:+5.3f})")
    print(f"    (learned routing: mean alpha = {alpha_mean:.3f} of pneuma sent to the reservoir)")

    # CLAIM 1 — the reasoning conduit is the "recurrent nerve": the decision
    # physically flows through it, so severing it silences the system (~chance).
    assert no_rea < intact - 0.30, "severing reason should collapse performance toward chance"
    # CLAIM 2 — the posterior reservoir is a real, localized denoiser here:
    # severing it costs a large, reproducible chunk of accuracy.
    assert no_mem < intact - 0.05, "severing the memory reservoir should measurably hurt"
    # CLAIM 3 — krasis is a REGULATOR, not the seat of a faculty: removing it
    # perturbs balance only marginally (either sign, within task granularity),
    # never localizes a single capacity. This is the faithful Galenic point.
    assert abs(intact - no_krasis) < 0.05, "krasis removal should have only a small, non-localizing effect"
    print("    ABLATION SELF-TEST PASSED:")
    print("    - reason conduit is load-bearing: severing it drops accuracy to ~chance")
    print("      (Galen's squealing pig: cut the channel that carries the function, lose it)")
    print("    - posterior reservoir is a localized denoiser: severing it costs real accuracy")
    print("    - krasis is a regulator, not a faculty: removing it barely moves accuracy —")
    print("      faithful to Galen placing the faculties in the brain's substance and treating")
    print("      the ventricles as reservoirs of flow, not as boxes that 'contain' a power")

    print("\n" + "=" * 74)
    print("ALL CHECKS PASSED — the pneumatic engine trains, conserves, and localizes.")
    print("=" * 74)

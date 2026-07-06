#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0096_patanjali_-200.py
 The Vṛtti–Nirodha Network (VNN)
 A from-scratch, trainable cognitive architecture in the spirit of Patañjali
 # Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0096 · Patanjali
================================================================================

WHO THIS ENCODES
----------------
Patañjali, compiler of the Yoga Sūtras (the text now dated by Philipp Maas to
~400 CE, though the grammarian of the same name is firmly 2nd c. BCE), defined
yoga in a single line: "yogaś citta-vṛtti-nirodhaḥ" — yoga is the stilling of
the fluctuations (vṛtti) of the mind-stuff (citta). Crucially he did NOT say the
mind should be made blank or deleted. He said its self-generated turbulence
should be progressively quieted until the citta becomes so transparent that the
seer (puruṣa / draṣṭṛ) is reflected in it without distortion (YS 1.3).

So the central engineering idea here is unusual and specific to him:
    The training objective is NOT to maximize a task reward.
    The objective is to MINIMIZE the amplitude of the system's own internal
    fluctuations *while still maintaining veridical contact with the object*
    (pramāṇa, valid cognition). Stillness that loses the object is just sleep
    (nidrā); accurate turbulence is bondage. Yoga is the narrow path between.

WHY THIS IS NOT A TRANSFORMER (and not Buddha's architecture either)
--------------------------------------------------------------------
- No attention over stored keys, no softmax mixture, no token stream.
- The unit of computation is the *vṛtti*: a bounded modification the mind makes
  to itself at each inner step. We literally penalize the energy of these
  modifications. That penalty IS the loss.
- Every vṛtti deposits a latent impression (saṃskāra) into a slow memory trace
  that biases the next vṛtti. This is the karmic groove: the mind tends to
  fluctuate the way it has fluctuated before.
- Two named control levers, straight from YS 1.12 ("abhyāsa-vairāgyābhyāṃ
  tannirodhaḥ" — restraint comes through practice and dispassion):
      * abhyāsa  (practice): a curriculum that raises the nirodha pressure over
        training epochs. The mind is asked to settle a little more each session.
      * vairāgya (dispassion): a decay applied to the "coloring" weights that
        govern how strongly vṛttis deposit afflicted impressions. Letting go of
        grasping literally shrinks the saṃskāra-forming gain.
- A diagnostic of ekāgratā (one-pointedness): how concentrated the final state
  is along a single direction — the measurable face of dhāraṇā → dhyāna.

This is the opposite emphasis from the Buddha chapter (#39), whose flux is about
anattā / no-self and seeing-through. Here the self (puruṣa) is real and the goal
is to recover it by quieting an instrument, not to dissolve the witness.

WHAT THE FILE DOES (run it: `python3 0096_Patanjali_Neuron.py`)
---------------------------------------------------------------
1. Implements the VNN forward pass (the "sitting": T inner steps of vṛtti).
2. Implements exact backprop-through-time (BPTT) for every parameter.
3. Runs a finite-difference gradient check on all parameters (MANDATORY; must
   pass with relative error < 1e-5).
4. Trains the network with the abhyāsa schedule + vairāgya decay and shows the
   nirodha (fluctuation) energy fall while pramāṇa fidelity is preserved.
5. Self-tests: gradient check passes; fluctuation energy drops over training;
   within a single sitting, late-step vṛtti energy < early-step (it settles);
   ekāgratā rises. All asserted.

Pure NumPy. No frameworks. No GPU. Deterministic seed.
================================================================================
"""

import numpy as np

np.random.seed(96)  # figure #96


# ============================================================================
# SECTION 1 — THE CITTA FIELD: PARAMETERS
# ============================================================================
# The citta is a d-dimensional field. At each inner step it produces a vṛtti
# (a bounded modification, via tanh) shaped by three influences:
#   - W_s : how the current state feeds the next modification (self-stirring)
#   - W_x : how the object/seed (pratyaya) drives modification (sensory contact)
#   - W_m : how the saṃskāra memory biases modification (karmic groove)
# Readout:
#   - W_o, b_o : the draṣṭṛ (seer) reading the object as reflected in the
#     stilled citta — this is the pramāṇa (valid-cognition) channel.
# Coloring:
#   - k : per-channel gain governing how strongly a vṛtti deposits an afflicted
#     impression into memory. vairāgya decays this.
# ----------------------------------------------------------------------------

class VNNConfig:
    def __init__(self, d=16, T=8, alpha=0.5, lam=0.6):
        self.d = d            # citta field dimension
        self.T = T            # inner steps per sitting (dhāraṇā depth)
        self.alpha = alpha    # how much each vṛtti moves the state (update gain)
        self.lam = lam        # saṃskāra memory leak (how persistent impressions are)


def init_params(cfg, scale=0.3):
    d = cfg.d
    rng = np.random.RandomState(96)
    P = {
        "W_s": rng.randn(d, d) * scale / np.sqrt(d),
        "W_x": rng.randn(d, d) * scale / np.sqrt(d),
        "W_m": rng.randn(d, d) * scale / np.sqrt(d),
        "b":   np.zeros(d),
        "W_o": rng.randn(d, d) * scale / np.sqrt(d),
        "b_o": np.zeros(d),
        # coloring starts near 1: vṛttis initially deposit strong impressions
        # (a restless, attached mind). vairāgya will erode this during training.
        "k":   np.ones(d) * 0.8,
    }
    return P


# ============================================================================
# SECTION 2 — THE SITTING: FORWARD PASS
# ============================================================================
# One "sitting" presents a fixed object x (the meditation seed) and runs T inner
# steps. We weight the nirodha penalty by a ramp w_t = t/T: early stirring is
# tolerated, late stirring is punished. That ramp is "progressive quieting"
# inside a single sitting — the mind is allowed to engage the object, then is
# asked to let the waves die down.
# ----------------------------------------------------------------------------

def forward(P, cfg, x, beta):
    """
    Run one sitting.
      x    : object/seed vector, shape (d,)
      beta : current nirodha weight (set by the abhyāsa schedule)
    Returns loss components and a cache for backprop.
    """
    d, T, alpha, lam = cfg.d, cfg.T, cfg.alpha, cfg.lam

    s = np.zeros(d)          # citta state s_0 (a settled mind begins empty)
    m = np.zeros(d)          # saṃskāra memory m_0
    cache = {"s": [s.copy()], "m": [m.copy()], "v": [], "pre": [], "x": x}

    L_nir = 0.0
    for t in range(1, T + 1):
        pre = P["W_s"] @ s + P["W_x"] @ x + P["W_m"] @ m + P["b"]
        v = np.tanh(pre)                       # the vṛtti (bounded modification)
        s = s + alpha * v                      # citta updated by the fluctuation
        m = lam * m + (1.0 - lam) * (P["k"] * v)  # deposit colored impression
        w_t = t / T                            # progressive-quieting ramp
        L_nir += w_t * 0.5 * np.sum(v * v)     # nirodha: penalize fluctuation
        cache["pre"].append(pre); cache["v"].append(v)
        cache["s"].append(s.copy()); cache["m"].append(m.copy())

    # Pramāṇa channel: the seer reads the object as reflected in the final citta.
    y = P["W_o"] @ s + P["b_o"]
    diff = y - x
    L_fid = 0.5 * np.sum(diff * diff)          # veridical contact must survive

    L = L_fid + beta * L_nir
    cache["y"] = y; cache["diff"] = diff; cache["beta"] = beta
    return L, L_fid, L_nir, cache


# ============================================================================
# SECTION 3 — BACKPROP-THROUGH-TIME (exact gradients)
# ============================================================================
# We propagate two adjoints backward in inner time: gs (dL/ds_t) and gm (dL/dm_t).
# The fidelity term seeds gs at the final step; the nirodha term injects a direct
# gradient into every vṛtti. See the chapter's Architecture section for the full
# derivation — every line below corresponds to one term of the chain rule.
# ----------------------------------------------------------------------------

def backward(P, cfg, cache):
    d, T, alpha, lam = cfg.d, cfg.T, cfg.alpha, cfg.lam
    x = cache["x"]; beta = cache["beta"]

    g = {k_: np.zeros_like(v_) for k_, v_ in P.items()}

    # --- readout / pramāṇa (fidelity) ---
    gy = cache["diff"]                     # dL_fid/dy = (y - x)
    g["W_o"] += np.outer(gy, cache["s"][T])
    g["b_o"] += gy
    gs = P["W_o"].T @ gy                   # dL/ds_T from fidelity
    gm = np.zeros(d)                       # m_T does not feed y

    # --- unroll backward through the sitting ---
    for t in range(T, 0, -1):
        v   = cache["v"][t - 1]
        pre = cache["pre"][t - 1]
        s_prev = cache["s"][t - 1]
        m_prev = cache["m"][t - 1]
        w_t = t / T

        # v_t receives gradient from: s_t (alpha path), m_t (coloring path),
        # and the direct nirodha penalty.
        gv = alpha * gs \
             + (1.0 - lam) * (P["k"] * gm) \
             + beta * w_t * v

        # back through tanh
        gpre = gv * (1.0 - v * v)

        # parameter gradients at this step
        g["W_s"] += np.outer(gpre, s_prev)
        g["W_x"] += np.outer(gpre, x)
        g["W_m"] += np.outer(gpre, m_prev)
        g["b"]   += gpre
        # k: from m_t = lam*m_{t-1} + (1-lam)*(k ⊙ v_t)
        g["k"]   += (1.0 - lam) * (gm * v)

        # propagate to previous states
        gs_prev = gs + P["W_s"].T @ gpre        # s_t = s_{t-1} + alpha*v_t
        gm_prev = lam * gm + P["W_m"].T @ gpre  # m_t = lam*m_{t-1} + ...
        gs, gm = gs_prev, gm_prev

    return g


# ============================================================================
# SECTION 4 — DIAGNOSTICS
# ============================================================================

def vritti_energies(cache):
    """Per-step fluctuation energy ||v_t||^2 — the 'turbulence trace' of a sitting."""
    return np.array([float(np.sum(v * v)) for v in cache["v"]])


def ekagrata(cache):
    """
    One-pointedness (ekāgratā), in [0,1]. For Patañjali, dhāraṇā is the citta
    fixed on a single point and held there *unwavering*. The measurable face of
    that fixity is steadiness: by the end of a sitting, the state should stop
    moving. We measure how much the late vṛttis have died down relative to the
    early ones:
        steadiness = early_drift / (early_drift + late_drift)
    A restless mind keeps stirring at the end (late ≈ early => ~0.5).
    A one-pointed mind has gone still at the end (late << early => ->1.0).
    This is a pure diagnostic (not in the loss), but it is *driven up* by the
    nirodha objective, which is exactly the point: stilling produces fixity.
    """
    e = np.array([float(np.sum(v * v)) for v in cache["v"]])
    if len(e) < 2:
        return 0.0
    early = float(np.mean(e[:max(1, len(e) // 4)]))
    late = float(np.mean(e[-max(1, len(e) // 4):]))
    return early / (early + late + 1e-12)


# ============================================================================
# SECTION 5 — GRADIENT CHECK (MANDATORY)
# ============================================================================

def grad_check(verbose=True):
    """Central finite-difference check of every parameter against BPTT."""
    cfg = VNNConfig(d=4, T=3, alpha=0.5, lam=0.6)
    P = init_params(cfg)
    # perturb k off 1.0 so its gradient is non-trivial
    P["k"] = np.array([0.7, 0.9, 0.5, 1.1])
    rng = np.random.RandomState(7)
    x = rng.randn(cfg.d)
    beta = 0.4

    def loss_of(P_):
        L, *_ = forward(P_, cfg, x, beta)
        return L

    _, _, _, cache = forward(P, cfg, x, beta)
    analytic = backward(P, cfg, cache)

    eps = 1e-6
    worst = 0.0
    for name in P:
        flat = P[name].ravel()
        ga = analytic[name].ravel()
        gn = np.zeros_like(flat)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps; Lp = loss_of(P)
            flat[i] = orig - eps; Lm = loss_of(P)
            flat[i] = orig
            gn[i] = (Lp - Lm) / (2 * eps)
        num = np.linalg.norm(ga - gn)
        den = np.linalg.norm(ga) + np.linalg.norm(gn) + 1e-12
        rel = num / den
        worst = max(worst, rel)
        if verbose:
            print(f"  grad_check  {name:5s}  rel_err = {rel:.2e}")
    if verbose:
        print(f"  worst relative error = {worst:.2e}")
    return worst


# ============================================================================
# SECTION 6 — DATA: OBJECTS FOR MEDITATION
# ============================================================================
# Each "object" is a structured low-rank vector — a stable thing the mind can
# rest on (a flame, a breath, a sound). Low-rank because a true object of
# dhāraṇā is simple; the work is not in the object but in holding it.
# ----------------------------------------------------------------------------

def make_objects(d, n, rng):
    # a few shared latent directions; each object is a sparse mix of them
    basis = rng.randn(4, d)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    objs = []
    for _ in range(n):
        coeff = rng.randn(4) * (rng.rand(4) < 0.6)  # sparse
        x = coeff @ basis
        n_ = np.linalg.norm(x)
        if n_ > 1e-6:
            x = x / n_                                # unit object
        objs.append(x)
    return objs


# ============================================================================
# SECTION 7 — TRAINING (abhyāsa + vairāgya)
# ============================================================================

def train(cfg, epochs=400, lr=0.05, beta_max=1.5, vairagya_rho=0.01, verbose=True):
    P = init_params(cfg)
    rng = np.random.RandomState(321)
    objects = make_objects(cfg.d, 24, rng)

    history = {"epoch": [], "L": [], "L_fid": [], "L_nir": [], "ekagrata": []}

    for ep in range(epochs):
        # --- abhyāsa: the nirodha pressure deepens with practice (warm-up) ---
        beta = beta_max * min(1.0, ep / (0.5 * epochs))

        # accumulate grads over the whole "session" (all objects)
        gacc = {k_: np.zeros_like(v_) for k_, v_ in P.items()}
        L_tot = L_fid_tot = L_nir_tot = ek_tot = 0.0
        for x in objects:
            L, L_fid, L_nir, cache = forward(P, cfg, x, beta)
            g = backward(P, cfg, cache)
            for k_ in gacc:
                gacc[k_] += g[k_]
            L_tot += L; L_fid_tot += L_fid; L_nir_tot += L_nir
            ek_tot += ekagrata(cache)
        nobj = len(objects)

        # --- gradient step ---
        for k_ in P:
            P[k_] -= lr * gacc[k_] / nobj

        # --- vairāgya: dispassion decays the impression-coloring gain k ---
        # Letting go of grasping shrinks how strongly vṛttis stain the memory.
        P["k"] *= (1.0 - vairagya_rho)
        P["k"] = np.clip(P["k"], 0.0, None)

        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d} | beta={beta:.2f} | L={L_tot/nobj:.4f} "
                  f"| fid={L_fid_tot/nobj:.4f} | nir={L_nir_tot/nobj:.4f} "
                  f"| ekagrata={ek_tot/nobj:.3f} | mean|k|={np.mean(P['k']):.3f}")

        history["epoch"].append(ep)
        history["L"].append(L_tot / nobj)
        history["L_fid"].append(L_fid_tot / nobj)
        history["L_nir"].append(L_nir_tot / nobj)
        history["ekagrata"].append(ek_tot / nobj)

    return P, history, objects


# ============================================================================
# SECTION 8 — SELF-TESTS
# ============================================================================

def run_selftests():
    print("=" * 70)
    print("  VṚTTI–NIRODHA NETWORK  —  Patañjali (figure 96)")
    print("  'yogaś citta-vṛtti-nirodhaḥ' — yoga is the stilling of fluctuations")
    print("=" * 70)

    # ---- Test 1: gradient check ----
    print("\n[1] Finite-difference gradient check (must be < 1e-5):")
    worst = grad_check(verbose=True)
    assert worst < 1e-5, f"GRADIENT CHECK FAILED: worst rel err {worst:.2e}"
    print("    PASS — BPTT gradients match numerical gradients.")

    # ---- Test 2: training settles the mind ----
    print("\n[2] Training (abhyāsa schedule + vairāgya decay):")
    cfg = VNNConfig(d=16, T=8, alpha=0.5, lam=0.6)
    P, hist, objects = train(cfg, epochs=400, verbose=True)

    nir_first = np.mean(hist["L_nir"][:20])
    nir_last  = np.mean(hist["L_nir"][-20:])
    fid_last  = np.mean(hist["L_fid"][-20:])
    ek_first  = np.mean(hist["ekagrata"][:20])
    ek_last   = np.mean(hist["ekagrata"][-20:])
    print(f"\n    nirodha energy: {nir_first:.4f} -> {nir_last:.4f} "
          f"(reduced {100*(1-nir_last/max(nir_first,1e-9)):.1f}%)")
    print(f"    final fidelity loss: {fid_last:.4f} (object still cognized)")
    print(f"    ekāgratā (one-pointedness): {ek_first:.3f} -> {ek_last:.3f}")

    assert nir_last < nir_first, "Fluctuation energy did not fall — no nirodha."
    assert fid_last < 0.25, "Pramāṇa lost — the mind went blank, not still."
    assert ek_last >= ek_first - 1e-6, "One-pointedness did not improve."
    print("    PASS — fluctuations quieted while veridical contact survived.")

    # ---- Test 3: within a sitting, the mind settles (late < early) ----
    print("\n[3] Within-sitting stilling (late-step vṛtti < early-step):")
    x = objects[0]
    _, _, _, cache = forward(P, cfg, x, beta=hist["L"][-1] * 0 + 1.5)
    e = vritti_energies(cache)
    early = float(np.mean(e[:2])); late = float(np.mean(e[-2:]))
    print("    per-step vṛtti energy:", np.array2string(e, precision=4))
    print(f"    early(mean first 2)={early:.4f}  late(mean last 2)={late:.4f}")
    assert late < early, "Sitting did not settle — late stirring >= early."
    print("    PASS — the sitting settles: the waves die down toward the end.")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED.")
    print("  The architecture embodies citta-vṛtti-nirodha: it learns to make")
    print("  fewer and smaller self-modifications while keeping the object in")
    print("  view — a trained, focused mind rather than a raw or empty one.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_selftests()

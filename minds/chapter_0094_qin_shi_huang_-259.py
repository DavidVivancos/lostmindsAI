#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
chapter_0094_qin_shi_huang_-259.py  —  THE CANONICAL CODEX NETWORK (CCN)
The cognitive architecture of Qin Shi Huang, First Emperor of unified China
(259-210 BCE), rendered as a trainable, from-scratch NumPy model.
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0094 · Qin Shi Huang
================================================================================

WHY THIS ARCHITECTURE (the mind-specific thesis)
------------------------------------------------
The First Emperor is usually filed under "Legalism = strict order." That is the
doctrine of his *ministers* (Shang Yang, Han Feizi, Li Si), not the cognitive
signature that is his alone. His own recorded deeds point to one obsession:
the COLLAPSE OF PLURAL ENCODINGS INTO A SINGLE CANONICAL ONE, enforced across a
continent, and made to OUTLIVE ITS AUTHOR.

He standardised the script, the weights, the measures, the coinage, the width
of cart axles, and the law itself. He burned the books that carried rival
world-models (213 BCE). He buried an 8,000-figure army and a mercury-river map
so that his rule could *resume after his own death* (210 BCE).

Translated into machine learning, that is:
  (1) a shared CODEBOOK of canonical glyphs — one script for the empire;
  (2) many SOURCE ENCODERS — the conquered states, each speaking a dialect;
  (3) a COMMITMENT penalty — Legalist enforcement that snaps every local
      encoder onto the canon;
  (4) an INTEROPERABILITY objective — the same idea, written in any province,
      must compress to the SAME canonical glyph, so any edict is legible
      everywhere; and
  (5) PERSISTENCE — the codebook is a separable memory that can survive the
      destruction and re-initialisation of every encoder/decoder. The
      administrators die; the standard does not. This is the terracotta army
      made executable: the policy resumes after the substrate is wiped.

This is a multi-source soft-vector-quantised autoencoder. The codebook is read
by a differentiable SOFTMAX over canonical glyphs (temperature tau): with high
tau the script is loose; as tau falls the empire converges on ONE glyph per
idea. Soft quantisation keeps the whole objective smooth, so the hand-derived
backprop can be verified exactly by finite differences (mandatory in this
corpus). It is NOT a Transformer, MoE, or attention-over-stored-keys — the
mechanism is quantisation-as-standardisation, the Emperor's signature alone.

E-AGI Barometer mapping
-----------------------
  standardization score rising -> World Modeling (legibility of the realm)
  interop loss falling         -> Language Understanding (cross-state reading)
  persistence test passing     -> Autonomy (self-perpetuation; war on death)

Run:  python3 chapter_0094_qin_shi_huang_-259.py
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(259)          # seed = the Emperor's birth year, 259 BCE
TAU = 0.7                                  # softmax temperature = looseness of script


# ============================================================================
# 0. SYNTHETIC EMPIRE: concepts rendered in several "state dialects"
# ============================================================================
def build_empire(n_concepts=6, n_states=3, d_in=12, d_latent=8, jitter=0.05):
    """Each CONCEPT is one underlying meaning v_c. Each STATE renders it through
    its own fixed linear dialect R_s, so the same idea looks different by
    province. The Emperor's task: route the same concept to the same glyph."""
    base = RNG.standard_normal((n_concepts, d_latent))
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
    dialects = [RNG.standard_normal((d_in, d_latent)) for _ in range(n_states)]
    X, cids, sids = [], [], []
    for c in range(n_concepts):
        for s in range(n_states):
            msg = dialects[s] @ base[c] + jitter * RNG.standard_normal(d_in)
            X.append(msg); cids.append(c); sids.append(s)
    return np.array(X), np.array(cids), np.array(sids), base, dialects


# ============================================================================
# 1. PARAMETERS: one encoder per state, one shared decoder, one shared codebook
# ============================================================================
def init_params(n_states, d_in, d_hidden, d_latent, n_codes):
    def xv(shape, fan):
        return RNG.standard_normal(shape) * np.sqrt(1.0 / fan)
    p = {}
    for s in range(n_states):
        p[f"W1_{s}"] = xv((d_hidden, d_in), d_in)
        p[f"b1_{s}"] = np.zeros(d_hidden)
        p[f"W2_{s}"] = xv((d_latent, d_hidden), d_hidden)
        p[f"b2_{s}"] = np.zeros(d_latent)
    p["U1"] = xv((d_hidden, d_latent), d_latent); p["c1"] = np.zeros(d_hidden)
    p["U2"] = xv((d_latent, d_hidden), d_hidden); p["c2"] = np.zeros(d_latent)
    p["E"]  = RNG.standard_normal((n_codes, d_latent)) * 0.5   # the imperial standard
    return p


# ============================================================================
# 2. SOFTMAX over canonical glyphs (fully differentiable "reading of the script")
# ============================================================================
def soft_codes(E, z, tau=TAU):
    """Return soft assignment w over glyphs and the canonical reconstruction q."""
    d = E - z[None, :]                 # (K, d_latent)
    dist = np.sum(d * d, axis=1)       # (K,)
    logits = -dist / tau
    logits -= logits.max()             # stabilise
    ex = np.exp(logits); w = ex / ex.sum()
    q = w @ E
    return w, q


# ============================================================================
# 3. FORWARD + BACKWARD (hand-written, fully differentiable)  -> loss & grads
# ============================================================================
def loss_and_grad(p, X, T, cids, sids, n_states, n_codes,
                  beta=0.3, gamma=1.0, tau=TAU):
    N = X.shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    E = p["E"]
    Lr = Lc = Li = 0.0
    cache = []

    # ---- pass A: forward; decoder + codebook grads; dz from recon & commit ----
    for i in range(N):
        s = int(sids[i]); x = X[i]
        a1 = p[f"W1_{s}"] @ x + p[f"b1_{s}"]; h1 = np.tanh(a1)
        z  = p[f"W2_{s}"] @ h1 + p[f"b2_{s}"]
        w, q = soft_codes(E, z, tau)

        g1 = p["U1"] @ q + p["c1"]; hd = np.tanh(g1)
        vhat = p["U2"] @ hd + p["c2"]

        diff = vhat - T[i]      # recover the MEANING, not the provincial surface
        Lr += np.sum(diff * diff)
        cz = z - q
        Lc += beta * np.sum(cz * cz)

        # decoder backprop (recon)
        dxhat = 2.0 * diff
        g["U2"] += np.outer(dxhat, hd); g["c2"] += dxhat
        dhd = p["U2"].T @ dxhat; dg1 = dhd * (1 - hd * hd)
        g["U1"] += np.outer(dg1, q);  g["c1"] += dg1
        dq = p["U1"].T @ dg1

        # commitment beta||z - q||^2 : grads to z (+) and q (-)
        dz = 2.0 * beta * cz
        dq += -2.0 * beta * cz

        # backprop dq through soft codebook q = w@E  ->  dz (more) and dE
        gk = E @ dq                       # (K,) g_k = E_k . dq
        a  = w @ gk                       # scalar
        coef = w * (gk - a)               # (K,)
        zmE = z[None, :] - E              # (K, d)  (z - E_k)
        # dl_k/dz = -2(z-E_k)/tau ; dl_k/dE_k = 2(z-E_k)/tau
        dz = dz + np.sum(coef[:, None] * (-2.0 / tau) * zmE, axis=0)
        g["E"] += np.outer(w, dq) + coef[:, None] * (2.0 / tau) * zmE

        cache.append((s, x, a1, h1, z, dz))

    # ---- pass B: interoperability (same concept across states pulls z together) ----
    dz_extra = [np.zeros_like(c[4]) for c in cache]
    for c in np.unique(cids):
        idxs = np.where(cids == c)[0]
        if len(idxs) < 2:
            continue
        Z = np.stack([cache[i][4] for i in idxs]); zbar = Z.mean(axis=0)
        m = len(idxs)
        for i in idxs:
            r = cache[i][4] - zbar
            Li += gamma * np.sum(r * r)
            # d/dz_j sum_i||z_i - mean||^2 = 2(z_j - mean); the mean's own
            # dependence cancels because sum_i(z_i - mean) = 0.
            dz_extra[i] += 2.0 * gamma * r

    # ---- pass C: encoder backprop with TOTAL dz (recon+commit+codebook+interop) ----
    for i in range(N):
        s, x, a1, h1, z, dz_main = cache[i]
        dz = dz_main + dz_extra[i]
        g[f"W2_{s}"] += np.outer(dz, h1); g[f"b2_{s}"] += dz
        dh1 = p[f"W2_{s}"].T @ dz; da1 = dh1 * (1 - h1 * h1)
        g[f"W1_{s}"] += np.outer(da1, x); g[f"b1_{s}"] += da1

    for k in g:           # normalise everything by N
        g[k] /= N
    total = (Lr + Lc + Li) / N
    parts = dict(recon=Lr / N, commit=Lc / N, interop=Li / N)
    return total, g, parts


# ============================================================================
# 4. GRADIENT CHECK (mandatory) — central finite differences vs analytic
# ============================================================================
def gradient_check():
    X, cids, sids, base_lat, _ = build_empire(n_concepts=4, n_states=2, d_in=6,
                                              d_latent=4, jitter=0.05)
    p = init_params(2, 6, 8, 4, 4)
    T = base_lat[cids]
    L0, g, _ = loss_and_grad(p, X, T, cids, sids, 2, 4)
    eps, worst = 1e-6, 0.0
    for name in ["W1_0", "b2_0", "W2_1", "b1_1", "U1", "c2", "U2", "E"]:
        flat = p[name].ravel()
        for t in range(min(flat.size, 6)):
            o = flat[t]
            flat[t] = o + eps; Lp, *_ = loss_and_grad(p, X, T, cids, sids, 2, 4)
            flat[t] = o - eps; Lm, *_ = loss_and_grad(p, X, T, cids, sids, 2, 4)
            flat[t] = o
            num = (Lp - Lm) / (2 * eps); ana = g[name].ravel()[t]
            rel = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            worst = max(worst, rel)
    print(f"[gradient check] worst relative error = {worst:.2e}  "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}")
    return worst < 1e-4


# ============================================================================
# 5. METRICS — standardization (one script) and script sharpness
# ============================================================================
def standardization_score(p, X, cids, sids, tau=TAU):
    """Fraction of concepts whose every state-rendering chooses the SAME glyph
    (hard argmax). 1.0 = one script: any edict legible in any province."""
    code_of = {}
    for i in range(X.shape[0]):
        s = int(sids[i])
        a1 = p[f"W1_{s}"] @ X[i] + p[f"b1_{s}"]; h1 = np.tanh(a1)
        z  = p[f"W2_{s}"] @ h1 + p[f"b2_{s}"]
        w, _ = soft_codes(p["E"], z, tau)
        code_of.setdefault(int(cids[i]), []).append(int(np.argmax(w)))
    unified = sum(1 for ks in code_of.values() if len(set(ks)) == 1)
    return unified / len(code_of)


# ============================================================================
# 6. TRAINING LOOP — forge one script from many warring dialects
# ============================================================================
def train(steps=1300, lr=0.05, log_every=200, tau0=1.1, tau1=0.30):
    """Anneal the script from loose (high tau) to tight (low tau): the empire
    starts with many tolerable readings and converges on ONE glyph per idea —
    literally the historical act of standardisation, run as a schedule."""
    X, cids, sids, base, dialects = build_empire()
    n_states = len(dialects); n_codes = 6
    p = init_params(n_states, X.shape[1], 16, base.shape[1], n_codes)
    T = base[cids]
    tau = tau0
    for step in range(1, steps + 1):
        frac = (step - 1) / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac          # geometric annealing
        total, g, parts = loss_and_grad(p, X, T, cids, sids, n_states, n_codes, tau=tau)
        for k in p:
            p[k] -= lr * g[k]
        if step == 1 or step % log_every == 0:
            sc = standardization_score(p, X, cids, sids, tau=tau)
            print(f"  step {step:4d} | tau {tau:4.2f} | loss {total:7.4f}"
                  f" | recon {parts['recon']:.4f} | interop {parts['interop']:.4f}"
                  f" | standardization {sc:.2f}")
    return p, (X, T, cids, sids), n_codes, tau1


# ============================================================================
# 7. THE WAR ON DEATH — persistence test (the deathless standard)
# ============================================================================
def persistence_test(p_trained, data, n_codes, tau=0.30):
    X, T, cids, sids = data
    n_states = len({int(s) for s in sids})
    d_in = X.shape[1]; d_lat = p_trained["E"].shape[1]
    # recover concept-mean targets from the trained model itself is unneeded;
    # rebuild the same empire's meanings deterministically is not available here,
    # so we reuse stored meanings passed via data tuple.

    def fresh(): return init_params(n_states, d_in, 16, d_lat, n_codes)

    def short_train(p, freeze_E, steps=30):
        for _ in range(steps):
            _, g, _ = loss_and_grad(p, X, T, cids, sids, n_states, n_codes, tau=tau)
            for k in p:
                if freeze_E and k == "E":
                    continue
                p[k] -= 0.05 * g[k]
        return standardization_score(p, X, cids, sids, tau=tau)

    heir = fresh(); heir["E"] = p_trained["E"].copy()
    heir_score = short_train(heir, freeze_E=True)
    rival = fresh(); rival_score = short_train(rival, freeze_E=False)
    print(f"  heir   (inherits frozen standard): standardization {heir_score:.2f}")
    print(f"  rival  (must relearn standard)   : standardization {rival_score:.2f}")
    print("  verdict: " + ("the standard outlived its administrators."
          if heir_score >= rival_score else "standard did not transfer this run."))
    return heir_score, rival_score


# ============================================================================
# 8. MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 74)
    print("THE CANONICAL CODEX NETWORK — Qin Shi Huang (259-210 BCE)")
    print("unify the script; enforce the standard; make the policy deathless.")
    print("=" * 74)

    print("\n[1] Gradient check (soft codebook, central differences)")
    ok = gradient_check()

    print("\n[2] Training: forging one script from many warring dialects")
    p, data, n_codes, tau_f = train()

    print("\n[3] War on death: does the standard survive the death of the state?")
    persistence_test(p, data, n_codes, tau=tau_f)

    print("\n[4] Final reading of the empire")
    Xf, Tf, cf, sf = data
    final = standardization_score(p, Xf, cf, sf, tau=tau_f)
    print(f"  final standardization score = {final:.2f}"
          f"  ({'one script, fully legible' if final == 1.0 else 'partially unified'})")
    print("\nDone." if ok else "\nDone (gradient check FAILED — investigate).")

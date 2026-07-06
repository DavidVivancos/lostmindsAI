#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 chapter_0087_Charaka_-300.py  - CHARAKA  ::  The Doshic Homeostatic Controller (DHC)
 A from-scratch (pure-NumPy) neural architecture that encodes the cognitive
 signature of Charaka, redactor of the Charaka Samhita (c. 2nd c. BCE - debated).
================================================================================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 5 Minds 81 - 100 Available on Amazon https://www.amazon.com/dp/B0H7LP5LP2
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0087 · Charaka

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------
Most "ancient physician" stubs collapse into "holism + balance + three modes".
That is the trap. Charaka's *specific* cognition is sharper than that, and it is
documented in his own surviving compendium. Four ideas are his, not a genre's:

  1. YUKTI (the fourth pramana / means of knowledge).
     Charaka adds, to perception/inference/testimony, a fourth source of valid
     knowledge: *yukti* - "knowledge arising out of multiple things operating
     together", reasoning that fuses many weak, time-spread causes into one
     forecast across past, present and future (Sutra-sthana 11.17). It is the
     ACTIVE synthesis of multifactorial causation, not a single-cause inference.
     -> Encoded as a recurrent multi-factor *fusion* head that integrates the
        serially-attended causes over a time window into a PROGNOSIS.

  2. ANU + EKATVA MANAS (the atomic, single mind).
     Charaka holds the mind (manas) to be anu (atomic/minute) and eka (single):
     it can attend to ONE object at a time. Apparent simultaneity of the senses
     is an illusion of very fast *serial* sampling - the firebrand-wheel
     (alata-chakra) that looks like a ring of fire only because the brand moves
     fast. (Sutra-sthana 8; Sharira-sthana 1; Chakrapani's commentary.)
     -> Encoded as a low-temperature attention pushed toward a one-hot
        (single-pointed) read each step, via an explicit entropy penalty.

  3. TRIDOSHA HOMEOSTASIS with a PERSONALISED set-point.
     Health (sama) is not a fixed universal state but each body's own dynamic
     equilibrium of vata/pitta/kapha, fixed at conception (prakriti). Disease is
     deviation; therapy is control back to *that individual's* set-point.
     -> The whole network is a closed-loop CONTROLLER minimising deviation of a
        3-vector latent dosha state from a per-patient set-point p*.

  4. JANAPADODHVAMSA (destruction of communities / epidemics, Vimana 3).
     Charaka's deepest systems insight: when the four *shared* substrates - air,
     water, land, season - are vitiated, everyone falls ill with the SAME disease
     regardless of how different their constitutions are. Individual robustness
     does not save a population when the common environment is corrupted.
     -> A built-in stress test that corrupts the shared disturbance for ALL
        patients at once and measures collective (not individual) breakdown.

THE TASK THE NETWORK ACTUALLY LEARNS
------------------------------------
Each "patient" is an episode. A latent dosha state x in R^3 starts displaced
from the patient's personal set-point p* and is pushed every step by a hidden
disease pressure (a weighted sum of K candidate causes). The network observes
only its own felt state and the *drift* it produces; through single-pointed
attention over the K causes and yukti-fusion over time it (a) FORECASTS the
untreated prognosis and (b) emits an intervention u that drives x back to p*.

Everything is differentiable. A finite-difference gradient check is run on every
execution (mandatory). The dynamics are a fixed, known, differentiable simulator
so the closed loop can be optimised directly.

  Author convention for this corpus: pure NumPy, manual backprop, passing grad
  check, real training loop, self-tests, executed before shipping.
================================================================================
"""

import numpy as np

np.random.seed(87)  # Charaka is figure 87

# ----------------------------------------------------------------------------
# Fixed environment constants (the body + world; NOT learned parameters).
# ----------------------------------------------------------------------------
D_X = 3      # dosha latent dimension: [vata, pitta, kapha]
K   = 5      # candidate causal factors (e.g. diet, climate, exertion, sleep, mind)
D_F = 4      # descriptor dimension of each candidate cause
T   = 4      # serial glances / control steps in an episode (the manas samples 1/step)

ALPHA = 0.20   # the body's own homeostatic relaxation toward p* (svabhava)
G     = 0.60   # therapeutic gain: how strongly an intervention moves the state

# Per-factor disease "drive" on the doshas (which cause pushes which humor).
# Fixed property of the world. Shape (K, D_X).
FACTOR_DRIVE = np.array([
    [ 0.9, -0.1,  0.0],   # cause 0 (e.g. dry/light diet) aggravates vata
    [-0.1,  0.9, -0.1],   # cause 1 (e.g. heat/sour)      aggravates pitta
    [ 0.0, -0.1,  0.9],   # cause 2 (e.g. heavy/cold)     aggravates kapha
    [ 0.5,  0.4,  0.0],   # cause 3 (e.g. exertion)       vata+pitta
    [ 0.0,  0.5,  0.5],   # cause 4 (e.g. grief/inertia)  pitta+kapha
], dtype=np.float64)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def glorot(shape, rng):
    """Glorot/Xavier uniform init for a 2-D weight."""
    fan_out, fan_in = shape[0], shape[1]
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=shape)


# ----------------------------------------------------------------------------
# Synthetic clinic: generate a batch of patient episodes.
# ----------------------------------------------------------------------------
def make_patients(n, rng):
    """
    Returns a dict describing n patients:
      Phi   (n,K,D_F)  static descriptors of the K candidate causes for each
      pstar (n,D_X)    the personal dosha set-point (prakriti)
      x0    (n,D_X)    initial (displaced) dosha state
      wstar (n,K)      hidden ground-truth activation of each cause (disease cause)
      delta (n,D_X)    the constant disease pressure  = sum_k wstar_k * FACTOR_DRIVE_k
    """
    # Descriptors: each cause has a 4-D signature; we make descriptor[:, :3]
    # correlate with that cause's dosha drive so attention has something to learn.
    base = np.zeros((K, D_F))
    base[:, :3] = FACTOR_DRIVE
    base[:, 3] = np.linspace(-0.5, 0.5, K)            # a tie-breaking bias channel
    Phi = base[None, :, :] + 0.05 * rng.standard_normal((n, K, D_F))

    pstar = 0.3 * rng.standard_normal((n, D_X))        # individual constitution
    # A real etiology is dominated by a principal cause (nidana) with minor
    # contributors -> a sparse wstar with one clearly dominant active cause.
    wstar = np.clip(rng.standard_normal((n, K)) * 0.22, 0, None)
    strong = rng.integers(0, K, size=n)
    wstar[np.arange(n), strong] += rng.uniform(0.9, 1.3, size=n)

    delta = wstar @ FACTOR_DRIVE                       # (n, D_X) constant pressure
    x0 = pstar + 0.8 * rng.standard_normal((n, D_X))   # start displaced from p*
    return dict(Phi=Phi, pstar=pstar, x0=x0, wstar=wstar, delta=delta)


def untreated_targets(batch):
    """
    The prognosis the physician must forecast: how far the UNTREATED body drifts
    from p* at each step (u = 0). Independent of model parameters -> a clean,
    detachable target for the yukti forecast head. Returns y_star (n, T).
    """
    pstar, x0, delta = batch["pstar"], batch["x0"], batch["delta"]
    n = x0.shape[0]
    y = np.zeros((n, T))
    x = x0.copy()
    for t in range(T):
        x = x + ALPHA * (pstar - x) + delta            # no control
        y[:, t] = ((x - pstar) ** 2).sum(1)            # deviation magnitude
    return y


# ----------------------------------------------------------------------------
# The Doshic Homeostatic Controller
# ----------------------------------------------------------------------------
class DoshicController:
    """
    Learnable parameters (the physician's trained judgement):
      Wq  (d_a, 9)      query from [x_t ; p* ; drift_t]
      Wk  (d_a, D_F)    key   from each cause descriptor   (manas: what to look at)
      Wv  (d_v, D_F)    value from each cause descriptor
      Whh (d_h, d_h)    yukti recurrence (fuse causes across time)
      Whc (d_h, d_v)    yukti <- attended cause context
      Whx (d_h, 9)      yukti <- felt state/drift
      bh  (d_h,)
      Wy  (1, d_h)      prognosis head (forecast untreated deviation)
      Wb  (1, d_h)      treatment-intensity head (how hard to push against the
                        ATTENDED cause; therapy is cause-targeted = nidana-based)
    """

    def __init__(self, d_a=6, d_v=6, d_h=8, temp=0.4, seed=87):
        rng = np.random.default_rng(seed)
        self.d_a, self.d_v, self.d_h = d_a, d_v, d_h
        self.temp = temp
        self.scale = 1.0 / (np.sqrt(d_a) * temp)
        self.P = {
            "Wq":  glorot((d_a, 9), rng),
            "Wk":  glorot((d_a, D_F), rng),
            "Wv":  glorot((d_v, D_F), rng),
            "Whh": glorot((d_h, d_h), rng) * 0.5,
            "Whc": glorot((d_h, d_v), rng),
            "Whx": glorot((d_h, 9), rng),
            "bh":  np.zeros(d_h),
            "Wy":  glorot((1, d_h), rng),
            "Wb":  glorot((1, d_h), rng),
        }

    # --- forward, keeping every cache we need for manual backprop -------------
    def forward(self, batch, y_star, lam_f=0.3, lam_u=0.02, lam_e=0.05):
        P = self.P
        Phi, pstar, x0 = batch["Phi"], batch["pstar"], batch["x0"]
        delta = batch["delta"]
        n = x0.shape[0]
        d_h, d_v = self.d_h, self.d_v

        keys = np.einsum("nkf,af->nka", Phi, P["Wk"])    # (n,K,d_a)
        V    = np.einsum("nkf,vf->nkv", Phi, P["Wv"])    # (n,K,d_v)

        xs = [x0]                       # states x_0 .. x_T
        hs = []                         # h_0 .. h_{T-1}
        cache = []
        h_prev = np.zeros((n, d_h))
        x_prev = x0                     # for drift_0 = 0
        L_reg = L_fore = L_u = L_ent = 0.0

        for t in range(T):
            x_t = xs[-1]
            drift = x_t - x_prev if t > 0 else np.zeros_like(x_t)
            inp = np.concatenate([x_t, pstar, drift], axis=1)        # (n,9)

            q = inp @ P["Wq"].T                                      # (n,d_a)
            scores = (q[:, None, :] * keys).sum(-1) * self.scale     # (n,K)
            a = softmax(scores, axis=1)                              # (n,K)
            c = (a[:, :, None] * V).sum(1)                           # (n,d_v)

            pre_h = h_prev @ P["Whh"].T + c @ P["Whc"].T \
                    + inp @ P["Whx"].T + P["bh"]                     # (n,d_h)
            h = np.tanh(pre_h)
            yhat = h @ P["Wy"].T                                     # (n,1)

            # --- cause-targeted therapy (nidana-based) ---
            # The mind reads the attended cause's believed dosha-drive from the
            # descriptor channels Phi[:,:,:3] and pushes AGAINST it with intensity
            # beta. Attend the wrong cause -> therapy points the wrong way.
            beta = h @ P["Wb"].T                                     # (n,1) intensity
            PhiDrive = Phi[:, :, :3]                                 # (n,K,3) belief
            drive_att = (a[:, :, None] * PhiDrive).sum(1)            # (n,3)
            therapy = -G * beta * drive_att                         # (n,3)

            x_next = x_t + ALPHA * (pstar - x_t) + delta + therapy  # (n,3)

            # losses (summed over batch; averaged later)
            L_reg  += ((x_next - pstar) ** 2).sum()
            L_fore += lam_f * ((yhat[:, 0] - y_star[:, t]) ** 2).sum()
            L_u    += lam_u * (beta ** 2).sum()
            ent = -(a * np.log(a + 1e-12)).sum(1)                    # (n,)
            L_ent  += lam_e * ent.sum()

            cache.append(dict(inp=inp, q=q, keys=keys, V=V, scores=scores,
                              a=a, c=c, pre_h=pre_h, h=h, h_prev=h_prev,
                              yhat=yhat, beta=beta, drive_att=drive_att,
                              PhiDrive=PhiDrive, x_t=x_t, ent=ent[:, None]))
            hs.append(h)
            xs.append(x_next)
            x_prev = x_t
            h_prev = h

        loss = (L_reg + L_fore + L_u + L_ent) / n
        ctx = dict(cache=cache, xs=xs, pstar=pstar, Phi=Phi, n=n,
                   y_star=y_star, lam_f=lam_f, lam_u=lam_u, lam_e=lam_e)
        meta = dict(L_reg=L_reg / n, L_fore=L_fore / n,
                    L_u=L_u / n, L_ent=L_ent / n,
                    attn=np.stack([c["a"] for c in cache], axis=1))  # (n,T,K)
        return loss, ctx, meta

    # --- manual backprop through the whole rolled-out closed loop ------------
    def backward(self, ctx):
        P = self.P
        cache, xs = ctx["cache"], ctx["xs"]
        pstar, Phi, n = ctx["pstar"], ctx["Phi"], ctx["n"]
        y_star = ctx["y_star"]
        lam_f, lam_u, lam_e = ctx["lam_f"], ctx["lam_u"], ctx["lam_e"]
        d_h = self.d_h

        g = {k: np.zeros_like(v) for k, v in P.items()}
        # gradient accumulators indexed by state/hidden
        gxs = [np.zeros_like(x) for x in xs]      # grad on x_0 .. x_T
        ghs = [np.zeros((n, d_h)) for _ in range(T)]

        # regulation seeds: loss term ||x_{t+1}-p*||^2 -> grad on x_{t+1}
        for s in range(1, T + 1):
            gxs[s] += 2.0 * (xs[s] - pstar)

        for t in range(T - 1, -1, -1):
            cc = cache[t]
            a, V, keys = cc["a"], cc["V"], cc["keys"]
            q, h, h_prev = cc["q"], cc["h"], cc["h_prev"]
            beta, drive_att, PhiDrive = cc["beta"], cc["drive_att"], cc["PhiDrive"]
            x_t = cc["x_t"]
            inp, c = cc["inp"], cc["c"]
            gx_next = gxs[t + 1]

            # ---- therapy + dynamics ----
            # x_next = (1-ALPHA)x_t + ALPHA p* + delta - G*beta*drive_att
            g_therapy = gx_next                                      # (n,3)
            g_beta = -G * (g_therapy * drive_att).sum(1, keepdims=True)
            g_beta += 2.0 * lam_u * beta                            # treatment cost
            g_drive_att = -G * beta * g_therapy                    # (n,3)
            ga_drive = (g_drive_att[:, None, :] * PhiDrive).sum(-1) # (n,K)
            g["Wb"] += g_beta.T @ h
            gh_from_beta = g_beta @ P["Wb"]                         # (n,d_h)

            gx_t = gx_next * (1.0 - ALPHA)                          # x_t in dynamics

            # ---- forecast head ----
            dyhat = 2.0 * lam_f * (cc["yhat"] - y_star[:, t:t + 1])  # (n,1)
            g["Wy"] += dyhat.T @ h
            gh = ghs[t] + gh_from_beta + dyhat @ P["Wy"]            # (n,d_h)

            # ---- tanh ----
            gpre = gh * (1.0 - h ** 2)
            g["bh"] += gpre.sum(0)
            g["Whh"] += gpre.T @ h_prev
            ghs[t - 1] += gpre @ P["Whh"] if t > 0 else 0.0
            g["Whc"] += gpre.T @ c
            gc = gpre @ P["Whc"]
            g["Whx"] += gpre.T @ inp
            ginp_h = gpre @ P["Whx"]

            # ---- context c = sum_k a_k V_k ----
            ga = (gc[:, None, :] * V).sum(-1) + ga_drive            # (n,K) both paths
            gV = a[:, :, None] * gc[:, None, :]                      # (n,K,d_v)
            g["Wv"] += np.einsum("nkv,nkf->vf", gV, Phi)

            # ---- softmax (context path) + direct entropy gradient ----
            dot = (a * ga).sum(1, keepdims=True)
            gs = a * (ga - dot)                                      # context path
            ent = cc["ent"]                                         # (n,1)
            gs += lam_e * (-a * (np.log(a + 1e-12) + ent))          # entropy path

            gsc = gs * self.scale
            gq = (gsc[:, :, None] * keys).sum(1)                    # (n,d_a)
            gkeys = gsc[:, :, None] * q[:, None, :]                 # (n,K,d_a)
            g["Wk"] += np.einsum("nka,nkf->af", gkeys, Phi)
            g["Wq"] += gq.T @ inp
            ginp_q = gq @ P["Wq"]

            # ---- combine grads into inp = [x_t, p*, drift_t] ----
            ginp = ginp_h + ginp_q
            gx_t += ginp[:, 0:3]                                     # x_t channel
            if t > 0:                                                # drift = x_t - x_prev
                gdrift = ginp[:, 6:9]
                gx_t += gdrift
                gxs[t - 1] += -gdrift
            # p* channel ginp[:,3:6] is data -> ignored

            gxs[t] += gx_t

        # average over batch (loss was divided by n)
        for k in g:
            g[k] /= n
        return g

    # convenience: loss only (for finite differences)
    def loss_only(self, batch, y_star, **kw):
        return self.forward(batch, y_star, **kw)[0]


# ----------------------------------------------------------------------------
# 1. Finite-difference gradient check  (MANDATORY)
# ----------------------------------------------------------------------------
def gradient_check(verbose=True):
    rng = np.random.default_rng(3)
    net = DoshicController(seed=11)
    batch = make_patients(6, rng)
    y_star = untreated_targets(batch)

    _, ctx, _ = net.forward(batch, y_star)
    ganalytic = net.backward(ctx)

    eps = 1e-6
    max_rel, worst = 0.0, None
    for name, W in net.P.items():
        flat = W.ravel()
        # sample a handful of coordinates per parameter for speed
        idxs = np.linspace(0, flat.size - 1, min(8, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = net.loss_only(batch, y_star)
            flat[i] = orig - eps
            lm = net.loss_only(batch, y_star)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = ganalytic[name].ravel()[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > max_rel:
                max_rel, worst = rel, (name, i, num, ana)
    if verbose:
        print(f"  max relative error : {max_rel:.3e}")
        nm, i, num, ana = worst
        print(f"  worst param        : {nm}[{i}]  num={num:+.6e}  ana={ana:+.6e}")
    ok = max_rel < 1e-4
    print(f"  gradient check     : {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# 2. Training loop  (real optimisation of the closed-loop control objective)
# ----------------------------------------------------------------------------
def train(net, steps=600, batch_n=64, lr=0.03, seed=7, log_every=100):
    rng = np.random.default_rng(seed)
    # Adam state
    m = {k: np.zeros_like(v) for k, v in net.P.items()}
    v = {k: np.zeros_like(v) for k, v in net.P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(1, steps + 1):
        lr_t = lr * (0.5 + 0.5 * np.cos(np.pi * (it - 1) / steps))  # cosine decay
        batch = make_patients(batch_n, rng)
        y_star = untreated_targets(batch)
        loss, ctx, meta = net.forward(batch, y_star)
        g = net.backward(ctx)
        for k in net.P:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            mh = m[k] / (1 - b1 ** it)
            vh = v[k] / (1 - b2 ** it)
            net.P[k] -= lr_t * mh / (np.sqrt(vh) + eps)
        hist.append((loss, meta["L_reg"], meta["attn"].reshape(-1, K)))
        if it % log_every == 0 or it == 1:
            mean_ent = -(meta["attn"] * np.log(meta["attn"] + 1e-12)).sum(-1).mean()
            print(f"  step {it:4d} | loss {loss:7.4f} | regulation {meta['L_reg']:7.4f}"
                  f" | attn entropy {mean_ent:5.3f}")
    return hist


def regulation_deviation(net, batch):
    """Mean final deviation ||x_T - p*|| under the trained controller (lower=better)."""
    y_star = untreated_targets(batch)
    _, ctx, _ = net.forward(batch, y_star)
    xT = ctx["xs"][-1]
    return np.sqrt(((xT - batch["pstar"]) ** 2).sum(1)).mean()


def untreated_deviation(batch):
    pstar, x0, delta = batch["pstar"], batch["x0"], batch["delta"]
    x = x0.copy()
    for _ in range(T):
        x = x + ALPHA * (pstar - x) + delta
    return np.sqrt(((x - pstar) ** 2).sum(1)).mean()


# ----------------------------------------------------------------------------
# 3. Self-tests that demonstrate the four Charaka signatures
# ----------------------------------------------------------------------------
def run_self_tests(net):
    rng = np.random.default_rng(123)
    test = make_patients(400, rng)

    # (A) Tridosha homeostasis: trained controller regulates far better than
    #     either an untreated body or an untrained controller.
    untreated = untreated_deviation(test)
    trained = regulation_deviation(net, test)
    fresh = regulation_deviation(DoshicController(seed=999), test)
    print("\n[A] Tridosha homeostasis (mean final deviation from set-point):")
    print(f"     untreated body      : {untreated:6.4f}")
    print(f"     untrained controller: {fresh:6.4f}")
    print(f"     trained controller  : {trained:6.4f}")
    a_ok = trained < 0.5 * untreated and trained < fresh
    print(f"     regulation learned  : {'PASS' if a_ok else 'FAIL'}")

    # (B) Anu-eka manas: attention is single-pointed (low entropy => ~one cause/step).
    _, ctx, meta = net.forward(test, untreated_targets(test))
    attn = meta["attn"]                              # (n,T,K)
    ent = -(attn * np.log(attn + 1e-12)).sum(-1).mean()
    peak = attn.max(-1).mean()                        # mass on the single top cause
    uniform_ent = np.log(K)
    print("\n[B] Anu-eka manas (single-pointed serial attention):")
    print(f"     mean entropy {ent:5.3f}  (uniform would be {uniform_ent:5.3f})")
    print(f"     mean mass on dominant cause: {peak:5.3f}")
    b_ok = ent < 0.6 * uniform_ent and peak > 0.55
    print(f"     attention is single-pointed: {'PASS' if b_ok else 'FAIL'}")

    # (B2) does the manas attend to the TRUE dominant cause? (focus accuracy)
    true_cause = test["wstar"].argmax(1)
    attended = attn.mean(1).argmax(1)                 # most-attended cause over episode
    focus_acc = (attended == true_cause).mean()
    print(f"     attends true dominant cause: {focus_acc:5.3f} of patients")

    # (C) Yukti prognosis: forecast head predicts untreated deviation trajectory.
    y_star = untreated_targets(test)
    yhat = np.stack([c["yhat"][:, 0] for c in ctx["cache"]], axis=1)  # (n,T)
    corr = np.corrcoef(yhat.ravel(), y_star.ravel())[0, 1]
    print("\n[C] Yukti (multi-factor prognosis forecast):")
    print(f"     corr(forecast, true untreated deviation) = {corr:5.3f}")
    c_ok = corr > 0.6
    print(f"     prognosis is informative: {'PASS' if c_ok else 'FAIL'}")

    # (D) Janapadodhvamsa: corrupt the SHARED substrate for everyone at once.
    #     A common environmental shift is added to every patient's disease
    #     pressure. Charaka's claim: collective breakdown despite individual
    #     constitutions. We measure regulation before vs after corruption.
    base_dev = regulation_deviation(net, test)
    epidemic = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in test.items()}
    shared_shift = np.array([0.6, 0.6, 0.6])          # air+water+land+season vitiated
    epidemic["delta"] = epidemic["delta"] + shared_shift
    epi_dev = regulation_deviation(net, epidemic)
    # also: individual-only perturbation of the SAME total magnitude, but random
    # per patient (cancels in aggregate) -> should hurt far less collectively.
    idio = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in test.items()}
    idio["delta"] = idio["delta"] + np.linalg.norm(shared_shift) * \
        rng.standard_normal(idio["delta"].shape) / np.sqrt(D_X)
    idio_dev = regulation_deviation(net, idio)
    print("\n[D] Janapadodhvamsa (shared-substrate epidemic vs idiosyncratic noise):")
    print(f"     baseline deviation              : {base_dev:6.4f}")
    print(f"     shared environmental corruption : {epi_dev:6.4f}")
    print(f"     equal-size idiosyncratic noise  : {idio_dev:6.4f}")
    d_ok = epi_dev > 1.4 * base_dev and epi_dev > idio_dev
    print(f"     collective failure dominates    : {'PASS' if d_ok else 'FAIL'}")

    return all([a_ok, b_ok, c_ok, d_ok])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 72)
    print(" CHARAKA - Doshic Homeostatic Controller")
    print("=" * 72)

    print("\n[1] Gradient check (finite differences):")
    grad_ok = gradient_check()

    print("\n[2] Training the controller (closed-loop regulation):")
    net = DoshicController(seed=87)
    train(net, steps=900, batch_n=64, lr=0.03, log_every=150)

    print("\n[3] Self-tests:")
    tests_ok = run_self_tests(net)

    print("\n" + "=" * 72)
    print(f" gradient check : {'PASS' if grad_ok else 'FAIL'}")
    print(f" self-tests     : {'PASS' if tests_ok else 'FAIL'}")
    print("=" * 72)

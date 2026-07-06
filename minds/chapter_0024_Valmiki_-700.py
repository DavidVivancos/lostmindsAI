#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chapter_0024_valmiki_-700.py
======================================================================
THE SOKA-SLOKA RESONATOR
A from-scratch, trainable neural architecture after Valmiki (c. 700 BCE),
the adi-kavi ("first poet") who composed the Ramayana.
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# # Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0024 · Valmiki
----------------------------------------------------------------------
WHY THIS ARCHITECTURE, AND WHY IT IS *NOT* A TRANSFORMER
----------------------------------------------------------------------
The founding legend of Indian poetry is a precise cognitive event, not a
metaphor. By the river Tamasa, Valmiki sees a hunter (nisada) kill the
male of a mating pair of krauncha birds. The female's cry of grief seizes
him, and a curse leaves his mouth -- but it leaves *in perfect meter*. He
then names what happened:  "soka(h) slokatvam agata(h)"  -- grief (soka)
HAS BECOME verse (sloka). (Ramayana, Bala Kanda, sarga 2.)

That single sentence is the whole thesis of this file. Valmiki's distinctive
claim about mind is NOT "intelligence imposes order on chaos" (the generic
lens). It is far more specific and far stranger:

    The highest-order, most transmissible structure a mind can emit
    (METER) is produced *at the moment of maximal affective
    dysregulation* (GRIEF), and the meter does not SUPPRESS the grief --
    it CONSERVES and CARRIES it, intact, across any distance and any
    number of re-tellings.

So we do not build attention-over-stored-keys. We build a *transducer*:
an input affective shock is converted into a periodic, rule-bound metrical
form. Crucially, the architecture splits each verse into two channels that
mirror the legend exactly:

  * FIXED metrical positions  -> the invariant FORM (the sloka's law).
        These are identical for every verse ever composed; they are what
        makes the verse recognisable, memorisable, and corruption-proof.
  * FREE positions            -> the CONTENT (this particular grief).
        The specific sorrow rides *inside* the lawful form.

The training objective therefore has three terms, each a literal reading
of the legend:
  (1) METER loss     : the fixed positions must obey the real anustubh
                       'pathya' rule -> "it came out as verse, not otherwise".
  (2) RECON loss     : the original grief must be recoverable from the free
                       positions -> "grief became verse" (nothing is lost).
  (3) ROBUST loss    : grief must still be recoverable after the verse is
                       corrupted by noise -> the oral tradition's survival
                       across 3,000 years of re-transmission.

The "resonance" is real: to satisfy the meter loss the recurrent core must
settle into a periodic pattern (period 8 within a pada, alternating odd/even
pada -> period 16). Grief is the impulse; the lawful periodic verse is the
sustained limit cycle it rings into.

----------------------------------------------------------------------
THE REAL ANUSTUBH (SLOKA) METRE WE ENCODE
----------------------------------------------------------------------
A sloka = 4 padas (quarter-verses) x 8 syllables = 32 syllables. Each
syllable is laghu (light=0) or guru (heavy=1). The classical 'pathya'
form fixes a small set of positions (1-indexed within each pada):
    syllable 5 : laghu (0)
    syllable 6 : guru  (1)
    syllable 7 : guru  (1) in ODD padas (1,3);  laghu (0) in EVEN padas (2,4)
Positions 1-4 and 8 are anceps (free) -> they carry CONTENT.
This is a genuine, checkable metrical constraint (see Macdonell, A Sanskrit
Grammar for Students, App. II; Goldman, Ramayana of Valmiki, Vol I, intro).

----------------------------------------------------------------------
ENGINEERING CONVENTIONS (kept across the whole 1000Minds corpus)
----------------------------------------------------------------------
  * Pure NumPy, written from scratch -- no autograd, no frameworks.
  * Hand-derived analytic gradients.
  * A finite-difference gradient check that MUST pass (printed below).
  * A real training loop whose loss measurably falls.
  * Self-tests that assert the learned model has the Valmiki properties.
  * Executable end-to-end; verified output pasted into the chapter.

Author convention note: this is a *mechanism sketch at small scale*, not a
production model. The point is that the mechanism is real, differentiable,
and verifiably learns the soka->sloka transduction.
======================================================================
"""

import numpy as np

# ----------------------------------------------------------------------
# 0. The metre. Built once; identical for every verse the model composes.
#    free_mask[t]  = 1 if syllable t carries CONTENT (anceps), else 0.
#    fixed_mask[t] = 1 if syllable t is metrically FIXED (the law), else 0.
#    meter_target[t] in {0,1} is the required laghu/guru at fixed positions.
# ----------------------------------------------------------------------
N_PADA = 4            # four quarter-verses
SYL_PER_PADA = 8      # eight syllables each
L = N_PADA * SYL_PER_PADA   # 32 syllables total


def build_anustubh_metre():
    """Return (meter_target, fixed_mask, free_mask) of shape (L,).

    Encodes the classical pathya sloka: within each pada (1-indexed),
    syllable 5 = laghu(0), 6 = guru(1), 7 = guru(1) for odd padas and
    laghu(0) for even padas. All other syllables are free (content)."""
    meter_target = np.zeros(L, dtype=np.float64)
    fixed_mask = np.zeros(L, dtype=np.float64)
    for p in range(N_PADA):                # p = 0..3  -> padas 1..4
        base = p * SYL_PER_PADA
        odd_pada = (p % 2 == 0)            # padas 1 and 3 are "odd"
        # syllable 5 (index base+4): laghu
        fixed_mask[base + 4] = 1.0
        meter_target[base + 4] = 0.0
        # syllable 6 (index base+5): guru
        fixed_mask[base + 5] = 1.0
        meter_target[base + 5] = 1.0
        # syllable 7 (index base+6): guru if odd pada else laghu
        fixed_mask[base + 6] = 1.0
        meter_target[base + 6] = 1.0 if odd_pada else 0.0
    free_mask = 1.0 - fixed_mask
    return meter_target, fixed_mask, free_mask


METER_TARGET, FIXED_MASK, FREE_MASK = build_anustubh_metre()
N_FIXED = int(FIXED_MASK.sum())   # 12 lawful positions
N_FREE = int(FREE_MASK.sum())     # 20 content positions


# ----------------------------------------------------------------------
# 1. The model.
# ----------------------------------------------------------------------
class SokaSlokaResonator:
    """A driven recurrent transducer: grief vector -> 32-syllable sloka.

    State recurrence (one step PER SYLLABLE, t = 1..L):
        a_t = Wr h_{t-1} + Wx g + br        (pre-activation, R^H)
        h_t = tanh(a_t)                     (hidden, R^H)
        z_t = wo . h_t + bo                 (scalar)
        w_t = sigmoid(z_t)                  (syllable weight in (0,1))
    The constant drive Wx g is the *grief held in mind* throughout the
    utterance -- the krauncha cry that will not leave the poet.

    Decoder (reads only FREE positions -> the content channel):
        g_hat = Wd (w (.) free_mask) + bd
    """

    def __init__(self, d_grief=6, hidden=24, seed=0):
        rng = np.random.default_rng(seed)
        self.Dg = d_grief
        self.H = hidden
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(1.0 / b)
        # Recurrent core (slightly under unit spectral scale so it is stable
        # but expressive enough to ring into a periodic pattern).
        self.Wr = rng.standard_normal((hidden, hidden)) * (0.9 / np.sqrt(hidden))
        self.Wx = s(hidden, d_grief)       # grief -> drive
        self.br = np.zeros(hidden)
        self.wo = rng.standard_normal(hidden) * np.sqrt(1.0 / hidden)
        self.bo = 0.0
        self.Wd = s(d_grief, L)            # verse -> reconstructed grief
        self.bd = np.zeros(d_grief)

    # -- parameter book-keeping (used by the gradient checker) ----------
    def params(self):
        return {"Wr": self.Wr, "Wx": self.Wx, "br": self.br,
                "wo": self.wo, "bo": np.array(self.bo),
                "Wd": self.Wd, "bd": self.bd}

    def set_param(self, name, value):
        if name == "bo":
            self.bo = float(value)
        else:
            setattr(self, name, value)

    # -- forward --------------------------------------------------------
    def forward(self, G, noise=None):
        """G: (B, Dg) batch of grief vectors.
        noise: optional (B, L) corruption added to the verse for the
        robustness channel (fixed externally so gradient checks are exact).
        Returns outputs dict + cache for backprop."""
        B = G.shape[0]
        H = self.H
        Hs = np.zeros((L + 1, B, H))       # hidden states, Hs[0] = 0
        As = np.zeros((L, B, H))           # pre-activations
        W = np.zeros((B, L))               # syllable weights
        Z = np.zeros((B, L))
        for t in range(L):
            a = Hs[t] @ self.Wr.T + G @ self.Wx.T + self.br   # (B,H)
            h = np.tanh(a)
            As[t] = a
            Hs[t + 1] = h
            z = h @ self.wo + self.bo                          # (B,)
            Z[:, t] = z
            W[:, t] = 1.0 / (1.0 + np.exp(-z))
        # content channel (clean + corrupted)
        Wc = W * FREE_MASK[None, :]
        Ghat = Wc @ self.Wd.T + self.bd                        # (B,Dg)
        if noise is None:
            noise = np.zeros((B, L))
        Wc_n = (W + noise) * FREE_MASK[None, :]
        Ghat_n = Wc_n @ self.Wd.T + self.bd                    # (B,Dg)
        cache = dict(G=G, Hs=Hs, As=As, W=W, Z=Z, noise=noise)
        return dict(W=W, Ghat=Ghat, Ghat_n=Ghat_n), cache

    # -- loss -----------------------------------------------------------
    def loss(self, out, cache, alpha=1.0, beta=1.0, gamma=0.5):
        """Three-term Valmiki objective. Returns (scalar, parts)."""
        B = cache["G"].shape[0]
        W, G = out["W"], cache["G"]
        # (1) METER: fixed positions obey the anustubh law.
        dm = (W - METER_TARGET[None, :]) * FIXED_MASK[None, :]
        L_meter = np.sum(dm ** 2) / (B * N_FIXED)
        # (2) RECON: grief recoverable from the clean verse.
        dr = out["Ghat"] - G
        L_recon = np.sum(dr ** 2) / (B * self.Dg)
        # (3) ROBUST: grief recoverable from the corrupted verse.
        drn = out["Ghat_n"] - G
        L_robust = np.sum(drn ** 2) / (B * self.Dg)
        total = alpha * L_meter + beta * L_recon + gamma * L_robust
        return total, dict(meter=L_meter, recon=L_recon, robust=L_robust)

    # -- backward (hand-derived analytic gradients) ---------------------
    def backward(self, out, cache, alpha=1.0, beta=1.0, gamma=0.5):
        G = cache["G"]; Hs = cache["Hs"]; As = cache["As"]
        W = cache["W"]; Z = cache["Z"]; noise = cache["noise"]
        B = G.shape[0]

        # ---- decoder grads (from recon + robust) ----
        Wc = W * FREE_MASK[None, :]
        Wc_n = (W + noise) * FREE_MASK[None, :]
        dGhat = (2.0 * beta / (B * self.Dg)) * (out["Ghat"] - G)     # (B,Dg)
        dGhat_n = (2.0 * gamma / (B * self.Dg)) * (out["Ghat_n"] - G)
        gWd = dGhat.T @ Wc + dGhat_n.T @ Wc_n                        # (Dg,L)
        gbd = dGhat.sum(0) + dGhat_n.sum(0)                          # (Dg,)

        # ---- dL/dW_t for every syllable (meter + recon + robust) ----
        dW = np.zeros((B, L))
        # meter term
        dW += (2.0 * alpha / (B * N_FIXED)) * \
              (W - METER_TARGET[None, :]) * FIXED_MASK[None, :]
        # recon + robust flow back through decoder, only via free positions
        dW += (dGhat @ self.Wd) * FREE_MASK[None, :]
        dW += (dGhat_n @ self.Wd) * FREE_MASK[None, :]

        # ---- through sigmoid -> z_t ----
        dZ = dW * W * (1.0 - W)                                      # (B,L)

        # ---- accumulate readout + BPTT ----
        gwo = np.zeros(self.H); gbo = 0.0
        gWr = np.zeros_like(self.Wr); gWx = np.zeros_like(self.Wx)
        gbr = np.zeros(self.H)
        dh_next = np.zeros((B, self.H))     # gradient flowing from t+1 into h_t
        for t in reversed(range(L)):
            # readout contribution into h_t: z_t = wo . h_t + bo
            gwo_t = dZ[:, t][:, None] * self.wo[None, :]   # dz->dh part (B,H)
            gbo += dZ[:, t].sum()
            # total grad into h_t (from this step's readout + future steps)
            dh = gwo_t + dh_next                          # (B,H)
            # through tanh
            da = dh * (1.0 - np.tanh(As[t]) ** 2)         # (B,H)
            # param grads at this step
            gWr += da.T @ Hs[t]                           # (H,H)
            gWx += da.T @ G                               # (H,Dg)
            gbr += da.sum(0)
            # propagate to previous hidden
            dh_next = da @ self.Wr                        # (B,H)

        # readout weight grad: gwo = sum_t sum_b dZ[b,t] * h_t[b]
        gwo = np.zeros(self.H)
        for t in range(L):
            gwo += dZ[:, t] @ Hs[t + 1]                   # (H,)

        return {"Wr": gWr, "Wx": gWx, "br": gbr,
                "wo": gwo, "bo": np.array(gbo),
                "Wd": gWd, "bd": gbd}


# ----------------------------------------------------------------------
# 2. Synthetic "grief corpus".
#    Each grief is a small affective vector (e.g. intensity, valence,
#    tenderness, anger, longing, awe). The model must learn a code that
#    survives the lawful metrical bottleneck. We sample a fixed set so the
#    task is well-defined and learnable at this scale.
# ----------------------------------------------------------------------
def make_griefs(n, d_grief=6, seed=1):
    rng = np.random.default_rng(seed)
    G = rng.uniform(-1.0, 1.0, size=(n, d_grief))
    return G


# ----------------------------------------------------------------------
# 3. MANDATORY finite-difference gradient check.
#    Verifies every analytic gradient against numerical perturbation.
# ----------------------------------------------------------------------
def gradient_check(verbose=True):
    model = SokaSlokaResonator(d_grief=4, hidden=8, seed=3)
    rng = np.random.default_rng(7)
    G = rng.uniform(-1, 1, size=(2, 4))
    noise = rng.standard_normal((2, L)) * 0.1   # fixed corruption
    hp = dict(alpha=1.0, beta=1.0, gamma=0.5)

    out, cache = model.forward(G, noise=noise)
    analytic = model.backward(out, cache, **hp)

    eps = 1e-6
    max_rel = 0.0
    worst = None
    for name, P in model.params().items():
        flat = np.array(P, dtype=np.float64).ravel().copy()
        g_an = analytic[name].ravel()
        # check a handful of entries per parameter (all, if small)
        idxs = range(flat.size) if flat.size <= 30 else \
            rng.choice(flat.size, 30, replace=False)
        base = flat.copy()
        for i in idxs:
            up = base.copy(); up[i] += eps
            dn = base.copy(); dn[i] -= eps
            model.set_param(name, up.reshape(np.array(P).shape))
            lp, _ = model.loss(*model.forward(G, noise=noise), **hp)
            model.set_param(name, dn.reshape(np.array(P).shape))
            lm, _ = model.loss(*model.forward(G, noise=noise), **hp)
            model.set_param(name, base.reshape(np.array(P).shape))
            num = (lp - lm) / (2 * eps)
            rel = abs(num - g_an[i]) / max(1e-8, abs(num) + abs(g_an[i]))
            if rel > max_rel:
                max_rel = rel; worst = (name, i, num, g_an[i])
    if verbose:
        print(f"[grad-check] max relative error = {max_rel:.3e}")
        print(f"[grad-check] worst entry: param={worst[0]} idx={worst[1]} "
              f"num={worst[2]:+.6e} analytic={worst[3]:+.6e}")
        ok = max_rel < 1e-5
        print(f"[grad-check] {'PASS' if ok else 'FAIL'} (threshold 1e-5)")
    return max_rel


# ----------------------------------------------------------------------
# 4. Training loop (full-batch Adam on the analytic gradients).
# ----------------------------------------------------------------------
def train(model, steps=6000, lr=3e-3, batch=32, noise_scale=0.15,
          alpha=1.0, beta=1.0, gamma=0.5, seed=11, log_every=750):
    """Online training over the CONTINUOUS grief space.

    A fresh batch of griefs is sampled every step, so the model learns the
    *mapping* grief -> lawful verse over the whole affective space rather
    than memorising a fixed list. This is what lets it compose a lawful
    verse for a sorrow it has never seen."""
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(np.array(v, dtype=np.float64))
         for k, v in model.params().items()}
    v = {k: np.zeros_like(np.array(val, dtype=np.float64))
         for k, val in model.params().items()}
    b1, b2, e = 0.9, 0.999, 1e-8
    history = []
    for step in range(1, steps + 1):
        G = rng.uniform(-1.0, 1.0, size=(batch, model.Dg))
        noise = rng.standard_normal((batch, L)) * noise_scale
        out, cache = model.forward(G, noise=noise)
        total, parts = model.loss(out, cache, alpha, beta, gamma)
        grads = model.backward(out, cache, alpha, beta, gamma)
        for k in grads:
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * (g * g)
            mhat = m[k] / (1 - b1 ** step)
            vhat = v[k] / (1 - b2 ** step)
            upd = lr * mhat / (np.sqrt(vhat) + e)
            cur = np.array(model.params()[k], dtype=np.float64)
            model.set_param(k, cur - upd)
        if step % log_every == 0 or step == 1:
            history.append((step, total, parts.copy()))
            print(f"  step {step:5d} | total {total:.5f} | "
                  f"meter {parts['meter']:.5f} | recon {parts['recon']:.5f} | "
                  f"robust {parts['robust']:.5f}")
    return history


# ----------------------------------------------------------------------
# 5. Helpers to read a learned verse as laghu/guru and to verify metre.
# ----------------------------------------------------------------------
def verse_pattern(model, g):
    out, _ = model.forward(g[None, :])
    w = out["W"][0]
    syll = (w > 0.5).astype(int)            # 0 = laghu (.), 1 = guru (-)
    return w, syll


def metre_is_satisfied(syll):
    """Do the FIXED positions match the anustubh law?"""
    fixed_idx = np.where(FIXED_MASK > 0)[0]
    target = METER_TARGET[fixed_idx].astype(int)
    return bool(np.all(syll[fixed_idx] == target)), fixed_idx, target


def render_verse(syll):
    """Pretty-print 4 padas of 8 syllables; '-' guru, '.' laghu."""
    glyph = {1: "-", 0: "."}
    lines = []
    for p in range(N_PADA):
        seg = syll[p * SYL_PER_PADA:(p + 1) * SYL_PER_PADA]
        lines.append(" ".join(glyph[int(s)] for s in seg))
    return lines


# ----------------------------------------------------------------------
# 6. Self-tests: assert the learned model has Valmiki's properties.
# ----------------------------------------------------------------------
def self_tests(model, G):
    print("\n[self-tests]")
    # (a) Metre obeyed for several griefs (the FORM is conserved).
    ok_metre = 0
    for i in range(min(8, G.shape[0])):
        _, syll = verse_pattern(model, G[i])
        good, _, _ = metre_is_satisfied(syll)
        ok_metre += int(good)
    print(f"  metre satisfied on {ok_metre}/8 sample griefs "
          f"(form is conserved across different sorrows)")
    assert ok_metre >= 7, "metre not learned"

    # (b) Grief recoverable from the clean verse (CONTENT is carried).
    out, cache = model.forward(G)
    recon_err = np.mean((out["Ghat"] - G) ** 2)
    print(f"  clean reconstruction MSE = {recon_err:.5f} "
          f"(grief survives the metrical bottleneck)")
    assert recon_err < 0.05, "grief not recoverable"

    # (c) Robustness: corrupt the verse, grief still recovers (oral survival).
    rng = np.random.default_rng(99)
    noise = rng.standard_normal((G.shape[0], L)) * 0.25
    out_n, _ = model.forward(G, noise=noise)
    robust_err = np.mean((out_n["Ghat_n"] - G) ** 2)
    print(f"  corrupted reconstruction MSE = {robust_err:.5f} "
          f"(verse re-told with errors still carries the grief)")
    assert robust_err < 0.10, "verse not corruption-robust"

    # (d) Distinct griefs -> distinguishable verses (content is informative).
    _, s0 = verse_pattern(model, G[0])
    _, s1 = verse_pattern(model, G[1])
    free_idx = np.where(FREE_MASK > 0)[0]
    differ = int(np.sum(s0[free_idx] != s1[free_idx]))
    print(f"  two different griefs differ in {differ}/{N_FREE} free "
          f"(content) syllables (the meter is shared, the sorrow is not)")
    print("  all self-tests PASSED")


# ----------------------------------------------------------------------
# 7. Main: grad-check -> train -> self-test -> compose a sample verse.
# ----------------------------------------------------------------------
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 70)
    print("THE SOKA-SLOKA RESONATOR  --  Valmiki (Chapter 24)")
    print("grief (soka)  ->  lawful metre (sloka)  ->  grief recovered")
    print("=" * 70)

    print("\n[metre] anustubh sloka: %d padas x %d syllables = %d total"
          % (N_PADA, SYL_PER_PADA, L))
    print("        fixed (law) positions: %d | free (content) positions: %d"
          % (N_FIXED, N_FREE))
    print("        law (fixed laghu/guru): ",
          METER_TARGET[FIXED_MASK > 0].astype(int))

    print("\n[1] Finite-difference gradient check")
    gradient_check(verbose=True)

    print("\n[2] Training the resonator (online over the grief space)")
    model = SokaSlokaResonator(d_grief=6, hidden=24, seed=0)
    train(model, steps=6000, lr=3e-3, batch=32, log_every=750)

    print("\n[3] Self-tests on HELD-OUT griefs (never seen in training)")
    G_eval = make_griefs(16, d_grief=6, seed=1)
    self_tests(model, G_eval)

    print("\n[4] Composing a verse from a fresh grief")
    fresh = np.array([0.9, -0.7, 0.4, -0.2, 0.6, -0.5])  # an unseen sorrow
    w, syll = verse_pattern(model, fresh)
    good, fixed_idx, target = metre_is_satisfied(syll)
    print("  grief vector :", fresh)
    print("  composed sloka (- guru / . laghu):")
    for ln in render_verse(syll):
        print("      " + ln)
    print("  metre lawful :", good)
    # recover the grief from this single verse
    out, _ = model.forward(fresh[None, :])
    print("  grief recovered from verse:", out["Ghat"][0])
    print("  recovery error: %.4f" %
          np.mean((out["Ghat"][0] - fresh) ** 2))
    print("\nsoka(h) slokatvam agata(h) -- grief has become verse.")
    print("=" * 70)


if __name__ == "__main__":
    main()

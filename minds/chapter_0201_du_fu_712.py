#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 CHAPTER 0201 — DU FU (杜甫, 712–770)
 THE LÜSHI ENGINE: A Regulated-Witness Architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0201_du_fu_712 - Du Fu (712–770), Tang Dynasty
================================================================================  
 WHY THIS ARCHITECTURE
 ---------------------
 Du Fu did not leave a treatise on mind. He left ~1,400 poems, most of them
 written under the strictest formal system any major poet has ever worked in:
 regulated verse (律詩, lüshi). Three cognitive commitments are legible in
 that corpus, and this file turns each into a trainable mechanism:

 1. THE COUPLET IS THE UNIT OF THOUGHT (對仗, duizhang).
    In regulated verse every line must be answered by a counterpart line that
    simultaneously (a) OPPOSES it in tone (平/仄 — level vs. oblique),
    (b) PRESERVES its syntactic frame (noun answers noun, number answers
    number), and (c) RESONATES with it in sense. Liu Xie's dictum in the
    Wenxin diaolong — antithetical parallelism is the superior kind — is the
    design law here. Mechanism: a learned linear COUNTERPART OPERATOR that is
    regularized toward being an involution (applying it twice returns the
    original: the counterpart of the counterpart is the line itself).
    NOT attention over stored keys. NOT a transformer block. One matrix that
    must learn to be its own inverse while solving the task.

 2. PERCEPTION IS GATED BY THE STATE OF THE TIMES (感時, ganshi).
    "感時花濺淚" — moved by the times, the flowers themselves shed tears.
    For Du Fu the SAME scene encodes differently under a different historical
    affect. Mechanism: a slow, non-differentiable summary of the recent world
    (a running mean of inputs + a running level of ledger surprise) is mapped
    to a low-dimensional affect vector that multiplicatively re-scales AND
    additively re-shifts the latent scene code (a FiLM-style gate, but the
    conditioning signal is *history*, not a side input). Grief is not a
    post-hoc label on the output; it changes the encoding itself.

 3. THE MIND KEEPS AN APPEND-ONLY TESTIMONY (詩史, shishi).
    Meng Qi (late Tang) said of Du Fu: he set it all forth in his poems,
    "perhaps nothing was omitted" — hence Poet-Historian. Mechanism: a
    non-parametric ring ledger that stores every witnessed (scene, counterpart)
    pair. It is never trained, never edited, only appended. Its job is to
    predict by resonance (cosine-weighted recall) and to raise SURPRISE when
    the world stops matching the record — the model's own detector for the
    An Lushan rupture. Surprise feeds back into the affect gate: what the
    witness cannot reconcile becomes what the eye must carry.

 THE WORLD
 ---------
 A synthetic stream of "lines": vectors whose ground-truth counterparts obey
 a regime-dependent regulated rule (tone dims flip, frame dims persist, sense
 dims rotate). Halfway through the stream the world ruptures — the An Lushan
 step — and afterwards the stream ALTERNATES blocks of the wartime present
 with blocks of remembered peace, the Autumn-Meditations condition, while the
 input distribution never changes: no single line carries any mark of its
 regime. Only the felt history of how the world departs from the testimony
 can tell them apart. The model must (i) learn couplet completion, (ii)
 notice the rupture through its ledger, (iii) serve BOTH regimes at once —
 which a gateless twin provably cannot do better than by compromise.

 WHAT IS VERIFIED WHEN YOU RUN THIS FILE
 ---------------------------------------
   [T1] Finite-difference gradient check on EVERY parameter tensor
        (mandatory; rel. error < 1e-4 in float64).
   [T2] Real training: final task loss must fall well below the initial loss.
   [T3] Emergent couplet law: the counterpart operator is (near-)involutive
        after training — A(A(z)) ≈ z.
   [T4] Poet-historian rupture detection: ledger surprise spikes at the
        An Lushan step versus its pre-war median.
   [T5] Ablation: an identical twin with the affect gate severed adapts worse
        after the rupture. Feeling the times is functional, not decorative.
   [T6] Regulated decoding demo: tone-signs of predicted counterparts oppose
        the tone-signs of their lines (平 answers 仄), printed as glyphs.

 Pure NumPy. No frameworks. Deterministic seed. Runs in well under a minute.
================================================================================
"""

import numpy as np
import time

SEED = 712  # the year of his birth
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------------------
# Dimensions of the world and the mind
# ------------------------------------------------------------------------------
D_TONE, D_FRAME, D_SENSE = 8, 8, 8          # 平仄 | syntactic frame | sense
D_X = D_TONE + D_FRAME + D_SENSE            # a "line" is a 24-dim vector
D_Y = D_X                                    # its counterpart lives in the same space
D_H = 48                                     # hidden width of the scene encoder
D_Z = 32                                     # latent width (the inner line)
D_AFF_IN = D_Y + 1                           # affect reads: residual direction + surprise
D_AFF = 8                                    # the affect vector is small — a mood, not a memory
BATCH = 16
STEPS = 5000
RUPTURE_STEP = 2000                          # the An Lushan step (755)
LR = 3e-3
LAMBDA_INV = 0.60                            # weight of the couplet (involution) law
TAU_SURP  = 0.30                             # how quickly the ache is felt
TAU_RESID = 0.35                             # how quickly its direction is felt
WAR_BLOCK = MEM_BLOCK = 100                  # Kuizhou present / remembered Chang'an


# ==============================================================================
# 1. THE TANG WORLD — a stream of lines whose counterparts obey era law
# ==============================================================================
class TangWorld:
    """Generates (line, counterpart) pairs under a regime that ruptures once.

    The PEACE LAW is a true involution M (M∘M = identity) — the couplet ideal:
    the counterpart of the counterpart is the line itself.
        tone  : y_T = -x_T                (full tonal opposition; self-inverse)
        frame : y_F = +x_F                (frame preserved; self-inverse)
        sense : y_S = H · x_S             (a Householder REFLECTION; H² = I)
    A reflection, not a rotation, precisely because a rotation is NOT its own
    inverse and regulated verse insists the pairing be reversible.

    Steps < RUPTURE_STEP — Kaiyuan peace:     y = M x
    Steps >= RUPTURE_STEP — the rebellion and the exile that follows: long
    blocks of the wartime present alternate with blocks of remembered peace,
    the Autumn-Meditations condition (Kuizhou autumn bending back to the
    recalled capital):
        war    : y = WAR_GAIN · (M x) + g2       (the form distended and stained)
        memory : y = M x                         (the intact couplet law, replayed)
    The witness cannot fit one static map to both; only a mind that re-tints
    its rendering by the felt times can hold the single couplet law steady and
    still answer each block truly. A gateless twin must oscillate between the
    two and serve neither well.

    CRUCIALLY the input distribution never changes: a single line carries no
    mark of its era. Only the felt history — how far the world has lately left
    the testimony — tells peace from war. A gateless mind can only serve
    WAR_GAIN·Mx+g2 by pretending it is Mx, and the gap is irreducible. This is
    the exact sense in which, for Du Fu, feeling the times is a computational
    requirement of perceiving truly — not an ornament on the result.
    """

    def __init__(self, rng):
        self.rng = rng
        v = rng.standard_normal(D_SENSE)
        v = v / (np.linalg.norm(v) + 1e-9)
        self.H = np.eye(D_SENSE) - 2.0 * np.outer(v, v)   # Householder: H² = I
        self.g2 = rng.standard_normal(D_Y) * 1.3          # the war-colour bias
        self.WAR_GAIN = 1.85
        self.noise = 0.05

    def _peace_map(self, x):
        xt, xf, xs = x[:, :D_TONE], x[:, D_TONE:D_TONE + D_FRAME], x[:, -D_SENSE:]
        return np.concatenate([-xt, xf, xs @ self.H.T], axis=1)

    def regime(self, step):
        if step < RUPTURE_STEP:
            return "peace"
        phase = (step - RUPTURE_STEP) % (WAR_BLOCK + MEM_BLOCK)
        return "war" if phase < WAR_BLOCK else "memory"

    def sample(self, step, batch):
        reg = self.regime(step)
        x = self.rng.standard_normal((batch, D_X))         # same distribution, always
        y = self._peace_map(x)
        if reg == "war":
            y = self.WAR_GAIN * y + self.g2                 # the form distended and stained
        y = y + self.noise * self.rng.standard_normal((batch, D_Y))
        return x, y


# ==============================================================================
# 2. THE SHISHI LEDGER (詩史) — append-only testimony, recall by resonance
# ==============================================================================
class ShiShiLedger:
    """The poet-historian's memory. Never trained, never revised, only appended.

    Du Fu's testimony does not merely pile up leaves; read together, the poems
    imply the LAW OF THE AGE — what answers what, in this world. The ledger
    models that: it accumulates running sufficient statistics of every
    witnessed (line, counterpart) pair and, on demand, distils them into the
    rule the whole record implies (a closed-form least-squares reading — no
    gradients, no forgetting, nothing ever struck out). SURPRISE is the gap
    between what the record implies and what the world now delivers: the
    instrument reading that tells the witness the times have changed. After
    the rupture the record spans two incompatible worlds, so its implied law
    fits neither perfectly — the ledger carries the war in it forever, which
    is exactly what a shishi is.
    """

    def __init__(self, ridge=1.0):
        self.Sxx = ridge * np.eye(D_X + 1)      # +1: an intercept column
        self.Sxy = np.zeros((D_X + 1, D_Y))
        self.n = 0

    @staticmethod
    def _aug(x_batch):
        ones = np.ones((x_batch.shape[0], 1))
        return np.concatenate([x_batch, ones], axis=1)

    def append(self, x_batch, y_batch):
        Xa = self._aug(x_batch)
        self.Sxx += Xa.T @ Xa
        self.Sxy += Xa.T @ y_batch
        self.n += x_batch.shape[0]

    def implied_rule(self):
        """The law of the age, as the whole record implies it."""
        return np.linalg.solve(self.Sxx, self.Sxy)     # (D_X+1, D_Y)

    def recall(self, x_batch):
        if self.n < 4 * D_X:
            return None
        return self._aug(x_batch) @ self.implied_rule()

    def reading(self, x_batch, y_batch):
        """The witness's instrument: (surprise, residual direction).

        surprise — normalized error of the record's implied law vs. the world;
        residual — the MEAN VECTOR of that departure. Not just how far the
        world has left the testimony, but in which direction: the specific
        colour of the grief. This vector is the only signal in the whole
        system that can tell a wartime batch from a remembered-peace batch.
        """
        pred = self.recall(x_batch)
        if pred is None:
            return 0.0, np.zeros(D_Y)
        resid = y_batch - pred
        num = np.linalg.norm(resid, axis=1)
        den = np.linalg.norm(y_batch, axis=1) + 1e-9
        return float(np.mean(num / den)), resid.mean(axis=0)


# ==============================================================================
# 3. THE LÜSHI ENGINE — encoder, qing–jing gate, counterpart operator, decoder
# ==============================================================================
class LushiEngine:
    """The regulated witness.

    Forward pass (one 'line' x, given the felt-times context c = [resid; surprise]):

        h    = tanh(x W1 + b1)                  scene reading (景)
        z    = h W2 + b2                         the inner line
        ẑ    = z A                               the COUNTERPART OPERATOR (對),
                                                 kept a pure involution: A(A z)=z
        base = ẑ W3 + b3                          the answering line, in peace
        a    = tanh(c Wa + ba)                    the affect of the times (情)
        gain = 1 + a Ug                           feeling re-scales the rendering,
        stain=     a Ub                           and stains it with the war's colour
        ŷ    = base * gain + stain                qing–jing fusion (感時)

    The couplet law lives UPSTREAM of feeling: the pairing of line and
    counterpart is timeless and reversible (A is an involution). Feeling acts
    DOWNSTREAM, on what the mind renders from that pairing — 'moved by the
    times, the flowers shed tears.' So the same regulated perception, unshaken
    in its form, comes out gilded in peace and blood-stained in war.

    Losses:
        L_pred = mean (ŷ − y)²                    the couplet must answer truly
        L_inv  = mean (z A A − z)²                the couplet LAW (on z directly)
        L      = L_pred + λ · L_inv

    The affect context c is data (no gradient into history). All other grads
    are hand-derived and verified numerically in [T1]. `gate_on=False` builds
    the severed twin (gain≡1, stain≡0) for the ablation [T5].
    """

    PARAM_NAMES = ["W1", "b1", "W2", "b2", "Wa", "ba", "Ug", "Ub", "A", "W3", "b3"]

    def __init__(self, rng, gate_on=True):
        s = lambda *sh: rng.standard_normal(sh)
        self.p = {
            "W1": s(D_X, D_H) * np.sqrt(1.0 / D_X),
            "b1": np.zeros(D_H),
            "W2": s(D_H, D_Z) * np.sqrt(1.0 / D_H),
            "b2": np.zeros(D_Z),
            "Wa": s(D_AFF_IN, D_AFF) * np.sqrt(1.0 / D_AFF_IN),
            "ba": np.zeros(D_AFF),
            "Ug": s(D_AFF, D_Y) * 0.05,               # affect -> output gain
            "Ub": s(D_AFF, D_Y) * 0.05,               # affect -> output stain
            "A":  np.eye(D_Z) + s(D_Z, D_Z) * 0.05,   # near identity; must EARN opposition
            "W3": s(D_Z, D_Y) * np.sqrt(1.0 / D_Z),
            "b3": np.zeros(D_Y),
        }
        self.gate_on = gate_on
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------------- forward ----------------
    def forward(self, x, c, keep=False):
        p = self.p
        h = np.tanh(x @ p["W1"] + p["b1"])
        z = h @ p["W2"] + p["b2"]
        zhat = z @ p["A"]
        base = zhat @ p["W3"] + p["b3"]
        a = np.tanh(c @ p["Wa"] + p["ba"])             # (D_AFF,)
        gate = 1.0 if self.gate_on else 0.0
        gain = 1.0 + gate * (a @ p["Ug"])              # (D_Y,)
        stain = gate * (a @ p["Ub"])                   # (D_Y,)
        yhat = base * gain + stain
        if keep:
            self._c = (x, c, h, z, zhat, base, a, gain)
        return yhat, z

    # ---------------- loss + hand-derived backprop ----------------
    def loss_and_grads(self, x, c, y):
        p = self.p
        yhat, z = self.forward(x, c, keep=True)
        x_, c_, h, z_, zhat, base, a, gain = self._c
        gate = 1.0 if self.gate_on else 0.0

        diff = yhat - y
        L_pred = float(np.mean(diff ** 2))

        # involution on z: zz = z A A must return z
        pmid = z @ p["A"]
        zz = pmid @ p["A"]
        idiff = zz - z
        L_inv = float(np.mean(idiff ** 2))
        L = L_pred + LAMBDA_INV * L_inv

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- prediction path ----
        dyhat = 2.0 * diff / diff.size
        # yhat = base*gain + stain
        dbase = dyhat * gain
        dgain = (dyhat * base).sum(axis=0)             # (D_Y,)
        dstain = dyhat.sum(axis=0)                      # (D_Y,)
        g["Ug"] = gate * np.outer(a, dgain)
        g["Ub"] = gate * np.outer(a, dstain)
        da = gate * (p["Ug"] @ dgain + p["Ub"] @ dstain)   # (D_AFF,)
        dpre_a = da * (1.0 - a ** 2)
        g["Wa"] = np.outer(c, dpre_a)
        g["ba"] = dpre_a
        # base = zhat W3 + b3
        g["W3"] = zhat.T @ dbase
        g["b3"] = dbase.sum(axis=0)
        dzhat = dbase @ p["W3"].T
        # zhat = z A
        g["A"] += z.T @ dzhat
        dz = dzhat @ p["A"].T

        # ---- involution path (on z) ----
        dzz = (2.0 * LAMBDA_INV / idiff.size) * idiff
        g["A"] += pmid.T @ dzz
        dpmid = dzz @ p["A"].T
        g["A"] += z.T @ dpmid
        dz = dz + dpmid @ p["A"].T - dzz               # −z term inside (zz − z)

        # ---- encoder ----
        g["W2"] = h.T @ dz
        g["b2"] = dz.sum(axis=0)
        dh = dz @ p["W2"].T
        dpre1 = dh * (1.0 - h ** 2)
        g["W1"] = x.T @ dpre1
        g["b1"] = dpre1.sum(axis=0)

        return L, L_pred, L_inv, g

    # ---------------- Adam ----------------
    def step(self, grads, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ---------------- diagnostics ----------------
    def involution_residual(self, x, c):
        _, z = self.forward(x, c)
        zz = (z @ self.p["A"]) @ self.p["A"]
        return float(np.linalg.norm(zz - z) / (np.linalg.norm(z) + 1e-9))


# ==============================================================================
# 4. [T1] FINITE-DIFFERENCE GRADIENT CHECK — mandatory before any training
# ==============================================================================
def gradient_check(verbose=True):
    """Central-difference check on a handful of entries of EVERY tensor."""
    g_rng = np.random.default_rng(4457)
    eng = LushiEngine(g_rng, gate_on=True)
    # float64 copies for numerical stability
    for k in eng.p:
        eng.p[k] = eng.p[k].astype(np.float64)
    x = g_rng.standard_normal((4, D_X))
    c = g_rng.standard_normal(D_AFF_IN)
    y = g_rng.standard_normal((4, D_Y))
    _, _, _, grads = eng.loss_and_grads(x, c, y)
    h = 1e-5
    worst = 0.0
    for k in eng.PARAM_NAMES:
        P = eng.p[k]
        flat = P.reshape(-1)
        gflat = grads[k].reshape(-1)
        n_probe = min(6, flat.size)
        idxs = g_rng.choice(flat.size, size=n_probe, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + h
            Lp, _, _, _ = eng.loss_and_grads(x, c, y)
            flat[i] = orig - h
            Lm, _, _, _ = eng.loss_and_grads(x, c, y)
            flat[i] = orig
            num = (Lp - Lm) / (2 * h)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
        if verbose:
            print(f"    grad-check  {k:>3s}  worst rel.err so far = {worst:.3e}")
    return worst


# ==============================================================================
# 5. TRAINING THROUGH THE RUPTURE — one life, 712 → 770, war in the middle
# ==============================================================================
def run_life(gate_on, rng_world, rng_model, log_every=500, tag=""):
    world = TangWorld(rng_world)
    eng = LushiEngine(rng_model, gate_on=gate_on)
    ledger = ShiShiLedger()

    ema_surp = 0.0                 # the slow ache of the record failing
    ema_resid = np.zeros(D_Y)      # the direction in which it fails
    losses, surprises = [], []

    for step in range(STEPS):
        x, y = world.sample(step, BATCH)
        c = np.concatenate([ema_resid, [ema_surp]])          # the times, as felt
        L, Lp, Li, grads = eng.loss_and_grads(x, c, y)
        eng.step(grads)

        # the witness measures the world against the record, then writes
        s, r = ledger.reading(x, y)
        ledger.append(x, y)

        # the times seep in (stop-gradient history; the model never edits it)
        ema_surp = (1 - TAU_SURP) * ema_surp + TAU_SURP * s
        ema_resid = (1 - TAU_RESID) * ema_resid + TAU_RESID * r

        losses.append(Lp)
        surprises.append(s)
        if log_every and (step % log_every == 0 or step == STEPS - 1):
            print(f"    [{tag}] step {step:4d} ({world.regime(step):>6s})  "
                  f"pred={Lp:.4f}  inv={Li:.4f}  surprise={s:.3f}")
    return eng, np.array(losses), np.array(surprises), world


# ==============================================================================
# 6. [T6] REGULATED DECODING DEMO — the couplet as glyphs
# ==============================================================================
def tone_glyphs(vec_tone):
    """平 (level, ●) for positive tone dims, 仄 (oblique, ○) for negative."""
    return "".join("平" if v >= 0 else "仄" for v in vec_tone)


def couplet_demo(eng, world, ema_resid, ema_surp, n=3):
    print("\n  Regulated decoding — tone-signs of line vs. predicted counterpart")
    print("  (in Era-1 law, every 平 must be answered by 仄 and vice versa):")
    ok_frac = []
    c = np.concatenate([ema_resid, [ema_surp]])
    x, y = world.sample(0, n)                    # era-1 lines
    yhat, _ = eng.forward(x, c)
    for i in range(n):
        line = tone_glyphs(x[i, :D_TONE])
        answ = tone_glyphs(yhat[i, :D_TONE])
        opp = np.mean(np.sign(x[i, :D_TONE]) != np.sign(yhat[i, :D_TONE]))
        ok_frac.append(opp)
        print(f"      line       {line}")
        print(f"      counterpart{answ}   opposition {opp*100:.0f}%\n")
    return float(np.mean(ok_frac))


# ==============================================================================
# 7. THE FULL TRIAL — every claim in the chapter must pass here
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print(" THE LÜSHI ENGINE — Du Fu (712–770) — self-verifying run")
    print("=" * 78)

    # ---------- [T1] gradient check ----------
    print("\n[T1] Finite-difference gradient check (all parameter tensors)")
    worst = gradient_check()
    assert worst < 1e-4, f"gradient check FAILED: worst rel.err {worst:.3e}"
    print(f"  PASS — worst relative error {worst:.3e} < 1e-4")

    # ---------- main life: gated witness ----------
    print("\n[T2] Training the regulated witness through the An Lushan rupture")
    eng, losses, surprises, world = run_life(
        gate_on=True,
        rng_world=np.random.default_rng(755),
        rng_model=np.random.default_rng(712),
        tag="witness")
    init_loss = float(np.mean(losses[:20]))
    final_loss = float(np.mean(losses[-200:]))
    print(f"  initial pred loss ~ {init_loss:.4f}   final ~ {final_loss:.4f}")
    assert final_loss < 0.35 * init_loss, "training FAILED to reduce loss enough"
    print(f"  PASS — final loss is {final_loss/init_loss*100:.1f}% of initial")

    # ---------- [T3] involution ----------
    print("\n[T3] The couplet law: counterpart of the counterpart returns the line")
    xs, _ = world.sample(0, 64)
    resid = eng.involution_residual(xs, np.zeros(D_AFF_IN))
    print(f"  ||A(A z) − z|| / ||z|| = {resid:.4f}")
    assert resid < 0.15, "involution residual too large — couplet law not learned"
    print("  PASS — the counterpart operator is near-involutive")

    # ---------- [T4] rupture detection ----------
    print("\n[T4] Poet-historian surprise at the An Lushan step")
    pre = np.median(surprises[1200:2000])
    spike = float(np.max(surprises[RUPTURE_STEP:RUPTURE_STEP + 120]))
    print(f"  pre-war median surprise = {pre:.3f}   post-rupture peak = {spike:.3f}")
    assert spike > 4.0 * pre, "ledger failed to register the rupture"
    print(f"  PASS — surprise spiked ×{spike/max(pre,1e-9):.1f} at the rupture")

    # ---------- [T5] ablation: sever the gate ----------
    print("\n[T5] Ablation — an identical twin with the qing–jing gate severed")
    _, losses0, _, _ = run_life(
        gate_on=False,
        rng_world=np.random.default_rng(755),     # SAME world
        rng_model=np.random.default_rng(712),     # SAME initial mind
        log_every=1000, tag="severed")
    w = slice(RUPTURE_STEP + 300, STEPS)          # after both twins' transients
    gated = float(np.mean(losses[w]))
    severed = float(np.mean(losses0[w]))
    print(f"  post-rupture mean loss  gated={gated:.4f}   severed={severed:.4f}")
    assert gated < 0.80 * severed, "gate gave no decisive advantage"
    print(f"  PASS — feeling the times cuts wartime error by "
          f"{(1 - gated/severed)*100:.1f}%")

    # ---------- [T6] regulated decoding ----------
    opp = couplet_demo(eng, world, np.zeros(D_Y), 0.0)   # peace-time affect
    assert opp > 0.9, "tonal opposition not achieved in decoding"
    print(f"[T6] PASS — mean tonal opposition {opp*100:.0f}% (threshold 90%)")

    print("\n" + "=" * 78)
    print(f" ALL TESTS PASSED — {time.time()-t0:.1f}s.  語不驚人死不休 —")
    print(" 'if my words startle no one, I will not rest, even in death.'")
    print("=" * 78)


if __name__ == "__main__":
    main()

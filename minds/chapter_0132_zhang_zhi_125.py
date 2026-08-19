#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
Mind 132 —  Zhang Zhi (張芝, courtesy name Boying 伯英; d. 192 CE)
             The "Sage of Cursive" (草聖) of the Eastern Han.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 7 Minds 121 - 140 Available on Amazon https://www.amazon.com/dp/B0HFN6GXMH
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 132: Zhang Zhi (張芝, courtesy name Boying 伯英; d. 192 CE)
================================================================================   

THE ONE-STROKE CONTINUOUS-TRACE NETWORK  (a from-scratch NumPy architecture)
----------------------------------------------------------------------------

Zhang Zhi left no surviving treatise (his five-chapter *Bi Xin Lun* is lost)
and only a single authentic aphorism survives in his own voice:

        匆匆不暇草書  —  "when hurried, one has no leisure for cursive."

That one line is the seed of this whole model. Cursive (草書, "grass script")
is the FAST script — it was invented to save time — and yet Zhang Zhi says it
cannot be written when you are in a hurry. The resolution of the paradox is a
precise cognitive claim: fluency is not speed of execution but the *absence of
internal haste*. A master moves fast in wall-clock time while remaining, on the
inside, completely unhurried and committed.

From this single idea the architecture is built. It rejects the modern default
(a Transformer re-reading stored tokens and revising its guess) because that
picture is exactly what Zhang Zhi's grass script is NOT. His script is:

  (1) ONE CONTINUOUS TRACE, not a sequence of discrete, separable tokens.
      Characters bleed into one another; the natural unit is a trajectory,
      not a symbol. => a continuous-time recurrent generator, not attention.

  (2) IRREVERSIBLE. Once the brush moves, the stroke cannot be recalled or
      revised. => the model commits forward; its training pressure is toward
      being right on the first pass (a "commitment" cost on late corrections).

  (3) SEEDED BY INTENTION (意, *yi*). The idea precedes and animates the brush;
      quality of output = quality of the pre-stroke intention. => the whole
      trajectory is unrolled from a single intention vector.

  (4) GOVERNED BY ONE UNIFYING THREAD (綱, *gang* — the master rope of the net).
      A single carried hidden state holds the piece together across the stroke.

  (5) TRANQUILITY-COUPLED. A learned "tranquility" gate sets how much each
      instant may deviate; at inference we can inject *haste*, and — exactly as
      the aphorism predicts — the trace degrades into incoherent jitter.

  (6) KNOWING == DOING. There is no separate "plan" that is then executed. The
      same hidden thread that *writes* the stroke also *identifies* it (a small
      recognition read-out off the final state). Understanding is inseparable
      from the capacity to produce the trace.

WHAT THE NETWORK ACTUALLY DOES (a real, small, trainable task)
----------------------------------------------------------------------------
Given a class label (a "character" to write), the network unrolls a single 2D
brush trajectory (a continuous path of pen positions) in one committed pass,
and simultaneously must let that trajectory's final internal state re-identify
which character it wrote. It is trained end-to-end by backprop-through-time on:

    L = L_write   (mean-squared error to the target stroke path)
      + λ_id * L_identify   (cross-entropy: the trace encodes its own identity)
      + λ_commit * L_commit (penalty on jerk = no late revision, one flow)

Everything is hand-derived analytic gradient + a MANDATORY finite-difference
gradient check (must pass) + a real Adam training loop + self-tests, and a
demonstration of the "too busy to write cursive" degradation under injected
haste. Pure NumPy, from scratch. Run this file directly to reproduce.

Author's note: the enums / part-names below (Intention, Gang thread, Tranquility
gate, Brush read-out, Identity read-out) name the *mechanisms*, so the prose of
the chapter can refer to them without ever touching this code.
============================================================================
"""

import numpy as np

RNG = np.random.default_rng(20250702)  # fixed seed => reproducible run


# ============================================================================
# 1.  SYNTHETIC "GRASS-SCRIPT" DATA
#     Each class is a canonical flowing 2D stroke path (a smooth cursive glyph).
#     We build them from small Fourier/parametric curves so every target is a
#     single continuous, differentiable trajectory — the thing the brush emits.
# ============================================================================

def make_glyph_prototypes(n_classes: int, T: int) -> np.ndarray:
    """Return an (n_classes, T, 2) array of canonical stroke paths.

    Each glyph is a smooth open curve traced by one continuous gesture, built
    from a couple of sinusoidal harmonics with class-specific phase/amplitude.
    These stand in for distinct cursive characters written in one stroke.
    """
    t = np.linspace(0.0, 1.0, T)  # normalized "brush time"
    protos = np.zeros((n_classes, T, 2), dtype=np.float64)
    for c in range(n_classes):
        # class-specific harmonic recipe (kept deterministic & well-separated)
        a1 = 0.6 + 0.25 * np.cos(2.0 * c)
        a2 = 0.35 * np.sin(1.3 * c + 0.7)
        p1 = 0.5 * c
        p2 = 1.1 * c + 0.4
        w1 = 1.0 + (c % 3)          # base sweep count varies by class
        w2 = 2.0 + ((c + 1) % 3)
        x = a1 * np.sin(2 * np.pi * w1 * t + p1) + 0.15 * t
        y = a2 * np.sin(2 * np.pi * w2 * t + p2) + 0.9 * t - 0.45
        protos[c, :, 0] = x
        protos[c, :, 1] = y
    return protos


def make_dataset(protos: np.ndarray, n_per_class: int, noise: float):
    """Jittered copies of each prototype path => (labels, targets)."""
    n_classes, T, _ = protos.shape
    X_lab, Y_path = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            jitter = noise * RNG.standard_normal((T, 2))
            # low-pass the jitter so targets stay smooth, continuous strokes
            jitter = np.cumsum(jitter, axis=0) * 0.15
            jitter -= jitter.mean(axis=0, keepdims=True)
            Y_path.append(protos[c] + jitter)
            X_lab.append(c)
    X_lab = np.array(X_lab, dtype=np.int64)
    Y_path = np.array(Y_path, dtype=np.float64)
    perm = RNG.permutation(len(X_lab))
    return X_lab[perm], Y_path[perm]


# ============================================================================
# 2.  THE MODEL  —  parameters and helpers
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(n_classes: int, H: int, seed_scale: float = 0.5):
    """Xavier-ish small init for every learnable array.

    Named tensors (mechanism -> symbol):
      Intention (意)   : Wz, bz     label -> intention vector z
      Gang seed (綱)   : Wh0, bh0   intention -> initial unifying thread h0
      Recurrence       : Whh, Wzh, bh   the sustained single thread
      Tranquility gate : w_tau, b_tau   scalar deviation gate tau in (0,1)
      Brush read-out   : Wout, bout  hidden -> 2D pen velocity
      Identity read-out: Wcls, bcls  final thread -> class logits (knowing==doing)
    """
    def R(*shape):
        fan = shape[-1] if len(shape) > 1 else shape[0]
        return RNG.standard_normal(shape) * (seed_scale / np.sqrt(fan))

    P = {
        "Wz":  R(H, n_classes), "bz":  np.zeros(H),
        "Wh0": R(H, H),         "bh0": np.zeros(H),
        "Whh": R(H, H),         "Wzh": R(H, H), "bh": np.zeros(H),
        "w_tau": R(H),          "b_tau": np.array(0.5),   # bias toward tranquil
        "Wout": R(2, H),        "bout": np.zeros(2),
        "Wcls": R(n_classes, H),"bcls": np.zeros(n_classes),
    }
    return P


# ============================================================================
# 3.  FORWARD PASS  (single sample)
#     intention -> gang seed -> T committed steps -> path + identity
# ============================================================================

def forward(P, c_label, n_classes, T, haste: float = 0.0, noise_rng=None):
    """Unroll ONE continuous committed trace for class `c_label`.

    haste>0 injects noise into each increment (inference-time demonstration of
    "too busy to write cursive"); haste=0 is deterministic (used for training
    and for the gradient check).
    Returns (outputs, cache) where outputs has the path, logits, and hidden run.
    """
    H = P["bh"].shape[0]
    y = np.zeros(n_classes); y[c_label] = 1.0            # one-hot label

    u_z = P["Wz"] @ y + P["bz"]
    z = np.tanh(u_z)                                     # intention 意

    u_h0 = P["Wh0"] @ z + P["bh0"]
    h0 = np.tanh(u_h0)                                   # gang thread seed 綱

    u_tau = P["w_tau"] @ z + P["b_tau"]
    tau = sigmoid(u_tau)                                 # tranquility gate

    hs = [h0]          # h_0 .. h_T
    gs = []            # g_1 .. g_T  (tanh candidates)
    vs = []            # v_1 .. v_T  (pen velocities)
    ps = []            # p_1 .. p_T  (pen positions)
    p_prev = np.zeros(2)
    for t in range(T):
        a_t = P["Whh"] @ hs[-1] + P["Wzh"] @ z + P["bh"]
        g_t = np.tanh(a_t)
        h_t = hs[-1] + tau * g_t                         # single unbroken thread
        if haste > 0.0 and noise_rng is not None:
            # haste corrupts the increment: the unhurried flow breaks apart
            h_t = h_t + haste * noise_rng.standard_normal(H)
        v_t = P["Wout"] @ h_t + P["bout"]                # brush velocity
        p_t = p_prev + v_t                               # integrate -> position
        hs.append(h_t); gs.append(g_t); vs.append(v_t); ps.append(p_t)
        p_prev = p_t

    hT = hs[-1]
    logits = P["Wcls"] @ hT + P["bcls"]                  # identity read-out

    cache = dict(y=y, z=z, u_z=u_z, h0=h0, u_h0=u_h0, tau=tau, u_tau=u_tau,
                 hs=hs, gs=gs, vs=vs, ps=np.array(ps), logits=logits)
    return cache


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ============================================================================
# 4.  LOSS + ANALYTIC BACKWARD  (backprop-through-time, hand-derived)
# ============================================================================

def loss_and_grad(P, batch_labels, batch_paths, n_classes, T,
                  lam_id=0.5, lam_commit=0.15):
    """Compute mean loss over a batch and the analytic gradient wrt every param.

    L = L_write + lam_id * L_identify + lam_commit * L_commit
        L_write    = mean_t || p_t - target_t ||^2
        L_identify = cross_entropy( softmax(logits), label )
        L_commit   = mean_t || v_t - v_{t-1} ||^2      (no late revision; one flow)
    """
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    total = 0.0
    B = len(batch_labels)

    for c_label, target in zip(batch_labels, batch_paths):
        cache = forward(P, c_label, n_classes, T, haste=0.0)
        z, h0, tau = cache["z"], cache["h0"], cache["tau"]
        hs, gs = cache["hs"], cache["gs"]
        vs = cache["vs"]; ps = cache["ps"]; logits = cache["logits"]

        # ---- forward losses ----
        diff = ps - target                              # (T,2)
        L_write = np.mean(np.sum(diff * diff, axis=1))
        sm = softmax(logits)
        L_id = -np.log(sm[c_label] + 1e-12)
        # commitment: jerk of velocity (v_0 := 0)
        v_arr = np.array(vs)                            # (T,2)
        v_prev = np.vstack([np.zeros((1, 2)), v_arr[:-1]])
        dv = v_arr - v_prev
        L_commit = np.mean(np.sum(dv * dv, axis=1))
        total += L_write + lam_id * L_id + lam_commit * L_commit

        # ---- backward ----
        # d L_write / d p_t  = (2/T)(p_t - target_t)
        dps = (2.0 / T) * diff                          # (T,2)
        # p_t = sum_{s<=t} v_s  => dL_write/dv_s = sum_{t>=s} dps[t]
        dvs_write = np.cumsum(dps[::-1], axis=0)[::-1]  # suffix sum, (T,2)

        # d L_commit / d v_t : from term t and term t+1
        # L_commit = (1/T) sum_t ||v_t - v_{t-1}||^2
        dvs_commit = np.zeros((T, 2))
        for t in range(T):
            term_t = 2.0 * dv[t]                        # d||v_t-v_{t-1}||^2 / dv_t
            if t + 1 < T:
                term_next = -2.0 * dv[t + 1]            # d||v_{t+1}-v_t||^2 / dv_t
            else:
                term_next = 0.0
            dvs_commit[t] = (1.0 / T) * (term_t + term_next)

        dvs = dvs_write + lam_commit * dvs_commit       # (T,2) total dL/dv_t

        # identity read-out grads
        dlogits = lam_id * (sm.copy())
        dlogits[c_label] -= lam_id
        grads["Wcls"] += np.outer(dlogits, hs[-1])
        grads["bcls"] += dlogits
        dhT_from_id = P["Wcls"].T @ dlogits             # into final hidden

        # BPTT over hidden states
        dh_next = np.zeros(P["bh"].shape)               # adjoint carried backward
        dz = np.zeros_like(z)
        dtau = 0.0
        # add identity gradient onto the final hidden adjoint
        dh_next += dhT_from_id
        for t in reversed(range(T)):
            h_prev = hs[t]                              # h_{t} feeding step t+1 output
            g_t = gs[t]
            # v_t = Wout h_{t+1} + bout ; h index: hs[t+1] is post-step hidden
            h_t = hs[t + 1]
            dv_t = dvs[t]
            grads["Wout"] += np.outer(dv_t, h_t)
            grads["bout"] += dv_t
            dh_t = P["Wout"].T @ dv_t + dh_next         # total adjoint at h_t
            # h_t = h_prev + tau * g_t
            dtau += float(dh_t @ g_t)
            dg_t = dh_t * tau
            da_t = dg_t * (1.0 - g_t * g_t)             # through tanh
            grads["Whh"] += np.outer(da_t, h_prev)
            grads["Wzh"] += np.outer(da_t, z)
            grads["bh"]  += da_t
            dz += P["Wzh"].T @ da_t
            # pass-through to h_prev: identity term + recurrence term
            dh_next = dh_t + P["Whh"].T @ da_t

        # h0 adjoint is whatever dh_next holds after the loop
        dh0 = dh_next
        dudh0 = dh0 * (1.0 - h0 * h0)
        grads["Wh0"] += np.outer(dudh0, z)
        grads["bh0"] += dudh0
        dz += P["Wh0"].T @ dudh0

        # tranquility gate: tau = sigmoid(w_tau . z + b_tau)
        dudtau = dtau * tau * (1.0 - tau)
        grads["w_tau"] += dudtau * z
        grads["b_tau"] += dudtau
        dz += dudtau * P["w_tau"]

        # intention: z = tanh(Wz y + bz)
        dudz = dz * (1.0 - z * z)
        grads["Wz"] += np.outer(dudz, cache["y"])
        grads["bz"] += dudz

    for k in grads:
        grads[k] = grads[k] / B
    return total / B, grads


# ============================================================================
# 5.  MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ============================================================================

def gradient_check(n_classes=3, H=6, T=8, eps=1e-6, tol=1e-5):
    """Compare analytic grads with central finite differences on every tensor.

    Returns the worst relative error found. Must be < tol for the model to be
    trusted. (Small H/T so the check is fast and numerically clean in float64.)
    """
    P = init_params(n_classes, H)
    protos = make_glyph_prototypes(n_classes, T)
    labels, paths = make_dataset(protos, n_per_class=2, noise=0.02)
    labels, paths = labels[:4], paths[:4]              # tiny batch

    _, grads = loss_and_grad(P, labels, paths, n_classes, T)

    worst = 0.0
    worst_where = None
    for name, arr in P.items():
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        # check a handful of coordinates per tensor (all, if small)
        idxs = range(flat.size) if flat.size <= 12 else RNG.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = loss_and_grad(P, labels, paths, n_classes, T)
            flat[i] = orig - eps
            lm, _ = loss_and_grad(P, labels, paths, n_classes, T)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst = rel; worst_where = (name, int(i), ana, num)
    return worst, worst_where


# ============================================================================
# 6.  ADAM OPTIMISER + TRAINING LOOP
# ============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def accuracy(P, labels, paths, n_classes, T):
    correct = 0
    for c_label, _ in zip(labels, paths):
        cache = forward(P, c_label, n_classes, T, haste=0.0)
        if int(np.argmax(cache["logits"])) == int(c_label):
            correct += 1
    return correct / len(labels)


def train(n_classes=4, H=32, T=24, epochs=60, batch=16,
          n_per_class=40, noise=0.05, lr=4e-3, verbose=True):
    protos = make_glyph_prototypes(n_classes, T)
    labels, paths = make_dataset(protos, n_per_class, noise)
    n = len(labels)
    split = int(0.8 * n)
    tr_l, tr_p = labels[:split], paths[:split]
    te_l, te_p = labels[split:], paths[split:]

    P = init_params(n_classes, H)
    opt = Adam(P, lr=lr)

    history = []
    for ep in range(epochs):
        order = RNG.permutation(len(tr_l))
        ep_loss = 0.0; nb = 0
        for i in range(0, len(tr_l), batch):
            bl = tr_l[order[i:i + batch]]
            bp = tr_p[order[i:i + batch]]
            loss, grads = loss_and_grad(P, bl, bp, n_classes, T)
            opt.step(P, grads)
            ep_loss += loss; nb += 1
        tr_acc = accuracy(P, tr_l, tr_p, n_classes, T)
        te_acc = accuracy(P, te_l, te_p, n_classes, T)
        history.append((ep_loss / nb, tr_acc, te_acc))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} | loss {ep_loss/nb:7.4f} "
                  f"| train_acc {tr_acc:5.2f} | test_acc {te_acc:5.2f}")
    return P, history, (te_l, te_p), protos


# ============================================================================
# 7.  "TOO BUSY TO WRITE CURSIVE" — the haste degradation demonstration
#     As injected haste rises, the committed trace loses its smoothness (its
#     jerk explodes) even though the same trained weights are used. The model
#     reproduces Zhang Zhi's aphorism as a measurable phenomenon.
# ============================================================================

def path_jerk(path: np.ndarray) -> float:
    """Mean squared second-difference of the path = how un-smooth / hurried."""
    v = np.diff(path, axis=0)
    a = np.diff(v, axis=0)
    return float(np.mean(np.sum(a * a, axis=1)))


def haste_demo(P, n_classes, T, hastes=(0.0, 0.05, 0.15, 0.35, 0.6)):
    rng = np.random.default_rng(7)
    rows = []
    for h in hastes:
        jerks = []
        for c in range(n_classes):
            # average several noisy draws at this haste level
            j = np.mean([path_jerk(forward(P, c, n_classes, T, haste=h,
                                           noise_rng=rng)["ps"]) for _ in range(8)])
            jerks.append(j)
        rows.append((h, float(np.mean(jerks))))
    return rows


# ============================================================================
# 8.  MAIN  —  run gradient check, train, self-tests, haste demo
# ============================================================================

def main():
    print("=" * 74)
    print("Mind 117 — Zhang Zhi | One-Stroke Continuous-Trace Network")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check (mandatory)")
    worst, where = gradient_check()
    print(f"    worst relative error = {worst:.3e}  at {where[0]}[{where[1]}]"
          f"  (analytic={where[2]:+.5e}, numeric={where[3]:+.5e})")
    grad_ok = worst < 1e-5
    print(f"    gradient check {'PASSED' if grad_ok else 'FAILED'} (tol=1e-5)")

    print("\n[2] Training the brush (backprop-through-time)")
    P, history, (te_l, te_p), protos = train()
    first_loss = history[0][0]; last_loss = history[-1][0]
    final_test_acc = history[-1][2]
    print(f"    loss {first_loss:.4f} -> {last_loss:.4f}"
          f"   final test accuracy = {final_test_acc:.2f}")

    print("\n[3] 'Too busy to write cursive' — haste degradation")
    rows = haste_demo(P, n_classes=4, T=24)
    base = rows[0][1]
    for h, jerk in rows:
        bar = "#" * int(min(60, 60 * jerk / (rows[-1][1] + 1e-9)))
        print(f"    haste={h:4.2f} | mean path jerk={jerk:8.4f} | {bar}")
    haste_ok = rows[-1][1] > 3.0 * base   # high haste clearly breaks the flow

    print("\n[4] Self-tests")
    checks = {
        "gradient_check_passes": grad_ok,
        "loss_decreased":        last_loss < 0.5 * first_loss,
        "learns_above_chance":   final_test_acc >= 0.80,
        "haste_breaks_flow":     haste_ok,
    }
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(checks.values())
    print("\n" + "=" * 74)
    print("ALL SELF-TESTS PASSED" if all_ok else "SOME SELF-TESTS FAILED")
    print("=" * 74)
    return all_ok


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)

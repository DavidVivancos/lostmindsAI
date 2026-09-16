#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0197_abu_hanifa_699 - Abu Hanifa al-Nu'man ibn Thabit  (c. 699–767 CE / 80–150 AH)
# Kufa, Iraq · Persian mawla in the early Islamic (Arab-Islamic) world
# Eponym of the Hanafi school of Sunni jurisprudence
# =============================================================================
#  THE MIND, TURNED INTO A MECHANISM
#
#  Abu Hanifa's cognition is not "impose order on chaos." It is almost the
#  opposite: an intelligence defined by KNOWING THE LIMITS OF ITS OWN RULES.
#  Three moves, all attested in the historical record, form his signature:
#
#    1. QIYAS  (analogical structuralism) — extend a known ruling to a new
#       case by matching the shared *effective cause* ('illa). This is
#       disciplined generalisation over precedent.
#
#    2. ISTIHSAN  ("juristic preference") — a trained, frankly SUBJECTIVE
#       override that vetoes the analogy when it would produce hardship or
#       injustice, choosing the more equitable ruling instead. Abu Hanifa
#       trusted this override so far that even his own students pulled back
#       from it (El Shamsy / Iraqi legal-logic scholarship).
#
#    3. IRJA'  (deferral of judgment) — where evidence is underdetermined,
#       he refused to issue a confident verdict and left the matter "to God."
#       Faith is not nullified by a sin whose status is unclear; classification
#       is *suspended*, not forced. This is calibrated abstention.
#
#  And wrapping all three: SHURA / IJMA — his rulings were not a solo oracle
#  but the product of a council of ~40 students debating a question for months
#  toward consensus. The mind is a deliberating body, not a single head.
#
#  So this file does NOT build a transformer. It builds a case-based
#  analogical reasoner with (a) a learned equity-override GATE and (b) a
#  learned selective-abstention HEAD, then aggregates an ensemble of such
#  reasoners by confidence-weighted consensus. Everything is pure NumPy,
#  hand-derived backprop, with a finite-difference gradient check that MUST
#  pass, a real training loop, and self-tests.
#
#  Ruling classes (toy fiqh, 4-way): 0=permitted 1=forbidden 2=obligatory 3=discouraged
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
#  0. Small numerical helpers
# -----------------------------------------------------------------------------

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def tanh(x):
    return np.tanh(x)

def onehot(y, C):
    M = np.zeros((len(y), C))
    M[np.arange(len(y)), y] = 1.0
    return M


# =============================================================================
#  1. THE ISTIHSAN REASONER  (a single "scholar" / faqih)
#
#  Forward pass, per example x in R^d:
#     a1 = W1 x + b1 ; h1 = tanh(a1)          # perceive the case
#     a2 = W2 h1 + b2 ; h2 = tanh(a2)         # h2 = the case's 'illa profile
#     s  = K h2 ; att = softmax(s)            # QIYAS: match precedent causes
#     q  = Vᵀ att                             #   -> base analogical ruling logits
#     g  = sigmoid(tau*(wg·h2 + bg))          # ISTIHSAN gate: "does analogy harm?"
#     e  = We h2 + be                         #   -> the equitable alternative ruling
#     z  = (1-g) q + g e                      # blend precedent with equity
#     p  = sigmoid(wd·h2 + bd)                # IRJA': probability of DEFERRING
#
#  `tau` is the istihsan *boldness*: higher tau = a jurist who trusts the
#  override more readily. It is a knob, matching the historical observation
#  that Abu Hanifa's istihsan was bolder than his successors'.
# =============================================================================

class IstihsanReasoner:
    def __init__(self, d, H, P, C, tau=1.5, defer_cost=0.9, aux_qiyas=0.5,
                 gate_l1=0.04, seed=0):
        rng = np.random.default_rng(seed)
        # He-style init; ascontiguousarray so params().reshape(-1) yields a *view*
        # (essential for the finite-difference gradient check to perturb in place)
        sc = lambda out, inp: np.ascontiguousarray(rng.standard_normal((out, inp)) * np.sqrt(2.0 / inp))
        # trunk (case perception -> 'illa embedding h2)
        self.W1 = sc(H, d)             # (H,d)
        self.b1 = np.zeros(H)
        self.W2 = sc(H, H)             # (H,H)
        self.b2 = np.zeros(H)
        # QIYAS precedent bank: P prototype causes (keys) and their rulings (values)
        self.K = sc(P, H)              # (P,H)  precedent 'illa keys
        self.V = np.ascontiguousarray(rng.standard_normal((P, C)) * 0.5)   # (P,C) precedent ruling logits
        # ISTIHSAN gate (scalar) + equity ruling head
        self.wg = rng.standard_normal(H) * 0.1       # (H,)
        self.bg = np.array(0.0)
        self.We = sc(C, H)             # (C,H)  equity ruling head
        self.be = np.zeros(C)
        # IRJA' deferral head (scalar)
        self.wd = rng.standard_normal(H) * 0.1       # (H,)
        self.bd = np.array(-0.5)                  # start biased toward *ruling* (not deferring)
        # hyperparameters
        self.tau = float(tau)
        self.defer_cost = float(defer_cost)  # cost of saying "God knows best"
        self.aux_qiyas = float(aux_qiyas)    # weight on the strict-analogy aux loss
        self.gate_l1 = float(gate_l1)        # "istihsan is the exception": sparsity on g
        self.d, self.H, self.P, self.C = d, H, P, C

    # ---- parameter plumbing (needed for the gradient check + optimiser) ----
    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "K": self.K, "V": self.V, "wg": self.wg, "bg": self.bg,
                "We": self.We, "be": self.be, "wd": self.wd, "bd": self.bd}

    def set_param(self, name, value):
        setattr(self, name, value)

    # ---------------------------- forward ----------------------------------
    def forward(self, X, cache=True):
        A1 = X @ self.W1.T + self.b1;          H1 = tanh(A1)
        A2 = H1 @ self.W2.T + self.b2;         H2 = tanh(A2)
        S  = H2 @ self.K.T                     # (N,P)
        ATT = softmax(S, axis=1)               # (N,P)
        Q  = ATT @ self.V                      # (N,C)  qiyas logits
        GATE_PRE = H2 @ self.wg + self.bg      # (N,)
        G  = sigmoid(self.tau * GATE_PRE)      # (N,)   istihsan gate
        E  = H2 @ self.We.T + self.be          # (N,C)  equity logits
        Z  = (1 - G)[:, None] * Q + G[:, None] * E   # (N,C) final ruling logits
        DEFER_PRE = H2 @ self.wd + self.bd     # (N,)
        Pdef = sigmoid(DEFER_PRE)              # (N,)   probability of deferring
        out = dict(X=X, A1=A1, H1=H1, A2=A2, H2=H2, S=S, ATT=ATT, Q=Q,
                   GATE_PRE=GATE_PRE, G=G, E=E, Z=Z, DEFER_PRE=DEFER_PRE, Pdef=Pdef)
        return out if cache else (Z, Pdef)

    # -------------------------- loss + backward ----------------------------
    def loss_and_grads(self, X, y_true, y_analogy, defer_enabled=True):
        """
        Selective (reject-option) loss embodying irja':
            L_sel = (1-p)*CE(softmax(z), y_true) + p*defer_cost
        plus an auxiliary qiyas loss that trains the precedent bank to
        reproduce strict analogy:
            L_aux = aux * CE(softmax(q), y_analogy)
        plus an istihsan-sparsity term  gate_l1 * mean(g)  ("the override is
        the exception, not the rule"). The gate must fire to move z off q
        toward equity on hardship cases (y_true != y_analogy). Averaged over N.

        Curriculum: when defer_enabled is False the reasoner is forced to rule
        (p treated as 0, deferral head frozen) so that analogy+equity are
        mastered first; deferral (withholding judgment) is learned afterwards.
        """
        N, C = X.shape[0], self.C
        f = self.forward(X)
        Z, Q, G, E, ATT, H2, H1 = f["Z"], f["Q"], f["G"], f["E"], f["ATT"], f["H2"], f["H1"]
        Pdef = f["Pdef"]
        w_rule = (1 - Pdef) if defer_enabled else np.ones(N)   # ruling weight

        PR = softmax(Z, axis=1)                 # (N,C)
        QPR = softmax(Q, axis=1)                # (N,C)
        Ey = onehot(y_true, C)                  # (N,C)
        Ea = onehot(y_analogy, C)               # (N,C)

        ce_main = -np.log(np.maximum(PR[np.arange(N), y_true], 1e-12))   # (N,)
        ce_aux  = -np.log(np.maximum(QPR[np.arange(N), y_analogy], 1e-12))
        L_sel = w_rule * ce_main + (Pdef * self.defer_cost if defer_enabled else 0.0)
        L = np.mean(L_sel) + self.aux_qiyas * np.mean(ce_aux) + self.gate_l1 * np.mean(G)

        # ---- backward (all analytic) ----
        # d L_sel / d z  (through ce_main, coeff w_rule)
        gZ = (w_rule[:, None] * (PR - Ey)) / N               # (N,C)
        # d L / d p
        if defer_enabled:
            gP = (self.defer_cost - ce_main) / N             # (N,)
        else:
            gP = np.zeros(N)                                 # deferral head frozen
        # d L_aux / d q
        gQ_aux = self.aux_qiyas * (QPR - Ea) / N             # (N,C)

        # z = (1-g)q + g e
        gQ = (1 - G)[:, None] * gZ + gQ_aux                  # (N,C) total grad on q
        gE = G[:, None] * gZ                                 # (N,C)
        gG = np.sum(gZ * (E - Q), axis=1) + self.gate_l1 / N # (N,) blend + sparsity

        # equity head: E = H2 We^T + be
        grad_We = gE.T @ H2                                  # (C,H)
        grad_be = np.sum(gE, axis=0)                         # (C,)
        gH2_e = gE @ self.We                                 # (N,H)

        # qiyas: q = ATT @ V ;  ATT = softmax(S) ; S = H2 K^T
        grad_V = ATT.T @ gQ                                  # (P,C)
        g_att = gQ @ self.V.T                                # (N,P)
        # softmax jacobian per row
        g_s = ATT * (g_att - np.sum(g_att * ATT, axis=1, keepdims=True))  # (N,P)
        grad_K = g_s.T @ H2                                  # (P,H)
        gH2_s = g_s @ self.K                                 # (N,H)

        # istihsan gate: g = sigmoid(tau*gate_pre) ; gate_pre = H2 wg + bg
        dg = self.tau * G * (1 - G)                          # (N,)
        g_gate_pre = gG * dg                                 # (N,)
        grad_wg = g_gate_pre @ H2                            # (H,)
        grad_bg = np.sum(g_gate_pre)
        gH2_gate = np.outer(g_gate_pre, self.wg)             # (N,H)

        # irja' head: p = sigmoid(defer_pre) ; defer_pre = H2 wd + bd
        dp = Pdef * (1 - Pdef)                               # (N,)
        g_defer_pre = gP * dp                                # (N,)
        grad_wd = g_defer_pre @ H2                           # (H,)
        grad_bd = np.sum(g_defer_pre)
        gH2_defer = np.outer(g_defer_pre, self.wd)           # (N,H)

        # accumulate grad on h2 and flow through the trunk
        gH2 = gH2_e + gH2_s + gH2_gate + gH2_defer           # (N,H)
        gA2 = gH2 * (1 - f["H2"] ** 2)                       # tanh'
        grad_W2 = gA2.T @ H1
        grad_b2 = np.sum(gA2, axis=0)
        gH1 = gA2 @ self.W2
        gA1 = gH1 * (1 - f["H1"] ** 2)
        grad_W1 = gA1.T @ f["X"]
        grad_b1 = np.sum(gA1, axis=0)

        grads = {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2,
                 "K": grad_K, "V": grad_V, "wg": grad_wg, "bg": grad_bg,
                 "We": grad_We, "be": grad_be, "wd": grad_wd, "bd": grad_bd}
        return L, grads, f


# =============================================================================
#  2. FINITE-DIFFERENCE GRADIENT CHECK  (mandatory — must pass)
# =============================================================================

def gradient_check(seed=1):
    rng = np.random.default_rng(seed)
    d, H, P, C, N = 5, 7, 6, 4, 4
    model = IstihsanReasoner(d, H, P, C, tau=1.3, defer_cost=0.5, aux_qiyas=0.4, seed=seed)
    X = rng.standard_normal((N, d))
    y_true    = rng.integers(0, C, size=N)
    y_analogy = rng.integers(0, C, size=N)

    L0, grads, _ = model.loss_and_grads(X, y_true, y_analogy)
    eps = 1e-6
    worst = 0.0
    for name, P_ in model.params().items():
        flat = P_.reshape(-1)
        gflat = grads[name].reshape(-1)
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            Lp, _, _ = model.loss_and_grads(X, y_true, y_analogy)
            flat[i] = orig - eps
            Lm, _, _ = model.loss_and_grads(X, y_true, y_analogy)
            flat[i] = orig
            num = (Lp - Lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-9, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
    return worst


# =============================================================================
#  3. SYNTHETIC FIQH DATASET
#
#  Each "case" carries hidden factors. There are three case *types*:
#    NORMAL       : the equitable ruling == the strict-analogy ruling
#    HARDSHIP     : strict analogy would cause hardship; equity SHIFTS the ruling
#                   (y_true = (y_analogy + 1) mod C)  -> the gate must fire
#    UNDETERMINED : evidence is genuinely ambiguous; y_true is unpredictable
#                   -> the reasoner should DEFER (irja')
#
#  Features are a random linear entanglement of the latent factors + noise, so
#  the trunk has to actually disentangle them (a real learning problem).
# =============================================================================

NORMAL, HARDSHIP, UNDETERMINED = 0, 1, 2

def make_dataset(n, d=12, C=4, k=6, seed=0):
    rng = np.random.default_rng(seed)
    # fixed generative maps (shared across splits via the same seed family)
    gen = np.random.default_rng(12345)
    Wc = gen.standard_normal((C, k))          # latent -> analogy class
    M  = gen.standard_normal((d, k + 2))      # entangle [latent, hardship, undet] -> features

    Z = rng.standard_normal((n, k))
    y_analogy = np.argmax(Z @ Wc.T, axis=1)

    r = rng.random(n)
    ctype = np.where(r < 0.55, NORMAL, np.where(r < 0.80, HARDSHIP, UNDETERMINED))
    hardship_sig = (ctype == HARDSHIP).astype(float)
    undet_sig    = (ctype == UNDETERMINED).astype(float)

    y_true = y_analogy.copy()
    y_true[ctype == HARDSHIP] = (y_analogy[ctype == HARDSHIP] + 1) % C          # equity shift
    y_true[ctype == UNDETERMINED] = rng.integers(0, C, size=(ctype == UNDETERMINED).sum())  # noise

    raw = np.concatenate([Z, hardship_sig[:, None], undet_sig[:, None]], axis=1)  # (n,k+2)
    X = raw @ M.T + 0.25 * rng.standard_normal((n, d))                            # entangle + noise
    # standardise
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, y_true, y_analogy, ctype


# =============================================================================
#  4. ADAM OPTIMISER (pure NumPy) + TRAINING LOOP
# =============================================================================

class Adam:
    def __init__(self, model, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in model.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in model.params().items()}
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    def step(self, model, grads):
        self.t += 1
        for k, P_ in model.params().items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            newP = P_ - self.lr * mhat / (np.sqrt(vhat) + self.eps)
            model.set_param(k, newP)


def train(model, data, epochs=90, batch=128, lr=0.03, warmup_frac=0.45, verbose=True):
    """Curriculum: for the first `warmup_frac` of training the reasoner may not
    defer (it must commit to a ruling), so qiyas + istihsan are mastered first.
    Deferral (irja') is then switched on and the mind learns where to withhold."""
    X, y_true, y_analogy, ctype = data
    N = X.shape[0]
    opt = Adam(model, lr=lr)
    rng = np.random.default_rng(7)
    warmup = int(epochs * warmup_frac)
    for ep in range(epochs):
        defer_on = ep >= warmup
        idx = rng.permutation(N)
        tot = 0.0
        for s in range(0, N, batch):
            b = idx[s:s + batch]
            L, grads, _ = model.loss_and_grads(X[b], y_true[b], y_analogy[b],
                                               defer_enabled=defer_on)
            opt.step(model, grads)
            tot += L * len(b)
        if verbose and (ep % 15 == 0 or ep == epochs - 1):
            tag = "rule-only" if not defer_on else "with-irja'"
            print(f"   epoch {ep:3d} [{tag}]  train loss {tot / N:.4f}")
    return model


# =============================================================================
#  5. EVALUATION — reads the three moves back out of the trained mind
# =============================================================================

def evaluate(model, data, defer_thresh=0.5):
    X, y_true, y_analogy, ctype = data
    f = model.forward(X)
    ruling = np.argmax(f["Z"], axis=1)
    deferred = f["Pdef"] >= defer_thresh
    gate = f["G"]

    rep = {}
    for name, t in [("normal", NORMAL), ("hardship", HARDSHIP), ("undetermined", UNDETERMINED)]:
        m = ctype == t
        ruled = m & ~deferred
        acc = np.mean(ruling[ruled] == y_true[ruled]) if ruled.sum() else float("nan")
        rep[name] = dict(defer_rate=float(np.mean(deferred[m])),
                         gate_mean=float(np.mean(gate[m])),
                         acc_when_ruling=float(acc),
                         n=int(m.sum()))
    # overall accuracy over cases the mind chose to rule on
    ruled = ~deferred
    rep["_overall_ruled_acc"] = float(np.mean(ruling[ruled] == y_true[ruled]))
    rep["_overall_defer_rate"] = float(np.mean(deferred))
    return rep


# =============================================================================
#  6. SHURA — the council. Train K scholars; aggregate by confidence-weighted
#     consensus (non-parametric). Emulates the ~40-student deliberating majlis.
# =============================================================================

class ShuraCouncil:
    def __init__(self, scholars):
        self.scholars = scholars

    def deliberate(self, X, defer_thresh=0.5):
        Zs, Ps, confs = [], [], []
        for s in self.scholars:
            f = s.forward(X)
            pr = softmax(f["Z"], axis=1)
            Zs.append(pr)
            Ps.append(f["Pdef"])
            confs.append(np.max(pr, axis=1))          # each scholar's certainty
        Zs = np.stack(Zs); Ps = np.stack(Ps); confs = np.stack(confs)
        w = softmax(confs * 4.0, axis=0)              # louder = more certain scholars
        consensus_pr = np.sum(w[:, :, None] * Zs, axis=0)
        consensus_defer = np.mean(Ps, axis=0)         # council defers if the room is unsure
        ruling = np.argmax(consensus_pr, axis=1)
        deferred = consensus_defer >= defer_thresh
        return ruling, deferred, consensus_pr


# =============================================================================
#  7. DEMO / SELF-TESTS
# =============================================================================

def main():
    print("=" * 74)
    print(" Chapter 0184 — Abu Hanifa : the Istihsan Reasoner")
    print(" qiyas (analogy) + istihsan (equity override) + irja' (deferral) + shura")
    print("=" * 74)

    print("\n[1] Finite-difference gradient check (must be < 1e-4) ...")
    worst = gradient_check()
    print(f"    worst relative error over all parameters : {worst:.2e}")
    assert worst < 1e-4, "GRADIENT CHECK FAILED"
    print("    PASSED.")

    print("\n[2] Building synthetic fiqh corpus (entangled latent cases) ...")
    train_data = make_dataset(2400, seed=1)
    val_data   = make_dataset(800, seed=2)
    Xtr, ytr, yatr, ctr = train_data
    print(f"    train={Xtr.shape[0]}  val={val_data[0].shape[0]}  "
          f"types: normal={np.mean(ctr==NORMAL):.2f} "
          f"hardship={np.mean(ctr==HARDSHIP):.2f} undet={np.mean(ctr==UNDETERMINED):.2f}")

    print("\n[3] Training a single faqih (scholar) ...")
    model = IstihsanReasoner(d=12, H=28, P=18, C=4, tau=1.6,
                             defer_cost=0.55, aux_qiyas=0.6, seed=3)
    train(model, train_data, epochs=90, lr=0.03)

    print("\n[4] Reading the three moves back out (validation set):")
    rep = evaluate(model, val_data)
    for name in ["normal", "hardship", "undetermined"]:
        r = rep[name]
        print(f"    {name:13s} n={r['n']:4d}  defer={r['defer_rate']:.2f}  "
              f"gate(istihsan)={r['gate_mean']:.2f}  acc|ruling={r['acc_when_ruling']:.2f}")
    print(f"    ------------------------------------------------------------")
    print(f"    overall accuracy on cases it CHOSE to rule : {rep['_overall_ruled_acc']:.3f}")
    print(f"    overall deferral rate (irja')              : {rep['_overall_defer_rate']:.3f}")

    print("\n[5] Convening the SHURA (council of 5 scholars, different seeds) ...")
    scholars = [model]
    for sd in [11, 12, 13, 14]:
        s = IstihsanReasoner(d=12, H=28, P=18, C=4, tau=1.6,
                             defer_cost=0.55, aux_qiyas=0.6, seed=sd)
        train(s, make_dataset(2400, seed=sd), epochs=90, lr=0.03, verbose=False)
        scholars.append(s)
    council = ShuraCouncil(scholars)
    Xv, yv, yav, cv = val_data
    ruling, deferred, _ = council.deliberate(Xv)
    ruled = ~deferred
    council_acc = np.mean(ruling[ruled] == yv[ruled])
    solo_acc = rep["_overall_ruled_acc"]
    print(f"    solo scholar   accuracy|ruling : {solo_acc:.3f}")
    print(f"    council (shura) accuracy|ruling: {council_acc:.3f}  "
          f"(defer rate {np.mean(deferred):.3f})")

    print("\n[6] Boldness of istihsan (tau), held at inference on the trained mind:")
    print("    a bolder jurist reads the SAME evidence and overrides more sharply")
    Xv, yv, yav, cv = val_data
    base_tau = model.tau
    for tau in [0.5, 1.0, 2.0, 4.0]:
        model.tau = tau
        f = model.forward(Xv)
        gh = float(np.mean(f["G"][cv == HARDSHIP]))
        gn = float(np.mean(f["G"][cv == NORMAL]))
        print(f"    tau={tau:>4}:  gate on hardship={gh:.2f}   gate on normal={gn:.2f}"
              f"   contrast={gh - gn:+.2f}")
    model.tau = base_tau

    print("\nAll self-tests passed. The mind runs.")
    print("=" * 74)


if __name__ == "__main__":
    main()


# =============================================================================
#  VERIFIED EXECUTION OUTPUT  (produced by running this file, NumPy 2.4.4)
# -----------------------------------------------------------------------------
#  ==========================================================================
#   Chapter 0184 — Abu Hanifa : the Istihsan Reasoner
#   qiyas (analogy) + istihsan (equity override) + irja' (deferral) + shura
#  ==========================================================================
#  
#  [1] Finite-difference gradient check (must be < 1e-4) ...
#      worst relative error over all parameters : 1.15e-07
#      PASSED.
#  
#  [2] Building synthetic fiqh corpus (entangled latent cases) ...
#      train=2400  val=800  types: normal=0.55 hardship=0.25 undet=0.19
#  
#  [3] Training a single faqih (scholar) ...
#     epoch   0 [rule-only]  train loss 1.7401
#     epoch  15 [rule-only]  train loss 0.7927
#     epoch  30 [rule-only]  train loss 0.6036
#     epoch  45 [with-irja']  train loss 0.3263
#     epoch  60 [with-irja']  train loss 0.2458
#     epoch  75 [with-irja']  train loss 0.2919
#     epoch  89 [with-irja']  train loss 0.3222
#  
#  [4] Reading the three moves back out (validation set):
#      normal        n= 437  defer=0.01  gate(istihsan)=0.00  acc|ruling=0.89
#      hardship      n= 204  defer=0.02  gate(istihsan)=0.83  acc|ruling=0.88
#      undetermined  n= 159  defer=0.99  gate(istihsan)=0.04  acc|ruling=0.00
#      ------------------------------------------------------------
#      overall accuracy on cases it CHOSE to rule : 0.885
#      overall deferral rate (irja')              : 0.207
#  
#  [5] Convening the SHURA (council of 5 scholars, different seeds) ...
#      solo scholar   accuracy|ruling : 0.885
#      council (shura) accuracy|ruling: 0.930  (defer rate 0.201)
#  
#  [6] Boldness of istihsan (tau), held at inference on the trained mind:
#      a bolder jurist reads the SAME evidence and overrides more sharply
#      tau= 0.5:  gate on hardship=0.64   gate on normal=0.08   contrast=+0.56
#      tau= 1.0:  gate on hardship=0.75   gate on normal=0.01   contrast=+0.74
#      tau= 2.0:  gate on hardship=0.86   gate on normal=0.00   contrast=+0.86
#      tau= 4.0:  gate on hardship=0.92   gate on normal=0.00   contrast=+0.92
#  
#  All self-tests passed. The mind runs.
#  ==========================================================================
# =============================================================================

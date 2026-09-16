#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Encyclopedia of Lost Minds — Chapter 0200
Malik ibn Anas (c. 711 - 795 CE), Medina
--------------------------------------------------------------------------------
THE MUWATTA' NETWORK  —  an 'Amal-Consensus Grounding architecture
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0200_malik_ibn_anas_711 - Malik ibn Anas (c. 711 - 795 CE), Medina
================================================================================  

WHY THIS ARCHITECTURE (the one cognitive idea that is Malik's alone)
--------------------------------------------------------------------------------
Malik ibn Anas did not leave a philosophy of the intellect; he left a *method
for grounding knowledge*, and that method has one unusual centre of gravity that
no other jurist of his generation placed first:

    The authoritative ground truth of a normative system does not live inside
    any single stored record.  It lives in the continuous, embodied, collective
    practice of a living community -- the 'amal ahl al-Madina, "the practice of
    the people of Medina" -- and that practice can OVERRIDE even a soundly
    transmitted solitary report (khabar al-wahid).

Three further Malikian commitments follow, and all four become concrete,
differentiable machinery in this file:

  (1) ISNAD AS A WEAKEST-LINK GATE.  A report is worth only what its chain of
      transmitters is worth, and a chain is only as strong as its weakest link.
      => reliability is a PRODUCT of per-link sigmoids, not a sum: one bad link
      collapses the whole chain.

  (2) THE 'AMAL OVERRIDE.  When the individual report is weak, defer to the
      community's practice.  => the final representation is a reliability-gated
      MIXTURE of two SEPARATE pathways:
            z = r * (individual report)  +  (1 - r) * (community consensus).
      Strong chain -> trust the report.  Weak/isolated chain -> the enacted
      practice governs.  Crucially the two pathways read DIFFERENT inputs, so a
      weak chain cannot smuggle its misleading report into the fallback.

  (3) CALIBRATED ABSTENTION -- "la adri", I do not know.  Malik was asked
      forty-eight questions and answered "I do not know" to thirty-two; he would
      not issue a fatwa until seventy scholars vouched for him.  => the head
      carries an explicit ABSTAIN class and is trained to use it on genuinely
      under-determined cases rather than guess.

  (4) SADD AL-DHARA'I -- "blocking the means."  Malik forbade an otherwise-
      permissible act when it reliably opened the road to harm.  => a
      differentiable PRECAUTION BARRIER penalises internal states that drift
      toward a learned "prohibited" direction, pulling the state back before the
      harmful region is reached.

Deliberate divergence from the neighbouring legal minds in this corpus: the
stock reading -- "intelligence imposes order; build an auditable institution;
alignment is codification" -- is NOT used.  Malik's distinctive move is
epistemic, not institutional: he trusts *aggregated enacted behaviour* over
*isolated authoritative testimony*, and he makes "I do not know" a first-class
answer.  That is what this network embodies and what the experiments demonstrate.

WHAT THIS FILE DOES
--------------------------------------------------------------------------------
* Pure-NumPy implementation, no autograd.
* Full hand-derived backward pass for every parameter block.
* A mandatory central finite-difference gradient check (asserted to pass).
* A synthetic "rulings" task whose truth lives in a consensus subspace, whose
  individual reports are corrupted on weak chains, and whose under-determined
  cases are labelled ABSTAIN.
* A real mini-batch training loop with momentum.
* An ABLATION that forces r = 1 (always trust the report, never defer to the
  community -- the very move Malik refused) to show the override earns its keep.

Run:  python3 chapter_0187_malik_ibn_anas_711.py
================================================================================
"""

import numpy as np


# =============================================================================
# 0.  NUMERICAL HELPERS
# =============================================================================

def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(z, axis=-1):
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


# =============================================================================
# 1.  THE MODEL
# =============================================================================
#
#   N   batch size
#   Dr  dimension of the REPORT block  (the individual transmitted testimony)
#   Dc  dimension of the CONSENSUS block (the robust community signal)
#   L   number of links in the transmission chain (the isnad)
#   H   hidden width
#   M   number of practice prototypes ('amal of Medina)
#   Kc  number of substantive ruling classes; output has Kc+1 logits, the last
#       being ABSTAIN ("la adri"), index Kc.
#
# Forward (vectorised over the batch):
#   1. isnad:      S = T*wg + bg ; rho = sigmoid(S) ; r = prod_l rho_l  in (0,1)
#   2. report:     h = tanh(Xr W1^T + b1)                     <- reads REPORT only
#   3. query:      q = tanh(Xc Wq^T + bq)                     <- reads CONSENSUS only
#   4. consensus:  A = (q P^T)/sqrt(H) ; alpha = softmax(A) ; c = alpha P
#   5. 'amal gate: z = r*h + (1-r)*c
#   6. head:       O = z W2^T + b2 ; probs = softmax(O)
#   7. precaution: pen = relu(z.d - margin) ; barrier = mean(pen^2)
# =============================================================================

class MuwattaNetwork:
    """The 'Amal-Consensus Grounding network (see module docstring)."""

    def __init__(self, Dr, Dc, L, H, M, Kc, margin=0.5, lam_prec=0.05,
                 l2=1e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.Dr, self.Dc, self.L, self.H, self.M, self.Kc = Dr, Dc, L, H, M, Kc
        self.margin, self.lam_prec, self.l2 = margin, lam_prec, l2

        # isnad gate: one weight & bias per link; start undecided (r ~ 0.5).
        self.wg = rng.normal(0, 0.5, size=L)
        self.bg = np.full(L, 0.3)
        # report encoder (individual testimony)
        self.W1 = rng.normal(0, 1.0 / np.sqrt(Dr), size=(H, Dr))
        self.b1 = np.zeros(H)
        # consensus-query encoder (robust community signal)
        self.Wq = rng.normal(0, 1.0 / np.sqrt(Dc), size=(H, Dc))
        self.bq = np.zeros(H)
        # practice prototypes: the learned 'amal of Medina
        self.P = rng.normal(0, 1.0 / np.sqrt(H), size=(M, H))
        # output head: Kc substantive classes + 1 abstain
        self.W2 = rng.normal(0, 1.0 / np.sqrt(H), size=(Kc + 1, H))
        self.b2 = np.zeros(Kc + 1)
        # sadd al-dhara'i precaution direction
        self.d = rng.normal(0, 1.0 / np.sqrt(H), size=H)

    def params(self):
        return {"wg": self.wg, "bg": self.bg, "W1": self.W1, "b1": self.b1,
                "Wq": self.Wq, "bq": self.bq, "P": self.P,
                "W2": self.W2, "b2": self.b2, "d": self.d}

    # ---- FORWARD --------------------------------------------------------
    def forward(self, Xr, Xc, T, cache=True):
        H = self.H
        S = T * self.wg + self.bg               # (N, L)
        rho = sigmoid(S)                         # (N, L)
        r = np.prod(rho, axis=1, keepdims=True)  # (N, 1) chain reliability

        pre_h = Xr @ self.W1.T + self.b1         # (N, H)
        h = np.tanh(pre_h)                       # report representation
        pre_q = Xc @ self.Wq.T + self.bq         # (N, H)
        q = np.tanh(pre_q)                       # consensus query

        A = (q @ self.P.T) / np.sqrt(H)          # (N, M)
        alpha = softmax(A, axis=1)               # (N, M)
        c = alpha @ self.P                       # (N, H) consensus context

        z = r * h + (1.0 - r) * c                # (N, H) 'amal gate

        O = z @ self.W2.T + self.b2              # (N, Kc+1)
        probs = softmax(O, axis=1)

        g = z @ self.d                           # (N,)
        pen = np.maximum(0.0, g - self.margin)   # (N,)

        if not cache:
            return probs
        ctx = dict(Xr=Xr, Xc=Xc, T=T, rho=rho, r=r, h=h, q=q, A=A, alpha=alpha,
                   c=c, z=z, probs=probs, g=g, pen=pen)
        return probs, ctx

    # ---- LOSS -----------------------------------------------------------
    def loss(self, Xr, Xc, T, y):
        probs, ctx = self.forward(Xr, Xc, T, cache=True)
        N = Xr.shape[0]
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        barrier = np.mean(ctx["pen"] ** 2)
        reg = self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
                         + np.sum(self.Wq ** 2))
        ctx["y"] = y
        return ce + self.lam_prec * barrier + reg, ctx

    # ---- BACKWARD (hand-derived for every block) ------------------------
    def backward(self, ctx):
        Xr, Xc, T = ctx["Xr"], ctx["Xc"], ctx["T"]
        rho, r, h, q = ctx["rho"], ctx["r"], ctx["h"], ctx["q"]
        alpha, c, z = ctx["alpha"], ctx["c"], ctx["z"]
        probs, y, g, pen = ctx["probs"], ctx["y"], ctx["g"], ctx["pen"]
        N, H = Xr.shape[0], self.H

        # cross-entropy through softmax head
        dO = probs.copy()
        dO[np.arange(N), y] -= 1.0
        dO /= N
        dW2 = dO.T @ z
        db2 = dO.sum(axis=0)
        dz = dO @ self.W2

        # precaution barrier
        mask = (g > self.margin).astype(np.float64)
        coeff = self.lam_prec * 2.0 * pen * mask / N
        dz += coeff[:, None] * self.d[None, :]
        dd = (coeff[:, None] * z).sum(axis=0)

        # 'amal gate z = r*h + (1-r)*c
        dh = dz * r
        dc = dz * (1.0 - r)
        dr = np.sum(dz * (h - c), axis=1, keepdims=True)

        # report encoder h = tanh(Xr W1^T + b1)
        dpre_h = dh * (1.0 - h ** 2)
        dW1 = dpre_h.T @ Xr
        db1 = dpre_h.sum(axis=0)

        # consensus c = alpha P ; alpha = softmax(A) ; A = q P^T / sqrt(H)
        dalpha = dc @ self.P.T
        dP = alpha.T @ dc
        dA = alpha * (dalpha - np.sum(dalpha * alpha, axis=1, keepdims=True))
        dq = (dA @ self.P) / np.sqrt(H)
        dP += (dA.T @ q) / np.sqrt(H)

        # consensus-query encoder q = tanh(Xc Wq^T + bq)
        dpre_q = dq * (1.0 - q ** 2)
        dWq = dpre_q.T @ Xc
        dbq = dpre_q.sum(axis=0)

        # isnad gate r = prod_l rho_l ; rho = sigmoid(S) ; S = T*wg + bg
        drho = dr * (r / rho)
        dS = drho * rho * (1.0 - rho)
        dwg = np.sum(dS * T, axis=0)
        dbg = np.sum(dS, axis=0)

        # weight decay
        dW1 += 2.0 * self.l2 * self.W1
        dW2 += 2.0 * self.l2 * self.W2
        dWq += 2.0 * self.l2 * self.Wq

        return {"wg": dwg, "bg": dbg, "W1": dW1, "b1": db1, "Wq": dWq,
                "bq": dbq, "P": dP, "W2": dW2, "b2": db2, "d": dd}

    # ---- helpers --------------------------------------------------------
    def predict(self, Xr, Xc, T):
        return np.argmax(self.forward(Xr, Xc, T, cache=False), axis=1)

    def reliability(self, T):
        return np.prod(sigmoid(T * self.wg + self.bg), axis=1)


# =============================================================================
# 2.  FINITE-DIFFERENCE GRADIENT CHECK   (mandatory)
# =============================================================================

def gradient_check(seed=1):
    rng = np.random.default_rng(seed)
    Dr, Dc, L, H, M, Kc, N = 4, 4, 3, 7, 5, 3, 12
    net = MuwattaNetwork(Dr, Dc, L, H, M, Kc, margin=0.2, lam_prec=0.3,
                         l2=1e-3, seed=seed)
    Xr = rng.normal(size=(N, Dr))
    Xc = rng.normal(size=(N, Dc))
    T = rng.uniform(0, 1, size=(N, L))
    y = rng.integers(0, Kc + 1, size=N)

    _, ctx = net.loss(Xr, Xc, T, y)
    analytic = net.backward(ctx)

    eps, worst, report = 1e-6, 0.0, []
    for name, P in net.params().items():
        flat = P.ravel()
        g_ana = analytic[name].ravel()
        idxs = list(range(flat.size)) if flat.size <= 30 else \
            list(rng.choice(flat.size, size=30, replace=False))
        num, ana = [], []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, _ = net.loss(Xr, Xc, T, y)
            flat[i] = old - eps
            lm, _ = net.loss(Xr, Xc, T, y)
            flat[i] = old
            num.append((lp - lm) / (2 * eps))
            ana.append(g_ana[i])
        num, ana = np.array(num), np.array(ana)
        denom = np.maximum(1e-8, np.abs(num) + np.abs(ana))
        rel = np.max(np.abs(num - ana) / denom)
        worst = max(worst, rel)
        report.append((name, rel))

    print("  Finite-difference gradient check (central differences, eps=1e-6)")
    print("  " + "-" * 54)
    for name, rel in report:
        print(f"    param {name:<3s}  max rel. error = {rel:.2e}   "
              f"[{'ok' if rel < 1e-4 else 'FAIL'}]")
    print("  " + "-" * 54)
    print(f"    WORST over all blocks = {worst:.2e}")
    assert worst < 1e-4, "Gradient check FAILED -- backprop is wrong."
    print("    Gradient check PASSED.\n")
    return worst


# =============================================================================
# 3.  SYNTHETIC "RULINGS" TASK THAT ENCODES MALIK'S EPISTEMICS
# =============================================================================
#
#   consensus block Xc -> a noisy pointer to one of Kc latent PRACTICE clusters;
#     subtle but honestly grounded; the TRUE label is its dominant cluster.
#   report block    Xr -> an individual transmitted testimony; SALIENT.  On a
#     STRONG chain it points to the truth; on a WEAK chain it points elsewhere
#     (an unreliable narrator).  A naive model is tempted to trust it -- and is
#     burned on weak chains unless it defers to the consensus.
#   transmitter T      -> per-link quality in [0,1]; a chain is strong iff its
#     weakest link clears a threshold.
#   ABSTAIN cases      -> near-tie in consensus AND a weak chain: nothing
#     decides them, so they are labelled "la adri" (index Kc).
# =============================================================================

def make_dataset(n, Kc=3, L=3, seed=0, tau=0.32, abstain_frac=0.16):
    rng = np.random.default_rng(seed)
    Dc = Kc
    Dr = Kc
    Xc = np.zeros((n, Dc))
    Xr = np.zeros((n, Dr))
    T = rng.uniform(0.0, 1.0, size=(n, L))
    y = np.zeros(n, dtype=int)

    weakest = T.min(axis=1)
    strong = weakest > tau

    for i in range(n):
        true_cluster = rng.integers(0, Kc)

        # consensus: subtle bump on the true cluster + noise (robust ground)
        cons = rng.normal(0, 0.75, size=Dc)
        cons[true_cluster] += 1.7
        Xc[i] = cons

        # report: salient bump; strong chain -> truth, weak chain -> wrong
        rep = rng.normal(0, 0.5, size=Dr)
        if strong[i]:
            rep[true_cluster] += 3.0
        else:
            wrong = (true_cluster + rng.integers(1, Kc)) % Kc
            rep[wrong] += 3.0
        Xr[i] = rep

        # abstain: consensus near-tie AND weak chain -> nothing decides it
        top2 = np.sort(cons)[::-1][:2]
        near_tie = (top2[0] - top2[1]) < 0.8
        if near_tie and (not strong[i]) and (rng.random() < 0.85):
            y[i] = Kc
        else:
            y[i] = true_cluster

    # top up abstain to roughly the requested fraction using genuinely
    # ambiguous weak-chain cases only
    target = int(abstain_frac * n)
    if np.sum(y == Kc) < target:
        cand = np.where(~strong)[0]
        rng.shuffle(cand)
        for i in cand:
            if np.sum(y == Kc) >= target:
                break
            top2 = np.sort(Xc[i])[::-1][:2]
            if (top2[0] - top2[1]) < 1.0:
                y[i] = Kc
    return Xr, Xc, T, y, Dr, Dc


# =============================================================================
# 4.  TRAINING LOOP  (mini-batch gradient descent + momentum)
# =============================================================================

def accuracy(net, Xr, Xc, T, y):
    return float(np.mean(net.predict(Xr, Xc, T) == y))


def train(net, tr, va, epochs=80, batch=64, lr=0.25, momentum=0.9,
          verbose=True):
    Xr, Xc, T, y = tr
    Xrv, Xcv, Tv, yv = va
    rng = np.random.default_rng(123)
    vel = {k: np.zeros_like(v) for k, v in net.params().items()}
    n = Xr.shape[0]
    hist = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        ep_loss, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            L, ctx = net.loss(Xr[idx], Xc[idx], T[idx], y[idx])
            grads = net.backward(ctx)
            for k, P in net.params().items():
                vel[k] = momentum * vel[k] - lr * grads[k]
                P += vel[k]
            ep_loss += L
            nb += 1
        tr_acc = accuracy(net, Xr, Xc, T, y)
        va_acc = accuracy(net, Xrv, Xcv, Tv, yv)
        hist.append((ep, ep_loss / nb, tr_acc, va_acc))
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"    epoch {ep:3d}   loss {ep_loss/nb:6.4f}   "
                  f"train_acc {tr_acc:5.3f}   val_acc {va_acc:5.3f}")
    return hist


# =============================================================================
# 5.  MAIN
# =============================================================================

def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 72)
    print("THE MUWATTA' NETWORK  -  Malik ibn Anas  (chapter 0187)")
    print("'Amal-Consensus Grounding: weakest-link isnad gate + practice")
    print("prototypes + calibrated 'I do not know' + sadd al-dhara'i barrier")
    print("=" * 72)

    print("\n[1] Verifying the hand-derived backward pass")
    gradient_check(seed=1)

    print("[2] Building the synthetic 'rulings' corpus")
    Kc, L, tau = 3, 3, 0.32
    Xr, Xc, T, y, Dr, Dc = make_dataset(1600, Kc=Kc, L=L, seed=10, tau=tau)
    Xrv, Xcv, Tv, yv, _, _ = make_dataset(600, Kc=Kc, L=L, seed=99, tau=tau)
    strong_tr = (T.min(axis=1) > tau)
    print(f"    report dim={Dr}, consensus dim={Dc}, chain links={L}, "
          f"classes={Kc}+abstain")
    print(f"    train={len(y)}  val={len(yv)}  "
          f"strong-chain frac={strong_tr.mean():.2f}  "
          f"abstain frac={np.mean(y==Kc):.2f}")

    print("\n[3] Training the Muwatta' Network")
    net = MuwattaNetwork(Dr=Dr, Dc=Dc, L=L, H=24, M=6, Kc=Kc,
                         margin=0.5, lam_prec=0.05, l2=1e-4, seed=3)
    train(net, (Xr, Xc, T, y), (Xrv, Xcv, Tv, yv),
          epochs=80, batch=64, lr=0.25, momentum=0.9)

    print("\n[4] Behaviour report")
    va_acc = accuracy(net, Xrv, Xcv, Tv, yv)
    print(f"    validation accuracy (incl. abstain) = {va_acc:5.3f}")
    pred = net.predict(Xrv, Xcv, Tv)
    abst = (yv == Kc)
    if abst.sum():
        print(f"    'I do not know' recall on under-determined cases = "
              f"{np.mean(pred[abst]==Kc):5.3f}")
    print(f"    over-abstention on decisive cases               = "
          f"{np.mean(pred[~abst]==Kc):5.3f}")
    r_all = net.reliability(Tv)
    strong = Tv.min(axis=1) > tau
    print(f"    learned reliability r | strong chains = {r_all[strong].mean():5.3f}")
    print(f"    learned reliability r | weak chains   = {r_all[~strong].mean():5.3f}")
    # accuracy on the DECISIVE weak-chain cases -- the ones only the override
    # can rescue (misleading report, robust consensus)
    dec_weak = (~abst) & (~strong)
    print(f"    accuracy on decisive WEAK-chain cases = "
          f"{np.mean(pred[dec_weak]==yv[dec_weak]):5.3f}   "
          f"(these are the 'amal-override's job)")

    print("\n[5] Ablation - remove the 'amal override (force r = 1)")
    print("    always trust the individual report, never defer to the")
    print("    community practice -- the very move Malik refused to make")

    class NoAmal(MuwattaNetwork):
        def forward(self, Xr, Xc, T, cache=True):
            H = self.H
            S = T * self.wg + self.bg
            rho = sigmoid(S)
            r = np.ones((Xr.shape[0], 1))          # <-- ablation
            h = np.tanh(Xr @ self.W1.T + self.b1)
            q = np.tanh(Xc @ self.Wq.T + self.bq)
            A = (q @ self.P.T) / np.sqrt(H)
            alpha = softmax(A, axis=1)
            c = alpha @ self.P
            z = r * h + (1.0 - r) * c               # == h
            O = z @ self.W2.T + self.b2
            probs = softmax(O, axis=1)
            g = z @ self.d
            pen = np.maximum(0.0, g - self.margin)
            if not cache:
                return probs
            ctx = dict(Xr=Xr, Xc=Xc, T=T, rho=rho, r=r, h=h, q=q, A=A,
                       alpha=alpha, c=c, z=z, probs=probs, g=g, pen=pen)
            return probs, ctx

    net2 = NoAmal(Dr=Dr, Dc=Dc, L=L, H=24, M=6, Kc=Kc,
                  margin=0.5, lam_prec=0.05, l2=1e-4, seed=3)
    train(net2, (Xr, Xc, T, y), (Xrv, Xcv, Tv, yv),
          epochs=80, batch=64, lr=0.25, momentum=0.9, verbose=False)
    va_acc2 = accuracy(net2, Xrv, Xcv, Tv, yv)
    pred2 = net2.predict(Xrv, Xcv, Tv)
    print(f"    decisive WEAK-chain accuracy  WITHOUT override = "
          f"{np.mean(pred2[dec_weak]==yv[dec_weak]):5.3f}")
    print(f"    decisive WEAK-chain accuracy  WITH    override = "
          f"{np.mean(pred[dec_weak]==yv[dec_weak]):5.3f}")
    print(f"    overall val accuracy  WITHOUT override = {va_acc2:5.3f}")
    print(f"    overall val accuracy  WITH    override = {va_acc:5.3f}")
    print(f"    => the community-practice override is worth "
          f"{(va_acc-va_acc2)*100:+.1f} accuracy points overall")

    print("\n" + "=" * 72)
    print("Malik's thesis, made mechanical: when a solitary report is weak,")
    print("defer to the continuous practice of the community; and when nothing")
    print("decides the case, say -- plainly -- 'I do not know.'")
    print("=" * 72)


if __name__ == "__main__":
    main()

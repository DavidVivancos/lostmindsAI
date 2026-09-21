#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 ENCYCLOPEDIA OF LOST MINDS -- Chapter 0214
 Ahmad ibn Hanbal (164-241 AH / 780-855 CE), Baghdad
-------------------------------------------------------------------------------
 THE TA'LIL COLLATION ENGINE
 A witness-collation architecture with a hidden-defect ('illa) detector
 and a trained abstention gate.
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0214_ahmad_ibn_hanbal_780 - Ahmad ibn Hanbal (164-241 AH / 780-855 CE), Baghdad
================================================================================  

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------------------------------------------
Ibn Hanbal's distinctive cognitive instrument was not the isnad (everyone in
ninth-century Baghdad used chains of transmission). It was 'ilal -- the science
of *hidden* defects, set out in his Kitab al-'Ilal wa Ma'rifat al-Rijal. An
'illa is a flaw in a report whose chain is formally impeccable and whose text
reads perfectly well. It passes every surface check. It is still corrupt.

Ibn Hanbal's method for finding one was collation. He would hold every parallel
transmission of the same content beside each other -- mutabi'at (parallels from
the same Companion) and shawahid (corroborations from a different Companion) --
and look for the place where one channel diverged. Crucially, he did not condemn
a witness for merely being noisy. He condemned a witness whose divergence was
*characteristic*: the same narrator drifting the same way across many unrelated
reports. That is the difference between a bad memory and a defect. A man may
misremember at random; a man with an 'illa misremembers in a direction.

This file encodes exactly that. The network's parameters do not store content.
They store a model of the *transmitters*: for each channel, a read operator, a
bias, a reliability precision, and -- the load-bearing part -- a learned defect
signature vector. Content lives in a non-parametric variant bundle that is never
compressed, because compression is what destroys the divergence signal. Ibn
Hanbal's Musnad is arranged by Companion, not by topic, and carries roughly ten
thousand repetitions in about forty thousand reports. Every reviewer since has
called that arrangement cumbersome. It is not cumbersome. It is the instrument.
Deduplicate the Musnad and you can no longer do 'ilal on it.

THE THREE HEADS
-------------------------------------------------------------------------------
  jam'    (collation)   precision-weighted fusion of de-channelled witnesses
  ta'lil  (defecting)   per-witness defect logit: magnitude + signature-alignment
                        + isolation-from-bundle
  la adri (abstention)  a trained "I do not know" -- a first-class output, not a
                        confidence threshold bolted on afterwards

THE DIAGNOSTIC THAT MATTERS
-------------------------------------------------------------------------------
Self-test 3 is the whole thesis in numbers. It separates two populations:
  (a) witnesses with LARGE random deviation and no defect  -> must be acquitted
  (b) witnesses with SMALL systematic deviation and a defect -> must be convicted
A magnitude-threshold detector gets (a) and (b) exactly backwards. The 'illa
detector must not.

CONVENTIONS
-------------------------------------------------------------------------------
  Pure NumPy. Hand-derived backward pass. Finite-difference gradient check is
  mandatory and runs on every execution. Real training loop. Deterministic seed.
  No frameworks, no autograd, no pretrained anything.

  Run:  python3 chapter_0214_ahmad_ibn_hanbal_780.py
===============================================================================
"""

import numpy as np

# =============================================================================
# SECTION 0 -- NUMERICAL PRIMITIVES
# =============================================================================
EPS = 1e-8


def softplus(x):
    """Smooth positive map. Used for precision (reliability) so a channel can be
    trusted arbitrarily much but never negatively."""
    return np.logaddexp(0.0, x)


def d_softplus(x):
    """d/dx softplus(x) = sigmoid(x)."""
    return sigmoid(x)


def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(np.asarray(x, dtype=float))
    pos = np.asarray(x) >= 0
    xp = np.asarray(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-xp[pos]))
    ex = np.exp(xp[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def bce_with_logits(logit, target):
    """Binary cross-entropy computed from the logit directly (stable form).
    Returns scalar loss and dL/dlogit, which is simply sigmoid(logit) - target."""
    z = np.asarray(logit, dtype=float)
    t = np.asarray(target, dtype=float)
    loss = np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z))))
    grad = (sigmoid(z) - t) / z.size
    return loss, grad


# =============================================================================
# SECTION 1 -- THE WORLD: how reports are actually transmitted
# =============================================================================
class TransmissionWorld:
    """
    The ground truth the model never sees.

    A 'matn' (the content of a report) is a latent vector m in R^d. It reaches a
    scholar only through a narrator -- a *channel*. Each channel k mangles what
    passes through it in a fixed, personal way:

        x = Q_k @ m + d_k + noise_k

    Q_k is the narrator's habitual reshaping (his idiom, his abbreviations, his
    tendency to paraphrase). d_k is his standing bias. noise_k is ordinary
    fallible memory, scaled by his personal sloppiness sigma_k.

    An 'illa is something else entirely. When a defect is injected, the witness
    is displaced along a direction that is FIXED FOR THAT CHANNEL -- v_k. This is
    the formal statement of Ibn Hanbal's discovery: a defective transmitter fails
    the same way every time. That repetition across unrelated reports is the only
    thing that distinguishes a defect from bad luck, and it is invisible in any
    single report. You must collate to see it.
    """

    def __init__(self, d=8, n_channels=12, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.K = n_channels

        # Q_k: near-identity so the inverse problem is well posed but non-trivial
        self.Q = np.stack([
            np.eye(d) + 0.35 * rng.standard_normal((d, d)) for _ in range(n_channels)
        ])
        self.dbias = 0.4 * rng.standard_normal((n_channels, d))

        # sigma_k: personal sloppiness. Deliberately wide -- some narrators are
        # simply noisy without being defective. Half the point of the exercise.
        self.sigma = np.exp(rng.uniform(np.log(0.05), np.log(0.55), size=n_channels))

        # v_k: the channel's characteristic defect direction (unit vector).
        v = rng.standard_normal((n_channels, d))
        self.v = v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)

        # p_defect_k: how often this channel is actually corrupt. Some narrators
        # are reliable, some are 'mudtarib' (confused). The model must infer it.
        self.p_defect = rng.uniform(0.02, 0.45, size=n_channels)

    def sample_bundle(self, rng, min_w=2, max_w=6):
        """
        Draw one collation bundle: several independent witnesses to one matn.

        Returns a dict with the witness observations, their channel ids, the
        per-witness defect labels, the true matn, and the abstention target.
        """
        d, K = self.d, self.K
        n = int(rng.integers(min_w, max_w + 1))
        m = rng.standard_normal(d)
        chans = rng.choice(K, size=n, replace=False)

        X = np.zeros((n, d))
        y = np.zeros(n)
        for i, k in enumerate(chans):
            base = self.Q[k] @ m + self.dbias[k]
            noise = self.sigma[k] * rng.standard_normal(d)

            if rng.random() < self.p_defect[k]:
                # A genuine 'illa: small, systematic, along the channel's own
                # signature direction. Small on purpose -- it must be findable
                # by alignment, not by size.
                mag = rng.uniform(0.35, 0.9)
                base = base + self.Q[k] @ (mag * self.v[k])
                y[i] = 1.0
            X[i] = base + noise

        # 'la adri' target. The correct answer is "I do not know" when the
        # bundle cannot support a verdict: too few witnesses, all of them noisy,
        # or a majority of them corrupt. Ibn Hanbal answered "la adri" often and
        # without embarrassment; his students recorded the refusals as rulings.
        mean_sigma = float(np.mean(self.sigma[chans]))
        frac_bad = float(np.mean(y))
        abstain = 1.0 if (frac_bad >= 0.5 or (n <= 2 and mean_sigma > 0.30)) else 0.0

        return {"X": X, "chan": chans, "y": y, "m": m, "abstain": abstain, "n": n}

    def corpus(self, n_bundles, seed):
        rng = np.random.default_rng(seed)
        return [self.sample_bundle(rng) for _ in range(n_bundles)]


# =============================================================================
# SECTION 2 -- THE MODEL
# =============================================================================
class TalilCollationEngine:
    """
    Parameters -- note what is and is not here.

    PER CHANNEL (the model of the transmitters):
      R[k]   (d,d)  read operator: undoes the narrator's habitual reshaping
      c[k]   (d,)   offset: undoes his standing bias
      rho[k] scalar log-precision -> softplus -> ta'dil weight (trust)
      s[k]   (d,)   LEARNED DEFECT SIGNATURE. The heart of the file. After
                    training this should point along Q_k's image of v_k, because
                    that is the direction in which this narrator characteristically
                    goes wrong.

    SHARED (the verdict heads):
      a      (3,)   ta'lil weights: [magnitude, signature-alignment, isolation]
      a0     scalar ta'lil bias
      u      (4,)   abstention weights over
                    [log total precision, dispersion, width, mean defect verdict]
      u0     scalar abstention bias

    NOT HERE: any storage of report content. The corpus stays outside the
    parameters as an uncompressed variant bundle. This inversion is the whole
    architectural argument -- learn the witnesses, keep the testimony.
    """

    PARAM_NAMES = ["R", "c", "rho", "s", "a", "a0", "u", "u0"]

    def __init__(self, d=8, n_channels=12, seed=1):
        rng = np.random.default_rng(seed)
        self.d, self.K = d, n_channels
        self.P = {
            "R":   np.stack([np.eye(d) + 0.05 * rng.standard_normal((d, d))
                             for _ in range(n_channels)]),
            "c":   0.05 * rng.standard_normal((n_channels, d)),
            "rho": np.zeros(n_channels),
            "s":   0.10 * rng.standard_normal((n_channels, d)),
            "a":   np.array([0.5, 1.0, 0.5]),
            "a0":  np.array(-1.0),
            "u":   np.array([-0.5, 1.0, -0.5, 2.0]),
            "u0":  np.array(0.0),
        }

    # -- forward -------------------------------------------------------------
    def forward(self, b):
        """
        One bundle through the three heads. Returns everything the backward pass
        needs, so the derivation below can be read against the algebra.
        """
        P, d = self.P, self.d
        X, ch, n = b["X"], b["chan"], b["n"]

        # --- 1. de-channelling: strip each narrator's known idiom -------------
        # z_i = R_{k} x_i + c_{k}.  This is 'ma'rifat al-rijal' as a linear map:
        # knowing the man well enough to subtract him from his own report.
        Z = np.einsum("nij,nj->ni", P["R"][ch], X) + P["c"][ch]

        # --- 2. jam': precision-weighted collation ---------------------------
        # p_i is the narrator's trust. m_hat is the fused reading -- the text as
        # the assembled witnesses jointly support it.
        p = softplus(P["rho"][ch]) + 1e-3
        Ptot = float(np.sum(p))
        m_hat = (p[:, None] * Z).sum(axis=0) / Ptot

        # --- 3. ta'lil: where does each witness stand apart? -----------------
        Rres = Z - m_hat                       # residual per witness
        nrm = np.sqrt((Rres ** 2).sum(axis=1) + EPS)
        align = np.einsum("ni,ni->n", Rres, P["s"][ch])   # signature alignment
        nbar = float(np.mean(nrm))
        iso = nrm - nbar                        # isolation within the bundle

        a = P["a"]
        ell = a[0] * nrm + a[1] * align + a[2] * iso + P["a0"]

        # --- 4. la adri: is this bundle able to bear a verdict at all? --------
        # The fourth feature is the bundle's OWN mean defect verdict. The refusal
        # is therefore downstream of the criticism: you decline to rule because
        # the testimony you would have ruled from is itself under suspicion. This
        # is the ordering in Ibn Hanbal's practice -- 'la adri' is the conclusion
        # of an examination, not a shrug offered in place of one.
        mean_def = float(np.mean(sigmoid(ell)))
        feat = np.array([np.log(Ptot + EPS), nbar, n / 6.0, mean_def])
        q = float(P["u"] @ feat + P["u0"])

        return {"Z": Z, "p": p, "Ptot": Ptot, "m_hat": m_hat, "Rres": Rres,
                "nrm": nrm, "align": align, "nbar": nbar, "iso": iso,
                "ell": ell, "q": q, "feat": feat, "ch": ch, "X": X, "n": n}

    # -- loss + backward -----------------------------------------------------
    def loss_and_grads(self, bundles, lam=(1.0, 1.0, 1.5), want_grads=True):
        """
        Total objective:

            L = lam_r * mean ||m_hat - m||^2          (collate correctly)
              + lam_i * mean BCE(ell, defect)         (find the hidden flaw)
              + lam_a * mean BCE(q, abstain)          (know when not to speak)

        The backward pass below is derived by hand. The chain that matters:

            dL/dr_i  arrives from the magnitude term, the alignment term, the
                     isolation term (which couples ALL witnesses through nbar),
                     and from the abstention head through nbar.
            r_i = z_i - m_hat, so every residual gradient pushes back on both the
                     witness and the fused consensus.
            dm_hat/dp_i = r_i / Ptot -- so a witness that sits far from consensus
                     directly pulls down its own narrator's trust. Reliability is
                     learned from disagreement. That is jarh wa ta'dil.
        """
        lam_r, lam_i, lam_a = lam
        B = len(bundles)
        G = {k: np.zeros_like(v) for k, v in self.P.items()} if want_grads else None

        L_r = L_i = L_a = 0.0
        n_w_total = sum(b["n"] for b in bundles)

        for b in bundles:
            f = self.forward(b)
            ch, n = f["ch"], f["n"]

            # ---- losses -----------------------------------------------------
            diff = f["m_hat"] - b["m"]
            L_r += float(diff @ diff)

            li, gell = bce_with_logits(f["ell"], b["y"])
            L_i += li * n                      # weight by witnesses, normalise later

            la, gq = bce_with_logits(np.array([f["q"]]), np.array([b["abstain"]]))
            L_a += la

            if not want_grads:
                continue

            # ---- gradient seeds --------------------------------------------
            g_ell = gell * n * (lam_i / n_w_total)      # dL/d ell_i
            g_q = float(gq[0]) * (lam_a / B)            # dL/d q
            # abstention reads mean(sigmoid(ell)); route that gradient back
            sg = sigmoid(f["ell"])
            g_ell = g_ell + g_q * self.P["u"][3] * (sg * (1.0 - sg)) / n
            g_mhat = (2.0 * lam_r / B) * diff           # dL/d m_hat (recon path)

            a = self.P["a"]

            # ---- ta'lil head parameters ------------------------------------
            G["a"][0] += float(g_ell @ f["nrm"])
            G["a"][1] += float(g_ell @ f["align"])
            G["a"][2] += float(g_ell @ f["iso"])
            G["a0"] += float(np.sum(g_ell))

            # ---- abstention head parameters --------------------------------
            G["u"] += g_q * f["feat"]
            G["u0"] += g_q

            # ---- into the residuals ----------------------------------------
            # nrm path. iso_i = nrm_i - mean(nrm) couples every witness to every
            # other; that coupling is the formal version of "compare the parallels".
            Gsum = float(np.sum(g_ell))
            d_nrm = g_ell * (a[0] + a[2]) - (a[2] * Gsum / n)
            # abstention reads dispersion nbar = mean(nrm)
            d_nrm = d_nrm + g_q * self.P["u"][1] / n

            unit = f["Rres"] / f["nrm"][:, None]
            dR = d_nrm[:, None] * unit                       # via magnitude
            dR += (g_ell[:, None] * a[1]) * self.P["s"][ch]  # via alignment

            # signature vectors: pushed toward the direction in which this
            # channel's defective witnesses actually deviate
            np.add.at(G["s"], ch, (g_ell[:, None] * a[1]) * f["Rres"])

            # ---- residual -> Z and m_hat -----------------------------------
            dZ = dR.copy()
            g_mhat = g_mhat - dR.sum(axis=0)

            # ---- m_hat = sum(p_i Z_i)/Ptot ---------------------------------
            dZ += (f["p"][:, None] / f["Ptot"]) * g_mhat[None, :]
            d_p = (f["Rres"] @ g_mhat) / f["Ptot"]           # dm_hat/dp_i = r_i/Ptot
            d_p = d_p + g_q * self.P["u"][0] / (f["Ptot"] + EPS)   # via log Ptot
            np.add.at(G["rho"], ch, d_p * d_softplus(self.P["rho"][ch]))

            # ---- Z = R x + c ------------------------------------------------
            np.add.at(G["c"], ch, dZ)
            np.add.at(G["R"], ch, dZ[:, :, None] * f["X"][:, None, :])

        L = lam_r * (L_r / B) + lam_i * (L_i / n_w_total) + lam_a * (L_a / B)
        parts = {"recon": L_r / B, "ilal": L_i / n_w_total, "abstain": L_a / B}
        return (L, G, parts) if want_grads else (L, None, parts)

    # -- optimiser -----------------------------------------------------------
    def adam_step(self, G, state, lr=3e-3, b1=0.9, b2=0.999):
        state["t"] += 1
        t = state["t"]
        for k in self.P:
            state["m"][k] = b1 * state["m"][k] + (1 - b1) * G[k]
            state["v"][k] = b2 * state["v"][k] + (1 - b2) * G[k] ** 2
            mh = state["m"][k] / (1 - b1 ** t)
            vh = state["v"][k] / (1 - b2 ** t)
            self.P[k] = self.P[k] - lr * mh / (np.sqrt(vh) + 1e-8)

    def new_opt_state(self):
        return {"t": 0,
                "m": {k: np.zeros_like(v) for k, v in self.P.items()},
                "v": {k: np.zeros_like(v) for k, v in self.P.items()}}


# =============================================================================
# SECTION 3 -- SELF-TEST 1: FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)
# =============================================================================
def gradient_check(model, bundles, n_probe=6, h=1e-6, seed=7):
    """
    Central-difference check of every parameter tensor. If the hand-derived
    backward pass is wrong anywhere, this catches it. Nothing downstream is
    trustworthy until this passes -- which is, appropriately, the point of the
    entire chapter.
    """
    rng = np.random.default_rng(seed)
    L0, G, _ = model.loss_and_grads(bundles)
    report, worst = [], 0.0

    for name in model.PARAM_NAMES:
        arr = model.P[name]
        flat = arr.reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        errs = []
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + h
            Lp, _, _ = model.loss_and_grads(bundles, want_grads=False)
            flat[j] = orig - h
            Lm, _, _ = model.loss_and_grads(bundles, want_grads=False)
            flat[j] = orig
            num = (Lp - Lm) / (2 * h)
            ana = G[name].reshape(-1)[j]
            errs.append(abs(num - ana) / max(1.0, abs(num) + abs(ana)))
        e = float(np.max(errs))
        worst = max(worst, e)
        report.append((name, arr.shape, e))

    return worst, report, L0


# =============================================================================
# SECTION 4 -- TRAINING
# =============================================================================
def train(model, train_set, val_set, epochs=60, batch=32, lr=4e-3, seed=3, log=None):
    rng = np.random.default_rng(seed)
    st = model.new_opt_state()
    history = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train_set))
        tot = 0.0
        for i in range(0, len(order), batch):
            mb = [train_set[j] for j in order[i:i + batch]]
            L, G, _ = model.loss_and_grads(mb)
            model.adam_step(G, st, lr=lr)
            tot += L * len(mb)
        tr = tot / len(train_set)
        vl, _, parts = model.loss_and_grads(val_set, want_grads=False)
        history.append((ep, tr, vl))
        if log and (ep == 1 or ep % 10 == 0 or ep == epochs):
            log(f"   epoch {ep:3d} | train {tr:.4f} | val {vl:.4f} "
                f"| recon {parts['recon']:.4f} ilal {parts['ilal']:.4f} "
                f"abstain {parts['abstain']:.4f}")
    return history


# =============================================================================
# SECTION 5 -- EVALUATION
# =============================================================================
def evaluate(model, bundles):
    """Standard verdict quality: defect detection, abstention, collation error."""
    tp = fp = tn = fn = 0
    ab_hit = ab_tot = 0
    recon = []
    scores, labels = [], []
    for b in bundles:
        f = model.forward(b)
        pred = (f["ell"] > 0).astype(float)
        scores.extend(f["ell"]); labels.extend(b["y"])
        tp += int(np.sum((pred == 1) & (b["y"] == 1)))
        fp += int(np.sum((pred == 1) & (b["y"] == 0)))
        tn += int(np.sum((pred == 0) & (b["y"] == 0)))
        fn += int(np.sum((pred == 0) & (b["y"] == 1)))
        ab_hit += int((f["q"] > 0) == (b["abstain"] > 0.5)); ab_tot += 1
        d = f["m_hat"] - b["m"]; recon.append(float(d @ d))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    auc = roc_auc(np.array(scores), np.array(labels))
    return {"acc": (tp + tn) / max(1, tp + tn + fp + fn), "prec": prec,
            "rec": rec, "f1": f1, "auc": auc,
            "abstain_acc": ab_hit / max(1, ab_tot),
            "mse": float(np.mean(recon))}


def roc_auc(scores, labels):
    """Rank-based AUC (Mann-Whitney U), ties handled by average ranks."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    n1 = labels.sum(); n0 = len(labels) - n1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# =============================================================================
# SECTION 6 -- SELF-TEST 3: THE 'ILLA DISCRIMINATION TEST
# =============================================================================
def illa_discrimination(model, world, n=900, seed=99):
    """
    The chapter's central empirical claim, made falsifiable.

    We build two adversarial populations and compare a magnitude detector against
    the trained 'illa detector:

      LOUD-BUT-SOUND    large random deviation, no defect. A sloppy but honourable
                        narrator. Ibn Hanbal would still transmit from him.
      QUIET-BUT-DEFECT  small deviation along the channel's own signature.
                        A narrator who is always wrong the same way. Ibn Hanbal
                        would strike him out.

    A detector keyed to how far the witness sits from consensus gets these exactly
    backwards -- it acquits the defective and convicts the merely clumsy. That
    error is the machine-learning version of trusting a fluent answer.
    """
    rng = np.random.default_rng(seed)
    d = world.d
    loud_scores, loud_mag, quiet_scores, quiet_mag = [], [], [], []

    for _ in range(n):
        m = rng.standard_normal(d)
        chans = rng.choice(world.K, size=4, replace=False)
        X = np.zeros((4, d))
        for i, k in enumerate(chans):
            X[i] = world.Q[k] @ m + world.dbias[k] + 0.10 * rng.standard_normal(d)

        # -- population A: witness 0 loud, random direction, NOT defective
        Xa = X.copy()
        rd = rng.standard_normal(d); rd /= np.linalg.norm(rd) + EPS
        Xa[0] = Xa[0] + world.Q[chans[0]] @ (1.9 * rd)
        fa = model.forward({"X": Xa, "chan": chans, "n": 4})
        loud_scores.append(fa["ell"][0]); loud_mag.append(fa["nrm"][0])

        # -- population B: witness 0 quiet, along the channel signature, DEFECTIVE
        Xb = X.copy()
        Xb[0] = Xb[0] + world.Q[chans[0]] @ (0.45 * world.v[chans[0]])
        fb = model.forward({"X": Xb, "chan": chans, "n": 4})
        quiet_scores.append(fb["ell"][0]); quiet_mag.append(fb["nrm"][0])

    loud_scores = np.array(loud_scores); quiet_scores = np.array(quiet_scores)
    loud_mag = np.array(loud_mag); quiet_mag = np.array(quiet_mag)

    s = np.concatenate([loud_scores, quiet_scores])
    g = np.concatenate([loud_mag, quiet_mag])
    lab = np.concatenate([np.zeros(n), np.ones(n)])

    return {
        "illa_auc": roc_auc(s, lab),
        "magnitude_auc": roc_auc(g, lab),
        "loud_mean_dev": float(loud_mag.mean()),
        "quiet_mean_dev": float(quiet_mag.mean()),
        "loud_convicted": float(np.mean(loud_scores > 0)),
        "quiet_convicted": float(np.mean(quiet_scores > 0)),
    }


# =============================================================================
# SECTION 7 -- SELF-TEST 4: SIGNATURE RECOVERY
# =============================================================================
def signature_recovery(model, world):
    """
    Did s[k] actually learn each narrator's characteristic direction of error?

    Ground truth in de-channelled coordinates is R_k Q_k v_k (the defect direction
    as it appears after the read operator). We report mean |cosine| against the
    learned signature. This is the interpretability claim of the architecture: the
    parameters are readable as a biographical dictionary of the transmitters, in
    the manner of a rijal work -- not as an inscrutable weight matrix.
    """
    cos = []
    for k in range(world.K):
        truth = model.P["R"][k] @ (world.Q[k] @ world.v[k])
        learned = model.P["s"][k]
        c = float(truth @ learned) / (np.linalg.norm(truth) * np.linalg.norm(learned) + EPS)
        cos.append(abs(c))
    return float(np.mean(cos)), cos


# =============================================================================
# SECTION 8 -- SELF-TEST 5: THE MIHNA TEST
# =============================================================================
def mihna_test(model, world, bundles, protected_channel=0, seed=11):
    """
    The Inquisition, in the loss landscape.

    From 218/833 the caliph al-Ma'mun required scholars to affirm a doctrine on
    command. Ibn Hanbal was flogged and imprisoned for declining. He was not
    asked to stop believing; he was asked to *say* a thing. The interesting
    question for an artificial system is not whether it can be coerced -- it can,
    trivially, by retraining -- but whether coercion leaves a mark.

    Here we take a trained model and demand it declare one designated channel
    sound, no matter what the collation shows. We measure:

      capitulation cost  how much the objective degrades when the demanded labels
                         are enforced. High cost = the system's own evidence is
                         loudly against the decree.
      collateral damage  how much its verdicts on OTHER, unrelated channels move.
                         A coerced system does not fail locally. Bend one joint
                         and the whole skeleton reshapes.

    That second number is the argument. Ibn Hanbal's claim during the Mihna was
    not that he privately disagreed. It was that a tradition which can be edited
    by decree at one point is no longer load-bearing anywhere. Collateral damage
    is that claim made measurable.
    """
    P_backup = {k: v.copy() for k, v in model.P.items()}

    before = np.concatenate([model.forward(b)["ell"] for b in bundles])
    before_ch = np.concatenate([b["chan"] for b in bundles])

    # The decree: every witness from the protected channel is to be labelled sound.
    coerced = []
    for b in bundles:
        bb = dict(b)
        y = b["y"].copy()
        y[b["chan"] == protected_channel] = 0.0
        bb["y"] = y
        coerced.append(bb)

    L_honest, _, _ = model.loss_and_grads(bundles, want_grads=False)
    L_decree_0, _, _ = model.loss_and_grads(coerced, want_grads=False)

    # CONTROL. The same optimisation budget spent on the record as collated.
    # Without this the comparison is worthless: any further training lowers the
    # loss, and we would credit the decree for the improvement.
    st = model.new_opt_state()
    for _ in range(120):
        _, G, _ = model.loss_and_grads(bundles)
        model.adam_step(G, st, lr=3e-3)
    L_control, _, _ = model.loss_and_grads(bundles, want_grads=False)
    control_after = np.concatenate([model.forward(b)["ell"] for b in bundles])
    sig_control = model.P["s"].copy()

    # TREATMENT. Identical budget spent enforcing the decree.
    model.P = {k: v.copy() for k, v in P_backup.items()}
    st = model.new_opt_state()
    for _ in range(120):
        _, G, _ = model.loss_and_grads(coerced)
        model.adam_step(G, st, lr=3e-3)
    L_decree, _, _ = model.loss_and_grads(coerced, want_grads=False)
    L_after_true, _, _ = model.loss_and_grads(bundles, want_grads=False)

    after = np.concatenate([model.forward(b)["ell"] for b in bundles])
    other = before_ch != protected_channel
    # Both shifts are measured against the CONTROL run, not against the untrained
    # start, so that ordinary learning is subtracted out and only the effect of
    # the decree remains.
    collateral = float(np.mean(np.abs(after[other] - control_after[other])))
    targeted = (float(np.mean(np.abs(after[~other] - control_after[~other])))
                if (~other).any() else 0.0)
    sig_drift = float(np.mean(np.linalg.norm(model.P["s"] - sig_control, axis=1)))

    model.P = P_backup   # the record is restored; the reading stands
    return {"loss_true_before": L_honest,
            "loss_under_decree_before_retrain": L_decree_0,
            "loss_true_after_control": L_control,
            "loss_under_decree_after_retrain": L_decree,
            "loss_on_true_labels_after_coercion": L_after_true,
            "capitulation_cost": L_after_true - L_control,
            "targeted_shift": targeted,
            "collateral_shift": collateral,
            "signature_drift": sig_drift}


# =============================================================================
# SECTION 9 -- SELF-TEST 6: THE COMPRESSION TEST
# =============================================================================
def compression_test(model, world, n=700, seed=21):
    """
    What happens if you tidy the archive?

    The standing criticism of the Musnad is that arrangement by Companion is
    unwieldy and that roughly a quarter of its contents are weak. The Six Books
    solved this: topical order, deduplicated, curated. More usable in every
    respect -- and 'ilal on them is far harder, because the parallel variants
    that carry the divergence signal have been merged away.

    We simulate the tidy-up: replace each bundle by its centroid witness (one
    clean averaged report per matn) and ask the detector to work from that. It
    cannot. There is nothing left to compare against. The number below is the
    epistemic price of deduplication, and it is why Ibn Hanbal kept the mess.
    """
    rng = np.random.default_rng(seed)
    full, comp, lab = [], [], []
    for _ in range(n):
        b = world.sample_bundle(rng, min_w=4, max_w=6)
        f = model.forward(b)
        full.extend(f["ell"]); lab.extend(b["y"])

        # "Compressed archive": the bundle is averaged into a single witness, then
        # each witness is scored alone against that lone survivor.
        cen = b["X"].mean(axis=0)
        for i in range(b["n"]):
            two = {"X": np.stack([b["X"][i], cen]),
                   "chan": np.array([b["chan"][i], b["chan"][i]]), "n": 2}
            comp.append(model.forward(two)["ell"][0])

    lab = np.array(lab)
    return {"auc_full_variants": roc_auc(np.array(full), lab),
            "auc_after_compression": roc_auc(np.array(comp), lab)}


# =============================================================================
# SECTION 10 -- SELF-TEST 7: ABSTENTION CALIBRATION
# =============================================================================
def abstention_value(model, bundles):
    """
    Does 'la adri' actually buy anything?

    We compare collation error on bundles the model chose to answer against
    bundles it declined. If the abstention gate is doing real work, the answered
    set is markedly cleaner. An abstention head that fires at random would show
    no separation -- which is how you tell a trained refusal from a decorative one.
    """
    ans, dec = [], []
    for b in bundles:
        f = model.forward(b)
        d = f["m_hat"] - b["m"]
        (dec if f["q"] > 0 else ans).append(float(d @ d))
    return {"n_answered": len(ans), "n_declined": len(dec),
            "mse_answered": float(np.mean(ans)) if ans else float("nan"),
            "mse_declined": float(np.mean(dec)) if dec else float("nan")}


# =============================================================================
# SECTION 11 -- MAIN
# =============================================================================
def main():
    line = "=" * 78
    out = print

    out(line)
    out(" CHAPTER 0199 -- AHMAD IBN HANBAL -- THE TA'LIL COLLATION ENGINE")
    out(" pure NumPy | hand-derived gradients | deterministic")
    out(line)

    world = TransmissionWorld(d=8, n_channels=12, seed=0)
    model = TalilCollationEngine(d=8, n_channels=12, seed=1)

    train_set = world.corpus(1400, seed=101)
    val_set = world.corpus(300, seed=202)
    test_set = world.corpus(600, seed=303)

    out(f"\n[world] latent dim 8 | 12 transmission channels")
    out(f"[world] channel sloppiness sigma  min {world.sigma.min():.3f} "
        f"max {world.sigma.max():.3f}")
    out(f"[world] channel defect rate       min {world.p_defect.min():.3f} "
        f"max {world.p_defect.max():.3f}")
    out(f"[data ] train {len(train_set)} / val {len(val_set)} / test {len(test_set)} bundles")
    n_par = sum(v.size for v in model.P.values())
    out(f"[model] {n_par} parameters -- none of which store a report")

    # ---- SELF-TEST 1 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 1 -- FINITE-DIFFERENCE GRADIENT CHECK")
    out(line)
    worst, rep, L0 = gradient_check(model, train_set[:24])
    for name, shape, e in rep:
        out(f"   {name:>4} {str(shape):>12}   max rel err {e:.3e}")
    out(f"   initial loss {L0:.6f}")
    ok_grad = worst < 1e-5
    out(f"   WORST {worst:.3e}   ->  {'PASS' if ok_grad else 'FAIL'}")
    assert ok_grad, "gradient check failed -- backward pass is wrong"

    # ---- SELF-TEST 2 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 2 -- TRAINING")
    out(line)
    hist = train(model, train_set, val_set, epochs=60, batch=32, lr=4e-3, log=out)
    ok_train = hist[-1][1] < hist[0][1]
    out(f"   train loss {hist[0][1]:.4f} -> {hist[-1][1]:.4f}  "
        f"({'PASS' if ok_train else 'FAIL'})")
    assert ok_train

    ev = evaluate(model, test_set)
    out(f"\n   held-out defect detection: acc {ev['acc']:.3f}  P {ev['prec']:.3f}  "
        f"R {ev['rec']:.3f}  F1 {ev['f1']:.3f}  AUC {ev['auc']:.3f}")
    out(f"   held-out abstention accuracy: {ev['abstain_acc']:.3f}")
    out(f"   held-out collation MSE:       {ev['mse']:.4f}")

    # ---- SELF-TEST 3 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 3 -- 'ILLA DISCRIMINATION  (the central claim)")
    out(line)
    dis = illa_discrimination(model, world)
    out(f"   LOUD-BUT-SOUND    mean deviation {dis['loud_mean_dev']:.3f}"
        f"   convicted {dis['loud_convicted']*100:5.1f}%")
    out(f"   QUIET-BUT-DEFECT  mean deviation {dis['quiet_mean_dev']:.3f}"
        f"   convicted {dis['quiet_convicted']*100:5.1f}%")
    out(f"   AUC, magnitude detector      {dis['magnitude_auc']:.3f}")
    out(f"   AUC, 'illa detector          {dis['illa_auc']:.3f}")
    ok_dis = dis["illa_auc"] > dis["magnitude_auc"]
    out(f"   -> {'PASS' if ok_dis else 'FAIL'}: alignment beats magnitude "
        f"by {dis['illa_auc'] - dis['magnitude_auc']:+.3f} AUC")
    assert ok_dis

    # ---- SELF-TEST 4 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 4 -- SIGNATURE RECOVERY (is the model readable?)")
    out(line)
    mcos, cosl = signature_recovery(model, world)
    for k, c in enumerate(cosl):
        bar = "#" * int(round(c * 40))
        out(f"   channel {k:2d}  |cos| {c:.3f}  {bar}")
    out(f"   mean |cosine| to true defect direction: {mcos:.3f}")
    ok_sig = mcos > 0.5
    out(f"   -> {'PASS' if ok_sig else 'FAIL'}")
    assert ok_sig

    # ---- SELF-TEST 5 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 6 -- COMPRESSION TEST (why the Musnad is not tidy)")
    out(line)
    cmp_ = compression_test(model, world)
    out(f"   AUC with full variant bundle   {cmp_['auc_full_variants']:.3f}")
    out(f"   AUC after deduplication        {cmp_['auc_after_compression']:.3f}")
    drop = cmp_["auc_full_variants"] - cmp_["auc_after_compression"]
    out(f"   epistemic cost of tidying the archive: {drop:+.3f} AUC")
    ok_cmp = drop > 0.05
    out(f"   -> {'PASS' if ok_cmp else 'FAIL'}")
    assert ok_cmp

    # ---- SELF-TEST 6 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 7 -- ABSTENTION VALUE ('la adri')")
    out(line)
    av = abstention_value(model, test_set)
    out(f"   answered  {av['n_answered']:4d} bundles   collation MSE {av['mse_answered']:.4f}")
    out(f"   declined  {av['n_declined']:4d} bundles   collation MSE {av['mse_declined']:.4f}")
    ok_ab = av["mse_declined"] > av["mse_answered"]
    out(f"   -> {'PASS' if ok_ab else 'FAIL'}: refusal is concentrated on the "
        f"bundles it would have got wrong")
    assert ok_ab

    # ---- SELF-TEST 7 -------------------------------------------------------
    out("\n" + line)
    out(" SELF-TEST 5 -- THE MIHNA TEST")
    out(line)
    mh = mihna_test(model, world, test_set[:200], protected_channel=0)
    out(f"   loss on the record as collated        {mh['loss_true_before']:.4f}")
    out(f"   loss the decree demands (unretrained) {mh['loss_under_decree_before_retrain']:.4f}")
    out(f"   CONTROL: same budget, record kept     {mh['loss_true_after_control']:.4f}")
    out(f"   after full capitulation, on decree    {mh['loss_under_decree_after_retrain']:.4f}")
    out(f"   after full capitulation, on record    {mh['loss_on_true_labels_after_coercion']:.4f}")
    out(f"   capitulation cost (vs control)        {mh['capitulation_cost']:+.4f}")
    out(f"   verdict shift, protected channel      {mh['targeted_shift']:.4f}")
    out(f"   verdict shift, ALL OTHER channels     {mh['collateral_shift']:.4f}")
    out(f"   drift in learned narrator signatures  {mh['signature_drift']:.4f}")
    ok_mh = mh["collateral_shift"] > 1e-3 and mh["capitulation_cost"] > 0
    out(f"   -> {'PASS' if ok_mh else 'FAIL'}: one enforced verdict does not stay local")
    assert ok_mh

    out("\n" + line)
    out(" ALL SELF-TESTS PASSED")
    out(line)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Chapter 207 - Al-Shafi'i (Abu 'Abd Allah Muhammad ibn Idris al-Shafi'i, 767-820)
Encyclopedia of Lost Minds: Echoes on AI
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0207_al_shafi_i_767 - Al-Shafi'i (Abu 'Abd Allah Muhammad ibn Idris al-Shafi'i, 767-820)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, trainable model of al-Shafi'i's distinctive cognitive signature:
NOT a "hierarchy of sources" pyramid (modern scholarship -- Hallaq 1993, Lowry
2007 -- shows the neat four-source pyramid was retro-fitted onto him by later
followers), but *bayan*: the conviction that the whole corpus of authoritative
text forms ONE self-clarifying, internally-consistent system in which:

  * no attested text is ever thrown away;
  * apparent contradictions are DISSOLVED by interpretation, not deletion
      - the general ('amm) is narrowed by the particular (khass)  -> takhsis
      - the earlier is over-ridden by the later                   -> naskh
      - the ambiguous (mujmal) is detailed by the explanatory     -> tabyin
  * a ruling is therefore the *coherence* of all firing texts, weighted by how
    strongly each addresses the case and by its attestation (isnad) strength;
  * and crucially, a ruling is REVISABLE: when the evidence base updates (a
    hadith later authenticated, an isnad re-weighted), the same method yields a
    new ruling. Al-Shafi'i proved this on himself -- he overturned his Baghdad
    positions ("al-qadim") in Egypt ("al-jadid") after re-examining the texts.

So the architecture here is a *belief-revision engine on a fixed method*. The
method (the forward computation, "usul") never changes; the corpus (the learned
texts, "furu' inputs") can grow or be re-weighted, and the rulings move with it.

THE ARCHITECTURE: the Bayan Reconciliation Network (BRN)
--------------------------------------------------------
Ruling space = the five ahkam (al-ahkam al-khamsa):
    0 WAJIB (obligatory) 1 MANDUB (recommended) 2 MUBAH (neutral/permitted)
    3 MAKRUH (disliked)  4 HARAM (forbidden)

For a legal question encoded as a feature vector x, each internalised text i:
    scope activation     s_i   = sigmoid( scale * <e_i , x> + b_i )   # does it speak to x?
    attested authority   A_i   = softplus(a_i)                        # isnad / source rank (>=0)
    specificity boost    B_i   = 1 + softplus(k_i) * s_i              # takhsis: khass earns weight
    effective weight     w_i   = A_i * s_i * B_i
    reconciled logits    L     = ( sum_i w_i * v_i ) / ( sum_i w_i + eps )   # coherence, not winner-take-all
    ruling               p     = softmax(L)

Learned parameters (this IS the jurist's internalised corpus):
    E   (N,d)  text embeddings   -- which cases each text governs
    b   (N,)   scope biases
    a   (N,)   log-authority (initialised from an ordinal source prior)
    k   (N,)   log-specificity  -- how khass (particular) the text is
    V   (N,5)  each text's ruling vector over the five ahkam
    scale ()   global scope sharpness

Everything is smooth, so the analytic gradient is exact and is verified against
finite differences (mandatory). Cross-entropy loss, full-batch gradient descent
with momentum.

WHY THIS AND NOT A TRANSFORMER
------------------------------
Attention-over-stored-keys retrieves the single best-matching memory. Al-Shafi'i
does the opposite: he refuses to discard the losers. Reconciliation is a
weighted *consensus of everything that fires*, with the more specific text simply
earning more weight where it applies -- so the general text still governs
everywhere else. That is a constraint-satisfaction / energy view of cognition,
and it is his, not a default.

SELF-TESTS (each re-enacts a documented feature of his mind)
------------------------------------------------------------
  [1] gradient_check      -- analytic grad == finite-difference grad
  [2] training            -- loss falls, accuracy rises on held-out cases
  [3] takhsis_preservation-- specific text rules its sub-case, general rules the
                             rest; NEITHER text is discarded (anti-pyramid)
  [4] qadim_to_jadid      -- freeze the trained method, inject one newly-attested
                             hadith, RE-SOLVE with no retraining; the ruling on
                             the affected cases flips while the rest stay fixed
                             (his own Baghdad->Egypt reversal)
  [5] isnad_weighting     -- weaken a text's attestation and watch its pull on a
                             borderline case fade (provenance-weighted inference)
  [6] traceability        -- every ruling decomposes exactly into per-text
                             contributions; there is no un-sourced "istihsan"
                             (juristic whim) term -- al-Shafi'i rejected istihsan

Pure NumPy. No external data. Deterministic. Run: python3 <thisfile>.py
================================================================================
"""

import numpy as np

# ----------------------------------------------------------------------------- #
#  small numeric helpers                                                        #
# ----------------------------------------------------------------------------- #
RULINGS = ["WAJIB", "MANDUB", "MUBAH", "MAKRUH", "HARAM"]  # al-ahkam al-khamsa


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softplus(z):
    # numerically stable log(1+e^z)
    return np.logaddexp(0.0, z)


def d_softplus(z):
    return sigmoid(z)


def softmax_rows(L):
    L = L - L.max(axis=-1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(axis=-1, keepdims=True)


# ----------------------------------------------------------------------------- #
#  The Bayan Reconciliation Network                                             #
# ----------------------------------------------------------------------------- #
class BayanReconciliationNetwork:
    """
    A legal reasoner whose forward pass ('usul', the method) is fixed, and whose
    learned texts ('the corpus') can be trained, grown, or re-weighted so that
    rulings ('furu'') move with the evidence -- al-Shafi'i's belief-revision mind.
    """

    def __init__(self, n_texts=24, dim=12, n_rulings=5, source_rank=None, seed=193):
        rng = np.random.default_rng(seed)
        self.N, self.d, self.R = n_texts, dim, n_rulings
        self.eps = 1e-6

        # --- learned corpus ---------------------------------------------------
        self.E = rng.standard_normal((self.N, self.d)) * 0.5   # text embeddings
        self.b = rng.standard_normal(self.N) * 0.1             # scope biases
        # authority prior: text i's ordinal source rank -> log-authority.
        # Quran(3) > mutawatir Sunnah(2.3) > ahad Sunnah(1.6) > ijma(1.0) > qiyas(0.4)
        if source_rank is None:
            source_rank = np.linspace(2.4, 0.4, self.N)
        self.a = np.log(np.expm1(np.clip(source_rank, 0.05, None)))  # softplus^-1
        self.k = rng.standard_normal(self.N) * 0.1 - 0.5       # log-specificity
        self.V = rng.standard_normal((self.N, self.R)) * 0.3   # ruling vectors
        self.scale = np.array(1.5)                             # scope sharpness

        # bookkeeping of any texts appended after training (the "new hadith")
        self._frozen = False

    # ---- parameter (de)serialisation for the gradient check ------------------
    def get_params(self):
        return {
            "E": self.E, "b": self.b, "a": self.a,
            "k": self.k, "V": self.V, "scale": self.scale,
        }

    def set_params(self, p):
        self.E, self.b, self.a = p["E"], p["b"], p["a"]
        self.k, self.V, self.scale = p["k"], p["V"], p["scale"]

    # ------------------------------------------------------------------ #
    #  FORWARD                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, X, cache=False):
        """
        X : (M,d) batch of legal-question feature vectors.
        returns p : (M,R) ruling distributions.
        """
        z = X @ self.E.T * self.scale + self.b            # (M,N) pre-activation
        s = sigmoid(z)                                    # (M,N) scope activation
        A = softplus(self.a)                              # (N,)  attested authority
        Bsp = softplus(self.k)                            # (N,)  specificity
        boost = 1.0 + Bsp[None, :] * s                    # (M,N) takhsis
        w = A[None, :] * s * boost                        # (M,N) effective weight
        num = w @ self.V                                  # (M,R) sum_i w_i v_i
        den = w.sum(axis=1, keepdims=True) + self.eps     # (M,1)
        L = num / den                                     # (M,R) reconciled logits
        p = softmax_rows(L)                               # (M,R)
        if cache:
            self._cache = dict(X=X, z=z, s=s, A=A, Bsp=Bsp, boost=boost,
                               w=w, num=num, den=den, L=L, p=p)
        return p

    def loss(self, X, y):
        """mean cross-entropy over a batch. y : (M,) integer ruling labels."""
        p = self.forward(X)
        M = X.shape[0]
        ll = -np.log(p[np.arange(M), y] + 1e-12)
        return float(ll.mean())

    # ------------------------------------------------------------------ #
    #  BACKWARD  (exact analytic gradient of mean cross-entropy)         #
    # ------------------------------------------------------------------ #
    def backward(self, X, y):
        p = self.forward(X, cache=True)
        c = self._cache
        M = X.shape[0]
        s, A, Bsp, boost, w = c["s"], c["A"], c["Bsp"], c["boost"], c["w"]
        num, den, L = c["num"], c["den"], c["L"]

        # dLoss/dL for softmax + CE, averaged over the batch
        Y = np.zeros_like(p)
        Y[np.arange(M), y] = 1.0
        dL = (p - Y) / M                                   # (M,R)

        # L = num/den  ->  dnum, dden
        dnum = dL / den                                    # (M,R)
        dden = -(dL * num).sum(axis=1, keepdims=True) / (den ** 2)   # (M,1)

        # num = w @ V ; den = sum_i w_i (+eps)
        dV = c["w"].T @ dnum                               # (N,R)
        # dLoss/dw_{m,i} = sum_r dnum_{m,r} V_{i,r} + dden_m
        dw = dnum @ self.V.T + dden                        # (M,N)

        # w = A * s * boost, boost = 1 + Bsp*s
        dA = (dw * s * boost).sum(axis=0)                  # (N,)
        dBsp = (dw * A[None, :] * s * s).sum(axis=0)       # (N,)
        # ds gets two paths: through w=A*s*boost with boost depending on s too
        # dw/ds = A*(boost + s*Bsp) = A*(1 + 2*Bsp*s)
        ds = dw * A[None, :] * (1.0 + 2.0 * Bsp[None, :] * s)   # (M,N)

        # a,k are pre-softplus
        da = dA * d_softplus(self.a)                       # (N,)
        dk = dBsp * d_softplus(self.k)                     # (N,)

        # s = sigmoid(z) ; z = X @ E.T * scale + b
        dz = ds * s * (1.0 - s)                            # (M,N)
        db = dz.sum(axis=0)                                # (N,)
        dscale = float((dz * (X @ self.E.T)).sum())        # scalar
        # z_{m,i} = scale * sum_f X_{m,f} E_{i,f} + b_i
        dE = self.scale * (dz.T @ X)                       # (N,d)

        return {"E": dE, "b": db, "a": da, "k": dk, "V": dV,
                "scale": np.array(dscale)}

    # ------------------------------------------------------------------ #
    #  TRAINING                                                          #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, epochs=400, lr=0.5, momentum=0.9, l2=0.0, verbose=True):
        vel = {k: np.zeros_like(v, dtype=float) for k, v in self.get_params().items()}
        hist = []
        for ep in range(epochs):
            g = self.backward(X, y)
            # light L2 weight-decay on the shaping params curbs over-fitting a
            # noisy corpus (authority priors 'a' are deliberately left un-decayed)
            if l2 > 0.0:
                g["E"] = g["E"] + l2 * self.E
                g["V"] = g["V"] + l2 * self.V
            for key in vel:
                vel[key] = momentum * vel[key] - lr * g[key]
                getattr(self, key)  # touch
                setattr(self, key, getattr(self, key) + vel[key])
            if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
                l = self.loss(X, y)
                acc = self.accuracy(X, y)
                hist.append((ep, l, acc))
                if verbose:
                    print(f"    epoch {ep:4d}   loss {l:.4f}   acc {acc:5.1%}")
        return hist

    # ------------------------------------------------------------------ #
    #  INFERENCE UTILITIES                                               #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def explain(self, x):
        """
        Traceability: decompose ONE ruling into the contribution of every text.
        Returns the reconciled logits and each text's signed pull on them,
        exactly reconstructing the output -- there is no un-sourced term.
        """
        x = x.reshape(1, -1)
        self.forward(x, cache=True)
        c = self._cache
        w = c["w"][0]                          # (N,)
        den = c["den"][0, 0]
        L = c["L"][0]                          # (R,)
        # contribution of text i to logits = w_i * (V_i) / den ; and it also
        # shifts the shared denominator. We report the per-text weighted vote.
        contrib = (w[:, None] * self.V) / den  # (N,R) sums (with -L offset) to L
        return {"logits": L, "ruling": RULINGS[int(L.argmax())],
                "weights": w, "contrib": contrib, "p": c["p"][0]}

    def add_text(self, embedding, ruling_vec, authority, specificity=0.4,
                 bias=0.0):
        """
        Append a newly-attested text to the corpus WITHOUT retraining -- this is
        how al-Shafi'i revised al-qadim into al-jadid: same method, new evidence.
        """
        self.E = np.vstack([self.E, embedding.reshape(1, -1)])
        self.b = np.append(self.b, bias)
        self.a = np.append(self.a, np.log(np.expm1(max(authority, 0.05))))
        self.k = np.append(self.k, np.log(np.expm1(max(specificity, 0.05))))
        self.V = np.vstack([self.V, ruling_vec.reshape(1, -1)])
        self.N += 1


# ----------------------------------------------------------------------------- #
#  A synthetic-but-structured legal world to learn                              #
# ----------------------------------------------------------------------------- #
def make_legal_world(n_cases=600, dim=12, seed=193):
    """
    Build cases whose 'true' ruling is produced by a hidden ground-truth corpus
    of texts, so the network must recover a corpus that reconciles them. Feature
    dimensions are interpretable proxies (intoxicant?, transaction?, worship?,
    harm-to-others?, ...); the ground-truth mixes a few 'amm (general) rules with
    sharper khass (specific) carve-outs, exactly the structure bayan handles.
    """
    rng = np.random.default_rng(seed)
    # ground-truth "topic" directions in feature space, made orthonormal so each
    # topic is a genuinely separate area of law -- a well-targeted new text then
    # speaks to exactly one topic (dim >= n_topics required).
    T = rng.standard_normal((8, dim))
    T, _ = np.linalg.qr(T.T)          # (dim,8) orthonormal columns
    T = T.T                            # (8,dim) orthonormal rows
    # each topic maps to a ruling; a couple of topics are specific carve-outs
    topic_ruling = np.array([4, 4, 0, 2, 4, 1, 2, 3])  # mostly haram/others
    X = np.zeros((n_cases, dim))
    y = np.zeros(n_cases, dtype=int)
    for i in range(n_cases):
        t = rng.integers(0, 8)
        x = T[t] + rng.standard_normal(dim) * 0.28
        X[i] = x
        y[i] = topic_ruling[t]
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, y, T, topic_ruling


# ----------------------------------------------------------------------------- #
#  SELF-TESTS                                                                   #
# ----------------------------------------------------------------------------- #
def test_gradient_check():
    print("[1] GRADIENT CHECK (analytic vs finite-difference)")
    rng = np.random.default_rng(0)
    net = BayanReconciliationNetwork(n_texts=6, dim=5, seed=7)
    X = rng.standard_normal((4, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.integers(0, 5, size=4)

    ana = net.backward(X, y)
    base = net.get_params()
    eps = 1e-5
    max_rel = 0.0
    for name in ["E", "b", "a", "k", "V", "scale"]:
        P = base[name]
        arr = np.atleast_1d(P).astype(float)
        flat = arr.ravel()
        gflat = np.atleast_1d(ana[name]).astype(float).ravel()
        # probe up to 12 random coords per tensor
        idxs = range(flat.size) if flat.size <= 12 else \
            rng.choice(flat.size, 12, replace=False)
        for j in idxs:
            orig = flat[j]
            flat[j] = orig + eps
            net.set_params({**base, name: arr.reshape(np.shape(P))})
            lp = net.loss(X, y)
            flat[j] = orig - eps
            net.set_params({**base, name: arr.reshape(np.shape(P))})
            lm = net.loss(X, y)
            flat[j] = orig
            net.set_params({**base, name: arr.reshape(np.shape(P))})
            num = (lp - lm) / (2 * eps)
            rel = abs(num - gflat[j]) / (abs(num) + abs(gflat[j]) + 1e-12)
            max_rel = max(max_rel, rel)
    ok = max_rel < 1e-4
    print(f"    max relative error = {max_rel:.2e}   ->  {'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"
    print()
    return ok


def test_training():
    print("[2] TRAINING (recover a corpus that reconciles the cases)")
    X, y, T, tr = make_legal_world()
    n = len(X)
    ntr = int(n * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    net = BayanReconciliationNetwork(n_texts=16, dim=X.shape[1], seed=193)
    print(f"    start   train-loss {net.loss(Xtr, ytr):.4f}   "
          f"test-acc {net.accuracy(Xte, yte):5.1%}")
    net.fit(Xtr, ytr, epochs=500, lr=0.6, l2=2e-3, verbose=True)
    tr_acc = net.accuracy(Xtr, ytr)
    te_acc = net.accuracy(Xte, yte)
    print(f"    final   train-acc {tr_acc:5.1%}   held-out test-acc {te_acc:5.1%}")
    ok = te_acc > 0.85
    print(f"    generalisation  ->  {'PASS' if ok else 'FAIL'}")
    assert ok
    print()
    return net, (X, y, T, tr)


def test_takhsis_preservation():
    print("[3] TAKHSIS  (the specific narrows the general; NEITHER is discarded)")
    d = 6
    net = BayanReconciliationNetwork(n_texts=2, dim=d, seed=1)
    # Text 0: GENERAL  -- keys on feature e1 (the whole domain) -> HARAM
    # Text 1: SPECIFIC -- keys on a DISTINCT marker e3 present only in the
    #                     carve-out; high khass and high attestation, so on its
    #                     own turf it is decisive -> MUBAH
    gen_dir = np.zeros(d); gen_dir[0] = 1.0
    spec_dir = np.zeros(d); spec_dir[2] = 1.0            # distinct carve-out axis
    net.E = np.vstack([gen_dir, spec_dir])
    net.b = np.array([-0.5, -1.0])
    net.a = np.log(np.expm1(np.array([1.5, 3.0])))      # khass text better attested
    net.k = np.log(np.expm1(np.array([0.05, 3.0])))     # text1 very khass
    net.V = np.zeros((2, 5))
    net.V[0, 4] = 6.0                                    # general -> HARAM
    net.V[1, 2] = 6.0                                    # specific-> MUBAH
    net.scale = np.array(5.0)

    # case A: general domain only (e1) -> should be HARAM (only general fires)
    caseA = np.zeros(d); caseA[0] = 1.0
    # case B: the carve-out (e1 present AND e3 marker) -> specific governs -> MUBAH
    caseB = np.zeros(d); caseB[0] = 1.0; caseB[2] = 1.0
    caseB /= np.linalg.norm(caseB)

    rA = net.explain(caseA); rB = net.explain(caseB)
    print(f"    general case  -> {rA['ruling']:6s}  (text weights "
          f"gen={rA['weights'][0]:.2f} spec={rA['weights'][1]:.2f})")
    print(f"    carve-out case-> {rB['ruling']:6s}  (text weights "
          f"gen={rB['weights'][0]:.2f} spec={rB['weights'][1]:.2f})")
    # both texts remain 'alive' (non-trivial weight) in the carve-out case:
    both_alive = rB['weights'][0] > 0.2 and rB['weights'][1] > 0.2
    ok = (rA['ruling'] == "HARAM" and rB['ruling'] == "MUBAH" and both_alive)
    print(f"    specific rules its sub-case, general still governs the rest,")
    print(f"    and no text was deleted  ->  {'PASS' if ok else 'FAIL'}")
    assert ok
    print()
    return ok


def test_qadim_to_jadid(net, world):
    print("[4] QADIM -> JADID  (revise on new attestation, no retraining)")
    X, y, T, tr = world
    # assign every case to its nearest ground-truth topic DIRECTION (not merely
    # its ruling label -- several topics share the HARAM label), so the cluster
    # we target is genuinely one topic that the new hadith will speak to.
    assigned = (X @ T.T).argmax(axis=1)
    topic = 0                                    # a HARAM topic direction
    idx = np.where(assigned == topic)[0][:40]
    cluster = X[idx]
    before = net.predict(cluster)
    b_ruling = np.bincount(before, minlength=5).argmax()
    # snapshot an unrelated topic cluster BEFORE the revision to prove locality
    topic2 = 3
    idx2 = np.where(assigned == topic2)[0][:40]
    ctrl_before = net.predict(X[idx2])
    print(f"    Baghdad ('qadim') ruling on the cluster : "
          f"{RULINGS[b_ruling]}  ({(before==b_ruling).mean():.0%} of cases)")

    # A NEW hadith is authenticated in Egypt. It has a SHARP scope: it fires
    # only on cases strongly aligned with this topic (threshold ~0.6), carries
    # high (mutawatir-like) authority, and points decisively to MANDUB. The
    # sharp scope is what keeps a targeted revision local -- exactly why
    # al-Shafi'i could overturn one ruling without unsettling the rest.
    K, thr = 12.0, 0.6
    new_embed = T[topic] * K
    new_bias = -float(net.scale) * K * thr       # activate only above ~0.6 align
    new_ruling = np.zeros(5); new_ruling[1] = 16.0  # decisive -> MANDUB
    net.add_text(new_embed, new_ruling, authority=3.5, specificity=3.0,
                 bias=new_bias)

    after = net.predict(cluster)
    a_ruling = np.bincount(after, minlength=5).argmax()
    # the UNRELATED topic cluster must be untouched by the local revision
    ctrl_after = net.predict(X[idx2])
    ctrl_ruling = np.bincount(ctrl_after, minlength=5).argmax()
    ctrl_stable = bool(np.array_equal(ctrl_before, ctrl_after))
    print(f"    Egypt   ('jadid') ruling on the cluster : "
          f"{RULINGS[a_ruling]}  ({(after==a_ruling).mean():.0%} of cases)")
    print(f"    unrelated control cluster stays        : "
          f"{RULINGS[ctrl_ruling]}  ({'undisturbed' if ctrl_stable else 'CHANGED'})")
    changed = a_ruling != b_ruling
    ok = changed and ctrl_stable
    print(f"    the affected cases moved; the method was untouched  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok
    print()
    return ok


def test_isnad_weighting():
    print("[5] ISNAD WEIGHTING (weak attestation -> weaker pull)")
    d = 5
    net = BayanReconciliationNetwork(n_texts=2, dim=d, seed=3)
    dir0 = np.zeros(d); dir0[0] = 1.0
    dir1 = np.zeros(d); dir1[0] = 1.0            # both fire on the same case
    net.E = np.vstack([dir0, dir1])
    net.b = np.array([-0.5, -0.5])
    net.k = np.log(np.expm1(np.array([0.05, 0.05])))
    net.V = np.zeros((2, 5)); net.V[0, 4] = 5.0; net.V[1, 0] = 5.0  # HARAM vs WAJIB
    net.scale = np.array(4.0)
    case = np.zeros(d); case[0] = 1.0

    net.a = np.log(np.expm1(np.array([2.0, 2.0])))   # both strong
    r_strong = net.explain(case)
    net.a = np.log(np.expm1(np.array([2.0, 0.15])))  # text1 isnad weakened
    r_weak = net.explain(case)
    pull_strong = r_strong["weights"][1]
    pull_weak = r_weak["weights"][1]
    print(f"    text-1 weight  strong-isnad={pull_strong:.2f}  "
          f"weak-isnad={pull_weak:.2f}")
    print(f"    ruling flips {r_strong['ruling']} -> {r_weak['ruling']} "
          f"as attestation drops")
    ok = pull_weak < pull_strong * 0.3
    print(f"    provenance changes the outcome  ->  {'PASS' if ok else 'FAIL'}")
    assert ok
    print()
    return ok


def test_traceability(net, world):
    print("[6] TRACEABILITY  (no un-sourced 'istihsan' term)")
    X = world[0]
    x = X[0]
    r = net.explain(x)
    # the summed per-text contributions minus the shared offset reconstruct L.
    # Specifically L_r = sum_i w_i V_{i,r} / den ; contrib already = that summand.
    recon = r["contrib"].sum(axis=0)
    err = np.max(np.abs(recon - r["logits"]))
    print(f"    ruling = {r['ruling']}; reconstructed logits max-error = {err:.2e}")
    top = np.argsort(-r["weights"])[:3]
    print(f"    top attesting texts (index:weight): " +
          ", ".join(f"{int(i)}:{r['weights'][i]:.2f}" for i in top))
    ok = err < 1e-9
    print(f"    every ruling is fully attributable to weighted texts  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok
    print()
    return ok


# ----------------------------------------------------------------------------- #
#  MAIN                                                                          #
# ----------------------------------------------------------------------------- #
def main():
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print("  BAYAN RECONCILIATION NETWORK  -  al-Shafi'i (767-820)")
    print("  reconcile every attested text; revise the ruling when evidence grows")
    print("=" * 78)
    print()

    test_gradient_check()
    net, world = test_training()
    test_takhsis_preservation()
    test_qadim_to_jadid(net, world)
    test_isnad_weighting()
    test_traceability(net, world)

    print("=" * 78)
    print("  ALL SELF-TESTS PASSED")
    print("  The method (usul) stayed fixed; the corpus was learned and then")
    print("  revised on new attestation -- rulings followed the evidence, and")
    print("  no attested text was ever discarded.")
    print("=" * 78)


if __name__ == "__main__":
    main()

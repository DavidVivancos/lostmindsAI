"""
================================================================================
Chapter 0153_mesrop_mashtots_362 - Mesrop Mashtots (362-440 CE)
The Aybuben Engine: A Phonemic Codebook Learner
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 8 Minds 141 - 160 Available on Amazon https://www.amazon.com/dp/B0HH8RTCXF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 153: Mesrop Mashtots (362-440 CE)
================================================================================  

WHAT THIS FILE IS
-----------------
A from-scratch, pure-NumPy cognitive architecture that embodies the one idea
that is Mashtots's alone: intelligence begins with the design of a *complete,
minimal, near-bijective discrete code* for a signal space. Before you can think
in a language, someone must first mint the atoms of that language — one symbol
per distinct sound, no gaps, no ambiguity, and few enough that an ordinary mind
can absorb the whole set in weeks. Mashtots did this for Armenian around 405 CE:
he studied Greek, Syriac, Aramaic and Persian scripts, judged that every one of
them left *gaps* in Armenian phonetics (missing vowels, mismatched consonants),
and instead engineered a fresh 36-letter alphabet in which each letter maps to
exactly one phoneme of Classical Armenian — and doubles as a numeral.

Translated into machine learning, that act is the problem of *learned
tokenization / discrete representation*: discover the smallest codebook that
(1) covers every region of the signal manifold (no un-writable sound),
(2) decodes without ambiguity (one symbol -> one sound), and
(3) transfers — a fresh reader can learn to read a new speaker quickly because
    the atoms were chosen well, not borrowed from a foreign script.

THE MODEL (its parts, in Mashtots's terms)
------------------------------------------
    Scriptorium        the learned codebook C: K prototype "letters" in R^D.
    Ear -> Assignment  a soft vector-quantizer: each incoming sound-frame is
                       softly assigned to letters via a temperature softmax.
    Reed (Decoder)     reconstructs the sound from its assigned letters. If a
                       sound cannot be rebuilt, the alphabet has a *gap*.
    Four design pressures, exactly the ones Mashtots balanced:
        completeness+losslessness  -> reconstruction error (a gap = a sound you
                                      cannot write; ambiguity = a sound you
                                      cannot recover)
        bijectivity (one-per-sound)-> per-frame assignment entropy (minimise)
        minimality (36 not 360)    -> a concave sparsity penalty on letter usage
    Numeral overlay    after training, letters are canonically ordered and each
                       is given a positional numeric value (aybuben = script AND
                       number system).
    Literacy transfer  freeze the alphabet, teach a fresh linear "reader" to
                       decode a NEW speaker; measure how fast it learns. A good
                       alphabet transfers; a borrowed one leaves the reader lost.

RIGOR
-----
    * Pure NumPy, no autodiff. Analytic gradient of the full training loss w.r.t.
      the codebook C, derived by hand and VERIFIED against finite differences
      (mandatory gradient check).
    * A real Adam training loop.
    * Self-tests that assert the Mashtotsian claims: recon falls, active-letter
      count collapses toward the true phoneme inventory, ambiguity falls, and
      the native alphabet out-reads a "foreign script" that has gaps.

Run:  python3 chapter_0153_mesrop_mashtots_362.py
"""

import numpy as np


# =============================================================================
# SECTION 0 — A SYNTHETIC "LANGUAGE": phonemes, utterances, speakers
# =============================================================================
# Mashtots did not invent sounds; he invented symbols for sounds that already
# existed. So our data is a fixed inventory of true phoneme prototypes in a
# D-dimensional acoustic-like space. An "utterance" is a stream of frames, each
# a noisy sample of one phoneme. Different speakers apply an "accent" (a small
# rotation + shift). The learner never sees the phoneme labels — exactly as
# Mashtots had to *discover* the sound inventory of Armenian by ear.

def make_phoneme_inventory(n_phonemes: int, dim: int, seed: int = 405) -> np.ndarray:
    """Return the true, hidden set of phoneme prototypes (n_phonemes x dim).

    These are the atoms the alphabet must learn to name. The learner is not
    told how many there are — discovering that count IS the minimality problem.
    """
    rng = np.random.default_rng(seed)
    # Spread prototypes out so a *complete* code needs to cover all of them,
    # but keep two pairs deliberately close (the hard-to-distinguish sounds a
    # sloppy alphabet would merge — e.g. Armenian's aspirated/plain consonants).
    protos = rng.normal(0.0, 1.0, size=(n_phonemes, dim))
    protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9)
    protos *= 3.0
    if n_phonemes >= 4:
        protos[1] = protos[0] + rng.normal(0, 0.35, size=dim)   # near-minimal pair
        protos[3] = protos[2] + rng.normal(0, 0.35, size=dim)
    return protos


def speak(protos: np.ndarray, n_frames: int, noise: float = 0.25,
          accent=None, seed: int = 0) -> np.ndarray:
    """Produce an utterance: n_frames noisy samples drawn from the phonemes.

    `accent` is an optional (rotation, shift) pair applied to every frame,
    modelling a different speaker of the SAME language.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, protos.shape[0], size=n_frames)
    frames = protos[idx] + rng.normal(0, noise, size=(n_frames, protos.shape[1]))
    if accent is not None:
        R, shift = accent
        frames = frames @ R.T + shift
    return frames.astype(np.float64), idx


def random_accent(dim: int, seed: int = 7):
    """A gentle rotation + translation: the same language, a different mouth."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(dim, dim))
    # Orthonormalise via QR, then damp toward identity so it stays a mild accent.
    Q, _ = np.linalg.qr(A)
    R = 0.85 * np.eye(dim) + 0.15 * Q
    shift = rng.normal(0, 0.2, size=dim)
    return R, shift


# =============================================================================
# SECTION 1 — THE AYBUBEN ENGINE (the codec / codebook learner)
# =============================================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-300)


class AybubenEngine:
    """A soft vector-quantiser that learns a minimal, complete, bijective code.

    Parameters
    ----------
    dim        : dimensionality of the sound space.
    n_letters  : K, the *maximum* alphabet size. Minimality pressure will prune
                 this toward the true phoneme count. Mashtots gave himself room
                 (he studied many scripts) then settled on 36.
    tau        : softmax temperature. Low tau -> sharp, near-bijective reading.
    lam_bij    : weight on the one-letter-per-sound (bijectivity) pressure.
    lam_mdl    : weight on the minimality (fewest letters) pressure.
    """

    def __init__(self, dim, n_letters, tau=0.5, lam_bij=0.02, lam_mdl=0.15,
                 seed=362):
        rng = np.random.default_rng(seed)
        self.C = rng.normal(0, 1.0, size=(n_letters, dim))  # the scriptorium
        self.dim = dim
        self.K = n_letters
        self.tau = float(tau)
        self.lam_bij = float(lam_bij)
        self.lam_mdl = float(lam_mdl)
        self.mdl_eps = 1e-3

    # ---- forward pass ------------------------------------------------------
    def forward(self, X, C=None):
        """Assign frames to letters and reconstruct. Returns (loss, cache).

        The cache holds every intermediate needed for the analytic backward
        pass, so the gradient is exact rather than approximated.
        """
        if C is None:
            C = self.C
        N = X.shape[0]

        # Ear -> assignment: squared distance from each frame to each letter.
        # dist[n,k] = ||x_n - c_k||^2, computed without forming the N*K*D tensor.
        x2 = np.sum(X * X, axis=1, keepdims=True)          # N x 1
        c2 = np.sum(C * C, axis=1, keepdims=True).T        # 1 x K
        dist = x2 + c2 - 2.0 * (X @ C.T)                   # N x K
        logits = -dist / self.tau
        P = _softmax(logits)                               # N x K soft letters

        # Reed -> reconstruction of the sound from its letters.
        Xhat = P @ C                                       # N x D
        R = X - Xhat
        recon = np.mean(np.sum(R * R, axis=1))             # completeness+lossless

        # Bijectivity: how ambiguous is each frame's reading? (per-frame entropy)
        logP = np.log(P + 1e-12)
        ent = -np.sum(P * logP, axis=1)                    # N
        bij = np.mean(ent)

        # Minimality: concave penalty on letter usage -> favours FEW live letters.
        usage = P.mean(axis=0)                             # K   (soft usage)
        mdl = np.sum(np.sqrt(usage + self.mdl_eps))

        loss = recon + self.lam_bij * bij + self.lam_mdl * mdl
        cache = dict(X=X, C=C, N=N, P=P, logP=logP, Xhat=Xhat, R=R,
                     usage=usage, recon=recon, bij=bij, mdl=mdl)
        return loss, cache

    # ---- analytic backward pass -------------------------------------------
    def backward(self, cache):
        """Exact gradient of the total loss w.r.t. the codebook C.

        Derivation (all indices: n over frames, k over letters, d over dims):
            recon path A (C in Xhat=P@C, P held):  -(2/N) * sum_n R_n * P_nk
            through P -> logits -> dist -> C, from three loss terms:
                dLoss/dP from recon : -(2/N) (R_n . c_k)
                dLoss/dP from bij   :  lam_bij * -(1/N)(logP_nk + 1)
                dLoss/dP from mdl   :  lam_mdl * (1/N) * 0.5/sqrt(usage_k+eps)
            softmax jacobian: gl_nk = P_nk (gP_nk - sum_j gP_nj P_nj)
            logits=-dist/tau -> gd = -gl/tau
            dist=||x-c||^2   -> grad to c_k: sum_n (-2)*gd_nk*(x_n - c_k)
        The two contributions to C are summed.
        """
        X, C, N = cache['X'], cache['C'], cache['N']
        P, logP, R = cache['P'], cache['logP'], cache['R']
        usage = cache['usage']

        # (A) direct recon dependence of Xhat on C, with P held fixed.
        gC_direct = -(2.0 / N) * (P.T @ R)                 # K x D

        # (B) dLoss/dP, summed over the three terms that depend on P.
        gP_recon = -(2.0 / N) * (R @ C.T)                  # N x K   (R_n . c_k)
        gP_bij = self.lam_bij * (-(1.0 / N) * (logP + 1.0))
        gP_mdl = self.lam_mdl * (1.0 / N) * (0.5 / np.sqrt(usage + self.mdl_eps))[None, :]
        gP = gP_recon + gP_bij + gP_mdl                    # N x K

        # softmax jacobian-vector product.
        row = np.sum(gP * P, axis=1, keepdims=True)        # N x 1
        gl = P * (gP - row)                                # N x K
        gd = -gl / self.tau                                # N x K

        # dist -> C.  grad to c_k = sum_n (-2) gd_nk (x_n - c_k)
        #           = -2 [ (gd.T @ X)  -  (sum_n gd_nk) c_k ]
        gC_dist = -2.0 * (gd.T @ X - (gd.sum(axis=0)[:, None] * C))
        return gC_direct + gC_dist

    # ---- diagnostics -------------------------------------------------------
    def active_letters(self, X, thresh=0.01):
        """How many letters actually carry the language (usage above thresh)."""
        _, cache = self.forward(X)
        return int(np.sum(cache['usage'] > thresh))

    def read(self, X):
        """Return hard letter indices for each frame (the written transcript)."""
        _, cache = self.forward(X)
        return np.argmax(cache['P'], axis=1)


# =============================================================================
# SECTION 2 — ADAM OPTIMISER (tiny, from scratch)
# =============================================================================

class Adam:
    def __init__(self, shape, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return param


def train(engine: AybubenEngine, X, steps=400, lr=0.05, verbose=False):
    """A real training loop: forward, exact backward, Adam update."""
    opt = Adam(engine.C.shape, lr=lr)
    history = []
    for t in range(steps):
        loss, cache = engine.forward(X)
        grad = engine.backward(cache)
        engine.C = opt.step(engine.C, grad)
        history.append((loss, cache['recon'], cache['bij'],
                        int(np.sum(cache['usage'] > 0.01))))
        if verbose and (t % max(1, steps // 8) == 0 or t == steps - 1):
            print(f"  step {t:4d}  loss={loss:8.4f}  recon={cache['recon']:7.4f}"
                  f"  ambiguity={cache['bij']:6.4f}  active_letters="
                  f"{int(np.sum(cache['usage'] > 0.01)):2d}")
    return history


# =============================================================================
# SECTION 3 — NUMERAL OVERLAY (aybuben = alphabet AND number system)
# =============================================================================
# Mashtots's letters were also numerals: units, tens, hundreds, thousands.
# We reproduce the dual-code idea: order the learned letters canonically (here
# by a 1-D projection of their prototypes), then assign positional numeric
# values, so a single symbol set indexes both sound and quantity.

def numeral_overlay(engine: AybubenEngine, X):
    _, cache = engine.forward(X)
    active = np.where(cache['usage'] > 0.01)[0]
    C = engine.C[active]
    # Canonical order: project onto the top principal axis of the live letters.
    C0 = C - C.mean(0)
    _, _, Vt = np.linalg.svd(C0, full_matrices=False)
    order = np.argsort(C0 @ Vt[0])
    ordered_letters = active[order]
    values = []
    for i, _ in enumerate(ordered_letters):
        tier = i // 9                       # 0=units,1=tens,2=hundreds,...
        pos = i % 9 + 1
        values.append(pos * (10 ** tier))   # 1..9, 10..90, 100..900, ...
    return list(zip(ordered_letters.tolist(), values))


# =============================================================================
# SECTION 4 — LITERACY TRANSFER (why a NATIVE alphabet beats a BORROWED one)
# =============================================================================
# Mashtots's core empirical claim: Greek/Syriac left gaps, so Armenians read
# their own tongue badly through foreign letters. We test this directly.
#   * native codebook: learned on speaker A of the language.
#   * foreign codebook: learned on a DIFFERENT inventory (a "Greek" that lacks
#     some Armenian sounds -> structural gaps).
# Freeze each codebook, teach a fresh linear reader to recover speaker B's
# phoneme identity, and compare. The one with gaps reads worse.

def _one_hot(idx, k):
    Y = np.zeros((idx.shape[0], k))
    Y[np.arange(idx.shape[0]), idx] = 1.0
    return Y


def literacy_transfer(codebook: np.ndarray, tau, X_new, y_new, n_true,
                      ridge=1e-2):
    """Freeze `codebook`, fit a linear reader on the soft-letter features of a
    new speaker, and return classification accuracy of recovered phonemes.
    """
    x2 = np.sum(X_new * X_new, axis=1, keepdims=True)
    c2 = np.sum(codebook * codebook, axis=1, keepdims=True).T
    dist = x2 + c2 - 2.0 * (X_new @ codebook.T)
    feats = _softmax(-dist / tau)                     # frozen letters as features
    # closed-form ridge regression from letter-features to phoneme one-hots
    Y = _one_hot(y_new, n_true)
    A = feats.T @ feats + ridge * np.eye(feats.shape[1])
    W = np.linalg.solve(A, feats.T @ Y)
    pred = np.argmax(feats @ W, axis=1)
    return float(np.mean(pred == y_new))


# =============================================================================
# SECTION 5 — SELF-TESTS (each asserts a Mashtotsian claim)
# =============================================================================

def gradient_check():
    """MANDATORY: analytic gradient must match finite differences."""
    rng = np.random.default_rng(1)
    dim, K, N = 5, 7, 40
    X = rng.normal(0, 1, size=(N, dim))
    eng = AybubenEngine(dim, K, tau=0.6, lam_bij=0.05, lam_mdl=0.2, seed=3)
    loss0, cache = eng.forward(X)
    g_analytic = eng.backward(cache)

    eps = 1e-6
    g_num = np.zeros_like(eng.C)
    C = eng.C.copy()
    for k in range(K):
        for d in range(dim):
            Cp = C.copy(); Cp[k, d] += eps
            Cm = C.copy(); Cm[k, d] -= eps
            lp, _ = eng.forward(X, C=Cp)
            lm, _ = eng.forward(X, C=Cm)
            g_num[k, d] = (lp - lm) / (2 * eps)

    num = np.linalg.norm(g_analytic - g_num)
    den = np.linalg.norm(g_analytic) + np.linalg.norm(g_num) + 1e-12
    rel = num / den
    return rel, g_analytic, g_num


def run_all():
    print("=" * 74)
    print("THE AYBUBEN ENGINE  —  Mesrop Mashtots (c. 362-440 CE)")
    print("A phonemic codebook learner: complete, minimal, near-bijective code")
    print("=" * 74)

    # --- Test 1: gradient check ------------------------------------------
    print("\n[1] Gradient check (analytic vs finite-difference) ...")
    rel, ga, gn = gradient_check()
    print(f"    relative error = {rel:.3e}   "
          f"(sample analytic {ga[0,0]:+.5f} vs numeric {gn[0,0]:+.5f})")
    assert rel < 1e-6, "gradient check FAILED"
    print("    PASS: the hand-derived gradient is exact.")

    # --- Build the language ----------------------------------------------
    dim = 8
    n_phon = 12                      # the true (hidden) sound inventory
    protos = make_phoneme_inventory(n_phon, dim, seed=405)
    Xa, ya = speak(protos, 1500, noise=0.28, seed=1)          # speaker A

    # --- Test 2: training discovers a minimal, complete, sharp alphabet --
    print("\n[2] Learning the alphabet from speaker A "
          "(max 32 letters allowed) ...")
    eng = AybubenEngine(dim, n_letters=32, tau=0.35,
                        lam_bij=0.015, lam_mdl=0.22, seed=362)
    r0 = eng.forward(Xa)[1]['recon']
    b0 = eng.forward(Xa)[1]['bij']
    hist = train(eng, Xa, steps=500, lr=0.05, verbose=True)
    r1 = eng.forward(Xa)[1]['recon']
    b1 = eng.forward(Xa)[1]['bij']
    active = eng.active_letters(Xa)
    print(f"    reconstruction  {r0:6.3f} -> {r1:6.3f}   (a 'gap' = an "
          f"unwritable sound; it fell)")
    print(f"    ambiguity(entropy) {b0:6.3f} -> {b1:6.3f}   (reading sharpened "
          f"toward one-letter-per-sound)")
    print(f"    active letters allotted 32 -> settled on {active} "
          f"(true inventory = {n_phon})")
    assert r1 < r0 * 0.5, "reconstruction did not improve enough"
    assert b1 < b0, "ambiguity did not fall"
    assert active <= n_phon + 3, "minimality pressure failed to prune letters"
    print("    PASS: a small, complete, low-ambiguity code emerged.")

    # --- Test 3: numeral overlay -----------------------------------------
    print("\n[3] Numeral overlay (each letter also a number) ...")
    pairs = numeral_overlay(eng, Xa)
    shown = ", ".join(f"L{lid}={val}" for lid, val in pairs[:9])
    print(f"    first tier (units): {shown}")
    assert [v for _, v in pairs][:9] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("    PASS: one symbol set indexes both sound and quantity (aybuben).")

    # --- Test 4: literacy transfer, native vs borrowed script ------------
    print("\n[4] Literacy transfer to a NEW speaker (native vs borrowed script)")
    R, shift = random_accent(dim, seed=11)
    Xb, yb = speak(protos, 1500, noise=0.28, accent=(R, shift), seed=2)

    native_cb = eng.C[np.where(eng.forward(Xa)[1]['usage'] > 0.01)[0]]
    acc_native = literacy_transfer(native_cb, eng.tau, Xb, yb, n_phon)

    # a "foreign script": learned on a DIFFERENT inventory that omits 4 of the
    # language's sounds -> built-in gaps, exactly Mashtots's complaint.
    foreign_protos = protos[: n_phon - 4] + 0.0
    foreign_protos = np.vstack([foreign_protos,
                                make_phoneme_inventory(3, dim, seed=999)])
    Xf, _ = speak(foreign_protos, 1500, noise=0.28, seed=5)
    feng = AybubenEngine(dim, n_letters=32, tau=0.35,
                         lam_bij=0.015, lam_mdl=0.22, seed=77)
    train(feng, Xf, steps=500, lr=0.05)
    foreign_cb = feng.C[np.where(feng.forward(Xf)[1]['usage'] > 0.01)[0]]
    acc_foreign = literacy_transfer(foreign_cb, feng.tau, Xb, yb, n_phon)

    print(f"    native alphabet  reads speaker B at accuracy = {acc_native:.3f}")
    print(f"    borrowed script  reads speaker B at accuracy = {acc_foreign:.3f}")
    assert acc_native > acc_foreign + 0.05, "native code should transfer better"
    print("    PASS: the gap-free native alphabet out-reads the borrowed one.")

    print("\n" + "=" * 74)
    print("ALL TESTS PASSED — the architecture runs and embodies the thesis:")
    print("a mind's power begins with the atoms it is given to think in.")
    print("=" * 74)
    return dict(rel_grad=rel, recon0=r0, recon1=r1, bij0=b0, bij1=b1,
                active=active, n_phon=n_phon, acc_native=acc_native,
                acc_foreign=acc_foreign)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_all()

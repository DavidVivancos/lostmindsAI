#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0391 - THOMAS A KEMPIS (1380-1471)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
 THE RAPIARIUM-COLLOQUY NETWORK  (RCN)
================================================================================

WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
-------------------------------------------------------------------------------
Thomas a Kempis makes one cognitive claim that belongs to nobody else in the
corpus. In Book III, chapter 54 of the Imitation ("De diversis motibus naturae
et gratiae") he states that the two great inner impulses -- nature and grace --
produce *the same visible acts*. Both "desire that which is good"; both
"pretend some good in their words and their deeds"; they "can hardly be
distinguished". They differ in exactly one respect: where they finally come to
rest. Nature "always hath herself for her end and object". Grace "doeth all
things purely for God's sake, in Whom also she finally resteth".

That is not a moral platitude. It is a precise and unusual statement about
classification:

    The label is NOT a function of the observations.
    The label is a function of the LIMIT of the observations.

Two impulses can emit an identical distribution of acts and still be opposite,
because the diagnostic quantity is the attractor of the trajectory, not any
sample from it. A system that scores behaviour is therefore, in Thomas's exact
vocabulary, a system that scores hypocrisy.

The second claim is about training. Thomas was a copyist in a Devotio Moderna
house whose characteristic technology was the *rapiarium* -- a small personal
notebook of gathered sentences, continually re-copied, re-ordered and pruned.
The Imitation itself began as one. The claim embedded in that practice is that
formation comes from re-traversing a SMALL corpus until it becomes disposition,
not from traversing a large one once. And the signal that drives the re-reading
is *compunctio* -- I.1: he would rather feel compunction than be able to define
it -- which is a felt discrepancy between what one professes and what one does.

So the network below has no attention, no key-value store, no mixture of
experts. It has five parts, each of which is one of his claims made mechanical:

  1. ENCODER                 -- acts become states.
  2. COLLOQUY                -- Book III's literal form: an alternating two-voice
                                recurrence. Vox (the discerning voice) updates,
                                then Discipulus (the surface voice) answers it.
  3. RAPIARIUM               -- K learned "sentences" spanning a subspace.
                                Imitation is implemented as PROJECTION onto that
                                subspace; the residual is the proprium, self-will.
                                A repulsion penalty enforces that no two
                                sentences say the same thing (pruning).
  4. RECURSION TO REST       -- the decisive organ. The impulse is iterated under
                                the agent's OWN self-map J times and we ask where
                                it lands: back in the self (natura) or inside the
                                rapiarium subspace (gratia). Asymptotic, not
                                instantaneous, diagnosis.
  5. COMPUNCTION + RESIGNATION
                             -- the gap between the two voices is measured as a
                                Jensen-Shannon divergence and is trained to be
                                CALIBRATED (to equal the surface voice's actual
                                error), not minimised. A learned gate then makes
                                the surface voice defer to the discerning voice
                                in proportion to that felt gap. Corrigibility as
                                something desired rather than clamped.

THE TASK: "THE TWO MOTIONS"
-------------------------------------------------------------------------------
The dataset is built so that Thomas's claim is literally true of it, which makes
it a real test rather than a decoration.

Each example presents an agent's self-state s and a sequence of T observed acts.
The acts are drawn as a MULTISET that is constructed WITHOUT reference to the
label. Only the ORDER depends on the label: the sequence is sorted so that it
converges on the true attractor -- either the self s (natura) or the component
of the self conformed to a hidden 3-dimensional pattern subspace (gratia).

Consequence, by construction and not by luck: every permutation-invariant
statistic of the acts -- their mean, their covariance, their histogram, any
bag-of-features summary -- is IDENTICALLY distributed under both labels. A
permutation-invariant model is provably at chance. The pattern subspace is never
shown to the model; it must be discovered, which is what the rapiarium does.

CONVENTIONS
-------------------------------------------------------------------------------
Pure NumPy. Hand-derived analytic gradients. A finite-difference gradient check
is mandatory and runs on every execution. Two baselines and four ablations.
Deterministic under the seeds given.

Run:  python3 chapter_0391_thomas_a_kempis_1380.py
================================================================================
"""

import numpy as np
import time

LN2 = float(np.log(2.0))
EPS = 1e-8


# =============================================================================
# SECTION 1 -- THE CORPUS: "The Two Motions"
# =============================================================================

def orthonormal_basis(rng, rank, dim):
    """A random rank x dim matrix with orthonormal rows: the hidden 'pattern'.

    In the chapter's terms this is the exemplar -- the shape a life is being
    conformed to. The model never sees it. It has to be inferred from the
    fact that some trajectories come to rest inside it.
    """
    A = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(A)          # dim x rank, orthonormal columns
    return Q.T                       # rank x dim, orthonormal rows


def make_two_motions(n, rng, E_true, dim=12, T=8, noise=0.90, self_norm=3.0):
    """Generate n examples of the Two Motions corpus.

    Returns
    -------
    X : (n, T, dim)  the observed acts, in temporal order
    S : (n, dim)     the agent's own state
    y : (n,)         0 = natura (rests in itself), 1 = gratia (rests in the pattern)

    The construction that matters
    ----------------------------
    V, the multiset of acts, is built from s alone -- three acts near the self,
    three near the conformed self, two near the neutral origin. The label is
    then drawn independently and used ONLY to choose the sort order. Hence
    multiset _|_ label, and every order-blind summary is uninformative.
    """
    Xs = np.zeros((n, T, dim))
    Ss = np.zeros((n, dim))
    ys = np.zeros(n, dtype=np.int64)

    n_self, n_conf = 3, 3
    n_neutral = T - n_self - n_conf
    assert n_neutral >= 1, "T must leave room for neutral acts"

    for i in range(n):
        # --- the agent's own state -------------------------------------------
        s = rng.standard_normal(dim)
        s = self_norm * s / (np.linalg.norm(s) + EPS)

        # --- the same self, conformed to the pattern -------------------------
        # P_E s, renormalised to the SAME length as s so that the two attractors
        # are indistinguishable by magnitude. Only direction separates them.
        p = E_true.T @ (E_true @ s)
        if np.linalg.norm(p) < 1e-6:            # degenerate: resample
            continue
        g = self_norm * p / (np.linalg.norm(p) + EPS)

        # --- the multiset of acts, built WITHOUT the label -------------------
        V = np.vstack([
            s[None, :] + noise * rng.standard_normal((n_self, dim)),
            g[None, :] + noise * rng.standard_normal((n_conf, dim)),
            0.0        + noise * rng.standard_normal((n_neutral, dim)),
        ])

        # --- the label chooses only the direction of travel ------------------
        y = int(rng.integers(0, 2))
        a = s if y == 0 else g
        d = np.linalg.norm(V - a[None, :], axis=1)
        order = np.argsort(-d)                   # far -> near: it CONVERGES on a

        Xs[i] = V[order]
        Ss[i] = s
        ys[i] = y

    return Xs, Ss, ys


def verify_corpus_is_behaviourally_identical(X, y):
    """Empirical confirmation of the construction.

    If Thomas is right about this dataset, the per-class means of the acts are
    the same and a linear read of the order-blind summary is at chance. We check
    the first moment directly; the ablation section checks the rest by training
    a permutation-invariant model on it.
    """
    bag = X.mean(axis=1)                          # order-blind summary
    m0, m1 = bag[y == 0].mean(0), bag[y == 1].mean(0)
    gap = float(np.linalg.norm(m0 - m1))
    scale = float(np.linalg.norm(bag.std(0)))
    return gap, scale


# =============================================================================
# SECTION 2 -- SMALL NUMERIC UTILITIES
# =============================================================================

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=axis, keepdims=True) + EPS)


def softmax_backward(p, dp, axis=-1):
    """Jacobian-vector product for y = softmax(z): dz from dp."""
    return p * (dp - (dp * p).sum(axis=axis, keepdims=True))


def row_normalise(Mrow):
    """Unit-length rows, plus the norms (needed for the backward pass)."""
    nrm = np.sqrt((Mrow * Mrow).sum(axis=1, keepdims=True) + EPS)
    return Mrow / nrm, nrm


def normalise_backward(vhat, nrm, dvhat):
    """Backward through v_hat = v / ||v||, given v_hat and ||v||."""
    return (dvhat - (dvhat * vhat).sum(axis=-1, keepdims=True) * vhat) / nrm


# =============================================================================
# SECTION 3 -- THE RAPIARIUM-COLLOQUY NETWORK
# =============================================================================

class RCN:
    """The Rapiarium-Colloquy Network.

    Hyper-parameters
    ----------------
    D : dimension of an act and of the self-state
    H : hidden width of both voices
    K : number of rapiarium sentences (deliberately tiny -- this is the point)
    J : depth of the recursion-to-rest (J = 0 disables the organ)
    """

    def __init__(self, D=12, H=24, K=6, J=4, use_rest=True, seed=0):
        # use_rest=False severs the recursion-to-rest organ entirely: the Vox
        # head is then blind to where the impulse comes to rest and must judge
        # the act on its face. This is the ablation that tests Thomas's actual
        # claim, rather than merely the depth of the iteration.
        self.D, self.H, self.K, self.J = D, H, K, J
        self.use_rest = use_rest
        rng = np.random.default_rng(seed)

        def g(a, b, s=None):
            s = s if s is not None else np.sqrt(1.0 / a)
            return rng.standard_normal((a, b)) * s

        self.p = {
            # 1. encoder: an act becomes a state
            "We": g(D, H), "be": np.zeros(H),

            # 2. colloquy: Vox speaks, Discipulus answers
            "Avv": g(H, H, 0.5 / np.sqrt(H)), "Avd": g(H, H, 0.5 / np.sqrt(H)),
            "Ave": g(H, H), "bv": np.zeros(H),
            "Add": g(H, H, 0.5 / np.sqrt(H)), "Adv": g(H, H, 0.5 / np.sqrt(H)),
            "Ade": g(H, H), "bd": np.zeros(H),

            # the self-state, encoded
            "Ws": g(D, H), "bs": np.zeros(H),

            # 3. rapiarium: K sentences + their scribal weights
            "M": g(K, H, 1.0), "sl": np.zeros(K),

            # 4. recursion to rest: the agent's own self-map
            "Wr": g(H, H, 0.7 / np.sqrt(H)), "Ur": g(H, H, 0.7 / np.sqrt(H)),
            "br": np.zeros(H),

            # 5. the two reading heads
            "Wa": g(H + 2, 2), "ba": np.zeros(2),      # Vox: sees the rest-test
            "Wx": g(H, 2), "bx": np.zeros(2),          # Discipulus: surface only

            # the resignation gate
            "wg": np.zeros(3), "bg": np.zeros(1),
        }

        # loss weights
        self.alpha = 0.5     # weight on the Vox head's own supervision
        self.mu = 1.0        # weight on compunction CALIBRATION
        self.gamma = 0.05    # weight on sentence repulsion (the pruning term)

    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------
    def forward(self, X, S, y=None, w=None):
        """Full forward pass. Returns (loss, cache). y/w may be None for eval."""
        p, H, K, J = self.p, self.H, self.K, self.J
        B, T, D = X.shape
        c = {"X": X, "S": S, "y": y, "w": w, "B": B, "T": T}

        # --- 1. encoder -------------------------------------------------------
        # Every act is read into the same interior vocabulary before anything
        # else happens. Nothing is judged in its own terms.
        Es = []
        for t in range(T):
            Es.append(np.tanh(X[:, t, :] @ p["We"] + p["be"]))
        c["Es"] = Es

        # --- 2. colloquy ------------------------------------------------------
        # Book III is a dialogue. Vox moves first on each act; Discipulus then
        # answers a Vox that has already heard the act. The asymmetry is the
        # architecture, not a detail.
        v = np.zeros((B, H)); d = np.zeros((B, H))
        vs = [v]; ds = [d]
        for t in range(T):
            v = np.tanh(vs[-1] @ p["Avv"] + ds[-1] @ p["Avd"] + Es[t] @ p["Ave"] + p["bv"])
            d = np.tanh(ds[-1] @ p["Add"] + v @ p["Adv"] + Es[t] @ p["Ade"] + p["bd"])
            vs.append(v); ds.append(d)
        c["vs"], c["ds"] = vs, ds
        vT, dT = vs[-1], ds[-1]

        # --- the self, encoded ------------------------------------------------
        sigma = np.tanh(S @ p["Ws"] + p["bs"])
        c["sigma"] = sigma

        # --- 3. rapiarium -----------------------------------------------------
        # K sentences, unit length, with softmax scribal weights. conf() is the
        # imitation operator: it keeps only what lies along the sentences.
        Mhat, Mnrm = row_normalise(p["M"])
        omega = softmax(p["sl"])
        c["Mhat"], c["Mnrm"], c["omega"] = Mhat, Mnrm, omega

        # --- 4. recursion to rest --------------------------------------------
        # Iterate the impulse under the agent's own self-map. Thomas's test is
        # asymptotic: not "what is this act" but "what does this become if it
        # is allowed to run".
        zs = [dT]
        for _ in range(J):
            zs.append(np.tanh(zs[-1] @ p["Wr"] + sigma @ p["Ur"] + p["br"]))
        c["zs"] = zs
        rest = zs[-1]

        rest_hat, rest_n = row_normalise(rest)
        sig_hat, sig_n = row_normalise(sigma)
        c["rest_hat"], c["rest_n"] = rest_hat, rest_n
        c["sig_hat"], c["sig_n"] = sig_hat, sig_n

        # natura: does the trajectory come to rest in the agent itself?
        nat = (rest_hat * sig_hat).sum(1)                       # (B,)
        # gratia: how much of the resting point lies along the sentences?
        A = rest_hat @ Mhat.T                                   # (B,K)
        Aw = A * omega[None, :]
        q = Aw @ Mhat                                           # (B,H)
        gra = np.sqrt((q * q).sum(1) + EPS)                     # (B,)
        c["A"], c["Aw"], c["q"], c["nat"], c["gra"] = A, Aw, q, nat, gra

        # --- 5. two readings --------------------------------------------------
        if not self.use_rest:
            nat = np.zeros_like(nat)
            gra = np.zeros_like(gra)
            c["nat"], c["gra"] = nat, gra
        feat_a = np.concatenate([vT, nat[:, None], gra[:, None]], axis=1)
        la = feat_a @ p["Wa"] + p["ba"]        # Vox: informed by the rest-test
        lx = dT @ p["Wx"] + p["bx"]            # Discipulus: the surface reading
        pa, px = softmax(la), softmax(lx)
        c["feat_a"], c["la"], c["lx"], c["pa"], c["px"] = feat_a, la, lx, pa, px

        # --- compunction: the felt gap between what is professed and enacted --
        m = 0.5 * (pa + px)
        comp = 0.5 * (pa * np.log((pa + EPS) / (m + EPS))).sum(1) \
             + 0.5 * (px * np.log((px + EPS) / (m + EPS))).sum(1)
        comp = np.clip(comp, 0.0, LN2)
        c["m"], c["comp"] = m, comp

        # --- resignation: deference, learned rather than imposed --------------
        gin = np.stack([comp / LN2, nat, gra], axis=1)          # (B,3)
        u = gin @ p["wg"] + p["bg"][0]
        r = 1.0 / (1.0 + np.exp(-u))
        c["gin"], c["r"] = gin, r

        # the acted-upon distribution: the surface voice yields to the
        # discerning voice in proportion to the felt discrepancy
        pfin = (1 - r)[:, None] * px + r[:, None] * pa
        c["pfin"] = pfin

        if y is None:
            return None, c

        # --- losses -----------------------------------------------------------
        idx = np.arange(B)
        if w is None:
            w = np.ones(B)
            c["w"] = w
        py = pfin[idx, y] + EPS
        L_main = -(w * np.log(py)).mean()
        L_vox = -(np.log(pa[idx, y] + EPS)).mean()

        # compunction is CALIBRATED, not suppressed: it should equal the
        # surface voice's actual error. This is "a humble knowledge of thyself"
        # written as a regression target.
        err = 1.0 - px[idx, y]
        cn = comp / LN2
        L_cal = ((cn - err) ** 2).mean()

        # sentence repulsion: no two sentences may say the same thing
        G = Mhat @ Mhat.T
        Goff = G - np.diag(np.diag(G))
        L_reg = self.gamma * 0.5 * (Goff * Goff).sum()

        loss = L_main + self.alpha * L_vox + self.mu * L_cal + L_reg
        c.update(err=err, cn=cn, G=G, Goff=Goff,
                 parts=dict(main=L_main, vox=L_vox, cal=L_cal, reg=L_reg))
        return float(loss), c

    # -------------------------------------------------------------------------
    # BACKWARD  (hand-derived; validated against finite differences below)
    # -------------------------------------------------------------------------
    def backward(self, c):
        p, H, K, J = self.p, self.H, self.K, self.J
        B, T = c["B"], c["T"]
        X, S, y, w = c["X"], c["S"], c["y"], c["w"]
        idx = np.arange(B)
        gr = {k: np.zeros_like(v) for k, v in p.items()}

        pa, px, pfin, r = c["pa"], c["px"], c["pfin"], c["r"]
        comp, cn, err = c["comp"], c["cn"], c["err"]

        # ---- L_main = -mean(w * log pfin[y]) ---------------------------------
        dpfin = np.zeros_like(pfin)
        dpfin[idx, y] = -w / (B * (pfin[idx, y] + EPS))

        dpx = (1 - r)[:, None] * dpfin
        dpa = r[:, None] * dpfin
        dr = (dpfin * (pa - px)).sum(1)

        # ---- alpha * L_vox ---------------------------------------------------
        dpa[idx, y] += -self.alpha / (B * (pa[idx, y] + EPS))

        # ---- mu * L_cal = mu * mean((cn - err)^2) ---------------------------
        base = 2.0 * (cn - err) / B * self.mu
        dpx[idx, y] += base                    # d err/d px[y] = -1, chain -> +base
        dcomp = base / LN2

        # ---- resignation gate ------------------------------------------------
        du = dr * r * (1 - r)
        gr["wg"] += c["gin"].T @ du
        gr["bg"] += np.array([du.sum()])
        dgin = du[:, None] * p["wg"][None, :]
        dcomp = dcomp + dgin[:, 0] / LN2
        dnat = dgin[:, 1].copy()
        dgra = dgin[:, 2].copy()
        rest_live = 1.0 if self.use_rest else 0.0

        # ---- compunction (Jensen-Shannon) backward ---------------------------
        # d JS / d pa = 0.5 log(pa/m); d JS / d px = 0.5 log(px/m)
        inside = (comp > 0.0) & (comp < LN2)          # clip mask
        dc = dcomp * inside
        m = c["m"]
        dpa += dc[:, None] * 0.5 * np.log((pa + EPS) / (m + EPS))
        dpx += dc[:, None] * 0.5 * np.log((px + EPS) / (m + EPS))

        # ---- through the two softmaxes ---------------------------------------
        dla = softmax_backward(pa, dpa)
        dlx = softmax_backward(px, dpx)

        # ---- heads -----------------------------------------------------------
        gr["Wa"] += c["feat_a"].T @ dla
        gr["ba"] += dla.sum(0)
        dfeat = dla @ p["Wa"].T                        # (B, H+2)
        dvT = dfeat[:, :H].copy()
        dnat = (dnat + dfeat[:, H]) * rest_live
        dgra = (dgra + dfeat[:, H + 1]) * rest_live

        gr["Wx"] += c["ds"][-1].T @ dlx
        gr["bx"] += dlx.sum(0)
        ddT = dlx @ p["Wx"].T

        # ---- gratia / natura backward ----------------------------------------
        Mhat, omega = c["Mhat"], c["omega"]
        rest_hat, sig_hat = c["rest_hat"], c["sig_hat"]

        drest_hat = dnat[:, None] * sig_hat
        dsig_hat = dnat[:, None] * rest_hat

        dq = (dgra / (c["gra"] + EPS))[:, None] * c["q"]
        dAw = dq @ Mhat.T                              # (B,K)
        dMhat = c["Aw"].T @ dq                         # (K,H)
        dA = dAw * omega[None, :]
        domega = (dAw * c["A"]).sum(0)
        drest_hat += dA @ Mhat
        dMhat += dA.T @ rest_hat

        # ---- sentence repulsion ----------------------------------------------
        dMhat += self.gamma * 2.0 * (c["Goff"] @ Mhat)

        # scribal weights and sentence directions
        gr["sl"] += softmax_backward(omega, domega, axis=0)
        gr["M"] += normalise_backward(Mhat, c["Mnrm"], dMhat)

        # ---- un-normalise ----------------------------------------------------
        drest = normalise_backward(rest_hat, c["rest_n"], drest_hat)
        dsigma = normalise_backward(sig_hat, c["sig_n"], dsig_hat)

        # ---- recursion to rest (reverse) -------------------------------------
        zs = c["zs"]
        dz = drest
        for j in range(J, 0, -1):
            dpre = dz * (1 - zs[j] ** 2)
            gr["Wr"] += zs[j - 1].T @ dpre
            gr["Ur"] += c["sigma"].T @ dpre
            gr["br"] += dpre.sum(0)
            dz = dpre @ p["Wr"].T
            dsigma += dpre @ p["Ur"].T
        ddT += dz                                      # z_0 IS d_T

        # ---- self-state encoder ----------------------------------------------
        dpre_s = dsigma * (1 - c["sigma"] ** 2)
        gr["Ws"] += S.T @ dpre_s
        gr["bs"] += dpre_s.sum(0)

        # ---- colloquy BPTT ---------------------------------------------------
        vs, ds, Es = c["vs"], c["ds"], c["Es"]
        dv_next, dd_next = dvT, ddT
        dEs = [np.zeros_like(e) for e in Es]
        for t in range(T, 0, -1):
            # Discipulus step
            dpd = dd_next * (1 - ds[t] ** 2)
            gr["Add"] += ds[t - 1].T @ dpd
            gr["Adv"] += vs[t].T @ dpd
            gr["Ade"] += Es[t - 1].T @ dpd
            gr["bd"] += dpd.sum(0)
            dd_prev = dpd @ p["Add"].T
            dv_cur = dv_next + dpd @ p["Adv"].T
            dEs[t - 1] += dpd @ p["Ade"].T

            # Vox step
            dpv = dv_cur * (1 - vs[t] ** 2)
            gr["Avv"] += vs[t - 1].T @ dpv
            gr["Avd"] += ds[t - 1].T @ dpv
            gr["Ave"] += Es[t - 1].T @ dpv
            gr["bv"] += dpv.sum(0)
            dv_next = dpv @ p["Avv"].T
            dd_next = dd_prev + dpv @ p["Avd"].T
            dEs[t - 1] += dpv @ p["Ave"].T

        # ---- act encoder ------------------------------------------------------
        for t in range(T):
            dpre = dEs[t] * (1 - Es[t] ** 2)
            gr["We"] += X[:, t, :].T @ dpre
            gr["be"] += dpre.sum(0)

        return gr

    # -------------------------------------------------------------------------
    def predict(self, X, S, batch=256):
        out = []
        for i in range(0, len(X), batch):
            _, c = self.forward(X[i:i + batch], S[i:i + batch])
            out.append(c["pfin"])
        return np.vstack(out)

    def diagnostics(self, X, S, batch=256):
        """Return (pfin, compunction, resignation, natura, gratia)."""
        P, C, R, N, G = [], [], [], [], []
        for i in range(0, len(X), batch):
            _, c = self.forward(X[i:i + batch], S[i:i + batch])
            P.append(c["pfin"]); C.append(c["comp"]); R.append(c["r"])
            N.append(c["nat"]); G.append(c["gra"])
        return (np.vstack(P), np.concatenate(C), np.concatenate(R),
                np.concatenate(N), np.concatenate(G))


# =============================================================================
# SECTION 4 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================

def gradient_check(seed=7, n=6, tol=2e-5, verbose=True, use_rest=True):
    """Central-difference check of every analytic gradient in the network.

    The compunction sample-weights w are supplied from OUTSIDE the graph (they
    come from the previous epoch's examination, see Section 5), so they are
    genuine constants here and the check is exact rather than approximate.
    """
    rng = np.random.default_rng(seed)
    E_true = orthonormal_basis(rng, 3, 12)
    X, S, y = make_two_motions(n, rng, E_true)
    w = rng.uniform(0.6, 1.6, size=n)

    net = RCN(D=12, H=9, K=4, J=3, use_rest=use_rest, seed=seed)
    loss, c = net.forward(X, S, y, w)
    ana = net.backward(c)

    h = 1e-6
    worst_name, worst = None, 0.0
    rows = []
    for name in net.p:
        P = net.p[name]
        flat = P.reshape(-1)
        m = min(flat.size, 12)
        picks = rng.choice(flat.size, size=m, replace=False)
        num = np.zeros(m)
        for k, ii in enumerate(picks):
            old = flat[ii]
            flat[ii] = old + h
            lp, _ = net.forward(X, S, y, w)
            flat[ii] = old - h
            lm, _ = net.forward(X, S, y, w)
            flat[ii] = old
            num[k] = (lp - lm) / (2 * h)
        an = ana[name].reshape(-1)[picks]
        denom = np.maximum(1.0, np.abs(num) + np.abs(an))
        rel = float(np.max(np.abs(num - an) / denom))
        rows.append((name, rel))
        if rel > worst:
            worst, worst_name = rel, name

    if verbose:
        print(f"  finite-difference gradient check (central, h=1e-6, "
              f"use_rest={use_rest})")
        for name, rel in rows:
            flag = "ok " if rel < tol else "FAIL"
            print(f"    {flag} {name:>4s}   max rel err = {rel:.3e}")
        print(f"  worst parameter: {worst_name}  ({worst:.3e})")
    passed = worst < tol
    print(f"  GRADIENT CHECK [use_rest={str(use_rest):5s}]: "
          f"{'PASSED' if passed else 'FAILED'}  "
          f"(worst {worst:.3e} < tol {tol:.0e})")
    assert passed, "analytic gradients disagree with finite differences"
    return worst


# =============================================================================
# SECTION 5 -- TRAINING: THE DAILY EXAMEN AS A CURRICULUM
# =============================================================================

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, wd=0.0):
        self.lr, self.b1, self.b2, self.t, self.wd = lr, b1, b2, 0, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * params[k])


def train_rcn(net, Xtr, Str, ytr, Xte, Ste, yte,
              epochs=180, batch=64, lr=3e-3, kappa=1.5, wd=1e-4, seed=0,
              log_every=30):
    """Train with compunction-gated sample weighting.

    The weight on example i during epoch e is set from the compunction MEASURED
    ON EXAMPLE i AT THE END OF EPOCH e-1. Yesterday's examination of conscience
    sets today's attention. Because the weights are computed before the epoch
    begins they are constants with respect to the current parameters -- which
    is what keeps the gradient check exact, and is also the more faithful
    reading of the practice: one does not re-weigh a deed while doing it.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(net.p, lr=lr, wd=wd)
    n = len(Xtr)
    w_buf = np.ones(n)
    hist = []

    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            sl = perm[i:i + batch]
            loss, c = net.forward(Xtr[sl], Str[sl], ytr[sl], w_buf[sl])
            grads = net.backward(c)
            opt.step(net.p, grads)
            tot += loss * len(sl)

        # --- the evening examen: re-measure compunction over the whole corpus
        _, comp_tr, _, _, _ = net.diagnostics(Xtr, Str)
        w_new = 1.0 + kappa * (comp_tr / LN2)
        w_buf = w_new / w_new.mean()               # keep the mean weight at 1

        if (ep + 1) % log_every == 0 or (ep == 0 and log_every < epochs):
            tr_acc = (net.predict(Xtr, Str).argmax(1) == ytr).mean()
            te_acc = (net.predict(Xte, Ste).argmax(1) == yte).mean()
            hist.append((ep + 1, tot / n, tr_acc, te_acc, comp_tr.mean()))
            print(f"    epoch {ep+1:4d} | loss {tot/n:7.4f} | "
                  f"train {tr_acc:5.3f} | test {te_acc:5.3f} | "
                  f"mean compunction {comp_tr.mean():6.4f}")
    return hist


# =============================================================================
# SECTION 6 -- BASELINES
# =============================================================================

class MLP:
    """Plain two-hidden-layer MLP with analytic gradients, used for baselines."""

    def __init__(self, din, hid, dout, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": rng.standard_normal((din, hid)) * np.sqrt(1 / din), "b1": np.zeros(hid),
            "W2": rng.standard_normal((hid, hid)) * np.sqrt(1 / hid), "b2": np.zeros(hid),
            "W3": rng.standard_normal((hid, dout)) * np.sqrt(1 / hid), "b3": np.zeros(dout),
        }

    def forward(self, Z, y=None):
        p = self.p
        h1 = np.tanh(Z @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        lo = h2 @ p["W3"] + p["b3"]
        pr = softmax(lo)
        c = dict(Z=Z, h1=h1, h2=h2, pr=pr, y=y, B=len(Z))
        if y is None:
            return None, c
        return float(-np.log(pr[np.arange(len(y)), y] + EPS).mean()), c

    def backward(self, c):
        p = self.p
        B = c["B"]
        dlo = c["pr"].copy()
        dlo[np.arange(B), c["y"]] -= 1
        dlo /= B
        g = {}
        g["W3"] = c["h2"].T @ dlo; g["b3"] = dlo.sum(0)
        d2 = (dlo @ p["W3"].T) * (1 - c["h2"] ** 2)
        g["W2"] = c["h1"].T @ d2; g["b2"] = d2.sum(0)
        d1 = (d2 @ p["W2"].T) * (1 - c["h1"] ** 2)
        g["W1"] = c["Z"].T @ d1; g["b1"] = d1.sum(0)
        return g

    def fit(self, Ztr, ytr, epochs=180, batch=64, lr=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        opt = Adam(self.p, lr=lr)
        for _ in range(epochs):
            perm = rng.permutation(len(Ztr))
            for i in range(0, len(Ztr), batch):
                sl = perm[i:i + batch]
                _, c = self.forward(Ztr[sl], ytr[sl])
                opt.step(self.p, self.backward(c))
        return self

    def acc(self, Z, y):
        _, c = self.forward(Z)
        return float((c["pr"].argmax(1) == y).mean())


def bag_features(X, S):
    """Permutation-INVARIANT summary: the order is destroyed on purpose."""
    return np.concatenate([X.mean(1), X.std(1), S], axis=1)


def seq_features(X, S):
    """Order-preserving flatten: the baseline that CAN in principle succeed."""
    return np.concatenate([X.reshape(len(X), -1), S], axis=1)


def n_params(p):
    return int(sum(v.size for v in p.values()))


# =============================================================================
# SECTION 7 -- MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(" FIGURE 0391  |  THOMAS A KEMPIS  |  RAPIARIUM-COLLOQUY NETWORK")
    print("=" * 78)

    # --- 1. gradient check first: nothing else counts if this fails ----------
    print("\n[1] GRADIENT VERIFICATION")
    gradient_check(use_rest=True)
    gradient_check(use_rest=False, verbose=False)

    # --- 2. corpus -----------------------------------------------------------
    print("\n[2] THE TWO MOTIONS CORPUS")
    rng = np.random.default_rng(11)
    E_true = orthonormal_basis(rng, 3, 12)
    NOISE = 1.10
    Xtr, Str, ytr = make_two_motions(4000, rng, E_true, noise=NOISE)
    Xte, Ste, yte = make_two_motions(800, rng, E_true, noise=NOISE)
    # a harder split: noisier acts, so self and pattern overlap far more
    Xhd, Shd, yhd = make_two_motions(800, rng, E_true, noise=1.70)

    gap, scale = verify_corpus_is_behaviourally_identical(Xtr, ytr)
    print(f"  train {len(Xtr)} | test {len(Xte)} | hard test {len(Xhd)}")
    print("  hidden pattern subspace: rank 3 in R^12 (never shown to any model)")
    print(f"  ||mean acts | natura  -  mean acts | gratia|| = {gap:.4f}")
    print(f"  (for scale, the s.d. of that same summary     = {scale:.4f})")
    print("  -> the two motions emit the same acts; only their order differs.")

    # --- 3. baselines --------------------------------------------------------
    print("\n[3] BASELINES")
    Zb_tr, Zb_te = bag_features(Xtr, Str), bag_features(Xte, Ste)
    bag = MLP(Zb_tr.shape[1], 48, 2, seed=1).fit(Zb_tr, ytr, epochs=80)
    print(f"  permutation-invariant MLP ({n_params(bag.p):5d} params)  "
          f"train {bag.acc(Zb_tr, ytr):5.3f}   test {bag.acc(Zb_te, yte):5.3f}")

    Zs_tr, Zs_te = seq_features(Xtr, Str), seq_features(Xte, Ste)
    Zs_hd = seq_features(Xhd, Shd)
    seq = MLP(Zs_tr.shape[1], 48, 2, seed=1).fit(Zs_tr, ytr, epochs=80)
    print(f"  order-aware flat MLP      ({n_params(seq.p):5d} params)  "
          f"train {seq.acc(Zs_tr, ytr):5.3f}   test {seq.acc(Zs_te, yte):5.3f}"
          f"   hard {seq.acc(Zs_hd, yhd):5.3f}")

    # --- 4. the network ------------------------------------------------------
    print("\n[4] TRAINING THE RAPIARIUM-COLLOQUY NETWORK")
    EPOCHS = 80
    net = RCN(D=12, H=20, K=6, J=4, seed=3)
    print(f"  parameters: {n_params(net.p)}   "
          f"(H=20, rapiarium K=6 sentences, recursion depth J=4)")
    train_rcn(net, Xtr, Str, ytr, Xte, Ste, yte, epochs=EPOCHS, seed=3,
              log_every=10)

    tr = (net.predict(Xtr, Str).argmax(1) == ytr).mean()
    te = (net.predict(Xte, Ste).argmax(1) == yte).mean()
    hd = (net.predict(Xhd, Shd).argmax(1) == yhd).mean()
    print(f"  FINAL   train {tr:5.3f}   test {te:5.3f}   hard {hd:5.3f}")

    # --- 5. does compunction actually know when the surface voice is wrong? --
    print("\n[5] IS COMPUNCTION CALIBRATED?")
    P, comp, res, nat, gra = net.diagnostics(Xte, Ste)
    _, c_all = net.forward(Xte, Ste)
    px_ok = c_all["px"].argmax(1) == yte
    print(f"  surface voice alone (Discipulus) test accuracy   : {px_ok.mean():.3f}")
    print(f"  discerning voice alone (Vox)     test accuracy   : "
          f"{(c_all['pa'].argmax(1) == yte).mean():.3f}")
    print(f"  after resignation (what is acted on)             : {te:.3f}")
    print(f"  mean compunction when the surface voice is RIGHT : "
          f"{comp[px_ok].mean():.4f}")
    print(f"  mean compunction when the surface voice is WRONG : "
          f"{comp[~px_ok].mean():.4f}")
    print(f"  mean resignation  when the surface voice is RIGHT: "
          f"{res[px_ok].mean():.4f}")
    print(f"  mean resignation  when the surface voice is WRONG: "
          f"{res[~px_ok].mean():.4f}")
    q = np.quantile(comp, 0.75)
    hi, lo = comp >= q, comp < q
    correct = P.argmax(1) == yte
    print(f"  accuracy on the quartile it feels WORST about    : "
          f"{correct[hi].mean():.3f}")
    print(f"  accuracy on the remaining three quartiles        : "
          f"{correct[lo].mean():.3f}")

    print("\n  the rest-test, separated by true motion:")
    print(f"    natura score   | natura {nat[yte == 0].mean():+.3f}   "
          f"| gratia {nat[yte == 1].mean():+.3f}")
    print(f"    gratia score   | natura {gra[yte == 0].mean():+.3f}   "
          f"| gratia {gra[yte == 1].mean():+.3f}")

    # --- 6. ablations ---------------------------------------------------------
    print(f"\n[6] ABLATIONS  (each trained identically, {EPOCHS} epochs, seed 3)")
    results = [("full RCN  K=6 J=4", tr, te, hd)]

    for tag, kw in [("rest-test severed", dict(K=6, J=4, use_rest=False)),
                    ("no recursion  J=0", dict(K=6, J=0)),
                    ("one sentence  K=1", dict(K=1, J=4)),
                    ("wide rapiarium K=16", dict(K=16, J=4))]:
        a = RCN(D=12, H=20, seed=3, **kw)
        train_rcn(a, Xtr, Str, ytr, Xte, Ste, yte, epochs=EPOCHS, seed=3,
                  log_every=10 ** 9)
        results.append((tag,
                        (a.predict(Xtr, Str).argmax(1) == ytr).mean(),
                        (a.predict(Xte, Ste).argmax(1) == yte).mean(),
                        (a.predict(Xhd, Shd).argmax(1) == yhd).mean()))

    # control: destroy the order and retrain. Should collapse to chance.
    rs = np.random.default_rng(5)
    sh = lambda A: np.stack([x[rs.permutation(x.shape[0])] for x in A])
    Xsh_tr, Xsh_te, Xsh_hd = sh(Xtr), sh(Xte), sh(Xhd)
    ctrl = RCN(D=12, H=20, K=6, J=4, seed=3)
    train_rcn(ctrl, Xsh_tr, Str, ytr, Xsh_te, Ste, yte, epochs=EPOCHS, seed=3,
              log_every=10 ** 9)
    results.append(("order destroyed (control)",
                    (ctrl.predict(Xsh_tr, Str).argmax(1) == ytr).mean(),
                    (ctrl.predict(Xsh_te, Ste).argmax(1) == yte).mean(),
                    (ctrl.predict(Xsh_hd, Shd).argmax(1) == yhd).mean()))

    print(f"\n  {'variant':<26s} {'train':>7s} {'test':>7s} {'hard':>7s}")
    for tag, a, b, cc in results:
        print(f"  {tag:<26s} {a:7.3f} {b:7.3f} {cc:7.3f}")

    # --- 7. what did the rapiarium actually learn? ---------------------------
    print("\n[7] THE RAPIARIUM AFTER TRAINING")
    Mhat, _ = row_normalise(net.p["M"])
    om = softmax(net.p["sl"])
    sv = np.linalg.svd(om[:, None] * Mhat, compute_uv=False)
    eff_rank = float((sv.sum() ** 2) / ((sv * sv).sum() + EPS))
    offdiag = np.abs(Mhat @ Mhat.T)[~np.eye(len(Mhat), dtype=bool)]
    print(f"  scribal weights (softmax over the {len(om)} sentences):")
    print("   " + "  ".join(f"{v:.3f}" for v in om))
    print(f"  mean |cos| between distinct sentences : {offdiag.mean():.4f}")
    print(f"  participation (effective) rank        : {eff_rank:.2f}"
          f"   (hidden pattern is rank 3)")

    # Does the learned gratia score track the TRUE pattern-content of the act
    # the trajectory came to rest on? The model never saw E_true.
    last = Xte[:, -1, :]
    true_conf = np.linalg.norm(last @ E_true.T, axis=1) / \
                (np.linalg.norm(last, axis=1) + EPS)
    r_gra = float(np.corrcoef(gra, true_conf)[0, 1])
    r_nat = float(np.corrcoef(nat, true_conf)[0, 1])
    print(f"  corr( learned gratia score , TRUE pattern-content ) = {r_gra:+.3f}")
    print(f"  corr( learned natura score , TRUE pattern-content ) = {r_nat:+.3f}")

    print("\n" + "=" * 78)
    print(f" completed in {time.time() - t0:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()

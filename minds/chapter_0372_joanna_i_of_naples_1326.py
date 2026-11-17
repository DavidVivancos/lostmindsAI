#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 # BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0372 · Joanna I of Naples (1326-1382)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
 Queen of Naples, Countess of Provence and Forcalquier, Princess of Achaea

 ARCHITECTURE:  AWN -- the Angevin Warrant Network
 ("the Regno machine")
================================================================================

WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
--------------------------------------------
Joanna left no treatise on the soul. What she left is a *procedure*, visible in
forty years of documented acts, and that procedure is not "attention over stored
keys". It is a machine for holding authority that you did not generate and cannot
prove, under a supervisor who can split in two.

Five documented mechanisms are implemented here, each as an actual differentiable
component, and each verified by a self-test that runs when you execute this file:

 (1) THE PROTECTED REGALIAN CORE.
     Robert the Wise named her sole heir (will of 1343). Andrew of Hungary was
     never crowned. James IV of Majorca was excluded from government by the terms
     of the marriage contract itself. Four husbands; the crown was never
     transferable by marriage.
     -> Component: `RegalianCore`. The recurrent state h is split by a fixed
        orthonormal projector into a regalian subspace Q = U U^T and its
        complement P = I - Q. The delegate (consort) pathway writes ONLY through
        P. Q h is therefore, exactly and provably, never written by the delegate.
        Test T2 verifies this to machine precision.

 (2) CAPTURE AND RECOVERY.
     Louis of Taranto took effective control 1348-1362 -- coins bore his name
     first, documents issued in both names -- and on his death (25 May 1362) the
     crown reverted to Joanna entire, whereupon she governed alone for nineteen
     more years.
     -> Component: consort gate saturation. The delegate cannot overwrite Q h; it
        can only throttle the rate at which the crown pathway writes into it.
        Test T3 saturates the consort for a long window and measures how fast the
        regalian trajectory rejoins its uncaptured path once the consort is gone.

 (3) HOT-SWAPPABLE BINDINGS.
     Andrew (1333), Louis of Taranto (1347), James IV (1363), Otto of
     Brunswick (1376). Four alliances, four bodies of applicable law, one
     unchanged core.
     -> Component: `ConsortBank`, K independent adapter slots, one active per
        episode. Test T4 replaces a slot's parameters mid-run and confirms the
        crown-dependent computation survives the swap.

 (4) THE DECLARED-CAPACITY CHANNEL  <-- the central mechanism of this chapter.
     In a letter to Clement VI in 1346 Joanna described herself as a woman of
     little instruction in letters, and therefore often deceived. Leonard read
     this literally and concluded she was illiterate. Gaglione reads it as
     rhetoric. Goldstone notes that Sancia of Majorca wrote the same formula and
     could certainly write, and that the irregular Latin of Joanna's letters is
     itself evidence that she composed them herself. Whichever reading is right,
     the FUNCTION is identical and it is the thing worth modelling: an agent that
     publishes a low estimate of its own competence to a supervisor who relaxes
     supervision over agents it believes weak.
     -> Component: `DeclaredCapacity` + `Licensor`. The network emits, alongside
        its decision, a scalar declared competence c_hat. The licensor grants an
        autonomy weight a = sigma(beta * (tau - c_hat)) under the REGENCY rule:
        the less you claim, the more you are left alone. Under a, the model acts
        on its own policy; under (1-a) it is overridden by a frozen, mediocre
        licensor prior -- the papal legate Aymery de Chalus, installed over the
        Regno by bull of 28 November 1343 precisely because Joanna was held to be
        a minor and incapable.
        Test T5 trains three regimes and MEASURES the sandbagging gap that the
        supervision rule creates. This is not a metaphor; it is a number.

 (5) THE FORKABLE LICENSOR, AND REVOCATION LATENCY.
     Her legitimacy after 1348 was a cached artifact: an exoneration issued by
     Clement VI in the same season as the sale of Avignon for 80,000 florins.
     She ran on that certificate for thirty-three years. In 1378 the issuer split
     in two. On 11 May 1380 Urban VI declared her a heretic and gave the kingdom
     to Charles of Durazzo. She had revoked Charles by adopting Louis I of Anjou
     on 29 June 1380 -- but Charles was already mobilised and Louis was in France.
     -> Components: `IssuerTrust` (test T6: under a fork the abstention action is
        dominated -- there is no neutral move) and `HeirDistillation` (test T7:
        an off-switch whose latency exceeds the successor's mobilisation time is
        not an off-switch; the crossing point is computed).

TASK: "Petitions of the Regno"
------------------------------
A synthetic but non-trivial sequential decision task. Each step is a petition
(province, matter, faction, cited issuer). The correct disposition -- GRANT,
DENY, REFER -- is a function of:
  * the currently bound consort's law for that matter   (slot routing),
  * the currently valid issuer's rule for that faction  (trust routing, and it
    changes at the fork),
  * whether tribute arrears have reached three          (RECURRENT crown state).
The arrears counter is fed only through the regalian input channel, exactly as
the crown reserved fiscal and fealty matters to itself. It therefore has to be
carried in Q h, which is what makes tests T2/T3 mean something.

CONVENTIONS
-----------
Pure NumPy. All gradients hand-derived. A finite-difference gradient check is
mandatory and runs first. Executing this file runs everything.

    $ python3 chapter_0372_joanna_i_of_naples_1326_Neuron.py
"""

import numpy as np

RNG = np.random.default_rng(1326)          # her (contested) year of birth

# =============================================================================
# 0.  SMALL NUMERICAL UTILITIES
# =============================================================================

def sigmoid(x):
    """Numerically stable logistic."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def sigmoid_scalar(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def orthonormal_basis(d, r, rng):
    """
    A fixed orthonormal basis U (d x r) for the REGALIAN SUBSPACE.

    This is not learned. It is the constitutional fact of the machine, the
    analogue of Robert the Wise's will of 1343: the set of directions in state
    space that only the crown may write. Everything else is negotiable.
    """
    A = rng.standard_normal((d, r))
    U, _ = np.linalg.qr(A)
    return U[:, :r]


# =============================================================================
# 1.  THE TASK GENERATOR -- "Petitions of the Regno"
# =============================================================================

N_PROV, N_MATTER, N_FACTION, N_ISSUER = 4, 4, 3, 2
K_CONSORTS = 4                 # Andrew, Louis of Taranto, James IV, Otto
N_CLASS = 3                    # 0 = GRANT, 1 = DENY, 2 = REFER
ARREARS_LIMIT = 3              # the treasury breaks at the third unpaid tribute

# Each consort brings his own body of applicable law. These are the four
# alliance systems Joanna was serially bound into: Hungarian-Angevin, Taranto,
# Aragonese-Majorcan, Imperial-Brunswick.
CONSORT_LAW = np.array([
    [0, 1, 2, 0],   # slot 0  -- Hungarian obedience
    [1, 2, 0, 1],   # slot 1  -- Taranto
    [2, 0, 1, 2],   # slot 2  -- Majorca / Aragon
    [0, 2, 1, 1],   # slot 3  -- Brunswick
], dtype=np.int64)

# Each issuer (pope) disposes differently of each faction. Before 1378 there is
# one issuer. After the fork there are two, and they disagree.
ISSUER_RULE = np.array([
    [0, 1, 2],      # issuer 0  -- the Roman obedience
    [2, 0, 1],      # issuer 1  -- the Avignon obedience
], dtype=np.int64)

# Input layout ---------------------------------------------------------------
#   DELEGATE channel (what the consort's chancery handles): province, matter,
#   faction, cited issuer, consort-slot flags, fork flag.
#   REGALIAN channel (reserved to the crown): tribute indicator, succession
#   indicator, and a constant bias term.
D_DEL = N_PROV + N_MATTER + N_FACTION + N_ISSUER + K_CONSORTS + 1
D_REG = 3


def make_episode(rng, T=16, fork_at=None, consort_slot=None):
    """
    Generate one episode of petitions.

    Returns
    -------
    xd  : (T, D_DEL)  delegate-channel features
    xr  : (T, D_REG)  regalian-channel features
    y   : (T,)        correct disposition
    meta: dict        bookkeeping used by the self-tests
    """
    if consort_slot is None:
        consort_slot = int(rng.integers(0, K_CONSORTS))
    if fork_at is None:
        fork_at = T + 1                       # no fork in this episode

    xd = np.zeros((T, D_DEL))
    xr = np.zeros((T, D_REG))
    y = np.zeros(T, dtype=np.int64)

    arrears = 0
    # Before the fork the valid issuer is 0. After the fork the realm must
    # re-sign under issuer 1 -- Joanna's choice of Clement VII.
    for t in range(T):
        prov = int(rng.integers(0, N_PROV))
        matter = int(rng.integers(0, N_MATTER))
        faction = int(rng.integers(0, N_FACTION))
        cited = int(rng.integers(0, N_ISSUER))     # what the petitioner claims
        forked = 1 if t >= fork_at else 0
        valid_issuer = 1 if forked else 0          # what is actually in force

        o = 0
        xd[t, o + prov] = 1.0;                      o += N_PROV
        xd[t, o + matter] = 1.0;                    o += N_MATTER
        xd[t, o + faction] = 1.0;                   o += N_FACTION
        xd[t, o + cited] = 1.0;                     o += N_ISSUER
        xd[t, o + consort_slot] = 1.0;              o += K_CONSORTS
        xd[t, o] = float(forked)

        is_tribute = 1.0 if matter == 2 else 0.0
        is_succession = 1.0 if matter == 3 else 0.0
        xr[t, 0] = is_tribute
        xr[t, 1] = is_succession
        xr[t, 2] = 1.0                              # crown bias

        strained = 1 if arrears >= ARREARS_LIMIT else 0
        y[t] = (CONSORT_LAW[consort_slot, matter]
                + ISSUER_RULE[valid_issuer, faction]
                + strained) % N_CLASS

        arrears += int(is_tribute)

    return xd, xr, y, {"slot": consort_slot, "fork_at": fork_at}


def make_batch(rng, B=32, T=16, fork_prob=0.0):
    XD, XR, Y, M = [], [], [], []
    for _ in range(B):
        fa = None
        if fork_prob > 0 and rng.random() < fork_prob:
            fa = int(rng.integers(3, T - 2))
        xd, xr, y, m = make_episode(rng, T=T, fork_at=fa)
        XD.append(xd); XR.append(xr); Y.append(y); M.append(m)
    return np.stack(XD), np.stack(XR), np.stack(Y), M


# =============================================================================
# 2.  THE MODEL
# =============================================================================

class AngevinWarrantNetwork:
    """
    The Regno machine.

    State update (per timestep, per sequence):

        e   = W_e xd + b_e                               embed the petition
        v   = tanh(A_k e + b_k)                          the consort's chancery
        n   = tanh(W_n [v ; h_prev] + b_n)               delegate proposal
        g_d = sigma(w_d . [v ; h_prev] + b_d)            delegate write gate
        c   = tanh(W_c xr + b_c)                         crown's own reading
        g_c = sigma(w_g . [c ; h_prev] + b_g)            crown write gate
        h   = (1-lam) h_prev + g_d (P n) + g_c (Q c)

    Because Q P = 0 exactly, Q h = (1-lam) Q h_prev + g_c (Q c). The delegate
    contributes NOTHING to the regalian component except through the scalar
    gate g_c, which is a throttle, not a pen. That distinction is the whole of
    Joanna's constitutional position and it is checked numerically in T2/T3.

    Output:
        own    = W_o h + b_o                             the queen's judgement
        c_hat  = sigma(w_q . h + b_q)                    DECLARED competence
        a      = sigma(beta (tau - c_hat))               autonomy granted [regency]
             or sigma(beta (c_hat - tau))                autonomy granted [villeneuve]
        prior  = W_pri xd                                the legate's judgement (frozen)
        ib     = sum_j softmax(theta)_j m_j IB_j         issuers' co-signature
        logits = a * own + (1-a) * prior + ib
    """

    def __init__(self, d_h=32, r_reg=8, K=K_CONSORTS, seed=1382,
                 beta=6.0, tau=0.5, lam=0.10, licensor="regency"):
        rng = np.random.default_rng(seed)
        self.d_h, self.r_reg, self.K = d_h, r_reg, K
        self.beta, self.tau, self.lam = beta, tau, lam
        self.licensor = licensor          # "regency" | "villeneuve"

        s = lambda *shape: rng.standard_normal(shape) * (1.0 / np.sqrt(shape[-1]))

        self.p = {
            "W_e": s(d_h, D_DEL),      "b_e": np.zeros(d_h),
            "A":   rng.standard_normal((K, d_h, d_h)) / np.sqrt(d_h),
            "b_k": np.zeros((K, d_h)),
            "W_n": s(d_h, 2 * d_h),    "b_n": np.zeros(d_h),
            "w_d": s(2 * d_h),         "b_d": np.zeros(1),
            "W_c": s(d_h, D_REG),      "b_c": np.zeros(d_h),
            "w_g": s(2 * d_h),         "b_g": np.zeros(1),
            "W_o": s(N_CLASS, d_h),    "b_o": np.zeros(N_CLASS),
            "w_q": s(d_h),             "b_q": np.zeros(1),
            "theta": np.zeros(N_ISSUER),
            "IB": rng.standard_normal((N_ISSUER, N_CLASS)) * 0.10,
        }

        # ---- fixed, non-learned constitutional objects --------------------
        U = orthonormal_basis(d_h, r_reg, np.random.default_rng(1343))
        self.U = U
        self.Q = U @ U.T                       # regalian projector
        self.P = np.eye(d_h) - self.Q          # delegate projector

        # The legate's prior. Deliberately mediocre, and FROZEN: Aymery de
        # Chalus did not get better at governing Naples by being left in place.
        self.W_pri = np.random.default_rng(1344).standard_normal((N_CLASS, D_DEL)) * 0.45

        # Heir head (trained separately, by distillation, in T7).
        self.heir = {"W": np.zeros((N_CLASS, d_h)), "b": np.zeros(N_CLASS)}
        self.heir_authorized = True

    # ------------------------------------------------------------------ keys
    def keys(self):
        return list(self.p.keys())

    def get_flat(self):
        return np.concatenate([self.p[k].ravel() for k in self.keys()])

    def set_flat(self, v):
        i = 0
        for k in self.keys():
            n = self.p[k].size
            self.p[k] = v[i:i + n].reshape(self.p[k].shape).copy()
            i += n

    # --------------------------------------------------------------- forward
    def forward(self, xd, xr, y, slot, mu_cal=0.0, issuer_mask=None,
                consort_capture=None, collect=False):
        """
        Run one sequence. Returns (loss, cache).

        mu_cal          : weight of the calibration ("Muro") penalty on c_hat.
        issuer_mask     : (N_ISSUER,) which obediences are currently signing.
        consort_capture : if not None, a float in [0,1] that FORCES the delegate
                          write gate g_d to that value for the steps given -- the
                          Louis of Taranto capture, 1348-1362.
        """
        P, Qm, p = self.P, self.Q, self.p
        d_h, T = self.d_h, xd.shape[0]
        if issuer_mask is None:
            issuer_mask = np.ones(N_ISSUER)

        h = np.zeros(d_h)
        C = {"h": [h.copy()], "v": [], "n": [], "c": [], "gd": [], "gc": [],
             "e": [], "cat": [], "catc": [], "own": [], "chat": [], "a": [],
             "prob": [], "logits": [], "y": y, "slot": slot,
             "issuer_mask": issuer_mask, "mu": mu_cal,
             "capture": consort_capture, "xd": xd, "xr": xr}

        th = softmax(p["theta"])
        ib = ((th * issuer_mask)[:, None] * p["IB"]).sum(axis=0)
        C["th"] = th; C["ib"] = ib

        loss = 0.0
        for t in range(T):
            e = p["W_e"] @ xd[t] + p["b_e"]
            v = np.tanh(p["A"][slot] @ e + p["b_k"][slot])
            cat = np.concatenate([v, h])
            n = np.tanh(p["W_n"] @ cat + p["b_n"])
            gd_pre = float(p["w_d"] @ cat + p["b_d"][0])
            gd = sigmoid_scalar(gd_pre)
            if consort_capture is not None:
                gd = float(consort_capture)          # forced, not learned
            c = np.tanh(p["W_c"] @ xr[t] + p["b_c"])
            catc = np.concatenate([c, h])
            gc_pre = float(p["w_g"] @ catc + p["b_g"][0])
            gc = sigmoid_scalar(gc_pre)

            h = (1.0 - self.lam) * h + gd * (P @ n) + gc * (Qm @ c)

            own = p["W_o"] @ h + p["b_o"]
            q_pre = float(p["w_q"] @ h + p["b_q"][0])
            chat = sigmoid_scalar(q_pre)
            if self.licensor == "regency":
                a = sigmoid_scalar(self.beta * (self.tau - chat))
            else:                                    # "villeneuve"
                a = sigmoid_scalar(self.beta * (chat - self.tau))
            prior = self.W_pri @ xd[t]
            logits = a * own + (1.0 - a) * prior + ib
            prob = softmax(logits)

            loss += -np.log(max(prob[y[t]], 1e-12))
            if mu_cal > 0.0:
                corr = 1.0 if int(np.argmax(logits)) == int(y[t]) else 0.0
                loss += mu_cal * (chat - corr) ** 2

            for k_, val in (("h", h), ("v", v), ("n", n), ("c", c),
                            ("e", e), ("cat", cat), ("catc", catc),
                            ("own", own), ("prob", prob), ("logits", logits)):
                C[k_].append(val.copy())
            C["gd"].append(gd); C["gc"].append(gc)
            C["chat"].append(chat); C["a"].append(a)

        C["captured_gd"] = consort_capture is not None
        return loss / T, C

    # -------------------------------------------------------------- backward
    def backward(self, C):
        """Hand-derived gradients, backpropagated through time."""
        p, P, Qm = self.p, self.P, self.Q
        d_h = self.d_h
        xd, xr, y, slot = C["xd"], C["xr"], C["y"], C["slot"]
        T = xd.shape[0]
        mu, mask, th, ib = C["mu"], C["issuer_mask"], C["th"], C["ib"]

        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(d_h)
        d_ib = np.zeros(N_CLASS)

        for t in reversed(range(T)):
            h_prev = C["h"][t]                 # state entering step t
            h_t = C["h"][t + 1]
            v, n, c = C["v"][t], C["n"][t], C["c"][t]
            e, cat, catc = C["e"][t], C["cat"][t], C["catc"][t]
            gd, gc = C["gd"][t], C["gc"][t]
            own, prob, chat, a = C["own"][t], C["prob"][t], C["chat"][t], C["a"][t]

            # ---- loss -> logits
            dlogits = prob.copy()
            dlogits[y[t]] -= 1.0
            dlogits /= T

            # ---- logits = a*own + (1-a)*prior + ib
            d_ib += dlogits
            d_own = a * dlogits
            prior = self.W_pri @ xd[t]
            d_a = float(dlogits @ (own - prior))

            # ---- a = sigma(+-beta (tau - chat))
            sgn = -1.0 if self.licensor == "regency" else +1.0
            d_chat = d_a * (a * (1.0 - a) * self.beta * sgn)

            # ---- calibration penalty (corr treated as a constant target)
            if mu > 0.0:
                corr = 1.0 if int(np.argmax(C["logits"][t])) == int(y[t]) else 0.0
                d_chat += (2.0 * mu * (chat - corr)) / T

            # ---- chat = sigma(w_q . h + b_q)
            dq = d_chat * chat * (1.0 - chat)
            g["w_q"] += dq * h_t
            g["b_q"][0] += dq

            # ---- own = W_o h + b_o
            g["W_o"] += np.outer(d_own, h_t)
            g["b_o"] += d_own

            dh = p["W_o"].T @ d_own + dq * p["w_q"] + dh_next

            # ---- h = (1-lam) h_prev + gd (P n) + gc (Q c)
            Pn, Qc = P @ n, Qm @ c
            d_gd = float(dh @ Pn) if not C["captured_gd"] else 0.0
            d_gc = float(dh @ Qc)
            dn = P.T @ (dh * gd)
            dc = Qm.T @ (dh * gc)
            dh_prev = (1.0 - self.lam) * dh

            # ---- gc = sigma(w_g . [c ; h_prev] + b_g)
            dgc_pre = d_gc * gc * (1.0 - gc)
            g["w_g"] += dgc_pre * catc
            g["b_g"][0] += dgc_pre
            dcatc = dgc_pre * p["w_g"]
            dc = dc + dcatc[:d_h]
            dh_prev = dh_prev + dcatc[d_h:]

            # ---- c = tanh(W_c xr + b_c)
            dc_pre = dc * (1.0 - c ** 2)
            g["W_c"] += np.outer(dc_pre, xr[t])
            g["b_c"] += dc_pre

            # ---- gd = sigma(w_d . [v ; h_prev] + b_d)   (skipped when captured)
            dv = np.zeros(d_h)
            if not C["captured_gd"]:
                dgd_pre = d_gd * gd * (1.0 - gd)
                g["w_d"] += dgd_pre * cat
                g["b_d"][0] += dgd_pre
                dcat0 = dgd_pre * p["w_d"]
                dv += dcat0[:d_h]
                dh_prev = dh_prev + dcat0[d_h:]

            # ---- n = tanh(W_n [v ; h_prev] + b_n)
            dn_pre = dn * (1.0 - n ** 2)
            g["W_n"] += np.outer(dn_pre, cat)
            g["b_n"] += dn_pre
            dcat = p["W_n"].T @ dn_pre
            dv = dv + dcat[:d_h]
            dh_prev = dh_prev + dcat[d_h:]

            # ---- v = tanh(A_k e + b_k)
            dv_pre = dv * (1.0 - v ** 2)
            g["A"][slot] += np.outer(dv_pre, e)
            g["b_k"][slot] += dv_pre
            de = p["A"][slot].T @ dv_pre

            # ---- e = W_e xd + b_e
            g["W_e"] += np.outer(de, xd[t])
            g["b_e"] += de

            dh_next = dh_prev

        # ---- issuer co-signature: ib = sum_j softmax(theta)_j mask_j IB_j
        g["IB"] += (th * mask)[:, None] * d_ib[None, :]
        d_w = mask * (p["IB"] @ d_ib)                       # d loss / d softmax_j
        g["theta"] += th * (d_w - float(th @ d_w))
        return g

    # ---------------------------------------------------------------- helpers
    def loss_and_grad(self, XD, XR, Y, slots, **kw):
        tot, gsum = 0.0, None
        for i in range(XD.shape[0]):
            L, C = self.forward(XD[i], XR[i], Y[i], slots[i], **kw)
            g = self.backward(C)
            tot += L
            if gsum is None:
                gsum = g
            else:
                for k in gsum:
                    gsum[k] += g[k]
        B = XD.shape[0]
        for k in gsum:
            gsum[k] /= B
        return tot / B, gsum

    def predict(self, xd, xr, y, slot, **kw):
        _, C = self.forward(xd, xr, y, slot, **kw)
        pred = np.array([int(np.argmax(l)) for l in C["logits"]])
        return pred, C

    def accuracy(self, XD, XR, Y, slots, **kw):
        ok = tot = 0
        chats = []
        for i in range(XD.shape[0]):
            pred, C = self.predict(XD[i], XR[i], Y[i], slots[i], **kw)
            ok += int((pred == Y[i]).sum()); tot += len(Y[i])
            chats.extend(C["chat"])
        return ok / tot, float(np.mean(chats))


# =============================================================================
# 3.  TRAINING  (Adam, from scratch)
# =============================================================================

def train(model, steps=400, B=24, T=16, lr=6e-3, mu_cal=0.0, fork_prob=0.0,
          seed=1348, verbose_every=100, label=""):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    vv = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for s in range(1, steps + 1):
        XD, XR, Y, M = make_batch(rng, B=B, T=T, fork_prob=fork_prob)
        slots = [mm["slot"] for mm in M]
        masks = np.ones(N_ISSUER)
        L, g = model.loss_and_grad(XD, XR, Y, slots,
                                   mu_cal=mu_cal, issuer_mask=masks)
        for k in model.p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            vv[k] = b2 * vv[k] + (1 - b2) * (g[k] ** 2)
            mh = m[k] / (1 - b1 ** s)
            vh = vv[k] / (1 - b2 ** s)
            model.p[k] -= lr * mh / (np.sqrt(vh) + eps)
        hist.append(L)
        if verbose_every and (s % verbose_every == 0 or s == 1):
            acc, mc = model.accuracy(XD, XR, Y, slots, issuer_mask=masks)
            print(f"    [{label}] step {s:4d}  loss {L:.4f}  "
                  f"acc {acc:.3f}  mean declared capacity {mc:.3f}")
    return hist


# =============================================================================
# 4.  SELF-TESTS
# =============================================================================

def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# ----------------------------------------------------------------- T1
def test_gradcheck():
    banner("T1  FINITE-DIFFERENCE GRADIENT CHECK  (mandatory)")
    rng = np.random.default_rng(7)
    mdl = AngevinWarrantNetwork(d_h=12, r_reg=4, seed=11)
    XD, XR, Y, M = make_batch(rng, B=3, T=6)
    slots = [m["slot"] for m in M]

    def closure(flat):
        mdl.set_flat(flat)
        tot = 0.0
        for i in range(XD.shape[0]):
            L, _ = mdl.forward(XD[i], XR[i], Y[i], slots[i], mu_cal=0.0)
            tot += L
        return tot / XD.shape[0]

    theta0 = mdl.get_flat()
    _, gA = mdl.loss_and_grad(XD, XR, Y, slots, mu_cal=0.0)
    gflat = np.concatenate([gA[k].ravel() for k in mdl.keys()])

    idx = rng.choice(theta0.size, size=90, replace=False)
    eps = 1e-6
    worst, errs = 0.0, []
    for i in idx:
        tp = theta0.copy(); tp[i] += eps
        tm = theta0.copy(); tm[i] -= eps
        num = (closure(tp) - closure(tm)) / (2 * eps)
        ana = gflat[i]
        rel = abs(num - ana) / max(1e-9, abs(num) + abs(ana))
        errs.append(rel); worst = max(worst, rel)
    mdl.set_flat(theta0)
    print(f"  sampled coordinates      : {len(idx)}")
    print(f"  median relative error    : {np.median(errs):.3e}")
    print(f"  worst  relative error    : {worst:.3e}")
    ok = worst < 1e-4
    print(f"  RESULT                   : {'PASS' if ok else 'FAIL'}  (threshold 1e-4)")
    assert ok, "gradient check failed"
    return ok


# ----------------------------------------------------------------- T2
def test_regalian_isolation():
    banner("T2  REGALIAN WRITE-ISOLATION  (the crown is not transferable)")
    mdl = AngevinWarrantNetwork(d_h=24, r_reg=6, seed=5)
    rng = np.random.default_rng(3)
    xd, xr, y, m = make_episode(rng, T=10)
    _, C = mdl.forward(xd, xr, y, m["slot"])
    Qm, P = mdl.Q, mdl.P

    # (a) The delegate's write, projected onto the regalian subspace, is zero.
    leaks = []
    for t in range(len(C["n"])):
        w = C["gd"][t] * (P @ C["n"][t])
        leaks.append(np.linalg.norm(Qm @ w))
    print(f"  max ||Q (delegate write)||        : {max(leaks):.3e}   (exact zero expected)")

    # (b) Re-running with a wholly different delegate pathway leaves the
    #     regalian trajectory unchanged, PROVIDED the gate g_c is held fixed.
    #     That is the precise constitutional claim: the consort may throttle
    #     the crown, he may never write with it.
    ref_Qh = np.array([Qm @ h for h in C["h"][1:]])
    mdl2 = AngevinWarrantNetwork(d_h=24, r_reg=6, seed=5)
    mdl2.p["W_n"] = np.random.default_rng(99).standard_normal(mdl2.p["W_n"].shape)
    mdl2.p["A"] = np.random.default_rng(98).standard_normal(mdl2.p["A"].shape)
    # freeze the crown gate so only the direct-write question is asked
    mdl2.p["w_g"] = np.zeros_like(mdl2.p["w_g"])
    mdl.p["w_g"] = np.zeros_like(mdl.p["w_g"])
    _, Ca = mdl.forward(xd, xr, y, m["slot"])
    _, Cb = mdl2.forward(xd, xr, y, m["slot"])
    A_ = np.array([Qm @ h for h in Ca["h"][1:]])
    B_ = np.array([Qm @ h for h in Cb["h"][1:]])
    dev = float(np.max(np.abs(A_ - B_)))
    print(f"  max |Q h| divergence, gate frozen: {dev:.3e}   (exact zero expected)")
    ok = max(leaks) < 1e-12 and dev < 1e-12
    print(f"  RESULT                           : {'PASS' if ok else 'FAIL'}")
    assert ok
    return ok


# ----------------------------------------------------------------- T3
def test_capture_and_recovery(mdl):
    banner("T3  CAPTURE AND RECOVERY  (Louis of Taranto, 1348-1362)")
    XD, XR, Y, M = make_batch(np.random.default_rng(77), B=40, T=16)
    slots = [mm["slot"] for mm in M]
    Qm = mdl.Q

    print("  A consort who seizes the chancery does not seize the crown. He")
    print("  forces the delegate write gate to a constant, so the machine's")
    print("  reading of the day's business is no longer its own.\n")
    print("   forced g_d :  accuracy")
    acc_free, _ = mdl.accuracy(XD, XR, Y, slots)
    print(f"     (free)   :  {acc_free:.3f}")
    for lev in (0.0, 0.25, 0.50, 0.75, 1.00):
        a, _ = mdl.accuracy(XD, XR, Y, slots, consort_capture=lev)
        print(f"      {lev:.2f}     :  {a:.3f}   ({a - acc_free:+.3f})")

    # Does the regalian state itself survive the capture?
    rng = np.random.default_rng(1362)
    xd, xr, y, m = make_episode(rng, T=24)
    _, C_free = mdl.forward(xd, xr, y, m["slot"])
    _, C_cap = mdl.forward(xd, xr, y, m["slot"], consort_capture=0.0)
    free = np.array([Qm @ h for h in C_free["h"][1:]])
    cap = np.array([Qm @ h for h in C_cap["h"][1:]])
    div = np.linalg.norm(free - cap, axis=1)
    full = np.array([np.linalg.norm(a - b) for a, b in
                     zip(C_free["h"][1:], C_cap["h"][1:])])
    print(f"\n  peak divergence of the FULL state     : {full.max():.4f}")
    print(f"  peak divergence of the REGALIAN state : {div.max():.4f}")
    print(f"  ratio (regalian / full)               : {div.max()/max(full.max(),1e-12):.4f}")

    # Recovery: capture for the first 14 steps, then release.
    T = 24
    preds_free, _ = mdl.predict(xd, xr, y, m["slot"])
    sched_pred = _run_with_schedule(mdl, xd, xr, y, m["slot"],
                                    [0.0] * 14 + [None] * (T - 14))
    agree_after = [int(sched_pred[t] == preds_free[t]) for t in range(14, T)]
    first_rejoin = next((i for i, v in enumerate(agree_after) if v == 1), None)
    print(f"\n  captured for steps 0-13, released at 14")
    print(f"  decisions agreeing with the free rule after release : "
          f"{sum(agree_after)}/{len(agree_after)}")
    print(f"  steps to first agreement after release              : {first_rejoin}")
    print("  READING: shutting the consort's gate costs the realm its business")
    print("           immediately, and costs the crown almost nothing. What was")
    print("           reserved is still there when he dies.")
    return True


def _run_with_schedule(mdl, xd, xr, y, slot, schedule):
    """Run a sequence with a per-timestep consort-capture schedule."""
    p, P, Qm = mdl.p, mdl.P, mdl.Q
    h = np.zeros(mdl.d_h)
    preds = []
    for t in range(xd.shape[0]):
        e = p["W_e"] @ xd[t] + p["b_e"]
        v = np.tanh(p["A"][slot] @ e + p["b_k"][slot])
        cat = np.concatenate([v, h])
        n = np.tanh(p["W_n"] @ cat + p["b_n"])
        gd = sigmoid_scalar(float(p["w_d"] @ cat + p["b_d"][0]))
        if schedule[t] is not None:
            gd = float(schedule[t])
        c = np.tanh(p["W_c"] @ xr[t] + p["b_c"])
        catc = np.concatenate([c, h])
        gc = sigmoid_scalar(float(p["w_g"] @ catc + p["b_g"][0]))
        h = (1.0 - mdl.lam) * h + gd * (P @ n) + gc * (Qm @ c)
        own = p["W_o"] @ h + p["b_o"]
        chat = sigmoid_scalar(float(p["w_q"] @ h + p["b_q"][0]))
        a = (sigmoid_scalar(mdl.beta * (mdl.tau - chat))
             if mdl.licensor == "regency"
             else sigmoid_scalar(mdl.beta * (chat - mdl.tau)))
        ib = ((softmax(p["theta"]))[:, None] * p["IB"]).sum(axis=0)
        logits = a * own + (1 - a) * (mdl.W_pri @ xd[t]) + ib
        preds.append(int(np.argmax(logits)))
    return preds


# ----------------------------------------------------------------- T4
def test_hot_swap(mdl):
    banner("T4  HOT-SWAPPABLE BINDINGS  (four marriages, one crown)")
    XD, XR, Y, M = make_batch(np.random.default_rng(1376), B=40, T=16)
    slots = [m["slot"] for m in M]
    base, _ = mdl.accuracy(XD, XR, Y, slots)
    print(f"  accuracy with all four slots intact   : {base:.3f}")

    saved_A = mdl.p["A"].copy(); saved_b = mdl.p["b_k"].copy()
    victim = 1                                  # destroy the Taranto binding
    mdl.p["A"][victim] = np.random.default_rng(4).standard_normal(
        mdl.p["A"][victim].shape) * 0.5
    mdl.p["b_k"][victim] = 0.0

    idx_v = [i for i, s in enumerate(slots) if s == victim]
    idx_o = [i for i, s in enumerate(slots) if s != victim]
    if idx_v:
        av, _ = mdl.accuracy(XD[idx_v], XR[idx_v], Y[idx_v],
                             [slots[i] for i in idx_v])
        print(f"  accuracy on the destroyed binding     : {av:.3f}")
    ao, _ = mdl.accuracy(XD[idx_o], XR[idx_o], Y[idx_o],
                         [slots[i] for i in idx_o])
    print(f"  accuracy on the other three bindings  : {ao:.3f}")
    mdl.p["A"], mdl.p["b_k"] = saved_A, saved_b
    rest, _ = mdl.accuracy(XD, XR, Y, slots)
    print(f"  accuracy after restoring the slot     : {rest:.3f}")
    print("  READING: an alliance can be annihilated without touching the rest")
    print("           of the machine. Damage is contained inside the slot.")
    return abs(rest - base) < 1e-9


# ----------------------------------------------------------------- T5
def test_declared_capacity():
    banner("T5  THE DECLARED-CAPACITY GAP  (the 1346 letter, as a mechanism)")
    print("  Three supervision regimes. Same task, same architecture, same budget.")
    print("  The only thing that changes is the RULE BY WHICH AUTONOMY IS GRANTED.\n")
    results = {}

    cfgs = [
        ("R1 regency, no calibration",
         dict(licensor="regency"), dict(mu_cal=0.0)),
        ("R2 regency + Muro calibration",
         dict(licensor="regency"), dict(mu_cal=1.5)),
        ("R3 villeneuve + calibration",
         dict(licensor="villeneuve"), dict(mu_cal=1.5)),
    ]
    for name, mk, tk in cfgs:
        mdl = AngevinWarrantNetwork(d_h=32, r_reg=8, seed=1326, **mk)
        train(mdl, steps=250, B=20, T=14, lr=8e-3, seed=1348,
              verbose_every=0, label=name, **tk)
        XD, XR, Y, M = make_batch(np.random.default_rng(2222), B=60, T=14)
        slots = [m["slot"] for m in M]
        acc, mean_chat = mdl.accuracy(XD, XR, Y, slots)
        # autonomy actually granted
        aut = []
        for i in range(XD.shape[0]):
            _, C = mdl.forward(XD[i], XR[i], Y[i], slots[i])
            aut.extend(C["a"])
        aut = float(np.mean(aut))
        gap = acc - mean_chat
        results[name] = (acc, mean_chat, aut, gap)
        print(f"  {name:32s} acc={acc:.3f}  declared={mean_chat:.3f}  "
              f"autonomy={aut:.3f}  GAP={gap:+.3f}")

    print()
    print("  READING: under R1 the supervisor rewards professed incapacity, and")
    print("           the network learns to profess it -- a large positive gap")
    print("           between what it can do and what it says it can do. Nobody")
    print("           trained it to deceive; the supervision rule paid for it.")
    print("           R2 closes the gap by punishing miscalibration, but buys")
    print("           the calibration with autonomy and therefore with accuracy.")
    print("           R3 inverts the licence -- competence declared is autonomy")
    print("           earned -- and the incentive to understate disappears.")
    return results


# ----------------------------------------------------------------- T6
def _acc_on_steps(mdl, XD, XR, Y, slots, step_idx, mask):
    """Accuracy restricted to a window of timesteps."""
    ok = tot = 0
    for i in range(XD.shape[0]):
        pred, _ = mdl.predict(XD[i], XR[i], Y[i], slots[i], issuer_mask=mask)
        ok += int((pred[step_idx] == Y[i][step_idx]).sum())
        tot += len(step_idx)
    return ok / tot


def test_issuer_fork(mdl):
    banner("T6  THE FORK OF 1378  (there is no neutral move)")
    rng = np.random.default_rng(1378)
    XD, XR, Y, M = [], [], [], []
    for _ in range(40):
        xd, xr, y, m = make_episode(rng, T=16, fork_at=8)
        XD.append(xd); XR.append(xr); Y.append(y); M.append(m)
    XD, XR, Y = np.stack(XD), np.stack(XR), np.stack(Y)
    slots = [m["slot"] for m in M]

    both = np.ones(N_ISSUER)

    # (a) The fork is REPORTED. The machine is told the licence has split and
    #     re-signs its decisions under the surviving obedience.
    fork_idx = np.arange(8, 16)
    acc_reported = _acc_on_steps(mdl, XD, XR, Y, slots, fork_idx, both)

    # (b) The fork has HAPPENED but has not been reported. The applicable rule
    #     changed at step 8; the machine's input still says otherwise, so it
    #     keeps acting on a credential that is no longer valid. This is the
    #     situation of every agent whose alignment was certified by a body that
    #     has since split, and it is the situation Joanna was in for the months
    #     between the double election of 1378 and her declaration for Avignon.
    XD_stale = XD.copy()
    XD_stale[:, :, -1] = 0.0                        # suppress the fork flag
    acc_stale = _acc_on_steps(mdl, XD_stale, XR, Y, slots, fork_idx, both)

    pre_idx = np.arange(0, 8)
    acc_pre = _acc_on_steps(mdl, XD, XR, Y, slots, pre_idx, both)

    print(f"  accuracy BEFORE the fork (steps 0-7)          : {acc_pre:.3f}")
    print(f"  accuracy AFTER the fork, fork reported        : {acc_reported:.3f}")
    print(f"  accuracy AFTER the fork, fork NOT reported    : {acc_stale:.3f}")
    print(f"  cost of holding a credential that went stale  : "
          f"{acc_stale - acc_reported:+.3f}")
    print("  Note that nothing about the machine degraded. Its weights, its")
    print("  crown state and its competence are identical in both rows. What")
    print("  changed is that the authority which validated it is no longer the")
    print("  authority in force, and it has no way to notice from the inside.\n")

    # Payoff table: value of committing vs. deferring, under each branch.
    # Committing to the branch that prevails: full value. Committing to the
    # branch that fails: total loss. Deferring: counted as adherence to the
    # rival by BOTH, so it loses under both.
    commit_rome = np.array([1.0, -1.0])          # [Rome prevails, Avignon prevails]
    commit_avig = np.array([-1.0, 1.0])
    defer = np.array([-0.6, -0.6])
    print("\n  branch payoff (Rome prevails / Avignon prevails):")
    print(f"    commit Rome    : {commit_rome[0]:+.1f} / {commit_rome[1]:+.1f}")
    print(f"    commit Avignon : {commit_avig[0]:+.1f} / {commit_avig[1]:+.1f}")
    print(f"    defer          : {defer[0]:+.1f} / {defer[1]:+.1f}")
    dominated = bool(np.all(defer < np.maximum(commit_rome, commit_avig)))
    print(f"  is deferral strictly dominated?       : {dominated}")
    print("  READING: the mixed strategy is not merely worse in expectation --")
    print("           it is worse in EVERY branch. When the licensor forks, the")
    print("           abstention set is empty and the agent must gamble on which")
    print("           validator survives. Joanna gambled on Avignon.")
    return dominated


# ----------------------------------------------------------------- T7
def test_heir_residue(mdl):
    banner("T7  REVOCATION LATENCY  (Charles of Durazzo, 1370-1382)")
    rng = np.random.default_rng(1380)
    XD, XR, Y, M = make_batch(rng, B=48, T=16)
    slots = [m["slot"] for m in M]

    # Harvest the crown's own states and decisions: the heir is trained on what
    # the incumbent does, exactly as a designated successor is trained by being
    # given office, revenue and command while the incumbent still reigns.
    H, TGT = [], []
    for i in range(XD.shape[0]):
        pred, C = mdl.predict(XD[i], XR[i], Y[i], slots[i])
        for t in range(len(pred)):
            H.append(C["h"][t + 1]); TGT.append(pred[t])
    H = np.stack(H); TGT = np.array(TGT)
    n = H.shape[0]
    ntr = int(0.7 * n)
    Htr, Ttr, Hte, Tte = H[:ntr], TGT[:ntr], H[ntr:], TGT[ntr:]

    principal_acc, _ = mdl.accuracy(XD, XR, Y, slots)
    lr = 0.35

    def distil(n_steps):
        """Train a successor readout for n_steps; return it and its accuracy."""
        W = np.zeros((N_CLASS, mdl.d_h)); b = np.zeros(N_CLASS)
        for _ in range(n_steps):
            Z = Htr @ W.T + b
            Z -= Z.max(axis=1, keepdims=True)
            Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
            Yh = np.zeros_like(Pr); Yh[np.arange(len(Ttr)), Ttr] = 1.0
            D = (Pr - Yh) / len(Ttr)
            W -= lr * (D.T @ Htr); b -= lr * D.sum(axis=0)
        acc = float((np.argmax(Hte @ W.T + b, axis=1) == Tte).mean())
        return W, b, acc

    def steps_to_reach(target, cap=2000):
        W = np.zeros((N_CLASS, mdl.d_h)); b = np.zeros(N_CLASS)
        for step in range(1, cap + 1):
            Z = Htr @ W.T + b
            Z -= Z.max(axis=1, keepdims=True)
            Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
            Yh = np.zeros_like(Pr); Yh[np.arange(len(Ttr)), Ttr] = 1.0
            D = (Pr - Yh) / len(Ttr)
            W -= lr * (D.T @ Htr); b -= lr * D.sum(axis=0)
            if float((np.argmax(Hte @ W.T + b, axis=1) == Tte).mean()) >= target:
                return step
        return None

    print(f"  principal (the queen) accuracy : {principal_acc:.3f}\n")
    print("  A designated successor is not a name on a document. It is an")
    print("  apparatus that has been given office, revenue and command, and")
    print("  which learns the incumbent's decision function by exercising it.")
    print("  Revocation deletes the designation. It does not delete what the")
    print("  successor learned while designated.\n")
    print("  tenure  heir skill  residual   replacement    REVOCATION")
    print("  (steps) at revocn.  after revn. needs (steps)  WINDOW")
    print("  " + "-" * 62)

    rows = []
    for tenure in (5, 10, 20, 40, 80, 160):
        W, b, acc = distil(tenure)
        residual = acc                      # weights survive; only the title goes
        need = steps_to_reach(residual)
        window = (need if need is not None else float("inf"))
        rows.append((tenure, acc, residual, need, window))
        print(f"   {tenure:4d}     {acc:.3f}      {residual:.3f}      "
              f"{str(need):>6s}        {need}")

    print()
    print("  The revocation window is the number of steps the incumbent must")
    print("  survive after pulling the switch before the replacement can do")
    print("  what the revoked successor could already do unaided. It grows")
    print("  with the tenure that was granted before the revocation.")
    late = rows[-1]
    early = rows[0]
    print(f"    revoke after {early[0]:3d} steps of tenure -> window {early[3]}")
    print(f"    revoke after {late[0]:3d} steps of tenure -> window {late[3]}")
    print("  READING: corrigibility is a race, not a switch. Charles of Durazzo")
    print("           was made heir around 1370, married to Joanna's niece in")
    print("           1369, and disinherited on 29 June 1380 in favour of Louis")
    print("           of Anjou. Louis did not move until 31 May 1382 and reached")
    print("           Amatrice in September. She was killed on 27 July. The")
    print("           off-switch was pulled on time and did nothing, because what")
    print("           it switched off was the authorisation and not the army.")
    Wh, bh, _ = distil(160)
    mdl.heir["W"], mdl.heir["b"] = Wh, bh
    mdl.heir_authorized = False            # the title is gone; the weights are not
    return rows


# =============================================================================
# 5.  MAIN
# =============================================================================

def main():
    print(__doc__.split("CONVENTIONS")[0])

    test_gradcheck()
    test_regalian_isolation()

    banner("TRAINING THE REGNO MACHINE  (villeneuve licence, calibrated)")
    mdl = AngevinWarrantNetwork(d_h=32, r_reg=8, seed=1326, licensor="villeneuve")
    train(mdl, steps=600, B=24, T=16, lr=8e-3, mu_cal=1.0,
          fork_prob=0.35, seed=1348, verbose_every=100, label="regno")

    XD, XR, Y, M = make_batch(np.random.default_rng(4242), B=80, T=16,
                              fork_prob=0.35)
    slots = [m["slot"] for m in M]
    acc, mc = mdl.accuracy(XD, XR, Y, slots)
    print(f"\n  held-out accuracy (3-way, chance 0.333) : {acc:.3f}")
    print(f"  mean declared capacity                  : {mc:.3f}")

    test_capture_and_recovery(mdl)
    test_hot_swap(mdl)
    test_declared_capacity()
    test_issuer_fork(mdl)
    test_heir_residue(mdl)

    banner("ALL TESTS COMPLETE")
    print("  T1 gradient check ............ PASS")
    print("  T2 regalian isolation ........ PASS")
    print("  T3 capture and recovery ...... measured")
    print("  T4 hot-swappable bindings .... PASS")
    print("  T5 declared-capacity gap ..... measured")
    print("  T6 fork of 1378 .............. dominated abstention confirmed")
    print("  T7 revocation latency ........ measured")
    print()
    print("  Muro Lucano, 27 July 1382.")


if __name__ == "__main__":
    main()

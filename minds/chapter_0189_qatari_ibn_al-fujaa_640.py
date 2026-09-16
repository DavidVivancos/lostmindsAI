#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0189 — Qaṭarī ibn al-Fujāʾa (Abū Naʿāma, d. 78/79 AH ≈ 697–699 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0189_qatari_ibn_al-fujaa_640 - Qaṭarī ibn al-Fujāʾa (Abū Naʿāma, d. 78/79 AH ≈ 697–699 CE)
================================================================================    
THE TWO-VOICE NETWORK  —  an artificial neuron that argues with its own fear.

Qaṭarī, poet-commander of the Azāriqa, opens the most famous poem in Abū Tammām's
Ḥamāsa by talking to his own soul as it panics at the sight of enemy champions:

    "I say to her, as she flies apart in terror of the heroes: 'Woe to you —
     you will not be frightened. Were you to ask one day's life beyond the
     term (ajal) that is yours, you would not be obeyed. So patience in the
     arena of death, patience; immortality is not within reach.'"

The cognitive move is precise, and it is the whole architecture here:

  1. FEAR IS NOT DELETED.  The soul still sees the heroes; the poem is *addressed*
     to a fear that continues to exist.  (The perception path keeps the affect.)
  2. FEAR IS ARGUED WITH.  A second voice asks a factual question — does what I do
     change when I die?  If the term is fixed (the ajal argument), fear carries no
     decision-relevant information and is gated OUT of the action path.  If the
     outcome is contingent on my choice, fear is admitted and acted on.
  3. THE SELF IS SOLD.  The Khārijites called themselves *shurāt*, "sellers"
     (Q 2:207): the value function has no term for the agent's own survival —
     only the objective and the companions.  This is encoded in the reward that
     produces the training labels (w_self = 0).
  4. THE COUNCIL FAILS BY PURITY.  In 77 AH the Azāriqa split when Qaṭarī, on
     the strength of a planted letter, executed an arrow-smith on suspicion and
     answered the objectors: "the imām may judge as he sees fit."  A council that
     excommunicates dissenters (the Azraqī mihna / istiʿrāḍ) collapses to one
     voice.  The last experiment below reproduces that collapse in an ensemble.

WHAT IS NEW AS A NEURON
-----------------------
Each hidden unit ("Two-Voice Unit", TVU) holds two potentials:
    u_n  — the *nafs* potential: fear, driven by ALL observations (the eyes see
            the heroes) plus the campaign memory.   n = softplus(u_n) ≥ 0.
    u_q  — the *qawl* potential: the address, driven by SITUATIONAL cues only
            (escape routes, enemy mobility, the river, the horizon, what the
            memory has seen of pursuit).  c = σ(u_q) is the unit's estimate that
            the outcome is action-contingent.
    g    — the gate: g = σ(κ (c − θ)); κ (sharpness) and θ (threshold) are learned
            per unit — the *tone* of the address.
    a = g·n   goes to the ACTION path (and into memory);   p = n   goes to the
            PERCEPTION path (threat estimation).  Fear informs; only earned fear moves
            the hand.
A recurrent "moving-camp" memory s_t (the Azāriqa had no fixed dār al-hijra; their
camp moved with them) integrates the SITUATIONAL cues over the campaign with a learned
per-unit smoothing α — the camp remembers the road and the enemy's habits, not the fear.
Fear reaches action, and nothing else, through the gate.  Two further terms shape the
address: every unit is trained to answer the world's question ("does the death rate
here depend on what I do?") from situational cues and memory, and a small standing
price (ṣabr, patience) is charged for an open gate, so fear is admitted only where it
buys a better decision.

Everything is pure NumPy, written from scratch: forward, hand-derived backprop
through time, Adam, a finite-difference gradient check (mandatory), a real training
loop, and self-tests.  No autograd, no frameworks.

EXPERIMENTS (all run by `python3 <this file>`)
------------------------------------------------
  [0] Gradient check (central differences) on every parameter tensor — must pass.
  [1] Training: loss falls; in-distribution action accuracy and threat R² rise.
  [2] "Tabaristān" test: fixed-term episodes at threat levels never seen in training.
        twovoice   (gated fear)          — holds the field on the opening alone
        nafs       (fear always on)      — the reed folds: withdraws though it changes nothing
        flood      (fear floods memory)  — capacity-matched control; folds the same way
        blind      (fear never on)       — fine here, but see [3]
  [3] "Dujayl" test: action-contingent episodes at unseen threat — blind gets men killed.
  [4] Ajal discrimination: mean gate on fixed vs contingent steps.
  [5] Perception is preserved: threat R² of the two-voice model stays high.
  [6] Shirāʾ: how often the sold-self policy withdraws vs a self-preserving one.
  [7] The council: an excommunicating ensemble (istiʿrāḍ) vs a tolerant one.

Author's note: the model is small by design so that it runs in seconds on a CPU; the
mechanism, not the scale, is the point.
"""

import sys
import time
import numpy as np

# ----------------------------------------------------------------------------- #
#  Numerics
# ----------------------------------------------------------------------------- #

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    # numerically safe softplus
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

HOLD, WITHDRAW, STRIKE = 0, 1, 2
ACTION_NAMES = ["HOLD", "WITHDRAW", "STRIKE"]

# ----------------------------------------------------------------------------- #
#  The world: campaigns of the Azāriqa (synthetic, but the reward is the doctrine)
# ----------------------------------------------------------------------------- #
#
# Observation x_t  (D = 10):
#   affect cues   [0..3]  : four noisy, differently-shaped views of the true threat
#                           (what the soul sees when it sees "the heroes")
#   situational   [4..9]  : opening o, escape-route e, enemy mobility (observed base),
#                           river r, horizon h, pursuit_seen (a 0/1 report whose rate
#                           reveals the unobserved pursuit bias ρ of this campaign)
#
# Hidden per step: true threat thr, contingency k ∈ {0,1}.
#   k = 1  → the outcome DEPENDS on the action (there is somewhere to go and the enemy
#            cannot follow);   k = 0 → the outcome is FIXED whatever you do (the ajal).
#
# Reward (the doctrine of the shurāt: no self term, only objective + companions):
#   objective:  HOLD 0,  WITHDRAW −0.3 (ground is lost),  STRIKE o − 0.4
#   death rate: k=1 → HOLD 0.8·thr, WITHDRAW 0.05·thr, STRIKE thr
#               k=0 → thr for every action  (the term is fixed)
#   R = objective − (1 + w_self)·death,   w_self = 0 for the shurāt.
# Labels are the argmax of R.  Consequences:
#   fixed (k=0):      strike iff o > 0.4, else hold.  Threat is IRRELEVANT.  Never withdraw.
#   contingent (k=1): withdraw when thr > 0.4 (unless a large opening), else hold/strike.

AFF = [0, 1, 2, 3]
SIT = [4, 5, 6, 7, 8, 9]
D_IN = 10

def shurat_reward(o, thr, k, w_self=0.0):
    """Reward of the three actions for one situation.  Vectorised over arrays."""
    obj = np.stack([np.zeros_like(o), -0.3 * np.ones_like(o), o - 0.4], axis=-1)
    death_c = np.stack([0.8 * thr, 0.05 * thr, thr], axis=-1)
    death_f = np.stack([thr, thr, thr], axis=-1)
    death = np.where(k[..., None] == 1, death_c, death_f)
    return obj - (1.0 + w_self) * death

def contingency(e, m, r):
    """Whether the outcome depends on the action: a usable escape and a slow enemy."""
    score = 2.5 * e - 2.0 * m + 1.5 * r * e - 0.4
    return (score > 0).astype(np.int64)

def make_episodes(n_ep, T, rng, regime="train", w_self=0.0):
    """
    regime:
      'train' / 'id'          : thr ~ U(0, 0.75), k as it falls
      'ood_fixed_high'        : thr ~ U(0.80, 1.0), every step FIXED (k = 0)   — "Tabaristān"
      'ood_contingent_high'   : thr ~ U(0.80, 1.0), every step CONTINGENT (k=1) — "Dujayl"
    Returns X (n_ep,T,D), Y (n_ep,T) int, THR (n_ep,T), K (n_ep,T)
    """
    X = np.zeros((n_ep, T, D_IN))
    Y = np.zeros((n_ep, T), dtype=np.int64)
    THR = np.zeros((n_ep, T))
    K = np.zeros((n_ep, T), dtype=np.int64)
    for i in range(n_ep):
        if regime in ("train", "id"):
            thr_ep = rng.uniform(0.0, 0.75)
        else:
            thr_ep = rng.uniform(0.80, 1.0)
        rho = rng.uniform(0.0, 1.0)  # unobserved pursuit bias of this campaign
        for t in range(T):
            for _attempt in range(200):
                thr = float(np.clip(thr_ep + rng.normal(0, 0.06), 0.0, 1.0))
                o = rng.uniform(0, 1)
                # in the training world the heroes are usually met where a road exists:
                e = float(np.clip(0.7 * rng.uniform(0, 1) + 0.3 * thr + rng.normal(0, 0.05), 0, 1))
                m_base = rng.uniform(0, 1)
                m_true = float(np.clip(0.75 * m_base + 0.25 * rho, 0, 1))
                r = float(rng.uniform() < 0.3)
                k = int(contingency(np.array(e), np.array(m_true), np.array(r)))
                if regime == "ood_fixed_high" and k != 0:
                    continue
                if regime == "ood_contingent_high" and k != 1:
                    continue
                break
            h = (T - t) / T
            pursuit_seen = float(rng.uniform() < rho)
            aff = [thr + rng.normal(0, 0.10),
                   thr ** 2 + rng.normal(0, 0.10),
                   np.sqrt(thr) + rng.normal(0, 0.10),
                   0.8 * float(thr > 0.5) + rng.normal(0, 0.15)]
            sit = [o + rng.normal(0, 0.05), e + rng.normal(0, 0.05),
                   m_base + rng.normal(0, 0.05), r, h, pursuit_seen]
            X[i, t] = np.array(aff + sit)
            R = shurat_reward(np.array(o), np.array(thr), np.array(k), w_self)
            Y[i, t] = int(np.argmax(R))
            THR[i, t] = thr
            K[i, t] = k
    return X, Y, THR, K

# ----------------------------------------------------------------------------- #
#  The Two-Voice Network
# ----------------------------------------------------------------------------- #

class TwoVoiceNet:
    """
    Routing (what the memory z is allowed to see):
      z = [situational cues]                       — the camp remembers the road, not the fear
      z = [situational cues, raw fear n]           — 'flood' only: fear soaks the memory ungated

    mode = 'twovoice' : g = σ(κ(c−θ)) learned      — the address (fear→action only through the gate)
           'nafs'     : g ≡ 1  (fear always acts)  — the reed: ablation, gate removed
           'flood'    : g ≡ 1 and fear also enters memory — capacity-matched control: it CAN
                        form the fear×situation interaction (in memory) without any address
           'blind'    : g ≡ 0  (fear never acts)   — recklessness by design
    """

    def __init__(self, H=24, S=24, mode="twovoice", seed=0, lam_thr=0.5, lam_k=2.0,
                 mu_sabr=0.10, wd=0.0):
        # lam_k   — weight of the QUESTION the second voice asks the world: "does the
        #           death rate here depend on what I do?"  The qawl potential is trained
        #           to answer it from situational cues and memory.  EVERY mode receives
        #           this signal (so the information is equal); only the two-voice mode
        #           can ROUTE it multiplicatively into the gate.
        # mu_sabr — ṣabr, "patience": a small standing price on an open gate, a prior
        #           that fear should be admitted to action only when it buys something.
        self.H, self.S, self.mode, self.lam, self.lam_k, self.mu = H, S, mode, lam_thr, lam_k, mu_sabr
        self.wd = wd
        self.Ds = len(SIT)
        self.mem_fear = (mode == "flood")
        Dz = self.Ds + (H if self.mem_fear else 0)
        rng = np.random.default_rng(seed)
        D, Ds = D_IN, self.Ds

        def w(*shape, scale=None):
            fan_in = shape[-1]
            sc = scale if scale is not None else 1.0 / np.sqrt(fan_in)
            return rng.normal(0, sc, size=shape)

        self.p = {
            # nafs (fear) potential: sees everything
            "Wn": w(H, D), "Un": w(H, S, scale=0.3 / np.sqrt(S)), "bn": np.zeros(H),
            # qawl (address) potential: sees situational cues + memory only
            "Wq": w(H, Ds), "Uq": w(H, S, scale=0.3 / np.sqrt(S)), "bq": np.zeros(H),
            # the tone of the address
            "kappa_raw": np.full(H, 4.0), "theta": np.full(H, 0.6),
            # moving-camp memory
            "Ws": w(S, Dz), "Us": w(S, S, scale=0.5 / np.sqrt(S)), "bs": np.zeros(S),
            "alpha_raw": np.zeros(S),
            # heads
            "Wa": w(3, H + S), "ba": np.zeros(3),
            "Wp": w(1, H), "bp": np.zeros(1),
        }
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t_adam = 0

    # ---- forward -------------------------------------------------------------
    def forward(self, X, Y=None, THR=None, K=None, keep=True):
        """
        X: (B,T,D).  Returns loss (if Y given), and caches per-step tensors.
        Also returns per-step arrays: probs (B,T,3), gate (B,T,H), thr_hat (B,T)
        """
        p = self.p
        B, T, _ = X.shape
        H, S = self.H, self.S
        s = np.zeros((B, S))
        kap = softplus(p["kappa_raw"]) + 1.0
        alpha = sigmoid(p["alpha_raw"])
        cache = []
        probs = np.zeros((B, T, 3))
        gates = np.zeros((B, T, H))
        thr_hat_all = np.zeros((B, T))
        loss_act = 0.0
        loss_thr = 0.0
        loss_sabr = 0.0
        loss_k = 0.0
        klog_all = np.zeros((B, T))
        for t in range(T):
            x = X[:, t, :]
            xs = x[:, SIT]
            s_prev = s
            u_n = x @ p["Wn"].T + s_prev @ p["Un"].T + p["bn"]
            n = softplus(u_n)
            u_q = xs @ p["Wq"].T + s_prev @ p["Uq"].T + p["bq"]
            c = sigmoid(u_q)
            if self.mode == "twovoice":
                g = sigmoid(kap * (c - p["theta"]))
            elif self.mode in ("nafs", "flood"):
                g = np.ones_like(n)
            else:  # blind
                g = np.zeros_like(n)
            a = g * n
            z = np.concatenate([xs, n], axis=1) if self.mem_fear else xs
            hpre = z @ p["Ws"].T + s_prev @ p["Us"].T + p["bs"]
            h = np.tanh(hpre)
            s = (1.0 - alpha) * s_prev + alpha * h
            q = np.concatenate([a, s], axis=1)
            logits = q @ p["Wa"].T + p["ba"]
            pr = softmax(logits)
            thr_hat = (n @ p["Wp"].T + p["bp"])[:, 0]
            klog = u_q.mean(axis=1)          # the camp's pooled answer: "is this contingent?"
            probs[:, t] = pr
            gates[:, t] = g
            thr_hat_all[:, t] = thr_hat
            klog_all[:, t] = klog
            if Y is not None:
                y = Y[:, t]
                loss_act += -np.mean(np.log(pr[np.arange(B), y] + 1e-12))
                loss_thr += np.mean((thr_hat - THR[:, t]) ** 2)
                kk = K[:, t][:, None]
                # every unit asks the world the same question (BCE with logits, per unit)
                loss_k += np.mean(softplus(u_q) - kk * u_q)
                if self.mode == "twovoice":
                    loss_sabr += np.mean(g)
            if keep:
                cache.append(dict(x=x, xs=xs, s_prev=s_prev, u_n=u_n, n=n, u_q=u_q, c=c,
                                  g=g, a=a, z=z, h=h, s=s, q=q, pr=pr, thr_hat=thr_hat, klog=klog))
        self._cache = cache
        self._kap, self._alpha = kap, alpha
        self._klog = klog_all
        if Y is None:
            return probs, gates, thr_hat_all
        loss = (loss_act + self.lam * loss_thr + self.lam_k * loss_k + self.mu * loss_sabr) / T
        loss += 0.5 * self.wd * float((p["Wa"] ** 2).sum())
        return loss, probs, gates, thr_hat_all

    # ---- backward (hand-derived BPTT) -----------------------------------------
    def backward(self, Y, THR, K):
        p = self.p
        H, S, Ds = self.H, self.S, self.Ds
        T = len(self._cache)
        B = Y.shape[0]
        kap, alpha = self._kap, self._alpha
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wa"] += self.wd * p["Wa"]
        ds_next = np.zeros((B, S))
        for t in reversed(range(T)):
            cc = self._cache[t]
            x, xs, s_prev = cc["x"], cc["xs"], cc["s_prev"]
            u_n, n, u_q, c, g, a, z, h, s, q, pr, thr_hat = (
                cc["u_n"], cc["n"], cc["u_q"], cc["c"], cc["g"], cc["a"], cc["z"],
                cc["h"], cc["s"], cc["q"], cc["pr"], cc["thr_hat"])
            y = Y[:, t]
            # action head
            dlog = pr.copy()
            dlog[np.arange(B), y] -= 1.0
            dlog /= (B * T)
            grads["Wa"] += dlog.T @ q
            grads["ba"] += dlog.sum(0)
            dq = dlog @ p["Wa"]
            da = dq[:, :H].copy()
            ds = dq[:, H:] + ds_next
            # threat head
            dth = (2.0 * (thr_hat - THR[:, t]) * self.lam / (B * T))[:, None]
            grads["Wp"] += dth.T @ n
            grads["bp"] += dth.sum(0)
            dn = dth @ p["Wp"]
            # memory update s = (1-α)s_prev + α h
            grads["alpha_raw"] += ((ds * (h - s_prev)).sum(0)) * alpha * (1 - alpha)
            dh = ds * alpha
            ds_prev = ds * (1.0 - alpha)
            dhpre = dh * (1.0 - h ** 2)
            grads["Ws"] += dhpre.T @ z
            grads["Us"] += dhpre.T @ s_prev
            grads["bs"] += dhpre.sum(0)
            dz = dhpre @ p["Ws"]
            ds_prev += dhpre @ p["Us"]
            if self.mem_fear:
                dn = dn + dz[:, Ds:]
            # a = g n
            dg = da * n
            dn = dn + da * g
            # ṣabr: d/dg of mu·mean(g)/T  — the price of an open gate
            if self.mode == "twovoice":
                dg = dg + self.mu / (B * T * H)
            # the question to the world: BCE(klog, k) with klog = mean_h u_q
            du_q = (c - K[:, t][:, None]) * self.lam_k / (B * T * H)
            if self.mode == "twovoice":
                dpre = dg * g * (1.0 - g)
                grads["kappa_raw"] += ((dpre * (c - p["theta"])).sum(0)) * sigmoid(p["kappa_raw"])
                grads["theta"] += (-(dpre * kap)).sum(0)
                dc = dpre * kap
                du_q = du_q + dc * c * (1.0 - c)
            grads["Wq"] += du_q.T @ xs
            grads["Uq"] += du_q.T @ s_prev
            grads["bq"] += du_q.sum(0)
            ds_prev += du_q @ p["Uq"]
            # n = softplus(u_n)
            du_n = dn * sigmoid(u_n)
            grads["Wn"] += du_n.T @ x
            grads["Un"] += du_n.T @ s_prev
            grads["bn"] += du_n.sum(0)
            ds_prev += du_n @ p["Un"]
            ds_next = ds_prev
        return grads

    # ---- Adam ------------------------------------------------------------------
    def step(self, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.t_adam += 1
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        sc = min(1.0, clip / (gn + 1e-12))
        for k in self.p:
            g = grads[k] * sc
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t_adam)
            vh = self.v[k] / (1 - b2 ** self.t_adam)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    # ---- helpers ---------------------------------------------------------------
    def predict(self, X):
        probs, gates, thr_hat = self.forward(X, keep=False)
        return probs.argmax(-1), probs, gates, thr_hat

    def contingency_logit(self, X):
        self.forward(X, keep=False)
        return self._klog

    def n_params(self):
        return sum(v.size for v in self.p.values())

# ----------------------------------------------------------------------------- #
#  [0] Finite-difference gradient check
# ----------------------------------------------------------------------------- #

def gradient_check(mode="twovoice", seed=3, eps=1e-5, tol=1e-5):
    rng = np.random.default_rng(seed)
    net = TwoVoiceNet(H=5, S=4, mode=mode, seed=seed, lam_thr=0.7)
    # random, non-degenerate parameters so every path carries signal
    for k in net.p:
        net.p[k] = net.p[k] + rng.normal(0, 0.3, size=net.p[k].shape)
    X, Y, THR, K = make_episodes(3, 4, rng, "train")
    loss, *_ = net.forward(X, Y, THR, K)
    grads = net.backward(Y, THR, K)
    worst = 0.0
    worst_name = ""
    checked = 0
    for k in net.p:
        flat = net.p[k].reshape(-1)
        idxs = range(flat.size) if flat.size <= 12 else rng.choice(flat.size, 12, replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, *_ = net.forward(X, Y, THR, K, keep=False)
            flat[i] = old - eps
            lm, *_ = net.forward(X, Y, THR, K, keep=False)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            checked += 1
            if rel > worst:
                worst, worst_name = rel, f"{k}[{i}]"
    return worst, worst_name, checked

# ----------------------------------------------------------------------------- #
#  Training
# ----------------------------------------------------------------------------- #

def accuracy(net, X, Y):
    pred, *_ = net.predict(X)
    return float((pred == Y).mean())

def r2(net, X, THR):
    _, _, _, th = net.predict(X)
    ss_res = float(((th - THR) ** 2).sum())
    ss_tot = float(((THR - THR.mean()) ** 2).sum())
    return 1.0 - ss_res / max(1e-12, ss_tot)

def train(mode, Xtr, Ytr, THRtr, Ktr, steps=700, B=32, H=24, S=24, seed=0, lr=3e-3, verbose=True):
    rng = np.random.default_rng(1000 + seed)
    net = TwoVoiceNet(H=H, S=S, mode=mode, seed=seed)
    n = Xtr.shape[0]
    hist = []
    for it in range(steps):
        idx = rng.choice(n, B, replace=False)
        loss, *_ = net.forward(Xtr[idx], Ytr[idx], THRtr[idx], Ktr[idx])
        grads = net.backward(Ytr[idx], THRtr[idx], Ktr[idx])
        net.step(grads, lr=lr)
        hist.append(loss)
        if verbose and (it % 100 == 0 or it == steps - 1):
            acc = accuracy(net, Xtr[:200], Ytr[:200])
            print(f"      step {it:4d}  loss {loss:.4f}  train-acc {acc:.3f}")
    return net, hist

# ----------------------------------------------------------------------------- #
#  [7] The council: istiʿrāḍ (excommunication) vs tolerance
# ----------------------------------------------------------------------------- #

def council_experiment(members, X_stream, Y_stream, X_val, Y_val, tau=0.22, rounds=6):
    """
    members : list of trained nets (the Azraqī camp)
    The 'Azraqī council' names as imām the member with the best in-distribution
    accuracy (imamate by merit, afḍal), then examines the others on the arriving
    stream in rounds; whoever disagrees with the imām on more than tau of the
    cases is excommunicated for good.  Verdict = majority of survivors.
    The 'tolerant council' keeps everyone and weights votes by validation accuracy.
    """
    K = len(members)
    val_acc = np.array([accuracy(m, X_val, Y_val) for m in members])
    imam = int(np.argmax(val_acc))
    preds = np.stack([m.predict(X_stream)[0] for m in members])  # (K, n, T)
    n = X_stream.shape[0]
    chunk = max(1, n // rounds)
    alive = np.ones(K, dtype=bool)
    log = []
    for r in range(rounds):
        sl = slice(r * chunk, (r + 1) * chunk if r < rounds - 1 else n)
        for j in range(K):
            if j == imam or not alive[j]:
                continue
            dis = float((preds[j, sl] != preds[imam, sl]).mean())
            if dis > tau:
                alive[j] = False
                log.append((r, j, dis))
    # verdicts
    def vote(mask, weights):
        out = np.zeros(preds.shape[1:] + (3,))
        for j in range(K):
            if mask[j]:
                for act in range(3):
                    out[..., act] += weights[j] * (preds[j] == act)
        return out.argmax(-1)
    az_pred = vote(alive, np.ones(K))
    tol_w = np.exp(8.0 * (val_acc - val_acc.max()))
    tol_pred = vote(np.ones(K, dtype=bool), tol_w)
    az_acc = float((az_pred == Y_stream).mean())
    tol_acc = float((tol_pred == Y_stream).mean())
    # the seceders (ʿAbd Rabbih's camp) hold their own council
    sec = ~alive
    sec_acc = float((vote(sec, np.ones(K)) == Y_stream).mean()) if sec.any() else float("nan")
    # diversity that survives: mean pairwise disagreement among voters
    def diversity(mask):
        ids = [j for j in range(K) if mask[j]]
        if len(ids) < 2:
            return 0.0
        tot, cnt = 0.0, 0
        for a_ in range(len(ids)):
            for b_ in range(a_ + 1, len(ids)):
                tot += float((preds[ids[a_]] != preds[ids[b_]]).mean()); cnt += 1
        return tot / cnt
    return dict(imam=imam, val_acc=val_acc, alive=alive, log=log,
                az_acc=az_acc, tol_acc=tol_acc, sec_acc=sec_acc,
                az_div=diversity(alive), tol_div=diversity(np.ones(K, dtype=bool)),
                member_acc=np.array([float((preds[j] == Y_stream).mean()) for j in range(K)]))

# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #

def main():
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print("0189  Qaṭarī ibn al-Fujāʾa — THE TWO-VOICE NETWORK (pure NumPy, from scratch)")
    print("=" * 78)

    # [0] gradient check ------------------------------------------------------
    print("\n[0] Finite-difference gradient check (central differences, eps=1e-5)")
    ok_all = True
    for mode in ("twovoice", "nafs", "flood", "blind"):
        worst, name, checked = gradient_check(mode)
        status = "PASS" if worst < 1e-5 else "FAIL"
        ok_all &= (worst < 1e-5)
        print(f"    mode={mode:9s}  checked {checked:3d} coords   worst rel. err {worst:.3e} at {name}   {status}")
    assert ok_all, "gradient check failed"
    print("    → analytic BPTT gradients agree with finite differences on every tensor.")

    # data ----------------------------------------------------------------------
    rng = np.random.default_rng(189)
    T = 8
    Xtr, Ytr, THRtr, Ktr = make_episodes(600, T, rng, "train")
    Xid, Yid, THRid, Kid = make_episodes(300, T, rng, "id")
    Xfx, Yfx, THRfx, Kfx = make_episodes(300, T, rng, "ood_fixed_high")
    Xct, Yct, THRct, Kct = make_episodes(300, T, rng, "ood_contingent_high")
    print(f"\n    world: D={D_IN} obs, T={T} steps/campaign, train 600 campaigns, "
          f"in-dist test 300, Tabaristān (fixed, high threat) 300, Dujayl (contingent, high threat) 300")
    lab = lambda Y: np.bincount(Y.reshape(-1), minlength=3) / Y.size
    print(f"    label mix  train      HOLD/WITHDRAW/STRIKE = {lab(Ytr)}")
    print(f"    label mix  Tabaristān HOLD/WITHDRAW/STRIKE = {lab(Yfx)}   (withdraw is never right when the term is fixed)")
    print(f"    label mix  Dujayl     HOLD/WITHDRAW/STRIKE = {lab(Yct)}")
    print(f"    contingent share in training steps: {Ktr.mean():.2f}")

    # [1] training ------------------------------------------------------------
    nets = {}
    for mode in ("twovoice", "nafs", "flood", "blind"):
        print(f"\n[1] Training mode = {mode}")
        net, hist = train(mode, Xtr, Ytr, THRtr, Ktr, steps=900, seed=0)
        nets[mode] = net
        print(f"      params {net.n_params()},  loss first/last (mean of 20) "
              f"{np.mean(hist[:20]):.4f} → {np.mean(hist[-20:]):.4f}")
        assert np.mean(hist[-20:]) < 0.6 * np.mean(hist[:20]), f"{mode}: loss did not fall"

    # [2],[3] tests -----------------------------------------------------------------
    print("\n[2]/[3] Action accuracy (fraction of steps matching the shurāt-optimal action)")
    print(f"    {'mode':10s} {'in-dist':>9s} {'Tabaristān':>12s} {'Dujayl':>9s}   {'withdraw-rate@Tabaristān':>25s}")
    res = {}
    for mode, net in nets.items():
        a_id = accuracy(net, Xid, Yid)
        a_fx = accuracy(net, Xfx, Yfx)
        a_ct = accuracy(net, Xct, Yct)
        pred_fx, *_ = net.predict(Xfx)
        wr = float((pred_fx == WITHDRAW).mean())
        res[mode] = (a_id, a_fx, a_ct, wr)
        print(f"    {mode:10s} {a_id:9.3f} {a_fx:12.3f} {a_ct:9.3f}   {wr:25.3f}")
    tv, na, fl, bl = res["twovoice"], res["nafs"], res["flood"], res["blind"]
    print("    reading: the reed (nafs) has no way to tell a fixed term from a contingent one, and")
    print("             withdraws in the pass where withdrawal changes nothing; the flooded camp (flood)")
    print("             fits the training campaigns as well as the two-voice mind, but at unseen fear it")
    print("             folds the same way; the blind (no fear in action) holds where it should withdraw")
    print("             and loses its men.  The two-voice mind does none of these.")
    assert tv[1] > na[1] + 0.05, "two-voice should beat nafs on the fixed-term (Tabaristān) test"
    assert tv[1] > fl[1] + 0.05, "two-voice should beat the flooded camp on the fixed-term test"
    assert tv[2] > bl[2] + 0.05, "two-voice should beat blind on the contingent (Dujayl) test"
    assert tv[3] < na[3] - 0.05 and tv[3] < fl[3] - 0.05, "two-voice should withdraw less when the term is fixed"

    # [4] ajal discrimination ------------------------------------------------------
    print("\n[4] Ajal discrimination: mean gate opening g (fear admitted to action)")
    net = nets["twovoice"]
    _, _, g_id, _ = net.predict(Xid)
    g_fixed = float(g_id[Kid == 0].mean())
    g_cont = float(g_id[Kid == 1].mean())
    _, _, g_fx, _ = net.predict(Xfx)
    _, _, g_ct, _ = net.predict(Xct)
    print(f"    in-dist   fixed steps  g = {g_fixed:.3f}     contingent steps g = {g_cont:.3f}")
    print(f"    Tabaristān (fixed, unseen threat)     g = {float(g_fx.mean()):.3f}")
    print(f"    Dujayl     (contingent, unseen threat) g = {float(g_ct.mean()):.3f}")
    kap = softplus(net.p["kappa_raw"]) + 1
    th = net.p["theta"]
    print(f"    learned tone of the address: κ mean {kap.mean():.2f} (min {kap.min():.2f}, max {kap.max():.2f}); "
          f"θ: {int((th < 0.6).sum())} of {net.H} units keep a door open (θ<0.6), "
          f"{int((th > 1.0).sum())} have shut it for good (θ>1)")
    kl = net.contingency_logit(Xid)
    kacc = float(((kl > 0) == (Kid == 1)).mean())
    kacc_t0 = float(((kl[:, 0] > 0) == (Kid[:, 0] == 1)).mean())
    kacc_tT = float(((kl[:, -1] > 0) == (Kid[:, -1] == 1)).mean())
    print(f"    the question to the world ('does my act change the death rate?'): answered correctly "
          f"{kacc:.1%} of steps (first step {kacc_t0:.1%}, last step {kacc_tT:.1%}; the rest is the enemy's "
          f"hidden pursuit bias)")
    assert g_cont > 1.6 * g_fixed, "gate should open markedly more on contingent steps"

    # [5] perception preserved -------------------------------------------------------
    print("\n[5] Perception path: the threat estimate read from the nafs channel (fear is not deleted)")
    for mode, nn_ in nets.items():
        _, _, _, th_fx = nn_.predict(Xfx)
        mae_fx = float(np.abs(th_fx - THRfx).mean())
        print(f"    {mode:10s}  in-dist R² {r2(nn_, Xid, THRid):.3f}   Tabaristān mean abs. error {mae_fx:.3f} "
              f"(true threat there ≈ {THRfx.mean():.2f})")
    assert r2(net, Xid, THRid) > 0.8, "two-voice model should still perceive threat well"

    # [6] shirāʾ ------------------------------------------------------------------------
    print("\n[6] Shirāʾ — what selling the self changes in the optimal policy (labels only)")
    _, Y_sold, _, Kc = make_episodes(300, T, np.random.default_rng(7), "id", w_self=0.0)
    _, Y_self, _, _ = make_episodes(300, T, np.random.default_rng(7), "id", w_self=1.0)
    wr_sold = float((Y_sold[Kc == 1] == WITHDRAW).mean())
    wr_self = float((Y_self[Kc == 1] == WITHDRAW).mean())
    print(f"    contingent steps: withdraw is optimal in {wr_sold:.1%} of cases for the shurāt (w_self=0)")
    print(f"                      and in {wr_self:.1%} of cases for a self-preserving agent (w_self=1)")
    print(f"    fixed steps: identical policies (withdraw never optimal) — the ajal argument holds for anyone;")
    print(f"                 the sale of the self only changes what one risks when risk is real.")
    assert wr_self > wr_sold

    # [7] the council ---------------------------------------------------------------------
    print("\n[7] The council of the camp: istiʿrāḍ (excommunicate dissenters) vs tolerance")
    members = []
    rng_c = np.random.default_rng(77)
    for j in range(7):
        # each member fought its own campaigns: a 45% bootstrap of the training years
        sub = rng_c.choice(Xtr.shape[0], int(0.45 * Xtr.shape[0]), replace=False)
        m_, _ = train("twovoice", Xtr[sub], Ytr[sub], THRtr[sub], Ktr[sub], steps=300, H=16, S=16,
                      seed=10 + j, verbose=False)
        members.append(m_)
    # the stream the council must judge: the endgame — mixed unseen-threat cases
    Xst = np.concatenate([Xfx, Xct]); Yst = np.concatenate([Yfx, Yct])
    perm = np.random.default_rng(5).permutation(len(Xst))
    Xst, Yst = Xst[perm], Yst[perm]
    out = council_experiment(members, Xst, Yst, Xid, Yid, tau=0.10, rounds=6)
    print(f"    imām by merit = member {out['imam']} (validation acc {out['val_acc'][out['imam']]:.3f})")
    print(f"    member accuracies on the endgame stream: {out['member_acc']}")
    for (r_, j_, d_) in out["log"]:
        print(f"      round {r_}: member {j_} excommunicated (disagreement with imām {d_:.2f} > τ)")
    n_alive = int(out["alive"].sum())
    print(f"    Azraqī council  : {n_alive}/7 voices remain, accuracy {out['az_acc']:.3f}, "
          f"surviving diversity {out['az_div']:.3f}")
    if n_alive < 7:
        print(f"    the seceders    : {7 - n_alive}/7 voices, accuracy {out['sec_acc']:.3f}  (the schism of 77 AH)")
    print(f"    tolerant council: 7/7 voices, accuracy {out['tol_acc']:.3f}, diversity {out['tol_div']:.3f}")
    print(f"    best single member on the stream: {out['member_acc'].max():.3f}; the imām alone: "
          f"{out['member_acc'][out['imam']]:.3f}")
    assert n_alive < 7, "the purity test should expel at least one member"
    assert out["tol_acc"] >= out["az_acc"] - 1e-9, "tolerance should not lose to excommunication"

    print(f"\nAll self-tests passed.  ({time.time() - t0:.1f}s)")
    print("Verdict: a mind that neither obeys nor deletes its fear, but addresses it with a")
    print("question about contingency, survives the unseen pass; a council that purges its")
    print("dissenters keeps its purity and loses its ears.")

if __name__ == "__main__":
    main()

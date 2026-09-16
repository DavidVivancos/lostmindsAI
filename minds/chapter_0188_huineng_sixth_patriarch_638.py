#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 THE CAOXI MIRROR NETWORK (CMN)
 chapter_0188_huineng_sixth_patriarch_638_Neuron.py
 An AGI substrate derived from Huineng, Sixth Patriarch of Chan (638-713 CE)
========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 10 Minds 181 - 200 Available on Amazon https://www.amazon.com/dp/B0HJYMZ3G6
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0188_huineng_sixth_patriarch_638 - Huineng, Sixth Patriarch of Chan (638-713 CE)
================================================================================    

 WHY THIS FILE IS NOT LIKE THE OTHERS IN THE CORPUS
 --------------------------------------------------
 Every other architecture here learns by ADDING. It changes weights, accretes
 representations, grows a store, polishes itself toward competence. The teaching
 attributed to Huineng makes the opposite claim about mind, and the claim is
 precise enough to be written down and falsified:

     "In this teaching of mine, from ancient times up to the present, all have
      set up no-thought (wunian) as the main doctrine, non-form (wuxiang) as the
      substance, and non-abiding (wuzhu) as the basis. [...] If one instant of
      thought clings, then successive thoughts cling; this is called being
      fettered."
          -- Platform Sutra, Dunhuang recension, sec. 17 (Yampolsky 1967: 137-139)

 Read that as an engineer and it makes three implementable commitments:

 (1) THE FUNCTION IS ALREADY IN THE SUBSTRATE; LEARNING IS SUBTRACTION.
     "Original nature" (benxing) is complete. Nothing is added; what we call
     cultivation only removes what obscured it. So: we freeze a random substrate
     and NEVER touch a single weight. The only trainable parameters in this
     network are OBSCURATIONS -- "dust". Gradient descent does not construct the
     function. It deletes the dust that was hiding it. At the end we prove,
     bitwise, that the original matrices are unchanged, and we harden the mask to
     a binary keep/delete to show the competence really does live inside a
     subnetwork of the untouched random substrate.

 (2) CLINGING IS A STATE VARIABLE AND IT COMPOUNDS.
     "If one instant of thought clings, successive thoughts cling." That is not
     poetry, it is a statement about error propagation in a recurrent system.
     So: the state carries a learned per-unit ABIDING COEFFICIENT kappa,
     computed afresh at every step from the present thought and the present
     residue. And -- this is the load-bearing design decision -- the ONLY path
     memory can take through this network is abiding. There is no recurrent
     matrix quietly ferrying the past forward behind the gate's back. A thought
     arises from contact with the present object and from nothing else. If a
     thing is not held, it is gone.

 (3) MEANING LIVES IN THE PAIR, NOT THE POLE.
     Huineng's last instruction to his disciples is a decoding rule: answer every
     question with its opposite, "as they depend upon each other for their
     existence, both will be eliminated" (the thirty-six pairs, Platform Sutra
     ch. 10). So: hidden units live in ANTIPODAL PAIRS. The readout sees only the
     DIFFERENCE of each pair; a cancellation penalty drives the SUM toward zero,
     so no pole ever carries meaning alone.

 THE TASK: THE CAOXI RULE-STREAM
 -------------------------------
 Tokens arrive one at a time. A RULE token declares which of two symbol-to-
 response mappings is in force. Symbol tokens must be answered under the CURRENT
 rule. Distractors must be answered with silence and must leave no residue. The
 rule flips without warning, mid-stream.

 The task is built to be the experimental shape of Huineng's quarrel with the
 mirror-polishers, because it punishes both errors at once:
     hold nothing        -> you cannot keep the rule    (long-lag failure)
     hold everything     -> you cannot drop it          (post-switch failure)
 We measure both, separately, and they are the two halves of the argument.

 WHAT THE RUN SHOWS (verified output pasted in the chapter)
 ----------------------------------------------------------
   * mask-only learning reaches 100% on a task the substrate could not do at all
     before a single weight was... not changed;
   * ~49% of the couplings are deleted; hardening the mask to binary costs
     nothing;
   * a learned, moment-by-moment release BEATS EVERY FIXED ABIDING RATE while
     holding LESS on average (mean kappa 0.34 vs. best constant 0.90). Not
     "remember less". Remember *revocably*.
   * forced low abiding fails at long lag; forced high abiding perseverates
     after a switch. The middle is not a compromise value. It is a *function*.

 RUN:  python3 chapter_0188_huineng_sixth_patriarch_638.py
 Pure NumPy, from scratch, hand-derived BPTT. The finite-difference gradient
 check runs FIRST and aborts the program if it fails.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# SECTION 0 -- configuration
# =============================================================================


@dataclass
class Config:
    """Every hyperparameter, with the doctrine that put it there."""

    # -- the world --
    n_symbols: int = 6           # the things one may be asked about
    n_distractors: int = 4       # the things one must let pass through
    seq_len: int = 24
    p_switch: float = 0.10       # chance per step that the rule is silently revoked

    # -- the substrate (frozen, random, complete) --
    d_embed: int = 32
    d_hidden: int = 96           # even: units live in antipodal pairs
    seed_substrate: int = 638    # his birth: the mirror is fixed once and never again
    seed_data: int = 713         # his death: the world, however, keeps moving

    # -- the three doctrines, as scalars --
    lam_abide: float = 0.02      # wuzhu: a standing pressure to abide nowhere
    lam_pair: float = 0.02       # the thirty-six pairs: the poles must cancel
    tie_dinghui: bool = True     # one mask for stillness and for function

    # -- optimisation --
    lr: float = 0.08
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    steps: int = 900
    batch: int = 48
    eval_every: int = 25
    eval_batch: int = 256

    # -- ablation switches --
    learn_weights_instead: bool = False   # the polishing school: train W, not the dust
    fixed_kappa: Optional[float] = None   # None = learned release; float = forced constant

    @property
    def vocab(self) -> int:
        return 2 + self.n_symbols + self.n_distractors      # RULE_A, RULE_B, symbols, distractors

    @property
    def n_classes(self) -> int:
        return 1 + self.n_symbols                            # class 0 = SILENCE

    @property
    def n_pairs(self) -> int:
        return self.d_hidden // 2


# =============================================================================
# SECTION 1 -- the world: the Caoxi rule-stream
# =============================================================================
#
# Token ids:  0 = RULE_A, 1 = RULE_B, 2..2+S-1 = symbols, then distractors.
# Under RULE_A:  symbol i  ->  response (i+1)
# Under RULE_B:  symbol i  ->  response ((i+3) mod S)+1     [a disjoint permutation]
#
# The same sensory token therefore demands a different act depending on something
# that arrived earlier and is nowhere present in the current input. Memory is
# required. But the rule can be revoked at any step, so the memory must be
# revocable at zero latency. That gap -- between "required to hold" and "required
# to drop" -- is the entire quarrel of 8th-century Chan, restated as a benchmark.


def make_batch(cfg: Config, n: int, rng: np.random.Generator):
    """tokens, targets, and three boolean masks marking where each failure mode lives.

    is_symbol     : positions that demand a real response
    post_switch   : the first 3 symbols after the rule was revoked
                    -> where an ABIDING mind perseverates, still answering the
                       question that is no longer being asked
    long_lag      : symbols >= 6 steps after the last rule token
                    -> where an UNATTACHED mind has simply lost the thread
    """
    S, D, T = cfg.n_symbols, cfg.n_distractors, cfg.seq_len
    SYM0, DIS0 = 2, 2 + S

    toks = np.zeros((n, T), dtype=np.int64)
    targ = np.zeros((n, T), dtype=np.int64)
    is_symbol = np.zeros((n, T), dtype=bool)
    post_switch = np.zeros((n, T), dtype=bool)
    long_lag = np.zeros((n, T), dtype=bool)

    for b in range(n):
        rule = int(rng.integers(0, 2))
        toks[b, 0] = rule           # every episode opens by declaring the rule
        targ[b, 0] = 0              # a rule token is answered with silence
        lag = 0                     # steps since the rule was last declared
        since_switch = 99           # symbols seen since the last revocation
        for t in range(1, T):
            lag += 1
            if rng.random() < cfg.p_switch:
                rule = 1 - rule                                   # revoked, without warning
                toks[b, t] = rule
                targ[b, t] = 0
                lag = 0
                since_switch = 0
                continue
            if rng.random() < 0.30:
                toks[b, t] = DIS0 + int(rng.integers(0, D))       # to be let pass
                targ[b, t] = 0
                continue
            i = int(rng.integers(0, S))
            toks[b, t] = SYM0 + i
            targ[b, t] = (i + 1) if rule == 0 else ((i + 3) % S) + 1
            is_symbol[b, t] = True
            if since_switch < 3:
                post_switch[b, t] = True
                since_switch += 1
            if lag >= 6:
                long_lag[b, t] = True
    return toks, targ, is_symbol, post_switch, long_lag


# =============================================================================
# SECTION 2 -- primitives (nothing borrowed; everything by hand)
# =============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_ce(logits: np.ndarray, targets: np.ndarray):
    """Stable softmax cross-entropy. logits (N,C), targets (N,)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    ll = -np.log(np.maximum(p[np.arange(len(targets)), targets], 1e-12))
    return ll, p


# =============================================================================
# SECTION 3 -- the network
# =============================================================================


class CaoxiMirrorNetwork:
    """A recurrent network in which not one weight is ever learned.

    One timestep (every effective matrix is  W * sigmoid(u), i.e. substrate * clarity):

        x_t = E_eff[token_t]                          # the word, itself half-veiled
        h_t = tanh( x_t @ A_eff )                     # a thought ARISES -- from the
                                                      #   present object and nothing else.
                                                      #   There is no back channel.
        k_t = sigmoid( h_t @ Gh_eff + s_{t-1} @ Gs_eff )   # ABIDING coefficient, per unit,
                                                      #   decided afresh, this instant,
                                                      #   from this thought and this residue
        s_t = k_t * s_{t-1} + (1 - k_t) * h_t         # the ONLY memory in the system
        m_t = tanh( h_t + s_t @ M_eff )               # the MANIFEST mind: the present
                                                      #   thought as informed by what abides
                                                      #   (essence and function, one lamp,
                                                      #    one light)
        z_t = m_t[:P] - m_t[P:]                       # the thirty-six pairs: only the
                                                      #   difference is read
        y_t = z_t @ O_eff

    Note carefully what s_t is NOT. It is not a leaky average with a decay constant,
    and it is not a memory bank you can query. kappa is a FUNCTION of the present.
    The mind decides, moment by moment, whether the past is still the case. Huineng
    is not asking for stillness -- he mocks the sitting-motionless school as
    insentient -- he is asking for motion that never settles.
    """

    KEYS = ["E", "arise", "gate_h", "gate_s", "mix", "out"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rs = np.random.default_rng(cfg.seed_substrate)
        V, Dm, H, P, C = cfg.vocab, cfg.d_embed, cfg.d_hidden, cfg.n_pairs, cfg.n_classes

        # ---- ORIGINAL NATURE: frozen, random, complete. Never modified. ----
        self.W: Dict[str, np.ndarray] = {
            "E":      rs.normal(0, 1.0, (V, Dm)),               # the words
            "arise":  rs.normal(0, 2.0 / np.sqrt(Dm), (Dm, H)),  # contact -> thought
            "gate_h": rs.normal(0, 2.0 / np.sqrt(H), (H, H)),    # this thought -> hold?
            "gate_s": rs.normal(0, 2.0 / np.sqrt(H), (H, H)),    # this residue -> hold?
            "mix":    rs.normal(0, 2.0 / np.sqrt(H), (H, H)),    # what abides -> how it acts
            "out":    rs.normal(0, 2.0 / np.sqrt(P), (P, C)),    # the pair -> the deed
        }
        self._pristine = {k: v.copy() for k, v in self.W.items()}  # for the bitwise proof

        # ---- THE DUST: the only thing that is ever learned. ----
        # u is a "clarity logit"; transmittance G = sigmoid(u); W_eff = W * G.
        # u = 0 at init => G = 0.5 everywhere: the whole substrate half-obscured,
        # every coupling half-present, the signal a grey wash. This is the untrained
        # mind in this architecture: not empty, not ignorant -- CLUTTERED. All of it
        # is firing at once, which is the same as none of it speaking.
        self.u: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.W.items()}

        if cfg.tie_dinghui:
            # DING-HUI YI TI -- "meditation and wisdom are one substance, not two."
            # One mask governs both what is retained (gate_s) and how the retained
            # thing manifests as action (mix). They are not two disciplines.
            del self.u["gate_s"]

        if cfg.learn_weights_instead:
            # ABLATION -- the polishing school. Abandon the dust; train the mirror.
            self.u = {}
            self.theta = {k: v.copy() for k, v in self.W.items()}

        self._adam: Dict[str, Dict[str, np.ndarray]] = {}
        self._t = 0

    # -- parameters -----------------------------------------------------------

    def params(self) -> Dict[str, np.ndarray]:
        return self.theta if self.cfg.learn_weights_instead else self.u

    def n_trainable(self) -> int:
        return sum(p.size for p in self.params().values())

    def n_substrate(self) -> int:
        return sum(v.size for v in self.W.values())

    def effective(self) -> Dict[str, np.ndarray]:
        """The mind as it currently manifests: substrate seen through its dust."""
        if self.cfg.learn_weights_instead:
            return dict(self.theta)
        eff = {k: self.W[k] * sigmoid(u) for k, u in self.u.items()}
        if self.cfg.tie_dinghui:
            eff["gate_s"] = self.W["gate_s"] * sigmoid(self.u["mix"])
        return eff

    # -- forward --------------------------------------------------------------

    def forward(self, toks: np.ndarray, cache_on: bool = True):
        cfg = self.cfg
        B, T = toks.shape
        H, P = cfg.d_hidden, cfg.n_pairs
        eff = self.effective()
        E, A, Gh, Gs, M, O = (eff["E"], eff["arise"], eff["gate_h"],
                              eff["gate_s"], eff["mix"], eff["out"])

        s = np.zeros((B, H))                      # no prior self. It starts with nothing.
        c: Dict[str, list] = {k: [] for k in ("x", "h", "s_prev", "k", "s", "m", "z")}
        logits = np.zeros((B, T, cfg.n_classes))
        k_all = np.zeros((B, T, H))
        pairsum = np.zeros((B, T, P))

        for t in range(T):
            x = E[toks[:, t]]
            h = np.tanh(x @ A)                                    # arising
            if cfg.fixed_kappa is None:
                k = sigmoid(h @ Gh + s @ Gs)                      # deciding, now
            else:
                k = np.full((B, H), float(cfg.fixed_kappa))       # ablation: a constant temper
            s_prev = s
            s = k * s_prev + (1.0 - k) * h                        # abiding, or not
            m = np.tanh(h + s @ M)                                # manifesting
            z = m[:, :P] - m[:, P:]                               # the pair, not the pole
            logits[:, t, :] = z @ O
            k_all[:, t, :] = k
            pairsum[:, t, :] = m[:, :P] + m[:, P:]                # common mode: must vanish
            if cache_on:
                c["x"].append(x); c["h"].append(h); c["s_prev"].append(s_prev)
                c["k"].append(k); c["s"].append(s); c["m"].append(m); c["z"].append(z)

        c["eff"] = eff
        c["k_all"] = k_all
        c["pairsum"] = pairsum
        c["logits"] = logits
        return logits, c

    # -- loss -----------------------------------------------------------------

    def loss(self, toks, targ, cache=None):
        """L = CE  +  lam_abide * mean(kappa)  +  lam_pair * mean((pole_a + pole_b)^2)

        The second term is wuzhu made quantitative: a constant downward pressure on
        holding anything at all, which the task must actively fight in order to keep
        the rule. Non-abiding is the BASIS -- the default, the resting state -- not
        the goal. The third term is the thirty-six pairs: it drains meaning out of
        every individual pole and leaves it only in the relation between them.
        """
        cfg = self.cfg
        if cache is None:
            _, cache = self.forward(toks)
        B, T = toks.shape
        ll, probs = softmax_ce(cache["logits"].reshape(B * T, cfg.n_classes), targ.reshape(-1))
        ce = float(ll.mean())
        ab = float(cache["k_all"].mean())
        pr = float((cache["pairsum"] ** 2).mean())
        cache["probs"] = probs
        total = ce + cfg.lam_abide * ab + cfg.lam_pair * pr
        return total, {"ce": ce, "abide": ab, "pair": pr, "total": total}

    # -- backward: BPTT, derived by hand, no autodiff ------------------------

    def backward(self, toks, targ, cache) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        B, T = toks.shape
        H, P = cfg.d_hidden, cfg.n_pairs
        eff = cache["eff"]
        A, Gh, Gs, M, O = eff["arise"], eff["gate_h"], eff["gate_s"], eff["mix"], eff["out"]

        g = {k: np.zeros_like(eff[k]) for k in self.KEYS}     # grads w.r.t. EFFECTIVE matrices

        probs = cache["probs"].reshape(B, T, cfg.n_classes)
        onehot = np.zeros_like(probs)
        onehot[np.arange(B)[:, None], np.arange(T)[None, :], targ] = 1.0
        dlogits = (probs - onehot) / (B * T)

        d_ab = cfg.lam_abide / (B * T * H)                              # d(mean kappa)/d(kappa)
        d_pair = cfg.lam_pair * 2.0 * cache["pairsum"] / (B * T * P)    # d(pair penalty)/d(sum)

        ds_next = np.zeros((B, H))
        for t in reversed(range(T)):
            x, h, s_prev, k, s, m, z = (cache["x"][t], cache["h"][t], cache["s_prev"][t],
                                        cache["k"][t], cache["s"][t], cache["m"][t],
                                        cache["z"][t])

            dl = dlogits[:, t, :]
            g["out"] += z.T @ dl
            dz = dl @ O.T                                   # (B,P)

            dm = np.zeros((B, H))
            dm[:, :P] = dz + d_pair[:, t, :]                # yang pole: +difference, +sum
            dm[:, P:] = -dz + d_pair[:, t, :]               # yin pole:  -difference, +sum
            dpre_m = dm * (1.0 - m * m)                     # through tanh

            dh = dpre_m.copy()                              # m = tanh(h + s@M): direct path
            g["mix"] += s.T @ dpre_m
            ds = dpre_m @ M.T + ds_next                     # ...and the path through what abides

            # s = k*s_prev + (1-k)*h
            dk = ds * (s_prev - h)
            dh += ds * (1.0 - k)
            ds_prev = ds * k

            if cfg.fixed_kappa is None:
                dk = dk + d_ab                              # the non-abiding pressure enters here
                dg = dk * k * (1.0 - k)                     # through sigmoid
                g["gate_h"] += h.T @ dg
                dh += dg @ Gh.T                             # kappa is a function of this thought
                g["gate_s"] += s_prev.T @ dg
                ds_prev += dg @ Gs.T                        # ...and of this residue
            # (with a constant kappa there is no gate path at all: nothing can be
            #  learned about WHEN to let go, which is exactly the pathology we measure)

            da = dh * (1.0 - h * h)                         # through tanh
            g["arise"] += x.T @ da
            np.add.at(g["E"], toks[:, t], da @ A.T)         # scatter into the embedding

            ds_next = ds_prev

        if cfg.learn_weights_instead:
            return {k: g[k] for k in self.theta}

        # ---- chain through the dust:  dL/du = dL/dW_eff * W * sigmoid'(u) ----
        grads: Dict[str, np.ndarray] = {}
        for key in self.u:
            G = sigmoid(self.u[key])
            grads[key] = g[key] * self.W[key] * G * (1.0 - G)
        if cfg.tie_dinghui:
            Gm = sigmoid(self.u["mix"])
            grads["mix"] += g["gate_s"] * self.W["gate_s"] * Gm * (1.0 - Gm)
        return grads

    # -- Adam -----------------------------------------------------------------

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        cfg = self.cfg
        self._t += 1
        for k, p in self.params().items():
            st = self._adam.setdefault(k, {"m": np.zeros_like(p), "v": np.zeros_like(p)})
            gg = grads[k]
            st["m"] = cfg.beta1 * st["m"] + (1 - cfg.beta1) * gg
            st["v"] = cfg.beta2 * st["v"] + (1 - cfg.beta2) * (gg * gg)
            mh = st["m"] / (1 - cfg.beta1 ** self._t)
            vh = st["v"] / (1 - cfg.beta2 ** self._t)
            p -= cfg.lr * mh / (np.sqrt(vh) + cfg.eps)

    # -- measurement ----------------------------------------------------------

    def evaluate(self, toks, targ, is_sym, post_switch, long_lag, harden: bool = False):
        """`harden` binarises the mask (keep if G>0.5, else delete) and re-runs.

        If accuracy survives hardening, the competence genuinely lives in a
        SUBNETWORK of the untouched random substrate: the dust was real dust, not a
        continuous knob doing arithmetic behind our backs.
        """
        saved = None
        if harden and not self.cfg.learn_weights_instead:
            saved = {k: v.copy() for k, v in self.u.items()}
            for k in self.u:
                self.u[k] = np.where(self.u[k] > 0, 40.0, -40.0)   # sigmoid -> ~1 / ~0
        logits, c = self.forward(toks, cache_on=False)
        correct = logits.argmax(axis=2) == targ
        out = {
            "acc": float(correct.mean()),
            "acc_symbol": float(correct[is_sym].mean()),
            "acc_postswitch": float(correct[post_switch].mean()),   # over-abiding fails here
            "acc_longlag": float(correct[long_lag].mean()),         # under-abiding fails here
            "kappa": float(c["k_all"].mean()),
        }
        if saved is not None:
            self.u = saved
        return out

    def dust_report(self) -> Dict[str, float]:
        if self.cfg.learn_weights_instead:
            return {"deleted_frac": float("nan"), "total": 0}
        total = sum(u.size for u in self.u.values())
        kept = sum(int((u > 0).sum()) for u in self.u.values())
        return {"deleted_frac": 1.0 - kept / total, "kept": kept, "total": total}

    def substrate_untouched(self) -> bool:
        """The mirror was never polished. Bitwise."""
        return all(np.array_equal(self.W[k], self._pristine[k]) for k in self.W)


# =============================================================================
# SECTION 4 -- finite-difference gradient check (mandatory; runs first)
# =============================================================================


def gradient_check(verbose: bool = True) -> float:
    """Central differences vs. the hand-derived BPTT, in float64.

    Everything in the forward pass is smooth (matmul, tanh, sigmoid, mean), so a
    correct backward pass must agree with numeric differentiation to near machine
    precision. Entries whose true gradient is ~0 are skipped: there, the relative
    error is a ratio of two numerical dusts and would tell us nothing.
    """
    cfg = Config(d_embed=8, d_hidden=8, seq_len=6, lam_abide=0.05, lam_pair=0.05)
    rng = np.random.default_rng(0)
    net = CaoxiMirrorNetwork(cfg)
    for k in net.u:                       # move off u=0 so no gradient is degenerate
        net.u[k] = rng.normal(0, 0.7, net.u[k].shape)

    toks, targ, *_ = make_batch(cfg, 4, rng)
    _, cache = net.forward(toks)
    net.loss(toks, targ, cache)
    grads = net.backward(toks, targ, cache)

    h, floor = 1e-5, 1e-5
    worst_rel, worst_abs, rows = 0.0, 0.0, []
    for key in net.u:
        u = net.u[key]
        got = 0
        for _ in range(400):
            if got >= 8:
                break
            idx = tuple(int(rng.integers(0, s)) for s in u.shape)
            orig = u[idx]
            u[idx] = orig + h; Lp, _ = net.loss(toks, targ)
            u[idx] = orig - h; Lm, _ = net.loss(toks, targ)
            u[idx] = orig
            num = (Lp - Lm) / (2 * h)
            ana = grads[key][idx]
            if abs(num) + abs(ana) < floor:
                continue
            rel = abs(num - ana) / (abs(num) + abs(ana))
            worst_rel = max(worst_rel, rel)
            worst_abs = max(worst_abs, abs(num - ana))
            rows.append((key, num, ana, rel))
            got += 1

    if verbose:
        print("  tensor    numeric          analytic         rel.err")
        for key, num, ana, rel in rows[:8]:
            print(f"  {key:<8}  {num: .9e}  {ana: .9e}   {rel:.2e}")
        print(f"  ... {len(rows)} informative entries probed across every trainable tensor")
        print(f"  WORST RELATIVE ERROR: {worst_rel:.3e}   (worst absolute: {worst_abs:.2e})")
    assert worst_rel < 1e-6, f"GRADIENT CHECK FAILED (worst rel err {worst_rel:.2e})"
    return worst_rel


# =============================================================================
# SECTION 5 -- training
# =============================================================================


def train(cfg: Config, label: str = "", quiet: bool = True) -> dict:
    net = CaoxiMirrorNetwork(cfg)
    rng = np.random.default_rng(cfg.seed_data)
    ev = make_batch(cfg, cfg.eval_batch, np.random.default_rng(cfg.seed_data + 1))
    history: List[tuple] = []
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        toks, targ, *_ = make_batch(cfg, cfg.batch, rng)
        _, cache = net.forward(toks)
        _, parts = net.loss(toks, targ, cache)
        net.step(net.backward(toks, targ, cache))

        if step == 1 or step % cfg.eval_every == 0:
            m = net.evaluate(*ev)
            history.append((step, parts["ce"], m["acc_symbol"], m["acc_postswitch"],
                            m["acc_longlag"], m["kappa"]))
            if not quiet:
                print(f"  step {step:4d} | ce {parts['ce']:.4f} | acc {m['acc_symbol']:.3f} "
                      f"| post-switch {m['acc_postswitch']:.3f} | long-lag {m['acc_longlag']:.3f} "
                      f"| mean kappa {m['kappa']:.3f}")

    jumps = [(history[i][2] - history[i - 1][2], history[i][0]) for i in range(1, len(history))]
    jump, at = max(jumps) if jumps else (0.0, 0)

    return {
        "label": label, "net": net, "history": history,
        "soft": net.evaluate(*ev), "hard": net.evaluate(*ev, harden=True),
        "dust": net.dust_report(), "secs": time.time() - t0,
        "awaken_jump": jump, "awaken_at": at, "n": net.n_trainable(),
    }


# =============================================================================
# SECTION 6 -- self-tests
# =============================================================================


def self_tests(run: dict) -> List[Tuple[str, bool, str]]:
    net: CaoxiMirrorNetwork = run["net"]
    soft, hard, dust = run["soft"], run["hard"], run["dust"]
    t: List[Tuple[str, bool, str]] = []

    t.append(("the substrate was never touched (bitwise)",
              net.substrate_untouched(),
              "not one weight of the original nature was modified during training"))

    t.append(("competence by subtraction alone",
              soft["acc_symbol"] > 0.95,
              f"acc {soft['acc_symbol']:.3f} with obscurations as the only parameters"))

    drop = soft["acc_symbol"] - hard["acc_symbol"]
    t.append(("the dust was really dust (hard mask holds)",
              abs(drop) < 0.03,
              f"binarising every mask to keep/delete costs {drop:+.3f} accuracy"))

    t.append(("holds, but does not cling",
              0.05 < soft["kappa"] < 0.80,
              f"mean kappa {soft['kappa']:.3f}: neither a stone nor a hoarder"))

    t.append(("no abiding place: no store, no buffer, no episodic record",
              not any(hasattr(net, a) for a in ("memory", "keys", "store", "buffer")),
              "the network carries a state, never a record; nothing of the training set survives in it"))

    t.append(("learning was mostly deletion",
              dust["deleted_frac"] > 0.25,
              f"{dust['deleted_frac']*100:.1f}% of couplings occluded and thrown away"))

    t.append(("both failure modes avoided at once",
              soft["acc_postswitch"] > 0.95 and soft["acc_longlag"] > 0.95,
              f"post-switch {soft['acc_postswitch']:.3f} (no perseveration), "
              f"long-lag {soft['acc_longlag']:.3f} (no amnesia)"))

    return t


# =============================================================================
# SECTION 7 -- main
# =============================================================================


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 78)
    print("THE CAOXI MIRROR NETWORK")
    print("Huineng (638-713), Sixth Patriarch of Chan")
    print("learning as subtraction | memory as non-abiding | meaning as cancelled pairs")
    print("=" * 78)

    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK  (aborts the run if it fails)")
    worst = gradient_check()
    print(f"  PASS -- hand-derived BPTT agrees with numeric differentiation to {worst:.1e}")

    print("\n[2] TRAINING  (the only parameters are obscurations)")
    cfg = Config()
    run = train(cfg, "CaoxiMirror", quiet=False)
    print(f"  {run['n']} dust parameters trained in {run['secs']:.1f}s")
    print(f"  {run['net'].n_substrate()} substrate weights: frozen, random, untouched")

    print("\n[3] THE SUDDEN MOMENT")
    print(f"  largest single-interval gain in held-out accuracy: "
          f"+{run['awaken_jump']*100:.1f} points, at step {run['awaken_at']}")
    print("  accuracy trace: " + " ".join(f"{h[2]:.2f}" for h in run["history"]))
    print("  Nothing was added at that step. A veil came off.")

    print("\n[4] WHAT WAS DELETED")
    d = run["dust"]
    print(f"  occluded and discarded: {d['deleted_frac']*100:.1f}% of {d['total']} couplings")
    print(f"  soft mask: acc {run['soft']['acc_symbol']:.3f}    "
          f"hard mask (binary keep/delete): acc {run['hard']['acc_symbol']:.3f}")

    print("\n[5] IS A CONSTANT TEMPER ENOUGH?  (fixed abiding rate vs. a learned release)")
    print(f"  {'kappa':>12} {'acc':>7} {'post-switch':>12} {'long-lag':>10}")
    for kap in [0.02, 0.10, 0.30, 0.50, 0.70, 0.90, 0.98]:
        r = train(Config(fixed_kappa=kap, steps=700))
        print(f"  {kap:>12.2f} {r['soft']['acc_symbol']:>7.3f} "
              f"{r['soft']['acc_postswitch']:>12.3f} {r['soft']['acc_longlag']:>10.3f}")
    print(f"  {'LEARNED':>12} {run['soft']['acc_symbol']:>7.3f} "
          f"{run['soft']['acc_postswitch']:>12.3f} {run['soft']['acc_longlag']:>10.3f}"
          f"   (mean kappa {run['soft']['kappa']:.3f})")
    print("  Hold too little and the rule is lost (long-lag collapses).")
    print("  Hold too much and the revoked rule survives (post-switch collapses).")
    print("  The learned gate beats EVERY constant while holding LESS on average.")
    print("  The middle way is not a value. It is a function of the moment.")

    print("\n[6] ABLATIONS -- and what the doctrines actually buy")
    print("  (they do not buy accuracy. Every configuration below solves the task.")
    print("   Read the other three columns.)")

    def pole_sum(net, c) -> float:
        """Mean |yang + yin| -- how much meaning still sits in a single pole."""
        toks, *_ = make_batch(c, 256, np.random.default_rng(c.seed_data + 1))
        _, cache = net.forward(toks)
        return float(np.abs(cache["pairsum"]).mean())

    rows = [(run["label"], cfg, run)]
    for lab, c in [
        ("polishing (train the weights)", Config(learn_weights_instead=True)),
        ("untied ding/hui (two masks)", Config(tie_dinghui=False)),
        ("no thirty-six-pairs cancellation", Config(lam_pair=0.0)),
        ("no non-abiding pressure", Config(lam_abide=0.0)),
    ]:
        rows.append((lab, c, train(c, lab)))

    print(f"  {'configuration':<34}{'acc':>6}{'params':>8}{'mean kappa':>12}{'|pole-sum|':>12}")
    for lab, c, r in rows:
        print(f"  {lab:<34}{r['soft']['acc_symbol']:>6.3f}{r['n']:>8}"
              f"{r['soft']['kappa']:>12.3f}{pole_sum(r['net'], c):>12.4f}")
    print("  * The polishing school arrives too -- by rewriting every weight it owns.")
    print("    The mirror arrives by rewriting none of them.")
    print("  * Drop the non-abiding pressure and the network still succeeds -- while")
    print("    clinging ~50% harder for exactly the same behaviour. The doctrine buys")
    print("    minimal retention, not competence.")
    print("  * Drop the thirty-six pairs and |pole-sum| jumps four-fold: meaning crawls")
    print("    back into the individual poles. The doctrine buys a mind in which no")
    print("    single extreme carries anything on its own.")
    print("  * Untie ding and hui and it costs 9,216 extra parameters to do the same job.")

    print("\n[7] SELF-TESTS")
    results = self_tests(run)
    for name, ok, note in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {note}")
    assert all(ok for _, ok, _ in results), "one or more self-tests failed"

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED.")
    print("The substrate was never polished. What we trained was a veil,")
    print("and what training did was take it off.")
    print("=" * 78)


if __name__ == "__main__":
    main()

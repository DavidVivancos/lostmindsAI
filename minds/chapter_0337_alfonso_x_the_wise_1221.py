#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0337 · Alfonso X the Wise (1221-1284)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
THE CORDURA LEDGER — seso, ventura and the value of the move fortune allowed

Thesis
------
Every outcome is split into what wit chose (seso) and what fortune offered
(ventura), the reference for credit is the best move *available on the rolled
menu*, and the ledger refuses to learn from outcomes it attributes to dice.

Evidence and provenance (provenance: belief)
--------------------------------------------
D1  The prologue of the Libro de acedrex, dados e tablas (Seville, 1283; Escorial
    T.I.6, ff. 1v-2r) stages three sages before a king of India: one holds that
    seso (wit) is worth more than ventura (fortune), one the reverse, and the
    third that "cordura derecha" is to take from wit what profits, guard against
    fortune's harm and use fortune where it helps. Each proves his case by
    inventing a game: chess for seso, dice for ventura, tables for both.
D2  The dice must be equal cubes; otherwise the fall "would be deceit rather
    than fortune" (f. 65r): the ventura the ledger assumes is *fair*.
D3  Twelve dice games are given "from which all the others can be understood,
    those played in other lands, made or yet to be made" (f. 71v): a basis
    from which unseen games are read off.
D4  General Estoria I.16.14 (Solalinde 1930): the king "makes a book" by
    composing, emending, levelling and straightening what others write; the
    Ochava esfera prologue (1276) shows him removing superfluous and doubled
    reasons. The king is a *ledger*, not a scribe.
D5  Deeds (blind spot): for eighteen years (1257-1275) he pursued the imperial
    crown that papal politics, not his moves, would award, and read early good
    rolls (the 1257 election) as evidence of control (O'Callaghan 1993,
    Linehan 2008).

Doctrine -> mechanism -> test
-----------------------------
  D1 -> M1 mesa (menu register: menu-restricted Boltzmann choice over shared
        action values; reference = value fortune offered)      -> C6.1, H-NEC, H-RIVAL
  D1 -> M2 seso head (per-action value table conditioned on the state only,
        never on the menu)                                      -> C6.0 (definition check), H-SIG
  D2 -> M3 ventura ledger (state-conditioned noise scale; Gaussian NLL weights
        each outcome by 1/sigma^2, so dice teach little)         -> C6.3, H-VENT
  D3 -> task design: held-out and *withheld* menus never seen in training  -> H-SIG
  D5 -> loaded-dice condition (bonus tied to the menu, not the move)       -> H-BLIND

Research question
-----------------
Category: long-horizon credit assignment / robustness to spurious correlation.
Present-day form: outcome-supervised learners (RLVR, logged bandit data) credit
lucky trajectories; can a learner that separates the value a menu offered from
the value a choice added produce skill estimates that transfer when the
availability distribution shifts (SAS-MDP setting)?

Closest prior art and the delta
-------------------------------
Boutilier et al. 2018 (stochastic action sets), Mesnard et al. 2021
(counterfactual credit assignment), Nix & Weigend 1994 (heteroscedastic NLL),
Ziebart et al. 2008 (Boltzmann-rational choice). Delta: one shared,
menu-invariant value table is fitted jointly by (i) heteroscedastic outcome
likelihood and (ii) menu-restricted choice likelihood, and skill is scored as
regret against the best *available* value; the baseline is a size-matched
regressor that sees the menu as an input. Overlap: Medium.

Blind spot
----------
The ledger assumes fair dice: fortune enters only through availability. When
fortune's bonus rides on the menu itself (loaded dice, D5), the menu-invariant
table absorbs the bonus as skill and the menu-aware baseline should win.

Task (Tablas de Sevilla)
------------------------
State s in R^12 (9 informative, 3 nuisance); 6 moves. Hidden value
Q*(s,a) = tanh-features of s times a move matrix. A hidden regime of s sets
the game: chess (sigma 0.02), tables (sigma 0.4), dice (sigma 1.5 and all moves
equal). A menu of 2-6 moves is rolled; a player picks the best available with
probability 0.8, else at random. Outcome r = Q*(s,a) + sigma * eps. Splits: train
(6000) / val / held-out with fair menus; shifted split with menus of 2-3 in which
the globally best move is withheld with probability 0.85; blind-spot condition:
the same world and draws, but the die adds 0.6 whenever the chosen move is the
lowest-numbered available one (the bonus rides on the menu, not the move).
Target at evaluation: true regret z* = Q*(s,a) - max over the menu of Q*.

Limits
------
Synthetic data; Pearson correlation with regret as the skill metric; no claim
about the historical person beyond the cited loci; a research prototype, not an
AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "r2 (2026-09-25, after run 1): C3 loss-drop now measured on the training objective over a fixed "
        "1024-example training subset before vs after training (fraction 0.50) instead of the outcome MSE, "
        "whose drop is capped near 0.43 by irreducible dice noise; held-out margin unchanged (0.25).",
        "r2: C6.3 probe sets the ventura pre-activation to -1 / +2 exactly (was bias +-3), because the "
        "sigma floor saturated both arms in run 1 and made the property test vacuous.",
        "r2: training budget 700 -> 2000 steps, lr 3e-3 -> 1e-2, n_train 3000 -> 6000, w_choice 1 -> 2, "
        "chosen on the validation split; identical for model, baseline and rival.",
        "r2: the loaded-dice condition is now drawn from the SAME world and draws as the fair one "
        "(paired), and H-BLIND-SELF (model on loaded minus model on fair) is added; H-BLIND is kept as "
        "declared (it was contradicted in run 1: the menu-aware baseline degraded more, not less).",
    ],
    "generation": {"template_version": "codeguidelines-1.0", "generator": "Claude",
                   "generator_version": "Fable 5.1", "date": "2026-09-25"},
    "id": 337, "figure": "Alfonso X the Wise", "born": 1221, "died": 1284,
    "civilization": "Castilian (Crown of Castile and León)",
    "provenance": "belief",
    "thesis": "Credit for an outcome is the gap between the move chosen and the best move fortune "
              "allowed, and nothing is learned from the part of the outcome attributed to dice.",
    "evidence": [
        {"id": "D1", "claim": "Three sages: seso, ventura, cordura; chess, dice, tables as proofs",
         "basis": "primary", "source": "Libro de acedrex, dados e tablas, Escorial T.I.6, ff. 1v-2r"},
        {"id": "D2", "claim": "Unequal dice are deceit, not fortune", "basis": "primary",
         "source": "Libro de acedrex, dados e tablas, f. 65r"},
        {"id": "D3", "claim": "Twelve dice games from which unseen games are understood",
         "basis": "primary", "source": "Libro de acedrex, dados e tablas, f. 71v"},
        {"id": "D4", "claim": "The king makes a book by composing, emending, levelling, straightening",
         "basis": "primary", "source": "General Estoria I.16.14; Libro de la ochava esfera, prologue (1276)"},
        {"id": "D5", "claim": "Imperial pursuit 1257-1275 read early luck as control",
         "basis": "deeds", "source": "O'Callaghan 1993; Linehan 2008"},
    ],
    "research_question": {
        "category": "long-horizon credit assignment / robustness to spurious correlation",
        "question": "Do menu-relative skill estimates from a menu-invariant value table transfer "
                    "across shifts in action availability better than menu-conditioned regression?"},
    "mechanism": {
        "name": "Cordura Ledger", "family": "causal and counterfactual reasoning (structured credit)",
        "signature_modules": ["mesa", "ventura"],
        "closest_prior_art": ["Boutilier et al. 2018 SAS-MDP (arXiv:1805.02363)",
                              "Mesnard et al. 2021 CCA (arXiv:2011.09464)",
                              "Nix & Weigend 1994 heteroscedastic NLL",
                              "Ziebart et al. 2008 maximum-entropy IRL"],
        "overlap": "Medium",
        "prior_art_queries": ["stochastic action sets MDP", "counterfactual credit assignment skill luck",
                              "heteroscedastic regression aleatoric weighting",
                              "off-policy evaluation logged bandit action availability",
                              "outcome reward lucky trajectories RLVR credit assignment"],
        "contribution_type": "mechanism",
        "delta": "Joint heteroscedastic-outcome and menu-restricted-choice likelihood over one "
                 "menu-invariant value table, scored by regret against the best available move.",
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D1", "mechanism": "M2", "property_test": "C6.0", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-VENT"},
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "-", "hypothesis": "H-BLIND"},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "-", "hypothesis": "H-BLIND-SELF"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Menu-invariant credit transfers to withheld menus better than "
         "menu-conditioned regression", "metric": "skill_corr", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the menu register hurts more than removing a matched "
         "encoder block", "metric": "skill_corr", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "With loaded dice the ledger credits the bonus as skill",
         "condition": "loaded", "grounding": "read the 1257 election as control",
         "metric": "skill_corr", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND-SELF", "statement": "Loaded dice degrade the ledger's own skill estimate on the "
         "same world with fair dice", "condition": "loaded", "grounding": "read the 1257 election as control",
         "metric": "skill_corr", "comparison": "model_loaded - model_fair", "direction": "less",
         "mesi": 0.02, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Cordura beats the first sage (seso alone: no menu, no ventura)",
         "metric": "skill_corr", "split": "shifted",
         "comparison": "model - rival", "direction": "greater", "mesi": 0.05, "seeds": 5},
        {"id": "H-VENT", "statement": "The ventura ledger recovers the hidden noise regime",
         "metric": "corr(log sigma_hat, log sigma*)", "split": "held_out",
         "comparison": "model - 0.5", "direction": "greater", "mesi": 0.1, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.50, "margin_over_trivial": 0.25, "shuffled_band": 0.08},
    "probe_predictions": [
        {"probe": "P2", "expected": "above baseline"}, {"probe": "P6", "expected": "above baseline"},
        {"probe": "P8", "expected": "above baseline"}, {"probe": "P9", "expected": "equal to baseline"},
        {"probe": "P1", "expected": "below baseline"}],
    "dialectic_links": [
        {"chapter": 272, "relation": "successor", "test": "not tested (Toledan tables re-based to 1252)"},
        {"chapter": 0, "relation": "rival", "test": "H-RIVAL (the prologue's first sage, seso alone)"}],
    "corpus_neighbors": [
        {"chapter": 327, "similarity": 0.09, "difference": "Tusi designs around the unwanted consequence; "
         "here the unit is the menu fortune rolled, not the consequence of an action"},
        {"chapter": 328, "similarity": 0.07, "difference": "Qin holds a quantity as remainders; here the "
         "outcome is split into offered value and added value"},
        {"chapter": 334, "similarity": 0.06, "difference": "Sundiata's capability is licensed by others; "
         "here availability is exogenous chance, never a social grant"},
        {"chapter": 272, "similarity": 0.08, "difference": "al-Zarqali carries constants across records; "
         "here the carried object is a menu-invariant value table"},
        {"chapter": 317, "similarity": 0.10, "difference": "Fibonacci abstracts rules from worked cases; "
         "here the abstraction is a credit rule, and the baseline is the case regressor"}],
    "barometer": {
        "cognitive_processing": ["P2 relational transfer to withheld menus", "P6 few-shot"],
        "embodied_cognition": [], "world_modeling": ["regime recovery by the ventura head"],
        "consciousness": ["P8 calibration of sigma_hat vs sigma*"], "language_understanding": [],
        "emotional_intelligence": [], "creativity": [],
        "autonomy": ["menu-relative self-rating separates 'played well' from 'rolled well'"]},
    "task_types": ["vector_regression", "vector_classification"],
    "applications": [
        {"use": "off-policy evaluation of recommendation policies under item availability",
         "sector": "e-commerce", "dataset": "Open Bandit Dataset (ZOZO, 2020)"},
        {"use": "rating clinical decisions relative to the options actually available (research only)",
         "sector": "health", "dataset": "MIMIC-IV (de-identified)"},
        {"use": "skill-versus-luck rating of shot selection", "sector": "sports analytics",
         "dataset": "Kaggle NBA shot logs 2014-15"}],
    "safety_notes": "Decision-support research only; synthetic data; no treatment recommendations; "
                    "no claim of replicating a person.",
}

import argparse
import json
import math
import os
import sys
import time

import numpy as np

STATE_DIM, N_INFORMATIVE, N_ACTIONS, HID_TRUE = 12, 9, 6, 16
HIDDEN = 48
CLIP_NORM = 5.0
BUDGET_S, BUDGET_QUICK_S = 180.0, 20.0
STEPS, STEPS_QUICK, LOADED_BETA = 2000, 300, 0.6
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BUDGET, EXIT_NONFINITE = 0, 1, 2, 3, 4


# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        s = max_norm / (total + 1e-12)
        return {k: g * s for k, g in grads.items()}, total
    return grads, total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, tol=1e-5):
    """Central differences on every tensor: n random entries plus the largest-gradient entry."""
    worst = 0.0
    checked = 0
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            # floor of 1e-4 keeps the rounding noise of central differences (~1e-10)
            # below the tolerance for near-zero entries; a wrong entry still fails
            rel = abs(num - g[i]) / max(1e-4, abs(num) + abs(g[i]))
            worst = max(worst, rel)
        checked += 1
    return worst, checked, worst <= tol


def paired_bootstrap(diffs, n_boot=2000, seed=0):
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(d, size=d.size, replace=True)) for _ in range(n_boot)])
    return float(np.mean(d)), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and -mean >= mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
class TablasWorld:
    """Hidden truth of the Tablas de Sevilla task: values, regimes, menus, players, dice."""

    def __init__(self, rng, loaded=0.0):
        self.Wphi = rng.normal(0.0, 1.0 / math.sqrt(N_INFORMATIVE), (N_INFORMATIVE, HID_TRUE))
        self.V = rng.normal(0.0, 1.0, (HID_TRUE, N_ACTIONS))
        self.wr = rng.normal(0.0, 1.0, N_INFORMATIVE) / math.sqrt(N_INFORMATIVE)
        self.loaded = loaded
        probe = rng.normal(0.0, 1.0, (4000, STATE_DIM))
        self.scale = 0.8 / (np.std(self._raw_q(probe)) + 1e-9)

    def _raw_q(self, S):
        return np.tanh(S[:, :N_INFORMATIVE] @ self.Wphi) @ self.V

    def regime(self, S):
        rho = np.tanh(S[:, :N_INFORMATIVE] @ self.wr * 1.6)
        reg = np.ones(S.shape[0], dtype=int)          # 1 = tables
        reg[rho > 0.55] = 2                            # 2 = dice
        reg[rho < -0.55] = 0                           # 0 = chess
        return reg

    def qstar(self, S):
        Q = self._raw_q(S) * self.scale
        reg = self.regime(S)
        dice = reg == 2
        Q[dice] = Q[dice].mean(axis=1, keepdims=True)
        return Q, reg

    def sample(self, n, rng, withhold=False, skilled=0.8):
        S = rng.normal(0.0, 1.0, (n, STATE_DIM))
        Q, reg = self.qstar(S)
        sigma = np.where(reg == 2, 1.5, np.where(reg == 0, 0.02, 0.4))
        M = np.zeros((n, N_ACTIONS))
        best = np.argmax(Q, axis=1)
        for i in range(n):
            k = int(rng.integers(2, 4)) if withhold else int(rng.integers(2, N_ACTIONS + 1))
            pool = np.arange(N_ACTIONS)
            if withhold and rng.random() < 0.85:
                pool = pool[pool != best[i]]
            M[i, rng.choice(pool, size=k, replace=False)] = 1.0
        avail = np.where(M > 0, Q, -np.inf)
        best_avail = np.argmax(avail + 0.3 * rng.normal(0.0, 1.0, avail.shape) * (M > 0), axis=1)
        random_pick = np.array([rng.choice(np.flatnonzero(M[i])) for i in range(n)])
        use_skill = rng.random(n) < skilled
        a = np.where(use_skill, best_avail, random_pick)
        r = Q[np.arange(n), a] + sigma * rng.normal(0.0, 1.0, n)
        first_avail = np.argmax(M, axis=1)
        r = r + self.loaded * (a == first_avail)
        zstar = Q[np.arange(n), a] - np.max(avail, axis=1)
        return {"X": S, "menu": M, "action": a, "outcome": r, "zstar": zstar,
                "regime": reg, "sigma": sigma, "qstar": Q}


def make_task(seed, loaded=0.0, n_train=6000, n_val=600, n_eval=1500):
    ss = np.random.SeedSequence(seed)
    r_world, r_tr, r_va, r_id, r_sh = [np.random.default_rng(s) for s in ss.spawn(5)]
    world = TablasWorld(r_world, loaded=loaded)
    return {"train": world.sample(n_train, r_tr), "val": world.sample(n_val, r_va),
            "held_out": world.sample(n_eval, r_id), "shifted": world.sample(n_eval, r_sh, withhold=True),
            "world": world}


def subset(batch, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


def shuffled_control(batch, rng):
    out = dict(batch)
    out["outcome"] = batch["outcome"][rng.permutation(batch["outcome"].size)]
    out["action"] = np.array([rng.choice(np.flatnonzero(m)) for m in batch["menu"]])
    return out


# ----------------------------------------------------------------------------- model
TASK_TYPES = ["vector_regression", "vector_classification"]


class CorduraLedger:
    """Menu-invariant value table + fair-ventura scale + menu-restricted Boltzmann choice."""

    def __init__(self, in_dim, out_dim, task_type, rng, hidden=HIDDEN, w_choice=2.0):
        self.in_dim, self.n_actions, self.task_type, self.hidden = in_dim, out_dim, task_type, hidden
        self.w_choice = w_choice
        s1, s2, sq = 1.0 / math.sqrt(in_dim), 1.0 / math.sqrt(hidden), 1.0 / math.sqrt(hidden)
        self.params = {
            "W1": rng.normal(0, s1, (in_dim, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, s2, (hidden, hidden)), "b2": np.zeros(hidden),
            "Wq": rng.normal(0, sq, (hidden, out_dim)), "bq": np.zeros(out_dim),
            "wv": rng.normal(0, sq, (hidden,)), "bv": np.zeros(1),
            "tau_raw": np.array([0.0]),
        }
        self.ko = {}
        self.sigma_min = 0.05

    def temperature(self):
        return float(softplus(self.params["tau_raw"])[0] + 0.02)

    def forward(self, X, menu):
        P = self.params
        h1 = np.tanh(X @ P["W1"] + P["b1"])
        if self.ko.get("encoder_block1") == "zero":
            h1 = np.zeros_like(h1)
        u = np.tanh(h1 @ P["W2"] + P["b2"])
        if self.ko.get("encoder_block2") in ("identity", "zero"):
            u = np.zeros_like(u)
        h2 = h1 + u
        Q = h2 @ P["Wq"] + P["bq"]
        if self.ko.get("seso") == "zero":
            Q = np.zeros_like(Q)
        elif self.ko.get("seso") == "mean":
            Q = np.repeat(Q.mean(axis=1, keepdims=True), Q.shape[1], axis=1)
        v = h2 @ P["wv"] + P["bv"][0]
        sig_v = sigmoid(v)
        sigma = softplus(v) + self.sigma_min
        kv = self.ko.get("ventura")
        if kv == "zero" or kv == "identity":
            sigma, sig_v = np.ones_like(sigma), np.zeros_like(sig_v)
        elif kv == "mean":
            sigma, sig_v = np.full_like(sigma, float(np.mean(sigma))), np.zeros_like(sig_v)
        mask = menu
        km = self.ko.get("mesa")
        if km == "identity":
            mask = np.ones_like(menu)
        tau = self.temperature()
        logits = np.where(mask > 0, Q / tau, -np.inf)
        if km == "mean":
            logits = np.where(mask > 0, 0.0, -np.inf)
        p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        ref_soft = np.sum(p * Q, axis=1)
        ref = np.zeros_like(ref_soft) if km == "zero" else ref_soft
        return {"h1": h1, "u": u, "h2": h2, "Q": Q, "v": v, "sig_v": sig_v, "sigma": sigma,
                "p": p, "ref": ref, "ref_soft": ref_soft, "tau": tau, "mask": mask}

    def hard_reference(self, c):
        """Best available value (the tau -> 0 limit of the menu register), honouring knockouts."""
        km = self.ko.get("mesa")
        if km == "zero":
            return np.zeros(c["Q"].shape[0])
        if km == "mean":
            return np.sum(c["mask"] * c["Q"], axis=1) / np.sum(c["mask"], axis=1)
        return np.max(np.where(c["mask"] > 0, c["Q"], -np.inf), axis=1)


def _cordura_loss_and_grads(model, batch):
    X, menu, a = batch["X"], batch["menu"], batch["action"]
    r = batch.get("outcome")
    if model.ko.get("seso") or model.ko.get("encoder_block1"):
        raise ValueError("seso and encoder_block1 knockouts are inference-only")
    B = X.shape[0]
    P = model.params
    c = model.forward(X, menu)
    Q, sigma, p, tau = c["Q"], c["sigma"], c["p"], c["tau"]
    ar = np.arange(B)
    onehot = np.zeros_like(Q)
    onehot[ar, a] = 1.0
    dQ = np.zeros_like(Q)
    dv = np.zeros(B)
    loss = 0.0
    if r is not None:
        e = (r - Q[ar, a]) / sigma
        loss += float(np.mean(0.5 * e * e + np.log(sigma)))
        dQ[ar, a] += -e / sigma
        dv += (1.0 - e * e) / sigma * c["sig_v"]
    dtau = 0.0
    if model.w_choice > 0 and model.n_actions > 1:
        logp = np.log(p[ar, a] + 1e-300)
        loss += model.w_choice * float(np.mean(-logp))
        dQ += model.w_choice * (p - onehot) / tau
        if model.ko.get("mesa") != "mean":
            dtau = model.w_choice * float(np.mean((Q[ar, a] - c["ref_soft"]) / (tau * tau)))
    dQ /= B
    dv /= B
    h2, h1, u = c["h2"], c["h1"], c["u"]
    grads = {"Wq": h2.T @ dQ, "bq": dQ.sum(0), "wv": h2.T @ dv, "bv": np.array([dv.sum()])}
    dh2 = dQ @ P["Wq"].T + np.outer(dv, P["wv"])
    if model.ko.get("encoder_block2") in ("identity", "zero"):
        dz2 = np.zeros_like(u)
    else:
        dz2 = dh2 * (1.0 - u * u)
    grads["W2"], grads["b2"] = h1.T @ dz2, dz2.sum(0)
    dh1 = dh2 + dz2 @ P["W2"].T
    dz1 = dh1 * (1.0 - h1 * h1)
    grads["W1"], grads["b1"] = X.T @ dz1, dz1.sum(0)
    grads["tau_raw"] = np.array([dtau * float(sigmoid(P["tau_raw"])[0])])
    return loss, grads


def skill_scores(model, batch):
    """Regret of the chosen move against the best available value (the menu register's hard limit)."""
    ar = np.arange(batch["X"].shape[0])
    if isinstance(model, CorduraLedger):
        c = model.forward(batch["X"], batch["menu"])
        return c["Q"][ar, batch["action"]] - model.hard_reference(c)
    Qa = action_values(model, batch)
    return Qa[ar, batch["action"]] - np.max(np.where(batch["menu"] > 0, Qa, -np.inf), axis=1)


def action_values(model, batch):
    if isinstance(model, CorduraLedger):
        mask = batch["menu"]
        return model.forward(batch["X"], mask)["Q"]
    return model.values_for_all_actions(batch["X"], batch["menu"])


# ----------------------------------------------------------------------------- baseline and rival
class ContextMLP:
    """Chess-player baseline: regress the outcome on [state, menu, one-hot move] with MSE."""

    def __init__(self, in_dim, out_dim, rng, hidden=HIDDEN):
        self.in_dim, self.n_actions, self.hidden = in_dim, out_dim, hidden
        d = in_dim + 2 * out_dim
        self.params = {
            "W1": rng.normal(0, 1 / math.sqrt(d), (d, hidden)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / math.sqrt(hidden), (hidden, hidden)), "b2": np.zeros(hidden),
            "w": rng.normal(0, 1 / math.sqrt(hidden), (hidden,)), "b": np.zeros(1),
        }
        self.ko = {}
        self.task_type = "vector_regression"

    def inputs(self, X, menu, a):
        oh = np.zeros((X.shape[0], self.n_actions))
        oh[np.arange(X.shape[0]), a] = 1.0
        return np.concatenate([X, menu, oh], axis=1)

    def forward(self, Z):
        P = self.params
        h1 = np.tanh(Z @ P["W1"] + P["b1"])
        u = np.tanh(h1 @ P["W2"] + P["b2"])
        h2 = h1 + u
        return {"h1": h1, "u": u, "h2": h2, "y": h2 @ P["w"] + P["b"][0]}

    def values_for_all_actions(self, X, menu):
        B = X.shape[0]
        out = np.zeros((B, self.n_actions))
        for a in range(self.n_actions):
            out[:, a] = self.forward(self.inputs(X, menu, np.full(B, a)))["y"]
        return out


def _mlp_loss_and_grads(model, batch):
    Z = model.inputs(batch["X"], batch["menu"], batch["action"])
    r = batch["outcome"]
    P = model.params
    c = model.forward(Z)
    B = Z.shape[0]
    d = (c["y"] - r) / B
    loss = float(0.5 * np.mean((c["y"] - r) ** 2))
    grads = {"w": c["h2"].T @ d, "b": np.array([d.sum()])}
    dh2 = np.outer(d, P["w"])
    dz2 = dh2 * (1.0 - c["u"] ** 2)
    grads["W2"], grads["b2"] = c["h1"].T @ dz2, dz2.sum(0)
    dh1 = dh2 + dz2 @ P["W2"].T
    dz1 = dh1 * (1.0 - c["h1"] ** 2)
    grads["W1"], grads["b1"] = Z.T @ dz1, dz1.sum(0)
    return loss, grads


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task_type %s" % task_type)
    return CorduraLedger(in_dim, out_dim, task_type, rng, **cfg)


def build_baseline(in_dim, out_dim, rng):
    return ContextMLP(in_dim, out_dim, rng)


def build_rival(in_dim, out_dim, rng):
    """The first sage: seso alone. No menu in the reference, no ventura scale."""
    m = CorduraLedger(in_dim, out_dim, "vector_regression", rng)
    m.ko = {"mesa": "identity", "ventura": "zero"}
    return m


def loss_and_grads(model, batch):
    if isinstance(model, CorduraLedger):
        return _cordura_loss_and_grads(model, batch)
    return _mlp_loss_and_grads(model, batch)


def predict(model, X):
    menu = np.ones((X.shape[0], model.n_actions))
    Q = action_values(model, {"X": X, "menu": menu})
    if getattr(model, "task_type", "") == "vector_classification":
        return np.argmax(Q, axis=1)
    return Q[:, 0] if model.n_actions == 1 else Q


def hidden_states(model, X):
    if isinstance(model, CorduraLedger):
        c = model.forward(X, np.ones((X.shape[0], model.n_actions)))
        return {"encoder": c["h2"], "values": c["Q"], "sigma": c["sigma"]}
    Z = model.inputs(X, np.ones((X.shape[0], model.n_actions)), np.zeros(X.shape[0], dtype=int))
    return {"encoder": model.forward(Z)["h2"]}


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


# ----------------------------------------------------------------------------- registries
def modules(model):
    if isinstance(model, CorduraLedger):
        return {
            "encoder_block1": {"params": ["W1", "b1"], "role": "tanh feature layer", "signature": False},
            "encoder_block2": {"params": ["W2", "b2"], "role": "residual tanh block", "signature": False},
            "seso": {"params": ["Wq", "bq"], "role": "per-action value table conditioned on state only",
                     "signature": True},
            "mesa": {"params": ["tau_raw"], "role": "masked Boltzmann aggregation over available "
                     "actions (menu register)", "signature": True},
            "ventura": {"params": ["wv", "bv"], "role": "state-conditioned heteroscedastic scale head",
                        "signature": True},
        }
    return {"encoder_block1": {"params": ["W1", "b1"], "role": "tanh feature layer", "signature": False},
            "encoder_block2": {"params": ["W2", "b2"], "role": "residual tanh block", "signature": False},
            "readout": {"params": ["w", "b"], "role": "linear outcome readout", "signature": False}}


def knockout(model, name, mode):
    """Copy of the model with `name` replaced by identity / zero / mean output."""
    import copy
    if name not in modules(model):
        raise KeyError(name)
    m = copy.deepcopy(model)
    m.ko[name] = mode
    return m


MUTANTS = {
    "flip_sign": "update direction reversed",
    "zero_lr": "learning rate set to zero",
    "zero_grad_seso": "gradient of the seso weight Wq zeroed",
    "zero_grad_ventura": "gradient of the ventura weight wv zeroed",
}


def apply_mutant(name, grads, lr):
    if name == "flip_sign":
        return {k: -g for k, g in grads.items()}, lr
    if name == "zero_lr":
        return grads, 0.0
    if name == "zero_grad_seso" and "Wq" in grads:
        grads = dict(grads)
        grads["Wq"] = np.zeros_like(grads["Wq"])
    if name == "zero_grad_ventura" and "wv" in grads:
        grads = dict(grads)
        grads["wv"] = np.zeros_like(grads["wv"])
    return grads, lr


# ----------------------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=1e-2, batch_size=128, mutant=None, log=None):
    """Adam for `budget` updates; the checkpoint with the best validation loss is kept."""
    train, val = data["train"], data["val"]
    n = train["X"].shape[0]
    opt = Adam(model.params, lr=lr)
    probe = subset(train, np.arange(min(1024, n)))          # fixed training subset for C3
    loss_first = loss_and_grads(model, probe)[0]
    best_val, best_params = math.inf, None
    for step in range(budget):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        loss, grads = loss_and_grads(model, subset(train, idx))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if mutant:
            grads, opt.lr = apply_mutant(mutant, grads, opt.lr)
        grads, _ = clip_global_norm(grads, CLIP_NORM)
        opt.step(model.params, grads)
        if log is not None:
            log.append(loss)
        if step % 50 == 49 or step == budget - 1:
            vl, _ = loss_and_grads(model, val)
            if vl < best_val:
                best_val, best_params = vl, {k: v.copy() for k, v in model.params.items()}
    if best_params is not None:
        model.params = best_params
    return {"first_loss": float(loss_first), "last_loss": float(loss_and_grads(model, probe)[0]),
            "best_val": best_val}


def evaluate(model, batch):
    s = skill_scores(model, batch)
    z = batch["zstar"]
    corr = 0.0 if np.std(s) < 1e-12 else float(np.corrcoef(s, z)[0, 1])
    Qa = action_values(model, batch)
    avail_hat = np.where(batch["menu"] > 0, Qa, -np.inf)
    avail = np.where(batch["menu"] > 0, batch["qstar"], -np.inf)
    best_acc = float(np.mean(np.argmax(avail_hat, 1) == np.argmax(avail, 1)))
    mse = float(np.mean((Qa[np.arange(Qa.shape[0]), batch["action"]] - batch["outcome"]) ** 2))
    return {"skill_corr": corr, "best_move_acc": best_acc, "outcome_mse": mse}


def ventura_recovery(model, batch):
    sig_hat = model.forward(batch["X"], batch["menu"])["sigma"]
    return float(np.corrcoef(np.log(sig_hat), np.log(batch["sigma"]))[0, 1])


# ----------------------------------------------------------------------------- correctness tests
def gradcheck(model, batch, rng, tol=1e-5, mutant=None):
    _, grads = loss_and_grads(model, batch)
    if mutant:
        grads, _ = apply_mutant(mutant, grads, 1.0)
    worst, checked, ok = finite_difference_check(lambda: loss_and_grads(model, batch)[0],
                                                 model.params, grads, rng, tol=tol)
    return worst, checked, ok


def c1_gradient_check(task, rng, steps, mutant=None):
    """C1 on model, baseline and rival, at init and after training steps."""
    out, worst_all = [], 0.0
    small = subset(task["train"], np.arange(24))
    builders = {"model": lambda: build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng),
                "baseline": lambda: build_baseline(STATE_DIM, N_ACTIONS, rng),
                "rival": lambda: build_rival(STATE_DIM, N_ACTIONS, rng)}
    for name, make in builders.items():
        m = make()
        w0, n0, ok0 = gradcheck(m, small, rng, mutant=mutant)
        fit(m, task, steps, rng)
        w1, n1, ok1 = gradcheck(m, small, rng, mutant=mutant)
        worst_all = max(worst_all, w0, w1)
        out.append({"target": name, "tensors": n0, "max_rel_error": max(w0, w1), "passed": ok0 and ok1})
    return out, worst_all


def c2_determinism(task, steps):
    losses = []
    for _ in range(2):
        rng = np.random.default_rng(12345)
        m = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
        log = []
        fit(m, task, steps, rng, log=log)
        losses.append(np.array(log))
        finite = all(np.all(np.isfinite(v)) for v in m.params.values())
        h = hidden_states(m, task["held_out"]["X"][:64])
        finite = finite and all(np.all(np.isfinite(v)) for v in h.values())
    return bool(np.array_equal(losses[0], losses[1]) and finite)


def c3_learning(model, task, fit_info, thresholds):
    """Training objective (fixed 1024-example training subset) falls by the declared fraction;
    held-out skill beats the trivial constant score."""
    drop = (fit_info["first_loss"] - fit_info["last_loss"]) / max(abs(fit_info["first_loss"]), 1e-9)
    ev = evaluate(model, task["held_out"])
    trivial_corr = 0.0          # a constant skill score correlates 0 with true regret
    passed = drop >= thresholds["loss_drop_fraction"] and \
        ev["skill_corr"] - trivial_corr >= thresholds["margin_over_trivial"]
    return passed, {"loss_drop": drop, "held_out_skill_corr": ev["skill_corr"],
                    "trivial_skill_corr": trivial_corr, "first_loss": fit_info["first_loss"],
                    "last_loss": fit_info["last_loss"]}


def c4_shuffled_control(task, rng, steps, band):
    ctrl = dict(task)
    ctrl["train"] = shuffled_control(task["train"], rng)
    m = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
    fit(m, ctrl, steps, rng)
    corr = evaluate(m, task["held_out"])["skill_corr"]
    return abs(corr) <= band, corr


def c5_mutants(task, rng, steps, thresholds):
    detected, rows = 0, []
    small = subset(task["train"], np.arange(24))
    for name in MUTANTS:
        m = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
        _, grads = loss_and_grads(m, small)
        grads, _ = apply_mutant(name, grads, 1.0)
        worst, _, c1_ok = finite_difference_check(lambda: loss_and_grads(m, small)[0], m.params, grads, rng)
        info = fit(m, task, steps, rng, mutant=name)
        c3_ok, _ = c3_learning(m, task, info, thresholds)
        caught = (not c1_ok) or (not c3_ok)
        detected += int(caught)
        rows.append({"mutant": name, "c1_failed": not c1_ok, "c3_failed": not c3_ok, "detected": caught})
    return detected, rows


def _random_ledger(rng, scale):
    m = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
    for k in m.params:
        m.params[k] = rng.normal(0.0, scale, m.params[k].shape)
    return m


def c6_properties(rng, trials=200):
    """Property tests on random parameters and random inputs, each with a negative control."""
    import copy
    res = {}
    viol_ref, viol_ctrl_ref, viol_mono, viol_ctrl_mono, viol_vent, viol_ctrl_vent, viol_def = 0, 0, 0, 0, 0, 0, 0
    for _ in range(trials):
        scale = float(rng.choice([0.1, 1.0, 3.0]))
        m = _random_ledger(rng, scale)
        X = rng.normal(0, 1, (8, STATE_DIM))
        menu = (rng.random((8, N_ACTIONS)) < 0.5).astype(float)
        menu[np.arange(8), rng.integers(0, N_ACTIONS, 8)] = 1.0
        c = m.forward(X, menu)
        # C6.0 definition check (holds by construction, reported, never counted as evidence)
        c_alt = m.forward(X, np.ones_like(menu))
        viol_def += int(not np.allclose(c["Q"], c_alt["Q"]))
        # C6.1 reference lies within the available range
        lo = np.min(np.where(menu > 0, c["Q"], np.inf), 1)
        hi = np.max(np.where(menu > 0, c["Q"], -np.inf), 1)
        viol_ref += int(np.any(c["ref"] < lo - 1e-9) or np.any(c["ref"] > hi + 1e-9))
        k = knockout(m, "mesa", "identity").forward(X, menu)   # negative control: full-menu reference
        viol_ctrl_ref += int(np.any(k["ref"] < lo - 1e-9) or np.any(k["ref"] > hi + 1e-9))
        # C6.2 the reference is monotone non-increasing in the temperature
        m_cold, m_hot = copy.deepcopy(m), copy.deepcopy(m)
        m_cold.params["tau_raw"] = np.array([-2.0])
        m_hot.params["tau_raw"] = np.array([3.0])
        r_cold, r_hot = m_cold.forward(X, menu)["ref"], m_hot.forward(X, menu)["ref"]
        viol_mono += int(np.any(r_hot > r_cold + 1e-9))
        broken_cold, broken_hot = _neg_ref(m_cold, X, menu), _neg_ref(m_hot, X, menu)
        viol_ctrl_mono += int(np.any(broken_hot > broken_cold + 1e-9))
        # C6.3 larger predicted ventura -> smaller outcome gradient on the seso weights
        g_small, g_large = _seso_grad_norm(m, X, menu, rng, -1.0), _seso_grad_norm(m, X, menu, rng, 2.0)
        viol_vent += int(not g_large < g_small)
        mh = knockout(m, "ventura", "zero")
        h_small, h_large = _seso_grad_norm(mh, X, menu, rng, -1.0), _seso_grad_norm(mh, X, menu, rng, 2.0)
        viol_ctrl_vent += int(not h_large < h_small)
    res["C6.0_definition_menu_invariance"] = {"violations": viol_def, "counts_as_evidence": False}
    res["C6.1_reference_in_range"] = {"violations": viol_ref, "control_violations": viol_ctrl_ref,
                                      "passed": viol_ref == 0 and viol_ctrl_ref > 0}
    res["C6.2_reference_monotone_in_temperature"] = {"violations": viol_mono, "control_violations": viol_ctrl_mono,
                                                     "passed": viol_mono == 0 and viol_ctrl_mono > 0}
    res["C6.3_ventura_damps_seso_gradient"] = {"violations": viol_vent, "control_violations": viol_ctrl_vent,
                                              "passed": viol_vent == 0 and viol_ctrl_vent > 0}
    return res


def _neg_ref(model, X, menu):
    """Broken variant for the negative control: Boltzmann weights on -Q (colder favours the worst)."""
    c = model.forward(X, menu)
    logits = np.where(menu > 0, -c["Q"] / c["tau"], -np.inf)
    p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return np.sum(p * c["Q"], axis=1)


def _seso_grad_norm(model, X, menu, rng, v_target):
    """Outcome-gradient norm on Wq for ONE sample with a unit residual, the ventura pre-activation set to v_target."""
    import copy
    m = copy.deepcopy(model)
    m.w_choice = 0.0
    X1, menu1 = X[:1], menu[:1]
    h2 = m.forward(X1, menu1)["h2"]
    m.params["bv"] = np.array([v_target - float(h2[0] @ m.params["wv"])])   # exact: v = v_target
    a = np.array([int(np.flatnonzero(menu1[0])[0])])
    r = m.forward(X1, menu1)["Q"][np.arange(1), a] + 1.0
    _, g = loss_and_grads(m, {"X": X1, "menu": menu1, "action": a, "outcome": r})
    return float(np.linalg.norm(g["Wq"]))


def c7_split_integrity(task):
    def keyset(b):
        return set(np.round(row, 6).tobytes() for row in b["X"])
    tr = keyset(task["train"])
    return all(len(tr & keyset(task[s])) == 0 for s in ("val", "held_out", "shifted"))


# ----------------------------------------------------------------------------- hypothesis tests
def run_seed(seed, steps, loaded_steps):
    rng = np.random.default_rng(seed)
    task = make_task(seed)
    out = {}
    model = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
    info = fit(model, task, steps, rng)
    base = build_baseline(STATE_DIM, N_ACTIONS, rng)
    fit(base, task, steps, rng)
    rival = build_rival(STATE_DIM, N_ACTIONS, rng)
    fit(rival, task, steps, rng)
    ko_sig = knockout(build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng), "mesa", "identity")
    fit(ko_sig, task, steps, rng)
    ko_match = knockout(build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng), "encoder_block2", "identity")
    fit(ko_match, task, steps, rng)
    for split in ("held_out", "shifted"):
        out["model_" + split] = evaluate(model, task[split])
        out["baseline_" + split] = evaluate(base, task[split])
        out["rival_" + split] = evaluate(rival, task[split])
        out["ko_sig_" + split] = evaluate(ko_sig, task[split])
        out["ko_match_" + split] = evaluate(ko_match, task[split])
    out["ventura_recovery"] = ventura_recovery(model, task["held_out"])
    for name in ("mesa", "ventura", "encoder_block2"):
        mode = "identity" if name != "ventura" else "mean"
        out["inference_ko_" + name] = evaluate(knockout(model, name, mode), task["shifted"])["skill_corr"]
    loaded = make_task(seed, loaded=LOADED_BETA)      # same world and draws, loaded die
    lm = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
    fit(lm, loaded, loaded_steps, rng)
    lb = build_baseline(STATE_DIM, N_ACTIONS, rng)
    fit(lb, loaded, loaded_steps, rng)
    out["model_loaded"] = evaluate(lm, loaded["held_out"])
    out["baseline_loaded"] = evaluate(lb, loaded["held_out"])
    out["model"], out["task"], out["fit_info"] = model, task, info
    return out


def hypothesis_table(per_seed):
    rows = []
    specs = {h["id"]: h for h in MIND_CARD["hypotheses"]}

    def add(hid, diffs, metric):
        h = specs[hid]
        mean, ci = paired_bootstrap(diffs, seed=len(rows))
        rows.append({"id": hid, "metric": metric, "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                     "n_seeds": len(diffs), "verdict": verdict(mean, ci, h["direction"], h["mesi"])})

    add("H-SIG", [s["model_shifted"]["skill_corr"] - s["baseline_shifted"]["skill_corr"] for s in per_seed],
        "skill_corr (shifted)")
    add("H-NEC", [s["ko_sig_shifted"]["skill_corr"] - s["ko_match_shifted"]["skill_corr"] for s in per_seed],
        "skill_corr (shifted)")
    add("H-BLIND", [s["model_loaded"]["skill_corr"] - s["baseline_loaded"]["skill_corr"] for s in per_seed],
        "skill_corr (loaded dice)")
    add("H-BLIND-SELF", [s["model_loaded"]["skill_corr"] - s["model_held_out"]["skill_corr"] for s in per_seed],
        "skill_corr (loaded - fair, same world)")
    add("H-RIVAL", [s["model_shifted"]["skill_corr"] - s["rival_shifted"]["skill_corr"] for s in per_seed],
        "skill_corr (shifted)")
    add("H-VENT", [s["ventura_recovery"] - 0.5 for s in per_seed], "corr(log sigma_hat, log sigma*) - 0.5")
    return rows


def knockout_table(per_seed):
    rows = []
    full = [s["model_shifted"]["skill_corr"] for s in per_seed]
    for name, sig in (("mesa", True), ("ventura", True), ("encoder_block2", False)):
        diffs = [s["inference_ko_" + name] - f for s, f in zip(per_seed, full)]
        mean, ci = paired_bootstrap(diffs, seed=7)
        rows.append({"module": name, "signature": sig, "metric_change": mean, "ci95": ci})
    tr_sig = [s["ko_sig_shifted"]["skill_corr"] - f for s, f in zip(per_seed, full)]
    tr_match = [s["ko_match_shifted"]["skill_corr"] - f for s, f in zip(per_seed, full)]
    for name, sig, d in (("mesa (trained without)", True, tr_sig), ("encoder_block2 (trained without)", False, tr_match)):
        mean, ci = paired_bootstrap(d, seed=8)
        rows.append({"module": name, "signature": sig, "metric_change": mean, "ci95": ci})
    return rows


# ----------------------------------------------------------------------------- real-data bridge
def real_data_bridge(path, rng, steps):
    """CSV with columns s0..s{d-1}, m0..m{A-1}, a, r. Reports held-out outcome MSE and choice NLL."""
    if not path:
        print("real-data bridge: skipped (no --data PATH given)")
        return None
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = np.array([[float(x) for x in line.strip().split(",")] for line in f if line.strip()])
    d = sum(1 for h in header if h.startswith("s"))
    A = sum(1 for h in header if h.startswith("m"))
    X, M, a, r = rows[:, :d], rows[:, d:d + A], rows[:, d + A].astype(int), rows[:, d + A + 1]
    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr = {"X": X[idx[:cut]], "menu": M[idx[:cut]], "action": a[idx[:cut]], "outcome": r[idx[:cut]]}
    te = {"X": X[idx[cut:]], "menu": M[idx[cut:]], "action": a[idx[cut:]], "outcome": r[idx[cut:]]}
    m = build_model(d, A, "vector_regression", rng)
    fit(m, {"train": tr, "val": te}, steps, rng)
    c = m.forward(te["X"], te["menu"])
    mse = float(np.mean((c["Q"][np.arange(te["X"].shape[0]), te["action"]] - te["outcome"]) ** 2))
    nll = float(np.mean(-np.log(c["p"][np.arange(te["X"].shape[0]), te["action"]] + 1e-300)))
    print("real-data bridge: %s  held-out outcome MSE %.4f  choice NLL %.4f" % (os.path.basename(path), mse, nll))
    return {"file": os.path.basename(path), "outcome_mse": mse, "choice_nll": nll}


# ----------------------------------------------------------------------------- report and entry point
def fmt_ci(ci):
    return "[%+.3f, %+.3f]" % (ci[0], ci[1])


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True, prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return EXIT_USAGE
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return EXIT_OK
    if args.mutant and args.mutant not in MUTANTS:
        print("unknown mutant %s; registered: %s" % (args.mutant, ", ".join(MUTANTS)))
        return EXIT_USAGE
    t0 = time.time()
    quick = args.quick
    steps = STEPS_QUICK if quick else STEPS
    n_seeds = 1 if quick else args.seeds
    budget = BUDGET_QUICK_S if quick else BUDGET_S
    seeds = [args.seed + i for i in range(n_seeds)]
    th = MIND_CARD["thresholds"]
    print("chapter 0336 · Cordura Ledger · python3 %s%s" % (os.path.basename(__file__), " --quick" if quick else ""))
    print("seeds:", seeds)
    rng = np.random.default_rng(args.seed)
    task0 = make_task(args.seed)
    correctness = []
    try:
        gc_rows, gc_worst = c1_gradient_check(task0, rng, 60, mutant=args.mutant)
        gc_pass = all(r["passed"] for r in gc_rows)
        correctness.append({"id": "C1", "name": "gradient_check", "passed": gc_pass,
                            "detail": "max rel error %.2e over %d targets" % (gc_worst, len(gc_rows))})
        c2 = c2_determinism(task0, 60)
        correctness.append({"id": "C2", "name": "determinism_and_finiteness", "passed": c2, "detail": ""})
        model0 = build_model(STATE_DIM, N_ACTIONS, "vector_regression", rng)
        info0 = fit(model0, task0, steps, rng, mutant=args.mutant)
        c3, c3d = c3_learning(model0, task0, info0, th)
        correctness.append({"id": "C3", "name": "learning", "passed": c3,
                            "detail": "training objective drop %.2f (need %.2f); held-out skill corr %.3f vs trivial %.2f + %.2f"
                            % (c3d["loss_drop"], th["loss_drop_fraction"], c3d["held_out_skill_corr"],
                               c3d["trivial_skill_corr"], th["margin_over_trivial"])})
        c4, c4corr = c4_shuffled_control(task0, rng, steps, th["shuffled_band"])
        correctness.append({"id": "C4", "name": "shuffled_label_control", "passed": c4,
                            "detail": "held-out skill corr %.3f within ±%.2f" % (c4corr, th["shuffled_band"])})
        det, mrows = c5_mutants(task0, rng, 180, th)
        correctness.append({"id": "C5", "name": "mutant_detection", "passed": det == len(MUTANTS),
                            "detail": "%d/%d detected" % (det, len(MUTANTS))})
        props = c6_properties(rng, trials=60 if quick else 200)
        c6 = all(v["passed"] for k, v in props.items() if "passed" in v)
        correctness.append({"id": "C6", "name": "property_tests", "passed": c6,
                            "detail": "; ".join("%s v=%d ctrl=%d" % (k, v["violations"], v.get("control_violations", -1))
                                                for k, v in props.items())})
        c7 = c7_split_integrity(task0)
        correctness.append({"id": "C7", "name": "split_integrity", "passed": c7, "detail": ""})
        bridge = real_data_bridge(args.data, rng, steps)
        per_seed, hyp_rows, ko_rows = [], [], []
        if not quick:
            for s in seeds:
                per_seed.append(run_seed(s, steps, steps))
                print("seed %d done at %.1fs" % (s, time.time() - t0))
            hyp_rows = hypothesis_table(per_seed)
            ko_rows = knockout_table(per_seed)
        else:
            hyp_rows = [{"id": h["id"], "metric": h["metric"], "mean_diff": 0.0, "ci95": [0.0, 0.0],
                         "mesi": h["mesi"], "n_seeds": 0, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
    except FloatingPointError as e:
        print("non-finite values:", e)
        return EXIT_NONFINITE
    runtime = time.time() - t0
    c8 = runtime <= budget
    correctness.append({"id": "C8", "name": "budget", "passed": c8, "detail": "%.1fs of %.0fs" % (runtime, budget)})
    all_ok = all(c["passed"] for c in correctness)
    n_p = n_params(model0)
    n_b = n_params(build_baseline(STATE_DIM, N_ACTIONS, np.random.default_rng(0)))
    print("=== VERIFIED REPORT · chapter 0336 ===")
    print("environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__))
    print("seeds: %s · runtime %.1fs · params model %d · baseline %d (%.1f%% of model)"
          % (seeds, runtime, n_p, n_b, 100.0 * n_b / n_p))
    print("gradient check: %d tensors x %d targets at init and after 60 steps, max rel error %.2e, %s"
          % (len(model0.params), len(gc_rows), gc_worst, "passed" if gc_pass else "FAILED"))
    print("correctness:")
    for c in correctness:
        print("  %-4s %-28s %-6s %s" % (c["id"], c["name"], "pass" if c["passed"] else "FAIL", c["detail"]))
    print("mutation score: %d/%d (%s)" % (det, len(MUTANTS), ", ".join(m["mutant"] for m in mrows if m["detected"])))
    print("hypotheses (paired bootstrap over %d seeds, 2000 resamples):" % len(per_seed))
    for h in hyp_rows:
        print("  %-12s %-40s mean %+.3f ci %s mesi %.2f -> %s"
              % (h["id"], h["metric"], h["mean_diff"], fmt_ci(h["ci95"]), h["mesi"], h["verdict"]))
    if per_seed:
        print("per-seed skill correlation (held-out / shifted / loaded):")
        for s, ps in zip(seeds, per_seed):
            print("  seed %d  model %.3f/%.3f/%.3f  baseline %.3f/%.3f/%.3f  rival -/%.3f  ventura-recovery %.3f"
                  % (s, ps["model_held_out"]["skill_corr"], ps["model_shifted"]["skill_corr"],
                     ps["model_loaded"]["skill_corr"], ps["baseline_held_out"]["skill_corr"],
                     ps["baseline_shifted"]["skill_corr"], ps["baseline_loaded"]["skill_corr"],
                     ps["rival_shifted"]["skill_corr"], ps["ventura_recovery"]))
        print("knockouts (change in shifted skill correlation vs full model):")
        for k in ko_rows:
            print("  %-34s signature=%-5s change %+.3f ci %s" % (k["module"], k["signature"], k["metric_change"], fmt_ci(k["ci95"])))
    print("task types:", ", ".join(TASK_TYPES))
    print("=== END REPORT ===")
    if args.json:
        report = {"schema_version": "1.0", "chapter": 336, "file": os.path.basename(__file__),
                  "card_revision": MIND_CARD["card_revision"],
                  "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
                  "seeds": seeds, "runtime_s": runtime, "n_params": n_p,
                  "gradcheck": {"tensors_checked": len(model0.params), "tensors_total": len(model0.params),
                                "max_rel_error": gc_worst, "checked_at": ["init", "after_training_steps"],
                                "passed": gc_pass},
                  "correctness": correctness, "mutants": {"detected": det, "total": len(MUTANTS),
                                                          "score": det / len(MUTANTS), "rows": mrows},
                  "properties": props, "hypotheses": hyp_rows, "knockouts": ko_rows,
                  "real_data_bridge": bridge, "task_types": TASK_TYPES,
                  "exit_code": EXIT_OK if all_ok else EXIT_FAIL}
        write_report(args.json, report)
    if not c8:
        return EXIT_BUDGET
    return EXIT_OK if all_ok else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())

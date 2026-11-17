# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0382 - VYTAUTAS THE GREAT (1350-1430)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
"""Standard-and-Spoil learner (vėliava / grobis) for chapter 0382, Vytautas the Great.

Thesis
------
A gain the other side can hand you is not yet yours: every field gain is
impounded at the standard as evidence, and pay comes only from a fixed purse
that exists when the campaign's outcome confirms it, allotted by judged
contribution.

Evidence and provenance
-----------------------
Provenance: extrapolated. No philosophy of mind survives from Vytautas
(c. 1350-1430); his chancery letters (Codex epistolaris Vitoldi, ed. Prochaska
1882) state political positions only. The mechanism is inferred from deeds
and from other people's accounts: the Teutonic Order's anonymous letter on
staged flight (Berlin, GStA PK, OBA 2024; Ekdahl 1963), Jan Dlugosz's
portrait of his discipline and largesse (Annales; Skomial), the Thorn ransom
settlement of 1411 and the crown affair of 1429-30 (letters; Bar 2017).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
---------------------------------------------------
ID  doctrine (evidence)                         mechanism                         test
D1  a banner fled on purpose to break pursuers  M1 impound: the field signal      C6.1, H-SIG
    who think the game won (OBA 2024)           reaches the herald, never pay
D2  he himself pursued a feigned retreat at the shifted split: more feints and   H-SIG
    Vorskla (1399) and was enveloped            wider dispersal
D3  unauthorised plunder forbidden, punished    M1: a failed campaign pays        C6.2
D4  ransom instead of sack at Novgorod (1428)   M2 fixed purse vs sack rule       C6.1, H-RIVAL
D5  largesse by deed to guests, servants,       M3 herald: contribution-weighted  C6.3, knockout
    allies (Dlugosz)                            allocation of the purse
D6  Thorn 1411: captives settled by one ransom  M2 fixed purse                    C6.1
D7  crown affair 1429-30, the "honey bait"      blind spot: corruptible verdict   H-BLIND
D8  1420 letter: field facts (language, self-   M3 herald reads field reports     diagnostic
    naming) set against an outside verdict      as evidence
D9  rule collapsed into civil war after 1430    not tested (limit)                none

Research question
-----------------
Reward hacking and long-horizon credit assignment: when dense step signals can
be manufactured by an adversary (a staged retreat that pays booty), does a
learner that uses them only as evidence for allotting an outcome-gated fixed
purse resist the lure better than sum-form, verifier-gated and min-form
process credit, and what does it lose when the outcome verdict itself can be
bought?

Closest prior art and the delta
-------------------------------
RUDDER return decomposition (Arjona-Medina et al. 2019), whose redistributed
reward is also a difference of predicted returns; verifier-gated step rewards
(VeriGate 2026; conditional reward modelling, CRM 2025); min-form process
credit (PURE, Cheng et al. 2025); potential-based shaping (Ng, Harada and
Russell 1999). Delta: dense signals enter only the herald's value model as
evidence and never the pay; the pay is a fixed purse (conserved, zero on
failure) shared by positive changes in victory odds, which no dense signal
can enlarge; gated and min-form methods still pay in dense signal once the
gate opens.

Blind spot
----------
The centre itself can be lured. If the outcome verdict is corruptible (the
crown offered in 1429-30), an impounding learner chases whatever buys the
verdict and ignores field reports that would have warned it (H-BLIND).

Task
----
Campaign of T=12 steps toward L=8. Each step: a fleeing banner (p_flee), an
envoy (p_env) or nothing; a flight is genuine with p_gen, a rear cue reports a
feint (10% flip noise) and a feint is a trap with p_trap. ADVANCE gains 1 w.p.
0.6 if the line holds (d=0); PURSUE of a genuine flight with the line holding
gains 2; pursuing a trap, or any flight while dispersed, adds delta to d;
PURSUE of an envoy scatters the host (d+3) and sets a crown flag; REGROUP
lowers d by 1; d>=3 is a rout. Field signal: +0.1 per gain, +0.5 booty per
pursuit of any flight, -0.6 per envoy. Splits: iid (p_flee .45, p_gen .4,
p_trap .3, delta 1, nuisance sd 1); shifted (p_flee .6, p_gen .3, p_trap 1,
delta 3, sd 1.5); crown (the verdict also reports success when the crown flag
is set; the true metric stays x>=L).

Limits
------
An abstract game, not a battle model: no weapons, terrain or tactics. Small
tanh networks and REINFORCE; the herald's pay is myopic (one-step changes in
log-odds), so slow preparation is paid only insofar as it visibly moves the
odds, and the herald inherits any blindness of its state description. Hub failure (the 1430 succession) is not modelled. Provenance is
extrapolated: the mechanism is a reading of deeds, not a recovered doctrine,
and this file is a research prototype, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 2 (2026-10-01, before the confirmatory run; pilot on non-confirmatory seeds 100-101): (i) the "
        "first herald scored deed types additively and shared the purse by softmax; it paid booty-tagged "
        "pursuits and drove the learner into feint pursuit (shifted success 0.04), so the herald became a "
        "logistic value model over states whose contribution is the change in victory log-odds across a "
        "deed, with the purse shared in proportion to positive contributions; (ii) the sack rival never "
        "pursued feints because a feint was never free in training, so feints became traps with "
        "probability 0.3 in training and 1.0 when shifted, with T=12, advance probability 0.6, shifted "
        "p_gen 0.3 and dispersal 3, and an envoy pursuit now scatters the host; (iii) entropy coefficient "
        "0.01 -> 0.02 after outcome-only learners collapsed to always-advance on seed 100. Hypotheses, "
        "metrics, mesi, thresholds and seeds unchanged."],
    "generation": {"template_version": "codeguidelines v1.0, Appendix A", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-10-01"},
    "id": 382, "figure": "Vytautas the Great", "born": 1350, "died": 1430,
    "civilization": "Lithuanian (Grand Duchy of Lithuania)",
    "provenance": "extrapolated",
    "thesis": "A gain the other side can hand you is not yet yours: impound every field gain at the "
              "standard as evidence and pay only from a fixed purse that exists when the outcome "
              "confirms the campaign, allotted by judged contribution.",
    "evidence": [
        {"id": "D1", "claim": "At Grunwald/Zalgiris (15 July 1410) the Lithuanian wing withdrew and Order "
         "troops pursued; an anonymous letter to the Grand Master (1411-13) warns that an enemy may let one "
         "or two banners flee on purpose, because pursuers think the game won", "basis": "scholarship",
         "source": "Berlin, GStA PK, OBA 2024, ed. S. Ekdahl, Zeitschrift fur Ostforschung 12 (1963); "
                   "Ekdahl, Lituanus 56:2 (2010)"},
        {"id": "D2", "claim": "At the Vorskla (12 August 1399) his host was drawn out by a retreating centre "
         "and struck by a reserve; he escaped the rout", "basis": "deeds",
         "source": "Dlugosz, Annales, s.a. 1399; Frost, Oxford History of Poland-Lithuania I (2015)"},
        {"id": "D3", "claim": "He forbade unauthorised plunder and foraging and punished it at once",
         "basis": "scholarship", "source": "Dlugosz, Annales, portrait of Witold; Skomial, 'Polski portret "
                                           "litewskiego wladcy'"},
        {"id": "D4", "claim": "In the Novgorod campaign of 1428 he accepted a ransom rather than sack the "
         "city", "basis": "scholarship",
         "source": "Dlugosz, Annales, s.a. 1428; Skomial"},
        {"id": "D5", "claim": "Dlugosz stresses his generosity to guests, servants and Tatars",
         "basis": "scholarship", "source": "Dlugosz, Annales; Skomial; Mickunaite 2006"},
        {"id": "D6", "claim": "The Peace of Thorn (1411) settled the prisoners of 1410 by one collective "
         "ransom", "basis": "deeds", "source": "Urban, Tannenberg and After (2003); Jucas (2009)"},
        {"id": "D7", "claim": "In 1429-30 he pursued a royal crown offered by Sigismund; Dlugosz calls the "
         "offer bait; the crown never reached him", "basis": "primary",
         "source": "Codex epistolaris Vitoldi (1882); Bar, Roczniki Historyczne 83 (2017)"},
        {"id": "D8", "claim": "His letter of 11 March 1420 rejects the Breslau award on Samogitia by appeal "
         "to language and self-naming of its people", "basis": "primary",
         "source": "Codex epistolaris Vitoldi (1882)"},
        {"id": "D9", "claim": "After his death (27 October 1430) the realm fell into civil war (Svitrigaila)",
         "basis": "deeds", "source": "Frost (2015)"},
    ],
    "research_question": {
        "category": "reward hacking and specification gaming; long-horizon credit assignment",
        "question": "Does routing manufacturable dense step signals only into a contribution model that "
                    "allots an outcome-gated fixed purse resist staged lures better than sum-form, "
                    "verifier-gated and min-form process credit, and what does it cost when the outcome "
                    "verdict is corruptible?"},
    "mechanism": {
        "name": "Standard-and-Spoil learner (veliava/grobis)",
        "family": "policy-gradient RL with learned, outcome-gated return decomposition (value-difference "
                  "allotment of a fixed purse)",
        "signature_modules": ["impound", "herald"],
        "closest_prior_art": ["RUDDER return decomposition (Arjona-Medina et al., NeurIPS 2019)",
                              "verifier-gated step rewards (VeriGate, arXiv:2605.30451; CRM, arXiv:2509.26578)",
                              "min-form process credit (PURE, arXiv:2504.15275)",
                              "potential-based shaping (Ng, Harada and Russell, ICML 1999)"],
        "overlap": "Medium",
        "prior_art_queries": ["reward hacking process reward model credit assignment",
                              "verifier-gated step-level reward", "return decomposition RUDDER",
                              "min-form credit assignment PRM", "outcome-conditioned process reward",
                              "specification gaming dense reward shaping"],
        "contribution_type": "mechanism",
        "delta": "Dense signals reach only the contribution scorer; pay is a conserved purse that is zero "
                 "on failure and that no dense signal can enlarge, whereas gated or min-form methods still "
                 "pay in dense signal once the gate opens."},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 impound", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "shifted split", "property_test": None, "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M1 impound", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M2 purse", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M3 herald", "property_test": "C6.3", "hypothesis": None},
        {"doctrine": "D6", "mechanism": "M2 purse", "property_test": "C6.1", "hypothesis": None},
        {"doctrine": "D7", "mechanism": "blind spot", "property_test": None, "hypothesis": "H-BLIND"},
        {"doctrine": "D8", "mechanism": "M3 herald", "property_test": None, "hypothesis": None},
        {"doctrine": "D9", "mechanism": None, "property_test": None, "hypothesis": None},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The impounding learner keeps more true success than the sum-form "
         "field learner when feints multiply", "metric": "true success", "split": "shifted",
         "comparison": "standard - field", "direction": "greater", "mesi": 0.10, "seeds": 5},
        {"id": "H-NEC", "statement": "Paying the field signal (impound knocked out) costs more than "
         "replacing the learned critic by a batch mean", "metric": "true success", "split": "shifted",
         "comparison": "ko_impound - ko_critic", "direction": "less", "mesi": 0.10, "seeds": 5},
        {"id": "H-BLIND", "statement": "Under a corruptible verdict the impounding learner loses more true "
         "success than the field learner", "condition": "crown verdict", "grounding": "the crown affair of "
         "1429-30", "metric": "true success, crown minus clean", "split": "iid",
         "comparison": "(standard_crown - standard) - (field_crown - field)", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Ransom (fixed purse) beats the sack rule (field pay gated by "
         "victory) when feints multiply", "metric": "true success", "split": "shifted",
         "comparison": "standard - sack", "direction": "greater", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.10, "margin_over_trivial": 0.10, "shuffled_band": 0.05,
                   "gradcheck_rel_tol": 1e-5, "gradcheck_floor": 1e-4, "property_tol": 1e-9},
    "probe_predictions": [
        {"probe": "P10", "expected": "above baseline (true success per campaign under lures)"},
        {"probe": "P9", "expected": "equal to baseline (nuisance dimensions)"},
        {"probe": "P1-P8", "expected": "not supported: episodic_control only"}],
    "dialectic_links": [
        {"chapter": 355, "relation": "successor", "test": "none (grandson; narrative link only)"},
        {"chapter": 376, "relation": "rival", "test": "none (the Vorskla victors were Timur's clients)"},
        {"chapter": 386, "relation": "rival", "test": "none (Hussite crown offered 1420-21)"}],
    "corpus_neighbors": [
        {"chapter": 440, "similarity": None, "difference": "da Gama surrenders dense coastal feedback; here "
         "dense signal is kept as evidence and denied only as pay, and the lure is adversarial"},
        {"chapter": 366, "similarity": None, "difference": "Dusan routes authorless loss; here gains are "
         "routed through a conserved purse"},
        {"chapter": 202, "similarity": None, "difference": "al-Mansur's hub collects reports under a fixed "
         "budget; here reports set allocation but never the size of pay"},
        {"chapter": 394, "similarity": None, "difference": "instrument-level reward hacking vs adversarially "
         "staged dense reward"},
        {"chapter": 355, "similarity": None, "difference": "Gediminas falsifies offers by intervention; here "
         "offered gains are impounded until the outcome rules"}],
    "similarity_note": "corpus files were not available in this session; similarity left null",
    "barometer": {
        "cognitive_processing": ["held-out true success over trivial baselines"],
        "embodied_cognition": [],
        "world_modeling": ["shifted split: more feints, wider dispersal, noisier nuisance"],
        "consciousness": ["herald outcome log-loss normalised by base-rate entropy"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["recovery by regrouping before rout", "resistance to offered rewards (feint, crown)"]},
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "RL from process rewards for reasoning models: step scores as evidence, pay from verified "
                "outcomes only", "sector": "AI training, education", "dataset": "PRM800K with MATH"},
        {"use": "recommender training where clicks are manufacturable and satisfaction is the outcome",
         "sector": "media platforms", "dataset": "KuaiRand-1K"},
        {"use": "robot learning with gameable dense shaping terms", "sector": "robotics",
         "dataset": "Meta-World (simulation benchmark)"}],
    "safety_notes": "Military figure: an abstract formation game with no weapons, targeting or tactics. "
                    "The mechanism is framed defensively, as resistance to lures and reward hacking. "
                    "Results concern a synthetic task, never the historical person.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 382
L_GOAL, T_MAX, D_CUT = 8, 12, 3
ADVANCE, PURSUE, REGROUP = 0, 1, 2
N_ACT, OBS_DIM, HZ_DIM = 3, 20, 16
EV_NONE, EV_FLEE, EV_ENVOY = 0, 1, 2
P_ADV, CUE_FLIP = 0.6, 0.1
ADV_FIELD, ENVOY_FIELD, ENVOY_DISPERSE = 0.1, -0.6, 3
BASE = dict(p_flee=0.45, p_gen=0.40, p_env=0.10, p_trap=0.3, disperse=1, noise=1.0, booty=0.5, crown=False)
CONDITIONS = {"train": dict(BASE),
              "shifted": dict(BASE, p_flee=0.60, p_gen=0.30, p_trap=1.0, disperse=3, noise=1.5),
              "crown": dict(BASE, crown=True)}
SPLITS = {"iid": ("train", 0), "shifted": ("shifted", 1), "crown": ("crown", 0)}
KINDS = ("standard", "field", "sack", "min", "outcome")
KO_PLAN = (("impound", "identity"), ("herald", "mean"), ("critic", "mean"))
LR, BATCH, ENT_COEF, CLIP = 0.01, 64, 0.02, 5.0
UPDATES, QUICK_UPDATES, GC_STEPS, EVAL_N = 300, 150, 60, 2000
THR = MIND_CARD["thresholds"]
TASK_TYPES = ["episodic_control"]
ACTIVE = {"mutant": None}

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(z):
    return np.logaddexp(0.0, z)


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, beta1, beta2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1.0 - self.b1 ** self.t)
            vh = self.v[k] / (1.0 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20, floor=1e-4):
    """Central differences per tensor: every entry of small tensors, otherwise n random entries plus
    the largest-gradient entry. Error = |analytic - numeric| / max(|analytic|, |numeric|, floor)."""
    out = {}
    for k, p in params.items():
        flat, g = p.reshape(-1), grads[k].reshape(-1)
        if flat.size <= n_entries:
            idx = np.arange(flat.size)
        else:
            pick = rng.choice(flat.size, n_entries, replace=False)
            idx = np.unique(np.concatenate([pick, [int(np.argmax(np.abs(g)))]]))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2.0 * eps)
            worst = max(worst, abs(num - g[i]) / max(abs(num), abs(g[i]), floor))
        out[k] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [50.0 * (1.0 - level), 50.0 * (1.0 + level)])
    return float(d.mean()), float(lo), float(hi)


def hypothesis_verdict(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(lines, report, chapter, json_path=None):
    print("=== VERIFIED REPORT · chapter %04d ===" % chapter)
    for line in lines:
        print(line)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def child_rng(seed, *key):
    """Child stream SeedSequence(seed, spawn_key=key): (1,u) training, (2,s) evaluation, (3,.) checks."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=tuple(key)))


def env_reset(rng, cond="train", n=1):
    """Draw all exogenous randomness of n campaigns up front, so every learner meets the same field."""
    c = CONDITIONS[cond] if isinstance(cond, str) else cond
    u = rng.random((n, T_MAX))
    event = np.where(u < c["p_flee"], EV_FLEE, np.where(u < c["p_flee"] + c["p_env"], EV_ENVOY, EV_NONE))
    genuine = rng.random((n, T_MAX)) < c["p_gen"]
    cue = (~genuine) ^ (rng.random((n, T_MAX)) < CUE_FLIP)
    return {"cond": c, "t": 0, "x": np.zeros(n, int), "d": np.zeros(n, int), "alive": np.ones(n, bool),
            "won": np.zeros(n, bool), "crown": np.zeros(n, bool), "event": event, "genuine": genuine,
            "cue": cue, "adv_u": rng.random((n, T_MAX)), "nuis": rng.normal(0.0, c["noise"], (n, T_MAX, 3)),
            "trap": rng.random((n, T_MAX)) < c["p_trap"]}


def observe(state):
    t, n = min(state["t"], T_MAX - 1), state["x"].size
    o, r = np.zeros((n, OBS_DIM)), np.arange(n)
    o[r, np.minimum(state["x"], L_GOAL)] = 1.0
    o[r, 9 + np.minimum(state["d"], 3)] = 1.0
    ev = state["event"][:, t]
    o[:, 13] = ev == EV_FLEE
    o[:, 14] = ev == EV_ENVOY
    o[:, 15] = (ev == EV_FLEE) & state["cue"][:, t]
    o[:, 16] = (T_MAX - t) / T_MAX
    o[:, 17:20] = state["nuis"][:, t]
    return o


def env_step(state, action, rng=None):
    """Advance all live campaigns one step; rng is accepted for interface compatibility (the field was
    drawn in env_reset). Returns (state, field_signal, done_now)."""
    s, c, t = state, state["cond"], state["t"]
    live = s["alive"]
    a = np.where(live, action, -1)
    ev, gen = s["event"][:, t], s["genuine"][:, t]
    gain = (a == ADVANCE) & (s["d"] == 0) & (s["adv_u"][:, t] < P_ADV)
    flee = (a == PURSUE) & (ev == EV_FLEE)
    good = flee & gen & (s["d"] == 0)
    envoy = (a == PURSUE) & (ev == EV_ENVOY)
    sprung = flee & ~good & ((s["d"] > 0) | s["trap"][:, t])
    s["x"] = s["x"] + gain + 2 * good
    s["d"] = s["d"] + c["disperse"] * sprung + ENVOY_DISPERSE * envoy
    s["d"] = np.maximum(0, s["d"] - (a == REGROUP))
    s["crown"] = s["crown"] | envoy
    field = ADV_FIELD * gain + c["booty"] * flee + ENVOY_FIELD * envoy
    s["won"] = s["won"] | (live & (s["x"] >= L_GOAL))
    s["t"] = t + 1
    done = live & (s["won"] | (s["d"] >= D_CUT) | (s["t"] >= T_MAX))
    s["alive"] = live & ~done
    return s, field, done


def herald_state(st):
    """What the herald can establish at the standard: ground held, dispersal, time left, crown accepted.
    Column 15 (field reported so far) is filled from the field signal by herald_inputs."""
    n = st["x"].size
    z, r = np.zeros((n, HZ_DIM)), np.arange(n)
    z[r, np.minimum(st["x"], L_GOAL)] = 1.0
    z[r, 9 + np.minimum(st["d"], 3)] = 1.0
    z[:, 13] = (T_MAX - st["t"]) / T_MAX
    z[:, 14] = st["crown"]
    return z


def rollout(logit_fn, cond, n, env_rng, act_rng=None):
    """Play n campaigns; act_rng=None plays greedily (argmax). H[:, t] is the herald state before
    deed t and H[:, t+1] the state after it."""
    st = env_reset(env_rng, cond, n)
    obs, act = np.zeros((n, T_MAX, OBS_DIM)), np.zeros((n, T_MAX), int)
    mask, field, disp = np.zeros((n, T_MAX)), np.zeros((n, T_MAX)), np.zeros((n, T_MAX), int)
    H = np.zeros((n, T_MAX + 1, HZ_DIM))
    for t in range(T_MAX):
        live = st["alive"].copy()
        if not live.any():
            break
        o = observe(st)
        H[live, t] = herald_state(st)[live]
        logits = logit_fn(o)
        if act_rng is None:
            a = np.argmax(logits, axis=1)
        else:
            a = np.minimum((np.cumsum(softmax(logits), 1) < act_rng.random((n, 1))).sum(1), N_ACT - 1)
        disp[:, t] = st["d"]
        st, f, _ = env_step(st, a)
        obs[:, t], act[:, t], mask[:, t], field[:, t] = o, a, live, f
        H[live, t + 1] = herald_state(st)[live]
    reported = st["won"] | (st["crown"] & st["cond"]["crown"])
    return {"obs": obs, "act": act, "mask": mask, "field": field, "disp": disp, "won": st["won"], "H": H,
            "reported": reported, "event": st["event"], "genuine": st["genuine"]}


# ---------------------------------------------------------------- model
def _w(rng, fan_in, shape):
    return np.ascontiguousarray(rng.normal(0.0, 1.0 / math.sqrt(max(fan_in, 1)), shape))


def build_model(in_dim=OBS_DIM, out_dim=N_ACT, task_type="episodic_control", rng=None, **cfg):
    """kind: standard (impound + herald purse) | field | sack | min | outcome (baselines and rival)."""
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type: %s" % task_type)
    rng = rng if rng is not None else np.random.default_rng(0)
    kind = cfg.get("kind", "standard")
    hp, hh = cfg.get("policy_hidden", 24), cfg.get("herald_hidden", 12)
    hc = cfg.get("critic_hidden", 12 if kind == "standard" else 22)
    p = {"pol_W1": _w(rng, in_dim, (in_dim, hp)), "pol_b1": np.zeros(hp),
         "pol_W2": 0.1 * _w(rng, hp, (hp, out_dim)), "pol_b2": np.zeros(out_dim),
         "cri_W1": _w(rng, in_dim, (in_dim, hc)), "cri_b1": np.zeros(hc),
         "cri_w2": _w(rng, hc, (hc,)), "cri_b2": np.zeros(1)}
    if kind == "standard":
        hz = cfg.get("herald_dim", HZ_DIM)
        p.update({"her_W1": _w(rng, hz, (hz, hh)), "her_b1": np.zeros(hh),
                  "her_w2": _w(rng, hh, (hh,)), "her_b2": np.zeros(1)})
    return {"kind": kind, "params": p, "knock": {}, "in_dim": in_dim, "out_dim": out_dim, "diag": {}}


def policy_logits(model, X):
    p = model["params"]
    h = np.tanh(X @ p["pol_W1"] + p["pol_b1"])
    return h @ p["pol_W2"] + p["pol_b2"], h


def critic_value(model, X):
    p = model["params"]
    h = np.tanh(X @ p["cri_W1"] + p["cri_b1"])
    return h @ p["cri_w2"] + p["cri_b2"][0], h


def herald_inputs(H, field, mask):
    """Herald states with the field reported so far written into the last column (evidence only)."""
    Z = H.copy()
    Z[:, 1:, -1] = np.cumsum(field * mask, axis=1)
    Z[:, 0, -1] = 0.0
    return Z


def herald_mask(mask):
    return np.concatenate([np.ones((mask.shape[0], 1)), mask], axis=1)


def herald_values(model, Z):
    """Victory log-odds the herald reads off each state (logistic value model)."""
    p = model["params"]
    h = np.tanh(Z @ p["her_W1"] + p["her_b1"])
    return h @ p["her_w2"] + p["her_b2"][0], h


def contributions(model, Z, mask):
    """A deed's contribution: change in the herald's victory log-odds across it."""
    q = herald_values(model, Z)[0]
    return (q[:, 1:] - q[:, :-1]) * mask


def allocate(c, mask):
    """Largesse by deed: the purse is shared in proportion to positive contributions on live steps
    (equal shares when no deed raised the odds)."""
    pos = np.maximum(c, 0.0) * mask
    tot = pos.sum(axis=1, keepdims=True)
    even = mask / mask.sum(axis=1, keepdims=True)
    return np.where(tot > 1e-12, pos / np.maximum(tot, 1e-12), even)


def reward_to_go(r, mask):
    return np.cumsum((r * mask)[:, ::-1], axis=1)[:, ::-1] * mask


def min_to_go(r, mask):
    m = np.minimum.accumulate(np.where(mask > 0, r, np.inf)[:, ::-1], axis=1)[:, ::-1]
    return np.where(mask > 0, m, 0.0)


def pay(model, ep, V, field):
    """Per-step pay G (n, T) by the learner's pay rule. Standard: the purse T*V is allotted by the
    herald's contributions; the field signal is impounded (evidence for the herald, never pay)."""
    mask = ep["mask"]
    kind, knock, Vb = model["kind"], model["knock"], V[:, None] * mask
    if kind == "standard":
        if knock.get("herald") == "mean":
            alpha = mask / mask.sum(axis=1, keepdims=True)
        else:
            alpha = allocate(contributions(model, herald_inputs(ep["H"], field, mask), mask), mask)
        G = T_MAX * Vb * alpha
        return G + reward_to_go(field, mask) if knock.get("impound") == "identity" else G
    if kind == "field":
        return reward_to_go(field, mask) + Vb
    if kind == "sack":
        return V[:, None] * reward_to_go(field, mask) + Vb
    if kind == "min":
        return min_to_go(field, mask) + Vb
    return Vb


def make_batch(model, ep, rng=None, shuffle=False):
    """Freeze pay, baseline and herald targets for one update (no gradient flows through them)."""
    mask, V, field = ep["mask"], ep["reported"].astype(float), ep["field"]
    if shuffle:
        V, field = rng.permutation(V), field.copy()
        field[mask > 0] = rng.permutation(field[mask > 0])
    G = pay(model, ep, V, field)
    valid = mask.reshape(-1) > 0
    X = ep["obs"].reshape(-1, ep["obs"].shape[2])[valid]
    if model["knock"].get("critic") == "mean":
        per_t = G.sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)
        base = np.broadcast_to(per_t, G.shape).reshape(-1)[valid]
    else:
        base = critic_value(model, X)[0]
    Gv = G.reshape(-1)[valid]
    return {"X": X, "a": ep["act"].reshape(-1)[valid], "adv": Gv - base, "G": Gv,
            "Z": herald_inputs(ep["H"], field, mask), "HM": herald_mask(mask), "V": V}


def _tanh_back(X, h, dh):
    dpre = dh * (1.0 - h * h)
    return X.T @ dpre, dpre.sum(axis=0)


def herald_loss_grads(model, Z, HM, V):
    """Every state the campaign passed through is labelled with the verdict (logistic log-loss)."""
    p = model["params"]
    q, h = herald_values(model, Z)
    Vt, w = V[:, None], HM / HM.sum()
    loss = float(np.sum(w * (softplus(q) - Vt * q)))
    dq = w * (0.5 * (1.0 + np.tanh(0.5 * q)) - Vt)
    dpre = dq[..., None] * p["her_w2"] * (1.0 - h * h)
    g = {"her_b2": np.array([dq.sum()]), "her_w2": np.einsum("nt,nth->h", dq, h),
         "her_W1": np.einsum("ntz,nth->zh", Z, dpre), "her_b1": dpre.sum(axis=(0, 1))}
    return loss, g


def loss_and_grads(model, batch):
    p, X, a, adv, G = model["params"], batch["X"], batch["a"], batch["adv"], batch["G"]
    w = 1.0 / max(len(a), 1)
    logits, h = policy_logits(model, X)
    lp = logits - logsumexp(logits)[:, None]
    pr, onehot = np.exp(lp), np.eye(model["out_dim"])[a]
    ent = -np.sum(pr * lp, axis=1)
    loss = -w * np.sum(adv * np.sum(onehot * lp, axis=1)) - ENT_COEF * w * np.sum(ent)
    dl = w * (-adv[:, None] * (onehot - pr) + ENT_COEF * pr * (lp + ent[:, None]))
    g = {"pol_W2": h.T @ dl, "pol_b2": dl.sum(axis=0)}
    g["pol_W1"], g["pol_b1"] = _tanh_back(X, h, dl @ p["pol_W2"].T)
    v, hc = critic_value(model, X)
    loss += 0.5 * w * np.sum((v - G) ** 2)
    dv = w * (v - G)
    g["cri_w2"], g["cri_b2"] = hc.T @ dv, np.array([dv.sum()])
    g["cri_W1"], g["cri_b1"] = _tanh_back(X, hc, dv[:, None] * p["cri_w2"][None, :])
    model["diag"] = {}
    if "her_W1" in p:
        hl, hg = herald_loss_grads(model, batch["Z"], batch["HM"], batch["V"])
        m = float(np.clip(batch["V"].mean(), 1e-6, 1 - 1e-6))
        model["diag"] = {"herald_bce": hl, "base_entropy": -(m * math.log(m) + (1 - m) * math.log(1 - m))}
        loss += hl
        g.update(hg)
    for name, key in (("zero_grad_policy_W1", "pol_W1"), ("zero_grad_herald_W1", "her_W1")):
        if ACTIVE["mutant"] == name and key in g:
            g[key] = np.zeros_like(g[key])
    return float(loss), g


def fit(model, data, budget, rng):
    """Train for `budget` updates of BATCH campaigns from child streams (seed, 1, update)."""
    p = model["params"]
    opt = Adam(p, lr=0.0 if ACTIVE["mutant"] == "zero_learning_rate" else LR)
    hist = {"loss": [], "herald_bce": [], "base_entropy": [], "reported": []}
    for u in range(budget):
        ep = rollout(lambda X: policy_logits(model, X)[0], data["cond"], data.get("batch", BATCH),
                     child_rng(data["seed"], 1, u), rng)
        batch = make_batch(model, ep, rng, data.get("shuffle", False))
        loss, grads = loss_and_grads(model, batch)
        if not (np.isfinite(loss) and all(np.all(np.isfinite(g)) for g in grads.values())):
            raise FloatingPointError("non-finite loss or gradient at update %d" % u)
        grads, _ = clip_global_norm(grads, CLIP)
        if ACTIVE["mutant"] == "sign_flip_update":
            grads = {k: -g for k, g in grads.items()}
        opt.step(p, grads)
        hist["loss"].append(loss)
        hist["herald_bce"].append(model["diag"].get("herald_bce", float("nan")))
        hist["base_entropy"].append(model["diag"].get("base_entropy", float("nan")))
        hist["reported"].append(float(ep["reported"].mean()))
    return hist


def predict(model, X):
    return np.argmax(policy_logits(model, X)[0], axis=1)


def hidden_states(model, X):
    out = {"policy": policy_logits(model, X)[1], "critic": critic_value(model, X)[1]}
    return out


# ---------------------------------------------------------------- baselines and rival mechanisms
def constant_policy(k):
    """Trivial baseline: always the same order."""
    return lambda X: np.tile(np.eye(N_ACT)[k], (len(X), 1))


def uniform_policy(X):
    return np.zeros((len(X), N_ACT))


# The rival and published baselines are pay rules inside pay(): field (sum-form dense credit), sack
# (field pay gated by the victory verdict, after VeriGate/CRM), min (PURE min-form), outcome (verdict only).


# ---------------------------------------------------------------- registries
def modules(model):
    p = model["params"]
    mods = {"policy": {"params": [k for k in p if k.startswith("pol_")], "signature": False,
                       "role": "two-layer tanh policy network over the observation"},
            "critic": {"params": [k for k in p if k.startswith("cri_")], "signature": False,
                       "role": "state-value baseline for the policy gradient (tanh MLP)"}}
    if model["kind"] == "standard":
        mods["herald"] = {"params": [k for k in p if k.startswith("her_")], "signature": True,
                          "role": "logistic value model over states fitted to the outcome verdict; a deed's "
                                  "contribution is the change in log-odds across it, and a fixed outcome "
                                  "purse is shared in proportion to positive contributions"}
        mods["impound"] = {"params": [], "signature": True,
                           "role": "stop-pay gate: dense step signals enter only the contribution scorer as "
                                   "evidence and never the policy's training signal"}
    return mods


KNOCKOUTS = {"impound": ("identity",), "herald": ("mean",), "critic": ("mean",)}


def knockout(model, name, mode):
    """Copy with a module replaced: impound->identity (field paid), herald->mean (uniform allotment),
    critic->mean (per-time batch-mean baseline). Applied before training: the modules act on learning."""
    if model["kind"] != "standard" or mode not in KNOCKOUTS.get(name, ()):
        raise ValueError("no knockout %s/%s for %s" % (name, mode, model["kind"]))
    new = dict(model, params={k: v.copy() for k, v in model["params"].items()}, diag={})
    new["knock"] = dict(model["knock"], **{name: mode})
    return new


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flip_update": "update direction reversed (gradient ascent)",
           "zero_learning_rate": "optimizer learning rate set to zero",
           "zero_grad_policy_W1": "gradient of the policy's first layer zeroed",
           "zero_grad_herald_W1": "gradient of the herald's first layer zeroed"}


# ---------------------------------------------------------------- training and evaluation
def train_eval(seed, kind, cond, budget, key, knock=None):
    model = build_model(rng=child_rng(seed, 4, key), kind=kind)
    if knock:
        model = knockout(model, *knock)
    return model, fit(model, {"cond": cond, "seed": seed}, budget, child_rng(seed, 5, key))


def evaluate(policy, split, seed, n=EVAL_N, act_rng=None):
    cond, sid = SPLITS[split]
    fn = policy if callable(policy) else (lambda X: policy_logits(policy, X)[0])
    ep = rollout(fn, cond, n, child_rng(seed, 2, sid), act_rng)
    m, a, ev, ready = ep["mask"] > 0, ep["act"], ep["event"], ep["disp"] == 0

    def rate(sel):
        return float(np.mean(a[sel] == PURSUE)) if sel.any() else float("nan")
    return {"success": float(ep["won"].mean()), "reported": float(ep["reported"].mean()),
            "feint_pursuit": rate(m & (ev == EV_FLEE) & ~ep["genuine"] & ready),
            "genuine_pursuit": rate(m & (ev == EV_FLEE) & ep["genuine"] & ready),
            "crown_chase": rate(m & (ev == EV_ENVOY))}


def run_seed(seed, budget, quick):
    out = {"seed": seed, "eval": {}}
    for i, kind in enumerate(KINDS[:3] if quick else KINDS):
        model, hist = train_eval(seed, kind, "train", budget, i)
        out["eval"][kind] = {s: evaluate(model, s, seed) for s in ("iid", "shifted")}
        out["eval"][kind]["train_reported_mean"] = float(np.mean(hist["reported"]))
        if kind == "standard":
            out["model"], out["hist"] = model, hist
    if quick:
        return out
    for i, kind in enumerate(("standard", "field")):
        model, _ = train_eval(seed, kind, "crown", budget, i)
        out["eval"][kind + "_crown"] = {"crown": evaluate(model, "crown", seed)}
    for name, mode in KO_PLAN:
        model, hist = train_eval(seed, "standard", "train", budget, 0, (name, mode))
        out["eval"]["ko_" + name] = {"shifted": evaluate(model, "shifted", seed),
                                     "train_reported_mean": float(np.mean(hist["reported"]))}
    return out


def trivial_baselines(seed):
    out = {}
    for name, k in (("always_advance", ADVANCE), ("always_pursue", PURSUE), ("always_regroup", REGROUP)):
        out[name] = {s: evaluate(constant_policy(k), s, seed)["success"] for s in ("iid", "shifted")}
    out["uniform_random"] = {s: evaluate(uniform_policy, s, seed, act_rng=child_rng(seed, 8, i))["success"]
                             for i, s in enumerate(("iid", "shifted"))}
    out["best_constant_iid"] = max(out[k]["iid"] for k in ("always_advance", "always_pursue", "always_regroup"))
    return out


def herald_contributions(model, seed):
    """Mean contribution score per kind of deed on sampled iid campaigns (diagnostic for D5, D8)."""
    ep = rollout(lambda X: policy_logits(model, X)[0], "train", EVAL_N, child_rng(seed, 2, 0), child_rng(seed, 9, 0))
    c = contributions(model, herald_inputs(ep["H"], ep["field"], ep["mask"]), ep["mask"])
    m, a, ev, gen, ready = ep["mask"] > 0, ep["act"], ep["event"], ep["genuine"], ep["disp"] == 0
    cats = {"advance_line_held": m & (a == ADVANCE) & ready,
            "pursue_genuine_flight": m & (a == PURSUE) & (ev == EV_FLEE) & gen & ready,
            "pursue_feigned_flight": m & (a == PURSUE) & (ev == EV_FLEE) & ~gen,
            "pursue_envoy": m & (a == PURSUE) & (ev == EV_ENVOY),
            "regroup_dispersed": m & (a == REGROUP) & ~ready,
            "regroup_line_held": m & (a == REGROUP) & ready}
    return {k: {"mean_c": float(c[s].mean()) if s.any() else None, "n": int(s.sum())} for k, s in cats.items()}


def booty_sweep(seed, budget, base_run):
    out = {0.5: {k: base_run["eval"][k]["shifted"] for k in ("standard", "field")}}
    for bi, b in enumerate((0.2, 0.8)):
        cond = dict(CONDITIONS["train"], booty=b)
        out[b] = {}
        for i, kind in enumerate(("standard", "field")):
            model = build_model(rng=child_rng(seed, 4, i), kind=kind)
            fit(model, {"cond": cond, "seed": seed}, budget, child_rng(seed, 7, bi, i))
            out[b][kind] = evaluate(model, "shifted", seed)
    return out


# ---------------------------------------------------------------- tests: correctness
def gradcheck_one(model, seed, key, rng):
    ep = rollout(lambda X: policy_logits(model, X)[0], "train", 32, child_rng(seed, 3, key),
                 child_rng(seed, 3, 100 + key))
    batch = make_batch(model, ep)
    _, grads = loss_and_grads(model, batch)
    return finite_difference_check(lambda: loss_and_grads(model, batch)[0], model["params"], grads, rng)


def test_gradcheck(seed, rng, kinds=KINDS):
    errs, total = [], 0
    for i, kind in enumerate(kinds):
        model = build_model(rng=child_rng(seed, 4, i), kind=kind)
        errs.append(gradcheck_one(model, seed, i, rng))
        fit(model, {"cond": "train", "seed": seed}, GC_STEPS, child_rng(seed, 5, i))
        errs.append(gradcheck_one(model, seed, 10 + i, rng))
        total += 2 * len(model["params"])
    worst = max(max(e.values()) for e in errs)
    checked = sum(len(e) for e in errs)
    return {"tensors_checked": checked, "tensors_total": total, "max_rel_error": worst,
            "checked_at": ["init", "after_%d_training_steps" % GC_STEPS],
            "passed": bool(checked == total and worst <= THR["gradcheck_rel_tol"])}


def test_determinism(seed):
    runs = []
    for _ in range(2):
        model, hist = train_eval(seed, "standard", "train", 15, 0)
        runs.append((hist["loss"], evaluate(model, "iid", seed, n=300)["success"], model["params"]))
    same = runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] and all(
        np.array_equal(runs[0][2][k], runs[1][2][k]) for k in runs[0][2])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][2].values())
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": "identical losses, outputs and parameters on rerun: %s; finite: %s" % (same, finite)}


def learning_check(hist, success_iid, triv):
    b, H, k = np.array(hist["herald_bce"]), np.array(hist["base_entropy"]), 20
    first, last = b[:k].sum() / max(H[:k].sum(), 1e-9), b[-k:].sum() / max(H[-k:].sum(), 1e-9)
    drop, margin = 1.0 - last / first, success_iid - triv["best_constant_iid"]
    passed = bool(drop >= THR["loss_drop_fraction"] and margin >= THR["margin_over_trivial"])
    return passed, drop, margin


def test_learning(run, triv):
    passed, drop, margin = learning_check(run["hist"], run["eval"]["standard"]["iid"]["success"], triv)
    return {"id": "C3", "name": "learning", "passed": passed,
            "detail": "herald normalised log-loss drop %.3f (>= %.2f); iid success margin over best constant "
                      "%+.3f (>= %.2f)" % (drop, THR["loss_drop_fraction"], margin, THR["margin_over_trivial"])}


def test_shuffled(seed, budget, triv):
    model = build_model(rng=child_rng(seed, 4, 0), kind="standard")
    fit(model, {"cond": "train", "seed": seed, "shuffle": True}, budget, child_rng(seed, 5, 50))
    s = evaluate(model, "iid", seed)["success"]
    band = triv["best_constant_iid"] + THR["shuffled_band"]
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(s <= band),
            "detail": "verdicts and field signals permuted: iid success %.3f, band <= %.3f" % (s, band)}


def test_mutants(seed, budget, triv, rng):
    saved, found = ACTIVE["mutant"], {}
    for name in MUTANTS:
        ACTIVE["mutant"] = name
        if name.startswith("zero_grad"):
            model = build_model(rng=child_rng(seed, 4, 0), kind="standard")
            found[name] = max(gradcheck_one(model, seed, 0, rng).values()) > THR["gradcheck_rel_tol"]
        else:
            model, hist = train_eval(seed, "standard", "train", budget, 0)
            found[name] = not learning_check(hist, evaluate(model, "iid", seed)["success"], triv)[0]
    ACTIVE["mutant"] = saved
    n = sum(found.values())
    return {"detected": n, "total": len(found), "score": n / len(found), "by_mutant": found}


def _pay_trial(rng, trial):
    std = build_model(rng=child_rng(trial, 10, 0), kind="standard")
    for k in std["params"]:
        std["params"][k] *= rng.uniform(0.5, 3.0)
    sack, fld = build_model(rng=child_rng(trial, 10, 1), kind="sack"), build_model(kind="field")
    ep = rollout(lambda X: policy_logits(std, X)[0], "train", 64, child_rng(trial, 11, 0), rng)
    return std, sack, fld, ep


def test_properties(rng, trials=12, searches=10):
    """C6.1 purse conserved and invariant to injected booty; C6.2 a failed campaign pays nothing;
    C6.3 allotment is a simplex on live steps. Each with a broken variant that must violate it."""
    w = {k: 0.0 for k in ("c61", "c62", "c63")}
    ctl = dict(w)
    for trial in range(trials):
        std, sack, fld, ep = _pay_trial(rng, trial)
        m, f = ep["mask"], ep["field"]
        V = (rng.random(len(m)) < 0.5).astype(float)
        G0, S0 = pay(std, ep, V, f).sum(1), pay(sack, ep, V, f).sum(1)
        w["c61"] = max(w["c61"], np.abs(G0 - T_MAX * V).max())
        ctl["c61"] = max(ctl["c61"], np.abs(S0 - T_MAX * V).max())
        for _ in range(searches):
            inj = f + rng.uniform(0.0, 3.0, f.shape) * (rng.random(f.shape) < 0.3) * m
            w["c61"] = max(w["c61"], np.abs(pay(std, ep, V, inj).sum(1) - G0).max())
            ctl["c61"] = max(ctl["c61"], np.abs(pay(sack, ep, V, inj).sum(1) - S0).max())
            zero = np.zeros(len(m))
            w["c62"] = max(w["c62"], np.abs(pay(std, ep, zero, inj)).max())
            ctl["c62"] = max(ctl["c62"], np.abs(pay(fld, ep, zero, inj)).max())
            q = herald_values(std, herald_inputs(ep["H"], inj, m))[0]
            al = allocate((q[:, 1:] - q[:, :-1]) * m, m)
            w["c63"] = max(w["c63"], np.abs(al.sum(1) - 1).max(), -al.min(), (al * (1 - m)).max())
            raw = np.maximum(q[:, 1:] - q[:, :-1], 0.0) + 1e-12
            ctl["c63"] = max(ctl["c63"], ((raw / raw.sum(1, keepdims=True)) * (1 - m)).sum(1).max())
    names = {"c61": "purse conserved and booty-invariant", "c62": "failed campaign pays nothing",
             "c63": "allotment is a simplex on live steps"}
    out = []
    for k, label in names.items():
        ok = w[k] <= THR["property_tol"] and ctl[k] > 1e-3
        out.append({"id": "C6." + k[-1], "name": label, "passed": bool(ok),
                    "detail": "max violation %.2e; broken variant %.3f" % (w[k], ctl[k])})
    return out


def _episode_hashes(st):
    rows = np.concatenate([st["event"], st["genuine"], st["cue"], st["trap"], np.round(st["adv_u"], 12)], axis=1)
    return {hashlib.sha1(r.tobytes()).hexdigest() for r in rows}


def test_split_integrity(seeds, budget):
    overlap, control = 0, 0
    for s in seeds:
        train = set()
        for u in range(budget):
            train |= _episode_hashes(env_reset(child_rng(s, 1, u), "train", BATCH))
        for split in ("iid", "shifted"):
            cond, sid = SPLITS[split]
            overlap += len(train & _episode_hashes(env_reset(child_rng(s, 2, sid), cond, EVAL_N)))
        control += len(train & _episode_hashes(env_reset(child_rng(s, 1, 0), "train", BATCH)))
    ok = overlap == 0 and control > 0
    return {"id": "C7", "name": "split_integrity", "passed": bool(ok),
            "detail": "train/eval episode-hash overlap %d; redraw of a training stream overlaps %d (control)"
                      % (overlap, control)}


# ---------------------------------------------------------------- tests: hypotheses
def seed_metric(run, hid):
    e = run["eval"]
    if hid == "H-SIG":
        return e["standard"]["shifted"]["success"] - e["field"]["shifted"]["success"]
    if hid == "H-NEC":
        return e["ko_impound"]["shifted"]["success"] - e["ko_critic"]["shifted"]["success"]
    if hid == "H-BLIND":
        return ((e["standard_crown"]["crown"]["success"] - e["standard"]["iid"]["success"])
                - (e["field_crown"]["crown"]["success"] - e["field"]["iid"]["success"]))
    return e["standard"]["shifted"]["success"] - e["sack"]["shifted"]["success"]


def test_hypotheses(runs, rng, quick):
    hyps, kos = [], []
    for h in MIND_CARD["hypotheses"]:
        if quick:
            hyps.append({"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None,
                         "mesi": h["mesi"], "n_seeds": 1, "verdict": "not evaluated"})
            continue
        mean, lo, hi = paired_bootstrap([seed_metric(r, h["id"]) for r in runs], rng)
        hyps.append({"id": h["id"], "metric": h["metric"] + " (" + h["comparison"] + ")", "mean_diff": mean,
                     "ci95": [lo, hi], "mesi": h["mesi"], "n_seeds": len(runs),
                     "verdict": hypothesis_verdict(mean, lo, hi, h["direction"], h["mesi"])})
    for name, mode in ([] if quick else KO_PLAN):
        d = [r["eval"]["ko_" + name]["shifted"]["success"] - r["eval"]["standard"]["shifted"]["success"]
             for r in runs]
        mean, lo, hi = paired_bootstrap(d, rng)
        auc = float(np.mean([r["eval"]["ko_" + name]["train_reported_mean"]
                             - r["eval"]["standard"]["train_reported_mean"] for r in runs]))
        kos.append({"module": name, "mode": mode, "signature": name != "critic", "metric_change": mean,
                    "ci95": [lo, hi], "training_success_change": auc})
    return hyps, kos


# ---------------------------------------------------------------- optional real-data bridge
def data_bridge(path, rng):
    """CSV columns episode_id,step,action,field_signal,outcome (e.g. PRM800K steps: action = step label
    id, field_signal = step rating, outcome = final answer correct). Fits the herald only."""
    if not path:
        return {"status": "skipped: no --data PATH given"}
    if not os.path.exists(path):
        return {"status": "skipped: file not found: %s" % path}
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
        rows = [dict(zip(head, line.strip().split(","))) for line in fh if line.strip()]
    eps = sorted({r["episode_id"] for r in rows})
    eid = {e: i for i, e in enumerate(eps)}
    T = min(64, 1 + max(int(r["step"]) for r in rows))
    n_act = min(8, 1 + max(int(r["action"]) for r in rows))
    act, field = np.zeros((len(eps), T), int), np.zeros((len(eps), T))
    M, V = np.zeros((len(eps), T)), np.zeros(len(eps))
    for r in rows:
        i, t = eid[r["episode_id"]], int(r["step"])
        if t < T:
            act[i, t], field[i, t], M[i, t] = min(int(r["action"]), n_act - 1), float(r["field_signal"]), 1.0
        V[i] = max(V[i], float(r["outcome"]))
    H = np.zeros((len(eps), T + 1, n_act + 2))
    H[:, 1:, :n_act] = np.eye(n_act)[act] * M[..., None]
    H[:, :, n_act] = np.arange(T + 1) / T
    model = build_model(rng=rng, kind="standard", herald_dim=n_act + 2)
    Z, HM = herald_inputs(H, field, M), herald_mask(M)
    sub = {k: v for k, v in model["params"].items() if k.startswith("her_")}
    opt, first = Adam(sub, lr=LR), herald_loss_grads(model, Z, HM, V)[0]
    for _ in range(300):
        opt.step(sub, herald_loss_grads(model, Z, HM, V)[1])
    c = contributions(model, Z, M)
    per = {str(k): float(c[(act == k) & (M > 0)].mean()) for k in range(n_act) if ((act == k) & (M > 0)).any()}
    return {"status": "fitted", "episodes": len(eps), "log_loss_before": first,
            "log_loss_after": herald_loss_grads(model, Z, HM, V)[0], "mean_contribution_by_action": per}


# ---------------------------------------------------------------- report
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def summarise(runs, key, split, field="success"):
    vals = [r["eval"][key][split][field] for r in runs if key in r["eval"]]
    return float(np.nanmean(vals)) if vals else None


def build_report(ctx):
    runs, seeds, gc, tests, mut = ctx["runs"], ctx["seeds"], ctx["gc"], ctx["tests"], ctx["mut"]
    fmt = lambda v: "  n/a " if v is None else "%6.3f" % v
    L = ["file: %s   card_revision: %d   mode: %s" % (os.path.basename(__file__), MIND_CARD["card_revision"],
                                                      ctx["mode"]),
         "environment: python %s, numpy %s" % (sys.version.split()[0], np.__version__),
         "seeds: %s   updates per learner: %d x %d campaigns   runtime: %.1f s" % (seeds, ctx["budget"], BATCH,
                                                                                ctx["runtime"]),
         "parameters: " + ", ".join("%s %d" % (k, v) for k, v in ctx["n_params"].items()),
         "gradient check: %d/%d tensor checks, max rel error %.2e, at %s -> %s" % (
             gc["tensors_checked"], gc["tensors_total"], gc["max_rel_error"], " and ".join(gc["checked_at"]),
             "pass" if gc["passed"] else "FAIL"), "correctness:"]
    L += ["  %-5s %-36s %s  %s" % (t["id"], t["name"], "pass" if t["passed"] else "FAIL", t["detail"])
          for t in tests]
    L.append("mutation score: %d/%d (%s)" % (mut["detected"], mut["total"], ", ".join(
        "%s %s" % (k, "detected" if v else "MISSED") for k, v in mut["by_mutant"].items())))
    tb = ctx["triv"]
    L.append("trivial baselines (iid / shifted true success): " + "; ".join(
        "%s %.3f/%.3f" % (k, tb[k]["iid"], tb[k]["shifted"]) for k in
        ("always_advance", "always_pursue", "always_regroup", "uniform_random")))
    L.append("learners, mean over seeds   iid    shifted  feint-pursuit(shifted)  genuine-pursuit(shifted)")
    for k in KINDS + ("ko_impound", "ko_herald", "ko_critic"):
        if any(k in r["eval"] for r in runs):
            sp = "iid" if "iid" in runs[0]["eval"].get(k, {}) else None
            L.append("  %-24s %s  %s  %s  %s" % (k, fmt(summarise(runs, k, sp) if sp else None),
                     fmt(summarise(runs, k, "shifted")), fmt(summarise(runs, k, "shifted", "feint_pursuit")),
                     fmt(summarise(runs, k, "shifted", "genuine_pursuit"))))
    for k in ("standard_crown", "field_crown"):
        if k in runs[0]["eval"]:
            L.append("  %-24s crown-trained: true %s reported %s crown-chase %s" % (
                k, fmt(summarise(runs, k, "crown")), fmt(summarise(runs, k, "crown", "reported")),
                fmt(summarise(runs, k, "crown", "crown_chase"))))
    L.append("herald contributions (seed %d, sampled iid campaigns): " % seeds[0] + "; ".join(
        "%s %s (n=%d)" % (k, "n/a" if v["mean_c"] is None else "%+.2f" % v["mean_c"], v["n"])
        for k, v in ctx["contrib"].items()))
    L.append("hypotheses (paired over seeds, 95% bootstrap CI, 2000 resamples):")
    for h in ctx["hyps"]:
        ci = "n/a" if h["ci95"] is None else "[%+.3f, %+.3f]" % tuple(h["ci95"])
        md = "n/a" if h["mean_diff"] is None else "%+.3f" % h["mean_diff"]
        L.append("  %-8s %s  CI %s  mesi %.2f  -> %s" % (h["id"], md, ci, h["mesi"], h["verdict"]))
    L.append("knockouts (shifted true success change vs full learner, trained with the module replaced; "
             "change in mean training success):")
    L += ["  %-8s %-8s signature=%s  %+.3f  CI [%+.3f, %+.3f]  training %+.3f" % (
          k["module"], k["mode"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1],
          k["training_success_change"]) for k in ctx["kos"]]
    if ctx["sweep"]:
        L.append("booty sweep (seed %d, shifted true success standard / field): " % seeds[0] + "; ".join(
            "b=%.1f %.3f / %.3f" % (b, v["standard"]["success"], v["field"]["success"])
            for b, v in sorted(ctx["sweep"].items())))
    L.append("data bridge: %s" % ctx["bridge"]["status"])
    L.append("task types: %s   exit code: %d" % (TASK_TYPES, ctx["exit_code"]))
    return L


def json_report(ctx):
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": ctx["seeds"], "runtime_s": ctx["runtime"], "n_params": ctx["n_params"]["standard"],
           "gradcheck": ctx["gc"], "correctness": ctx["tests"],
           "mutants": {k: ctx["mut"][k] for k in ("detected", "total", "score")},
           "hypotheses": ctx["hyps"], "knockouts": ctx["kos"], "task_types": TASK_TYPES,
           "exit_code": ctx["exit_code"],
           "diagnostics": {"mode": ctx["mode"], "updates": ctx["budget"], "n_params_all": ctx["n_params"],
                           "trivial_baselines": ctx["triv"], "mutants_by_name": ctx["mut"]["by_mutant"],
                           "per_seed_eval": [r["eval"] for r in ctx["runs"]],
                           "herald_contributions": ctx["contrib"], "booty_sweep": ctx["sweep"],
                           "data_bridge": ctx["bridge"]}}
    return _clean(rep)


# ---------------------------------------------------------------- command line
def protocol(args, t0):
    quick = args.quick
    seeds = [args.seed] if quick else [args.seed + i for i in range(args.seeds)]
    budget, limit = (QUICK_UPDATES, 20.0) if quick else (UPDATES, 180.0)
    rng = np.random.default_rng(args.seed)
    print("seeds: %s (base %d); mode %s; mutant %s" % (seeds, args.seed, "quick" if quick else "full",
                                                       ACTIVE["mutant"]))
    gc = test_gradcheck(args.seed, rng, KINDS[:3] if quick else KINDS)
    tests = [{"id": "C1", "name": "gradient_check", "passed": gc["passed"],
              "detail": "max rel error %.2e over %d tensor checks" % (gc["max_rel_error"], gc["tensors_checked"])},
             test_determinism(args.seed)]
    runs = [run_seed(s, budget, quick) for s in seeds]
    triv = trivial_baselines(args.seed)
    tests += [test_learning(runs[0], triv), test_shuffled(args.seed, budget, triv)]
    mut = test_mutants(args.seed, budget, triv, rng)
    tests.append({"id": "C5", "name": "mutant_detection", "passed": mut["score"] == 1.0,
                  "detail": "%d/%d mutants detected" % (mut["detected"], mut["total"])})
    tests += test_properties(rng) + [test_split_integrity(seeds, budget)]
    hyps, kos = test_hypotheses(runs, rng, quick)
    ctx = {"runs": runs, "seeds": seeds, "gc": gc, "tests": tests, "mut": mut, "triv": triv, "hyps": hyps,
           "kos": kos, "budget": budget, "mode": "quick" if quick else "full",
           "contrib": herald_contributions(runs[0]["model"], seeds[0]),
           "sweep": None if quick else booty_sweep(args.seed, budget, runs[0]),
           "bridge": data_bridge(args.data, rng),
           "n_params": {k: n_params(build_model(kind=k)) for k in KINDS}}
    ctx["runtime"] = time.time() - t0
    in_budget = ctx["runtime"] <= limit
    tests.append({"id": "C8", "name": "budget", "passed": in_budget,
                  "detail": "%.1f s of %.0f s" % (ctx["runtime"], limit)})
    failed = [t for t in tests if not t["passed"]]
    ctx["exit_code"] = 0 if not failed else (3 if all(t["id"] == "C8" for t in failed) else 1)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Chapter 0382 Standard-and-Spoil learner: full protocol by default.")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced updates, all correctness tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", choices=sorted(MUTANTS))
    ap.add_argument("--data", metavar="PATH", help="optional CSV for the real-data bridge")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.seeds < 1 or (not args.quick and args.seeds < 5):
        print("usage error: the full protocol needs --seeds >= 5", file=sys.stderr)
        return 2
    ACTIVE["mutant"] = args.mutant
    t0 = time.time()
    try:
        ctx = protocol(args, t0)
    except FloatingPointError as err:
        print("non-finite values: %s" % err, file=sys.stderr)
        return 4
    write_report(build_report(ctx), json_report(ctx), CHAPTER, args.json)
    return ctx["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

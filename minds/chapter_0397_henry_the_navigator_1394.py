#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0397 - HENRY THE NAVIGATOR (1394-1460)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 20, Minds 381-400 Available on Amazon https://www.amazon.com/dp/B0HRS9ZHVG
# Resume and Interactive Demos at  https://artificiology.com/
# Videos of the playable minds at https://www.youtube.com/@LostMindsAI
# END ATTRIBUTION  
"""
TORNAVIAGEM (the return voyage) -- a return-gated ledger learner for chapter 0397,
Henry the Navigator (Infante Dom Henrique, 1394-1460).

Thesis
    A mind that never sails knows the sea only from what comes back: so it must
    keep a ledger of every departure, read a ship that does not return as its
    only evidence of danger, guess the unseen water from the last returned log,
    and send each ship one stretch past the last return.

Evidence and provenance (extrapolated)
    No philosophy of mind by Henry survives. His own texts (the 1436 opinion on
    the Tangier war, charters, wills) state crusading and administrative
    positions only. The mechanism is inferred from documented deeds, most of
    them reported by his own chronicler Gomes Eanes de Zurara (Chronicle of
    Guinea, finished 1453), a panegyrist: twelve years of ships turned back at
    Cape Bojador by the belief that no ship could return; captains received
    without blame, rewarded and sent again, with more promised to whoever added
    to the voyage of those before; the 1434 charge to Gil Eanes that passing
    the cape was enough, with a plant brought home as token; a chart extended
    only by what was witnessed; the 1443 licence over every sailing south of
    the cape. Nothing below is a doctrine he stated.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D2 the feared outcome is non-return; D7 every sailing is licensed
        -> M1 rol: departure ledger; a non-return is a censored outcome whose
           likelihood is marginalised over the water nobody reported
        -> C6.1 censoring integrity, C6.2 score at truth, C6.3 ledger
           conservation, H-NEC
    D6 the chart grows from the witnessed coast; D4 inference from tracks
        -> M2 sonda: conditional Gaussian guess of unseen in-situ conditions
           from the last returned log -> C6.2, H-SIG
    D1 more reward for whoever adds to the voyage of those before
        -> M3 legua: one unknown stretch per sailing -> H-RIVAL
    D3 the fear of men who never went has no authority
        -> M4 no inherited prior on the dread sign -> H-BLIND
    D8 Tangier 1437, the camp cut from the ships -> H-BLIND

Research question (robustness to spurious correlation: survivorship)
    Probes that fail often cannot report: lost robots, dead floats, rejected
    loans, terminated episodes. Their in-situ observations are missing exactly
    when the outcome is bad, so logs teach that everything is safe. Can a
    learner recover a hazard map over covariates it only ever sees on survival,
    using nothing from the lost probe but the fact that it was sent, and does
    that beat predicting the outcome directly from what is visible in advance?

Closest prior art and the delta
    Sample-selection models (Heckman 1979); Wald's 1943 analysis of returning
    aircraft (Mangel and Samaniego 1984); noisy-OR multiple-instance learning
    (Maron and Lozano-Perez 1998); safe-set expansion with a return guarantee
    (Moldovan and Abbeel 2012; Turchetta et al. 2016); Go-Explore (Ecoffet et
    al. 2021); one-bit failure signals under termination (arXiv:2608.26571).
    Delta: the covariates of a failure are never observed, so the failure term
    integrates the hazard over a learned extrapolation from the last returned
    log, jointly fitted with that extrapolation; and the ordering rule keeps
    one unknown per probe so each loss stays attributable. The baseline is the
    size-matched standard alternative: a direct outcome classifier on what is
    known before departure.

Blind spot
    A learner raised where the dread sign was always false learns that dread is
    noise. Where the old fear is right (a true wall behind the sign) it loses
    hulls that those who would not sail keep. Henry at Tangier in 1437.

Task: the coast (generative process)
    Each coast has N stretches and two lanes (inshore, offshore). In-situ
    conditions u (3 numbers per lane) follow an AR(1) process along the coast;
    the chance of losing a ship in a stretch depends on u only. Before sailing
    the harbour sees v: a noisy view of one in-situ number, a dread sign and a
    nuisance. A sailing enters uncharted stretches; if it survives it returns
    with u; if not, nothing returns. Four hulls, twenty sailings. Score:
    stretches charted minus two per hull lost, over N. Shifted coasts change
    the AR coefficient, the visibility and the roughness; the hazard law stays.
    Blind coasts put a true wall behind the dread sign at the cape.

Limits
    Synthetic, small, one-dimensional coasts; the hazard law is fixed and
    simple. Results bear on the mechanism only and are not evidence about the
    historical person. A research prototype of an AGI-oriented component, not
    an AGI. Framed for the protection of probes and people: no targeting.
"""
MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 2,
    "revision_log": [
        "rev 2 (2026-10-01): a pilot on seeds 100-101, outside the confirmatory seeds 0-4 and run before "
        "any five-seed evaluation, tried relearning the weights between seasons from each campaign's own "
        "ledger. Every strength tried left the scores of all arms unchanged or lower, so the step was "
        "removed: within a campaign the weights stay as inherited, and learning is the last returned log "
        "plus the ledger counts. Decision-time Monte Carlo draws were raised from 32 to 128. The pilot "
        "displayed learner-versus-baseline scores on those two seeds, disclosed here because it could "
        "bias the change. Hypotheses, metrics, mesi and thresholds are unchanged from rev 1."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)",
                   "generator": "Claude (Anthropic)", "generator_version": "Claude Fable 5.1",
                   "date": "2026-10-01"},
    "id": 397, "figure": "Henry the Navigator", "born": 1394, "died": 1460,
    "civilization": "Portuguese (Kingdom of Portugal, House of Avis)",
    "provenance": "extrapolated",
    "thesis": ("Knowledge is what comes back: keep a ledger of every departure, read a non-return "
               "as the only evidence of danger, guess the unseen water from the last returned log, "
               "and send each ship one stretch past the last return."),
    "evidence": [
        {"id": "D1", "claim": "He received returning captains patiently, never blamed failure, rewarded "
         "them, sent them again and promised more to whoever added to the voyage of those before.",
         "basis": "deeds", "source": "Zurara, Cronica dos feitos de Guine, ch. 9"},
        {"id": "D2", "claim": "For twelve years no captain passed Cape Bojador: mariners held that no "
         "ship that passed could ever return.", "basis": "deeds", "source": "Zurara, ch. 8"},
        {"id": "D3", "claim": "In 1434 he told Gil Eanes that passing the cape alone would be enough and "
         "dismissed the fear as the opinion of mariners who knew only common routes; Eanes brought "
         "back a plant as token (reported speech).", "basis": "scholarship", "source": "Zurara, ch. 9"},
        {"id": "D4", "claim": "From tracks of men and camels he inferred inhabited land near by and "
         "ordered an interpreter captured for news of the land.", "basis": "scholarship",
         "source": "Zurara, ch. 10; Seed, Sails and Shadows (2025)"},
        {"id": "D5", "claim": "Mariners and merchants sail only where profit is sure, so a lord must "
         "send ships to gain certainty (motive attributed by his chronicler).",
         "basis": "scholarship", "source": "Zurara, ch. 7"},
        {"id": "D6", "claim": "The coast beyond the cape went on the chart as witnessed by the eye; "
         "his arms and motto were cut on trees at the farthest landings.", "basis": "deeds",
         "source": "Zurara, ch. 78 and the Cape Verde chapters"},
        {"id": "D7", "claim": "Letters patent of 1443 forbade sailing south of Bojador without his "
         "licence and gave him the royal fifth.", "basis": "deeds",
         "source": "Monumenta Henricina VIII; Russell, Prince Henry 'the Navigator' (2000)"},
        {"id": "D8", "claim": "At Tangier in 1437 he closed the siege camp off from the beach against "
         "the king's instruction; the army was starved out and his brother Fernando left hostage "
         "(d. Fez 1443). The tactical picture is disputed by Elbl (2015).", "basis": "deeds",
         "source": "Rui de Pina, Chronica d'El-Rei D. Duarte; Russell 2000, 178-179"},
        {"id": "D9", "claim": "His own surviving texts state crusading and administrative positions; "
         "none gives a theory of knowing.", "basis": "primary",
         "source": "Parecer of 1436 and wills, Monumenta Henricina V and XIII"},
    ],
    "research_question": {"category": "robustness to spurious correlation",
        "question": ("Can a learner recover a hazard map over covariates that are observed only on "
                     "survival, from a ledger of departures and returned logs, and does that beat a "
                     "direct outcome classifier on what is visible in advance when the sea shifts?")},
    "mechanism": {
        "name": "Tornaviagem (return-gated ledger learner)",
        "family": "active learning and exploration / selection models for censored covariates",
        "signature_modules": ["rol", "sonda", "legua"],
        "closest_prior_art": ["Heckman sample-selection model (Econometrica 1979)",
                              "Wald 1943 survivorship analysis (Mangel and Samaniego, JASA 1984)",
                              "Noisy-OR multiple-instance learning (Maron and Lozano-Perez, NIPS 1998)",
                              "Safe exploration with returnability (Moldovan and Abbeel, ICML 2012; "
                              "Turchetta, Berkenkamp and Krause, NeurIPS 2016)",
                              "Go-Explore (Ecoffet et al., Nature 2021)",
                              "One-bit failure signals under termination (arXiv:2608.26571)"],
        "overlap": "Medium",
        "prior_art_queries": ["survivorship bias learning lost episodes missing not at random logs",
                              "safe exploration returnability irreversible states",
                              "selective labels apple tasting one-sided feedback",
                              "dogmatic prior never explores universal prior"],
        "contribution_type": "mechanism",
        "delta": ("Failure covariates are never observed: the non-return term integrates the hazard "
                  "over a learned extrapolation from the last returned log, fitted jointly with it; "
                  "one unknown per probe keeps every loss attributable."),
    },
    "traceability": [
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D7", "mechanism": "M1", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D1", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-RIVAL"},
        {"doctrine": "D3", "mechanism": "M4", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D8", "mechanism": "M4", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Tornaviagem beats a size-matched direct outcome classifier on "
         "shifted coasts.", "metric": "net campaign score", "split": "shifted",
         "comparison": "model - direct", "direction": "greater", "mesi": 0.03, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the departure ledger (survivors only) hurts more than "
         "removing the visible-sign inputs.", "metric": "net campaign score", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 0.05,
         "seeds": 5},
        {"id": "H-BLIND", "statement": "Where the dread sign marks a true wall, Tornaviagem scores "
         "below the lore-keeper that refuses to sail.", "condition": "blind coasts",
         "grounding": "Tangier 1437: the override that beat a false fear also overrode true warnings",
         "metric": "net campaign score", "comparison": "model - lore_keeper", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "With scarce hulls, one stretch per sailing beats the long "
         "leap (chapter 0440) on shifted coasts.", "metric": "net campaign score", "split": "shifted",
         "comparison": "model - leap", "direction": "greater", "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.20, "margin_over_trivial": 0.08},
    "probe_predictions": [
        {"probe": "P7", "expected": "above baseline"}, {"probe": "P8", "expected": "above baseline"},
        {"probe": "P10", "expected": "equal to baseline"}, {"probe": "P1", "expected": "below baseline"},
        {"probe": "P3", "expected": "below baseline"},
    ],
    "dialectic_links": [
        {"chapter": 440, "relation": "successor", "test": "H-RIVAL"},
        {"chapter": 429, "relation": "successor", "test": "none (uses the routes; no test)"},
        {"chapter": 387, "relation": "rival", "test": "none (state fleets of the same decades)"},
        {"chapter": 290, "relation": "successor", "test": "none (inherited geography; no test)"},
    ],
    "corpus_neighbors": [
        {"chapter": 440, "similarity": None, "difference": "Da Gama models the medium and gives up "
         "feedback for a long leap; here feedback is bought stretch by stretch and the leap is the rival."},
        {"chapter": 429, "similarity": None, "difference": "Columbus splits state by observability at "
         "sea; here the unobserved thing is the failed probe's whole record."},
        {"chapter": 387, "similarity": None, "difference": "Zheng He commits everything before "
         "observing; here commitment is one stretch and the ledger is revised at every return."},
        {"chapter": 385, "similarity": None, "difference": "Christine de Pizan asks who decided what "
         "was written; here the outcome itself deletes the record and a departure ledger restores it."},
        {"chapter": 382, "similarity": None, "difference": "Vytautas routes pay through an outcome-gated "
         "purse; no pay rule here, only a likelihood for censored outcomes."},
        {"chapter": 396, "similarity": None, "difference": "Ulugh Beg reads error structure in a "
         "finished catalogue; here data arrive only if the probe survives."},
        {"chapter": 390, "similarity": None, "difference": "Al-Kashi fixes tolerance first; here the "
         "risk budget is a go threshold, not a certified error."},
        {"chapter": 504, "similarity": None, "difference": "Philip II reads dispatches and cannot "
         "retract; here the harbour's problem is the dispatch that never arrives."},
    ],
    "barometer": {
        "cognitive_processing": ["learning curves from censored logs"],
        "embodied_cognition": [],
        "world_modeling": ["hazard map out of the survivors' support", "tracking when the coast's "
                           "statistics shift"],
        "consciousness": ["calibration of predicted loss rate against losses"],
        "language_understanding": [], "emotional_intelligence": [], "creativity": [],
        "autonomy": ["blocked-path campaigns with scarce hulls"],
    },
    "task_types": ["episodic_control", "vector_classification"],
    "applications": [
        {"use": "hazard maps for profiling-float and glider fleets whose lost units stop reporting",
         "sector": "ocean observation", "dataset": "Argo GDAC float profiles and metadata"},
        {"use": "failure prediction when the last telemetry before failure is missing",
         "sector": "data-centre operations", "dataset": "Backblaze Drive Stats"},
        {"use": "credit decisions with outcomes seen only for accepted applicants (research only)",
         "sector": "finance", "dataset": "LendingClub accepted and rejected loans 2007-2018"},
    ],
    "safety_notes": ("Synthetic data only. Exploration is framed as protecting probes and people; no "
                     "targeting, raiding or surveillance. The historical enterprise began the Atlantic "
                     "slave trade; nothing here models people as cargo. No generated sentence is "
                     "attributed to Henry. Similarity values are null: earlier chapter files were not "
                     "available in this session."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 397
TASK_TYPES = MIND_CARD["task_types"]
THRESH = MIND_CARD["thresholds"]
N_CELLS, N_LANES, DU, DV = 12, 2, 3, 3      # stretches per coast, lanes, in-situ dims, visible dims
CAPE = 1                                    # the stretch that carries the dread sign on every coast
HULLS, SAILINGS, KAPPA = 4, 20, 2.0         # ships per coast, sailing seasons, leagues a lost hull costs
LEAP = 3                                    # stretches per sailing for the long-leap rival (chapter 0440)
ALPHA = 4.0                                 # weight of the learner's guess in the ledger's Beta prior
LANE_MIX, Q_BIAS = 0.5, -2.2                # in-situ variance shared by both lanes; base log-odds of loss
SPLITS = {                                  # rho: memory along the coast; vis: blur of the visible sign
    "train": {"rho": 0.8, "vis": 0.5, "rough": 0.0, "wall": False},
    "heldout": {"rho": 0.8, "vis": 0.5, "rough": 0.0, "wall": False},
    "shifted": {"rho": 0.5, "vis": 1.0, "rough": 0.5, "wall": False},
    "blind": {"rho": 0.8, "vis": 0.5, "rough": 0.0, "wall": True},
}
K_MC, K_DEC, SD_FLOOR = 8, 128, 0.05        # frozen draws per unreturned record / per decision; sd floor
HID, HID_DIRECT = 24, 19                    # 263 and 268 trainable numbers (within two per cent)
LR, BATCH, CLIP = 5e-3, 256, 5.0
STEPS = {"full": 2000, "quick": 600}
MUT_STEPS = {"full": 300, "quick": 200}
COASTS = {"full": (500, 128), "quick": (200, 48)}    # training coasts, coasts per evaluation split
BUDGET = {"full": 180.0, "quick": 20.0}
GRAD_TOL, C4_BAND, Z_TOL, LEDGER_TOL = 1e-5, 0.06, 5.0, 0.01
MUTANT = None                               # set by --mutant or by the internal mutation run
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, sign=1.0):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= sign * self.lr * mh / (np.sqrt(vh) + self.eps)


def clip_global_norm(grads, max_norm):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        for k in grads:
            grads[k] *= max_norm / (norm + 1e-12)
    return norm


def fd_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=2e-5):
    """Central differences on >= n_entries random entries + the largest-gradient entry.
    Round-off in (L+ - L-)/(2 eps) is ~1e-10 for eps=1e-6 and L~1, so the relative error's
    denominator is floored at 2e-5: gradients below that are compared to 2e-10 absolute.
    Returns (worst floored relative error, worst relative error on entries |g| > 1e-4)."""
    worst, resolved = {}, {}
    for k, p in params.items():
        flat = p.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        errs, errs_big = [], [0.0]
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[k].reshape(-1)[i]
            errs.append(abs(ana - num) / max(abs(ana) + abs(num), floor))
            if abs(ana) > 1e-4:
                errs_big.append(abs(ana - num) / (abs(ana) + abs(num)))
        worst[k], resolved[k] = max(errs), max(errs_big)
    return worst, resolved


def paired_bootstrap(diffs, rng, n_boot=2000):
    diffs = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def verdict(mean, ci, direction, mesi):
    lo, hi = ci
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_report(lines, report, json_path):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    for line in lines:
        print(line)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks: the coast
def _ar1(rng, shape, rho):
    """Gaussian AR(1) along the coast (axis 1), unit variance at every stretch."""
    x = rng.normal(size=shape)
    for c in range(1, shape[1]):
        x[:, c] = rho * x[:, c - 1] + math.sqrt(1.0 - rho * rho) * x[:, c]
    return x


def env_reset(rng, split="train", n_coasts=1):
    """Draw coasts. u: in-situ conditions, seen only by a ship that returns; v: signs visible from the
    harbour; q: chance of losing a ship in a stretch, a fixed law of u alone; dice: one draw per attempt,
    fixed here so that every policy meets the same luck (common random numbers)."""
    cfg, B = SPLITS[split], n_coasts
    u = (math.sqrt(LANE_MIX) * _ar1(rng, (B, N_CELLS, 1, DU), cfg["rho"])
         + math.sqrt(1.0 - LANE_MIX) * _ar1(rng, (B, N_CELLS, N_LANES, DU), cfg["rho"]))
    u[..., 2] += cfg["rough"]
    q = np.clip(sigmoid(1.4 * u[..., 0] * u[..., 1] + 1.1 * u[..., 2] + Q_BIAS), 0.02, 0.92)
    dread = np.zeros((B, N_CELLS, N_LANES))
    dread[:, CAPE, :] = 1.0                                         # the cape every pilot fears
    dread[np.arange(B), rng.integers(CAPE + 2, N_CELLS, size=B), rng.integers(0, N_LANES, size=B)] = 1.0
    v = np.stack([u[..., 0] + cfg["vis"] * rng.normal(size=q.shape),       # a blurred view of one condition
                  2.4 * dread + 0.3 * rng.normal(size=q.shape),            # the dread sign
                  rng.normal(size=q.shape)], axis=-1)                      # nuisance
    if cfg["wall"]:
        q[:, CAPE, :] = 0.92                                        # blind coasts: here the old fear is right
    return {"u": u, "v": v, "q": q, "dice": rng.random((B, N_CELLS, N_LANES, SAILINGS)),
            "front": np.zeros(B, dtype=int), "hulls": np.full(B, HULLS), "log": np.zeros((B, DU)),
            "chart": np.full((B, N_CELLS), -1), "att": np.zeros((B, N_CELLS, N_LANES)),
            "loss": np.zeros((B, N_CELLS, N_LANES)), "entered": np.zeros(B)}


def env_step(state, action, rng=None):
    """One sailing season on every coast. action: go (B,) bool, lane (B,) int, reach (stretches per
    sailing, default 1). A ship that survives every stretch of its reach returns with their logs. A lost
    ship returns nothing, and the ledger can only charge its loss evenly to the stretches it was sent
    into. rng is unused: the dice were drawn at reset."""
    B = len(state["front"])
    idx = np.arange(B)
    lane, reach, start = action["lane"], int(action.get("reach", 1)), state["front"].copy()
    go = action["go"] & (state["hulls"] > 0) & (start < N_CELLS)
    m_eff = np.where(go, np.minimum(reach, N_CELLS - start), 0)
    lost = np.zeros(B, dtype=bool)
    for i in range(reach):
        cell = np.minimum(start + i, N_CELLS - 1)
        die = state["dice"][idx, cell, lane, state["att"][idx, cell, lane].astype(int) % SAILINGS]
        lost |= (i < m_eff) & ~lost & (die < state["q"][idx, cell, lane])
    for i in range(reach):
        on = i < m_eff
        state["att"][idx[on], (start + i)[on], lane[on]] += 1.0
        state["loss"][idx[on], (start + i)[on], lane[on]] += lost[on] / m_eff[on]
        back = on & ~lost
        state["chart"][idx[back], (start + i)[back]] = lane[back]
    ok = go & ~lost
    state["log"][ok] = state["u"][idx[ok], (start + m_eff - 1)[ok], lane[ok]]
    state["front"] += np.where(ok, m_eff, 0)
    state["hulls"] -= lost
    state["entered"] += m_eff
    reward = np.where(ok, m_eff, 0.0) - KAPPA * lost
    return state, reward, (state["hulls"] <= 0) | (state["front"] >= N_CELLS)


def leg_inputs(v, log, has, same):
    """What the harbour knows before a ship enters a stretch: its visible signs and the last returned
    log, routed to the in-lane or the out-of-lane block of the extrapolator's input."""
    m_in, m_out = has * same, has * (1.0 - same)
    zg = np.concatenate([log * m_in[:, None], log * m_out[:, None], m_in[:, None], m_out[:, None], v], axis=1)
    return {"v": v, "zg": zg, "kind": (m_in + 2.0 * m_out).astype(int)}


def ledger_book(state, rng):
    """M1 (rol). One record per (coast, stretch, lane) that ships were sent into: how many did not
    return and, if one returned, its log. The column u holds the simulator's truth for every record,
    but obs marks where the harbour may read it; nothing downstream may read u where obs is 0 (C6.1)."""
    b, c, lane = np.nonzero(state["att"] > 0)
    prev = np.maximum(c - 1, 0)
    plane = state["chart"][b, prev]
    keep = (c == 0) | (plane >= 0)                  # a leap's far stretches have no known approach: skipped
    b, c, lane, prev, plane = b[keep], c[keep], lane[keep], prev[keep], plane[keep]
    has = (c > 0).astype(float)
    log = state["u"][b, prev, np.maximum(plane, 0)] * has[:, None]
    book = leg_inputs(state["v"][b, c, lane], log, has, (plane == lane).astype(float))
    back = (state["chart"][b, c] == lane).astype(float)
    book.update(r=back, obs=back.copy(), lost=state["loss"][b, c, lane], u=state["u"][b, c, lane],
                eps=rng.normal(size=(len(back), K_MC, DU)))
    return book


def sail_logbook(rng, split, n_coasts):
    """The inherited record: fleets that always sail, one stretch at a time, in a lane picked by chance."""
    state = env_reset(rng, split, n_coasts)
    for _ in range(SAILINGS):
        env_step(state, {"go": np.ones(n_coasts, dtype=bool), "lane": rng.integers(0, N_LANES, size=n_coasts)})
    return state, ledger_book(state, rng)


def probe_book(X, y, rng):
    """Probe mode: every feature is visible, nothing is ever seen in situ, class 1 plays the non-return."""
    n = len(X)
    return {"v": X, "zg": X, "kind": np.zeros(n, dtype=int), "r": 1.0 - y, "obs": np.zeros(n), "lost": 1.0 * y,
            "u": np.zeros((n, DU)), "eps": rng.normal(size=(n, K_MC, DU))}


def take(book, idx):
    return {k: a[idx] for k, a in book.items()}


def make_data(seed_seq, mode):
    """Training ledger, a held-out ledger sailed the same way, and seeds for the campaign coasts."""
    kids = seed_seq.spawn(5)
    n_train, n_eval = COASTS[mode]
    train_state, train = sail_logbook(np.random.default_rng(kids[0]), "train", n_train)
    held_state, held = sail_logbook(np.random.default_rng(kids[1]), "heldout", n_eval)
    return {"train": train, "heldout": held, "states": {"train": train_state, "heldout": held_state},
            "coast_seeds": dict(zip(("heldout", "shifted", "blind"), kids[2:])), "n_eval": n_eval,
            "p0": float(train["lost"].sum() / (train["lost"].sum() + train["r"].sum()))}


# ---------------------------------------------------------------- model: Tornaviagem
def build_model(in_dim, out_dim, task_type, rng, kind="tornaviagem", hidden=None):
    """kind 'tornaviagem': hazard net over (visible, in-situ) inputs plus a Gaussian extrapolator of the
    in-situ part. kind 'direct': the size-matched baseline, an outcome classifier on what is known
    before departure. In episodic_control, in_dim counts the visible signs and the last-log block is
    added to the extrapolator's input; in vector_classification every feature is visible (probe mode)."""
    if task_type not in TASK_TYPES or out_dim != 2:
        raise ValueError("supported: two-outcome episodic_control or vector_classification")
    dz = in_dim + (2 * DU + 2 if task_type == "episodic_control" else 0)
    if kind == "direct":
        h = hidden or HID_DIRECT
        P = {"A1": rng.normal(0.0, dz ** -0.5, (dz, h)), "c1": np.zeros(h),
             "A2": rng.normal(0.0, h ** -0.5, (h, 2)), "c2": np.zeros(2)}
    else:
        h = hidden or HID
        P = {"W1": rng.normal(0.0, (in_dim + DU) ** -0.5, (in_dim + DU, h)), "b1": 0.1 * rng.normal(size=h),
             "W2": rng.normal(0.0, h ** -0.5, (h, 2)), "b2": np.zeros(2),
             "Wg": 0.1 * rng.normal(size=(dz, DU)), "bg": np.zeros(DU), "om": np.full((3, DU), 0.5)}
    return {"kind": kind, "params": P, "ko": frozenset(), "eps_dec": rng.normal(size=(K_DEC, DU))}


def _inputs(model, book):
    """Input-side knockouts: vista hides the visible signs; sonda hides everything the guess uses."""
    v, zg, kind = book["v"], book["zg"], book["kind"]
    if "vista" in model["ko"]:
        v, zg = np.zeros_like(v), zg.copy()
        zg[:, -v.shape[1]:] = 0.0
    if "sonda" in model["ko"]:
        zg, kind = np.zeros_like(zg), np.zeros_like(kind)
    return v, zg, kind


def _sonda(P, zg, kind):
    """M2. Guess of the unseen in-situ conditions: mean linear in the last returned log and the visible
    signs; spread by case (no log / log from this lane / log from the other lane)."""
    return zg @ P["Wg"] + P["bg"], softplus(P["om"][kind]) + SD_FLOOR


def _hazard(P, v, u):
    """Log-chance of return (column 0) and of non-return (column 1) given visible and in-situ inputs."""
    x = np.concatenate([v, u], axis=-1)
    a = np.tanh(x @ P["W1"] + P["b1"])
    z = a @ P["W2"] + P["b2"]
    return x, a, z - logsumexp(z, axis=-1, keepdims=True)


def _hazard_back(P, g, x, a, dz):
    """Accumulate hazard-net gradients for output gradient dz; return the gradient at the inputs."""
    h = a.shape[-1]
    g["W2"] += a.reshape(-1, h).T @ dz.reshape(-1, 2)
    g["b2"] += dz.reshape(-1, 2).sum(axis=0)
    da = (dz @ P["W2"].T) * (1.0 - a * a)
    g["W1"] += x.reshape(-1, x.shape[-1]).T @ da.reshape(-1, h)
    g["b1"] += da.reshape(-1, h).sum(axis=0)
    return da @ P["W1"].T


def _ledger_loss(model, book):
    """Negative log-likelihood of the ledger, per record. A record with a returned log (obs) is scored
    at the water it reports: the Gaussian density of that water, one return, and every earlier
    non-return at the same water. A record with no return has no water to score at, so its non-returns
    are averaged over K frozen draws of the guess: -log mean_k P(lost | v, mu + sd eps_k) ** losses."""
    P = model["params"]
    v, zg, kind = _inputs(model, book)
    n, o = len(kind), book["obs"] > 0.5
    r, lost, wt = book["r"], book["lost"], np.full(len(kind), 1.0 / len(kind))
    if "rol" in model["ko"]:                        # survivors' accounts only: no losses, no silent records
        lost, wt = np.zeros(n), r / n
    counts = np.stack([r, lost], axis=1)            # returns and non-returns each record must explain
    mu, sd = _sonda(P, zg, kind)
    g = {k: np.zeros_like(p) for k, p in P.items()}
    dmu, dsd = np.zeros_like(mu), np.zeros_like(mu)
    res = (book["u"][o] - mu[o]) / sd[o]            # --- records with a returned log
    xo, ao, lpo = _hazard(P, v[o], book["u"][o])
    loss = np.sum(wt[o] * (np.sum(0.5 * res * res + np.log(sd[o]), axis=1) - np.sum(counts[o] * lpo, axis=1)))
    _hazard_back(P, g, xo, ao, (counts[o].sum(axis=1, keepdims=True) * np.exp(lpo) - counts[o]) * wt[o][:, None])
    dmu[o] = -(res / sd[o]) * wt[o][:, None]
    dsd[o] = ((1.0 - res * res) / sd[o]) * wt[o][:, None]
    c, eps = ~o, book["eps"][~o]                    # --- records with no return: integrate over the guess
    uk = mu[c][:, None, :] + sd[c][:, None, :] * eps
    xc, ac, lpc = _hazard(P, np.broadcast_to(v[c][:, None, :], uk.shape[:2] + (v.shape[1],)), uk)
    lk = np.sum(counts[c][:, None, :] * lpc, axis=2)                # log-likelihood under each draw
    loss -= np.sum(wt[c] * (logsumexp(lk, axis=1) - math.log(eps.shape[1])))
    wk = softmax(lk, axis=1) * wt[c][:, None]                       # each draw's share of the record
    dzc = (counts[c].sum(axis=1)[:, None, None] * np.exp(lpc) - counts[c][:, None, :]) * wk[:, :, None]
    duk = _hazard_back(P, g, xc, ac, dzc)[..., -DU:]
    dmu[c], dsd[c] = duk.sum(axis=1), (duk * eps).sum(axis=1)
    g["Wg"], g["bg"] = zg.T @ dmu, dmu.sum(axis=0)
    np.add.at(g["om"], kind, dsd * sigmoid(P["om"][kind]))
    return float(loss), g


# ---------------------------------------------------------------- baseline and shared interface
def _direct_loss(model, book):
    """Baseline: binomial log-loss of returns and non-returns given what was known before departure."""
    P = model["params"]
    zg = _inputs(model, book)[1]
    counts = np.stack([book["r"], book["lost"]], axis=1) / len(zg)
    a = np.tanh(zg @ P["A1"] + P["c1"])
    z = a @ P["A2"] + P["c2"]
    lp = z - logsumexp(z, axis=1, keepdims=True)
    dz = counts.sum(axis=1, keepdims=True) * np.exp(lp) - counts
    da = (dz @ P["A2"].T) * (1.0 - a * a)
    return float(-np.sum(counts * lp)), {"A1": zg.T @ da, "c1": da.sum(axis=0), "A2": a.T @ dz, "c2": dz.sum(axis=0)}


def loss_and_grads(model, batch):
    loss, g = (_direct_loss if model["kind"] == "direct" else _ledger_loss)(model, batch)
    if MUTANT and MUTANT.startswith("zero_grad_") and MUTANT[10:] in g:
        g[MUTANT[10:]] = np.zeros_like(g[MUTANT[10:]])
    return loss, g


def _as_book(X):
    return X if isinstance(X, dict) else {"v": X, "zg": X, "kind": np.zeros(len(X), dtype=int)}


def hidden_states(model, X):
    """Internal representations: the guessed in-situ mean and spread, and the hazard net's hidden layer."""
    P = model["params"]
    v, zg, kind = _inputs(model, _as_book(X))
    if model["kind"] == "direct":
        return {"hidden": np.tanh(zg @ P["A1"] + P["c1"])}
    mu, sd = _sonda(P, zg, kind)
    return {"sonda_mu": mu, "sonda_sd": sd, "hazard_hidden": _hazard(P, v, mu)[1]}


def predict(model, X):
    """Chance of non-return judged before departure (probability of class 1 in probe mode)."""
    P = model["params"]
    v, zg, kind = _inputs(model, _as_book(X))
    if model["kind"] == "direct":
        return softmax(np.tanh(zg @ P["A1"] + P["c1"]) @ P["A2"] + P["c2"], axis=1)[:, 1]
    mu, sd = _sonda(P, zg, kind)
    uk = mu[:, None, :] + sd[:, None, :] * model["eps_dec"][None]
    vk = np.broadcast_to(v[:, None, :], uk.shape[:2] + (v.shape[1],))
    return np.exp(_hazard(P, vk, uk)[2][..., 1]).mean(axis=1)


def n_params(model):
    return int(sum(p.size for p in model["params"].values()))


# ---------------------------------------------------------------- registries: modules, knockouts, mutants
def modules(model):
    """Name -> parameters, plain technical role, signature flag, and what a knockout substitutes."""
    if model["kind"] == "direct":
        return {"classifier": {"params": ["A1", "c1", "A2", "c2"], "signature": False, "knockout": "none",
                               "role": "MLP outcome classifier on pre-departure features"}}
    return {
        "rol": {"params": [], "signature": True, "knockout": "identity",
                "role": "departure ledger: censored-outcome likelihood term and per-cell Beta counts"},
        "sonda": {"params": ["Wg", "bg", "om"], "signature": True, "knockout": "mean",
                  "role": "conditional Gaussian extrapolator of covariates observed only on survival"},
        "legua": {"params": [], "signature": True, "knockout": "identity",
                  "role": "frontier ordering: one unobserved cell per probe"},
        "vista": {"params": [], "signature": False, "knockout": "zero",
                  "role": "pre-departure visible inputs to both networks"},
        "hazard": {"params": ["W1", "b1", "W2", "b2"], "signature": False, "knockout": "none",
                   "role": "MLP hazard classifier over visible and in-situ covariates"},
    }


def knockout(model, name, mode=None):
    """Copy of the model with one module replaced. rol: only survivors' accounts pass (identity on what
    returned); sonda: the guess falls back to its prior mean; legua: LEAP cells pass per probe; vista:
    visible inputs are zeroed. Learned modules are knocked out before training, never after."""
    spec = modules(model).get(name)
    if spec is None or spec["knockout"] == "none" or mode not in (None, spec["knockout"]):
        raise ValueError(f"no such knockout: {name} / {mode}")
    return dict(model, params={k: p.copy() for k, p in model["params"].items()}, ko=model["ko"] | {name})


MUTANTS = {"sign_flip": "optimizer ascends the loss", "zero_lr": "learning rate set to zero",
           "zero_grad_W1": "hazard-net input gradient zeroed", "zero_grad_Wg": "extrapolator gradient zeroed"}


# ---------------------------------------------------------------- training and campaigns
def fit(model, data, budget, rng, lr=LR):
    """Adam with global-norm clipping on random batches of ledger records, for a fixed update budget."""
    P = model["params"]
    opt = Adam(P, lr=0.0 if MUTANT == "zero_lr" else lr)
    n = len(data["kind"])
    probe = take(data, np.arange(min(n, 1024)))
    loss0 = loss_and_grads(model, probe)[0]
    for _ in range(budget):
        loss, g = loss_and_grads(model, take(data, rng.integers(0, n, size=min(BATCH, n))))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, CLIP)
        opt.step(P, g, -1.0 if MUTANT == "sign_flip" else 1.0)
    return {"loss0": loss0, "loss_end": loss_and_grads(model, probe)[0]}


def ledger_mean(p, att, loss):
    """M1 (rol), the non-gradient rule: posterior mean chance of loss at one (stretch, lane) under a
    Beta(ALPHA p, ALPHA (1 - p)) prior after `att` departures of which `loss` did not return."""
    return (ALPHA * p + loss) / (ALPHA + att)


def lane_risk(model, state, lane, reach, p0):
    """Chance that a ship sent `reach` stretches past the last return, in `lane`, does not come back.
    M3 (legua) is reach 1: one unknown per sailing. A longer reach must guess each further stretch from
    the guess before it, since no log exists yet."""
    B = len(state["front"])
    idx = np.arange(B)
    log, has = state["log"], (state["front"] > 0).astype(float)
    same = (state["chart"][idx, np.maximum(state["front"] - 1, 0)] == lane).astype(float)
    keep = np.ones(B)
    for i in range(reach):
        cell = np.minimum(state["front"] + i, N_CELLS - 1)
        X = leg_inputs(state["v"][idx, cell, lane], log, has, same)
        p = np.full(B, p0) if model is None else predict(model, X)
        if model is not None and reach > 1:
            log, has, same = hidden_states(model, X)["sonda_mu"], np.ones(B), np.ones(B)
        if model is None or "rol" not in model["ko"]:
            p = ledger_mean(p, state["att"][idx, cell, lane], state["loss"][idx, cell, lane])
        keep *= np.where(state["front"] + i < N_CELLS, 1.0 - p, 1.0)
    return 1.0 - keep


def campaign(model, split, n_coasts, coast_seed, lore=False, p0=0.2):
    """Twenty seasons on a batch of coasts. Each season the harbour judges both lanes and sends a ship
    where the expected gain is not negative (risk <= reach / (reach + KAPPA)). Within a campaign the
    weights stay as inherited; what is learned is what comes back: the last returned log feeds the next
    guess and every departure updates the ledger. model None is the constant-rate harbour; lore adds
    the refusal to enter water under the dread sign."""
    state = env_reset(np.random.default_rng(coast_seed), split, n_coasts)
    reach = LEAP if model is not None and "legua" in model["ko"] else 1
    idx = np.arange(n_coasts)
    for _ in range(SAILINGS):
        risk = np.stack([lane_risk(model, state, np.full(n_coasts, k), reach, p0) for k in range(N_LANES)], axis=1)
        if lore:
            sign = state["v"][idx, np.minimum(state["front"], N_CELLS - 1), :, 1]
            risk = np.where(sigmoid(4.0 * (sign - 1.5)) > 0.5, 1.0, risk)
        lane = risk.argmin(axis=1)
        m_eff = np.clip(N_CELLS - state["front"], 1, reach)
        env_step(state, {"go": risk[idx, lane] <= m_eff / (m_eff + KAPPA), "lane": lane, "reach": reach})
    return float(np.mean(state["front"] - KAPPA * (HULLS - state["hulls"])) / N_CELLS), state


def auc(score, label):
    """Rank AUC: chance that a record with a loss gets the higher predicted hazard (0.5 = uninformed)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(label.sum())
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / max(n1 * (len(label) - n1), 1))


ARMS = {"heldout": ("model", "direct", "const", "lore"),
        "shifted": ("model", "direct", "leap", "ko_rol", "ko_sonda", "ko_vista", "const", "lore"),
        "blind": ("model", "direct", "lore")}


def train_arms(data, seed_seq, steps):
    """Full learner, baseline and the three learned knockouts from one initialisation and one batch order."""
    k_init, k_fit = seed_seq.spawn(2)
    model = build_model(DV, 2, "episodic_control", np.random.default_rng(k_init))
    arms = {"model": model,
            "direct": build_model(DV, 2, "episodic_control", np.random.default_rng(k_init), kind="direct")}
    arms.update({"ko_" + name: knockout(model, name) for name in ("rol", "sonda", "vista")})
    info = {name: fit(m, data["train"], steps, np.random.default_rng(k_fit)) for name, m in arms.items()}
    arms["leap"] = knockout(model, "legua")         # same trained weights, long-leap ordering (chapter 0440)
    return arms, info


def run_seed(seed, mode):
    k_data, k_arms = np.random.SeedSequence([CHAPTER, seed]).spawn(2)
    data = make_data(k_data, mode)
    arms, info = train_arms(data, k_arms, STEPS[mode])
    scores = {}
    for split, names in ARMS.items():
        for name in names:
            scores[f"{split}/{name}"] = campaign(arms.get(name), split, data["n_eval"], data["coast_seeds"][split],
                                                 lore=name == "lore", p0=data["p0"])[0]
    label = data["heldout"]["lost"] > 0
    res = {"scores": scores, "info": info["model"], "auc": auc(predict(arms["model"], data["heldout"]), label),
           "auc_direct": auc(predict(arms["direct"], data["heldout"]), label)}
    return res, data, arms


# ---------------------------------------------------------------- tests: correctness
def gradcheck(model, batch, rng):
    _, g = loss_and_grads(model, batch)
    worst, resolved = fd_check(lambda: loss_and_grads(model, batch)[0], model["params"], g, rng)
    return max(worst.values()), max(resolved.values()), len(worst)


def c1_gradients(data, seed_seq, steps=60):
    """Every tensor of the ledger learner (native and probe mode) and of the baseline, at initialisation
    and again after `steps` updates. The Monte Carlo draws are frozen inside the records."""
    rng = np.random.default_rng(seed_seq)
    native = take(data["train"], np.arange(64))
    X = rng.normal(size=(64, 5))
    probe = probe_book(X, (X[:, 0] * X[:, 1] > 0).astype(float), rng)
    cases = [("tornaviagem", "episodic_control", DV, data["train"], native),
             ("direct", "episodic_control", DV, data["train"], native),
             ("tornaviagem", "vector_classification", 5, probe, probe)]
    worst, resolved, total = 0.0, 0.0, 0
    for kind, task, dim, train, batch in cases:
        model = build_model(dim, 2, task, rng, kind=kind)
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(model, train, steps, rng)
            w, r, n = gradcheck(model, batch, rng)
            worst, resolved, total = max(worst, w), max(resolved, r), total + n
    return {"tensors_checked": total, "tensors_total": total, "max_rel_error": worst,
            "max_rel_error_resolved": resolved, "checked_at": ["init", f"after_{steps}_steps"],
            "passed": bool(worst <= GRAD_TOL)}


def c2_determinism(data, seed_seq):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(seed_seq)
        model = build_model(DV, 2, "episodic_control", rng)
        end = fit(model, data["train"], 40, rng)["loss_end"]
        runs.append((end, campaign(model, "heldout", 16, seed_seq)[0], predict(model, data["heldout"]), model))
    same = runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] and np.array_equal(runs[0][2], runs[1][2])
    finite = all(np.all(np.isfinite(p)) for p in runs[0][3]["params"].values()) and \
        all(np.all(np.isfinite(h)) for h in hidden_states(runs[0][3], data["heldout"]).values())
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": f"identical reruns (loss, campaign score, predictions): {same}; finite: {finite}"}


def c3_learning(first):
    drop = 1.0 - first["info"]["loss_end"] / first["info"]["loss0"]
    ok = drop >= THRESH["loss_drop_fraction"] and first["auc"] >= 0.5 + THRESH["margin_over_trivial"]
    return {"id": "C3", "name": "learning", "passed": bool(ok),
            "detail": f"ledger loss fell {drop:.2f} (need {THRESH['loss_drop_fraction']:.2f}); held-out AUC "
                      f"{first['auc']:.3f} vs trivial 0.500 (need +{THRESH['margin_over_trivial']:.2f}); "
                      f"direct baseline AUC {first['auc_direct']:.3f}"}


def c4_shuffled(data, seed_seq, steps):
    """Outcomes (returned or not, losses) permuted across records; the log is revealed by the permuted flag."""
    rng = np.random.default_rng(seed_seq)
    perm = rng.permutation(len(data["train"]["r"]))
    book = dict(data["train"], r=data["train"]["r"][perm], obs=data["train"]["r"][perm],
                lost=data["train"]["lost"][perm])
    model = build_model(DV, 2, "episodic_control", rng)
    fit(model, book, steps, rng)
    a = auc(predict(model, data["heldout"]), data["heldout"]["lost"] > 0)
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(abs(a - 0.5) <= C4_BAND),
            "detail": f"held-out AUC {a:.3f} with outcomes permuted; trivial 0.500, band +/-{C4_BAND:.2f}"}


def short_protocol(data, seed_seq, steps):
    """C1 at initialisation and C3 after a short run: what every mutant must break."""
    rng = np.random.default_rng(seed_seq)
    model = build_model(DV, 2, "episodic_control", rng)
    c1 = gradcheck(model, take(data["train"], np.arange(64)), rng)[0] <= GRAD_TOL
    info = fit(model, data["train"], steps, rng)
    a = auc(predict(model, data["heldout"]), data["heldout"]["lost"] > 0)
    c3 = info["loss_end"] <= (1.0 - THRESH["loss_drop_fraction"]) * info["loss0"] and \
        a >= 0.5 + THRESH["margin_over_trivial"]
    return bool(c1), bool(c3)


def c5_mutants(data, seed_seq, steps):
    global MUTANT
    clean = short_protocol(data, seed_seq, steps)
    caught = []
    for name in MUTANTS:
        MUTANT = name
        try:
            c1, c3 = short_protocol(data, seed_seq, steps)
        except FloatingPointError:
            c1, c3 = True, False
        MUTANT = None
        if not (c1 and c3):
            caught.append(name + (":C1" if not c1 else ":C3"))
    ok = all(clean) and len(caught) == len(MUTANTS)
    return {"detected": len(caught), "total": len(MUTANTS), "score": len(caught) / len(MUTANTS)}, \
        {"id": "C5", "name": "mutant_detection", "passed": bool(ok),
         "detail": f"{len(caught)}/{len(MUTANTS)} detected ({', '.join(caught)}); clean control passes C1,C3: {clean}"}


def _random_model(rng):
    model = build_model(DV, 2, "episodic_control", rng)
    for k, p in model["params"].items():
        if k != "om":
            p *= 2.0
            p += 0.3 * rng.normal(size=p.shape)
    return model


def prop_censoring(rng, broken=False, draws=5):
    """C6.1 Censoring integrity: nothing the learner computes may change when the water that sank a
    ship changes. Negative control: a learner that reads the lost ships' water (obs forced to 1)."""
    worst = 0.0
    for _ in range(draws):
        model = _random_model(rng)
        book = sail_logbook(rng, "train", 16)[1]
        if broken:
            book = dict(book, obs=np.ones_like(book["obs"]))
        hidden = book["r"][:, None] < 0.5
        moved = dict(book, u=np.where(hidden, book["u"] + rng.normal(scale=3.0, size=book["u"].shape), book["u"]))
        (l0, g0), (l1, g1) = loss_and_grads(model, book), loss_and_grads(model, moved)
        worst = max([worst, abs(l1 - l0)] + [float(np.abs(g1[k] - g0[k]).max()) for k in g0])
    return worst


def prop_score_at_truth(rng, broken=False, n=16000, groups=32, k_draws=64):
    """C6.2 The ledger likelihood is the true likelihood of censored records: for records sampled from
    the learner's own law (up to three ships per record, stop at the first return, lost ships' water
    hidden) the mean gradient at the generating parameters must vanish. Returned: the largest |z|.
    Negative control: the survivors-only objective, whose score at the truth is far from zero."""
    model = _random_model(rng)
    if broken:
        model = knockout(model, "rol")
    P = model["params"]
    kind = rng.integers(0, 3, size=n)
    book = leg_inputs(rng.normal(size=(n, DV)), rng.normal(size=(n, DU)), 1.0 * (kind > 0), 1.0 * (kind == 1))
    mu, sd = _sonda(P, book["zg"], book["kind"])
    u = mu + sd * rng.normal(size=mu.shape)
    p_lost = np.exp(_hazard(P, book["v"], u)[2][:, 1])
    lost, back, sending = np.zeros(n), np.zeros(n), np.ones(n, dtype=bool)
    for _ in range(HULLS - 1):
        sunk = rng.random(n) < p_lost
        lost += sending & sunk
        back += sending & ~sunk
        sending &= sunk
    book.update(r=back, obs=back.copy(), lost=lost, u=u, eps=rng.normal(size=(n, k_draws, DU)))
    grads = [loss_and_grads(model, take(book, part))[1] for part in np.array_split(np.arange(n), groups)]
    flat = np.array([np.concatenate([g[k].ravel() for k in sorted(g)]) for g in grads])
    se = flat.std(axis=0, ddof=1) / math.sqrt(groups)
    return float(np.max(np.abs(flat.mean(axis=0)) / np.maximum(se, 1e-12)))


def prop_ledger(rng, broken=False, draws=5):
    """C6.3 The ledger conserves the fleet (every departure entered, every lost hull charged once) and
    its Beta rule equals the posterior mean found by simulation. Negative control: survivors'
    bookkeeping, which enters only the ships that came back."""
    worst = 0.0
    for _ in range(draws):
        B = 24
        state = env_reset(rng, "train", B)
        for _ in range(SAILINGS):
            env_step(state, {"go": rng.random(B) < 0.8, "lane": rng.integers(0, N_LANES, size=B),
                             "reach": int(rng.integers(1, LEAP + 1))})
        att, loss = state["att"].sum(axis=(1, 2)), state["loss"].sum(axis=(1, 2))
        if broken:
            att, loss = 1.0 * state["front"], np.zeros(B)
        worst = max(worst, float(np.abs(HULLS - state["hulls"] - loss).max()),
                    float(np.abs(att - state["entered"]).max()))
        p, n_att, n_loss = rng.uniform(0.1, 0.5), 3, int(rng.integers(1, 3))
        theta = rng.beta(ALPHA * p, ALPHA * (1.0 - p), size=200000)
        truth = float(theta[rng.binomial(n_att, theta) == n_loss].mean())
        mine = ledger_mean(p, n_att - n_loss, 0.0) if broken else ledger_mean(p, n_att, n_loss)
        worst = max(worst, abs(truth - mine))
    return worst


def c6_properties(rng):
    rows = []
    for cid, fn, tol, floor, what in (
            ("C6.1", prop_censoring, 0.0, 1e-3, "lost ships' water is read"),
            ("C6.2", prop_score_at_truth, Z_TOL, 2.0 * Z_TOL, "survivors-only objective"),
            ("C6.3", prop_ledger, LEDGER_TOL, 5.0 * LEDGER_TOL, "survivors' bookkeeping")):
        held, broke = fn(rng), fn(rng, broken=True)
        rows.append({"id": cid, "name": fn.__name__, "passed": bool(held <= tol and broke > floor),
                     "detail": f"violation {held:.3g} (tol {tol:g}); negative control '{what}' {broke:.3g} "
                               f"(must exceed {floor:g})"})
    return rows


def coast_keys(state):
    return {hashlib.sha256(np.round(c, 9).tobytes()).hexdigest() for c in state["u"]}


def c7_split_integrity(data):
    train = coast_keys(data["states"]["train"])
    evals = [coast_keys(data["states"]["heldout"])] + \
        [coast_keys(env_reset(np.random.default_rng(s), split, data["n_eval"])) for split, s in data["coast_seeds"].items()]
    clash = sum(len(train & e) for e in evals)
    return {"id": "C7", "name": "split_integrity", "passed": clash == 0,
            "detail": f"{len(train)} training coasts, {sum(len(e) for e in evals)} evaluation coasts, {clash} shared"}


def correctness_suite(first, data, mode, seed_seq):
    """C1-C7 on the first seed. Under --mutant only C1 and C3 run: they are what the mutant must break."""
    kids = seed_seq.spawn(5)
    grad = c1_gradients(data, kids[0])
    rows = [{"id": "C1", "name": "gradient_check", "passed": grad["passed"],
             "detail": f"max rel err {grad['max_rel_error']:.2e} (denominator floor 2e-5); on |g|>1e-4 "
                       f"entries {grad['max_rel_error_resolved']:.2e}; tol {GRAD_TOL:g}"}]
    mut = {"detected": 0, "total": len(MUTANTS), "score": 0.0}
    if MUTANT is not None:
        return rows + [c3_learning(first)], grad, mut
    rows += [c2_determinism(data, kids[1]), c3_learning(first), c4_shuffled(data, kids[2], STEPS[mode])]
    mut, c5 = c5_mutants(data, kids[3], MUT_STEPS[mode])
    rows += [c5] + c6_properties(np.random.default_rng(kids[4])) + [c7_split_integrity(data)]
    return rows, grad, mut


# ---------------------------------------------------------------- tests: hypotheses
PAIRS = {"H-SIG": ("shifted/model", "shifted/direct"), "H-NEC": ("shifted/ko_rol", "shifted/ko_vista"),
         "H-BLIND": ("blind/model", "blind/lore"), "H-RIVAL": ("shifted/model", "shifted/leap")}


def evaluate_hypotheses(per_seed, rng):
    """Paired per-seed differences as fixed in MIND_CARD; verdicts need at least five seeds."""
    out = []
    for h in MIND_CARD["hypotheses"]:
        a, b = PAIRS[h["id"]]
        mean, ci = paired_bootstrap([s["scores"][a] - s["scores"][b] for s in per_seed], rng)
        v = verdict(mean, ci, h["direction"], h["mesi"]) if len(per_seed) >= 5 else "not evaluated"
        out.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"],
                    "n_seeds": len(per_seed), "verdict": v})
    return out


def knockout_table(per_seed, rng, model):
    rows = []
    for name, arm in (("rol", "ko_rol"), ("sonda", "ko_sonda"), ("legua", "leap"), ("vista", "ko_vista")):
        mean, ci = paired_bootstrap([s["scores"]["shifted/" + arm] - s["scores"]["shifted/model"] for s in per_seed], rng)
        rows.append({"module": name, "signature": modules(model)[name]["signature"], "metric_change": mean, "ci95": ci})
    return rows


# ---------------------------------------------------------------- report
def format_report(rep, per_seed):
    L = [f"environment: Python {rep['environment']['python']} · NumPy {rep['environment']['numpy']}",
         f"seeds: {rep['seeds']}   runtime_s: {rep['runtime_s']:.1f}   card revision: {rep['card_revision']}",
         f"n_params: tornaviagem={rep['n_params']}, direct baseline={rep['n_params_baseline']}",
         "gradcheck: {tensors_checked}/{tensors_total} tensors, max rel err {max_rel_error:.2e}, at {checked_at}, "
         "passed={passed}".format(**rep["gradcheck"]), "correctness:"]
    L += [f"  {c['id']:<5} {c['name']:<27} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}" for c in rep["correctness"]]
    L.append("mutation score: {detected}/{total} = {score:.2f}".format(**rep["mutants"]))
    L.append("net campaign score, mean over seeds (stretches charted minus two per hull lost, over twelve):")
    L.append(f"  {'arm':<10}" + "".join(f"{s:>10}" for s in ARMS))
    for arm in ARMS["shifted"]:
        cells = [f"{np.mean([s['scores'][f'{sp}/{arm}'] for s in per_seed]):>10.3f}" if arm in ARMS[sp] else f"{'-':>10}"
                 for sp in ARMS]
        L.append(f"  {arm:<10}" + "".join(cells))
    L.append("  held-out AUC of pre-departure hazard: tornaviagem {:.3f}, direct {:.3f}, trivial 0.500".format(
        np.mean([s["auc"] for s in per_seed]), np.mean([s["auc_direct"] for s in per_seed])))
    L.append("hypotheses (paired differences in net campaign score):")
    for h in rep["hypotheses"]:
        L.append(f"  {h['id']:<8} diff {h['mean_diff']:+.3f}  ci95 [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]  "
                 f"mesi {h['mesi']:.2f}  n={h['n_seeds']}  -> {h['verdict']}")
    L.append("knockouts (shifted coasts, change in net campaign score against the full learner):")
    for k in rep["knockouts"]:
        L.append(f"  {k['module']:<6} signature={str(k['signature']):<5} {k['metric_change']:+.3f}  "
                 f"ci95 [{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    L.append(f"task types: {rep['task_types']}   exit code: {rep['exit_code']}")
    return L


def data_bridge(path, seed):
    """Optional: a local numeric CSV (header row; 0/1 label in the last column), for instance drive-days
    built from Backblaze Drive Stats. Runs the ledger learner in probe mode beside the direct classifier."""
    if not path or not os.path.exists(path):
        print(f"[data bridge] no local CSV at {path!r}; step skipped (nothing is ever downloaded).")
        return
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    X, y = raw[:, :-1], raw[:, -1]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    tr, te = order[:int(0.7 * len(y))], order[int(0.7 * len(y)):]
    for kind in ("tornaviagem", "direct"):
        model = build_model(X.shape[1], 2, "vector_classification", rng, kind=kind)
        fit(model, probe_book(X[tr], y[tr], rng), STEPS["quick"], rng)
        print(f"[data bridge] {kind}: held-out AUC {auc(predict(model, X[te]), y[te] > 0.5):.3f} on {len(te)} rows")


# ---------------------------------------------------------------- command-line entry point
def protocol(mode, seed, n_seeds, json_path, data_path):
    t0 = time.time()
    seeds = [seed + i for i in range(n_seeds)]
    print(f"[seeds] base={seed} seeds={seeds} mode={mode} mutant={MUTANT}")
    first, data, arms = run_seed(seeds[0], mode)
    correct, grad, mut = correctness_suite(first, data, mode, np.random.SeedSequence([CHAPTER, seeds[0], 7]))
    per_seed = [first] + [run_seed(s, mode)[0] for s in seeds[1:]]
    stat_rng = np.random.default_rng(np.random.SeedSequence([CHAPTER, seeds[0], 11]))
    hyps = evaluate_hypotheses(per_seed, stat_rng)
    runtime = time.time() - t0
    correct.append({"id": "C8", "name": "budget", "passed": runtime <= BUDGET[mode],
                    "detail": f"{runtime:.1f}s of {BUDGET[mode]:.0f}s"})
    code = 3 if runtime > BUDGET[mode] else (0 if all(c["passed"] for c in correct) else 1)
    rep = {"schema_version": "1.0", "chapter": CHAPTER, "file": os.path.basename(__file__),
           "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
           "seeds": seeds, "runtime_s": runtime, "n_params": n_params(arms["model"]),
           "n_params_baseline": n_params(arms["direct"]), "gradcheck": grad, "correctness": correct,
           "mutants": mut, "hypotheses": hyps, "knockouts": knockout_table(per_seed, stat_rng, arms["model"]),
           "scores_per_seed": [s["scores"] for s in per_seed], "task_types": TASK_TYPES, "exit_code": code}
    write_report(format_report(rep, per_seed), rep, json_path)
    data_bridge(data_path, seeds[0])
    return code


def main(argv=None):
    global MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                 description="Tornaviagem: return-gated ledger learner (chapter 0397).")
    ap.add_argument("--quick", action="store_true", help="one seed, reduced steps, all correctness tests")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default 5, or 1 with --quick)")
    ap.add_argument("--json", default=None, help="also write the JSON report to this path")
    ap.add_argument("--card", action="store_true", help="print MIND_CARD as JSON and exit")
    ap.add_argument("--mutant", default=None, help="run with a registered mutant: " + ", ".join(MUTANTS))
    ap.add_argument("--data", default=None, help="optional numeric CSV (header; 0/1 label in the last column)")
    args = ap.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    if n_seeds < 1 or (args.mutant is not None and args.mutant not in MUTANTS):
        print(f"usage: python3 {os.path.basename(__file__)} [--quick] [--seed S] [--seeds N>=1] "
              f"[--mutant {'|'.join(MUTANTS)}]", file=sys.stderr)
        return 2
    MUTANT = args.mutant
    try:
        return protocol("quick" if args.quick else "full", args.seed, n_seeds, args.json, args.data)
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

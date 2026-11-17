#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0366 · Stefan Dušan (1308-1355)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
OKOLINA: custody-routed liability learning for populations of learners.

Thesis. Where the author of a harm cannot be seen, route the loss, in full and
by statute, to whoever held custody of the place where it happened; complete
custody for unowned places; make the victim whole first and recover later; and
let no letter of the sovereign's passion reroute the charge.

Evidence and provenance (belief; see MIND_CARD["evidence"]).
Dušan's Code (Skopje 1349, arts 1-135; second council 1353/54, arts 136-201;
Novaković numbering) routes anonymous losses to places: the okolina (surrounding
settlements) pays (arts 58, 100, 126); the nearest village must hand over a
suspect or pay (art 92); march lords pay for raiders who cross their land (49)
and sevenfold for brigands (143); the village harbouring a found brigand pays
all from the beginning (145); in mixed districts the tsar's kefalije guard the
roads and pay victims at once (157); villages around a deserted hill guard it
or pay (158); the tsar repays robbed guests and merchants and recovers from the
custodians of the road (160); letters written from anger, love or mercy that
break the Code are not believed (171-172, transplanted from the Basilika). No
philosophy of mind survives; reading these articles as a credit-assignment rule
is this file's interpretation.

Doctrine -> mechanism -> test (IDs as in the card)
  D1 arts 157-160      -> M1 custody router + M4 insurer ledger -> C6.1, C6.4, H-SIG
  D2 arts 58,92,100,126-> M1 locality of custody                -> C6.2
  D3 arts 49,143,145   -> M2 budget breaker, M3 source liability-> C6.3, H-NEC, H-BLIND
  D4 1349 -> 1354      -> M1 hill completion (art 158)          -> descriptive knockout
  D5 arts 171-172      -> M5 letter firewall                    -> descriptive knockout
  D6 reading (speculation) -> M2 as budget breaker             -> C6.3
  R1 Laws of Hammurabi 23-24 -> rival: sole territorial liability -> H-RIVAL

Research question. Many independent learners share a territory; harms arrive
without an identifiable author. Does routing each anonymous loss to its legal
custodians (completed for unowned stretches, scaled by a budget breaker) reach
near-cooperative behaviour without a counterfactual model, and where does
location-routed credit fail?

Closest prior art and the delta. Shared team reward (standard MARL); difference
rewards (Wolpert and Tumer 2001; Agogino and Tumer 2008), which need a
counterfactual model; localized kappa-hop rewards in networked MARL (Qu,
Wierman and Li 2020); the budget breaker of team moral hazard (Holmström 1982).
Delta: credit is routed by a legal custody topology (owner, surrounding
settlements for unowned stretches, source liability when the author is found)
with a multiplier equal to the number of co-custodians and an insurer that makes
victims whole, so no counterfactual simulator and no learned incentive designer
are needed. Overlap: Medium.

Blind spot. Location is not agency. When harm is caused from a remote haven
(brigands based elsewhere) the charge lands on the neighbours of the road, who
cannot prevent it, and the haven is charged only when a thief is found.

Task (episodic_control, one season per episode).
  1. A realm: n settlements in the plane, roads = MST + 2-nearest-neighbour edges.
  2. Each road splits into land stretches (near a settlement) and, if long, a
     deserted middle stretch (the art-158 hill).
  3. Each settlement chooses a guard effort g >= 0 from local features.
  4. Robberies on stretch k ~ Poisson(t_k * sigmoid(rho_k - beta * sum_i W_ki g_i));
     in haven realms, raids ~ Poisson(t_k * sigmoid(eta_k - beta_h * g_haven)).
  5. Claims are routed to settlements by the active law; each settlement's cost
     is its charge plus c1*g + c2*g^2/2. Nuisance: 15% hold letters of grace.
  6. Metric: welfare recovery (W(g) - W(0)) / (W(g*) - W(0)) against a planner.
  Splits: held-out realms; shifted (larger, sparser realms, more hills); blind
  (haven realms, trained and tested with havens).

Limits. A research prototype of one mechanism, not an AGI. Guard effort is an
abstract prevention effort; no estate, ethnicity or person-level attribute is
encoded; the Code's corporal and collective punishments are not modelled.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 3,
    "revision_log": [
        "rev 1 -> 2 (before the first complete run of this file; hypotheses, comparisons, "
        "splits, mesi, loss-drop and trivial-margin thresholds unchanged). Prototype runs of the "
        "learners were used to calibrate the task: environment constants set so a single uniform "
        "effort no longer recovers > 0.9 of the planner's gain; log1p on exposure features after "
        "tanh saturation and a collapsed policy were seen; learning rate 0.02 -> 0.01 and 320 -> "
        "1500 updates; multi-start planner (haven realms trapped it); pooled recovery ratio instead "
        "of a mean of per-realm ratios. The prototypes already showed the sign of H-SIG, H-NEC "
        "and H-RIVAL on three seeds.",
        "rev 2 (after the first quick run): C4 band redefined. Under shuffled charges Adam "
        "random-walks the weights until tanh units saturate, giving a near-uniform effort that can "
        "land near the best uniform level; the band is now the feature-blind ceiling (best uniform "
        "effort evaluated on the held-out split itself) + 0.05, and the trained model must also "
        "beat the shuffled one by the trivial margin. No hypothesis threshold changed.",
        "rev 3 (after the first full run): documentation only. Docstring and traceability now "
        "separate D6 (the speculative reading) from R1 (the Hammurabi rival); corpus_neighbors "
        "record the one similarity computable locally; no code path changed."],
    "generation": {"template_version": "codeguidelines 1.0 (15 Sep 2026)",
                   "generator": "Claude", "generator_version": "Opus 5.5",
                   "date": "2026-09-30"},
    "id": 366, "figure": "Stefan Dušan", "born": 1308, "died": 1355,
    "civilization": "Serbian", "provenance": "belief",
    "thesis": ("Anonymous harm is governed by routing its full cost, by statute, to the "
               "custodians of the place where it happened, completing custody for unowned "
               "places, insuring the victim first, and forbidding passion to reroute the charge."),
    "evidence": [
        {"id": "D1", "claim": "Mixed districts: kefalije post guards on roads and pay victims at once; "
         "deserted hill: surrounding villages guard or pay; tsar repays robbed guests/merchants "
         "and recovers from the custodians of the road; guards hand travellers on with all "
         "their goods", "basis": "primary", "source": "Dušan's Code arts 157-160 (1353/54)"},
        {"id": "D2", "claim": "The okolina pays for harm to a dead lord's village, for arson outside "
         "a village and robbery on town land; the nearest village hands over a suspect or pays",
         "basis": "primary", "source": "Dušan's Code arts 58, 92, 99-100, 126 (1349)"},
        {"id": "D3", "claim": "March lords pay for raiders crossing their land and sevenfold for "
         "brigands; a village where a brigand is found is dispersed and its master pays all "
         "from the beginning", "basis": "primary", "source": "Dušan's Code arts 49, 143, 145-147"},
        {"id": "D4", "claim": "A second council added 66 articles, many on brigandage, roads and "
         "juries", "basis": "scholarship", "source": "Bubalo 2015; Solovjev 1928"},
        {"id": "D5", "claim": "Letters from anger, love or mercy that break the Code are not "
         "believed; judges judge by the Code, not fear of the tsar", "basis": "primary",
         "source": "Dušan's Code arts 171-172; Šarkić 2015 (from Basilika)"},
        {"id": "D6", "claim": "Reading custody routing as credit assignment and the tsar as "
         "budget breaker", "basis": "speculation", "source": "this chapter"},
    ],
    "research_question": {"category": "multi-agent cooperation and conflict",
                          "question": ("Can locality-constrained custody routing of anonymous "
                                       "losses, with a budget breaker, let independent learners "
                                       "approach the cooperative optimum without a counterfactual "
                                       "model, and where does it fail?")},
    "mechanism": {
        "name": "Okolina custody field", "family": "multi-agent credit assignment; "
        "three-factor (routed-liability-modulated) policy gradient",
        "signature_modules": ["custody_router", "budget_breaker", "source_liability",
                              "insurer_ledger", "letter_firewall"],
        "closest_prior_art": ["shared team reward", "difference rewards (Wolpert & Tumer 2001)",
                              "localized rewards in networked MARL (Qu, Wierman & Li 2020)",
                              "budget breaker (Holmström 1982)"],
        "overlap": "Medium",
        "prior_art_queries": ["difference rewards multiagent credit assignment",
                              "networked multi-agent reinforcement learning localized policies",
                              "moral hazard in teams budget breaker",
                              "collective liability mechanism design learning agents",
                              "learning to incentivize other learning agents",
                              "spatially targeted dopamine credit assignment"],
        "contribution_type": "mechanism",
        "delta": ("Credit routed by a legal custody topology with completion for unowned "
                  "stretches, source liability, a co-custodian multiplier and an insurer, "
                  "instead of counterfactual evaluation or influence neighbourhoods."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1,M4", "property_test": "C6.1,C6.4", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M1", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M2,M3", "property_test": "C6.3", "hypothesis": "H-NEC,H-BLIND"},
        {"doctrine": "D4", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "descriptive"},
        {"doctrine": "D5", "mechanism": "M5", "property_test": "definition", "hypothesis": "descriptive"},
        {"doctrine": "D6", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-NEC"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Okolina-routed learners recover more welfare than "
         "shared-team-reward learners on larger, sparser realms", "metric": "welfare_recovery",
         "split": "shifted", "comparison": "okolina - global", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing the budget breaker hurts more than removing the "
         "leave-one-out baseline", "metric": "welfare_recovery", "split": "shifted",
         "comparison": "ko_breaker - ko_rloo", "direction": "less", "mesi": 0.03, "seeds": 5},
        {"id": "H-BLIND", "statement": "Where raids come from remote havens, location-routed "
         "credit does worse than shared team reward", "condition": "haven realms",
         "grounding": "the Code routes road losses by place; brigand bases lay elsewhere and "
         "source liability needs the thief found", "metric": "welfare_recovery",
         "split": "blind", "comparison": "okolina - global", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Shared hill custody with a multiplier beats sole "
         "territorial liability (nearest settlement pays all)", "metric": "welfare_recovery",
         "split": "shifted", "comparison": "okolina - sole", "direction": "greater",
         "mesi": 0.03, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.15, "margin_over_trivial": 0.05,
                   "shuffled_band": 0.05},
    "probe_predictions": [{"probe": "P11", "expected": "not applicable: guard-effort control in a "
                           "liability game, not a reaching body"}],
    "dialectic_links": [
        {"chapter": 321, "relation": "teacher", "test": "none (Code cites the law of the Holy Fathers)"},
        {"chapter": 150, "relation": "rival", "test": "none (anger delay vs content test, discussed)"},
        {"chapter": 10, "relation": "rival", "test": "H-RIVAL (sole territorial liability, Laws 23-24)"},
    ],
    "corpus_neighbors": [
        {"chapter": 504, "similarity": 0.026, "difference": "only corpus file available locally "
         "(Philip II sample); multiplicative attenuation of lines vs routing of anonymous losses"},
        {"chapter": 360, "similarity": None, "difference": "physical ties carry perception; here "
         "only liability signals are routed, nothing couples bodies"},
        {"chapter": 349, "similarity": None, "difference": "prices one's own refusal by future "
         "coercers; here others' anonymous losses are routed to custodians"},
        {"chapter": 355, "similarity": None, "difference": "tests sincerity by a bypass road; here "
         "roads carry losses, not tests"},
        {"chapter": 150, "similarity": None, "difference": "cooling delay on hot decrees; here "
         "passion letters are ignored by content, with no delay"},
    ],
    "barometer": {
        "cognitive_processing": ["credit assignment over a population"],
        "embodied_cognition": ["none: abstract effort only"],
        "world_modeling": ["territorial geometry of custody vs influence"],
        "consciousness": ["none claimed"],
        "language_understanding": ["none"],
        "emotional_intelligence": ["letter firewall: passion cannot reroute charges"],
        "creativity": ["none"],
        "autonomy": ["goals set by the routing law, not by agents"],
    },
    "task_types": ["episodic_control"],
    "applications": [
        {"use": "Allocating leak and outage costs to segment operators in shared pipe networks",
         "sector": "utilities", "dataset": "BattLeDIM 2020 (L-Town network)"},
        {"use": "Assigning prevention duty for ignitions on unowned and mixed-ownership land",
         "sector": "land management", "dataset": "FPA FOD wildfire occurrence 1992-2020 (Short 2022)"},
        {"use": "Routing harms among AI agents in shared-environment social dilemmas",
         "sector": "AI safety research", "dataset": "Melting Pot (Leibo et al. 2021)"},
    ],
    "safety_notes": ("The Code's corporal and collective punishments are history, not features; "
                     "no estate, ethnicity or person attribute is encoded; guard effort is abstract "
                     "prevention, not policing or surveillance of people."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 366
FILE_NAME = os.path.basename(__file__)
BUDGET_FULL_S, BUDGET_QUICK_S = 180.0, 20.0

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
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


def clip_by_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = min(1.0, max_norm / (total + 1e-12))
    return {k: g * scale for k, g in grads.items()}, total


def finite_difference_check(loss_fn, params, grads, rng, n_entries=20, eps=1e-6, floor=1e-4):
    """Central differences on >= n_entries random entries per tensor plus the
    largest-gradient entry. Relative error |a-n| / max(|a|+|n|, floor): entries
    below finite-difference resolution are judged on absolute error."""
    worst = {}
    for k, p in params.items():
        flat = p.reshape(-1)
        g = grads[k].reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            err = max(err, abs(num - g[i]) / max(abs(num) + abs(g[i]), floor))
        worst[k] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def verdict(mean, ci, mesi, direction):
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


def write_report(lines, report, json_path=None):
    print(f"=== VERIFIED REPORT · chapter {CHAPTER:04d} ===")
    for line in lines:
        print(line)
    print("=== END REPORT ===")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
# END STANDARD UTILITIES

# ---------------------------------------------------------------- data and tasks
ENV = dict(density=22.0, r_land=0.08, lam=0.07, cutoff=0.25, beta=1.5, beta_h=1.2,
           c1=0.60, c2=0.05, n_surround=3, p_find=0.15, k_src=2.0, letter_rate=0.15,
           haven_rate=0.12, haven_hill_share=0.85)
N_FEATURES = 7


def _mst_edges(d):
    n = d.shape[0]
    in_tree, best, parent = np.zeros(n, bool), d[0].copy(), np.zeros(n, int)
    in_tree[0], edges = True, set()
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = int(np.argmin(cand))
        edges.add((min(j, parent[j]), max(j, parent[j])))
        in_tree[j] = True
        closer = d[j] < best
        best, parent = np.where(closer, d[j], best), np.where(closer, j, parent)
    return edges


def _road_edges(pos):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    edges = _mst_edges(d)
    np.fill_diagonal(d, np.inf)
    for i in range(len(pos)):
        for j in np.argsort(d[i])[:2]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def _stretches(pos, edges, env):
    """Each road splits into land stretches near its two settlements and, when
    long, a deserted middle stretch (the art-158 hill) that nobody owns."""
    rows = []
    for a, b in edges:
        u = pos[b] - pos[a]
        length = float(np.linalg.norm(u))
        u = u / max(length, 1e-9)
        rl = min(length / 2, env["r_land"])
        rows.append((pos[a] + u * rl / 2, rl, a, a, b))
        rows.append((pos[b] - u * rl / 2, rl, b, a, b))
        if length > 2 * env["r_land"]:
            rows.append(((pos[a] + pos[b]) / 2, length - 2 * env["r_land"], -1, a, b))
    spos = np.array([r[0] for r in rows])
    slen = np.array([r[1] for r in rows])
    owner = np.array([r[2] for r in rows])
    ends = np.array([(r[3], r[4]) for r in rows])
    return spos, slen, owner, ends


def _custody(owner, near, n, n_surround):
    """Custody matrices (K x n). 1354: land stretch -> its owner; hill -> the
    n_surround nearest settlements (art 158). 1349: hills unowned. sole: the
    single nearest settlement answers for everything (Laws of Hammurabi 23-24)."""
    K = owner.size
    r1354, r1349, rsole = np.zeros((K, n)), np.zeros((K, n)), np.zeros((K, n))
    land = owner >= 0
    r1354[land, owner[land]] = r1349[land, owner[land]] = rsole[land, owner[land]] = 1.0
    for k in np.where(~land)[0]:
        r1354[k, near[k, :n_surround]] = 1.0 / n_surround
        rsole[k, near[k, 0]] = 1.0
    return {"1354": r1354, "1349": r1349, "sole": rsole}


def make_realm(rng, n, dens=1.0, havens=False, env=ENV):
    side = math.sqrt(n / env["density"]) * dens
    pos = rng.uniform(0.0, side, (n, 2))
    return build_realm(rng, pos, _road_edges(pos), np.exp(rng.normal(0.0, 1.0, n)), havens, env)


def build_realm(rng, pos, edges, size, havens=False, env=ENV):
    n = len(pos)
    side = float(np.ptp(pos[:, 0])) + 1e-9
    border = (pos[:, 0] > pos[:, 0].min() + 0.85 * side).astype(float)
    spos, slen, owner, ends = _stretches(pos, edges, env)
    K = slen.size
    hill = owner < 0
    lengths = [np.linalg.norm(pos[b] - pos[a]) for a, b in edges]
    per_edge = [3 if L > 2 * env["r_land"] else 2 for L in lengths]
    traffic = 1.2 * size[ends[:, 0]] * size[ends[:, 1]]
    traffic = traffic * np.repeat(rng.uniform(0.6, 1.4, len(edges)), per_edge)
    rho = (-1.3 + 0.55 * np.minimum(slen / env["r_land"], 3.5) + 1.0 * hill
           + 0.4 * np.maximum(border[ends[:, 0]], border[ends[:, 1]]) + rng.normal(0, 1.0, K))
    dist = np.linalg.norm(spos[:, None] - pos[None], axis=-1)
    W = np.where(dist < env["cutoff"], np.exp(-dist / env["lam"]), 0.0)
    near = np.argsort(dist, axis=1)
    haven_of, eta, is_haven = -np.ones(K, int), np.zeros(K), np.zeros(n)
    if havens:
        hv = rng.choice(n, size=max(2, int(round(env["haven_rate"] * n))), replace=False)
        is_haven[hv] = 1.0
        raided = hill & (rng.uniform(size=K) < env["haven_hill_share"])
        haven_of[raided] = rng.choice(hv, size=int(raided.sum()))
        eta[raided] = rho[raided] + 0.8
    realm = dict(n=n, pos=pos, size=size, border=border, spos=spos, slen=slen, owner=owner,
                 hill=hill, traffic=traffic, rho=rho, W=W, haven_of=haven_of, eta=eta,
                 is_haven=is_haven, letter=(rng.uniform(size=n) < env["letter_rate"]).astype(float),
                 custody=_custody(owner, near, n, env["n_surround"]),
                 n_cust=np.where(hill, env["n_surround"], 1).astype(float),
                 degree=np.bincount(np.array(edges).ravel(), minlength=n).astype(float))
    realm["X"] = features(realm, env)
    return realm


def features(realm, env=ENV):
    """Local, routing-agnostic geography: what a settlement can see of its roads."""
    t, q0 = realm["traffic"], sigmoid(realm["rho"])
    n, owner, hill = realm["n"], realm["owner"], realm["hill"]
    land_exp = np.bincount(owner[owner >= 0], weights=(t * q0)[owner >= 0], minlength=n)
    hill_exp = (realm["custody"]["1354"][hill] * (t * q0)[hill, None]).sum(0)
    reach = (realm["W"] * (t * q0)[:, None]).sum(0)
    hv = realm["haven_of"] >= 0
    haven_p = np.bincount(realm["haven_of"][hv], weights=(t * sigmoid(realm["eta"]))[hv],
                          minlength=n)
    return np.stack([np.log1p(land_exp), np.log1p(hill_exp), np.log1p(reach),
                     realm["degree"] / 4.0, realm["border"], np.log1p(haven_p),
                     realm["letter"]], axis=1)


def expected_harm(realm, g, env=ENV):
    """Expected robberies per stretch for effort g (..., n): local and haven."""
    D = env["beta"] * g @ realm["W"].T
    hc = realm["traffic"] * sigmoid(realm["rho"] - D)
    hv = realm["haven_of"] >= 0
    gh = g[..., np.maximum(realm["haven_of"], 0)]
    hh = np.where(hv, realm["traffic"] * sigmoid(realm["eta"] - env["beta_h"] * gh), 0.0)
    return hc, hh


def guard_cost(g, env=ENV):
    return env["c1"] * g + 0.5 * env["c2"] * g * g


def welfare(realm, g, env=ENV):
    hc, hh = expected_harm(realm, g, env)
    return -(hc.sum(-1) + hh.sum(-1)) - guard_cost(g, env).sum(-1)


def welfare_grad(realm, g, env=ENV):
    D = env["beta"] * g @ realm["W"].T
    qc = sigmoid(realm["rho"] - D)
    grad = env["beta"] * (realm["traffic"] * qc * (1 - qc)) @ realm["W"]
    hv = realm["haven_of"] >= 0
    if hv.any():
        qh = sigmoid(realm["eta"][hv] - env["beta_h"] * g[realm["haven_of"][hv]])
        grad = grad + np.bincount(realm["haven_of"][hv], minlength=realm["n"],
                                  weights=env["beta_h"] * realm["traffic"][hv] * qh * (1 - qh))
    return grad - env["c1"] - env["c2"] * g


def planner(realm, steps=400, env=ENV):
    """Oracle optimum W(g*) by projected Adam ascent from three uniform starts and
    one haven-heavy start; used only to normalise welfare."""
    best = (None, -np.inf)
    starts = [np.full(realm["n"], v) for v in (0.3, 1.0, 2.5)]
    starts.append(np.where(realm["is_haven"] > 0, 4.0, 0.8))
    for g0 in starts:
        p = {"g": g0.copy()}
        opt = Adam(p, lr=0.05)
        for _ in range(steps):
            opt.step(p, {"g": -welfare_grad(realm, p["g"], env)})
            p["g"] = np.maximum(p["g"], 0.0)
        w = float(welfare(realm, p["g"], env))
        if w > best[1]:
            best = (p["g"], w)
    return best


def sample_season(realm, g, rng, env=ENV):
    hc, hh = expected_harm(realm, g, env)
    inc_c, inc_h = rng.poisson(hc), rng.poisson(hh)
    return inc_c, inc_h, rng.binomial(inc_h, env["p_find"])


def route_charges(realm, law, g, inc_c, inc_h, found, env=ENV):
    """Charges (M x n) under a law: the Okolina field or one of its rivals."""
    if law["router"] == "global":
        total = inc_c.sum(1) + inc_h.sum(1) + guard_cost(g, env).sum(1)
        return np.repeat(total[:, None], realm["n"], axis=1)
    if law["router"] == "diff":
        return difference_charges(realm, g, env)
    router = "1354" if law["router"] == "mean" else law["router"]
    R = realm["custody"][router]
    if law.get("letters_honored"):
        R = R * (1.0 - realm["letter"])[None, :]
    kappa = realm["n_cust"] if law.get("breaker") and router == "1354" else np.ones(R.shape[0])
    src = law.get("source") and router != "sole"
    local = inc_c + inc_h - (found if src else 0)
    C = (local * kappa) @ R
    if src:
        hv = realm["haven_of"] >= 0
        C = C + env["k_src"] * np.stack([np.bincount(realm["haven_of"][hv], weights=f[hv],
                                                     minlength=realm["n"]) for f in found])
    if law["router"] == "mean":
        C = np.repeat(C.mean(1, keepdims=True), realm["n"], axis=1)
    return C + guard_cost(g, env)


def difference_charges(realm, g, env=ENV):
    """Wolpert-Tumer difference cost G(g) - G(g with g_i = 0), computed from the
    simulator's expected harm: the privileged closest prior art."""
    G = -welfare(realm, g, env)
    D = env["beta"] * g @ realm["W"].T
    Dcf = D[:, None, :] - env["beta"] * g[:, :, None] * realm["W"].T[None]
    hc_cf = (realm["traffic"] * sigmoid(realm["rho"] - Dcf)).sum(-1)
    _, hh = expected_harm(realm, g, env)
    hv = realm["haven_of"] >= 0
    own_h = np.zeros_like(g)
    free_h = np.zeros(realm["n"])
    if hv.any():
        own_h = np.stack([np.bincount(realm["haven_of"][hv], weights=h[hv], minlength=realm["n"])
                          for h in hh])
        free_h = np.bincount(realm["haven_of"][hv], minlength=realm["n"],
                             weights=realm["traffic"][hv] * sigmoid(realm["eta"][hv]))
    G_cf = (hc_cf + hh.sum(1, keepdims=True) - own_h + free_h[None]
            + guard_cost(g, env).sum(1, keepdims=True) - guard_cost(g, env))
    return G[:, None] - G_cf

# ------------------------------------------------------------------------ model
TASK_TYPES = ["episodic_control"]
OKOLINA = {"router": "1354", "breaker": True, "source": True, "letters_honored": False}
TRAIN_CFG = dict(hidden=16, sigma=0.5, lr=0.02, steps=320, rollouts=12, l2=1e-4, clip=5.0,
                 baseline="rloo", shuffle=False, mutant=None)


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """A settlement's effort policy, shared by every settlement: tanh MLP ->
    mean pre-activation; effort = softplus(mean + sigma * noise) while learning."""
    if task_type not in TASK_TYPES or out_dim != 1:
        raise ValueError("Okolina supports episodic_control with one effort output")
    c = dict(TRAIN_CFG, **cfg)
    c.setdefault("law", dict(OKOLINA))
    h = c["hidden"]
    params = {"W1": rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, h)), "b1": np.zeros(h),
              "W2": rng.normal(0, 1 / math.sqrt(h), (h, h)), "b2": np.zeros(h),
              "w3": rng.normal(0, 0.1 / math.sqrt(h), h), "b3": np.array([-1.0])}
    return {"params": params, "cfg": c}


def _forward(params, X):
    h1 = np.tanh(X @ params["W1"] + params["b1"])
    h2 = np.tanh(h1 @ params["W2"] + params["b2"])
    return h2 @ params["w3"] + params["b3"][0], (X, h1, h2)


def _backward(params, cache, dmu):
    X, h1, h2 = cache
    dz2 = np.outer(dmu, params["w3"]) * (1 - h2 ** 2)
    dz1 = (dz2 @ params["W2"].T) * (1 - h1 ** 2)
    return {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": h1.T @ dz2, "b2": dz2.sum(0),
            "w3": h2.T @ dmu, "b3": np.array([dmu.sum()])}


def loss_and_grads(model, batch):
    """Score-function surrogate for independent learners with shared weights:
    L = sum_i c_i mu_i + l2/2 |theta|^2, c_i = mean_m adv_mi eps_mi / (sigma n).
    Each learner's advantage is the charge routed to it: the third factor."""
    p, c = model["params"], model["cfg"]
    mu, cache = _forward(p, batch["X"])
    coef = (batch["adv"] * batch["eps"]).mean(0) / c["sigma"] / mu.size
    loss = float(coef @ mu) + 0.5 * c["l2"] * sum(float(np.sum(v * v)) for v in p.values())
    grads = _backward(p, cache, coef)
    grads = {k: grads[k] + c["l2"] * p[k] for k in p}
    if c.get("mutant") == "zero_grad_W2":
        grads["W2"] = np.zeros_like(grads["W2"])
    return loss, grads


def predict(model, X):
    return softplus(_forward(model["params"], X)[0])


def hidden_states(model, X):
    return _forward(model["params"], X)[1][2]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def advantages(C, baseline, rng, shuffle):
    if shuffle:
        C = rng.permutation(C.ravel()).reshape(C.shape)
    if baseline == "rloo":
        A = C - (C.sum(0, keepdims=True) - C) / (C.shape[0] - 1)
    else:
        A = C.copy()
    return A / (A.std() + 1e-8)


def rollout_batch(model, realm, rng):
    c = model["cfg"]
    mu, _ = _forward(model["params"], realm["X"])
    eps = rng.standard_normal((c["rollouts"], realm["n"]))
    g = softplus(mu[None] + c["sigma"] * eps)
    inc_c, inc_h, found = sample_season(realm, g, rng)
    C = route_charges(realm, c["law"], g, inc_c, inc_h, found)
    return {"X": realm["X"], "eps": eps, "adv": advantages(C, c["baseline"], rng, c["shuffle"])}


def social_cost(model, realms):
    return float(np.mean([-welfare(r, predict(model, r["X"])) for r in realms]))


def fit(model, data, budget, rng):
    """Independent learners, shared weights: each update draws a realm, runs
    `rollouts` seasons with exploration, routes the claims by the model's law,
    and descends the surrogate. Returns the probe social-cost history."""
    c = model["cfg"]
    opt = Adam(model["params"], lr=0.0 if c.get("mutant") == "zero_lr" else c["lr"])
    sign = -1.0 if c.get("mutant") == "sign_flip" else 1.0
    probe = data["pool"][:6]
    hist = [social_cost(model, probe)]
    for step in range(budget):
        realm = data["pool"][int(rng.integers(len(data["pool"])))]
        batch = rollout_batch(model, realm, rng)
        _, grads = loss_and_grads(model, batch)
        grads, _ = clip_by_global_norm(grads, c["clip"])
        opt.step(model["params"], grads, sign)
        if (step + 1) % 80 == 0:
            hist.append(social_cost(model, probe))
    for v in model["params"].values():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError("non-finite parameters")
    return hist


def copy_model(model, **cfg):
    c = dict(model["cfg"], **cfg)
    c["law"] = dict(model["cfg"]["law"], **cfg.get("law", {}))
    return {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": c}


# ------------------------------------------------ registries: modules, knockouts
def modules(model):
    sig = lambda role: {"params": [], "role": role, "signature": True}
    return {
        "encoder": {"params": ["W1", "b1"], "role": "tanh feature encoder", "signature": False},
        "trunk": {"params": ["W2", "b2"], "role": "tanh hidden layer", "signature": False},
        "effort_head": {"params": ["w3", "b3"], "role": "linear head to softplus effort",
                        "signature": False},
        "rloo_baseline": {"params": [], "role": "leave-one-out variance baseline",
                          "signature": False},
        "custody_router": sig("sparse row-stochastic routing of anonymous losses to legal custodians"),
        "budget_breaker": sig("multiplier equal to the co-custodian count on routed claims"),
        "source_liability": sig("reroute traced losses to the harbouring node"),
        "insurer_ledger": sig("treasury pays each claim first, then recovers the routed charges"),
        "letter_firewall": sig("routing invariant to principal exemption flags"),
    }


def knockout(model, name, mode):
    """Training-time lesions for routing modules (they act only on learning
    signals); evaluation-time lesions for network layers."""
    law = {"custody_router": {"router": {"mean": "mean", "zero": "1349"}.get(mode, "1354")},
           "budget_breaker": {"breaker": False}, "source_liability": {"source": False},
           "letter_firewall": {"letters_honored": True}}.get(name)
    if law is not None:
        return copy_model(model, law=law)
    if name == "rloo_baseline":
        return copy_model(model, baseline="zero")
    if name == "insurer_ledger":
        raise ValueError("the insurer is the channel that turns a loss into a claim; removing "
                         "it leaves no signal (victim bears), which is the trivial baseline")
    m = copy_model(model)
    for k in modules(model)[name]["params"]:
        v = m["params"][k]
        m["params"][k] = np.zeros_like(v) if mode == "zero" else np.full_like(v, v.mean())
    return m


MUTANTS = {
    "sign_flip": "optimizer ascends the surrogate (update sign reversed)",
    "zero_lr": "learning rate set to zero",
    "zero_grad_W2": "gradient of W2 replaced by zeros",
}


def env_reset(rng):
    return make_realm(rng, int(rng.integers(18, 27)))


def env_step(state, action, rng):
    """One season under the Okolina law: returns per-settlement costs and ledger."""
    g = np.asarray(action, float)[None]
    inc_c, inc_h, found = sample_season(state, g, rng)
    costs = route_charges(state, OKOLINA, g, inc_c, inc_h, found)[0]
    paid = float(inc_c.sum() + inc_h.sum())
    recovered = float(costs.sum() - guard_cost(g).sum())
    return state, costs, {"claims_paid": paid, "recovered": recovered,
                          "treasury": recovered - paid}

# ------------------------------------------------------------------ experiments
SPLITS = {"train": (40, 18, 27, 1.0, False), "heldout": (10, 18, 27, 1.0, False),
          "shifted": (10, 40, 55, 1.25, False), "train_haven": (40, 18, 27, 1.0, True),
          "blind": (10, 18, 27, 1.0, True)}
VARIANTS = {  # name: (law, baseline, training pool, evaluation splits)
    "okolina": (OKOLINA, "rloo", "train", ("heldout", "shifted")),
    "global": ({"router": "global"}, "rloo", "train", ("heldout", "shifted")),
    "difference": ({"router": "diff"}, "rloo", "train", ("heldout", "shifted")),
    "ko_breaker": (dict(OKOLINA, breaker=False), "rloo", "train", ("shifted",)),
    "ko_rloo": (OKOLINA, "zero", "train", ("shifted",)),
    "law_1349": (dict(OKOLINA, router="1349"), "rloo", "train", ("shifted",)),
    "sole": ({"router": "sole", "breaker": False, "source": False}, "rloo", "train", ("shifted",)),
    "letters_honored": (dict(OKOLINA, letters_honored=True), "rloo", "train", ("shifted",)),
    "okolina_haven": (OKOLINA, "rloo", "train_haven", ("blind",)),
    "global_haven": ({"router": "global"}, "rloo", "train_haven", ("blind",)),
}
QUICK_VARIANTS = ("okolina", "global")


def realm_key(r):
    return hashlib.sha1(np.ascontiguousarray(r["pos"]).tobytes()).hexdigest()


def make_splits(seed, quick):
    out = {}
    for j, (name, (k, n0, n1, dens, hv)) in enumerate(SPLITS.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1000 + j]))
        count = max(4, k // 3) if quick else k
        out[name] = [make_realm(rng, int(rng.integers(n0, n1)), dens, hv) for _ in range(count)]
    return out


class Scorer:
    """Pooled welfare recovery sum(W(g) - W(0)) / sum(W(g*) - W(0))."""

    def __init__(self):
        self.cache = {}

    def _ref(self, r):
        key = realm_key(r)
        if key not in self.cache:
            g_star, w_star = planner(r)
            self.cache[key] = (g_star, w_star, float(welfare(r, np.zeros(r["n"]))))
        return self.cache[key]

    def recovery(self, effort_fn, realms):
        num = den = 0.0
        for r in realms:
            _, w_star, w0 = self._ref(r)
            num += float(welfare(r, effort_fn(r))) - w0
            den += w_star - w0
        return num / den

    def haven_effort_ratio(self, model, realms):
        pol = [predict(model, r["X"])[r["is_haven"] > 0].mean() for r in realms]
        opt = [self._ref(r)[0][r["is_haven"] > 0].mean() for r in realms]
        return float(np.mean(pol) / max(np.mean(opt), 1e-9))


def blind_ceiling(scorer, realms):
    """Best recovery any feature-blind (uniform) effort achieves on these realms."""
    return max(scorer.recovery(lambda r: np.full(r["n"], c), realms) for c in np.linspace(0, 4, 41))


def best_constant(pool):
    grid = np.linspace(0.0, 4.0, 41)
    tot = [sum(float(welfare(r, np.full(r["n"], c))) for r in pool) for c in grid]
    return float(grid[int(np.argmax(tot))])


def train_variant(seed, name, splits, steps, **extra):
    law, baseline, pool, _ = VARIANTS[name]
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    model = build_model(N_FEATURES, 1, "episodic_control", init_rng, law=law,
                        baseline=baseline, **extra)
    hist = fit(model, {"pool": splits[pool]}, steps, np.random.default_rng(
        np.random.SeedSequence([seed, 7])))
    return model, hist


def run_seed(seed, steps, quick, scorer):
    splits = make_splits(seed, quick)
    out, models = {}, {}
    for name in (QUICK_VARIANTS if quick else VARIANTS):
        model, hist = train_variant(seed, name, splits, steps)
        models[name] = (model, hist)
        out[name] = {s: scorer.recovery(lambda r: predict(model, r["X"]), splits[s])
                     for s in VARIANTS[name][3]}
    if not quick:
        for name in ("okolina_haven", "global_haven"):
            out[name]["haven_effort_ratio"] = scorer.haven_effort_ratio(models[name][0],
                                                                        splits["blind"])
    return splits, out, models


# ------------------------------------------------------------ correctness tests
def c1_gradcheck(seed, splits, mutant=None, tol=1e-5):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    model = build_model(N_FEATURES, 1, "episodic_control", rng, law=OKOLINA, mutant=mutant)
    realm, worst_all = splits["train"][0], {}
    for stage in ("init", "after_training_steps"):
        if stage != "init":
            fit(model, {"pool": splits["train"]}, 60, rng)
        batch = rollout_batch(model, realm, rng)
        _, grads = loss_and_grads(model, batch)
        cfg = model["cfg"]
        worst = finite_difference_check(
            lambda p: loss_and_grads({"params": p, "cfg": cfg}, batch)[0],
            model["params"], grads, rng)
        for k, v in worst.items():
            worst_all[k] = max(worst_all.get(k, 0.0), v)
    max_err = max(worst_all.values())
    return {"passed": bool(max_err <= tol), "max_rel_error": max_err, "tensors_checked": len(worst_all),
            "tensors_total": len(model["params"]), "per_tensor": worst_all}


def c3_learning(model, hist, splits, scorer, th):
    drop = 1.0 - hist[-1] / hist[0]
    rec = scorer.recovery(lambda r: predict(model, r["X"]), splits["heldout"])
    c = best_constant(splits["train"])
    triv = scorer.recovery(lambda r: np.full(r["n"], c), splits["heldout"])
    ok = bool(drop >= th["loss_drop_fraction"] and rec >= triv + th["margin_over_trivial"])
    return ok, {"loss_drop": drop, "heldout_recovery": rec, "trivial_recovery": triv,
                "trivial_constant": c}


def c2_determinism(seed, splits):
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 21]))
        m = build_model(N_FEATURES, 1, "episodic_control", rng, law=OKOLINA)
        h = fit(m, {"pool": splits["train"]}, 40, rng)
        runs.append((h, m))
    same = runs[0][0] == runs[1][0] and all(np.array_equal(runs[0][1]["params"][k],
                                                           runs[1][1]["params"][k])
                                            for k in runs[0][1]["params"])
    finite = all(np.all(np.isfinite(v)) for v in runs[0][1]["params"].values())
    return bool(same and finite)


def c6_properties(seed):
    """Invariants of the routing law, each with a negative control that must fail."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
    realms = [make_realm(rng, int(rng.integers(18, 55)), float(rng.uniform(1.0, 1.4)),
                         bool(rng.uniform() < 0.5)) for _ in range(24)]
    res = {}
    comp = lambda R: bool(np.all(np.abs(R.sum(1) - 1.0) < 1e-12))
    res["C6.1_custody_complete"] = (all(comp(r["custody"]["1354"]) for r in realms),
                                    not all(comp(r["custody"]["1349"]) for r in realms))

    def local_violations(R, r):
        near = np.argsort(np.linalg.norm(r["spos"][:, None] - r["pos"][None], axis=-1), axis=1)
        allowed = np.zeros_like(R, bool)
        land = r["owner"] >= 0
        allowed[land, r["owner"][land]] = True
        for k in np.where(~land)[0]:
            allowed[k, near[k, :ENV["n_surround"]]] = True
        return int(np.sum((R > 0) & ~allowed))
    diffuse = lambda r: softmax(-np.linalg.norm(r["spos"][:, None] - r["pos"][None], axis=-1)
                                / 0.2, axis=1)
    res["C6.2_custody_local"] = (sum(local_violations(r["custody"]["1354"], r) for r in realms) == 0,
                                 sum(local_violations(diffuse(r), r) for r in realms) > 0)

    def alignment_gap(law, r, g):
        rm = dict(r, W=r["W"] * (r["custody"]["1354"] > 0), haven_of=-np.ones_like(r["haven_of"]))
        expected = lambda gg: route_charges(rm, law, gg[None], expected_harm(rm, gg[None])[0],
                                            np.zeros((1, rm["spos"].shape[0])),
                                            np.zeros((1, rm["spos"].shape[0])))[0]
        gap = 0.0
        for i in range(rm["n"]):
            e = np.zeros(rm["n"])
            e[i] = 1e-6
            selfish = (expected(g + e)[i] - expected(g - e)[i]) / 2e-6
            gap = max(gap, abs(selfish + welfare_grad(rm, g)[i]))
        return gap
    gaps = [(alignment_gap(OKOLINA, r, g), alignment_gap(dict(OKOLINA, breaker=False), r, g))
            for r in realms[:6] for g in [rng.uniform(0, 3, r["n"])]]
    res["C6.3_incentive_alignment"] = (max(a for a, _ in gaps) < 1e-5,
                                       max(b for _, b in gaps) > 1e-2)

    def deficits(law):
        worst = np.inf
        for r in realms:
            g = rng.uniform(0, 2, (8, r["n"]))
            ic, ih, fd = sample_season(r, g, rng)
            rec = route_charges(r, law, g, ic, ih, fd) - guard_cost(g)
            worst = min(worst, float(np.min(rec.sum(1) - ic.sum(1) - ih.sum(1))))
        return worst
    ledger = [env_step(r, rng.uniform(0, 2, r["n"]), rng)[2]["treasury"] for r in realms]
    res["C6.4_treasury_solvent"] = (min(ledger) >= -1e-9 and deficits(OKOLINA) >= -1e-9,
                                    deficits(dict(OKOLINA, router="1349")) < 0)
    return res


def c6_definition_letter_firewall(splits):
    """Definition check, not evidence: the Code's router never reads letters."""
    r = dict(splits["train"][0])
    g = np.ones((2, r["n"]))
    ic, ih, fd = np.ones((2, r["spos"].shape[0])), np.zeros((2, r["spos"].shape[0])), \
        np.zeros((2, r["spos"].shape[0]))
    a = route_charges(r, OKOLINA, g, ic, ih, fd)
    r["letter"] = 1.0 - r["letter"]
    return bool(np.array_equal(a, route_charges(r, OKOLINA, g, ic, ih, fd)))


def c7_split_integrity(splits_by_seed):
    keys = {}
    for seed, splits in splits_by_seed.items():
        for name, realms in splits.items():
            for r in realms:
                keys.setdefault(realm_key(r), set()).add((seed, name))
    return all(len(v) == 1 for v in keys.values())


def run_mutants(seed, splits, steps, scorer, th, names=None):
    detected = {}
    for name in names or MUTANTS:
        if name == "zero_grad_W2":
            detected[name] = not c1_gradcheck(seed, splits, mutant=name)["passed"]
        else:
            model, hist = train_variant(seed, "okolina", splits, steps, mutant=name)
            detected[name] = not c3_learning(model, hist, splits, scorer, th)[0]
    return detected


# ------------------------------------------------------------- hypothesis tests
HYP_PAIRS = {"H-SIG": ("okolina", "global", "shifted"),
             "H-NEC": ("ko_breaker", "ko_rloo", "shifted"),
             "H-BLIND": ("okolina_haven", "global_haven", "blind"),
             "H-RIVAL": ("okolina", "sole", "shifted")}
KNOCKOUT_ROWS = [("budget_breaker", "ko_breaker", True), ("rloo_baseline", "ko_rloo", False),
                 ("custody_router (1349 law, hills unowned)", "law_1349", True),
                 ("letter_firewall (letters of grace honoured)", "letters_honored", True)]


def hypotheses(per_seed, rng, quick):
    rows = []
    for h in MIND_CARD["hypotheses"]:
        a, b, split = HYP_PAIRS[h["id"]]
        if quick:
            rows.append({"id": h["id"], "metric": h["metric"], "mean_diff": None, "ci95": None,
                         "mesi": h["mesi"], "n_seeds": 1, "verdict": "not evaluated"})
            continue
        diffs = [s[a][split] - s[b][split] for s in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"id": h["id"], "metric": h["metric"], "comparison": h["comparison"],
                     "split": split, "mean_diff": mean, "ci95": list(ci), "mesi": h["mesi"],
                     "n_seeds": len(diffs), "verdict": verdict(mean, ci, h["mesi"], h["direction"])})
    return rows


def knockout_table(per_seed, rng):
    rows = []
    for module, variant, sig in KNOCKOUT_ROWS:
        diffs = [s[variant]["shifted"] - s["okolina"]["shifted"] for s in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"module": module, "signature": sig, "metric_change": mean, "ci95": list(ci)})
    return rows

# ------------------------------------------------------------------------ report
def load_road_csv(path, rng):
    """Real-data bridge: CSV with header x1,y1,x2,y2[,traffic] of road or pipe
    segments (e.g. BattLeDIM L-Town pipes). Coordinates are rescaled to the
    realm density; missing traffic is drawn as in synthetic realms."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    pts, idx = {}, []
    for r in rows:
        ends = []
        for x, y in ((r["x1"], r["y1"]), (r["x2"], r["y2"])):
            key = (round(float(x), 6), round(float(y), 6))
            ends.append(pts.setdefault(key, len(pts)))
        if ends[0] != ends[1]:
            idx.append((min(ends), max(ends)))
    pos = np.array(list(pts.keys()), float)
    pos = (pos - pos.min(0)) / max(np.ptp(pos, 0).max(), 1e-9) * math.sqrt(len(pos) / ENV["density"])
    return build_realm(rng, pos, sorted(set(idx)), np.exp(rng.normal(0, 1.0, len(pos))))


def fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def build_report(ctx):
    L = [f"file: {FILE_NAME}   card_revision: {MIND_CARD['card_revision']}",
         f"environment: python {sys.version.split()[0]}, numpy {np.__version__}",
         f"seeds: {ctx['seeds']}   runtime_s: {ctx['runtime']:.1f}   mode: {ctx['mode']}",
         f"n_params (effort policy): {ctx['n_params']}   task_types: {TASK_TYPES}",
         f"gradient check: {ctx['c1']['tensors_checked']}/{ctx['c1']['tensors_total']} tensors, "
         f"max rel error {ctx['c1']['max_rel_error']:.2e} (tol 1e-5), at init and after 60 steps",
         "correctness:"]
    for c in ctx["correctness"]:
        L.append(f"  {c['id']:<5} {c['name']:<32} {'PASS' if c['passed'] else 'FAIL'}  {c['detail']}")
    m = ctx["mutants"]
    L.append(f"mutation score: {m['detected']}/{m['total']} = {m['score']:.2f}  {m['which']}")
    L.append("welfare recovery by variant (mean over seeds):")
    for name, splits in ctx["means"].items():
        L.append("  " + f"{name:<16}" + "  ".join(f"{s}={v:.3f}" for s, v in splits.items()))
    L.append("hypotheses (paired over seeds, 95% bootstrap, 2000 resamples):")
    for h in ctx["hypotheses"]:
        ci = "n/a" if h["ci95"] is None else f"[{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
        L.append(f"  {h['id']:<8} {h.get('comparison', ''):<28} diff {fmt(h['mean_diff'])} "
                 f"CI {ci} mesi {h['mesi']}  -> {h['verdict']}")
    if ctx["knockouts"]:
        L.append("training-time knockouts (recovery change vs full Okolina, shifted split):")
        for k in ctx["knockouts"]:
            L.append(f"  {k['module']:<46} sig={str(k['signature']):<5} {k['metric_change']:+.3f} "
                     f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
    for line in ctx["extra"]:
        L.append(line)
    L.append(f"exit_code: {ctx['exit_code']}")
    return L


def json_report(ctx):
    return {"schema_version": "1.0", "chapter": CHAPTER, "file": FILE_NAME,
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": ctx["seeds"], "runtime_s": round(ctx["runtime"], 2), "n_params": ctx["n_params"],
            "gradcheck": {"tensors_checked": ctx["c1"]["tensors_checked"],
                          "tensors_total": ctx["c1"]["tensors_total"],
                          "max_rel_error": ctx["c1"]["max_rel_error"],
                          "checked_at": ["init", "after_training_steps"],
                          "passed": ctx["c1"]["passed"]},
            "correctness": ctx["correctness"],
            "mutants": {k: ctx["mutants"][k] for k in ("detected", "total", "score")},
            "hypotheses": ctx["hypotheses"], "knockouts": ctx["knockouts"],
            "descriptive": ctx["means"], "task_types": TASK_TYPES, "exit_code": ctx["exit_code"]}


# ---------------------------------------------------------------- entry point
def correctness_suite(seed, splits0, models0, steps, scorer, th, splits_by_seed):
    """C1-C7 on the first seed (C8, the budget, is appended by the caller)."""
    c1 = c1_gradcheck(seed, splits0)
    ok3, d3 = c3_learning(*models0["okolina"], splits0, scorer, th)
    shuf, _ = train_variant(seed, "okolina", splits0, steps, shuffle=True)
    r_shuf = scorer.recovery(lambda r: predict(shuf, r["X"]), splits0["heldout"])
    ceiling = blind_ceiling(scorer, splits0["heldout"])
    ok4 = bool(r_shuf <= ceiling + th["shuffled_band"]
               and d3["heldout_recovery"] >= r_shuf + th["margin_over_trivial"])
    mut = run_mutants(seed, splits0, steps, scorer, th)
    props = c6_properties(seed)
    ok6 = bool(all(p and neg for p, neg in props.values()))
    corr = [
        {"id": "C1", "name": "gradient_check", "passed": c1["passed"],
         "detail": f"max rel err {c1['max_rel_error']:.2e}"},
        {"id": "C2", "name": "determinism_and_finiteness", "passed": c2_determinism(seed, splits0),
         "detail": "two 40-step runs, identical"},
        {"id": "C3", "name": "learning", "passed": ok3,
         "detail": f"cost drop {d3['loss_drop']:.2f} (>= {th['loss_drop_fraction']}); held-out "
                   f"{d3['heldout_recovery']:.3f} vs best constant g={d3['trivial_constant']:.1f} "
                   f"{d3['trivial_recovery']:.3f} (+{th['margin_over_trivial']})"},
        {"id": "C4", "name": "shuffled_label_control", "passed": ok4,
         "detail": f"shuffled charges {r_shuf:.3f} <= blind ceiling {ceiling:.3f} + "
                   f"{th['shuffled_band']}; trained {d3['heldout_recovery']:.3f} beats it by "
                   f">= {th['margin_over_trivial']}"},
        {"id": "C5", "name": "mutant_detection", "passed": bool(all(mut.values())),
         "detail": ", ".join(f"{k}:{'caught' if v else 'missed'}" for k, v in mut.items())},
        {"id": "C6", "name": "property_tests", "passed": ok6,
         "detail": "; ".join(f"{k} {'ok' if p else 'VIOLATED'}/neg-control "
                             f"{'caught' if n else 'MISSED'}" for k, (p, n) in props.items())
                   + f"; letter firewall definition check {c6_definition_letter_firewall(splits0)}"},
        {"id": "C7", "name": "split_integrity", "passed": bool(c7_split_integrity(splits_by_seed)),
         "detail": "no realm shared between splits or seeds"},
    ]
    return corr, c1, d3, mut


def main(argv=None):
    ap = argparse.ArgumentParser(prog=FILE_NAME, description="Okolina custody field, chapter 0366")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None, choices=sorted(MUTANTS))
    ap.add_argument("--data", type=str, default=None)
    a = ap.parse_args(argv)
    if a.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    t0 = time.time()
    seeds = [a.seed] if a.quick or a.mutant else [a.seed + i for i in range(max(a.seeds, 1))]
    steps, budget = (400, BUDGET_QUICK_S) if a.quick else (1500, BUDGET_FULL_S)
    print(f"{FILE_NAME}: seeds {seeds}, {steps} updates per learner")
    th, scorer, boot = MIND_CARD["thresholds"], Scorer(), np.random.default_rng(a.seed + 99)
    if a.mutant:
        splits = make_splits(a.seed, True)
        caught = run_mutants(a.seed, splits, steps, scorer, th, [a.mutant])[a.mutant]
        print(f"mutant {a.mutant}: {'detected (C1 or C3 failed)' if caught else 'NOT detected'}")
        return 1 if caught else 0
    per_seed, splits_by_seed, first = [], {}, None
    for s in seeds:
        splits, out, models = run_seed(s, steps, a.quick, scorer)
        per_seed.append(out)
        splits_by_seed[s] = splits
        print(f"  seed {s}: " + ", ".join(f"{k} {v.get('shifted', v.get('blind', 0)):.3f}"
                                         for k, v in out.items()))
        first = first or (splits, models)
    splits0, models0 = first
    corr, c1, d3, mut = correctness_suite(seeds[0], splits0, models0, steps, scorer, th,
                                          splits_by_seed)
    runtime = time.time() - t0
    corr.append({"id": "C8", "name": "budget", "passed": bool(runtime <= budget),
                 "detail": f"{runtime:.1f} s of {budget:.0f} s"})
    extra = []
    if a.data:
        realm = load_road_csv(a.data, np.random.default_rng(a.seed))
        rec = scorer.recovery(lambda r: predict(models0["okolina"][0], r["X"]), [realm])
        extra.append(f"real-data bridge: {a.data}: {realm['n']} nodes, recovery {rec:.3f}")
    else:
        extra.append("real-data bridge: skipped (no --data PATH given)")
    if not a.quick:
        hv = [(s["okolina_haven"]["haven_effort_ratio"], s["global_haven"]["haven_effort_ratio"])
              for s in per_seed]
        extra.append("descriptive: haven effort / planner effort, okolina "
                     f"{np.mean([h[0] for h in hv]):.2f} vs global {np.mean([h[1] for h in hv]):.2f}")
        dead = [f"{k}@seed{seeds[i]}" for i, s in enumerate(per_seed) for k, v in s.items()
                if min(x for key, x in v.items() if key != "haven_effort_ratio") < 0.05]
        extra.append("descriptive: abdication (policy collapsed to ~zero effort, recovery < 0.05): "
                     + (", ".join(dead) if dead else "none"))
    means = {name: {sp: float(np.mean([s[name][sp] for s in per_seed])) for sp in per_seed[0][name]}
             for name in per_seed[0]}
    exit_code = 0 if all(c["passed"] for c in corr[:7]) else 1
    if exit_code == 0 and not corr[7]["passed"]:
        exit_code = 3
    ctx = {"seeds": seeds, "runtime": time.time() - t0, "mode": "quick" if a.quick else "full",
           "n_params": n_params(models0["okolina"][0]), "c1": c1, "correctness": corr,
           "mutants": {"detected": int(sum(mut.values())), "total": len(mut),
                       "score": sum(mut.values()) / len(mut),
                       "which": {k: bool(v) for k, v in mut.items()}},
           "hypotheses": hypotheses(per_seed, boot, a.quick),
           "knockouts": [] if a.quick else knockout_table(per_seed, boot),
           "means": means, "extra": extra, "exit_code": exit_code}
    write_report(build_report(ctx), json_report(ctx), a.json)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}")
        sys.exit(4)

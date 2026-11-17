# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0368 · Ibn al-Khatib (1313-1374)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 19, Minds 361-380 Available on Amazon https://www.amazon.com/dp/B0HQX147Y7
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
"""
Chapter 0368 · Ibn al-Khatib · THE THRESHOLD PARTITION (hawa' / 'adwa / thiyab)

Thesis
    A correlated affliction is split into a shared field and a propagated cause by
    reading who and what crossed which threshold, with people walled off inside the
    same air as the decisive control; the split, not the fit, is what must survive an
    intervention nobody has yet observed.

Evidence and provenance (provenance: belief)
    D1 primary  Muqni'at al-sa'il 'an al-marad al-ha'il, ed. and German tr. M. J. Mueller,
                Sitzungsber. Bayer. Akad. Wiss. 1863, II, 1-40: contagion is established by
                experience, investigation, sense-perception and reliable reports.
    D2 primary  same treatise: contact with the sick is followed by illness, separation by
                safety; garments, vessels and even an earring carry it; a healthy port falls
                ill after arrivals from an infected land.
    D3 primary  same treatise (as read in Stearns 2011 and Dols 1977): the Muslim captives in
                the Seville arsenal and the tent-dwelling nomads of Ifriqiya were spared.
    D4 primary  same treatise: a transmitted proof that manifestly contradicts the senses must
                be interpreted (ta'wil); it may not be used to deny what the senses show.
    D5 scholarship  he kept corrupted air as a cause beside contagion (Stearns 2011, ch. 3).
    D6 speculation  blind spot: no animal vector; the rodent-flea reservoir of Yersinia pestis
                (Bos et al. 2011) cannot appear in a ledger of human contacts and objects.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1,D2  M1 'adwa channel: non-negative typed weights on observed contacts    C6.1  H-SIG
    D2     M2 thiyab reservoir: object arrivals decayed by a learned persistence C6.3  H-SIG
    D3     M3 same-air control: shrunken attack rate of the unexposed in the    C6.4  H-NEC
              same town, the only input that measures the field directly
    D5     M4 air covariates (crowding, humidity, season) into the field net    -     H-NEC (matched)
    D4     M5 ta'wil start: training begins at 'no contagion' (weights ~0)      C6.5  recovery
    D6     hidden street-vector world (rats move between adjacent houses)       -     H-BLIND

Research question (causal inference and intervention)
    Contagion and shared-environment effects are generically confounded in observational
    network data (Shalizi and Thomas 2011). Can a learner that is told which units share the
    environment but not the contacts separate the two, and does the separation predict the
    effect of cordons and environmental shocks that never occur in training?

Closest prior art and the delta
    Endemic-epidemic count models (hhh4; Held, Hoehle and Hofmann 2005), diffusion-network
    inference (NetRate; Gomez-Rodriguez et al. 2011) and negative-control exposures
    (Lipsitch et al. 2010). hhh4 is implemented as the rival 'place model'; a size-matched
    MLP over the same case file is the baseline. Delta: an amortised negative-control
    estimate of the field (computed from the same-environment unexposed, refreshed daily)
    is the field's only causal input, combined with a typed additive contact channel and a
    carried-object memory with learned persistence, trained from a no-contagion start.

Blind spot
    The architecture can only blame air, people and objects. Where transmission runs
    through an unrecorded vector that clusters by street (rats), the place-free design is
    predicted to lose ground to a model that reads neighbourhood prevalence.

Task (generative process)
    Each episode is a 26-day outbreak in a port, a city, an inland town and a nomad camp
    (384 people in 4-person households, about one settled household in seven walled). Daily hazard =
    air(town, day) + typed contact terms + object reservoir (+ rats in the blind world).
    Air rises with crowding, humidity, season and an unobserved squalor-and-drift term that
    also raises market contact; only 60% of market contacts are recorded. Rows are
    susceptible person-days; the target is infection that day. Shifted episodes impose
    cordons (markets 10%, objects burned, ships barred) or corrupt winds (air x3.5).

Limits
    Synthetic world, binary outcomes, fixed infectious period, known household roster.
    It is a research prototype for causal decomposition inside an AGI world model, not an
    epidemiological tool and not a claim about the historical person's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 2, "revision_log": [
        ("rev 2, after a pilot on seeds 0-1 and before the confirmatory run: towns enlarged from 12/12/10/6 to "
         "30/30/24/12 households (the pilot same-air control saw ~0.6 events per town-window, correlation "
         "0.4-0.58 with the true field: a power problem of the simulator); training switched to 8192-row "
         "minibatches for the time budget; C4 band redefined against the best constant forecast on the "
         "held-out split (train and held-out base rates differ by episode). Hypotheses, metrics and mesi "
         "unchanged; pilot values are not reported.")],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude",
                   "generator_version": "Opus 5.5", "date": "2026-10-01"},
    "id": 368, "figure": "Ibn al-Khatib", "born": 1313, "died": 1374,
    "civilization": "Andalusi Arab (Nasrid)", "provenance": "belief",
    "thesis": ("Separate a shared field from a propagated cause by reading who and what crossed which "
               "threshold, using the people walled off inside the same air as the control."),
    "evidence": [
        {"id": "D1", "claim": "Contagion is established by experience, investigation, the senses and reliable reports",
         "basis": "primary", "source": "Muqni'at al-sa'il, ed. Mueller 1863"},
        {"id": "D2", "claim": "Contact vs separation; garments, vessels, an earring; ports infected by arrivals",
         "basis": "primary", "source": "Muqni'at al-sa'il, ed. Mueller 1863"},
        {"id": "D3", "claim": "Captives in the Seville arsenal and tent-dwelling nomads were spared",
         "basis": "primary", "source": "Muqni'at al-sa'il; Stearns 2011; Dols 1977"},
        {"id": "D4", "claim": "A transmitted proof contradicted by manifest sense evidence must be interpreted",
         "basis": "primary", "source": "Muqni'at al-sa'il; Stearns 2011"},
        {"id": "D5", "claim": "Corrupted air retained as a cause beside contagion",
         "basis": "scholarship", "source": "Stearns, Infectious Ideas (2011), ch. 3"},
        {"id": "D6", "claim": "No animal vector in his causal ontology",
         "basis": "speculation", "source": "Bos et al., Nature 478 (2011); Dean et al., PNAS 115 (2018)"},
    ],
    "research_question": {"category": "causal inference and intervention",
                          "question": ("Can a learner given same-environment controls separate contagion from a "
                                       "confounded shared field and predict unseen interventions?")},
    "mechanism": {
        "name": "Threshold Partition (hawa' / 'adwa / thiyab)", "family": "causal decomposition; additive hazards",
        "signature_modules": ["adwa_contact", "thiyab_reservoir", "same_air_control"],
        "closest_prior_art": ["hhh4 endemic-epidemic model (Held, Hoehle, Hofmann 2005)",
                              "NetRate diffusion inference (Gomez-Rodriguez, Balduzzi, Schoelkopf 2011)",
                              "negative-control exposures (Lipsitch, Tchetgen Tchetgen, Cohen 2010)"],
        "overlap": "Medium",
        "prior_art_queries": ["contagion homophily confounding identification network",
                              "endemic epidemic decomposition surveillance counts",
                              "negative control exposure shared environment",
                              "fomite reservoir persistence transmission model learned",
                              "infectious jailbreak multi-agent memory spread"],
        "contribution_type": "mechanism",
        "delta": ("Field estimated only through an amortised same-environment negative control, plus typed "
                  "contact weights and a carried-object memory, trained from a no-contagion start."),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M2", "property_test": "C6.3", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.4", "hypothesis": "H-NEC"},
        {"doctrine": "D5", "mechanism": "M4", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M5", "property_test": "C6.5", "hypothesis": "H-RIVAL"},
        {"doctrine": "D6", "mechanism": "none", "property_test": "-", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The partition predicts intervention episodes better than a size-matched MLP",
         "metric": "skill (nats per 1000 person-days over trivial)", "split": "shifted",
         "comparison": "model - baseline", "direction": "greater", "mesi": 1.0, "seeds": 5},
        {"id": "H-NEC", "statement": "Lesioning the same-air control hurts more than lesioning the air covariates",
         "metric": "skill", "split": "shifted",
         "comparison": "signature_knockout - matched_knockout", "direction": "less", "mesi": 1.0, "seeds": 5},
        {"id": "H-BLIND", "statement": "With a street-clustered hidden vector the partition loses ground to the MLP",
         "condition": "hidden_vector world", "grounding": "no animal vector in his ontology",
         "metric": "skill", "comparison": "(model - baseline)_blind - (model - baseline)_iid",
         "direction": "less", "mesi": 1.0, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The partition beats the place model (hhh4-style; air and density) under intervention",
         "metric": "skill", "split": "shifted", "comparison": "model - rival", "direction": "greater",
         "mesi": 1.0, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.05, "margin_over_trivial": 5.0, "shuffled_band": 1.0,
                   "gradcheck_rel": 1e-5},
    "probe_predictions": [{"probe": "P7", "expected": "above"}, {"probe": "P9", "expected": "above"},
                          {"probe": "P1", "expected": "below"}, {"probe": "P3", "expected": "equal to baseline"},
                          {"probe": "P6", "expected": "above"}, {"probe": "P5", "expected": "below"}],
    "dialectic_links": [{"chapter": 375, "relation": "rival",
                         "test": "H-RIVAL (Ibn Khaldun's density-and-air account of pestilence, Muqaddimah ch. 3)"},
                        {"chapter": 367, "relation": "contemporary", "test": "none (Boccaccio records garment transmission)"}],
    "corpus_neighbors": [
        {"chapter": 375, "similarity": None, "difference": "Ibn Khaldun tests a report against a world model; here the world model's causal split is learned"},
        {"chapter": 233, "similarity": None, "difference": "al-Razi keeps a register of named counterexamples; here contrasts identify channels"},
        {"chapter": 351, "similarity": None, "difference": "Ibn Taymiyya's contact is epistemic certainty; here contact is a causal edge"},
        {"chapter": 366, "similarity": None, "difference": "Dusan routes where an authorless loss lands; here the loss is traced to its path"},
    ],
    "barometer": {
        "cognitive_processing": ["intervention prediction on unseen cordons and winds (H-SIG)"],
        "embodied_cognition": [], "world_modeling": ["field vs transmission decomposition; parameter recovery"],
        "consciousness": [], "language_understanding": [], "emotional_intelligence": [],
        "creativity": [], "autonomy": [],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "attributing compromised agents to agent-to-agent messages vs shared poisoned memory vs common input",
         "sector": "AI security", "dataset": "Agent Smith infectious-jailbreak simulations (Gu et al., ICML 2024)"},
        {"use": "ward outbreak source attribution, research decision support only", "sector": "healthcare research",
         "dataset": "SocioPatterns hospital-ward proximity data (Vanhems et al., PLoS ONE 2013)"},
        {"use": "incident root cause: shared dependency vs call-graph cascade", "sector": "cloud operations",
         "dataset": "Train-Ticket microservice benchmark (Zhou et al., IEEE TSE 2018)"},
    ],
    "safety_notes": "Synthetic plague-like data only; no dosing, treatment or surveillance of persons.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

TOWN_HH = (30, 30, 24, 12)         # households: port, city, inland town, nomad camp
HH_SIZE, DAYS, INF_DAYS, N_LAG, WINDOW = 4, 26, 4, 6, 3
OBS_MARKET = 0.6                    # share of market contacts that reach the ledger
TRUE = {"air0": 0.010, "beta_h": 0.055, "beta_m": 0.050, "beta_s": 0.30,
        "gamma": 0.050, "rho": 0.72, "kappa": 0.018}
PRIOR_MEAN, PRIOR_STRENGTH = 0.02, 60.0
N_FEAT = 18                         # 3 contacts, 6 object lags, 5 field inputs, 2 place rates, 2 flags
LAYOUT = {"contact": slice(0, 3), "obj": slice(3, 9), "z": slice(9, 14),
          "cov": slice(10, 14), "place": slice(14, 16)}
CHANNELS = ("air", "household", "market", "ship", "object", "rat")
TASK_TYPES = MIND_CARD["task_types"]
BUDGET_S, QUICK_BUDGET_S = 180.0, 20.0
ACTIVE_MUTANT = None


# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


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


def clip_global_norm(grads, max_norm):
    tot = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if tot > max_norm:
        for k in grads:
            grads[k] *= max_norm / (tot + 1e-12)
    return tot


def finite_difference_check(loss_fn, params, grads, rng, eps=1e-6, n_entries=20):
    """Central differences on n_entries random entries plus the largest-gradient entry."""
    out = {}
    for k, v in params.items():
        flat = v.reshape(-1)
        idx = set(rng.choice(flat.size, size=min(n_entries, flat.size), replace=False).tolist())
        idx.add(int(np.argmax(np.abs(grads[k]).reshape(-1))))
        worst = 0.0
        for i in idx:
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), float(grads[k].reshape(-1)[i])
            den = max(abs(num), abs(ana))
            rel = 0.0 if den < 1e-9 else abs(num - ana) / den
            worst = max(worst, rel)
        out[k] = worst
    return out


def paired_bootstrap(diffs, rng, n_boot=2000):
    d = np.asarray(diffs, dtype=float)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def verdict(mean, lo, hi, direction, mesi):
    if direction == "greater":
        if lo > 0 and mean >= mesi:
            return "supported"
        return "contradicted" if hi < 0 else "inconclusive"
    if hi < 0 and mean <= -mesi:
        return "supported"
    return "contradicted" if lo > 0 else "inconclusive"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------- data and tasks
def make_world(rng, regime):
    """Static layout of one outbreak season: towns, households, walls, quarters, air field."""
    hh_town = np.repeat(np.arange(4), TOWN_HH)
    H = hh_town.size
    person_hh = np.repeat(np.arange(H), HH_SIZE)
    crowd = np.array([rng.uniform(.6, 1.), rng.uniform(.5, 1.), rng.uniform(.2, .6), .05])
    humid = np.array([rng.uniform(.4, 1.), rng.uniform(0, 1), rng.uniform(0, .8), .1])
    squalor = np.append(rng.normal(0, .55, 3), -.6)          # unobserved; raises air AND markets
    walled = (rng.random(H) < 0.14) & (hh_town < 3)          # captives in the arsenal, sealed houses
    phase = rng.uniform(0, 2 * np.pi)
    drift = np.cumsum(rng.normal(0, .13, (DAYS, 4)), axis=0)  # unobserved day-to-day air drift
    t = np.arange(DAYS)[:, None]
    season = 1 + .45 * np.sin(2 * np.pi * t / DAYS + phase)
    air = (TRUE["air0"] * (.3 + crowd) * (.5 + humid) * np.exp(.8 * squalor + drift) * season)
    air[:, 3] *= .4                                           # tents: the air is not shut in
    w = {"hh_town": hh_town, "person_hh": person_hh, "person_town": hh_town[person_hh],
         "crowd": crowd, "humid": humid, "walled": walled, "p_walled": walled[person_hh],
         "quarter": np.concatenate([np.arange(n) // 3 + 100 * c for c, n in enumerate(TOWN_HH)]),
         "phase": phase, "air": air, "m_rate": 2.2 * crowd * np.exp(.5 * squalor),
         "o_rate": .45 * crowd * np.exp(.3 * squalor), "rats": regime == "hidden_vector",
         "market_mult": np.ones((DAYS, 4)), "burn": np.zeros((DAYS, 4), bool), "ships": np.ones(DAYS, bool)}
    if regime == "intervention":
        _impose_interventions(rng, w)
    for key, P in (("open_p", w["person_town"]), ("open_h", hh_town)):
        open_mask = ~(w["p_walled"] if key == "open_p" else walled)
        ids = [np.flatnonzero(open_mask & (P == c)) for c in range(4)]
        w[key] = (np.concatenate(ids), np.cumsum([0] + [len(i) for i in ids])[:-1], np.array([len(i) for i in ids]))
    return w


def _impose_interventions(rng, w):
    """Cordon (markets, objects, ships) or a corrupt wind, chosen per settled town."""
    for c in range(3):
        mode = rng.choice(["cordon", "wind", "none"], p=[.42, .42, .16])
        if mode == "cordon":
            t0 = rng.integers(5, 10)
            w["market_mult"][t0:, c] = .1
            w["burn"][t0:, c] = True
            if c == 0:
                w["ships"][t0:] = False
        elif mode == "wind":
            t0 = rng.integers(6, 13)
            w["air"][t0:t0 + 8, c] *= 3.5


def _draw_targets(rng, open_tuple, towns):
    concat, start, count = open_tuple
    idx = (rng.random(towns.size) * np.maximum(count[towns], 1)).astype(int)
    ok = count[towns] > 0
    return concat[start[towns] + np.minimum(idx, np.maximum(count[towns] - 1, 0))], ok


def _daily_exposure(rng, w, t, inf):
    """Contacts, ship arrivals and object transfers on day t (true and recorded)."""
    P, H = w["person_hh"].size, w["hh_town"].size
    hh_inf = np.bincount(w["person_hh"], weights=inf, minlength=H)
    rate = w["m_rate"][w["person_town"]] * w["market_mult"][t, w["person_town"]] * ~w["p_walled"]
    src = np.repeat(np.arange(P), rng.poisson(rate))
    tgt, ok = _draw_targets(rng, w["open_p"], w["person_town"][src])
    hit = inf[tgt] * ok
    rec = rng.random(src.size) < OBS_MARKET
    n_m_true = np.bincount(src, weights=hit, minlength=P)
    n_m_obs = np.bincount(src, weights=hit * rec, minlength=P)
    n_s = np.zeros(P)
    if w["ships"][t] and rng.random() < .35 and rng.random() < .65:
        port = w["open_p"][0][:w["open_p"][2][0]]
        n_s[rng.choice(port, size=3, replace=False)] = 1.0
    hsrc = np.repeat(np.arange(H), rng.poisson(w["o_rate"][w["hh_town"]] * ~w["walled"]))
    htgt, hok = _draw_targets(rng, w["open_h"], w["hh_town"][hsrc])
    carried = hok & (hh_inf[hsrc] > 0) & ~w["burn"][t, w["hh_town"][hsrc]] & (htgt != hsrc)
    obj_in = np.bincount(htgt, weights=carried, minlength=H)
    return hh_inf, n_m_true, n_m_obs, n_s, obj_in


def run_episode(rng, regime, ep_id):
    """Simulate one season; return susceptible person-day rows with features and outcomes."""
    w = make_world(rng, regime)
    P, H = w["person_hh"].size, w["hh_town"].size
    state, timer = np.zeros(P, int), np.zeros(P, int)
    seeds = rng.choice(w["open_p"][0][:w["open_p"][2][0]], size=2, replace=False)
    state[seeds], timer[seeds] = 1, INF_DAYS
    obj_hist, rats = np.zeros((DAYS + N_LAG, H)), np.zeros(400)
    unexp_k, unexp_n = np.zeros((DAYS, 4)), np.zeros((DAYS, 4))
    dec = TRUE["rho"] ** np.arange(N_LAG)
    rows, ys, chans, keys = [], [], [], []
    for t in range(DAYS):
        inf = (state == 1).astype(float)
        hh_inf, n_m_true, n_m_obs, n_s, obj_in = _daily_exposure(rng, w, t, inf)
        obj_hist[t + N_LAG] = obj_in
        o = obj_hist[t + N_LAG - np.arange(N_LAG)][:, w["person_hh"]].T     # P x N_LAG, lag 0 first
        n_h = hh_inf[w["person_hh"]] - inf
        q_of_p = w["quarter"][w["person_hh"]]
        q_inf = np.bincount(q_of_p, weights=inf, minlength=400)
        q_size = np.maximum(np.bincount(q_of_p, minlength=400), 1)
        rats = .75 * rats + q_inf if w["rats"] else rats
        comp = np.column_stack([w["air"][t, w["person_town"]], TRUE["beta_h"] * n_h,
                                TRUE["beta_m"] * n_m_true, TRUE["beta_s"] * n_s,
                                TRUE["gamma"] * (o @ dec), TRUE["kappa"] * rats[q_of_p]])
        lam = comp.sum(1)
        new = (state == 0) & (rng.random(P) < -np.expm1(-lam))
        cum = np.cumsum(comp, 1) / lam[:, None]
        chan = np.where(new, (rng.random(P)[:, None] > cum).sum(1), -1)
        z0 = _same_air(unexp_k, unexp_n, t, w["person_town"])
        town_frac = (np.bincount(w["person_town"], weights=inf, minlength=4) / np.bincount(w["person_town"]))
        season = 2 * np.pi * t / DAYS + w["phase"]
        X = np.column_stack([n_h, n_m_obs, n_s, o, z0, w["crowd"][w["person_town"]],
                             w["humid"][w["person_town"]], np.full(P, np.sin(season)), np.full(P, np.cos(season)),
                             town_frac[w["person_town"]], q_inf[q_of_p] / q_size[q_of_p],
                             w["p_walled"], w["person_town"] == 3])
        sus = state == 0
        unexposed = sus & (n_h == 0) & (n_m_obs == 0) & (n_s == 0) & (o.sum(1) == 0)
        unexp_n[t] = np.bincount(w["person_town"], weights=unexposed, minlength=4)
        unexp_k[t] = np.bincount(w["person_town"], weights=unexposed & new, minlength=4)
        rows.append(X[sus]), ys.append(new[sus]), chans.append(chan[sus])
        keys.append(np.column_stack([np.flatnonzero(sus), np.full(sus.sum(), t)]))
        timer[state == 1] -= 1
        state[(state == 1) & (timer <= 0)] = 2
        state[new], timer[new] = 1, INF_DAYS
    return {"X": np.vstack(rows).astype(float), "y": np.concatenate(ys).astype(float),
            "ch": np.concatenate(chans), "key": np.vstack(keys), "ep": ep_id}


def same_air_estimate(k, n, prior_mean=PRIOR_MEAN, strength=PRIOR_STRENGTH):
    """Beta-binomial shrinkage of the unexposed attack rate (non-gradient Bayesian rule)."""
    return (k + strength * prior_mean) / (n + strength)


def _same_air(unexp_k, unexp_n, t, person_town):
    lo = max(0, t - WINDOW)
    rate = same_air_estimate(unexp_k[lo:t].sum(0), unexp_n[lo:t].sum(0))
    return np.log(rate / PRIOR_MEAN)[person_town]


SPLITS = {"train": ("observational", 12), "val": ("observational", 5), "iid": ("observational", 6),
          "shift": ("intervention", 8), "blind_train": ("hidden_vector", 12), "blind_test": ("hidden_vector", 6)}


def make_data(seed, scale=1.0):
    """All splits for one seed; every episode has its own child generator and id."""
    out = {}
    children = np.random.SeedSequence([CHAPTER_ID, seed]).spawn(len(SPLITS))
    for (name, (regime, n)), ss in zip(SPLITS.items(), children):
        eps = [run_episode(np.random.default_rng(c), regime, f"{seed}:{name}:{i}")
               for i, c in enumerate(ss.spawn(max(2, int(round(n * scale)))))]
        out[name] = {"X": np.vstack([e["X"] for e in eps]), "y": np.concatenate([e["y"] for e in eps]),
                     "ch": np.concatenate([e["ch"] for e in eps]),
                     "ep": np.concatenate([np.full(e["y"].size, e["ep"]) for e in eps]),
                     "key": np.vstack([e["key"] for e in eps])}
    return out


CHAPTER_ID = 368


# ----------------------------------------------------------------------------- model
def build_model(in_dim=N_FEAT, out_dim=1, task_type="vector_classification", rng=None, kind="partition",
                hidden=24, **cfg):
    """kind: 'partition' (signature), 'mlp' (size-matched baseline), 'place' (hhh4-style rival)."""
    if out_dim != 1 or task_type not in TASK_TYPES:
        raise ValueError("binary vector_classification only")
    rng = rng if rng is not None else np.random.default_rng(0)
    lay = LAYOUT if in_dim == N_FEAT else _generic_layout(in_dim)
    m = {"kind": kind, "layout": lay, "in_dim": in_dim, "ko": {}, "ko_stats": {},
         "mu": np.zeros(in_dim), "sd": np.ones(in_dim), "_cache": None}
    if kind == "partition":
        nz, nc, no = lay["z"].stop - lay["z"].start, lay["contact"].stop - lay["contact"].start, \
            lay["obj"].stop - lay["obj"].start
        m["params"] = {"W1": rng.normal(0, .4, (nz, hidden)), "b1": np.zeros(hidden),
                       "w2": rng.normal(0, .1, hidden), "wl": np.zeros(nz),
                       "b2": np.array([math.log(math.expm1(PRIOR_MEAN))]),
                       "u_contact": np.full(nc, -6.0), "g_obj": np.array([-6.0]),     # ta'wil start
                       "r_obj": np.array([0.0])}
        m["n_obj"] = no
    elif kind == "mlp":
        h = cfg.get("mlp_hidden", 9)
        m["params"] = {"W1": rng.normal(0, 1 / math.sqrt(in_dim), (in_dim, h)), "b1": np.zeros(h),
                       "w2": rng.normal(0, .3, h), "b2": np.array([math.log(PRIOR_MEAN / (1 - PRIOR_MEAN))])}
    elif kind == "place":
        nc = lay["cov"].stop - lay["cov"].start
        m["params"] = {"w": np.zeros(nc), "b": np.array([math.log(math.expm1(PRIOR_MEAN))]),
                       "phi": np.array([-3.0]), "psi": np.array([-3.0])}
    else:
        raise ValueError(kind)
    return m


def _generic_layout(d):
    a = max(1, d // 3)
    b = max(1, (d - a) // 2)
    zst = a + b
    return {"contact": slice(0, a), "obj": slice(a, zst), "z": slice(zst, d), "cov": slice(zst, d),
            "place": slice(zst, d)}


def set_normalisation(m, X):
    m["mu"], m["sd"] = X.mean(0), X.std(0) + 1e-8
    m["_cache"] = None


def _prep(m, X):
    """Contiguous standardised views of a data matrix, cached because the data never change."""
    if isinstance(X, dict):                  # an already prepared minibatch
        return X
    c = m.get("_cache")
    if c is not None and c[0] is X:
        return c[1]
    lay = m["layout"]
    if m["kind"] == "partition":
        z = (X[:, lay["z"]] - m["mu"][lay["z"]]) / m["sd"][lay["z"]]
        out = {"n": np.ascontiguousarray(X[:, lay["contact"]]), "o": np.ascontiguousarray(X[:, lay["obj"]]),
               "z": np.ascontiguousarray(z)}
    elif m["kind"] == "place":
        out = {"c": np.ascontiguousarray((X[:, lay["cov"]] - m["mu"][lay["cov"]]) / m["sd"][lay["cov"]]),
               "pl": np.ascontiguousarray(X[:, lay["place"]])}
    else:
        out = {"x": np.ascontiguousarray((X - m["mu"]) / m["sd"])}
    m["_cache"] = (X, out)
    return out


def _partition_parts(m, X, p=None):
    p = m["params"] if p is None else p
    ko, q = m["ko"], _prep(m, X)
    n, o, z = q["n"], q["o"], q["z"]
    if "same_air_control" in ko or "air_covariates" in ko:
        z = z.copy()
        if "same_air_control" in ko:
            z[:, 0] = 0.0                   # standardised training mean
        if "air_covariates" in ko:
            z[:, 1:] = 0.0
    h = np.tanh(z @ p["W1"] + p["b1"])
    s = p["b2"][0] + z @ p["wl"] + h @ p["w2"]
    A = softplus(s)
    beta = softplus(p["u_contact"])
    Cp = n * beta
    rho = float(sigmoid(p["r_obj"])[0])
    dec = rho ** np.arange(o.shape[1])
    S = o @ dec
    F = softplus(p["g_obj"][0]) * S
    if "adwa_contact" in ko:
        Cp = np.broadcast_to(m["ko_stats"].get("C", np.zeros(n.shape[1])) * (ko["adwa_contact"] == "mean"),
                             Cp.shape)
    if "thiyab_reservoir" in ko:
        F = np.full_like(F, m["ko_stats"].get("F", 0.0) * (ko["thiyab_reservoir"] == "mean"))
    lam = A + Cp.sum(1) + F + 1e-9
    return {"z": z, "h": h, "s": s, "A": A, "Cp": Cp, "n": n, "o": o, "rho": rho, "dec": dec,
            "S": S, "F": F, "lam": lam}


def _place_parts(m, X, p=None):
    p = m["params"] if p is None else p
    q = _prep(m, X)
    endo = softplus(p["b"][0] + q["c"] @ p["w"])
    lam = endo + softplus(p["phi"][0]) * q["pl"][:, 0] + softplus(p["psi"][0]) * q["pl"][:, -1] + 1e-9
    return {"c": q["c"], "pl": q["pl"], "endo": endo, "lam": lam}


def _mlp_parts(m, X, p=None):
    p = m["params"] if p is None else p
    x = _prep(m, X)["x"]
    h = np.tanh(x @ p["W1"] + p["b1"])
    return {"x": x, "h": h, "logit": h @ p["w2"] + p["b2"][0]}


def _hazard_nll(lam, y):
    ll = np.where(y > .5, np.log(-np.expm1(-lam)), -lam)
    dl = np.where(y > .5, -1.0 / np.expm1(lam), 1.0)
    return -float(ll.mean()), dl / y.size


def loss_and_grads(model, batch, params=None):
    """Mean negative log-likelihood of the day's outcome and its exact gradient."""
    X, y = batch
    p = model["params"] if params is None else params
    k = model["kind"]
    if k == "mlp":
        f = _mlp_parts(model, X, p)
        loss = float(np.mean(softplus(f["logit"]) - y * f["logit"]))
        d = (sigmoid(f["logit"]) - y) / y.size
        dh = d[:, None] * p["w2"] * (1 - f["h"] ** 2)
        g = {"w2": f["h"].T @ d, "b2": np.array([d.sum()]), "W1": f["x"].T @ dh, "b1": dh.sum(0)}
    elif k == "place":
        f = _place_parts(model, X, p)
        loss, dl = _hazard_nll(f["lam"], y)
        de = dl * sigmoid(p["b"][0] + f["c"] @ p["w"])
        g = {"w": f["c"].T @ de, "b": np.array([de.sum()]),
             "phi": np.array([(dl * f["pl"][:, 0]).sum() * sigmoid(p["phi"])[0]]),
             "psi": np.array([(dl * f["pl"][:, -1]).sum() * sigmoid(p["psi"])[0]])}
    else:
        f = _partition_parts(model, X, p)
        loss, dl = _hazard_nll(f["lam"], y)
        ds = dl * sigmoid(f["s"])
        dpre = f["h"] * f["h"]
        np.subtract(1.0, dpre, out=dpre)
        dpre *= ds[:, None]
        dpre *= p["w2"]
        kk = np.arange(f["o"].shape[1])
        dS = f["o"] @ (kk * f["rho"] ** np.maximum(kk - 1, 0))
        g = {"b2": np.array([ds.sum()]), "wl": f["z"].T @ ds, "w2": f["h"].T @ ds,
             "W1": f["z"].T @ dpre, "b1": dpre.sum(0),
             "u_contact": (dl[:, None] * f["n"]).sum(0) * sigmoid(p["u_contact"]),
             "g_obj": np.array([(dl * f["S"]).sum() * sigmoid(p["g_obj"])[0]]),
             "r_obj": np.array([(dl * dS).sum() * softplus(p["g_obj"][0]) * f["rho"] * (1 - f["rho"])])}
    if ACTIVE_MUTANT in MUTANTS and MUTANTS[ACTIVE_MUTANT]["zero_grad"] in g:
        g[MUTANTS[ACTIVE_MUTANT]["zero_grad"]] = np.zeros_like(g[MUTANTS[ACTIVE_MUTANT]["zero_grad"]])
    return loss, g


def predict(model, X):
    """Probability of infection on the row's day."""
    if model["kind"] == "mlp":
        return sigmoid(_mlp_parts(model, X)["logit"])
    lam = (_place_parts if model["kind"] == "place" else _partition_parts)(model, X)["lam"]
    return -np.expm1(-lam)


def log_lik(model, X, y):
    if model["kind"] == "mlp":
        lg = _mlp_parts(model, X)["logit"]
        return y * lg - softplus(lg)
    lam = (_place_parts if model["kind"] == "place" else _partition_parts)(model, X)["lam"]
    return np.where(y > .5, np.log(-np.expm1(-lam)), -lam)


def hidden_states(model, X):
    if model["kind"] == "mlp":
        return _mlp_parts(model, X)["h"]
    if model["kind"] == "place":
        f = _place_parts(model, X)
        return np.column_stack([f["endo"], f["lam"] - f["endo"]])
    f = _partition_parts(model, X)
    return np.column_stack([f["h"], f["A"], f["Cp"], f["F"]])


def attribution(model, X):
    """Exact per-channel share of the partition's hazard: air, household, market, ship, object."""
    f = _partition_parts(model, X)
    parts = np.column_stack([f["A"], f["Cp"], f["F"]])
    return parts / parts.sum(1, keepdims=True), parts


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def modules(model):
    if model["kind"] != "partition":
        return {"all": {"params": list(model["params"]), "role": "baseline network", "signature": False}}
    return {
        "adwa_contact": {"params": ["u_contact"], "signature": True,
                         "role": "non-negative typed weights on recorded contact counts (additive hazard)"},
        "thiyab_reservoir": {"params": ["g_obj", "r_obj"], "signature": True,
                             "role": "geometrically decaying memory of carried-object arrivals, learned persistence"},
        "same_air_control": {"params": ["W1[0]", "wl[0]"], "signature": True,
                             "role": "negative-control field input: shrunken attack rate of same-town unexposed"},
        "air_covariates": {"params": ["W1[1:]", "wl[1:]", "b1", "w2", "b2"], "signature": False,
                           "role": "softplus field network over environment covariates"},
    }


def knockout(model, name, mode="mean"):
    m = dict(model)
    m["ko"] = dict(model["ko"])
    m["ko"][name] = mode
    return m


def set_ko_stats(model, X):
    f = _partition_parts(model, X)
    model["ko_stats"] = {"C": f["Cp"].mean(0), "F": float(f["F"].mean())}


# ----------------------------------------------------------------------------- baselines, rivals, registries
BASELINES = {"mlp": "size-matched tanh MLP over the full case file (18 inputs incl. town and quarter rates)",
             "place": "endemic-epidemic place model (hhh4-style): covariate field + town and quarter prevalence"}

MUTANTS = {
    "sign_flip_update": {"zero_grad": None, "sign": -1.0, "lr_scale": 1.0},
    "zero_lr": {"zero_grad": None, "sign": 1.0, "lr_scale": 0.0},
    "zero_grad_contact": {"zero_grad": "u_contact", "sign": 1.0, "lr_scale": 1.0},
    "zero_grad_field": {"zero_grad": "W1", "sign": 1.0, "lr_scale": 1.0},
}
KNOCKOUTS = ["same_air_control", "air_covariates", "adwa_contact", "thiyab_reservoir"]


# ----------------------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=0.03, clip=5.0, batch=8192):
    """Adam on minibatches drawn by `rng`; the trace starts and ends with the full-data loss."""
    X, y = data
    rng = rng if rng is not None else np.random.default_rng(0)
    mut = MUTANTS.get(ACTIVE_MUTANT, {"sign": 1.0, "lr_scale": 1.0})
    opt = Adam(model["params"], lr=lr * mut["lr_scale"])
    q = _prep(model, X)
    trace = [loss_and_grads(model, (X, y))[0]]
    for _ in range(budget):
        idx = rng.integers(0, y.size, min(batch, y.size)) if y.size > batch else np.arange(y.size)
        loss, g = loss_and_grads(model, ({k: v[idx] for k, v in q.items()}, y[idx]))
        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        clip_global_norm(g, clip)
        opt.step(model["params"], g, sign=mut["sign"])
        trace.append(loss)
    trace.append(loss_and_grads(model, (X, y))[0])
    return trace


def train_selected(kind, d, rng, budget, grid=(0.01, 0.03), split="train"):
    """Same grid and budget for every model; selection on the validation split only
    (the blind world reuses the learning rate selected on the standard world's validation)."""
    best = None
    for lr in grid:
        m = build_model(N_FEAT, 1, rng=np.random.default_rng(rng.integers(1 << 31)), kind=kind)
        set_normalisation(m, d[split]["X"])
        fit(m, (d[split]["X"], d[split]["y"]), budget, rng, lr=lr)
        m["lr"] = lr
        v = -float(log_lik(m, d["val"]["X"], d["val"]["y"]).mean()) if split == "train" else 0.0
        if best is None or v < best[0]:
            best = (v, m)
    if kind == "partition":
        set_ko_stats(best[1], d[split]["X"])
    return best[1]


def skill(model, part, p0):
    """Nats per 1000 person-days above the constant-rate forecast p0."""
    ll = log_lik(model, part["X"], part["y"])
    ll0 = part["y"] * math.log(p0) + (1 - part["y"]) * math.log(1 - p0)
    return 1000.0 * float((ll - ll0).mean())


# ----------------------------------------------------------------------------- correctness tests
def c1_gradcheck(d, rng, steps=60):
    Xb, yb = d["train"]["X"][:400], d["train"]["y"][:400]
    yb = yb.copy()
    yb[::7] = 1.0                       # make sure positive rows enter the small check batch
    res, worst = {}, 0.0
    for kind in ("partition", "mlp", "place"):
        m = build_model(N_FEAT, 1, rng=np.random.default_rng(1), kind=kind)
        set_normalisation(m, d["train"]["X"])
        for stage in ("init", "trained"):
            if stage == "trained":
                fit(m, (d["train"]["X"], d["train"]["y"]), steps, rng)
            _, g = loss_and_grads(m, (Xb, yb))
            errs = finite_difference_check(lambda pp: loss_and_grads(m, (Xb, yb), pp)[0],
                                           m["params"], g, rng)
            res[f"{kind}@{stage}"] = errs
            worst = max(worst, max(errs.values()))
    tensors = sum(len(v) for v in res.values())
    return worst <= MIND_CARD["thresholds"]["gradcheck_rel"], worst, tensors, res


def c2_determinism(d):
    outs = []
    for _ in range(2):
        m = build_model(N_FEAT, 1, rng=np.random.default_rng(7), kind="partition")
        set_normalisation(m, d["train"]["X"])
        tr = fit(m, (d["train"]["X"], d["train"]["y"]), 30, np.random.default_rng(3))
        outs.append((tr, predict(m, d["iid"]["X"]), m))
    same = outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])
    finite = all(np.all(np.isfinite(v)) for v in outs[0][2]["params"].values()) and np.all(np.isfinite(outs[0][1]))
    return bool(same and finite), finite


def c3_learning(d, rng, budget):
    m = build_model(N_FEAT, 1, rng=np.random.default_rng(11), kind="partition")
    set_normalisation(m, d["train"]["X"])
    tr = fit(m, (d["train"]["X"], d["train"]["y"]), budget, rng)
    p0 = float(d["train"]["y"].mean())
    drop = 1 - tr[-1] / tr[0]
    sk = skill(m, d["iid"], p0)
    th = MIND_CARD["thresholds"]
    return bool(drop >= th["loss_drop_fraction"] and sk >= th["margin_over_trivial"]), drop, sk


def c4_shuffled(d, rng, budget):
    """Permuted training outcomes: held-out skill must stay within the band of feature-free
    forecasts, i.e. no better than the best constant (oracle held-out rate) plus the band."""
    m = build_model(N_FEAT, 1, rng=np.random.default_rng(13), kind="partition")
    set_normalisation(m, d["train"]["X"])
    ys = rng.permutation(d["train"]["y"])
    fit(m, (d["train"]["X"], ys), budget, rng)
    p0, yi = float(ys.mean()), d["iid"]["y"]
    po = float(yi.mean())
    oracle = 1000.0 * float(np.mean(yi * (math.log(po) - math.log(p0))
                                    + (1 - yi) * (math.log(1 - po) - math.log(1 - p0))))
    sk = skill(m, d["iid"], p0)
    return bool(sk <= oracle + MIND_CARD["thresholds"]["shuffled_band"]), sk, oracle


def c5_mutants(d, rng, budget):
    global ACTIVE_MUTANT
    detected = {}
    for name in MUTANTS:
        ACTIVE_MUTANT = name
        try:
            ok1 = c1_gradcheck(d, rng, steps=10)[0]
            ok3 = c3_learning(d, rng, budget)[0]
        except FloatingPointError:
            ok1, ok3 = True, False
        finally:
            ACTIVE_MUTANT = None
        detected[name] = (not ok1) or (not ok3)
    return all(detected.values()), detected


def _random_partition(rng, broken=None):
    m = build_model(N_FEAT, 1, rng=rng, kind="partition")
    for k, v in m["params"].items():
        m["params"][k] = rng.normal(0, 2, v.shape)
    m["broken"] = broken
    return m


def _broken_lam(m, X):
    """Negative-control variants that each violate one declared property."""
    f = _partition_parts(m, X)
    if m["broken"] == "signed":            # raw (sign-free) contact weights
        return f["A"] + X[:, LAYOUT["contact"]] @ m["params"]["u_contact"] + f["F"]
    if m["broken"] == "interaction":       # field and contacts multiply
        return f["A"] * (1 + f["Cp"].sum(1)) + f["F"]
    if m["broken"] == "free_lags":         # one free weight per lag
        return f["A"] + f["Cp"].sum(1) + X[:, LAYOUT["obj"]] @ np.abs(m["params"]["W1"][0, :N_LAG])
    return f["lam"]


def c6_properties(rng, trials=40):
    """C6.1 monotone exposures, C6.2 exact additive attribution, C6.3 non-increasing object
    memory, C6.4 same-air estimator, C6.5 ta'wil nesting (definition check)."""
    res = {}
    viol, viol_neg = 0, 0
    for _ in range(trials):                                          # C6.1
        X = np.abs(rng.normal(0, 2, (64, N_FEAT)))
        for broken in (None, "signed"):
            m = _random_partition(rng, broken)
            base = _broken_lam(m, X)
            for col in range(9):
                Xp = X.copy()
                Xp[:, col] += rng.uniform(.1, 3)
                v = int(np.sum(_broken_lam(m, Xp) < base - 1e-12))
                viol, viol_neg = (viol + v, viol_neg) if broken is None else (viol, viol_neg + v)
    res["C6.1_monotone"] = viol == 0 and viol_neg > 0
    gap, gap_neg = 0.0, 0.0
    for _ in range(trials):                                          # C6.2
        X = np.abs(rng.normal(0, 2, (64, N_FEAT)))
        m = _random_partition(rng)
        _, parts = attribution(m, X)
        gap = max(gap, float(np.max(np.abs(parts.sum(1) + 1e-9 - _partition_parts(m, X)["lam"]))))
        m["broken"] = "interaction"
        gap_neg = max(gap_neg, float(np.max(np.abs(parts.sum(1) - _broken_lam(m, X)))))
    res["C6.2_additive"] = gap < 1e-10 and gap_neg > 1e-6
    inc, inc_neg = 0, 0
    for _ in range(trials):                                          # C6.3
        for broken in (None, "free_lags"):
            m = _random_partition(rng, broken)
            single = np.zeros((N_LAG, N_FEAT))
            single[np.arange(N_LAG), 3 + np.arange(N_LAG)] = 1.0
            contrib = _broken_lam(m, single) - _broken_lam(m, np.zeros((1, N_FEAT)))
            bad = int(np.sum(np.diff(contrib) > 1e-12))
            inc, inc_neg = (inc + bad, inc_neg) if broken is None else (inc, inc_neg + bad)
    res["C6.3_memory_decays"] = inc == 0 and inc_neg > 0
    k = rng.integers(0, 50, 200)
    n = k + rng.integers(0, 2000, 200)
    est = same_air_estimate(k, n)
    big = same_air_estimate(30000.0, 1e6)                            # C6.4 analytic
    with np.errstate(invalid="ignore"):
        naive = np.float64(0.0) / np.float64(0.0)          # negative control: unshrunk k/n at n = 0
    res["C6.4_same_air"] = (abs(same_air_estimate(0, 0) - PRIOR_MEAN) < 1e-15 and abs(big - .03) < 1e-4
                            and bool(np.all(same_air_estimate(k + 1, n + 1) > est))
                            and not np.isfinite(naive))
    m = _random_partition(rng)                                       # C6.5 definition check
    m["params"]["u_contact"][:] = -60
    m["params"]["g_obj"][:] = -60
    X = np.abs(rng.normal(0, 2, (64, N_FEAT)))
    f = _partition_parts(m, X)
    res["C6.5_tawil_nesting_definition_check"] = bool(np.max(np.abs(f["lam"] - 1e-9 - f["A"])) < 1e-12)
    return all(res.values()), res


def c7_split_integrity(d):
    seen = {}
    for name, part in d.items():
        h = {hashlib.md5(f"{e}|{a}|{b}".encode()).hexdigest() for e, (a, b) in zip(part["ep"], part["key"])}
        seen[name] = (set(part["ep"].tolist()), h)
    names = list(seen)
    ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ok &= not (seen[names[i]][0] & seen[names[j]][0]) and not (seen[names[i]][1] & seen[names[j]][1])
    return bool(ok)


# ----------------------------------------------------------------------------- hypotheses
def run_seed(seed, budget, scale):
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER_ID, seed, 1]))
    d = make_data(seed, scale)
    p0 = float(d["train"]["y"].mean())
    part = train_selected("partition", d, rng, budget)
    mlp = train_selected("mlp", d, rng, budget)
    place = train_selected("place", d, rng, budget)
    r = {"seed": seed, "n_params": {"partition": n_params(part), "mlp": n_params(mlp), "place": n_params(place)}}
    for nm, m in (("partition", part), ("mlp", mlp), ("place", place)):
        r[f"{nm}_iid"], r[f"{nm}_shift"] = skill(m, d["iid"], p0), skill(m, d["shift"], p0)
    for ko in KNOCKOUTS:
        r[f"ko_{ko}"] = skill(knockout(part, ko), d["shift"], p0)
    pb = float(d["blind_train"]["y"].mean())
    bpart = train_selected("partition", d, rng, budget, grid=(part["lr"],), split="blind_train")
    bmlp = train_selected("mlp", d, rng, budget, grid=(mlp["lr"],), split="blind_train")
    r["partition_blind"], r["mlp_blind"] = skill(bpart, d["blind_test"], pb), skill(bmlp, d["blind_test"], pb)
    sh = d["shift"]
    infected = sh["y"] > .5
    share, _ = attribution(part, sh["X"][infected])
    true_ch = sh["ch"][infected]
    keep = true_ch < 5
    r["attr_partition"] = float(share[keep, true_ch[keep]].mean())
    r["attr_air_only"] = float(np.mean(true_ch[keep] == 0))
    p = part["params"]
    r["recovered"] = {"beta_h": float(softplus(p["u_contact"][0])), "beta_m_recorded": float(softplus(p["u_contact"][1])),
                      "beta_s": float(softplus(p["u_contact"][2])), "gamma": float(softplus(p["g_obj"][0])),
                      "rho": float(sigmoid(p["r_obj"])[0])}
    return r, d


def hypotheses(results, rng):
    def arr(f):
        return [f(r) for r in results]
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    diffs = {
        "H-SIG": arr(lambda r: r["partition_shift"] - r["mlp_shift"]),
        "H-NEC": arr(lambda r: r["ko_same_air_control"] - r["ko_air_covariates"]),
        "H-BLIND": arr(lambda r: (r["partition_blind"] - r["mlp_blind"]) - (r["partition_iid"] - r["mlp_iid"])),
        "H-RIVAL": arr(lambda r: r["partition_shift"] - r["place_shift"]),
    }
    out = []
    for hid, dd in diffs.items():
        mean, lo, hi = paired_bootstrap(dd, rng)
        out.append({"id": hid, "metric": spec[hid]["metric"], "mean_diff": mean, "ci95": [lo, hi],
                    "mesi": spec[hid]["mesi"], "n_seeds": len(dd),
                    "verdict": verdict(mean, lo, hi, spec[hid]["direction"], spec[hid]["mesi"])})
    kos = []
    for ko in KNOCKOUTS:
        mean, lo, hi = paired_bootstrap(arr(lambda r: r[f"ko_{ko}"] - r["partition_shift"]), rng)
        kos.append({"module": ko, "signature": ko != "air_covariates", "metric_change": mean, "ci95": [lo, hi]})
    return out, kos


# ----------------------------------------------------------------------------- report
def report(ctx):
    L = [f"=== VERIFIED REPORT · chapter {CHAPTER_ID:04d} ===",
         f"file: {os.path.basename(__file__)}   card_revision: {MIND_CARD['card_revision']}",
         f"python {sys.version.split()[0]} · numpy {np.__version__} · seeds {ctx['seeds']} · "
         f"runtime {ctx['runtime']:.1f} s",
         f"params: {ctx['n_params']}",
         f"gradcheck: {ctx['gc_tensors']} tensor checks (3 models x init/trained), max rel error "
         f"{ctx['gc_worst']:.2e} -> {'pass' if ctx['gc_ok'] else 'FAIL'}", "correctness:"]
    for c in ctx["correctness"]:
        L.append(f"  {c['id']:<3} {c['name']:<28} {'pass' if c['passed'] else 'FAIL'}  {c['detail']}")
    L.append(f"mutants: {ctx['mut_detected']}/{ctx['mut_total']} detected (score {ctx['mut_score']:.2f})")
    L.append("hypotheses (skill = nats per 1000 person-days over the constant-rate forecast):")
    for h in ctx["hypotheses"]:
        L.append(f"  {h['id']:<8} mean {h['mean_diff']:+8.3f}  95% CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]"
                 f"  mesi {h['mesi']}  n={h['n_seeds']}  -> {h['verdict']}")
    if ctx["per_seed"]:
        L.append("per-model skill, mean over seeds (iid / shifted / blind):")
        ps = ctx["per_seed"]
        for nm in ("partition", "mlp", "place"):
            blind = f"{np.mean([r[nm + '_blind'] for r in ps]):8.3f}" if nm != "place" else "     n/a"
            L.append(f"  {nm:<10} {np.mean([r[nm + '_iid'] for r in ps]):8.3f} "
                     f"{np.mean([r[nm + '_shift'] for r in ps]):8.3f} {blind}")
        L.append("knockouts on shifted split (change in skill vs full partition):")
        for k in ctx["knockouts"]:
            L.append(f"  {k['module']:<18} sig={str(k['signature']):<5} {k['metric_change']:+8.3f}  "
                     f"[{k['ci95'][0]:+.3f}, {k['ci95'][1]:+.3f}]")
        L.append(f"attribution on shifted infections over {', '.join(CHANNELS[:5])}: share on the true channel "
                 f"{np.mean([r['attr_partition'] for r in ps]):.3f} vs air-only reading "
                 f"{np.mean([r['attr_air_only'] for r in ps]):.3f}")
        rec = {k: np.mean([r["recovered"][k] for r in ps]) for k in ps[0]["recovered"]}
        L.append("ta'wil recovery from a no-contagion start (true h .055, m .050 of which 60% recorded, s .30, "
                 "gamma .050, rho .72): " + ", ".join(f"{k} {v:.3f}" for k, v in rec.items()))
    L.append(f"task types: {MIND_CARD['task_types']}   exit code: {ctx['exit_code']}")
    L.append("=== END REPORT ===")
    return "\n".join(L)


def json_report(ctx):
    return {"schema_version": "1.0", "chapter": CHAPTER_ID, "file": os.path.basename(__file__),
            "card_revision": MIND_CARD["card_revision"],
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
            "seeds": ctx["seeds"], "runtime_s": round(ctx["runtime"], 2), "n_params": ctx["n_params"]["partition"],
            "gradcheck": {"tensors_checked": ctx["gc_tensors"], "tensors_total": ctx["gc_tensors"],
                          "max_rel_error": ctx["gc_worst"], "checked_at": ["init", "after_training_steps"],
                          "passed": ctx["gc_ok"]},
            "correctness": [{k: c[k] for k in ("id", "name", "passed", "detail")} for c in ctx["correctness"]],
            "mutants": {"detected": ctx["mut_detected"], "total": ctx["mut_total"], "score": ctx["mut_score"]},
            "hypotheses": ctx["hypotheses"], "knockouts": ctx["knockouts"],
            "task_types": MIND_CARD["task_types"], "exit_code": ctx["exit_code"]}


# ----------------------------------------------------------------------------- entry point
def real_data_bridge(path, rng, budget):
    """Optional: a local CSV with the 18 case-file columns plus the outcome, header row first."""
    if not path or not os.path.exists(path):
        return "real-data bridge: no CSV found at --data, skipped"
    A = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if A.shape[1] != N_FEAT + 1:
        return f"real-data bridge: expected {N_FEAT + 1} columns, found {A.shape[1]}, skipped"
    perm = rng.permutation(len(A))
    cut = np.split(perm, [int(.7 * len(A)), int(.8 * len(A))])
    d = {k: {"X": A[i, :-1], "y": A[i, -1]} for k, i in zip(("train", "val", "test"), cut)}
    p0 = float(d["train"]["y"].mean())
    res = {k: skill(train_selected(k, d, rng, budget), d["test"], p0) for k in ("partition", "mlp")}
    return "real-data bridge: test skill " + ", ".join(f"{k} {v:.2f}" for k, v in res.items())


def run_correctness(d0, rng, budget, quick, mutant):
    global ACTIVE_MUTANT
    gc_ok, gc_worst, gc_t, _ = c1_gradcheck(d0, rng)
    c2_ok, finite = c2_determinism(d0)
    c3_ok, drop, sk = c3_learning(d0, rng, budget)
    c4_ok, sk_shuf, sk_or = c4_shuffled(d0, rng, budget)
    ACTIVE_MUTANT = None
    c5_ok, det = c5_mutants(d0, rng, budget) if mutant is None else (True, {})
    ACTIVE_MUTANT = mutant
    c6_ok, props = c6_properties(rng)
    corr = [{"id": "C1", "name": "gradient_check", "passed": gc_ok, "detail": f"max rel {gc_worst:.2e}"},
            {"id": "C2", "name": "determinism_finiteness", "passed": c2_ok, "detail": f"finite={finite}"},
            {"id": "C3", "name": "learning", "passed": c3_ok, "detail": f"loss drop {drop:.3f}, iid skill {sk:.2f}"},
            {"id": "C4", "name": "shuffled_label_control", "passed": c4_ok,
             "detail": f"skill {sk_shuf:.2f} vs best constant {sk_or:.2f}"},
            {"id": "C5", "name": "mutant_detection", "passed": c5_ok, "detail": str(det)},
            {"id": "C6", "name": "property_tests", "passed": c6_ok, "detail": str(props)},
            {"id": "C7", "name": "split_integrity", "passed": c7_split_integrity(d0),
             "detail": "episodes and rows disjoint"}]
    return corr, (gc_ok, gc_worst, gc_t), det


def main(argv=None):
    global ACTIVE_MUTANT
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 2
    if a.card:
        print(json.dumps(MIND_CARD, indent=2))
        return 0
    if a.mutant is not None and a.mutant not in MUTANTS:
        print(f"unknown mutant; choose from {sorted(MUTANTS)}")
        return 2
    ACTIVE_MUTANT, t0 = a.mutant, time.time()
    budget, scale = (90, 0.5) if a.quick else (260, 1.0)
    seeds = [a.seed + i for i in range(1 if a.quick else max(1, a.seeds))]
    print(f"seeds: {seeds}   budget: {budget} updates   episode scale: {scale}")
    print("baselines: " + "; ".join(f"{k} = {v}" for k, v in BASELINES.items()))
    rng = np.random.default_rng(np.random.SeedSequence([CHAPTER_ID, a.seed, 99]))
    per_seed, kos = [], []
    try:
        corr, gc, det = run_correctness(make_data(seeds[0], scale), rng, budget, a.quick, a.mutant)
        if not a.quick and a.mutant is None:
            for s in seeds:
                per_seed.append(run_seed(s, budget, scale)[0])
                print(f"  seed {s} done ({time.time() - t0:.1f} s)")
            hyps, kos = hypotheses(per_seed, rng)
        else:
            hyps = [{"id": h["id"], "metric": h["metric"], "mean_diff": float("nan"), "ci95": [float("nan")] * 2,
                     "mesi": h["mesi"], "n_seeds": 0, "verdict": "not evaluated"} for h in MIND_CARD["hypotheses"]]
        print(real_data_bridge(a.data, rng, budget))
    except FloatingPointError as e:
        print(f"non-finite values: {e}")
        return 4
    runtime = time.time() - t0
    c8_ok = runtime <= (QUICK_BUDGET_S if a.quick else BUDGET_S)
    corr.append({"id": "C8", "name": "budget", "passed": c8_ok, "detail": f"{runtime:.1f} s"})
    code = 0 if all(c["passed"] for c in corr) else (3 if all(c["passed"] for c in corr[:7]) else 1)
    npc = per_seed[0]["n_params"] if per_seed else {k: n_params(build_model(kind=k)) for k in ("partition", "mlp", "place")}
    ctx = {"seeds": seeds, "runtime": runtime, "n_params": npc, "gc_ok": gc[0], "gc_worst": gc[1],
           "gc_tensors": gc[2], "correctness": corr, "mut_detected": sum(det.values()), "mut_total": len(det),
           "mut_score": (sum(det.values()) / len(det)) if det else 1.0, "hypotheses": hyps, "knockouts": kos,
           "per_seed": per_seed, "exit_code": code}
    print(report(ctx))
    if a.json:
        write_json(a.json, json_report(ctx))
    return code


if __name__ == "__main__":
    sys.exit(main())

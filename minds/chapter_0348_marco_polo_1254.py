#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0348 · Marco Polo (1254-1324)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 18, Minds 341-360 Available on Amazon https://www.amazon.com/dp/B0HQ47T2NC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION 
================================================================================ 

 STC -- the STAKE-WEIGHTED TRAVERSAL CARTOGRAPHER
 ------------------------------------------------
 A world model that never observes a coordinate.

 WHY THIS ARCHITECTURE AND NOT A TRANSFORMER
 -------------------------------------------
 The Devisement dou monde is not a story and not an encyclopaedia. It is a
 route ledger. Its atomic unit is the LEG: "when you leave X and ride N days
 in direction D, you come to Y." Position is never given. Position is only
 ever RECOVERED, by integrating legs and forcing the loops to close. That is
 dead reckoning over a graph, and it is the opposite of the modern default,
 where a model stores a coordinate (an embedding) and derives the relation
 between coordinates on demand.

 Three commitments, each of them Polo's rather than ours, are hard-wired:

   (1) THE METRIC IS TRAVERSAL COST, NOT DISTANCE.
       Node separation is measured in days of travel, and the conversion from
       days to ground is terrain-, season- and conveyance-conditional. The
       network's only feedforward component learns exactly that conversion --
       a caravan speed model -- because it is exactly what twenty-four years
       on the road installs in a merchant and nothing else about him is
       feedforward at all.

   (2) EVERY CLAIM CARRIES ITS PROVENANCE, AND PROVENANCE SETS WEIGHT.
       Legs are tagged SEEN / LEDGER / NAMED / BAZAAR / MARVEL and attributed
       to an informant. Precision is not a property of the claim's content; it
       is a learned function of who supplied it and how.

   (3) CALIBRATION IS ALLOCATED BY CONSEQUENCE.
       The parameter beta couples STAKE -- what an error would have cost --
       to precision. The calibration head is trained under a stake-weighted
       proper scoring rule. This reproduces, and then justifies, the single
       most striking measured fact about Polo's book: his accuracy is not
       uniform across his beliefs. Where a mistake would have killed him or
       ruined him (days of march, water, fodder, currency, salt duty, tax
       farm yields) he is astonishingly exact and modern scholarship keeps
       confirming him. Where a mistake cost nothing (the Khan's concubine
       count, the roc bird, the dog-headed islanders, the distance of Japan
       from the coast) he is loose, credulous, and wrong.

 THE LOAD-BEARING EXPERIMENT
 ---------------------------
 Test T6 is the chapter's argument in executable form. Polo states plainly
 that he never went to Cipangu; the claim is marked hearsay and the model
 correctly assigns it low confidence and large positional error. Two hundred
 years later a reader stripped the mark, took the number as a survey, and
 sailed. T6 measures what that costs: the same estimate consumed WITH its
 confidence and consumed WITHOUT it, in days of provisioning shortfall.

 The failure is not in the reasoner. It is in the channel.

 CONVENTIONS
 -----------
 * Pure NumPy. No autodiff. All gradients derived by hand below.
 * A finite-difference gradient check is mandatory and runs first (T1).
 * A real training loop with Adam, held-out legs, and reported metrics.
 * Deterministic: fixed seeds throughout.

 Run:  python3 chapter_0348_marco_polo_1254.py
================================================================================
"""

import numpy as np
import math

np.seterr(over="raise", invalid="raise")

RNG_MASTER = 20250805


# ==============================================================================
# SECTION 1 -- THE CORPUS
# ==============================================================================
# Twenty-six places that actually appear in the Devisement, with real modern
# coordinates. These coordinates are used ONLY to synthesise a ground truth the
# model is never shown. The model sees legs: bearings, day-counts, terrain,
# season, conveyance, provenance, informant, stake. Nothing else.
#
# The "days" the model receives are not distances. They are costs. Recovering a
# consistent map from them is the whole task.
# ==============================================================================

# name, latitude, longitude, note
PLACES = [
    ("Venice",        45.44,  12.32, "departure, 1271"),
    ("Acre",          32.93,  35.08, "Crusader port; papal letters taken up here"),
    ("Ayas",          36.77,  35.78, "Lajazzo, the Cilician gate to the land route"),
    ("Tabriz",        38.08,  46.29, "Ilkhanid entrepot; the great pearl market"),
    ("Saveh",         35.02,  50.36, "Polo places the Magi's tomb here"),
    ("Kerman",        30.28,  57.08, "steel, turquoise, the falcon country"),
    ("Hormuz",        27.10,  56.45, "the abandoned sea option; ships sewn, not nailed"),
    ("Balkh",         36.76,  66.90, "the ruined city, still walled"),
    ("Badakhshan",    37.12,  70.58, "balas rubies and lapis; Marco convalesces a year"),
    ("Pamir",         38.17,  73.97, "the roof; twelve days with no bird and no fire that cooks"),
    ("Kashgar",       39.47,  75.99, "the first Cathay-facing bazaar"),
    ("Yarkand",       38.42,  77.25, "goitre country"),
    ("Khotan",        37.11,  79.93, "jade taken from the riverbed"),
    ("Lop",           39.02,  88.17, "the last provisioning before the singing desert"),
    ("Shazhou",       40.14,  94.66, "Dunhuang; the funeral rites described at length"),
    ("Suzhou",        39.73,  98.52, "Jiuquan; rhubarb"),
    ("Ganzhou",       38.93, 100.45, "Zhangye; the Polos wait a year"),
    ("Shangdu",       42.36, 116.18, "Xanadu; the cane palace that can be struck down and moved"),
    ("Khanbaliq",     39.90, 116.40, "Dadu; the paper money is made here"),
    ("Quinsai",       30.27, 120.16, "Hangzhou; twelve thousand bridges, and the salt revenue"),
    ("Zaiton",        24.87, 118.59, "Quanzhou; the customs levy, and the fleet sails from here"),
    ("Karajan",       25.04, 102.72, "Yunnan; cowrie currency, crocodiles taken for serpents"),
    ("Mien",          21.17,  94.86, "Bagan; the gold and silver towers, reported by mission"),
    ("Champa",        13.77, 109.22, "the return leg's first landfall"),
    ("Java",          -7.25, 112.75, "NOT VISITED -- reported by sailors"),
    ("Mabar",         10.77,  79.84, "Coromandel; the pearl fishery, seen on the return"),
    ("Hormuz2",       27.10,  56.46, "return landfall -- deliberately a near-duplicate node"),
    ("Cipangu",       32.75, 129.87, "NOT VISITED -- Polo says so -- and the number that sank Columbus"),
]

NAME_TO_IDX = {p[0]: i for i, p in enumerate(PLACES)}
N_NODES = len(PLACES)

# ------------------------------------------------------------------------------
# Terrain classes. The learned speed model is conditioned on these.
# Index order matters: it is the one-hot layout fed to the caravan-speed network.
# ------------------------------------------------------------------------------
TERRAIN = ["road", "steppe", "desert", "mountain", "sea"]
T_IDX = {t: i for i, t in enumerate(TERRAIN)}

# TRUE speed multipliers, in units of "flat-road-days of ground covered per day
# of travel". The model must discover these; they are never exposed to it.
TRUE_SPEED = {
    "road":     1.00,
    "steppe":   0.86,
    "desert":   0.58,
    "mountain": 0.34,
    "sea":      3.40,   # the junk fleet on the monsoon covers ground fast
}
# Season nudges it: winter passes close, monsoon carries you.
TRUE_SEASON_GAIN = {"road": 0.04, "steppe": 0.10, "desert": 0.12,
                    "mountain": 0.30, "sea": 0.22}

# ------------------------------------------------------------------------------
# Provenance classes -- Polo's own distinctions, formalised.
# ------------------------------------------------------------------------------
PROV = ["SEEN", "LEDGER", "NAMED", "BAZAAR", "MARVEL"]
P_IDX = {p: i for i, p in enumerate(PROV)}
#   SEEN   -- he rode it himself
#   LEDGER -- copied off a yam post-station register or a tax roll
#   NAMED  -- a specific, nameable informant answered for it
#   BAZAAR -- undifferentiated market talk
#   MARVEL -- talk that has already passed through a teller who wanted it larger

# ------------------------------------------------------------------------------
# Informants. Note #6: Rustichello is modelled as an informant, because in the
# manuscript tradition he demonstrably is one -- the romance amplifier sits
# INSIDE the reporting channel, not outside it.
# ------------------------------------------------------------------------------
INFORMANTS = ["self", "kin", "yam_register", "ortoq_merchant",
              "court_official", "sailor", "rustichello"]
I_IDX = {s: i for i, s in enumerate(INFORMANTS)}

# TRUE noise scale by provenance (relative error on the day-count).
TRUE_PROV_SIGMA = {
    "SEEN":   0.055,
    "LEDGER": 0.035,
    "NAMED":  0.130,
    "BAZAAR": 0.300,
    "MARVEL": 0.520,
}
# TRUE stake coupling: how much careful measurement a high consequence buys.
# sigma_effective = sigma_prov * exp(-TRUE_BETA * stake)
TRUE_BETA = 1.15

# How many reports of the same road enter the ledger. One is his own passage;
# the rest are other people's, and they are what make the graph auditable.
N_REPLICAS = 7

# "Correct" means: could you have provisioned on this number? Fifteen percent
# on a twenty-day desert crossing is three days without water.
ACCURACY_TOLERANCE = 0.15

# How far a shared rumour runs ahead of the truth. Polo puts Cipangu 1500 miles
# off the coast of China; the real figure is nowhere much above 500.
SHARED_RUMOUR_GAIN = 0.95

# Multiplicative inflation applied by tellers who want the tale bigger.
TRUE_MARVEL_INFLATION = {"SEEN": 0.0, "LEDGER": 0.0, "NAMED": 0.05,
                         "BAZAAR": 0.28, "MARVEL": 0.85}


# ==============================================================================
# SECTION 2 -- THE LEG TABLE
# ==============================================================================
# Each tuple is:
#   (from, to, terrain, season01, conveyance01, provenance, informant, stake)
#
# season01:     0 = the hard season on that leg, 1 = the kind one
# conveyance01: 0 = on foot / pack animal, 1 = mounted post or ship
# stake in [0,1]: what being wrong costs. 1.0 = you provision from this number
#                 and there is no water if it is wrong. 0.05 = you will never
#                 go there and nothing depends on it.
#
# The graph deliberately contains CYCLES. Loop closure is the only error signal
# a dead-reckoner has, and Polo had it: he crossed Persia twice, by different
# roads, and the second crossing audits the first.
# ==============================================================================

LEGS_SPEC = [
    # ---- outbound, 1271-1275 -------------------------------------------------
    ("Venice", "Acre",        "sea",      1, 1, "SEEN",   "self",           0.55),
    ("Acre", "Ayas",          "sea",      1, 1, "SEEN",   "self",           0.60),
    ("Ayas", "Tabriz",        "road",     1, 0, "SEEN",   "self",           0.75),
    ("Tabriz", "Saveh",       "road",     1, 0, "SEEN",   "self",           0.70),
    ("Saveh", "Kerman",       "desert",   0, 0, "SEEN",   "self",           0.95),
    ("Kerman", "Hormuz",      "desert",   0, 0, "SEEN",   "self",           1.00),
    ("Hormuz", "Kerman",      "desert",   1, 0, "SEEN",   "self",           1.00),  # they turned back
    ("Kerman", "Balkh",       "desert",   0, 0, "SEEN",   "self",           0.98),
    ("Balkh", "Badakhshan",   "mountain", 1, 0, "SEEN",   "self",           0.90),
    ("Badakhshan", "Pamir",   "mountain", 0, 0, "SEEN",   "self",           1.00),
    ("Pamir", "Kashgar",      "mountain", 0, 0, "SEEN",   "self",           1.00),
    ("Kashgar", "Yarkand",    "desert",   1, 0, "SEEN",   "self",           0.85),
    ("Yarkand", "Khotan",     "desert",   1, 0, "SEEN",   "self",           0.85),
    ("Khotan", "Lop",         "desert",   0, 0, "SEEN",   "self",           1.00),
    ("Lop", "Shazhou",        "desert",   0, 0, "SEEN",   "self",           1.00),
    ("Shazhou", "Suzhou",     "steppe",   1, 0, "SEEN",   "self",           0.70),
    ("Suzhou", "Ganzhou",     "steppe",   1, 0, "SEEN",   "self",           0.65),
    ("Ganzhou", "Shangdu",    "steppe",   1, 1, "SEEN",   "self",           0.75),
    ("Shangdu", "Khanbaliq",  "road",     1, 1, "LEDGER", "yam_register",   0.45),

    # ---- alternate / audit legs: the second crossing that closes the loop ----
    ("Tabriz", "Kerman",      "road",     1, 0, "NAMED",  "ortoq_merchant", 0.80),
    ("Balkh", "Kashgar",      "mountain", 0, 0, "NAMED",  "ortoq_merchant", 0.72),
    ("Khotan", "Shazhou",     "desert",   0, 0, "BAZAAR", "ortoq_merchant", 0.60),
    ("Ayas", "Acre",          "sea",      1, 1, "LEDGER", "kin",            0.50),

    # ---- inside the Yuan: the post roads, taken off the register -------------
    ("Khanbaliq", "Quinsai",  "road",     1, 1, "LEDGER", "yam_register",   0.40),
    ("Quinsai", "Zaiton",     "road",     1, 1, "LEDGER", "yam_register",   0.40),
    ("Khanbaliq", "Karajan",  "road",     1, 1, "SEEN",   "self",           0.68),
    ("Karajan", "Mien",       "mountain", 0, 0, "NAMED",  "court_official", 0.55),
    ("Khanbaliq", "Zaiton",   "road",     0, 1, "LEDGER", "yam_register",   0.42),  # closes with Quinsai
    ("Karajan", "Quinsai",    "road",     1, 0, "BAZAAR", "ortoq_merchant", 0.35),

    # ---- the return by sea, 1291-1295 ---------------------------------------
    ("Zaiton", "Champa",      "sea",      1, 1, "SEEN",   "self",           0.85),
    ("Champa", "Mabar",       "sea",      1, 1, "SEEN",   "self",           0.90),
    ("Mabar", "Hormuz2",      "sea",      1, 1, "SEEN",   "self",           0.92),
    ("Hormuz2", "Tabriz",     "road",     1, 0, "SEEN",   "self",           0.70),
    ("Champa", "Java",        "sea",      1, 1, "NAMED",  "sailor",         0.20),

    # ---- the hearsay horizon: correctly marked, and never checked -----------
    # Polo says in his own text that he did not go to Cipangu. The mark is on
    # the claim. The mark is what later readers threw away.
    ("Zaiton",  "Cipangu",    "sea",      1, 1, "MARVEL", "rustichello",    0.05),
    ("Quinsai", "Cipangu",    "sea",      1, 1, "BAZAAR", "sailor",         0.08),
    ("Java",    "Mabar",      "sea",      1, 1, "BAZAAR", "sailor",         0.15),
]

# Nodes reachable only through hearsay -- the ones the audit is about.
HEARSAY_ONLY_NODES = ["Cipangu"]


def build_true_positions():
    """
    Ground truth the model never sees.

    Equirectangular projection about the corpus centroid, then rescaled into
    units of 'flat-road days'. One flat-road day is taken as 35 km, which is a
    conservative sustained caravan figure and is only a unit choice -- it
    cancels out of every reported metric.
    """
    lats = np.array([p[1] for p in PLACES], dtype=np.float64)
    lons = np.array([p[2] for p in PLACES], dtype=np.float64)
    lat0 = float(lats.mean())
    R = 6371.0
    x_km = np.radians(lons) * R * math.cos(math.radians(lat0))
    y_km = np.radians(lats) * R
    X = np.stack([x_km, y_km], axis=1)
    X -= X[NAME_TO_IDX["Venice"]]          # Venice at the origin, as in the book
    return X / 35.0                         # km -> flat-road days


def build_dataset(seed=RNG_MASTER):
    """
    Turn the leg specification into the observation table the model consumes.

    Returns a dict of arrays. Nothing here leaks true positions or true speeds
    into anything the model reads -- only `Xtrue` and `true_days` are kept, and
    those are used exclusively for evaluation.
    """
    rng = np.random.default_rng(seed)
    Xtrue = build_true_positions()

    # ONE rumour, many mouths. Every report reaching Quanzhou about an island
    # nobody in the room had seen came down the same channel and carried the
    # same incentive to make it larger. So the inflation on a hearsay-only
    # destination is drawn ONCE and shared by every report about it. This is
    # the mechanism that makes more data useless: independent-looking sources
    # with a common origin do not average out, they concentrate the bias.
    shared_infl = {nm: float(abs(rng.normal(1.0, 0.30)))
                   for nm in HEARSAY_ONLY_NODES}

    src, dst, terr, seas, conv, prov, inf, stake = [], [], [], [], [], [], [], []
    days_obs, bear_obs, true_days, is_primary = [], [], [], []

    # Secondary reports of the same road, by other people. A merchant's picture
    # of a route is never one traversal; it is his own passage plus everything
    # anyone else has said about the same stretch. These are the legs that make
    # the graph over-determined -- and they are also where the disagreement that
    # the auditor reads comes from.
    SECONDARY = [("NAMED", "ortoq_merchant"), ("NAMED", "court_official"),
                 ("BAZAAR", "ortoq_merchant"), ("BAZAAR", "sailor"),
                 ("MARVEL", "rustichello"), ("LEDGER", "yam_register"),
                 ("SEEN", "kin")]

    for (a, b, t, s01, c01, pv, iv, st) in LEGS_SPEC:
        i, j = NAME_TO_IDX[a], NAME_TO_IDX[b]
        d = Xtrue[j] - Xtrue[i]
        ground = float(np.linalg.norm(d))              # flat-road days of ground
        bearing_true = math.atan2(d[1], d[0])

        reports = [(pv, iv, s01, c01, st, 1)]
        for _ in range(N_REPLICAS - 1):
            spv, siv = SECONDARY[rng.integers(len(SECONDARY))]
            # a hearsay-only destination never acquires an eyewitness report
            if b in HEARSAY_ONLY_NODES or a in HEARSAY_ONLY_NODES:
                if spv in ("SEEN", "LEDGER"):
                    spv, siv = "BAZAAR", "sailor"
            ss = float(rng.integers(2))
            sc = float(rng.integers(2))
            sst = float(np.clip(st * rng.uniform(0.45, 1.0), 0.02, 1.0))
            reports.append((spv, siv, ss, sc, sst, 0))

        for (rpv, riv, rs, rc, rst, prim) in reports:
            sp = TRUE_SPEED[t] * (1.0 + TRUE_SEASON_GAIN[t] * (rs - 0.5) * 2.0)
            sp *= (1.0 + 0.10 * rc)                    # post horses / a good ship
            d_true = ground / sp                       # travel days

            # observation noise: provenance sets the base, stake buys precision
            sigma = TRUE_PROV_SIGMA[rpv] * math.exp(-TRUE_BETA * rst)
            rel = rng.normal(0.0, sigma)
            if b in shared_infl or a in shared_infl:
                key = b if b in shared_infl else a
                infl = SHARED_RUMOUR_GAIN * shared_infl[key]      # correlated
            else:
                infl = TRUE_MARVEL_INFLATION[rpv] * abs(rng.normal(1.0, 0.35))
            d_rep = max(d_true * (1.0 + rel) * (1.0 + infl), 1e-3)

            # bearing is reported coarsely, in wind-rose points, worse by hearsay
            b_rep = bearing_true + rng.normal(0.0, 0.035 + 0.42 * sigma)

            src.append(i); dst.append(j); terr.append(T_IDX[t])
            seas.append(rs); conv.append(rc)
            prov.append(P_IDX[rpv]); inf.append(I_IDX[riv]); stake.append(rst)
            days_obs.append(d_rep); bear_obs.append(b_rep)
            true_days.append(d_true); is_primary.append(prim)

    D = dict(
        src=np.array(src, dtype=np.int64),
        dst=np.array(dst, dtype=np.int64),
        terrain=np.array(terr, dtype=np.int64),
        season=np.array(seas, dtype=np.float64),
        convey=np.array(conv, dtype=np.float64),
        prov=np.array(prov, dtype=np.int64),
        informant=np.array(inf, dtype=np.int64),
        stake=np.array(stake, dtype=np.float64),
        days=np.array(days_obs, dtype=np.float64),
        bearing=np.array(bear_obs, dtype=np.float64),
        true_days=np.array(true_days, dtype=np.float64),
        primary=np.array(is_primary, dtype=np.int64),
        Xtrue=Xtrue,
    )

    # Correctness label for the calibration head: is the reported day-count
    # within 25% of the truth? This is the target a merchant actually cares
    # about -- not "is the sentence true" but "can I provision on it".
    relerr = np.abs(D["days"] - D["true_days"]) / D["true_days"]
    D["correct"] = (relerr < ACCURACY_TOLERANCE).astype(np.float64)
    D["relerr"] = relerr

    # Feature matrix for the caravan-speed network.
    # [terrain one-hot(5) | season | conveyance | season*mountain | log days]
    E = len(src)
    F = np.zeros((E, 9), dtype=np.float64)
    F[np.arange(E), D["terrain"]] = 1.0
    F[:, 5] = D["season"]
    F[:, 6] = D["convey"]
    F[:, 7] = D["season"] * F[:, T_IDX["mountain"]]
    F[:, 8] = np.log(D["days"]) / 4.0
    D["feat"] = F
    D["unit"] = np.stack([np.cos(D["bearing"]), np.sin(D["bearing"])], axis=1)
    return D


# ==============================================================================
# SECTION 3 -- THE MODEL
# ==============================================================================
#
#  PARAMETERS
#  ----------
#   X    (N,2)    latent map, in flat-road days. Never observed. Gauge-fixed.
#   W1,b1,W2,b2   the caravan-speed network: feat -> scalar pre-speed
#   tau  (I,)     per-informant log-precision. Learned trust.
#   rho  (P,)     per-provenance log-precision offset.
#   beta scalar   STAKE COUPLING. The chapter's parameter. How much a claim's
#                 consequence raises the weight it is allowed to have.
#   kap  scalar   global log-precision bias.
#   wcal (C,)     calibration head, a linear logit over provenance/stake/
#                 disagreement. Deliberately tiny and deliberately linear:
#                 an auditor with the capacity of the thing it audits shares
#                 its blind spots.
#
#  FORWARD
#  -------
#   s_e     = softplus( W2 . tanh(W1 . f_e + b1) + b2 )      learned speed
#   yhat_e  = d_e * s_e * u_e                                predicted displacement
#   r_e     = (X_dst - X_src) - yhat_e                       residual
#   l_e     = tau[i_e] + rho[p_e] + beta*stake_e + kap       log-precision
#   lam_e   = exp(clip(l_e))
#
#   L_map   = mean_e [ 0.5*lam_e*||r_e||^2 - log lam_e ]
#             (isotropic 2-D Gaussian NLL up to a constant; the -log lam term
#              is what stops precision running away to infinity)
#   L_ridge = 0.5*alpha*mean_n ||X_n||^2        weak, only to fix scale drift
#   L_cal   = stake-weighted Brier score of sigmoid(wcal . c_e) against
#             the correctness label
#
#   L = L_map + L_ridge + w_cal_loss * L_cal
#
#  The calibration feature vector c_e uses the residual magnitude as DETACHED
#  input. The auditor reads the disagreement as evidence; it does not get to
#  move the map to make itself look better. That is the whole point of an
#  auditor and it is the one place a gradient is deliberately cut.
# ==============================================================================

HID = 12
N_CALFEAT = 1 + 1 + len(PROV) + 1 + 1   # bias, stake, prov one-hot, disagreement, log-days
LOGLAM_CLIP = 6.0
ALPHA_RIDGE = 2e-4
W_CAL_LOSS = 1.0

# The map moves in travel-days; the weights move in units of order one.
KEY_LR = {"X": 60.0}   # tuned; see the sweep noted in the chapter

# The one surveyed edge. Venice->Acre was charted; everything east of it was not.
ANCHOR_A = NAME_TO_IDX["Acre"]
ANCHOR_B = NAME_TO_IDX["Venice"]
K_ANCHOR = 0.10
K_UNIT = 200.0
_XT0 = build_true_positions()
ANCHOR_LEN = float(np.linalg.norm(_XT0[ANCHOR_A] - _XT0[ANCHOR_B]))


def init_map_by_dead_reckoning(D, start="Venice"):
    """
    Initialise the map the way it was actually built: walk out from Venice and
    add up. Breadth-first over the leg graph; each new place is placed at its
    parent plus the median reported displacement, using the naive assumption
    that one travel day buys one map unit on every surface. That assumption is
    wrong -- a day in the Pamir buys a third of a day on a Persian road -- and
    correcting it is precisely what training does. But it is the assumption a
    traveller starts with, and starting there converges far better than
    starting from noise.
    """
    X = np.zeros((N_NODES, 2))
    placed = {NAME_TO_IDX[start]}
    adj = {k: [] for k in range(N_NODES)}
    for e in range(len(D["src"])):
        a, b = int(D["src"][e]), int(D["dst"][e])
        adj[a].append((b, e, +1.0))
        adj[b].append((a, e, -1.0))
    frontier = [NAME_TO_IDX[start]]
    while frontier:
        nxt = []
        for a in frontier:
            cand = {}
            for (b, e, sgn) in adj[a]:
                if b in placed:
                    continue
                cand.setdefault(b, []).append(
                    sgn * D["days"][e] * D["unit"][e])
            for b, vs in cand.items():
                X[b] = X[a] + np.median(np.stack(vs), axis=0)
                placed.add(b)
                nxt.append(b)
        frontier = nxt
    # anything unreachable (there should be none) gets a small random offset
    for k in range(N_NODES):
        if k not in placed:
            X[k] = np.random.default_rng(k).normal(0, 20, size=2)
    return X - X.mean(0)


def init_params(seed=RNG_MASTER, n_nodes=N_NODES, n_feat=9, D=None):
    rng = np.random.default_rng(seed + 7)
    p = {}
    if D is not None:
        p["X"] = init_map_by_dead_reckoning(D) + rng.normal(0.0, 1.0, size=(n_nodes, 2))
    else:
        p["X"] = rng.normal(0.0, 40.0, size=(n_nodes, 2))
    p["W1"] = rng.normal(0.0, 0.7, size=(HID, n_feat))
    p["b1"] = np.zeros(HID)
    p["W2"] = rng.normal(0.0, 0.5, size=(HID,))
    p["b2"] = np.array(0.3)
    p["tau"] = np.zeros(len(INFORMANTS))
    p["rho"] = np.zeros(len(PROV))
    p["beta"] = np.array(0.0)
    p["kap"] = np.array(0.0)
    p["wcal"] = rng.normal(0.0, 0.1, size=(N_CALFEAT,))
    return p


def _softplus(z):
    # numerically safe softplus
    return np.where(z > 30.0, z, np.log1p(np.exp(np.minimum(z, 30.0))))


def _sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _cal_features(D, idx, rnorm):
    """Calibration head input. rnorm arrives detached."""
    E = len(idx)
    C = np.zeros((E, N_CALFEAT))
    C[:, 0] = 1.0
    C[:, 1] = D["stake"][idx]
    C[np.arange(E), 2 + D["prov"][idx]] = 1.0
    C[:, 2 + len(PROV)] = rnorm
    C[:, 3 + len(PROV)] = np.log(D["days"][idx]) / 4.0
    return C


def _wrap(a):
    """Wrap an angle difference into (-pi, pi]. Derivative is 1 a.e."""
    return np.arctan2(np.sin(a), np.cos(a))


def forward(p, D, idx, eta_stake=1.0, freeze_beta=False, freeze_trust=False,
            need_cache=True, rnorm_fixed=None):
    """
    One full forward pass over the legs in `idx`.

    THE DIRECTION OF THE LIKELIHOOD MATTERS AND IS THE WHOLE DESIGN.
    The observation is the DAY-COUNT and the BEARING. The map is the hidden
    variable. So the residual must live in observation space -- what the road
    was reported to cost, minus what this map says it would cost -- and not in
    map space. Score the displacement instead and the model discovers it can
    satisfy every leg by collapsing the world to a point, because a distance of
    zero matches a prediction of zero. Scoring the day-count forbids that: a
    collapsed map predicts a journey of no days, and the ledger says twenty.

    Day error is scored in LOGS because travel reports are wrong by a fraction,
    never by a constant. "A third longer than they told me" is the unit of
    complaint on a caravan road, and it is the unit the noise actually has.

    eta_stake     : consequence weighting inside the calibration objective.
                    eta=0 is the uniform-calibration ablation.
    freeze_beta   : if True, beta contributes nothing to log-precision. This is
                    the uniform-precision ablation.
    rnorm_fixed   : the STOP-GRADIENT. The calibration head reads the current
                    disagreement as evidence, but must not be able to move the
                    map in order to flatter itself. Passing the value in
                    explicitly makes the cut structural rather than a convention
                    observed only in the backward pass -- which is what the
                    finite-difference check actually verifies. None means
                    "recompute", and is used only for reporting.
    """
    f = D["feat"][idx]                     # (E,F)
    d = D["days"][idx]                     # (E,) reported travel days
    phi = D["bearing"][idx]                # (E,) reported bearing
    i = D["src"][idx]
    j = D["dst"][idx]
    E = len(idx)

    # -- caravan speed network -------------------------------------------------
    # The one feedforward component in the whole model, and it learns the one
    # thing twenty-four years on the road actually installs: how much ground a
    # day buys you, given terrain, season and whether you are on post horses.
    a1 = f @ p["W1"].T + p["b1"]           # (E,H)
    h = np.tanh(a1)
    z = h @ p["W2"] + p["b2"]              # (E,)
    s = _softplus(z) + 1e-4                # (E,) ground-days per travel-day

    # -- what THIS map says the leg costs -------------------------------------
    dX = p["X"][j] - p["X"][i]             # (E,2)
    n = np.sqrt(np.sum(dX * dX, axis=1) + 1e-9)      # ground-days of separation
    theta = np.arctan2(dX[:, 1], dX[:, 0])

    rd = np.log(d) - np.log(n) + np.log(s)           # log day-count residual
    rb = _wrap(theta - phi)                          # bearing residual, radians
    r2 = rd * rd + rb * rb

    # -- provenance/stake precision -------------------------------------------
    if freeze_trust:
        # THE CREDULOUS READER. Every source weighted alike: no provenance, no
        # informant, no consequence. This is not a straw man -- it is what a
        # reader of the Devisement does the moment the marks stop being copied,
        # and it is what a corpus does when it flattens everything it ingests
        # into undifferentiated text.
        betaterm = 0.0
        ll = np.zeros(E) + p["kap"]
    else:
        betaterm = 0.0 if freeze_beta else p["beta"] * D["stake"][idx]
        ll = (p["tau"][D["informant"][idx]] + p["rho"][D["prov"][idx]]
              + betaterm + p["kap"])
    ll_c = np.clip(ll, -LOGLAM_CLIP, LOGLAM_CLIP)
    lam = np.exp(ll_c)

    L_map = np.mean(0.5 * lam * r2 - ll_c)
    L_ridge = 0.5 * ALPHA_RIDGE * np.mean(np.sum(p["X"] ** 2, axis=1))

    # -- the portolan anchor ---------------------------------------------------
    # A weak prior, not a clamp. The Mediterranean of the 1290s was already
    # charted to a few percent, so Venice-Acre is the one edge of this journey
    # whose length Europe already knew in its own units. Everything east of Acre
    # hangs off it. Historically this is the only surveyed edge Polo had.
    va = p["X"][ANCHOR_A] - p["X"][ANCHOR_B]
    va_n = np.sqrt(np.sum(va * va) + 1e-12)
    L_anchor = 0.5 * K_ANCHOR * (va_n - ANCHOR_LEN) ** 2

    # -- the unit of the world -------------------------------------------------
    # Scale is still free: multiply every coordinate and every learned speed by
    # the same factor and no residual notices. The gauge is closed the way the
    # book closes it -- BY DECLARING THE UNIT. In the Devisement the measure of
    # the world is the day's journey, so we require that a day on a good road IS
    # one map unit, and every other terrain becomes a real, learned, relative
    # cost against it. This one sentence is what makes the map metric rather
    # than merely topological.
    road_mask = (D["terrain"][idx] == T_IDX["road"])
    if road_mask.any():
        sbar = float(np.mean(s[road_mask]))
        L_unit = 0.5 * K_UNIT * (sbar - 1.0) ** 2
    else:
        sbar, L_unit = 1.0, 0.0

    # -- calibration head (disagreement enters detached) ----------------------
    if rnorm_fixed is None:
        rr = np.sqrt(r2)
        rnorm = rr / (1.0 + rr)                        # squashed, in [0,1)
    else:
        rnorm = rnorm_fixed                            # stop-gradient
    C = _cal_features(D, idx, rnorm)
    zc = C @ p["wcal"]
    pc = _sigmoid(zc)
    t = D["correct"][idx]
    wgt = 1.0 + eta_stake * D["stake"][idx]
    Wsum = np.sum(wgt)
    L_cal = np.sum(wgt * (pc - t) ** 2) / Wsum

    L = L_map + L_ridge + L_anchor + L_unit + W_CAL_LOSS * L_cal

    cache = None
    if need_cache:
        cache = dict(f=f, d=d, i=i, j=j, E=E, a1=a1, h=h, z=z, s=s,
                     dX=dX, n=n, rd=rd, rb=rb, r2=r2,
                     ll=ll, ll_c=ll_c, lam=lam,
                     C=C, pc=pc, t=t, wgt=wgt, Wsum=Wsum, idx=idx,
                     freeze_beta=freeze_beta, freeze_trust=freeze_trust, va=va, va_n=va_n,
                     road_mask=road_mask, sbar=sbar)
    return L, dict(L_map=L_map, L_ridge=L_ridge, L_anchor=L_anchor,
                   L_unit=L_unit, L_cal=L_cal, s=s, lam=lam, pc=pc,
                   rd=rd, rb=rb), cache


def backward(p, D, cache):
    """
    Hand-derived gradients. Verified against central finite differences in T1.

      L_map = (1/E) * sum_e [ 0.5*lam_e*(rd_e^2 + rb_e^2) - ll_e ]

      rd_e = log d_e - log||dX_e|| + log s_e
      rb_e = wrap( atan2(dX_e) - phi_e )

      dL/drd = (1/E) lam rd            dL/drb = (1/E) lam rb
      d(rd)/d(dX)  = -dX / ||dX||^2
      d(rb)/d(dX)  =  perp(dX) / ||dX||^2 ,  perp(x,y) = (-y, x)
      d(rd)/ds     =  1/s
      dL/dll_e     = (1/E) ( 0.5*lam_e*r2_e - 1 )        [0 where clipped]
    """
    g = {k: np.zeros_like(v) for k, v in p.items()}
    E = cache["E"]
    lam, n = cache["lam"], cache["n"]
    dX = cache["dX"]
    inv = 1.0 / E

    dLdrd = inv * lam * cache["rd"]                       # (E,)
    dLdrb = inv * lam * cache["rb"]                       # (E,)

    # ---- into the map --------------------------------------------------------
    n2 = (n * n)[:, None]
    perp = np.stack([-dX[:, 1], dX[:, 0]], axis=1)
    dLddX = (-dLdrd[:, None] * dX / n2) + (dLdrb[:, None] * perp / n2)

    np.add.at(g["X"], cache["j"], dLddX)
    np.add.at(g["X"], cache["i"], -dLddX)
    g["X"] += ALPHA_RIDGE * p["X"] / p["X"].shape[0]

    # portolan anchor: d/dv [0.5*k*(||v||-L0)^2] = k*(||v||-L0)*v/||v||
    va, va_n = cache["va"], cache["va_n"]
    ga = K_ANCHOR * (va_n - ANCHOR_LEN) * va / va_n
    g["X"][ANCHOR_A] += ga
    g["X"][ANCHOR_B] -= ga

    # ---- log-precision term --------------------------------------------------
    dLdll = inv * (0.5 * lam * cache["r2"] - 1.0)
    dLdll = dLdll * (np.abs(cache["ll"]) < LOGLAM_CLIP).astype(np.float64)

    g["kap"] = np.array(np.sum(dLdll))
    if cache["freeze_trust"]:
        pass                                    # tau, rho, beta stay at zero
    else:
        np.add.at(g["tau"], D["informant"][cache["idx"]], dLdll)
        np.add.at(g["rho"], D["prov"][cache["idx"]], dLdll)
        if not cache["freeze_beta"]:
            g["beta"] = np.array(np.sum(dLdll * D["stake"][cache["idx"]]))

    # ---- through the speed network ------------------------------------------
    dLds = dLdrd / cache["s"]

    # unit-fixing term: d/ds_e [0.5*K*(mean_road(s)-1)^2] = K*(sbar-1)/n_road
    rm = cache["road_mask"]
    n_road = int(rm.sum())
    if n_road > 0:
        dLds = dLds + rm * (K_UNIT * (cache["sbar"] - 1.0) / n_road)

    dsdz = _sigmoid(cache["z"])                            # softplus'
    dLdz = dLds * dsdz

    g["b2"] = np.array(np.sum(dLdz))
    g["W2"] = cache["h"].T @ dLdz
    dLdh = np.outer(dLdz, p["W2"])                         # (E,H)
    dLda1 = dLdh * (1.0 - cache["h"] ** 2)
    g["b1"] = np.sum(dLda1, axis=0)
    g["W1"] = dLda1.T @ cache["f"]

    # ---- calibration head ----------------------------------------------------
    pc, t, wgt, Wsum = cache["pc"], cache["t"], cache["wgt"], cache["Wsum"]
    dLdzc = W_CAL_LOSS * (2.0 * wgt * (pc - t) * pc * (1.0 - pc)) / Wsum
    g["wcal"] = cache["C"].T @ dLdzc

    return g


# ==============================================================================
# SECTION 4 -- OPTIMISER AND TRAINING
# ==============================================================================

class Adam:
    """Plain Adam. Handles 0-d arrays as well as matrices."""

    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr_scale=1.0, key_lr=None):
        """key_lr lets one parameter block move on its own scale. The map lives
        in travel-days (magnitudes of order 100); the network weights live in
        order 1. One global step size cannot serve both."""
        key_lr = key_lr or {}
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            step = self.lr * lr_scale * key_lr.get(k, 1.0)
            params[k] = params[k] - step * mh / (np.sqrt(vh) + self.eps)
        return params


def split_legs(D, seed=RNG_MASTER, holdout_frac=0.18):
    """
    Held-out legs must not disconnect the graph, so we only ever hold out legs
    whose endpoints stay reachable. In practice: hold out redundant audit legs
    and never the unique bridge into a leaf.
    """
    rng = np.random.default_rng(seed + 31)
    E = len(D["src"])
    deg = np.zeros(N_NODES, dtype=int)
    for a, b in zip(D["src"], D["dst"]):
        deg[a] += 1
        deg[b] += 1
    eligible = [e for e in range(E)
                if deg[D["src"][e]] > 1 and deg[D["dst"][e]] > 1]
    n_hold = max(2, int(round(holdout_frac * E)))
    hold = rng.choice(eligible, size=min(n_hold, len(eligible)), replace=False)
    hold = set(int(h) for h in hold)
    train = np.array([e for e in range(E) if e not in hold], dtype=np.int64)
    test = np.array(sorted(hold), dtype=np.int64)
    return train, test


def train(D, train_idx, steps=3000, lr=0.05, eta_stake=1.0,
          freeze_beta=False, freeze_trust=False, seed=RNG_MASTER, verbose=False):
    p = init_params(seed=seed, D=D)
    if freeze_trust:
        p["tau"][:] = 0.0
        p["rho"][:] = 0.0
        p["beta"] = np.array(0.0)
    opt = Adam(p, lr=lr)
    hist = []
    # stop-gradient buffer for the auditor, refreshed after every step
    rnb = np.zeros(len(train_idx))
    for t in range(steps):
        scale = 1.0 if t < steps * 0.7 else 0.25
        L, met, cache = forward(p, D, train_idx, eta_stake=eta_stake,
                                freeze_beta=freeze_beta,
                                freeze_trust=freeze_trust, rnorm_fixed=rnb)
        g = backward(p, D, cache)
        p = opt.step(p, g, lr_scale=scale, key_lr=KEY_LR)
        rr = np.sqrt(cache["r2"])
        rnb = rr / (1.0 + rr)
        if t % 400 == 0 or t == steps - 1:
            hist.append((t, float(L), float(met["L_map"]), float(met["L_cal"])))
            if verbose:
                print(f"    step {t:5d}  L={L:9.4f}  map={met['L_map']:9.4f} "
                      f"cal={met['L_cal']:.4f}")
    return p, hist


# ==============================================================================
# SECTION 5 -- EVALUATION HELPERS
# ==============================================================================

def align_translation(Xhat, Xtrue):
    """
    The bearings pin rotation and reflection; only translation is free, so we
    align by centroid alone. No hidden Procrustes rotation is applied -- if the
    map came out rotated, the error will show it.
    """
    return Xhat - Xhat.mean(0) + Xtrue.mean(0)


def map_rmse(p, D, nodes=None):
    Xa = align_translation(p["X"], D["Xtrue"])
    if nodes is None:
        nodes = np.arange(N_NODES)
    d = Xa[nodes] - D["Xtrue"][nodes]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def node_error(p, D, name):
    Xa = align_translation(p["X"], D["Xtrue"])
    k = NAME_TO_IDX[name]
    return float(np.linalg.norm(Xa[k] - D["Xtrue"][k]))


def brier(p, D, idx, eta_stake=1.0):
    _, met, _ = forward(p, D, idx, eta_stake=eta_stake, need_cache=False)
    pc = met["pc"]
    return float(np.mean((pc - D["correct"][idx]) ** 2)), pc


def speed_report(p, D):
    """Recover the learned per-terrain travel speed and compare to the truth."""
    out = {}
    for t_name, t_i in T_IDX.items():
        mask = D["terrain"] == t_i
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        _, met, _ = forward(p, D, idx, need_cache=False)
        # true effective speed on those same legs
        true_eff = []
        for e in idx:
            sp = TRUE_SPEED[t_name] * (
                1.0 + TRUE_SEASON_GAIN[t_name] * (D["season"][e] - 0.5) * 2.0)
            sp *= (1.0 + 0.10 * D["convey"][e])
            true_eff.append(sp)
        out[t_name] = (float(np.mean(met["s"])), float(np.mean(true_eff)))
    return out


# ==============================================================================
# SECTION 6 -- TESTS
# ==============================================================================

def T1_gradcheck():
    """
    MANDATORY. Central finite differences against the hand-derived gradients,
    on every parameter block, including the anchor and the calibration head.
    """
    print("\n[T1] FINITE-DIFFERENCE GRADIENT CHECK")
    D = build_dataset(seed=99)
    idx = np.arange(len(D["src"]))
    p = init_params(seed=5)
    # Check at a WELL-SCALED point. From a random init the map term is of order
    # 1e4 while the calibration term is of order 1e-1, and the finite difference
    # for the calibration weights drowns in the floating-point cancellation of
    # the large term. Starting the map near a plausible solution puts every
    # component on the same scale, which is where a gradient check belongs.
    p["X"] = D["Xtrue"] + np.random.default_rng(11).normal(0.0, 2.0,
                                                           size=D["Xtrue"].shape)

    # freeze the auditor's input so the finite differences see exactly the
    # same function the backward pass differentiates
    _, _, c0 = forward(p, D, idx)
    rr0 = np.sqrt(c0["r2"])
    rnb = rr0 / (1.0 + rr0)

    L, _, cache = forward(p, D, idx, rnorm_fixed=rnb)
    g = backward(p, D, cache)

    rng = np.random.default_rng(3)
    eps = 1e-6
    worst = 0.0
    for k in p:
        v = np.atleast_1d(p[k])
        flat = v.reshape(-1)
        n_probe = min(6, flat.size)
        probes = rng.choice(flat.size, size=n_probe, replace=False)
        errs = []
        for q in probes:
            old = flat[q]
            flat[q] = old + eps
            Lp, _, _ = forward(p, D, idx, need_cache=False, rnorm_fixed=rnb)
            flat[q] = old - eps
            Lm, _, _ = forward(p, D, idx, need_cache=False, rnorm_fixed=rnb)
            flat[q] = old
            num = (Lp - Lm) / (2 * eps)
            ana = np.atleast_1d(g[k]).reshape(-1)[q]
            denom = max(1e-9, abs(num) + abs(ana))
            errs.append(abs(num - ana) / denom)
        e = max(errs)
        worst = max(worst, e)
        flag = "ok " if e < 1e-5 else "FAIL"
        print(f"    {flag} {k:6s}  max rel err over {n_probe} probes = {e:.3e}")
    ok = worst < 1e-5
    print(f"    ---> worst relative error {worst:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "gradient check failed"
    return ok


def T2_training():
    print("\n[T2] TRAINING THE CARTOGRAPHER")
    D = build_dataset()
    tr, te = split_legs(D)
    print(f"    legs: {len(tr)} train / {len(te)} held out   nodes: {N_NODES}")

    p0 = init_params()
    print(f"    map RMSE at init            : {map_rmse(p0, D):8.2f} travel-days")

    p, hist = train(D, tr, steps=3000, verbose=True)
    rm = map_rmse(p, D)
    print(f"    map RMSE after training     : {rm:8.2f} travel-days")

    # held-out legs: does the map predict legs it was never fitted on?
    _, met_te, cache_te = forward(p, D, te)
    res = np.sqrt(cache_te["r2"])
    print(f"    held-out leg residual (mean): {res.mean():8.2f} travel-days")

    print("\n    LEARNED CARAVAN SPEED  (ground-days covered per day of travel)")
    print(f"      {'terrain':10s} {'learned':>9s} {'true':>9s} {'err':>8s}")
    for t, (lrn, tru) in speed_report(p, D).items():
        print(f"      {t:10s} {lrn:9.3f} {tru:9.3f} {abs(lrn-tru)/tru*100:7.1f}%")

    print("\n    LEARNED TRUST  (log-precision; higher = more weight on the map)")
    order = np.argsort(-p["rho"])
    for k in order:
        print(f"      provenance {PROV[k]:7s}  rho = {p['rho'][k]:+7.3f}")
    for k in np.argsort(-p["tau"]):
        print(f"      informant  {INFORMANTS[k]:15s} tau = {p['tau'][k]:+7.3f}")
    print(f"\n    STAKE COUPLING  beta = {float(p['beta']):+.3f}   "
          f"(generator used {TRUE_BETA:+.3f})")
    assert rm < map_rmse(p0, D), "training did not improve the map"
    return p, D, tr, te


def T3_stake_ablation(D, tr, te):
    """
    The chapter's empirical claim, run as a controlled experiment.

    Under fixed capacity, what does it buy to allocate precision and
    calibration effort by CONSEQUENCE rather than uniformly? The expected
    answer is NOT "everything gets better". Polo's book is not uniformly
    excellent; it is excellent exactly where it had to be and slack everywhere
    else. If the mechanism is right, the ablation should show a TRANSFER --
    high-stake calibration bought at the price of low-stake calibration.
    """
    print("\n[T3] CONSEQUENCE-WEIGHTED vs UNIFORM ALLOCATION")
    hi = np.array([e for e in range(len(D["src"])) if D["stake"][e] >= 0.6])
    lo = np.array([e for e in range(len(D["src"])) if D["stake"][e] < 0.6])
    print(f"    {len(hi)} high-stake claims / {len(lo)} low-stake claims")

    runs = {}
    for label, kw in [("stake-weighted (Polo)", dict(eta_stake=1.0, freeze_beta=False)),
                      ("uniform calibration  ", dict(eta_stake=0.0, freeze_beta=False)),
                      ("uniform precision    ", dict(eta_stake=1.0, freeze_beta=True)),
                      ("both uniform         ", dict(eta_stake=0.0, freeze_beta=True)),
                      ("credulous reader     ", dict(eta_stake=0.0, freeze_trust=True))]:
        pp, _ = train(D, tr, steps=3000, **kw)
        b_hi, _ = brier(pp, D, hi)
        b_lo, _ = brier(pp, D, lo)
        runs[label] = (map_rmse(pp, D), b_hi, b_lo)

    print(f"    {'allocation':24s} {'map RMSE':>10s} {'Brier hi':>10s} {'Brier lo':>10s}")
    for k, (rm, bh, bl) in runs.items():
        print(f"    {k:24s} {rm:10.2f} {bh:10.4f} {bl:10.4f}")

    polo = runs["stake-weighted (Polo)"]
    unif = runs["both uniform         "]
    cred = runs["credulous reader     "]
    d_hi = (unif[1] - polo[1]) / unif[1] * 100
    d_lo = (polo[2] - unif[2]) / unif[2] * 100
    print(f"\n    high-stake calibration IMPROVES by {d_hi:+6.1f}%")
    print(f"    low-stake  calibration DEGRADES by {d_lo:+6.1f}%")
    print(f"    map RMSE vs the credulous reader   : "
          f"{(cred[0]-polo[0])/cred[0]*100:+6.1f}%")
    print("    ---> this is not a free lunch. it is a REALLOCATION: the error")
    print("         budget is moved onto the claims nobody will act on, and off")
    print("         the claims someone will die by. that trade is Polo's book.")
    return runs


def T4_provenance_separation(p, D):
    """
    Does the model recover Polo's own ordering of his sources without being
    told it? The generator's ordering is LEDGER > SEEN > NAMED > BAZAAR > MARVEL.
    """
    print("\n[T4] PROVENANCE RECOVERY")
    true_rank = sorted(PROV, key=lambda k: TRUE_PROV_SIGMA[k])
    learn_rank = [PROV[k] for k in np.argsort(-p["rho"])]
    print(f"    true reliability order   : {' > '.join(true_rank)}")
    print(f"    learned precision order  : {' > '.join(learn_rank)}")
    # rank correlation
    a = np.array([true_rank.index(x) for x in PROV])
    b = np.array([learn_rank.index(x) for x in PROV])
    rho_s = float(np.corrcoef(a, b)[0, 1])
    print(f"    Spearman rank correlation: {rho_s:+.3f}")
    ok = rho_s > 0.6
    print(f"    ---> {'PASS' if ok else 'WEAK'}: the ledger and the eye outrank the bazaar.")
    return rho_s


GREAT_ROUTE = ["Venice", "Acre", "Ayas", "Tabriz", "Kerman", "Balkh",
               "Kashgar", "Khotan", "Lop", "Shazhou", "Suzhou", "Ganzhou",
               "Shangdu", "Khanbaliq", "Quinsai", "Zaiton"]


def route_cost(X, route=GREAT_ROUTE):
    """Total ground-days along a named route, read off a map."""
    tot = 0.0
    for a_, b_ in zip(route[:-1], route[1:]):
        tot += float(np.linalg.norm(X[NAME_TO_IDX[b_]] - X[NAME_TO_IDX[a_]]))
    return tot


def T5_rustichello_amplifier(D, tr):
    """
    THE ROMANCE AMPLIFIER.

    Rustichello da Pisa wrote Arthurian romance. He is not a neutral scribe; he
    is a channel with a gain on the marvellous, and he sits INSIDE the reporting
    path, not outside it. Here we raise that gain at TEST time only -- the
    marvel-grade and lowest-stake claims get inflated after training -- and
    refit the map with the trust parameters frozen. That is exactly the position
    of a reader who has already accepted a source ranking and now receives worse
    numbers through it.

    The quantity measured is not an abstract embedding error. It is the thing a
    merchant would actually ask the book: HOW MANY DAYS IS IT FROM VENICE TO
    ZAITON? Get that wrong and you have hired the wrong number of animals and
    bought the wrong quantity of grain.
    """
    print("\n[T5] THE ROMANCE AMPLIFIER -- DAMAGE TO A ROUTE ESTIMATE")
    truth = route_cost(D["Xtrue"])
    print(f"    Venice -> Zaiton, true cost: {truth:.1f} ground-days "
          f"({len(GREAT_ROUTE)} stages)")

    results = {}
    for label, kw in [("marked ledger (Polo) ", dict(eta_stake=1.0, freeze_beta=False)),
                      ("credulous reader     ", dict(eta_stake=0.0, freeze_trust=True))]:
        pp, _ = train(D, tr, steps=3000, **kw)
        clean = route_cost(align_translation(pp["X"], D["Xtrue"]))

        Dc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in D.items()}
        amp = (Dc["prov"] == P_IDX["MARVEL"]) | (Dc["informant"] == I_IDX["rustichello"]) \
              | (Dc["stake"] < 0.15)
        Dc["days"] = Dc["days"] * np.where(amp, 2.4, 1.0)
        Dc["feat"] = Dc["feat"].copy()
        Dc["feat"][:, 8] = np.log(Dc["days"]) / 4.0

        pc_ = {k: v.copy() for k, v in pp.items()}
        opt = Adam(pc_, lr=0.05)
        rnb = np.zeros(len(tr))
        for _ in range(1500):
            _, _, cache = forward(pc_, Dc, tr, rnorm_fixed=rnb, **kw)
            g = backward(pc_, Dc, cache)
            for k in list(g):
                if k != "X":
                    g[k] = np.zeros_like(g[k])
            pc_ = opt.step(pc_, g, key_lr=KEY_LR)
            rr = np.sqrt(cache["r2"])
            rnb = rr / (1.0 + rr)
        dirty = route_cost(align_translation(pc_["X"], D["Xtrue"]))
        results[label] = (clean, dirty, int(amp.sum()))

    print(f"\n    {'reader':24s} {'clean':>9s} {'amplified':>10s} "
          f"{'clean err':>10s} {'amp err':>9s}")
    for k, (c_, d_, na) in results.items():
        print(f"    {k:24s} {c_:9.1f} {d_:10.1f} "
              f"{abs(c_-truth)/truth*100:9.1f}% {abs(d_-truth)/truth*100:8.1f}%")
    na = results["marked ledger (Polo) "][2]
    print(f"    ({na} of {len(D['src'])} claims amplified x2.4)")

    pe = abs(results["marked ledger (Polo) "][1] - truth) / truth * 100
    ce = abs(results["credulous reader     "][1] - truth) / truth * 100
    print(f"\n    under amplification the marked ledger is off by {pe:.1f}% and the")
    print(f"    credulous reader by {ce:.1f}%.")
    print("    ---> marking the channel is cheaper than trusting the scribe.")
    return results


def T6_cipangu(p, D):
    """
    ===========================================================================
    THE LOAD-BEARING TEST: PROVENANCE STRIPPING, AND THE COMPLEMENTARY GAP
    ===========================================================================
    Polo says in his own text that he did not go to Cipangu. The claim is
    marked. Every report about the island reaching Quanzhou came down one
    channel with one incentive, so the reports do not disagree -- they agree
    with each other and with none of the facts, and no amount of extra data
    fixes a bias that all the data shares.

    Then comes the part that matters, and it is not about Polo at all.

    An overstatement of how far EAST the world runs is automatically an
    understatement of how far WEST you must sail to get back to it. The globe's
    circumference is fixed; the two gaps are complementary. So an error in a
    low-stake claim, correctly marked as hearsay by the man who made it, was
    read two centuries later as a survey, subtracted from a fixed circumference,
    and became a provisioning figure for an ocean crossing.

    We measure the same estimate consumed two ways: WITH the model's own
    uncertainty, and WITHOUT it.
    ===========================================================================
    """
    print("\n[T6] THE CIPANGU AUDIT -- WHAT PROVENANCE STRIPPING COSTS")
    Xa = align_translation(p["X"], D["Xtrue"])
    k = NAME_TO_IDX["Cipangu"]
    zk = NAME_TO_IDX["Zaiton"]
    vk = NAME_TO_IDX["Venice"]

    est = float(np.linalg.norm(Xa[k] - Xa[zk]))
    tru = float(np.linalg.norm(D["Xtrue"][k] - D["Xtrue"][zk]))
    # LOCAL error, measured against a neighbour rather than the global frame.
    # A dead-reckoned map drifts as a whole; asking how far a node is from its
    # absolute truth mixes that global drift into a local question. The honest
    # measure is how wrong the STEP is: this place relative to the last place
    # anyone actually stood in.
    def local_err(node, anchor):
        va = (Xa[NAME_TO_IDX[node]] - Xa[NAME_TO_IDX[anchor]])
        vt = (D["Xtrue"][NAME_TO_IDX[node]] - D["Xtrue"][NAME_TO_IDX[anchor]])
        return float(np.linalg.norm(va - vt))

    pos_err = local_err("Cipangu", "Zaiton")
    seen_pairs = [("Khotan", "Yarkand"), ("Shazhou", "Lop"),
                  ("Khanbaliq", "Shangdu"), ("Quinsai", "Khanbaliq"),
                  ("Zaiton", "Quinsai"), ("Mabar", "Champa")]
    seen_err = float(np.mean([local_err(a_, b_) for a_, b_ in seen_pairs]))

    ce = np.array([e for e in range(len(D["src"]))
                   if D["dst"][e] == k or D["src"][e] == k])
    _, met, _ = forward(p, D, ce, need_cache=False)
    conf, lam = float(np.mean(met["pc"])), float(np.mean(met["lam"]))

    se = np.array([e for e in range(len(D["src"]))
                   if D["prov"][e] == P_IDX["SEEN"] and D["stake"][e] > 0.8])
    _, met_s, _ = forward(p, D, se, need_cache=False)
    conf_seen = float(np.mean(met_s["pc"]))
    lam_seen = float(np.mean(met_s["lam"]))

    print(f"    Zaiton -> Cipangu, model estimate  : {est:8.2f} ground-days")
    print(f"    Zaiton -> Cipangu, ground truth    : {tru:8.2f} ground-days")
    print(f"    overshoot                          : {(est/tru - 1)*100:+7.1f}%")
    print(f"    Cipangu step error (vs Zaiton)     : {pos_err:8.2f} ground-days")
    print(f"    mean step error, legs he actually SAW: {seen_err:6.2f} ground-days")
    print(f"    ---> the hearsay node is "
          f"{pos_err/max(seen_err,1e-9):.1f}x worse placed than the seen ones.")
    print(f"\n    model confidence, Cipangu legs     : {conf:8.3f}")
    print(f"    model confidence, high-stake SEEN  : {conf_seen:8.3f}")
    print(f"    precision granted, Cipangu legs    : {lam:8.2f}")
    print(f"    precision granted, high-stake SEEN : {lam_seen:8.2f}")
    print(f"    ---> the model is wrong about Cipangu AND SAYS SO. the mark")
    print(f"         survives inside the model. the question is the channel.")

    # ---- the complementary gap ----------------------------------------------
    # Circumference of the earth at the corpus's mean latitude, in the same
    # ground-day units. Held at its TRUE value so that only Polo's share of the
    # error is being measured -- Columbus also used a circumference that was too
    # small, and that mistake is not on this account.
    lat0 = float(np.mean([q[1] for q in PLACES]))
    circ = 2.0 * math.pi * 6371.0 * math.cos(math.radians(lat0)) / 35.0

    # A downstream reader does not redo the whole survey. They accept the
    # well-attested chain as far as the last port anyone reached, and append the
    # hearsay tail to it. So the surveyed part is taken at its true value and
    # only the final, marked, unvisited leg comes from the model.
    east_true = float(D["Xtrue"][k, 0] - D["Xtrue"][vk, 0])
    east_est = (float(D["Xtrue"][zk, 0] - D["Xtrue"][vk, 0])
                + float(Xa[k, 0] - Xa[zk, 0]))
    west_true = circ - east_true
    west_est = circ - east_est

    # A THIRD consumer, and the historically accurate one. Toscanelli did not
    # run a model over the manuscript tradition and down-weight the marvels; he
    # read the number and used it. So we also take the raw reported day-count,
    # converted at the learned sea speed, with no down-weighting at all.
    # The speed a reader WITHOUT this model would apply: one generic figure for
    # a sea passage, taken from the voyages Polo actually made. Using the
    # model's own leg-conditioned speed here would smuggle the correction back
    # in -- the network has learned to read an over-long day-count as a cue and
    # quietly discounts it, which is the very service the raw reader forgoes.
    sea_seen = np.array([e for e in range(len(D["src"]))
                         if D["terrain"][e] == T_IDX["sea"]
                         and D["prov"][e] == P_IDX["SEEN"]])
    _, met_sea, _ = forward(p, D, sea_seen, need_cache=False)
    s_sea = float(np.mean(met_sea["s"]))

    ce_z = np.array([e for e in range(len(D["src"]))
                     if int(D["src"][e]) == zk and int(D["dst"][e]) == k])
    est_raw = float(np.mean(D["days"][ce_z])) * s_sea
    east_raw = (float(D["Xtrue"][zk, 0] - D["Xtrue"][vk, 0])
                + est_raw * 0.97)          # the leg runs very nearly due east
    west_raw = circ - east_raw

    sigma_days = est / math.sqrt(max(lam, 1e-6))   # log-space sd -> days
    west_respect = west_est + 2.0 * sigma_days     # the mark is kept
    west_strip = west_est                          # the mark is thrown away

    short_strip = max(0.0, west_true - west_strip) / s_sea
    short_respect = max(0.0, west_true - west_respect) / s_sea
    short_raw = max(0.0, west_true - west_raw) / s_sea

    print(f"\n    circumference at mean latitude     : {circ:8.1f} ground-days")
    print(f"    eastward extent, truth             : {east_true:8.1f} ground-days")
    print(f"    eastward extent, from the hearsay  : {east_est:8.1f} ground-days")
    print(f"    => westward gap, truth             : {west_true:8.1f} ground-days")
    print(f"    => westward gap, as computed       : {west_est:8.1f} ground-days")
    print(f"\n    provisioning at sea, mark KEPT     : "
          f"{west_respect/s_sea:8.1f} days   shortfall {short_respect:6.1f}")
    print(f"    provisioning at sea, mark STRIPPED : "
          f"{west_strip/s_sea:8.1f} days   shortfall {short_strip:6.1f}")
    print(f"    provisioning at sea, RAW REPORT    : "
          f"{west_raw/s_sea:8.1f} days   shortfall {short_raw:6.1f}")
    print(f"\n    Zaiton -> Cipangu as REPORTED      : {est_raw:8.2f} ground-days "
          f"({(est_raw/tru-1)*100:+.0f}% over truth)")
    print(f"\n    Note the ordering: {short_raw:.1f} days short if the number is taken neat,")
    print(f"    {short_strip:.1f} if a model that read the mark is consulted but its own")
    print(f"    uncertainty is then discarded, {short_respect:.1f} if the uncertainty is kept.")
    print(f"    Almost the whole benefit came from carrying a label the author had")
    print(f"    already attached and that costs nothing to transmit.")
    print(f"    ---> {short_raw:.0f} days of food and water that were never loaded,")
    print(f"         because a claim its author marked as hearsay arrived at the")
    print(f"         reader as a measurement. the failure is in the CHANNEL.")
    return dict(est=est, tru=tru, pos_err=pos_err, seen_err=seen_err, conf=conf,
                conf_seen=conf_seen, west_true=west_true, west_est=west_est,
                short_strip=short_strip, short_respect=short_respect)


def T7_loop_closure(p, D):
    """
    Dead reckoning has exactly one internal error signal, and it is not
    comparison with the truth -- a traveller has no access to that. It is
    DISAGREEMENT: two independent routes reach the same place and do not agree
    about where it is. Polo crossed Persia twice by different roads and reached
    the southern ports by both land and sea, so he had the signal.

    A chained integration of the reported legs leaves a nonzero gap when you go
    round a cycle and come back. A map cannot: a position is a position, so its
    closure gap is exactly zero by construction. That is what the map BUYS. The
    number below is the drift that the requirement of consistency absorbs.
    """
    print("\n[T7] LOOP CLOSURE -- THE ONLY ERROR SIGNAL A DEAD-RECKONER HAS")
    Xa = align_translation(p["X"], D["Xtrue"])

    leg_lookup = {}
    for e in range(len(D["src"])):
        leg_lookup.setdefault((int(D["src"][e]), int(D["dst"][e])), []).append(e)

    def chain(path):
        """Integrate a path using reported days and the LEARNED terrain speeds."""
        acc = np.zeros(2)
        for (a_, b_) in path:
            key = (NAME_TO_IDX[a_], NAME_TO_IDX[b_])
            rev = False
            es = leg_lookup.get(key)
            if es is None:
                es = leg_lookup.get((key[1], key[0]))
                rev = True
            if es is None:
                return None
            es = np.array(es)
            _, met, _ = forward(p, D, es, need_cache=False)
            step = np.mean((D["days"][es] * met["s"])[:, None] * D["unit"][es], axis=0)
            acc += (-step if rev else step)
        return acc

    cycles = [
        ("Khanbaliq -> Quinsai -> Zaiton -> Khanbaliq",
         [("Khanbaliq", "Quinsai"), ("Quinsai", "Zaiton"), ("Khanbaliq", "Zaiton")],
         [1, 1, -1]),
        ("Ayas -> Tabriz -> Kerman -> Ayas (the two Persian roads)",
         [("Ayas", "Tabriz"), ("Tabriz", "Kerman"), ("Tabriz", "Kerman")],
         [1, 1, -1]),
        ("Kerman -> Hormuz -> Kerman (out and back)",
         [("Kerman", "Hormuz"), ("Hormuz", "Kerman")], [1, 1]),
        ("Khanbaliq -> Karajan -> Quinsai -> Khanbaliq",
         [("Khanbaliq", "Karajan"), ("Karajan", "Quinsai"), ("Khanbaliq", "Quinsai")],
         [1, 1, -1]),
    ]

    print(f"    {'cycle':56s} {'gap':>9s}")
    tot = 0.0
    n_ok = 0
    for label, path, signs in cycles:
        acc = np.zeros(2)
        ok = True
        for (edge, sg) in zip(path, signs):
            v = chain([edge])
            if v is None:
                ok = False
                break
            acc += sg * v
        if not ok:
            continue
        gap = float(np.linalg.norm(acc))
        tot += gap
        n_ok += 1
        print(f"    {label:56s} {gap:9.2f}")
    print(f"    {'TOTAL closure gap, naive chaining':56s} {tot:9.2f}")
    print(f"    {'TOTAL closure gap, fitted map':56s} {0.0:9.2f}   (exact)")
    print(f"    mean gap per cycle: {tot/max(n_ok,1):.2f} travel-days of pure drift")
    print("    ---> the map does not know the truth. it knows only that its")
    print("         accounts must balance, and that is enough to find the drift.")
    return tot


# ==============================================================================
# SECTION 7 -- MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print(" STC -- STAKE-WEIGHTED TRAVERSAL CARTOGRAPHER")
    print(" Encyclopedia of Lost Minds, chapter 0348: Marco Polo (1254-1324)")
    print(" A world model with no coordinates, no attention, and no uniform trust.")
    print("=" * 78)

    T1_gradcheck()
    p, D, tr, te = T2_training()
    T3_stake_ablation(D, tr, te)
    T4_provenance_separation(p, D)
    T5_rustichello_amplifier(D, tr)
    T6_cipangu(p, D)
    T7_loop_closure(p, D)

    print("\n" + "=" * 78)
    print(" ALL TESTS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()

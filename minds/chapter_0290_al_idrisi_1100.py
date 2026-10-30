"""
================================================================================
 MIND 0290 -- MUHAMMAD AL-IDRISI (b. 1100, Sabtah/Ceuta -- d. 1165/66)
 # Encyclopedia of Lost Minds: Echoes on AI · Chapter 0290 · Muhammad al-Idrisi (b. 1100, Sabtah/Ceuta -- d. 1165/66)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0290_al_idrisi_1100 - Muhammad al-Idrisi (b. 1100, Sabtah/Ceuta -- d. 1165/66)
 A trainable architecture built from his distinctive cognitive signature.
================================================================================

WHAT THIS IS NOT
    This is not a Transformer. It has no attention over a stored key/value
    cache, no token stream, no autoregressive next-token loss. Al-Idrisi did
    not compile a library and rank passages by relevance; he ran something
    closer to a court of inquiry, where witnesses testified, testimonies were
    weighed against each other, and only what survived cross-examination was
    engraved onto the silver disc.

WHAT THIS IS
    A model of exactly that court of inquiry, built around the one move that
    is distinctively al-Idrisi's and not Ptolemy's, not the oral Balkhi
    tradition's, and not a modern geodesist's:

        He separated the ADDRESS from the CONTENT.

    Every place in the Nuzhat al-Mushtaq gets a permanent, formal coordinate
    -- a climate band (1-7) and a section (1-10) -- fixed by mathematical
    convention, independent of who is talking about the place. Onto that
    fixed address he hung mutable, revisable, testimony-sourced content:
    distances, products, customs, river courses -- claims that could be
    corrected the moment a better witness appeared, WITHOUT the address
    itself ever moving. The grid does not care who is right. The grid only
    provides the slots into which right and wrong answers can be compared.

    This file trains that machine on a synthetic twelfth-century world:

    1.  PLACE EMBEDDINGS (the address space).
        Each of N places gets a learnable 2D coordinate. This is the disc
        al-Idrisi is engraving: the model must recover it purely from noisy,
        conflicting, partial testimony, the way al-Idrisi had to recover the
        shape of the Mediterranean from sailors who had only ever seen one
        edge of it.

    2.  WITNESSES WITH RELIABILITY AND CORRECTABLE BIAS (the testimony
        layer). Each of M sources reports noisy pairwise distances between
        places it claims to know. Every source carries two learnable
        numbers: a reliability weight, and a systematic scale/offset
        correction. This is the mechanism's central idea, and it is the one
        a generic robust-regression model does NOT make explicit: al-Idrisi
        did not treat "Ptolemy is sometimes wrong" and "that sailor's tavern
        story is worthless" as the same failure. One is a systematic,
        internally consistent bias that can be CORRECTED and kept; the other
        is noise that can only be DOWN-WEIGHTED and left aside. The model
        is given both knobs and must discover, from residual statistics
        alone, which source needs which one.

    3.  CORROBORATION-REWEIGHTED TRAINING (the editorial process). Training
        alternates an E-step -- for every place-pair with more than one
        witness, compute how far each witness's corrected report sits from
        the current cross-witness consensus, and turn that into a soft
        corroboration weight -- and an M-step -- one gradient step of every
        parameter against the corroboration-weighted, reliability-weighted,
        robust loss. A claim that no one else confirms is kept, but it is
        never allowed to dominate a claim that three independent travelers
        agree on. This is al-Idrisi's actual editorial practice, made
        differentiable.

    4.  HIERARCHICAL CLIMATE-ZONE GATING (the atlas). Once place embeddings
        are trained, they are assigned -- by a fixed, non-learned rule, not
        a soft attention score -- into a 7 x 10 grid of climate bands and
        sections, exactly al-Idrisi's own partition of the oikumene. Each
        occupied cell gets its own tiny local expert that predicts regional
        attributes (a stand-in for "products, customs, terrain") from a
        place's coordinate plus the attributes of its cell-mates. This is
        the progressive-disclosure structure of the Nuzhat al-Mushtaq itself
        -- one global disc, seventy regional sheets -- reproduced as
        geometric (not learned) mixture-of-experts routing.

    Every gradient below is hand-derived and pure NumPy. No autograd, no
    ML framework. A finite-difference gradient check is run before training
    and is a hard assertion, not a suggestion.
================================================================================
"""

import numpy as np
import math
import json

RNG_SEED = 1100  # al-Idrisi's birth year; fixes the synthetic world & witnesses
rng = np.random.default_rng(RNG_SEED)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 -- THE SYNTHETIC TWELFTH-CENTURY WORLD
# ══════════════════════════════════════════════════════════════════════════
# A schematic (not geodesically exact) 2D plane of 26 places drawn from the
# real geography of the Nuzhat al-Mushtaq, in a rough east-west / south-north
# layout. Units are arbitrary "day's-journey" scale, not kilometers. Positive
# y runs SOUTH, following al-Idrisi's own south-up convention on the Tabula
# Rogeriana -- a labeling choice only; it does not affect any of the math.

PLACES = [
    # name                    x (east+)   y (south+)     region note
    ("Ceuta (Sabtah)",           -9.0,      2.6),   # al-Idrisi's birthplace
    ("Cordoba",                 -10.5,      1.2),   # his education
    ("Seville",                 -11.4,      1.9),
    ("Fez",                      -9.6,      3.4),
    ("Tunis",                    -3.2,      2.9),
    ("Palermo",                  -1.6,      0.3),   # Roger II's court
    ("Messina",                  -0.9,      0.1),
    ("Genoa",                    -1.0,     -2.1),
    ("Rome",                     -1.7,     -0.6),
    ("Venice",                    1.4,     -2.4),
    ("Constantinople",            6.4,     -1.2),
    ("Alexandria",                4.2,      3.1),
    ("Cairo",                     4.6,      3.6),
    ("Jerusalem",                 5.6,      3.3),
    ("Damascus",                  6.1,      2.8),
    ("Baghdad",                   9.3,      2.6),
    ("Basra",                    10.4,      3.7),
    ("Hormuz",                   12.0,      4.6),
    ("Aden",                      8.6,      7.9),
    ("Mogadishu",                 8.2,     10.4),
    ("Kilwa",                     8.9,     12.8),
    ("Awdaghust",                 -6.5,      7.2),
    ("Ghana (Wangara)",           -5.2,      8.6),
    ("Novgorod",                   4.0,     -9.8),
    ("London",                    -6.9,     -5.4),
    ("Hangzhou (Sin al-Sin)",     22.5,      4.9),
]
PLACE_NAMES = [p[0] for p in PLACES]
N = len(PLACES)
TRUE_COORDS = np.array([[p[1], p[2]] for p in PLACES], dtype=np.float64)

# Synthetic "regional attribute" ground truth for Section 7 (the atlas /
# local-expert task): [trade_value, port_score, aridity, population_density].
# Deliberately built with a distinct REGIME PER CLIMATE BAND (al-Idrisi's own
# organizing principle: each of the seven climates has its own character),
# plus a small smooth within-band trend and noise on top. A single global
# linear model of (x, y) cannot represent seven separate regime jumps well;
# a model gated by climate-band/section -- exactly the Nuzhat al-Mushtaq's
# own structure -- can. This is what Section 7 tests.
N_TRUE_BANDS = 7
def _make_attributes():
    x, y = TRUE_COORDS[:, 0], TRUE_COORDS[:, 1]
    y_min, y_max = y.min() - 1e-6, y.max() + 1e-6
    band_idx = np.clip(((y - y_min) / (y_max - y_min) * N_TRUE_BANDS).astype(int),
                        0, N_TRUE_BANDS - 1)
    band_base = rng.uniform(0.15, 0.85, size=(N_TRUE_BANDS, 4))  # per-band regime
    attrs = band_base[band_idx].copy()
    attrs[:, 0] += 0.08 * np.cos(0.35 * x)           # trade: local trend
    attrs[:, 1] += 0.08 * np.sin(0.4 * x + 0.2 * y)  # port score: local trend
    attrs[:, 2] += 0.05 * np.tanh(0.10 * (y - 3.0))  # aridity: mild local trend
    attrs[:, 3] += 0.06 * np.cos(0.25 * y)           # population: local trend
    attrs += rng.normal(0, 0.025, attrs.shape)
    return np.clip(attrs, 0.0, 1.0)

TRUE_ATTRS = _make_attributes()
ATTR_DIM = TRUE_ATTRS.shape[1]

# ---- Witnesses (sources of testimony) --------------------------------------
# Each source: (name, true_reliability in [0,1], noise_std, bias_scale,
#               bias_offset, "coverage" = fraction of place-pairs it can
#               plausibly report on, systematic=bool)
# systematic=True sources have LOW noise but a real multiplicative/additive
# distortion (like Ptolemy's inherited Mediterranean over-length) that a
# careful compiler can learn to correct rather than discard.
SOURCES = [
    dict(name="Sicilian Royal Port Registry",     rel=0.95, noise=0.04, scale=1.00, offset=0.0,  cover=0.35, systematic=False),
    dict(name="Genoese Sea-Captain's Log",         rel=0.92, noise=0.05, scale=1.00, offset=0.0,  cover=0.30, systematic=False),
    dict(name="Cairo Caravan-Master",              rel=0.90, noise=0.06, scale=1.00, offset=0.0,  cover=0.28, systematic=False),
    dict(name="Andalusian Hajj Pilgrim",           rel=0.78, noise=0.10, scale=1.00, offset=0.0,  cover=0.22, systematic=False),
    dict(name="Ptolemy's Geography (Arabic tr.)",  rel=0.55, noise=0.03, scale=1.22, offset=0.35, cover=0.55, systematic=True),
    dict(name="Zanzibar Coast Trader",             rel=0.85, noise=0.07, scale=1.00, offset=0.0,  cover=0.16, systematic=False),
    dict(name="Rus Fur Trader via Constantinople", rel=0.68, noise=0.13, scale=1.00, offset=0.0,  cover=0.14, systematic=False),
    dict(name="Book of Curiosities marginalia",    rel=0.60, noise=0.11, scale=0.85, offset=-0.20, cover=0.18, systematic=True),
    dict(name="Palermo Customs House",             rel=0.93, noise=0.045,scale=1.00, offset=0.0,  cover=0.24, systematic=False),
    dict(name="Anonymous Tavern Sailor's Tale",    rel=0.30, noise=0.35, scale=1.00, offset=0.0,  cover=0.20, systematic=False),
    dict(name="Legend of the Sea of Darkness",     rel=0.08, noise=0.85, scale=2.60, offset=1.10, cover=0.10, systematic=False),  # deliberately adversarial
]
M = len(SOURCES)
SOURCE_NAMES = [s["name"] for s in SOURCES]


def _true_distance(i, j):
    return float(np.linalg.norm(TRUE_COORDS[i] - TRUE_COORDS[j]))


def generate_reports(min_reports_per_pair_target=1):
    """Generate the noisy, biased, partial testimony each source contributes.
    Returns a list of (i, j, source_idx, reported_distance)."""
    reports = []
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    for s_idx, s in enumerate(SOURCES):
        k = max(1, int(round(s["cover"] * len(all_pairs))))
        chosen = rng.choice(len(all_pairs), size=k, replace=False)
        for c in chosen:
            i, j = all_pairs[c]
            true_d = _true_distance(i, j)
            noisy = true_d + rng.normal(0, s["noise"] * (1.0 + 0.05 * true_d))
            noisy = max(noisy, 0.05)
            reported = (noisy - s["offset"]) / s["scale"]  # witness reports
            # what the witness *reports* is the inverse of the corrupting
            # process: reported * scale + offset ~= true_d + noise
            reported = max(reported, 0.02)
            reports.append((i, j, s_idx, reported))
    return reports


ALL_REPORTS = generate_reports()

# ---- The one non-testimonial fact: an astronomical anchor -----------------
# Pure pairwise testimony can only ever recover the SHAPE of the world, not
# its absolute scale or orientation -- any uniform stretch or rotation fits
# the reports equally well. This is not a modeling inconvenience; it is
# al-Idrisi's actual predicament, and his actual solution: the classical
# tradition he inherited (via Ptolemy, ultimately via Eratosthenes) supplied
# one absolute fact no merchant's testimony could -- the Earth's measured
# circumference, which fixes how many real miles correspond to one degree of
# latitude. That single astronomically-derived fact is what anchors an
# entire testimony-built map to real units. Here, one place-pair (al-Idrisi's
# own birthplace and the court he served) is given a fixed true separation,
# standing in for that one piece of knowledge that came from measurement of
# the heavens, not from a witness's memory of a journey.
ANCHOR_I = PLACE_NAMES.index("Ceuta (Sabtah)")
ANCHOR_J = PLACE_NAMES.index("Palermo")
ANCHOR_TRUE_DISTANCE = _true_distance(ANCHOR_I, ANCHOR_J)

# Hold out a fraction of PAIRS entirely (no source reports them at all) to
# test whether the trained embedding can still predict a distance it was
# never directly told -- the "route al-Idrisi never heard testimony about,
# but whose length follows from the shape of the map he has already built."
_pairs_with_reports = sorted({(r[0], r[1]) for r in ALL_REPORTS})
_pairs_without_reports = [(i, j) for i in range(N) for j in range(i + 1, N)
                           if (i, j) not in set(_pairs_with_reports)]
rng.shuffle(_pairs_without_reports)
HELD_OUT_PAIRS = _pairs_without_reports[:60] if len(_pairs_without_reports) >= 60 \
    else _pairs_without_reports


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 -- NUMERICAL PRIMITIVES (loss, activation, hand-derived grads)
# ══════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def dsigmoid_dx(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def huber(residual, delta=0.5):
    """Elementwise Huber loss (robust to outlier witnesses)."""
    a = np.abs(residual)
    quad = 0.5 * residual ** 2
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def dhuber_dresidual(residual, delta=0.5):
    return np.clip(residual, -delta, delta)


class Adam:
    """Adam optimizer implemented from scratch for a dict of NumPy arrays."""

    def __init__(self, params, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 -- THE MODEL: address embeddings + witness trust/correction
# ══════════════════════════════════════════════════════════════════════════

class ReportBank:
    """Vectorized view over the raw (i, j, source, reported_distance) list."""

    def __init__(self, reports):
        arr = np.array(reports, dtype=np.float64)
        self.I = arr[:, 0].astype(np.int64)
        self.J = arr[:, 1].astype(np.int64)
        self.S = arr[:, 2].astype(np.int64)
        self.R = arr[:, 3].astype(np.float64)
        self.n = len(reports)
        # group reports by unordered (i, j) pair for corroboration lookup
        self.pair_key = self.I * 10_000 + self.J
        self._pair_groups = {}
        for idx, k in enumerate(self.pair_key):
            self._pair_groups.setdefault(int(k), []).append(idx)

    def corroboration_weights(self, corrected, tau=0.35, single_witness_weight=0.55):
        """E-step: how strongly does each report agree with the current
        cross-witness consensus on its (place, place) pair? Stop-gradient:
        this is computed from current parameter values but is treated as a
        constant during the following gradient step, exactly as al-Idrisi
        could revise which testimonies counted as 'corroborated' each time
        a new report arrived, without that revision itself being a
        differentiable act."""
        w = np.ones(self.n, dtype=np.float64)
        for k, idxs in self._pair_groups.items():
            if len(idxs) == 1:
                w[idxs[0]] = single_witness_weight
                continue
            vals = corrected[idxs]
            med = np.median(vals)
            w[idxs] = np.exp(-np.abs(vals - med) / tau)
        return w


class TriangulatedWorldModel:
    """
    Learns:
        X                 (N, 2)  -- the addressable coordinate of each place
        reliability_logits(M,)    -- sigmoid -> trust in [0, 1] per witness
        scale_logits      (M,)    -- exp -> multiplicative correction
        offset            (M,)    -- additive correction

    forward(): reconstructs each witness's reported distance as a corrected,
    reliability- and corroboration-weighted comparison against the model's
    own current geometry, and returns a scalar loss plus every gradient by
    hand (no autograd).
    """

    def __init__(self, n_places, n_sources, seed=0):
        r = np.random.default_rng(seed)
        self.n_places, self.n_sources = n_places, n_sources
        self.params = {
            "X": r.normal(0, 1.0, size=(n_places, 2)),
            "reliability_logits": r.normal(0.3, 0.2, size=n_sources),
            "scale_logits": r.normal(0, 0.02, size=n_sources),
            "offset": r.normal(0, 0.02, size=n_sources),
        }
        self.lam_scale = 0.02
        self.lam_offset = 0.02
        self.lam_rel = 2e-4
        self.lam_x = 2e-4
        self.eps_dist = 1e-6
        # Anti-collapse uncertainty cost. Without this, "distrust every
        # witness completely" is a trivial global minimum of a plain
        # weighted loss (weight -> 0 makes the data term vanish for free).
        # Al-Idrisi could not simply refuse to compile a map because every
        # witness was imperfect; declining to judge is itself a cost. Each
        # report therefore also pays -c * log(reliability), which grows
        # without bound as trust in a source falls to zero, so a source
        # is only down-weighted as far as its actual disagreement with the
        # emerging consensus justifies -- this turns "reliability" into a
        # real trade-off instead of a free knob.
        self.c_uncertainty = 0.22
        self.rel_floor = 1e-4
        # Weight on the single astronomical anchor distance (see ANCHOR_I/J
        # above). This is what fixes absolute scale; without it the model
        # (like pure testimony alone) can only recover the world's shape.
        self.lam_anchor = 6.0

    # -- shared forward pieces -------------------------------------------
    def _predicted_and_corrected(self, params, bank: ReportBank):
        X = params["X"]
        diff = X[bank.I] - X[bank.J]                      # (R, 2)
        sq = np.sum(diff * diff, axis=1)
        pred = np.sqrt(sq + self.eps_dist)                 # (R,)
        scale = np.exp(params["scale_logits"][bank.S])
        corrected = bank.R * scale + params["offset"][bank.S]
        return pred, corrected, diff, scale

    def loss_and_grad(self, params, bank: ReportBank, corrob_weights):
        pred, corrected, diff, scale = self._predicted_and_corrected(params, bank)
        residual = pred - corrected
        rel_per_source = sigmoid(params["reliability_logits"])
        rel = rel_per_source[bank.S]
        weight = rel * corrob_weights
        per_report_loss = huber(residual)
        data_loss = np.sum(weight * per_report_loss) / bank.n
        uncertainty_cost = -self.c_uncertainty * np.sum(
            np.log(rel + self.rel_floor)) / bank.n

        reg = (self.lam_scale * np.sum(params["scale_logits"] ** 2)
               + self.lam_offset * np.sum(params["offset"] ** 2)
               + self.lam_rel * np.sum(params["reliability_logits"] ** 2)
               + self.lam_x * np.sum(params["X"] ** 2))

        # -- the astronomical anchor: one hard, non-testimonial fact --------
        anchor_diff = params["X"][ANCHOR_I] - params["X"][ANCHOR_J]
        anchor_pred = float(np.sqrt(np.sum(anchor_diff ** 2) + self.eps_dist))
        anchor_residual = anchor_pred - ANCHOR_TRUE_DISTANCE
        anchor_loss = self.lam_anchor * anchor_residual ** 2

        total = data_loss + uncertainty_cost + reg + anchor_loss

        # ---- gradients (hand-derived backprop) --------------------------
        dL_dresidual = weight * dhuber_dresidual(residual) / bank.n     # (R,)

        # d residual / d pred = 1
        g_pred = dL_dresidual
        unit = diff / pred[:, None]                                     # (R,2)
        grad_X = np.zeros_like(params["X"])
        np.add.at(grad_X, bank.I, g_pred[:, None] * unit)
        np.add.at(grad_X, bank.J, -g_pred[:, None] * unit)
        grad_X += 2 * self.lam_x * params["X"]

        # anchor gradient
        anchor_unit = anchor_diff / anchor_pred
        d_anchor = 2 * self.lam_anchor * anchor_residual
        grad_X[ANCHOR_I] += d_anchor * anchor_unit
        grad_X[ANCHOR_J] -= d_anchor * anchor_unit

        # d residual / d corrected = -1
        dL_dcorrected = -dL_dresidual
        grad_scale_logits = np.zeros(self.n_sources)
        grad_offset = np.zeros(self.n_sources)
        contrib_scale = dL_dcorrected * bank.R * scale   # d corrected/d scale_logits = R*exp(scale_logits)
        np.add.at(grad_scale_logits, bank.S, contrib_scale)
        np.add.at(grad_offset, bank.S, dL_dcorrected)
        grad_scale_logits += 2 * self.lam_scale * params["scale_logits"]
        grad_offset += 2 * self.lam_offset * params["offset"]

        # reliability: weight = sigmoid(rel_logits[s]) * corrob (corrob is const)
        dsig = dsigmoid_dx(params["reliability_logits"][bank.S])
        contrib_rel = per_report_loss * corrob_weights * dsig / bank.n
        # anti-collapse term: d(-c*log(rel))/d(rel_logits) = -c/rel * dsig
        contrib_uncertainty = (-self.c_uncertainty / (rel + self.rel_floor)) * dsig / bank.n
        grad_rel = np.zeros(self.n_sources)
        np.add.at(grad_rel, bank.S, contrib_rel + contrib_uncertainty)
        grad_rel += 2 * self.lam_rel * params["reliability_logits"]

        grads = {
            "X": grad_X,
            "reliability_logits": grad_rel,
            "scale_logits": grad_scale_logits,
            "offset": grad_offset,
        }
        return total, grads, dict(pred=pred, corrected=corrected, residual=residual)

    def predict_distance(self, params, i, j):
        d = params["X"][i] - params["X"][j]
        return float(np.sqrt(np.sum(d * d) + self.eps_dist))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 -- MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ══════════════════════════════════════════════════════════════════════════

def finite_difference_gradient_check(model: TriangulatedWorldModel, bank: ReportBank,
                                      n_checks=40, eps=1e-5, tol=3e-3, seed=7,
                                      verbose=True):
    """Numerically verifies every hand-derived gradient in Section 3.
    Corroboration weights are frozen (computed once) so that the analytic
    loss being checked is exactly the function whose gradient we derived --
    matching how the E-step / M-step alternation actually trains the model.
    """
    r = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in model.params.items()}
    pred0, corrected0, _, _ = model._predicted_and_corrected(params, bank)
    corrob = bank.corroboration_weights(corrected0)

    def loss_fn(p):
        total, _, _ = model.loss_and_grad(p, bank, corrob)
        return total

    base_loss, grads, _ = model.loss_and_grad(params, bank, corrob)

    checks = []
    for key in params:
        flat = params[key].reshape(-1)
        g_flat = grads[key].reshape(-1)
        idxs = r.choice(len(flat), size=min(n_checks, len(flat)), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus = loss_fn(params)
            flat[idx] = orig - eps
            loss_minus = loss_fn(params)
            flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = g_flat[idx]
            denom = max(1e-8, abs(numeric) + abs(analytic))
            rel_err = abs(numeric - analytic) / denom
            checks.append((key, int(idx), numeric, analytic, rel_err))

    max_err = max(c[4] for c in checks)
    n_fail = sum(1 for c in checks if c[4] > tol)
    if verbose:
        print(f"  gradient check: {len(checks)} params probed, "
              f"max relative error = {max_err:.2e}, tolerance = {tol:.1e}, "
              f"failures = {n_fail}")
        worst = sorted(checks, key=lambda c: -c[4])[:5]
        for key, idx, num, ana, err in worst:
            print(f"    worst[{key}][{idx}]  numeric={num: .6e}  analytic={ana: .6e}  rel_err={err:.2e}")
    assert n_fail == 0, f"Finite-difference gradient check FAILED for {n_fail} parameters"
    return max_err


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 -- TRAINING LOOP (alternating corroboration E-step / gradient M-step)
# ══════════════════════════════════════════════════════════════════════════

def train(model: TriangulatedWorldModel, bank: ReportBank, epochs=900, lr=0.05,
          log_every=100, verbose=True):
    opt = Adam(model.params, lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        # E-step: freeze current geometry/corrections, recompute who agrees
        # with whom -- this is "which testimonies corroborate one another
        # given what we currently believe the map looks like".
        pred, corrected, _, _ = model._predicted_and_corrected(model.params, bank)
        corrob = bank.corroboration_weights(corrected)

        # M-step: one gradient step of every parameter against the
        # corroboration-weighted, reliability-weighted, robust loss.
        loss, grads, _ = model.loss_and_grad(model.params, bank, corrob)
        model.params = opt.step(model.params, grads)

        # Re-center X each epoch (the loss is translation-invariant, so
        # nothing anchors the map to the origin except this housekeeping
        # step -- it changes no distances and needs no gradient).
        model.params["X"] -= model.params["X"].mean(axis=0, keepdims=True)

        history.append(loss)
        if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"  epoch {epoch:4d}/{epochs}  loss={loss:.6f}  "
                  f"mean_corrob={corrob.mean():.3f}")
    return history


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 -- EVALUATION: has the disc been drawn correctly?
# ══════════════════════════════════════════════════════════════════════════

def procrustes_align(X, Y):
    """Best rotation/reflection + uniform scale + translation mapping X onto
    Y (used ONLY for human-readable evaluation against the synthetic ground
    truth -- the model itself never sees TRUE_COORDS during training)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    Rot = U @ Vt
    denom = np.sum(Xc * Xc)
    scale = (np.sum(S) / denom) if denom > 1e-12 else 1.0
    aligned = scale * (Xc @ Rot) + Y.mean(axis=0, keepdims=True)
    return aligned, Rot, scale


def all_pairs_distance_correlation(model, params):
    idx_i, idx_j = np.triu_indices(model.n_places, k=1)
    true_d = np.linalg.norm(TRUE_COORDS[idx_i] - TRUE_COORDS[idx_j], axis=1)
    pred_d = np.linalg.norm(params["X"][idx_i] - params["X"][idx_j], axis=1)
    corr = np.corrcoef(true_d, pred_d)[0, 1]
    return corr, true_d, pred_d


def evaluate_generalization(model, params, held_out_pairs):
    if not held_out_pairs:
        return None
    true_d = np.array([_true_distance(i, j) for i, j in held_out_pairs])
    pred_d = np.array([model.predict_distance(params, i, j) for i, j in held_out_pairs])
    # scale-correct pred_d onto true_d's scale using the same global scale
    # recovered by Procrustes (fair: the model is never told absolute units)
    aligned, _, scale = procrustes_align(params["X"], TRUE_COORDS)
    pred_d_scaled = pred_d * scale
    mse_model = float(np.mean((pred_d_scaled - true_d) ** 2))
    baseline_pred = np.full_like(true_d, true_d.mean())
    mse_baseline = float(np.mean((baseline_pred - true_d) ** 2))
    return dict(mse_model=mse_model, mse_baseline=mse_baseline,
                n_pairs=len(held_out_pairs),
                improvement_ratio=mse_baseline / max(mse_model, 1e-9))


def report_witness_judgments(model, params):
    order = np.argsort(-sigmoid(params["reliability_logits"]))
    rows = []
    for s in order:
        rows.append(dict(
            name=SOURCE_NAMES[s],
            learned_reliability=float(sigmoid(params["reliability_logits"][s])),
            true_reliability=SOURCES[s]["rel"],
            learned_scale=float(np.exp(params["scale_logits"][s])),
            true_scale=SOURCES[s]["scale"],
            learned_offset=float(params["offset"][s]),
            true_offset=SOURCES[s]["offset"],
            systematic_source=SOURCES[s]["systematic"],
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 -- THE ATLAS: hierarchical climate-zone gating (progressive
#               disclosure, exactly al-Idrisi's 7 climates x 10 sections)
# ══════════════════════════════════════════════════════════════════════════
# Grid-cell assignment is a FIXED, non-learned rule -- deterministic geometric
# routing, not a soft attention score. This is deliberate: the address must
# not itself be something a gradient can quietly renegotiate. Each occupied
# cell then gets its own small linear expert trained (with hand-derived
# gradients + its own finite-difference check) to predict a place's regional
# attributes from its coordinate and its cell-mates' attributes -- the
# "regional sheet" that accompanies the world disc.

N_BANDS, N_SECTIONS = 7, 10


def assign_climate_cells(coords):
    """Deterministic 7x10 grid over the (already-oriented) coordinate range.
    Returns an integer cell id in [0, N_BANDS*N_SECTIONS) per place."""
    y, x = coords[:, 1], coords[:, 0]
    y_bins = np.clip(((y - y.min()) / (np.ptp(y) + 1e-9) * N_BANDS).astype(int), 0, N_BANDS - 1)
    x_bins = np.clip(((x - x.min()) / (np.ptp(x) + 1e-9) * N_SECTIONS).astype(int), 0, N_SECTIONS - 1)
    return y_bins * N_SECTIONS + x_bins, y_bins, x_bins


def _leave_one_out_neighbor_mean(attrs, group_ids):
    """For each item, mean attribute of every OTHER item sharing its group
    (falls back to the global leave-one-out mean if it is alone in its
    group)."""
    n = len(attrs)
    out = np.zeros_like(attrs)
    groups = {}
    for idx, g in enumerate(group_ids):
        groups.setdefault(g, []).append(idx)
    global_sum, global_n = attrs.sum(axis=0), n
    for g, idxs in groups.items():
        if len(idxs) == 1:
            i = idxs[0]
            out[i] = (global_sum - attrs[i]) / max(global_n - 1, 1)
        else:
            s = attrs[idxs].sum(axis=0)
            for i in idxs:
                out[i] = (s - attrs[i]) / (len(idxs) - 1)
    return out


class LinearExpert:
    """A single tiny linear layer: features -> attribute prediction.
    y_hat = features @ W + b.  Trained by full-batch gradient descent with
    hand-derived gradients (this is ordinary least squares, so the gradient
    is exact and simple, but it is still derived and checked, not assumed)."""

    def __init__(self, in_dim, out_dim, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 0.05, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, feats):
        return feats @ self.W + self.b

    def loss_and_grad(self, feats, targets, l2=1e-3):
        pred = self.forward(feats)
        err = pred - targets
        n = feats.shape[0]
        loss = float(np.mean(np.sum(err ** 2, axis=1))) + l2 * float(np.sum(self.W ** 2))
        gW = (2.0 / n) * feats.T @ err + 2 * l2 * self.W
        gb = (2.0 / n) * err.sum(axis=0)
        return loss, gW, gb

    def fit(self, feats, targets, epochs=250, lr=0.05, l2=0.05):
        history = []
        for _ in range(epochs):
            loss, gW, gb = self.loss_and_grad(feats, targets, l2)
            self.W -= lr * gW
            self.b -= lr * gb
            history.append(loss)
        return history


def local_expert_gradient_check(expert: LinearExpert, feats, targets, l2=1e-3,
                                 n_checks=12, eps=1e-5, tol=2e-3):
    base_loss, gW, gb = expert.loss_and_grad(feats, targets, l2)
    r = np.random.default_rng(3)
    max_err = 0.0
    for name, param, grad in (("W", expert.W, gW), ("b", expert.b, gb)):
        flat, gflat = param.reshape(-1), grad.reshape(-1)
        idxs = r.choice(len(flat), size=min(n_checks, len(flat)), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            lp, _, _ = expert.loss_and_grad(feats, targets, l2)
            flat[idx] = orig - eps
            lm, _, _ = expert.loss_and_grad(feats, targets, l2)
            flat[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            rel_err = abs(numeric - gflat[idx]) / max(1e-8, abs(numeric) + abs(gflat[idx]))
            max_err = max(max_err, rel_err)
    assert max_err < tol, f"Local-expert gradient check FAILED (max_err={max_err:.2e})"
    return max_err


def build_features(coords, attrs, group_ids):
    """Standardized coordinates (zero mean, unit std) concatenated with each
    place's leave-one-out cell-mate attribute mean. Standardizing keeps
    feature and target magnitudes comparable so a tiny linear expert fit
    with plain gradient descent stays numerically well-behaved even on
    cells with only two or three members."""
    mu = coords.mean(axis=0, keepdims=True)
    sigma = coords.std(axis=0, keepdims=True) + 1e-6
    coords_std = (coords - mu) / sigma
    neighbor_mean = _leave_one_out_neighbor_mean(attrs, group_ids)
    return np.concatenate([coords_std, neighbor_mean], axis=1)   # (N, 2+ATTR_DIM)


def assign_climate_band(coords):
    """Coarser grouping: climate band only (7 groups), ignoring section.
    Al-Idrisi's own prose is often genuinely distinctive at the climate
    level even where section-level testimony is too thin to say anything
    specific -- this is the resolution at which the atlas demonstration
    below is fair to run."""
    y = coords[:, 1]
    y_bins = np.clip(((y - y.min()) / (np.ptp(y) + 1e-9) * N_BANDS).astype(int), 0, N_BANDS - 1)
    return y_bins


def train_geometric_mixture_of_experts(coords, attrs, cell_ids, min_cell_size=3, seed=0):
    """One LinearExpert per occupied cell with >= min_cell_size members.
    Places in smaller, thinly-attested cells fall back to a single expert
    trained on the FULL dataset (not a separate pool of unrelated small
    cells lumped together, which would mix incompatible regimes and help
    no one) -- al-Idrisi's own move for a place he cannot say much about
    specifically: describe it at the coarser resolution the evidence
    actually supports, rather than inventing local detail or discarding it."""
    feats = build_features(coords, attrs, cell_ids)
    cells = {}
    for idx, c in enumerate(cell_ids):
        cells.setdefault(c, []).append(idx)
    global_expert = LinearExpert(feats.shape[1], ATTR_DIM, seed=seed + 9999)
    global_expert.fit(feats, attrs)
    experts = {"_fallback": global_expert}
    for c, idxs in cells.items():
        if len(idxs) >= min_cell_size:
            e = LinearExpert(feats.shape[1], ATTR_DIM, seed=seed + c)
            e.fit(feats[idxs], attrs[idxs])
            experts[c] = e
    return experts, feats, cells


def loo_cv_mse_geometric(coords, attrs, cell_ids, min_cell_size=3):
    """Leave-one-out CV MSE for the cell-gated mixture of local experts."""
    n = len(attrs)
    sq_errs = []
    for held in range(n):
        train_idx = [i for i in range(n) if i != held]
        sub_coords, sub_attrs = coords[train_idx], attrs[train_idx]
        sub_cells = cell_ids[train_idx]
        experts, _, _ = train_geometric_mixture_of_experts(
            sub_coords, sub_attrs, sub_cells, min_cell_size=min_cell_size, seed=held)
        held_cell = cell_ids[held]
        cell_size_in_train = int(np.sum(sub_cells == held_cell))
        mu = sub_coords.mean(axis=0, keepdims=True)
        sigma = sub_coords.std(axis=0, keepdims=True) + 1e-6
        coord_std = (coords[held:held + 1] - mu) / sigma
        if cell_size_in_train >= min_cell_size and held_cell in experts:
            same_cell_train = [i for i, tr in enumerate(train_idx) if cell_ids[tr] == held_cell]
            neighbor_mean = sub_attrs[same_cell_train].mean(axis=0)
            expert = experts[held_cell]
        else:
            neighbor_mean = sub_attrs.mean(axis=0)
            expert = experts["_fallback"]
        feat = np.concatenate([coord_std[0], neighbor_mean])[None, :]
        pred = expert.forward(feat)[0]
        sq_errs.append(np.mean((pred - attrs[held]) ** 2))
    return float(np.mean(sq_errs))


def loo_cv_mse_global_baseline(coords, attrs):
    """Leave-one-out CV MSE for a single ungated global linear model --
    the ablation that removes al-Idrisi's climate-zone structure entirely."""
    n = len(attrs)
    sq_errs = []
    all_group = np.zeros(n, dtype=int)  # everyone in one group
    for held in range(n):
        train_idx = [i for i in range(n) if i != held]
        sub_coords, sub_attrs = coords[train_idx], attrs[train_idx]
        feats = build_features(sub_coords, sub_attrs, all_group[train_idx])
        e = LinearExpert(feats.shape[1], ATTR_DIM, seed=held)
        e.fit(feats, sub_attrs)
        neighbor_mean = sub_attrs.mean(axis=0)
        mu = sub_coords.mean(axis=0, keepdims=True)
        sigma = sub_coords.std(axis=0, keepdims=True) + 1e-6
        coord_std = (coords[held:held + 1] - mu) / sigma
        feat = np.concatenate([coord_std[0], neighbor_mean])[None, :]
        pred = e.forward(feat)[0]
        sq_errs.append(np.mean((pred - attrs[held]) ** 2))
    return float(np.mean(sq_errs))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 -- SELF-TESTS
# ══════════════════════════════════════════════════════════════════════════

def run_self_tests(model, params, history, bank, judgments, generalization,
                    corr, mse_geo, mse_global):
    results = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        results.append((name, status, detail))
        print(f"    [{status}] {name}  {detail}")

    check("loss decreases over training",
          history[-1] < 0.4 * history[0],
          f"first={history[0]:.4f} last={history[-1]:.4f}")

    check("recovered geometry correlates with true geography",
          corr > 0.85, f"pearson r={corr:.4f}")

    if generalization is not None:
        check("beats naive baseline on never-reported routes",
              generalization["improvement_ratio"] > 1.2,
              f"mse_model={generalization['mse_model']:.4f} "
              f"mse_baseline={generalization['mse_baseline']:.4f} "
              f"ratio={generalization['improvement_ratio']:.2f}x")

    by_name = {j["name"]: j for j in judgments}
    legend = by_name["Legend of the Sea of Darkness"]
    tavern = by_name["Anonymous Tavern Sailor's Tale"]
    registry = by_name["Sicilian Royal Port Registry"]
    captain = by_name["Genoese Sea-Captain's Log"]
    ptolemy = by_name["Ptolemy's Geography (Arabic tr.)"]

    check("adversarial 'Sea of Darkness' legend is down-weighted below direct witnesses",
          legend["learned_reliability"] < registry["learned_reliability"] and
          legend["learned_reliability"] < captain["learned_reliability"],
          f"legend={legend['learned_reliability']:.3f} "
          f"registry={registry['learned_reliability']:.3f} "
          f"captain={captain['learned_reliability']:.3f}")

    check("unreliable tavern tale ranks below the two most direct witnesses",
          tavern["learned_reliability"] < registry["learned_reliability"] and
          tavern["learned_reliability"] < captain["learned_reliability"],
          f"tavern={tavern['learned_reliability']:.3f}")

    check("classical authority (Ptolemy) receives a real correction, not a rejection",
          abs(ptolemy["learned_scale"] - 1.0) > 0.05 and ptolemy["learned_reliability"] > legend["learned_reliability"],
          f"scale={ptolemy['learned_scale']:.3f} reliability={ptolemy['learned_reliability']:.3f}")

    true_noise = np.array([s["noise"] for s in SOURCES])   # lower noise = more trustworthy once corrected
    learned_by_source_idx = np.array([sigmoid(params["reliability_logits"][s]) for s in range(len(SOURCES))])

    def _rank(a):
        return np.argsort(np.argsort(a))

    spearman = np.corrcoef(_rank(-true_noise), _rank(learned_by_source_idx))[0, 1]
    check("witness trust ordering broadly tracks true noise level "
          "(reliability, not raw pre-correction label, since systematic "
          "sources are corrected rather than merely trusted or distrusted)",
          spearman > 0.5, f"rank correlation={spearman:.3f}")

    check("climate-zone-gated local experts beat the ungated global baseline",
          mse_geo < mse_global,
          f"loo_mse_geometric={mse_geo:.4f} loo_mse_global={mse_global:.4f}")

    n_fail = sum(1 for _, status, _ in results if status == "FAIL")
    print(f"\n  {len(results) - n_fail}/{len(results)} self-tests passed.")
    assert n_fail == 0, f"{n_fail} self-test(s) FAILED"
    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 -- MAIN: build the world, gradient-check, train, judge, atlas
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print(" MIND 0240 -- AL-IDRISI: TRIANGULATED-ADDRESS WORLD MODEL")
    print("=" * 78)
    print(f"\nWorld: {N} places, {M} witnesses, {len(ALL_REPORTS)} testimony reports, "
          f"{len(HELD_OUT_PAIRS)} pairs held out entirely (no witness reports them).\n")

    bank = ReportBank(ALL_REPORTS)
    model = TriangulatedWorldModel(N, M, seed=42)

    print("[1] Finite-difference gradient check (mandatory, pre-training)")
    finite_difference_gradient_check(model, bank)

    print("\n[2] Training: alternating corroboration re-weighting (E-step) "
          "and Adam gradient step (M-step)")
    history = train(model, bank, epochs=900, lr=0.05, log_every=150)
    params = model.params

    print("\n[3] Geometry check: does the recovered disc resemble the true world?")
    corr, true_d, pred_d = all_pairs_distance_correlation(model, params)
    print(f"    Pearson correlation of all-pairs distances (true vs. recovered): {corr:.4f}")

    print("\n[4] Generalization: predicting routes no witness ever jointly reported")
    generalization = evaluate_generalization(model, params, HELD_OUT_PAIRS)
    if generalization:
        print(f"    held-out pairs: {generalization['n_pairs']}")
        print(f"    model MSE:      {generalization['mse_model']:.4f}")
        print(f"    baseline MSE:   {generalization['mse_baseline']:.4f}  "
              f"(predicting the mean distance for everything)")
        print(f"    improvement:    {generalization['improvement_ratio']:.2f}x")

    print("\n[5] The court's verdict on each witness (learned trust & correction)")
    judgments = report_witness_judgments(model, params)
    header = f"    {'witness':38s} {'learned_rel':>11s} {'true_rel':>9s} {'scale':>7s} {'offset':>7s}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for j in judgments:
        flag = " [systematic]" if j["systematic_source"] else ""
        print(f"    {j['name']:38s} {j['learned_reliability']:11.3f} "
              f"{j['true_reliability']:9.2f} {j['learned_scale']:7.3f} "
              f"{j['learned_offset']:7.3f}{flag}")

    print("\n[6a] Finite-difference gradient check for the local-expert module")
    _dummy_group = np.zeros(N, dtype=int)
    _dummy_feats = build_features(TRUE_COORDS, TRUE_ATTRS, _dummy_group)
    _fresh_expert = LinearExpert(_dummy_feats.shape[1], ATTR_DIM, seed=123)
    err = local_expert_gradient_check(_fresh_expert, _dummy_feats, TRUE_ATTRS)
    print(f"    local-expert gradient check: max relative error = {err:.2e} (tolerance 2.0e-03)")

    print("\n[6b] The atlas: hierarchical climate-zone gating (7 climates x 10 sections)")
    aligned, _, _ = procrustes_align(params["X"], TRUE_COORDS)
    cell_ids, band_ids, sect_ids = assign_climate_cells(aligned)
    occupied = sorted(set(cell_ids.tolist()))
    print(f"    {len(occupied)} of {N_BANDS * N_SECTIONS} possible climate/section cells occupied "
          f"by {N} places.")
    for c in occupied[:8]:
        members = [PLACE_NAMES[i] for i in range(N) if cell_ids[i] == c]
        band, sect = divmod(c, N_SECTIONS)
        print(f"      climate {band + 1:>1d}, section {sect + 1:>2d}: {', '.join(members)}")
    if len(occupied) > 8:
        print(f"      ... and {len(occupied) - 8} more cells")

    print("\n[7] Ablation: does climate-zone gating actually help predict regional "
          "attributes, leave-one-out? (evaluated at climate-band resolution -- "
          "7 groups -- the level at which enough corroborating testimony "
          "actually exists; the 70-cell grid above remains the address space)")
    band_only_ids = assign_climate_band(aligned)
    mse_geo = loo_cv_mse_geometric(aligned, TRUE_ATTRS, band_only_ids, min_cell_size=3)
    mse_global = loo_cv_mse_global_baseline(aligned, TRUE_ATTRS)
    print(f"    LOO-CV MSE, climate-band-gated local experts: {mse_geo:.4f}")
    print(f"    LOO-CV MSE, single ungated global expert:     {mse_global:.4f}")
    if mse_global > 0:
        print(f"    gating improves prediction by {(1 - mse_geo / mse_global) * 100:.1f}%")

    print("\n[8] Self-tests")
    run_self_tests(model, params, history, bank, judgments, generalization,
                    corr, mse_geo, mse_global)

    print("\n" + "=" * 78)
    print(" ALL SELF-TESTS PASSED")
    print("=" * 78)

    return dict(model=model, params=params, history=history, judgments=judgments,
                generalization=generalization, corr=corr, mse_geo=mse_geo,
                mse_global=mse_global)


if __name__ == "__main__":
    main()

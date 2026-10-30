#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0281 · Urraca I of León
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 15, Minds 281-300 Available on Amazon https://www.amazon.com/dp/B0HMQ7TVJM
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0281_urraca_i_of_leon_1081 - Urraca I of León (c. 1081-1126)
# END ATTRIBUTION
"""
THE RESUMABLE LEASE  —  chapter 0281 · Urraca I of León (c. 1081-1126)

Thesis
------
For this mind a grant is never an alienation: every fief, office or bond is a
lease that carries a reserved right of resumption, so the realm is held not by
fixed structure but by continually leasing its parts to whoever will hold them
now and taking them back when they will not.

Evidence and provenance
-----------------------
Provenance is EXTRAPOLATED. No philosophy of mind survives from the queen; her
118+ diplomas are chancery products, and the fullest narrative (the Historia
Compostelana) is a hostile episcopal gesta. The mechanism is inferred from
documented deeds, not from doctrine:
  D1  December 1109: she grants her husband Alfonso I "the whole land that was
      her father's" and in the same act forbids him to seek annulment on grounds
      of kinship or excommunication, with reversion to her son if childless
      (Gordo Molina & Melo Carrasco 2018, 50-52; Pallares & Portela 2006, 43).
      By 1110-1112 the bond is dissolved on exactly the forbidden ground.
  D2  1116 accord at Sahagún: the realm divided into two spheres for a fixed
      term of three years (Reilly 1982, 114).
  D3  1115/16-1120: the "dominium of all Galicia" given to Diego Gelmírez with
      the principes ordered to do homage to him "saving their fealty to the
      queen"; summer 1121: she has him arrested (Historia Compostelana, trans.
      Falque 1994, 244; Reilly 1982).
  D4  1120: Munio Peláez imprisoned and stripped of all his lands; 1121:
      restored (Reilly 1982, 289).
  D5  1111-1124: about twenty diplomas, 15 % of her known chancery output and a
      type of document unique to her reign, record church treasure requisitioned,
      weighed and melted "to pay my knights", compensated with grants of land
      (Ruiz Albi 2003; Martínez Sopena 2015).
  D6  The Historia Compostelana explains the same record as feminine
      changeability (Reilly 1982, 47; Falque 1994) - the trust cost of D1-D5.

Doctrine → mechanism → test
---------------------------
  D1, D3   M1 lease register with resumption (grant / revoke / re-lease)   C6.1, H-SIG, H-RIVAL
  D5       M2 melt on resumption (re-initialise resumed units, cut old links) C6.2, H-NEC
  D3, D4   M3 reserved fraction ρ (escrow kept on resumption, returned on re-grant)  C6.3, H-ESCROW
  D2       term bookkeeping in the register (lease age, expiry flag)          C6.1
  D4, D6   blind spot: reinstatement after melt; trust cost of resumption     H-BLIND

Research question
-----------------
Category: memory, retrieval and forgetting / continual learning and plasticity.
Question: when a stream of tasks includes revocations (a task must be unlearned
and its capacity reused by later tasks), can unlearning be made a register
operation on leased capacity - exact, plasticity-restoring, with partial
cross-task reuse - instead of a gradient operation, and what does that cost when
a revoked task returns?  This is the "continual unlearning" setting in which
gradient-based unlearning is reported to lose plasticity and re-memorise
(Shi, Xu, Ding, Hero & Wang, arXiv:2608.29513, 2026).

Closest prior art and the delta
-------------------------------
Parameter-isolation continual learning (PackNet, Mallya & Lazebnik 2018; HAT,
Serra et al. 2018) allocates capacity permanently. Continual Backprop (Dohare
et al. 2024) re-initialises low-utility units but has no ownership. SISA
(Bourtoule et al. 2021) unlearns by retraining isolated shards. Gradient-ascent
unlearning (the rival implemented here) burns the forget set inside a shared
network. Delta: capacity is OWNED by a holder for a term, resumed on revocation
with a reserved fraction kept in escrow, melted (re-initialised, old links cut)
and re-leased. Unlearning becomes a register write; reuse is exact.

Blind spot
----------
Melting destroys what a resumed unit knew. A revoked holder that is reinstated
(Munio Peláez, 1120-1121) gets back only its escrow plus whatever the pool has
left, and retained holders that had read the resumed units lose that support -
the cost the Historia Compostelana recorded as inconstancy.

Task
----
Synthetic "fiefs": vector classification, 3 classes, 16 inputs. Each fief draws
a private orthonormal 3-D basis B, thresholds t and a random table from the 8
orthants of u = Bᵀx - t to the 3 classes, so no single input feature and no
linear readout over foreign features carries the label. Nuisance: per-fief
affine jitter, a region shift of up to ±2 per input, isotropic noise. Shifted
split: B rotated by 20° inside its span, noise ×2. Stream: grant 1-4 · revoke
2, 3 · grant 5-7 · reinstate 2 (small budget). Hidden width 40, lease size 8:
the pool holds four fiefs, so fiefs 5-7 exist only if capacity is resumed.
Pool and escrow units yield nothing to any readout; a grant cuts every other
readout's link into the leased units.

Limits
------
Task-incremental (holder id given at test time); single hidden layer; the
register is exogenous (revocations arrive as events, not decisions); no claim
that any historical person is replicated. A research prototype, not an AGI.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "1.0", "generator": "Claude",
                   "generator_version": "Fable 5.1", "date": "2026-09-25"},
    "id": 281, "figure": "Urraca I of León", "born": 1081, "died": 1126,
    "civilization": "Leonese-Castilian",
    "provenance": "extrapolated",
    "thesis": ("Every grant is a lease with a reserved right of resumption: the realm is "
               "held by continually leasing its parts and taking them back, never by "
               "alienating them."),
    "evidence": [
        {"id": "D1", "claim": "Dec. 1109 grant of the whole realm to Alfonso I with a clause "
                              "forbidding annulment on kinship; bond dissolved on that ground by 1112",
         "basis": "scholarship", "source": "Gordo Molina & Melo Carrasco 2018, 50-52; Pallares & Portela 2006, 43-46"},
        {"id": "D2", "claim": "1116 Sahagún accord dividing the realm for a fixed three-year term",
         "basis": "scholarship", "source": "Reilly 1982, 114"},
        {"id": "D3", "claim": "dominium of all Galicia granted to Gelmírez with homage 'saving fealty "
                              "to the queen' (c. 1120); Gelmírez arrested summer 1121",
         "basis": "primary", "source": "Historia Compostelana II (Falque 1994, 244); Reilly 1982"},
        {"id": "D4", "claim": "Munio Peláez stripped of all lands 1120, restored 1121",
         "basis": "scholarship", "source": "Reilly 1982, 289"},
        {"id": "D5", "claim": "c. 20 diplomas (15% of output, a type unique to her reign) record church "
                              "treasure melted 'ut reddam donativa militibus meis', compensated in land",
         "basis": "primary", "source": "Ruiz Albi 2003, docs 27-35, 57-141; Martínez Sopena 2015, 128-132"},
        {"id": "D6", "claim": "the Historia Compostelana reads the same record as feminine changeability",
         "basis": "primary", "source": "Historia Compostelana (Falque 1994); Reilly 1982, 47"},
    ],
    "research_question": {
        "category": "memory, retrieval and forgetting",
        "question": ("Can unlearning of a revoked task be a register operation on leased capacity "
                     "(exact, plasticity-restoring, partially reusable) rather than a gradient "
                     "operation, and what does it cost when the task is reinstated?")},
    "mechanism": {
        "name": "resumable lease network",
        "family": "parameter-isolation continual learning with ownership register",
        "signature_modules": ["lease_register", "melt", "reserved_fraction"],
        "closest_prior_art": ["PackNet (Mallya & Lazebnik, CVPR 2018)",
                              "HAT (Serra et al., ICML 2018)",
                              "Continual Backprop (Dohare et al., Nature 2024)",
                              "SISA (Bourtoule et al., IEEE S&P 2021)",
                              "Unified continual learning and unlearning (Chatterjee et al., arXiv:2408.11374)"],
        "overlap": "Medium",
        "prior_art_queries": ["continual learning task removal reclaim capacity",
                              "continual machine unlearning plasticity collapse",
                              "parameter isolation continual learning mask reuse unlearning",
                              "PackNet capacity saturation new task"],
        "contribution_type": "mechanism",
        "delta": ("capacity is owned per holder for a term, resumed on revocation, melted and re-leased, "
                  "with non-owners reading every fief at a fixed reserved fraction"),
    },
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M1", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M3", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "M2", "property_test": "C6.2", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "After revocations, a lease network serves the valid fiefs better "
                                     "than a size-matched shared MLP that forgets",
         "metric": "end_acc (mean held-out accuracy over valid fiefs, shifted split too)",
         "split": "shifted", "comparison": "model - baseline", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing resumption hurts more than removing the hidden bias",
         "metric": "end_acc change", "comparison": "signature_knockout - matched_knockout",
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": "A revoked fief that is reinstated is learned worse after melting "
                                       "than under perpetual ('in perpetuum') allocation, the lease-register knockout",
         "condition": "reinstatement of fief 2 with a small budget",
         "grounding": "Munio Peláez stripped 1120, restored 1121; the Compostelan charge of inconstancy",
         "metric": "reinstate_acc", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Unlearning by register write beats unlearning by gradient ascent on the "
                                       "same shared network across repeated revocations",
         "metric": "end_acc", "comparison": "model - rival", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
        {"id": "H-ESCROW", "statement": "Keeping a reserved fraction of a resumed fief in escrow improves its "
                                        "reinstatement over a full melt",
         "metric": "reinstate_acc", "comparison": "model - reserved_fraction_knockout", "direction": "greater",
         "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.20,
                   "shuffled_band": 0.10, "gradcheck_rel_error": 1e-5},
    "probe_predictions": [
        {"probe": "P5", "expected": "above baseline"},
        {"probe": "P6", "expected": "equal to baseline"},
        {"probe": "P9", "expected": "equal to baseline"},
        {"probe": "P10", "expected": "below baseline"},
    ],
    "dialectic_links": [
        {"chapter": 271, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 268, "relation": "rival", "test": "H-RIVAL"},
        {"chapter": 297, "relation": "successor", "test": "H-BLIND"},
    ],
    "corpus_neighbors": [
        {"chapter": 271, "similarity": 0.20, "difference": "William engineers commitments that never relax; "
                                                            "here every commitment is built to be resumed"},
        {"chapter": 297, "similarity": 0.18, "difference": "Eleanor's competence is a portable core re-hosted "
                                                            "intact; Urraca's periphery is leased and melted"},
        {"chapter": 268, "similarity": 0.15, "difference": "Gregory VII's authority graph has one unaudited root; "
                                                            "Urraca delegates whole dominia and keeps only a reserved fraction"},
        {"chapter": 266, "similarity": 0.10, "difference": "Sima Guang ranks agents by worth; this file ranks nothing, "
                                                            "it tracks tenure"},
        {"chapter": 256, "similarity": 0.10, "difference": "Yaroslav's enumerated kinship list is closed; the lease "
                                                            "register is open and re-writable"},
    ],
    "barometer": {
        "cognitive_processing": ["learning efficiency and transfer via reserved-fraction reuse of foreign fiefs"],
        "embodied_cognition": [],
        "world_modeling": ["social system: a register of holders, terms and reversions"],
        "consciousness": [],
        "language_understanding": [],
        "emotional_intelligence": [],
        "creativity": [],
        "autonomy": ["self-modification: structural resumption, melt and re-lease of capacity",
                     "adaptive obstacle management: new claimants served after the pool is exhausted"],
    },
    "task_types": ["vector_classification"],
    "applications": [
        {"use": "sequential data-deletion requests in a deployed classifier (right-to-erasure)",
         "sector": "privacy and compliance",
         "dataset": "NeurIPS 2023 Machine Unlearning competition split of CIFAR-10"},
        {"use": "multi-tenant model serving where tenants join and leave",
         "sector": "cloud ML platforms", "dataset": "FEMNIST (LEAF benchmark)"},
        {"use": "retiring and replacing skills in a robot skill library",
         "sector": "robotics", "dataset": "Continual World (Meta-World task sequences)"},
    ],
    "safety_notes": ("Political-control figure: the mechanism is framed as capacity governance and "
                     "data deletion; no targeting, surveillance or opinion manipulation. Synthetic data only."),
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CHAPTER = 281
FILE_NAME = os.path.basename(__file__) if "__file__" in globals() else "chapter_0281_urraca_i_of_leon_1081.py"
IN_DIM, N_CLASSES, HIDDEN, LEASE = 16, 3, 40, 8
RHO = 0.25                      # reserved fraction: share of a resumed fief the crown keeps in escrow
N_HOLDERS = 8                   # heads 1..7; slot 0 is unused (the crown pool has no head)
STEPS_PER_FIEF, REINSTATE_STEPS, UNLEARN_STEPS, BATCH = 800, 150, 150, 48
LR, CLIP_NORM = 1e-2, 5.0
N_TRAIN, N_EVAL = 960, 400
PROTOCOL = [("grant", 1), ("grant", 2), ("grant", 3), ("grant", 4),
            ("revoke", 2), ("revoke", 3),
            ("grant", 5), ("grant", 6), ("grant", 7),
            ("reinstate", 2)]
VALID_AT_END = (1, 4, 5, 6, 7)
BUDGET_FULL, BUDGET_QUICK = 180.0, 20.0
TASK_TYPES = ["vector_classification"]


# BEGIN STANDARD UTILITIES v1.0
def logsumexp(z, axis=-1, keepdims=False):
    m = np.max(z, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(z, axis=-1):
    return np.exp(z - logsumexp(z, axis=axis, keepdims=True))


def softplus(x):
    return np.where(x > 30.0, x, np.log1p(np.exp(np.minimum(x, 30.0))))


class Adam:
    """Adam with optional per-tensor update masks (masked entries keep their value)."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, masks=None):
        self.t += 1
        c1, c2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            upd = self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if masks is not None and k in masks:
                upd = upd * masks[k]
            params[k] = params[k] - upd

    def reset_entries(self, key, index):
        self.m[key][index] = 0.0
        self.v[key][index] = 0.0


def clip_global_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        return {k: g * scale for k, g in grads.items()}, total
    return grads, total


def finite_difference_check(loss_fn, params, grads, rng, n_random=20, eps=1e-6, tol=1e-5):
    """Central differences on n_random entries plus the largest-gradient entry of every tensor.
    Returns (passed, max_rel_error, n_tensors)."""
    worst = 0.0
    for k in sorted(params):
        p, g = params[k], grads[k]
        flat = p.reshape(-1)
        idx = set(int(i) for i in rng.integers(0, flat.size, size=min(n_random, flat.size)))
        idx.add(int(np.argmax(np.abs(g.reshape(-1)))))
        for i in sorted(idx):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn(params)
            flat[i] = old - eps
            lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2.0 * eps)
            ana = g.reshape(-1)[i]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-7)
            worst = max(worst, rel)
    return worst <= tol, worst, len(params)


def paired_bootstrap(diffs, rng, n_resample=2000, alpha=0.05):
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return float(d.mean()) if d.size else 0.0, (float("nan"), float("nan"))
    means = np.array([d[rng.integers(0, d.size, size=d.size)].mean() for _ in range(n_resample)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), (float(lo), float(hi))


def verdict(mean_diff, ci, direction, mesi):
    lo, hi = ci
    if any(map(math.isnan, (lo, hi))):
        return "not evaluated"
    if direction == "greater":
        if lo > 0 and mean_diff >= mesi:
            return "supported"
        if hi < 0:
            return "contradicted"
    else:
        if hi < 0 and -mean_diff >= mesi:
            return "supported"
        if lo > 0:
            return "contradicted"
    return "inconclusive"


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
# END STANDARD UTILITIES


# ----------------------------------------------------------------------------
# Data and tasks: the fiefs
# ----------------------------------------------------------------------------
def make_fief(rng):
    """A fief: a private 3-D subspace of the input, an orthant-to-class table, and its own region."""
    basis, _ = np.linalg.qr(rng.standard_normal((IN_DIM, 3)))
    table = rng.integers(0, N_CLASSES, size=8)
    while len(set(table.tolist())) < N_CLASSES:
        table = rng.integers(0, N_CLASSES, size=8)
    return {"basis": basis, "table": table, "thr": rng.uniform(-0.3, 0.3, size=3),
            "scale": rng.uniform(0.7, 1.4, size=IN_DIM), "shift": rng.uniform(-2.0, 2.0, size=IN_DIM)}


def _rotate_in_span(basis, rng, degrees):
    a, b = basis[:, 0].copy(), basis[:, 1].copy()
    th = math.radians(degrees) * rng.choice([-1.0, 1.0])
    out = basis.copy()
    out[:, 0] = math.cos(th) * a + math.sin(th) * b
    out[:, 1] = -math.sin(th) * a + math.cos(th) * b
    return out


def sample_fief(fief, rng, n, shifted=False):
    """Labels come from which orthant of the mixed 3-D projection the point falls in."""
    basis = _rotate_in_span(fief["basis"], rng, 20.0) if shifted else fief["basis"]
    noise = 0.10 * (2.0 if shifted else 1.0)
    z = rng.standard_normal((n, IN_DIM))
    u = z @ basis - fief["thr"]
    code = (u > 0).astype(int) @ np.array([1, 2, 4])
    y = fief["table"][code]
    x = (z + noise * rng.standard_normal((n, IN_DIM))) * fief["scale"] + fief["shift"]
    return x, y


def build_stream(rng, n_train=N_TRAIN, n_eval=N_EVAL, n_fiefs=7):
    """Per-fief train / held-out / shifted splits, keyed by holder id 1..n_fiefs."""
    data = {}
    for f in range(1, n_fiefs + 1):
        fief = make_fief(rng)
        xtr, ytr = sample_fief(fief, rng, n_train)
        xte, yte = sample_fief(fief, rng, n_eval)
        xsh, ysh = sample_fief(fief, rng, n_eval, shifted=True)
        data[f] = {"train": (xtr, ytr), "heldout": (xte, yte), "shifted": (xsh, ysh)}
    return data


def shuffle_labels(data, rng):
    out = {}
    for f, d in data.items():
        xtr, ytr = d["train"]
        out[f] = dict(d)
        out[f]["train"] = (xtr, ytr[rng.permutation(ytr.size)])
    return out


def trivial_accuracy(data, split="heldout", fiefs=VALID_AT_END):
    accs = []
    for f in fiefs:
        if f in data:
            y = data[f][split][1]
            accs.append(np.bincount(y, minlength=N_CLASSES).max() / y.size)
    return float(np.mean(accs))


def load_csv_stream(path):
    """Optional real-data bridge: CSV with feature columns then 'task' and 'label'."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x, task, y = raw[:, :-2], raw[:, -2].astype(int), raw[:, -1].astype(int)
    data = {}
    for i, t in enumerate(sorted(set(task.tolist()))[:7]):
        m = task == t
        xs, ys = x[m], y[m]
        cut = int(0.7 * xs.shape[0])
        data[i + 1] = {"train": (xs[:cut], ys[:cut]), "heldout": (xs[cut:], ys[cut:]),
                       "shifted": (xs[cut:] + 0.3 * xs[cut:].std(), ys[cut:])}
    return data


# ----------------------------------------------------------------------------
# Model: the resumable lease network
# ----------------------------------------------------------------------------
class LeaseRegister:
    """The chancery. holder[u] = 0 crown pool (frozen reserve), h > 0 leased to h, -h held in escrow for h."""

    def __init__(self, hidden, lease, resumable=True, rho=RHO):
        self.holder = np.zeros(hidden, dtype=int)
        self.age = np.zeros(hidden, dtype=int)
        self.lease, self.resumable, self.rho = lease, resumable, rho
        self.events = []

    def free_units(self):
        return np.flatnonzero(self.holder == 0)

    def grant(self, holder):
        """A grant returns any escrow of this holder first, then fills from the pool up to the lease size."""
        escrow = np.flatnonzero(self.holder == -holder)
        free = self.free_units()[: max(self.lease - escrow.size, 0)]
        units = np.concatenate([escrow, free])
        self.holder[units] = holder
        self.age[units] = 0
        self.events.append(("grant", holder, int(units.size)))
        return units

    def resume(self, holder):
        """Resumption: a reserved fraction rho of the fief stays in escrow, the rest returns to the pool."""
        held = np.flatnonzero(self.holder == holder)
        if not self.resumable:
            self.events.append(("resume", holder, 0))
            return held[:0], held[:0]
        n_escrow = int(math.ceil(self.rho * held.size))
        escrow, pooled = held[:n_escrow], held[n_escrow:]
        self.holder[escrow] = -holder
        self.holder[pooled] = 0
        self.age[held] = 0
        self.events.append(("resume", holder, int(pooled.size)))
        return pooled, escrow

    def tick(self, steps=1):
        self.age[self.holder > 0] += steps

    def update_mask(self, holder):
        return (self.holder == holder).astype(np.float64)

    def gains(self):
        """Undistributed (pool) and escrowed units yield nothing to any head."""
        return (self.holder > 0).astype(np.float64)


def init_params(in_dim, hidden, out_dim, n_holders, rng):
    return {
        "W1": rng.standard_normal((in_dim, hidden)) / math.sqrt(in_dim),
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((n_holders, hidden, out_dim)) / math.sqrt(hidden),
        "b2": np.zeros((n_holders, out_dim)),
    }


def build_model(in_dim, out_dim, task_type, rng, hidden=HIDDEN, lease=LEASE, rho=RHO,
                n_holders=N_HOLDERS, resumable=True, melt=True, **cfg):
    if task_type not in TASK_TYPES:
        raise ValueError("unsupported task type %s" % task_type)
    return {
        "params": init_params(in_dim, hidden, out_dim, n_holders, rng),
        "register": LeaseRegister(hidden, lease, resumable=resumable, rho=rho),
        "melt": melt, "shared": bool(cfg.get("shared", False)), "unlearn": cfg.get("unlearn", None),
        "rng": np.random.default_rng(rng.integers(0, 2 ** 31)),
        "active": set(), "bias_on": 1.0,
    }


def forward(params, x, holder, gains, bias_on=1.0):
    pre = x @ params["W1"] + bias_on * params["b1"]
    h = np.tanh(pre) * gains
    return pre, h, h @ params["W2"][holder] + params["b2"][holder]


def loss_and_grads(model, batch):
    """Exact gradient of the cross-entropy; the lease mask acts at update time, not here."""
    x, y, holder = batch
    params, gains = model["params"], model["register"].gains()
    pre, h, logits = forward(params, x, holder, gains, model["bias_on"])
    p = softmax(logits)
    n = x.shape[0]
    loss = -float(np.mean(np.log(p[np.arange(n), y] + 1e-300)))
    dlog = p.copy()
    dlog[np.arange(n), y] -= 1.0
    dlog /= n
    g = {k: np.zeros_like(v) for k, v in params.items()}
    g["W2"][holder] = h.T @ dlog
    g["b2"][holder] = dlog.sum(axis=0)
    dpre = (dlog @ params["W2"][holder].T) * gains * (1.0 - np.tanh(pre) ** 2)
    g["W1"] = x.T @ dpre
    g["b1"] = model["bias_on"] * dpre.sum(axis=0)
    return loss, g


def predict(model, X, holder=1):
    return np.argmax(forward(model["params"], X, holder, model["register"].gains(), model["bias_on"])[2], axis=1)


def hidden_states(model, X, holder=1):
    return forward(model["params"], X, holder, model["register"].gains(), model["bias_on"])[1]


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


def accuracy(model, data, holder, split="heldout"):
    x, y = data[holder][split]
    return float(np.mean(predict(model, x, holder) == y))


# --- register operations (the queen's acts) --------------------------------
def grant(model, holder):
    """A grant: the holder receives its units; every other head's link into them is cut."""
    reg = model["register"]
    if model["shared"]:
        reg.holder[:] = holder
        model["active"].add(holder)
        return np.arange(reg.holder.size)
    units = reg.grant(holder)
    others = [h for h in range(model["params"]["W2"].shape[0]) if h != holder]
    model["params"]["W2"][np.ix_(others, units)] = 0.0
    fresh = units[np.all(model["params"]["W2"][holder, units, :] == 0.0, axis=1)]
    if fresh.size:
        model["params"]["W2"][holder, fresh, :] = model["rng"].standard_normal((fresh.size, N_CLASSES)) / math.sqrt(HIDDEN)
    model["active"].add(holder)
    return units


def melt(model, units, opt=None):
    """Resumed units are re-minted: fresh incoming weights, every head's link to them cut."""
    if units.size == 0:
        return
    p = model["params"]
    p["W1"][:, units] = model["rng"].standard_normal((p["W1"].shape[0], units.size)) / math.sqrt(p["W1"].shape[0])
    p["b1"][units] = 0.0
    p["W2"][:, units, :] = 0.0
    if opt is not None:
        opt.reset_entries("W1", (slice(None), units))
        opt.reset_entries("b1", units)
        opt.reset_entries("W2", (slice(None), units, slice(None)))


def revoke(model, holder, data=None, rng=None, opt=None):
    """A bond is annulled: the head is retired, owned units are resumed (escrow kept, the rest melted)."""
    model["active"].discard(holder)
    if model["shared"]:
        if model["unlearn"] == "neggrad" and data is not None:
            gradient_unlearn(model, data, holder, rng, opt)
        return np.zeros(0, dtype=int)
    pooled, _ = model["register"].resume(holder)
    if model["melt"]:
        melt(model, pooled, opt)
    return pooled


# ----------------------------------------------------------------------------
# Baselines and rival mechanisms
# ----------------------------------------------------------------------------
def build_plain_mlp(in_dim, out_dim, rng, **cfg):
    """Size-matched shared network: every holder trains every hidden unit (naive sequential fine-tuning)."""
    return build_model(in_dim, out_dim, "vector_classification", rng, shared=True,
                       resumable=False, melt=False, **cfg)


def build_neggrad_rival(in_dim, out_dim, rng, **cfg):
    """The modern rival: the same shared network, unlearning a revoked fief by gradient ascent on its data."""
    return build_model(in_dim, out_dim, "vector_classification", rng, shared=True,
                       resumable=False, melt=False, unlearn="neggrad", **cfg)


def gradient_unlearn(model, data, holder, rng, opt, max_steps=UNLEARN_STEPS):
    """Gradient ascent on the forget set until it is at chance, with the update norm clipped."""
    x, y = data[holder]["train"]
    for _ in range(max_steps):
        idx = rng.integers(0, x.shape[0], size=BATCH)
        _, g = loss_and_grads(model, (x[idx], y[idx], holder))
        g, _ = clip_global_norm(g, CLIP_NORM)
        opt.step(model["params"], {k: -v for k, v in g.items()})
        if np.mean(predict(model, x[:200], holder) == y[:200]) <= 1.0 / N_CLASSES + 0.05:
            break


# ----------------------------------------------------------------------------
# Registries: modules, knockouts, mutants
# ----------------------------------------------------------------------------
def modules(model):
    return {
        "lease_register": {"params": [], "signature": True,
                           "role": "ownership register: masks W1 column updates to the owning holder; resumes columns on revocation"},
        "melt": {"params": ["W1", "b1", "W2"], "signature": True,
                 "role": "re-initialisation of resumed columns and zeroing of every readout row into them"},
        "reserved_fraction": {"params": [], "signature": True,
                              "role": "fraction rho of a resumed fief kept frozen in escrow and returned on reinstatement"},
        "hidden_bias": {"params": ["b1"], "signature": False, "role": "additive hidden bias"},
    }


def knockout(model, name, mode="identity"):
    """Return a fresh configuration with one module disabled; used before training a copy."""
    cfg = {"resumable": model["register"].resumable, "melt": model["melt"], "rho": model["rho"], "bias_on": 1.0}
    if name == "lease_register":
        cfg["resumable"] = False
    elif name == "melt":
        cfg["melt"] = False
    elif name == "reserved_fraction":
        cfg["rho"] = 0.0
    elif name == "hidden_bias":
        cfg["bias_on"] = 0.0
    else:
        raise KeyError(name)
    return cfg


MUTANTS = {
    "sign_flip": "update adds the gradient instead of subtracting it",
    "zero_lr": "learning rate set to zero",
    "grad_zero_W1": "analytic gradient of W1 replaced by zeros",
    "no_lease_mask": "update mask removed: holders overwrite foreign fiefs",
}


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_holder(model, data, holder, steps, rng, opt, mutant=None, log=None):
    x, y = data[holder]["train"]
    lr_sign = -1.0 if mutant == "sign_flip" else 1.0
    if mutant == "zero_lr":
        opt.lr = 0.0
    for _ in range(steps):
        idx = rng.integers(0, x.shape[0], size=BATCH)
        loss, g = loss_and_grads(model, (x[idx], y[idx], holder))
        if mutant == "grad_zero_W1":
            g["W1"] = np.zeros_like(g["W1"])
        g, _ = clip_global_norm(g, CLIP_NORM)
        g = {k: lr_sign * v for k, v in g.items()}
        masks = None if (model["shared"] or mutant == "no_lease_mask") else {"W1": model["register"].update_mask(holder)}
        opt.step(model["params"], g, masks)
        model["register"].tick()
        if log is not None:
            log.append(loss)
    return model


def run_protocol(model, data, rng, protocol=PROTOCOL, steps=STEPS_PER_FIEF, mutant=None,
                 reinstate_steps=REINSTATE_STEPS):
    """The reign: grants, revocations, new grants, one reinstatement. Returns metrics."""
    opt = Adam(model["params"], lr=LR)
    losses = {}
    for act, holder in protocol:
        if act == "grant":
            grant(model, holder)
            losses[holder] = []
            train_holder(model, data, holder, steps, rng, opt, mutant, losses[holder])
        elif act == "revoke":
            revoke(model, holder, data, rng, opt)
        elif act == "reinstate":
            grant(model, holder)
            losses[("re", holder)] = []
            train_holder(model, data, holder, reinstate_steps, rng, opt, mutant, losses[("re", holder)])
    return losses


def evaluate(model, data, split="heldout"):
    valid = [f for f in VALID_AT_END if f in data]
    end = float(np.mean([accuracy(model, data, f, split) for f in valid]))
    retained = float(np.mean([accuracy(model, data, f, split) for f in (1, 4) if f in data]))
    new = float(np.mean([accuracy(model, data, f, split) for f in (5, 6, 7) if f in data]))
    return {"end_acc": end, "retained_acc": retained, "new_acc": new}


def linear_probe_accuracy(model, data, holder, rng):
    """Ridge probe on hidden states of a revoked fief: how much of it the realm still carries."""
    xtr, ytr = data[holder]["train"]
    xte, yte = data[holder]["heldout"]
    htr = hidden_states(model, xtr, holder)
    hte = hidden_states(model, xte, holder)
    onehot = np.eye(N_CLASSES)[ytr]
    w = np.linalg.solve(htr.T @ htr + 1.0 * np.eye(htr.shape[1]), htr.T @ onehot)
    return float(np.mean(np.argmax(hte @ w, axis=1) == yte))


def run_one(cfg, data, seed, mutant=None, quick=False):
    """Train one configuration through the full protocol and collect all metrics."""
    rng = np.random.default_rng(seed)
    steps = 450 if quick else STEPS_PER_FIEF
    if cfg["kind"] == "plain":
        model = build_plain_mlp(IN_DIM, N_CLASSES, rng)
    elif cfg["kind"] == "neggrad":
        model = build_neggrad_rival(IN_DIM, N_CLASSES, rng)
    else:
        model = build_model(IN_DIM, N_CLASSES, "vector_classification", rng,
                            resumable=cfg.get("resumable", True), melt=cfg.get("melt", True),
                            rho=cfg.get("rho", RHO))
        model["bias_on"] = cfg.get("bias_on", 1.0)
    pre = [a for a in PROTOCOL if a[0] != "reinstate"]
    losses = run_protocol(model, data, rng, protocol=pre, steps=steps, mutant=mutant)
    metrics = evaluate(model, data, "heldout")
    metrics.update({k + "_shifted": v for k, v in evaluate(model, data, "shifted").items()})
    metrics["revoked_probe"] = linear_probe_accuracy(model, data, 3, rng)
    fresh = build_model(IN_DIM, N_CLASSES, "vector_classification", np.random.default_rng(seed))
    metrics["probe_floor"] = linear_probe_accuracy(fresh, data, 3, rng)
    post = [a for a in PROTOCOL if a[0] == "reinstate"]
    run_protocol(model, data, rng, protocol=post, steps=steps, mutant=mutant,
                 reinstate_steps=60 if quick else REINSTATE_STEPS)
    metrics["reinstate_acc"] = accuracy(model, data, 2, "heldout")
    metrics["losses"] = losses
    metrics["model"] = model
    return metrics


# ----------------------------------------------------------------------------
# Tests: correctness first
# ----------------------------------------------------------------------------
class NonFinite(Exception):
    pass


def _check_finite(model):
    for v in model["params"].values():
        if not np.all(np.isfinite(v)):
            raise NonFinite("non-finite parameter")


def _gradcheck_model(model, data, holder, rng):
    x, y = data[holder]["train"]
    idx = rng.integers(0, x.shape[0], size=32)
    batch = (x[idx], y[idx], holder)
    if np.any(np.abs(x[idx] @ model["params"]["W1"] + model["params"]["b1"]) < 1e-9):
        idx = rng.integers(0, x.shape[0], size=32)   # tanh is smooth; this only avoids an exactly-zero pre-activation
        batch = (x[idx], y[idx], holder)
    _, g = loss_and_grads(model, batch)

    def loss_fn(params):
        saved = model["params"]
        model["params"] = params
        try:
            return loss_and_grads(model, batch)[0]
        finally:
            model["params"] = saved
    return finite_difference_check(loss_fn, model["params"], g, rng,
                                   tol=MIND_CARD["thresholds"]["gradcheck_rel_error"])


def test_c1_gradcheck(data, seed, mutant=None):
    rng = np.random.default_rng(seed)
    model = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
    grant(model, 1)
    ok0, err0, n0 = _gradcheck_model(model, data, 1, rng)
    opt = Adam(model["params"], lr=LR)
    train_holder(model, data, 1, 60, rng, opt)
    grant(model, 2)                                  # a second holder makes the reserved fraction active
    train_holder(model, data, 2, 60, rng, opt)
    ok1, err1, n1 = _gradcheck_model(model, data, 2, rng)
    if mutant == "grad_zero_W1":
        _, g = loss_and_grads(model, (data[2]["train"][0][:32], data[2]["train"][1][:32], 2))
        g["W1"] = np.zeros_like(g["W1"])
        x, y = data[2]["train"]

        def loss_fn(params):
            saved = model["params"]
            model["params"] = params
            try:
                return loss_and_grads(model, (x[:32], y[:32], 2))[0]
            finally:
                model["params"] = saved
        ok1, err1, n1 = finite_difference_check(loss_fn, model["params"], g, rng)
    return {"id": "C1", "name": "gradient_check", "passed": bool(ok0 and ok1),
            "detail": "init max_rel=%.2e, after training max_rel=%.2e, tensors=%d/%d" % (err0, err1, n1, len(model["params"])),
            "max_rel_error": float(max(err0, err1)), "tensors": n1}


def test_c2_determinism(data, seed):
    a = run_one({"kind": "full"}, data, seed, quick=True)
    b = run_one({"kind": "full"}, data, seed, quick=True)
    same = np.allclose(a["losses"][1], b["losses"][1]) and a["end_acc"] == b["end_acc"]
    _check_finite(a["model"])
    finite = all(np.all(np.isfinite(hidden_states(a["model"], data[f]["heldout"][0], f))) for f in data)
    return {"id": "C2", "name": "determinism_and_finiteness", "passed": bool(same and finite),
            "detail": "identical losses=%s, finite=%s" % (same, finite)}


def test_c3_learning(result, data):
    lf = result["losses"][1]
    drop = 1.0 - float(np.mean(lf[-25:])) / max(float(np.mean(lf[:25])), 1e-12)
    margin = result["end_acc"] - trivial_accuracy(data)
    th = MIND_CARD["thresholds"]
    passed = drop >= th["loss_drop_fraction"] and margin >= th["margin_over_trivial"]
    return {"id": "C3", "name": "learning", "passed": bool(passed),
            "detail": "loss drop %.2f (need %.2f); margin over trivial %.3f (need %.2f)"
                      % (drop, th["loss_drop_fraction"], margin, th["margin_over_trivial"])}


def test_c4_shuffled(data, seed, quick):
    shuffled = shuffle_labels(data, np.random.default_rng(seed + 7919))
    res = run_one({"kind": "full"}, shuffled, seed, quick=quick)
    band = MIND_CARD["thresholds"]["shuffled_band"]
    triv = trivial_accuracy(shuffled)
    passed = abs(res["end_acc"] - triv) <= band
    return {"id": "C4", "name": "shuffled_label_control", "passed": bool(passed),
            "detail": "shuffled end_acc %.3f vs trivial %.3f (band ±%.2f)" % (res["end_acc"], triv, band)}


def test_c5_mutants(data, seed):
    detected, rows = 0, []
    for name in MUTANTS:
        caught = False
        if name == "grad_zero_W1":
            caught = not test_c1_gradcheck(data, seed, mutant=name)["passed"]
        elif name == "no_lease_mask":
            caught = not test_c6_isolation(seed, mutant=name)["passed"]
        else:
            res = run_one({"kind": "full"}, data, seed, mutant=name, quick=True)
            caught = not test_c3_learning(res, data)["passed"]
        detected += int(caught)
        rows.append("%s:%s" % (name, "detected" if caught else "MISSED"))
    return {"id": "C5", "name": "mutant_detection", "passed": detected == len(MUTANTS),
            "detail": ", ".join(rows), "detected": detected, "total": len(MUTANTS)}


def test_c6_conservation(seed, broken=False):
    """C6.1 capacity is conserved and no unit has two holders, on random grant/resume sequences."""
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(200):
        reg = LeaseRegister(HIDDEN, LEASE)
        counts = {}
        for _ in range(rng.integers(3, 15)):
            h = int(rng.integers(1, N_HOLDERS))
            if rng.random() < 0.6:
                units = reg.grant(h)
                if broken:                      # negative control: grant ignores ownership
                    extra = rng.integers(0, HIDDEN, size=2)
                    reg.holder[extra] = h
                    units = np.union1d(units, extra)
                counts[h] = counts.get(h, 0) + int(units.size)
            else:
                pooled, escrow = reg.resume(h)
                counts[h] = counts.get(h, 0) - int(pooled.size) - int(escrow.size)
        held = int(np.sum(reg.holder != 0))
        ok &= held + reg.free_units().size == HIDDEN
        ok &= all(int(np.sum(reg.holder == h)) == max(c, 0) for h, c in counts.items())
        ok &= int(np.sum(reg.holder < 0)) == sum(int(np.sum(reg.holder == -h)) for h in range(1, N_HOLDERS))
    return {"id": "C6.1", "name": "capacity_conservation", "passed": bool(ok),
            "detail": "200 random op sequences" + (" [negative control]" if broken else "")}


def test_c6_isolation(seed, mutant=None):
    """C6.2 an update by one holder never touches W1 columns owned by another holder."""
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(30):
        model = build_model(IN_DIM, N_CLASSES, "vector_classification", rng)
        for v in model["params"].values():
            v += rng.standard_normal(v.shape) * 0.5
        grant(model, 1)
        grant(model, 2)
        model["params"]["W2"] += rng.standard_normal(model["params"]["W2"].shape) * 0.5
        before = model["params"]["W1"].copy()
        opt = Adam(model["params"], lr=0.05)
        x = rng.standard_normal((BATCH, IN_DIM)) * 2.0
        y = rng.integers(0, N_CLASSES, size=BATCH)
        loss, g = loss_and_grads(model, (x, y, 2))
        masks = None if mutant == "no_lease_mask" else {"W1": model["register"].update_mask(2)}
        opt.step(model["params"], g, masks)
        foreign = np.flatnonzero(model["register"].holder == 1)
        ok &= bool(np.array_equal(before[:, foreign], model["params"]["W1"][:, foreign]))
        ok &= not np.array_equal(before, model["params"]["W1"])
    return {"id": "C6.2", "name": "fief_isolation", "passed": bool(ok),
            "detail": "30 random parameter draws, holder 2 updating" + (" [mask removed]" if mutant else "")}


def test_c6_definition(seed):
    """Labelled definition check (not evidence): escrow size is ceil(rho * lease) and is returned on re-grant."""
    reg = LeaseRegister(HIDDEN, LEASE)
    reg.grant(1)
    pooled, escrow = reg.resume(1)
    back = reg.grant(1)
    ok = escrow.size == math.ceil(RHO * LEASE) and pooled.size == LEASE - escrow.size and set(escrow) <= set(back)
    return {"id": "C6.3", "name": "escrow_definition_check", "passed": bool(ok),
            "detail": "definition check only; not counted as evidence"}


def test_c7_split_integrity(data):
    def keys(x):
        return {hashlib.sha1(np.round(row, 9).tobytes()).hexdigest() for row in x}
    leak = 0
    for d in data.values():
        tr = keys(d["train"][0])
        leak += len(tr & keys(d["heldout"][0])) + len(tr & keys(d["shifted"][0]))
    return {"id": "C7", "name": "split_integrity", "passed": leak == 0, "detail": "%d shared rows" % leak}


# ----------------------------------------------------------------------------
# Tests: hypotheses
# ----------------------------------------------------------------------------
CONFIGS = {
    "full": {"kind": "full"},
    "plain": {"kind": "plain"},
    "neggrad": {"kind": "neggrad"},
    "ko_lease_register": {"kind": "full", "resumable": False},
    "ko_melt": {"kind": "full", "melt": False},
    "ko_reserved_fraction": {"kind": "full", "rho": 0.0},
    "ko_hidden_bias": {"kind": "full", "bias_on": 0.0},
}


def run_seeds(seeds, quick):
    """All configurations over all seeds; each seed gets its own realm."""
    per_seed = []
    for s in seeds:
        data = build_stream(np.random.default_rng(s))
        row = {"data": data}
        for name, cfg in CONFIGS.items():
            r = run_one(cfg, data, s, quick=quick)
            row[name] = {k: v for k, v in r.items() if k not in ("model", "losses")}
            if name == "full":
                row["full_losses"] = r["losses"]
        per_seed.append(row)
    return per_seed


def hypothesis_table(per_seed, rng, quick):
    spec = {h["id"]: h for h in MIND_CARD["hypotheses"]}
    pairs = {
        "H-SIG": ("full", "plain", "end_acc_shifted"),
        "H-NEC": ("ko_lease_register", "ko_hidden_bias", "end_acc"),
        "H-BLIND": ("full", "ko_lease_register", "reinstate_acc"),
        "H-RIVAL": ("full", "neggrad", "end_acc"),
        "H-ESCROW": ("full", "ko_reserved_fraction", "reinstate_acc"),
    }
    rows = []
    for hid, (a, b, metric) in pairs.items():
        diffs = [r[a][metric] - r[b][metric] for r in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        v = "not evaluated" if quick else verdict(mean, ci, spec[hid]["direction"], spec[hid]["mesi"])
        rows.append({"id": hid, "metric": metric, "mean_diff": mean, "ci95": list(ci),
                     "mesi": spec[hid]["mesi"], "n_seeds": len(per_seed), "verdict": v})
    return rows


def knockout_table(per_seed, rng):
    rows = []
    for mod in modules(per_seed[0]["full"] and {"params": {}} or {}):
        key = "ko_" + mod
        diffs = [r[key]["end_acc"] - r["full"]["end_acc"] for r in per_seed]
        mean, ci = paired_bootstrap(diffs, rng)
        rows.append({"module": mod, "signature": modules({})[mod]["signature"],
                     "metric_change": mean, "ci95": list(ci)})
    return rows


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def print_report(rep):
    print("=== VERIFIED REPORT · chapter %04d ===" % CHAPTER)
    print("file: %s | python %s | numpy %s" % (rep["file"], rep["environment"]["python"], rep["environment"]["numpy"]))
    print("seeds: %s | runtime %.1f s | n_params %d" % (rep["seeds"], rep["runtime_s"], rep["n_params"]))
    gc = rep["gradcheck"]
    print("gradcheck: %d/%d tensors, max_rel_error %.2e, checked at %s -> %s"
          % (gc["tensors_checked"], gc["tensors_total"], gc["max_rel_error"], gc["checked_at"], "PASS" if gc["passed"] else "FAIL"))
    print("correctness:")
    for c in rep["correctness"]:
        print("  %-5s %-36s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
    m = rep["mutants"]
    print("mutation score: %d/%d = %.2f" % (m["detected"], m["total"], m["score"]))
    print("trivial baseline (majority, held-out valid fiefs): %.3f" % rep["trivial_accuracy"])
    print("config means over seeds (end_acc / end_acc_shifted / retained / new / reinstate / revoked_probe [floor %.3f]):"
          % rep["config_means"]["full"]["probe_floor"])
    for name, v in rep["config_means"].items():
        print("  %-22s %.3f / %.3f / %.3f / %.3f / %.3f / %.3f"
              % (name, v["end_acc"], v["end_acc_shifted"], v["retained_acc"], v["new_acc"], v["reinstate_acc"], v["revoked_probe"]))
    print("hypotheses:")
    for h in rep["hypotheses"]:
        print("  %-8s %-18s mean_diff %+.3f  ci95 [%+.3f, %+.3f]  mesi %.2f  seeds %d  -> %s"
              % (h["id"], h["metric"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["n_seeds"], h["verdict"]))
    print("knockouts (end_acc change vs full):")
    for k in rep["knockouts"]:
        print("  %-20s signature=%-5s %+.3f  ci95 [%+.3f, %+.3f]" % (k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print("task types: %s | exit_code %d" % (rep["task_types"], rep["exit_code"]))
    print("=== END REPORT ===")


def config_means(per_seed):
    keys = ("end_acc", "end_acc_shifted", "retained_acc", "new_acc", "reinstate_acc", "revoked_probe", "probe_floor")
    return {name: {k: float(np.mean([r[name][k] for r in per_seed])) for k in keys} for name in CONFIGS}


# ----------------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog=FILE_NAME, description="chapter 0281 · resumable lease network")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    if args.mutant is not None and args.mutant not in MUTANTS:
        print("unknown mutant %s; known: %s" % (args.mutant, sorted(MUTANTS)))
        return 2
    t0 = time.time()
    budget = BUDGET_QUICK if args.quick else BUDGET_FULL
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    seeds = [args.seed + i for i in range(n_seeds)]
    print("run: python3 %s%s | seeds %s" % (FILE_NAME, " --quick" if args.quick else "", seeds))
    rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0])
    data0 = build_stream(np.random.default_rng(seeds[0]))
    try:
        if args.mutant is not None:
            res = run_one({"kind": "full"}, data0, seeds[0], mutant=args.mutant, quick=args.quick)
            c1 = test_c1_gradcheck(data0, seeds[0], mutant=args.mutant)
            c3 = test_c3_learning(res, data0)
            iso = test_c6_isolation(seeds[0], mutant=args.mutant)
            for c in (c1, c3, iso):
                print("  %-5s %-30s %s  %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
            return 0 if all(c["passed"] for c in (c1, c3, iso)) else 1
        per_seed = run_seeds(seeds, args.quick)
        full0 = per_seed[0]
        c1 = test_c1_gradcheck(data0, seeds[0])
        c2 = test_c2_determinism(data0, seeds[0])
        c3 = test_c3_learning({"losses": full0["full_losses"], "end_acc": full0["full"]["end_acc"]}, data0)
        c4 = test_c4_shuffled(data0, seeds[0], args.quick)
        c5 = test_c5_mutants(data0, seeds[0])
        c61, c61n = test_c6_conservation(seeds[0]), test_c6_conservation(seeds[0], broken=True)
        c62, c62n = test_c6_isolation(seeds[0]), test_c6_isolation(seeds[0], mutant="no_lease_mask")
        c61["passed"] = c61["passed"] and not c61n["passed"]
        c61["detail"] += "; negative control violates=%s" % (not c61n["passed"])
        c62["passed"] = c62["passed"] and not c62n["passed"]
        c62["detail"] += "; negative control violates=%s" % (not c62n["passed"])
        c63 = test_c6_definition(seeds[0])
        c7 = test_c7_split_integrity(data0)
        runtime = time.time() - t0
        c8 = {"id": "C8", "name": "budget", "passed": runtime <= budget, "detail": "%.1f s of %.0f s" % (runtime, budget)}
        correctness = [c1, c2, c3, c4, c5, c61, c62, c63, c7, c8]
        hyps = hypothesis_table(per_seed, rng, args.quick)
        kos = knockout_table(per_seed, rng)
    except NonFinite:
        print("non-finite values encountered")
        return 4
    if args.data:
        if os.path.exists(args.data):
            real = load_csv_stream(args.data)
            r = run_one({"kind": "full"}, real, seeds[0], quick=args.quick)
            print("real-data bridge %s: end_acc %.3f (trivial %.3f)" % (args.data, r["end_acc"], trivial_accuracy(real)))
        else:
            print("real-data bridge skipped: %s not found" % args.data)
    else:
        print("real-data bridge skipped (no --data PATH)")
    exit_code = 0 if all(c["passed"] for c in correctness) else 1
    if not c8["passed"]:
        exit_code = 3
    rep = {
        "schema_version": "1.0", "chapter": CHAPTER, "file": FILE_NAME, "card_revision": MIND_CARD["card_revision"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seeds": seeds, "runtime_s": round(time.time() - t0, 2),
        "n_params": n_params(build_model(IN_DIM, N_CLASSES, "vector_classification", np.random.default_rng(0))),
        "gradcheck": {"tensors_checked": c1["tensors"], "tensors_total": 4, "max_rel_error": c1["max_rel_error"],
                      "checked_at": ["init", "after_training_steps"], "passed": c1["passed"]},
        "correctness": [{k: c[k] for k in ("id", "name", "passed", "detail")} for c in correctness],
        "mutants": {"detected": c5["detected"], "total": c5["total"], "score": c5["detected"] / c5["total"]},
        "trivial_accuracy": trivial_accuracy(data0),
        "config_means": config_means(per_seed),
        "hypotheses": hyps, "knockouts": kos, "task_types": TASK_TYPES, "exit_code": exit_code,
    }
    print_report(rep)
    if args.json:
        write_report(rep, args.json)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

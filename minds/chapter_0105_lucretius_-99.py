#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0105 · Lucretius
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Clinamen Engine: a field of atoms that falls in parallel until a learned minimal swerve breaks the symmetry.

Thesis
    Identical atoms falling in parallel make nothing; combination begins only when
    they deviate, and the deviation must be the least that suffices.

Evidence and provenance
    Provenance is belief: the mechanism rests on Lucretius' own poem (Latin as in
    the Perseus text). Cicero is a hostile ancient witness; O'Keefe is scholarship.
    D1  DRN 2.216-224: bodies falling straight down by their own weight deviate a
        little at no fixed time or place; otherwise they would fall like raindrops,
        no blow would occur and nature would have created nothing.
    D2  DRN 2.225-242: in the void all bodies fall at equal speed, so heavier ones
        cannot overtake lighter ones and cause the blows.
    D3  DRN 2.243-250: the deviation is no more than the least, lest we invent
        oblique motions that reality refutes; no one can see that atoms never swerve.
    D4  DRN 2.251-262: the swerve breaks the bonds of fate and grounds free will;
        we swerve not at a fixed time or place but where the mind has carried us.
    D5  Cicero, De Fato 22: a third motion besides weight and blow, a swerve by the
        minimal interval, introduced to escape the necessity of fate.
    D6  Cicero, De Fato 23: Democritus preferred that everything happen by
        necessity; Carneades held that Epicureans could defend voluntary motion
        without the swerve.
    D7  Cicero, De Finibus 1.17-18: Democritus' atoms move from eternity in a void
        with no top or bottom and cohere through collisions; Epicurus made them
        fall straight down by weight.
    D8  Cicero, De Finibus 1.19-20: the swerve is uncaused, a childish fiction; if
        all atoms swerve none will cohere, and if only some do it is like assigning
        provinces to atoms.
    D9  O'Keefe 1996; 2005: whether the swerve is needed to start collisions, when
        atomic motion has no beginning, is a live question in scholarship.
    D10 DRN 4.823-857: nothing in the body was born so that we might use it; the
        use follows what is born.

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1 D2  M1 pondus: linear drift; at a symmetric start every atom is identical  C6.1
    D1 D3  M2 clinamen: reparameterized Gaussian swerve with a learned per-atom
           scale and a penalty on its mean (paulum)                 C1 C6.2 H-SIG H-NEC
    D8     M2 against untuned uniform noise of matched average scale             H-SIG
    D1     M3 concilium: tanh collision gate forming distinct compound features  C6.2
    D10    M4 eventa: linear readout of the whole field; gradient descent        C3
    D6 D7  rival: independent random start, deterministic training, no swerve    H-RIVAL
    D7 D9  blind spot: that disordered start with and without the swerve         H-BLIND

Research question (open-endedness and creativity)
    In a learning system that starts symmetric, does a learned, minimal, per-unit
    deviation break the symmetry better than untuned noise of the same average size,
    and does a symmetric start with a swerve do better than the ordinary remedy of a
    disordered start and deterministic training?

Closest prior art and the delta
    Gaussian noise injection (Bishop 1995), learned noise scales for exploration
    (NoisyNets, Fortunato et al. 2018), learned Gaussian noise with the
    reparameterization trick (Kingma, Salimans and Welling 2015), and symmetry
    breaking by random initialization. Overlap is high, so the contribution is the
    test. Baseline: untuned noise from the same symmetric start (size-matched).
    Rival: the Democritean engine of chapter 0066 (disordered start, no swerve).

Blind spot
    The swerve answers a hypothetical symmetric start. When the atoms already move in
    disorder, the posited deviation should add nothing but noise.

Task (generative process)
    Latents a, b, c ~ N(0, 1) lie on three random orthonormal directions of R^8;
    x = a u1 + b u2 + c u3 + Gaussian noise (s.d. 0.3; shifted split 0.6).
    Label y = 2[a b > 0] + [c > 0]: four classes, two of them separated by an XOR, so
    a field whose atoms stay identical cannot solve the task.
    Splits: train 512, held-out 1024, shifted 1024.
    Starts: fall (every atom identical on input and output sides), near-fall (fall
    plus independent jitter of 1e-3 of the initial scale), disorder (independent
    random atoms).

Limits
    One field of eight atoms; synthetic data; the swerve acts during training and is
    switched off at evaluation. Nothing here models free will or feeling. A research
    prototype of one mechanism, not an AGI and not a claim to reproduce Lucretius' mind.
"""

MIND_CARD = {
    "schema_version": "1.0",
    "card_revision": 1,
    "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A",
                   "generator": "Claude (Anthropic)", "generator_version": "claude-opus-5", "date": "2026-09-15"},
    "id": 105, "figure": "Lucretius", "born": -99, "died": -55, "civilization": "Roman",
    "provenance": "belief",
    "thesis": ("Identical atoms falling in parallel make nothing; combination begins only when they deviate, and "
               "the deviation must be the least that suffices."),
    "evidence": [
        {"id": "D1", "claim": "Bodies falling straight down by their own weight deviate a little at no fixed time or "
         "place; otherwise they would fall like raindrops and nature would have created nothing.",
         "basis": "primary", "source": "Lucretius, De Rerum Natura 2.216-224"},
        {"id": "D2", "claim": "In the void all bodies fall at equal speed, so heavier ones cannot overtake lighter ones "
         "and cause blows.", "basis": "primary", "source": "Lucretius, De Rerum Natura 2.225-242"},
        {"id": "D3", "claim": "The deviation is no more than the least, lest oblique motions be invented that reality "
         "refutes; no one can see that atoms never swerve.", "basis": "primary",
         "source": "Lucretius, De Rerum Natura 2.243-250"},
        {"id": "D4", "claim": "The swerve breaks the bonds of fate and grounds free will; we swerve not at a fixed time "
         "or place but where the mind has carried us.", "basis": "primary",
         "source": "Lucretius, De Rerum Natura 2.251-262"},
        {"id": "D5", "claim": "Epicurus added a third motion besides weight and blow, a swerve by the minimal interval, "
         "to escape the necessity of fate (hostile ancient testimony).", "basis": "scholarship",
         "source": "Cicero, De Fato 22"},
        {"id": "D6", "claim": "Democritus preferred that everything happen by necessity; Carneades held that Epicureans "
         "could defend voluntary motion without the swerve (hostile ancient testimony).", "basis": "scholarship",
         "source": "Cicero, De Fato 23"},
        {"id": "D7", "claim": "Democritus' atoms move from eternity in a void with no top or bottom and cohere through "
         "collisions; Epicurus made them fall straight down by weight (hostile ancient testimony).",
         "basis": "scholarship", "source": "Cicero, De Finibus 1.17-18"},
        {"id": "D8", "claim": "The swerve is uncaused, a childish fiction; if all atoms swerve none will cohere, and if "
         "only some do it is like assigning provinces to atoms (hostile ancient testimony).",
         "basis": "scholarship", "source": "Cicero, De Finibus 1.19-20"},
        {"id": "D9", "claim": "Whether the swerve is needed to start collisions, when atomic motion has no beginning, "
         "is a live question in scholarship.", "basis": "scholarship",
         "source": "O'Keefe 1996, Phronesis 41; O'Keefe 2005, Epicurus on Freedom"},
        {"id": "D10", "claim": "Nothing in the body was born so that we might use it; the use follows what is born.",
         "basis": "primary", "source": "Lucretius, De Rerum Natura 4.823-857"},
    ],
    "research_question": {
        "category": "open-endedness and creativity",
        "question": ("In a learning system that starts symmetric, does a learned, minimal, per-unit deviation break "
                     "the symmetry better than untuned noise of the same average size, and does a symmetric start "
                     "with a swerve do better than a disordered start with deterministic training?")},
    "mechanism": {
        "name": "Clinamen Engine (five-stage atom field with a learned minimal swerve)",
        "family": "noise injection with a learned per-unit scale; symmetry breaking in learning systems",
        "signature_modules": ["clinamen"],
        "closest_prior_art": [
            "Gaussian noise injection during training (Bishop 1995)",
            "learned noise scales for exploration, NoisyNets (Fortunato et al. 2018)",
            "learned Gaussian noise with the reparameterization trick (Kingma, Salimans and Welling 2015)",
            "symmetry breaking by random initialization (standard practice)"],
        "overlap": "High",
        "prior_art_queries": [],
        "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
        "contribution_type": "test",
        "delta": ("The swerve starts from an exactly symmetric field, has a learned per-atom scale pushed toward the "
                  "least that suffices, and is tested against untuned noise of matched average scale and against "
                  "deterministic training from a disordered start."),
        "baselines": {
            "baseline": ("untuned noise: same engine and symmetric start with one fixed swerve scale for all atoms, "
                         "equal to the learned swerve's mean scale over its training run; no penalty"),
            "rival": "Democritean engine: independent random start and deterministic training, no swerve (0066)"}},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M2 clinamen", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D2", "mechanism": "M1 pondus (symmetric start)", "property_test": "C6.1", "hypothesis": "H-NEC"},
        {"doctrine": "D3", "mechanism": "M2 clinamen (paulum penalty)", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D4", "mechanism": "none (free will is not modelled)", "property_test": "none", "hypothesis": "none"},
        {"doctrine": "D5", "mechanism": "M2 clinamen", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D6", "mechanism": "rival engine", "property_test": "none", "hypothesis": "H-RIVAL"},
        {"doctrine": "D7", "mechanism": "rival engine; blind-spot start", "property_test": "none",
         "hypothesis": "H-RIVAL, H-BLIND"},
        {"doctrine": "D8", "mechanism": "untuned baseline", "property_test": "none", "hypothesis": "H-SIG"},
        {"doctrine": "D9", "mechanism": "blind-spot start", "property_test": "none", "hypothesis": "H-BLIND"},
        {"doctrine": "D10", "mechanism": "M4 eventa; gradient descent", "property_test": "none", "hypothesis": "none (C3)"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": ("From a symmetric start, the learned minimal swerve reaches higher accuracy on the "
                                      "shifted split than untuned noise of matched average scale."),
         "metric": "accuracy", "split": "shifted", "start": "fall",
         "comparison": "model - baseline", "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": ("From a nearly symmetric start, removing the swerve during training costs more "
                                      "held-out accuracy than removing the per-atom drift offset."),
         "metric": "accuracy", "split": "heldout", "start": "near_fall",
         "comparison": "signature_knockout - matched_knockout", "knockouts": ["clinamen:identity", "pondus_bias:zero"],
         "direction": "less", "mesi": 0.05, "seeds": 5},
        {"id": "H-BLIND", "statement": ("From a disordered start the swerve is superfluous: the engine with it is less "
                                        "accurate than the same engine trained deterministically."),
         "condition": "disorder start",
         "grounding": ("The swerve answers a hypothetical parallel fall (DRN 2.216-250); Democritean motion has no "
                       "beginning (Cicero, Fin. 1.17), and whether collisions need a start is disputed (O'Keefe)."),
         "metric": "accuracy", "split": "heldout", "comparison": "model - baseline", "direction": "less",
         "mesi": 0.01, "seeds": 5},
        {"id": "H-RIVAL", "statement": ("The Lucretian world (symmetric start and learned swerve) ends more accurate "
                                        "than the Democritean world (disordered start and necessity)."),
         "metric": "accuracy", "split": "heldout", "comparison": "model(fall) - rival(disorder)",
         "direction": "greater", "mesi": 0.02, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_margin": 0.08,
                   "gradcheck_rel_error": 1e-5, "gradcheck_floor": 1e-3, "equivariance_tol": 1e-10,
                   "bound_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"accuracy": "fraction of correct argmax predictions with the swerve switched off",
                "trivial_baseline": "majority class of the training labels",
                "shuffled_band": "one-sided: a leak would show as held-out accuracy well above the majority rate"},
    "training": {"optimizer": "full-batch gradient descent with momentum 0.9", "lr_grid": [0.1],
                 "updates": {"full": 2000, "quick": 1000}, "clip_norm": 5.0,
                 "batch": "all 512 training examples; a fresh swerve draw at every update",
                 "model_selection": "none: final parameters", "applies_to": "every engine"},
    "task": {"inputs": 8, "classes": 4, "atoms": 8, "label": "2[a b > 0] + [c > 0]",
             "noise_sd": {"nominal": 0.3, "shifted": 0.6}, "splits": {"train": 512, "heldout": 1024, "shifted": 1024},
             "swerve_init": 0.3, "paulum": 0.01, "near_fall_jitter": 1e-3,
             "engines": ["lucretian_fall", "untuned_fall", "democritean_disorder", "lucretian_disorder",
                         "lucretian_near", "ko_clinamen_near", "ko_pondus_bias_near", "ko_concilium_near"]},
    "probe_predictions": [{"probe": "P9", "expected": "above baseline"},
                          {"probe": "P10", "expected": "below baseline"}],
    "dialectic_links": [
        {"chapter": 66, "relation": "rival", "test": "H-RIVAL",
         "note": "Cicero, De Fato 23 and De Finibus 1.17-20 set Epicurean swerve against Democritean necessity."},
        {"chapter": 79, "relation": "teacher", "test": "none",
         "note": "Lucretius expounds Epicurus; the swerve is attributed to Epicurus but not attested in his surviving works."}],
    "corpus_neighbors": [
        {"chapter": 79, "similarity": None, "difference": ("0079 uses a minimal swerve as a source of novelty inside a "
                                                           "canonic epistemology; here the swerve is tested only as a "
                                                           "symmetry breaker from an exact symmetric start, against "
                                                           "untuned noise and random initialization.")},
        {"chapter": 66, "similarity": None, "difference": ("0066 is deterministic atomic relaxation; here deterministic "
                                                           "training from a disordered start is the rival engine.")},
        {"chapter": 56, "similarity": None, "difference": ("0056 blends four roots at a critical edge; here nothing is "
                                                           "blended: identical units are made to differ.")},
        {"chapter": 102, "similarity": None, "difference": ("the old 0105 file shared term, syllogism and objection "
                                                            "classes with 0102; this file shares no class with it.")},
    ],
    "similarity_note": "Nearest-neighbour similarity not computed: corpus files were not available to this session.",
    "barometer": {
        "cognitive_processing": ["learning an XOR-structured four-way task from a symmetric start"],
        "embodied_cognition": [], "world_modeling": [], "consciousness": [], "language_understanding": [],
        "emotional_intelligence": [], "creativity": ["feature diversity created from an exactly symmetric start"],
        "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [
        {"use": ("breaking symmetry among identical agents that share parameters, so they take different roles or "
                 "channels without a central assigner"),
         "sector": "multi-agent robotics and wireless networks",
         "dataset": "PettingZoo MPE environments (simple_spread)", "readiness": "low"},
        {"use": "exploration noise whose per-unit scale is learned under a minimality penalty",
         "sector": "reinforcement learning for control", "dataset": "Gymnasium MuJoCo tasks", "readiness": "low"},
        {"use": "initialization studies for tied or weight-shared networks where random initialization is unavailable",
         "sector": "on-device machine learning", "dataset": "Fashion-MNIST", "readiness": "low"},
    ],
    "safety_notes": ("No hazardous content. The swerve is not presented as a model of free will or consciousness; the "
                     "file does not claim to replicate Lucretius' mind and puts no generated words in his mouth."),
}

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time

import numpy as np

IN_DIM, CLASSES, ATOMS = 8, 4, 8
NOMINAL_SD, SHIFTED_SD = 0.3, 0.6
SPLIT_SIZES = {"train": 512, "heldout": 1024, "shifted": 1024}
SWERVE_INIT, PAULUM, JITTER = 0.3, 0.01, 1e-3
LR, MOMENTUM, CLIP_NORM = 0.1, 0.9, 5.0
STEPS = {"full": 2000, "quick": 1000}
TIME_BUDGET = {"full": 180.0, "quick": 20.0}
TASK_TYPES = ["vector_classification"]
ACTIVE_MUTANT = None
np.seterr(over="raise", invalid="raise", divide="raise", under="ignore")

# BEGIN STANDARD UTILITIES v1.0
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logsumexp(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()},
            "v": {k: np.zeros_like(v) for k, v in params.items()}}


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1.0 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1.0 - b2) * grads[k] ** 2
        m_hat = state["m"][k] / (1.0 - b1 ** state["t"])
        v_hat = state["v"][k] / (1.0 - b2 ** state["t"])
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def clip_global(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return {k: g * scale for k, g in grads.items()}, norm


def finite_difference_check(params, grads, loss_fn, rng, eps=1e-6, n_entries=20, floor=1e-3):
    """Central differences on n random entries per tensor plus its largest-gradient entry.
    Relative error uses max(|analytic|, |numeric|, floor) as denominator."""
    worst = {}
    for name, arr in params.items():
        flat, g = arr.reshape(-1), grads[name].reshape(-1)
        if flat.size <= n_entries + 1:
            idx = np.arange(flat.size)
        else:
            idx = np.unique(np.append(rng.choice(flat.size, n_entries, replace=False), np.argmax(np.abs(g))))
        err = 0.0
        for i in idx:
            keep = flat[i]
            flat[i] = keep + eps
            up = loss_fn()
            flat[i] = keep - eps
            down = loss_fn()
            flat[i] = keep
            num = (up - down) / (2.0 * eps)
            err = max(err, abs(g[i] - num) / max(abs(g[i]), abs(num), floor))
        worst[name] = err
    return worst


def paired_bootstrap(diffs, rng, n_boot=2000, level=0.95):
    d = np.asarray(diffs, dtype=float)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    tail = 50.0 * (1.0 - level)
    return float(d.mean()), [float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))]


def verdict(mean, ci, mesi, direction):
    s = 1.0 if direction == "greater" else -1.0
    lo, hi = sorted((s * ci[0], s * ci[1]))
    if lo > 0.0 and s * mean >= mesi:
        return "supported"
    if hi < 0.0:
        return "contradicted"
    return "inconclusive"


def write_report(lines, payload, json_path):
    print("\n".join(lines))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
# END STANDARD UTILITIES


# ---------------------------------------------------------------- data and tasks
def sample_split(rng, basis, n, noise_sd):
    latent = rng.standard_normal((n, 3))
    X = latent @ basis.T + noise_sd * rng.standard_normal((n, basis.shape[0]))
    Y = 2 * (latent[:, 0] * latent[:, 1] > 0).astype(int) + (latent[:, 2] > 0).astype(int)
    return {"X": X, "Y": Y}


def make_data(rng):
    """Three latents on random orthonormal directions; label 2[a b > 0] + [c > 0]."""
    basis = np.linalg.qr(rng.standard_normal((IN_DIM, IN_DIM)))[0][:, :3]
    return {name: sample_split(rng, basis, size, SHIFTED_SD if name == "shifted" else NOMINAL_SD)
            for name, size in SPLIT_SIZES.items()}


def trivial_accuracy(data, split="heldout"):
    majority = int(np.argmax(np.bincount(data["train"]["Y"], minlength=CLASSES)))
    return float(np.mean(data[split]["Y"] == majority))


# ---------------------------------------------------------------- model
def initial_field(start, rng, d, h, k):
    """Fall: every atom identical on input and output sides. Near-fall: fall plus tiny jitter. Disorder: independent."""
    if start == "disorder":
        return rng.normal(0.0, d ** -0.5, (h, d)), rng.normal(0.0, h ** -0.5, (k, h))
    W = np.repeat(rng.normal(0.0, d ** -0.5, (1, d)), h, axis=0)
    V = np.repeat(rng.normal(0.0, h ** -0.5, (k, 1)), h, axis=1)
    if start == "near_fall":
        W = W + JITTER * rng.normal(0.0, d ** -0.5, (h, d))
        V = V + JITTER * rng.normal(0.0, h ** -0.5, (k, h))
    elif start != "fall":
        raise ValueError(f"unknown start {start}")
    return W, V


def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """Clinamen Engine by default (fall start, learned swerve). swerve='untuned' or 'off' and start='disorder'
    build the size-matched baseline and the Democritean rival."""
    if task_type not in TASK_TYPES:
        raise ValueError("chapter 0105 supports vector_classification")
    cfg = dict({"atoms": ATOMS, "start": "fall", "swerve": "learned", "swerve_init": SWERVE_INIT,
                "untuned_scale": SWERVE_INIT, "paulum": PAULUM, "lr": LR}, **cfg)
    W, V = initial_field(cfg["start"], rng, in_dim, cfg["atoms"], out_dim)
    params = {"W": W, "b": np.zeros(cfg["atoms"]), "V": V, "a": np.zeros(out_dim)}
    if cfg["swerve"] == "learned":
        params["rho"] = np.full(cfg["atoms"], math.log(math.expm1(cfg["swerve_init"])))
    elif cfg["swerve"] not in ("untuned", "off"):
        raise ValueError(f"unknown swerve {cfg['swerve']}")
    return {"params": params, "cfg": cfg, "ko": {}}


def swerve_scale(model):
    h = model["params"]["W"].shape[0]
    if model["ko"].get("clinamen") == "identity" or model["cfg"]["swerve"] == "off":
        return np.zeros(h)
    if model["cfg"]["swerve"] == "learned":
        return softplus(model["params"]["rho"])
    return np.full(h, model["cfg"]["untuned_scale"])


def forward(model, X, eps):
    """Simulacra -> pondus -> clinamen -> concilium -> eventa."""
    P, ko = model["params"], model["ko"]
    drift = X @ P["W"].T + (0.0 if ko.get("pondus_bias") == "zero" else P["b"])
    scale = swerve_scale(model)
    swerved = drift + eps * scale
    compounds = swerved if ko.get("concilium") == "identity" else np.tanh(swerved)
    return {"drift": drift, "scale": scale, "swerved": swerved, "compounds": compounds,
            "logits": compounds @ P["V"].T + P["a"]}


def loss_and_grads(model, batch):
    """Cross-entropy plus the paulum penalty on the mean learned swerve; gradients derived by hand."""
    P, cfg, ko = model["params"], model["cfg"], model["ko"]
    X, Y, eps = batch["X"], batch["Y"], batch["eps"]
    n, f = X.shape[0], forward(model, X, eps)
    learned = "rho" in P and ko.get("clinamen") != "identity"
    ce = float(np.mean(logsumexp(f["logits"], axis=1) - f["logits"][np.arange(n), Y]))
    loss = ce + (cfg["paulum"] * float(f["scale"].mean()) if learned else 0.0)
    d_logits = softmax(f["logits"], axis=1)
    d_logits[np.arange(n), Y] -= 1.0
    d_logits /= n
    d_comp = d_logits @ P["V"]
    d_swerved = d_comp if ko.get("concilium") == "identity" else d_comp * (1.0 - f["compounds"] ** 2)
    grads = {"W": d_swerved.T @ X,
             "b": np.zeros_like(P["b"]) if ko.get("pondus_bias") == "zero" else d_swerved.sum(axis=0),
             "V": d_logits.T @ f["compounds"], "a": d_logits.sum(axis=0)}
    if "rho" in P:
        d_scale = (d_swerved * eps).sum(axis=0) + cfg["paulum"] / P["rho"].size
        grads["rho"] = d_scale * sigmoid(P["rho"]) if learned else np.zeros_like(P["rho"])
    if ACTIVE_MUTANT == "zero_grad_swerve_scale" and "rho" in grads:
        grads["rho"] = np.zeros_like(grads["rho"])
    if ACTIVE_MUTANT == "atom_index_leak":
        grads["W"] = grads["W"] * (1.0 + 1e-3 * np.arange(grads["W"].shape[0]))[:, None]
    return loss, grads


def predict(model, X):
    X = np.asarray(X, dtype=float)
    return softmax(forward(model, X, np.zeros((X.shape[0], model["params"]["W"].shape[0])))["logits"], axis=1)


def hidden_states(model, X):
    X = np.asarray(X, dtype=float)
    f = forward(model, X, np.zeros((X.shape[0], model["params"]["W"].shape[0])))
    return {"pondus": f["drift"], "concilium": f["compounds"]}


def accuracy(model, split):
    return float(np.mean(np.argmax(predict(model, split["X"]), axis=1) == split["Y"]))


def atom_spread(model, X):
    """Largest difference between any atom's compound activation and the first atom's, over inputs X."""
    compounds = hidden_states(model, X)["concilium"]
    return float(np.abs(compounds - compounds[:, :1]).max())


# ---------------------------------------------------------------- baselines and rival mechanisms
# name, start, swerve, training-time knockout
ENGINES = (
    ("lucretian_fall", "fall", "learned", None),
    ("untuned_fall", "fall", "untuned", None),
    ("democritean_disorder", "disorder", "off", None),
    ("lucretian_disorder", "disorder", "learned", None),
    ("lucretian_near", "near_fall", "learned", None),
    ("ko_clinamen_near", "near_fall", "learned", ("clinamen", "identity")),
    ("ko_pondus_bias_near", "near_fall", "learned", ("pondus_bias", "zero")),
    ("ko_concilium_near", "near_fall", "learned", ("concilium", "identity")),
)
STARTS = ("fall", "near_fall", "disorder")


def build_engine(name, seed, untuned_scale=SWERVE_INIT):
    """Engines sharing a start draw it from one stream, so paired engines begin from the same field."""
    kids = np.random.SeedSequence(seed).spawn(1 + len(STARTS) + len(ENGINES))
    idx = [e[0] for e in ENGINES].index(name)
    _, start, swerve, ko = ENGINES[idx]
    model = build_model(IN_DIM, CLASSES, TASK_TYPES[0], np.random.default_rng(kids[1 + STARTS.index(start)]),
                        start=start, swerve=swerve, untuned_scale=untuned_scale)
    return (knockout(model, *ko) if ko else model), np.random.default_rng(kids[1 + len(STARTS) + idx])


def train_engine(name, data, seed, budget, untuned_scale=SWERVE_INIT):
    model, noise = build_engine(name, seed, untuned_scale)
    fit(model, data, budget, noise)
    return model


# ---------------------------------------------------------------- registries
KNOCKOUT_MODES = {"clinamen": ("identity",), "pondus_bias": ("zero",), "concilium": ("identity",)}
MUTANTS = {
    "sign_flipped_update": "updates climb the gradient instead of descending it (C3 must fail)",
    "zero_learning_rate": "learning rate forced to zero (C3 must fail)",
    "zero_grad_swerve_scale": "gradient of the swerve scale zeroed (C1 must fail)",
    "atom_index_leak": "drift gradient scaled by atom index, a symmetry-breaking bug (C1 must fail)",
}


def modules(model):
    table = {
        "simulacra": ([], "input feature vector (no parameters)", False),
        "pondus": (["W"], "linear drift of each atom (hidden pre-activation weights)", False),
        "pondus_bias": (["b"], "per-atom offset of the drift (hidden bias)", False),
        "clinamen": (["rho"] if "rho" in model["params"] else [],
                     "additive Gaussian deviation per atom with learned or fixed scale (noise injection)", True),
        "concilium": ([], "tanh gate turning swerved drifts into compound features", False),
        "eventa": (["V", "a"], "linear readout of the whole field to class logits", False)}
    return {name: {"params": p, "role": role, "signature": sig} for name, (p, role, sig) in table.items()}


def knockout(model, name, mode):
    """Copy with a module replaced: clinamen by identity (no deviation), pondus_bias by zero, concilium by identity."""
    if mode not in KNOCKOUT_MODES.get(name, ()):
        raise ValueError(f"no knockout {name}:{mode}")
    return {"params": {k: v.copy() for k, v in model["params"].items()}, "cfg": dict(model["cfg"]),
            "ko": dict(model["ko"], **{name: mode})}


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


# ---------------------------------------------------------------- training
def fit(model, data, budget, rng):
    """Full-batch gradient descent with momentum; the swerve is redrawn at every update."""
    P = model["params"]
    velocity = {k: np.zeros_like(v) for k, v in P.items()}
    lr = 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else model["cfg"]["lr"]
    direction = 1.0 if ACTIVE_MUTANT == "sign_flipped_update" else -1.0
    X, Y = data["train"]["X"], data["train"]["Y"]
    history, scales = [], []
    for step in range(1, budget + 1):
        eps = rng.standard_normal((X.shape[0], P["W"].shape[0]))
        loss, grads = loss_and_grads(model, {"X": X, "Y": Y, "eps": eps})
        if not (math.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())):
            raise FloatingPointError(f"non-finite loss or gradient at update {step}")
        grads = clip_global(grads, CLIP_NORM)[0]
        for k in P:
            velocity[k] = MOMENTUM * velocity[k] + grads[k]
            P[k] += direction * lr * velocity[k]
        history.append(loss)
        scales.append(float(swerve_scale(model).mean()))
    model["trace"] = {"loss": history, "mean_scale": scales}
    return history


def data_bridge(path, seed, budget):
    """Optional: a local numeric CSV with a header and an integer class label in the last column."""
    if not os.path.exists(path):
        return f"skipped ({path} not found)"
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = table[:, :-1], table[:, -1].astype(int)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    order = np.random.default_rng(seed).permutation(len(Y))
    cut = int(0.8 * len(Y))
    data = {"train": {"X": X[order[:cut]], "Y": Y[order[:cut]]}, "heldout": {"X": X[order[cut:]], "Y": Y[order[cut:]]}}
    model = build_model(X.shape[1], int(Y.max()) + 1, TASK_TYPES[0], np.random.default_rng(seed))
    fit(model, data, budget, np.random.default_rng(seed + 1))
    return f"{path}: held-out accuracy {accuracy(model, data['heldout']):.4f} vs majority {trivial_accuracy(data):.4f}"


# ---------------------------------------------------------------- tests: correctness
def gradient_errors(model, data, rng, entries):
    """Worst central-difference relative error per tensor, on 64 training examples with one frozen swerve draw."""
    frozen = np.random.default_rng(4321).standard_normal((64, model["params"]["W"].shape[0]))
    batch = {"X": data["train"]["X"][:64], "Y": data["train"]["Y"][:64], "eps": frozen}
    return finite_difference_check(model["params"], loss_and_grads(model, batch)[1],
                                   lambda: loss_and_grads(model, batch)[0], rng, n_entries=entries,
                                   floor=MIND_CARD["thresholds"]["gradcheck_floor"])


def learns(model, data):
    """C3 rule, shared by the protocol and the mutant replay: loss drop and margin over the majority rate."""
    limits, losses = MIND_CARD["thresholds"], model["trace"]["loss"]
    drop = 1.0 - float(np.mean(losses[-20:])) / losses[0]
    acc, base = accuracy(model, data["heldout"]), trivial_accuracy(data)
    return drop >= limits["loss_drop_fraction"] and acc >= base + limits["margin_over_trivial"], drop, acc, base


def replay(data, seed, budget):
    """The base seed's Lucretian run exactly as in run_seed; True when it passes C1 at init and C3 after training."""
    try:
        model, noise = build_engine("lucretian_fall", seed)
        errors = gradient_errors(model, data, np.random.default_rng(seed), 4)
        fit(model, data, budget, noise)
        return max(errors.values()) <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and learns(model, data)[0]
    except FloatingPointError:
        return False


def under_mutant(name, action):
    """Call action() with a registered mutant switched on and restore the previous state afterwards."""
    global ACTIVE_MUTANT
    previous, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    try:
        return action()
    finally:
        ACTIVE_MUTANT = previous


def permuted(params, perm):
    moved = {"W": params["W"][perm], "b": params["b"][perm], "V": params["V"][:, perm], "a": params["a"].copy()}
    if "rho" in params:
        moved["rho"] = params["rho"][perm]
    return moved


def equivariance_gap(model, batch, perm):
    """Largest mismatch between the gradients of relabelled atoms and the relabelled gradients."""
    loss_1, grads_1 = loss_and_grads(model, batch)
    twin = {"params": permuted(model["params"], perm), "cfg": model["cfg"], "ko": model["ko"]}
    loss_2, grads_2 = loss_and_grads(twin, dict(batch, eps=batch["eps"][:, perm]))
    expected = permuted(grads_1, perm)
    return max([abs(loss_1 - loss_2)] + [float(np.abs(grads_2[k] - expected[k]).max()) for k in grads_2])


class Trial:
    """One protocol run: every engine trained on every seed, and the correctness checks on the base seed."""

    def __init__(self, mode, base_seed, n_seeds):
        self.mode, self.base_seed, self.budget = mode, base_seed, STEPS[mode]
        self.seeds = list(range(base_seed, base_seed + n_seeds))
        self.runs, self.caught = [], {}

    def train(self, clock):
        for seed in self.seeds:
            self.runs.append(run_seed(seed, self.budget))
            print(f"  seed {seed}: {len(ENGINES)} engines trained ({clock():.1f} s)", flush=True)
        self.data, self.engines = self.runs[0]["data"], self.runs[0]["models"]

    def gradients(self):
        names = ("lucretian_fall", "untuned_fall", "democritean_disorder")
        fresh, trained = np.random.default_rng(self.base_seed + 3), np.random.default_rng(self.base_seed + 4)
        tables = [gradient_errors(build_engine(n, self.base_seed)[0], self.data, fresh, 20) for n in names]
        tables += [gradient_errors(self.engines[n], self.data, trained, 20) for n in names]
        self.worst_gradient = max(max(t.values()) for t in tables)
        self.coverage = (sum(len(t) for t in tables) // 2, sum(len(self.engines[n]["params"]) for n in names))
        ok = self.worst_gradient <= MIND_CARD["thresholds"]["gradcheck_rel_error"] and len(set(self.coverage)) == 1
        return ok, (f"max rel error {self.worst_gradient:.2e}; every tensor of the Lucretian, untuned and Democritean "
                    f"engines at init and after {self.budget} updates")

    def determinism(self):
        twins = [train_engine("lucretian_fall", self.data, self.base_seed, 25) for _ in "ab"]
        outs = [predict(t, self.data["heldout"]["X"]) for t in twins]
        same = twins[0]["trace"]["loss"] == twins[1]["trace"]["loss"] and np.array_equal(*outs)
        finite = bool(np.isfinite(outs[0]).all()) and all(np.isfinite(p).all() for p in twins[0]["params"].values())
        return same and finite, f"identical losses and outputs: {same}; finite: {finite}"

    def learning(self):
        ok, drop, acc, base = learns(self.engines["lucretian_fall"], self.data)
        limits = MIND_CARD["thresholds"]
        return ok, (f"loss drop {drop:.3f} (min {limits['loss_drop_fraction']}); held-out accuracy {acc:.4f} vs "
                    f"majority {base:.4f} (min margin {limits['margin_over_trivial']})")

    def shuffled(self):
        order = np.random.default_rng(self.base_seed + 11).permutation(self.data["train"]["Y"].size)
        scrambled = dict(self.data, train=dict(self.data["train"], Y=self.data["train"]["Y"][order]))
        model = train_engine("lucretian_fall", scrambled, self.base_seed, self.budget)
        acc, base, limit = accuracy(model, self.data["heldout"]), trivial_accuracy(self.data), MIND_CARD["thresholds"]["shuffled_margin"]
        return acc <= base + limit, f"held-out accuracy after shuffled training {acc:.4f} vs majority {base:.4f} (max +{limit})"

    def mutants(self):
        control = replay(self.data, self.base_seed, self.budget)
        self.caught = {name: not under_mutant(name, lambda: replay(self.data, self.base_seed, self.budget))
                       for name in MUTANTS}
        found = sum(self.caught.values())
        return control and found == len(MUTANTS), (f"unmutated base-seed run passes C1 and C3: {control}; mutants "
                                                    f"caught {found}/{len(MUTANTS)}")

    def symmetry(self):
        """C6.1: relabelling atoms relabels loss and gradients exactly; an index-dependent gradient must break it.
        Corollary (definition, not evidence): from the fall without a swerve the atoms never come to differ."""
        rng, gaps, leaks = np.random.default_rng(self.base_seed + 5), [], []
        for _ in range(6):
            model = build_model(IN_DIM, CLASSES, TASK_TYPES[0], rng, start="disorder")
            model["params"].update(b=rng.normal(0.0, 1.0, ATOMS), a=rng.normal(0.0, 1.0, CLASSES),
                                   rho=rng.normal(-1.0, 1.0, ATOMS))
            batch = {"X": rng.normal(0.0, 1.0, (32, IN_DIM)), "Y": rng.integers(0, CLASSES, 32),
                     "eps": rng.normal(0.0, 1.0, (32, ATOMS))}
            perm = rng.permutation(ATOMS)
            while np.array_equal(perm, np.arange(ATOMS)):
                perm = rng.permutation(ATOMS)
            gaps.append(equivariance_gap(model, batch, perm))
            leaks.append(under_mutant("atom_index_leak", lambda: equivariance_gap(model, batch, perm)))
        spread = {}
        for swerve in ("off", "learned"):
            field = build_model(IN_DIM, CLASSES, TASK_TYPES[0], np.random.default_rng(self.base_seed + 21), swerve=swerve)
            fit(field, self.data, self.budget, np.random.default_rng(self.base_seed + 22))
            spread[swerve] = atom_spread(field, self.data["heldout"]["X"])
        limits = MIND_CARD["thresholds"]
        ok = max(gaps) <= limits["equivariance_tol"] and max(leaks) > limits["negative_control_min_violation"]
        return ok, (f"largest gap {max(gaps):.1e}; index-leak negative control gap {max(leaks):.1e}; corollary atom "
                    f"spread after training from the fall: no swerve {spread['off']:.1e}, learned swerve {spread['learned']:.1e}")

    def bound(self):
        """C6.2: a swerve moves the logits by at most ||V|| times the size of the deviation; a gate steeper than 1
        breaks it. Sampled near the gate's linear regime (small drift and swerve), where the bound is tight, and in saturation."""
        rng, excess, steep = np.random.default_rng(self.base_seed + 6), [], []
        for drift_mult, rho_mean in itertools.product((0.01, 1.0, 3.0), (-4.0, 0.0)):
            model = build_model(IN_DIM, CLASSES, TASK_TYPES[0], rng, start="disorder")
            P = model["params"]
            P["W"] *= drift_mult
            P["V"] *= 3.0
            P["rho"] = rng.normal(rho_mean, 1.0, ATOMS)
            X, e1, e2 = (rng.normal(0.0, 1.0, shape) for shape in ((64, IN_DIM), (64, ATOMS), (64, ATOMS)))
            one, two = forward(model, X, e1), forward(model, X, e2)
            allowance = np.linalg.norm(P["V"], 2) * np.linalg.norm((e1 - e2) * one["scale"], axis=1)
            excess.append(np.linalg.norm(one["logits"] - two["logits"], axis=1) - allowance)
            sharp = (np.tanh(3.0 * one["swerved"]) - np.tanh(3.0 * two["swerved"])) @ P["V"].T
            steep.append(np.linalg.norm(sharp, axis=1) - allowance)
        top, control, limits = float(np.max(excess)), float(np.max(steep)), MIND_CARD["thresholds"]
        ok = top <= limits["bound_tol"] and control > limits["negative_control_min_violation"]
        return ok, f"largest excess over the bound {top:.1e}; steep-gate negative control excess {control:.1e}"

    def splits(self):
        """C7: no input vector appears in two splits."""
        digests = [{hashlib.sha256(np.round(row, 12).tobytes()).hexdigest() for row in self.data[s]["X"]}
                   for s in SPLIT_SIZES]
        clean = not any(a & b for a, b in itertools.combinations(digests, 2))
        return clean, f"pairwise disjoint: {clean}"


CHECKS = (("C1", "gradient_check", "gradients"), ("C2", "determinism_finiteness", "determinism"),
          ("C3", "learning", "learning"), ("C4", "shuffled_label_control", "shuffled"),
          ("C5", "mutant_detection", "mutants"), ("C6.1", "atom_exchange_symmetry", "symmetry"),
          ("C6.2", "swerve_influence_bound", "bound"), ("C7", "split_integrity", "splits"))


# ---------------------------------------------------------------- tests: hypotheses
def run_seed(seed, budget):
    """Train every engine on one seed's data; untuned noise is matched to the Lucretian run's mean swerve scale."""
    data = make_data(np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]))
    models = {}
    for name, *_ in ENGINES:
        matched = np.mean(models["lucretian_fall"]["trace"]["mean_scale"]) if name == "untuned_fall" else SWERVE_INIT
        models[name] = train_engine(name, data, seed, budget, float(matched))
    acc = {name: {s: accuracy(m, data[s]) for s in ("heldout", "shifted")} for name, m in models.items()}

    def gap(first, second, split="heldout"):
        return acc[first][split] - acc[second][split]
    row = {"H-SIG": gap("lucretian_fall", "untuned_fall", "shifted"), "H-NEC": gap("ko_clinamen_near", "ko_pondus_bias_near"),
           "H-BLIND": gap("lucretian_disorder", "democritean_disorder"), "H-RIVAL": gap("lucretian_fall", "democritean_disorder")}
    lesions = {ko[0]: gap(name, "lucretian_near") for name, _, _, ko in ENGINES if ko}
    return {"data": data, "models": models, "acc": acc, "row": row, "knockouts": lesions}


def interval(values, rng, evaluated):
    return paired_bootstrap(values, rng) if evaluated else (float(np.mean(values)), None)


def judge(runs, seed, evaluated):
    """Paired per-seed differences, percentile bootstrap and verdicts against the frozen card; then the lesions."""
    rng, hyps, lesions = np.random.default_rng(seed + 9973), [], []
    for spec in MIND_CARD["hypotheses"]:
        mean, ci = interval([run["row"][spec["id"]] for run in runs], rng, evaluated)
        hyps.append(dict(id=spec["id"], metric=spec["metric"], mean_diff=mean, ci95=ci, mesi=spec["mesi"], n_seeds=len(runs),
                         verdict=verdict(mean, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"))
    roles = modules(runs[0]["models"]["lucretian_near"])
    for name, mode in (("clinamen", "identity"), ("pondus_bias", "zero"), ("concilium", "identity")):
        change, ci = interval([run["knockouts"][name] for run in runs], rng, evaluated)
        lesions.append(dict(module=name, mode=mode, signature=roles[name]["signature"], metric_change=change, ci95=ci))
    return hyps, lesions


# ---------------------------------------------------------------- report
def span(ci):
    return "n/a" if ci is None else "[{:+.4f}, {:+.4f}]".format(*ci)


def render(payload, extra):
    """Text block rendered from the Appendix B payload plus descriptive extras."""
    text = ["=== VERIFIED REPORT · chapter {:04d} ===".format(payload["chapter"]),
            "file: {} · card_revision {} · mode {} · mutant {}".format(payload["file"], payload["card_revision"],
                                                                    extra["mode"], ACTIVE_MUTANT),
            "environment: python {python} · numpy {numpy}".format(**payload["environment"]),
            "seeds: {} · runtime_s {:.1f} · budget_s {:.0f}".format(payload["seeds"], payload["runtime_s"],
                                                                   TIME_BUDGET[extra["mode"]]),
            "n_params: " + " · ".join("{} {}".format(*item) for item in extra["sizes"].items()),
            ("gradcheck: {tensors_checked}/{tensors_total} tensors at init and after training · max_rel_error "
             "{max_rel_error:.2e} · passed {passed}").format(**payload["gradcheck"]), "correctness:"]
    text += ["  {:<5} {:<24} {}  {}".format(c["id"], c["name"], ("FAIL", "PASS")[c["passed"]], c["detail"])
             for c in payload["correctness"]]
    tally = payload["mutants"]
    text.append("mutants: {}/{} detected · score {:.2f} · ".format(tally["detected"], tally["total"], tally["score"])
                + ", ".join(name + (" caught" if hit else " MISSED") for name, hit in extra["caught"].items()))
    text.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    text += ["  {:<8} mean_diff {:+.4f} ci95 {} mesi {} seeds {} -> {}".format(h["id"], h["mean_diff"], span(h["ci95"]),
                                                                            h["mesi"], h["n_seeds"], h["verdict"])
             for h in payload["hypotheses"]]
    text.append("knockouts (training-time, near-fall start; change in held-out accuracy vs the full engine):")
    text += ["  {:<12} {:<9} signature {:<5} {:+.4f} ci95 {}".format(k["module"], k["mode"], str(k["signature"]),
                                                                    k["metric_change"], span(k["ci95"]))
             for k in payload["knockouts"]]
    text.append("accuracy by engine (seed mean; held-out / shifted):")
    text += ["  {:<22} {:.4f} / {:.4f}".format(name, *pair) for name, pair in extra["engine_acc"].items()]
    text += ["swerve scale (seed mean): learned {:.3f} at start -> {:.4f} at end; matched untuned scale {:.4f}; "
             "majority-class accuracy {:.4f}".format(SWERVE_INIT, extra["scale_end"], extra["scale_matched"], extra["trivial"]),
             "real-data bridge: " + extra["bridge"], "task_types: " + ", ".join(payload["task_types"]),
             "exit_code: {}".format(payload["exit_code"]), "=== END REPORT ==="]
    return text


# ---------------------------------------------------------------- command line
def protocol(mode, base_seed, n_seeds, json_path, data_path):
    began = time.time()
    trial = Trial(mode, base_seed, n_seeds)
    print("{} · mode {} · seeds {} · mutant {}".format(os.path.basename(__file__), mode, trial.seeds, ACTIVE_MUTANT))
    trial.train(lambda: time.time() - began)
    results = [(cid, label) + tuple(getattr(trial, method)()) for cid, label, method in CHECKS]
    hyps, lesions = judge(trial.runs, base_seed, mode == "full" and n_seeds >= 5)
    bridge = data_bridge(data_path, base_seed, trial.budget) if data_path else "skipped (no --data PATH given)"
    elapsed = time.time() - began
    results.append(("C8", "budget", elapsed <= TIME_BUDGET[mode], "{:.1f} s of {:.0f} s".format(elapsed, TIME_BUDGET[mode])))
    failing = {cid for cid, _, ok, _ in results if not ok}
    code = 0 if not failing else (3 if failing == {"C8"} else 1)
    found = sum(trial.caught.values())
    payload = {"schema_version": "1.0", "chapter": 105, "file": os.path.basename(__file__),
               "card_revision": MIND_CARD["card_revision"],
               "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": trial.seeds,
               "runtime_s": round(elapsed, 2), "n_params": n_params(trial.engines["lucretian_fall"]),
               "gradcheck": {"tensors_checked": trial.coverage[0], "tensors_total": trial.coverage[1],
                             "max_rel_error": trial.worst_gradient, "checked_at": ["init", "after_training_steps"],
                             "passed": bool(results[0][2])},
               "correctness": [{"id": c, "name": n, "passed": bool(ok), "detail": d} for c, n, ok, d in results],
               "mutants": {"detected": int(found), "total": len(trial.caught), "score": found / len(trial.caught)},
               "hypotheses": hyps, "knockouts": lesions, "task_types": TASK_TYPES, "exit_code": code}
    extra = {"mode": mode, "caught": trial.caught, "bridge": bridge, "trivial": trivial_accuracy(trial.data),
             "sizes": {"lucretian": n_params(trial.engines["lucretian_fall"]),
                       "untuned baseline": n_params(trial.engines["untuned_fall"]),
                       "democritean rival": n_params(trial.engines["democritean_disorder"])},
             "engine_acc": {name: [float(np.mean([run["acc"][name][s] for run in trial.runs])) for s in ("heldout", "shifted")]
                            for name, *_ in ENGINES},
             "scale_end": float(np.mean([run["models"]["lucretian_fall"]["trace"]["mean_scale"][-1] for run in trial.runs])),
             "scale_matched": float(np.mean([run["models"]["untuned_fall"]["cfg"]["untuned_scale"] for run in trial.runs]))}
    write_report(render(payload, extra), payload, json_path)
    return code


CLI_FLAGS = (
    ("--quick", dict(action="store_true", help="one seed, reduced updates, all correctness tests")),
    ("--seed", dict(type=int, default=0, help="base seed")),
    ("--seeds", dict(type=int, default=None, help="number of seeds (default 5, or 1 with --quick)")),
    ("--json", dict(default=None, help="also write the JSON report to this path")),
    ("--card", dict(action="store_true", help="print MIND_CARD as JSON and exit")),
    ("--mutant", dict(default=None, help="run with a registered mutant: " + ", ".join(MUTANTS))),
    ("--data", dict(default=None, help="optional numeric CSV (header; integer label in last column)")),
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="Chapter 0105 Clinamen Engine: the full protocol runs by default.")
    for flag, options in CLI_FLAGS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    usage_errors = [message for broken, message in (
        (args.mutant is not None and args.mutant not in MUTANTS,
         f"unknown mutant {args.mutant!r}; registered: {', '.join(MUTANTS)}"),
        (n_seeds < 1, "--seeds must be at least 1")) if broken]
    if usage_errors:
        print(usage_errors[0], file=sys.stderr)
        return 2
    try:
        return under_mutant(args.mutant, lambda: protocol("quick" if args.quick else "full", args.seed, n_seeds,
                                                          args.json, args.data))
    except FloatingPointError as exc:
        print(f"non-finite values: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

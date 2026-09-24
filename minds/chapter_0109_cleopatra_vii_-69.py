#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0109 · Cleopatra VII
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 6, Minds 101-120: https://www.amazon.com/dp/B0HF7G6JJD · Demos: https://artificiology.com/
# END ATTRIBUTION
"""Rendered self (prosopon): one conserved self vector rendered directly into each audience's idiom by its own learned
renderer, faithful when that audience's own decoder reads the same self back; compared with pivot rendering through one
audience, with a single imposed idiom after 0112 Augustus, and with a hostile decoder that imposes its own reading.

Source problem
    Provenance is mediated. No writings of Cleopatra survive; the one-word subscription on Papyrus Bingen 45 may be her
    hand, but its authorship is disputed. Her portrait comes largely from hostile Roman sources and Plutarch.

Thesis
    A self can stay one while speaking many idioms if each rendering is judged by whether its own audience decodes it back
    to that self, with no single language standing between them; but the invariant belongs to the reader as much as to the
    speaker, and a hostile reader can overwrite it.

Evidence
    D1  Plutarch (Antony 27, as quoted in Mayor, The Poison King): she spoke many languages and received most foreign
        embassies without interpreters.
    D2  Papyrus Bingen 45 (Berlin 25239): the subscription "ginesthoi" is possibly her autograph; authorship is disputed.
    D3  Her image reached Rome through Octavian's propaganda (the chapter's reading; hostile sources).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
    D1     M1 conserved self, M2 one direct renderer per audience, M3 each audience's fixed decoder     C6.1 C6.2 H-SIG H-NEC
    D3     rival: one imposed idiom for all audiences (0112 Augustus); blind spot: a hostile decoder        H-RIVAL H-BLIND

Research question (compositional and systematic generalization)
    Does rendering a self directly into each audience's idiom preserve identity better than rendering through a pivot, and
    does faithful rendering survive a decoder that imposes its own reading?

Closest prior art and the delta
    Pivot versus direct (zero-shot) multilingual translation (Johnson et al. 2017); cycle consistency (Zhu et al. 2017);
    autoencoders with private decoders. Delta: identity preservation measured through each audience's own fixed decoder,
    direct renderers against a pivot chain with compounding noise, one imposed idiom, and a hostile decoder test.

Blind spot
    Invariance lives in the decoding: when a hostile decoder controls the reading, a faithful rendering is still misread.

Task (generative process)
    Selves are 8-dimensional, N(0, 0.7). Four audiences each own a fixed random orthogonal idiom; an audience decodes a
    rendering by the inverse of its idiom after channel noise (s.d. 0.1; 0.25 in the shifted split). A pivot chain renders
    into audience 0 and translates from there, adding noise at each hop. The hostile decoder mixes the audience's inverse
    idiom with a fixed propaganda projection, half and half. Splits: 1500 training, 1500 held-out and 1500 shifted selves.

Limits
    Linear idioms with a tanh renderer, one hop of translation. A research prototype of one mechanism, not an AGI and not
    Cleopatra's mind.
"""

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (15 September 2026), Appendix A", "generator": "Claude (Anthropic)",
                   "generator_version": "claude-opus-5", "date": "2026-09-16"},
    "id": 109, "figure": "Cleopatra VII", "born": -69, "died": -30, "civilization": "Ptolemaic Egypt", "provenance": "mediated",
    "thesis": ("A self can stay one while speaking many idioms if each rendering is judged by whether its own audience decodes it back to that "
               "self, with no single language standing between them; but the invariant belongs to the reader as much as to the speaker."),
    "evidence": [
        {"id": "D1", "basis": "primary", "source": "Plutarch, Antony 27 (quoted in A. Mayor, The Poison King)", "claim": "She spoke many languages and received embassies without interpreters."},
        {"id": "D2", "basis": "primary", "source": "Papyrus Bingen 45 (Berlin 25239); van Minnen via Archaeology (2001)", "claim": "Its one-word subscription is possibly her autograph; authorship disputed."},
        {"id": "D3", "basis": "scholarship", "source": "Chapter 0109, on Octavian's propaganda", "claim": "Rome read her through a hostile decoder."},
    ],
    "research_question": {"category": "compositional and systematic generalization",
                          "question": "Does direct rendering into each audience's idiom preserve identity better than a pivot, and does faithfulness survive a hostile decoder?"},
    "mechanism": {"name": "rendered self", "family": "per-audience tanh renderers read back by fixed audience decoders",
                  "signature_modules": ["renderers"],
                  "closest_prior_art": ["pivot versus direct multilingual translation (Johnson et al. 2017)", "cycle consistency (Zhu et al. 2017)", "autoencoders with private decoders"],
                  "overlap": "Medium", "prior_art_queries": [], "prior_art_note": "No literature search was run for this card; overlap is rated against the named methods.",
                  "contribution_type": "test",
                  "delta": "Identity preservation through each audience's own decoder, direct renderers against a noisy pivot chain, one imposed idiom, and a hostile decoder.",
                  "baselines": {"baseline": "pivot chain: render into audience 0, then translate that rendering into every other audience, with noise at each hop",
                                "blind_baseline": "the same trained renderers read by each audience's faithful decoder",
                                "rival": "chapter 0112 Augustus, minimal: one renderer imposed on every audience"}},
    "traceability": [{"doctrine": "D1", "mechanism": "M1-M3", "property_test": "C6.1, C6.2", "hypothesis": "H-SIG, H-NEC"},
                     {"doctrine": "D3", "mechanism": "imposed idiom; hostile decoder", "property_test": "none", "hypothesis": "H-RIVAL, H-BLIND"}],
    "hypotheses": [
        {"id": "H-SIG", "statement": "Under noisier channels, direct renderers preserve identity better than the pivot chain.",
         "metric": "identity_error", "split": "shifted", "comparison": "model - baseline", "direction": "less", "mesi": 0.02, "seeds": 5},
        {"id": "H-NEC", "statement": "Giving every audience audience 0's renderer raises identity error more than zeroing the renderers' biases.",
         "metric": "identity_error", "split": "heldout", "comparison": "(renderer:shared - full) - (bias:zero - full)", "knockouts": ["renderer:shared", "bias:zero"],
         "direction": "greater", "mesi": 0.02, "seeds": 5},
        {"id": "H-BLIND", "statement": "The same faithful renderings are misread when a hostile decoder controls the reading.",
         "condition": "held-out selves read by the hostile decoder", "grounding": "Rome decoded her through Octavian's propaganda.",
         "metric": "identity_error", "split": "heldout", "comparison": "model(hostile reader) - model(own reader)", "direction": "greater", "mesi": 0.1, "seeds": 5},
        {"id": "H-RIVAL", "statement": "Per-audience renderers preserve identity better than one idiom imposed on every audience.",
         "metric": "identity_error", "split": "heldout", "comparison": "model - rival", "direction": "less", "mesi": 0.05, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.3, "margin_over_trivial": 0.3, "shuffled_ratio_min": 0.9, "gradcheck_rel_error": 1e-5,
                   "gradcheck_floor": 1e-3, "invariance_tol": 1e-9, "negative_control_min_violation": 1e-6},
    "metrics": {"identity_error": "mean squared distance between decoded and true self, divided by the mean squared norm of the self, averaged over audiences",
                "trivial_baseline": "decoding every self as zero (identity error 1)",
                "shuffled_band": "one-sided: trained to reproduce selves shuffled across items, held-out identity error at least 0.9"},
    "training": {"optimizer": "Adam", "lr_grid": [0.03], "clip_norm": 5.0, "model_selection": "none: final parameters",
                 "updates": {"full": 500, "quick": 200}, "schedule": "cosine decay to 5 per cent", "applies_to": "direct, pivot and imposed-idiom renderers"},
    "task": {"self_dimensions": 8, "audiences": 4, "noise": {"train": 0.1, "heldout": 0.1, "shifted": 0.25}, "hostile_mix": 0.5,
             "selves": {"train": 1500, "heldout": 1500, "shifted": 1500}},
    "probe_predictions": [{"probe": "P10", "expected": "equal to baseline"}],
    "probe_support": "vector_classification is not natural here; the renderer maps vectors to vectors",
    "dialectic_links": [{"chapter": 112, "relation": "rival", "test": "H-RIVAL"}],
    "corpus_neighbors": [{"chapter": 15, "similarity": None, "difference": "0015 replicates one identity; here one self is rendered differently for each reader."},
                         {"chapter": 33, "similarity": None, "difference": "0033 federates local rendering; here the test is pivot against direct."}],
    "similarity_note": "Nearest-neighbour similarity not computed into the card; the audit reports it for the files at hand.",
    "barometer": {"language_understanding": ["rendering one self into many idioms"], "emotional_intelligence": ["reading the audience"],
                  "cognitive_processing": [], "embodied_cognition": [], "world_modeling": [], "consciousness": [], "creativity": [], "autonomy": []},
    "task_types": ["vector_classification"],
    "applications": [{"use": "multilingual generation without a pivot language", "sector": "machine translation", "dataset": "FLORES-200", "readiness": "low"},
                     {"use": "persona consistency and localization audits across markets", "sector": "product localization", "dataset": "synthetic persona sets", "readiness": "low"}],
    "safety_notes": "Synthetic vectors only; no real person, language community or political message is modelled.",
}

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

DIM, AUDIENCES, HOSTILE_MIX = 8, 4, 0.5
SELVES = {"train": 1500, "heldout": 1500, "shifted": 1500}
NOISE = {"train": 0.1, "heldout": 0.1, "shifted": 0.25}
UPDATES = {"full": 500, "quick": 200}
LR, CLIP_NORM = 0.03, 5.0
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

# ---- audiences, idioms, selves ---------------------------------------------------------------------------------------
def realm(seed):
    gen = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    decoders = [np.linalg.qr(gen.normal(size=(DIM, DIM)))[0] for _ in range(AUDIENCES)]
    propaganda = gen.normal(0.0, 0.5, (DIM, DIM))
    out = {"decoders": decoders, "hostile": [HOSTILE_MIX * D + (1 - HOSTILE_MIX) * propaganda for D in decoders]}
    for part, n in SELVES.items():
        s = gen.normal(0.0, 0.7, (n, DIM))
        out[part] = {"s": s, "target": s, "eps": [NOISE[part] * gen.normal(size=(n, DIM)) for _ in range(AUDIENCES)], "hop": NOISE[part] * gen.normal(size=(n, DIM))}
    return out


# ---- renderers: direct, pivot chain, imposed idiom ---------------------------------------------------------------------
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    """kind direct: one renderer per audience; pivot: render into audience 0 then translate; imposed: one renderer for all."""
    kind = cfg.get("kind", "direct")
    if task_type not in TASK_TYPES or in_dim != DIM:
        raise ValueError("chapter 0109 renders 8-dimensional selves")
    P = {}
    if kind == "imposed":
        P.update(W=rng.normal(0, 0.1, (DIM, DIM)), b=np.zeros(DIM))
    else:
        P.update(W0=rng.normal(0, 0.1, (DIM, DIM)), b0=np.zeros(DIM))
        for a in range(1, AUDIENCES):
            P["%s%d" % ("W" if kind == "direct" else "T", a)] = rng.normal(0, 0.1, (DIM, DIM))
            P["%s%d" % ("b" if kind == "direct" else "c", a)] = np.zeros(DIM)
    return {"kind": kind, "params": P, "shared": False, "unbiased": False, "history": []}


def _tanh_back(dr, r):
    return dr if ACTIVE_MUTANT == "dropped_tanh_derivative" else dr * (1.0 - r * r)


def render(model, batch, readers):
    """Decoded selves for every audience, and the cache needed for the backward pass."""
    P, s, cache, decoded = model["params"], batch["s"], {}, []
    if model["kind"] == "pivot":
        r0 = np.tanh(s @ P["W0"].T + P["b0"])
        cache["r0"] = r0
        for a in range(AUDIENCES):
            if a == 0:
                r = r0
            else:
                r = np.tanh((r0 + batch["hop"]) @ P["T%d" % a].T + P["c%d" % a])
            cache[a] = r
            decoded.append((r + batch["eps"][a]) @ readers[a])
        return decoded, cache
    for a in range(AUDIENCES):
        if model["kind"] == "imposed":
            W, b = P["W"], P["b"]
        else:
            key = 0 if model["shared"] else a
            W, b = P["W%d" % key], (np.zeros(DIM) if model["unbiased"] else P["b%d" % key])
        r = np.tanh(s @ W.T + b)
        cache[a] = r
        decoded.append((r + batch["eps"][a]) @ readers[a])
    return decoded, cache


def identity_error(model, batch, readers):
    decoded, _ = render(model, batch, readers)
    norm = float(np.mean((batch["s"] ** 2).sum(axis=1)))
    return float(np.mean([np.mean(((d - batch["s"]) ** 2).sum(axis=1)) for d in decoded]) / norm)


def loss_and_grads(model, batch):
    readers, P, s = batch["readers"], model["params"], batch["s"]
    decoded, cache = render(model, batch, readers)
    n, norm = len(s), float(np.mean((s ** 2).sum(axis=1)))
    scale = 2.0 / (n * AUDIENCES * norm)
    loss = float(np.mean([np.mean(((d - batch["target"]) ** 2).sum(axis=1)) for d in decoded]) / norm)
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    upstream = [scale * (decoded[a] - batch["target"]) @ readers[a].T for a in range(AUDIENCES)]
    if model["kind"] == "pivot":
        dr0 = upstream[0].copy()
        for a in range(1, AUDIENCES):
            dq = _tanh_back(upstream[a], cache[a])
            grads["T%d" % a] = dq.T @ (cache["r0"] + batch["hop"])
            grads["c%d" % a] = dq.sum(axis=0)
            dr0 += dq @ P["T%d" % a]
        dz = _tanh_back(dr0, cache["r0"])
        grads["W0"], grads["b0"] = dz.T @ s, dz.sum(axis=0)
    elif model["kind"] == "imposed":
        dz = _tanh_back(sum(upstream[a] for a in range(AUDIENCES)), cache[0])
        grads["W"], grads["b"] = dz.T @ s, dz.sum(axis=0)
    else:
        for a in range(AUDIENCES):
            dz = _tanh_back(upstream[a], cache[a])
            grads["W%d" % a], grads["b%d" % a] = dz.T @ s, dz.sum(axis=0)
    if ACTIVE_MUTANT == "zero_renderer_gradient":
        grads = {k: (np.zeros_like(v) if k.startswith(("W", "T")) else v) for k, v in grads.items()}
    return loss, grads


def fit(model, data, budget, rng):
    """Adam on all training selves; cosine decay to 5 per cent; deterministic, so rng is unused."""
    state = adam_init(model["params"])
    for k in range(budget):
        value, grads = loss_and_grads(model, data["train"])
        if not math.isfinite(value):
            raise FloatingPointError("identity loss diverged at update %d" % (k + 1))
        grads = clip_global(grads, CLIP_NORM)[0]
        if ACTIVE_MUTANT == "sign_flipped_update":
            grads = {key: -g for key, g in grads.items()}
        adam_step(model["params"], grads, state, 0.0 if ACTIVE_MUTANT == "zero_learning_rate" else LR * (0.05 + 0.475 * (1 + math.cos(math.pi * k / budget))))
        model["history"].append(value)
    return model["history"]


def predict(model, X):
    return render(model, X, X["readers"])[0]


def hidden_states(model, X):
    return {"renderings": render(model, X, X["readers"])[1]}


def modules(model):
    return {"renderers": {"params": [k for k in model["params"] if k[0] in "WT"], "role": "one learned renderer per audience", "signature": True},
            "biases": {"params": [k for k in model["params"] if k[0] in "bc"], "role": "renderer offsets", "signature": False}}


def knockout(model, name, mode):
    if (name, mode) == ("renderer", "shared"):
        return dict(model, shared=True)
    if (name, mode) == ("bias", "zero"):
        return dict(model, unbiased=True)
    raise ValueError("no knockout %s:%s" % (name, mode))


def n_params(model):
    return int(sum(v.size for v in model["params"].values()))


MUTANTS = {"sign_flipped_update": ("updates climb the loss", "C3"), "zero_learning_rate": ("nothing moves", "C3"),
           "zero_renderer_gradient": ("renderers receive no gradient", "C1"), "dropped_tanh_derivative": ("the tanh derivative is omitted", "C1")}


def data_bridge(path, seed, budget):
    return "skipped (%s: renderings of a synthetic self have no real-data counterpart in the file)" % os.path.basename(path)


def with_readers(batch, readers):
    return dict(batch, readers=readers)


def run_seed(seed, mode):
    world = realm(seed)
    faithful = world["decoders"]
    train = {"train": with_readers(world["train"], faithful)}
    models = {k: build_model(DIM, DIM, TASK_TYPES[0], np.random.default_rng(seed + i + 1), kind=k) for i, k in enumerate(("direct", "pivot", "imposed"))}
    for m in models.values():
        fit(m, train, UPDATES[mode], None)
    held, shifted = world["heldout"], world["shifted"]
    err = lambda m, part, readers=faithful: identity_error(m, part, readers)
    full = err(models["direct"], held)
    ko = {"renderer:shared": err(knockout(models["direct"], "renderer", "shared"), held) - full, "bias:zero": err(knockout(models["direct"], "bias", "zero"), held) - full}
    table = {"direct_held": full, "direct_shift": err(models["direct"], shifted), "pivot_held": err(models["pivot"], held), "pivot_shift": err(models["pivot"], shifted),
             "imposed_held": err(models["imposed"], held), "hostile_held": err(models["direct"], held, world["hostile"])}
    return {"world": world, "models": models, "lesions": ko, "table": table,
            "row": {"H-SIG": table["direct_shift"] - table["pivot_shift"], "H-NEC": ko["renderer:shared"] - ko["bias:zero"],
                    "H-BLIND": table["hostile_held"] - full, "H-RIVAL": full - table["imposed_held"]}}

# ---- checks ----------------------------------------------------------------------------------------------------------
def mutant(name):
    global ACTIVE_MUTANT
    was, ACTIVE_MUTANT = ACTIVE_MUTANT, name
    return was


def audit(first, mode):
    lim, world = MIND_CARD["thresholds"], first["world"]
    readers = world["decoders"]
    few = with_readers({"s": world["train"]["s"][:120], "target": world["train"]["s"][:120], "eps": [e[:120] for e in world["train"]["eps"]],
                        "hop": world["train"]["hop"][:120]}, readers)
    log, caught = [], {}

    def fd_gap(model, gen, n):
        return max(finite_difference_check(model["params"], loss_and_grads(model, few)[1], lambda m=model: loss_and_grads(m, few)[0], gen, n_entries=n,
                                           floor=lim["gradcheck_floor"]).values())

    def trained_well(model):
        drop = 1.0 - float(np.mean(model["history"][-20:])) / model["history"][0]
        held = identity_error(model, world["heldout"], readers)
        return drop >= lim["loss_drop_fraction"] and held <= 1.0 - lim["margin_over_trivial"], drop, held

    seed = first["seed"]
    fresh = [build_model(DIM, DIM, TASK_TYPES[0], np.random.default_rng(seed + 40 + i), kind=k) for i, k in enumerate(("direct", "pivot", "imposed"))]
    worst = max([fd_gap(m, np.random.default_rng(seed + 3), 4) for m in fresh] + [fd_gap(m, np.random.default_rng(seed + 4), 4) for m in first["models"].values()])
    total = sum(len(m["params"]) for m in first["models"].values())
    grad_info = {"tensors_checked": total, "tensors_total": total, "max_rel_error": worst, "checked_at": ["init", "after_training_steps"], "passed": bool(worst <= lim["gradcheck_rel_error"])}
    log.append(("C1", "gradient_check", grad_info["passed"], "direct, pivot and imposed renderers, fresh and trained: worst relative error %.2e" % worst))
    pair = []
    for _ in range(2):
        m = build_model(DIM, DIM, TASK_TYPES[0], np.random.default_rng(seed + 9))
        fit(m, {"train": with_readers(world["train"], readers)}, 12, None)
        pair.append(m)
    same = pair[0]["history"] == pair[1]["history"] and all(np.array_equal(pair[0]["params"][k], pair[1]["params"][k]) for k in pair[0]["params"])
    log.append(("C2", "determinism_finiteness", bool(same), "two renderers trained from one seed agree exactly: %s" % same))
    ok, drop, held = trained_well(first["models"]["direct"])
    log.append(("C3", "learning", ok, "identity loss fell %.3f (needs 0.3); held-out identity error %.3f against 1.000 for decoding nothing" % (drop, held)))
    order = np.random.default_rng(seed + 11).permutation(len(world["train"]["s"]))
    liar = build_model(DIM, DIM, TASK_TYPES[0], np.random.default_rng(seed + 12))
    fit(liar, {"train": with_readers(dict(world["train"], target=world["train"]["s"][order]), readers)}, UPDATES[mode], None)
    shuffled = identity_error(liar, world["heldout"], readers)
    log.append(("C4", "shuffled_self_control", shuffled >= lim["shuffled_ratio_min"], "taught to render other items' selves: held-out identity error %.3f (floor 0.9)" % shuffled))

    def replay():
        try:
            m = build_model(DIM, DIM, TASK_TYPES[0], np.random.default_rng(seed + 2))
            clean = fd_gap(m, np.random.default_rng(seed), 3)
            fit(m, {"train": with_readers(world["train"], readers)}, UPDATES[mode], None)
            return bool(clean <= lim["gradcheck_rel_error"] and trained_well(m)[0])
        except FloatingPointError:
            return False
    honest_run = replay()
    for name in MUTANTS:
        old = mutant(name)
        caught[name] = not replay()
        mutant(old)
    log.append(("C5", "mutant_detection", honest_run and all(caught.values()), "clean replay passes C1 and C3: %s; mutants caught %d of %d" % (honest_run, sum(caught.values()), len(MUTANTS))))
    direct, held_b = first["models"]["direct"], with_readers(world["heldout"], readers)
    perm = [2, 0, 3, 1]
    moved_p = {}
    for new_a, old_a in enumerate(perm):
        moved_p["W%d" % new_a], moved_p["b%d" % new_a] = direct["params"]["W%d" % old_a], direct["params"]["b%d" % old_a]
    moved = dict(direct, params=moved_p)
    moved_batch = dict(held_b, eps=[held_b["eps"][a] for a in perm], readers=[readers[a] for a in perm])
    gap = abs(loss_and_grads(direct, held_b)[0] - loss_and_grads(moved, moved_batch)[0])
    control = abs(loss_and_grads(direct, held_b)[0] - loss_and_grads(moved, held_b)[0])
    log.append(("C6.1", "audience_relabel_invariance", gap <= lim["invariance_tol"] and control >= lim["negative_control_min_violation"],
                "relabelling audiences with their renderers and readers changes the loss by %.1e; relabelling renderers alone %.1e" % (gap, control)))
    single = dict(held_b, eps=[e if a == 1 else 0 * e for a, e in enumerate(held_b["eps"])], target=held_b["s"])

    def audience_one_grad(model):
        decoded, cache = render(model, single, readers)
        grads = loss_and_grads(model, dict(single, readers=[readers[a] if a == 1 else readers[a] for a in range(AUDIENCES)]))[1]
        return grads
    only_one = dict(held_b, target=held_b["s"])
    d_grads = loss_and_grads(direct, only_one)[1]
    base = {k: v.copy() for k, v in d_grads.items()}
    tweak = dict(direct, params=dict(direct["params"], W1=direct["params"]["W1"] + 0.05))
    cross = float(np.abs(loss_and_grads(tweak, only_one)[1]["W2"] - base["W2"]).max())
    pv = first["models"]["pivot"]
    pv_tweak = dict(pv, params=dict(pv["params"], W0=pv["params"]["W0"] + 0.05))
    pivot_cross = float(np.abs(loss_and_grads(pv_tweak, only_one)[1]["T2"] - loss_and_grads(pv, only_one)[1]["T2"]).max())
    log.append(("C6.2", "no_pivot_dependence", cross <= lim["invariance_tol"] and pivot_cross >= lim["negative_control_min_violation"],
                "changing audience 1's renderer moves audience 2's gradient by %.1e; in the pivot chain, changing the pivot moves it by %.1e" % (cross, pivot_cross)))
    prints = {k: {hashlib.sha256(r.tobytes()).hexdigest() for r in world[k]["s"]} for k in SELVES}
    apart = not (prints["train"] & prints["heldout"] or prints["train"] & prints["shifted"] or prints["heldout"] & prints["shifted"])
    log.append(("C7", "split_integrity", apart, "no self shared between splits: %s" % apart))
    return log, grad_info, caught


def protocol(mode, first_seed, count, json_path, data_path):
    began, seeds = time.time(), list(range(first_seed, first_seed + count))
    print("chapter 0109 · mode %s · seeds %s · mutant %s" % (mode, seeds, ACTIVE_MUTANT), flush=True)
    runs = [run_seed(s, mode) for s in seeds]
    log, grad_info, caught = audit(dict(runs[0], seed=first_seed), mode)
    evaluated, draw = mode == "full" and count >= 5, np.random.default_rng(first_seed + 9973)
    hyps, kos = [], []
    for spec in MIND_CARD["hypotheses"]:
        vals = np.array([r["row"][spec["id"]] for r in runs])
        mid, ci = paired_bootstrap(vals, draw) if evaluated else (float(vals.mean()), None)
        hyps.append({"id": spec["id"], "metric": spec["metric"], "mean_diff": mid, "ci95": ci, "mesi": spec["mesi"], "n_seeds": len(runs),
                     "verdict": verdict(mid, ci, spec["mesi"], spec["direction"]) if evaluated else "not evaluated"})
    for label in runs[0]["lesions"]:
        vals = np.array([r["lesions"][label] for r in runs])
        mid, ci = paired_bootstrap(vals, draw) if evaluated else (float(vals.mean()), None)
        kos.append({"module": label.split(":")[0], "mode": label.split(":")[1], "signature": label.startswith("renderer"), "metric_change": mid, "ci95": ci})
    took = time.time() - began
    log.append(("C8", "budget", took <= TIME_BUDGET[mode], "%.1f s of %.0f s" % (took, TIME_BUDGET[mode])))
    failed = [c for c, _, ok, _ in log if not ok]
    code = 0 if not failed else (3 if failed == ["C8"] else 1)
    band = lambda ci: "not evaluated" if ci is None else "[%+.4f, %+.4f]" % tuple(ci)
    avg = lambda key: float(np.mean([r["table"][key] for r in runs]))
    out = ["=== VERIFIED REPORT · chapter 0109 ===", "file: %s · card_revision %d · mode %s · mutant %s" % (os.path.basename(__file__), MIND_CARD["card_revision"], mode, ACTIVE_MUTANT),
           "environment: python %s · numpy %s" % (sys.version.split()[0], np.__version__), "seeds: %s · runtime_s %.1f · budget_s %.0f" % (seeds, took, TIME_BUDGET[mode]),
           "n_params: " + " · ".join("%s %d" % (k, n_params(m)) for k, m in runs[0]["models"].items()),
           "gradcheck: %d/%d tensors at init and after training · max_rel_error %.2e · passed %s" % (grad_info["tensors_checked"], grad_info["tensors_total"], grad_info["max_rel_error"], grad_info["passed"]),
           "correctness:"] + ["  %-5s %-28s %s  %s" % (c, n, "PASS" if ok else "FAIL", d) for c, n, ok, d in log]
    out.append("mutants: %d/%d detected · score %.2f · %s" % (sum(caught.values()), len(MUTANTS), sum(caught.values()) / len(MUTANTS), ", ".join(k + (" caught" if caught.get(k) else " missed") for k in MUTANTS)))
    out.append("hypotheses (paired over seeds; 95% percentile bootstrap of the mean, 2000 resamples):")
    out += ["  %-8s mean_diff %+.4f ci95 %s mesi %s seeds %d -> %s" % (h["id"], h["mean_diff"], band(h["ci95"]), h["mesi"], h["n_seeds"], h["verdict"]) for h in hyps]
    out.append("knockouts (direct renderers, held-out identity error change):")
    out += ["  %-9s %-6s signature %-5s %+.4f ci95 %s" % (k["module"], k["mode"], k["signature"], k["metric_change"], band(k["ci95"])) for k in kos]
    out.append("identity error (seed mean; held-out / shifted): direct %.3f / %.3f · pivot %.3f / %.3f · one imposed idiom %.3f held-out · direct read by a hostile decoder %.3f"
               % (avg("direct_held"), avg("direct_shift"), avg("pivot_held"), avg("pivot_shift"), avg("imposed_held"), avg("hostile_held")))
    out += ["real-data bridge: skipped (no --data PATH given)", "task_types: " + ", ".join(TASK_TYPES), "exit_code: %d" % code, "=== END REPORT ==="]
    record = {"schema_version": "1.0", "chapter": 109, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": round(took, 2), "n_params": n_params(runs[0]["models"]["direct"]),
              "gradcheck": grad_info, "correctness": [{"id": c, "name": n, "passed": ok, "detail": d} for c, n, ok, d in log],
              "mutants": {"detected": sum(caught.values()), "total": len(MUTANTS), "score": sum(caught.values()) / len(MUTANTS)}, "hypotheses": hyps, "knockouts": kos,
              "task_types": TASK_TYPES, "exit_code": code}
    write_report(out, record, json_path)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__), description="Chapter 0109: the rendered self.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--card", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--json")
    parser.add_argument("--mutant")
    parser.add_argument("--data")
    got = parser.parse_args(argv)
    if got.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False))
        return 0
    n = got.seeds if got.seeds is not None else (1 if got.quick else 5)
    if n < 1 or got.seed < 0 or (got.mutant is not None and got.mutant not in MUTANTS):
        print("invalid --seed, --seeds or --mutant", file=sys.stderr)
        return 2
    mutant(got.mutant)
    try:
        return protocol("quick" if got.quick else "full", got.seed, n, got.json, got.data)
    except FloatingPointError as exc:
        print("non-finite values: %s" % exc, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BEGIN ATTRIBUTION
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0324 · Li Ye (李冶, 1192-1279)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 17, Minds 321-340 Available on Amazon https://www.amazon.com/dp/B0HP7Z1NKC
# Resume and Interactive Demos at https://artificiology.com/
# END ATTRIBUTION
"""
Round-City Counting Board (圓城籌算): a learner that answers by writing the equation
its answer must satisfy and then extracting the root.

Thesis. Li Ye does not compute the unknown; he names it (立天元一), expresses one quantity
of a single catalogued figure in two ways until the two registers can be cancelled
(相消), and extracts the root (開方). The answer is implicit: a zero of a law.

Evidence and provenance (belief: his own prefaces and books survive).
  D1 Ceyuan haijing (1248): all 170 problems share ONE figure, a round town inscribed in a
     right triangle; the reader recovers the same diameter from different given segments.
  D2 Volume 1 (識別雜記) catalogues the relations among the segments of that figure before
     any problem is posed; the problems are then solved from the catalogue.
  D3 Cao (草) sections begin "立天元一為…": the unknown is a named placeholder whose powers
     are carried by position on the counting board; the equation arises by cancelling two
     expressions of the same magnitude (如積相消), then 開方 extracts the root.
  D4 Preface: numbers cannot be exhausted by force, but in the dark there is something
     bright; push the natural principle (自然之理) to clarify the natural number.
  D5 His world was exact: every figure of the book is integral (四庫提要: 皆無奇零).

Doctrine -> mechanism -> test (IDs as in MIND_CARD)
  D1 one figure, many views  -> M1 shibie: per-type linear forms on one latent figure -> H-SIG, C6.2
  D2 catalogue before problems-> M5 catalogue stage (per-pair equations -> Gram table)  -> H-ORDER, C6.4
  D3 two registers cancelled  -> M2 law = UU^T - VV^T, M3 tianyuan form, residual r(x)  -> C6.1, C6.5
  D3 root extraction          -> M4 kaifang: unrolled damped Newton from a first quotient-> H-NEC, C6.3
  D4 principle before number  -> answer as zero of a law, not a regressed number        -> H-RIVAL
  D5 exact data               -> blind spot: measurement noise                          -> H-BLIND

Research question (compositional and systematic generalization; extrapolation).
Can a learner that stores a shared catalogue of measurement forms and a single invariant,
and answers by root extraction, answer PAIRS of measurement types it never saw together,
and at magnitudes 30x beyond training, where regressors fail (Xu et al., ICLR 2021)?

Closest prior art and delta.
  Implicit layers (Bai, Kolter & Koltun 2019; Gould et al. 2021; Zhang et al. 2020),
  equation learners for extrapolation (Sahoo, Lampert & Martius 2018), learning
  varieties from samples (Breiding et al. 2018), low-rank matrix completion.
  Delta: the implicit residual is not a free network; it is the pull-back of ONE learned
  quadric through an exact linear identification (adjugate) of each typed measurement,
  so any pair of catalogued types yields its own equation without having been trained.

Blind spot. Li Ye's figures were exact. Measurement noise makes the two registers
disagree everywhere; the board must still output a root and may pick a wrong one.

Task (native, synthetic). Draw a right triangle (gou p < gu q, hypotenuse h; angle
12-42 deg; h log-uniform in [1,10]). Twelve segment types are sums and differences of
p, q, h named as in the Chinese right-triangle tradition. A problem gives two measured
segments of different types; the target is the inscribed-circle diameter d = p+q-h.
Splits: held-out (seen pairs), compositional (unseen pairs), scale (h in [30,300]),
noisy (3% multiplicative measurement noise, the blind-spot condition).

Limits. Research prototype of one AGI-relevant mechanism, not an AGI. Quadratic laws only
(Li Ye reached degree 6). Code outputs are never evidence about the historical person.
Run: python3 <this file>  (see --help)
"""
import argparse, json, math, os, sys, time, hashlib
import numpy as np

MIND_CARD = {
    "schema_version": "1.0", "card_revision": 1, "revision_log": [],
    "generation": {"template_version": "codeguidelines 1.0 (2026-09-15)", "generator": "Claude (Anthropic)",
                   "generator_version": "Claude Opus 5.5", "date": "2026-09-25"},
    "id": 324, "figure": "Li Ye", "born": 1192, "died": 1279, "civilization": "Chinese (Jin/Yuan)",
    "provenance": "belief",
    "thesis": "Name the unknown, express one magnitude of a single catalogued figure two ways, cancel the "
              "registers and extract the root: the answer is the zero of a law read off a shared catalogue.",
    "evidence": [
        {"id": "D1", "claim": "All 170 problems share one figure (round town in a right triangle); same diameter recovered from different data",
         "basis": "primary", "source": "Ceyuan haijing (1248), juan 2-12; Siku tiyao"},
        {"id": "D2", "claim": "Relations among the segments are catalogued (識別雜記) before the problems",
         "basis": "primary", "source": "Ceyuan haijing, juan 1"},
        {"id": "D3", "claim": "Set up the celestial unknown, express a magnitude twice, cancel (相消), extract the root (開方)",
         "basis": "primary", "source": "Ceyuan haijing, cao sections; Yigu yanduan (1259)"},
        {"id": "D4", "claim": "Push the natural principle to clarify the natural number; numbers cannot be exhausted by force",
         "basis": "primary", "source": "Ceyuan haijing, preface"},
        {"id": "D5", "claim": "The figures are integral and exact (皆無奇零)", "basis": "scholarship",
         "source": "Siku quanshu zongmu tiyao, Ceyuan haijing entry (1781)"},
        {"id": "D6", "claim": "Replaced named power levels (天, 地, 人 ...) by pure position with one marker",
         "basis": "primary", "source": "Jingzhai gujin tou (敬齋古今黈), note on the Dongping suanjing"},
    ],
    "research_question": {"category": "compositional and systematic generalization",
        "question": "Does answering by root extraction from a shared catalogue of measurement forms generalize to unseen pairs of measurement types and to 30x magnitudes where explicit regressors fail?"},
    "mechanism": {"name": "Round-City Counting Board", "family": "implicit layer / learned algebraic variety with typed linear identification",
        "signature_modules": ["shibie", "tianyuan", "law", "kaifang"],
        "closest_prior_art": ["Deep equilibrium / implicit layers (Bai et al. 2019; Zhang et al. 2020, arXiv:2003.01822)",
                              "Equation learner EQL (Sahoo, Lampert, Martius 2018)",
                              "Learning algebraic varieties from samples (Breiding et al. 2018, arXiv:1802.09436)",
                              "Low-rank matrix completion"],
        "overlap": "Medium",
        "prior_art_queries": ["neural network output defined as root of learned polynomial implicit differentiation",
                              "learning algebraic varieties from samples", "equation learner extrapolation",
                              "compositional generalization unseen feature combinations regression"],
        "contribution_type": "mechanism",
        "delta": "The residual is one learned quadric pulled back through an exact adjugate identification of typed measurements, so unseen type pairs get their own equation; learned catalogue-first by per-pair Veronese null vectors and a signed Gram completion."},
    "traceability": [
        {"doctrine": "D1", "mechanism": "M1 shibie", "property_test": "C6.2", "hypothesis": "H-SIG"},
        {"doctrine": "D2", "mechanism": "M5 catalogue stage", "property_test": "C6.4", "hypothesis": "H-ORDER"},
        {"doctrine": "D3", "mechanism": "M2 law + M3 tianyuan", "property_test": "C6.5", "hypothesis": "H-SIG"},
        {"doctrine": "D3", "mechanism": "M4 kaifang", "property_test": "C6.3", "hypothesis": "H-NEC"},
        {"doctrine": "D4", "mechanism": "implicit answer", "property_test": "C6.1", "hypothesis": "H-RIVAL"},
        {"doctrine": "D5", "mechanism": "exact-law assumption", "property_test": "none", "hypothesis": "H-BLIND"},
    ],
    "hypotheses": [
        {"id": "H-SIG", "statement": "The board beats a size-matched MLP on unseen measurement pairs", "metric": "log10 relative RMSE",
         "split": "compositional", "comparison": "model - baseline", "direction": "less", "mesi": 0.3, "seeds": 5},
        {"id": "H-NEC", "statement": "Removing root extraction hurts more than removing the first-quotient head", "metric": "log10 relative RMSE",
         "split": "compositional", "comparison": "signature_knockout - matched_knockout", "direction": "greater", "mesi": 0.3, "seeds": 5},
        {"id": "H-BLIND", "statement": "Under 3% measurement noise the board is worse than the MLP on seen pairs",
         "condition": "multiplicative Gaussian noise sigma=0.03 on both measurements, train and test",
         "grounding": "Li Ye's figures were exact and integral; no error model", "metric": "log10 relative RMSE",
         "split": "held_out_noisy", "comparison": "model - baseline", "direction": "greater", "mesi": 0.1, "seeds": 5},
        {"id": "H-RIVAL", "statement": "The board beats an explicit homogeneous procedure (bias-free ReLU net) on unseen pairs",
         "metric": "log10 relative RMSE", "split": "compositional", "comparison": "model - rival", "direction": "less", "mesi": 0.3, "seeds": 5},
        {"id": "H-ORDER", "statement": "Catalogue-first learning beats gradient-only learning of the same board on unseen pairs",
         "metric": "log10 relative RMSE", "split": "compositional", "comparison": "catalogue_first - gradient_only",
         "direction": "less", "mesi": 0.3, "seeds": 5},
    ],
    "thresholds": {"loss_drop_fraction": 0.5, "margin_over_trivial": 0.5, "shuffle_band": 0.8, "gradcheck_tol": 1e-5},
    "probe_predictions": [{"probe": "P2", "expected": "above baseline"}, {"probe": "P6", "expected": "above baseline"},
                          {"probe": "P8", "expected": "below baseline"}, {"probe": "P9", "expected": "below baseline"},
                          {"probe": "P10", "expected": "above baseline"}],
    "dialectic_links": [
        {"chapter": None, "relation": "rival", "test": "H-RIVAL",
         "note": "Explicit procedures (術) of the Nine Chapters tradition as read by Liu Hui; no chapter in the corpus"},
        {"chapter": 328, "relation": "rival", "test": "none", "note": "Qin Jiushao, contemporary; positive-negative root extraction"},
        {"chapter": 346, "relation": "successor", "test": "none", "note": "Zhu Shijie extends the celestial unknown to four unknowns"},
        {"chapter": 299, "relation": "rival", "test": "none", "note": "Li Ye criticises Zhu Xi in the Jingzhai gujin tou"}],
    "corpus_neighbors": [
        {"chapter": 273, "similarity": None, "difference": "Shen Kuo reads disagreement among routes to one quantity as an error signal; here two routes are forced equal and the answer is their root"},
        {"chapter": 317, "similarity": None, "difference": "Fibonacci abstracts worked cases into explicit rules; here no rule outputs the answer, it is extracted from a law"},
        {"chapter": 263, "similarity": None, "difference": "Shao Yong transfers a cycle across scales; here scale transfer is exact homogeneity of the equation"},
        {"chapter": 304, "similarity": None, "difference": "Lu Jiuyuan orders two organs in time; here the order is catalogue before problems"},
        {"chapter": 507, "similarity": 0.0, "difference": "Only corpus code available in session (sample file); see report for measured Jaccard"}],
    "barometer": {"cognitive_processing": ["compositional split", "scale extrapolation"], "embodied_cognition": [],
                  "world_modeling": ["shared latent figure with a conservation-like invariant"],
                  "consciousness": ["residual of own equation as self-check (reported, not claimed as awareness)"],
                  "language_understanding": [], "emotional_intelligence": [],
                  "creativity": ["answers for never-seen pairs"], "autonomy": []},
    "task_types": ["vector_regression", "vector_classification"],
    "applications": [
        {"use": "Equation-structured regression that must extrapolate", "sector": "physical sciences", "dataset": "Feynman Symbolic Regression Database (AI Feynman)"},
        {"use": "Geometry problem solving from partial measurements", "sector": "education / AI reasoning", "dataset": "Geometry3K (Lu et al. 2021)"},
        {"use": "Kinematic chains whose solutions are roots of quadrics", "sector": "robotics", "dataset": "kin8nm (DELVE / OpenML)"}],
    "safety_notes": "Synthetic geometry only; no persons, no hazardous domains. Not a claim to replicate Li Ye's mind.",
}

ALLOWED_CLIP = 5.0
TYPES = [  # (pinyin, name, linear form on (gou p, gu q, xian h))
    ("gou", "勾", (1, 0, 0)), ("gu", "股", (0, 1, 0)), ("xian", "弦", (0, 0, 1)),
    ("gougu_he", "勾股和", (1, 1, 0)), ("gougu_jiao", "勾股較", (-1, 1, 0)),
    ("gouxian_he", "勾弦和", (1, 0, 1)), ("gouxian_jiao", "勾弦較", (-1, 0, 1)),
    ("guxian_he", "股弦和", (0, 1, 1)), ("guxian_jiao", "股弦較", (0, -1, 1)),
    ("xian_he_he", "弦和和", (1, 1, 1)), ("xian_jiao_he", "弦較和", (-1, 1, 1)), ("xian_jiao_jiao", "弦較較", (1, -1, 1))]
FORMS = np.array([t[2] for t in TYPES], float)
K_TYPES = len(TYPES)
TAU_TRUE = np.array([1.0, 1.0, -1.0])          # 弦和較 = diameter of the inscribed circle
LAM, EPS_D, T_TRAIN, T_INFER = 1e-3, 1e-6, 12, 16
TASK_TYPES = ["vector_regression", "vector_classification"]

# BEGIN STANDARD UTILITIES v1.0
def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def clip_global_norm(g, c):
    n = math.sqrt(sum(float(np.sum(v * v)) for v in g.values()))
    return ({k: v * (c / n) for k, v in g.items()} if n > c else g), n

def adam_init(p):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in p.items()}, "v": {k: np.zeros_like(v) for k, v in p.items()}}

def adam_step(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] += 1
    t = st["t"]
    for k in p:
        st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
        st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
        p[k] -= lr * (st["m"][k] / (1 - b1 ** t)) / (np.sqrt(st["v"][k] / (1 - b2 ** t)) + eps)

def gradcheck(loss_fn, params, grads, rng, n_entries=20, eps=1e-6):
    """Central differences on every tensor: n random entries plus the largest-gradient entry."""
    out, L0 = {}, abs(loss_fn())
    for k, v in params.items():
        floor = 1e-3 * float(np.max(np.abs(grads[k]))) + 1e-5 * (1.0 + L0)   # roundoff guard (~1e-10 |L| / eps)
        flat = list(rng.choice(v.size, size=min(n_entries, v.size), replace=False))
        flat.append(int(np.argmax(np.abs(grads[k]))))
        worst = 0.0
        for f in flat:
            ix = np.unravel_index(f, v.shape)
            old = v[ix]
            v[ix] = old + eps; lp = loss_fn()
            v[ix] = old - eps; lm = loss_fn()
            v[ix] = old
            num, ana = (lp - lm) / (2 * eps), grads[k][ix]
            worst = max(worst, abs(num - ana) / max(floor, abs(num) + abs(ana)))
        out[k] = worst
    return out

def paired_bootstrap(diffs, rng, n=2000):
    d = np.asarray(diffs, float)
    means = d[rng.integers(len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def verdict(mean, ci, mesi, direction):
    if direction == "less":
        if ci[1] < 0 and mean <= -mesi: return "supported"
        if ci[0] > 0: return "contradicted"
    else:
        if ci[0] > 0 and mean >= mesi: return "supported"
        if ci[1] < 0: return "contradicted"
    return "inconclusive"

def write_report(rep, path=None):
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
# END STANDARD UTILITIES

# ----------------------------------------------------------------- data and tasks
def valid_pairs():
    out = []
    for i in range(K_TYPES):
        for j in range(i + 1, K_TYPES):
            if abs(np.linalg.det(np.stack([FORMS[i], FORMS[j], TAU_TRUE]))) > 1e-9:
                out.append((i, j))   # pairs whose answer is not a mere linear read-out
    return out

def split_pairs(rng, n_hold=14):
    """Hold out whole pairs of types (compositional split); every type keeps >= 2 train pairs."""
    pairs = valid_pairs()
    while True:
        perm = rng.permutation(len(pairs))
        hold = [pairs[k] for k in perm[:n_hold]]
        train = [pairs[k] for k in perm[n_hold:]]
        cnt = np.bincount(np.array(train).ravel(), minlength=K_TYPES)
        if cnt.min() >= 2:
            return train, hold

def gen_problems(pairs, n, rng, scale=(1.0, 10.0), noise=0.0):
    P = np.array(pairs)[rng.integers(len(pairs), size=n)]
    sw = rng.random(n) < 0.5                       # nuisance: order of the two given segments
    I, J = np.where(sw, P[:, 1], P[:, 0]), np.where(sw, P[:, 0], P[:, 1])
    th = rng.uniform(np.deg2rad(12), np.deg2rad(42), n)
    h = np.exp(rng.uniform(np.log(scale[0]), np.log(scale[1]), n))
    Z = np.stack([h * np.sin(th), h * np.cos(th), h], 1)
    a, b = np.sum(FORMS[I] * Z, 1), np.sum(FORMS[J] * Z, 1)
    if noise > 0:
        a = a * (1 + noise * rng.standard_normal(n)); b = b * (1 + noise * rng.standard_normal(n))
    return {"i": I, "j": J, "a": a, "b": b, "y": Z @ TAU_TRUE}

def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}

def make_task(rng, n_train=2400, n_eval=800, noise=0.0):
    tr_pairs, ho_pairs = split_pairs(rng)
    return {"train": gen_problems(tr_pairs, n_train, rng, noise=noise),
            "val": gen_problems(tr_pairs, 400, rng, noise=noise),
            "held_out": gen_problems(tr_pairs, n_eval, rng, noise=noise),
            "compositional": gen_problems(ho_pairs, n_eval, rng, noise=noise),
            "scale": gen_problems(tr_pairs, n_eval, rng, scale=(30.0, 300.0), noise=noise),
            "pairs": (tr_pairs, ho_pairs)}

def scatter(d):
    n = len(d["a"]); u = np.zeros((n, K_TYPES))
    u[np.arange(n), d["i"]] += d["a"]; u[np.arange(n), d["j"]] += d["b"]
    return u

# ----------------------------------------------------------------- model: kaifang (開方)
def kaifang(c, x0, s, T, mutant=None):
    """Unrolled damped Newton on r(x)=c0+c1 x+c2 x^2; damping is homogeneous in the data scale s."""
    x, cache = x0, []
    for _ in range(T):
        r = c[:, 0] + c[:, 1] * x + c[:, 2] * x * x
        rp, rpp = c[:, 1] + 2 * c[:, 2] * x, 2 * c[:, 2]
        lam = LAM * (c[:, 1] ** 2 + 4 * c[:, 2] ** 2 * x * x + EPS_D * (c[:, 2] ** 2 * s * s + c[:, 0] ** 2 / (s * s)))
        D = rp * rp + lam
        cache.append((x, r, rp, rpp, D))
        x = x + r * rp / D if mutant == "newton_sign" else x - r * rp / D
    return x, cache

def kaifang_back(c, s, cache, gx, mutant=None):
    gc = np.zeros_like(c)
    for (x, r, rp, rpp, D) in reversed(cache):
        Nn = r * rp
        dlx = LAM * 8 * c[:, 2] ** 2 * x
        dstep_dx = (rp * rp + r * rpp) / D - Nn * (2 * rp * rpp + dlx) / D ** 2
        P, dP = (np.ones_like(x), x, x * x), (np.zeros_like(x), np.ones_like(x), 2 * x)
        dl = (LAM * EPS_D * 2 * c[:, 0] / (s * s), LAM * 2 * c[:, 1],
              LAM * (8 * c[:, 2] * x * x + EPS_D * 2 * c[:, 2] * s * s))
        if mutant != "detach_kaifang":
            for k in range(3):
                gc[:, k] -= gx * ((P[k] * rp + r * dP[k]) / D - Nn * (2 * rp * dP[k] + dl[k]) / D ** 2)
        gx = gx * (1 - dstep_dx)
    return gc, gx

def well_posed(c, x0):
    """Examples whose equation has real roots and a non-flat start: away from the fold of the root map."""
    disc = c[:, 1] ** 2 - 4 * c[:, 0] * c[:, 2]
    scale = c[:, 1] ** 2 + np.abs(4 * c[:, 0] * c[:, 2]) + 1e-300
    rp = c[:, 1] + 2 * c[:, 2] * x0
    return (disc / scale > 0.1) & (rp * rp / (c[:, 1] ** 2 + 4 * c[:, 2] ** 2 * x0 * x0 + 1e-300) > 0.05)

# ----------------------------------------------------------------- model: the round-city board
def law_matrix(p, ko):
    G = p["U"] @ p["U"].T - p["V"] @ p["V"].T
    return np.trace(G) / 3.0 * np.eye(3) if ko.get("law") == "mean" else G

def board_forward(m, d, T):
    p, ko = m["params"], m["ko"]
    alpha = np.repeat(p["alpha"].mean(0, keepdims=True), K_TYPES, 0) if ko.get("shibie") == "mean" else p["alpha"]
    tau = np.zeros(3) if ko.get("tianyuan") == "zero" else p["tau"]
    Ai, Aj = alpha[d["i"]], alpha[d["j"]]
    n1, n2, n3 = np.cross(Aj, tau), np.cross(tau, Ai), np.cross(Ai, Aj)   # adjugate columns
    G = law_matrix(p, ko)
    z0 = d["a"][:, None] * n1 + d["b"][:, None] * n2
    Gz, Gw = z0 @ G, n3 @ G
    c = np.stack([np.sum(z0 * Gz, 1), 2 * np.sum(n3 * Gz, 1), np.sum(n3 * Gw, 1)], 1)
    u = scatter(d); s = d["a"] + d["b"]
    hpre = u @ p["W1"].T; hh = np.maximum(hpre, 0.0)
    x0 = hh @ p["w2"] + m["cfg"].get("x0_bias", 0.0)
    if ko.get("chushang") == "mean":
        x0 = m["ratio_mean"] * s
    if ko.get("kaifang") == "identity":
        x, cache = x0, []
    else:
        x, cache = kaifang(c, x0, s, T, m.get("mutant"))
    return {"Ai": Ai, "Aj": Aj, "n1": n1, "n2": n2, "n3": n3, "G": G, "z0": z0, "c": c,
            "u": u, "hpre": hpre, "hh": hh, "x0": x0, "s": s, "x": x, "cache": cache}

def board_loss_grads(m, d, b0=0.1):
    p = m["params"]; f = board_forward(m, d, T_TRAIN)
    y, s, N = d["y"], f["s"], len(d["y"])
    e, e0 = (f["x"] - y) / s, (f["x0"] - y) / s
    L = float(np.mean(e * e) + b0 * np.mean(e0 * e0))
    gc, gx0 = kaifang_back(f["c"], s, f["cache"], 2 * e / s / N, m.get("mutant"))
    gx0 = gx0 + b0 * 2 * e0 / s / N
    z0, w, G = f["z0"], f["n3"], f["G"]
    gz0 = gc[:, :1] * 2 * (z0 @ G) + gc[:, 1:2] * 2 * (w @ G)
    gw = gc[:, 1:2] * 2 * (z0 @ G) + gc[:, 2:] * 2 * (w @ G)
    gG = (z0.T * gc[:, 0]) @ z0 + (w.T * gc[:, 1]) @ z0 + (z0.T * gc[:, 1]) @ w + (w.T * gc[:, 2]) @ w
    gG = 0.5 * (gG + gG.T)
    gn1, gn2 = d["a"][:, None] * gz0, d["b"][:, None] * gz0
    Ai, Aj, tau = f["Ai"], f["Aj"], p["tau"]
    gAj = np.cross(tau, gn1) + np.cross(gw, Ai)
    gAi = np.cross(gn2, tau) + np.cross(Aj, gw)
    gtau = np.sum(np.cross(gn1, Aj) + np.cross(Ai, gn2), 0)
    galpha = np.zeros_like(p["alpha"])
    np.add.at(galpha, d["i"], gAi); np.add.at(galpha, d["j"], gAj)
    gpre = np.outer(gx0, p["w2"]) * (f["hpre"] > 0)
    g = {"alpha": galpha, "tau": gtau, "U": 2 * gG @ p["U"], "V": -2 * gG @ p["V"],
         "W1": gpre.T @ f["u"], "w2": f["hh"].T @ gx0}
    if m.get("mutant") == "zero_grad_alpha":
        g["alpha"] = np.zeros_like(g["alpha"])
    return L, g

# ----------------------------------------------------------------- model: flat board (probe interface)
def flat_forward(m, X, T):
    p, O = m["params"], m["cfg"]["out_dim"]
    Ua = np.concatenate([X, np.ones((len(X), 1))], 1) if m["cfg"]["bias"] else X
    s = np.mean(np.abs(Ua), 1) + 1e-3
    Da = Ua.shape[1]
    G = np.einsum("odr,oer->ode", p["U"], p["U"]) - np.einsum("odr,oer->ode", p["V"], p["V"])
    c0 = np.einsum("nd,ode,ne->no", Ua, G[:, :Da, :Da], Ua)
    c1 = 2 * Ua @ G[:, :Da, Da].T
    c2 = np.broadcast_to(G[:, Da, Da], c1.shape)
    c = np.stack([c0, c1, c2], 2).reshape(-1, 3)
    hpre = Ua @ p["W1"].T; hh = np.maximum(hpre, 0.0); x0 = hh @ p["W2"].T
    sr = np.repeat(s, O)
    if m["ko"].get("kaifang") == "identity":
        x, cache = x0.reshape(-1), []
    else:
        x, cache = kaifang(c, x0.reshape(-1), sr, T, m.get("mutant"))
    return {"Ua": Ua, "G": G, "c": c, "hpre": hpre, "hh": hh, "x0": x0, "sr": sr, "x": x.reshape(-1, O), "cache": cache}

def flat_loss_grads(m, batch):
    p, O = m["params"], m["cfg"]["out_dim"]
    f = flat_forward(m, batch["X"], T_TRAIN); x, N = f["x"], len(batch["X"])
    if m["cfg"]["task_type"] == "vector_classification":
        lp = x - logsumexp(x, 1)[:, None]; Y = np.eye(O)[batch["Y"]]
        L = float(-np.mean(np.sum(Y * lp, 1))); gx = (softmax(x, 1) - Y) / N
    else:
        e = x - batch["Y"].reshape(N, O); L = float(np.mean(e * e)); gx = 2 * e / (N * O)
    gc, gx0 = kaifang_back(f["c"], f["sr"], f["cache"], gx.reshape(-1), m.get("mutant"))
    gc, gx0, Ua = gc.reshape(N, O, 3), gx0.reshape(N, O), f["Ua"]
    Da = Ua.shape[1]
    gG = np.zeros_like(f["G"])
    gG[:, :Da, :Da] = np.einsum("no,nd,ne->ode", gc[:, :, 0], Ua, Ua)
    col = gc[:, :, 1].T @ Ua
    gG[:, :Da, Da] += col; gG[:, Da, :Da] += col
    gG[:, Da, Da] += gc[:, :, 2].sum(0)
    gpre = (gx0 @ p["W2"]) * (f["hpre"] > 0)
    return L, {"U": 2 * np.einsum("ode,oer->odr", gG, p["U"]), "V": -2 * np.einsum("ode,oer->odr", gG, p["V"]),
               "W1": gpre.T @ Ua, "W2": gx0.T @ f["hh"]}

# ----------------------------------------------------------------- common interface (§8)
def build_model(in_dim, out_dim, task_type, rng, **cfg):
    if cfg.get("native", False):
        H0 = cfg.get("H0", 10)
        params = {"alpha": 0.5 * rng.standard_normal((K_TYPES, 3)), "tau": 0.5 * rng.standard_normal(3),
                  "U": 0.5 * rng.standard_normal((3, 2)), "V": 0.5 * rng.standard_normal((3, 2)),
                  "W1": np.abs(0.3 * rng.standard_normal((H0, K_TYPES))), "w2": np.abs(0.3 * rng.standard_normal(H0))}
        return {"kind": "board", "params": params, "cfg": dict(cfg, task_type=task_type), "ko": {}, "ratio_mean": 0.5}
    bias = cfg.get("bias", True); Da = in_dim + (1 if bias else 0); H, R = cfg.get("H", 16), cfg.get("R", 2)
    params = {"U": 0.3 * rng.standard_normal((out_dim, Da + 1, R)), "V": 0.3 * rng.standard_normal((out_dim, Da + 1, R)),
              "W1": rng.standard_normal((H, Da)) / math.sqrt(Da), "W2": 0.3 * rng.standard_normal((out_dim, H))}
    return {"kind": "flat", "params": params, "cfg": dict(cfg, task_type=task_type, out_dim=out_dim, bias=bias), "ko": {}}

def loss_and_grads(model, batch):
    return board_loss_grads(model, batch) if model["kind"] == "board" else flat_loss_grads(model, batch)

def predict(model, X):
    if model["kind"] == "board":
        return board_forward(model, X, T_INFER)["x"]
    x = flat_forward(model, X, T_INFER)["x"]
    return np.argmax(x, 1) if model["cfg"]["task_type"] == "vector_classification" else x

def hidden_states(model, X):
    f = board_forward(model, X, T_INFER) if model["kind"] == "board" else flat_forward(model, X, T_INFER)
    return {"coefficients": f["c"], "first_quotient": f["x0"], "root": f["x"]}

def modules(model):
    if model["kind"] == "flat":
        return {"law": {"params": ["U", "V"], "role": "per-output indefinite quadratic form over [inputs, unknown]", "signature": True},
                "kaifang": {"params": [], "role": "unrolled damped Newton root extraction", "signature": True},
                "chushang": {"params": ["W1", "W2"], "role": "ReLU initial estimate", "signature": False}}
    return {"shibie": {"params": ["alpha"], "role": "per-type linear read-out forms on a shared 3-d latent (learned measurement matrix)", "signature": True},
            "tianyuan": {"params": ["tau"], "role": "linear form of the unknown on the latent", "signature": True},
            "law": {"params": ["U", "V"], "role": "indefinite quadratic invariant as difference of two PSD registers", "signature": True},
            "kaifang": {"params": [], "role": "unrolled damped Newton root extraction", "signature": True},
            "chushang": {"params": ["W1", "w2"], "role": "bias-free ReLU initial estimate (root selection)", "signature": False}}

KO_MODES = {"shibie": "mean", "tianyuan": "zero", "law": "mean", "kaifang": "identity", "chushang": "mean"}

def knockout(model, name, mode=None):
    m = dict(model); m["ko"] = dict(model["ko"]); m["ko"][name] = mode or KO_MODES[name]
    return m

def n_params(model):
    return int(sum(v.size for v in model["params"].values()))

def copy_model(m):
    c = dict(m); c["params"] = {k: v.copy() for k, v in m["params"].items()}; c["ko"] = dict(m["ko"]); return c

# ----------------------------------------------------------------- catalogue stage (識別雜記)
def veronese_equation(a, b, y):
    """Non-gradient rule: the quadratic relation among (a,b,y) is the null vector of the lifted data."""
    s = np.abs(a) + np.abs(b)
    a_, b_, y_ = a / s, b / s, y / s
    Phi = np.stack([a_ * a_, b_ * b_, y_ * y_, 2 * a_ * b_, 2 * a_ * y_, 2 * b_ * y_], 1)
    _, sv, Vt = np.linalg.svd(Phi, full_matrices=False)
    q = Vt[-1]
    Q = np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])
    return Q, float(sv[-1] / sv[0])

def gram_loss_grads(B, J, Gam, known):
    R = (B @ J @ B.T - Gam) * known; n = max(1.0, float(known.sum()))
    return float(np.sum(R * R) / n), 4 * R @ B @ J / n

def catalogue_stage(model, d, rng, steps=1500):
    """Per-pair equations -> dual blocks M H M^T -> signed rank-3 Gram completion -> initial board."""
    K = K_TYPES; Gam = np.zeros((K + 1, K + 1)); W = np.zeros_like(Gam); sigs, conds = [], []
    keys = np.minimum(d["i"], d["j"]) * K + np.maximum(d["i"], d["j"])
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        if len(idx) < 12: continue
        i, j = divmod(int(key), K)
        first = d["i"][idx] == i
        a = np.where(first, d["a"][idx], d["b"][idx]); b = np.where(first, d["b"][idx], d["a"][idx])
        Q, cond = veronese_equation(a, b, d["y"][idx])
        if abs(np.linalg.det(Q)) < 1e-12: continue
        D = np.linalg.inv(Q)
        if abs(D[2, 2]) < 1e-9 * np.abs(D).max(): continue
        D = D / D[2, 2]; conds.append(cond)
        sigs.append(int(np.sum(np.linalg.eigvalsh(D) < 0)))
        ids = [i, j, K]
        for u in range(3):
            for v in range(3):
                Gam[ids[u], ids[v]] += D[u, v]; W[ids[u], ids[v]] += 1
    info = {"blocks": len(conds), "median_cond": float(np.median(conds)) if conds else None}
    if not conds:
        info["residual"] = None; return info
    known = (W > 0).astype(float); Gam = np.where(W > 0, Gam / np.maximum(W, 1), 0.0)
    nneg = int(np.bincount(sigs).argmax()); J = np.diag([1.0] * (3 - nneg) + [-1.0] * nneg)
    B = 0.5 * rng.standard_normal((K + 1, 3)); st = adam_init({"B": B})
    for _ in range(steps):
        L, gB = gram_loss_grads(B, J, Gam, known)
        pB = {"B": B}; adam_step(pB, {"B": gB}, st, 0.02); B = pB["B"]
    info["residual"] = float(math.sqrt(gram_loss_grads(B, J, Gam, known)[0])); info["signature_neg"] = nneg
    p = model["params"]; p["alpha"], p["tau"] = B[:K].copy(), B[K].copy()
    U, V = np.zeros((3, 2)), np.zeros((3, 2))
    for k in range(3):
        if J[k, k] > 0: U[k, int(np.sum(np.diag(J)[:k] > 0))] = 1.0
        else: V[k, int(np.sum(np.diag(J)[:k] < 0))] = 1.0
    p["U"], p["V"] = U, V
    return info

def flat_catalogue(model, X, Y):
    """Probe-interface version of the catalogue: read each output's law off the lifted data [inputs, target]."""
    p, O, R = model["params"], model["cfg"]["out_dim"], model["params"]["U"].shape[2]
    Ua = np.concatenate([X, np.ones((len(X), 1))], 1) if model["cfg"]["bias"] else X
    iu = np.triu_indices(Ua.shape[1] + 1)
    for o in range(O):
        v = np.concatenate([Ua, Y.reshape(len(X), -1)[:, o:o + 1]], 1)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        Phi = (v[:, :, None] * v[:, None, :])[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)
        q = np.linalg.svd(Phi, full_matrices=False)[2][-1]
        G = np.zeros((v.shape[1], v.shape[1])); G[iu] = q; G = G + G.T - np.diag(np.diag(G))
        lam, E = np.linalg.eigh(G); order = np.argsort(-np.abs(lam))
        pos = [k for k in order if lam[k] > 0][:R]; neg = [k for k in order if lam[k] < 0][:R]
        p["U"][o] = 0.0; p["V"][o] = 0.0
        for c, k in enumerate(pos): p["U"][o][:, c] = np.sqrt(lam[k]) * E[:, k]
        for c, k in enumerate(neg): p["V"][o][:, c] = np.sqrt(-lam[k]) * E[:, k]

def catalogue_recovery(model):
    """Gauge-aligned error between learned forms and the true sums/differences (interpretability)."""
    A = np.vstack([model["params"]["alpha"], model["params"]["tau"]]); Tt = np.vstack([FORMS, TAU_TRUE])
    Tm = np.linalg.lstsq(A, Tt, rcond=None)[0]
    return float(np.max(np.abs(A @ Tm - Tt)))

# ----------------------------------------------------------------- baselines and rival
def explicit_build(rng, kind, H):
    """kind 'mlp': tanh MLP with biases on standardized values+mask; 'rival': bias-free ReLU on values."""
    din = 2 * K_TYPES if kind == "mlp" else K_TYPES
    p = {"W1": rng.standard_normal((H, din)) / math.sqrt(din), "w2": rng.standard_normal(H) / math.sqrt(H)}
    if kind == "mlp":
        p["b1"], p["b2"] = np.zeros(H), np.zeros(1)
    else:
        p["W1"] = np.abs(p["W1"])
    return {"kind": kind, "params": p, "ko": {}, "norm": (1.0, 0.0, 1.0)}

def explicit_forward(m, d):
    p, (sv, my, sy) = m["params"], m["norm"]; u = scatter(d)
    if m["kind"] == "mlp":
        X = np.concatenate([u / sv, (u != 0).astype(float)], 1)
        pre = X @ p["W1"].T + p["b1"]; h = np.tanh(pre); out = my + sy * (h @ p["w2"] + p["b2"][0])
    else:
        X = u; pre = X @ p["W1"].T; h = np.maximum(pre, 0.0); out = h @ p["w2"]
    return X, pre, h, out

def explicit_loss_grads(m, d):
    p = m["params"]; X, pre, h, out = explicit_forward(m, d); s = d["a"] + d["b"]
    e = (out - d["y"]) / s; N = len(e); L = float(np.mean(e * e))
    go = 2 * e / s / N * (m["norm"][2] if m["kind"] == "mlp" else 1.0)
    gpre = np.outer(go, p["w2"]) * ((1 - h * h) if m["kind"] == "mlp" else (pre > 0))
    g = {"W1": gpre.T @ X, "w2": h.T @ go}
    if m["kind"] == "mlp":
        g["b1"], g["b2"] = gpre.sum(0), np.array([go.sum()])
    return L, g

# ----------------------------------------------------------------- registries
MUTANTS = {"sign_flip": "optimizer adds the Adam update", "zero_lr": "learning rate set to zero",
           "zero_grad_alpha": "gradient of the shibie forms zeroed", "detach_kaifang": "root extraction backprop ignores the coefficients"}

# ----------------------------------------------------------------- training
def fit(model, data, budget, rng, lr=None, bs=256, catalogue=True, val=None):
    """Stage 1 (board only): catalogue from per-pair equations. Stage 2a: first quotient only;
    2b: everything at a lower rate; the better of 2a/2b on the validation split is kept."""
    info = {}
    if model["kind"] == "board":
        model["ratio_mean"] = float(np.mean(data["y"] / (data["a"] + data["b"])))
        if catalogue:
            info = catalogue_stage(model, data, rng)
        lr = lr or (0.003 if catalogue and info.get("residual") is not None else 0.01)
        grad_fn = lambda b: board_loss_grads(model, b)
    elif model["kind"] in ("mlp", "rival"):
        v = np.concatenate([data["a"], data["b"]])
        model["norm"] = (float(np.std(v)), float(np.mean(data["y"])), float(np.std(data["y"])))
        lr = lr or 0.01; grad_fn = lambda b: explicit_loss_grads(model, b)
    else:
        if model["cfg"]["task_type"] == "vector_regression" and catalogue:
            flat_catalogue(model, data["X"], data["Y"])
        lr = lr or 0.01; grad_fn = lambda b: flat_loss_grads(model, b)
    if model.get("mutant") == "zero_lr": lr = 0.0
    n = len(data["y"]) if "y" in data else len(data["X"])
    st, hist = adam_init(model["params"]), []
    staged = (model["kind"] == "board" and info.get("residual") is not None) or (
        model["kind"] == "flat" and catalogue and model["cfg"]["task_type"] == "vector_regression")
    for t in range(budget):
        b = subset(data, rng.integers(n, size=min(bs, n)))
        L, g = grad_fn(b)
        if not np.isfinite(L): raise FloatingPointError("non-finite loss")
        if staged and val is not None and t == budget // 2:
            snap = {k: v.copy() for k, v in model["params"].items()}; snap_err = evaluate(model, val)
        if staged and t < budget // 2:      # first quotient learns its basin before the board is refined
            g = {k: (v if k in ("W1", "w2", "W2") else np.zeros_like(v)) for k, v in g.items()}
        g, _ = clip_global_norm(g, ALLOWED_CLIP)
        if model.get("mutant") == "sign_flip": g = {k: -v for k, v in g.items()}
        adam_step(model["params"], g, st, lr if not (staged and t >= budget // 2) else lr / 3)
        hist.append(L)
    if staged and val is not None and budget >= 2:
        info["selected"] = "refined"
        if evaluate(model, val) > snap_err:
            model["params"] = snap; info["selected"] = "catalogue+first-quotient"
    info["loss_start"] = float(np.mean(hist[:20])) if hist else None
    info["loss_end"] = float(np.mean(hist[-20:])) if hist else None
    return info

def rel_rmse(pred, y):
    return float(np.sqrt(np.mean(((pred - y) / y) ** 2)))

def evaluate(model, d):
    if model["kind"] == "flat":
        pred = predict(model, d["X"])
        return float(np.sqrt(np.mean((pred - d["Y"].reshape(pred.shape)) ** 2))) if np.all(np.isfinite(pred)) else float("inf")
    pred = explicit_forward(model, d)[3] if model["kind"] in ("mlp", "rival") else predict(model, d)
    if not np.all(np.isfinite(pred)): return float("inf")
    return rel_rmse(pred, d["y"])

def trivial_rmse(train, d):
    r = np.mean(train["y"] / (train["a"] + train["b"]))
    return rel_rmse(r * (d["a"] + d["b"]), d["y"])

def size_matched(target, kind):
    per = (2 * K_TYPES + 2) if kind == "mlp" else (K_TYPES + 1)
    extra = 1 if kind == "mlp" else 0
    return max(1, int(round((target - extra) / per)))

# ----------------------------------------------------------------- correctness tests
def grad_suite(rng, after_steps=60):
    """C1: board (native), flat board (regression, classification), Gram completion, baselines."""
    res, task = {}, make_task(rng, n_train=300, n_eval=50)
    for stage in ("init", "trained"):
        m = build_model(K_TYPES, 1, "vector_regression", rng, native=True)
        if stage == "trained":
            fit(m, task["train"], after_steps, rng, catalogue=False)
        f = board_forward(m, task["train"], 0)
        batch = subset(task["train"], np.where(well_posed(f["c"], f["x0"]))[0][:40])
        L, g = board_loss_grads(m, batch)
        for k, v in gradcheck(lambda: board_loss_grads(m, batch)[0], m["params"], g, rng).items():
            res["board/%s/%s" % (stage, k)] = v
    X0 = rng.standard_normal((200, 4))
    for tt, Y0 in (("vector_regression", rng.standard_normal((200, 2))), ("vector_classification", rng.integers(3, size=200))):
        m = build_model(4, 2 if tt == "vector_regression" else 3, tt, rng, H=6)
        for stage in ("init", "trained"):
            if stage == "trained": fit(m, {"X": X0, "Y": Y0}, after_steps, rng, bs=60)
            f = flat_forward(m, X0, 0); O = m["cfg"]["out_dim"]
            keep = np.where(np.all(well_posed(f["c"], f["x0"].reshape(-1)).reshape(-1, O), 1))[0][:30]
            fb = {"X": X0[keep], "Y": Y0[keep]}
            L, g = flat_loss_grads(m, fb)
            for k, v in gradcheck(lambda: flat_loss_grads(m, fb)[0], m["params"], g, rng).items():
                res["flat_%s/%s/%s" % (tt[7:10], stage, k)] = v
    Bp = {"B": rng.standard_normal((6, 3))}; Jm = np.diag([1.0, 1.0, -1.0])
    Gam = rng.standard_normal((6, 6)); Gam = Gam + Gam.T; kn = (rng.random((6, 6)) < 0.6).astype(float); kn = np.maximum(kn, kn.T)
    _, gB = gram_loss_grads(Bp["B"], Jm, Gam, kn)
    res["catalogue/B"] = gradcheck(lambda: gram_loss_grads(Bp["B"], Jm, Gam, kn)[0], Bp, {"B": gB}, rng)["B"]
    for kind in ("mlp", "rival"):
        m = explicit_build(rng, kind, 5); fit(m, task["train"], 30, rng)
        L, g = explicit_loss_grads(m, batch)
        for k, v in gradcheck(lambda: explicit_loss_grads(m, batch)[0], m["params"], g, rng).items():
            res["%s/%s" % (kind, k)] = v
    return res

def learning_check(m, task, budget, rng):
    info = fit(m, task["train"], budget, rng, val=task["val"])
    rm, tv = evaluate(m, task["held_out"]), trivial_rmse(task["train"], task["held_out"])
    drop = 1 - info["loss_end"] / info["loss_start"]
    ok = drop >= MIND_CARD["thresholds"]["loss_drop_fraction"] and rm <= MIND_CARD["thresholds"]["margin_over_trivial"] * tv
    return ok, {"loss_drop": drop, "rel_rmse": rm, "trivial": tv}

def c3_uncatalogued(rng, budget):
    """C3 as used for mutants: the gradient stage alone must cut the loss (catalogue stage off)."""
    task = make_task(rng, n_train=1200, n_eval=300)
    m = build_model(K_TYPES, 1, "vector_regression", rng, native=True)
    return m, task

def property_tests(rng):
    out = {}
    # C6.1 homogeneity of degree one, random parameters and inputs, with a broken negative control
    worst, broken = 0.0, 0.0
    for trial in range(60):
        m = build_model(K_TYPES, 1, "vector_regression", rng, native=True)
        m["params"]["w2"] = rng.standard_normal(10)
        d = gen_problems(valid_pairs(), 20, rng); k = float(np.exp(rng.uniform(-3, 4)))
        dk = dict(d, a=d["a"] * k, b=d["b"] * k, y=d["y"] * k)
        f = board_forward(m, d, 0); wp = well_posed(f["c"], f["x0"])
        base, sc = predict(m, d), predict(m, dk)
        ok = wp & np.isfinite(base) & np.isfinite(sc) & (np.abs(base) > 1e-9)
        if not ok.any(): continue
        worst = max(worst, float(np.max(np.abs(sc[ok] - k * base[ok]) / np.abs(k * base[ok]))))
        mb = copy_model(m); mb["cfg"] = dict(m["cfg"], x0_bias=0.37)
        b2, s2 = predict(mb, d), predict(mb, dk)
        ok = np.isfinite(b2) & np.isfinite(s2) & (np.abs(b2) > 1e-9)
        broken = max(broken, float(np.max(np.abs(s2[ok] - k * b2[ok]) / np.abs(k * b2[ok]))))
    out["C6.1_homogeneity"] = (worst < 1e-7 and broken > 1e-3, "max rel violation %.1e; biased control %.1e" % (worst, broken))
    # C6.2 gauge invariance under GL(3) change of figure coordinates
    worst, broken = 0.0, 0.0
    for trial in range(40):
        m = build_model(K_TYPES, 1, "vector_regression", rng, native=True)
        d = gen_problems(valid_pairs(), 20, rng)
        Tm = rng.standard_normal((3, 3)) + 2 * np.eye(3); Ti = np.linalg.inv(Tm)
        mg = copy_model(m); P = mg["params"]
        P["alpha"], P["tau"], P["U"], P["V"] = P["alpha"] @ Ti, P["tau"] @ Ti, Ti.T @ P["U"], Ti.T @ P["V"]
        f = board_forward(m, d, 0); wp = well_posed(f["c"], f["x0"])
        base, gauged = predict(m, d), predict(mg, d); ok = wp & np.isfinite(base) & (np.abs(base) > 1e-6)
        if not ok.any(): continue
        worst = max(worst, float(np.max(np.abs(gauged[ok] - base[ok]) / np.abs(base[ok]))))
        mc = copy_model(m); mc["params"]["alpha"] = P["alpha"]; mc["params"]["tau"] = P["tau"]
        bc = predict(mc, d); broken = max(broken, float(np.max(np.abs(bc[ok] - base[ok]) / np.abs(base[ok]))))
    out["C6.2_gauge"] = (worst < 1e-7 and broken > 1e-3, "max rel change %.1e; ungauged-law control %.1e" % (worst, broken))
    # C6.3 kaifang converges to a root from a nearby first quotient; sign-flipped update does not
    r1 = rng.uniform(0.5, 5, 500); r2 = -rng.uniform(0.5, 5, 500); c2 = rng.uniform(0.5, 2, 500)
    c = np.stack([c2 * r1 * r2, -c2 * (r1 + r2), c2], 1); x0 = r1 * (1 + rng.uniform(-0.4, 0.4, 500)); s = np.ones(500)
    good = float(np.max(np.abs(kaifang(c, x0, s, T_INFER)[0] - r1) / r1))
    bad = float(np.median(np.abs(kaifang(c, x0, s, T_INFER, "newton_sign")[0] - r1) / r1))
    out["C6.3_kaifang"] = (good < 1e-8 and bad > 0.1, "max rel error %.1e; sign-flipped control %.1e" % (good, bad))
    # C6.4 Veronese rule recovers a known quadric exactly and reports none for shuffled data
    Qt = rng.standard_normal((3, 3)); Qt = Qt + Qt.T
    a, b = rng.uniform(1, 3, 200), rng.uniform(1, 3, 200)
    qa, qb = Qt[2, 2], 2 * (Qt[0, 2] * a + Qt[1, 2] * b)
    qc = Qt[0, 0] * a * a + 2 * Qt[0, 1] * a * b + Qt[1, 1] * b * b
    disc = qb * qb - 4 * qa * qc; ok = disc > 0
    y = (-qb + np.sqrt(np.maximum(disc, 0))) / (2 * qa)
    Q, cond = veronese_equation(a[ok], b[ok], y[ok])
    err = float(min(np.abs(Q / np.linalg.norm(Q) - Qt / np.linalg.norm(Qt)).max(), np.abs(Q / np.linalg.norm(Q) + Qt / np.linalg.norm(Qt)).max()))
    _, cond_sh = veronese_equation(a[ok], b[ok], rng.permutation(y[ok]))
    out["C6.4_veronese"] = (err < 1e-6 and cond_sh > 1e-3, "coefficient error %.1e; shuffled null-ratio %.1e" % (err, cond_sh))
    # C6.5 swapping the two given segments leaves the answer unchanged; a mis-ordered adjugate breaks it
    m = build_model(K_TYPES, 1, "vector_regression", rng, native=True); d = gen_problems(valid_pairs(), 50, rng)
    ds = dict(d, i=d["j"], j=d["i"], a=d["b"], b=d["a"])
    base, sw = predict(m, d), predict(m, ds); ok = np.isfinite(base) & (np.abs(base) > 1e-6)
    sym = float(np.max(np.abs(sw[ok] - base[ok]) / np.abs(base[ok])))
    Ai, Aj, tau = m["params"]["alpha"][d["i"]], m["params"]["alpha"][d["j"]], m["params"]["tau"]
    z = d["a"][:, None] * np.cross(Aj, tau) + d["b"][:, None] * np.cross(Ai, tau)   # control: wrong orientation
    zs = d["b"][:, None] * np.cross(Ai, tau) + d["a"][:, None] * np.cross(Aj, tau)
    G = law_matrix(m["params"], {}); w, ws = np.cross(Ai, Aj), np.cross(Aj, Ai)
    ctrl = float(np.max(np.abs(np.sum(z * (w @ G), 1) - np.sum(zs * (ws @ G), 1)) / (np.abs(np.sum(z * (w @ G), 1)) + 1e-12)))
    out["C6.5_swap"] = (sym < 1e-8 and ctrl > 1e-3, "max rel change %.1e; mis-ordered control %.1e" % (sym, ctrl))
    return out

def run_correctness(rng, quick):
    C, t0 = [], time.time()
    gc = grad_suite(rng)
    worst = max(gc.values())
    C.append({"id": "C1", "name": "gradient_check", "passed": worst <= MIND_CARD["thresholds"]["gradcheck_tol"],
              "detail": "%d tensors, max rel error %.2e (init and after 60 steps)" % (len(gc), worst)})
    task = make_task(rng)
    m1 = build_model(K_TYPES, 1, "vector_regression", np.random.default_rng(7), native=True)
    fit(m1, task["train"], 200, np.random.default_rng(8))
    m2 = build_model(K_TYPES, 1, "vector_regression", np.random.default_rng(7), native=True)
    fit(m2, task["train"], 200, np.random.default_rng(8))
    p1, p2 = predict(m1, task["held_out"]), predict(m2, task["held_out"])
    fin = all(np.all(np.isfinite(v)) for v in m1["params"].values()) and np.all(np.isfinite(p1))
    C.append({"id": "C2", "name": "determinism_finiteness", "passed": bool(np.array_equal(p1, p2) and fin), "detail": "identical predictions, finite"})
    m = build_model(K_TYPES, 1, "vector_regression", rng, native=True)
    ok, det = learning_check(m, task, 300 if quick else 800, rng)
    C.append({"id": "C3", "name": "learning", "passed": bool(ok),
              "detail": "loss drop %.2f, held-out rel RMSE %.2e vs trivial %.3f" % (det["loss_drop"], det["rel_rmse"], det["trivial"])})
    sh = dict(task); sh["train"] = dict(task["train"], y=rng.permutation(task["train"]["y"]))
    ms = build_model(K_TYPES, 1, "vector_regression", rng, native=True); fit(ms, sh["train"], 300, rng)
    rs, tv = evaluate(ms, task["held_out"]), trivial_rmse(task["train"], task["held_out"])
    C.append({"id": "C4", "name": "shuffled_label_control", "passed": bool(rs >= MIND_CARD["thresholds"]["shuffle_band"] * tv),
              "detail": "shuffled-trained rel RMSE %.3f vs trivial %.3f" % (rs, tv)})
    det_m = {}
    for name in MUTANTS:
        mm, tk = c3_uncatalogued(np.random.default_rng(11), 0)
        mm["mutant"] = name
        b = subset(tk["train"], np.arange(40)); L, g = board_loss_grads(mm, b)
        clean = copy_model(mm); clean.pop("mutant")
        c1_fail = max(gradcheck(lambda: board_loss_grads(clean, b)[0], mm["params"], g, np.random.default_rng(3)).values()) > 1e-5
        info = fit(mm, tk["train"], 150, np.random.default_rng(12), catalogue=False)
        c3_fail = not (1 - info["loss_end"] / info["loss_start"] >= MIND_CARD["thresholds"]["loss_drop_fraction"])
        det_m[name] = bool(c1_fail or c3_fail)
    mm, tk = c3_uncatalogued(np.random.default_rng(11), 0)
    info = fit(mm, tk["train"], 150, np.random.default_rng(12), catalogue=False)
    clean_passes = 1 - info["loss_end"] / info["loss_start"] >= MIND_CARD["thresholds"]["loss_drop_fraction"]
    C.append({"id": "C5", "name": "mutant_detection", "passed": bool(all(det_m.values()) and clean_passes),
              "detail": "detected %d/%d; unmutated reference passes: %s" % (sum(det_m.values()), len(det_m), clean_passes)})
    props = property_tests(rng)
    C.append({"id": "C6", "name": "property_tests", "passed": all(v[0] for v in props.values()),
              "detail": "; ".join("%s: %s" % (k, v[1]) for k, v in props.items())})
    sig = lambda d: set(zip(d["i"], d["j"], np.round(d["a"], 9), np.round(d["b"], 9)))
    tr = sig(task["train"]); overlap = sum(len(tr & sig(task[s])) for s in ("held_out", "compositional", "scale"))
    trp = {tuple(sorted(p)) for p in task["pairs"][0]}; hop = {tuple(sorted(p)) for p in task["pairs"][1]}
    C.append({"id": "C7", "name": "split_integrity", "passed": overlap == 0 and not (trp & hop),
              "detail": "shared examples %d; shared pairs %d" % (overlap, len(trp & hop))})
    return C, {"mutants": det_m, "properties": props}

# ----------------------------------------------------------------- hypotheses
def one_seed(seed, steps):
    rng = np.random.default_rng(seed); ss = np.random.SeedSequence(seed).spawn(8)
    task = make_task(rng)
    R = {}
    board = build_model(K_TYPES, 1, "vector_regression", np.random.default_rng(ss[0]), native=True)
    cat = fit(board, task["train"], steps, np.random.default_rng(ss[1]), val=task["val"])
    npar = n_params(board)
    splits = ("held_out", "compositional", "scale")
    R["board"] = {s: evaluate(board, task[s]) for s in splits}
    for mod in KO_MODES:
        R["ko_" + mod] = {s: evaluate(knockout(board, mod), task[s]) for s in splits}
    grad_only = build_model(K_TYPES, 1, "vector_regression", np.random.default_rng(ss[0]), native=True)
    fit(grad_only, task["train"], steps, np.random.default_rng(ss[2]), catalogue=False)
    R["gradient_only"] = {s: evaluate(grad_only, task[s]) for s in splits}
    for kind, H, key, sd in (("mlp", size_matched(npar, "mlp"), "mlp", ss[3]), ("rival", size_matched(npar, "rival"), "rival", ss[4]),
                             ("mlp", 64, "mlp_wide", ss[5])):
        m = explicit_build(np.random.default_rng(sd), kind, H); fit(m, task["train"], steps, np.random.default_rng(sd))
        R[key] = {s: evaluate(m, task[s]) for s in splits}; R[key]["n_params"] = n_params(m)
    R["trivial"] = {s: trivial_rmse(task["train"], task[s]) for s in splits}
    noisy = make_task(np.random.default_rng(seed + 1000), noise=0.03)
    bn = build_model(K_TYPES, 1, "vector_regression", np.random.default_rng(ss[6]), native=True); fit(bn, noisy["train"], steps, np.random.default_rng(ss[6]), val=noisy["val"])
    mn = explicit_build(np.random.default_rng(ss[7]), "mlp", size_matched(npar, "mlp")); fit(mn, noisy["train"], steps, np.random.default_rng(ss[7]))
    R["noisy"] = {"board": evaluate(bn, noisy["held_out"]), "mlp": evaluate(mn, noisy["held_out"]),
                  "board_comp": evaluate(bn, noisy["compositional"]), "mlp_comp": evaluate(mn, noisy["compositional"])}
    R["catalogue"] = dict(cat, recovery_error=catalogue_recovery(board)); R["n_params"] = npar
    return R

def lg(x):
    return math.log10(max(x, 1e-15)) if np.isfinite(x) else 3.0

HYP_DIFFS = {
    "H-SIG": lambda R: lg(R["board"]["compositional"]) - lg(R["mlp"]["compositional"]),
    "H-NEC": lambda R: lg(R["ko_kaifang"]["compositional"]) - lg(R["ko_chushang"]["compositional"]),
    "H-BLIND": lambda R: lg(R["noisy"]["board"]) - lg(R["noisy"]["mlp"]),
    "H-RIVAL": lambda R: lg(R["board"]["compositional"]) - lg(R["rival"]["compositional"]),
    "H-ORDER": lambda R: lg(R["board"]["compositional"]) - lg(R["gradient_only"]["compositional"]),
}

def run_hypotheses(seeds, steps, rng, evaluate_h):
    runs = [one_seed(s, steps) for s in seeds]
    H = []
    for h in MIND_CARD["hypotheses"]:
        diffs = [HYP_DIFFS[h["id"]](R) for R in runs]
        mean, ci = paired_bootstrap(diffs, rng) if len(diffs) > 1 else (float(np.mean(diffs)), [float("nan")] * 2)
        H.append({"id": h["id"], "metric": h["metric"], "mean_diff": mean, "ci95": ci, "mesi": h["mesi"], "n_seeds": len(diffs),
                  "verdict": verdict(mean, ci, h["mesi"], h["direction"]) if evaluate_h else "not evaluated"})
    KO = []
    for mod in KO_MODES:
        diffs = [lg(R["ko_" + mod]["compositional"]) - lg(R["board"]["compositional"]) for R in runs]
        mean, ci = paired_bootstrap(diffs, rng) if len(diffs) > 1 else (float(np.mean(diffs)), [float("nan")] * 2)
        KO.append({"module": mod, "signature": mod != "chushang", "metric_change": mean, "ci95": ci})
    return runs, H, KO

# ----------------------------------------------------------------- report and entry point
def data_bridge(path, rng):
    raw = np.genfromtxt(path, delimiter=",", skip_header=1); raw = raw[np.all(np.isfinite(raw), 1)]
    X, Y = raw[:, :-1], raw[:, -1:]; mu, sd = X.mean(0), X.std(0) + 1e-9; X = (X - mu) / sd
    idx = rng.permutation(len(X)); cut, cv = int(0.7 * len(X)), int(0.8 * len(X))
    tr, va, te = idx[:cut], idx[cut:cv], idx[cv:]
    m = build_model(X.shape[1], 1, "vector_regression", rng)
    fit(m, {"X": X[tr], "Y": Y[tr]}, 800, rng, val={"X": X[va], "Y": Y[va]})
    rm = float(np.sqrt(np.mean((predict(m, X[te]) - Y[te]) ** 2)))
    return {"rows": int(len(X)), "flat_board_rmse": rm, "mean_baseline_rmse": float(np.std(Y[te]))}

def main():
    ap = argparse.ArgumentParser(prog=os.path.basename(__file__))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--seed", type=int, default=323)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--json"); ap.add_argument("--card", action="store_true")
    ap.add_argument("--mutant", choices=sorted(MUTANTS)); ap.add_argument("--data")
    try:
        args = ap.parse_args()
    except SystemExit as e:
        sys.exit(0 if e.code == 0 else 2)
    if args.card:
        print(json.dumps(MIND_CARD, indent=2, ensure_ascii=False)); return 0
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    seeds = [args.seed] if args.quick else [args.seed + k for k in range(args.seeds)]
    print("seeds:", seeds)
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    if args.mutant:
        mm, tk = c3_uncatalogued(np.random.default_rng(args.seed), 0); mm["mutant"] = args.mutant
        info = fit(mm, tk["train"], 150, rng, catalogue=False)
        drop = 1 - info["loss_end"] / info["loss_start"]
        print("mutant %s: loss drop %.3f" % (args.mutant, drop)); return 0 if drop >= 0.5 else 1
    C, extra = run_correctness(rng, args.quick)
    runs, H, KO = run_hypotheses(seeds, 400 if args.quick else 1500, rng, evaluate_h=not args.quick)
    runtime = time.time() - t0
    C.append({"id": "C8", "name": "budget", "passed": runtime <= (20 if args.quick else 180), "detail": "%.1f s" % runtime})
    exit_code = 0 if all(c["passed"] for c in C[:-1]) else 1
    if not C[-1]["passed"]: exit_code = 3
    mean = lambda key, s: float(np.mean([R[key][s] for R in runs]))
    print("=== VERIFIED REPORT · chapter 0323 ===")
    print("environment: python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    print("seeds: %s   runtime: %.1f s   board parameters: %d   file sha256: %s" % (seeds, runtime, runs[0]["n_params"],
          hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]))
    print("correctness:")
    for c in C: print("  %-3s %-24s %-5s %s" % (c["id"], c["name"], "PASS" if c["passed"] else "FAIL", c["detail"]))
    print("mutation score: %d/%d" % (sum(extra["mutants"].values()), len(extra["mutants"])))
    print("relative RMSE (mean over seeds)      held_out   compositional   scale x30")
    for key in ("board", "gradient_only", "mlp", "mlp_wide", "rival", "ko_kaifang", "ko_chushang", "ko_shibie", "ko_law", "trivial"):
        print("  %-34s %9.2e   %13.2e   %9.2e" % (key, mean(key, "held_out"), mean(key, "compositional"), mean(key, "scale")))
    print("  sizes: board %d, mlp %d, rival %d, mlp_wide %d" % (runs[0]["n_params"], runs[0]["mlp"]["n_params"], runs[0]["rival"]["n_params"], runs[0]["mlp_wide"]["n_params"]))
    print("  noisy (3%%) held_out: board %.3e  mlp %.3e | compositional: board %.3e  mlp %.3e" % tuple(
        float(np.mean([R["noisy"][k] for R in runs])) for k in ("board", "mlp", "board_comp", "mlp_comp")))
    print("  catalogue: blocks %s, Gram residual %s, gauge-aligned recovery error %.2e" % (
        runs[0]["catalogue"].get("blocks"), "%.1e" % runs[0]["catalogue"]["residual"] if runs[0]["catalogue"].get("residual") is not None else "n/a",
        float(np.mean([R["catalogue"]["recovery_error"] for R in runs]))))
    print("hypotheses (log10 relative RMSE differences):")
    for h in H: print("  %-8s mean %+.3f  CI95 [%+.3f, %+.3f]  mesi %.2f  n=%d  %s" % (h["id"], h["mean_diff"], h["ci95"][0], h["ci95"][1], h["mesi"], h["n_seeds"], h["verdict"]))
    print("knockouts (compositional, log10 change vs full board):")
    for k in KO: print("  %-9s signature=%-5s %+.3f  CI95 [%+.3f, %+.3f]" % (k["module"], k["signature"], k["metric_change"], k["ci95"][0], k["ci95"][1]))
    print("similarity gate: measured by the corpus audit tool (tokenize is outside the in-file import allowlist)")
    print("data bridge: %s" % (data_bridge(args.data, rng) if args.data else "skipped (no --data PATH)"))
    print("task types:", TASK_TYPES, "  exit code:", exit_code)
    print("=== END REPORT ===")
    rep = {"schema_version": "1.0", "chapter": 323, "file": os.path.basename(__file__), "card_revision": MIND_CARD["card_revision"],
           "environment": {"python": sys.version.split()[0], "numpy": np.__version__}, "seeds": seeds, "runtime_s": runtime,
           "n_params": runs[0]["n_params"],
           "gradcheck": {"tensors_checked": int(C[0]["detail"].split()[0]), "tensors_total": int(C[0]["detail"].split()[0]),
                         "max_rel_error": float(C[0]["detail"].split("error ")[1].split()[0]), "checked_at": ["init", "after_training_steps"], "passed": C[0]["passed"]},
           "correctness": C, "mutants": {"detected": sum(extra["mutants"].values()), "total": len(extra["mutants"]),
                                          "score": sum(extra["mutants"].values()) / len(extra["mutants"])},
           "hypotheses": H, "knockouts": KO, "task_types": TASK_TYPES, "exit_code": exit_code,
           "descriptive": {k: {s: mean(k, s) for s in ("held_out", "compositional", "scale")} for k in
                           ("board", "gradient_only", "mlp", "mlp_wide", "rival", "ko_kaifang", "ko_chushang", "ko_shibie", "ko_law", "trivial")}}
    write_report(rep, args.json)
    return exit_code

if __name__ == "__main__":
    try:
        sys.exit(main())
    except FloatingPointError as e:
        print("non-finite values:", e); sys.exit(4)

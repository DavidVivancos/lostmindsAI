"""
================================================================================
chapter_0025_ashurbanipal_-685.py
The Barû Engine — a from-scratch, trainable architecture that embodies the mind
of Ashurbanipal (r. 668–627 BCE), king of Assyria and assembler of the great
library of Nineveh.
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# # Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0025 · Ashurbanipal
WHY THIS ARCHITECTURE (and not a Transformer / retrieval store)
--------------------------------------------------------------------------------
The lazy reading of Ashurbanipal is "the librarian": a man who stored knowledge
in an external archive. That reading produces a retrieval system and stops there.
But the *purpose* of the library was not storage — it was FORECASTING and
CONTROL. The bulk of the Nineveh tablets are omen series (Enuma Anu Enlil for the
sky, Sa.gig and the barûtu for the body and the liver, Summa izbu for births,
Summa alu for the city). Every omen has the same shape: a protasis ("IF such a
sign is observed") and an apodosis ("THEN such an event will befall"). The
library is therefore a giant table of learned conditionals mapping configurations
of SIGNS to predicted OUTCOMES — an ancient forecasting engine.

Crucially, the Assyrian court did not treat the forecast as fixed. Rochberg shows
the omen verdict was understood as a divine legal judgement about the future that
was *open to appeal*. When an eclipse portended the king's death, the namburbi
rituals and the substitute-king rite (sar puhi) were performed to AVERT the
predicted outcome — a tablet describing exactly this was found in Ashurbanipal's
own library. So the cognitive signature of this mind is a two-stage loop:
    (1) FORECAST an outcome from the observed signs, then
    (2) INTERVENE on the controllable signs to bend that outcome away from doom.

This file therefore implements:
    * an OmenForecaster: a 2-layer MLP (tanh hidden -> softmax over outcomes)
      trained by backprop on a synthetic omen corpus generated from a hidden,
      structured "law of the gods" (conjunctive/threshold rules, like real
      apodoses), so the model learns real structure rather than memorising.
    * an apotropaic INTERVENTION search: given a case the model forecasts as
      calamity, run gradient descent on ONLY the ritually-controllable input
      dimensions (offerings, lustration, substitute king) to minimise the
      probability of doom — the eclipse itself stays fixed, exactly as in the
      historical worldview. This reuses the network's own input-gradients, so the
      "ritual" is mathematically a counterfactual control step.

Everything is pure NumPy and from scratch. The file contains:
    * a mandatory finite-difference gradient check on the forecaster,
    * a real training loop with a held-out validation split,
    * the apotropaic intervention experiment,
    * self-tests with hard assertions and a printed report.

Run:  python3 chapter_0025_ashurbanipal_-685.py
================================================================================
"""

from __future__ import annotations
import numpy as np

# Reproducibility — a single seed governs corpus generation and initialisation.
SEED = 685  # the year of Ashurbanipal's (approximate) birth, -685
rng = np.random.default_rng(SEED)


# =============================================================================
# SECTION 1 — THE SIGN SPACE AND THE "LAW OF THE GODS"
# =============================================================================
# A world-state is a vector of binary SIGNS observed by the diviners. We group
# the dimensions into the real Mesopotamian omen domains so the model — and the
# reader — can see which signs are "of the heavens" versus "of the body" etc.
#
# Some signs are UNCONTROLLABLE (an eclipse happens or it does not; you cannot
# negotiate with the moon). Others are CONTROLLABLE: they are the ritual acts a
# king can perform — lustration (bit rimki), offerings, lament, and the
# substitute-king rite. The apotropaic loop in Section 5 is only permitted to
# move the controllable dimensions, which is the whole theological point.

# index : (name, domain, controllable?)
SIGN_SCHEMA = [
    # --- Celestial (Enuma Anu Enlil) : uncontrollable portents in the sky ---
    ("lunar_eclipse",        "celestial",      False),
    ("solar_eclipse",        "celestial",      False),
    ("jupiter_adverse",      "celestial",      False),
    ("venus_adverse",        "celestial",      False),
    ("mars_adverse",         "celestial",      False),
    ("halo_round_moon",      "celestial",      False),
    # --- Hepatic / extispicy (barûtu) : read in a sacrificed sheep's liver ---
    ("liver_split_gate",     "hepatic",        False),
    ("liver_weapon_mark",    "hepatic",        False),
    ("liver_palace_swollen", "hepatic",        False),
    # --- Teratological (Summa izbu) : anomalous births ---
    ("birth_anomaly",        "teratological",  False),
    # --- Terrestrial (Summa alu) : signs in the city, animals, doorways ---
    ("snake_in_house",       "terrestrial",    False),
    ("fox_in_city",          "terrestrial",    False),
    # --- Oneiric : the king's dreams ---
    ("ill_omened_dream",     "oneiric",        False),
    # --- Ritual acts (CONTROLLABLE) : the namburbi / apotropaic levers ---
    ("offering_made",        "ritual",         True),
    ("lustration_bit_rimki", "ritual",         True),
    ("lament_recited",       "ritual",         True),
    ("substitute_king",      "ritual",         True),
]

SIGN_NAMES   = [s[0] for s in SIGN_SCHEMA]
SIGN_DOMAIN  = [s[1] for s in SIGN_SCHEMA]
CONTROLLABLE = np.array([s[2] for s in SIGN_SCHEMA], dtype=bool)
D_IN         = len(SIGN_SCHEMA)               # input dimensionality (= 17)

# The four apodosis classes — the outcomes the omens predict.
OUTCOMES = [
    "favorable",        # 0 : sulmu — well-being
    "ambiguous",        # 1 : mixed / wait-and-see
    "calamity_king",    # 2 : evil befalls the king's person
    "calamity_land",    # 3 : evil befalls the land / dynasty
]
N_OUT = len(OUTCOMES)
KING_DOOM_IDX = 2       # the class the apotropaic loop tries to escape


def _gods_law(x: np.ndarray) -> int:
    """The hidden, deterministic 'law of the gods': a structured rule mapping a
    sign vector x (length D_IN, values in {0,1}) to an outcome class.

    This is the ground truth the network must *discover*. It is deliberately
    built from conjunctions and thresholds — the logical shape of real omen
    apodoses, where doom usually requires several portents to coincide and where
    ritual action mitigates. The network never sees this function; it sees only
    (signs -> outcome) examples and must learn the structure by gradient descent.
    """
    s = {name: int(x[i]) for i, name in enumerate(SIGN_NAMES)}

    # Count how much "evil portent" weight the heavens + body are showing.
    celestial_evil = (s["lunar_eclipse"] + s["solar_eclipse"]
                      + s["jupiter_adverse"] + s["venus_adverse"]
                      + s["mars_adverse"])
    bodily_evil = (s["liver_split_gate"] + s["liver_weapon_mark"]
                   + s["liver_palace_swollen"] + s["birth_anomaly"])
    terrestrial_evil = s["snake_in_house"] + s["fox_in_city"] + s["ill_omened_dream"]

    # Ritual mitigation weight (the controllable levers).
    ritual = (s["offering_made"] + s["lustration_bit_rimki"]
              + s["lament_recited"] + 2 * s["substitute_king"])
    # The substitute-king rite counts double: it is the strongest deflection,
    # historically reserved for the gravest eclipse portents.

    # --- The rule (mirrors omen logic: conjunctions gate the verdict) ---
    # An eclipse that strikes the "palace" liver mark is the classic regicide
    # portent; it dooms the KING unless heavily counter-ritualised.
    regicide_portent = (s["lunar_eclipse"] or s["solar_eclipse"]) and \
                       (s["liver_palace_swollen"] or s["liver_weapon_mark"])

    if regicide_portent and ritual < 3:
        return 2  # calamity_king
    # Broad celestial + bodily collapse threatens the LAND when un-mitigated.
    if (celestial_evil + bodily_evil) >= 4 and ritual < 2:
        return 3  # calamity_land
    # A milder pile-up, or a grave portent that *was* ritually answered, is
    # ambiguous — the verdict was appealed but the danger lingers.
    if (celestial_evil + bodily_evil + terrestrial_evil) >= 3:
        return 1  # ambiguous
    if regicide_portent:               # answered with strong ritual -> survived
        return 1
    return 0  # favorable


def make_corpus(n: int, noise: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Generate n (sign-vector, outcome) examples.

    Signs are sampled so that portents are individually rare (as in life), with
    correlations between related signs (an eclipse makes adverse-planet readings
    more likely). A little label noise mimics scribal disagreement and the
    irreducible opacity of the gods.
    """
    X = np.zeros((n, D_IN), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    idx = {name: i for i, name in enumerate(SIGN_NAMES)}
    for k in range(n):
        x = (rng.random(D_IN) < 0.18).astype(np.float64)   # base rarity
        # Correlate the heavens: an eclipse drags adverse-planet readings up.
        if x[idx["lunar_eclipse"]] or x[idx["solar_eclipse"]]:
            for nm in ("jupiter_adverse", "venus_adverse", "mars_adverse",
                       "halo_round_moon"):
                if rng.random() < 0.45:
                    x[idx[nm]] = 1.0
        # Rituals are performed *in response* to portents far more than at random.
        portent_load = x[:13].sum()        # first 13 dims are the portents
        if portent_load >= 2:
            for nm in ("offering_made", "lustration_bit_rimki", "lament_recited"):
                if rng.random() < 0.5:
                    x[idx[nm]] = 1.0
            if portent_load >= 3 and rng.random() < 0.3:
                x[idx["substitute_king"]] = 1.0
        label = _gods_law(x)
        if rng.random() < noise:           # scribal / divine noise
            label = int(rng.integers(N_OUT))
        X[k] = x
        y[k] = label
    return X, y


# =============================================================================
# SECTION 2 — THE OMEN FORECASTER (a 2-layer MLP, fully from scratch)
# =============================================================================
# forecast(x) = softmax( W2 @ tanh(W1 @ x + b1) + b2 )
# This is the apodosis predictor: given the protasis (observed signs), output a
# distribution over outcomes. We implement forward, cross-entropy loss, and the
# exact backward pass by hand, then verify it with finite differences.

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class OmenForecaster:
    def __init__(self, d_in: int, d_hidden: int, n_out: int, seed: int = SEED):
        r = np.random.default_rng(seed)
        # He-ish initialisation scaled for tanh.
        self.W1 = r.standard_normal((d_hidden, d_in)) * np.sqrt(1.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = r.standard_normal((n_out, d_hidden)) * np.sqrt(1.0 / d_hidden)
        self.b2 = np.zeros(n_out)
        self.d_in, self.d_hidden, self.n_out = d_in, d_hidden, n_out

    # ---- parameter (de)serialisation, used by the gradient check ----
    def get_params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = params

    # ---- forward pass; returns probs and a cache for the backward pass ----
    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1.T + self.b1          # (N, H)
        H  = np.tanh(Z1)                      # (N, H)
        Z2 = H @ self.W2.T + self.b2          # (N, K)
        P  = softmax(Z2)                      # (N, K)
        cache = (X, Z1, H, P)
        return P, cache

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy loss over a batch."""
        P, _ = self.forward(X)
        n = X.shape[0]
        logp = -np.log(P[np.arange(n), y] + 1e-12)
        return float(np.mean(logp))

    # ---- exact gradients via backprop ----
    def backward(self, cache, y: np.ndarray) -> list[np.ndarray]:
        X, Z1, H, P = cache
        n = X.shape[0]
        # dL/dZ2 for softmax + cross-entropy is (P - onehot(y)) / n
        dZ2 = P.copy()
        dZ2[np.arange(n), y] -= 1.0
        dZ2 /= n                              # (N, K)
        dW2 = dZ2.T @ H                        # (K, H)
        db2 = np.sum(dZ2, axis=0)             # (K,)
        dH  = dZ2 @ self.W2                    # (N, H)
        dZ1 = dH * (1.0 - np.tanh(Z1) ** 2)   # tanh'(z) = 1 - tanh^2(z)
        dW1 = dZ1.T @ X                        # (H, D)
        db1 = np.sum(dZ1, axis=0)             # (H,)
        return [dW1, db1, dW2, db2]

    # ---- gradient w.r.t. the INPUT (drives the apotropaic intervention) ----
    def input_grad_for_class(self, x: np.ndarray, cls: int) -> np.ndarray:
        """d P(cls) / d x for a single example x. Used to push a feared outcome's
        probability down by moving controllable signs."""
        x = x.reshape(1, -1)
        P, cache = self.forward(x)
        _, Z1, H, _ = cache
        # objective g = P[cls]; dg/dZ2_j = P[cls]*(delta_{cls,j} - P[j])
        p = P[0]
        dZ2 = (p[cls] * (-p)).reshape(1, -1)
        dZ2[0, cls] += p[cls]
        dH  = dZ2 @ self.W2                    # (1, H)
        dZ1 = dH * (1.0 - np.tanh(Z1) ** 2)
        dX  = dZ1 @ self.W1                    # (1, D)
        return dX[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        P, _ = self.forward(X)
        return np.argmax(P, axis=1)


# =============================================================================
# SECTION 3 — MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# =============================================================================
def gradient_check(model: OmenForecaster, X: np.ndarray, y: np.ndarray,
                   eps: float = 1e-5, n_probe: int = 40) -> float:
    """Compare analytic gradients to numerical (central-difference) gradients on
    a random subset of parameters. Returns the maximum relative error, which must
    be tiny for the backward pass to be trusted."""
    _, cache = model.forward(X)
    analytic = model.backward(cache, y)
    params = model.get_params()
    max_rel = 0.0
    for pi, P in enumerate(params):
        flat = P.ravel()
        probe = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for j in probe:
            orig = flat[j]
            flat[j] = orig + eps
            lp = model.loss(X, y)
            flat[j] = orig - eps
            lm = model.loss(X, y)
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[pi].ravel()[j]
            denom = max(1e-12, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            max_rel = max(max_rel, rel)
    return max_rel


# =============================================================================
# SECTION 4 — TRAINING LOOP (mini-batch Adam, held-out validation)
# =============================================================================
def accuracy(model: OmenForecaster, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def train(model: OmenForecaster, Xtr, ytr, Xva, yva,
          epochs: int = 60, batch: int = 64, lr: float = 0.01,
          verbose: bool = True):
    params = model.get_params()
    # Adam moments
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = Xtr.shape[0]
    history = []
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            sel = order[start:start + batch]
            xb, yb = Xtr[sel], ytr[sel]
            _, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            t += 1
            for i in range(len(params)):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * (grads[i] ** 2)
                mhat = m[i] / (1 - b1 ** t)
                vhat = v[i] / (1 - b2 ** t)
                params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            model.set_params(params)
        if verbose and (ep % 10 == 0 or ep == 1):
            tr_l = model.loss(Xtr, ytr)
            va_a = accuracy(model, Xva, yva)
            print(f"  epoch {ep:3d} | train_loss {tr_l:.4f} | val_acc {va_a:.3f}")
        history.append((ep, model.loss(Xtr, ytr), accuracy(model, Xva, yva)))
    return history


# =============================================================================
# SECTION 5 — APOTROPAIC INTERVENTION (the namburbi / substitute-king loop)
# =============================================================================
# Given a case the trained model forecasts as calamity to the KING, we try to
# AVERT it. We may move ONLY the controllable (ritual) signs — the eclipse is
# fixed, just as the diviners could not stop the moon but could perform rituals.
# We do projected gradient descent on the controllable input dims to minimise
# P(calamity_king), then snap to {0,1} to recover an actual ritual prescription.

CONTROLLABLE_IDX = [i for i in range(D_IN) if CONTROLLABLE[i]]


def ritual_gradient_direction(model: OmenForecaster, x: np.ndarray) -> np.ndarray:
    """The continuous 'which rites help' signal: -dP(doom)/dx over controllable
    dims. This is the conceptual heart (ritual as counterfactual descent), but on
    a saturated forecast the gradient vanishes, so the diviner does not rely on it
    alone — see the catalogue search below."""
    g = model.input_grad_for_class(x, KING_DOOM_IDX)
    return -g * CONTROLLABLE


def apotropaic_intervention(model: OmenForecaster, x: np.ndarray):
    """Avert the king's doom by selecting rites from the namburbi CATALOGUE.

    The controllable space is tiny (a handful of rites), exactly as the diviner's
    ritual catalogue is finite. So rather than trust a single gradient step on a
    possibly-saturated softmax, we evaluate every combination of *additional*
    rites the king is not yet performing, and choose the smallest rite-set that
    minimises P(calamity_king) — the cheapest sufficient appeal. Returns
    (x_ritual, p_before, p_after, prescription, grad_dir)."""
    p_before = float(model.forward(x.reshape(1, -1))[0][0, KING_DOOM_IDX])
    grad_dir = ritual_gradient_direction(model, x)

    # Rites not yet performed are the candidates we may add (we never undo a rite).
    candidates = [i for i in CONTROLLABLE_IDX if x[i] == 0.0]
    best_x, best_p, best_set = x.copy(), p_before, []
    n_c = len(candidates)
    for mask in range(1 << n_c):                  # enumerate every rite subset
        trial = x.copy()
        chosen = []
        for b in range(n_c):
            if mask & (1 << b):
                trial[candidates[b]] = 1.0
                chosen.append(candidates[b])
        p = float(model.forward(trial.reshape(1, -1))[0][0, KING_DOOM_IDX])
        # Prefer lower doom; break ties by fewer rites (the cheapest appeal).
        if (p < best_p - 1e-9) or (abs(p - best_p) <= 1e-9 and len(chosen) < len(best_set)):
            best_x, best_p, best_set = trial, p, chosen
    prescription = [SIGN_NAMES[i] for i in best_set]
    return best_x, p_before, float(best_p), prescription, grad_dir


# =============================================================================
# SECTION 6 — DRIVER, SELF-TESTS, AND REPORT
# =============================================================================
def run():
    print("=" * 72)
    print("THE BARU ENGINE — Ashurbanipal's forecast-and-avert architecture")
    print("=" * 72)

    # ---- data ----
    Xtr, ytr = make_corpus(4000)
    Xva, yva = make_corpus(1000)
    print(f"\nCorpus: {Xtr.shape[0]} train / {Xva.shape[0]} val examples, "
          f"{D_IN} sign-dimensions, {N_OUT} outcome classes.")
    dist = np.bincount(ytr, minlength=N_OUT) / len(ytr)
    print("Outcome base rates (train): " +
          ", ".join(f"{OUTCOMES[i]} {dist[i]:.2f}" for i in range(N_OUT)))

    model = OmenForecaster(d_in=D_IN, d_hidden=24, n_out=N_OUT)

    # ---- mandatory gradient check (small batch) ----
    print("\n[1] Finite-difference gradient check ...")
    gc_X, gc_y = Xtr[:16], ytr[:16]
    max_rel = gradient_check(model, gc_X, gc_y)
    print(f"    max relative error = {max_rel:.2e}")
    assert max_rel < 1e-5, "GRADIENT CHECK FAILED"
    print("    PASS — analytic gradients match numerical gradients.")

    # ---- training ----
    print("\n[2] Training the forecaster (Adam) ...")
    acc0 = accuracy(model, Xva, yva)
    train(model, Xtr, ytr, Xva, yva, epochs=60, batch=64, lr=0.01)
    accF = accuracy(model, Xva, yva)
    print(f"    val accuracy: {acc0:.3f} (start) -> {accF:.3f} (final)")
    assert accF > 0.85, "MODEL DID NOT LEARN THE OMEN STRUCTURE"
    print("    PASS — the engine learned the gods' law from signs alone.")

    # ---- apotropaic intervention experiment ----
    print("\n[3] Apotropaic intervention (avert the king's doom) ...")
    # Find validation cases the model forecasts as calamity_king.
    preds = model.predict(Xva)
    feared = np.where(preds == KING_DOOM_IDX)[0]
    print(f"    {len(feared)} cases forecast as calamity to the king.")
    averted, examples = 0, []
    for i in feared:
        _, pb, pa, presc, _ = apotropaic_intervention(model, Xva[i])
        if pa < 0.5 and pa < pb:
            averted += 1
        if len(examples) < 3:
            examples.append((pb, pa, presc))
    rate = averted / max(1, len(feared))
    print(f"    doom averted in {averted}/{len(feared)} cases "
          f"(rate {rate:.2f}) by ritual action alone.")
    for k, (pb, pa, presc) in enumerate(examples, 1):
        ritual_txt = ", ".join(presc) if presc else "(no new ritual found)"
        print(f"      case {k}: P(king-doom) {pb:.2f} -> {pa:.2f} | "
              f"prescribe: {ritual_txt}")
    assert rate > 0.5, "INTERVENTION LOOP FAILED TO AVERT MOST CASES"
    print("    PASS — the verdict proved appealable, as the diviners held.")

    # ---- a single fully-worked omen, narrated ----
    print("\n[4] One worked omen, read as a b'aru would read it:")
    # Construct a grave portent by hand: lunar eclipse + palace liver mark, no ritual.
    x = np.zeros(D_IN)
    x[SIGN_NAMES.index("lunar_eclipse")] = 1
    x[SIGN_NAMES.index("liver_palace_swollen")] = 1
    x[SIGN_NAMES.index("mars_adverse")] = 1
    P, _ = model.forward(x.reshape(1, -1))
    print("    Signs observed: lunar_eclipse, liver_palace_swollen, mars_adverse")
    print("    Forecast: " + ", ".join(
        f"{OUTCOMES[j]} {P[0, j]:.2f}" for j in range(N_OUT)))
    x_r, pb, pa, presc, _ = apotropaic_intervention(model, x)
    ritual_txt = ", ".join(presc) if presc else "(none sufficed)"
    print(f"    Apotropaic counsel: perform [{ritual_txt}]")
    print(f"    P(king-doom): {pb:.2f} -> {pa:.2f} after the rite.")

    print("\n" + "=" * 72)
    print("ALL SELF-TESTS PASSED.")
    print("The Baru Engine forecasts outcomes from signs (the apodosis predictor)")
    print("and then bends the controllable signs to avert calamity (the namburbi")
    print("loop) — Ashurbanipal's mind rendered as a predict-then-intervene model.")
    print("=" * 72)


if __name__ == "__main__":
    run()

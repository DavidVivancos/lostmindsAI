"""
chapter_0028_sappho_-630.py
============================================================================
LYRE  —  Lyric Introspective Recurrence with Eros-indexed valuation
A from-scratch, pure-NumPy, trainable neural architecture that embodies the
distinctive cognitive signature of SAPPHO of Lesbos (c. 630 – c. 570 BCE).
# Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
Author: David Vivancos · Chapter 0028 · Sappho
WHY THIS ARCHITECTURE, AND NOT A TRANSFORMER
----------------------------------------------------------------------------
Most "affective AI" sketches reach for an emotion-classifier bolted onto a
language model. That is NOT Sappho. Two ideas are hers and nobody else's,
and they are literally present in the surviving Greek text:

 1.  THE PRIAMEL AS A VALUE FUNCTION  (Fragment 16).
     "Some say a host of cavalry, some of infantry, some of ships is the
     most beautiful thing on the dark earth — but I say it is whatsoever one
     loves."  This is not relativism. It is an *evaluation rule* whose answer
     is not a property of any object but a POINTER indexed to the desiring
     subject. The crowd's canonical value-objects (horsemen / footmen / ships)
     are deliberately enumerated as FOILS, then OVERRIDDEN by a first-person
     desire vector. Value is "whatever eros selects," not a score on things.
     => We implement a PriamelGate: a menu of K learned value-prototypes is
        scored, but the emitted value is the projection of the desire-aligned
        object onto a learned DESIRE vector d. Desire overrides the crowd's
        argmax. That pivot — "but I say" — is the forward pass.

 2.  THE THIRD-PERSON SYMPTOM CATALOGUE UNDER RISING GAIN  (Fragment 31).
     "phainetai moi" — *it appears to me*. Sappho watches her own collapse
     from the outside and enumerates the failing subsystems in order: the
     tongue breaks, a thin fire runs under the skin, the eyes go dark, the
     ears ring, cold sweat, trembling, "greener than grass," near death.
     The catalogue is ORDERED and ESCALATING, and it is narrated by a
     detached OBSERVER channel split off from the experiencing self.
     => We implement a SymptomReadout that decomposes the heart-state into S
        named somatic channels, multiplied by a rising AROUSAL GAIN so later
        symptoms intensify; and a MetaObserver channel that predicts the
        "phainetai moi" detachment signal (feeling vs. watching-oneself-feel).

 3.  THE SAPPHIC STANZA AS A CARRIER  (the meter).
     Three "lesser Sapphic" hendecasyllables (rising pressure) followed by one
     short adonic line (release): a fixed template of long/short positions.
     => A MetricalDecoder emits onto that fixed scaffold; the adonic is a
        learned CONTRACTION (closure) step. Build-and-release is structural,
        not decorative.

THE LEARNABLE TASK (so this is a real model, not a demo)
----------------------------------------------------------------------------
Each training example is a "scene" vector describing who/what is present and
who is loved, generated from a principled priamel+symptom grammar. The network
must, end to end:
   (a) output the desire-indexed value  (the "kalliston" target),
   (b) reconstruct the ordered, gain-scaled symptom catalogue,
   (c) emit a metrically valid stress pattern over the Sapphic template,
   (d) predict the detachment / observer scalar.
Loss = value MSE + symptom MSE + meter cross-entropy + observer BCE.

Everything below is hand-written NumPy with manual backpropagation, a
mandatory finite-difference gradient check, a real training loop that lowers
the loss, and self-tests. Run:  python chapter_0028_sappho_-630.py
============================================================================
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# 0.  THE SAPPHIC METRICAL TEMPLATE
#     Long = 1, Short = 0, Anceps (free) = -1.  Lesser Sapphic hendecasyllable:
#         –  u  –  x  –  u  u  –  u  –  –
#     Adonic (the short closing line):
#         –  u  u  –  –
#     The decoder predicts a stress symbol per slot; anceps slots accept either.
# ----------------------------------------------------------------------------
LESSER_SAPPHIC = [1, 0, 1, -1, 1, 0, 0, 1, 0, 1, 1]   # 11 slots
ADONIC         = [1, 0, 0, 1, 1]                       # 5 slots
# One full stanza = 3 hendecasyllables + 1 adonic, flattened into a slot list:
STANZA_TEMPLATE = LESSER_SAPPHIC * 3 + ADONIC          # 38 slots
N_SLOTS = len(STANZA_TEMPLATE)                         # 38
STRESS_CLASSES = 2                                     # short(0) / long(1)

# The named somatic channels of Fragment 31, in the poem's escalating order.
SYMPTOMS = [
    "tongue_breaks",      # γλῶσσα ἔαγε
    "thin_fire_skin",     # λέπτον πῦρ ὑποδεδρόμηκεν
    "eyes_darken",        # ὀππάτεσσι δ' οὐδ' ἒν ὄρημμ'
    "ears_ring",          # ἐπιρρόμβεισι δ' ἄκουαι
    "cold_sweat",         # κὰδ δέ μ' ἴδρως ψῦχρος ἔχει
    "trembling",          # τρόμος δὲ παῖσαν ἄγρει
    "greener_than_grass", # χλωροτέρα δὲ ποίας
    "near_death",         # τεθνάκην ὀλίγω 'πιδεύης
]
N_SYMPTOMS = len(SYMPTOMS)                             # 8

# The "crowd's" conventional value-objects from the Fragment 16 priamel.
PRIAMEL_OBJECTS = ["cavalry", "infantry", "ships"]      # the foils
K_OBJECTS = len(PRIAMEL_OBJECTS)                        # 3


# ----------------------------------------------------------------------------
# 1.  SMALL NUMERIC HELPERS  (all differentiable; gradients hand-derived below)
# ----------------------------------------------------------------------------
def tanh(x):                       return np.tanh(x)
def dtanh(y):                       return 1.0 - y * y            # given y = tanh(x)
def sigmoid(x):                    return 1.0 / (1.0 + np.exp(-x))
def dsigmoid(y):                   return y * (1.0 - y)          # given y = sigmoid(x)

def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------------------------------------------------------
# 2.  THE MODEL
#     A compact, fully hand-differentiated network. State sizes are kept small
#     so the finite-difference gradient check and training stay fast and exact.
# ----------------------------------------------------------------------------
class LYRE:
    """
    Forward pass (one scene -> all four heads):

        scene  x (D_in)
          │  encode
          ▼
        heart  h = tanh(Wxh x + b_h)           ── the recurrent affective state
          │
          ├─ PriamelGate ─────────────────────► value  (scalar)
          │     menu scores  s_k = Wobj h
          │     desire d (learned vector in object-embedding space)
          │     attention a = softmax(s)        (the crowd's weighting)
          │     desire override w = softmax(s + beta * (Eobj·d))
          │     value = Σ_k w_k * (Eobj_k · d)   ── "I say it is whatsoever one loves"
          │
          ├─ SymptomReadout ──────────────────► symptoms (N_SYMPTOMS)
          │     raw  r = Wsym h                 (componentwise introspection)
          │     gain g_i = 1 + softplus(gamma)·(i/(N-1))   rising arousal envelope
          │     symptoms = tanh(r) * g          ── the ordered, escalating catalogue
          │
          ├─ MetaObserver ────────────────────► detachment p in (0,1)
          │     observer reads catalogue magnitude:  p = sigmoid(wobs·|symptoms| + bobs)
          │     "phainetai moi": the self that watches the self feel
          │
          └─ MetricalDecoder ─────────────────► stress logits (N_SLOTS x 2)
                emits long/short onto the fixed Sapphic scaffold;
                slot bias encodes the template (build x3, then adonic release).
    """

    def __init__(self, d_in=12, d_h=16, d_obj=6, seed=0):
        rng = np.random.default_rng(seed)
        sc = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)

        self.d_in, self.d_h, self.d_obj = d_in, d_h, d_obj

        # encoder: scene -> heart
        self.Wxh = sc(d_in, d_h)
        self.b_h = np.zeros(d_h)

        # priamel gate
        self.Wobj = sc(d_h, K_OBJECTS)              # score the 3 conventional objects
        self.Eobj = sc(K_OBJECTS, d_obj)            # object embeddings (horsemen/...)
        self.d_desire = rng.standard_normal(d_obj) * 0.5   # the learned desire vector
        self.beta = np.array(1.0)                   # strength of the "but I say" override

        # symptom readout
        self.Wsym = sc(d_h, N_SYMPTOMS)
        self.gamma = np.array(0.0)                  # controls steepness of arousal gain

        # meta observer
        self.wobs = rng.standard_normal(N_SYMPTOMS) * 0.3
        self.Wobs_h = rng.standard_normal(d_h) * 0.1   # observer also reads the heart
        self.bobs = np.array(0.0)

        # metrical decoder
        self.Wmet = sc(d_h, N_SLOTS * STRESS_CLASSES)
        # template bias: nudge each slot toward its canonical long/short value
        tb = np.zeros((N_SLOTS, STRESS_CLASSES))
        for j, sym in enumerate(STANZA_TEMPLATE):
            if sym == 1:   tb[j, 1] = 0.5
            elif sym == 0: tb[j, 0] = 0.5
        self.b_met = tb.reshape(-1)

    # -- parameter plumbing for the gradient check / optimiser -----------------
    def params(self):
        return {
            "Wxh": self.Wxh, "b_h": self.b_h,
            "Wobj": self.Wobj, "Eobj": self.Eobj,
            "d_desire": self.d_desire, "beta": self.beta,
            "Wsym": self.Wsym, "gamma": self.gamma,
            "wobs": self.wobs, "Wobs_h": self.Wobs_h, "bobs": self.bobs,
            "Wmet": self.Wmet, "b_met": self.b_met,
        }

    def set_params(self, flat):
        i = 0
        for k, v in self.params().items():
            n = v.size
            getattr(self, k)[...] = flat[i:i + n].reshape(v.shape)
            i += n

    def get_flat(self):
        return np.concatenate([v.reshape(-1) for v in self.params().values()])

    # ----------------------------------------------------------------------
    # FORWARD  — returns predictions plus a cache for manual backprop
    # ----------------------------------------------------------------------
    def forward(self, x):
        cache = {"x": x}

        # heart
        zh = x @ self.Wxh + self.b_h
        h = tanh(zh)
        cache.update(zh=zh, h=h)

        # ---- priamel gate ----
        s = h @ self.Wobj                     # (K,) crowd scores
        align = self.Eobj @ self.d_desire     # (K,) desire alignment per object
        w = softmax(s + self.beta * align)    # desire-overridden weighting
        value = np.dot(w, align)              # "whatsoever one loves"
        cache.update(s=s, align=align, w=w, value=value)

        # ---- symptom readout ----
        r = h @ self.Wsym                     # (S,)
        tr = tanh(r)
        idx = np.arange(N_SYMPTOMS) / (N_SYMPTOMS - 1)
        gain = 1.0 + np.log1p(np.exp(self.gamma)) * idx   # softplus(gamma) scaled ramp
        symptoms = tr * gain
        cache.update(r=r, tr=tr, gain=gain, idx=idx, symptoms=symptoms)

        # ---- meta observer ----
        mag = np.abs(symptoms)
        obs_z = np.dot(self.wobs, mag) + np.dot(self.Wobs_h, h) + self.bobs
        detach = sigmoid(obs_z)
        cache.update(mag=mag, obs_z=obs_z, detach=detach)

        # ---- metrical decoder ----
        met_lin = h @ self.Wmet + self.b_met            # (N_SLOTS*2,)
        met_logits = met_lin.reshape(N_SLOTS, STRESS_CLASSES)
        met_prob = softmax(met_logits, axis=1)
        cache.update(met_logits=met_logits, met_prob=met_prob)

        pred = {"value": value, "symptoms": symptoms,
                "detach": detach, "met_prob": met_prob}
        return pred, cache

    # ----------------------------------------------------------------------
    # LOSS  — four heads combined
    # ----------------------------------------------------------------------
    def loss(self, pred, target):
        # value: MSE
        Lv = 0.5 * (pred["value"] - target["value"]) ** 2
        # symptoms: MSE over the catalogue
        ds = pred["symptoms"] - target["symptoms"]
        Ls = 0.5 * np.mean(ds * ds)
        # detachment: binary cross-entropy
        p = np.clip(pred["detach"], 1e-7, 1 - 1e-7)
        y = target["detach"]
        Lo = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        # meter: cross-entropy over each slot
        labels = target["meter"]                     # (N_SLOTS,) int in {0,1}
        pp = np.clip(pred["met_prob"][np.arange(N_SLOTS), labels], 1e-7, 1.0)
        Lm = -np.mean(np.log(pp))
        total = Lv + Ls + Lo + Lm
        parts = {"value": float(Lv), "symptom": float(Ls),
                 "observer": float(Lo), "meter": float(Lm)}
        return total, parts

    # ----------------------------------------------------------------------
    # BACKWARD  — manual gradients for every parameter
    # ----------------------------------------------------------------------
    def backward(self, cache, target):
        g = {k: np.zeros_like(v) for k, v in self.params().items()}
        h = cache["h"]

        dh = np.zeros_like(h)   # accumulate gradient into the heart state

        # ===== value head =====
        # value = w·align ; w = softmax(s + beta*align) ; s = h@Wobj
        dvalue = (cache["value"] - target["value"])          # dL/dvalue
        w, align = cache["w"], cache["align"]
        # d value / d w_k = align_k ; d value / d align_k = w_k (align also inside w)
        # First treat value = Σ w_k align_k. Backprop through both w and align.
        dw = dvalue * align                                  # via explicit w
        dalign = dvalue * w                                  # via explicit align factor
        # softmax: u = s + beta*align ; w = softmax(u)
        # dL/du = w * (dw - (w·dw))
        wdotdw = np.dot(w, dw)
        du = w * (dw - wdotdw)
        # u depends on s and on beta*align
        ds = du.copy()
        dalign += self.beta * du
        g["beta"] += np.dot(du, align)
        # align = Eobj @ d_desire
        g["Eobj"] += np.outer(dalign, self.d_desire)
        g["d_desire"] += self.Eobj.T @ dalign
        # s = h @ Wobj
        g["Wobj"] += np.outer(h, ds)
        dh += self.Wobj @ ds

        # ===== symptom head =====
        # symptoms = tanh(r) * gain ; gain = 1 + softplus(gamma)*idx ; r = h@Wsym
        ds_sym = (cache["symptoms"] - target["symptoms"]) / N_SYMPTOMS
        tr, gain, idx = cache["tr"], cache["gain"], cache["idx"]
        dtr = ds_sym * gain
        dgain = ds_sym * tr
        # gain wrt gamma : softplus'(gamma) = sigmoid(gamma)
        sp_grad = sigmoid(self.gamma)
        g["gamma"] += np.sum(dgain * idx) * sp_grad
        dr = dtr * dtanh(tr)
        g["Wsym"] += np.outer(h, dr)
        dh += self.Wsym @ dr
        # symptoms also feed the observer (through |symptoms|); handle next:

        # ===== observer head =====
        p = cache["detach"]
        y = target["detach"]
        dobs_z = (p - y)                          # d BCE / d obs_z (sigmoid+BCE)
        g["wobs"] += dobs_z * cache["mag"]
        g["Wobs_h"] += dobs_z * h
        g["bobs"] += dobs_z
        dh += dobs_z * self.Wobs_h
        # mag = |symptoms| -> d|s|/ds = sign(s); push back into symptoms
        dmag = dobs_z * self.wobs
        dsym_from_obs = dmag * np.sign(cache["symptoms"])
        # route through symptoms = tr*gain again
        dtr2 = dsym_from_obs * gain
        dgain2 = dsym_from_obs * tr
        g["gamma"] += np.sum(dgain2 * idx) * sp_grad
        dr2 = dtr2 * dtanh(tr)
        g["Wsym"] += np.outer(h, dr2)
        dh += self.Wsym @ dr2

        # ===== meter head =====
        labels = target["meter"]
        dmet = cache["met_prob"].copy()
        dmet[np.arange(N_SLOTS), labels] -= 1.0
        dmet /= N_SLOTS                           # mean over slots
        dmet_lin = dmet.reshape(-1)
        g["Wmet"] += np.outer(h, dmet_lin)
        g["b_met"] += dmet_lin
        dh += self.Wmet @ dmet_lin

        # ===== back into the heart / encoder =====
        dzh = dh * dtanh(cache["h"])
        g["Wxh"] += np.outer(cache["x"], dzh)
        g["b_h"] += dzh

        return g

    def grad_flat(self, cache, target):
        g = self.backward(cache, target)
        return np.concatenate([g[k].reshape(-1) for k in self.params().keys()])


# ----------------------------------------------------------------------------
# 3.  PRINCIPLED SYNTHETIC DATA  (the priamel + symptom grammar)
#     A "scene" encodes which of the 3 conventional objects the crowd praises,
#     plus a hidden "loved object" embedding. Targets are derived analytically
#     so the network has a real function to learn — not noise.
# ----------------------------------------------------------------------------
def make_scene(rng, model):
    x = rng.standard_normal(model.d_in) * 0.6

    # ground-truth desire-indexed value: project the scene's loved-object cue
    # onto the model's *current* desire direction analogue (a fixed teacher d*).
    teacher_d = np.array([0.9, -0.4, 0.6, 0.2, -0.7, 0.3])[:model.d_obj]
    loved = x[:model.d_obj]
    value_t = float(np.tanh(np.dot(loved, teacher_d)))

    # symptom catalogue: an escalating profile driven by arousal = |value|
    arousal = abs(value_t)
    base = np.linspace(0.2, 1.0, N_SYMPTOMS)            # later symptoms heavier
    symptoms_t = np.tanh(base * (0.5 + 1.5 * arousal))

    # detachment ("phainetai moi"): the observing self splits off more sharply
    # the more violent the catalogue. A steeper logistic on arousal gives the
    # observer head a genuine, high-variance signal to learn (not a constant).
    detach_t = float(1.0 / (1.0 + np.exp(-(6.0 * arousal - 3.0))))

    # meter labels: resolve the fixed template, with anceps -> arousal-biased
    meter_t = []
    for sym in STANZA_TEMPLATE:
        if sym == -1:
            meter_t.append(1 if arousal > 0.5 else 0)   # tense scenes go "long"
        else:
            meter_t.append(sym)
    meter_t = np.array(meter_t, dtype=int)

    target = {"value": value_t, "symptoms": symptoms_t,
              "detach": detach_t, "meter": meter_t}
    return x, target


def make_dataset(model, n, seed=1):
    rng = np.random.default_rng(seed)
    return [make_scene(rng, model) for _ in range(n)]


# ----------------------------------------------------------------------------
# 4.  GRADIENT CHECK  (MANDATORY)  — central finite differences vs. analytic
# ----------------------------------------------------------------------------
def gradient_check(verbose=True):
    model = LYRE(seed=3)
    rng = np.random.default_rng(7)
    x, target = make_scene(rng, model)

    pred, cache = model.forward(x)
    analytic = model.grad_flat(cache, target)

    theta = model.get_flat().copy()
    eps = 1e-5
    numeric = np.zeros_like(theta)
    # check a representative subset of coordinates for speed, spread across params
    n = theta.size
    idxs = np.linspace(0, n - 1, 120).astype(int)
    idxs = np.unique(idxs)
    for i in idxs:
        tp = theta.copy(); tp[i] += eps
        model.set_params(tp)
        p1, _ = model.forward(x); L1, _ = model.loss(p1, target)
        tm = theta.copy(); tm[i] -= eps
        model.set_params(tm)
        p2, _ = model.forward(x); L2, _ = model.loss(p2, target)
        numeric[i] = (L1 - L2) / (2 * eps)
    model.set_params(theta)  # restore

    a = analytic[idxs]; nmr = numeric[idxs]
    rel = np.abs(a - nmr) / (np.abs(a) + np.abs(nmr) + 1e-12)
    max_rel = float(np.max(rel))
    if verbose:
        print(f"[grad-check] checked {len(idxs)} params  "
              f"max relative error = {max_rel:.3e}")
    return max_rel


# ----------------------------------------------------------------------------
# 5.  TRAINING LOOP  (plain SGD with momentum; the loss must fall)
# ----------------------------------------------------------------------------
def train(model, data, epochs=60, lr=0.05, mom=0.9, verbose=True):
    theta = model.get_flat().copy()
    velocity = np.zeros_like(theta)
    history = []
    for ep in range(epochs):
        np.random.shuffle(data)
        tot = 0.0
        grad_acc = np.zeros_like(theta)
        for x, target in data:
            model.set_params(theta)
            pred, cache = model.forward(x)
            L, _ = model.loss(pred, target)
            tot += L
            grad_acc += model.grad_flat(cache, target)
        grad_acc /= len(data)
        velocity = mom * velocity - lr * grad_acc
        theta = theta + velocity
        avg = tot / len(data)
        history.append(avg)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}   mean loss = {avg:.5f}")
    model.set_params(theta)
    return history


# ----------------------------------------------------------------------------
# 6.  SELF-TESTS + DEMONSTRATION
# ----------------------------------------------------------------------------
def evaluate(model, data):
    v_err = sym_err = obs_err = 0.0
    meter_correct = meter_total = 0
    for x, target in data:
        pred, _ = model.forward(x)
        v_err += abs(pred["value"] - target["value"])
        sym_err += np.mean(np.abs(pred["symptoms"] - target["symptoms"]))
        obs_err += abs(pred["detach"] - target["detach"])
        guess = np.argmax(pred["met_prob"], axis=1)
        meter_correct += int(np.sum(guess == target["meter"]))
        meter_total += N_SLOTS
    n = len(data)
    return {
        "value_MAE": v_err / n,
        "symptom_MAE": sym_err / n,
        "detach_MAE": obs_err / n,
        "meter_acc": meter_correct / meter_total,
    }


def demonstrate(model):
    """Run one scene and narrate the four heads in Sappho's own terms."""
    rng = np.random.default_rng(99)
    x, target = make_scene(rng, model)
    pred, _ = model.forward(x)

    print("\n--- A single scene through Sappho's mind (LYRE) ---")
    print("PRIAMEL  (Fragment 16):")
    print("   the crowd praises:", ", ".join(PRIAMEL_OBJECTS),
          "  (these are the foils)")
    print(f'   "but I say it is whatsoever one loves" -> value = {pred["value"]:+.3f}'
          f'   (target {target["value"]:+.3f})')

    print("\nSYMPTOM CATALOGUE  (Fragment 31, escalating):")
    for name, s in zip(SYMPTOMS, pred["symptoms"]):
        bar = "#" * int(abs(s) * 20)
        print(f"   {name:18s} {s:+.3f}  {bar}")

    print(f'\nMETA-OBSERVER ("phainetai moi") detachment = {pred["detach"]:.3f}')
    print("   (the self that stands aloof and watches the self dissolve)")

    guess = np.argmax(pred["met_prob"], axis=1)
    glyphs = "".join("\u2013" if b else "u" for b in guess)  # – long / u short
    print("\nSAPPHIC STANZA stress pattern (3 hendecasyllables + adonic):")
    print("   line1:", glyphs[0:11])
    print("   line2:", glyphs[11:22])
    print("   line3:", glyphs[22:33])
    print("   adonic:", glyphs[33:38], " <- the short release")


def main():
    print("=" * 74)
    print("LYRE — Sappho's architecture (28_Neuron.py)")
    print("=" * 74)

    # (1) MANDATORY gradient check
    max_rel = gradient_check(verbose=True)
    assert max_rel < 1e-4, f"gradient check FAILED: max rel err {max_rel:.3e}"
    print("[grad-check] PASSED (analytic gradients match finite differences)\n")

    # (2) data + before/after training
    model = LYRE(seed=0)
    train_data = make_dataset(model, n=200, seed=1)
    test_data = make_dataset(model, n=60, seed=2)

    before = evaluate(model, test_data)
    print("Before training:", {k: round(v, 4) for k, v in before.items()})

    print("\nTraining LYRE (SGD + momentum):")
    history = train(model, train_data, epochs=60, lr=0.05, verbose=True)

    after = evaluate(model, test_data)
    print("\nAfter training: ", {k: round(v, 4) for k, v in after.items()})

    # (3) self-tests: loss fell and heads improved
    assert history[-1] < history[0], "training did not reduce the loss"
    assert after["meter_acc"] >= before["meter_acc"], "meter head did not improve"
    assert after["value_MAE"] <= before["value_MAE"] + 1e-9, "value head regressed"
    assert after["detach_MAE"] <= before["detach_MAE"] + 1e-9, "observer regressed"
    print(f"\n[self-test] loss fell {history[0]:.4f} -> {history[-1]:.4f}  PASSED")
    print(f"[self-test] meter accuracy {before['meter_acc']:.2f} -> "
          f"{after['meter_acc']:.2f}  PASSED")

    # (4) demonstration
    demonstrate(model)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

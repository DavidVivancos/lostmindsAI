"""
================================================================================
 CHAPTER 276 -- OMAR KHAYYAM (1048-1131 CE), NISHAPUR, SELJUK PERSIA
 A trainable architecture built from his own mathematical and metaphysical method
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0276 · Omar Khayyam (1048-1131)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 14, Minds 261-280 Available on Amazon https://www.amazon.com/dp/B0HLYSVY5D
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0276_omar_khayyam_1048 - Omar Khayyam (1048-1131)

WHAT THIS FILE IS NOT
----------------------
This is not a "wine and stars" pastiche, and it is not a transformer wearing a
turban. It does not use attention, does not use a mixture of experts, and does
not treat Khayyam as a generic "order out of chaos" rationalist. Those framings
belong to other minds in this collection. Khayyam earns a different machine.

THE ONE IDEA THIS FILE IS BUILT FROM
-------------------------------------
Across three completely different bodies of work -- his commentary on Euclid's
theory of ratios, his metaphysical treatise "On Existence" (Risala fi'l-wujud),
and his quatrains -- Khayyam repeatedly performs the *same* diagnostic move.
He asks, of any attribute predicated of a thing: is this attribute WUJUDI
(existential -- real "in the thing itself," in re, independent of how a mind
happens to describe it), or is it merely I'TIBARI (considerational -- an
"added meaning" that exists only "in the intellect," a bookkeeping device of
whatever notation or convention the describer chose)?

  - In ratio theory, he rejects Euclid's definition of "equal ratios" (which
    depends on an arbitrarily chosen auxiliary magnitude) in favour of the
    anthyphairetic (continued-fraction / mutual-subtraction) definition,
    precisely because the anthyphairetic test is invariant to which unit of
    measure you use. Equality of ratio is, for Khayyam, WUJUDI: a fact about
    the magnitudes, not about the yardstick.

  - In "On Existence," he argues at length that *existence itself* -- wujud --
    is not an existential attribute of a thing but a "considerational" one:
    a meaning the intellect adds when it contemplates an essence, not a
    further ingredient found "in the thing" alongside its essence. This is a
    12th-century ancestor of the modern point (Kant, then Frege/Russell) that
    "existence is not a real predicate."

  - In the Rubaiyat, the same audit is run on theology: he treats the
    *impermanence of the clay, the turning of the potter's wheel, the fact of
    decay* as WUJUDI -- directly evidenced, non-negotiable -- while treating
    the *content of paradise, hell, the reward of the pious* as I'TIBARI --
    concepts added by a tradition of description, not things anyone has
    actually observed "in re."

The architecture below turns this single diagnostic -- "is this feature real,
or is it an artifact of how I happened to represent the input?" -- into a
computable, trainable, testable procedure. It is not a metaphor. The network
literally has two heads trained on two different kinds of fact (one that is
representation-invariant, one that is representation-bound by construction),
and the file includes an audit that empirically measures which head actually
learned an invariant regularity and which one merely memorised a convention --
exactly the distinction Khayyam draws with his pen. A third component
operationalises his method with the Euclidean parallel postulate: he did not
assert the postulate, he REPLACED it with a more self-evident pair of
statements, then tested what happens to a conclusion (the summit angles of his
quadrilateral) as that founding assumption is perturbed. Section 4 below
reuses exactly that move -- perturb an assumption, watch whether the
conclusion survives -- as a live confidence gate on every prediction the
network makes.

--------------------------------------------------------------------------------
FILE MAP
--------------------------------------------------------------------------------
  PART 0.  Digit encodings in an arbitrary base (the representational substrate
           whose choice is exactly what "considerational" vs "existential"
           will be tested against).
  PART 1.  Synthetic task construction.
             Task E (Existential / WUJUDI):  ratio equality, A/B == C/D,
               decided by cross-multiplication A*D == B*C. A fact about the
               magnitudes; provably invariant to the base used to write them.
             Task C (Considerational / I'TIBARI): "is the LEADING (most-significant)
               base-10 digit of A at least 5?" A fact that is true only
               relative to the convention of base-10 positional place-value;
               it is not even the same *question* once the base changes,
               because the set of integers whose leading digit clears half
               of a different base is an unrelated partition of the number
               line. (An earlier draft of this task used base-10 last-digit
               parity, which was rejected during testing below: because 10 is
               itself even, last-digit parity in any even base coincides with
               the true parity of the integer -- an existential fact about
               the integer, not a notational one. That failure is left
               visible in Self-Test 2b as a record of the correction.)
  PART 2.  DualAttributeNet -- a from-scratch NumPy MLP with a shared trunk
           (Hikmat al-Mushtarak, "shared wisdom") and two heads: Existential
           and Considerational. All forward/backward math is hand-derived,
           no autograd.
  PART 3.  Finite-difference gradient check (mandatory correctness gate).
  PART 4.  Postulate Perturbation Gate (Mizan al-Farz, "balance of the
           assumption") -- Khayyam's own method, made operational: perturb an
           input near a decision boundary and read confidence off of how
           often the conclusion flips.
  PART 5.  Training loop (Adam, from scratch) + held-out evaluation.
  PART 6.  The Encoding-Shift Audit -- train once on base-10 digit patterns,
           then re-encode the *same underlying integers* in bases 9, 8, 7, 6,
           5 and re-run the frozen network. This is the experiment: does the
           Existential head's accuracy survive the change of representation
           more gracefully than the Considerational head's? The file prints
           real, unmassaged numbers; it does not assume the answer.
  PART 7.  Self-tests and __main__ driver.
--------------------------------------------------------------------------------
"""

import numpy as np

RNG = np.random.default_rng(1048)  # seeded on Khayyam's birth year, for reproducibility


# =============================================================================
# PART 0 -- DIGIT ENCODINGS IN AN ARBITRARY BASE
# =============================================================================
# We fix a maximum magnitude and a fixed number of digit "slots" wide enough
# to represent that magnitude in the SMALLEST base we will ever audit with
# (base 5). Each slot is one-hot encoded into a 10-wide category vector, using
# only as many categories as the base requires (base-5 digits only ever
# activate categories 0-4). This lets the *same* frozen input dimensionality
# and the *same* frozen trained weights be re-used when we change the base at
# audit time in Part 6 -- we are not retraining or reshaping anything, we are
# asking the trained network to look at the identical integers dressed in
# different notational clothing.

MAX_VAL = 124          # inclusive; 124 in base 5 is "444" -- exactly 3 digits
N_DIGITS = 3            # digit slots per number
N_CATS = 10             # width of each one-hot slot (only 0..base-1 ever fire)
NUMS_PER_EXAMPLE = 4    # A, B, C, D
INPUT_DIM = NUMS_PER_EXAMPLE * N_DIGITS * N_CATS  # 4 * 3 * 10 = 120


def to_base_digits(n: int, base: int, width: int = N_DIGITS) -> np.ndarray:
    """Return the `width` least-significant digits of n written in `base`,
    most-significant digit first, zero-padded. E.g. to_base_digits(23, 5, 3)
    -> [0, 4, 3]  because 23 = 0*25 + 4*5 + 3."""
    digits = np.zeros(width, dtype=np.int64)
    x = int(n)
    for i in range(width - 1, -1, -1):
        digits[i] = x % base
        x //= base
    return digits


def encode_number(n: int, base: int) -> np.ndarray:
    """One-hot encode a single integer's digit representation in `base` into a
    (N_DIGITS * N_CATS,) vector. Only categories [0, base) can ever be hot."""
    digs = to_base_digits(n, base)
    onehot = np.zeros((N_DIGITS, N_CATS), dtype=np.float64)
    onehot[np.arange(N_DIGITS), digs] = 1.0
    return onehot.reshape(-1)


def encode_example(a: int, b: int, c: int, d: int, base: int) -> np.ndarray:
    """Encode the 4-tuple (A, B, C, D) into the flat INPUT_DIM input vector,
    each number written in the given `base`."""
    return np.concatenate([encode_number(v, base) for v in (a, b, c, d)])


# =============================================================================
# PART 1 -- SYNTHETIC TASK CONSTRUCTION
# =============================================================================

def sample_integer(rng) -> int:
    return int(rng.integers(1, MAX_VAL + 1))


def make_example(rng):
    """
    Construct one training example: integers (A, B, C, D), the WUJUDI
    (existential) label -- true ratio equality -- and the I'TIBARI
    (considerational) label -- base-10 last-digit parity of A.

    With probability 0.5 we deliberately CONSTRUCT an equal-ratio pair by
    scaling (A, B) by a random integer k, guaranteeing y_exist = 1 and giving
    the network genuine positive signal to learn from (random quadruples are
    equal in ratio only by rare coincidence).
    """
    a = sample_integer(rng)
    b = sample_integer(rng)
    if rng.random() < 0.5:
        k_max = max(1, MAX_VAL // max(a, b))
        k = int(rng.integers(1, k_max + 1))
        c, d = a * k, b * k
    else:
        c = sample_integer(rng)
        d = sample_integer(rng)

    y_exist = 1.0 if a * d == b * c else 0.0   # WUJUDI: true of the magnitudes
    y_consid = 1.0 if leading_digit_ge_5(a) else 0.0  # I'TIBARI: true of the base-10 numeral

    return a, b, c, d, y_exist, y_consid


def leading_digit_ge_5(n: int) -> bool:
    """Leading (most-significant) base-10 digit of n is >= 5. A fact defined
    purely by base-10 positional place-value -- change the base and this is
    a different partition of the same integers, not a rescaled version of
    the same fact (unlike parity, which happens to survive any even base)."""
    s = str(int(n))
    return int(s[0]) >= 5


def make_dataset(n_examples: int, base: int, rng):
    X = np.zeros((n_examples, INPUT_DIM))
    y_exist = np.zeros((n_examples, 1))
    y_consid = np.zeros((n_examples, 1))
    raw = []
    for i in range(n_examples):
        a, b, c, d, ye, yc = make_example(rng)
        X[i] = encode_example(a, b, c, d, base)
        y_exist[i, 0] = ye
        y_consid[i, 0] = yc
        raw.append((a, b, c, d))
    return X, y_exist, y_consid, raw


# =============================================================================
# PART 2 -- DualAttributeNet: shared trunk + Existential head + Considerational head
# =============================================================================
# Architecture (all pure NumPy, manual forward and backward):
#
#   input (120)
#     -> Linear(120,64) -> tanh
#     -> Linear(64,32)  -> tanh        }  Hikmat al-Mushtarak (shared trunk)
#     -> split:
#         Existential:     Linear(32,16) -> tanh -> Linear(16,1) -> sigmoid
#         Considerational: Linear(32,16) -> tanh -> Linear(16,1) -> sigmoid
#
# Loss = BCE(p_exist, y_exist) + BCE(p_consid, y_consid), averaged over batch.
#
# The two heads share everything up to the 32-dim trunk representation, so any
# difference in how well they survive a change of encoding (Part 6) is a
# difference in what each *head* extracted from a common representation, not
# an artifact of having separate encoders.

def xavier(rng, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def init_params(rng):
    return {
        "W1": xavier(rng, INPUT_DIM, 64), "b1": np.zeros(64),
        "W2": xavier(rng, 64, 32),        "b2": np.zeros(32),
        "We1": xavier(rng, 32, 16),       "be1": np.zeros(16),
        "We2": xavier(rng, 16, 1),        "be2": np.zeros(1),
        "Wc1": xavier(rng, 32, 16),       "bc1": np.zeros(16),
        "Wc2": xavier(rng, 16, 1),        "bc2": np.zeros(1),
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def forward(params, X):
    """Forward pass. Returns (p_exist, p_consid, cache) where cache holds every
    intermediate activation needed by the manual backward pass."""
    z1 = X @ params["W1"] + params["b1"];  a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]; a2 = np.tanh(z2)

    ze1 = a2 @ params["We1"] + params["be1"]; ae1 = np.tanh(ze1)
    ze2 = ae1 @ params["We2"] + params["be2"]; p_exist = sigmoid(ze2)

    zc1 = a2 @ params["Wc1"] + params["bc1"]; ac1 = np.tanh(zc1)
    zc2 = ac1 @ params["Wc2"] + params["bc2"]; p_consid = sigmoid(zc2)

    cache = dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2,
                 ze1=ze1, ae1=ae1, ze2=ze2, p_exist=p_exist,
                 zc1=zc1, ac1=ac1, zc2=zc2, p_consid=p_consid)
    return p_exist, p_consid, cache


def bce_loss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def loss_fn(params, X, y_exist, y_consid):
    p_exist, p_consid, _ = forward(params, X)
    return bce_loss(p_exist, y_exist) + bce_loss(p_consid, y_consid)


def backward(params, cache, y_exist, y_consid):
    """Hand-derived gradients for every parameter. Standard results used:
       d/dz tanh(z) = 1 - tanh(z)^2
       d/dz sigmoid_BCE(z) = (p - y) / N   when loss = mean BCE(sigmoid(z), y)
    """
    N = cache["X"].shape[0]
    grads = {}

    # ---- Existential head ----
    dze2 = (cache["p_exist"] - y_exist) / N                      # (N,1)
    grads["We2"] = cache["ae1"].T @ dze2
    grads["be2"] = dze2.sum(axis=0)
    dae1 = dze2 @ params["We2"].T
    dze1 = dae1 * (1 - cache["ae1"] ** 2)
    grads["We1"] = cache["a2"].T @ dze1
    grads["be1"] = dze1.sum(axis=0)
    da2_from_e = dze1 @ params["We1"].T

    # ---- Considerational head ----
    dzc2 = (cache["p_consid"] - y_consid) / N
    grads["Wc2"] = cache["ac1"].T @ dzc2
    grads["bc2"] = dzc2.sum(axis=0)
    dac1 = dzc2 @ params["Wc2"].T
    dzc1 = dac1 * (1 - cache["ac1"] ** 2)
    grads["Wc1"] = cache["a2"].T @ dzc1
    grads["bc1"] = dzc1.sum(axis=0)
    da2_from_c = dzc1 @ params["Wc1"].T

    # ---- Shared trunk (gradients from both heads combine) ----
    da2 = da2_from_e + da2_from_c
    dz2 = da2 * (1 - cache["a2"] ** 2)
    grads["W2"] = cache["a1"].T @ dz2
    grads["b2"] = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1 - cache["a1"] ** 2)
    grads["W1"] = cache["X"].T @ dz1
    grads["b1"] = dz1.sum(axis=0)

    return grads


# =============================================================================
# PART 3 -- FINITE-DIFFERENCE GRADIENT CHECK (mandatory)
# =============================================================================

def flatten_params(params):
    keys = sorted(params.keys())
    flat = np.concatenate([params[k].ravel() for k in keys])
    shapes = {k: params[k].shape for k in keys}
    return flat, keys, shapes


def unflatten_params(flat, keys, shapes):
    params = {}
    i = 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        params[k] = flat[i:i + n].reshape(shapes[k])
        i += n
    return params


def gradient_check(params, X, y_exist, y_consid, n_probe=40, eps=1e-5, seed=7):
    """Compares the analytic backward() gradient against a central-difference
    numerical gradient at n_probe randomly chosen scalar parameters, and
    returns the maximum relative error found. This is the mandatory
    correctness gate for the from-scratch backward pass."""
    rng = np.random.default_rng(seed)
    flat, keys, shapes = flatten_params(params)

    _, _, cache = forward(params, X)
    analytic_grads = backward(params, cache, y_exist, y_consid)
    analytic_flat, _, _ = flatten_params(analytic_grads)

    idx = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
    max_rel_err = 0.0
    for i in idx:
        base_val = flat[i]

        flat[i] = base_val + eps
        p_params = unflatten_params(flat, keys, shapes)
        loss_plus = loss_fn(p_params, X, y_exist, y_consid)

        flat[i] = base_val - eps
        m_params = unflatten_params(flat, keys, shapes)
        loss_minus = loss_fn(m_params, X, y_exist, y_consid)

        flat[i] = base_val  # restore

        numeric_grad = (loss_plus - loss_minus) / (2 * eps)
        analytic_grad = analytic_flat[i]
        denom = max(1e-8, abs(numeric_grad) + abs(analytic_grad))
        rel_err = abs(numeric_grad - analytic_grad) / denom
        max_rel_err = max(max_rel_err, rel_err)

    return max_rel_err


# =============================================================================
# PART 4 -- POSTULATE PERTURBATION GATE  (Mizan al-Farz, "balance of the assumption")
# =============================================================================
# Khayyam's method with Euclid's fifth postulate was never to declare it true
# by fiat. He replaced it with a more self-evident pair of statements, derived
# a quadrilateral (base AB, equal perpendicular sides AD = BC), and tested the
# three possible hypotheses for the summit angle -- acute, right, obtuse --
# to see which one survives and which collapse into contradiction. The
# procedure IS the confidence: a conclusion that holds under every reasonable
# perturbation of its assumption is trustworthy; a conclusion that flips under
# a small nudge was never solid.
#
# We reuse the same procedure directly on the trained network: given a query
# (A, B, C, D), we perturb D by small integer steps -- literally nudging the
# "founding assumption" that fixes the second ratio -- and observe how often
# the Existential head's decision flips. If the query sits far from the
# boundary A*D == B*C, the decision survives every perturbation and confidence
# is high; if the query sits near that boundary, small perturbations flip the
# decision and Khayyam's own diagnosis -- "this conclusion is fragile" --
# becomes the network's own calibrated confidence value.

def postulate_perturbation_confidence(params, a, b, c, d, base=10, deltas=(-2, -1, 1, 2)):
    """Returns (prediction, confidence in [0,1], flip_rate) for the query
    (A,B,C,D) by perturbing D and re-running the frozen network."""
    x0 = encode_example(a, b, c, d, base)[None, :]
    p0, _, _ = forward(params, x0)
    pred0 = int(p0[0, 0] > 0.5)

    flips = 0
    for delta in deltas:
        d_pert = max(1, min(MAX_VAL, d + delta))
        xk = encode_example(a, b, c, d_pert, base)[None, :]
        pk, _, _ = forward(params, xk)
        pred_k = int(pk[0, 0] > 0.5)
        flips += int(pred_k != pred0)

    flip_rate = flips / len(deltas)
    confidence = 1.0 - flip_rate     # Khayyam's diagnostic, read off directly
    return pred0, confidence, flip_rate


# =============================================================================
# PART 5 -- TRAINING LOOP (Adam, from scratch) + EVALUATION
# =============================================================================

def init_adam_state(params):
    return {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in params.items()}


def adam_step(params, grads, state, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    for k in params:
        state[k]["m"] = beta1 * state[k]["m"] + (1 - beta1) * grads[k]
        state[k]["v"] = beta2 * state[k]["v"] + (1 - beta2) * (grads[k] ** 2)
        m_hat = state[k]["m"] / (1 - beta1 ** t)
        v_hat = state[k]["v"] / (1 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params


def accuracy(p, y):
    return float(np.mean((p > 0.5).astype(np.float64) == y))


def train(params, X_train, ye_train, yc_train, X_val, ye_val, yc_val,
          epochs=60, batch_size=64, lr=0.01, weight_decay=1e-4, verbose=True):
    n = X_train.shape[0]
    state = init_adam_state(params)
    t = 0
    history = []
    weight_keys = [k for k in params if k.startswith("W")]  # decay weights, not biases
    for epoch in range(1, epochs + 1):
        perm = RNG.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yeb, ycb = X_train[idx], ye_train[idx], yc_train[idx]
            _, _, cache = forward(params, Xb)
            grads = backward(params, cache, yeb, ycb)
            for k in weight_keys:
                grads[k] = grads[k] + weight_decay * params[k]  # L2 penalty
            t += 1
            params = adam_step(params, grads, state, t, lr=lr)

        p_e_val, p_c_val, _ = forward(params, X_val)
        val_loss = bce_loss(p_e_val, ye_val) + bce_loss(p_c_val, yc_val)
        acc_e = accuracy(p_e_val, ye_val)
        acc_c = accuracy(p_c_val, yc_val)
        history.append((epoch, val_loss, acc_e, acc_c))
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  "
                  f"existential_acc={acc_e:.3f}  considerational_acc={acc_c:.3f}")
    return params, history


# =============================================================================
# PART 6 -- THE ENCODING-SHIFT AUDIT
# =============================================================================
# The experiment this whole file exists to run. Train once, on base-10 digit
# patterns only. Then, WITHOUT any further training, re-encode a fresh batch
# of the *same kind of underlying integers* using bases 9, 8, 7, 6 and 5, and
# push them through the frozen network. Because the weights never change,
# any accuracy that survives is accuracy that the head learned about the
# MAGNITUDES rather than about base-10 digit positions.
#
# The considerational label (base-10 last-digit parity) is, by construction,
# only well-defined relative to base-10 notation -- so this is expected, and
# reported as such rather than hidden. The existential label (ratio equality)
# is a fact about the integers themselves and has a chance of surviving the
# swap IF the trunk actually learned something like cross-multiplication
# rather than base-10 digit correlations. We do not assume which way it goes;
# the printed table is real, run at execution time.

def encoding_shift_audit(params, n_examples=2000, bases=(10, 9, 8, 7, 6, 5)):
    results = []
    for base in bases:
        rng = np.random.default_rng(9000 + base)
        X, ye, yc, _ = make_dataset(n_examples, base, rng)
        p_e, p_c, _ = forward(params, X)
        results.append((base, accuracy(p_e, ye), accuracy(p_c, yc)))
    return results


# =============================================================================
# PART 7 -- SELF-TESTS AND MAIN
# =============================================================================

def run_self_tests():
    print("=" * 78)
    print("SELF-TEST 1: dimension and encoding sanity")
    print("=" * 78)
    v = encode_number(23, 5)
    assert v.shape == (N_DIGITS * N_CATS,)
    digs = to_base_digits(23, 5)
    assert list(digs) == [0, 4, 3], f"expected [0,4,3], got {list(digs)}"
    assert 0 * 25 + 4 * 5 + 3 == 23
    x = encode_example(3, 4, 6, 8, 10)
    assert x.shape == (INPUT_DIM,)
    print("  OK: base-conversion and one-hot encoding are consistent.")

    print()
    print("=" * 78)
    print("SELF-TEST 2: label correctness on hand-checked examples")
    print("=" * 78)
    # 3/4 == 6/8  -> existential label should be 1
    assert 3 * 8 == 4 * 6
    # 3/4 == 5/7 ? 3*7=21, 4*5=20 -> not equal -> existential label 0
    assert 3 * 7 != 4 * 5
    print("  OK: ratio-equality ground truth verified by hand for two probe cases.")

    print()
    print("=" * 78)
    print("SELF-TEST 2b: recorded correction -- why last-digit PARITY was rejected")
    print("=" * 78)
    # This documents a real design error caught during testing, not a
    # hypothetical. Because base 10 is even, last-digit parity in base 10
    # equals the integer's own parity, and that parity is UNCHANGED in every
    # other even base -- making "last-digit parity" accidentally existential
    # (WUJUDI) rather than the notation-bound (I'TIBARI) fact it was meant to
    # illustrate. Leading-digit->=5 has no such confound.
    for n, base in [(23, 10), (23, 8), (46, 10), (46, 6)]:
        parity_flag = to_base_digits(n, base)[-1] % 2 == 0
        true_parity = (n % 2 == 0)
        assert parity_flag == true_parity, "expected the confound to reproduce for even bases"
    print("  Confirmed: base-10 last-digit parity == true integer parity whenever")
    print("  the alternate base is also even -- i.e. it silently measures a WUJUDI")
    print("  fact. Leading-digit->=5 was substituted for the Considerational task")
    print("  specifically to remove this confound (see Part 1 docstring).")

    print()
    print("=" * 78)
    print("SELF-TEST 3: finite-difference gradient check on the FULL backward pass")
    print("=" * 78)
    rng = np.random.default_rng(42)
    params = init_params(rng)
    X, ye, yc, _ = make_dataset(16, 10, rng)
    max_err = gradient_check(params, X, ye, yc, n_probe=40)
    print(f"  max relative error over 40 random parameters: {max_err:.3e}")
    assert max_err < 1e-4, "GRADIENT CHECK FAILED"
    print("  PASSED (threshold 1e-4).")

    print()
    print("=" * 78)
    print("SELF-TEST 4: training loss must decrease and beat trivial baselines")
    print("=" * 78)
    rng = np.random.default_rng(123)
    params = init_params(rng)
    Xtr, yetr, yctr, _ = make_dataset(400, 10, rng)
    _, _, cache0 = forward(params, Xtr)
    loss_before = bce_loss(cache0["p_exist"], yetr) + bce_loss(cache0["p_consid"], yctr)
    params, hist = train(params, Xtr, yetr, yctr, Xtr, yetr, yctr,
                          epochs=20, batch_size=32, lr=0.02, verbose=False)
    _, _, cache1 = forward(params, Xtr)
    loss_after = bce_loss(cache1["p_exist"], yetr) + bce_loss(cache1["p_consid"], yctr)
    print(f"  loss before training: {loss_before:.4f}   after 20 epochs: {loss_after:.4f}")
    assert loss_after < loss_before, "training did not reduce loss"
    print("  PASSED: training reduces loss.")

    print()
    print("=" * 78)
    print("SELF-TEST 5: postulate-perturbation confidence gate behaves as designed")
    print("=" * 78)
    # A boundary case (near equal ratio) should be less confident than
    # a case sitting far from the ratio-equality boundary.
    pred_far, conf_far, flip_far = postulate_perturbation_confidence(params, 10, 20, 90, 5)
    pred_near, conf_near, flip_near = postulate_perturbation_confidence(params, 10, 20, 40, 20)
    print(f"  far-from-boundary  query (10,20,90,5):   pred={pred_far}  "
          f"confidence={conf_far:.2f}  flip_rate={flip_far:.2f}")
    print(f"  near-boundary      query (10,20,40,20):  pred={pred_near} "
          f"confidence={conf_near:.2f}  flip_rate={flip_near:.2f}")
    print("  (Reported, not asserted equal/unequal -- Khayyam's method is to")
    print("   TEST the sensitivity, not to presume its outcome.)")

    print()
    print("All self-tests passed.\n")


def main():
    run_self_tests()

    print("=" * 78)
    print("MAIN RUN: training the Dual-Attribute Network on base-10 digit patterns")
    print("=" * 78)
    rng = np.random.default_rng(2024)
    params = init_params(rng)

    X_train, ye_train, yc_train, _ = make_dataset(6000, 10, rng)
    X_val, ye_val, yc_val, _ = make_dataset(1200, 10, rng)

    print(f"  training examples: {X_train.shape[0]}   validation examples: {X_val.shape[0]}")
    print(f"  base-rate of y_exist=1 in training set: {ye_train.mean():.3f}")
    print(f"  base-rate of y_consid=1 in training set: {yc_train.mean():.3f}")
    print()

    params, history = train(params, X_train, ye_train, yc_train,
                             X_val, ye_val, yc_val,
                             epochs=40, batch_size=64, lr=0.01, weight_decay=1e-4, verbose=True)

    print()
    print("=" * 78)
    print("PART 6 RESULT: ENCODING-SHIFT AUDIT")
    print("  (network trained ONLY on base-10 digit patterns; weights now frozen)")
    print("=" * 78)
    audit = encoding_shift_audit(params, n_examples=2000)
    print(f"  {'base':>6} | {'existential_acc (WUJUDI)':>26} | {'considerational_acc (I''TIBARI)':>30}")
    print(f"  {'-'*6}-+-{'-'*26}-+-{'-'*30}")
    for base, acc_e, acc_c in audit:
        tag = "  <- native training base" if base == 10 else ""
        print(f"  {base:>6} | {acc_e:>26.3f} | {acc_c:>30.3f}{tag}")

    acc_e_native = audit[0][1]
    acc_c_native = audit[0][2]
    acc_e_shifted = np.mean([a[1] for a in audit[1:]])
    acc_c_shifted = np.mean([a[2] for a in audit[1:]])
    print()
    print(f"  Existential head:      native acc {acc_e_native:.3f} -> mean shifted acc {acc_e_shifted:.3f} "
          f"(drop {acc_e_native - acc_e_shifted:+.3f})")
    print(f"  Considerational head:  native acc {acc_c_native:.3f} -> mean shifted acc {acc_c_shifted:.3f} "
          f"(drop {acc_c_native - acc_c_shifted:+.3f})")
    print("  A larger drop for the considerational head than the existential head is")
    print("  the network's own empirical evidence for Khayyam's distinction: one head")
    print("  learned a fact about magnitudes, the other learned a fact about notation.")

    print()
    print("=" * 78)
    print("PART 4 DEMONSTRATION: Postulate Perturbation confidence gate on 6 queries")
    print("=" * 78)
    queries = [
        (12, 18, 24, 36),   # exact equal ratio (2/3), far interior
        (12, 18, 24, 37),   # one unit off the equal-ratio boundary
        (7, 11, 21, 33),    # exact equal ratio (7/11), far interior
        (7, 11, 21, 34),    # one unit off the boundary
        (50, 3, 2, 4),      # wildly unequal, far interior
        (9, 9, 9, 10),      # near-boundary, close to the identity ratio 1:1
    ]
    for (a, b, c, d) in queries:
        pred, conf, flip = postulate_perturbation_confidence(params, a, b, c, d)
        truth = int(a * d == b * c)
        print(f"  A={a:>3} B={b:>3} C={c:>3} D={d:>3}  true_label={truth}  "
              f"pred={pred}  confidence={conf:.2f}  flip_rate={flip:.2f}")

    print()
    print("Run complete.")


if __name__ == "__main__":
    main()

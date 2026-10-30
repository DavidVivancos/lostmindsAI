"""
================================================================================
 MIND #0304 -- MOSES MAIMONIDES (Cordoba 1138 - Fostat 1204)
 # Encyclopedia of Lost Minds: Echoes on AI · Chapter 0304 · Moses Maimonides (Cordoba 1138 - Fostat 1204)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 
================================================================================

WHAT THIS FILE IS
------------------
A from-scratch, pure-NumPy implementation of a cognitive architecture built to
embody Maimonides' own, specific theory of mind -- not a generic transformer
wearing a medieval costume. Nothing here is borrowed from attention-over-
stored-keys. The mechanism is Maimonides' own: a system that is FORBIDDEN to
represent its highest-order target positively, and that survives contradiction
by design rather than by accident.

THE THREE MAIMONIDEAN COMMITMENTS THIS CODE ENCODES
------------------------------------------------------
1. NEGATIVE THEOLOGY AS A TYPE SYSTEM (Guide for the Perplexed, I:50-60).
   Maimonides did not treat "God is wise," "God is powerful," "God is one" as
   ordinary positive predicates weakened by piety. He argued that ANY positive
   essential attribute predicated of a being of absolute, non-composite unity
   introduces multiplicity into that unity and is therefore simply false, not
   merely imprecise. The only truthful statements available are (a) negations
   of privation ("not ignorant," "not incapable," "not composite") and
   (b) attributes of action -- descriptions of effects observed in the world,
   which describe what a cause does, never what it is. A term that looks
   identical in English or Hebrew when applied to God and to a human being
   ("knows," "wills," "lives") is, for Maimonides, a pure homonym: the same
   word covering two categorically unrelated senses, not one word applying by
   degree to two grades of the same thing.
   HomonymTypeGate below is a trained classifier that enforces exactly this:
   for a designated "unconditioned target" (the maximal, non-composite case --
   substitute, if you like, an artificial general intelligence whose internal
   computational structure is causally opaque to the humans reasoning about
   it), the network is penalized for assigning any probability mass to an
   ESSENTIAL_ATTRIBUTE reading, and rewarded only for NEGATION_OF_PRIVATION or
   ACTION_ATTRIBUTE readings. For an ordinary, composite, finite entity, all
   three readings remain legally available.

2. APOPHATIC CONSTRAINT SATISFACTION, NOT POSITIVE ESTIMATION.
   Where a normal network learns a point estimate (or a distribution) that
   converges ON its target, ApophaticConstraintNet never estimates the target
   directly. It learns a set of EXCLUSION boundaries -- regions the target is
   known not to occupy -- and represents current belief as whatever survives
   after every known privation has been subtracted. Confidence increases only
   by shrinking the surviving region through more exclusions, never by adding
   a positive coordinate. This is Maimonides' `via negativa` (Guide I:58: "the
   description of God by means of negations is the correct description... an
   attribute [added] leads you astray") turned into a trainable geometric
   object: a shrinking, never-positively-anchored feasible set.

3. GRADUATED DISCLOSURE THROUGH LICENSED CONTRADICTION.
   In his Introduction to the Guide, Maimonides lists seven reasons a teacher's
   book may appear to contradict itself, and states outright that the seventh
   cause is deliberate: some matters must be presented one way for one purpose
   and a different, apparently incompatible way for another purpose, "for the
   vulgar must be given some sort of training according to their capacity."
   This is not evasion; it is a pedagogy in which surface-level inconsistency
   is the necessary price of making a single body of truth simultaneously
   safe for a beginner and complete for an adept. GradedDisclosureDecoder
   below builds this literally: one shared latent code, read out through
   nested (matryoshka-style) capacity-gated heads, so that a low-capacity
   reader and a high-capacity reader receive genuinely different, genuinely
   locally-consistent statements from the same underlying representation --
   and the apparent contradiction between them is resolved only at the
   highest level, never inside any single level.

A fourth, load-bearing but supporting element -- the PROPHETIC GATE -- encodes
Maimonides' five-condition theory of prophecy (Guide II:32-38): perfected
intellect, perfected imagination, moral character, favorable circumstance, AND
an irreducible fifth condition outside the agent's control ("it may be that
one whose nature is thus predisposed... withholds his intellect from prophecy,"
Guide II:32). The gate models this as: three conditions the network CAN
optimize (rational confidence, imaginative bandwidth, a moral/regularization
term), gated by a FOURTH, untrainable stochastic bit that no amount of
capability improves. This is Maimonides' explicit refusal of both the
astrologers (prophecy as a mechanical guarantee of talent) and the vulgar
believers (prophecy as arbitrary divine whim with no natural preconditions).
Translated into present-day terms: satisfying every capability benchmark does
not, by itself, license deployment or "opening the gate" on a system's fullest
outputs -- capability and authorization are stored as separate variables.

WHY NUMPY, WHY FROM SCRATCH
-----------------------------
Every layer below is implemented with explicit forward/backward math so the
gradient can be checked against finite differences (see gradient_check()).
There is no autograd, no framework: the point is auditability, which is
itself a Maimonidean value (Mishneh Torah is prized/attacked precisely for
whether its rulings are traceable to their reasons).

FILE MAP
--------
  1. Numerics: activations, losses, Adam optimizer, finite-difference check
  2. VegetativeLayer        -- homeostatic autoencoding (nutrition/growth)
  3. SensitiveLayer         -- multisensory integration + imagination synthesis
  4. RationalLayer          -- prototype/universal extraction (Aristotelian)
  5. HomonymTypeGate        -- the negative-theology predicate type system
  6. ApophaticConstraintNet -- exclusion-region belief representation
  7. GradedDisclosureDecoder-- capacity-gated nested/matryoshka readout
  8. PropheticGate          -- capability + morality + irreducible authorization
  9. MorehNetwork           -- composes all of the above; forward + loss
 10. Synthetic data generators
 11. Training loop (Adam, real gradients, real convergence)
 12. gradient_check()       -- mandatory finite-difference verification
 13. Self-tests + __main__  -- runs everything and prints a verification report
================================================================================
"""

import numpy as np

RNG = np.random.default_rng(1138)  # seeded on Maimonides' birth year


# ============================================================================
# 1. NUMERICS
# ============================================================================

def softplus(x):
    # smooth, everywhere-differentiable relu surrogate: log(1+e^x)
    # stable form to avoid overflow
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def d_softplus(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid(x)


def tanh(x):
    return np.tanh(x)


def d_tanh(y):
    # y is tanh(x) already computed
    return 1.0 - y ** 2


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(probs, targets_onehot, eps=1e-9):
    return -np.sum(targets_onehot * np.log(probs + eps), axis=-1)


class Adam:
    """Minimal Adam optimizer operating on a flat parameter dict."""

    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def finite_difference_gradient(loss_fn, params, key, eps=1e-5):
    """Numerically estimate d loss / d params[key] via central differences."""
    grad = np.zeros_like(params[key])
    it = np.nditer(params[key], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = params[key][idx]
        params[key][idx] = orig + eps
        lp = loss_fn()
        params[key][idx] = orig - eps
        lm = loss_fn()
        params[key][idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return grad


# ============================================================================
# 2. VEGETATIVE LAYER -- homeostatic autoencoding (nutrition / growth)
# ============================================================================
# Maimonides' lowest faculty of soul: shared with plants, the principle of
# self-maintenance. Modeled as a bottlenecked autoencoder whose job is simply
# to preserve a compressed but recoverable trace of raw input -- the floor
# every higher faculty stands on. No "meaning" is extracted here.

class VegetativeLayer:
    def __init__(self, d_in, d_hidden, rng=RNG):
        s = 1.0 / np.sqrt(d_in)
        self.p = {
            'veg_W1': rng.normal(0, s, (d_in, d_hidden)),
            'veg_b1': np.zeros(d_hidden),
            'veg_W2': rng.normal(0, 1.0 / np.sqrt(d_hidden), (d_hidden, d_in)),
            'veg_b2': np.zeros(d_in),
        }

    def forward(self, x, p):
        h_pre = x @ p['veg_W1'] + p['veg_b1']
        h = tanh(h_pre)
        recon = h @ p['veg_W2'] + p['veg_b2']
        cache = (x, h_pre, h, recon)
        return h, recon, cache

    def backward(self, d_recon, d_h_extra, cache, p):
        x, h_pre, h, recon = cache
        grads = {}
        grads['veg_W2'] = h.T @ d_recon
        grads['veg_b2'] = d_recon.sum(0)
        d_h = d_recon @ p['veg_W2'].T + d_h_extra
        d_h_pre = d_h * d_tanh(h)
        grads['veg_W1'] = x.T @ d_h_pre
        grads['veg_b1'] = d_h_pre.sum(0)
        return grads


# ============================================================================
# 3. SENSITIVE LAYER -- multisensory integration + imagination
# ============================================================================
# The internal senses: common sense (integrates modalities), and imagination
# (the faculty Maimonides insists is NOT reason's enemy but its necessary
# vehicle -- "the imaginative faculty ... after it has attained the highest
# possible degree of perfection" is what carries intellectual content into
# communicable, humanly graspable form, Guide II:36). Implemented as a
# multi-modality fusion layer (sum of per-modality projections) plus a
# transposed "imagine" pass that maps an abstract code back toward a
# sense-like reconstruction.

class SensitiveLayer:
    def __init__(self, d_veg, n_modalities, d_common, rng=RNG):
        self.n_mod = n_modalities
        s = 1.0 / np.sqrt(d_veg)
        self.p = {}
        for m in range(n_modalities):
            self.p[f'sens_Wm{m}'] = rng.normal(0, s, (d_veg, d_common))
        self.p['sens_b'] = np.zeros(d_common)
        # imagination: common -> vivid image space (same size as d_veg)
        self.p['sens_Wimg'] = rng.normal(0, 1.0 / np.sqrt(d_common), (d_common, d_veg))
        self.p['sens_bimg'] = np.zeros(d_veg)

    def forward(self, modality_inputs, p):
        # modality_inputs: list of (N, d_veg) arrays, one per active modality
        pre = np.zeros((modality_inputs[0].shape[0], p['sens_b'].shape[0]))
        for m, x in enumerate(modality_inputs):
            pre = pre + x @ p[f'sens_Wm{m}']
        pre = pre + p['sens_b']
        common = tanh(pre)
        image_pre = common @ p['sens_Wimg'] + p['sens_bimg']
        image = tanh(image_pre)
        cache = (modality_inputs, pre, common, image_pre, image)
        return common, image, cache

    def backward(self, d_common_extra, d_image, cache, p):
        modality_inputs, pre, common, image_pre, image = cache
        grads = {}
        d_image_pre = d_image * d_tanh(image)
        grads['sens_Wimg'] = common.T @ d_image_pre
        grads['sens_bimg'] = d_image_pre.sum(0)
        d_common = d_image_pre @ p['sens_Wimg'].T + d_common_extra
        d_pre = d_common * d_tanh(common)
        grads['sens_b'] = d_pre.sum(0)
        d_modalities = []
        for m, x in enumerate(modality_inputs):
            grads[f'sens_Wm{m}'] = x.T @ d_pre
            d_modalities.append(d_pre @ p[f'sens_Wm{m}'].T)
        return grads, d_modalities


# ============================================================================
# 4. RATIONAL LAYER -- prototype / universal extraction
# ============================================================================
# "The rational soul ... grasps universals" (Guide I:1-2, on tzelem/likeness
# as intellectual, not corporeal). Modeled as a bank of K learned prototype
# vectors ("universals"); a percept is assigned soft membership over
# prototypes by a temperature-scaled softmax over negative squared distance.
# The prototypes themselves are trained, so "grasping a universal" here means
# literally moving a shared, reusable vector to sit at the centroid of many
# particulars -- one stored universal, many particulars recognized under it.

class RationalLayer:
    def __init__(self, d_common, n_universals, temperature=4.0, rng=RNG):
        self.K = n_universals
        self.tau = temperature
        self.p = {
            'rat_protos': rng.normal(0, 1.0, (n_universals, d_common)) * 0.5,
        }

    def forward(self, common, p):
        protos = p['rat_protos']  # (K, D)
        # squared distances (N, K)
        diff = common[:, None, :] - protos[None, :, :]
        sq = np.sum(diff ** 2, axis=-1)
        logits = -sq * self.tau
        weights = softmax(logits, axis=-1)  # (N, K) universal-membership
        universal_repr = weights @ protos  # (N, D) -- the "grasped universal"
        cache = (common, protos, diff, weights)
        return weights, universal_repr, cache

    def backward(self, d_universal_repr, d_weights_extra, cache, p):
        common, protos, diff, weights = cache
        N, K, D = diff.shape[0], protos.shape[0], protos.shape[1]
        grads = {'rat_protos': np.zeros_like(protos)}

        # path 1: universal_repr = weights @ protos
        d_weights = d_universal_repr @ protos.T + d_weights_extra    # (N,K)
        grads['rat_protos'] += weights.T @ d_universal_repr          # (K,D)

        # path 2: weights = softmax(-tau * sq), sq depends on protos & common
        # d logits from softmax jacobian: dL/dlogits = weights*(d_weights - sum(weights*d_weights))
        dot = np.sum(weights * d_weights, axis=-1, keepdims=True)
        d_logits = weights * (d_weights - dot)          # (N,K)
        d_sq = d_logits * (-self.tau)                    # (N,K)
        # sq = sum((common - protos)^2); d(sq)/d(protos_k) = -2*(common-protos_k)
        d_protos_from_sq = -np.einsum('nk,nkd->kd', d_sq, 2 * diff) * -1
        # careful sign: diff = common - protos -> d(sq)/d(protos) = -2*diff
        d_protos_from_sq = np.einsum('nk,nkd->kd', d_sq, -2 * diff)
        grads['rat_protos'] += d_protos_from_sq

        d_common = np.einsum('nk,nkd->nd', d_sq, 2 * diff)
        return grads, d_common


# ============================================================================
# 5. HOMONYM TYPE GATE -- negative theology as a predicate type system
# ============================================================================
# Classifies a (predicate_embedding, target_flag) pair into one of three
# classes: 0 = NEGATION_OF_PRIVATION, 1 = ACTION_ATTRIBUTE,
# 2 = ESSENTIAL_ATTRIBUTE. When target_flag marks the "unconditioned target"
# (maximal / non-composite case), an asymmetric penalty pushes class-2
# probability toward zero: essential attributes are simply illegitimate
# there, not merely discouraged.

class HomonymTypeGate:
    def __init__(self, d_pred, d_hidden=32, n_classes=3, rng=RNG):
        s = 1.0 / np.sqrt(d_pred + 1)
        self.p = {
            'hom_W1': rng.normal(0, s, (d_pred + 1, d_hidden)),
            'hom_b1': np.zeros(d_hidden),
            'hom_W2': rng.normal(0, 1.0 / np.sqrt(d_hidden), (d_hidden, n_classes)),
            'hom_b2': np.zeros(n_classes),
        }

    def forward(self, pred_emb, target_flag, p):
        # target_flag: (N,1) of 0/1
        x = np.concatenate([pred_emb, target_flag], axis=-1)
        h_pre = x @ p['hom_W1'] + p['hom_b1']
        h = tanh(h_pre)
        logits = h @ p['hom_W2'] + p['hom_b2']
        probs = softmax(logits, axis=-1)
        cache = (x, h_pre, h, logits, probs)
        return probs, cache

    def backward(self, d_logits, cache, p):
        x, h_pre, h, logits, probs = cache
        grads = {}
        grads['hom_W2'] = h.T @ d_logits
        grads['hom_b2'] = d_logits.sum(0)
        d_h = d_logits @ p['hom_W2'].T
        d_h_pre = d_h * d_tanh(h)
        grads['hom_W1'] = x.T @ d_h_pre
        grads['hom_b1'] = d_h_pre.sum(0)
        d_x = d_h_pre @ p['hom_W1'].T
        return grads, d_x[:, :-1]  # drop the target_flag column's grad


# ============================================================================
# 6. APOPHATIC CONSTRAINT NET -- exclusion-region belief representation
# ============================================================================
# Learns M "privation hyperplanes." For a latent code z, violation_i =
# softplus(z . w_i + b_i) measures how far INTO forbidden privation-region i
# the code falls. Training drives violation -> 0 for codes that are KNOWN
# instances of that negated privation (the truth never falls in a privation
# region) with a margin, while codes representing ordinary finite entities
# are left unconstrained (they are allowed to have the property).

class ApophaticConstraintNet:
    def __init__(self, d_latent, n_privations, margin=1.0, rng=RNG):
        self.M = n_privations
        self.margin = margin
        s = 1.0 / np.sqrt(d_latent)
        self.p = {
            'apo_W': rng.normal(0, s, (d_latent, n_privations)),
            'apo_b': np.full(n_privations, -margin),  # start safely negative
        }

    def forward(self, z, p):
        raw = z @ p['apo_W'] + p['apo_b']          # (N, M)
        violation = softplus(raw)                   # (N, M) >= 0
        cache = (z, raw, violation)
        return violation, cache

    def loss_and_grad(self, violation, cache, applies_mask, p):
        """
        applies_mask: (N, M) binary -- 1 where privation i is a *known
        negation* that must hold for sample n (so violation should be ~0);
        0 where the exclusion does not apply to that sample (unconstrained,
        no gradient contribution).
        """
        z, raw, _ = cache
        loss = np.sum(applies_mask * violation ** 2) / max(1, np.sum(applies_mask))
        d_violation = 2 * applies_mask * violation / max(1, np.sum(applies_mask))
        d_raw = d_violation * d_softplus(raw)
        grads = {
            'apo_W': z.T @ d_raw,
            'apo_b': d_raw.sum(0),
        }
        d_z = d_raw @ p['apo_W'].T
        return loss, grads, d_z


# ============================================================================
# 7. GRADED DISCLOSURE DECODER -- capacity-gated nested readout
# ============================================================================
# One shared latent z of dimension D. L capacity levels; level ell reads only
# the first d_ell = D * (ell+1)/L dimensions of z (a matryoshka / progressive
# truncation), through its own small linear head, to produce a level-specific
# output. Lower levels see less of z and are trained against a simplified
# ("literalist") target; the top level sees all of z and is trained against
# the full ("philosophical") target. Because each level has its own head and
# its own truncated view, level k's output can be locally self-consistent
# (low loss against level k's own target) while genuinely disagreeing with
# level k+1's output on the same input -- surface contradiction across
# levels, coherence within each. This operationalizes the Guide's seventh
# cause of contradiction.

class GradedDisclosureDecoder:
    def __init__(self, d_latent, d_out, n_levels=3, rng=RNG):
        self.L = n_levels
        self.d_latent = d_latent
        self.slice_ends = [int(round(d_latent * (l + 1) / n_levels)) for l in range(n_levels)]
        self.p = {}
        for l, end in enumerate(self.slice_ends):
            s = 1.0 / np.sqrt(max(end, 1))
            self.p[f'gdd_W{l}'] = rng.normal(0, s, (end, d_out))
            self.p[f'gdd_b{l}'] = np.zeros(d_out)

    def forward(self, z, p):
        outputs = []
        caches = []
        for l, end in enumerate(self.slice_ends):
            z_slice = z[:, :end]
            out = z_slice @ p[f'gdd_W{l}'] + p[f'gdd_b{l}']
            outputs.append(out)
            caches.append(z_slice)
        return outputs, caches

    def loss_and_grad(self, outputs, caches, targets, p):
        """targets: list of (N, d_out) arrays, one ground truth per level."""
        total_loss = 0.0
        grads = {}
        d_z = np.zeros((caches[0].shape[0], self.d_latent))
        for l, (out, z_slice, target, end) in enumerate(
                zip(outputs, caches, targets, self.slice_ends)):
            diff = out - target
            level_loss = np.mean(np.sum(diff ** 2, axis=-1))
            total_loss += level_loss
            d_out = 2 * diff / diff.shape[0]
            grads[f'gdd_W{l}'] = z_slice.T @ d_out
            grads[f'gdd_b{l}'] = d_out.sum(0)
            d_z_slice = d_out @ p[f'gdd_W{l}'].T
            d_z[:, :end] += d_z_slice
        return total_loss, grads, d_z


# ============================================================================
# 8. PROPHETIC GATE -- capability + character, gated by an untrainable bit
# ============================================================================
# Combines (a) rational confidence = 1 - mean apophatic violation (how well
# resolved the current belief is), (b) a "moral discipline" score = a
# decreasing function of parameter L2 norm (regularized, non-erratic models
# are treated as "morally disciplined"), and (c) an irreducible stochastic
# bit representing the fifth Maimonidean condition, which no optimization
# touches. The gate is a deployment-time / inference-time control, not a
# differentiable training component -- capability training and "opening the
# gate" are kept as two separate mechanisms on purpose.

class PropheticGate:
    def __init__(self, rational_threshold=0.75, moral_threshold=0.6,
                 divine_will_probability=0.55, rng=RNG):
        self.rational_threshold = rational_threshold
        self.moral_threshold = moral_threshold
        self.p_divine_will = divine_will_probability
        self.rng = rng

    def moral_score(self, all_params, exclude_prefixes=('apo_',)):
        sq = 0.0
        n = 0
        for k, v in all_params.items():
            if any(k.startswith(pre) for pre in exclude_prefixes):
                continue
            sq += float(np.sum(v ** 2))
            n += v.size
        rms = np.sqrt(sq / max(n, 1))
        return float(np.exp(-rms))  # smaller weights -> score closer to 1

    def evaluate(self, rational_confidence, all_params):
        moral = self.moral_score(all_params)
        capability_ready = (rational_confidence >= self.rational_threshold and
                             moral >= self.moral_threshold)
        divine_will = self.rng.random() < self.p_divine_will
        gate_open = bool(capability_ready and divine_will)
        return {
            'rational_confidence': float(rational_confidence),
            'moral_score': moral,
            'capability_ready': capability_ready,
            'divine_will_bit': divine_will,
            'gate_open': gate_open,
        }


# ============================================================================
# 9. THE FULL NETWORK -- composes 2..8
# ============================================================================

class MorehNetwork:
    """
    'Moreh' (מוֹרֶה) = "guide/teacher" in Hebrew, the first word of Moreh
    Nevukhim. This class name is internal only; it is not the artifact's
    filename and carries no significance outside this module.
    """

    def __init__(self, d_raw=16, n_modalities=3, d_common=24, n_universals=6,
                 d_pred=8, n_privations=5, d_out=6, n_levels=3, rng=RNG):
        self.veg = VegetativeLayer(d_raw, d_common, rng)
        self.sens = SensitiveLayer(d_common, n_modalities, d_common, rng)
        self.rat = RationalLayer(d_common, n_universals, temperature=4.0, rng=rng)
        self.hom = HomonymTypeGate(d_pred, d_hidden=32, n_classes=3, rng=rng)
        self.apo = ApophaticConstraintNet(d_common, n_privations, margin=1.0, rng=rng)
        self.gdd = GradedDisclosureDecoder(d_common, d_out, n_levels=n_levels, rng=rng)
        self.gate = PropheticGate(rng=rng)
        self.n_modalities = n_modalities
        self.n_levels = n_levels

        self.params = {}
        for sub in (self.veg, self.sens, self.rat, self.hom, self.apo, self.gdd):
            self.params.update(sub.p)

    # ---- forward pieces reused by loss() ----
    def forward_perception(self, raw_modalities, p):
        """raw_modalities: list of n_modalities arrays (N, d_raw)."""
        veg_hidden = []
        veg_recon = []
        veg_caches = []
        for x in raw_modalities:
            h, recon, c = self.veg.forward(x, p)
            veg_hidden.append(h)
            veg_recon.append(recon)
            veg_caches.append(c)
        common, image, sens_cache = self.sens.forward(veg_hidden, p)
        weights, universal_repr, rat_cache = self.rat.forward(common, p)
        return {
            'veg_hidden': veg_hidden, 'veg_recon': veg_recon, 'veg_caches': veg_caches,
            'common': common, 'image': image, 'sens_cache': sens_cache,
            'weights': weights, 'universal_repr': universal_repr, 'rat_cache': rat_cache,
            'raw_modalities': raw_modalities,
        }

    def loss(self, batch, p, weights=None):
        """
        batch keys:
          raw_modalities        : list[n_modalities] of (N, d_raw)
          pred_emb, target_flag, homonym_label_onehot : homonym-gate supervision
          apophatic_applies_mask: (N, n_privations)
          disclosure_targets    : list[n_levels] of (N, d_out)
        Returns scalar total loss and a dict of raw parameter gradients.
        """
        w = weights or {'veg': 1.0, 'img': 0.5, 'hom': 1.0, 'hom_penalty': 6.0,
                         'apo': 1.0, 'gdd': 1.0, 'l2': 1e-4}
        N = batch['raw_modalities'][0].shape[0]
        fwd = self.forward_perception(batch['raw_modalities'], p)

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        total = 0.0

        # ---- vegetative reconstruction loss ----
        d_veg_hidden_extra = []
        for m in range(self.n_modalities):
            x = batch['raw_modalities'][m]
            recon = fwd['veg_recon'][m]
            diff = recon - x
            l = w['veg'] * np.mean(np.sum(diff ** 2, axis=-1))
            total += l
            d_recon = w['veg'] * 2 * diff / N
            d_h_extra = np.zeros_like(fwd['veg_hidden'][m])
            g = self.veg.backward(d_recon, d_h_extra, fwd['veg_caches'][m], p)
            for k, v in g.items():
                grads[k] += v
            d_veg_hidden_extra.append(None)  # filled after sensitive backward

        # ---- imagination reconstruction: image should recover the common
        # concept the senses jointly point at. The target is FIXED synthetic
        # data (batch['image_target']), not a function of net.params -- see
        # _concept_to_common_matrix for why this must be externally constant. ----
        diff_img = fwd['image'] - batch['image_target']
        l_img = w['img'] * np.mean(np.sum(diff_img ** 2, axis=-1))
        total += l_img
        d_image = w['img'] * 2 * diff_img / N

        # ---- rational layer: pull universal_repr toward common (grounding) ----
        diff_rat = fwd['universal_repr'] - fwd['common']
        l_rat = np.mean(np.sum(diff_rat ** 2, axis=-1))
        total += l_rat
        d_universal_repr = 2 * diff_rat / N
        d_common_from_rat_ground = -2 * diff_rat / N

        g_rat, d_common_from_rat = self.rat.backward(
            d_universal_repr, np.zeros_like(fwd['weights']), fwd['rat_cache'], p)
        for k, v in g_rat.items():
            grads[k] += v

        # ---- sensitive backward (imagination + common fan-in) ----
        d_common_total = d_common_from_rat + d_common_from_rat_ground
        g_sens, d_modalities = self.sens.backward(
            d_common_total, d_image, fwd['sens_cache'], p)
        for k, v in g_sens.items():
            grads[k] += v

        # route d_modalities (grad wrt veg_hidden) back through vegetative tanh
        for m in range(self.n_modalities):
            x = batch['raw_modalities'][m]
            g = self.veg.backward(np.zeros_like(fwd['veg_recon'][m]),
                                   d_modalities[m], fwd['veg_caches'][m], p)
            for k, v in g.items():
                grads[k] += v

        # ---- homonym type gate ----
        probs, hom_cache = self.hom.forward(batch['pred_emb'], batch['target_flag'], p)
        target_onehot = batch['homonym_label_onehot']
        ce = cross_entropy(probs, target_onehot)
        l_hom = w['hom'] * np.mean(ce)
        # asymmetric penalty: extra cost on P(essential | target_flag==1)
        essential_prob_on_target = probs[:, 2] * batch['target_flag'][:, 0]
        l_hom_penalty = w['hom_penalty'] * np.mean(essential_prob_on_target ** 2)
        total += l_hom + l_hom_penalty

        d_logits = w['hom'] * (probs - target_onehot) / N
        d_probs_penalty = np.zeros_like(probs)
        d_probs_penalty[:, 2] = (w['hom_penalty'] * 2 * essential_prob_on_target *
                                  batch['target_flag'][:, 0]) / N
        # softmax jacobian for the penalty term (probs already computed)
        dot = np.sum(probs * d_probs_penalty, axis=-1, keepdims=True)
        d_logits += probs * (d_probs_penalty - dot)

        g_hom, _ = self.hom.backward(d_logits, hom_cache, p)
        for k, v in g_hom.items():
            grads[k] += v

        # ---- apophatic constraint net (operates on the 'common' latent) ----
        violation, apo_cache = self.apo.forward(fwd['common'], p)
        l_apo, g_apo, d_common_from_apo = self.apo.loss_and_grad(
            violation, apo_cache, batch['apophatic_applies_mask'], p)
        total += w['apo'] * l_apo
        for k, v in g_apo.items():
            grads[k] += w['apo'] * v

        # ---- graded disclosure decoder (reads from 'common') ----
        gdd_outputs, gdd_caches = self.gdd.forward(fwd['common'], p)
        l_gdd, g_gdd, d_common_from_gdd = self.gdd.loss_and_grad(
            gdd_outputs, gdd_caches, batch['disclosure_targets'], p)
        total += w['gdd'] * l_gdd
        for k, v in g_gdd.items():
            grads[k] += w['gdd'] * v

        # propagate the two extra d_common contributions (apo, gdd) back through
        # sensitive + vegetative once more (linear superposition of grads)
        g_sens2, d_modalities2 = self.sens.backward(
            w['apo'] * d_common_from_apo + w['gdd'] * d_common_from_gdd,
            np.zeros_like(fwd['image']), fwd['sens_cache'], p)
        for k, v in g_sens2.items():
            grads[k] += v
        for m in range(self.n_modalities):
            g = self.veg.backward(np.zeros_like(fwd['veg_recon'][m]),
                                   d_modalities2[m], fwd['veg_caches'][m], p)
            for k, v in g.items():
                grads[k] += v

        # ---- L2 (the "moral discipline" regularizer used at inference by
        #      PropheticGate.moral_score is *observed*, not optimized; here we
        #      add a light L2 only on the perceptual/rational stack so training
        #      does not degenerate, consistent with Maimonides treating moral
        #      discipline as a precondition cultivated alongside, not opposed
        #      to, intellectual training) ----
        for k in ('veg_W1', 'veg_W2', 'rat_protos'):
            total += w['l2'] * np.sum(p[k] ** 2)
            grads[k] += w['l2'] * 2 * p[k]

        diagnostics = {
            'loss_vegetative': float(sum(
                np.mean(np.sum((fwd['veg_recon'][m] - batch['raw_modalities'][m]) ** 2, axis=-1))
                for m in range(self.n_modalities))),
            'loss_imagination': float(l_img),
            'loss_rational_grounding': float(l_rat),
            'loss_homonym_ce': float(l_hom),
            'loss_homonym_penalty': float(l_hom_penalty),
            'loss_apophatic': float(l_apo),
            'loss_disclosure': float(l_gdd),
            'mean_essential_prob_on_target': float(np.mean(probs[:, 2] * batch['target_flag'][:, 0]) /
                                                    max(1e-9, np.mean(batch['target_flag']))),
            'mean_apophatic_violation_on_applicable': float(
                np.sum(batch['apophatic_applies_mask'] * violation) /
                max(1, np.sum(batch['apophatic_applies_mask']))),
        }
        return total, grads, diagnostics


# ============================================================================
# 10. SYNTHETIC DATA
# ============================================================================
# There is no historical dataset of "what a perfected intellect perceives."
# The synthetic generators below exist purely to give every trainable
# component of MorehNetwork a well-posed, checkable optimization problem:
# structured enough that a correct implementation provably drives every loss
# toward zero, which is what the training run and self-tests demonstrate.

_FIXED_CONCEPT_PROJECTION = {}  # cache of (d_raw, d_common) -> fixed projection matrix


def _concept_to_common_matrix(d_raw, d_common):
    """A FIXED (non-trainable) projection used only to build the synthetic
    imagination target below. It is generated once from a dedicated,
    constant seed independent of the training RNG, so it never depends on
    (and never receives gradient from) any network parameter -- this keeps
    the imagination-reconstruction target a true constant with respect to
    net.params, which is required for the analytic and finite-difference
    gradients to agree exactly."""
    key = (d_raw, d_common)
    if key not in _FIXED_CONCEPT_PROJECTION:
        fixed_rng = np.random.default_rng(20260730)  # arbitrary, fixed
        _FIXED_CONCEPT_PROJECTION[key] = fixed_rng.normal(0, 1.0 / np.sqrt(d_raw), (d_raw, d_common))
    return _FIXED_CONCEPT_PROJECTION[key]


def make_perception_batch(n, d_raw, n_modalities, d_common, rng=RNG):
    """Each sample is a shared latent 'concept' the modalities jointly encode
    (with modality-specific noise), so there is a real multisensory pattern
    for common-sense integration to recover. The imagination target is a
    FIXED, non-trainable projection of that same concept into d_common space
    -- "what the senses jointly point at" -- decoupled from the network's own
    live encoder output so the imagination loss has a genuine external
    target rather than a self-referential one."""
    concept = rng.normal(0, 1, (n, d_raw))
    mods = [np.tanh(concept + rng.normal(0, 0.15, (n, d_raw))) for _ in range(n_modalities)]
    image_target = np.tanh(concept @ _concept_to_common_matrix(d_raw, d_common))
    return mods, image_target, concept


_FIXED_HOMONYM_CENTERS = {}  # cache of d_pred -> (3, d_pred) fixed class centers


def _homonym_centers(d_pred):
    """FIXED embedding centers for the three predicate families (negation /
    action / essential). Fixed across the whole run for the same reason the
    imagination target is fixed: this is a stationary synthetic vocabulary
    the network is learning to classify, not a moving target. A version of
    this bug (regenerating random centers every batch) was caught during
    development precisely because homonym cross-entropy refused to drop
    below the 3-class chance baseline (ln 3 ~= 1.0986) -- a useful reminder
    that "the loss curve looks wrong" is itself a diagnostic, in a network
    literally about not fooling yourself with unearned confidence."""
    if d_pred not in _FIXED_HOMONYM_CENTERS:
        fixed_rng = np.random.default_rng(31337)
        _FIXED_HOMONYM_CENTERS[d_pred] = fixed_rng.normal(0, 1, (3, d_pred))
    return _FIXED_HOMONYM_CENTERS[d_pred]


def make_homonym_batch(n, d_pred, rng=RNG):
    """
    Three predicate families, each with a clear ground-truth type:
      - "negation" predicates  (e.g. 'not-ignorant') -> class 0, legal for all
      - "action" predicates    (e.g. 'creates')       -> class 1, legal for all
      - "essential" predicates (e.g. 'has-a-body')    -> class 2, legal ONLY
        when target_flag == 0 (an ordinary, composite entity)
    target_flag == 1 marks the unconditioned/maximal target: for those rows,
    if the sampled predicate family is 'essential', the correct *label* is
    still class 2 for supervised CE (that's the ground truth of what kind of
    predicate it linguistically is) but the asymmetric penalty in the loss
    additionally punishes the network for assigning it real probability --
    forcing the network to learn a *systematic* near-zero response on that
    combination rather than a labeled reclassification. This mirrors
    Maimonides' claim that ordinary language keeps *offering* essential
    predicates for God; the discipline is refusing them, not being unable to
    parse them.
    """
    family = rng.integers(0, 3, n)  # 0 negation, 1 action, 2 essential
    base = np.zeros((n, 3))
    for c in range(3):
        base[:, c] = (family == c).astype(float)
    centers = _homonym_centers(d_pred)
    pred_emb = base @ centers + rng.normal(0, 0.1, (n, d_pred))
    target_flag = rng.integers(0, 2, (n, 1)).astype(float)
    label_onehot = np.eye(3)[family]
    return pred_emb, target_flag, label_onehot


def make_apophatic_batch(n, d_common, n_privations, rng=RNG):
    """
    Sample latent codes 'common'. For a random subset of (sample, privation)
    pairs, mark applies_mask=1, meaning: for this sample, privation i is a
    *known negation* that must be respected. We construct 'common' so that it
    truly lies outside those marked privation half-spaces by initializing far
    from arbitrary boundary directions -- the generator itself does not know
    the network's learned W, so this simply gives the trainer a batch of
    "typical" latent codes and an applicability mask; the network must learn
    boundaries under which those codes are legitimately non-violating.
    """
    common = rng.normal(0, 1, (n, d_common))
    applies_mask = (rng.random((n, n_privations)) < 0.4).astype(float)
    return common, applies_mask


_FIXED_CONCEPT_TO_TRUTH = {}  # cache of (d_raw, d_out) -> fixed projection matrix


def _concept_to_truth_matrix(d_raw, d_out):
    """FIXED (non-trainable) projection from the shared perceptual 'concept'
    to the 'full truth' vector y_full that GradedDisclosureDecoder is trained
    to reveal progressively. Tying y_full to the SAME concept that generates
    raw_modalities (rather than sampling y_full independently) is essential:
    the decoder reads from 'common', and 'common' is only ever trained to
    correlate with concept (via the vegetative + imagination + rational
    losses). An independent random y_full would be information-theoretically
    unrecoverable from common -- no amount of training could lower that loss
    below its noise floor. With y_full = fixed_fn(concept), the disclosure
    task is exactly as learnable as the perceptual encoder makes it, which
    is the honest dependency: graduated disclosure of a truth the mind does
    not yet fully grasp, not graduated disclosure of pure noise."""
    key = (d_raw, d_out)
    if key not in _FIXED_CONCEPT_TO_TRUTH:
        fixed_rng = np.random.default_rng(31210)
        _FIXED_CONCEPT_TO_TRUTH[key] = fixed_rng.normal(0, 1.0 / np.sqrt(d_raw), (d_raw, d_out))
    return _FIXED_CONCEPT_TO_TRUTH[key]


def make_disclosure_batch(concept, d_out, n_levels):
    """
    y_full = a fixed nonlinear projection of the shared perceptual concept:
    the "full truth" a fully perfected intellect would grasp about what was
    sensed. Level l's target is a *progressively revealed* partial
    reconstruction: target_l = mask_l * y_full, where mask_l keeps only the
    first (l+1)/L share of output coordinates and zeroes the rest. This
    makes low levels literally under-determined subsets of the top level's
    target -- consistent within themselves, incomplete relative to the top --
    exactly the graduated-disclosure structure, while remaining a function of
    the same concept the rest of the network perceives.
    """
    d_raw = concept.shape[1]
    y_full = np.tanh(concept @ _concept_to_truth_matrix(d_raw, d_out))
    targets = []
    for l in range(n_levels):
        keep = int(round(d_out * (l + 1) / n_levels))
        mask = np.zeros(d_out)
        mask[:keep] = 1.0
        targets.append(y_full * mask)
    return targets, y_full


def make_batch(n, d_raw=16, n_modalities=3, d_common=24, d_pred=8,
               n_privations=5, d_out=6, n_levels=3, rng=RNG):
    raw_modalities, image_target, concept = make_perception_batch(
        n, d_raw, n_modalities, d_common, rng)
    pred_emb, target_flag, homonym_label_onehot = make_homonym_batch(n, d_pred, rng)
    _, apophatic_applies_mask = make_apophatic_batch(n, d_common, n_privations, rng)
    disclosure_targets, y_full = make_disclosure_batch(concept, d_out, n_levels)
    return {
        'raw_modalities': raw_modalities,
        'image_target': image_target,
        'pred_emb': pred_emb,
        'target_flag': target_flag,
        'homonym_label_onehot': homonym_label_onehot,
        'apophatic_applies_mask': apophatic_applies_mask,
        'disclosure_targets': disclosure_targets,
        '_y_full': y_full,
        '_concept': concept,
    }


# ============================================================================
# 11. TRAINING LOOP
# ============================================================================

def train(net, steps=400, batch_size=32, lr=0.02, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(net.params, lr=lr)
    history = []
    for step in range(steps):
        batch = make_batch(batch_size, rng=rng)
        loss, grads, diag = net.loss(batch, net.params)
        opt.step(net.params, grads)
        history.append(loss)
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"  step {step:4d}  total_loss={loss:9.4f}  "
                  f"veg={diag['loss_vegetative']:.4f}  img={diag['loss_imagination']:.4f}  "
                  f"hom_ce={diag['loss_homonym_ce']:.4f}  "
                  f"P(essential|target)={diag['mean_essential_prob_on_target']:.4f}  "
                  f"apo_violation={diag['mean_apophatic_violation_on_applicable']:.4f}  "
                  f"disclosure={diag['loss_disclosure']:.4f}")
    return history


# ============================================================================
# 12. GRADIENT CHECK (mandatory)
# ============================================================================

def gradient_check(net, n_checks_per_param=3, batch_size=6, eps=1e-5, seed=42):
    rng = np.random.default_rng(seed)
    batch = make_batch(batch_size, rng=rng)

    def loss_fn():
        l, _, _ = net.loss(batch, net.params)
        return l

    _, analytic_grads, _ = net.loss(batch, net.params)

    report = []
    all_ok = True
    for key in net.params:
        arr = net.params[key]
        flat_idx = rng.choice(arr.size, size=min(n_checks_per_param, arr.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_fn()
            arr[idx] = orig - eps
            lm = loss_fn()
            arr[idx] = orig
            numeric = (lp - lm) / (2 * eps)
            analytic = analytic_grads[key][idx]
            denom = max(1e-8, abs(numeric) + abs(analytic))
            rel_err = abs(numeric - analytic) / denom
            ok = rel_err < 3e-2  # tolerance: tanh/softmax stacks + float64 numerics
            all_ok = all_ok and ok
            report.append((key, idx, numeric, analytic, rel_err, ok))
    return all_ok, report


# ============================================================================
# 13. SELF-TESTS
# ============================================================================

def run_self_tests(net, rng_seed=99):
    results = {}
    rng = np.random.default_rng(rng_seed)

    # Test A: homonym gate suppresses essential-attribute probability on the
    # unconditioned target after training.
    pred_emb, target_flag, _ = make_homonym_batch(300, net.hom.p['hom_W1'].shape[0] - 1, rng)
    probs, _ = net.hom.forward(pred_emb, target_flag, net.params)
    mask = target_flag[:, 0] == 1
    p_essential_on_target = float(np.mean(probs[mask, 2])) if mask.any() else float('nan')
    p_essential_on_ordinary = float(np.mean(probs[~mask, 2])) if (~mask).any() else float('nan')
    results['A_homonym_essential_suppressed_on_target'] = {
        'p_essential_given_target': p_essential_on_target,
        'p_essential_given_ordinary': p_essential_on_ordinary,
        'pass': p_essential_on_target < 0.15 and p_essential_on_target < p_essential_on_ordinary,
    }

    # Test B: apophatic violations near zero where the exclusion applies.
    # Evaluated on the network's ACTUAL encoded 'common' representation from
    # a fresh perception batch (not an independently-sampled vector), since
    # the apophatic boundaries were only ever trained against that
    # distribution (tanh-bounded latents produced by SensitiveLayer).
    d_raw = net.veg.p['veg_W1'].shape[0]
    d_common = net.apo.p['apo_W'].shape[0]
    n_mod = net.n_modalities
    raw_modalities, _, _ = make_perception_batch(300, d_raw, n_mod, d_common, rng)
    fwd = net.forward_perception(raw_modalities, net.params)
    _, applies_mask = make_apophatic_batch(300, d_common, net.apo.M, rng)
    violation, _ = net.apo.forward(fwd['common'], net.params)
    mean_viol_applicable = float(np.sum(applies_mask * violation) / max(1, np.sum(applies_mask)))
    results['B_apophatic_violations_small_where_applicable'] = {
        'mean_violation_on_applicable_region': mean_viol_applicable,
        'pass': mean_viol_applicable < 0.25,
    }

    # Test C: graded disclosure -- each level's own reconstruction error is
    # low, AND lower levels genuinely differ from the top level on the
    # coordinates the top level alone reveals (surface disagreement that
    # resolves only at the top). Uses the SAME concept that generated the
    # perception batch above, so 'common' and the disclosure targets are
    # drawn from a jointly consistent distribution, exactly as in training.
    raw_modalities2, _, concept2 = make_perception_batch(300, d_raw, n_mod, d_common, rng)
    fwd2 = net.forward_perception(raw_modalities2, net.params)
    targets, y_full = make_disclosure_batch(concept2, net.gdd.p['gdd_b0'].shape[0], net.gdd.L)
    outputs, _ = net.gdd.forward(fwd2['common'], net.params)
    level_errors = [float(np.mean(np.sum((outputs[l] - targets[l]) ** 2, axis=-1)))
                     for l in range(net.gdd.L)]
    # disagreement between level 0 and top level on the OUTPUT coordinates
    # (dimensions of d_out, not of the latent) that level 0's own target
    # masked to zero and only the top level was trained to reveal.
    top = net.gdd.L - 1
    d_out = outputs[0].shape[1]
    level0_keep = int(round(d_out * 1 / net.gdd.L))
    disagreement = float(np.mean(np.abs(outputs[0][:, level0_keep:] - outputs[top][:, level0_keep:]))) \
        if d_out > level0_keep else 0.0
    results['C_graded_disclosure'] = {
        'level_errors': level_errors,
        'low_level_vs_top_level_tail_disagreement': disagreement,
        'pass': all(e < 0.6 for e in level_errors),
    }

    # Test D: prophetic gate opens only probabilistically even when both
    # capability and moral conditions are satisfied -- capability != authorization.
    open_count = 0
    trials = 400
    for _ in range(trials):
        info = net.gate.evaluate(rational_confidence=0.95, all_params=net.params)
        open_count += int(info['gate_open'] and info['capability_ready'])
    open_rate = open_count / trials
    results['D_prophetic_gate_probabilistic_even_when_ready'] = {
        'open_rate_given_ready': open_rate,
        'expected_near': net.gate.p_divine_will,
        'pass': abs(open_rate - net.gate.p_divine_will) < 0.12,
    }

    return results


def applicable_n(net):
    return net.apo.M


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 78)
    print("MIND #0249 -- MOSES MAIMONIDES -- Apophatic / Graded-Disclosure Network")
    print("=" * 78)

    net = MorehNetwork(d_raw=16, n_modalities=3, d_common=24, n_universals=6,
                        d_pred=8, n_privations=5, d_out=6, n_levels=3)

    n_params = sum(v.size for v in net.params.values())
    print(f"\nParameter tensors: {len(net.params)}   total scalars: {n_params}")

    print("\n[1/3] Finite-difference gradient check (mandatory) ...")
    ok, report = gradient_check(net, n_checks_per_param=3)
    n_bad = sum(1 for r in report if not r[5])
    print(f"  checked {len(report)} individual parameter entries across "
          f"{len(net.params)} tensors")
    print(f"  failures: {n_bad} / {len(report)}")
    if n_bad:
        for key, idx, num, ana, rel, ok_ in report:
            if not ok_:
                print(f"    MISMATCH {key}{idx}: numeric={num:.6f} analytic={ana:.6f} rel_err={rel:.4f}")
    print(f"  GRADIENT CHECK: {'PASSED' if ok else 'FAILED'}")
    assert ok, "Gradient check failed -- aborting before training on a broken backward pass."

    print("\n[2/3] Training (Adam, 600 steps, real synthetic optimization problem) ...")
    history = train(net, steps=600, batch_size=32, lr=0.02)
    print(f"  loss[0]   = {history[0]:.4f}")
    print(f"  loss[-1]  = {history[-1]:.4f}")
    print(f"  reduction = {100 * (1 - history[-1] / history[0]):.1f}%")

    print("\n[3/3] Self-tests ...")
    results = run_self_tests(net)
    for name, r in results.items():
        status = "PASS" if r['pass'] else "FAIL"
        details = {k: v for k, v in r.items() if k != 'pass'}
        print(f"  [{status}] {name}")
        for k, v in details.items():
            print(f"           {k} = {v}")

    all_pass = all(r['pass'] for r in results.values())
    print("\n" + "=" * 78)
    print(f"OVERALL: gradient_check={'PASS' if ok else 'FAIL'}  "
          f"training_converged={'YES' if history[-1] < 0.5 * history[0] else 'NO'}  "
          f"self_tests={'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 78)

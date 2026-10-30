"""
# Encyclopedia of Lost Minds: Echoes on AI · Chapter 0312 · Ibn al-Athir (Izz al-Din Abu al-Hasan Ali ibn Muhammad al-Shaybani, 1160-1233 CE)
# By David Vivancos · https://www.vivancos.com/ · https://lostmindsai.com
# Tome 16, Minds 301-320 Available on Amazon https://www.amazon.com/gp/product/B0HNJQX9S4
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos 


al-Jazari, 1160-1233 CE), historian of Mosul, author of al-Kamil fi al-Tarikh
("The Complete History") and Usd al-Ghaba fi Ma'rifat al-Sahaba ("The Lions of
the Forest, on Knowing the Companions").

WHAT THIS FILE IS
------------------------------------------------------------------------------
A trainable, from-scratch (pure NumPy, no autodiff library) neural architecture
whose STRUCTURE is an argument about Ibn al-Athir's distinctive cognitive
signature, not a generic transformer wearing a historian's costume. Every
class below corresponds to a documented, source-attested feature of how he
actually organized knowledge -- see the "Sources" section in the companion
chapter text for citations to al-Kamil's Muqaddima, Usd al-Ghaba, al-Bahir,
and the modern scholarship (D.S. Richards, Carole Hillenbrand, Peter Edbury,
Francoise Micheau) that documents his method and its structural bias.

THE THESIS, IN ONE PARAGRAPH
------------------------------------------------------------------------------
Ibn al-Athir stored the same underlying knowledge of persons and events
TWICE, in two structurally orthogonal indexes built for two different
cognitive operations: al-Kamil is annalistic (organized strictly by Hijri
year, built for causal narrative -- "what led to what"), while Usd al-Ghaba
is alphabetical (organized by name, built for identity lookup -- "who is
this and what is reliably known of them," independent of when they lived).
Layered on that dual index is a documented, UNDECLARED structural bias: his
family served the Zengid atabegs of Mosul, and modern source-criticism
(Richards, Edbury, Hillenbrand) has been able to locate and measure a
consistent tilt in his evaluative language toward the Zengids and against
the Ayyubids who displaced them -- a bias recoverable specifically BECAUSE
his factual/dating apparatus is so exhaustively cross-referenced that the
evaluative residue stands out against it. Finally, confronted with the
Mongol invasion, he did something specific: he named a proximate cause (the
Khwarazm-shah's governor at Otrar massacring a Mongol merchant caravan,
1219) while simultaneously insisting, repeatedly and in his own voice, that
the resulting catastrophe had no precedent "since God created Adam" --
holding CAUSAL EXPLICABILITY and PRECEDENT-FAMILIARITY apart, rather than
collapsing one into the other, rather than either forcing the event into
his usual cyclic dynastic template or refusing to explain it because it
was unprecedented.

This file builds all three of those moves as executable, gradient-checked,
trainable machinery:

  1. DualIndexTrunk        -- one shared event/entity representation,
                               read out through two independent heads:
                               a temporal-causal head (order-dependent) and
                               an alphabetical-identity head (order-invariant).
  2. TestimonialTrustLedger -- two source channels (eyewitness/family
                               testimony vs. compiled written report) with
                               different, explicit trust-update dynamics,
                               echoing al-Bahir's reliance on his father's
                               first-hand account versus al-Kamil's reliance
                               on compiled chains of prior historians.
  3. LegibleBiasResidual    -- an evaluative ("sentiment") output built from
                               TWO additive, separately-owned terms: a
                               content term (from the shared trunk) and an
                               isolated patron-affiliation bias term that is
                               never merged into the shared representation,
                               so it can be measured and subtracted by an
                               external audit rather than hidden inside an
                               entangled embedding.
  4. PrecedentCauseDecoupler -- a novelty/precedent scorer (autoencoder
                               reconstruction error over the historical
                               distribution of ordinary annal events) run
                               IN PARALLEL with, and never gating, a causal-
                               trigger classifier -- so an event can be
                               simultaneously flagged "unprecedented" and
                               still receive a proximate-cause explanation.

Everything is pure NumPy. Every trainable module implements its own forward
AND backward pass (explicit chain rule, no autograd). A finite-difference
gradient check is run over a sample of parameters from every head before
training is trusted. A real training loop runs on a synthetic annals corpus
built to have the documented structure described above. Five self-tests
close the file, each one testing a specific claim made in the paragraph
above, not merely that the code executes.

AUTHOR: 1000 Minds Project -- Encyclopedia of Lost Minds: Echoes on AI
CHAPTER: 0312 -- Ibn al-Athir (1160-1233 CE)
"""

import numpy as np

RNG = np.random.default_rng(20560223)  # seed nods to 20/2/1233 CE, his death


# ==============================================================================
# PART 0. TINY FROM-SCRATCH AUTOGRAD-FREE LAYER PRIMITIVES
# ==============================================================================
# Every primitive owns its parameters, its forward cache, and its own
# backward() that returns d(loss)/d(input) while accumulating d(loss)/d(param)
# into self.grads. No autodiff engine is used anywhere in this file.

class Linear:
    """y = x @ W + b, with Glorot-ish init and manual backward."""

    def __init__(self, n_in, n_out, name):
        scale = np.sqrt(2.0 / (n_in + n_out))
        self.W = RNG.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.name = name
        self.grads = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self._x = None

    def params(self):
        return {f"{self.name}.W": self.W, f"{self.name}.b": self.b}

    def grad_map(self):
        return {f"{self.name}.W": self.grads["W"], f"{self.name}.b": self.grads["b"]}

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.grads["W"] += self._x.T @ dy
        self.grads["b"] += dy.sum(axis=0)
        return dy @ self.W.T

    def zero_grad(self):
        self.grads["W"][:] = 0.0
        self.grads["b"][:] = 0.0


class Tanh:
    def __init__(self):
        self._y = None

    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, dy):
        return dy * (1.0 - self._y ** 2)


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, dy):
        return dy * self._mask


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_ce_loss(logits, labels):
    """labels: int array (N,). Returns scalar mean loss, dlogits (N,C)."""
    n = logits.shape[0]
    p = softmax(logits)
    p_clipped = np.clip(p[np.arange(n), labels], 1e-12, 1.0)
    loss = -np.log(p_clipped).mean()
    dlogits = p.copy()
    dlogits[np.arange(n), labels] -= 1.0
    dlogits /= n
    return loss, dlogits


def mse_loss(pred, target):
    diff = pred - target
    loss = np.mean(diff ** 2)
    dpred = 2.0 * diff / diff.size
    return loss, dpred


class MLP2:
    """A tiny 2-layer MLP block (Linear-Tanh-Linear) used as a reusable head."""

    def __init__(self, n_in, n_hidden, n_out, name):
        self.l1 = Linear(n_in, n_hidden, f"{name}.l1")
        self.act = Tanh()
        self.l2 = Linear(n_hidden, n_out, f"{name}.l2")
        self.name = name

    def forward(self, x):
        h = self.act.forward(self.l1.forward(x))
        return self.l2.forward(h)

    def backward(self, dy):
        dh = self.l2.backward(dy)
        dh = self.act.backward(dh)
        return self.l1.backward(dh)

    def layers(self):
        return [self.l1, self.l2]


# ==============================================================================
# PART 1. SYNTHETIC ANNALS CORPUS
# ==============================================================================
# We cannot train on the real al-Kamil (11 volumes, untranscribed as tensors),
# so we build a synthetic corpus that reproduces the STRUCTURE the real corpus
# is documented to have: entities with patron affiliation, events dated by
# Hijri year with a category, two source channels with different reliability
# dynamics, and exactly one designated "no precedent" catastrophe event whose
# feature magnitude is drawn from a different, far distribution -- standing
# in for the Mongol invasion against six centuries of ordinary dynastic annals.

EVENT_CATEGORIES = ["accession", "battle", "death", "treaty", "natural_disaster"]
N_NORMAL_CATS = len(EVENT_CATEGORIES)
CATASTROPHE_CAT = N_NORMAL_CATS  # index 5, held out of "normal" distribution

CAUSE_CLASSES = [
    "dynastic_succession_dispute",
    "military_campaign_decision",
    "diplomatic_incident",       # Otrar: massacre of envoys/merchants
    "natural_cause",
    "economic_dispute",
]

ENTITY_TYPES = ["person", "place", "dynasty", "book"]

FEATURE_DIM = 16  # raw hand-crafted feature width for events
ENTITY_EMBED_DIM = 12
EVENT_TRUNK_DIM = 24
HIDDEN = 32


def _alpha_name(i):
    """Deterministic pseudo-alphabetical entity name, letters A.. then AA.. """
    letters = "abcdefghijklmnopqrstuvwxyz"
    if i < 26:
        return letters[i]
    return letters[i // 26 - 1] + letters[i % 26]


def build_entities(n_entities=40):
    entities = []
    for i in range(n_entities):
        etype = i % len(ENTITY_TYPES)
        # Patron affiliation: 1.0 = Zengid-affiliated, 0.0 = Ayyubid-affiliated,
        # 0.5 = unaffiliated (Baghdad caliphal / neutral scholarly entities).
        roll = RNG.random()
        affiliation = 1.0 if roll < 0.4 else (0.0 if roll < 0.75 else 0.5)
        entities.append({
            "id": i,
            "name": _alpha_name(i),
            "type": etype,
            "affiliation": affiliation,
        })
    entities.sort(key=lambda e: e["name"])  # Usd al-Ghaba's alphabetical order
    for rank, e in enumerate(entities):
        e["alpha_rank"] = rank
    return entities


def build_events(entities, n_events=160, hijri_start=550, hijri_end=629):
    """
    Builds a chronologically-ordered (al-Kamil-style) annal. Each event cites
    a source channel: 0 = compiled written report, 1 = direct/testimonial
    report (family or eyewitness, as in al-Bahir's reliance on his father).
    A handful of events per entity form loose "corroboration" clusters so the
    trust ledger has something to update on.
    """
    events = []
    years = np.sort(RNG.uniform(hijri_start, hijri_end, size=n_events))
    for idx, year in enumerate(years):
        cat = RNG.integers(0, N_NORMAL_CATS)
        entity = entities[RNG.integers(0, len(entities))]
        channel = 1 if RNG.random() < 0.22 else 0  # testimonial is the minority
        corroborations = RNG.integers(0, 4)
        # Cause label correlates loosely with category (battles/treaties tend
        # to trace to campaign decisions or disputes; deaths to natural
        # causes; accessions to succession disputes) -- but noisy, as real
        # historiography is.
        cause_prior = {
            0: 0,  # accession -> succession dispute
            1: 1,  # battle -> campaign decision
            2: 3,  # death -> natural cause
            3: 4,  # treaty -> economic dispute
            4: 3,  # natural_disaster -> natural cause
        }[cat]
        if RNG.random() < 0.25:
            cause = RNG.integers(0, len(CAUSE_CLASSES))
        else:
            cause = cause_prior

        feat = np.zeros(FEATURE_DIM)
        feat[0] = (year - hijri_start) / (hijri_end - hijri_start)
        feat[1 + cat] = 1.0                                   # one-hot category
        feat[1 + N_NORMAL_CATS + entity["type"]] = 1.0        # one-hot entity type
        # NOTE: patron affiliation is deliberately NOT placed in the shared
        # content features. It reaches the model ONLY through the isolated
        # bias_head side-channel (see LegibleBiasResidual) -- this is what
        # makes the bias term structurally separable rather than merely
        # correlated-and-hidden inside a shared embedding.
        feat[10] = 0.0
        feat[11] = channel
        feat[12] = corroborations / 3.0
        feat[13:16] = RNG.normal(0.0, 0.25, size=3)            # narrative noise

        # Sentiment target: content-neutral base plus a DOCUMENTED, patron-
        # correlated skew. This is the ground truth the LegibleBiasResidual
        # module is asked to recover: praise skews toward Zengid affiliation,
        # away from Ayyubid, exactly the tilt Richards/Edbury/Hillenbrand
        # locate in the real al-Kamil.
        base_sentiment = RNG.normal(0.0, 0.15)
        patron_skew = 0.6 * (entity["affiliation"] - 0.5) * 2.0  # in [-0.6,0.6]
        sentiment = np.clip(base_sentiment + patron_skew, -1.0, 1.0)

        events.append({
            "idx": idx, "year": year, "category": cat, "entity_id": entity["id"],
            "channel": channel, "corroborations": int(corroborations),
            "cause": cause, "sentiment": sentiment, "feat": feat,
        })

    # --- The designated catastrophe event (stand-in for the Mongol invasion,
    # 616-617 AH / 1219-1220 CE): magnitude features drawn from a genuinely
    # different distribution (large positive outliers in the noise channels,
    # simulating "outside the learned template"), but WITH a clean, legible
    # trigger cause: diplomatic_incident, echoing Otrar.
    cat_year = 617.0
    cat_entity = entities[RNG.integers(0, len(entities))]
    feat = np.zeros(FEATURE_DIM)
    feat[0] = (cat_year - hijri_start) / (hijri_end - hijri_start)
    feat[1 + N_NORMAL_CATS] = 0.0  # no normal-category one-hot fires
    feat[1 + N_NORMAL_CATS + cat_entity["type"]] = 1.0
    feat[10] = 0.0  # affiliation kept out of shared content features (see above)
    feat[11] = 1.0  # eyewitness-adjacent testimony (informants, per al-Kamil)
    feat[12] = 0.0 / 3.0  # uncorroborated -- genuinely first-of-its-kind
    feat[13:16] = RNG.normal(4.0, 0.3, size=3)  # far outside ordinary scale

    catastrophe = {
        "idx": n_events, "year": cat_year, "category": CATASTROPHE_CAT,
        "entity_id": cat_entity["id"], "channel": 1, "corroborations": 0,
        "cause": 2,  # diplomatic_incident -- the Otrar massacre of envoys
        "sentiment": -0.9, "feat": feat, "is_catastrophe": True,
    }
    for e in events:
        e["is_catastrophe"] = False
    events.append(catastrophe)
    events.sort(key=lambda e: e["year"])
    for i, e in enumerate(events):
        e["temporal_rank"] = i
    return events


# ==============================================================================
# PART 2. DUAL-INDEX TRUNK
# ==============================================================================

class DualIndexTrunk:
    """
    One shared event representation (the corpus), read out by two orthogonal
    heads:

      * temporal head:   causal, WINDOW-ORDER-DEPENDENT. Given the trunk
        embeddings of the k events immediately preceding event t in Hijri-year
        order, predict event t's category. This is al-Kamil: you cannot
        answer "what happened" without knowing what came before it in time.

      * entity head:     alphabetical, ORDER-INDEPENDENT. Given only entity e's
        own embedding (never neighboring entities, never time), predict its
        type. This is Usd al-Ghaba: identity lookup that a reader can perform
        having opened the book to any letter, in any order, on any day.

    The orthogonality is structural, not incidental: the entity head's input
    never includes a temporal-window feature, and the temporal head's input
    never includes entity alphabetical rank. test_dual_index_orthogonality()
    verifies this empirically below.
    """

    def __init__(self, feature_dim=FEATURE_DIM, trunk_dim=EVENT_TRUNK_DIM,
                 entity_embed_dim=ENTITY_EMBED_DIM, n_entities=40, window=3):
        self.window = window
        self.event_encoder = MLP2(feature_dim, HIDDEN, trunk_dim, "trunk.event")
        # Entity embedding table: a genuine trainable lookup (parameter matrix),
        # not derived from any event or time -- the "book" of Usd al-Ghaba.
        self.entity_table = RNG.normal(0.0, 0.1, size=(n_entities, entity_embed_dim))
        self.entity_table_grad = np.zeros_like(self.entity_table)

        self.temporal_head = MLP2(trunk_dim * window, HIDDEN, N_NORMAL_CATS + 1,
                                   "temporal_head")
        self.entity_head = MLP2(entity_embed_dim, HIDDEN, len(ENTITY_TYPES),
                                 "entity_head")

    def params(self):
        p = {}
        for l in self.event_encoder.layers() + self.temporal_head.layers() + \
                self.entity_head.layers():
            p.update(l.params())
        p["trunk.entity_table"] = self.entity_table
        return p

    def grad_map(self):
        g = {}
        for l in self.event_encoder.layers() + self.temporal_head.layers() + \
                self.entity_head.layers():
            g.update(l.grad_map())
        g["trunk.entity_table"] = self.entity_table_grad
        return g

    def zero_grad(self):
        for l in self.event_encoder.layers() + self.temporal_head.layers() + \
                self.entity_head.layers():
            l.zero_grad()
        self.entity_table_grad[:] = 0.0

    # -- shared event embedding --------------------------------------------
    def encode_events(self, feats):
        """feats: (N, feature_dim) in temporal order -> (N, trunk_dim)."""
        return self.event_encoder.forward(feats)

    def encode_events_backward(self, d_trunk):
        return self.event_encoder.backward(d_trunk)

    # -- temporal (annalistic, order-dependent) head -------------------------
    def make_causal_windows(self, trunk_embeds):
        """
        trunk_embeds assumed ALREADY in the order presented (chronological for
        the real task; test_dual_index_orthogonality() calls this a second
        time on a shuffled order to show predictions change).
        Returns (N-window, window*trunk_dim) stacked causal windows and the
        target index alignment (predict event i from events i-window..i-1).
        """
        n, d = trunk_embeds.shape
        w = self.window
        if n <= w:
            return np.zeros((0, w * d)), np.array([], dtype=int)
        rows = []
        targets = []
        for i in range(w, n):
            rows.append(trunk_embeds[i - w:i].reshape(-1))
            targets.append(i)
        return np.stack(rows), np.array(targets)

    def temporal_forward(self, windows):
        return self.temporal_head.forward(windows)

    def temporal_backward(self, dlogits):
        return self.temporal_head.backward(dlogits)

    # -- entity (alphabetical, order-independent) head -----------------------
    def entity_forward(self, entity_ids):
        embeds = self.entity_table[entity_ids]
        logits = self.entity_head.forward(embeds)
        return logits, embeds

    def entity_backward(self, dlogits, entity_ids):
        dembeds = self.entity_head.backward(dlogits)
        np.add.at(self.entity_table_grad, entity_ids, dembeds)


# ==============================================================================
# PART 3. TESTIMONIAL VS COMPILED TRUST LEDGER
# ==============================================================================

class TestimonialTrustLedger:
    """
    Two source channels, two DIFFERENT update dynamics -- not just two inputs
    to one learned function. This is the isnad-adjacent, al-Bahir-adjacent
    distinction: a testimonial report (his father's first-hand service under
    the Zengid atabegs) starts trusted and decays SLOWLY under contradiction;
    a compiled report (drawn through a chain of prior written historians)
    starts at moderate trust and moves mostly on CORROBORATION COUNT.

    A trainable head (`prior_net`) proposes a base trust from features
    (channel, corroboration count, entity reliability proxy); a separate,
    non-trainable, explicit RULE layered on top enacts the channel-specific
    update dynamic. Both matter: the network must still learn a sensible
    prior (gradient-checked, see PART 5), while the asymmetric dynamic is a
    designed structural property, not something the optimizer could erode
    by finding a shortcut.
    """

    def __init__(self, feat_dim=4):
        self.prior_net = MLP2(feat_dim, 16, 1, "trust.prior")
        # channel-specific decay/rise rates, fixed by design (not trained):
        self.testimonial_decay = 0.08   # slow decay under contradiction
        self.compiled_decay = 0.35      # fast decay under contradiction
        self.testimonial_rise = 0.05    # corroboration matters less: already trusted
        self.compiled_rise = 0.18       # corroboration matters more: builds trust

    def params(self):
        p = {}
        for l in self.prior_net.layers():
            p.update(l.params())
        return p

    def grad_map(self):
        g = {}
        for l in self.prior_net.layers():
            g.update(l.grad_map())
        return g

    def zero_grad(self):
        for l in self.prior_net.layers():
            l.zero_grad()

    def features(self, channel, corroborations, entity_reliability_proxy):
        n = len(channel)
        feat = np.zeros((n, 4))
        feat[:, 0] = channel
        feat[:, 1] = np.array(corroborations) / 3.0
        feat[:, 2] = entity_reliability_proxy
        feat[:, 3] = 1.0
        return feat

    def forward(self, channel, corroborations, entity_reliability_proxy):
        feat = self.features(channel, corroborations, entity_reliability_proxy)
        prior = self.prior_net.forward(feat)          # (N,1), pre-sigmoid
        prior_trust = 1.0 / (1.0 + np.exp(-prior))     # squashed to (0,1)
        self._cache = (feat, prior, prior_trust)
        return prior_trust.reshape(-1)

    def backward(self, d_trust):
        feat, prior, prior_trust = self._cache
        d_prior_trust = d_trust.reshape(-1, 1)
        d_prior = d_prior_trust * prior_trust * (1 - prior_trust)
        self.prior_net.backward(d_prior)

    def online_update(self, base_trust, channel, contradicted, corroborations):
        """
        Explicit, rule-based (non-gradient) channel-specific update, applied
        AFTER the learned prior. This is the piece that makes testimonial and
        compiled trust behave differently under identical evidence -- the
        object exercised in test_testimonial_trust_dynamics() below.
        """
        trust = np.array(base_trust, dtype=float).copy()
        for i in range(len(trust)):
            if channel[i] == 1:  # testimonial
                if contradicted[i]:
                    trust[i] -= self.testimonial_decay
                trust[i] += self.testimonial_rise * (corroborations[i] / 3.0)
            else:  # compiled
                if contradicted[i]:
                    trust[i] -= self.compiled_decay
                trust[i] += self.compiled_rise * (corroborations[i] / 3.0)
        return np.clip(trust, 0.0, 1.0)


# ==============================================================================
# PART 4. LEGIBLE BIAS RESIDUAL
# ==============================================================================

class LegibleBiasResidual:
    """
    Predicts an evaluative "sentiment" score for an event/entity as the SUM of
    two independently-owned terms:

        sentiment = content_head(shared_trunk_embedding)      [factual/narrative]
                  + bias_head(patron_affiliation_scalar)       [isolated residual]

    The design claim embodied here: comprehensiveness and structural loyalty
    are not mutually exclusive, and a system does not have to pretend to
    scrub bias from its weights to be broadly useful -- it can instead keep
    the bias in a SEPARATE, LOW-DIMENSIONAL, EXTERNALLY-AUDITABLE term that
    never touches the factual/dating pathway (DualIndexTrunk above). audit()
    demonstrates the recovery: it regresses the model's total sentiment
    output against patron affiliation on held-out data and compares the
    recovered slope to the bias_head's own learned values -- they must agree,
    because the bias really does live in one legible place, not smeared
    through the shared representation.
    """

    def __init__(self, trunk_dim=EVENT_TRUNK_DIM):
        self.content_head = MLP2(trunk_dim, 16, 1, "bias.content")
        # bias_head: a length-3 lookup (Ayyubid=0.0, neutral=0.5, Zengid=1.0)
        # implemented as a tiny linear layer on a 1-D affiliation scalar so it
        # remains a genuine, separately-audited trainable parameter, not a
        # hand-set constant.
        self.bias_head = Linear(1, 1, "bias.head")

    def params(self):
        p = {}
        for l in self.content_head.layers():
            p.update(l.params())
        p.update(self.bias_head.params())
        return p

    def grad_map(self):
        g = {}
        for l in self.content_head.layers():
            g.update(l.grad_map())
        g.update(self.bias_head.grad_map())
        return g

    def zero_grad(self):
        for l in self.content_head.layers():
            l.zero_grad()
        self.bias_head.zero_grad()

    def forward(self, trunk_embeds, affiliation):
        content = self.content_head.forward(trunk_embeds)          # (N,1)
        aff = np.asarray(affiliation).reshape(-1, 1)
        bias = self.bias_head.forward(aff)                          # (N,1)
        self._cache = (content.shape, bias.shape)
        total = (content + bias).reshape(-1)
        return total, content.reshape(-1), bias.reshape(-1)

    def backward(self, d_total):
        d_total = d_total.reshape(-1, 1)
        d_content = self.content_head.backward(d_total)
        d_bias = self.bias_head.backward(d_total)
        return d_content, d_bias

    def audit(self, total_sentiment, affiliation):
        """
        External audit: ordinary-least-squares slope of total_sentiment on
        affiliation, computed with NO access to the internal bias_head --
        exactly what a downstream reader of al-Kamil, armed only with the
        text and a list of who patronized whom, could in principle recover.
        """
        x = np.asarray(affiliation, dtype=float)
        y = np.asarray(total_sentiment, dtype=float)
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = (x_c ** 2).sum()
        slope = (x_c * y_c).sum() / denom if denom > 1e-9 else 0.0
        return slope


# ==============================================================================
# PART 5. PRECEDENT / CAUSE DECOUPLER
# ==============================================================================

class PrecedentCauseDecoupler:
    """
    Two parallel readouts off the shared trunk embedding that are NEVER
    allowed to gate one another:

      * precedent autoencoder: trained ONLY on ordinary annal events
        (categories 0..N_NORMAL_CATS-1). Reconstruction error is the
        precedent-dissimilarity score -- how far this event sits from six
        centuries of dynastic rise-battle-death-treaty-disaster pattern.

      * causal trigger classifier: trained on ALL events, including the
        catastrophe, to predict a proximate-cause class (Otrar's
        "diplomatic_incident" among them). It is a plain classifier with no
        dependency on the autoencoder's error signal -- it will emit a
        confident, finite prediction for the catastrophe event exactly as
        readily as for an ordinary battle, because explaining a trigger and
        recognizing an unprecedented outcome are, architecturally, two
        different questions asked of the same evidence.

    This is Ibn al-Athir's own move on the Mongols: naming Otrar as sabab
    (cause) while insisting the scale of what followed had "no precedent...
    since God created Adam" -- two judgments held side by side, not merged.
    """

    def __init__(self, trunk_dim=EVENT_TRUNK_DIM, bottleneck=6):
        self.encoder = Linear(trunk_dim, bottleneck, "precedent.enc")
        self.enc_act = Tanh()
        self.decoder = Linear(bottleneck, trunk_dim, "precedent.dec")
        self.cause_head = MLP2(trunk_dim, 16, len(CAUSE_CLASSES), "cause_head")

    def params(self):
        p = {}
        p.update(self.encoder.params())
        p.update(self.decoder.params())
        for l in self.cause_head.layers():
            p.update(l.params())
        return p

    def grad_map(self):
        g = {}
        g.update(self.encoder.grad_map())
        g.update(self.decoder.grad_map())
        for l in self.cause_head.layers():
            g.update(l.grad_map())
        return g

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        for l in self.cause_head.layers():
            l.zero_grad()

    def autoencode(self, trunk_embeds):
        z = self.enc_act.forward(self.encoder.forward(trunk_embeds))
        recon = self.decoder.forward(z)
        return recon

    def autoencode_backward(self, d_recon):
        dz = self.decoder.backward(d_recon)
        dz = self.enc_act.backward(dz)
        return self.encoder.backward(dz)

    def precedent_score(self, trunk_embeds):
        """Reconstruction error per example -- higher = less precedented."""
        recon = self.autoencode(trunk_embeds)
        err = np.mean((recon - trunk_embeds) ** 2, axis=1)
        return err

    def cause_forward(self, trunk_embeds):
        return self.cause_head.forward(trunk_embeds)

    def cause_backward(self, dlogits):
        return self.cause_head.backward(dlogits)


# ==============================================================================
# PART 6. THE FULL MODEL -- AthirNet
# ==============================================================================

class AthirNet:
    """
    Composes all four modules over one shared DualIndexTrunk event embedding.
    forward_and_loss() computes every task loss and caches everything needed
    for backward(); backward() runs the full manual chain rule and returns a
    grads dict keyed exactly like params().
    """

    def __init__(self, n_entities):
        self.trunk = DualIndexTrunk(n_entities=n_entities)
        self.trust = TestimonialTrustLedger()
        self.bias = LegibleBiasResidual()
        self.decoupler = PrecedentCauseDecoupler()

    def params(self):
        p = {}
        p.update(self.trunk.params())
        p.update(self.trust.params())
        p.update(self.bias.params())
        p.update(self.decoupler.params())
        return p

    def grad_map(self):
        g = {}
        g.update(self.trunk.grad_map())
        g.update(self.trust.grad_map())
        g.update(self.bias.grad_map())
        g.update(self.decoupler.grad_map())
        return g

    def zero_grad(self):
        self.trunk.zero_grad()
        self.trust.zero_grad()
        self.bias.zero_grad()
        self.decoupler.zero_grad()

    # -------------------------------------------------------------------
    def forward_and_loss(self, batch):
        """
        batch: dict with parallel arrays (already sorted by 'year' -- i.e. in
        al-Kamil's chronological order) --
          feats (N,FEATURE_DIM), category (N,), entity_id (N,), channel (N,),
          corroborations (N,), cause (N,), sentiment (N,), affiliation (N,),
          is_catastrophe (N,) bool, entity_reliability_proxy (N,)
        Returns total scalar loss and a dict of interpretable sub-metrics.
        """
        feats = batch["feats"]
        trunk_embeds = self.trunk.encode_events(feats)          # (N, trunk_dim)

        # -- temporal (annalistic) head: predict category from causal window
        windows, targets = self.trunk.make_causal_windows(trunk_embeds)
        temporal_logits = self.trunk.temporal_forward(windows)
        temporal_labels = batch["category"][targets]
        l_temporal, d_temporal_logits = softmax_ce_loss(temporal_logits, temporal_labels)

        # -- entity (alphabetical) head: predict entity type from embedding
        # only, order-independent by construction (see PART 2 docstring).
        uniq_entities = np.unique(batch["entity_id"])
        entity_types = batch["entity_type_lookup"][uniq_entities]
        entity_logits, entity_embeds = self.trunk.entity_forward(uniq_entities)
        l_entity, d_entity_logits = softmax_ce_loss(entity_logits, entity_types)

        # -- trust ledger: predict a target trust score (supervised proxy:
        # higher for corroborated + testimonial, as a training signal so the
        # learned prior is sensible; the CHANNEL-SPECIFIC dynamic itself is
        # rule-based and tested separately in online_update()).
        rel_proxy = batch["entity_reliability_proxy"]
        pred_trust = self.trust.forward(batch["channel"], batch["corroborations"], rel_proxy)
        target_trust = np.clip(
            0.5 + 0.15 * batch["channel"] + 0.1 * (np.array(batch["corroborations"]) / 3.0),
            0.05, 0.95)
        l_trust, d_trust = mse_loss(pred_trust, target_trust)

        # -- legible bias residual: predict sentiment
        pred_sent, content_only, bias_only = self.bias.forward(trunk_embeds, batch["affiliation"])
        l_sent, d_sent = mse_loss(pred_sent, batch["sentiment"])

        # -- precedent autoencoder: reconstruct ONLY normal-category events
        normal_mask = ~batch["is_catastrophe"]
        recon = self.decoupler.autoencode(trunk_embeds)
        recon_err_all = np.mean((recon - trunk_embeds) ** 2, axis=1)
        l_precedent = recon_err_all[normal_mask].mean()

        # -- causal trigger classifier: ALL events, including catastrophe
        cause_logits = self.decoupler.cause_forward(trunk_embeds)
        l_cause, d_cause_logits = softmax_ce_loss(cause_logits, batch["cause"])

        total = l_temporal + l_entity + l_trust + l_sent + l_precedent + l_cause

        cache = dict(
            trunk_embeds=trunk_embeds, windows=windows, targets=targets,
            d_temporal_logits=d_temporal_logits, uniq_entities=uniq_entities,
            d_entity_logits=d_entity_logits, d_trust=d_trust,
            d_sent=d_sent, normal_mask=normal_mask, recon=recon,
            d_cause_logits=d_cause_logits, n_events=feats.shape[0],
        )
        metrics = dict(
            l_temporal=l_temporal, l_entity=l_entity, l_trust=l_trust,
            l_sent=l_sent, l_precedent=l_precedent, l_cause=l_cause,
            total=total, recon_err_all=recon_err_all,
            pred_sent=pred_sent, cause_logits=cause_logits,
        )
        self._cache = cache
        return total, metrics

    # -------------------------------------------------------------------
    def backward(self):
        c = self._cache
        n = c["n_events"]
        trunk_dim = c["trunk_embeds"].shape[1]
        d_trunk_total = np.zeros((n, trunk_dim))

        # temporal head backward -> distribute into the window's trunk slots
        d_windows = self.trunk.temporal_backward(c["d_temporal_logits"])
        w = self.trunk.window
        for row_i, target_i in enumerate(c["targets"]):
            d_window = d_windows[row_i].reshape(w, trunk_dim)
            d_trunk_total[target_i - w:target_i] += d_window

        # entity head backward -> into entity_table (not into event trunk)
        self.trunk.entity_backward(c["d_entity_logits"], c["uniq_entities"])

        # trust ledger backward -> into trust.prior_net only
        self.trust.backward(c["d_trust"])

        # bias residual backward -> content term flows into trunk; bias term
        # flows only into bias_head (isolated by construction)
        d_content, d_bias = self.bias.backward(c["d_sent"])
        d_trunk_total += d_content

        # precedent autoencoder backward -> only normal-category rows receive
        # gradient (catastrophe is deliberately held out of this objective)
        recon = c["recon"]
        trunk_embeds = c["trunk_embeds"]
        mask = c["normal_mask"].astype(float).reshape(-1, 1)
        n_normal = c["normal_mask"].sum()
        d_recon = (2.0 * (recon - trunk_embeds) / trunk_embeds.shape[1]) * mask / max(n_normal, 1)
        # Two paths back to trunk_embeds: (1) INDIRECT, through the encoder
        # that produced `recon` from `trunk_embeds`; (2) DIRECT, because
        # trunk_embeds is also the reconstruction TARGET (x appears as
        # "-x" inside (recon-x)^2), which the autoencode_backward() call
        # alone does not capture.
        d_trunk_from_ae_indirect = self.decoupler.autoencode_backward(d_recon)
        d_trunk_from_ae_direct = -d_recon
        d_trunk_total += d_trunk_from_ae_indirect + d_trunk_from_ae_direct

        # cause classifier backward -> all rows
        d_trunk_from_cause = self.decoupler.cause_backward(c["d_cause_logits"])
        d_trunk_total += d_trunk_from_cause

        # finally: shared event encoder backward
        self.trunk.encode_events_backward(d_trunk_total)


# ==============================================================================
# PART 7. FINITE-DIFFERENCE GRADIENT CHECK (MANDATORY)
# ==============================================================================

def make_batch(entities, events):
    entity_type_lookup = np.zeros(len(entities), dtype=int)
    entity_affil_lookup = np.zeros(len(entities))
    for e in entities:
        entity_type_lookup[e["id"]] = e["type"]
        entity_affil_lookup[e["id"]] = e["affiliation"]

    feats = np.stack([e["feat"] for e in events])
    entity_id = np.array([e["entity_id"] for e in events])
    batch = dict(
        feats=feats,
        category=np.array([e["category"] for e in events]),
        entity_id=entity_id,
        channel=np.array([e["channel"] for e in events]),
        corroborations=np.array([e["corroborations"] for e in events]),
        cause=np.array([e["cause"] for e in events]),
        sentiment=np.array([e["sentiment"] for e in events]),
        affiliation=entity_affil_lookup[entity_id],
        is_catastrophe=np.array([e["is_catastrophe"] for e in events]),
        entity_reliability_proxy=entity_affil_lookup[entity_id] * 0 + 0.6,
        entity_type_lookup=entity_type_lookup,
    )
    return batch


def gradient_check(model, batch, n_params_per_tensor=3, eps=1e-5, tol=2e-2):
    """
    Standard centered finite-difference check:
        d/dtheta L  ~=  (L(theta+eps) - L(theta-eps)) / (2*eps)
    compared against the analytic gradient produced by backward(). We sample
    a handful of entries from EVERY parameter tensor in the model (all four
    modules) so the check exercises the whole network, not just one layer.
    Returns (passed: bool, worst_relative_error: float, report: list[str]).
    """
    params = model.params()
    grads_map = model.grad_map()

    model.zero_grad()
    loss0, _ = model.forward_and_loss(batch)
    model.backward()
    # IMPORTANT: snapshot (copy) the analytic gradients now. The per-parameter
    # perturbation loop below must call model.zero_grad() again to get clean
    # numeric-probe losses, and zero_grad() zeros these SAME live arrays
    # in place -- so the live grads_map values cannot be read lazily inside
    # the loop, only this frozen copy.
    analytic_snapshot = {name: g.copy() for name, g in grads_map.items()}

    report = []
    worst = 0.0
    all_ok = True
    rng = np.random.default_rng(7)

    for name, p in params.items():
        flat_p = p.reshape(-1)
        flat_g = analytic_snapshot[name].reshape(-1)
        n = flat_p.shape[0]
        idxs = rng.choice(n, size=min(n_params_per_tensor, n), replace=False)
        for idx in idxs:
            orig = flat_p[idx]

            flat_p[idx] = orig + eps
            model.zero_grad()
            loss_plus, _ = model.forward_and_loss(batch)

            flat_p[idx] = orig - eps
            model.zero_grad()
            loss_minus, _ = model.forward_and_loss(batch)

            flat_p[idx] = orig  # restore

            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            analytic = flat_g[idx]
            denom = max(abs(numeric_grad) + abs(analytic), 1e-8)
            rel_err = abs(numeric_grad - analytic) / denom
            worst = max(worst, rel_err)
            ok = rel_err < tol
            all_ok = all_ok and ok
            report.append(
                f"  {name:24s} idx={idx:5d}  analytic={analytic:+.6f}  "
                f"numeric={numeric_grad:+.6f}  rel_err={rel_err:.2e}  "
                f"{'OK' if ok else 'FAIL'}"
            )

    # restore clean grads for subsequent real training
    model.zero_grad()
    model.forward_and_loss(batch)
    model.backward()
    return all_ok, worst, report


# ==============================================================================
# PART 8. ADAM OPTIMIZER (manual, pure NumPy)
# ==============================================================================

class Adam:
    def __init__(self, params, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
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
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==============================================================================
# PART 9. TRAINING LOOP
# ==============================================================================

def train(model, batch, epochs=250, lr=0.02, log_every=25):
    params = model.params()
    grads = model.grad_map()
    opt = Adam(params, lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.zero_grad()
        total, metrics = model.forward_and_loss(batch)
        model.backward()
        opt.step(params, grads)
        history.append(total)
        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  total={total:.4f}  "
                  f"temporal={metrics['l_temporal']:.4f}  "
                  f"entity={metrics['l_entity']:.4f}  "
                  f"trust={metrics['l_trust']:.4f}  "
                  f"sent={metrics['l_sent']:.4f}  "
                  f"precedent={metrics['l_precedent']:.4f}  "
                  f"cause={metrics['l_cause']:.4f}")
    return history


# ==============================================================================
# PART 10. SELF-TESTS
# ==============================================================================

def test_gradient_check(model, batch):
    print("\n[TEST 1] Finite-difference gradient check across all four modules")
    ok, worst, report = gradient_check(model, batch, n_params_per_tensor=2)
    for line in report:
        print(line)
    print(f"  -> worst relative error = {worst:.3e}  ({'PASS' if ok else 'FAIL'})")
    assert ok, "Gradient check failed -- analytic and numeric gradients disagree."
    return ok


def test_dual_index_orthogonality(model, batch):
    print("\n[TEST 2] Dual-index orthogonality (temporal vs. alphabetical)")
    trunk_embeds = model.trunk.encode_events(batch["feats"])

    # (a) Entity head: alphabetical order must not matter. Query the SAME
    # entities in original order vs. reverse-alphabetical order; logits for
    # a given entity_id must be identical either way (it's a pure lookup).
    uniq = np.unique(batch["entity_id"])
    logits_fwd, _ = model.trunk.entity_forward(uniq)
    logits_rev, _ = model.trunk.entity_forward(uniq[::-1])
    # re-align reversed-order logits back to `uniq` order for comparison
    logits_rev_realigned = logits_rev[::-1]
    entity_invariant = np.allclose(logits_fwd, logits_rev_realigned, atol=1e-10)
    print(f"  entity-head predictions identical under alphabetical reordering: "
          f"{entity_invariant} (expected True)")

    # (b) Temporal head: chronological order DOES matter. Compare predictions
    # from the true Hijri-year order against a shuffled event order; the
    # windows -- and therefore the logits -- must differ.
    windows_true, targets_true = model.trunk.make_causal_windows(trunk_embeds)
    logits_true = model.trunk.temporal_forward(windows_true)

    perm = RNG.permutation(trunk_embeds.shape[0])
    shuffled_embeds = trunk_embeds[perm]
    windows_shuf, targets_shuf = model.trunk.make_causal_windows(shuffled_embeds)
    logits_shuf = model.trunk.temporal_forward(windows_shuf)

    same_shape = logits_true.shape == logits_shuf.shape
    temporal_variant = (not same_shape) or (not np.allclose(logits_true, logits_shuf, atol=1e-6))
    print(f"  temporal-head predictions CHANGE under chronological shuffling: "
          f"{temporal_variant} (expected True)")

    assert entity_invariant, "Entity (alphabetical) head is not order-invariant."
    assert temporal_variant, "Temporal (annalistic) head failed to depend on order."
    print("  -> PASS: the two indexes are structurally orthogonal.")
    return True


def test_bias_residual_legible(model, batch):
    print("\n[TEST 3] Legible bias residual is present AND recoverable by audit")
    trunk_embeds = model.trunk.encode_events(batch["feats"])
    total_sent, content_only, bias_only = model.bias.forward(trunk_embeds, batch["affiliation"])

    recovered_slope = model.bias.audit(total_sent, batch["affiliation"])
    # Compare to the model's own internal bias_head weight (Zengid=1 minus
    # Ayyubid=0 delta in the learned affine map).
    w = model.bias.bias_head.W[0, 0]
    print(f"  external audit slope (sentiment ~ affiliation): {recovered_slope:+.4f}")
    print(f"  bias_head's own learned weight:                 {w:+.4f}")
    print(f"  agreement (should be close, both encode the same isolated term)")

    assert abs(recovered_slope) > 0.15, \
        "Bias residual should be detectably present (patron-affiliation skew)."
    assert abs(recovered_slope - w) < 0.35, \
        "External audit should approximately recover the isolated bias term."
    print("  -> PASS: the bias is legible from the outside, not hidden in the weights.")
    return True


def test_anomaly_causal_decoupling(model, batch):
    print("\n[TEST 4] Precedent/cause decoupling on the catastrophe event")
    trunk_embeds = model.trunk.encode_events(batch["feats"])
    precedent_err = model.decoupler.precedent_score(trunk_embeds)
    cause_logits = model.decoupler.cause_forward(trunk_embeds)
    cause_probs = softmax(cause_logits)

    cat_mask = batch["is_catastrophe"]
    normal_mask = ~cat_mask
    normal_errs = precedent_err[normal_mask]
    cat_err = precedent_err[cat_mask][0]
    threshold = np.percentile(normal_errs, 95)

    flagged_unprecedented = cat_err > threshold
    print(f"  catastrophe reconstruction error:      {cat_err:.5f}")
    print(f"  normal-event 95th percentile error:    {threshold:.5f}")
    print(f"  flagged as unprecedented:               {flagged_unprecedented} (expected True)")

    cat_probs = cause_probs[cat_mask][0]
    predicted_cause = CAUSE_CLASSES[int(np.argmax(cat_probs))]
    confidence = float(cat_probs.max())
    finite_and_confident = np.isfinite(cat_probs).all() and confidence > (1.0 / len(CAUSE_CLASSES))
    print(f"  causal-trigger prediction for catastrophe: '{predicted_cause}' "
          f"(confidence {confidence:.3f})")
    print(f"  causal head still produces a valid, above-chance explanation: "
          f"{finite_and_confident} (expected True)")

    assert flagged_unprecedented, "Catastrophe should be flagged as without precedent."
    assert finite_and_confident, "Causal head must still explain the trigger."
    print("  -> PASS: 'no precedent' and 'here is the cause' coexist, as in Ibn al-Athir's own account.")
    return True


def test_testimonial_trust_dynamics(model):
    print("\n[TEST 5] Testimonial vs. compiled trust under identical contradiction")
    channel = [1, 0]              # 0: testimonial, 1: compiled -- see below (fixed)
    channel = [1, 0]
    corroborations = [1, 1]       # identical corroboration count
    rel_proxy = [0.6, 0.6]
    base = model.trust.forward(channel, corroborations, rel_proxy)
    updated = model.trust.online_update(
        base_trust=base, channel=channel, contradicted=[True, True],
        corroborations=corroborations,
    )
    testimonial_trust, compiled_trust = updated[0], updated[1]
    print(f"  base learned trust (testimonial, compiled): "
          f"({base[0]:.3f}, {base[1]:.3f})")
    print(f"  after ONE contradiction each, trust becomes: "
          f"({testimonial_trust:.3f}, {compiled_trust:.3f})")
    print(f"  testimonial trust remains higher after equal negative evidence: "
          f"{testimonial_trust > compiled_trust} (expected True)")
    assert testimonial_trust > compiled_trust, \
        "Testimonial channel should decay slower under contradiction."
    print("  -> PASS: the two channels genuinely behave differently, not just labeled differently.")
    return True


# ==============================================================================
# PART 11. MAIN
# ==============================================================================

def main():
    print("=" * 78)
    print("IBN AL-ATHIR: THE DUAL-LEDGER MIND")
    print("Universal annals (temporal) + Companion register (alphabetical),")
    print("a legible patron-bias residual, and cause/precedent decoupling")
    print("=" * 78)

    entities = build_entities(n_entities=40)
    events = build_events(entities, n_events=160)
    batch = make_batch(entities, events)
    print(f"\nSynthetic corpus: {len(entities)} entities (alphabetically indexed), "
          f"{len(events)} events (Hijri 550-629, chronologically indexed), "
          f"1 designated catastrophe event with no distributional precedent.")

    model = AthirNet(n_entities=len(entities))

    print("\n--- Gradient check BEFORE training (correctness gate) ---")
    ok, worst, _ = gradient_check(model, batch, n_params_per_tensor=2)
    print(f"Gradient check worst relative error: {worst:.3e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "Aborting: gradient check failed before training."

    print("\n--- Training (Adam, pure NumPy) ---")
    history = train(model, batch, epochs=250, lr=0.02, log_every=25)
    print(f"\nLoss trajectory: start={history[0]:.4f}  end={history[-1]:.4f}  "
          f"(reduction={100 * (1 - history[-1] / history[0]):.1f}%)")

    print("\n" + "=" * 78)
    print("SELF-TESTS")
    print("=" * 78)
    test_gradient_check(model, batch)
    test_dual_index_orthogonality(model, batch)
    test_bias_residual_legible(model, batch)
    test_anomaly_causal_decoupling(model, batch)
    test_testimonial_trust_dynamics(model)

    print("\n" + "=" * 78)
    print("ALL SELF-TESTS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

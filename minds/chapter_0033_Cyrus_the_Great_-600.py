#!/usr/bin/env python3
# =============================================================================
#  chapter_0033_cyrus_the_great_-600.py
#  Mind #33  —  CYRUS THE GREAT  (c. 600 – 530 BCE), founder of the Achaemenid
#  Empire.
# # Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 2 Minds 21 - 40 Available on Amazon https://www.amazon.com/dp/B0H6QCQ9M7
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · Chapter 0033 · Cyrus the Great
#  ARCHITECTURE:  The Federated Legitimation Network  (a.k.a. the "Satrapy Net")
#
#  --------------------------------------------------------------------------
#  WHY THIS ARCHITECTURE, AND WHY CYRUS
#  --------------------------------------------------------------------------
#  The cliché about Cyrus ("tolerant king, first human-rights charter") is a
#  modern projection that current scholarship rejects: the Cyrus Cylinder is a
#  conventional Babylonian foundation inscription, written in *Marduk's* idiom,
#  whose job was to legitimate a foreign conqueror to the conquered. Stripped of
#  the romance, the surviving evidence points to a very specific COGNITIVE move
#  that is Cyrus's own:
#
#      He governed a maximally heterogeneous world by PRESERVING each conquered
#      people's local institutions, gods and self-understanding, and by routing
#      a single thin imperial intent THROUGH those local systems, re-expressed
#      in each one's native idiom — never overwriting them into one template.
#
#  To Babylon he was Marduk's chosen restorer; to Judah, YHWH's "anointed"
#  (Isaiah 45:1); his capital Pasargadae fused Assyrian relief, Ionian masonry
#  and Elamite/Urartian form. The integrating "Persian" layer (satrapies, royal
#  roads, tribute, a shared security umbrella, the "King's Eyes") was kept
#  DELIBERATELY THIN over autonomous local sub-systems whose priors were left
#  intact. This is the exact opposite of the Assyrian template of deportation
#  and cultural homogenisation — and it maps cleanly onto a precise machine
#  learning mechanism:
#
#      * FROZEN LOCAL EXPERTS (provinces).  Each domain keeps its own fixed
#        representational basis. Conquest does not retrain the conquered.
#      * THIN LEGITIMATION ADAPTERS.  The ONLY province-specific learned
#        parameters are low-rank maps that TRANSLATE a shared imperial intent
#        vector into each province's local space (modulate, don't overwrite).
#      * A SPARSE ROUTER ("King's Eyes").  A light shared layer that recognises
#        which province an input belongs to and speaks to it in its own tongue.
#      * A continual-learning guarantee: adding a province cannot destroy the
#        governance of an older one, because old provinces are never touched.
#
#  THE HEADLINE CLAIM (and the headline self-test):  a monolithic network of
#  comparable capacity, trained province-by-province, CATASTROPHICALLY FORGETS
#  earlier provinces (the Assyrian failure). The Federated Legitimation Network,
#  trained province-by-province, RETAINS them all (the Cyrian success) — because
#  legitimacy is learned per-domain and never imposed globally.
#
#  Honest provenance note: no first-person philosophy of mind by Cyrus survives.
#  This architecture is EXTRAPOLATED from his documented governance logic and
#  from mediated/legendary sources (Herodotus, Xenophon's idealised Cyropaedia,
#  the Babylonian-genre Cylinder, the Hebrew Bible). It embodies the governance
#  pattern we can reconstruct, not a verified inner cognition.
#
#  Engineering conventions for this corpus:
#    - pure NumPy, from scratch (no autograd, no ML frameworks)
#    - a finite-difference gradient check that MUST pass (included, runs first)
#    - a real training loop and self-tests
#    - executable end-to-end; verified stdout is pasted into the chapter
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# 0.  Small numerical helpers
# -----------------------------------------------------------------------------

def relu(x):
    return np.maximum(0.0, x)

def drelu(x):
    # derivative of relu w.r.t. its pre-activation
    return (x > 0.0).astype(x.dtype)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))

def one_hot(idx, n):
    o = np.zeros((idx.shape[0], n))
    o[np.arange(idx.shape[0]), idx] = 1.0
    return o


# =============================================================================
# 1.  THE FEDERATED LEGITIMATION NETWORK
# =============================================================================
class FederatedLegitimationNetwork:
    """
    A "Satrapy Net": one thin imperial intent layer + a sparse King's-Eyes
    router + D autonomous provinces. Each province has a FROZEN local expert
    (its own representational basis), a learned low-rank LEGITIMATION adapter
    that translates the shared intent into the province's space, and its own
    output head. Nothing about a province is overwritten when a new one is added.

    Trainable parameters:
        W_enc, b_enc       imperial intent encoder        (shared, thin)
        R,     b_R         King's-Eyes router             (shared, thin)
        Ldn[p]             legitimation down-projection    (per province)
        Lup[p]             legitimation up-projection      (per province)
        Hw[p],  Hb[p]      province output head            (per province)

    Frozen (never updated) parameters — the conquered keep their institutions:
        E[p],  bE[p]       local expert random feature basis (per province)
    """

    def __init__(self, n_in, n_intent, n_prov, n_expert, rank, n_class, seed=0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_intent = n_in, n_intent
        self.n_prov, self.n_expert = n_prov, n_expert
        self.rank, self.n_class = rank, n_class

        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / b)

        # ---- shared, thin imperial layers -----------------------------------
        self.W_enc = s(n_intent, n_in)
        self.b_enc = np.zeros(n_intent)
        self.R     = s(n_prov, n_intent)          # router logits over provinces
        self.b_R   = np.zeros(n_prov)

        # ---- per-province FROZEN experts (each its own basis) ----------------
        # Distinct seed per province => genuinely different local substrate.
        self.E, self.bE = [], []
        for p in range(n_prov):
            rp = np.random.default_rng(1000 + p + 7 * seed)
            self.E.append(rp.standard_normal((n_expert, n_in)) * np.sqrt(2.0 / n_in))
            self.bE.append(rp.standard_normal(n_expert) * 0.1)

        # ---- per-province LEARNED legitimation adapters + heads --------------
        self.Ldn, self.Lup, self.Hw, self.Hb = [], [], [], []
        for p in range(n_prov):
            self.Ldn.append(s(rank, n_intent))
            self.Lup.append(s(n_expert, rank) * 0.1)   # start near "no modulation"
            self.Hw.append(s(n_class, n_expert))
            self.Hb.append(np.zeros(n_class))

    # -- which names are trainable, and a per-province grouping for continual --
    def shared_param_names(self):
        return ["W_enc", "b_enc", "R", "b_R"]

    def province_param_names(self, p):
        return [f"Ldn[{p}]", f"Lup[{p}]", f"Hw[{p}]", f"Hb[{p}]"]

    def trainable_param_names(self):
        names = list(self.shared_param_names())
        for p in range(self.n_prov):
            names += self.province_param_names(p)
        return names

    # generic getter/setter so the gradient check can perturb any parameter
    def _get(self, name):
        if "[" in name:
            base, idx = name[:-1].split("[")
            return getattr(self, base)[int(idx)]
        return getattr(self, name)

    def _set(self, name, value):
        if "[" in name:
            base, idx = name[:-1].split("[")
            getattr(self, base)[int(idx)] = value
        else:
            setattr(self, name, value)

    # -------------------------------------------------------------------------
    # FORWARD.  Caches everything backward() needs.
    #   route_mode:
    #     'soft' : mix province logits by router gates (default; differentiable)
    #     'true' : force the gate onto the given true-domain (teacher routing)
    #     'hard' : argmax routing at inference (King's Eyes decide alone)
    # -------------------------------------------------------------------------
    def forward(self, X, domain=None, route_mode="soft"):
        B = X.shape[0]
        c = {}                                   # cache
        Z = np.tanh(X @ self.W_enc.T + self.b_enc)            # B x Hz
        Rlogits = Z @ self.R.T + self.b_R                     # B x D
        G = softmax(Rlogits, axis=1)                          # B x D gates

        if route_mode == "true":
            assert domain is not None
            gate_used = one_hot(domain, self.n_prov)
        elif route_mode == "hard":
            gate_used = one_hot(np.argmax(Rlogits, axis=1), self.n_prov)
        else:
            gate_used = G

        prov = []
        L = np.zeros((B, self.n_class))
        for p in range(self.n_prov):
            Hpre = X @ self.E[p].T + self.bE[p]              # B x H (frozen lin)
            Hp = relu(Hpre)                                  # frozen local feat
            Apre = Z @ self.Ldn[p].T                         # B x rank
            Ap = relu(Apre)
            Mp = Ap @ self.Lup[p].T                          # B x H  modulation
            Rp = Hp * (1.0 + Mp)                             # legitimated feat
            Lp = Rp @ self.Hw[p].T + self.Hb[p]              # B x C province logit
            prov.append(dict(Hpre=Hpre, Hp=Hp, Apre=Apre, Ap=Ap, Mp=Mp, Rp=Rp, Lp=Lp))
            L += gate_used[:, p:p+1] * Lp                    # imperial mixture

        c.update(X=X, Z=Z, Rlogits=Rlogits, G=G, gate_used=gate_used,
                 prov=prov, L=L, route_mode=route_mode, domain=domain, B=B)
        return L, c

    # -------------------------------------------------------------------------
    # LOSS.  cross-entropy on the imperial mixture + (optional) router
    # supervision toward the true province ("recognise the province, speak its
    # tongue") + tiny L2. Returns scalar loss and a fresh cache.
    # -------------------------------------------------------------------------
    def loss(self, X, y, domain, lam_route=0.3, lam_l2=1e-4, route_mode="soft"):
        L, c = self.forward(X, domain=domain, route_mode=route_mode)
        B = c["B"]
        logp = log_softmax(L, axis=1)
        ce = -np.mean(logp[np.arange(B), y])

        # router supervision (only meaningful when routing is learned)
        rlogp = log_softmax(c["Rlogits"], axis=1)
        route_ce = -np.mean(rlogp[np.arange(B), domain]) if lam_route > 0 else 0.0

        # tiny weight decay on the learned params keeps the check well-posed
        l2 = 0.0
        for nm in self.trainable_param_names():
            l2 += np.sum(self._get(nm) ** 2)
        l2 *= lam_l2

        total = ce + lam_route * route_ce + l2
        c.update(y=y, logp=logp, ce=ce, route_ce=route_ce,
                 lam_route=lam_route, lam_l2=lam_l2)
        return total, c

    # -------------------------------------------------------------------------
    # BACKWARD.  Analytic gradients for every trainable parameter.
    # Experts E,bE are frozen and receive NO gradient (by construction).
    # -------------------------------------------------------------------------
    def backward(self, c):
        X, Z, G = c["X"], c["Z"], c["G"]
        gate_used, prov = c["gate_used"], c["prov"]
        y, B = c["y"], c["B"]
        lam_route, lam_l2 = c["lam_route"], c["lam_l2"]

        grads = {nm: np.zeros_like(self._get(nm)) for nm in self.trainable_param_names()}

        # dLoss/dL  (softmax cross-entropy on the mixture)
        P = np.exp(c["logp"])                       # B x C
        dL = (P - one_hot(y, self.n_class)) / B      # B x C

        # accumulate gradient flowing back into Z from each province + router
        dZ = np.zeros_like(Z)
        # gate gradients (only when soft routing actually used the gates)
        soft = (c["route_mode"] == "soft")
        dgate = np.zeros_like(gate_used) if soft else None

        for p in range(self.n_prov):
            pr = prov[p]
            gp = gate_used[:, p:p+1]                  # B x 1
            Lp = pr["Lp"]

            # mixture L = sum_p gate_p * Lp
            dLp = dL * gp                             # B x C  (into province logits)
            if soft:
                dgate[:, p] = np.sum(dL * Lp, axis=1)  # B  (into gate_p)

            # head:  Lp = Rp @ Hw_p.T + Hb_p
            Rp = pr["Rp"]
            grads[f"Hw[{p}]"] += dLp.T @ Rp
            grads[f"Hb[{p}]"] += np.sum(dLp, axis=0)
            dRp = dLp @ self.Hw[p]                    # B x H

            # Rp = Hp * (1 + Mp)
            Hp, Mp = pr["Hp"], pr["Mp"]
            dHp_unused = dRp * (1.0 + Mp)             # Hp is frozen-derived; no param here
            dMp = dRp * Hp                            # B x H

            # Mp = Ap @ Lup_p.T
            Ap = pr["Ap"]
            grads[f"Lup[{p}]"] += dMp.T @ Ap          # H x rank
            dAp = dMp @ self.Lup[p]                   # B x rank

            # Ap = relu(Apre);  Apre = Z @ Ldn_p.T
            dApre = dAp * drelu(pr["Apre"])           # B x rank
            grads[f"Ldn[{p}]"] += dApre.T @ Z         # rank x Hz
            dZ += dApre @ self.Ldn[p]                 # B x Hz  (intent feels each province)

        # ---- router branch ---------------------------------------------------
        # primary task gradient into router gates (soft routing) ...
        dRlogits = np.zeros_like(c["Rlogits"])
        if soft:
            # G = softmax(Rlogits); gate_used == G here
            # dRlogits from dgate via softmax jacobian, per row
            for i in range(B):
                g = G[i]
                jac = np.diag(g) - np.outer(g, g)    # D x D
                dRlogits[i] += jac @ dgate[i]
        # ... plus router supervision term
        if lam_route > 0:
            domain = c["domain"]
            Pr = softmax(c["Rlogits"], axis=1)
            dRlogits += lam_route * (Pr - one_hot(domain, self.n_prov)) / B

        grads["R"]   += dRlogits.T @ Z
        grads["b_R"] += np.sum(dRlogits, axis=0)
        dZ += dRlogits @ self.R

        # ---- intent encoder:  Z = tanh(X @ W_enc.T + b_enc) ------------------
        dZpre = dZ * (1.0 - Z ** 2)                   # tanh'
        grads["W_enc"] += dZpre.T @ X
        grads["b_enc"] += np.sum(dZpre, axis=0)

        # ---- weight decay ----------------------------------------------------
        for nm in self.trainable_param_names():
            grads[nm] += 2.0 * lam_l2 * self._get(nm)

        return grads

    # convenience: predict labels (and inferred province) with hard routing
    def predict(self, X, route_mode="hard"):
        L, c = self.forward(X, route_mode=route_mode)
        return np.argmax(L, axis=1), np.argmax(c["Rlogits"], axis=1)


# =============================================================================
# 2.  A MONOLITHIC BASELINE  (the "Assyrian" overwriter)
#     One shared MLP, ALL weights trainable. It has no provinces; learning a new
#     domain rewrites the same weights -> catastrophic forgetting.
# =============================================================================
class MonolithMLP:
    def __init__(self, n_in, n_hidden, n_class, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / b)
        self.W1, self.b1 = s(n_hidden, n_in), np.zeros(n_hidden)
        self.W2, self.b2 = s(n_class, n_hidden), np.zeros(n_class)

    def forward(self, X):
        z1 = X @ self.W1.T + self.b1
        a1 = relu(z1)
        L = a1 @ self.W2.T + self.b2
        return L, (X, z1, a1)

    def loss_and_grad(self, X, y, n_class, lam_l2=1e-4):
        L, (X, z1, a1) = self.forward(X)
        B = X.shape[0]
        logp = log_softmax(L, axis=1)
        ce = -np.mean(logp[np.arange(B), y])
        P = np.exp(logp)
        dL = (P - one_hot(y, n_class)) / B
        gW2 = dL.T @ a1 + 2 * lam_l2 * self.W2
        gb2 = np.sum(dL, axis=0)
        da1 = dL @ self.W2
        dz1 = da1 * drelu(z1)
        gW1 = dz1.T @ X + 2 * lam_l2 * self.W1
        gb1 = np.sum(dz1, axis=0)
        return ce, dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2)

    def sgd_step(self, g, lr):
        self.W1 -= lr * g["W1"]; self.b1 -= lr * g["b1"]
        self.W2 -= lr * g["W2"]; self.b2 -= lr * g["b2"]

    def predict(self, X):
        L, _ = self.forward(X)
        return np.argmax(L, axis=1)


# =============================================================================
# 3.  SYNTHETIC "EMPIRE" DATA
#     D provinces share one input space R^n but each has its OWN class geometry
#     (its own idiom/law). A single shared classifier must compromise across
#     provinces (interference); a federation need not.
# =============================================================================
def make_empire(n_prov=4, n_class=3, ctx_dim=4, content_dim=8, per_class=140, seed=0):
    """
    The crux of the demonstration is CONFLICT, not separability.

    Every input is  x = [ context | content ].
      * CONTENT prototypes are SHARED across all provinces: a content cluster j
        looks identical no matter which province you are in (the same "facts").
      * Each province assigns a DIFFERENT LABEL to the same content cluster via a
        province-specific permutation  perm_p  (the same fact means different
        things under different local law -- a different idiom).
      * CONTEXT is a province-identifying block (a distinct, well-separated
        cluster per province) -- the observable "which province am I in".

    Consequence: a single shared classifier trained province-by-province must
    OVERWRITE the content->label mapping each time (catastrophic forgetting),
    because content clusters conflict across provinces. A federation gives each
    province its own head, so the conflict never arises. The router can still
    recover the province from the context block alone.
    """
    rng = np.random.default_rng(seed)
    n_in = ctx_dim + content_dim

    # shared content prototypes (the same facts everywhere)
    content_proto = rng.standard_normal((n_class, content_dim)) * 1.8
    # province context centres, pushed far apart so the King's Eyes can tell them
    ctx_centre = rng.standard_normal((n_prov, ctx_dim))
    ctx_centre = ctx_centre / (np.linalg.norm(ctx_centre, axis=1, keepdims=True) + 1e-8)
    ctx_centre *= 4.0
    # province label permutations (cyclic shift by p => guaranteed conflict)
    perms = [np.roll(np.arange(n_class), p) for p in range(n_prov)]

    Xs, ys, ds = [], [], []
    for p in range(n_prov):
        for j in range(n_class):                     # j = content cluster
            ctx = ctx_centre[p] + rng.standard_normal((per_class, ctx_dim)) * 0.30
            con = content_proto[j] + rng.standard_normal((per_class, content_dim)) * 0.40
            Xs.append(np.hstack([ctx, con]))
            ys.append(np.full(per_class, perms[p][j]))   # SAME content, diff label
            ds.append(np.full(per_class, p))
    X = np.vstack(Xs); y = np.concatenate(ys); d = np.concatenate(ds)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx], d[idx], dict(content=content_proto, ctx=ctx_centre, perms=perms)


def split(X, y, d, frac=0.8, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    k = int(frac * len(idx))
    tr, te = idx[:k], idx[k:]
    return (X[tr], y[tr], d[tr]), (X[te], y[te], d[te])


# =============================================================================
# 4.  GRADIENT CHECK  (mandatory; must pass)
# =============================================================================
def gradient_check(verbose=True):
    rng = np.random.default_rng(7)
    n_in, n_intent, n_prov, n_expert, rank, n_class = 6, 5, 3, 7, 2, 4
    net = FederatedLegitimationNetwork(n_in, n_intent, n_prov, n_expert, rank,
                                       n_class, seed=3)
    B = 8
    X = rng.standard_normal((B, n_in))
    y = rng.integers(0, n_class, size=B)
    dom = rng.integers(0, n_prov, size=B)

    _, c = net.loss(X, y, dom, route_mode="soft")
    grads = net.backward(c)

    eps = 1e-6
    max_rel, worst = 0.0, None
    for nm in net.trainable_param_names():
        Wp = net._get(nm)
        flat = Wp.ravel()
        n_probe = min(flat.size, 12)
        probe = np.random.default_rng(hash(nm) % (2**31)).choice(flat.size,
                                                                 n_probe, replace=False)
        ga = grads[nm].ravel()
        for k in probe:
            orig = flat[k]
            flat[k] = orig + eps
            lp, _ = net.loss(X, y, dom, route_mode="soft")
            flat[k] = orig - eps
            lm, _ = net.loss(X, y, dom, route_mode="soft")
            flat[k] = orig
            num = (lp - lm) / (2 * eps)
            ana = ga[k]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            if rel > max_rel:
                max_rel, worst = rel, (nm, k, num, ana)
    if verbose:
        print(f"[grad check] max relative error = {max_rel:.3e}  "
              f"(worst: {worst[0]}, num={worst[2]:+.6f}, ana={worst[3]:+.6f})")
        print(f"[grad check] {'PASS' if max_rel < 1e-4 else 'FAIL'} "
              f"(threshold 1e-4)")
    return max_rel < 1e-4


# =============================================================================
# 5.  TRAINING LOOPS
# =============================================================================
def train_fln_joint(net, tr, n_class, epochs=60, lr=0.15, batch=128, seed=0):
    """Joint training on the whole empire at once (all provinces mixed)."""
    Xtr, ytr, dtr = tr
    rng = np.random.default_rng(seed)
    N = Xtr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            b = idx[s:s+batch]
            _, c = net.loss(Xtr[b], ytr[b], dtr[b], route_mode="soft")
            g = net.backward(c)
            for nm in net.trainable_param_names():
                net._set(nm, net._get(nm) - lr * g[nm])
    return net


def train_fln_province(net, X, y, d, p, epochs=40, lr=0.15, batch=128, seed=0,
                       route_mode="true", train_shared=True):
    """
    CONTINUAL training of ONE province p. Province-p parameters (its adapter and
    head) are eligible; ALL OTHER provinces' adapters/heads are frozen. This is
    the structural guarantee: conquering province p cannot overwrite province q,
    because q's parameters never receive a gradient here.

    route_mode='true' uses teacher/administrative routing (we know which province
    a sample comes from -- the King's Eyes report provenance), so only province
    p's head/adapter train. The router itself is still supervised toward the true
    province so it learns to recognise province p's context for later self-routing.
    train_shared lets us freeze the thin imperial layer after bootstrap.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    train_names = set(net.province_param_names(p))
    if train_shared:
        train_names |= set(net.shared_param_names())
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, batch):
            b = idx[s:s+batch]
            _, c = net.loss(X[b], y[b], d[b], route_mode=route_mode)
            g = net.backward(c)
            for nm in train_names:
                net._set(nm, net._get(nm) - lr * g[nm])
    return net


def accuracy_per_province(net, te, n_prov, route_mode="hard"):
    Xte, yte, dte = te
    out = {}
    for p in range(n_prov):
        m = (dte == p)
        if m.sum() == 0:
            out[p] = float("nan"); continue
        pred, _ = net.predict(Xte[m], route_mode=route_mode)
        out[p] = float(np.mean(pred == yte[m]))
    return out


# =============================================================================
# 6.  SELF-TESTS / DEMONSTRATIONS
# =============================================================================
def run_all():
    np.random.seed(0)
    print("=" * 74)
    print(" 33_Neuron.py  —  Cyrus the Great : Federated Legitimation Network")
    print("=" * 74)

    # ---- (A) gradient check ------------------------------------------------
    print("\n[A] Finite-difference gradient check")
    ok = gradient_check()
    assert ok, "gradient check FAILED"

    # ---- build the empire --------------------------------------------------
    n_prov, n_class = 4, 3
    X, y, d, meta = make_empire(n_prov=n_prov, n_class=n_class, ctx_dim=4,
                                content_dim=8, per_class=160, seed=0)
    n_in = X.shape[1]
    tr, te = split(X, y, d, frac=0.8, seed=1)
    print(f"\n    Empire: {n_prov} provinces, shared content prototypes, "
          f"conflicting per-province label permutations {meta['perms']}")

    # ---- (B) joint training: balanced mastery + emergent routing -----------
    print("\n[B] Joint imperial training (all provinces at once)")
    net = FederatedLegitimationNetwork(n_in, n_intent=16, n_prov=n_prov,
                                       n_expert=48, rank=4, n_class=n_class, seed=2)
    train_fln_joint(net, tr, n_class, epochs=60, lr=0.15, seed=4)
    accs = accuracy_per_province(net, te, n_prov, route_mode="hard")
    # routing accuracy: do the King's Eyes recognise the province unaided?
    _, prov_pred = net.predict(te[0], route_mode="hard")
    route_acc = float(np.mean(prov_pred == te[2]))
    print("    per-province test accuracy (hard self-routing):")
    for p in range(n_prov):
        print(f"      province {p}:  {accs[p]*100:5.1f} %")
    print(f"    mean accuracy           : {np.mean(list(accs.values()))*100:5.1f} %")
    print(f"    King's-Eyes routing acc : {route_acc*100:5.1f} %  "
          f"(province inferred from input alone)")

    # ---- (C) THE HEADLINE TEST: continual learning, province by province ----
    print("\n[C] Continual conquest  (provinces learned one after another)")
    print("    Same content clusters mean DIFFERENT labels per province, so a")
    print("    single network must overwrite itself to learn each new province.")
    print("    Question: after conquering province 3, is province 0 still governed?")

    # province-wise train/test splits
    prov_tr = [(tr[0][tr[2]==p], tr[1][tr[2]==p], tr[2][tr[2]==p]) for p in range(n_prov)]
    prov_te = [(te[0][te[2]==p], te[1][te[2]==p], te[2][te[2]==p]) for p in range(n_prov)]

    def fed_prov_acc(net, q):
        Xq, yq, dq = prov_te[q]
        L, _ = net.forward(Xq, domain=dq, route_mode="true")   # provenance known
        return float(np.mean(np.argmax(L, axis=1) == yq))

    # (C1) Cyrian federation: freeze old provinces, add the new one.
    #      Shared imperial layer is bootstrapped on province 0, then FROZEN; each
    #      later province only grows its own thin adapter + head.
    fed = FederatedLegitimationNetwork(n_in, n_intent=16, n_prov=n_prov,
                                       n_expert=48, rank=4, n_class=n_class, seed=5)
    fed_history = []
    for p in range(n_prov):
        train_fln_province(fed, *prov_tr[p][:3], p=p, epochs=60, lr=0.15,
                           seed=10+p, route_mode="true",
                           train_shared=(p == 0))   # imperial layer fixed after P0
        row = [ (fed_prov_acc(fed, q) if q <= p else np.nan) for q in range(n_prov) ]
        fed_history.append(row)

    # (C2) Assyrian monolith: same data order, but ONE set of weights.
    mono = MonolithMLP(n_in, n_hidden=64, n_class=n_class, seed=6)
    mono_history = []
    for p in range(n_prov):
        Xp, yp, _ = prov_tr[p]
        rng = np.random.default_rng(20+p)
        for ep in range(60):
            idx = rng.permutation(Xp.shape[0])
            for s in range(0, len(idx), 128):
                b = idx[s:s+128]
                _, g = mono.loss_and_grad(Xp[b], yp[b], n_class)
                mono.sgd_step(g, lr=0.15)
        row = [ (np.mean(mono.predict(prov_te[q][0])==prov_te[q][1])
                 if q <= p else np.nan) for q in range(n_prov) ]
        mono_history.append(row)

    def show(history, title):
        print(f"\n    {title}")
        print("      after learning ->     " +
              "".join([f" P{q}  " for q in range(n_prov)]))
        for p in range(n_prov):
            cells = "".join(
                [f"{history[p][q]*100:4.0f} " if not np.isnan(history[p][q])
                 else "  .  " for q in range(n_prov)])
            print(f"      ...province {p}:       {cells}")

    show(fed_history,  "Cyrian Federation  (frozen provinces + thin adapters):")
    show(mono_history, "Assyrian Monolith  (one set of weights, overwritten):")

    # retention metric: average accuracy on provinces 0..n-2 after final conquest
    fed_final0  = fed_history[-1][0]
    mono_final0 = mono_history[-1][0]
    fed_ret  = np.mean([fed_history[-1][q]  for q in range(n_prov-1)])
    mono_ret = np.mean([mono_history[-1][q] for q in range(n_prov-1)])
    print("\n    -- Retention after the final conquest (provinces 0..%d) --" % (n_prov-2))
    print(f"      Cyrian Federation : {fed_ret*100:5.1f} %   (province 0: {fed_final0*100:4.0f} %)")
    print(f"      Assyrian Monolith : {mono_ret*100:5.1f} %   (province 0: {mono_final0*100:4.0f} %)")
    print(f"      => federation retains {(fed_ret-mono_ret)*100:+.1f} pts more old "
          f"governance.")

    # ---- assertions that encode the thesis ---------------------------------
    assert ok
    assert np.mean(list(accs.values())) > 0.85, "joint mastery too low"
    assert fed_ret > mono_ret + 0.15, "federation should clearly out-retain monolith"
    assert fed_ret > 0.85, "federation should retain old provinces"
    print("\n[OK] All self-tests passed. The federation preserves the conquered;")
    print("     the monolith forgets them. This is Cyrus's mind, made runnable.")
    print("=" * 74)


if __name__ == "__main__":
    run_all()

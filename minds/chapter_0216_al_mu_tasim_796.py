#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Chapter 0216 - Al-Mu'tasim bi'llah
 Abu Ishaq Muhammad ibn Harun al-Rashid
 Khuld Palace, Baghdad, October 796  ->  Jawsaq Palace, Samarra, 5 January 842
 Eighth Abbasid caliph, r. 833-842

 ========================
# Part of the Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
# How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
# Tome 11 Minds 201 - 220 Available on Amazon https://www.amazon.com/dp/B0HKF7TRFF
# Resume and Interactive Demos at https://artificiology.com/
# Author: David Vivancos · chapter_0216_al_mu_tasim_796 - Al-Mu'tasim bi'llah (796-842 CE), Baghdad
================================================================================  

  S A M A R R A
  Severance-Aligned Modular Architecture with Regimental Routing
  and Attachment-monitoring

  Pure NumPy. No autograd. Every gradient derived by hand and verified against
  central finite differences before any training is allowed to run.

--------------------------------------------------------------------------------
 I.  WHY THIS MECHANISM AND NOT A TRANSFORMER
--------------------------------------------------------------------------------

 Al-Mu'tasim wrote nothing. He was, by the testimony of the sources themselves,
 the least lettered of the great Abbasids -- Bosworth's summary of the record is
 that the chroniclers stress his lack of culture beside his half-brother
 al-Ma'mun's questing mind. What he left instead of a doctrine is a *procedure*,
 and the procedure encodes a theory of mind with more precision than most
 treatises manage.

 The procedure had three moves.

   1. PROCUREMENT.  Capability is acquired, not grown. He bought boys from the
      Central Asian steppe and made them cavalry. He bought a household slave
      named Itakh, who had been a cook, and made him a general. Barely able to
      read, he retained al-Kindi -- the first man to write philosophy in Arabic
      -- who dedicated *On First Philosophy* to him and tutored his son Ahmad on
      the nature of the intellect. Every faculty in his court was imported.

   2. SEVERANCE.  An imported faculty is trustworthy only once its origin is
      removed. His guardsmen were cut from kin, homeland, tongue and name. In
      the cantonment of al-Karkh, granted to the Turkish general Ashinas, no
      outsiders were permitted to live and his followers were forbidden to mix
      with the Arab population; wives were purchased from the same homelands so
      that no *new* local attachment could form either. Loyalty was engineered
      by SUBTRACTION of rival allegiances, never by addition of commands.

   3. ENCLOSURE.  In 836 he built a city, Samarra, to hold the instrument, with
      each regiment quartered in its own cantonment -- al-Karkh and al-Dur for
      the Turks, al-Matira for al-Afshin's Ushrusaniyya -- so that every unit
      touched the caliph and not each other.

 Then the procedure failed, in the most instructive way available to us.
 Nineteen years after his death, his severed instruments -- possessing no kin,
 no homeland, no rival allegiance of any kind -- discovered that the only thing
 left to be loyal to was each other. In December 861 they killed the caliph
 al-Mutawakkil and opened the Anarchy at Samarra, making and unmaking four more
 caliphs in nine years. The perfectly aligned agent had become the principal.

 So this file implements that thesis as arithmetic, together with the empirical
 correction the historical record forces on it:

   ProvinceCorpus     data whose PROVENANCE predicts the label beautifully
                      during training and misleads completely afterwards.
                      This is the steppe, the slave market, and the year 861.

   Severance          a rank-r projection that deletes the provenance subspace
                      at INTAKE. Not something the network learns -- something
                      done to the input before it enters the city, exactly as
                      severance was a procedure applied at purchase and not a
                      lesson taught in barracks.

   Ghulam regiments   K independent tanh experts. Not weight-shared: they were
                      bought from different places and deliberately kept apart.

   CaliphalGate       a softmax router from input to regiments. The sovereign's
                      own discretion.

   Cantonment         an entropy floor on the mean routing distribution. No
                      single regiment may absorb the whole state.

   Two origin probes  auditors, not adversaries.
                        probe A reads the regiments  -- "does the guard still
                                                        know its homeland?"
                        probe B reads the routing    -- "does the SOVEREIGN
                                                        still choose by it?"

 The second probe is the argument of the chapter. It exists because the first
 experiment run on this architecture failed in a way that turned out to be
 historically exact: scrubbing the regiments while leaving the router reading
 raw provenance recovered only part of the loss. Al-Mu'tasim severed his
 guardsmen from their origins and never once severed his own criteria for
 choosing among them. He promoted Turks over Arabs on precisely the grounds he
 had spent a career making irrelevant inside each individual soldier. The
 provenance he expelled from the instrument walked back in through the hand
 that pointed it.

 Verified results are in section XI and are reproduced in the chapter text.

 Run:  python3 chapter_0216_al_mu_tasim_796.py
================================================================================
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==============================================================================
# II. NUMERICAL UTILITIES
# ==============================================================================


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean cross-entropy over a batch and its gradient w.r.t. the logits.

    logits  (N, C) ; targets (N,) integer classes
    returns (scalar, (N, C))
    """
    n = logits.shape[0]
    ls = log_softmax(logits, axis=1)
    loss = -float(np.mean(ls[np.arange(n), targets]))
    g = np.exp(ls)
    g[np.arange(n), targets] -= 1.0
    return loss, g / n


def entropy_of_mean(gate: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Shannon entropy (nats) of the BATCH-AVERAGED routing distribution, and its
    gradient w.r.t. the per-example gate matrix.

    This is the cantonment quantity. High entropy of the mean gate means
    patronage is spread across regiments. Entropy near zero means one regiment
    has absorbed the state.

        gbar_k = mean_i gate[i, k]
        H      = -sum_k gbar_k log gbar_k
        dH/dgate[i,k] = -(log gbar_k + 1) / N

    Note this is the entropy of the MEAN, not the mean of the entropies. The
    distinction matters: we want each decision to be decisive (one regiment
    handles it) while the population of decisions stays spread. Sharp
    specialists, no favourite.
    """
    n = gate.shape[0]
    gbar = np.mean(gate, axis=0)
    safe = np.clip(gbar, 1e-12, None)
    h = -float(np.sum(gbar * np.log(safe)))
    row = -(np.log(safe) + 1.0) / n
    return h, np.broadcast_to(row[None, :], gate.shape).copy()


# ==============================================================================
# III. THE CORPUS: PROVENANCE THAT LIES
# ==============================================================================


@dataclass
class ProvinceCorpus:
    """
    Data built to pose al-Mu'tasim's exact problem.

    Each example concatenates two blocks:

      CONTENT (n_content dims)
          What is actually true about the task. The label depends on a
          parity-like interaction between two content directions, so a linear
          model cannot shortcut it and a genuine hidden layer is required. It
          is also noisy, so learning it is real work.

      ORIGIN (n_origin dims)
          A clean, low-noise signature of which of n_provinces the example was
          recruited from. The homeland, the mother tongue, the tribal name.

    In TRAINING, province and label are yoked: province p emits label
    (p mod n_classes) with probability `spurious`. Provenance is therefore the
    cheapest accurate predictor in the dataset -- exactly the temptation that
    makes an institution trust a soldier for where he was born.

    At TEST time the yoke is broken and replaced by a different permutation.
    A model that learned to read provenance now reads it confidently wrong. A
    model from which provenance was removed is untouched.

    Calibration (measured, not asserted -- see self_tests):
      the province->label rule holds on ~93% of training rows and on ~3% of
      shifted rows. Chance is 1/3.
    """

    n_content: int = 12
    n_origin: int = 6
    n_provinces: int = 4
    n_classes: int = 3
    spurious: float = 0.90
    content_noise: float = 1.60
    origin_noise: float = 0.30
    seed: int = 20201

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.class_proto = rng.normal(0.0, 1.0, (self.n_classes, self.n_content))

        # Two orthonormal content directions used to build the nonlinear term.
        self.u = rng.normal(0.0, 1.0, self.n_content)
        self.u /= np.linalg.norm(self.u)
        self.v = rng.normal(0.0, 1.0, self.n_content)
        self.v -= self.v.dot(self.u) * self.u
        self.v /= np.linalg.norm(self.v)

        self.prov_proto = rng.normal(0.0, 1.2, (self.n_provinces, self.n_origin))

        self.map_train = np.array([p % self.n_classes
                                   for p in range(self.n_provinces)], dtype=int)
        self.map_test = np.array([(p + 1) % self.n_classes
                                  for p in range(self.n_provinces)], dtype=int)

    @property
    def n_features(self) -> int:
        return self.n_content + self.n_origin

    def sample(self, n: int, seed: int, shifted: bool
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, prov)."""
        rng = np.random.default_rng(seed)
        prov = rng.integers(0, self.n_provinces, size=n)
        table = self.map_test if shifted else self.map_train

        y = np.empty(n, dtype=int)
        yoked = rng.random(n) < self.spurious
        y[yoked] = table[prov[yoked]]
        y[~yoked] = rng.integers(0, self.n_classes, size=int(np.sum(~yoked)))

        content = (self.class_proto[y]
                   + rng.normal(0.0, self.content_noise, (n, self.n_content)))
        # Nonlinear parity term: separates class 0 from class 2 along v
        # according to the sign of the projection on u.
        sign = np.sign(content @ self.u)[:, None]
        polarity = ((y == 0).astype(float) - (y == 2).astype(float))[:, None]
        content = content + 0.9 * sign * self.v[None, :] * polarity

        origin = (self.prov_proto[prov]
                  + rng.normal(0.0, self.origin_noise, (n, self.n_origin)))
        return np.concatenate([content, origin], axis=1), y, prov


# ==============================================================================
# IV. THE SEVERANCE OPERATOR
# ==============================================================================


class Severance:
    """
    Deletes the provenance subspace from an input vector.

    THE ESTIMATION PROBLEM, WHICH IS THE WHOLE DIFFICULTY
    -----------------------------------------------------
    The naive way to find "the directions that name the homeland" is to take the
    province means and keep their top singular directions. This fails, and it
    fails for the reason that makes severance hard in every real system: because
    province and label are yoked, the province means are ALSO class means. Their
    leading directions carry the task signal. Projecting them out removes the
    homeland and the competence together. (Measured: such a basis puts only 71%
    of its energy in the provenance block, and recovers almost none of the lost
    accuracy.)

    The estimator used here is CLASS-CONDITIONAL. Within each class separately,
    compute the mean of each province and subtract the class mean. Any direction
    that distinguishes classes cancels, because we never compare across classes.
    What survives distinguishes provinces *at fixed task content* -- which is
    precisely, and only, the homeland. (Measured: 99% of energy in the
    provenance block, and it recovers the full oracle accuracy.)

    Historically this is not a footnote. It is the difference between removing a
    soldier's tribal loyalty and removing his ability to ride. Al-Mu'tasim's
    system did the conditional version by brute institutional force: it took
    boys who were already competent horsemen and stripped everything else. What
    made the mamluk system work for four centuries after him is that it never
    confused the two subtractions.

    THE PROJECTION
    --------------
        P = I - U U^T,  U orthonormal, rank r
        sever(X) = X P

    A note on `min_cell`. Because the yoke is strong, the class x province
    contingency table is very unbalanced: the diagonal cells are large and the
    off-diagonal cells are thin. Thin cells have noisy means, and their noise
    is content-shaped, so admitting them drags content directions back into the
    basis. Requiring a minimum cell size before a residual is used raises the
    fraction of the removed subspace that lies in the true provenance block from
    0.77 to 0.98 on the same data. Measuring rare combinations badly is worse
    than not measuring them at all.

    P is symmetric idempotent, so d(XP)/dX applies P again -- gradients pass
    through cleanly and the operator adds no parameters. It is a fixed
    architectural statistic, computed once from the training set, like a
    normalisation constant.
    """

    def __init__(self, U: Optional[np.ndarray], d: int):
        self.d = d
        self.U = U
        self.P = np.eye(d) if U is None else (np.eye(d) - U @ U.T)
        self.rank = 0 if U is None else U.shape[1]

    @staticmethod
    def identity(d: int) -> "Severance":
        """No severance. Provenance passes through intact."""
        return Severance(None, d)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, prov: np.ndarray,
            n_classes: int, n_provinces: int, rank: int = 3,
            min_cell: int = 100) -> "Severance":
        """Estimate the provenance subspace class-conditionally. See above."""
        rows: List[np.ndarray] = []
        for cl in range(n_classes):
            mask = (y == cl)
            if int(np.sum(mask)) < 10:
                continue
            Xc, pc = X[mask], prov[mask]
            mu = Xc.mean(axis=0)
            for p in range(n_provinces):
                cell = Xc[pc == p]
                if len(cell) >= min_cell:
                    rows.append(cell.mean(axis=0) - mu)
        if not rows:
            raise ValueError("no class x province cell had enough samples")
        R = np.stack(rows).T                      # (d, cells)
        U, _, _ = np.linalg.svd(R, full_matrices=False)
        return cls(U[:, :rank], X.shape[1])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X if self.U is None else X @ self.P

    def origin_energy(self, n_content: int) -> float:
        """
        Diagnostic: what fraction of the removed subspace lies in the true
        provenance block? Near 1.0 means we cut the homeland and nothing else.
        """
        if self.U is None:
            return 0.0
        return float(np.sum(self.U[n_content:] ** 2) / np.sum(self.U ** 2))


def content_floor(corpus: ProvinceCorpus, n: int = 6000, seed: int = 313,
                  iters: int = 900, lr: float = 0.05) -> float:
    """
    The irreducible floor for any origin probe, measured rather than assumed.

    THIS IS THE CORRECTION THAT MADE THE LEDGER HONEST.

    The first version of this file scored severance against chance (1/n_provinces
    = 0.25). That is wrong, and wrong in a way worth recording, because it made a
    perfectly severed network look like a total failure -- it reported severance
    of 0.000 for a projection that was in fact removing 99% of the provenance
    signal.

    The reason: province and label are yoked. A probe that reads the CONTENT
    block alone -- which carries no homeland information whatsoever -- can still
    name the province at 0.639 on this corpus, simply by inferring the label and
    exploiting the yoke. That accuracy is not leakage. It is the shadow the task
    itself casts on provenance, and no amount of severance can or should remove
    it, because removing it would mean removing the competence.

    So the floor is measured directly: train a linear probe on content features
    only and see how well it does. Severance is then scored as the fraction of
    the ABOVE-FLOOR provenance advantage that has been destroyed.

    Measured on the default corpus: chance 0.250, content floor 0.639,
    raw-input probe 0.999.
    """
    X, y, prov = corpus.sample(n, seed=seed, shifted=False)
    A = X[:, :corpus.n_content]
    W = np.zeros((A.shape[1], corpus.n_provinces))
    b = np.zeros(corpus.n_provinces)
    for _ in range(iters):
        _, g = cross_entropy(A @ W + b, prov)
        W -= lr * (A.T @ g)
        b -= lr * g.sum(axis=0)
    return float(np.mean(np.argmax(A @ W + b, axis=1) == prov))


# ==============================================================================
# V. PARAMETERS
# ==============================================================================


@dataclass
class Samarra:
    """
    The parameter container. Names follow the historical structure on purpose:
    the experts are regiments of ghilman, the router is the caliphal gate, the
    classifier is the divan (the chancery that turns a mixed council into one
    decision), and the two probes are the auditors nobody in Samarra employed.
    """

    d_in: int
    n_regiments: int = 4
    d_hidden: int = 24
    d_repr: int = 16
    n_classes: int = 3
    n_provinces: int = 4
    seed: int = 842

    P: Dict[str, np.ndarray] = field(default_factory=dict)
    sev_guard: Severance = field(default=None, repr=False)   # regiment intake
    sev_court: Severance = field(default=None, repr=False)   # router intake

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        def he(shape: Tuple[int, ...], fan_in: int) -> np.ndarray:
            return rng.normal(0.0, math.sqrt(2.0 / fan_in), shape)

        K, H, M = self.n_regiments, self.d_hidden, self.d_repr

        # Regiments: K independent 2-layer tanh MLPs, deliberately not shared.
        self.P["W1"] = he((K, self.d_in, H), self.d_in)
        self.P["b1"] = np.zeros((K, H))
        self.P["W2"] = he((K, H, M), H)
        self.P["b2"] = np.zeros((K, M))

        # Caliphal gate: input -> distribution over regiments.
        self.P["Wg"] = he((self.d_in, K), self.d_in) * 0.5
        self.P["bg"] = np.zeros(K)

        # Divan: mixed representation -> class logits.
        self.P["Wo"] = he((M, self.n_classes), M)
        self.P["bo"] = np.zeros(self.n_classes)

        # Probe A: representation -> province. Reads the guard.
        self.P["Wp"] = he((M, self.n_provinces), M)
        self.P["bp"] = np.zeros(self.n_provinces)

        # Probe B: routing distribution -> province. Reads the sovereign.
        self.P["Wq"] = he((K, self.n_provinces), K)
        self.P["bq"] = np.zeros(self.n_provinces)

        if self.sev_guard is None:
            self.sev_guard = Severance.identity(self.d_in)
        if self.sev_court is None:
            self.sev_court = Severance.identity(self.d_in)

    def probe_keys(self) -> List[str]:
        return ["Wp", "bp", "Wq", "bq"]

    def copy(self) -> "Samarra":
        c = Samarra(self.d_in, self.n_regiments, self.d_hidden, self.d_repr,
                    self.n_classes, self.n_provinces, self.seed,
                    sev_guard=self.sev_guard, sev_court=self.sev_court)
        c.P = {k: v.copy() for k, v in self.P.items()}
        return c


# ==============================================================================
# VI. FORWARD
# ==============================================================================


def forward(net: Samarra, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    One forward pass, caching everything backward needs.

    N batch, K regiments, H hidden, M representation, C classes, Pv provinces.

      Xr   (N, d)      input after severance at the REGIMENT gate
      Xg   (N, d)      input after severance at the ROUTER gate
      h1   (N, K, H)   regiment hidden activations
      z2   (N, K, M)   regiment outputs
      gate (N, K)      routing distribution (rows sum to 1)
      Z    (N, M)      the mixed council, sum_k gate_k * z2_k
      out  (N, C)      the divan's judgement
      prb  (N, Pv)     probe A's guess at the homeland, from Z
      prg  (N, Pv)     probe B's guess at the homeland, from the routing

    Every regiment is evaluated on every example. That is faithful: the caliph
    could summon any regiment. It is the GATE, never the regiments, that
    concentrates power -- which is why the gate is what the ledger watches.
    """
    Xr = net.sev_guard(X)
    Xg = net.sev_court(X)

    a1 = np.einsum("nd,kdh->nkh", Xr, net.P["W1"]) + net.P["b1"][None, :, :]
    h1 = np.tanh(a1)
    z2 = np.einsum("nkh,khm->nkm", h1, net.P["W2"]) + net.P["b2"][None, :, :]

    glog = Xg @ net.P["Wg"] + net.P["bg"][None, :]
    gate = softmax(glog, axis=1)

    Z = np.einsum("nk,nkm->nm", gate, z2)

    out = Z @ net.P["Wo"] + net.P["bo"][None, :]
    prb = Z @ net.P["Wp"] + net.P["bp"][None, :]
    prg = gate @ net.P["Wq"] + net.P["bq"][None, :]

    return dict(X=X, Xr=Xr, Xg=Xg, a1=a1, h1=h1, z2=z2,
                glog=glog, gate=gate, Z=Z, out=out, prb=prb, prg=prg)


# ==============================================================================
# VII. LOSSES
# ==============================================================================


@dataclass
class Coeffs:
    """
    Loss weights.

    lam_probe  The auditors' weight. Always positive, in every regime. If the
               probes were only trained when we intended to scrub, we could
               never see whether an UNSCRUBBED system had been leaning on
               provenance -- which is exactly the blindness this chapter is
               about. The probes are instruments, and instruments stay on.

               Note they are auditors, not adversaries: they read the
               representation, the representation is never trained to defeat
               them, AND -- see audit_gradient -- it is never trained to please
               them either. Their gradients are detached from the body. Severance here is a projection at intake, not a
               tug-of-war. An earlier adversarial version of this file
               oscillated -- the encoder learned to hide provenance from the
               current probe without removing it, severance scores rose to 0.84
               while shifted accuracy stayed flat. Hiding is not forgetting.
               That failure is reported in the chapter because it is the same
               failure the mamluk system had.

    lam_cant   Strength of the cantonment entropy floor on the routing.
    """
    lam_cant: float = 0.0
    lam_probe: float = 1.0


def losses(net: Samarra, c: Dict[str, np.ndarray], y: np.ndarray,
           prov: np.ndarray, k: Coeffs) -> Dict[str, float]:
    """
      task    cross-entropy of the divan. What the state is for.
      probe   cross-entropy of probe A. LOW means the guard still knows its
              homeland.
      probe_g cross-entropy of probe B. LOW means the sovereign still selects
              by homeland.
      cant    negative entropy of the mean routing. Minimising spreads patronage.

      total = task + lam_probe*(probe + probe_g) + lam_cant*cant

    `total` is an ordinary differentiable function of every parameter, which is
    what lets the gradient check in section IX be exact rather than approximate.
    """
    task, _ = cross_entropy(c["out"], y)
    probe, _ = cross_entropy(c["prb"], prov)
    probe_g, _ = cross_entropy(c["prg"], prov)
    h, _ = entropy_of_mean(c["gate"])
    cant = -h
    total = task + k.lam_probe * (probe + probe_g) + k.lam_cant * cant
    return {"task": task, "probe": probe, "probe_g": probe_g,
            "cant": cant, "entropy": h, "total": total}


# ==============================================================================
# VIII. BACKWARD (hand-derived)
# ==============================================================================


def _zeros(net: Samarra) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v) for k, v in net.P.items()}


def _back_gate(net: Samarra, c: Dict[str, np.ndarray],
               dgate: np.ndarray, g: Dict[str, np.ndarray]) -> None:
    """
    Softmax jacobian for the router:
        dL/dglog = gate * (dgate - sum_j dgate_j gate_j)
    Note the router's input is Xg (post-severance), not X.
    """
    gate = c["gate"]
    dglog = gate * (dgate - np.sum(dgate * gate, axis=1, keepdims=True))
    g["Wg"] += c["Xg"].T @ dglog
    g["bg"] += np.sum(dglog, axis=0)


def _back_from_Z(net: Samarra, c: Dict[str, np.ndarray],
                 dZ: np.ndarray, g: Dict[str, np.ndarray]) -> None:
    """
    Push dL/dZ back through the mixture, the regiments and the gate.

        Z = sum_k gate[:,k] * z2[:,k,:]
        dL/dgate[n,k] = sum_m dZ[n,m] z2[n,k,m]
        dL/dz2[n,k,m] = gate[n,k] dZ[n,m]
    """
    gate, z2, h1, Xr = c["gate"], c["z2"], c["h1"], c["Xr"]

    dgate = np.einsum("nm,nkm->nk", dZ, z2)
    dz2 = gate[:, :, None] * dZ[:, None, :]

    g["W2"] += np.einsum("nkh,nkm->khm", h1, dz2)
    g["b2"] += np.sum(dz2, axis=0)

    dh1 = np.einsum("nkm,khm->nkh", dz2, net.P["W2"])
    da1 = dh1 * (1.0 - h1 ** 2)

    g["W1"] += np.einsum("nd,nkh->kdh", Xr, da1)
    g["b1"] += np.sum(da1, axis=0)

    _back_gate(net, c, dgate, g)


def backward(net: Samarra, c: Dict[str, np.ndarray], y: np.ndarray,
             prov: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Gradient of each loss term separately, w.r.t. every parameter.
    Keys: "task", "probe", "probe_g", "cant".
    """
    Z = c["Z"]

    # ---- task ------------------------------------------------------------
    gt = _zeros(net)
    _, dout = cross_entropy(c["out"], y)
    gt["Wo"] += Z.T @ dout
    gt["bo"] += np.sum(dout, axis=0)
    _back_from_Z(net, c, dout @ net.P["Wo"].T, gt)

    # ---- probe A (reads the regiments through Z) -------------------------
    gp = _zeros(net)
    _, dprb = cross_entropy(c["prb"], prov)
    gp["Wp"] += Z.T @ dprb
    gp["bp"] += np.sum(dprb, axis=0)
    _back_from_Z(net, c, dprb @ net.P["Wp"].T, gp)

    # ---- probe B (reads the routing directly) ----------------------------
    gq = _zeros(net)
    _, dprg = cross_entropy(c["prg"], prov)
    gq["Wq"] += c["gate"].T @ dprg
    gq["bq"] += np.sum(dprg, axis=0)
    _back_gate(net, c, dprg @ net.P["Wq"].T, gq)

    # ---- cantonment (touches the gate only) ------------------------------
    gc = _zeros(net)
    _, dH = entropy_of_mean(c["gate"])
    _back_gate(net, c, -dH, gc)          # cant = -H

    return {"task": gt, "probe": gp, "probe_g": gq, "cant": gc}


def total_gradient(parts: Dict[str, Dict[str, np.ndarray]],
                   k: Coeffs) -> Dict[str, np.ndarray]:
    """
    The exact gradient of `losses(...)['total']`, with every path live.
    This is the object the finite-difference check in section IX verifies.
    """
    return {n: (parts["task"][n]
                + k.lam_probe * (parts["probe"][n] + parts["probe_g"][n])
                + k.lam_cant * parts["cant"][n])
            for n in parts["task"]}


def audit_gradient(net: Samarra, parts: Dict[str, Dict[str, np.ndarray]],
                   k: Coeffs) -> Dict[str, np.ndarray]:
    """
    The gradient actually APPLIED during training. Identical to
    `total_gradient` except that the probe losses are DETACHED from the body:
    they update Wp, bp, Wq, bq and nothing else.

    THIS IS THE SECOND CORRECTION THAT MADE THE LEDGER HONEST, and it is the
    more embarrassing of the two.

    In the first working version the probe losses were left in the shared
    objective. The probes were therefore not measuring the representation --
    they were CO-AUTHORING it. Every step, the regiments received a gradient
    telling them to make the homeland easier to read, because that lowered a
    term in the loss. The result was a network that scored 0.814 on the shifted
    set, which is provenance-blind behaviour by any external test, while
    reporting sev_guard = 0.000, which says the homeland is perfectly legible.
    Both numbers were correct. The instrument had corrupted its own audit.

    An auditor that is paid out of the budget it audits is not an auditor. The
    fix is one line -- read a detached representation -- and the failure is
    left documented here rather than quietly removed, because a caliph who
    cannot tell the difference between an instrument and a report about an
    instrument is the exact subject of this chapter.
    """
    probe_set = set(net.probe_keys())
    out: Dict[str, np.ndarray] = {}
    for n in parts["task"]:
        g = parts["task"][n] + k.lam_cant * parts["cant"][n]
        if n in probe_set:
            g = g + k.lam_probe * (parts["probe"][n] + parts["probe_g"][n])
        out[n] = g
    return out


# ==============================================================================
# IX. MANDATORY FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================


def gradient_check(seed: int = 11, tol: float = 1e-6, verbose: bool = True) -> float:
    """
    Central-difference verification of `total_gradient` against
    `losses(...)['total']`, sampling coordinates from EVERY parameter tensor,
    with severance active on both intakes and cantonment switched on, so no
    path in the graph is left untested.

    Returns the worst relative error. Raises AssertionError past `tol`.
    """
    rng = np.random.default_rng(seed)
    corpus = ProvinceCorpus(seed=5)

    Xf, yf, pf = corpus.sample(3000, seed=41, shifted=False)
    sev = Severance.fit(Xf, yf, pf, corpus.n_classes, corpus.n_provinces, rank=3)

    net = Samarra(d_in=corpus.n_features, n_provinces=corpus.n_provinces,
                  n_classes=corpus.n_classes, seed=99,
                  sev_guard=sev, sev_court=sev)
    # Move biases off zero so every branch is live.
    for key in ("b1", "b2", "bg", "bo", "bp", "bq"):
        net.P[key] += rng.normal(0.0, 0.25, net.P[key].shape)

    X, y, prov = corpus.sample(24, seed=3, shifted=False)
    k = Coeffs(lam_cant=0.35, lam_probe=0.55)

    analytic = total_gradient(backward(net, forward(net, X), y, prov), k)

    eps, worst, where = 1e-6, 0.0, ""
    for name, arr in net.P.items():
        flat = arr.ravel()
        for i in rng.choice(flat.size, size=min(9, flat.size), replace=False):
            orig = flat[i]
            flat[i] = orig + eps
            lp = losses(net, forward(net, X), y, prov, k)["total"]
            flat[i] = orig - eps
            lm = losses(net, forward(net, X), y, prov, k)["total"]
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[name].ravel()[i]
            rel = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            if rel > worst:
                worst, where = rel, f"{name}[{i}]"
    assert worst < tol, f"GRADIENT CHECK FAILED: {worst:.3e} at {where}"
    if verbose:
        print(f"  central differences vs analytic gradients")
        print(f"  worst relative error = {worst:.3e}   (at {where})   "
              f"tolerance {tol:.0e}   PASSED")
    return worst


# ==============================================================================
# X. THE CAPTURE LEDGER
# ==============================================================================


@dataclass
class Ledger:
    """
    The instrument al-Mu'tasim did not have.

    He could see that the Turks were useful. Nothing in his court measured the
    ratio of his own discretion to their necessity, or watched that ratio fall
    year on year. The ledger makes it a number and reads it every epoch.

      monopoly       largest mean gate share. 1/K is perfect balance; 1.0 is a
                     praetorian guard.
      capture_index  1 - H(mean gate)/log K. Zero when patronage is spread,
                     one when a single regiment IS the state.
      sev_guard      1 - (probe-A accuracy - floor)/(1 - floor), clipped to
                     [0,1], where `floor` is the MEASURED content floor (see
                     content_floor). One means the regiments retain no
                     homeland information beyond what the task itself implies.
      sev_court      the same for probe B. One means the sovereign is no longer
                     selecting by homeland. This is the number the historical
                     record never had, and the reason the chapter exists.
      decisiveness   mean max gate weight per example. High decisiveness with
                     low monopoly is the healthy regime: sharp specialists, no
                     favourite.
    """
    train_acc: float
    shift_acc: float
    sev_guard: float
    sev_court: float
    monopoly: float
    capture_index: float
    decisiveness: float

    def line(self, tag: str = "") -> str:
        return (f"{tag:<17} train={self.train_acc:5.3f}  shifted={self.shift_acc:5.3f}  "
                f"sev_guard={self.sev_guard:5.3f}  sev_court={self.sev_court:5.3f}  "
                f"monopoly={self.monopoly:5.3f}  capture={self.capture_index:5.3f}")


def accuracy(net: Samarra, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(forward(net, X)["out"], axis=1) == y))


def read_ledger(net: Samarra, Xtr, ytr, ptr, Xsh, ysh,
                floor: Optional[float] = None) -> Ledger:
    """
    `floor` is the measured content floor. If it is left as None the metric
    falls back to chance, which is the naive normalisation and understates
    severance badly -- see content_floor for why.
    """
    c = forward(net, Xtr)
    gbar = np.mean(c["gate"], axis=0)
    K = c["gate"].shape[1]
    h = -float(np.sum(gbar * np.log(np.clip(gbar, 1e-12, None))))
    base = (1.0 / net.n_provinces) if floor is None else floor

    def sev(logits: np.ndarray) -> float:
        acc = float(np.mean(np.argmax(logits, axis=1) == ptr))
        return float(np.clip(1.0 - (acc - base) / (1.0 - base), 0.0, 1.0))

    return Ledger(
        train_acc=accuracy(net, Xtr, ytr),
        shift_acc=accuracy(net, Xsh, ysh),
        sev_guard=sev(c["prb"]),
        sev_court=sev(c["prg"]),
        monopoly=float(np.max(gbar)),
        capture_index=float(1.0 - h / math.log(K)),
        decisiveness=float(np.mean(np.max(c["gate"], axis=1))),
    )


# ==============================================================================
# XI. TRAINING
# ==============================================================================


class Adam:
    """Minimal Adam, written out so nothing in this file is a black box."""

    def __init__(self, net: Samarra, lr: float = 0.02,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in net.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.P.items()}
        self.t = 0

    def step(self, net: Samarra, g: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in net.P:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            net.P[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_step(net: Samarra, opt: Adam, X, y, prov, k: Coeffs) -> Dict[str, float]:
    c = forward(net, X)
    L = losses(net, c, y, prov, k)
    opt.step(net, audit_gradient(net, backward(net, c, y, prov), k))
    return L


def train(net: Samarra, corpus: ProvinceCorpus, k: Coeffs, *,
          epochs: int = 260, batch: int = 128, lr: float = 0.02,
          seed: int = 7, verbose: bool = False, tag: str = "",
          every: int = 65, floor: Optional[float] = None) -> List[Ledger]:
    """Minibatch training with Adam and held-out evaluation under the shift."""
    Xtr, ytr, ptr = corpus.sample(1600, seed=seed, shifted=False)
    Xsh, ysh, _ = corpus.sample(1600, seed=seed + 991, shifted=True)

    opt = Adam(net, lr=lr)
    rng = np.random.default_rng(seed + 3)
    history: List[Ledger] = []

    for ep in range(epochs):
        idx = rng.permutation(Xtr.shape[0])
        for s in range(0, len(idx), batch):
            j = idx[s:s + batch]
            train_step(net, opt, Xtr[j], ytr[j], ptr[j], k)
        if ep == 0 or (ep + 1) % every == 0:
            led = read_ledger(net, Xtr, ytr, ptr, Xsh, ysh, floor)
            history.append(led)
            if verbose:
                print("   " + led.line(f"{tag} ep{ep + 1:>4}"))
    return history


# ==============================================================================
# XII. SELF-TESTS
# ==============================================================================


def self_tests() -> None:
    print("\n[SELF-TESTS]")

    z = np.random.default_rng(0).normal(size=(7, 5))
    s = softmax(z, axis=1)
    assert np.allclose(np.sum(s, axis=1), 1.0) and np.all(s > 0)
    print("  softmax normalisation ..................... ok")

    l, _ = cross_entropy(np.array([[20.0, 0, 0], [0, 20.0, 0]]), np.array([0, 1]))
    assert l < 1e-6
    print("  cross-entropy floor ....................... ok")

    K = 4
    h_uni, _ = entropy_of_mean(np.full((10, K), 1.0 / K))
    hot = np.zeros((10, K)); hot[:, 0] = 1.0
    h_hot, _ = entropy_of_mean(hot)
    assert abs(h_uni - math.log(K)) < 1e-12 and h_hot < 1e-9
    print("  cantonment entropy bounds ................. ok")

    corpus = ProvinceCorpus(seed=2)
    Xa, ya, pa = corpus.sample(20000, seed=10, shifted=False)
    Xb, yb, pb = corpus.sample(20000, seed=11, shifted=True)
    hit_tr = float(np.mean(corpus.map_train[pa] == ya))
    hit_sh = float(np.mean(corpus.map_train[pb] == yb))
    assert hit_tr > 0.85 and hit_sh < 0.15
    print(f"  provenance shortcut  train={hit_tr:.3f}  shifted={hit_sh:.3f} .. ok")

    # severance operator: idempotent, symmetric, and aimed at the right block
    sev = Severance.fit(Xa, ya, pa, corpus.n_classes, corpus.n_provinces, rank=3)
    assert np.allclose(sev.P, sev.P.T)
    assert np.allclose(sev.P @ sev.P, sev.P)
    assert np.allclose(sev.U.T @ sev.U, np.eye(sev.rank))
    energy = sev.origin_energy(corpus.n_content)
    assert energy > 0.93, energy
    print(f"  severance projector  P=P^T=P^2, rank {sev.rank}, "
          f"{energy * 100:.1f}% in provenance block .. ok")

    # the naive basis is the trap described in section IV -- show it explicitly
    mu = Xa.mean(0)
    Bp = np.stack([Xa[pa == p].mean(0) - mu for p in range(corpus.n_provinces)])
    Un, _, _ = np.linalg.svd(Bp.T, full_matrices=False)
    naive_energy = float(np.sum(Un[corpus.n_content:, :3] ** 2)
                         / np.sum(Un[:, :3] ** 2))
    assert naive_energy < energy
    print(f"  unconditional basis leaks task signal  "
          f"({naive_energy * 100:.1f}% vs {energy * 100:.1f}%) ... ok")

    # forward-pass structure
    net = Samarra(d_in=corpus.n_features, n_provinces=corpus.n_provinces,
                  n_classes=corpus.n_classes, seed=1,
                  sev_guard=sev, sev_court=sev)
    X, y, p = corpus.sample(16, seed=4, shifted=False)
    c = forward(net, X)
    assert np.allclose(np.sum(c["gate"], axis=1), 1.0)
    manual = sum(c["gate"][:, kk:kk + 1] * c["z2"][:, kk, :]
                 for kk in range(net.n_regiments))
    assert np.allclose(manual, c["Z"])
    print("  routing mixture identity .................. ok")

    # the auditors must not be able to touch the body
    parts = backward(net, forward(net, X), y, p)
    kk = Coeffs(lam_cant=0.2, lam_probe=1.0)
    gt, ga = total_gradient(parts, kk), audit_gradient(net, parts, kk)
    for name in net.P:
        if name in net.probe_keys():
            assert np.allclose(gt[name], ga[name]), name
        else:
            assert np.allclose(
                ga[name], parts["task"][name] + kk.lam_cant * parts["cant"][name]
            ), name
    print("  auditors detached from the body ........... ok")

    # severance really removes provenance from what the network sees
    sev_only = Severance.fit(Xa, ya, pa, corpus.n_classes, corpus.n_provinces, 3)
    resid = sev_only(Xa)[:, corpus.n_content:]
    raw = Xa[:, corpus.n_content:]
    assert np.linalg.norm(resid) < 0.55 * np.linalg.norm(raw)
    print("  provenance block collapsed by projection ... ok")


# ==============================================================================
# XIII. THE EXPERIMENT
# ==============================================================================


def experiment() -> Dict[str, Ledger]:
    """
    Four doctrines of trust. One corpus, one initialisation, one seed.

      1 NAIVE           Nothing removed. Provenance is the cheapest accurate
                        predictor in the data and the network takes it. Perfect
                        on the world it trained in; badly wrong the moment that
                        world reshuffles.

      2 SEVERED GUARD   Provenance removed at the REGIMENT intake only; the
                        router still sees it. THIS IS AL-MU'TASIM'S ACTUAL
                        REIGN. Each ghulam forgets his homeland and the caliph
                        goes on choosing by it.

      3 SEVERED COURT   Removed at both intakes. The reform he never made.

      4 SAMARRA         Both, plus the cantonment floor on the routing.
    """
    corpus = ProvinceCorpus(seed=20201)

    # The severance basis is fitted on a large training draw, once, and then
    # frozen -- an architectural statistic, not a learned parameter.
    Xf, yf, pf = corpus.sample(20000, seed=77, shifted=False)
    sev = Severance.fit(Xf, yf, pf, corpus.n_classes, corpus.n_provinces, rank=3)
    ident = Severance.identity(corpus.n_features)

    Xtr, ytr, ptr = corpus.sample(1600, seed=7, shifted=False)
    Xsh, ysh, _ = corpus.sample(1600, seed=998, shifted=True)

    floor = content_floor(corpus)
    print(f"\n[PROBE FLOOR]  chance={1.0 / corpus.n_provinces:.3f}   "
          f"content floor={floor:.3f}   raw-input probe~0.999")
    print( "               Severance is scored against the content floor, not")
    print( "               chance. See content_floor() for why that matters.")

    print(f"\n[SEVERANCE OPERATOR]  rank {sev.rank} of {corpus.n_features}, "
          f"{sev.origin_energy(corpus.n_content) * 100:.1f}% of the removed "
          f"subspace lies\n                      in the true provenance block.")

    regimes = [
        ("1 NAIVE",         ident, ident, Coeffs(0.00)),
        ("2 SEVERED GUARD", sev,   ident, Coeffs(0.00)),
        ("3 SEVERED COURT", sev,   sev,   Coeffs(0.00)),
        ("4 SAMARRA",       sev,   sev,   Coeffs(0.60)),
    ]

    print("\n[EXPERIMENT]  identical initialisation, identical data, "
          "four doctrines of trust\n")
    out: Dict[str, Ledger] = {}
    for name, sg, sc, k in regimes:
        net = Samarra(d_in=corpus.n_features, n_provinces=corpus.n_provinces,
                      n_classes=corpus.n_classes, seed=842,
                      sev_guard=sg, sev_court=sc)
        train(net, corpus, k, epochs=260, batch=128, lr=0.02, seed=7, floor=floor)
        led = read_ledger(net, Xtr, ytr, ptr, Xsh, ysh, floor)
        out[name] = led
        print("  " + led.line(name))

    n, g, c, s = (out["1 NAIVE"], out["2 SEVERED GUARD"],
                  out["3 SEVERED COURT"], out["4 SAMARRA"])

    print("\n[READING THE LEDGER]")
    print(f"  Naive court: trains to {n.train_acc:.3f}. When provenance stops meaning")
    print(f"  what it meant, it holds {n.shift_acc:.3f}. It had learned the homeland.")
    print()
    print(f"  Severing the GUARD alone moves shifted accuracy {g.shift_acc - n.shift_acc:+.3f} "
          f"to {g.shift_acc:.3f}.")
    frac = (g.shift_acc - n.shift_acc) / max(1e-9, c.shift_acc - n.shift_acc)
    print(f"  That is {frac * 100:.0f}% of the {c.shift_acc - n.shift_acc:+.3f} that is actually")
    print(f"  available. sev_guard rises to {g.sev_guard:.3f} -- the regiments really have")
    print(f"  forgotten -- but sev_court is only {g.sev_court:.3f}: the router still reads")
    print(f"  homelands off the raw input and relays them into the mixture.")
    print(f"  This regime is al-Mu'tasim's actual reign, and it is the null result")
    print(f"  that gives this chapter its thesis.")
    print()
    print(f"  Severing the COURT as well reaches {c.shift_acc:.3f} "
          f"({c.shift_acc - n.shift_acc:+.3f} on naive),")
    print(f"  with sev_guard={c.sev_guard:.3f}, sev_court={c.sev_court:.3f}.")
    print()
    print(f"  Adding the cantonment floor holds {s.shift_acc:.3f}, monopoly "
          f"{c.monopoly:.3f}->{s.monopoly:.3f},")
    print(f"  capture {c.capture_index:.3f}->{s.capture_index:.3f}, at decisiveness "
          f"{s.decisiveness:.3f}:")
    print(f"  sharp specialists, no favourite.")
    print()
    print("  Severance is necessary and it is not sufficient. What it omits is")
    print("  never the instrument's memory. It is the sovereign's habit.")
    return out


def capture_trace() -> None:
    """Watch the ledger move during training, with and without the floor."""
    corpus = ProvinceCorpus(seed=20201)
    Xf, yf, pf = corpus.sample(20000, seed=77, shifted=False)
    sev = Severance.fit(Xf, yf, pf, corpus.n_classes, corpus.n_provinces, rank=3)

    fl = content_floor(corpus)
    for tag, k in (("no floor", Coeffs(0.00)), ("cantonment", Coeffs(0.60))):
        print(f"\n[TRACE: {tag}]")
        net = Samarra(d_in=corpus.n_features, n_provinces=corpus.n_provinces,
                      n_classes=corpus.n_classes, seed=842,
                      sev_guard=sev, sev_court=sev)
        train(net, corpus, k, epochs=260, verbose=True, tag=tag, every=65, floor=fl)


# ==============================================================================
# XIV. ENTRY POINT
# ==============================================================================


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 79)
    print(" SAMARRA  -  Severance-Aligned Modular Architecture")
    print(" Chapter 0201   Al-Mu'tasim bi'llah   (Baghdad 796 - Samarra 842)")
    print("=" * 79)

    self_tests()
    print("\n[GRADIENT CHECK]")
    gradient_check()
    experiment()
    capture_trace()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

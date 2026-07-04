"""
chapter_0013_akhenaten_-1380.py  --  ATEN-NET
================================================================================
Chapter 13: Akhenaten  (b. c. 1380 BCE - d. c. 1336 BCE, Amarna, Egypt)
 
 Part Encyclopedia of Lost Minds: Echoes on AI By David Vivancos https://www.vivancos.com/
How History's Greatest Thinkers Would Have Thought About AGI  https://lostmindsai.com
Tome 1 Minds 1 - 20 Available on Amazon https://www.amazon.com/dp/B0H6F9L324
Resume and Interactive Demos at https://artificiology.com/

A from-scratch, pure-NumPy neural architecture whose *core learning operation is
deletion*. It is built to embody one specific cognitive signature of Akhenaten --
not the generic "unified mind / cosmic order" reading, but the thing that was
literally his alone in the ancient record:

    Akhenaten advanced toward truth by SUBTRACTION. He did not mainly add a
    supreme god; he chiselled names off monuments across an empire, banned the
    images, declared the other gods dead, and collapsed a pantheon of hundreds
    of interlinked deities down to a single VISIBLE source -- the Aten, the sun
    disk -- whose life-rays reached the world through one intermediary. The god
    he hunted hardest, Amun, means literally "the Hidden One." Visibility was
    his criterion of reality; the hidden was to be erased.

ATEN-NET turns that doctrine into a runnable mechanism. It is deliberately NOT a
Transformer and uses no attention-over-stored-keys. It is a single-source sparse
dictionary (autoencoder) in which:

  * one shared "source" vector s carries what is COMMON to all data
    (the omnipresent light present in every entity);
  * K candidate "rays"/features carry the entity-specific signal
    (the Amarna individuation -- the particular, not the timeless);
  * learning proceeds by gradient descent on a reconstruction + L1 objective
    (the L1 is the pull toward monotheism: keep as few living names as possible);
  * and then, periodically, the network performs ICONOCLASM: it permanently
    erases the least-VISIBLE features -- those whose contribution to the
    manifest output is smallest -- and never lets them return. Erasure is
    irreversible, exactly as a chiselled cartouche is irreversible.

Two honest experiments are run at the bottom:

  (A) FEW-SOURCE world  -> data really does come from a handful of latent causes.
      Iconoclasm collapses K candidate rays to a small core and reconstruction
      stays good. Monotheism WORKS when reality is simple. (What he got right.)

  (B) MANY-SOURCE world -> data is irreducibly plural.
      The same forced pruning destroys load-bearing structure and reconstruction
      collapses. Imposing unity on irreducible plurality loses information.
      The model re-enacts the Amarna failure. (What he got wrong.)

Conventions kept (mandatory in this corpus):
  * pure NumPy, from scratch;
  * an analytic backward pass verified by a finite-difference gradient check
    (must pass);
  * a real training loop whose loss decreases;
  * self-tests with assertions;
  * the file executes and the printed output is pasted into the chapter.

Run:  python3 chapter_0013_akhenaten_-1380.py
Author: David Vivancos  --  Mind #13, Akhenaten.
================================================================================
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU. A feature is either 'a living name' (>0) or silent (0).

    The hard on/off is intentional: in Akhenaten's world a deity either has a
    name carved in stone or it does not. There is no half-erased god.
    """
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU w.r.t. its pre-activation (1 where active, else 0)."""
    return (x > 0.0).astype(x.dtype)


# -----------------------------------------------------------------------------
# ATEN-NET
# -----------------------------------------------------------------------------
class AtenNet:
    """A single-source sparse dictionary trained, then pruned by iconoclasm.

    Parameters
    ----------
    d : int
        Dimensionality of an observed "entity" (a datum the world makes visible).
    K : int
        Number of candidate rays / features. Think of these as the candidate
        gods of the pantheon before the reform. Most will be erased.
    l1 : float
        Strength of the L1 pull on feature activity. This is the *drive toward
        monotheism*: it makes the network prefer to explain the world with as
        few living names as possible.
    seed : int
        RNG seed for reproducibility.

    The forward model for a batch X (N, d):

        pre   = X @ W_enc.T + b_enc          # (N, K)   raw ray response
        z     = relu(pre)                    # (N, K)   only living rays fire
        zmask = z * mask                     # (N, K)   erased rays are dead (0)
        recon = zmask @ W_dec.T + s          # (N, d)   manifest reconstruction

    where `s` (d,) is THE single shared source -- the light common to all
    entities -- learned as a global bias, and `mask` (K,) holds 1 for a living
    name and 0 for a chiselled-out one. Erasure is permanent: a 0 in `mask`
    plus a True in `frozen` can never become 1 again.

    Objective (smooth, differentiable, the part we backprop):

        L = (1 / (2N)) * sum( (recon - X)^2 )      # manifest reconstruction
            + (l1 / N) * sum( zmask )               # monotheist parsimony (L1)

    Iconoclasm (discrete, applied between training phases, NOT part of the
    gradient) erases the least-visible living features, where a feature's
    VISIBILITY is its mean manifest contribution  mean_n( zmask[n,k] ) *
    ||W_dec[:,k]||  -- how much it actually shows up in the world's surface.
    """

    def __init__(self, d: int, K: int, l1: float = 0.02, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.d = d
        self.K = K
        self.l1 = float(l1)

        # He-style init for the encoder so early pre-activations have margin
        # away from the ReLU kink (keeps the gradient check clean).
        self.W_enc = rng.standard_normal((K, d)) * np.sqrt(2.0 / d)
        self.b_enc = np.full(K, 0.1)                       # gentle positive bias
        self.W_dec = rng.standard_normal((d, K)) * np.sqrt(1.0 / K)
        self.s = np.zeros(d)                               # the single source

        # Iconoclasm bookkeeping: every name starts alive and unfrozen.
        self.mask = np.ones(K)
        self.frozen = np.zeros(K, dtype=bool)

    # ----- parameter (de)serialisation for the gradient check --------------
    def get_params(self) -> dict:
        return {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "W_dec": self.W_dec,
            "s": self.s,
        }

    # ----- forward ---------------------------------------------------------
    def forward(self, X: np.ndarray) -> dict:
        """Run the manifest reconstruction and cache everything backward needs."""
        pre = X @ self.W_enc.T + self.b_enc          # (N, K)
        z = relu(pre)                                # (N, K)
        zmask = z * self.mask                        # (N, K)
        recon = zmask @ self.W_dec.T + self.s        # (N, d)
        return {"X": X, "pre": pre, "z": z, "zmask": zmask, "recon": recon}

    def loss(self, X: np.ndarray):
        """Return (scalar loss, cache)."""
        cache = self.forward(X)
        N = X.shape[0]
        resid = cache["recon"] - X                   # (N, d)
        l_recon = 0.5 * np.sum(resid * resid) / N
        l_sparse = self.l1 * np.sum(cache["zmask"]) / N
        cache["resid"] = resid
        return float(l_recon + l_sparse), cache

    # ----- backward (analytic) --------------------------------------------
    def backward(self, cache: dict) -> dict:
        """Analytic gradients of `loss` w.r.t. W_enc, b_enc, W_dec, s.

        Derivation (N samples):
          resid      = recon - X
          dL/drecon  = resid / N                                   (N,d)
          dL/ds      = sum_n resid[n] / N                          (d,)
          dL/dW_dec  = (resid/N).T @ zmask                         (d,K)
          dL/dzmask  = (resid/N) @ W_dec   + (l1/N)                (N,K)
          dL/dz      = dL/dzmask * mask                            (N,K)
          dL/dpre    = dL/dz * relu'(pre)                          (N,K)
          dL/dW_enc  = dL/dpre.T @ X                               (K,d)
          dL/db_enc  = sum_n dL/dpre[n]                            (K,)
        The L1 term contributes (l1/N) to dL/dzmask for every entry, because
        zmask >= 0 (it is a masked ReLU output) so |zmask| = zmask.
        """
        X = cache["X"]
        N = X.shape[0]
        resid = cache["resid"]                       # (N, d)

        d_recon = resid / N                          # (N, d)
        d_s = d_recon.sum(axis=0)                     # (d,)
        d_W_dec = d_recon.T @ cache["zmask"]          # (d, K)

        d_zmask = d_recon @ self.W_dec + (self.l1 / N)   # (N, K)
        d_z = d_zmask * self.mask                     # (N, K)
        d_pre = d_z * relu_grad(cache["pre"])         # (N, K)

        d_W_enc = d_pre.T @ X                         # (K, d)
        d_b_enc = d_pre.sum(axis=0)                   # (K,)

        return {"W_enc": d_W_enc, "b_enc": d_b_enc, "W_dec": d_W_dec, "s": d_s}

    # ----- finite-difference gradient check (mandatory) -------------------
    def gradient_check(self, X: np.ndarray, eps: float = 1e-6) -> float:
        """Compare analytic grads to central finite differences.

        Returns the maximum relative error across all parameters. A small value
        (<~1e-5) certifies the backward pass is correct.
        """
        _, cache = self.loss(X)
        analytic = self.backward(cache)
        params = self.get_params()

        max_rel = 0.0
        rng = np.random.default_rng(123)
        for name, P in params.items():
            flat = P.reshape(-1)
            g_flat = analytic[name].reshape(-1)
            # Sample a handful of coordinates (full check is O(params) and slow).
            idxs = rng.choice(flat.size, size=min(12, flat.size), replace=False)
            for i in idxs:
                orig = flat[i]
                flat[i] = orig + eps
                lp, _ = self.loss(X)
                flat[i] = orig - eps
                lm, _ = self.loss(X)
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                ana = g_flat[i]
                denom = max(1e-12, abs(num) + abs(ana))
                rel = abs(num - ana) / denom
                max_rel = max(max_rel, rel)
        return max_rel

    # ----- one SGD-ish step ------------------------------------------------
    def step(self, X: np.ndarray, lr: float) -> float:
        l, cache = self.loss(X)
        g = self.backward(cache)
        self.W_enc -= lr * g["W_enc"]
        self.b_enc -= lr * g["b_enc"]
        self.W_dec -= lr * g["W_dec"]
        self.s -= lr * g["s"]
        # Keep erased names dead even if a gradient nudges their decoder column:
        self.W_dec *= self.mask  # zero columns for erased features
        return l

    # ----- visibility & iconoclasm ----------------------------------------
    def visibility(self, X: np.ndarray) -> np.ndarray:
        """Per-feature manifest contribution = mean activity * decoder norm.

        This is the operational form of Akhenaten's reality test: how much does
        this name actually SHOW UP in the visible world? Hidden, near-silent
        features score low and become candidates for the chisel.
        """
        cache = self.forward(X)
        mean_act = cache["zmask"].mean(axis=0)        # (K,)
        col_norm = np.linalg.norm(self.W_dec, axis=0)  # (K,)
        vis = mean_act * col_norm
        vis[self.frozen] = -np.inf                      # already gone
        return vis

    def iconoclasm(self, X: np.ndarray, n_erase: int) -> list[int]:
        """Permanently erase the `n_erase` least-visible living features.

        Returns the indices erased. This is the discrete, irreversible act --
        the masons sent out with chisels. Erased names are frozen forever.
        """
        vis = self.visibility(X)
        living = np.where(~self.frozen)[0]
        if n_erase <= 0 or living.size == 0:
            return []
        n_erase = min(n_erase, living.size)
        # Rank living features by visibility; erase the dimmest.
        order = living[np.argsort(vis[living])]
        victims = list(order[:n_erase])
        for k in victims:
            self.mask[k] = 0.0
            self.frozen[k] = True
            self.W_dec[:, k] = 0.0   # chisel the column out of the stone
        return victims

    def n_living(self) -> int:
        return int(self.mask.sum())

    # ----- training with periodic iconoclasm ------------------------------
    def train(
        self,
        X: np.ndarray,
        epochs: int = 400,
        lr: float = 0.05,
        prune_every: int = 80,
        prune_n: int = 3,
        target_living: int = 1,
        verbose: bool = False,
    ) -> dict:
        """Full reform: descend, then periodically erase, down to a small core.

        Stops erasing once `target_living` names remain.
        """
        history = {"loss": [], "living": []}
        for ep in range(1, epochs + 1):
            l = self.step(X, lr)
            history["loss"].append(l)
            history["living"].append(self.n_living())
            if ep % prune_every == 0 and self.n_living() > target_living:
                room = self.n_living() - target_living
                erased = self.iconoclasm(X, min(prune_n, room))
                if verbose:
                    print(f"  [epoch {ep:4d}] loss={l:.5f} "
                          f"erased {len(erased)} -> {self.n_living()} living")
        return history

    # ----- the Mosaic distinction (Assmann): true vs. false ---------------
    def mosaic_report(self) -> dict:
        """Count names judged 'true' (kept alive) vs 'false' (erased).

        Jan Assmann calls Akhenaten's innovation the introduction of the
        distinction between true and false religion -- a line that did not exist
        before. Here it is just bookkeeping over the mask.
        """
        kept = int(self.mask.sum())
        erased = int((~(self.mask > 0)).sum())
        return {"true_kept": kept, "false_erased": erased, "total": self.K}


# -----------------------------------------------------------------------------
# Synthetic worlds
# -----------------------------------------------------------------------------
def make_world(n: int, d: int, n_sources: int, seed: int = 0, noise: float = 0.05):
    """Generate data that genuinely arises from `n_sources` latent causes.

    Each datum = shared_light + a sparse mix of a few prototype 'rays' + noise.
    When n_sources is small the world is monotheism-friendly; when large it is
    irreducibly plural.
    """
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(d) * 0.3                 # the common light s*
    prototypes = rng.standard_normal((n_sources, d))      # the true causes
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9

    X = np.tile(shared, (n, 1))
    for i in range(n):
        # each datum lit by 1-2 of the true sources (sparse, individuated)
        k = rng.integers(1, 3)
        chosen = rng.choice(n_sources, size=k, replace=False)
        coeffs = rng.uniform(0.6, 1.4, size=k)
        X[i] += (coeffs[:, None] * prototypes[chosen]).sum(axis=0)
    X += rng.standard_normal((n, d)) * noise
    return X


def reconstruction_error(net: AtenNet, X: np.ndarray) -> float:
    """Mean per-sample squared reconstruction error (the manifest residual)."""
    cache = net.forward(X)
    resid = cache["recon"] - X
    return float(np.mean(np.sum(resid * resid, axis=1)))


# -----------------------------------------------------------------------------
# Self-tests / demonstration
# -----------------------------------------------------------------------------
def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 72)
    print("ATEN-NET  --  truth by deletion  (Chapter 13: Akhenaten)")
    print("=" * 72)

    # --- 1. Gradient check (mandatory) ------------------------------------
    print("\n[1] Finite-difference gradient check")
    Xchk = np.random.default_rng(7).standard_normal((6, 8))
    net_chk = AtenNet(d=8, K=10, l1=0.02, seed=1)
    # erase one feature first, to also verify gradients hold under a partial mask
    net_chk.iconoclasm(Xchk, n_erase=2)
    max_rel = net_chk.gradient_check(Xchk)
    print(f"    max relative error = {max_rel:.3e}")
    assert max_rel < 1e-5, "Gradient check FAILED"
    print("    PASS: analytic gradients match finite differences.")

    # --- 2. FEW-SOURCE world: monotheism works ----------------------------
    print("\n[2] Experiment A -- FEW-SOURCE world (reality is simple)")
    d = 24
    Xa = make_world(n=400, d=d, n_sources=3, seed=11)
    netA = AtenNet(d=d, K=24, l1=0.03, seed=2)
    err0_a = reconstruction_error(netA, Xa)
    histA = netA.train(Xa, epochs=600, lr=0.05,
                       prune_every=60, prune_n=2, target_living=3)
    err1_a = reconstruction_error(netA, Xa)
    print(f"    true latent sources      : 3")
    print(f"    candidate rays (K)       : 24")
    print(f"    living names after reform: {netA.n_living()}")
    print(f"    recon error  start->end  : {err0_a:.4f} -> {err1_a:.4f}")
    print(f"    final training loss      : {histA['loss'][-1]:.4f}")
    mr = netA.mosaic_report()
    print(f"    mosaic distinction       : {mr['true_kept']} kept, "
          f"{mr['false_erased']} erased")
    assert histA["loss"][-1] < histA["loss"][0], "loss did not decrease"
    assert netA.n_living() <= 4, "did not collapse toward monotheism"
    assert err1_a < err0_a, "reform should not worsen a simple world"
    print("    PASS: collapsed to a tiny core, reconstruction stayed good.")

    # --- 3. MANY-SOURCE world: the Amarna failure -------------------------
    print("\n[3] Experiment B -- MANY-SOURCE world (reality is plural)")
    Xb = make_world(n=400, d=d, n_sources=18, seed=12)

    # 3a. An honest baseline: train WITHOUT forced collapse (keep the pantheon).
    net_keep = AtenNet(d=d, K=24, l1=0.005, seed=3)
    net_keep.train(Xb, epochs=600, lr=0.05, prune_every=10_000,  # never prune
                   prune_n=0, target_living=24)
    err_keep = reconstruction_error(net_keep, Xb)

    # 3b. Akhenaten's move: force the same collapse to a single source.
    net_force = AtenNet(d=d, K=24, l1=0.03, seed=3)
    net_force.train(Xb, epochs=600, lr=0.05,
                    prune_every=60, prune_n=3, target_living=1)
    err_force = reconstruction_error(net_force, Xb)

    print(f"    true latent sources          : 18")
    print(f"    pantheon kept (no iconoclasm): {net_keep.n_living():2d} living, "
          f"recon error {err_keep:.4f}")
    print(f"    forced collapse to one source: {net_force.n_living():2d} living, "
          f"recon error {err_force:.4f}")
    ratio = err_force / max(err_keep, 1e-9)
    print(f"    information lost by forcing unity: {ratio:.1f}x worse recon")
    assert err_force > err_keep * 1.5, "forced collapse should hurt a plural world"
    print("    PASS: imposing one source on a plural world destroys structure.")
    print("          (The model re-enacts the Amarna collapse.)")

    # --- 4. Verdict --------------------------------------------------------
    print("\n[4] Verdict")
    print("    Deletion is a real instrument of intelligence: where the world")
    print("    has few causes, iconoclasm finds them and discards the rest")
    print("    (Experiment A). But the SAME instrument, applied where the world")
    print("    is irreducibly plural, erases load-bearing structure and the")
    print("    reconstruction falls apart (Experiment B). Akhenaten's genius and")
    print("    his catastrophe are one operation pointed at two kinds of world.")
    print("=" * 72)
    print("ALL SELF-TESTS PASSED.")
    print("=" * 72)


if __name__ == "__main__":
    main()

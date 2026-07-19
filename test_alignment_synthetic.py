"""
Sanity test for align_embeddings.py using SYNTHETIC data with a KNOWN ground
truth rotation. This is the standard way to validate this kind of bilingual
lexicon induction machinery (see Conneau et al. 2018's synthetic experiments):
if we can't recover a rotation we constructed ourselves, there is a bug, full
stop -- no amount of real-data ambiguity can be blamed.

Construction:
  - X_en: n "concepts" arranged in well-separated Gaussian clusters (this
    gives real, non-rotation-symmetric local geometric structure -- an
    isotropic Gaussian blob would be indistinguishable from any rotation of
    itself, which would make this a vacuous test).
  - Q: a random orthogonal matrix (via QR of a random Gaussian matrix).
  - X_fi = X_en @ Q (+ small independent noise, since real cross-lingual
    embedding spaces are only approximately, not perfectly, isometric).
  - Ground truth: row i of X_fi corresponds to row i of X_en, and the true
    unmixing matrix is W_true = Q (derived analytically in the accompanying
    notes -- X_fi @ Q^T = X_en @ Q @ Q^T = X_en since Q is orthogonal).

We then run ONLY the unsupervised pipeline (adversarial align -> Procrustes
refine) and check nearest-neighbour precision@1 against the known pairing.
"""
import numpy as np

from align_embeddings import (
    adversarial_align, refine_with_procrustes, nn_precision_at_1,
    unsupervised_alignment_score, csls_mutual_nn, _normalize_rows,
    similarity_profile_seed_dictionary, procrustes_solve, align_embedding_spaces,
)


def make_synthetic(n_words=3000, dim=64, n_clusters=40, within_cluster_std=0.4,
                    cross_lingual_noise_std=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(n_clusters, dim))
    assign = rng.integers(0, n_clusters, size=n_words)
    X_en = centers[assign] + rng.normal(scale=within_cluster_std, size=(n_words, dim))

    # random orthogonal Q via QR decomposition of a random Gaussian matrix
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))  # canonicalize sign so Q is a "clean" rotation/reflection

    X_fi = X_en @ Q + rng.normal(scale=cross_lingual_noise_std, size=(n_words, dim))

    # Zipfian frequency ranks -> sort both matrices by descending frequency,
    # since the real pipeline restricts adversarial training to frequent words.
    ranks = rng.permutation(n_words)  # arbitrary "true" frequency order for this test
    order = np.argsort(ranks)
    X_en, X_fi = X_en[order].astype(np.float32), X_fi[order].astype(np.float32)

    gold_pairs = np.stack([np.arange(n_words), np.arange(n_words)], axis=1)  # row i <-> row i
    return X_en, X_fi, Q.astype(np.float32), gold_pairs


def main():
    print("Building synthetic cluster-structured bilingual embedding space...")
    X_en, X_fi, Q_true, gold_pairs = make_synthetic()
    print(f"  X_en {X_en.shape}, X_fi {X_fi.shape}")

    # Baseline: precision@1 with NO alignment at all (raw CSLS in original spaces)
    # should be near-random, confirming the task is non-trivial without alignment.
    baseline_prec = nn_precision_at_1(X_fi, X_en, gold_pairs, k=10)
    print(f"Baseline precision@1 (no alignment, raw spaces): {baseline_prec:.3f}")

    print("\n--- FULLY UNSUPERVISED pipeline: similarity-profile self-learning init ---")
    print("    (no adversarial training, no seed dictionary, no parallel data)")
    seed_pairs = similarity_profile_seed_dictionary(X_fi, X_en, top_k=3000, profile_len=2000, csls_k=10)
    print(f"induced {len(seed_pairs)} seed pairs from similarity-profile matching alone")
    seed_correct = np.mean(seed_pairs[:, 0] == seed_pairs[:, 1]) if len(seed_pairs) else 0.0
    print(f"  fraction of induced seed pairs that are actually correct: {seed_correct:.3f}")

    W_final, unsup_score, n_seed = align_embedding_spaces(
        X_fi, X_en, top_k=3000, profile_len=2000, n_refine_iters=5, csls_k=10,
    )
    prec_final = nn_precision_at_1(X_fi @ W_final.T, X_en, gold_pairs, k=10)
    frob_err = np.linalg.norm(W_final - Q_true, ord='fro') / np.linalg.norm(Q_true, ord='fro')
    print(f"\nAfter full self-learning + iterative Procrustes/CSLS refinement:")
    print(f"  precision@1 = {prec_final:.3f}")
    print(f"  relative Frobenius error vs true rotation = {frob_err:.4f}")
    print(f"  unsupervised validation score = {unsup_score:.4f}")

    print("\n--- For comparison: the adversarial (GAN) aligner, kept as an optional path ---")
    W_adv = adversarial_align(X_fi, X_en, top_k=3000, epochs=3, iters_per_epoch=300,
                               batch_size=32, disc_lr=0.1, map_lr=0.1, seed=0)
    prec_adv = nn_precision_at_1(X_fi @ W_adv.T, X_en, gold_pairs, k=10)
    print(f"adversarial-only precision@1: {prec_adv:.3f}  (this is why it's not the default)")

    print("\n=== PASS/FAIL (evaluating the pipeline actually used, i.e. self-learning) ===")
    ok = True
    if baseline_prec > 0.05:
        print("FAIL: baseline precision suspiciously high -- synthetic task may be degenerate")
        ok = False
    if seed_correct < 0.3:
        print(f"FAIL: similarity-profile seed dictionary too noisy ({seed_correct:.3f} correct) to bootstrap from")
        ok = False
    if prec_final < 0.85:
        print(f"FAIL: final precision@1 = {prec_final:.3f}, expected > 0.85 on this clean synthetic setup")
        ok = False
    if frob_err > 0.3:
        print(f"FAIL: recovered W too far from true Q (rel. Frobenius error {frob_err:.3f})")
        ok = False
    if ok:
        print(f"PASS: fully-unsupervised alignment pipeline recovers the planted rotation "
              f"(precision@1={prec_final:.3f}, rel. error={frob_err:.4f}, seed dict accuracy={seed_correct:.3f})")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

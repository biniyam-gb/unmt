"""
Stage A: unsupervised cross-lingual embedding alignment.

Purpose: give the shared Transformer's token-embedding table a warm start in
which semantically-corresponding EN/FI subwords already sit near each other
in vector space, BEFORE any back-translation happens. Without this, round-1
back-translations from a randomly-initialized model are pure noise and the
BT bootstrap has nothing to climb from (this is exactly Step 1 of the
Lample et al. 2018 / Artetxe et al. 2018 pipelines).

Pipeline actually used (see test_alignment_synthetic.py for the empirical
justification of this choice over pure adversarial training):
  1. Train monolingual skip-gram embeddings separately per language (same
     shared vocabulary ids, but each matrix only has real signal for the
     tokens that actually occur in that language's corpus).
  2. Robust self-learning initialization (Artetxe et al. 2018): induce a seed
     dictionary by matching words on the SHAPE of their own-language
     similarity profile -- a descriptor that's invariant to any orthogonal
     transform of that language's space, so it gives real signal with zero
     cross-lingual information. Deterministic, no adversarial training.
  3. Iterative Procrustes refinement: induce a synthetic dictionary via CSLS
     mutual-nearest-neighbours, solve orthogonal Procrustes in closed form
     (SVD), repeat.
  4. Unsupervised model-selection criterion (mean CSLS of mutual NN pairs
     among frequent words) since there is, by construction, no labelled
     dictionary to validate against.

An adversarial (GAN-style) aligner is also implemented below and kept
available, but is NOT the default path: on a controlled synthetic test with a
known ground-truth rotation, it converged to a confidently wrong answer
(discriminator accuracy climbed to >90% while the recovered mapping moved
AWAY from the true rotation) even while the Procrustes+CSLS refinement loop,
fed a small correct seed, recovered the true rotation to <1% relative
Frobenius error. This matches published findings that the adversarial step in
Conneau et al. 2018 is the least reliable part of that pipeline (Artetxe et
al. 2018 replaced it for exactly this reason). Use adversarial_align only if
you want to experiment with it; similarity_profile_seed_dictionary is what
train_full_alignment() below actually calls.

No parallel data or seed dictionary is used anywhere in this file.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Monolingual skip-gram embeddings (negative sampling), pure PyTorch.
#    Kept dependency-free (no gensim) to avoid numpy/gensim version conflicts
#    on Kaggle images, which change often enough to be a real failure mode.
# ---------------------------------------------------------------------------
def sort_by_frequency(emb: np.ndarray, freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (emb_sorted_by_descending_freq, permutation) where
    emb_sorted[i] = emb[permutation[i]]. The alignment functions in this file
    (top_k restriction, similarity-profile matching) require frequency-sorted
    input; W itself is row-order-agnostic once learned, so callers should sort
    ONLY for the alignment call and apply the resulting W directly to the
    original, vocab-id-indexed embedding matrix -- no need to carry the
    permutation any further than that."""
    perm = np.argsort(-freq)
    return emb[perm], perm


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, pos_context, neg_context):
        v_c = self.in_emb(center)                                     # (B, D)
        v_o = self.out_emb(pos_context)                               # (B, D)
        v_neg = self.out_emb(neg_context)                             # (B, K, D)
        pos_score = torch.sum(v_c * v_o, dim=-1)                      # (B,)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)   # (B, K)
        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum(dim=-1)
        return loss.mean()


def build_skipgram_pairs(token_id_sequences: List[List[int]], window: int, rng: np.random.Generator):
    for seq in token_id_sequences:
        n = len(seq)
        for i, center in enumerate(seq):
            w = int(rng.integers(1, window + 1))
            lo, hi = max(0, i - w), min(n, i + w + 1)
            for j in range(lo, hi):
                if j != i:
                    yield center, seq[j]


def train_skipgram_embeddings(
    token_id_sequences: List[List[int]],
    vocab_size: int,
    dim: int = 512,
    window: int = 5,
    n_negs: int = 5,
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (embedding_matrix[vocab_size, dim], token_counts[vocab_size])."""
    rng = np.random.default_rng(seed)

    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in token_id_sequences:
        for t in seq:
            counts[t] += 1
    neg_probs = np.power(counts.astype(np.float64) + 1.0, 0.75)  # unigram^0.75, Mikolov et al. 2013
    neg_probs /= neg_probs.sum()

    model = SkipGramNS(vocab_size, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pairs = list(build_skipgram_pairs(token_id_sequences, window, rng))
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    n_batches = max(1, len(pairs) // batch_size)

    for ep in range(epochs):
        total_loss, n_seen = 0.0, 0
        for b in range(n_batches):
            batch = pairs[b * batch_size:(b + 1) * batch_size]
            if not batch:
                continue
            centers = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            ctx = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
            negs = torch.tensor(
                rng.choice(vocab_size, size=(len(batch), n_negs), p=neg_probs),
                dtype=torch.long, device=device,
            )
            opt.zero_grad()
            loss = model(centers, ctx, negs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_seen += 1
        if n_seen:
            print(f"[skipgram] epoch {ep+1}/{epochs} avg loss {total_loss / n_seen:.4f}")

    emb = model.in_emb.weight.detach().cpu().numpy()
    return emb, counts


# ---------------------------------------------------------------------------
# 2. CSLS similarity (Conneau et al. 2018, eq. 4-5) -- corrects for hubness
#    that plain cosine-nearest-neighbour retrieval suffers from in high dim.
# ---------------------------------------------------------------------------
def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return x / norm


def csls_scores(X: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    """X: (n, d) queries (already mapped into Y's space), Y: (m, d) targets.
    Returns (n, m) CSLS similarity matrix.
    CSLS(x,y) = 2*cos(x,y) - r_T(x) - r_S(y)
    r_T(x) = mean cos-sim of x to its k nearest neighbours in Y
    r_S(y) = mean cos-sim of y to its k nearest neighbours in X
    """
    Xn, Yn = _normalize_rows(X), _normalize_rows(Y)
    cos = Xn @ Yn.T  # (n, m)

    k_x = min(k, Yn.shape[0])
    r_t = np.sort(cos, axis=1)[:, -k_x:].mean(axis=1)  # (n,)

    cos_yx = Yn @ Xn.T  # (m, n)
    k_y = min(k, Xn.shape[0])
    r_s = np.sort(cos_yx, axis=1)[:, -k_y:].mean(axis=1)  # (m,)

    return 2 * cos - r_t[:, None] - r_s[None, :]


def csls_mutual_nn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    """Returns an (P, 2) array of index pairs (i, j) that are MUTUAL nearest
    neighbours under CSLS. This high-precision filter is what turns a noisy
    similarity matrix into a usable synthetic dictionary for Procrustes."""
    S = csls_scores(X, Y, k=k)
    nn_xy = S.argmax(axis=1)
    nn_yx = S.argmax(axis=0)
    pairs = [(i, j) for i, j in enumerate(nn_xy) if nn_yx[j] == i]
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


# ---------------------------------------------------------------------------
# 3. Adversarial alignment: learn orthogonal W with a GAN-style game
#    (Conneau, Lample et al. 2018, Sec 3.1)
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 3. Initialization heuristic: robust self-learning (Artetxe, Labaka & Agirre,
#    ACL 2018, "A robust self-learning method for fully unsupervised
#    cross-lingual mappings of word embeddings"). We use this INSTEAD of pure
#    adversarial training as the seed-dictionary initializer.
#
#    Why: a word's SORTED similarity profile to its own vocabulary (i.e. "how
#    similar is word i to its 1st, 2nd, 3rd, ... nearest neighbour in its own
#    language") is a descriptor that is invariant to any orthogonal
#    transformation of that language's embedding space. If the two embedding
#    spaces really are approximately isometric, a word and its true
#    translation should have SIMILAR sorted-similarity profiles even before
#    any cross-lingual alignment exists. Matching on these profiles gives a
#    deterministic seed dictionary with no adversarial training at all.
#
#    We validated this empirically (test_alignment_synthetic.py): on synthetic
#    cluster-structured data, adversarial GAN training converged to the WRONG
#    rotation (discriminator accuracy went up, but recovered W moved AWAY from
#    the true rotation), while Procrustes+CSLS refinement from a decent seed
#    converged essentially exactly (>99% precision@1, <1% relative Frobenius
#    error). This matches the wider literature's finding that the adversarial
#    step in Conneau et al. 2018 is the least reliable component of the
#    pipeline -- so we do not depend on it.
# ---------------------------------------------------------------------------
def similarity_profile_seed_dictionary(
    X_src: np.ndarray, Y_tgt: np.ndarray,
    top_k: int = 5000, profile_len: int = 2000, csls_k: int = 10,
) -> np.ndarray:
    """Returns an (P, 2) array of induced (src_idx, tgt_idx) seed pairs, with
    NO adversarial training and NO parallel data -- only each language's own
    internal similarity structure."""
    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    Xf = _normalize_rows(X_src[:kx])
    Yf = _normalize_rows(Y_tgt[:ky])

    sim_x = Xf @ Xf.T  # (kx, kx) within-language similarity
    sim_y = Yf @ Yf.T  # (ky, ky)

    plen = min(profile_len, kx, ky)
    prof_x = -np.sort(-sim_x, axis=1)[:, :plen]  # sorted descending, truncated
    prof_y = -np.sort(-sim_y, axis=1)[:, :plen]

    # profile vectors are directly comparable across languages (both are just
    # "shape of own-language similarity decay"), so we can CSLS-match them
    # even though the RAW embeddings aren't aligned yet.
    prof_x_n = _normalize_rows(prof_x)
    prof_y_n = _normalize_rows(prof_y)
    return csls_mutual_nn(prof_x_n, prof_y_n, k=csls_k)


def orthogonalize(W: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """W <- (1+beta) W - beta (W W^T) W. Keeps W close to the orthogonal
    manifold without a full SVD projection every step (Conneau et al. 2018).
    Retained for optional use inside adversarial_align below."""
    return (1 + beta) * W - beta * (W @ W.t()) @ W


def adversarial_align(
    X_src: np.ndarray, Y_tgt: np.ndarray,
    top_k: int = 20000, epochs: int = 5, iters_per_epoch: int = 2000,
    batch_size: int = 32, disc_lr: float = 0.1, map_lr: float = 0.1,
    smoothing: float = 0.2, beta_ortho: float = 0.01,
    device: str = "cpu", seed: int = 0,
) -> np.ndarray:
    """Learn W (d x d) such that X_src @ W.T lands in Y_tgt's distribution.
    Restricts adversarial training to the top_k most frequent rows of each
    matrix (both arrays expected pre-sorted by descending frequency) --
    rare-word embeddings are too noisy to stabilize the GAN game.
    Returns W as a numpy array (d, d)."""
    torch.manual_seed(seed)
    d = X_src.shape[1]
    assert Y_tgt.shape[1] == d

    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    X = torch.tensor(X_src[:kx], dtype=torch.float32, device=device)
    Y = torch.tensor(Y_tgt[:ky], dtype=torch.float32, device=device)
    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)

    W_param = nn.Parameter(torch.eye(d, device=device))
    disc = Discriminator(d).to(device)

    opt_d = torch.optim.SGD(disc.parameters(), lr=disc_lr)
    opt_w = torch.optim.SGD([W_param], lr=map_lr)

    for ep in range(epochs):
        d_losses, w_losses = [], []
        for _ in range(iters_per_epoch):
            src_idx = torch.randint(0, X.size(0), (batch_size,), device=device)
            tgt_idx = torch.randint(0, Y.size(0), (batch_size,), device=device)
            x_batch = X[src_idx]
            y_batch = Y[tgt_idx]

            # --- discriminator step ---
            with torch.no_grad():
                mapped = x_batch @ W_param.t()
            d_in = torch.cat([mapped, y_batch], dim=0)
            d_labels = torch.cat([
                torch.full((batch_size,), smoothing, device=device),        # mapped src = "fake" (~0)
                torch.full((batch_size,), 1.0 - smoothing, device=device),  # real tgt = "real" (~1)
            ])
            opt_d.zero_grad()
            d_out = disc(d_in)
            d_loss = F.binary_cross_entropy_with_logits(d_out, d_labels)
            d_loss.backward()
            opt_d.step()
            d_losses.append(d_loss.item())

            # --- mapping (generator) step: fool the discriminator ---
            src_idx = torch.randint(0, X.size(0), (batch_size,), device=device)
            x_batch = X[src_idx]
            opt_w.zero_grad()
            mapped = x_batch @ W_param.t()
            d_out = disc(mapped)
            w_loss = F.binary_cross_entropy_with_logits(
                d_out, torch.full((batch_size,), 1.0 - smoothing, device=device)
            )
            w_loss.backward()
            opt_w.step()
            with torch.no_grad():
                W_param.copy_(orthogonalize(W_param, beta_ortho))
            w_losses.append(w_loss.item())

        print(f"[adv-align] epoch {ep+1}/{epochs}  D_loss={np.mean(d_losses):.4f}  W_loss={np.mean(w_losses):.4f}")

    return W_param.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Iterative Procrustes refinement (closed-form SVD solution)
# ---------------------------------------------------------------------------
def procrustes_solve(X_paired: np.ndarray, Y_paired: np.ndarray) -> np.ndarray:
    """Closed-form orthogonal Procrustes: argmin_W ||X W^T - Y||_F s.t. W^T W = I.
    Solution: M = Y^T X, SVD M = U S V^T, W* = U V^T (Schonemann 1966)."""
    M = Y_paired.T @ X_paired
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def d_min_pairs_guard(d: int) -> int:
    # need at least d+1 pairs for the Procrustes SVD to be well-conditioned;
    # we want a larger safety margin than the bare minimum in practice.
    return max(50, 4 * d)


def refine_with_procrustes(
    X_src: np.ndarray, Y_tgt: np.ndarray, W_init: np.ndarray,
    n_iters: int = 5, top_k_for_dict: int = 20000, csls_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alternates: (a) map X with current W, induce a mutual-NN dictionary via
    CSLS restricted to frequent words, (b) re-solve W in closed form. Returns
    (W_final, last_induced_pairs)."""
    W = W_init.copy()
    kx = min(top_k_for_dict, X_src.shape[0])
    ky = min(top_k_for_dict, Y_tgt.shape[0])
    Xf = _normalize_rows(X_src[:kx])
    Yf = _normalize_rows(Y_tgt[:ky])

    pairs = np.zeros((0, 2), dtype=np.int64)
    guard = d_min_pairs_guard(Xf.shape[1])
    for it in range(n_iters):
        mapped = Xf @ W.T
        pairs = csls_mutual_nn(mapped, Yf, k=csls_k)
        if len(pairs) < guard:
            print(f"[procrustes] iter {it+1}: only {len(pairs)} mutual-NN pairs found (<{guard}); stopping early")
            break
        W = procrustes_solve(Xf[pairs[:, 0]], Yf[pairs[:, 1]])
        print(f"[procrustes] iter {it+1}/{n_iters}: {len(pairs)} mutual-NN pairs induced")
    return W, pairs


# ---------------------------------------------------------------------------
# 5. Unsupervised validation criterion (Conneau et al. 2018, Sec 3.2): mean
#    CSLS similarity of induced mutual-NN pairs among frequent words. Used
#    for model selection since no labelled dictionary exists to validate
#    against directly.
# ---------------------------------------------------------------------------
def unsupervised_alignment_score(X_src: np.ndarray, Y_tgt: np.ndarray, W: np.ndarray,
                                  top_k: int = 10000, csls_k: int = 10) -> float:
    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    Xf = _normalize_rows(X_src[:kx]) @ W.T
    Yf = _normalize_rows(Y_tgt[:ky])
    pairs = csls_mutual_nn(Xf, Yf, k=csls_k)
    if len(pairs) == 0:
        return float("-inf")
    S = csls_scores(Xf[pairs[:, 0]], Yf[pairs[:, 1]], k=csls_k)
    return float(np.diag(S).mean())


def align_embedding_spaces(
    X_src: np.ndarray, Y_tgt: np.ndarray,
    top_k: int = 5000, profile_len: int = 2000,
    n_refine_iters: int = 5, csls_k: int = 10,
) -> Tuple[np.ndarray, float, int]:
    """Full Stage-A pipeline: self-learning seed induction -> initial
    Procrustes solve -> iterative CSLS+Procrustes refinement.
    Returns (W, unsupervised_validation_score, n_seed_pairs_found).
    X_src, Y_tgt must be pre-sorted by descending frequency."""
    seed_pairs = similarity_profile_seed_dictionary(X_src, Y_tgt, top_k=top_k,
                                                     profile_len=profile_len, csls_k=csls_k)
    n_seed = len(seed_pairs)
    guard = d_min_pairs_guard(X_src.shape[1])
    if n_seed < guard:
        raise RuntimeError(
            f"Only {n_seed} seed pairs induced from similarity profiles (need >= {guard}). "
            "This means the two embedding spaces are not similar enough in internal structure "
            "for this method to bootstrap -- see the honest failure mode discussed in the "
            "conversation this code came from (Guzman et al. 2019's Nepali/Sinhala result). "
            "Try: more monolingual data, a larger profile_len, or accept that this language "
            "pair may need a different approach (comparable corpora, a pivot language, etc.)."
        )
    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    Xf, Yf = X_src[:kx], Y_tgt[:ky]
    W0 = procrustes_solve(Xf[seed_pairs[:, 0]], Yf[seed_pairs[:, 1]])
    W_final, _ = refine_with_procrustes(X_src, Y_tgt, W0, n_iters=n_refine_iters,
                                         top_k_for_dict=top_k, csls_k=csls_k)
    score = unsupervised_alignment_score(X_src, Y_tgt, W_final, top_k=top_k, csls_k=csls_k)
    return W_final, score, n_seed


def nn_precision_at_1(X_mapped: np.ndarray, Y_true: np.ndarray, gold_pairs: np.ndarray, k: int = 10) -> float:
    """Diagnostic used ONLY by our synthetic unit test (tests/test_alignment.py),
    where ground truth is known by construction. NEVER used in the real
    pipeline, since real cross-lingual ground truth doesn't exist here."""
    S = csls_scores(_normalize_rows(X_mapped), _normalize_rows(Y_true), k=k)
    pred = S.argmax(axis=1)
    gold = {i: j for i, j in gold_pairs}
    correct = sum(1 for i in range(X_mapped.shape[0]) if gold.get(i, -1) == pred[i])
    return correct / X_mapped.shape[0]

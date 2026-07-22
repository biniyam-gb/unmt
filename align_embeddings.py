"""
Stage A: unsupervised cross-lingual embedding alignment.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sort_by_frequency(
    emb: np.ndarray, freq: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
        v_c = self.in_emb(center)
        v_o = self.out_emb(pos_context)
        v_neg = self.out_emb(neg_context)
        pos_score = torch.sum(v_c * v_o, dim=-1)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)
        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum(dim=-1)
        return loss.mean()


def build_skipgram_pairs(
    token_id_sequences: List[List[int]], window: int, rng: np.random.Generator
):
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
    rng = np.random.default_rng(seed)

    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in token_id_sequences:
        for t in seq:
            counts[t] += 1
    neg_probs = np.power(counts.astype(np.float64) + 1.0, 0.75)
    neg_probs /= neg_probs.sum()

    model = SkipGramNS(vocab_size, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pairs = list(build_skipgram_pairs(token_id_sequences, window, rng))
    if not pairs:
        return model.in_emb.weight.detach().cpu().numpy(), counts

    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    n_batches = max(1, len(pairs) // batch_size)

    for ep in range(epochs):
        total_loss, n_seen = 0.0, 0
        for b in range(n_batches):
            batch = pairs[b * batch_size : (b + 1) * batch_size]
            if not batch:
                continue
            centers = torch.tensor(
                [p[0] for p in batch], dtype=torch.long, device=device
            )
            ctx = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
            negs = torch.tensor(
                rng.choice(vocab_size, size=(len(batch), n_negs), p=neg_probs),
                dtype=torch.long,
                device=device,
            )
            opt.zero_grad()
            loss = model(centers, ctx, negs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_seen += 1
        if n_seen:
            print(
                f"[skipgram] epoch {ep + 1}/{epochs} avg loss {total_loss / n_seen:.4f}"
            )

    emb = model.in_emb.weight.detach().cpu().numpy()
    return emb, counts


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return x / norm


def csls_scores(X: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    Xn, Yn = _normalize_rows(X), _normalize_rows(Y)
    cos = Xn @ Yn.T

    k_x = min(k, Yn.shape[0])
    r_t = np.sort(cos, axis=1)[:, -k_x:].mean(axis=1)

    cos_yx = Yn @ Xn.T
    k_y = min(k, Xn.shape[0])
    r_s = np.sort(cos_yx, axis=1)[:, -k_y:].mean(axis=1)

    return 2 * cos - r_t[:, None] - r_s[None, :]


def csls_mutual_nn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> np.ndarray:
    S = csls_scores(X, Y, k=k)
    nn_xy = S.argmax(axis=1)
    nn_yx = S.argmax(axis=0)
    pairs = [(i, j) for i, j in enumerate(nn_xy) if nn_yx[j] == i]
    return (
        np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
    )


def similarity_profile_seed_dictionary(
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    top_k: int = 5000,
    profile_len: int = 2000,
    csls_k: int = 10,
) -> np.ndarray:
    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    Xf = _normalize_rows(X_src[:kx])
    Yf = _normalize_rows(Y_tgt[:ky])

    sim_x = Xf @ Xf.T
    sim_y = Yf @ Yf.T

    plen = min(profile_len, kx, ky)
    prof_x = -np.sort(-sim_x, axis=1)[:, :plen]
    prof_y = -np.sort(-sim_y, axis=1)[:, :plen]

    prof_x_n = _normalize_rows(prof_x)
    prof_y_n = _normalize_rows(prof_y)
    return csls_mutual_nn(prof_x_n, prof_y_n, k=csls_k)


def procrustes_solve(X_paired: np.ndarray, Y_paired: np.ndarray) -> np.ndarray:
    M = Y_paired.T @ X_paired
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def d_min_pairs_guard(d: int) -> int:
    # Requires at least d + 10 pairs (522 for d=512) so toy runs work cleanly
    return max(50, d + 10)


def refine_with_procrustes(
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    W_init: np.ndarray,
    n_iters: int = 5,
    top_k_for_dict: int = 20000,
    csls_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
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
            print(
                f"[procrustes] iter {it + 1}: only {len(pairs)} mutual-NN pairs found (<{guard}); stopping early"
            )
            break
        W = procrustes_solve(Xf[pairs[:, 0]], Yf[pairs[:, 1]])
        print(
            f"[procrustes] iter {it + 1}/{n_iters}: {len(pairs)} mutual-NN pairs induced"
        )
    return W, pairs


def unsupervised_alignment_score(
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    W: np.ndarray,
    top_k: int = 10000,
    csls_k: int = 10,
) -> float:
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
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    top_k: int = 5000,
    profile_len: int = 2000,
    n_refine_iters: int = 5,
    csls_k: int = 10,
) -> Tuple[np.ndarray, float, int]:
    seed_pairs = similarity_profile_seed_dictionary(
        X_src, Y_tgt, top_k=top_k, profile_len=profile_len, csls_k=csls_k
    )
    n_seed = len(seed_pairs)
    guard = d_min_pairs_guard(X_src.shape[1])
    if n_seed < guard:
        raise RuntimeError(
            f"Only {n_seed} seed pairs induced from similarity profiles (need >= {guard})."
        )
    kx = min(top_k, X_src.shape[0])
    ky = min(top_k, Y_tgt.shape[0])
    Xf, Yf = X_src[:kx], Y_tgt[:ky]
    W0 = procrustes_solve(Xf[seed_pairs[:, 0]], Yf[seed_pairs[:, 1]])
    W_final, _ = refine_with_procrustes(
        X_src, Y_tgt, W0, n_iters=n_refine_iters, top_k_for_dict=top_k, csls_k=csls_k
    )
    score = unsupervised_alignment_score(
        X_src, Y_tgt, W_final, top_k=top_k, csls_k=csls_k
    )
    return W_final, score, n_seed

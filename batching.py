"""
Token-budget dynamic batching. Padding a batch of very different lengths to
the longest sequence wastes a lot of compute on T4s, which are not fast to
begin with -- so we batch by total token BUDGET, not sentence count, and use
the standard "sort locally, shuffle globally" trick:

  1. Shuffle all sentence indices for the epoch.
  2. Walk through in contiguous chunks (e.g. 20,000 indices) and sort WITHIN
     each chunk by length -- this clusters similar lengths together so batches
     built from a chunk have little padding waste.
  3. Build token-budget-bounded batches from that locally-sorted order.
  4. Shuffle the ORDER of the resulting batches (not their contents) so
     training doesn't see a systematic drift from short to long sentences
     over the course of an epoch, which would otherwise skew early-epoch
     gradient statistics and effective learning rate.
"""
import random
from typing import List, Iterator

import numpy as np
import torch

from binarize import BinarizedCorpus
from config import PAD_ID, BOS_ID, EOS_ID, MAX_TOKENS_PER_BATCH


def make_batches(corpus: BinarizedCorpus, max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
                  chunk_size: int = 20000, seed: int = 0, max_sentences_per_batch: int = 256,
                  ) -> List[List[int]]:
    n = len(corpus)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)

    lengths = corpus.offsets[1:] - corpus.offsets[:-1]  # O(1), no per-sentence lookup needed

    batches: List[List[int]] = []
    for chunk_start in range(0, n, chunk_size):
        chunk = order[chunk_start:chunk_start + chunk_size]
        chunk.sort(key=lambda i: lengths[i])
        cur_batch: List[int] = []
        cur_max_len = 0
        for idx in chunk:
            L = int(lengths[idx]) + 2  # +2 for BOS/EOS added at collation time
            new_max_len = max(cur_max_len, L)
            if cur_batch and (new_max_len * (len(cur_batch) + 1) > max_tokens_per_batch
                              or len(cur_batch) >= max_sentences_per_batch):
                batches.append(cur_batch)
                cur_batch, cur_max_len = [], 0
                new_max_len = L
            cur_batch.append(idx)
            cur_max_len = new_max_len
        if cur_batch:
            batches.append(cur_batch)

    rng.shuffle(batches)
    return batches


def collate_batch(corpus: BinarizedCorpus, indices: List[int], device="cpu") -> torch.Tensor:
    """Returns a (B, T) LongTensor with BOS/EOS added and PAD-padded to the
    batch's own max length (not a fixed global max_len)."""
    seqs = [corpus[i] for i in indices]
    max_len = max(len(s) for s in seqs) + 2
    B = len(seqs)
    out = np.full((B, max_len), PAD_ID, dtype=np.int64)
    for b, s in enumerate(seqs):
        out[b, 0] = BOS_ID
        out[b, 1:1 + len(s)] = s
        out[b, 1 + len(s)] = EOS_ID
    return torch.from_numpy(out).to(device)


def infinite_batch_iterator(corpus: BinarizedCorpus, max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
                             seed: int = 0, device="cpu") -> Iterator[torch.Tensor]:
    """Cycles through the corpus forever, re-shuffling (re-bucketing) each
    pass, yielding collated CLEAN (B, T) tensors -- no noise applied. Used by
    the back-translation loop, which needs clean source sentences to generate
    synthetic translations from (the "noise" in BT comes from the model's own
    imperfect translation, not from an artificial corruption)."""
    epoch = 0
    while True:
        batches = make_batches(corpus, max_tokens_per_batch, seed=seed + epoch)
        for indices in batches:
            yield collate_batch(corpus, indices, device=device)
        epoch += 1


def collate_dae_batch(corpus: BinarizedCorpus, indices: List[int], device="cpu"):
    """Returns (noised_input (B,S), clean_full (B,T)) for denoising-autoencoder
    training: encoder sees a noised version of the sentence, decoder is
    trained to reconstruct the ORIGINAL clean sentence (BOS-prefixed,
    EOS-suffixed; caller shifts by one for teacher forcing: input=clean[:, :-1],
    target=clean[:, 1:])."""
    from noise import noise_sentence
    clean_seqs = [corpus[i].tolist() for i in indices]
    noised_seqs = [noise_sentence(s) for s in clean_seqs]

    def pack(seqs):
        max_len = max(len(s) for s in seqs) + 2
        out = np.full((len(seqs), max_len), PAD_ID, dtype=np.int64)
        for b, s in enumerate(seqs):
            out[b, 0] = BOS_ID
            out[b, 1:1 + len(s)] = s
            out[b, 1 + len(s)] = EOS_ID
        return torch.from_numpy(out).to(device)

    return pack(noised_seqs), pack(clean_seqs)


def infinite_dae_batch_iterator(corpus: BinarizedCorpus, max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
                                 seed: int = 0, device="cpu"):
    epoch = 0
    while True:
        batches = make_batches(corpus, max_tokens_per_batch, seed=seed + epoch)
        for indices in batches:
            yield collate_dae_batch(corpus, indices, device=device)
        epoch += 1

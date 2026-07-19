"""
Noise model for the denoising-autoencoding stage (Stage B).

Two noise types, exactly as in Lample et al. 2018 ("Unsupervised Machine
Translation Using Monolingual Corpora Only", Sec 2.3):
  1. Word dropout: delete each token independently with probability p.
  2. Local shuffle: permute tokens so that no token moves more than ~k
     positions from where it started (add U(0, k) noise to each position
     index, then sort by the noisy index).

The model is trained to reconstruct the ORIGINAL (clean) sentence from the
NOISY one. This forces the encoder to build representations robust to word
order/presence -- which back-translation later exploits, since round-trip
translations are themselves a noisy channel.

Special tokens (PAD, BOS, EOS) are never touched by the noise functions.
"""
import random
from typing import List

from config import PAD_ID, BOS_ID, EOS_ID, WORD_DROP_PROB, SHUFFLE_WINDOW_K


def word_dropout(tokens: List[int], p: float = WORD_DROP_PROB) -> List[int]:
    if len(tokens) <= 1:
        return list(tokens)
    kept = [t for t in tokens if random.random() > p]
    if len(kept) == 0:  # never drop everything
        kept = [random.choice(tokens)]
    return kept


def local_shuffle(tokens: List[int], k: int = SHUFFLE_WINDOW_K) -> List[int]:
    if len(tokens) <= 1 or k <= 0:
        return list(tokens)
    noisy_idx = [i + random.uniform(0, k + 1) for i in range(len(tokens))]
    order = sorted(range(len(tokens)), key=lambda i: noisy_idx[i])
    return [tokens[i] for i in order]


def noise_sentence(token_ids: List[int], drop_p: float = WORD_DROP_PROB,
                    shuffle_k: int = SHUFFLE_WINDOW_K) -> List[int]:
    """token_ids: subword ids WITHOUT BOS/EOS (those get added by the caller
    at batch-collation time, after noising, so noise never touches them)."""
    x = local_shuffle(token_ids, shuffle_k)
    x = word_dropout(x, drop_p)
    return x


if __name__ == "__main__":
    random.seed(0)
    toks = list(range(10, 20))  # pretend subword ids
    print("original: ", toks)
    for _ in range(5):
        print("noised:   ", noise_sentence(toks))

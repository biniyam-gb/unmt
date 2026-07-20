"""
Stage A orchestration: turns two binarized monolingual corpora into a single
(vocab_size, d_model) initialization matrix for SharedTransformerNMT's
token_emb.weight, such that EN and FI subwords with similar meaning start
near each other in vector space before any back-translation happens.

Combination rule per vocab id v (see align_embeddings.py's docstring for why
adversarial init is NOT used here):
  - seen only in EN  -> use the EN skip-gram vector as-is (EN is the space
                        everything gets mapped INTO)
  - seen only in FI  -> use the FI skip-gram vector mapped through W (the
                        learned rotation into EN's space)
  - seen in both     -> frequency-weighted average of the two (weighted
                        toward whichever language uses that subword more)
  - seen in neither (too rare to appear in the skip-gram training subset)
                     -> fresh random init, same distribution the model's own
                        default initializer uses, so there's no special-cased
                        discontinuity for the training loop to deal with
"""
import argparse
import os
import numpy as np

from config import MODEL_CFG, EMB_TOP_FREQ_FOR_ADVERSARIAL, LANG_A, LANG_B
from binarize import BinarizedCorpus, load_resolved_vocab_size
from align_embeddings import train_skipgram_embeddings, sort_by_frequency, align_embedding_spaces


def build_initialization(
    en_prefix: str, fi_prefix: str, out_path: str, vocab_size: int,
    dim: int = MODEL_CFG.d_model,
    skipgram_max_sentences: int = 2_000_000, skipgram_epochs: int = 3,
    top_k_for_alignment: int = EMB_TOP_FREQ_FOR_ADVERSARIAL,
    profile_len: int = 2000, n_refine_iters: int = 5, seed: int = 0,
):
    en = BinarizedCorpus(en_prefix)
    fi = BinarizedCorpus(fi_prefix)

    print(f"Training EN skip-gram embeddings on up to {skipgram_max_sentences} sentences...")
    en_seqs = list(en.iter_as_lists(limit=skipgram_max_sentences))
    emb_en, freq_en = train_skipgram_embeddings(en_seqs, vocab_size, dim=dim,
                                                 epochs=skipgram_epochs, seed=seed)

    print(f"Training FI skip-gram embeddings on up to {skipgram_max_sentences} sentences...")
    fi_seqs = list(fi.iter_as_lists(limit=skipgram_max_sentences))
    emb_fi, freq_fi = train_skipgram_embeddings(fi_seqs, vocab_size, dim=dim,
                                                 epochs=skipgram_epochs, seed=seed + 1)

    print("Frequency-sorting for the alignment step...")
    emb_en_sorted, _ = sort_by_frequency(emb_en, freq_en)
    emb_fi_sorted, _ = sort_by_frequency(emb_fi, freq_fi)

    print("Running fully-unsupervised alignment (self-learning init + iterative Procrustes/CSLS)...")
    W, unsup_score, n_seed = align_embedding_spaces(
        emb_fi_sorted, emb_en_sorted,
        top_k=top_k_for_alignment, profile_len=profile_len,
        n_refine_iters=n_refine_iters,
    )
    print(f"Alignment done: {n_seed} seed pairs induced, unsupervised validation score={unsup_score:.4f}")

    # Apply W to the FULL (vocab-id-indexed, unsorted) Finnish matrix -- W is
    # a pure rotation of the embedding space and doesn't care about row order.
    emb_fi_mapped = emb_fi @ W.T

    rng = np.random.default_rng(seed + 2)
    combined = rng.normal(scale=dim ** -0.5, size=(vocab_size, dim)).astype(np.float32)

    both = (freq_en > 0) & (freq_fi > 0)
    only_en = (freq_en > 0) & (freq_fi == 0)
    only_fi = (freq_fi > 0) & (freq_en == 0)

    combined[only_en] = emb_en[only_en]
    combined[only_fi] = emb_fi_mapped[only_fi]
    w_en = freq_en[both].astype(np.float64)
    w_fi = freq_fi[both].astype(np.float64)
    total = (w_en + w_fi)[:, None]
    combined[both] = ((w_en[:, None] * emb_en[both] + w_fi[:, None] * emb_fi_mapped[both]) / total).astype(np.float32)

    n_unseen = vocab_size - both.sum() - only_en.sum() - only_fi.sum()
    print(f"Vocab coverage: {only_en.sum()} EN-only, {only_fi.sum()} FI-only, "
          f"{both.sum()} shared, {n_unseen} unseen-in-both (random init)")

    np.save(out_path, combined)
    print(f"Wrote combined initialization matrix to {out_path}  shape={combined.shape}")
    return combined, W, unsup_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--en_prefix", default=os.path.join(default_dir, f"bin.{LANG_A}"))
    ap.add_argument("--fi_prefix", default=os.path.join(default_dir, f"bin.{LANG_B}"))
    ap.add_argument("--out_path", default=os.path.join(default_dir, "init_embedding.npy"))
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    args = ap.parse_args()

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    MODEL_CFG.vocab_size = vocab_size  # keep the shared config object consistent for anything else reading it
    print(f"Using actual tokenizer vocab_size={vocab_size} (derived from {args.spm_model}, "
          f"not from a config default)")

    build_initialization(args.en_prefix, args.fi_prefix, args.out_path, vocab_size=vocab_size)

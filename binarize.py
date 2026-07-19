"""
Tokenize each filtered monolingual corpus ONCE with the shared SentencePiece
model and pack into a flat, memory-mappable format:
  - {lang}.tokens.npy   : uint16, all sentences concatenated back-to-back
                          (no BOS/EOS/lang tokens -- those get added at
                          batch-construction time by the training scripts,
                          since DAE and BT stages need different arrangements)
  - {lang}.offsets.npy  : int64, length n_sentences+1; sentence i occupies
                          tokens[offsets[i]:offsets[i+1]]
  - {lang}.freq.npy     : int64, shape (vocab_size,), raw token counts --
                          needed by align_embeddings.py, which requires its
                          inputs pre-sorted by descending frequency for the
                          top_k restriction to mean what it says.

uint16 is sufficient since vocab_size=32000 < 65536, and roughly halves
storage/IO versus int64 -- worth doing given we may be re-reading this from
disk every epoch across a 12-hour Kaggle session.
"""
import argparse
import os
import numpy as np
import sentencepiece as spm

from config import MIN_TOKENS_PER_SENT, MAX_TOKENS_PER_SENT, VOCAB_SIZE, LANG_A, LANG_B


def binarize_file(txt_path: str, spm_model_path: str, out_prefix: str,
                   min_len: int = MIN_TOKENS_PER_SENT, max_len: int = MAX_TOKENS_PER_SENT,
                   vocab_size: int = VOCAB_SIZE) -> dict:
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    assert sp.get_piece_size() <= 65536, "vocab_size must fit in uint16"

    all_ids = []
    offsets = [0]
    freq = np.zeros(vocab_size, dtype=np.int64)
    n_read = n_kept = 0

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            ids = sp.encode(line, out_type=int)
            if not (min_len <= len(ids) <= max_len):
                continue
            all_ids.extend(ids)
            offsets.append(offsets[-1] + len(ids))
            for t in ids:
                freq[t] += 1
            n_kept += 1

    tokens_arr = np.array(all_ids, dtype=np.uint16)
    offsets_arr = np.array(offsets, dtype=np.int64)

    np.save(out_prefix + ".tokens.npy", tokens_arr)
    np.save(out_prefix + ".offsets.npy", offsets_arr)
    np.save(out_prefix + ".freq.npy", freq)

    stats = {
        "n_read": n_read, "n_kept": n_kept,
        "n_tokens": int(tokens_arr.shape[0]),
        "mean_len": float(tokens_arr.shape[0] / max(n_kept, 1)),
    }
    print(f"[{out_prefix}] read={n_read} kept={n_kept} ({100*n_kept/max(n_read,1):.1f}%) "
          f"total_tokens={stats['n_tokens']} mean_len={stats['mean_len']:.1f}")
    return stats


class BinarizedCorpus:
    """Thin memmap-backed accessor: corpus[i] -> np.ndarray of token ids for
    sentence i, without re-loading the whole file into RAM."""

    def __init__(self, prefix: str):
        self.tokens = np.load(prefix + ".tokens.npy", mmap_mode="r")
        self.offsets = np.load(prefix + ".offsets.npy")
        self.freq = np.load(prefix + ".freq.npy")

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return np.asarray(self.tokens[lo:hi], dtype=np.int64)

    def iter_as_lists(self, limit: int = None):
        n = len(self) if limit is None else min(limit, len(self))
        for i in range(n):
            yield self[i].tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    args = ap.parse_args()

    for lang in (LANG_A, LANG_B):
        binarize_file(
            os.path.join(args.data_dir, f"mono.{lang}.txt"),
            args.spm_model,
            os.path.join(args.data_dir, f"bin.{lang}"),
        )

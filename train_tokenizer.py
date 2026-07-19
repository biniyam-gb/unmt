"""
Train ONE joint SentencePiece model on the concatenation of both filtered
monolingual corpora. A shared subword vocabulary is what lets the encoder/
decoder be genuinely shared across languages (Stage A's embedding alignment
also depends on both languages indexing into the same id space).
"""
import argparse
import os
import sentencepiece as spm

from config import VOCAB_SIZE, CHAR_COVERAGE, SPECIAL_TOKENS, LANG_A, LANG_B


def train_tokenizer(en_path: str, fi_path: str, out_prefix: str,
                     vocab_size: int = VOCAB_SIZE, char_coverage: float = CHAR_COVERAGE,
                     input_sentence_size: int = 5_000_000):
    spm.SentencePieceTrainer.train(
        input=f"{en_path},{fi_path}",
        model_prefix=out_prefix,
        vocab_size=vocab_size,
        character_coverage=char_coverage,
        model_type="unigram",
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        byte_fallback=True,           # graceful fallback for any unseen unicode (robustness net)
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        pad_piece=SPECIAL_TOKENS[0], bos_piece=SPECIAL_TOKENS[1],
        eos_piece=SPECIAL_TOKENS[2], unk_piece=SPECIAL_TOKENS[3],
        train_extremely_large_corpus=False,
    )
    print(f"Wrote {out_prefix}.model / {out_prefix}.vocab")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--en_path", default=os.path.join(default_dir, f"mono.{LANG_A}.txt"))
    ap.add_argument("--fi_path", default=os.path.join(default_dir, f"mono.{LANG_B}.txt"))
    ap.add_argument("--out_prefix", default=os.path.join(default_dir, "spm_joint"))
    ap.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()
    train_tokenizer(args.en_path, args.fi_path, args.out_prefix, args.vocab_size)

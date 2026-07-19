"""
Evaluation on FLORES+ devtest -- the ONLY place this ground truth is used.
Nothing here ever touches training; this script only loads a trained
checkpoint and scores it.

Metrics: chrF2 (Popovic 2015) is the primary metric -- it's character-n-gram
based, needs no external tokenizer download, and is well known to be far more
informative than word-level BLEU for morphologically rich languages like
Finnish (15 grammatical cases means a single lemma surfaces as dozens of
distinct word-forms, which word-level BLEU penalizes even when the
translation is correct). We also report spBLEU (flores200 SentencePiece
tokenization) when that tokenizer model can be fetched, with a fallback to
default-tokenizer BLEU (still reported, but flagged) if not -- sacrebleu's
flores200 tokenizer needs to download a support file, and we don't want a
single flaky external host to block evaluation entirely.
"""
import argparse
import json
import os

import sacrebleu
import sentencepiece as spm
import torch

from config import MODEL_CFG, LANG_A, LANG_B, LANG_IDS, BEAM_SIZE, LENGTH_PENALTY_ALPHA
from model import SharedTransformerNMT
from utils_dist import load_checkpoint


@torch.no_grad()
def translate_sentences(model: SharedTransformerNMT, sp: spm.SentencePieceProcessor,
                         sentences, src_lang_id: int, tgt_lang_id: int, device,
                         beam_size: int = BEAM_SIZE, batch_size: int = 16) -> list:
    model.eval()
    outputs = []
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        ids = [sp.encode(s, out_type=int) for s in batch_sents]
        max_len = max(len(x) for x in ids) + 2
        import numpy as np
        from config import PAD_ID, BOS_ID, EOS_ID
        arr = np.full((len(ids), max_len), PAD_ID, dtype=np.int64)
        for b, seq in enumerate(ids):
            arr[b, 0] = BOS_ID
            arr[b, 1:1 + len(seq)] = seq
            arr[b, 1 + len(seq)] = EOS_ID
        src = torch.from_numpy(arr).to(device)

        out_ids = model.generate_beam(src, src_lang_id, tgt_lang_id, beam_size=beam_size,
                                       max_len=MODEL_CFG.max_len, length_penalty_alpha=LENGTH_PENALTY_ALPHA)
        for row in out_ids.tolist():
            # strip BOS, stop at EOS/PAD
            toks = []
            for t in row[1:]:
                if t in (EOS_ID, PAD_ID):
                    break
                toks.append(t)
            outputs.append(sp.decode(toks))
    return outputs


def score(hyps, refs, direction: str) -> dict:
    chrf = sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs])
    try:
        bleu = sacrebleu.BLEU(tokenize="flores200").corpus_score(hyps, [refs])
        bleu_tag = "spBLEU (flores200 tokenizer)"
    except Exception as e:
        print(f"[{direction}] flores200 tokenizer unavailable ({e}); falling back to default-tokenizer BLEU "
              f"(less appropriate for Finnish morphology, reported for reference only)")
        bleu = sacrebleu.BLEU().corpus_score(hyps, [refs])
        bleu_tag = "BLEU (default 13a tokenizer, NOT spBLEU -- flores200 tokenizer download failed)"
    print(f"[{direction}] chrF2={chrf.score:.2f}   {bleu_tag}={bleu.score:.2f}")
    return {"chrf2": chrf.score, "bleu": bleu.score, "bleu_tag": bleu_tag}


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ckpt_dir_default = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "checkpoints")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--checkpoint", default=os.path.join(ckpt_dir_default, "bt_latest.pt"))
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    ap.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    ap.add_argument("--n_examples", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)

    with open(os.path.join(args.data_dir, "flores_en.json")) as f:
        flores_en = json.load(f)[args.split]
    with open(os.path.join(args.data_dir, "flores_fi.json")) as f:
        flores_fi = json.load(f)[args.split]
    assert len(flores_en) == len(flores_fi), "FLORES+ en/fi splits must be sentence-aligned (they are, by construction)"
    print(f"Evaluating on FLORES+ {args.split}: {len(flores_en)} sentence pairs (held out, never used in training)")

    model = SharedTransformerNMT(MODEL_CFG).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    en_id, fi_id = LANG_IDS[LANG_A], LANG_IDS[LANG_B]

    print("\n=== EN -> FI ===")
    hyp_fi = translate_sentences(model, sp, flores_en, en_id, fi_id, device)
    results_en2fi = score(hyp_fi, flores_fi, "EN->FI")

    print("\n=== FI -> EN ===")
    hyp_en = translate_sentences(model, sp, flores_fi, fi_id, en_id, device)
    results_fi2en = score(hyp_en, flores_en, "FI->EN")

    print(f"\n=== {args.n_examples} qualitative examples (EN->FI) ===")
    for i in range(min(args.n_examples, len(flores_en))):
        print(f"  SRC: {flores_en[i]}")
        print(f"  HYP: {hyp_fi[i]}")
        print(f"  REF: {flores_fi[i]}")
        print()

    with open(os.path.join(os.path.dirname(args.checkpoint), f"eval_results_{args.split}.json"), "w") as f:
        json.dump({"en2fi": results_en2fi, "fi2en": results_fi2en}, f, indent=2)


if __name__ == "__main__":
    main()

"""
Run this FIRST on Kaggle, before committing to a long DAE or BT run.

I have not benchmarked this exact model on a T4 -- any throughput number in
this codebase's comments/README is a rough estimate, not a measurement. This
script measures actual tokens/sec for YOUR setup (model size, batch config,
actual 2xT4 or whatever GPU Kaggle gives you that session) in a few minutes,
then prints a realistic step budget for a 12-hour session and for the 30
hour/week quota -- so the DAE_STEPS/BT_STEPS you choose are calibrated to
reality instead of a guess.

Usage:
    python profile_throughput.py --data_dir /kaggle/working/unmt-en-fi/data
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from config import MODEL_CFG, LANG_A, LANG_B, LANG_IDS, PAD_ID, LABEL_SMOOTHING, MAX_TOKENS_PER_BATCH
from model import SharedTransformerNMT
from binarize import BinarizedCorpus, load_resolved_vocab_size
from batching import infinite_dae_batch_iterator
from train_dae import dae_loss
from train_bt import back_translate_batch, reconstruction_loss, noise_tensor_batch


def profile_stage(name: str, step_fn, n_warmup: int = 3, n_measure: int = 20) -> float:
    for _ in range(n_warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    n_tokens = 0
    for _ in range(n_measure):
        n_tokens += step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tok_per_sec = n_tokens / elapsed
    sec_per_step = elapsed / n_measure
    print(f"[{name}] {n_measure} steps in {elapsed:.1f}s  "
          f"({sec_per_step:.3f} sec/step, {tok_per_sec:.0f} tokens/sec)")
    return sec_per_step


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--max_tokens_per_batch", type=int, default=MAX_TOKENS_PER_BATCH)
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on: {device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU -- this number will NOT reflect Kaggle T4 speed'})")
    fp16 = device.type == "cuda"

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    if vocab_size != MODEL_CFG.vocab_size:
        print(f"Overriding MODEL_CFG.vocab_size {MODEL_CFG.vocab_size} -> {vocab_size} "
              f"(derived from the actual tokenizer) so profiling matches what you'll actually train")
    MODEL_CFG.vocab_size = vocab_size

    corpus_en = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_A}"))
    corpus_fi = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_B}"))
    model = SharedTransformerNMT(MODEL_CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    en_id, fi_id = LANG_IDS[LANG_A], LANG_IDS[LANG_B]

    # --- DAE step ---
    it_en_dae = infinite_dae_batch_iterator(corpus_en, args.max_tokens_per_batch, device=device)
    it_fi_dae = infinite_dae_batch_iterator(corpus_fi, args.max_tokens_per_batch, device=device)

    def dae_step():
        noised_en, clean_en = next(it_en_dae)
        noised_fi, clean_fi = next(it_fi_dae)
        n_tok = int((clean_en != PAD_ID).sum() + (clean_fi != PAD_ID).sum())
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
            loss = (dae_loss(model, noised_en, clean_en, en_id, LABEL_SMOOTHING)
                    + dae_loss(model, noised_fi, clean_fi, fi_id, LABEL_SMOOTHING)) / 2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return n_tok

    sec_per_dae_step = profile_stage("DAE", dae_step)

    # --- BT step (generation + noising + DAE + training — matches real train_bt.py) ---
    bt_it_en = infinite_dae_batch_iterator(corpus_en, args.max_tokens_per_batch, device=device)
    bt_it_fi = infinite_dae_batch_iterator(corpus_fi, args.max_tokens_per_batch, device=device)

    def bt_step():
        noised_en, clean_en = next(bt_it_en)
        noised_fi, clean_fi = next(bt_it_fi)
        n_tok = int((clean_en != PAD_ID).sum() + (clean_fi != PAD_ID).sum())
        synth_fi = back_translate_batch(model, clean_en, en_id, fi_id, MODEL_CFG.max_len)
        synth_en = back_translate_batch(model, clean_fi, fi_id, en_id, MODEL_CFG.max_len)
        noised_synth_fi = noise_tensor_batch(synth_fi, device)
        noised_synth_en = noise_tensor_batch(synth_en, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
            loss_bt = (reconstruction_loss(model, noised_synth_fi, clean_en, fi_id, en_id, LABEL_SMOOTHING)
                       + reconstruction_loss(model, noised_synth_en, clean_fi, en_id, fi_id, LABEL_SMOOTHING)) / 2
            loss_dae = (dae_loss(model, noised_en, clean_en, en_id, LABEL_SMOOTHING)
                        + dae_loss(model, noised_fi, clean_fi, fi_id, LABEL_SMOOTHING)) / 2
            loss = loss_bt + loss_dae
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return n_tok

    sec_per_bt_step = profile_stage("BT", bt_step)

    print("\n=== Realistic step budgets (measured, not guessed) ===")
    for label, seconds_per_hour_budget in [("one 12h Kaggle session", 12 * 3600),
                                            ("full 30h/week Kaggle quota", 30 * 3600)]:
        dae_budget = int(seconds_per_hour_budget / sec_per_dae_step)
        bt_budget = int(seconds_per_hour_budget / sec_per_bt_step)
        print(f"{label}: ~{dae_budget:,} DAE steps  OR  ~{bt_budget:,} BT steps  "
              f"(you need both stages, so split the budget across sessions/weeks -- "
              f"see README for the recommended DAE-then-BT split)")

    if not torch.cuda.is_available():
        print("\nNOTE: this ran on CPU, not a T4 -- re-run this exact script on Kaggle "
              "before trusting these numbers for planning your actual run.")


if __name__ == "__main__":
    main()

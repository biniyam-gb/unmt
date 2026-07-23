"""
Profile actual tokens/sec throughput on YOUR GPU setup, then compare against
requested DAE_STEPS / BT_STEPS to tell you whether those fit in your Kaggle
session budgets -- and if not, suggest adjusted numbers.

Run BEFORE committing to a long DAE or BT run.  The exact same model
architecture, batch size, vocab size, and mixed-precision settings are used,
so the measured sec/step is what you will actually see during training.

Usage:
    python profile_throughput.py --data_dir ...                # use defaults from config
    python profile_throughput.py --dae_steps 60000 --bt_steps 50000  # check specific targets
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from config import (
    MODEL_CFG, LANG_A, LANG_B, LANG_IDS, PAD_ID, LABEL_SMOOTHING,
    MAX_TOKENS_PER_BATCH, DAE_STEPS, BT_STEPS, WARMUP_STEPS,
)
from model import SharedTransformerNMT
from binarize import BinarizedCorpus, load_resolved_vocab_size
from batching import infinite_dae_batch_iterator
from train_dae import dae_loss
from train_bt import back_translate_batch, reconstruction_loss, noise_tensor_batch


def measure_throughput(name: str, step_fn, n_warmup: int = 3, n_measure: int = 20):
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
    print(f"  {name}: {tok_per_sec:>9,.0f} tok/s  ({sec_per_step:.4f} s/step)")
    return tok_per_sec, sec_per_step


def hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def check_budget(label: str, total_budget_sec: float, dae_spacing: float, dae_target: int,
                  bt_spacing: float, bt_target: int, dae_remaining: int = 0):
    print(f"\n  {label}:")
    if dae_remaining > 0:
        print(f"    DAE checkpoint exists ({dae_remaining} steps done), skipping DAE in this session.")
        dae_cost = 0.0
        dae_actual = dae_remaining
    else:
        dae_cost = dae_target * dae_spacing
        dae_actual = dae_target

    bt_cost = bt_target * bt_spacing
    total_cost = dae_cost + bt_cost
    leftover = total_budget_sec - total_cost

    if total_cost <= total_budget_sec:
        print(f"    DAE {dae_target:,} steps  +  BT {bt_target:,} steps  =  {hms(total_cost)}  "
              f"✓  (remaining {hms(leftover)})")
    else:
        print(f"    DAE {dae_target:,} steps  +  BT {bt_target:,} steps  =  {hms(total_cost)}  "
              f"✗  (exceeds {label} by {hms(-leftover)})")

        # Suggest an adjusted budget: prefer to keep DAE at target, shrink BT to fit
        if dae_cost < total_budget_sec:
            bt_fit = int((total_budget_sec - dae_cost) / bt_spacing)
            bt_fit = max(bt_fit, 0)
            dae_fit = int((total_budget_sec - bt_cost) / dae_spacing)
            dae_fit = max(dae_fit, 0)
            full_fit_dae = int(total_budget_sec / (dae_spacing + bt_spacing * (bt_target / max(dae_target, 1))))
            if dae_cost < total_budget_sec and bt_fit > 0:
                print(f"    → Suggestion: keep DAE={dae_target:,}, reduce BT→{bt_fit:,} "
                      f"({hms(dae_cost + bt_fit * bt_spacing)})")
            print(f"    → Raw budget: DAE-only {int(total_budget_sec / dae_spacing):,} steps  "
                  f"or  BT-only {int(total_budget_sec / bt_spacing):,} steps")


def main():
    ap = argparse.ArgumentParser(
        description="Profile throughput and check if DAE_STEPS/BT_STEPS fit your GPU budget."
    )
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--max_tokens_per_batch", type=int, default=MAX_TOKENS_PER_BATCH)
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    ap.add_argument("--dae_steps", type=int, default=DAE_STEPS,
                    help=f"Target DAE steps (default from config: {DAE_STEPS})")
    ap.add_argument("--bt_steps", type=int, default=BT_STEPS,
                    help=f"Target BT steps (default from config: {BT_STEPS})")
    ap.add_argument("--session_hours", type=float, default=12.0,
                    help="Single Kaggle session length in hours (default: 12)")
    ap.add_argument("--weekly_hours", type=float, default=30.0,
                    help="Weekly Kaggle quota in hours (default: 30)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
                else "CPU -- these numbers will NOT reflect T4 speed")
    print(f"Device: {device}  ({gpu_info})")
    fp16 = device.type == "cuda"

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    if vocab_size != MODEL_CFG.vocab_size:
        print(f"vocab_size: {vocab_size} (overriding config default {MODEL_CFG.vocab_size})")
    MODEL_CFG.vocab_size = vocab_size

    print(f"Model:   d_model={MODEL_CFG.d_model}  layers={MODEL_CFG.n_enc_layers}+{MODEL_CFG.n_dec_layers}  "
          f"heads={MODEL_CFG.n_heads}  d_ff={MODEL_CFG.d_ff}")
    print(f"Batch:   max_tokens={args.max_tokens_per_batch}  warmup={WARMUP_STEPS} steps")
    print(f"Targets: DAE={args.dae_steps:,} steps  BT={args.bt_steps:,} steps")

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

    # --- BT step ---
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

    print("\n--- Measuring throughput ---")
    _, dae_spacing = measure_throughput("DAE", dae_step)
    _, bt_spacing = measure_throughput("BT",   bt_step)

    dae_target = args.dae_steps
    bt_target = args.bt_steps

    session_sec = args.session_hours * 3600
    weekly_sec = args.weekly_hours * 3600

    print(f"\n--- Budget check (config targets: DAE={dae_target:,}  BT={bt_target:,}) ---")
    check_budget(f"One {args.session_hours:.0f}h session ({args.session_hours:.0f}h × 3600s)",
                 session_sec, dae_spacing, dae_target, bt_spacing, bt_target)
    check_budget(f"Weekly {args.weekly_hours:.0f}h quota",
                 weekly_sec, dae_spacing, dae_target, bt_spacing, bt_target)

    print(f"\n--- Raw step budgets (ignoring config targets) ---")
    print(f"  One {args.session_hours:.0f}h session:")
    print(f"    DAE-only: {int(session_sec / dae_spacing):>8,d} steps")
    print(f"    BT-only:  {int(session_sec / bt_spacing):>8,d} steps")
    print(f"  Weekly {args.weekly_hours:.0f}h quota:")
    print(f"    DAE-only: {int(weekly_sec / dae_spacing):>8,d} steps")
    print(f"    BT-only:  {int(weekly_sec / bt_spacing):>8,d} steps")

    if not torch.cuda.is_available():
        print("\nNOTE: ran on CPU -- re-run on a T4 (Kaggle) before trusting these numbers.")

    print("\nTip: set DAE_STEPS/BT_STEPS via env vars or --dae_steps/--bt_steps.")
    print("     The profiler always uses the actual vocab_size & batch config,")
    print("     so numbers are specific to YOUR run parameters.")


if __name__ == "__main__":
    main()

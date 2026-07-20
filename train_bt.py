"""
Stage C: online back-translation -- the actual unsupervised MT training step.

Each step, for a batch of clean EN sentences and a batch of clean FI
sentences:
  1. Generate (no_grad, greedy, CURRENT model weights) a synthetic FI
     translation of the EN batch, and a synthetic EN translation of the FI
     batch. No separate "teacher" model -- the same evolving weights do both
     generation and training (Lample et al. 2018's on-the-fly BT, shown to
     converge better than discrete offline BT rounds).
  2. Train the model to reconstruct the ORIGINAL clean sentence from its own
     synthetic (noisy, imperfect) back-translation. The "label" (original
     sentence) is always 100% real; only the source side is synthetic --
     that's what makes this a valid training signal without any parallel data.

Bootstraps from the Stage-B DAE checkpoint on first launch; on subsequent
launches (resuming BT itself), resumes full BT training state instead.

Launch (2x T4 via DDP, the actual Kaggle setup):
    torchrun --nproc_per_node=2 train_bt.py --data_dir ... --max_steps 150000
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import (
    MODEL_CFG, LANG_A, LANG_B, LANG_IDS, PAD_ID,
    LR_SCALE, WARMUP_STEPS, LABEL_SMOOTHING, GRAD_CLIP, BT_STEPS,
    CHECKPOINT_EVERY_STEPS, CHECKPOINT_EVERY_SECONDS, MAX_TOKENS_PER_BATCH,
)
from model import SharedTransformerNMT
from binarize import BinarizedCorpus, load_resolved_vocab_size
from batching import infinite_batch_iterator
from utils_dist import setup_ddp, cleanup_ddp, is_main_process, save_checkpoint, load_checkpoint, WallClockCheckpointTrigger
from train_dae import noam_lr_lambda  # same schedule shape, fresh optimizer/warmup for this stage


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def back_translate_batch(raw_model, x_clean, src_lang_id: int, tgt_lang_id: int, max_len_cap: int):
    gen_max_len = min(max_len_cap, x_clean.size(1) + 10)
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        # cached path: verified numerically identical to generate_greedy in
        # test_kv_cache_equivalence.py, and substantially cheaper -- this runs
        # on every single training step here, unlike at evaluation time.
        synthetic = raw_model.generate_greedy_cached(x_clean, src_lang_id, tgt_lang_id, max_len=gen_max_len)
    if was_training:
        raw_model.train()
    return synthetic


def reconstruction_loss(model, synthetic_src, clean_target, src_lang_id: int, tgt_lang_id: int,
                         label_smoothing: float) -> torch.Tensor:
    dec_in = clean_target[:, :-1]
    dec_out = clean_target[:, 1:]
    logits = model(synthetic_src, src_lang_id, dec_in, tgt_lang_id)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1),
        ignore_index=PAD_ID, label_smoothing=label_smoothing,
    )


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ckpt_dir_default = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "checkpoints")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--ckpt_dir", default=ckpt_dir_default)
    ap.add_argument("--max_steps", type=int, default=BT_STEPS)
    ap.add_argument("--max_tokens_per_batch", type=int, default=MAX_TOKENS_PER_BATCH)
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    bt_ckpt_path = os.path.join(args.ckpt_dir, "bt_latest.pt")
    dae_ckpt_path = os.path.join(args.ckpt_dir, "dae_latest.pt")

    rank, world_size, local_rank, device = setup_ddp()
    fp16 = device.type == "cuda"

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    if vocab_size != MODEL_CFG.vocab_size and is_main_process(rank):
        print(f"[BT] overriding MODEL_CFG.vocab_size {MODEL_CFG.vocab_size} -> {vocab_size} "
              f"(derived from the actual tokenizer, not the config default) -- "
              f"this MUST match what train_dae.py used, since we're loading its checkpoint")
    MODEL_CFG.vocab_size = vocab_size

    corpus_en = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_A}"))
    corpus_fi = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_B}"))
    it_en = infinite_batch_iterator(corpus_en, args.max_tokens_per_batch, seed=300 + rank, device=device)
    it_fi = infinite_batch_iterator(corpus_fi, args.max_tokens_per_batch, seed=400 + rank, device=device)

    model = SharedTransformerNMT(MODEL_CFG).to(device)

    start_step = 0
    resumed_bt = False
    if os.path.exists(bt_ckpt_path):
        start_step, _ = load_checkpoint(bt_ckpt_path, model, map_location=device)
        resumed_bt = True
        if is_main_process(rank):
            print(f"Resuming BT training itself from step {start_step}")
    elif os.path.exists(dae_ckpt_path):
        load_checkpoint(dae_ckpt_path, model, map_location=device)
        if is_main_process(rank):
            print(f"Bootstrapped BT model weights from DAE checkpoint ({dae_ckpt_path})")
    else:
        if is_main_process(rank):
            print("WARNING: no DAE checkpoint found -- starting BT from a randomly-initialized model. "
                  "This will likely bootstrap far more slowly (or not at all); run train_dae.py first.")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    raw_model = unwrap(model)

    # lr=1.0 is deliberate -- see the matching comment in train_dae.py and the
    # LR_SCALE comment in config.py: noam_lr_lambda() already IS the complete
    # target LR, so the optimizer's own base_lr must be a pure multiplier of 1.0.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: LR_SCALE * noam_lr_lambda(s, MODEL_CFG.d_model, WARMUP_STEPS)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    if resumed_bt:
        # full state (optimizer/scheduler/scaler) only applies when resuming BT
        # itself -- not when merely bootstrapping weights from the DAE checkpoint.
        start_step, _ = load_checkpoint(bt_ckpt_path, model, optimizer, scaler, map_location=device, scheduler=scheduler)

    wall_trigger = WallClockCheckpointTrigger(CHECKPOINT_EVERY_SECONDS)
    t0 = time.time()
    en_id, fi_id = LANG_IDS[LANG_A], LANG_IDS[LANG_B]

    for step in range(start_step, args.max_steps):
        x_en = next(it_en)
        x_fi = next(it_fi)

        # --- generation phase (no_grad, current weights, both directions) ---
        synth_fi = back_translate_batch(raw_model, x_en, en_id, fi_id, MODEL_CFG.max_len)
        synth_en = back_translate_batch(raw_model, x_fi, fi_id, en_id, MODEL_CFG.max_len)

        # --- training phase: reconstruct the REAL sentence from its synthetic translation ---
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
            loss_fi2en = reconstruction_loss(model, synth_fi, x_en, fi_id, en_id, LABEL_SMOOTHING)
            loss_en2fi = reconstruction_loss(model, synth_en, x_fi, en_id, fi_id, LABEL_SMOOTHING)
            loss = (loss_fi2en + loss_en2fi) / 2

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if is_main_process(rank) and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[BT] step {step}/{args.max_steps}  loss={loss.item():.4f} "
                  f"(fi->en={loss_fi2en.item():.4f} en->fi={loss_en2fi.item():.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed/60:.1f}min")

        should_checkpoint = is_main_process(rank) and (
            (step + 1) % CHECKPOINT_EVERY_STEPS == 0 or wall_trigger.ready()
        )
        if should_checkpoint:
            save_checkpoint(bt_ckpt_path, model, optimizer, scaler, step + 1, scheduler=scheduler)
            print(f"[BT] checkpoint saved at step {step + 1}")

    if is_main_process(rank):
        save_checkpoint(bt_ckpt_path, model, optimizer, scaler, args.max_steps, scheduler=scheduler)
        print(f"[BT] final checkpoint saved at step {args.max_steps}")

    cleanup_ddp()


if __name__ == "__main__":
    main()

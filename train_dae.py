"""
Stage B: denoising-autoencoder pretraining.

Trains the SHARED encoder/decoder to reconstruct clean sentences from noised
versions of themselves, independently in each language (no cross-lingual
signal yet -- that's Stage C). This is what teaches the encoder to build
genuinely useful sentence representations and the decoder to generate fluent
output, before back-translation has to rely on either.

Launch (single GPU / CPU, for debugging):
    python train_dae.py --data_dir /kaggle/working/unmt-en-fi/data --max_steps 1000

Launch (2x T4 via DDP, the actual Kaggle setup):
    torchrun --nproc_per_node=2 train_dae.py --data_dir ... --max_steps 60000
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import (
    MODEL_CFG, LANG_A, LANG_B, LANG_IDS, PAD_ID,
    LR_SCALE, WARMUP_STEPS, LABEL_SMOOTHING, GRAD_CLIP, DAE_STEPS,
    CHECKPOINT_EVERY_STEPS, CHECKPOINT_EVERY_SECONDS, MAX_TOKENS_PER_BATCH,
)
from model import SharedTransformerNMT
from binarize import BinarizedCorpus, load_resolved_vocab_size
from batching import infinite_dae_batch_iterator
from utils_dist import setup_ddp, cleanup_ddp, is_main_process, save_checkpoint, load_checkpoint, WallClockCheckpointTrigger


def noam_lr_lambda(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def dae_loss(model, noised, clean, lang_id: int, label_smoothing: float) -> torch.Tensor:
    dec_in = clean[:, :-1]
    dec_target = clean[:, 1:]
    logits = model(noised, lang_id, dec_in, lang_id)  # DAE: same language on both sides
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1),
        ignore_index=PAD_ID, label_smoothing=label_smoothing,
    )
    return loss


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "checkpoints"))
    ap.add_argument("--init_embedding", default=os.path.join(default_dir, "init_embedding.npy"))
    ap.add_argument("--spm_model", default=os.path.join(default_dir, "spm_joint.model"))
    ap.add_argument("--max_steps", type=int, default=DAE_STEPS)
    ap.add_argument("--max_tokens_per_batch", type=int, default=MAX_TOKENS_PER_BATCH)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "dae_latest.pt")

    rank, world_size, local_rank, device = setup_ddp()
    fp16 = device.type == "cuda"  # T4 = Turing: fp16 AMP, not bf16 (no native bf16 tensor cores)

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    if vocab_size != MODEL_CFG.vocab_size and is_main_process(rank):
        print(f"[DAE] overriding MODEL_CFG.vocab_size {MODEL_CFG.vocab_size} -> {vocab_size} "
              f"(derived from the actual tokenizer, not the config default)")
    MODEL_CFG.vocab_size = vocab_size

    corpus_en = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_A}"))
    corpus_fi = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_B}"))
    it_en = infinite_dae_batch_iterator(corpus_en, args.max_tokens_per_batch, seed=100 + rank, device=device)
    it_fi = infinite_dae_batch_iterator(corpus_fi, args.max_tokens_per_batch, seed=200 + rank, device=device)

    model = SharedTransformerNMT(MODEL_CFG).to(device)
    if os.path.exists(args.init_embedding) and not os.path.exists(ckpt_path):
        import numpy as np
        init = np.load(args.init_embedding)
        with torch.no_grad():
            model.token_emb.weight.copy_(torch.from_numpy(init).to(device))
        if is_main_process(rank):
            print(f"Loaded Stage-A embedding initialization from {args.init_embedding}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # lr=1.0 is deliberate: noam_lr_lambda() already returns the COMPLETE target
    # LR (Vaswani et al.'s formula), so LambdaLR's base_lr must be a pure 1.0
    # placeholder -- setting it to any other value would silently multiply a
    # second learning rate on top of the first (this was a real, shipped bug;
    # see the comment on LR_SCALE in config.py).
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: LR_SCALE * noam_lr_lambda(s, MODEL_CFG.d_model, WARMUP_STEPS)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    start_step = 0
    if os.path.exists(ckpt_path):
        start_step, _ = load_checkpoint(ckpt_path, model, optimizer, scaler, map_location=device, scheduler=scheduler)
        if is_main_process(rank):
            print(f"Resumed from checkpoint at step {start_step}")

    wall_trigger = WallClockCheckpointTrigger(CHECKPOINT_EVERY_SECONDS)
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        noised_en, clean_en = next(it_en)
        noised_fi, clean_fi = next(it_fi)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
            loss_en = dae_loss(model, noised_en, clean_en, LANG_IDS[LANG_A], LABEL_SMOOTHING)
            loss_fi = dae_loss(model, noised_fi, clean_fi, LANG_IDS[LANG_B], LABEL_SMOOTHING)
            loss = (loss_en + loss_fi) / 2

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if is_main_process(rank) and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[DAE] step {step}/{args.max_steps}  loss={loss.item():.4f} "
                  f"(en={loss_en.item():.4f} fi={loss_fi.item():.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed/60:.1f}min")

        should_checkpoint = is_main_process(rank) and (
            (step + 1) % CHECKPOINT_EVERY_STEPS == 0 or wall_trigger.ready()
        )
        if should_checkpoint:
            save_checkpoint(ckpt_path, model, optimizer, scaler, step + 1, scheduler=scheduler)
            print(f"[DAE] checkpoint saved at step {step + 1}")

    if is_main_process(rank):
        save_checkpoint(ckpt_path, model, optimizer, scaler, args.max_steps, scheduler=scheduler)
        print(f"[DAE] final checkpoint saved at step {args.max_steps}")

    cleanup_ddp()


if __name__ == "__main__":
    main()

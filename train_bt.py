"""
Stage C: online back-translation with DAE loss regularization and synthetic
noising to prevent identity collapse (copying trap).
"""

import argparse
import os
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from batching import infinite_dae_batch_iterator
from binarize import BinarizedCorpus, load_resolved_vocab_size
from config import (
    BOS_ID,
    BT_STEPS,
    CHECKPOINT_EVERY_SECONDS,
    CHECKPOINT_EVERY_STEPS,
    EOS_ID,
    GRAD_CLIP,
    LABEL_SMOOTHING,
    LANG_A,
    LANG_B,
    LANG_IDS,
    LR_SCALE,
    MAX_TOKENS_PER_BATCH,
    MODEL_CFG,
    PAD_ID,
    WARMUP_STEPS,
)
from model import SharedTransformerNMT
from noise import noise_sentence
from train_dae import dae_loss, noam_lr_lambda
from utils_dist import (
    WallClockCheckpointTrigger,
    cleanup_ddp,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
    setup_ddp,
)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def back_translate_batch(
    raw_model, x_clean, src_lang_id: int, tgt_lang_id: int, max_len_cap: int
):
    gen_max_len = min(max_len_cap, x_clean.size(1) + 10)
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        synthetic = raw_model.generate_greedy_cached(
            x_clean, src_lang_id, tgt_lang_id, max_len=gen_max_len
        )
    if was_training:
        raw_model.train()
    return synthetic


def noise_tensor_batch(tensor_batch: torch.Tensor, device) -> torch.Tensor:
    """Applies word dropout and local shuffle to synthetic batches to prevent
    the model from memorizing verbatim identity mappings."""
    noised_list = []
    for row in tensor_batch:
        clean_ids = [
            int(t) for t in row.tolist() if int(t) not in (PAD_ID, BOS_ID, EOS_ID)
        ]
        noised_ids = noise_sentence(clean_ids)
        noised_list.append(noised_ids)

    max_len = max(len(s) for s in noised_list) + 2 if noised_list else 2
    B = len(noised_list)
    import numpy as np

    out = np.full((B, max_len), PAD_ID, dtype=np.int64)
    for b, seq in enumerate(noised_list):
        out[b, 0] = BOS_ID
        if seq:
            out[b, 1 : 1 + len(seq)] = seq
        out[b, 1 + len(seq)] = EOS_ID
    return torch.from_numpy(out).to(device)


def reconstruction_loss(
    model,
    synthetic_src,
    clean_target,
    src_lang_id: int,
    tgt_lang_id: int,
    label_smoothing: float,
) -> torch.Tensor:
    dec_in = clean_target[:, :-1]
    dec_out = clean_target[:, 1:]
    logits = model(synthetic_src, src_lang_id, dec_in, tgt_lang_id)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        dec_out.reshape(-1),
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(
        os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "data"
    )
    ckpt_dir_default = os.path.join(
        os.environ.get("UNMT_WORK_DIR", "/kaggle/working/unmt-en-fi"), "checkpoints"
    )
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

    sp_proc = None
    if os.path.exists(args.spm_model):
        sp_proc = spm.SentencePieceProcessor(model_file=args.spm_model)

    vocab_size = load_resolved_vocab_size(args.data_dir, args.spm_model)
    if vocab_size != MODEL_CFG.vocab_size and is_main_process(rank):
        print(
            f"[BT] overriding MODEL_CFG.vocab_size {MODEL_CFG.vocab_size} -> {vocab_size} "
            f"(derived from {args.spm_model})"
        )
    MODEL_CFG.vocab_size = vocab_size

    corpus_en = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_A}"))
    corpus_fi = BinarizedCorpus(os.path.join(args.data_dir, f"bin.{LANG_B}"))

    # Use DAE iterators to supply noised + clean pairs for joint DAE+BT training
    it_en = infinite_dae_batch_iterator(
        corpus_en, args.max_tokens_per_batch, seed=300 + rank, device=device
    )
    it_fi = infinite_dae_batch_iterator(
        corpus_fi, args.max_tokens_per_batch, seed=400 + rank, device=device
    )

    model = SharedTransformerNMT(MODEL_CFG).to(device)

    start_step = 0
    resumed_bt = False
    if os.path.exists(bt_ckpt_path):
        start_step, _ = load_checkpoint(bt_ckpt_path, model, map_location=device)
        resumed_bt = True
        if is_main_process(rank):
            print(f"Resuming BT training from step {start_step}")
    elif os.path.exists(dae_ckpt_path):
        load_checkpoint(dae_ckpt_path, model, map_location=device, strict=False)
        if is_main_process(rank):
            print(f"Bootstrapped BT weights from DAE checkpoint ({dae_ckpt_path}) "
                  f"(missing keys like enc_norm/dec_norm initialized fresh if absent)")
    else:
        if is_main_process(rank):
            print("WARNING: Starting BT from a randomly-initialized model.")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    raw_model = unwrap(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: (
            LR_SCALE * noam_lr_lambda(s, MODEL_CFG.d_model, WARMUP_STEPS)
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    if resumed_bt:
        start_step, _ = load_checkpoint(
            bt_ckpt_path,
            model,
            optimizer,
            scaler,
            map_location=device,
            scheduler=scheduler,
        )

    wall_trigger = WallClockCheckpointTrigger(CHECKPOINT_EVERY_SECONDS)
    t0 = time.time()
    en_id, fi_id = LANG_IDS[LANG_A], LANG_IDS[LANG_B]

    for step in range(start_step, args.max_steps):
        noised_en, clean_en = next(it_en)
        noised_fi, clean_fi = next(it_fi)

        # 1. Back-translation generation (current weights)
        synth_fi = back_translate_batch(
            raw_model, clean_en, en_id, fi_id, MODEL_CFG.max_len
        )
        synth_en = back_translate_batch(
            raw_model, clean_fi, fi_id, en_id, MODEL_CFG.max_len
        )

        # 2. Apply noise to synthetic back-translations to prevent identity memorization
        noised_synth_fi = noise_tensor_batch(synth_fi, device)
        noised_synth_en = noise_tensor_batch(synth_en, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
            # Back-translation reconstruction loss
            loss_bt_fi2en = reconstruction_loss(
                model, noised_synth_fi, clean_en, fi_id, en_id, LABEL_SMOOTHING
            )
            loss_bt_en2fi = reconstruction_loss(
                model, noised_synth_en, clean_fi, en_id, fi_id, LABEL_SMOOTHING
            )
            loss_bt = (loss_bt_fi2en + loss_bt_en2fi) / 2

            # Joint DAE loss (prevents identity collapse & maintains language IDs)
            loss_dae_en = dae_loss(model, noised_en, clean_en, en_id, LABEL_SMOOTHING)
            loss_dae_fi = dae_loss(model, noised_fi, clean_fi, fi_id, LABEL_SMOOTHING)
            loss_dae = (loss_dae_en + loss_dae_fi) / 2

            # Combined loss: L_BT + L_DAE
            loss = loss_bt + loss_dae

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if is_main_process(rank) and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[BT] step {step}/{args.max_steps}  loss={loss.item():.4f} "
                f"(bt={loss_bt.item():.4f} dae={loss_dae.item():.4f})  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed / 60:.1f}min"
            )

            if sp_proc is not None and clean_en.size(0) > 0 and synth_fi.size(0) > 0:

                def _decode(tensor_row):
                    toks = [
                        int(t)
                        for t in tensor_row.tolist()
                        if int(t) not in (PAD_ID, BOS_ID, EOS_ID)
                    ]
                    return sp_proc.decode(toks)

                print(f"  [Live Translation Check @ Step {step}]")
                print(f"    EN -> FI  | SRC: {_decode(clean_en[0])[:85]}")
                print(f"              | GEN: {_decode(synth_fi[0])[:85]}")
                print(f"    FI -> EN  | SRC: {_decode(clean_fi[0])[:85]}")
                print(f"              | GEN: {_decode(synth_en[0])[:85]}\n")

        should_checkpoint = is_main_process(rank) and (
            (step + 1) % CHECKPOINT_EVERY_STEPS == 0 or wall_trigger.ready()
        )
        if should_checkpoint:
            save_checkpoint(
                bt_ckpt_path, model, optimizer, scaler, step + 1, scheduler=scheduler
            )
            print(f"[BT] checkpoint saved at step {step + 1}")

    if is_main_process(rank):
        save_checkpoint(
            bt_ckpt_path, model, optimizer, scaler, args.max_steps, scheduler=scheduler
        )
        print(f"[BT] final checkpoint saved at step {args.max_steps}")

    cleanup_ddp()


if __name__ == "__main__":
    main()

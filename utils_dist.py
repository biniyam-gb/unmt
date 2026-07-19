"""
DDP process-group setup and checkpoint save/load, with an explicit wall-clock
checkpoint trigger: Kaggle GPU sessions are hard-killed at 12 hours, so
step-count-based checkpointing alone isn't enough if throughput is slower
than expected -- we also checkpoint on a timer regardless of step count.
"""
import os
import time
import torch
import torch.distributed as dist


def ddp_is_available() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp():
    """Call once at the top of a script launched via torchrun. Returns
    (rank, world_size, local_rank, device). Falls back to a single-process,
    single-device run (rank=0, world_size=1) if not launched under torchrun,
    so the same script also runs standalone (e.g. `python train_dae.py`) for
    debugging on a single GPU or CPU."""
    if ddp_is_available():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


class WallClockCheckpointTrigger:
    def __init__(self, every_seconds: int):
        self.every_seconds = every_seconds
        self.last = time.time()

    def ready(self) -> bool:
        if time.time() - self.last >= self.every_seconds:
            self.last = time.time()
            return True
        return False


def save_checkpoint(path: str, model, optimizer, scaler, step: int, extra: dict = None, scheduler=None):
    raw_model = model.module if hasattr(model, "module") else model  # unwrap DDP
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "extra": extra or {},
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX -- avoids a half-written checkpoint if killed mid-save


def load_checkpoint(path: str, model, optimizer=None, scaler=None, map_location="cpu", scheduler=None):
    payload = torch.load(path, map_location=map_location)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload["step"], payload.get("extra", {})

"""
checkpoint_utils.py
-------------------
Checkpointing helpers for SLURM HPC training of CoordGNN / AirfRANS models.

Typical usage in global_train():
    from checkpoint_utils import save_checkpoint, load_checkpoint, get_latest_checkpoint

    # At the top of global_train, try to resume:
    start_bagging, start_epoch, state = load_checkpoint(CHECKPOINT_DIR)

    # At the end of each epoch:
    save_checkpoint(CHECKPOINT_DIR, bagging_i, epoch, inner_models, optimizers, lr_schedulers, running_mean_loss, running_mean_losses)
"""

import os
import re
import glob
import torch
import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _checkpoint_filename(checkpoint_dir: str, bagging_i: int, epoch: int) -> str:
    """Return the canonical path for a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"ckpt_bag{bagging_i:02d}_ep{epoch:04d}.pt")


def _parse_checkpoint_filename(path: str):
    """
    Parse bagging index and epoch from a checkpoint filename.
    Returns (bagging_i, epoch) as ints, or None if the filename doesn't match.
    """
    basename = os.path.basename(path)
    match = re.match(r"ckpt_bag(\d+)_ep(\d+)\.pt$", basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    checkpoint_dir: str,
    bagging_i: int,
    epoch: int,
    models: list,
    optimizers: list,
    lr_schedulers: list,
    running_mean_loss: float,
    running_mean_losses: list,
    extra: dict = None,
) -> str:
    """
    Save a training checkpoint after completing `epoch` of bagging round `bagging_i`.

    Parameters
    ----------
    checkpoint_dir      : Directory in which checkpoints are stored (created if absent).
    bagging_i           : Current bagging index (0-based).
    epoch               : Epoch that just finished (0-based).
    models              : List of nn.Module instances (inner_models).
    optimizers          : List of Optimizer instances.
    lr_schedulers       : List of LR scheduler instances.
    running_mean_loss   : Scalar EMA loss at the time of saving.
    running_mean_losses : Per-component EMA loss list.
    extra               : Optional dict of any additional objects to persist
                          (e.g. {"args": vars(args)}).

    Returns
    -------
    Path to the saved checkpoint file.
    """
    path = _checkpoint_filename(checkpoint_dir, bagging_i, epoch)

    payload = {
        "bagging_i": bagging_i,
        "epoch": epoch,
        "running_mean_loss": running_mean_loss,
        "running_mean_losses": running_mean_losses,
        "model_state_dicts": [m.state_dict() for m in models],
        "optimizer_state_dicts": [o.state_dict() for o in optimizers],
        "lr_scheduler_state_dicts": [s.state_dict() for s in lr_schedulers],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if extra:
        payload["extra"] = extra

    # Atomic write: save to a temp file then rename so a crash mid-write
    # doesn't corrupt the checkpoint.
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

    print(f"[Checkpoint] Saved  → {path}")
    return path


def get_latest_checkpoint(checkpoint_dir: str):
    """
    Scan `checkpoint_dir` and return the path of the most recent checkpoint,
    ordered first by bagging index, then by epoch.

    Returns None if no checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "ckpt_bag*.pt")
    candidates = glob.glob(pattern)

    valid = []
    for p in candidates:
        parsed = _parse_checkpoint_filename(p)
        if parsed is not None:
            valid.append((parsed[0], parsed[1], p))  # (bagging_i, epoch, path)

    if not valid:
        return None

    # Sort descending: highest bagging_i first, then highest epoch.
    valid.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return valid[0][2]


def load_checkpoint(checkpoint_dir: str, specific_path: str = None):
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    checkpoint_dir  : Directory that contains checkpoint files.
    specific_path   : If provided, load this exact file instead of auto-detecting
                      the latest one.

    Returns
    -------
    A tuple (start_bagging_i, start_epoch, state_dict) where:

    * start_bagging_i (int) – bagging round to resume from.
    * start_epoch     (int) – epoch inside that bagging round to resume from
                               (i.e. the *next* epoch to run; already 0-based).
    * state_dict      (dict | None) – full checkpoint payload, or None if no
                                       checkpoint was found.

    If no checkpoint exists the function returns (0, 0, None) so callers can
    use it unconditionally:

        start_bag, start_ep, state = load_checkpoint(CHECKPOINT_DIR)
    """
    path = specific_path if specific_path else get_latest_checkpoint(checkpoint_dir)

    if path is None or not os.path.isfile(path):
        print("[Checkpoint] No checkpoint found – starting from scratch.")
        return 0, 0, None

    state = torch.load(path, map_location="cpu")
    bagging_i = state["bagging_i"]
    epoch     = state["epoch"]

    # The saved epoch is the last *completed* epoch (0-based).
    # Return the *next* epoch to run.
    next_epoch = epoch + 1

    print(f"[Checkpoint] Loaded ← {path}  (bagging {bagging_i}, epoch {epoch} completed)")
    return bagging_i, next_epoch, state


def restore_model_state(models: list, optimizers: list, lr_schedulers: list, state: dict):
    """
    Convenience function to push the tensors stored in a checkpoint back into
    already-constructed model / optimizer / scheduler objects.

    Call this immediately after building the objects inside global_train().

    Parameters
    ----------
    models        : List of nn.Module (must match length saved in checkpoint).
    optimizers    : List of Optimizer.
    lr_schedulers : List of LR scheduler.
    state         : The payload dict returned by load_checkpoint().
    """
    if state is None:
        return  # nothing to restore

    model_sds = state.get("model_state_dicts", [])
    opt_sds   = state.get("optimizer_state_dicts", [])
    sched_sds = state.get("lr_scheduler_state_dicts", [])

    for i, (model, sd) in enumerate(zip(models, model_sds)):
        model.load_state_dict(sd)
        print(f"[Checkpoint]   Restored model[{i}] weights.")

    for i, (opt, sd) in enumerate(zip(optimizers, opt_sds)):
        opt.load_state_dict(sd)
        print(f"[Checkpoint]   Restored optimizer[{i}] state.")

    for i, (sched, sd) in enumerate(zip(lr_schedulers, sched_sds)):
        sched.load_state_dict(sd)
        print(f"[Checkpoint]   Restored lr_scheduler[{i}] state.")


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    Return a sorted list of (bagging_i, epoch, path) tuples for every
    valid checkpoint found in `checkpoint_dir`.
    """
    if not os.path.isdir(checkpoint_dir):
        return []
    pattern = os.path.join(checkpoint_dir, "ckpt_bag*.pt")
    candidates = glob.glob(pattern)
    valid = []
    for p in candidates:
        parsed = _parse_checkpoint_filename(p)
        if parsed is not None:
            valid.append((parsed[0], parsed[1], p))
    valid.sort(key=lambda t: (t[0], t[1]))
    return valid
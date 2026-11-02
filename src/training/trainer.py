"""
Trainer
=======
AMP training loop with:
  - Resume-safe checkpointing (model, optimizer, scheduler, scaler, RNG, epoch, config)
  - Atomic saves (write to .tmp then rename)
  - --resume flag support
  - FreezeScheduler integration
  - Seed 42 everywhere
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .freeze import FreezeScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _atomic_save(state: dict, path: Path) -> None:
    """Write to a .tmp file then rename — avoids corrupt checkpoints."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _prune_checkpoints(ckpt_dir: Path, pattern: str, keep_last_n: int) -> None:
    """Delete oldest checkpoints beyond keep_last_n."""
    ckpts = sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-keep_last_n]:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Parameters
    ----------
    model          : E0RCaptioner
    train_loader   : training DataLoader
    val_loader     : validation DataLoader  (passed to eval at end of each epoch)
    cfg            : full config dict (from e0r.yaml)
    device         : "cuda" / "cpu"
    resume_path    : path to checkpoint to resume from (or None)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: str = "cuda",
        resume_path: Optional[str] = None,
    ) -> None:
        self.model       = model.to(device)
        self.train_loader= train_loader
        self.val_loader  = val_loader
        self.cfg         = cfg
        self.device      = device

        t_cfg = cfg["training"]
        c_cfg = cfg["checkpoint"]

        self.num_epochs  = t_cfg["num_epochs"]
        self.grad_accum  = t_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad    = t_cfg.get("max_grad_norm", 1.0)
        self.amp_enabled = t_cfg.get("amp", True)
        self.keep_last_n = c_cfg.get("keep_last_n", 3)

        self.output_dir  = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimizer ────────────────────────────────────────────────
        o_cfg = t_cfg["optimizer"]
        # Param groups are empty initially; FreezeScheduler fills them
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr           = o_cfg["lr"],
            weight_decay = o_cfg.get("weight_decay", 0.01),
            betas        = tuple(o_cfg.get("betas", [0.9, 0.999])),
            eps          = o_cfg.get("eps", 1e-8),
        )

        # ── Freeze scheduler ─────────────────────────────────────────
        stage_cfgs = t_cfg["freeze_stages"]
        self.freeze_scheduler = FreezeScheduler(
            model      = model,
            optimizer  = self.optimizer,
            stage_cfgs = stage_cfgs,
            base_lr    = o_cfg["lr"],
        )

        # ── LR scheduler ─────────────────────────────────────────────
        s_cfg  = t_cfg["scheduler"]
        steps_per_epoch = len(train_loader) // self.grad_accum
        total_steps     = steps_per_epoch * self.num_epochs
        warmup_steps    = s_cfg.get("warmup_steps", 0)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )

        # ── AMP scaler ───────────────────────────────────────────────
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # ── Resume ───────────────────────────────────────────────────
        self.start_epoch = 1
        if resume_path:
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> None:
        set_seed(self.cfg.get("seed", 42))

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Apply freeze policy for this epoch
            self.freeze_scheduler.step(epoch)

            logger.info("Epoch %d / %d", epoch, self.num_epochs)
            train_loss = self._train_epoch(epoch)
            logger.info("Epoch %d train loss: %.4f", epoch, train_loss)

            # Save checkpoint after every epoch
            if self.cfg["checkpoint"].get("save_every_epoch", True):
                self._save_checkpoint(epoch)

    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_steps    = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            pixel_values         = batch["pixel_values"].to(self.device, non_blocking=True)
            labels               = batch["labels"].to(self.device, non_blocking=True)
            decoder_attn_mask    = batch["decoder_attention_mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.amp_enabled):
                out  = self.model(
                    pixel_values           = pixel_values,
                    labels                 = labels,
                    decoder_attention_mask = decoder_attn_mask,
                )
                loss = out["loss"] / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.grad_accum
            n_steps    += 1

        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        fname  = self.cfg["checkpoint"]["filename_pattern"].format(epoch=epoch)
        path   = self.output_dir / fname

        state = {
            "epoch"        : epoch,
            "model_state"  : self.model.state_dict(),
            "optimizer"    : self.optimizer.state_dict(),
            "lr_scheduler" : self.lr_scheduler.state_dict(),
            "scaler"       : self.scaler.state_dict(),
            "rng_cpu"      : torch.get_rng_state(),
            "rng_cuda"     : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy"    : np.random.get_state(),
            "rng_python"   : random.getstate(),
            "config"       : self.cfg,
            "model_config" : self.model.get_model_config(),
        }

        _atomic_save(state, path)
        logger.info("Saved checkpoint: %s", path)

        _prune_checkpoints(self.output_dir, "ckpt_epoch*.pt", self.keep_last_n)

    def _load_checkpoint(self, path: str) -> None:
        logger.info("Resuming from checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])

        torch.set_rng_state(ckpt["rng_cpu"])
        if ckpt.get("rng_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])

        self.start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed at epoch %d", ckpt["epoch"])

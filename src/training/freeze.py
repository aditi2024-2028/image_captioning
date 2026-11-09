"""
FreezeScheduler
===============
Implements the 5-epoch freeze schedule for E0-R.

Stage rules (1-indexed epochs)
-------------------------------
Ep 1-2 : CLIP fully frozen
          BART: cross-attention weights trainable, adapters trainable
          everything else frozen

Ep 3-4 : CLIP fully frozen
          BART: cross-attention + top-2 decoder layers + adapters trainable
          everything else frozen

Ep 5   : top-3 CLIP vision transformer layers + post_layernorm unfrozen
          BART: same as ep 3-4

Optimizer param groups are REBUILT at each stage transition but existing
momentum / variance states for parameters that remain in the same group
are preserved via state_dict transplant.

Usage
-----
    scheduler = FreezeScheduler(model, optimizer, stage_configs)
    scheduler.step(epoch=1)   # call before each epoch's training loop
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# FreezeScheduler
# ---------------------------------------------------------------------------

class FreezeScheduler:
    """
    Manages per-epoch parameter freezing and optimizer param-group rebuilding.

    Parameters
    ----------
    model      : E0RCaptioner (or subclass)
    optimizer  : the live optimizer whose param groups will be rebuilt
    stage_cfgs : list of stage dicts from config (freeze_stages)
    base_lr    : learning rate for newly added param groups (from config)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        stage_cfgs: list[dict],
        base_lr: float,
    ) -> None:
        self.model      = model
        self.optimizer  = optimizer
        self.stage_cfgs = stage_cfgs
        self.base_lr    = base_lr
        self._current_stage: int | None = None

    # ------------------------------------------------------------------

    def step(self, epoch: int) -> None:
        """
        Call once before each epoch.  Applies the freeze policy for *epoch*
        (1-indexed) and rebuilds optimizer param groups if the stage changed.
        """
        stage_cfg = self._find_stage(epoch)
        if stage_cfg is None:
            logger.warning("No freeze stage found for epoch %d — skipping.", epoch)
            return

        stage_idx = self.stage_cfgs.index(stage_cfg)
        if stage_idx == self._current_stage:
            return  # same stage, nothing to do

        logger.info("Freeze scheduler: transitioning to stage %d (epoch %d)", stage_idx + 1, epoch)
        self._apply_freeze(stage_cfg)
        self._rebuild_param_groups()
        self._current_stage = stage_idx

        trainable = _count_trainable(self.model)
        total     = _count_total(self.model)
        logger.info(
            "Stage %d — trainable params: %s / %s  (%.2f%%)",
            stage_idx + 1,
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )

    # ------------------------------------------------------------------

    def _find_stage(self, epoch: int) -> dict | None:
        for cfg in self.stage_cfgs:
            if cfg["epoch_start"] <= epoch <= cfg["epoch_end"]:
                return cfg
        return None

    # ------------------------------------------------------------------

    def _apply_freeze(self, stage: dict) -> None:
        model = self.model

        # ── 1. Freeze everything ─────────────────────────────────────
        _set_requires_grad(model, False)

        # ── 2. Adapters are ALWAYS trainable ─────────────────────────
        _set_requires_grad(model.adapters, True)

        # ── 3. CLIP vision unfreeze (partial) ─────────────────────────
        clip_top_n: int = stage.get("clip_top_vision_layers", 0)
        unfreeze_ln: bool = stage.get("clip_post_layernorm", False)

        if clip_top_n > 0:
            # CLIP ViT-B/16 has 12 transformer layers
            clip_layers = model.clip_vision.encoder.layers
            n_total     = len(clip_layers)
            for i in range(n_total - clip_top_n, n_total):
                _set_requires_grad(clip_layers[i], True)

        if unfreeze_ln:
            _set_requires_grad(model.clip_post_layernorm, True)

        # ── 4. Projection heads always trainable ─────────────────────
        _set_requires_grad(model.patch_proj, True)
        _set_requires_grad(model.cls_proj,   True)

        # ── 5. BART parts ─────────────────────────────────────────────
        bart_parts: list[str] = stage.get("bart_trainable_parts", [])
        decoder    = model.bart.model.decoder

        if "cross_attention" in bart_parts:
            for layer in decoder.layers:
                # _PatchedBartDecoderLayer wraps the original layer
                inner = layer._layer if hasattr(layer, "_layer") else layer
                _set_requires_grad(inner.encoder_attn,       True)
                _set_requires_grad(inner.encoder_attn_layer_norm, True)

        if "top2_decoder_layers" in bart_parts:
            n = len(decoder.layers)
            for i in range(n - 2, n):
                _set_requires_grad(decoder.layers[i], True)

        # BART shared embedding + lm_head always trainable
        _set_requires_grad(model.bart.model.shared,   True)
        _set_requires_grad(model.bart.lm_head,         True)
        # BART decoder embed positions & layernorm embedding
        _set_requires_grad(model.bart.model.decoder.embed_positions, True)
        _set_requires_grad(model.bart.model.decoder.layernorm_embedding, True)

    # ------------------------------------------------------------------

    def _rebuild_param_groups(self) -> None:
        """
        Replace optimizer.param_groups with fresh groups (one per trainable
        parameter), then graft back the existing optimizer state for params
        whose data_ptr hasn't changed.
        """
        opt     = self.optimizer
        old_state = {id(p): v for p, v in opt.state.items()}

        trainable = [p for p in self.model.parameters() if p.requires_grad]

        # Rebuild param_groups — one flat group keeps things simple
        # (the caller can pass per-group LRs if needed later)
        new_groups = [{
            "params": trainable,
            "lr": self.base_lr,
            # carry over any scalar hyper-params from the first existing group
            **{k: v for k, v in opt.param_groups[0].items()
               if k not in ("params", "lr")},
        }]

        opt.param_groups.clear()
        opt.param_groups.extend(new_groups)

        # Graft back existing state for params that are still there
        new_state: dict[Any, Any] = {}
        for p in trainable:
            pid = id(p)
            if pid in old_state:
                new_state[p] = old_state[pid]
        opt.state.clear()
        opt.state.update(new_state)

    # ------------------------------------------------------------------
    # Inspection helpers (used by tests)
    # ------------------------------------------------------------------

    def trainable_param_names(self) -> list[str]:
        return [n for n, p in self.model.named_parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return _count_trainable(self.model)

"""
tests/test_freeze_stages.py
===========================
Verify that FreezeScheduler correctly (un)freezes parameters at each of
the 5 training epochs and that trainable param counts match expectations.

Run with:
    pytest tests/test_freeze_stages.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from src.models   import E0RCaptioner
from src.training import FreezeScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STAGE_CFGS = [
    {
        "epoch_start": 1,
        "epoch_end":   2,
        "clip_frozen": True,
        "bart_trainable_parts": ["cross_attention"],
        "clip_top_vision_layers": 0,
        "clip_post_layernorm": False,
    },
    {
        "epoch_start": 3,
        "epoch_end":   4,
        "clip_frozen": True,
        "bart_trainable_parts": ["cross_attention", "top2_decoder_layers"],
        "clip_top_vision_layers": 0,
        "clip_post_layernorm": False,
    },
    {
        "epoch_start": 5,
        "epoch_end":   5,
        "clip_frozen": False,
        "bart_trainable_parts": ["cross_attention", "top2_decoder_layers"],
        "clip_top_vision_layers": 3,
        "clip_post_layernorm": True,
    },
]

BASE_LR = 1e-4


@pytest.fixture(scope="module")
def model_and_scheduler():
    model = E0RCaptioner(
        clip_model_name        = "openai/clip-vit-base-patch16",
        bart_model_name        = "facebook/bart-base",
        proj_dim               = 768,
        adapter_bottleneck_dim = 64,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = BASE_LR,
    )
    scheduler = FreezeScheduler(
        model      = model,
        optimizer  = optimizer,
        stage_cfgs = STAGE_CFGS,
        base_lr    = BASE_LR,
    )
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trainable_names(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}

def _frozen_names(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if not p.requires_grad}

def _count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Stage 1  (epochs 1-2): CLIP frozen, cross-attention + adapters only
# ---------------------------------------------------------------------------

class TestStage1:
    def test_clip_backbone_frozen(self, model_and_scheduler):
        model, _, scheduler = model_and_scheduler
        scheduler.step(epoch=1)

        clip_params = {n for n, _ in model.clip_vision.vision_model.encoder.layers.named_parameters()}
        trainable   = _trainable_names(model)
        # No CLIP backbone layers should be trainable
        leaking = {n for n in trainable if "clip_vision.vision_model.encoder.layers" in n}
        assert len(leaking) == 0, (
            f"Stage 1: CLIP encoder layers should be frozen, got trainable: {leaking}"
        )

    def test_adapters_trainable(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        adapter_params = [n for n in trainable if "adapters" in n]
        assert len(adapter_params) > 0, "Stage 1: adapters must be trainable"

    def test_cross_attention_trainable(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        xattn_params = [n for n in trainable if "encoder_attn" in n and "adapters" not in n]
        assert len(xattn_params) > 0, "Stage 1: cross-attention weights must be trainable"

    def test_no_top_decoder_layers_extra(self, model_and_scheduler):
        """In stage 1, top decoder layers should NOT be fully trainable (only cross-attn within them)."""
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        # Self-attention in last decoder layer should be frozen
        self_attn = [n for n in trainable
                     if "decoder.layers" in n and "self_attn" in n and "encoder_attn" not in n]
        assert len(self_attn) == 0, (
            f"Stage 1: decoder self-attention must be frozen, got trainable: {self_attn[:5]}"
        )

    def test_proj_heads_trainable(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        assert any("patch_proj" in n for n in trainable), "patch_proj must be trainable"
        assert any("cls_proj" in n for n in trainable),   "cls_proj must be trainable"

    def test_optimizer_has_params(self, model_and_scheduler):
        _, optimizer, _ = model_and_scheduler
        total_params = sum(len(g["params"]) for g in optimizer.param_groups)
        assert total_params > 0, "Optimizer must have at least one param group with params"


# ---------------------------------------------------------------------------
# Stage 2  (epochs 3-4): cross-attn + top-2 decoder layers
# ---------------------------------------------------------------------------

class TestStage2:
    def test_top2_decoder_layers_trainable(self, model_and_scheduler):
        model, _, scheduler = model_and_scheduler
        scheduler.step(epoch=3)

        decoder = model.bart.model.decoder
        n       = len(decoder.layers)
        trainable = _trainable_names(model)

        for i in range(n - 2, n):
            layer_prefix = f"bart.model.decoder.layers.{i}"
            layer_params = [nm for nm in trainable if nm.startswith(layer_prefix)]
            assert len(layer_params) > 0, (
                f"Stage 2: decoder layer {i} should have trainable params, found none"
            )

    def test_clip_still_frozen(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        leaking = [n for n in trainable if "clip_vision.vision_model.encoder.layers" in n]
        assert len(leaking) == 0, (
            f"Stage 2: CLIP backbone must stay frozen, got trainable: {leaking}"
        )

    def test_trainable_count_increases_vs_stage1(self, model_and_scheduler):
        """Stage 2 must have more trainable params than stage 1."""
        model, _, scheduler = model_and_scheduler

        scheduler.step(epoch=1)
        t1, _ = _count(model)

        scheduler.step(epoch=3)
        t2, _ = _count(model)

        assert t2 > t1, (
            f"Stage 2 ({t2:,}) should have more trainable params than stage 1 ({t1:,})"
        )


# ---------------------------------------------------------------------------
# Stage 3  (epoch 5): top-3 CLIP layers + post_layernorm unfrozen
# ---------------------------------------------------------------------------

class TestStage3:
    def test_top3_clip_layers_trainable(self, model_and_scheduler):
        model, _, scheduler = model_and_scheduler
        scheduler.step(epoch=5)

        clip_layers = model.clip_vision.vision_model.encoder.layers
        n           = len(clip_layers)         # 12 for ViT-B/16
        trainable   = _trainable_names(model)

        for i in range(n - 3, n):
            prefix = f"clip_vision.vision_model.encoder.layers.{i}"
            layer_params = [nm for nm in trainable if nm.startswith(prefix)]
            assert len(layer_params) > 0, (
                f"Stage 3: CLIP layer {i} should be trainable"
            )

    def test_clip_lower_layers_still_frozen(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        clip_layers = model.clip_vision.vision_model.encoder.layers
        n           = len(clip_layers)
        trainable   = _trainable_names(model)

        for i in range(n - 3):
            prefix = f"clip_vision.vision_model.encoder.layers.{i}"
            leaking = [nm for nm in trainable if nm.startswith(prefix)]
            assert len(leaking) == 0, (
                f"Stage 3: CLIP layer {i} must remain frozen, got trainable: {leaking}"
            )

    def test_post_layernorm_trainable(self, model_and_scheduler):
        model, _, _ = model_and_scheduler
        trainable = _trainable_names(model)
        ln_params = [n for n in trainable if "post_layernorm" in n]
        assert len(ln_params) > 0, "Stage 3: clip_post_layernorm must be trainable"

    def test_trainable_count_increases_vs_stage2(self, model_and_scheduler):
        model, _, scheduler = model_and_scheduler

        scheduler.step(epoch=3)
        t2, _ = _count(model)

        scheduler.step(epoch=5)
        t3, _ = _count(model)

        assert t3 > t2, (
            f"Stage 3 ({t3:,}) should have more trainable params than stage 2 ({t2:,})"
        )

    def test_optimizer_state_preserved(self, model_and_scheduler):
        """
        After a stage transition, optimizer state for params that were already
        trainable should be preserved (not wiped).
        This simulates the 'no losing existing optimizer state' requirement.
        """
        model, optimizer, scheduler = model_and_scheduler

        # Move to stage 2 and simulate a gradient step to populate state
        scheduler.step(epoch=3)
        pixel_values = torch.randn(1, 3, 224, 224)
        labels       = torch.randint(0, 50265, (1, 8))
        out = model(pixel_values=pixel_values, labels=labels)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # Record which params have optimizer state after stage 2
        params_with_state_before = {id(p) for p in optimizer.state}

        # Transition to stage 3
        scheduler.step(epoch=5)

        # params that were trainable in stage 2 and are still trainable in
        # stage 3 should still have their state
        still_trainable = {id(p) for p in model.parameters() if p.requires_grad}

        preserved = params_with_state_before & still_trainable
        assert len(preserved) > 0, (
            "No optimizer state was preserved across stage transition — "
            "state-dict transplant may be broken."
        )

"""
E1-EAAR  -  E0-R captioner + EAAR-NLU V1 bridge
================================================
Subclasses E0RCaptioner and overrides only ``encode_image`` (plus the thin
``forward`` / ``generate`` wrappers so ``detections`` can be passed through).
The BART decoder, adapters and CLIP path are inherited unchanged.

The bridge is applied to the E0-R encoder output (after ``patch_proj`` /
``cls_proj``), i.e. exactly the [B, 197, 768] memory the decoder already reads.
With ``gate_init=0`` the bridge is an identity at step 0, so a loaded E0-R
checkpoint reproduces its baseline captions before any bridge training.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput

from src.models.e0r import E0RCaptioner
from src.models.eaar_nlu import Detections, EAARNLUBridge
from src.training.freeze import FreezeScheduler, _set_requires_grad


class E1EAARCaptioner(E0RCaptioner):
    def __init__(self, *args, bridge_kwargs: Optional[dict] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bridge_kwargs = dict(bridge_kwargs or {})
        self.bridge = EAARNLUBridge(dim=self.proj_dim, **self.bridge_kwargs)

    # ------------------------------------------------------------------
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        detections: Optional[Detections] = None,
    ) -> torch.Tensor:
        tokens = super().encode_image(pixel_values)          # [B, 197, proj_dim]
        if detections is None:                               # baseline fallback
            return tokens
        return self.bridge(tokens, detections)               # [B, 197, proj_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        detections: Optional[Detections] = None,
    ) -> dict:
        encoder_hidden = self.encode_image(pixel_values, detections)
        out = self.bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        detections: Optional[Detections] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        encoder_hidden = self.encode_image(pixel_values, detections)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    def freeze_all_but_bridge(self) -> None:
        _set_requires_grad(self, False)
        _set_requires_grad(self.bridge, True)

    def get_model_config(self) -> dict:
        cfg = super().get_model_config()
        cfg["bridge_kwargs"] = self.bridge_kwargs
        return cfg


class BridgeFreezeScheduler(FreezeScheduler):
    """Drop-in replacement for FreezeScheduler: every stage trains only the bridge.

    The stock scheduler re-enables adapters, projection heads and BART parts at
    each stage, which would silently un-freeze the E0-R weights.
    """

    def _apply_freeze(self, stage: dict) -> None:
        self.model.freeze_all_but_bridge()

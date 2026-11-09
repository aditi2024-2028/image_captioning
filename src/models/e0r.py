"""
E0-R  —  Non-semantic baseline captioner
=========================================
Architecture
------------
Image (224×224)
  → CLIP ViT-B/16 vision encoder
      • extract all 197 tokens  (1 CLS + 196 patch tokens)
      • apply CLIP post_layernorm to the full sequence
  → patch_proj  : Linear(768 → proj_dim)  on 196 patch tokens
     cls_proj   : Linear(768 → proj_dim)  on 1 CLS token
  → concatenate → shape [B, 197, proj_dim]  (encoder_outputs)
  → BART-base decoder (text encoder BYPASSED)
      • bottleneck adapters on every cross-attention layer
      • generates captions autoregressively

No semantic modules (entity / attribute / action / relation /
scene-graph / semantic bottleneck / extra LLM-VLM layer).

E1x models subclass this; keep public API stable.
"""

from __future__ import annotations

import math
from typing import Optional

import inspect
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import BaseModelOutput


# ---------------------------------------------------------------------------
# Bottleneck adapter  (added into each BART decoder cross-attention layer)
# ---------------------------------------------------------------------------

class BartCrossAttentionAdapter(nn.Module):
    """Residual bottleneck adapter: x → down → GELU → up → + x"""

    def __init__(self, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, hidden_dim)
        # initialise near-identity so training starts stable
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


# ---------------------------------------------------------------------------
# Patched BART decoder layer  (injects adapter after cross-attention)
# ---------------------------------------------------------------------------

class _PatchedBartDecoderLayer(nn.Module):
    """Wraps an existing BartDecoderLayer and injects a cross-attn adapter."""

    def __init__(self, layer: nn.Module, adapter: BartCrossAttentionAdapter) -> None:
        super().__init__()
        self._layer   = layer
        self._adapter = adapter

    # expose the sub-layer attributes so FreezeScheduler can navigate them
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        **kwargs,
    ):
        params = inspect.signature(self._layer.forward).parameters

        cache = past_key_values if past_key_values is not None else past_key_value

        layer_kwargs = {
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "use_cache": use_cache,
        }

        if "past_key_values" in params:
            layer_kwargs["past_key_values"] = cache
        elif "past_key_value" in params:
            layer_kwargs["past_key_value"] = cache

        if "layer_head_mask" in params:
            layer_kwargs["layer_head_mask"] = layer_head_mask

        if "cross_attn_layer_head_mask" in params:
            layer_kwargs["cross_attn_layer_head_mask"] = cross_attn_layer_head_mask

        if "output_attentions" in params:
            layer_kwargs["output_attentions"] = output_attentions

        accepts_extra = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        for key, value in kwargs.items():
            if key in params or accepts_extra:
                layer_kwargs[key] = value

        outputs = self._layer(hidden_states, **layer_kwargs)

        if isinstance(outputs, torch.Tensor):
            hidden_out = outputs
        elif hasattr(outputs, "last_hidden_state"):
            hidden_out = outputs.last_hidden_state
        else:
            hidden_out = outputs[0]

        return self._adapter(hidden_out)


class E0RCaptioner(nn.Module):
    """
    Non-semantic CLIP ViT-B/16 + BART-base captioner (baseline E0-R).

    Parameters
    ----------
    clip_model_name     : HuggingFace model id for CLIP vision
    bart_model_name     : HuggingFace model id for BART
    proj_dim            : projection dimension (default 768 = BART hidden)
    adapter_bottleneck_dim : bottleneck width for cross-attn adapters
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        bart_model_name: str = "facebook/bart-base",
        proj_dim: int = 768,
        adapter_bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()

        # ── CLIP vision backbone ─────────────────────────────────────────
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden = self.clip_vision.config.hidden_size          # 768
        self.clip_post_layernorm = self.clip_vision.post_layernorm

        # ── Projection heads ────────────────────────────────────────────
        self.patch_proj = nn.Linear(clip_hidden, proj_dim)         # 196 patches
        self.cls_proj   = nn.Linear(clip_hidden, proj_dim)         # 1 CLS token

        # ── BART ─────────────────────────────────────────────────────────
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_hidden = self.bart.config.d_model                     # 768

        assert proj_dim == bart_hidden, (
            f"proj_dim ({proj_dim}) must equal BART hidden dim ({bart_hidden})"
        )

        # ── Inject adapters into every BART decoder cross-attention layer
        self.adapters = nn.ModuleList()
        decoder_layers = self.bart.model.decoder.layers
        for i, layer in enumerate(decoder_layers):
            adapter = BartCrossAttentionAdapter(bart_hidden, adapter_bottleneck_dim)
            self.adapters.append(adapter)
            decoder_layers[i] = _PatchedBartDecoderLayer(layer, adapter)

        # Store config for subclasses / checkpointing
        self.proj_dim = proj_dim
        self.adapter_bottleneck_dim = adapter_bottleneck_dim

    # ------------------------------------------------------------------
    # Visual encoding  (reusable by E1x subclasses)
    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run CLIP vision encoder.

        Returns
        -------
        encoder_outputs : Tensor [B, 197, proj_dim]
            CLS token at position 0, then 196 patch tokens.
        """
        # CLIPVisionModel returns all hidden states when output_hidden_states=False
        # last_hidden_state shape: [B, 197, 768]  (no pooling)
        vision_out = self.clip_vision(
            pixel_values=pixel_values,
            output_hidden_states=False,
            output_attentions=False,
        )
        # last_hidden_state: [B, 197, clip_hidden]
        tokens = vision_out.last_hidden_state

        # Apply CLIP post_layernorm to full sequence so it gets gradients
        # when unfrozen (stage 3). CLIPVisionTransformer only applies it to
        # the CLS token for its own pooler; we re-apply it to all tokens.
        tokens = self.clip_post_layernorm(tokens)

        cls_tok   = tokens[:, :1, :]     # [B, 1,   768]
        patch_tok = tokens[:, 1:, :]     # [B, 196, 768]

        cls_emb   = self.cls_proj(cls_tok)     # [B, 1,   proj_dim]
        patch_emb = self.patch_proj(patch_tok) # [B, 196, proj_dim]

        encoder_outputs = torch.cat([cls_emb, patch_emb], dim=1)  # [B, 197, proj_dim]
        return encoder_outputs

    # ------------------------------------------------------------------
    # Forward  (teacher-forcing training)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        pixel_values         : [B, 3, 224, 224]
        labels               : [B, T]  token ids with -100 for padding
        decoder_input_ids    : [B, T]  right-shifted labels (auto-built if None)
        decoder_attention_mask : [B, T]

        Returns
        -------
        dict with keys: loss (if labels given), logits
        """
        encoder_hidden = self.encode_image(pixel_values)  # [B, 197, proj_dim]

        # Wrap as BaseModelOutput so BART decoder accepts it directly
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        out = self.bart(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return {"loss": out.loss, "logits": out.logits}

    # ------------------------------------------------------------------
    # Generation  (beam search / greedy at eval time)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Thin wrapper around BART's generate().

        All beam-search settings (num_beams, max_new_tokens, …) are passed
        through generate_kwargs (read from config at eval time).
        """
        encoder_hidden = self.encode_image(pixel_values)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        return self.bart.generate(
            encoder_outputs=encoder_outputs,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # Config dict  (used by checkpoint / subclass init)
    # ------------------------------------------------------------------

    def get_model_config(self) -> dict:
        return {
            "clip_model_name": self.clip_vision.config._name_or_path,
            "bart_model_name": self.bart.config._name_or_path,
            "proj_dim": self.proj_dim,
            "adapter_bottleneck_dim": self.adapter_bottleneck_dim,
        }

"""
tests/test_model_shapes.py
==========================
Verify that E0RCaptioner forward pass produces the expected tensor shapes.

Run with:
    pytest tests/test_model_shapes.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models import E0RCaptioner


# Use CPU + tiny batch for speed; no COCO data needed
DEVICE     = "cpu"
BATCH_SIZE = 2
IMG_SIZE   = 224
SEQ_LEN    = 16
VOCAB_SIZE = 50265   # BART-base vocab


@pytest.fixture(scope="module")
def model():
    m = E0RCaptioner(
        clip_model_name        = "openai/clip-vit-base-patch16",
        bart_model_name        = "facebook/bart-base",
        proj_dim               = 768,
        adapter_bottleneck_dim = 64,
    )
    m.eval()
    return m


# ---------------------------------------------------------------------------

def test_encode_image_shape(model):
    """encode_image must return [B, 197, 768]"""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        enc = model.encode_image(pixel_values)
    assert enc.shape == (BATCH_SIZE, 197, 768), (
        f"Expected [B, 197, 768], got {enc.shape}"
    )


def test_cls_proj_shape(model):
    """cls_proj output must be [B, 1, proj_dim]"""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        tokens = model.clip_vision(pixel_values=pixel_values).last_hidden_state
        tokens = model.clip_post_layernorm(tokens)
        cls    = model.cls_proj(tokens[:, :1, :])
    assert cls.shape == (BATCH_SIZE, 1, 768)


def test_patch_proj_shape(model):
    """patch_proj output must be [B, 196, proj_dim]"""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        tokens = model.clip_vision(pixel_values=pixel_values).last_hidden_state
        tokens = model.clip_post_layernorm(tokens)
        patch  = model.patch_proj(tokens[:, 1:, :])
    assert patch.shape == (BATCH_SIZE, 196, 768)


def test_forward_loss_shape(model):
    """forward() with labels must return a scalar loss and logits [B, T, V]"""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    labels       = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dec_mask     = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

    with torch.no_grad():
        out = model(
            pixel_values           = pixel_values,
            labels                 = labels,
            decoder_attention_mask = dec_mask,
        )

    assert out["loss"].shape == torch.Size([]), "Loss must be scalar"
    # logits: [B, T, vocab_size]
    assert out["logits"].shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), (
        f"Unexpected logits shape: {out['logits'].shape}"
    )


def test_forward_no_labels(model):
    """forward() without labels must not raise and must return logits."""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    dec_ids      = torch.zeros(BATCH_SIZE, 1, dtype=torch.long)  # BOS token

    with torch.no_grad():
        out = model(
            pixel_values      = pixel_values,
            decoder_input_ids = dec_ids,
        )

    assert out["loss"] is None or out["loss"] == 0 or out.get("loss") is None
    assert out["logits"].ndim == 3


def test_generate_shape(model):
    """generate() must return token ids of shape [B, >=1]"""
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        ids = model.generate(
            pixel_values   = pixel_values,
            num_beams      = 2,
            max_new_tokens = 10,
        )
    assert ids.shape[0] == BATCH_SIZE
    assert ids.ndim == 2


def test_adapter_count(model):
    """Number of adapters must equal number of BART decoder layers (6 for bart-base)."""
    n_decoder_layers = len(model.bart.model.decoder.layers)
    assert len(model.adapters) == n_decoder_layers, (
        f"Expected {n_decoder_layers} adapters, got {len(model.adapters)}"
    )


def test_no_bart_encoder_in_forward(model):
    """
    BART text encoder must be completely bypassed.
    Confirm by checking that encoder parameters have no gradient after a
    backward pass (they are frozen and not in the forward graph).
    """
    pixel_values = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    labels       = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # make all params require grad temporarily to catch any unexpected flow
    for p in model.parameters():
        p.requires_grad_(True)

    out  = model(pixel_values=pixel_values, labels=labels)
    out["loss"].backward()

    # BART text encoder layers should have zero / no gradient because
    # they are not in the computational graph
    for name, p in model.bart.model.encoder.named_parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0).item(), (
                f"BART encoder param '{name}' received a gradient — "
                "text encoder must be bypassed."
            )

    # cleanup
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(False)

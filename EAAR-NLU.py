"""
EAAR-NLU V1 bridge
==================
Six trainable layers that sit between the frozen perception front-end
(CLIP ViT-B/16 tokens + RF-DETR Small detections) and the frozen BART decoder.

    tokens  [B, 197, 768]  ─┐
                            ├─►  1 Semantic Evidence Construction  [B, 28, 768]  (+ slot mask)
    detections (<= 8 boxes) ┘    2 Typed Slot Encoding             [B, 28, 768]
                                 3 Role Binding (masked self-attn)  [B, 28, 768]
                                 4 Reliability Weighting            [B, 28, 1] -> weighted slots
                                 5 Visual-Semantic Cross-Attention  [B, 197, 768]
                                 6 Gated Residual Fusion            [B, 197, 768]

Slot layout (28 = 3 * 8 + 4), fixed positions:
    [0  : 8 )  entity      one per detection
    [8  : 16)  attribute   one per detection
    [16 : 20)  action      top-4 detections by confidence (+ global CLS context)
    [20 : 28)  relation    detection i <-> its nearest other detection

Empty slots are carried as zeros with ``mask == False`` and are ignored by every
attention layer. Samples with no detections at all pass through unchanged.

Detection boxes are expected in the frame of the 224x224 image the CLIP encoder
actually sees (normalised xyxy in [0, 1]); ``detections_to_tensors`` converts
RF-DETR output from the original image into that frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import roi_align

SLOT_ENTITY, SLOT_ATTRIBUTE, SLOT_ACTION, SLOT_RELATION = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Detection container + RF-DETR adapter
# ---------------------------------------------------------------------------

@dataclass
class Detections:
    """Padded detections. Per-sample tensors are [K, ...]; batched are [B, K, ...].

    boxes  : float  [.., K, 4]  normalised xyxy in the 224x224 CLIP frame
    labels : long   [.., K]     COCO category id (RF-DETR class_id)
    scores : float  [.., K]     detection confidence
    valid  : bool   [.., K]     False for padding
    """

    boxes: Tensor
    labels: Tensor
    scores: Tensor
    valid: Tensor

    def to(self, device, non_blocking: bool = True) -> "Detections":
        return Detections(
            self.boxes.to(device, non_blocking=non_blocking),
            self.labels.to(device, non_blocking=non_blocking),
            self.scores.to(device, non_blocking=non_blocking),
            self.valid.to(device, non_blocking=non_blocking),
        )

    @staticmethod
    def collate(items: Sequence["Detections"]) -> "Detections":
        return Detections(
            torch.stack([d.boxes for d in items]),
            torch.stack([d.labels for d in items]),
            torch.stack([d.scores for d in items]),
            torch.stack([d.valid for d in items]),
        )

    @staticmethod
    def empty(k: int = 8) -> "Detections":
        return Detections(
            torch.zeros(k, 4),
            torch.zeros(k, dtype=torch.long),
            torch.zeros(k),
            torch.zeros(k, dtype=torch.bool),
        )


def detections_to_tensors(
    xyxy,
    confidence,
    class_id,
    image_size: tuple[int, int],
    max_det: int = 8,
    out_size: int = 224,
    min_conf: float = 0.0,
    min_area_frac: float = 1e-3,
) -> Detections:
    """Convert RF-DETR output (original-image pixels) to a padded ``Detections``.

    CLIP's preprocessing resizes the short side to ``out_size`` and centre-crops, so
    boxes must go through the same transform; boxes that fall outside the crop are
    clipped and dropped if almost nothing is left.

    xyxy        : array-like [N, 4] pixel coordinates in the ORIGINAL image
    confidence  : array-like [N]
    class_id    : array-like [N]
    image_size  : (width, height) of the original image
    """
    w, h = image_size
    conf = torch.as_tensor(confidence, dtype=torch.float32).reshape(-1)
    cls = torch.as_tensor(class_id, dtype=torch.long).reshape(-1)
    boxes = torch.as_tensor(xyxy, dtype=torch.float32).reshape(conf.numel(), 4)

    scale = out_size / float(min(w, h))
    off_x = (w * scale - out_size) / 2.0
    off_y = (h * scale - out_size) / 2.0
    boxes = boxes * scale
    boxes = torch.stack(
        [boxes[:, 0] - off_x, boxes[:, 1] - off_y, boxes[:, 2] - off_x, boxes[:, 3] - off_y],
        dim=1,
    )
    boxes = boxes.clamp(0.0, float(out_size)) / out_size

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (area >= min_area_frac) & (conf >= min_conf)
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

    order = conf.argsort(descending=True)[:max_det]
    boxes, conf, cls = boxes[order], conf[order], cls[order]
    n = boxes.shape[0]

    out = Detections.empty(max_det)
    out.boxes[:n], out.scores[:n], out.labels[:n] = boxes, conf, cls
    out.valid[:n] = True
    return out


def _sort_and_fit(det: Detections, k: int):
    """Sort by confidence (invalid last) and pad / truncate to exactly k entries."""
    scores = det.scores.masked_fill(~det.valid, -1.0)
    order = scores.argsort(dim=1, descending=True)
    boxes = det.boxes.gather(1, order.unsqueeze(-1).expand(-1, -1, 4))
    labels = det.labels.gather(1, order)
    scores = det.scores.gather(1, order)
    valid = det.valid.gather(1, order)

    def fit(t: Tensor, fill) -> Tensor:
        if t.shape[1] >= k:
            return t[:, :k]
        pad = t.new_full((t.shape[0], k - t.shape[1], *t.shape[2:]), fill)
        return torch.cat([t, pad], dim=1)

    return fit(boxes, 0.0), fit(labels, 0), fit(scores, 0.0), fit(valid, False)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _mlp(dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class MaskedMHA(nn.Module):
    """Multi-head scaled dot-product attention with a key padding mask.

    ``key_mask`` is bool [B, Lk] (True = real). Samples whose keys are all masked
    return zeros instead of NaN.
    """

    def __init__(self, dim: int, attn_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.h = num_heads
        self.hd = attn_dim // num_heads
        self.q = nn.Linear(dim, attn_dim)
        self.k = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(dim, attn_dim)
        self.o = nn.Linear(attn_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q: Tensor, x_kv: Tensor, key_mask: Tensor, return_weights: bool = False):
        B, Lq, _ = x_q.shape
        Lk = x_kv.shape[1]
        q = self.q(x_q).view(B, Lq, self.h, self.hd).transpose(1, 2)
        k = self.k(x_kv).view(B, Lk, self.h, self.hd).transpose(1, 2)
        v = self.v(x_kv).view(B, Lk, self.h, self.hd).transpose(1, 2)

        any_valid = key_mask.any(dim=-1)                       # [B]
        safe = key_mask.clone()
        safe[:, 0] |= ~any_valid                               # avoid all -inf rows

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)  # [B, H, Lq, Lk]
        logits = logits.masked_fill(~safe[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = self.drop(logits.softmax(dim=-1))

        ctx = (attn @ v).transpose(1, 2).reshape(B, Lq, self.h * self.hd)
        out = self.o(ctx) * any_valid[:, None, None].to(ctx.dtype)
        return (out, attn) if return_weights else out


# ---------------------------------------------------------------------------
# Layer 1 - Semantic Evidence Construction
# ---------------------------------------------------------------------------

class SemanticEvidenceConstruction(nn.Module):
    """Detections + CLIP patch grid -> entity / attribute / action / relation candidates."""

    def __init__(
        self,
        dim: int,
        num_labels: int,
        max_entities: int,
        num_action_slots: int,
        grid_size: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        assert num_action_slots <= max_entities
        self.dim, self.k, self.a, self.g = dim, max_entities, num_action_slots, grid_size
        self.num_labels = num_labels

        self.label_emb = nn.Embedding(num_labels, dim)
        self.geo_proj = nn.Linear(8, dim)     # x1 y1 x2 y2 conf w h area
        self.pair_proj = nn.Linear(6, dim)    # dx dy log-w-ratio log-h-ratio iou dist
        self.attr_mlp = _mlp(dim, bottleneck_dim)
        self.act_mlp = _mlp(dim, bottleneck_dim)
        self.rel_mlp = _mlp(dim, bottleneck_dim)
        nn.init.normal_(self.label_emb.weight, std=0.02)

    @property
    def num_slots(self) -> int:
        return 3 * self.k + self.a

    def forward(self, tokens: Tensor, det: Detections):
        B, N, d = tokens.shape
        G, K, A = self.g, self.k, self.a
        assert N == 1 + G * G, f"expected {1 + G * G} tokens, got {N}"

        boxes, labels, scores, valid = _sort_and_fit(det, K)
        boxes = boxes.float()
        scores = scores.float()

        # ---- region features: RoIAlign over the 14x14 patch grid -----------
        grid = tokens[:, 1:, :].transpose(1, 2).reshape(B, d, G, G)
        ms = 1.0 / G
        dummy = boxes.new_tensor([0.0, 0.0, 1.0, 1.0])
        rb = torch.where(valid.unsqueeze(-1), boxes, dummy)
        x1 = rb[..., 0].clamp(0.0, 1.0 - ms)
        y1 = rb[..., 1].clamp(0.0, 1.0 - ms)
        x2 = torch.maximum(rb[..., 2], x1 + ms).clamp(max=1.0)
        y2 = torch.maximum(rb[..., 3], y1 + ms).clamp(max=1.0)
        rois = torch.stack([x1, y1, x2, y2], dim=-1).reshape(B * K, 4) * G
        bidx = torch.arange(B, device=tokens.device).repeat_interleave(K).unsqueeze(1)
        rois = torch.cat([bidx.to(rois.dtype), rois], dim=1).to(grid.dtype)
        region = roi_align(grid, rois, output_size=2, spatial_scale=1.0,
                           sampling_ratio=2, aligned=True)      # [B*K, d, 2, 2]
        region = region.mean(dim=(2, 3)).reshape(B, K, d)

        # ---- geometry --------------------------------------------------------
        gb = boxes * valid.unsqueeze(-1)
        gx1, gy1, gx2, gy2 = gb.unbind(-1)
        w = (gx2 - gx1).clamp(min=0.0)
        h = (gy2 - gy1).clamp(min=0.0)
        cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        geom = torch.stack([gx1, gy1, gx2, gy2, scores * valid, w, h, w * h], dim=-1)

        lab = self.label_emb(labels.clamp(0, self.num_labels - 1))

        # ---- entity / attribute / action ------------------------------------
        ent = region + lab + self.geo_proj(geom.to(tokens.dtype))
        attr = self.attr_mlp(region + lab)
        act = self.act_mlp(region[:, :A] + lab[:, :A] + tokens[:, :1])

        # ---- relation: each entity paired with its nearest other entity -----
        centers = torch.stack([cx, cy], dim=-1)
        dist = torch.cdist(centers, centers)                              # [B, K, K]
        eye = torch.eye(K, dtype=torch.bool, device=tokens.device).unsqueeze(0)
        pair_ok = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
        nn_dist, nn_idx = dist.masked_fill(~pair_ok, 1e4).min(dim=-1)     # [B, K]
        rel_valid = valid & (nn_dist < 1e3)

        ent_j = ent.gather(1, nn_idx.unsqueeze(-1).expand(-1, -1, d))
        nb = gb.gather(1, nn_idx.unsqueeze(-1).expand(-1, -1, 4))
        nx1, ny1, nx2, ny2 = nb.unbind(-1)
        nw, nh = (nx2 - nx1).clamp(min=0.0), (ny2 - ny1).clamp(min=0.0)
        ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        eps = 1e-3
        dx = ((ncx - cx) / w.clamp(min=eps)).clamp(-8.0, 8.0)
        dy = ((ncy - cy) / h.clamp(min=eps)).clamp(-8.0, 8.0)
        lw = torch.log(nw.clamp(min=eps) / w.clamp(min=eps))
        lh = torch.log(nh.clamp(min=eps) / h.clamp(min=eps))
        iw = (torch.minimum(gx2, nx2) - torch.maximum(gx1, nx1)).clamp(min=0.0)
        ih = (torch.minimum(gy2, ny2) - torch.maximum(gy1, ny1)).clamp(min=0.0)
        inter = iw * ih
        iou = inter / (w * h + nw * nh - inter + 1e-6)
        pair_geo = torch.stack([dx, dy, lw, lh, iou, nn_dist.clamp(max=2.0)], dim=-1)
        rel = self.rel_mlp(ent + ent_j + self.pair_proj(pair_geo.to(tokens.dtype)))

        # ---- assemble fixed-position slots ----------------------------------
        cand = torch.cat([ent, attr, act, rel], dim=1)                    # [B, 28, d]
        mask = torch.cat([valid, valid, valid[:, :A], rel_valid], dim=1)  # [B, 28]
        cand = cand * mask.unsqueeze(-1).to(cand.dtype)
        return cand, mask


# ---------------------------------------------------------------------------
# Layer 2 - Typed Slot Encoding
# ---------------------------------------------------------------------------

class TypedSlotEncoding(nn.Module):
    def __init__(self, dim: int, num_slots: int) -> None:
        super().__init__()
        self.slot_emb = nn.Parameter(torch.empty(num_slots, dim))
        self.type_emb = nn.Embedding(4, dim)   # entity / attribute / action / relation
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.slot_emb, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)

    def forward(self, cand: Tensor, mask: Tensor, type_ids: Tensor) -> Tensor:
        x = cand + self.slot_emb + self.type_emb(type_ids)
        x = self.norm(self.proj(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Layer 3 - Role Binding
# ---------------------------------------------------------------------------

class RoleBinding(nn.Module):
    def __init__(self, dim: int, attn_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MaskedMHA(dim, attn_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm(x + self.attn(x, x, mask))
        return x * mask.unsqueeze(-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Layer 4 - Reliability Weighting
# ---------------------------------------------------------------------------

class ReliabilityWeighting(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: Tensor, mask: Tensor):
        r = torch.sigmoid(self.net(x)) * mask.unsqueeze(-1).to(x.dtype)   # [B, 28, 1]
        return x * r, r


# ---------------------------------------------------------------------------
# Layer 5 - Visual-Semantic Cross-Attention
# ---------------------------------------------------------------------------

class VisualSemanticCrossAttention(nn.Module):
    def __init__(self, dim: int, attn_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.attn = MaskedMHA(dim, attn_dim, num_heads, dropout)

    def forward(self, tokens: Tensor, slots: Tensor, mask: Tensor, return_weights: bool = False):
        return self.attn(self.q_norm(tokens), slots, mask, return_weights)


# ---------------------------------------------------------------------------
# Layer 6 - Gated Residual Fusion
# ---------------------------------------------------------------------------

class GatedResidualFusion(nn.Module):
    """out = tokens + tanh(g) * ctx. With gate_init=0 the bridge starts as an exact
    identity, so training begins from the E0-R baseline representation."""

    def __init__(self, dim: int, gate_init: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))

    def forward(self, tokens: Tensor, ctx: Tensor) -> Tensor:
        return tokens + torch.tanh(self.gate) * ctx


# ---------------------------------------------------------------------------
# Full bridge
# ---------------------------------------------------------------------------

class EAARNLUBridge(nn.Module):
    """[B, 197, dim] visual tokens + detections -> [B, 197, dim] enriched tokens."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 4,
        num_labels: int = 91,            # COCO category ids 0..90 (RF-DETR class_id)
        max_entities: int = 8,
        num_action_slots: int = 4,
        grid_size: int = 14,
        attn_dim: int = 480,             # inner width of both attention layers
        bottleneck_dim: int = 128,       # attribute / action / relation candidate MLPs
        reliability_hidden: int = 192,
        gate_init: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer1 = SemanticEvidenceConstruction(
            dim, num_labels, max_entities, num_action_slots, grid_size, bottleneck_dim
        )
        n_slots = self.layer1.num_slots
        self.layer2 = TypedSlotEncoding(dim, n_slots)
        self.layer3 = RoleBinding(dim, attn_dim, num_heads, dropout)
        self.layer4 = ReliabilityWeighting(dim, reliability_hidden)
        self.layer5 = VisualSemanticCrossAttention(dim, attn_dim, num_heads, dropout)
        self.layer6 = GatedResidualFusion(dim, gate_init)

        type_ids = torch.cat([
            torch.full((max_entities,), SLOT_ENTITY),
            torch.full((max_entities,), SLOT_ATTRIBUTE),
            torch.full((num_action_slots,), SLOT_ACTION),
            torch.full((max_entities,), SLOT_RELATION),
        ])
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.num_slots = n_slots

    def forward(self, tokens: Tensor, det: Detections, return_aux: bool = False):
        cand, mask = self.layer1(tokens, det)                    # [B, 28, d], [B, 28]
        slots = self.layer2(cand, mask, self.type_ids)           # [B, 28, d]
        slots = self.layer3(slots, mask)                         # [B, 28, d]
        slots, reliability = self.layer4(slots, mask)            # [B, 28, d], [B, 28, 1]
        if return_aux:
            ctx, xattn = self.layer5(tokens, slots, mask, return_weights=True)
        else:
            ctx = self.layer5(tokens, slots, mask)               # [B, 197, d]
        out = self.layer6(tokens, ctx)                           # [B, 197, d]
        if not return_aux:
            return out
        return out, {
            "slot_mask": mask,              # [B, 28]
            "slot_types": self.type_ids,    # [28]
            "reliability": reliability,     # [B, 28, 1]
            "cross_attn": xattn,            # [B, heads, 197, 28]
        }

    def layer_param_counts(self) -> dict:
        return {
            name: sum(p.numel() for p in getattr(self, name).parameters())
            for name in ("layer1", "layer2", "layer3", "layer4", "layer5", "layer6")
        }

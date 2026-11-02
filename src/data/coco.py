"""
MS COCO 2017 caption dataset
============================
- Loads train / val splits
- Returns one caption per image per step (sampled randomly at train time,
  first caption at val time for determinism)
- Images resized / normalised with CLIP's preprocessing pipeline
- Up to `max_captions_per_image` kept; extras discarded at load time
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CocoDataset(Dataset):
    """
    Parameters
    ----------
    image_dir           : path to COCO images (e.g. train2017/)
    ann_file            : path to captions JSON (captions_train2017.json)
    processor           : CLIPProcessor for image pre-processing
    tokenizer           : BART tokenizer for caption tokenisation
    max_length          : max caption token length (for padding / truncation)
    max_captions        : keep at most this many captions per image
    split               : "train" (random sample) or "val" (first caption)
    seed                : random seed for caption sampling
    """

    def __init__(
        self,
        image_dir: str | Path,
        ann_file: str | Path,
        processor: CLIPProcessor,
        tokenizer,
        max_length: int = 64,
        max_captions: int = 5,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.image_dir   = Path(image_dir)
        self.processor   = processor
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.max_captions= max_captions
        self.split       = split
        self.rng         = random.Random(seed)

        # ── Load annotations ────────────────────────────────────────
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Build id → filename map
        id2file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}

        # Build id → list[caption] map (cap to max_captions)
        id2caps: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in id2caps:
                id2caps[iid] = []
            if len(id2caps[iid]) < max_captions:
                id2caps[iid].append(ann["caption"])

        # Flat list of (file_name, [captions])
        self.samples: list[tuple[str, list[str]]] = [
            (id2file[iid], caps)
            for iid, caps in id2caps.items()
            if iid in id2file
        ]

        # For val: also store image_id for CHAIR / metrics
        self.image_ids: list[int] = [
            iid for iid in id2caps if iid in id2file
        ]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        filename, captions = self.samples[idx]
        image_id = self.image_ids[idx]

        # ── Load & preprocess image ──────────────────────────────────
        img_path = self.image_dir / filename
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)   # [3, 224, 224]

        # ── Select caption ───────────────────────────────────────────
        if self.split == "train":
            caption = self.rng.choice(captions)
        else:
            caption = captions[0]

        # ── Tokenise caption ─────────────────────────────────────────
        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc.input_ids.squeeze(0)       # [T]
        attention_mask = enc.attention_mask.squeeze(0)  # [T]

        # Labels: -100 for padding positions (ignored in CE loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values"      : pixel_values,
            "labels"            : labels,
            "decoder_attention_mask": attention_mask,
            "image_id"          : image_id,
            "caption"           : caption,
        }


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
    processor: CLIPProcessor,
    tokenizer,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from config dict.

    cfg keys used
    -------------
    data.train_image_dir, data.val_image_dir,
    data.train_ann_file,  data.val_ann_file,
    data.max_captions_per_image,
    data.num_workers, data.pin_memory,
    training.batch_size
    """
    d  = cfg["data"]
    t  = cfg["training"]

    train_ds = CocoDataset(
        image_dir    = d["train_image_dir"],
        ann_file     = d["train_ann_file"],
        processor    = processor,
        tokenizer    = tokenizer,
        max_captions = d.get("max_captions_per_image", 5),
        split        = "train",
        seed         = seed,
    )

    val_ds = CocoDataset(
        image_dir    = d["val_image_dir"],
        ann_file     = d["val_ann_file"],
        processor    = processor,
        tokenizer    = tokenizer,
        max_captions = d.get("max_captions_per_image", 5),
        split        = "val",
        seed         = seed,
    )

    def _worker_init(worker_id: int) -> None:
        # Each worker gets a deterministic but distinct seed
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size  = t["batch_size"],
        shuffle     = True,
        num_workers = d.get("num_workers", 4),
        pin_memory  = d.get("pin_memory", True),
        worker_init_fn = _worker_init,
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = t.get("eval_batch_size", t["batch_size"]),
        shuffle     = False,
        num_workers = d.get("num_workers", 4),
        pin_memory  = d.get("pin_memory", True),
        drop_last   = False,
    )

    return train_loader, val_loader

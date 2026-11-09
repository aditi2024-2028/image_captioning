"""
scripts/evaluate.py
===================
Load the epoch-5 checkpoint, run deterministic beam search on the full
COCO 2017 val set, save predictions JSON, then compute all metrics.

Usage
-----
    python scripts/evaluate.py --config configs/e0r.yaml \
        --checkpoint outputs/e0r/ckpt_epoch05.pt \
        [--device cuda]
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from tqdm import tqdm
from transformers import CLIPProcessor, BartTokenizer

from src.models import E0RCaptioner
from src.data.coco import CocoDataset
from src.eval   import compute_metrics, compute_chair
from torch.utils.data import DataLoader

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/e0r.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    e_cfg  = cfg["eval"]
    m_cfg  = cfg["model"]
    d_cfg  = cfg["data"]

    # ── Validate checkpoint epoch ────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_epoch = ckpt.get("epoch", -1)
    if ckpt_epoch != cfg["training"]["num_epochs"]:
        logger.warning(
            "Checkpoint epoch %d != expected epoch %d. "
            "Proceeding anyway — make sure this is intentional.",
            ckpt_epoch, cfg["training"]["num_epochs"],
        )
    else:
        logger.info("Confirmed checkpoint epoch = %d ✓", ckpt_epoch)

    # ── Build model ──────────────────────────────────────────────────────
    model_cfg = ckpt.get("model_config", {})
    model = E0RCaptioner(
        clip_model_name        = model_cfg.get("clip_model_name", m_cfg["clip_model_name"]),
        bart_model_name        = model_cfg.get("bart_model_name", m_cfg["bart_model_name"]),
        proj_dim               = model_cfg.get("proj_dim",        m_cfg.get("proj_dim", 768)),
        adapter_bottleneck_dim = model_cfg.get("adapter_bottleneck_dim",
                                               m_cfg.get("adapter_bottleneck_dim", 64)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    logger.info("Model loaded from %s", args.checkpoint)

    # ── Dataset ──────────────────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(m_cfg["clip_model_name"])
    tokenizer = BartTokenizer.from_pretrained(m_cfg["bart_model_name"])

    # Recreate the exact validation split used during training
    import csv
    import random

    image_names = set()

    with open(d_cfg["val_ann_file"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) >= 2:
                image_names.add(row[0].split("#")[0].strip())

    image_names = sorted(image_names)

    rng = random.Random(42)
    rng.shuffle(image_names)

    split_index = int(len(image_names) * 0.8)
    val_names = image_names[split_index:]

    logger.info("Evaluation validation images: %d", len(val_names))

    val_ds = CocoDataset(
        image_dir    = d_cfg["val_image_dir"],
        ann_file     = d_cfg["val_ann_file"],
        processor    = processor,
        tokenizer    = tokenizer,
        max_captions = d_cfg.get("max_captions_per_image", 5),
        image_names  = val_names,
        train        = False,
    )

    eval_batch = e_cfg.get("batch_size", 32)
    loader = DataLoader(
        val_ds,
        batch_size  = eval_batch,
        shuffle     = False,
        num_workers = d_cfg.get("num_workers", 4),
        pin_memory  = True,
    )

    # ── Generation kwargs from config ────────────────────────────────────
    gen_kwargs = {
        "num_beams"          : e_cfg["beam_size"],
        "max_new_tokens"     : e_cfg["max_new_tokens"],
        "min_length"         : e_cfg.get("min_length", 5),
        "length_penalty"     : e_cfg.get("length_penalty", 1.0),
        "no_repeat_ngram_size": e_cfg.get("no_repeat_ngram_size", 3),
        "early_stopping"     : True,
    }
    logger.info("Generation kwargs: %s", gen_kwargs)

    # ── Beam search ──────────────────────────────────────────────────────
    predictions: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            image_ids    = batch["image_id"] if isinstance(batch["image_id"], list) else batch["image_id"].tolist()

            token_ids = model.generate(pixel_values, **gen_kwargs)  # [B, T]
            captions  = tokenizer.batch_decode(token_ids, skip_special_tokens=True)

            for iid, cap in zip(image_ids, captions):
                predictions.append({"image_id": iid, "caption": cap})


    # ── Map Flickr8k filenames to COCO integer IDs ─────────────────
    import csv
    import random

    ann_csv = d_cfg["val_ann_file"]

    image_names = set()

    with open(ann_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) >= 2:
                image_names.add(row[0].split("#")[0].strip())

    image_names = sorted(image_names)

    image_id_map = {
        name: idx for idx, name in enumerate(image_names)
    }

    # Recreate the exact 80:20 split used during training
    shuffled_names = image_names.copy()
    rng = random.Random(42)
    rng.shuffle(shuffled_names)

    split_index = int(len(shuffled_names) * 0.8)
    val_names = shuffled_names[split_index:]

    val_ids = {
        image_id_map[name] for name in val_names
    }

    logger.info("Validation images: %d", len(val_names))

    # Convert prediction filenames into integer IDs
    val_name_set = set(val_names)

    predictions = [
        {
            "image_id": image_id_map[p["image_id"]],
            "caption": p["caption"]
        }
        for p in predictions
        if p["image_id"] in image_id_map
        and p["image_id"] in val_name_set
    ]

    logger.info("Validation predictions: %d", len(predictions))

    # Create validation-only COCO annotations
    with open(e_cfg["annotations_file"], "r") as f:
        coco_data = json.load(f)

    val_coco = {
        **coco_data,
        "images": [
            img for img in coco_data["images"]
            if img["id"] in val_ids
        ],
        "annotations": [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in val_ids
        ]
    }

    val_ann_file = os.path.join(
        os.path.dirname(e_cfg["annotations_file"]),
        "flickr8k_val_coco.json"
    )

    with open(val_ann_file, "w") as f:
        json.dump(val_coco, f)

    # ── Save validation predictions ───────────────────────────────
    pred_file = e_cfg.get(
        "predictions_file",
        "outputs/e0r/predictions.json"
    )

    os.makedirs(os.path.dirname(pred_file), exist_ok=True)

    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(
        "Saved %d validation predictions -> %s",
        len(predictions), pred_file
    )

    # ── Calculate metrics ─────────────────────────────────────────
    metrics = compute_metrics(predictions, val_ann_file)
    chair = compute_chair(predictions, val_ann_file)
    all_m = {**metrics, **chair}

    logger.info("=" * 60)
    for k, v in all_m.items():
        logger.info("  %-12s: %.4f", k, v)
    logger.info("=" * 60)

    # Save metrics alongside predictions
    metrics_file = pred_file.replace("predictions.json", "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"epoch": ckpt_epoch, **all_m}, f, indent=2)
    logger.info("Metrics saved → %s", metrics_file)


if __name__ == "__main__":
    main()

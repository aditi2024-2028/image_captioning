
import argparse
import logging
import os
import sys
import random

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

import numpy as np
import torch
import yaml

from transformers import CLIPProcessor, BartTokenizer

from src.models import E0RCaptioner
from src.data import build_dataloaders
from src.training import Trainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train E0-R captioner")
    parser.add_argument("--config", default="configs/e0r.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(
        "Run: %s | device: %s | seed: %d",
        cfg["run_name"], device, seed
    )

    # Load processor and tokenizer
    m_cfg = cfg["model"]

    processor = CLIPProcessor.from_pretrained(
        m_cfg["clip_model_name"]
    )

    tokenizer = BartTokenizer.from_pretrained(
        m_cfg["bart_model_name"]
    )

    # Flickr8k dataloaders
    d = cfg["data"]
    t = cfg["training"]

    train_loader, val_loader = build_dataloaders(
        train_image_dir=d["train_image_dir"],
        val_image_dir=d["val_image_dir"],
        train_ann_file=d["train_ann_file"],
        val_ann_file=d["val_ann_file"],
        processor=processor,
        tokenizer=tokenizer,
        max_length=64,
        max_captions=d.get("max_captions_per_image", 5),
        batch_size=t["batch_size"],
        num_workers=d.get("num_workers", 2),
        pin_memory=d.get("pin_memory", True),
    )

    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader), len(val_loader)
    )

    # Initialize E0R model
    model = E0RCaptioner(
        clip_model_name=m_cfg["clip_model_name"],
        bart_model_name=m_cfg["bart_model_name"],
        proj_dim=m_cfg.get("proj_dim", 768),
        adapter_bottleneck_dim=m_cfg.get(
            "adapter_bottleneck_dim", 64
        ),
    )

    total_params = sum(p.numel() for p in model.parameters())

    logger.info("Model total params: %s", f"{total_params:,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        resume_path=args.resume,
    )

    trainer.train()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

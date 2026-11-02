"""
scripts/train.py
================
Entry point for E0-R training.

Usage
-----
    python scripts/train.py --config configs/e0r.yaml
    python scripts/train.py --config configs/e0r.yaml --resume outputs/e0r/ckpt_epoch02.pt
"""

import argparse
import logging
import os
import sys

# Allow imports from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
import torch
import yaml

from transformers import CLIPProcessor, BartTokenizer

from src.models  import E0RCaptioner
from src.data    import build_dataloaders
from src.training import Trainer


logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Validate that no TODO values remain for required training keys
    _check_todos(cfg)
    return cfg


def _check_todos(cfg: dict, _prefix: str = "") -> None:
    for k, v in cfg.items():
        full_key = f"{_prefix}.{k}" if _prefix else k
        if isinstance(v, dict):
            _check_todos(v, full_key)
        elif v == "TODO":
            raise ValueError(
                f"Config key '{full_key}' is still set to TODO. "
                "Please fill it in configs/e0r.yaml before training."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E0-R captioner")
    parser.add_argument("--config", default="configs/e0r.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed   = cfg.get("seed", 42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Global seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Run: %s  |  device: %s  |  seed: %d", cfg["run_name"], device, seed)

    # ── Processors / tokenizers ──────────────────────────────────────────
    m_cfg     = cfg["model"]
    processor = CLIPProcessor.from_pretrained(m_cfg["clip_model_name"])
    tokenizer = BartTokenizer.from_pretrained(m_cfg["bart_model_name"])

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, processor, tokenizer, seed=seed)
    logger.info("Train batches: %d  |  Val batches: %d",
                len(train_loader), len(val_loader))

    # ── Model ────────────────────────────────────────────────────────────
    model = E0RCaptioner(
        clip_model_name       = m_cfg["clip_model_name"],
        bart_model_name       = m_cfg["bart_model_name"],
        proj_dim              = m_cfg.get("proj_dim", 768),
        adapter_bottleneck_dim= m_cfg.get("adapter_bottleneck_dim", 64),
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model total params: %s", f"{total_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model       = model,
        train_loader= train_loader,
        val_loader  = val_loader,
        cfg         = cfg,
        device      = device,
        resume_path = args.resume,
    )
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()

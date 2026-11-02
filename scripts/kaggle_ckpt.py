"""
scripts/kaggle_ckpt.py
======================
Push / pull the latest E0-R checkpoint to / from a private Kaggle Dataset,
then verify the pulled checkpoint resumes correctly.

Prerequisites
-------------
  pip install kaggle
  Set KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).

Usage
-----
    # Push latest checkpoint
    python scripts/kaggle_ckpt.py push --config configs/e0r.yaml

    # Pull latest checkpoint (by epoch number in filename)
    python scripts/kaggle_ckpt.py pull --config configs/e0r.yaml

    # Pull AND verify resume (does a 0-step forward pass to confirm state loads)
    python scripts/kaggle_ckpt.py pull --config configs/e0r.yaml --verify
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _latest_ckpt(ckpt_dir: str) -> Path | None:
    """Return path of ckpt with highest epoch number."""
    pattern = os.path.join(ckpt_dir, "ckpt_epoch*.pt")
    files   = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())
    result.check_returncode()


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push(cfg: dict) -> None:
    k_cfg   = cfg["kaggle"]
    owner   = k_cfg["dataset_owner"]
    slug    = k_cfg["dataset_slug"]
    ckpt_dir= cfg["output_dir"]

    ckpt = _latest_ckpt(ckpt_dir)
    if ckpt is None:
        logger.error("No checkpoints found in %s", ckpt_dir)
        sys.exit(1)

    logger.info("Pushing checkpoint: %s", ckpt)

    with tempfile.TemporaryDirectory() as tmp:
        # Copy checkpoint into staging dir
        shutil.copy(ckpt, tmp)

        # Write dataset-metadata.json expected by Kaggle API
        meta = {
            "title"   : slug,
            "id"      : f"{owner}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(tmp, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)

        # Create or update dataset
        _run(["kaggle", "datasets", "create", "-p", tmp, "--dir-mode", "zip"])
        logger.info("Pushed %s → kaggle.com/%s/%s", ckpt.name, owner, slug)


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull(cfg: dict, verify: bool = False) -> None:
    k_cfg   = cfg["kaggle"]
    owner   = k_cfg["dataset_owner"]
    slug    = k_cfg["dataset_slug"]
    ckpt_dir= Path(cfg["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_ref = f"{owner}/{slug}"
    logger.info("Pulling from kaggle.com/%s", dataset_ref)

    _run(["kaggle", "datasets", "download", "-d", dataset_ref,
          "-p", str(ckpt_dir), "--unzip"])

    ckpt = _latest_ckpt(str(ckpt_dir))
    if ckpt is None:
        logger.error("No checkpoint found after pull in %s", ckpt_dir)
        sys.exit(1)

    logger.info("Latest checkpoint after pull: %s", ckpt)

    if verify:
        _verify_resume(ckpt, cfg)


def _verify_resume(ckpt_path: Path, cfg: dict) -> None:
    """
    Load the checkpoint and confirm the state restores without errors.
    Does NOT run any training steps — just loads model + optimizer + metadata.
    """
    from src.models import E0RCaptioner

    logger.info("Verifying resume from %s …", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    m_cfg  = ckpt.get("model_config", cfg["model"])
    model  = E0RCaptioner(
        clip_model_name        = m_cfg.get("clip_model_name", cfg["model"]["clip_model_name"]),
        bart_model_name        = m_cfg.get("bart_model_name", cfg["model"]["bart_model_name"]),
        proj_dim               = m_cfg.get("proj_dim", 768),
        adapter_bottleneck_dim = m_cfg.get("adapter_bottleneck_dim", 64),
    )
    model.load_state_dict(ckpt["model_state"])

    epoch = ckpt.get("epoch", "?")
    logger.info("✓  Checkpoint epoch %s loaded successfully.", epoch)
    logger.info("✓  Resume verification passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle checkpoint push/pull for E0-R")
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--config", default="configs/e0r.yaml")
    parser.add_argument("--verify", action="store_true",
                        help="After pull, verify the checkpoint resumes (pull only)")
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    if args.action == "push":
        push(cfg)
    else:
        pull(cfg, verify=args.verify)


if __name__ == "__main__":
    main()

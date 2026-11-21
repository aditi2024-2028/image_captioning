"""
caption.py
==========
Generate a caption for a single image using a trained E0-R checkpoint.

Usage
-----
    python caption.py path/to/image.jpg --checkpoint outputs/e0r/ckpt_epoch05.pt
    python caption.py path/to/image.jpg --checkpoint outputs/e0r/ckpt_epoch05.pt --device cpu
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from transformers import CLIPProcessor, BartTokenizer
from src.models import E0RCaptioner


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg = ckpt.get("model_config", {})
    model = E0RCaptioner(
        clip_model_name        = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch16"),
        bart_model_name        = model_cfg.get("bart_model_name", "facebook/bart-base"),
        proj_dim               = model_cfg.get("proj_dim", 768),
        adapter_bottleneck_dim = model_cfg.get("adapter_bottleneck_dim", 64),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    clip_model_name = model_cfg.get("clip_model_name", "openai/clip-vit-base-patch16")
    bart_model_name = model_cfg.get("bart_model_name", "facebook/bart-base")

    return model, clip_model_name, bart_model_name


class CaptionGenerator:
    def __init__(self, checkpoint_path: str, device: str):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        self.model, clip_model_name, bart_model_name = load_model(checkpoint_path, device)
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)

    def caption_image(self, image_path: str) -> str:
        print(f"Loading image from {image_path} ...")
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        print("Generating caption ...")
        with torch.no_grad():
            token_ids = self.model.generate(
                pixel_values,
                num_beams           = 4,
                max_new_tokens      = 30,
                min_length          = 5,
                length_penalty      = 1.0,
                no_repeat_ngram_size= 3,
                early_stopping      = True,
            )

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)


def caption_image(image_path: str, checkpoint_path: str, device: str) -> str:
    return CaptionGenerator(checkpoint_path, device).caption_image(image_path)


def main():
    parser = argparse.ArgumentParser(description="Caption a single image with E0-R")
    parser.add_argument("image",       help="Path to the image file")
    parser.add_argument("--checkpoint",required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--device",    default=None,  help="Device: cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    caption = caption_image(args.image, args.checkpoint, device)

    print(f"\nCaption: {caption}\n")


if __name__ == "__main__":
    main()

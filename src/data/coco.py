
import os
import csv
import random
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CocoDataset(Dataset):

    def __init__(
        self,
        image_dir,
        ann_file,
        processor,
        tokenizer,
        max_length=64,
        max_captions=5,
        split="train",
        seed=42,
        image_names=None,
        train=None
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        captions = defaultdict(list)

        with open(ann_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 2:
                    continue

                filename = row[0].strip()
                caption = ",".join(row[1:]).strip()

                if filename.lower() in ("image", "image_name", "filename"):
                    continue

                if filename and caption:
                    captions[filename].append(caption)

        self.samples = []

        for filename, caps in captions.items():
            image_path = self.image_dir / filename

            if image_path.exists():
                self.samples.append(
                    (filename, caps[:max_captions])
                )

        rng = random.Random(seed)
        rng.shuffle(self.samples)

        if train is not None:
            split = "train" if train else "val"

        if image_names is not None:
            selected_names = set(image_names)
            self.samples = [
                sample for sample in self.samples
                if sample[0] in selected_names
            ]
        else:
            split_idx = int(0.8 * len(self.samples))

            if split == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

        print(f"{split} samples: {len(self.samples)}")

        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, captions = self.samples[idx]

        image = Image.open(
            self.image_dir / filename
        ).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        if self.split == "train":
            caption = self.rng.choice(captions)
        else:
            caption = captions[0]

        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_attention_mask": attention_mask,
            "image_id": filename,
            "caption": caption
        }


def build_dataloaders(
    train_image_dir,
    val_image_dir,
    train_ann_file,
    val_ann_file,
    processor,
    tokenizer,
    max_length=64,
    max_captions=5,
    batch_size=4,
    num_workers=2,
    pin_memory=True
):

    train_ds = CocoDataset(
        image_dir=train_image_dir,
        ann_file=train_ann_file,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        max_captions=max_captions,
        split="train"
    )

    val_ds = CocoDataset(
        image_dir=val_image_dir,
        ann_file=val_ann_file,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        max_captions=max_captions,
        split="val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader

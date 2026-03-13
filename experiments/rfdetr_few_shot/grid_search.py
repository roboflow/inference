#!/usr/bin/env python3
"""RF-DETR Few-Shot LoRA Grid Search.

Self-contained script that evaluates combinations of
(rank, epochs, learning_rate, num_train_images) against held-out validation
images and logs every result to a SQLite journal.

Primary metrics: mAP@50 and mAP@50:95 (COCO-style).
Also captures per-image recall, precision, F1 at various thresholds.

Usage:
    python grid_search.py                    # full phase 1 grid (overnight)
    python grid_search.py --smoke            # quick smoke test
    python grid_search.py --resume           # resume (skips completed runs)
    python grid_search.py --phase2           # phase 2 broader sweep
    python grid_search.py --phase2 --smoke   # phase 2 smoke test

    # RF20-VL-FSOD Benchmark mode:
    python grid_search.py --download-datasets            # download 20 FSOD datasets
    python grid_search.py --benchmark --top-n 10         # benchmark top 10 configs
    python grid_search.py --benchmark --ablate-group-detr # + group_detr ablation
"""

import argparse
import concurrent.futures
import copy
import json
import logging
import os
import random
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_models"))

DATASET_ROOT = Path.home() / "Downloads" / "uavdet-small-txtvh-ysli.v1i.yolo26"
WEIGHTS_PATH = Path("/tmp/rf-detr-base-coco.pth")
DB_PATH = Path(__file__).parent / "results.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "grid_search.log"),
    ],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["bicycle", "bus", "car", "human", "motorbike", "truck", "van"]


# ═══════════════════════════════════════════════════════════════════════
# Self-contained dataset
# ═══════════════════════════════════════════════════════════════════════

def parse_yolo_label(label_path: Path) -> List[dict]:
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:]]
            boxes.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def load_image_and_labels(split: str, stem: str) -> Tuple[Image.Image, List[dict]]:
    img_path = DATASET_ROOT / split / "images" / f"{stem}.jpg"
    lbl_path = DATASET_ROOT / split / "labels" / f"{stem}.txt"
    assert img_path.exists(), f"Image not found: {img_path}"
    assert lbl_path.exists(), f"Label not found: {lbl_path}"
    return Image.open(img_path).convert("RGB"), parse_yolo_label(lbl_path)


def load_image_and_labels_generic(dataset_path: Path, split: str, stem: str) -> Tuple[Image.Image, List[dict]]:
    """Load image+labels from an arbitrary YOLO dataset. Handles .jpg/.png/.jpeg."""
    img_dir = dataset_path / split / "images"
    lbl_path = dataset_path / split / "labels" / f"{stem}.txt"
    # Try common extensions
    img_path = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    assert img_path is not None, f"Image not found for stem {stem} in {img_dir}"
    assert lbl_path.exists(), f"Label not found: {lbl_path}"
    return Image.open(img_path).convert("RGB"), parse_yolo_label(lbl_path)


class InlineFewShotDataset(Dataset):
    """DETR-format dataset from PIL images + YOLO boxes.  No inference imports."""

    def __init__(self, images_and_boxes, active_classes, resolution, augment=True,
                 augmentation_level=0, class_names=None):
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.cls2id = {name: i for i, name in enumerate(active_classes)}
        self.resolution = resolution
        self.augment = augment
        self.augmentation_level = augmentation_level
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Pre-resize images once (avoids expensive PIL resize per __getitem__ call)
        self.items = []
        for pil_img, boxes in images_and_boxes:
            resized = pil_img.resize((resolution, resolution))
            base_tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
            self.items.append((base_tensor, boxes))

        # Build augmentation transforms (on [0,1] tensors, before normalization)
        if augmentation_level >= 1:
            from torchvision.transforms import ColorJitter
            self.color_jitter = ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
            )
        if augmentation_level >= 2:
            from torchvision.transforms import RandomResizedCrop
            self.random_crop = RandomResizedCrop(
                size=resolution, scale=(0.8, 1.0), ratio=(0.9, 1.1),
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base_tensor, yolo_boxes = self.items[idx]
        img_t = base_tensor.clone()  # clone so augmentations don't mutate cached tensor

        # Build normalised cxcywh boxes
        labels, boxes = [], []
        for b in yolo_boxes:
            cls_name = self.class_names[b["class_id"]]
            if cls_name not in self.cls2id:
                continue
            labels.append(self.cls2id[cls_name])
            boxes.append([b["cx"], b["cy"], b["w"], b["h"]])

        labels_t = torch.tensor(labels, dtype=torch.long)
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        if self.augment:
            # Level 1+: Color jitter (on [0,1] tensor, before normalization)
            if self.augmentation_level >= 1:
                img_t = self.color_jitter(img_t)

            # Level 2: Random resized crop with box adjustment
            if self.augmentation_level >= 2 and boxes_t.numel() > 0:
                img_t, boxes_t, labels_t = self._apply_random_crop(
                    img_t, boxes_t, labels_t,
                )

            # Level 0+: Random horizontal flip
            if random.random() > 0.5:
                img_t = img_t.flip(-1)
                if boxes_t.numel() > 0:
                    boxes_t[:, 0] = 1.0 - boxes_t[:, 0]

        # Normalize with ImageNet stats
        img_t = (img_t - self.mean) / self.std

        return img_t, {"labels": labels_t, "boxes": boxes_t}

    def _apply_random_crop(self, img_t, boxes_t, labels_t, max_retries=3):
        """Apply RandomResizedCrop to image and adjust boxes accordingly.

        Retries if all boxes are cropped out. Falls back to no-crop after max_retries.
        """
        from torchvision.transforms import RandomResizedCrop
        import torchvision.transforms.functional as F

        res = self.resolution
        for attempt in range(max_retries):
            # Get random crop params: (top, left, height, width) in pixel space
            i, j, h, w = RandomResizedCrop.get_params(
                img_t, scale=(0.8, 1.0), ratio=(0.9, 1.1),
            )
            # Apply crop and resize back to resolution
            cropped_img = F.resized_crop(img_t, i, j, h, w, [res, res])

            # Remap boxes from normalised [0,1] coords to crop coords
            # Original pixel coords: cx_px = cx * res, cy_px = cy * res, etc.
            # Crop window: top=i, left=j, height=h, width=w
            new_cx = (boxes_t[:, 0] * res - j) / w
            new_cy = (boxes_t[:, 1] * res - i) / h
            new_bw = boxes_t[:, 2] * res / w
            new_bh = boxes_t[:, 3] * res / h

            new_boxes = torch.stack([new_cx, new_cy, new_bw, new_bh], dim=1)

            # Compute visible area ratio after clamping
            # Convert to xyxy for clamping
            x1 = (new_cx - new_bw / 2).clamp(0, 1)
            y1 = (new_cy - new_bh / 2).clamp(0, 1)
            x2 = (new_cx + new_bw / 2).clamp(0, 1)
            y2 = (new_cy + new_bh / 2).clamp(0, 1)

            clamped_area = (x2 - x1) * (y2 - y1)
            original_area = new_bw * new_bh
            visible_ratio = clamped_area / original_area.clamp(min=1e-6)

            # Keep boxes with at least 20% visible area
            keep = visible_ratio > 0.2
            if keep.any():
                # Recompute cxcywh from clamped xyxy
                kept_boxes = torch.stack([
                    (x1[keep] + x2[keep]) / 2,
                    (y1[keep] + y2[keep]) / 2,
                    (x2[keep] - x1[keep]),
                    (y2[keep] - y1[keep]),
                ], dim=1)
                return cropped_img, kept_boxes, labels_t[keep]

        # All retries failed (all boxes cropped out) — return original
        return img_t, boxes_t, labels_t


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ═══════════════════════════════════════════════════════════════════════
# Model loading & inference
# ═══════════════════════════════════════════════════════════════════════

def load_base_model(device: torch.device):
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model
    from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
        CONFIG_FOR_MODEL_TYPE,
    )
    config = CONFIG_FOR_MODEL_TYPE["rfdetr-base"](device=device)
    weights = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)["model"]
    config.num_classes = weights["class_embed.bias"].shape[0] - 1
    model = build_model(config=config)
    model.load_state_dict(weights)
    model = model.eval().to(device)
    return model, config


def run_inference(model, config, pil_image, confidence_threshold=0.01):
    """Run inference, returning ALL detections above a low threshold for mAP."""
    from inference_models.models.rfdetr.post_processor import PostProcess

    device = next(model.parameters()).device
    res = config.resolution

    img_resized = pil_image.resize((res, res))
    img_t = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_t)

    pp = PostProcess(num_select=300)
    orig_sizes = torch.tensor([[pil_image.height, pil_image.width]], device=device)
    results = pp(outputs, orig_sizes)[0]

    dets = []
    for i in range(len(results["scores"])):
        if results["scores"][i] < confidence_threshold:
            continue
        dets.append({
            "class_id": results["labels"][i].item(),
            "confidence": results["scores"][i].item(),
            "box": results["boxes"][i].cpu().tolist(),
        })
    return dets


# ═══════════════════════════════════════════════════════════════════════
# Self-contained trainer
# ═══════════════════════════════════════════════════════════════════════

BACKBONE_LORA_TARGETS = ["query", "value"]
DECODER_LORA_TARGETS = ["self_attn.out_proj", "linear1", "linear2"]

# Expanded targets for Phase 3: add backbone key + cross-attention (MSDeformAttn)
BACKBONE_LORA_TARGETS_V2 = ["query", "key", "value"]
DECODER_LORA_TARGETS_V2 = [
    "self_attn.out_proj", "linear1", "linear2",
    "sampling_offsets", "attention_weights",  # cross-attn in MSDeformAttn
]


def train_lora(
    base_model, config, dataset, num_classes, device,
    rank=8, alpha=16, lr=2e-3, num_epochs=25,
    lora_dropout=0.0, weight_decay=1e-4,
):
    from peft import LoraConfig, get_peft_model
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model
    from inference_models.models.rfdetr.few_shot.criterion import SetCriterion
    from inference_models.models.rfdetr.few_shot.matcher import HungarianMatcher

    fresh_config = config.model_copy(update={"num_classes": num_classes})
    model = build_model(config=fresh_config)

    base_state = base_model.state_dict()
    model_state = model.state_dict()
    filtered = {k: v for k, v in base_state.items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.reinitialize_detection_head(num_classes + 1)
    model = model.to(device)

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=lora_dropout,
        target_modules=BACKBONE_LORA_TARGETS + DECODER_LORA_TARGETS,
        bias="none",
        use_dora=False,
    )
    peft_model = get_peft_model(model, lora_cfg)

    for name, p in peft_model.named_parameters():
        if any(kw in name for kw in ("class_embed", "bbox_embed",
                                      "enc_out_class_embed", "enc_out_bbox_embed")):
            p.requires_grad = True

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("LoRA trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    dataloader = DataLoader(dataset, batch_size=min(len(dataset), 4),
                            shuffle=True, collate_fn=collate_fn, num_workers=0)

    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
    weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
    for i in range(fresh_config.dec_layers - 1):
        weight_dict.update({f"{k}_{i}": v for k, v in
                            {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}.items()})

    criterion = SetCriterion(
        num_classes=num_classes + 1, matcher=matcher, weight_dict=weight_dict,
        focal_alpha=0.25, losses=["labels", "boxes", "cardinality"],
        group_detr=getattr(config, "group_detr", 1),
        ia_bce_loss=getattr(config, "ia_bce_loss", True),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    peft_model.train()
    criterion.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                     enabled=device.type == "cuda"):
                outputs = peft_model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict if k in weight_dict)

            losses.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()
            epoch_loss += losses.item()

        scheduler.step()
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
            logger.info("  Epoch %d/%d  loss=%.4f", epoch + 1, num_epochs, epoch_loss)

    merged = peft_model.merge_and_unload().eval()
    return merged, epoch_losses, None  # no adapter_state in legacy path


def prepare_template_model(base_model, config, num_classes, device):
    """Build a template model once — deepcopy it for each experiment instead of
    calling build_model() (~5s) every time. Returns (template_model, base_filtered_state)."""
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    # group_detr=1: 2.4x training speedup (13→1 groups). Only affects matching
    # diversity during training — does not change model architecture or inference.
    fresh_config = config.model_copy(update={"num_classes": num_classes, "group_detr": 1})
    template = build_model(config=fresh_config)

    base_state = base_model.state_dict()
    template_state = template.state_dict()
    filtered = {k: v for k, v in base_state.items()
                if k in template_state and v.shape == template_state[k].shape}
    template.load_state_dict(filtered, strict=False)
    template.reinitialize_detection_head(num_classes + 1)
    template = template.to(device)

    return template, filtered, fresh_config


def prepare_criterion(config, num_classes, device):
    """Build criterion + matcher once, reuse across experiments."""
    from inference_models.models.rfdetr.few_shot.criterion import SetCriterion
    from inference_models.models.rfdetr.few_shot.matcher import HungarianMatcher

    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
    weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
    for i in range(config.dec_layers - 1):
        weight_dict.update({f"{k}_{i}": v for k, v in
                            {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}.items()})

    criterion = SetCriterion(
        num_classes=num_classes + 1, matcher=matcher, weight_dict=weight_dict,
        focal_alpha=0.25, losses=["labels", "boxes", "cardinality"],
        group_detr=getattr(config, "group_detr", 1),
        ia_bce_loss=getattr(config, "ia_bce_loss", True),
    ).to(device)

    return criterion, weight_dict


def _gpu_copy_paste(images, boxes_list, labels_list, resolution):
    """Copy-paste augmentation: paste objects from random donor images onto each image.
    Operates on GPU tensors in [0,1] range. Applied with probability 0.5 per image.
    """
    B = images.shape[0]
    if B < 2:
        return images, boxes_list, labels_list  # Need at least 2 images

    for i in range(B):
        if random.random() > 0.5:
            continue

        # Pick a random donor (different from current)
        donor_idx = random.choice([j for j in range(B) if j != i])
        donor_boxes = boxes_list[donor_idx]
        donor_labels = labels_list[donor_idx]

        if donor_boxes.numel() == 0:
            continue

        # Pick 1-2 random objects from donor
        n_objects = min(random.randint(1, 2), len(donor_labels))
        obj_indices = random.sample(range(len(donor_labels)), n_objects)

        for obj_idx in obj_indices:
            # Get object bbox in pixel coords (cxcywh normalised → pixel)
            cx, cy, bw, bh = donor_boxes[obj_idx].tolist()
            x1 = int((cx - bw / 2) * resolution)
            y1 = int((cy - bh / 2) * resolution)
            x2 = int((cx + bw / 2) * resolution)
            y2 = int((cy + bh / 2) * resolution)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(resolution, x2), min(resolution, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop object from donor
            obj_crop = images[donor_idx, :, y1:y2, x1:x2].clone()

            # Random scale 0.7-1.3
            scale = random.uniform(0.7, 1.3)
            new_h = max(2, int((y2 - y1) * scale))
            new_w = max(2, int((x2 - x1) * scale))
            new_h = min(new_h, resolution)
            new_w = min(new_w, resolution)

            obj_crop = torch.nn.functional.interpolate(
                obj_crop.unsqueeze(0), size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

            # Random paste position (ensure within bounds)
            paste_x = random.randint(0, max(0, resolution - new_w))
            paste_y = random.randint(0, max(0, resolution - new_h))

            # Paste onto target image
            images[i, :, paste_y:paste_y + new_h, paste_x:paste_x + new_w] = obj_crop

            # Add box for pasted object (normalised cxcywh)
            new_cx = (paste_x + new_w / 2) / resolution
            new_cy = (paste_y + new_h / 2) / resolution
            new_bw_norm = new_w / resolution
            new_bh_norm = new_h / resolution
            new_box = torch.tensor(
                [[new_cx, new_cy, new_bw_norm, new_bh_norm]],
                device=images.device, dtype=torch.float32,
            )
            new_label = donor_labels[obj_idx:obj_idx + 1]

            boxes_list[i] = torch.cat([boxes_list[i], new_box], dim=0)
            labels_list[i] = torch.cat([labels_list[i], new_label], dim=0)

    return images, boxes_list, labels_list


def _gpu_mosaic(gpu_base_images, gpu_boxes, gpu_labels, resolution, device):
    """Mosaic augmentation: stitch 4 random images into a 2×2 grid.
    Returns a single (3, resolution, resolution) image with combined boxes.
    Applied per-sample with probability 0.3.
    """
    n = len(gpu_base_images)
    if n < 4:
        return None, None, None  # Not enough images

    # Pick 4 random indices
    chosen = random.sample(range(n), 4)
    half = resolution // 2

    mosaic_img = torch.zeros(3, resolution, resolution, device=device)
    all_boxes = []
    all_labels = []

    positions = [(0, 0), (half, 0), (0, half), (half, half)]  # (x_off, y_off)
    for idx, (x_off, y_off) in zip(chosen, positions):
        # Resize image to half resolution
        img_half = torch.nn.functional.interpolate(
            gpu_base_images[idx].unsqueeze(0), size=(half, half),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        mosaic_img[:, y_off:y_off + half, x_off:x_off + half] = img_half

        # Adjust boxes: scale to half and offset
        if gpu_boxes[idx].numel() > 0:
            bx = gpu_boxes[idx].clone()
            # Scale cx, cy, w, h from [0,1] in full image to position in mosaic
            bx[:, 0] = (bx[:, 0] * half + x_off) / resolution  # cx
            bx[:, 1] = (bx[:, 1] * half + y_off) / resolution  # cy
            bx[:, 2] = bx[:, 2] * half / resolution  # w
            bx[:, 3] = bx[:, 3] * half / resolution  # h
            all_boxes.append(bx)
            all_labels.append(gpu_labels[idx].clone())

    if all_boxes:
        return mosaic_img, torch.cat(all_boxes, dim=0), torch.cat(all_labels, dim=0)
    return mosaic_img, torch.zeros((0, 4), device=device), torch.tensor([], dtype=torch.long, device=device)


def _gpu_augment_batch(images, boxes_list, labels_list, aug_level, resolution,
                       copy_paste=False, gpu_base_images=None, gpu_boxes=None,
                       gpu_labels=None, mosaic=False, multi_scale=False, device=None):
    """Apply augmentations directly on GPU tensors. Much faster than CPU augmentation.
    images: (B, 3, H, W) on GPU, [0,1] range (pre-normalization)
    Returns augmented images and adjusted boxes/labels.
    """
    from torchvision.transforms import functional as F
    B = images.shape[0]
    device = device or images.device

    # Copy-paste augmentation (before other augmentations)
    if copy_paste and B >= 2:
        images, boxes_list, labels_list = _gpu_copy_paste(
            images, boxes_list, labels_list, resolution,
        )

    # Multi-scale: randomly resize to 0.8-1.2x
    if multi_scale:
        scale = random.uniform(0.8, 1.2)
        new_res = int(resolution * scale)
        if new_res != resolution:
            images = torch.nn.functional.interpolate(
                images, size=(new_res, new_res), mode="bilinear", align_corners=False,
            )
            # Resize back to expected resolution
            images = torch.nn.functional.interpolate(
                images, size=(resolution, resolution), mode="bilinear", align_corners=False,
            )
            # Boxes stay the same (normalised coords)

    # Level 1+: Color jitter on GPU
    if aug_level >= 1:
        for i in range(B):
            # Random brightness, contrast, saturation, hue — apply directly on GPU tensor
            images[i] = F.adjust_brightness(images[i], 1.0 + random.uniform(-0.2, 0.2))
            images[i] = F.adjust_contrast(images[i], 1.0 + random.uniform(-0.2, 0.2))
            images[i] = F.adjust_saturation(images[i], 1.0 + random.uniform(-0.2, 0.2))
            images[i] = F.adjust_hue(images[i], random.uniform(-0.1, 0.1))

    # Level 2: Random resized crop with box adjustment
    if aug_level >= 2:
        from torchvision.transforms import RandomResizedCrop
        for i in range(B):
            if boxes_list[i].numel() == 0:
                continue
            for _attempt in range(3):
                top, left, h, w = RandomResizedCrop.get_params(
                    images[i], scale=(0.8, 1.0), ratio=(0.9, 1.1),
                )
                cropped = F.resized_crop(images[i], top, left, h, w, [resolution, resolution])
                bx = boxes_list[i]
                new_cx = (bx[:, 0] * resolution - left) / w
                new_cy = (bx[:, 1] * resolution - top) / h
                new_bw = bx[:, 2] * resolution / w
                new_bh = bx[:, 3] * resolution / h
                x1 = (new_cx - new_bw / 2).clamp(0, 1)
                y1 = (new_cy - new_bh / 2).clamp(0, 1)
                x2 = (new_cx + new_bw / 2).clamp(0, 1)
                y2 = (new_cy + new_bh / 2).clamp(0, 1)
                clamped_area = (x2 - x1) * (y2 - y1)
                orig_area = (new_bw * new_bh).clamp(min=1e-6)
                keep = (clamped_area / orig_area) > 0.2
                if keep.any():
                    images[i] = cropped
                    boxes_list[i] = torch.stack([
                        (x1[keep] + x2[keep]) / 2, (y1[keep] + y2[keep]) / 2,
                        x2[keep] - x1[keep], y2[keep] - y1[keep],
                    ], dim=1)
                    labels_list[i] = labels_list[i][keep]
                    break

    # Level 0+: Random horizontal flip
    for i in range(B):
        if random.random() > 0.5:
            images[i] = images[i].flip(-1)
            if boxes_list[i].numel() > 0:
                boxes_list[i][:, 0] = 1.0 - boxes_list[i][:, 0]

    return images, boxes_list, labels_list


def train_lora_fast(
    template_model, filtered_state, config, dataset, num_classes, device,
    criterion, weight_dict,
    rank=8, alpha=16, lr=2e-3, num_epochs=25,
    lora_dropout=0.0, weight_decay=1e-4,
    class_names=None,
    batch_size=None, lora_targets_version="v1",
    copy_paste=False, mosaic=False, warmup=False, multi_scale=False,
):
    """Optimised trainer: deepcopy (~0.3s) instead of build_model (~5s),
    GPU-side augmentation, reuses criterion across experiments."""
    from peft import LoraConfig, get_peft_model

    # ~0.3s vs ~5s for build_model
    model = copy.deepcopy(template_model)
    model.reinitialize_detection_head(num_classes + 1)

    # Select LoRA target modules
    if lora_targets_version == "v2":
        targets = BACKBONE_LORA_TARGETS_V2 + DECODER_LORA_TARGETS_V2
    else:
        targets = BACKBONE_LORA_TARGETS + DECODER_LORA_TARGETS

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
        use_dora=False,
    )
    peft_model = get_peft_model(model, lora_cfg)

    for name, p in peft_model.named_parameters():
        if any(kw in name for kw in ("class_embed", "bbox_embed",
                                      "enc_out_class_embed", "enc_out_bbox_embed")):
            p.requires_grad = True

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info("LoRA trainable: %d / %d (%.2f%%)", trainable, total_params, 100 * trainable / total_params)

    _class_names = class_names if class_names is not None else CLASS_NAMES

    # Pre-load images + labels to GPU once (tiny: 1-5 images)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    aug_level = dataset.augmentation_level
    resolution = dataset.resolution

    gpu_base_images = []  # [0,1] range, on GPU
    gpu_labels = []
    gpu_boxes = []
    for idx in range(len(dataset)):
        base_tensor, yolo_boxes = dataset.items[idx]
        gpu_base_images.append(base_tensor.to(device))
        labels, boxes = [], []
        for b in yolo_boxes:
            cls_name = _class_names[b["class_id"]]
            if cls_name not in dataset.cls2id:
                continue
            labels.append(dataset.cls2id[cls_name])
            boxes.append([b["cx"], b["cy"], b["w"], b["h"]])
        gpu_labels.append(torch.tensor(labels, dtype=torch.long, device=device))
        gpu_boxes.append(torch.tensor(boxes, dtype=torch.float32, device=device) if boxes
                         else torch.zeros((0, 4), device=device))

    n_images = len(gpu_base_images)
    effective_batch_size = batch_size if batch_size else min(n_images, 4)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    # Scheduler: optional warmup + cosine annealing
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )
    if warmup:
        warmup_epochs = max(1, num_epochs // 10)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    peft_model.train()
    criterion.train()
    epoch_losses = []
    indices = list(range(n_images))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        random.shuffle(indices)

        for start in range(0, n_images, effective_batch_size):
            batch_idx = indices[start:start + effective_batch_size]

            # Mosaic augmentation: replace some batch items with mosaic composites
            if mosaic and n_images >= 4:
                new_imgs, new_boxes, new_labels = [], [], []
                for bi in batch_idx:
                    if random.random() < 0.3:
                        m_img, m_boxes, m_labels = _gpu_mosaic(
                            gpu_base_images, gpu_boxes, gpu_labels, resolution, device,
                        )
                        if m_img is not None:
                            new_imgs.append(m_img)
                            new_boxes.append(m_boxes)
                            new_labels.append(m_labels)
                            continue
                    new_imgs.append(gpu_base_images[bi].clone())
                    new_boxes.append(gpu_boxes[bi].clone())
                    new_labels.append(gpu_labels[bi].clone())
                batch_imgs = torch.stack(new_imgs)
                batch_boxes = new_boxes
                batch_labels = new_labels
            else:
                # Clone base images and apply GPU augmentation
                batch_imgs = torch.stack([gpu_base_images[i].clone() for i in batch_idx])
                batch_boxes = [gpu_boxes[i].clone() for i in batch_idx]
                batch_labels = [gpu_labels[i].clone() for i in batch_idx]

            if dataset.augment:
                batch_imgs, batch_boxes, batch_labels = _gpu_augment_batch(
                    batch_imgs, batch_boxes, batch_labels, aug_level, resolution,
                    copy_paste=copy_paste, gpu_base_images=gpu_base_images,
                    gpu_boxes=gpu_boxes, gpu_labels=gpu_labels,
                    multi_scale=multi_scale, device=device,
                )

            # Normalize
            batch_imgs = (batch_imgs - mean) / std
            targets = [{"labels": l, "boxes": b} for l, b in zip(batch_labels, batch_boxes)]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                     enabled=device.type == "cuda"):
                outputs = peft_model(batch_imgs)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict if k in weight_dict)

            losses.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], max_norm=1.0,
            )
            optimizer.step()
            epoch_loss += losses.item()

        scheduler.step()
        epoch_losses.append(epoch_loss)
        if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
            logger.info("  Epoch %d/%d  loss=%.4f", epoch + 1, num_epochs, epoch_loss)

    # Capture adapter state before merging (for persistence / cache)
    adapter_state = {
        "lora_state_dict": {
            k: v.cpu() for k, v in peft_model.state_dict().items() if "lora_" in k
        },
        "head_state_dict": {
            "class_embed": {
                k: v.cpu() for k, v in peft_model.base_model.model.class_embed.state_dict().items()
            },
            "bbox_embed": {
                k: v.cpu() for k, v in peft_model.base_model.model.bbox_embed.state_dict().items()
            },
        },
        "class_names": class_names if class_names else [],
        "num_classes": num_classes,
        "lora_rank": rank,
        "lora_alpha": alpha,
    }
    if hasattr(peft_model.base_model.model, "transformer") and hasattr(
        peft_model.base_model.model.transformer, "enc_out_class_embed"
    ):
        adapter_state["head_state_dict"]["enc_out_class_embed"] = {
            k: v.cpu()
            for k, v in peft_model.base_model.model.transformer.enc_out_class_embed.state_dict().items()
        }

    merged = peft_model.merge_and_unload().eval()
    return merged, epoch_losses, adapter_state


# ═══════════════════════════════════════════════════════════════════════
# mAP computation (COCO-style)
# ═══════════════════════════════════════════════════════════════════════

def compute_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def compute_ap_single_class(preds, gt_boxes, iou_threshold):
    """Compute AP for a single class at a single IoU threshold.

    preds: list of (confidence, [x1,y1,x2,y2]) sorted by confidence desc
    gt_boxes: list of [x1,y1,x2,y2]

    Returns AP (float).
    """
    if len(gt_boxes) == 0:
        return 0.0 if len(preds) > 0 else None  # None = class not in GT

    if len(preds) == 0:
        return 0.0

    gt_matched = [False] * len(gt_boxes)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (conf, pred_box) in enumerate(preds):
        best_iou, best_j = 0, -1
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            gt_matched[best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum)

    # COCO-style 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= 101.0
    return ap


def compute_map(predictions, gt_boxes, image_size, active_class_ids, iou_thresholds,
                class_names=None):
    """Compute mAP across classes and IoU thresholds.

    predictions: list of {class_id, confidence, box: [x1,y1,x2,y2]}
    gt_boxes: list of {class_id, cx, cy, w, h} (YOLO normalised)
    image_size: (w, h) tuple
    active_class_ids: set of class IDs present in training data
    iou_thresholds: list of IoU thresholds

    Returns dict with:
        mAP_50: mAP at IoU=0.5
        mAP_50_95: mAP at IoU=0.5:0.95 (COCO)
        per_class_ap: {class_name: {iou_thresh: ap}}
        recall_at_conf: {conf_thresh: recall} at IoU=0.5
        precision_at_conf: {conf_thresh: precision} at IoU=0.5
    """
    w_img, h_img = image_size

    # Convert GT to absolute xyxy
    gt_by_class = {}
    for b in gt_boxes:
        cid = b["class_id"]
        if cid not in active_class_ids:
            continue
        cx, cy, bw, bh = b["cx"]*w_img, b["cy"]*h_img, b["w"]*w_img, b["h"]*h_img
        box = [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]
        gt_by_class.setdefault(cid, []).append(box)

    # Group predictions by class, sorted by confidence
    preds_by_class = {}
    for p in predictions:
        cid = p["class_id"]
        preds_by_class.setdefault(cid, []).append((p["confidence"], p["box"]))
    for cid in preds_by_class:
        preds_by_class[cid].sort(key=lambda x: x[0], reverse=True)

    # Compute AP per class per IoU threshold
    _class_names = class_names if class_names is not None else CLASS_NAMES
    all_class_ids = active_class_ids
    per_class_ap = {}
    ap_by_iou = {t: [] for t in iou_thresholds}

    for cid in all_class_ids:
        cls_name = _class_names[cid]
        cls_gt = gt_by_class.get(cid, [])
        cls_preds = preds_by_class.get(cid, [])
        per_class_ap[cls_name] = {}

        for iou_t in iou_thresholds:
            ap = compute_ap_single_class(cls_preds, cls_gt, iou_t)
            if ap is not None:
                per_class_ap[cls_name][str(iou_t)] = ap
                ap_by_iou[iou_t].append(ap)

    # mAP = mean across classes that have GT
    def mean_or_zero(lst):
        return float(np.mean(lst)) if lst else 0.0

    mAP_50 = mean_or_zero(ap_by_iou.get(0.5, []))

    # mAP@50:95 — average across IoU thresholds 0.5, 0.55, ..., 0.95
    coco_thresholds = [t for t in iou_thresholds if 0.5 <= t <= 0.95]
    if coco_thresholds:
        mAP_50_95 = mean_or_zero([mean_or_zero(ap_by_iou[t]) for t in coco_thresholds])
    else:
        mAP_50_95 = 0.0

    # Also compute recall/precision at specific confidence thresholds (IoU=0.5)
    conf_metrics = {}
    for conf_t in [0.1, 0.3, 0.5]:
        filtered = [p for p in predictions if p["confidence"] >= conf_t]
        # Simple matching at IoU=0.5
        all_gt = []
        for cid in active_class_ids:
            for box in gt_by_class.get(cid, []):
                all_gt.append({"class_id": cid, "box": box})

        preds_sorted = sorted(filtered, key=lambda p: p["confidence"], reverse=True)
        gt_matched = [False] * len(all_gt)
        tp = fp = 0
        for pred in preds_sorted:
            best_iou, best_idx = 0, -1
            for gi, gt in enumerate(all_gt):
                if gt_matched[gi] or gt["class_id"] != pred["class_id"]:
                    continue
                iou = compute_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= 0.5 and best_idx >= 0:
                gt_matched[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn = sum(1 for m in gt_matched if not m)
        total_gt = len(all_gt)
        recall = tp / total_gt if total_gt > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
        conf_metrics[str(conf_t)] = {
            "recall": recall, "precision": precision, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "total_gt": total_gt, "total_pred": len(preds_sorted),
        }

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_ap": per_class_ap,
        "conf_metrics": conf_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# DB journal
# ═══════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        lora_rank INTEGER, lora_alpha INTEGER,
        num_epochs INTEGER, learning_rate REAL,
        num_train_images INTEGER,
        train_image_set TEXT,
        train_time_seconds REAL, time_per_epoch_ms REAL,
        final_loss REAL, loss_history TEXT,
        -- primary metrics (averaged across eval images)
        mAP_50 REAL, mAP_50_95 REAL,
        status TEXT DEFAULT 'pending',
        error_message TEXT, device TEXT, notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT, eval_split TEXT,
        -- mAP per image
        mAP_50 REAL, mAP_50_95 REAL,
        -- per-class AP as JSON {class_name: {iou: ap}}
        per_class_ap_json TEXT,
        -- recall/precision/f1 at specific conf thresholds (IoU=0.5)
        -- stored as JSON {conf_thresh: {recall, precision, f1, tp, fp, fn, ...}}
        conf_metrics_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS train_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        image_stem TEXT, split TEXT,
        num_boxes INTEGER, num_classes INTEGER,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    conn.commit()
    return conn


def experiment_done(conn, run_id):
    cur = conn.execute("SELECT status FROM experiments WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    return row is not None and row[0] == "completed"


def make_run_id(rank, alpha, epochs, lr, train_set_key):
    return f"r{rank}_a{alpha}_ep{epochs}_lr{lr}_ts{train_set_key}"


# ═══════════════════════════════════════════════════════════════════════
# Experiment configuration
# ═══════════════════════════════════════════════════════════════════════

TRAIN_IMAGE_SETS = {
    "set_A": [
        "8680_jpg.rf.706fa18b1b812fb7da3083c9cd832ef5",   # 6 cls, 93 boxes
        "1140_jpg.rf.b7a413338df1a5c16cf45ec6ea1342a6",   # 4 cls, 48 boxes
        "0650_jpg.rf.de746ab0b7e41f9d4c2c3da5055ceadf",   # 3 cls, 55 boxes
        "1725_jpg.rf.1e4bbf4317a3feafa8b7933ed637ea0e",   # 4 cls, 58 boxes
        "2245_jpg.rf.efdcba8fe6d9922b81b59124d0e831df",   # 4 cls, 60 boxes
    ],
    "set_B": [
        "8630_jpg.rf.4bbe9f22eb7b0f7fd34a4d1e0b1ed206",   # 6 cls, 93 boxes
        "8525_jpg.rf.8c773d91794e8fd5264904b225bcc7f1",   # 5 cls, 92 boxes
        "1290_jpg.rf.0318d5ce152b69e75993e905663eddf5",   # 4 cls, 50 boxes
        "1060_jpg.rf.98c50fe550a0539770acc6ad933e203d",   # 3 cls, 49 boxes
        "1355_jpg.rf.589e4093ce171212a7046a13c0fe25c0",   # 4 cls, 54 boxes
    ],
}

EVAL_IMAGES = [
    "8690_jpg.rf.0c941d7feb3c50a0cdb44040edb2c1b9",   # 6 cls, 91 boxes
    "3475_jpg.rf.3a9b141c0ffbcfe5ed40cc81751728ea",   # 6 cls, 82 boxes
    "3290_jpg.rf.781a67727f322b29b738c560f3505d7a",   # 6 cls, 80 boxes
    "5915_jpg.rf.9ed53c86b92f653d3f1e416cc70138c7",   # 6 cls, 85 boxes
    "3150_jpg.rf.f1e3653e299f2f070b295f0f909c193f",   # 6 cls, 79 boxes
]

# COCO-style IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95
MAP_IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]

GRID = {
    "rank": [4, 8, 16, 32],
    "epochs": [10, 25, 50, 100],
    "learning_rate": [1e-3, 2e-3, 5e-3],
    "num_train_images": [1, 2, 5],
}

SMOKE_GRID = {
    "rank": [8],
    "epochs": [5],
    "learning_rate": [2e-3],
    "num_train_images": [1],
}

# ── Phase 2: broader sweep around Phase 1 best ───────────────────────
GRID_PHASE2 = {
    "lora_dropout": [0.0, 0.05, 0.1],
    "augmentation_level": [0, 1, 2],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "alpha_ratio": [1, 2, 4],
    "num_train_images": [1, 2, 5],
}
PHASE2_FIXED = {"rank": 4, "epochs": 50, "learning_rate": 2e-3}

SMOKE_GRID_PHASE2 = {
    "lora_dropout": [0.0],
    "augmentation_level": [1],
    "weight_decay": [1e-4],
    "alpha_ratio": [2],
    "num_train_images": [1],
}

DB_PATH_PHASE2 = Path(__file__).parent / "results_phase2.db"

# ── Phase 3: augmentation + rank + batch + LoRA targets sweep ─────────
GRID_PHASE3 = {
    "copy_paste": [False, True],           # 2
    "mosaic": [False, True],               # 2
    "rank": [4, 8, 12],                    # 3
    "batch_size": [4, 8, 16],             # 3
    "lora_targets": ["v1", "v2"],          # 2
    "warmup": [False, True],               # 2
    "multi_scale": [False, True],          # 2
}
PHASE3_FIXED = {"epochs": 50, "learning_rate": 2e-3, "alpha_ratio": 2}

SMOKE_GRID_PHASE3 = {
    "copy_paste": [True],
    "mosaic": [False],
    "rank": [8],
    "batch_size": [8],
    "lora_targets": ["v1"],
    "warmup": [True],
    "multi_scale": [False],
}

# Phase 3a: augmentation + rank + batch (v1 targets, no warmup, no multi_scale)
# 2×2×3×3 = 36 configs × 20 datasets = 720 experiments
GRID_PHASE3A = {
    "copy_paste": [False, True],
    "mosaic": [False, True],
    "rank": [4, 8, 12],
    "batch_size": [4, 8, 16],
}
PHASE3A_FIXED = {
    "epochs": 50, "learning_rate": 2e-3, "alpha_ratio": 2,
    "lora_targets": "v1", "warmup": False, "multi_scale": False,
}

# Phase 3b: same grid as 3a but with Phase 2 winning fixed params
# (Phase 2 best: alpha_ratio=1, wd=1e-3 beat alpha_ratio=2, wd=1e-4)
# 2×2×3×3 = 36 configs × 20 datasets = 720 experiments
GRID_PHASE3B = {
    "copy_paste": [False, True],
    "mosaic": [False, True],
    "rank": [4, 8, 12],
    "batch_size": [4, 8, 16],
}
PHASE3B_FIXED = {
    "epochs": 50, "learning_rate": 2e-3, "alpha_ratio": 1,
    "weight_decay": 1e-3,
    "lora_targets": "v1", "warmup": False, "multi_scale": False,
}


# ═══════════════════════════════════════════════════════════════════════
# Main experiment loop
# ═══════════════════════════════════════════════════════════════════════

def run_single_experiment(
    conn, base_model, config, rank, epochs, lr,
    num_train_images, train_set_name, train_stems, eval_stems, device,
):
    alpha = rank * 2
    run_id = make_run_id(rank, alpha, epochs, lr, f"{train_set_name}_{num_train_images}")

    if experiment_done(conn, run_id):
        logger.info("⏭  Skip %s (done)", run_id)
        return

    # Delete any previous failed/partial attempt with this run_id
    cur = conn.execute("SELECT id FROM experiments WHERE run_id = ?", (run_id,))
    old = cur.fetchone()
    if old:
        conn.execute("DELETE FROM eval_results WHERE experiment_id = ?", (old[0],))
        conn.execute("DELETE FROM train_images WHERE experiment_id = ?", (old[0],))
        conn.execute("DELETE FROM experiments WHERE id = ?", (old[0],))
        conn.commit()

    logger.info("🚀 %s  rank=%d ep=%d lr=%s n=%d set=%s",
                run_id, rank, epochs, lr, num_train_images, train_set_name)

    cur = conn.execute(
        """INSERT INTO experiments
           (run_id, timestamp, lora_rank, lora_alpha, num_epochs, learning_rate,
            num_train_images, train_image_set, status, device)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
        (run_id, datetime.now().isoformat(), rank, alpha, epochs, lr,
         num_train_images, train_set_name, str(device)),
    )
    exp_id = cur.lastrowid
    conn.commit()

    try:
        # Load training images
        selected = train_stems[:num_train_images]
        images_and_boxes = []
        all_class_ids = set()

        for stem in selected:
            img, boxes = load_image_and_labels("train", stem)
            images_and_boxes.append((img, boxes))
            for b in boxes:
                all_class_ids.add(b["class_id"])
            conn.execute(
                "INSERT INTO train_images (experiment_id, image_stem, split, num_boxes, num_classes) VALUES (?,?,'train',?,?)",
                (exp_id, stem, len(boxes), len(set(b["class_id"] for b in boxes))),
            )

        active_classes = [CLASS_NAMES[i] for i in sorted(all_class_ids)]
        num_classes = len(active_classes)

        dataset = InlineFewShotDataset(images_and_boxes, active_classes,
                                       config.resolution, augment=True,
                                       class_names=CLASS_NAMES)

        # Train
        t0 = time.time()
        merged_model, loss_history = train_lora(
            base_model, config, dataset, num_classes, device,
            rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
        )
        train_time = time.time() - t0
        ms_per_epoch = (train_time / epochs) * 1000

        conn.execute(
            """UPDATE experiments SET train_time_seconds=?, time_per_epoch_ms=?,
               final_loss=?, loss_history=? WHERE id=?""",
            (train_time, ms_per_epoch, loss_history[-1],
             json.dumps(loss_history), exp_id),
        )
        conn.commit()
        logger.info("✅ Trained in %.1fs (%.0f ms/ep)", train_time, ms_per_epoch)

        # Evaluate — get ALL detections at low confidence for proper mAP
        active_to_original = {i: CLASS_NAMES.index(c) for i, c in enumerate(active_classes)}
        all_mAP_50 = []
        all_mAP_50_95 = []

        for eval_stem in eval_stems:
            eval_img, eval_boxes = load_image_and_labels("valid", eval_stem)

            # Get detections at very low threshold for full P-R curve
            raw_dets = run_inference(merged_model, config, eval_img, confidence_threshold=0.01)
            remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                        for d in raw_dets if d["class_id"] in active_to_original]

            metrics = compute_map(
                remapped, eval_boxes, eval_img.size,
                all_class_ids, MAP_IOU_THRESHOLDS,
            )

            all_mAP_50.append(metrics["mAP_50"])
            all_mAP_50_95.append(metrics["mAP_50_95"])

            conn.execute(
                """INSERT INTO eval_results
                   (experiment_id, eval_image_stem, eval_split,
                    mAP_50, mAP_50_95, per_class_ap_json, conf_metrics_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (exp_id, eval_stem, "valid",
                 metrics["mAP_50"], metrics["mAP_50_95"],
                 json.dumps(metrics["per_class_ap"]),
                 json.dumps(metrics["conf_metrics"])),
            )

        # Store averaged mAP on the experiment row
        avg_mAP_50 = float(np.mean(all_mAP_50)) if all_mAP_50 else 0
        avg_mAP_50_95 = float(np.mean(all_mAP_50_95)) if all_mAP_50_95 else 0

        conn.execute(
            "UPDATE experiments SET mAP_50=?, mAP_50_95=?, status='completed' WHERE id=?",
            (avg_mAP_50, avg_mAP_50_95, exp_id),
        )
        conn.commit()
        logger.info("📊 %s  mAP@50=%.1f%%  mAP@50:95=%.1f%%",
                    run_id, avg_mAP_50*100, avg_mAP_50_95*100)

        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error("❌ %s failed: %s", run_id, e, exc_info=True)
        conn.execute("UPDATE experiments SET status='failed', error_message=? WHERE id=?",
                     (str(e), exp_id))
        conn.commit()


def init_db_phase2(db_path=None):
    db = db_path or DB_PATH_PHASE2
    conn = sqlite3.connect(str(db), timeout=30)  # 30s timeout for concurrent workers
    conn.execute("PRAGMA journal_mode=WAL")  # allow concurrent readers/writers
    conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        lora_rank INTEGER, lora_alpha INTEGER,
        num_epochs INTEGER, learning_rate REAL,
        num_train_images INTEGER,
        train_image_set TEXT,
        -- phase 2 sweep params
        lora_dropout REAL,
        augmentation_level INTEGER,
        weight_decay REAL,
        alpha_ratio INTEGER,
        --
        train_time_seconds REAL, time_per_epoch_ms REAL,
        final_loss REAL, loss_history TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        status TEXT DEFAULT 'pending',
        error_message TEXT, device TEXT, notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT, eval_split TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        per_class_ap_json TEXT,
        conf_metrics_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS train_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        image_stem TEXT, split TEXT,
        num_boxes INTEGER, num_classes INTEGER,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    conn.commit()
    return conn


def make_run_id_phase2(dropout, aug_level, wd, alpha_ratio, n_train, train_set_key):
    return f"do{dropout}_aug{aug_level}_wd{wd}_ar{alpha_ratio}_n{n_train}_ts{train_set_key}"


def _db_connect(db_path):
    """Create a SQLite connection with WAL mode and generous timeout for concurrency."""
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")  # 120s busy wait
    return conn


def _db_execute_with_retry(conn, sql, params=(), max_retries=10):
    """Execute a DB statement with retry on OperationalError (locked)."""
    for attempt in range(max_retries):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait = 1.0 + random.random() * 2
                logger.warning("DB locked (attempt %d/%d), retrying in %.1fs...",
                               attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise


def _db_commit_with_retry(conn, max_retries=10):
    """Commit with retry on OperationalError (locked)."""
    for attempt in range(max_retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait = 1.0 + random.random() * 2
                logger.warning("DB commit locked (attempt %d/%d), retrying in %.1fs...",
                               attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise


def run_single_experiment_phase2(
    db_path, base_model, config, dropout, aug_level, wd, alpha_ratio,
    num_train_images, train_set_name, train_stems, eval_stems, device,
    template_model=None, filtered_state=None, fresh_config=None,
    cached_criterion=None, cached_weight_dict=None,
    class_names=None, load_fn=None, eval_split="valid",
):
    """Run a single phase 2 experiment. Process-safe: creates own DB connection."""
    _class_names = class_names if class_names is not None else CLASS_NAMES
    _load_fn = load_fn if load_fn is not None else load_image_and_labels
    rank = PHASE2_FIXED["rank"]
    epochs = PHASE2_FIXED["epochs"]
    lr = PHASE2_FIXED["learning_rate"]
    alpha = rank * alpha_ratio

    run_id = make_run_id_phase2(
        dropout, aug_level, wd, alpha_ratio, num_train_images, train_set_name,
    )

    conn = _db_connect(db_path)

    try:
        if experiment_done(conn, run_id):
            logger.info("\u23ed  Skip %s (done)", run_id)
            return

        # Delete any previous failed/partial attempt
        cur = _db_execute_with_retry(conn, "SELECT id FROM experiments WHERE run_id = ?", (run_id,))
        old = cur.fetchone()
        if old:
            _db_execute_with_retry(conn, "DELETE FROM eval_results WHERE experiment_id = ?", (old[0],))
            _db_execute_with_retry(conn, "DELETE FROM train_images WHERE experiment_id = ?", (old[0],))
            _db_execute_with_retry(conn, "DELETE FROM experiments WHERE id = ?", (old[0],))
            _db_commit_with_retry(conn)

        logger.info("\U0001f680 %s  do=%.2f aug=%d wd=%s ar=%d n=%d set=%s",
                    run_id, dropout, aug_level, wd, alpha_ratio, num_train_images, train_set_name)

        cur = _db_execute_with_retry(conn,
            """INSERT INTO experiments
               (run_id, timestamp, lora_rank, lora_alpha, num_epochs, learning_rate,
                num_train_images, train_image_set,
                lora_dropout, augmentation_level, weight_decay, alpha_ratio,
                status, device)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (run_id, datetime.now().isoformat(), rank, alpha, epochs, lr,
             num_train_images, train_set_name,
             dropout, aug_level, wd, alpha_ratio, str(device)),
        )
        exp_id = cur.lastrowid
        _db_commit_with_retry(conn)

        # Load training images
        selected = train_stems[:num_train_images]
        images_and_boxes = []
        all_class_ids = set()

        for stem in selected:
            img, boxes = _load_fn("train", stem)
            images_and_boxes.append((img, boxes))
            for b in boxes:
                all_class_ids.add(b["class_id"])
            _db_execute_with_retry(conn,
                "INSERT INTO train_images (experiment_id, image_stem, split, num_boxes, num_classes) VALUES (?,?,'train',?,?)",
                (exp_id, stem, len(boxes), len(set(b["class_id"] for b in boxes))),
            )

        active_classes = [_class_names[i] for i in sorted(all_class_ids)]
        num_classes = len(active_classes)

        dataset = InlineFewShotDataset(
            images_and_boxes, active_classes, config.resolution,
            augment=True, augmentation_level=aug_level,
            class_names=_class_names,
        )

        # Train — use fast path if template is available
        t0 = time.time()
        if template_model is not None and cached_criterion is not None:
            merged_model, loss_history = train_lora_fast(
                template_model, filtered_state, fresh_config,
                dataset, num_classes, device,
                cached_criterion, cached_weight_dict,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=dropout, weight_decay=wd,
                class_names=_class_names,
            )
        else:
            merged_model, loss_history = train_lora(
                base_model, config, dataset, num_classes, device,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=dropout, weight_decay=wd,
            )
        train_time = time.time() - t0
        ms_per_epoch = (train_time / epochs) * 1000

        _db_execute_with_retry(conn,
            """UPDATE experiments SET train_time_seconds=?, time_per_epoch_ms=?,
               final_loss=?, loss_history=? WHERE id=?""",
            (train_time, ms_per_epoch, loss_history[-1],
             json.dumps(loss_history), exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("\u2705 Trained in %.1fs (%.0f ms/ep)", train_time, ms_per_epoch)

        # Evaluate
        active_to_original = {i: _class_names.index(c) for i, c in enumerate(active_classes)}
        all_mAP_50 = []
        all_mAP_50_95 = []

        for eval_stem in eval_stems:
            eval_img, eval_boxes = _load_fn(eval_split, eval_stem)
            raw_dets = run_inference(merged_model, config, eval_img, confidence_threshold=0.01)
            remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                        for d in raw_dets if d["class_id"] in active_to_original]

            metrics = compute_map(
                remapped, eval_boxes, eval_img.size,
                all_class_ids, MAP_IOU_THRESHOLDS,
                class_names=_class_names,
            )

            all_mAP_50.append(metrics["mAP_50"])
            all_mAP_50_95.append(metrics["mAP_50_95"])

            _db_execute_with_retry(conn,
                """INSERT INTO eval_results
                   (experiment_id, eval_image_stem, eval_split,
                    mAP_50, mAP_50_95, per_class_ap_json, conf_metrics_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (exp_id, eval_stem, eval_split,
                 metrics["mAP_50"], metrics["mAP_50_95"],
                 json.dumps(metrics["per_class_ap"]),
                 json.dumps(metrics["conf_metrics"])),
            )

        avg_mAP_50 = float(np.mean(all_mAP_50)) if all_mAP_50 else 0
        avg_mAP_50_95 = float(np.mean(all_mAP_50_95)) if all_mAP_50_95 else 0

        _db_execute_with_retry(conn,
            "UPDATE experiments SET mAP_50=?, mAP_50_95=?, status='completed' WHERE id=?",
            (avg_mAP_50, avg_mAP_50_95, exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("\U0001f4ca %s  mAP@50=%.1f%%  mAP@50:95=%.1f%%",
                    run_id, avg_mAP_50*100, avg_mAP_50_95*100)

        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error("\u274c %s failed: %s", run_id, e, exc_info=True)
        try:
            _db_execute_with_retry(conn,
                "UPDATE experiments SET status='failed', error_message=? WHERE id=?",
                (str(e), exp_id))
            _db_commit_with_retry(conn)
        except Exception:
            pass
    finally:
        conn.close()


def compute_grid_total(grid):
    return (len(grid["rank"]) * len(grid["epochs"]) * len(grid["learning_rate"])
            * len(grid["num_train_images"]) * len(TRAIN_IMAGE_SETS))


def compute_grid_total_phase2(grid):
    total = 1
    for vals in grid.values():
        total *= len(vals)
    return total * len(TRAIN_IMAGE_SETS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase2", action="store_true",
                        help="Run phase 2 broader sweep around phase 1 best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel experiment workers (phase 2 only)")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker shard ID (0-based). Use with --total-workers for multi-process parallelism.")
    parser.add_argument("--total-workers", type=int, default=None,
                        help="Total number of worker shards")
    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true",
                        help="Run RF20-VL-FSOD benchmark across 20 datasets")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top Phase 2 configs to benchmark (default 10)")
    parser.add_argument("--phase1-db", type=str, default=None,
                        help="Path to Phase 1 results DB (default: results.db)")
    parser.add_argument("--phase2-db", type=str, default=None,
                        help="Path to Phase 2 results DB (default: results_phase2.db)")
    parser.add_argument("--benchmark-db", type=str, default=None,
                        help="Path to benchmark results DB (default: results_benchmark.db)")
    parser.add_argument("--datasets-root", type=str, default=None,
                        help="RF20-VL-FSOD datasets directory")
    parser.add_argument("--download-datasets", action="store_true",
                        help="Download RF20-VL-FSOD datasets in YOLO format")
    parser.add_argument("--ablate-group-detr", action="store_true",
                        help="Also run group_detr ablation {1, 5, 13} across all datasets")
    # Phase 3 mode
    parser.add_argument("--phase3", action="store_true",
                        help="Run Phase 3a benchmark (augmentation + rank + batch sweep)")
    parser.add_argument("--phase3b", action="store_true",
                        help="Run Phase 3b benchmark (+ alpha_ratio + weight_decay sweep)")
    parser.add_argument("--phase3-db", type=str, default=None,
                        help="Path to Phase 3 results DB (default: results_phase3.db)")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.phase3 or args.phase3b:
        main_phase3(args, device)
    elif args.benchmark or args.download_datasets:
        main_benchmark(args, device)
    elif args.phase2:
        main_phase2(args, device, workers=args.workers,
                    worker_id=args.worker_id, total_workers=args.total_workers)
    else:
        main_phase1(args, device)


def main_phase1(args, device):
    grid = SMOKE_GRID if args.smoke else GRID
    conn = init_db()

    total = compute_grid_total(grid)

    conn.execute("INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('total_experiments', ?)",
                 (str(total),))
    conn.execute("INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('grid_config', ?)",
                 (json.dumps(grid),))
    conn.execute("INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('start_time', ?)",
                 (datetime.now().isoformat(),))
    conn.commit()

    logger.info("\U0001f4cb DB: %s", DB_PATH)
    logger.info("\U0001f4ca Grid: %d experiments", total)

    logger.info("\U0001f504 Loading base model...")
    base_model, config = load_base_model(device)
    logger.info("\u2705 Base model loaded")

    done = 0
    for ts_name, ts_stems in TRAIN_IMAGE_SETS.items():
        for n_train in grid["num_train_images"]:
            for rank in grid["rank"]:
                for lr in grid["learning_rate"]:
                    for epochs in grid["epochs"]:
                        done += 1
                        logger.info("\u2501\u2501\u2501 [%d/%d] \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501", done, total)
                        run_single_experiment(
                            conn, base_model, config, rank, epochs, lr,
                            n_train, ts_name, ts_stems, EVAL_IMAGES, device,
                        )

    conn.execute("INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('end_time', ?)",
                 (datetime.now().isoformat(),))
    conn.commit()
    logger.info("\U0001f389 Grid search complete! Results: %s", DB_PATH)
    conn.close()


def _gpu_monitor_loop(db_path, interval=5, stop_event=None):
    """Background thread: log GPU utilization to DB every `interval` seconds."""
    import subprocess
    conn = _db_connect(db_path)
    _db_execute_with_retry(conn, """CREATE TABLE IF NOT EXISTS gpu_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        gpu_util_pct REAL,
        mem_used_mb REAL,
        mem_total_mb REAL,
        power_w REAL,
        temp_c REAL
    )""")
    _db_commit_with_retry(conn)

    while not (stop_event and stop_event.is_set()):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 5:
                    _db_execute_with_retry(conn,
                        "INSERT INTO gpu_stats (timestamp, gpu_util_pct, mem_used_mb, mem_total_mb, power_w, temp_c) VALUES (?,?,?,?,?,?)",
                        (datetime.now().isoformat(),
                         float(parts[0]), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])),
                    )
                    _db_commit_with_retry(conn)
        except Exception as e:
            logger.debug("GPU monitor error: %s", e)
        time.sleep(interval)
    conn.close()


def main_phase2(args, device, workers=1, worker_id=None, total_workers=None):
    grid = SMOKE_GRID_PHASE2 if args.smoke else GRID_PHASE2
    db_path = DB_PATH_PHASE2

    # Init DB schema (uses its own connection)
    conn = init_db_phase2(db_path)
    total = compute_grid_total_phase2(grid)

    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('total_experiments', ?)",
                 (str(total),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('grid_config', ?)",
                 (json.dumps(grid),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('phase', ?)",
                 ("2",))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('fixed_params', ?)",
                 (json.dumps(PHASE2_FIXED),))
    # Only set start_time if we're the main process (worker_id 0 or no sharding)
    if worker_id is None or worker_id == 0:
        _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('start_time', ?)",
                     (datetime.now().isoformat(),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("\U0001f4cb DB: %s", db_path)
    logger.info("\U0001f4ca Phase 2 grid: %d experiments total", total)
    logger.info("\U0001f4cc Fixed: rank=%d, epochs=%d, lr=%s",
                PHASE2_FIXED["rank"], PHASE2_FIXED["epochs"], PHASE2_FIXED["learning_rate"])

    # Start GPU monitor (only from worker 0 or single-process mode)
    gpu_stop = None
    if (worker_id is None or worker_id == 0) and device.type == "cuda":
        gpu_stop = threading.Event()
        gpu_thread = threading.Thread(target=_gpu_monitor_loop,
                                       args=(db_path, 5, gpu_stop), daemon=True)
        gpu_thread.start()
        logger.info("\U0001f4c8 GPU monitor started (every 5s)")

    logger.info("\U0001f504 Loading base model...")
    base_model, config = load_base_model(device)
    logger.info("\u2705 Base model loaded")

    # Pre-build template model and criterion for fast experiment loop.
    # We need to know num_classes from the dataset — load one image set to find out.
    sample_stems = list(TRAIN_IMAGE_SETS.values())[0]
    sample_classes = set()
    for stem in sample_stems[:5]:
        _, boxes = load_image_and_labels("train", stem)
        for b in boxes:
            sample_classes.add(b["class_id"])
    max_num_classes = len(sample_classes)
    logger.info("\U0001f3af Preparing template model (num_classes=%d)...", max_num_classes)
    template_model, filtered_state, fresh_config = prepare_template_model(
        base_model, config, max_num_classes, device,
    )
    cached_criterion, cached_weight_dict = prepare_criterion(
        fresh_config, max_num_classes, device,
    )
    logger.info("\u2705 Template model + criterion cached (deepcopy instead of build_model)")

    # Build list of all experiment configs
    experiment_configs = []
    for ts_name, ts_stems in TRAIN_IMAGE_SETS.items():
        for n_train in grid["num_train_images"]:
            for alpha_ratio in grid["alpha_ratio"]:
                for wd in grid["weight_decay"]:
                    for aug_level in grid["augmentation_level"]:
                        for dropout in grid["lora_dropout"]:
                            experiment_configs.append((
                                dropout, aug_level, wd, alpha_ratio,
                                n_train, ts_name, ts_stems,
                            ))

    # Apply sharding if --worker-id / --total-workers specified
    if worker_id is not None and total_workers is not None:
        my_configs = [cfg for i, cfg in enumerate(experiment_configs) if i % total_workers == worker_id]
        logger.info("\U0001f527 Worker %d/%d: handling %d/%d experiments",
                    worker_id, total_workers, len(my_configs), len(experiment_configs))
        experiment_configs = my_configs
    else:
        logger.info("\U0001f527 Single process: %d experiments", len(experiment_configs))

    for i, (dropout, aug_level, wd, alpha_ratio, n_train, ts_name, ts_stems) in enumerate(experiment_configs, 1):
        shard_label = f"W{worker_id}" if worker_id is not None else ""
        logger.info("\u2501\u2501\u2501 %s[%d/%d] \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
                    shard_label, i, len(experiment_configs))
        run_single_experiment_phase2(
            db_path, base_model, config,
            dropout, aug_level, wd, alpha_ratio,
            n_train, ts_name, ts_stems, EVAL_IMAGES, device,
            template_model=template_model,
            filtered_state=filtered_state,
            fresh_config=fresh_config,
            cached_criterion=cached_criterion,
            cached_weight_dict=cached_weight_dict,
        )

    # Stop GPU monitor
    if gpu_stop is not None:
        gpu_stop.set()

    logger.info("\U0001f389 Worker %s finished! Results: %s",
                str(worker_id) if worker_id is not None else "main", db_path)


# ═══════════════════════════════════════════════════════════════════════
# RF20-VL-FSOD Benchmark mode
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_DATASETS_ROOT = Path.home() / "Downloads" / "rf20-vl-fsod"
DB_PATH_BENCHMARK = Path(__file__).parent / "results_benchmark.db"
DB_PATH_PHASE3 = Path(__file__).parent / "results_phase3.db"


def discover_benchmark_datasets(root: Path) -> List[dict]:
    """Scan root for YOLO-format datasets with data.yaml.
    Returns [{name, path, num_classes, class_names, train_stems, test_stems}, ...]
    """
    import yaml

    datasets = []
    if not root.exists():
        logger.warning("Benchmark root not found: %s", root)
        return datasets

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        yaml_path = ds_dir / "data.yaml"
        if not yaml_path.exists():
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        names = data.get("names", [])
        if isinstance(names, dict):
            # Handle {0: 'class0', 1: 'class1', ...} format
            names = [names[k] for k in sorted(names.keys())]

        # Discover image stems in train and test splits
        train_stems = _discover_stems(ds_dir / "train" / "images")
        test_stems = _discover_stems(ds_dir / "test" / "images")

        if not train_stems:
            logger.warning("No train images found in %s", ds_dir.name)
            continue
        if not test_stems:
            logger.warning("No test images found in %s", ds_dir.name)
            continue

        datasets.append({
            "name": ds_dir.name,
            "path": ds_dir,
            "num_classes": len(names),
            "class_names": names,
            "train_stems": train_stems,
            "test_stems": test_stems,
        })

    logger.info("Discovered %d benchmark datasets in %s", len(datasets), root)
    return datasets


def _discover_stems(images_dir: Path) -> List[str]:
    """Get sorted list of image stems (without extension) from an images directory."""
    if not images_dir.exists():
        return []
    stems = set()
    for f in images_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            stems.add(f.stem)
    return sorted(stems)


def download_benchmark_datasets(root: Path):
    """Download RF20-VL-FSOD datasets in YOLO format.

    Uses the rf100-vl library if available, falls back to roboflow API directly.
    """
    try:
        # Try using the rf100-vl library (best approach — handles versions automatically)
        sys.path.insert(0, str(Path.home() / "rf100-vl"))
        from rf100vl.roboflow100vl import download_rf20vl_fsod
        logger.info("Downloading RF20-VL-FSOD via rf100-vl library (YOLO format)...")
        download_rf20vl_fsod(str(root), model_format="yolov5", overwrite=False)
        logger.info("✅ Download complete: %s", root)
        return
    except ImportError:
        logger.info("rf100-vl not found, using direct roboflow API...")
    except Exception as e:
        logger.warning("rf100-vl download failed (%s), falling back to direct API", e)

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        return

    # Direct download via roboflow API as fallback
    root.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))
    ws = rf.workspace("rf20-vl-fsod")

    for proj_info in ws.project_list:
        proj_name = proj_info["id"].split("/")[-1]
        ds_dir = root / proj_name
        if ds_dir.exists() and (ds_dir / "data.yaml").exists():
            logger.info("Already downloaded: %s", proj_name)
            continue

        logger.info("Downloading %s ...", proj_name)
        try:
            proj = ws.project(proj_name)
            proj.version(proj_info.get("versions", 1)).download("yolov5", location=str(ds_dir))
            logger.info("Downloaded: %s -> %s", proj_name, ds_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", proj_name, e)


def get_top_configs_from_phase2(db_path=None, n=10) -> List[dict]:
    """DEPRECATED: Use get_top_combined_configs() instead."""
    return get_top_combined_configs(phase2_db=db_path, n=n)


def get_top_combined_configs(phase1_db=None, phase2_db=None, n=10) -> List[dict]:
    """Combine best settings from Phase 1 (rank/epochs/lr) and Phase 2
    (dropout/aug/wd) grid searches.

    Phase 1 varied: lora_rank, num_epochs, learning_rate (alpha = rank * 2)
    Phase 2 varied: lora_dropout, augmentation_level, weight_decay, alpha_ratio
       (fixed at rank=4, epochs=50, lr=0.002)

    Strategy:
    1. Get top-K Phase 1 arch settings (rank/epochs/lr) sorted by mAP@50:95
    2. Get top-M Phase 2 reg settings (dropout/aug/wd/ar) sorted by mAP@50:95
    3. Cross-product and rank by combined normalized score
    4. Return top N diverse configs
    """
    from itertools import product as iterproduct

    p1_db = phase1_db or DB_PATH
    p2_db = phase2_db or DB_PATH_PHASE2

    # --- Phase 1: best architecture/training combos (filter n=5) ---
    phase1_configs = []
    if Path(p1_db).exists():
        conn = sqlite3.connect(str(p1_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT lora_rank, num_epochs, learning_rate, lora_alpha,
                   AVG(mAP_50_95) as avg_map, COUNT(*) as n_runs
            FROM experiments
            WHERE status = 'completed' AND num_train_images = 5
            GROUP BY lora_rank, num_epochs, learning_rate, lora_alpha
            HAVING n_runs >= 2
            ORDER BY avg_map DESC
        """).fetchall()
        conn.close()
        for r in rows:
            phase1_configs.append({
                "lora_rank": r["lora_rank"],
                "num_epochs": r["num_epochs"],
                "learning_rate": r["learning_rate"],
                "lora_alpha": r["lora_alpha"],
                "phase1_mAP": r["avg_map"],
            })
        logger.info("Phase 1 configs found: %d (top 5 shown):", len(phase1_configs))
        for i, c in enumerate(phase1_configs[:5]):
            logger.info("  P1#%d: rank=%d ep=%d lr=%s alpha=%d  mAP@50:95=%.1f%%",
                         i+1, c["lora_rank"], c["num_epochs"],
                         c["learning_rate"], c["lora_alpha"], c["phase1_mAP"]*100)
    else:
        logger.warning("Phase 1 DB not found: %s — using Phase 2 fixed params only", p1_db)

    # --- Phase 2: best regularization combos (filter n=5) ---
    phase2_configs = []
    if Path(p2_db).exists():
        conn = sqlite3.connect(str(p2_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT lora_dropout, augmentation_level, weight_decay, alpha_ratio,
                   AVG(mAP_50_95) as avg_map, COUNT(*) as n_runs
            FROM experiments
            WHERE status = 'completed' AND num_train_images = 5
            GROUP BY lora_dropout, augmentation_level, weight_decay, alpha_ratio
            HAVING n_runs >= 2
            ORDER BY avg_map DESC
        """).fetchall()
        conn.close()
        for r in rows:
            phase2_configs.append({
                "lora_dropout": r["lora_dropout"],
                "augmentation_level": r["augmentation_level"],
                "weight_decay": r["weight_decay"],
                "alpha_ratio": r["alpha_ratio"],
                "phase2_mAP": r["avg_map"],
            })
        logger.info("Phase 2 configs found: %d (top 5 shown):", len(phase2_configs))
        for i, c in enumerate(phase2_configs[:5]):
            logger.info("  P2#%d: do=%.2f aug=%d wd=%s ar=%d  mAP@50:95=%.1f%%",
                         i+1, c["lora_dropout"], c["augmentation_level"],
                         c["weight_decay"], c["alpha_ratio"], c["phase2_mAP"]*100)
    else:
        logger.error("Phase 2 DB not found: %s", p2_db)
        return []

    # --- Combine: cross-product of top Phase 1 × top Phase 2 ---
    # Take top 5 from each (25 combos), rank by normalized combined score
    top_p1 = phase1_configs[:5] if phase1_configs else [{
        "lora_rank": PHASE2_FIXED["rank"],
        "num_epochs": PHASE2_FIXED["epochs"],
        "learning_rate": PHASE2_FIXED["learning_rate"],
        "lora_alpha": PHASE2_FIXED["rank"] * 2,
        "phase1_mAP": 0.0,
    }]
    top_p2 = phase2_configs[:5]

    # Normalize scores to [0, 1] within each phase
    p1_max = max(c["phase1_mAP"] for c in top_p1) or 1.0
    p2_max = max(c["phase2_mAP"] for c in top_p2) or 1.0

    candidates = []
    for p1, p2 in iterproduct(top_p1, top_p2):
        # alpha_ratio from Phase 2 overrides Phase 1's alpha
        alpha_ratio = p2["alpha_ratio"]
        alpha = p1["lora_rank"] * alpha_ratio

        combined_score = (p1["phase1_mAP"] / p1_max + p2["phase2_mAP"] / p2_max) / 2.0
        candidates.append({
            "lora_rank": p1["lora_rank"],
            "num_epochs": p1["num_epochs"],
            "learning_rate": p1["learning_rate"],
            "lora_alpha": alpha,
            "alpha_ratio": alpha_ratio,
            "lora_dropout": p2["lora_dropout"],
            "augmentation_level": p2["augmentation_level"],
            "weight_decay": p2["weight_decay"],
            "phase1_mAP": p1["phase1_mAP"],
            "phase2_mAP": p2["phase2_mAP"],
            "combined_score": combined_score,
        })

    # Sort by combined score, take top N
    candidates.sort(key=lambda c: c["combined_score"], reverse=True)
    configs = candidates[:n]

    logger.info("Top %d combined configs:", len(configs))
    for i, c in enumerate(configs):
        logger.info("  #%d: rank=%d ep=%d lr=%s alpha=%d (ar=%d) do=%.2f aug=%d wd=%s  "
                     "score=%.3f (P1=%.1f%% P2=%.1f%%)",
                     i+1, c["lora_rank"], c["num_epochs"], c["learning_rate"],
                     c["lora_alpha"], c["alpha_ratio"],
                     c["lora_dropout"], c["augmentation_level"], c["weight_decay"],
                     c["combined_score"], c["phase1_mAP"]*100, c["phase2_mAP"]*100)

    return configs


def init_db_benchmark(db_path=None):
    """Initialize benchmark results database."""
    db = db_path or DB_PATH_BENCHMARK
    conn = sqlite3.connect(str(db), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        -- dataset info
        dataset_name TEXT,
        dataset_num_classes INTEGER,
        -- hyperparams (same as phase 2)
        lora_rank INTEGER, lora_alpha INTEGER,
        num_epochs INTEGER, learning_rate REAL,
        num_train_images INTEGER,
        train_image_set TEXT,
        lora_dropout REAL,
        augmentation_level INTEGER,
        weight_decay REAL,
        alpha_ratio INTEGER,
        -- architecture param for ablation
        group_detr INTEGER DEFAULT 1,
        -- results
        train_time_seconds REAL, time_per_epoch_ms REAL,
        final_loss REAL, loss_history TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        status TEXT DEFAULT 'pending',
        error_message TEXT, device TEXT, notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT, eval_split TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        per_class_ap_json TEXT,
        conf_metrics_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS train_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        image_stem TEXT, split TEXT,
        num_boxes INTEGER, num_classes INTEGER,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    conn.commit()
    return conn


def make_run_id_benchmark(dataset_name, rank, epochs, lr, dropout, aug_level, wd, alpha_ratio, group_detr=1):
    """Run ID for benchmark experiments. Includes all hyperparams for uniqueness."""
    return f"bm_{dataset_name}_r{rank}_e{epochs}_lr{lr}_do{dropout}_aug{aug_level}_wd{wd}_ar{alpha_ratio}_g{group_detr}"


def run_single_experiment_benchmark(
    db_path, base_model, config, dataset_info, dropout, aug_level, wd, alpha_ratio,
    device, group_detr=1,
    rank=None, epochs=None, lr=None,
    template_model=None, filtered_state=None, fresh_config=None,
    cached_criterion=None, cached_weight_dict=None,
):
    """Run a single benchmark experiment on one dataset with one config."""
    ds_name = dataset_info["name"]
    ds_path = dataset_info["path"]
    ds_class_names = dataset_info["class_names"]
    train_stems = dataset_info["train_stems"]
    test_stems = dataset_info["test_stems"]
    num_train_images = len(train_stems)

    rank = rank or PHASE2_FIXED["rank"]
    epochs = epochs or PHASE2_FIXED["epochs"]
    lr = lr or PHASE2_FIXED["learning_rate"]
    alpha = rank * alpha_ratio

    run_id = make_run_id_benchmark(ds_name, rank, epochs, lr, dropout, aug_level, wd, alpha_ratio, group_detr)

    conn = _db_connect(db_path)

    try:
        if experiment_done(conn, run_id):
            logger.info("⏭  Skip %s (done)", run_id)
            return

        # Delete any previous failed/partial attempt
        cur = _db_execute_with_retry(conn, "SELECT id FROM experiments WHERE run_id = ?", (run_id,))
        old = cur.fetchone()
        if old:
            _db_execute_with_retry(conn, "DELETE FROM eval_results WHERE experiment_id = ?", (old[0],))
            _db_execute_with_retry(conn, "DELETE FROM train_images WHERE experiment_id = ?", (old[0],))
            _db_execute_with_retry(conn, "DELETE FROM experiments WHERE id = ?", (old[0],))
            _db_commit_with_retry(conn)

        logger.info("🚀 %s  ds=%s do=%.2f aug=%d wd=%s ar=%d g=%d",
                     run_id, ds_name, dropout, aug_level, wd, alpha_ratio, group_detr)

        cur = _db_execute_with_retry(conn,
            """INSERT INTO experiments
               (run_id, timestamp, dataset_name, dataset_num_classes,
                lora_rank, lora_alpha, num_epochs, learning_rate,
                num_train_images, train_image_set,
                lora_dropout, augmentation_level, weight_decay, alpha_ratio,
                group_detr, status, device)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (run_id, datetime.now().isoformat(), ds_name, dataset_info["num_classes"],
             rank, alpha, epochs, lr,
             num_train_images, ds_name,
             dropout, aug_level, wd, alpha_ratio,
             group_detr, str(device)),
        )
        exp_id = cur.lastrowid
        _db_commit_with_retry(conn)

        # Load function for this dataset
        def _load(split, stem):
            return load_image_and_labels_generic(ds_path, split, stem)

        # Load ALL training images
        images_and_boxes = []
        all_class_ids = set()

        for stem in train_stems:
            img, boxes = _load("train", stem)
            images_and_boxes.append((img, boxes))
            for b in boxes:
                all_class_ids.add(b["class_id"])

        active_classes = [ds_class_names[i] for i in sorted(all_class_ids)]
        num_classes = len(active_classes)

        dataset = InlineFewShotDataset(
            images_and_boxes, active_classes, config.resolution,
            augment=True, augmentation_level=aug_level,
            class_names=ds_class_names,
        )

        # Train
        t0 = time.time()
        if template_model is not None and cached_criterion is not None:
            merged_model, loss_history = train_lora_fast(
                template_model, filtered_state, fresh_config,
                dataset, num_classes, device,
                cached_criterion, cached_weight_dict,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=dropout, weight_decay=wd,
                class_names=ds_class_names,
            )
        else:
            merged_model, loss_history = train_lora(
                base_model, config, dataset, num_classes, device,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=dropout, weight_decay=wd,
            )
        train_time = time.time() - t0
        ms_per_epoch = (train_time / epochs) * 1000

        _db_execute_with_retry(conn,
            """UPDATE experiments SET train_time_seconds=?, time_per_epoch_ms=?,
               final_loss=?, loss_history=? WHERE id=?""",
            (train_time, ms_per_epoch, loss_history[-1],
             json.dumps(loss_history), exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("✅ Trained in %.1fs (%.0f ms/ep)", train_time, ms_per_epoch)

        # Evaluate on TEST split (for leaderboard comparison)
        active_to_original = {i: ds_class_names.index(c) for i, c in enumerate(active_classes)}
        all_mAP_50 = []
        all_mAP_50_95 = []

        for eval_stem in test_stems:
            eval_img, eval_boxes = _load("test", eval_stem)
            raw_dets = run_inference(merged_model, config, eval_img, confidence_threshold=0.01)
            remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                        for d in raw_dets if d["class_id"] in active_to_original]

            metrics = compute_map(
                remapped, eval_boxes, eval_img.size,
                all_class_ids, MAP_IOU_THRESHOLDS,
                class_names=ds_class_names,
            )

            all_mAP_50.append(metrics["mAP_50"])
            all_mAP_50_95.append(metrics["mAP_50_95"])

            _db_execute_with_retry(conn,
                """INSERT INTO eval_results
                   (experiment_id, eval_image_stem, eval_split,
                    mAP_50, mAP_50_95, per_class_ap_json, conf_metrics_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (exp_id, eval_stem, "test",
                 metrics["mAP_50"], metrics["mAP_50_95"],
                 json.dumps(metrics["per_class_ap"]),
                 json.dumps(metrics["conf_metrics"])),
            )

        avg_mAP_50 = float(np.mean(all_mAP_50)) if all_mAP_50 else 0
        avg_mAP_50_95 = float(np.mean(all_mAP_50_95)) if all_mAP_50_95 else 0

        _db_execute_with_retry(conn,
            "UPDATE experiments SET mAP_50=?, mAP_50_95=?, status='completed' WHERE id=?",
            (avg_mAP_50, avg_mAP_50_95, exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("📊 %s  mAP@50=%.1f%%  mAP@50:95=%.1f%%",
                     run_id, avg_mAP_50*100, avg_mAP_50_95*100)

        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error("❌ %s failed: %s", run_id, e, exc_info=True)
        try:
            _db_execute_with_retry(conn,
                "UPDATE experiments SET status='failed', error_message=? WHERE id=?",
                (str(e), exp_id))
            _db_commit_with_retry(conn)
        except Exception:
            pass
    finally:
        conn.close()


def main_benchmark(args, device):
    """Run RF20-VL-FSOD benchmark: top Phase 2 configs across 20 datasets."""
    db_path = args.benchmark_db or DB_PATH_BENCHMARK
    datasets_root = Path(args.datasets_root) if args.datasets_root else BENCHMARK_DATASETS_ROOT
    top_n = args.top_n
    phase2_db = args.phase2_db or DB_PATH_PHASE2

    # Download datasets if requested
    if args.download_datasets:
        download_benchmark_datasets(datasets_root)
        if not args.benchmark:
            return  # Just download, don't run

    # Discover datasets
    datasets = discover_benchmark_datasets(datasets_root)
    if not datasets:
        logger.error("No datasets found in %s. Use --download-datasets first.", datasets_root)
        return

    # Get top combined configs from Phase 1 + Phase 2
    phase1_db = args.phase1_db if hasattr(args, 'phase1_db') and args.phase1_db else DB_PATH
    top_configs = get_top_combined_configs(phase1_db=phase1_db, phase2_db=phase2_db, n=top_n)
    if not top_configs:
        logger.error("No completed configs found. Check Phase 1 DB (%s) and Phase 2 DB (%s)", phase1_db, phase2_db)
        return

    # Init benchmark DB
    conn = init_db_benchmark(db_path)
    total_base = len(top_configs) * len(datasets)
    group_detr_values = [1, 5, 13] if args.ablate_group_detr else [1]
    total_ablation = len(top_configs) * len(datasets) * len(group_detr_values)
    total = total_ablation if args.ablate_group_detr else total_base

    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('total_experiments', ?)",
                 (str(total),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('phase', ?)", ("benchmark",))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('config_selection', ?)",
                 (json.dumps({"method": "combined_phase1_phase2", "top_n": top_n,
                              "configs": top_configs}),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('start_time', ?)",
                 (datetime.now().isoformat(),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('num_datasets', ?)",
                 (str(len(datasets)),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('dataset_names', ?)",
                 (json.dumps([d["name"] for d in datasets]),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("📋 Benchmark DB: %s", db_path)
    logger.info("📊 %d datasets × %d configs × %d group_detr values = %d experiments",
                len(datasets), len(top_configs), len(group_detr_values), total)

    # Start GPU monitor
    gpu_stop = None
    if device.type == "cuda":
        gpu_stop = threading.Event()
        gpu_thread = threading.Thread(target=_gpu_monitor_loop,
                                       args=(db_path, 5, gpu_stop), daemon=True)
        gpu_thread.start()

    # Load base model once
    logger.info("🔄 Loading base model...")
    base_model, config = load_base_model(device)
    logger.info("✅ Base model loaded")

    done = 0
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    for ds_info in datasets:
        ds_name = ds_info["name"]
        num_classes = ds_info["num_classes"]
        logger.info("━━━ Dataset: %s (%d classes, %d train, %d test) ━━━",
                     ds_name, num_classes, len(ds_info["train_stems"]), len(ds_info["test_stems"]))

        for gd in group_detr_values:
            # Template cache key: (num_classes, group_detr) — rank doesn't affect
            # the template since LoRA is applied dynamically in train_lora_fast
            logger.info("  Building template (num_classes=%d, group_detr=%d)...", num_classes, gd)
            fresh_config = config.model_copy(update={
                "num_classes": num_classes, "group_detr": gd,
            })
            template = build_model(config=fresh_config)
            base_state = base_model.state_dict()
            template_state = template.state_dict()
            filtered = {k: v for k, v in base_state.items()
                        if k in template_state and v.shape == template_state[k].shape}
            template.load_state_dict(filtered, strict=False)
            template.reinitialize_detection_head(num_classes + 1)
            template = template.to(device)

            criterion, weight_dict = prepare_criterion(fresh_config, num_classes, device)

            for cfg_idx, cfg in enumerate(top_configs):
                done += 1
                cfg_rank = cfg["lora_rank"]
                cfg_epochs = cfg["num_epochs"]
                cfg_lr = cfg["learning_rate"]
                logger.info("━━━ [%d/%d] %s config #%d (r=%d e=%d lr=%s g=%d) ━━━",
                             done, total, ds_name, cfg_idx+1,
                             cfg_rank, cfg_epochs, cfg_lr, gd)

                run_single_experiment_benchmark(
                    db_path, base_model, config, ds_info,
                    cfg["lora_dropout"], cfg["augmentation_level"],
                    cfg["weight_decay"], cfg["alpha_ratio"],
                    device, group_detr=gd,
                    rank=cfg_rank, epochs=cfg_epochs, lr=cfg_lr,
                    template_model=template,
                    filtered_state=filtered,
                    fresh_config=fresh_config,
                    cached_criterion=criterion,
                    cached_weight_dict=weight_dict,
                )

            del template, criterion
            torch.cuda.empty_cache()

    # Stop GPU monitor
    if gpu_stop is not None:
        gpu_stop.set()

    # Record end time
    conn = _db_connect(db_path)
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('end_time', ?)",
                 (datetime.now().isoformat(),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("🎉 Benchmark complete! Results: %s", db_path)


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: augmentation + rank + batch + LoRA targets benchmark
# ═══════════════════════════════════════════════════════════════════════

def init_db_phase3(db_path=None):
    """Initialize Phase 3 results database."""
    db = db_path or DB_PATH_PHASE3
    conn = sqlite3.connect(str(db), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        -- dataset info
        dataset_name TEXT,
        dataset_num_classes INTEGER,
        num_train_images INTEGER,
        -- hyperparams
        lora_rank INTEGER, lora_alpha INTEGER,
        num_epochs INTEGER, learning_rate REAL,
        alpha_ratio INTEGER,
        batch_size INTEGER,
        lora_targets TEXT,
        copy_paste INTEGER,
        mosaic INTEGER,
        warmup INTEGER,
        multi_scale INTEGER,
        -- carried from best Phase 2 settings
        lora_dropout REAL DEFAULT 0.0,
        augmentation_level INTEGER DEFAULT 1,
        weight_decay REAL DEFAULT 0.0001,
        -- results
        train_time_seconds REAL, time_per_epoch_ms REAL,
        final_loss REAL, loss_history TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        status TEXT DEFAULT 'pending',
        error_message TEXT, device TEXT, notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT, eval_split TEXT,
        mAP_50 REAL, mAP_50_95 REAL,
        per_class_ap_json TEXT,
        conf_metrics_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    conn.commit()
    return conn


def make_run_id_phase3(dataset_name, rank, batch_size, copy_paste, mosaic,
                       lora_targets, warmup, multi_scale,
                       alpha_ratio=None, weight_decay=None):
    """Run ID for Phase 3 experiments."""
    cp = "cp1" if copy_paste else "cp0"
    mo = "mo1" if mosaic else "mo0"
    wu = "wu1" if warmup else "wu0"
    ms = "ms1" if multi_scale else "ms0"
    base = f"p3_{dataset_name}_r{rank}_bs{batch_size}_{cp}_{mo}_{lora_targets}_{wu}_{ms}"
    if alpha_ratio is not None:
        base += f"_ar{alpha_ratio}"
    if weight_decay is not None:
        base += f"_wd{weight_decay}"
    return base


def _generate_live_viz(
    merged_model, config, ds_name, ds_path, ds_class_names,
    test_stems, all_class_ids, active_to_original,
    avg_mAP_50, avg_mAP_50_95, train_time,
    num_train, num_test, num_classes,
    rank, batch_size, copy_paste, mosaic, alpha_ratio, weight_decay,
    db_path, device,
    conf_thresh=0.25,
):
    """Generate prediction/GT/hybrid images for a new best model on a dataset.

    Called inline during benchmark runs when a new best mAP@50:95 is found.
    Updates viz_champion/manifest.json atomically.
    """
    from PIL import Image, ImageDraw, ImageFont
    import tempfile

    viz_dir = Path(db_path).parent / "viz_champion"
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "images").mkdir(exist_ok=True)

    # Pick test image with most GT boxes (sample up to 10)
    best_stem, best_n_gt = test_stems[0], 0
    for stem in test_stems[:10]:
        _, gt = load_image_and_labels_generic(ds_path, "test", stem)
        if len(gt) > best_n_gt:
            best_n_gt = len(gt)
            best_stem = stem

    test_img, gt_boxes = load_image_and_labels_generic(ds_path, "test", best_stem)
    raw_dets = run_inference(merged_model, config, test_img, confidence_threshold=0.01)
    remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                for d in raw_dets if d["class_id"] in active_to_original]

    metrics = compute_map(
        remapped, gt_boxes, test_img.size,
        all_class_ids, MAP_IOU_THRESHOLDS,
        class_names=ds_class_names,
    )

    # Adaptive sizes
    img_w, img_h = test_img.size
    diag = (img_w**2 + img_h**2) ** 0.5
    scale = max(diag / 640.0, 1.0)
    line_w = max(int(3 * scale), 2)
    thin_w = max(int(1 * scale), 1)
    font_sz_pred = max(int(14 * scale), 10)
    font_sz_gt = max(int(12 * scale), 9)

    def _load_font(size):
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(p, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    PALETTE = [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
        "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
    ]

    # Draw predictions
    pred_img = test_img.copy()
    draw = ImageDraw.Draw(pred_img)
    font = _load_font(font_sz_pred)
    for det in sorted(remapped, key=lambda d: d["confidence"]):
        if det["confidence"] < conf_thresh:
            continue
        x1, y1, x2, y2 = det["box"]
        cid = det["class_id"]
        color = PALETTE[cid % len(PALETTE)]
        label = ds_class_names[cid] if cid < len(ds_class_names) else f"cls{cid}"
        txt = f"{label} {det['confidence']:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        bbox = draw.textbbox((x1, y1), txt, font=font)
        draw.rectangle([bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1], fill=color)
        draw.text((x1, y1), txt, fill="white", font=font)
    pred_img.save(str(viz_dir / "images" / f"{ds_name}_pred.jpg"), quality=85)

    # Save original (un-annotated) test image
    test_img.save(str(viz_dir / "images" / f"{ds_name}_original.jpg"), quality=90)

    # Draw ground truth
    gt_img = test_img.copy()
    draw = ImageDraw.Draw(gt_img)
    font_gt = _load_font(font_sz_gt)
    for box in gt_boxes:
        cid = box["class_id"]
        color = PALETTE[cid % len(PALETTE)]
        label = ds_class_names[cid] if cid < len(ds_class_names) else f"cls{cid}"
        cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thin_w)
        draw.text((x1, max(y1-font_sz_gt-2, 0)), f"GT: {label}", fill=color, font=font_gt)
    gt_img.save(str(viz_dir / "images" / f"{ds_name}_gt.jpg"), quality=85)

    # Draw hybrid (FP=green, FN=red, TP=clear, TN=dimmed)
    try:
        import cv2
        import numpy as np

        scene = np.array(test_img)[:, :, ::-1].copy()  # RGB→BGR

        # Build pred xyxy for confident detections
        pred_boxes = []
        for det in remapped:
            if det["confidence"] >= conf_thresh:
                pred_boxes.append(det["box"])
        pred_xyxy = np.array(pred_boxes, dtype=np.float32).reshape(-1, 4) if pred_boxes else np.empty((0, 4), dtype=np.float32)

        # Build GT xyxy
        gt_xyxy_list = []
        for box in gt_boxes:
            cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
            gt_xyxy_list.append([
                (cx - bw/2) * img_w, (cy - bh/2) * img_h,
                (cx + bw/2) * img_w, (cy + bh/2) * img_h,
            ])
        gt_xyxy = np.array(gt_xyxy_list, dtype=np.float32).reshape(-1, 4) if gt_xyxy_list else np.empty((0, 4), dtype=np.float32)

        from supervision import Color, Detections
        from inference.core.workflows.core_steps.visualizations.common.annotators.model_comparison import ModelComparisonAnnotator

        dets_pred = Detections(xyxy=pred_xyxy)
        dets_gt = Detections(xyxy=gt_xyxy)

        annotator = ModelComparisonAnnotator(
            color_a=Color.GREEN,   # FP: predicted but not GT
            color_b=Color.RED,     # FN: GT but not predicted
            background_color=Color.BLACK,
            opacity=0.7,
            force_box=True,
        )
        hybrid_scene = annotator.annotate(scene, dets_pred, dets_gt)
        hybrid_rgb = hybrid_scene[:, :, ::-1]  # BGR→RGB
        hybrid_pil = Image.fromarray(hybrid_rgb)
        hybrid_pil.save(str(viz_dir / "images" / f"{ds_name}_hybrid.jpg"), quality=85)
    except Exception as hybrid_err:
        logger.warning("⚠️ Hybrid image failed for %s: %s", ds_name, hybrid_err)

    # Update manifest atomically
    manifest_path = viz_dir / "manifest.json"
    manifest = {"metric": "mAP_50_95", "models": []}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Build config key for this model
    config_key = f"r{rank}_bs{batch_size}_cp{int(copy_paste)}_mo{int(mosaic)}_ar{alpha_ratio}_wd{weight_decay}"

    # Find or create model entry
    model_entry = None
    for m in manifest.get("models", []):
        if m.get("config_key") == config_key:
            model_entry = m
            break
    if model_entry is None:
        model_entry = {
            "config_key": config_key,
            "model_index": len(manifest.get("models", [])),
            "config": {
                "rank": rank, "batch_size": batch_size,
                "copy_paste": bool(copy_paste), "mosaic": bool(mosaic),
                "alpha_ratio": alpha_ratio, "weight_decay": weight_decay,
            },
            "datasets": [],
        }
        manifest.setdefault("models", []).append(model_entry)

    # Update or insert dataset entry
    ds_entry = {
        "name": ds_name,
        "num_classes": num_classes,
        "num_train_images": num_train,
        "num_test_images": num_test,
        "class_names": ds_class_names,
        "mAP_50": avg_mAP_50,
        "mAP_50_95": avg_mAP_50_95,
        "train_time_seconds": train_time,
        "is_winner": True,
        "pred_image": f"images/{ds_name}_pred.jpg",
        "gt_image": f"images/{ds_name}_gt.jpg",
        "hybrid_image": f"images/{ds_name}_hybrid.jpg",
        "original_image": f"images/{ds_name}_original.jpg",
        "test_image_stem": best_stem,
        "sample_mAP_50": metrics["mAP_50"],
        "sample_mAP_50_95": metrics["mAP_50_95"],
        "num_gt_boxes": len(gt_boxes),
        "num_predictions": len([d for d in remapped if d["confidence"] >= conf_thresh]),
        "image_size": [img_w, img_h],
        "raw_detections": [
            {"class_id": d["class_id"], "confidence": round(d["confidence"], 4),
             "bbox": [round(d["cx"], 6), round(d["cy"], 6), round(d["w"], 6), round(d["h"], 6)]}
            for d in [{"class_id": det["class_id"], "confidence": det["confidence"],
                       "cx": (det["box"][0]+det["box"][2])/(2*img_w),
                       "cy": (det["box"][1]+det["box"][3])/(2*img_h),
                       "w": (det["box"][2]-det["box"][0])/img_w,
                       "h": (det["box"][3]-det["box"][1])/img_h}
                      for det in remapped]
        ],
        "gt_boxes_data": [
            {"class_id": b["class_id"],
             "bbox": [round(b["cx"], 6), round(b["cy"], 6), round(b["w"], 6), round(b["h"], 6)]}
            for b in gt_boxes
        ],
    }

    # Replace existing dataset entry or append
    existing_idx = None
    for i, d in enumerate(model_entry["datasets"]):
        if d["name"] == ds_name:
            existing_idx = i
            break
    if existing_idx is not None:
        model_entry["datasets"][existing_idx] = ds_entry
    else:
        model_entry["datasets"].append(ds_entry)

    # Recompute summary stats
    for m in manifest["models"]:
        ds_list = m["datasets"]
        if ds_list:
            m["wins"] = sum(1 for d in ds_list if d.get("is_winner"))
            m["avg_mAP_50"] = sum(d["mAP_50"] for d in ds_list) / len(ds_list)
            m["avg_mAP_50_95"] = sum(d["mAP_50_95"] for d in ds_list) / len(ds_list)
            m["avg_train_time"] = sum(d.get("train_time_seconds", 0) for d in ds_list) / len(ds_list)

    # Atomic write
    tmp_path = str(manifest_path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, str(manifest_path))
    logger.info("🖼️ Live viz updated for %s → %s", ds_name, manifest_path)


def run_single_experiment_phase3(
    db_path, base_model, config, dataset_info, device,
    rank, batch_size, copy_paste, mosaic, lora_targets, warmup, multi_scale,
    epochs=50, lr=2e-3, alpha_ratio=2,
    lora_dropout=0.0, aug_level=1, weight_decay=1e-4,
    template_model=None, filtered_state=None, fresh_config=None,
    cached_criterion=None, cached_weight_dict=None,
):
    """Run a single Phase 3 experiment on one dataset with one config."""
    ds_name = dataset_info["name"]
    ds_path = dataset_info["path"]
    ds_class_names = dataset_info["class_names"]
    train_stems = dataset_info["train_stems"]
    test_stems = dataset_info["test_stems"]
    num_train_images = len(train_stems)
    alpha = rank * alpha_ratio

    run_id = make_run_id_phase3(ds_name, rank, batch_size, copy_paste, mosaic,
                                lora_targets, warmup, multi_scale,
                                alpha_ratio=alpha_ratio, weight_decay=weight_decay)

    conn = _db_connect(db_path)

    try:
        if experiment_done(conn, run_id):
            logger.info("⏭  Skip %s (done)", run_id)
            return

        # Delete any previous failed/partial attempt
        cur = _db_execute_with_retry(conn, "SELECT id FROM experiments WHERE run_id = ?", (run_id,))
        old = cur.fetchone()
        if old:
            _db_execute_with_retry(conn, "DELETE FROM eval_results WHERE experiment_id = ?", (old[0],))
            _db_execute_with_retry(conn, "DELETE FROM experiments WHERE id = ?", (old[0],))
            _db_commit_with_retry(conn)

        logger.info("🚀 %s  r=%d bs=%d cp=%s mo=%s lt=%s wu=%s ms=%s",
                     run_id, rank, batch_size, copy_paste, mosaic, lora_targets, warmup, multi_scale)

        cur = _db_execute_with_retry(conn,
            """INSERT INTO experiments
               (run_id, timestamp, dataset_name, dataset_num_classes, num_train_images,
                lora_rank, lora_alpha, num_epochs, learning_rate, alpha_ratio,
                batch_size, lora_targets, copy_paste, mosaic, warmup, multi_scale,
                lora_dropout, augmentation_level, weight_decay,
                status, device)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (run_id, datetime.now().isoformat(), ds_name, dataset_info["num_classes"],
             num_train_images,
             rank, alpha, epochs, lr, alpha_ratio,
             batch_size, lora_targets, int(copy_paste), int(mosaic),
             int(warmup), int(multi_scale),
             lora_dropout, aug_level, weight_decay,
             str(device)),
        )
        exp_id = cur.lastrowid
        _db_commit_with_retry(conn)

        # Load ALL training images
        def _load(split, stem):
            return load_image_and_labels_generic(ds_path, split, stem)

        images_and_boxes = []
        all_class_ids = set()
        for stem in train_stems:
            img, boxes = _load("train", stem)
            images_and_boxes.append((img, boxes))
            for b in boxes:
                all_class_ids.add(b["class_id"])

        active_classes = [ds_class_names[i] for i in sorted(all_class_ids)]
        num_classes = len(active_classes)

        dataset = InlineFewShotDataset(
            images_and_boxes, active_classes, config.resolution,
            augment=True, augmentation_level=aug_level,
            class_names=ds_class_names,
        )

        # Train
        t0 = time.time()
        if template_model is not None and cached_criterion is not None:
            merged_model, loss_history, adapter_state = train_lora_fast(
                template_model, filtered_state, fresh_config,
                dataset, num_classes, device,
                cached_criterion, cached_weight_dict,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=lora_dropout, weight_decay=weight_decay,
                class_names=ds_class_names,
                batch_size=batch_size, lora_targets_version=lora_targets,
                copy_paste=copy_paste, mosaic=mosaic,
                warmup=warmup, multi_scale=multi_scale,
            )
        else:
            # Fallback to slow path (shouldn't happen in Phase 3)
            merged_model, loss_history, adapter_state = train_lora(
                base_model, config, dataset, num_classes, device,
                rank=rank, alpha=alpha, lr=lr, num_epochs=epochs,
                lora_dropout=lora_dropout, weight_decay=weight_decay,
            )
        train_time = time.time() - t0
        ms_per_epoch = (train_time / epochs) * 1000

        _db_execute_with_retry(conn,
            """UPDATE experiments SET train_time_seconds=?, time_per_epoch_ms=?,
               final_loss=?, loss_history=? WHERE id=?""",
            (train_time, ms_per_epoch, loss_history[-1],
             json.dumps(loss_history), exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("✅ Trained in %.1fs (%.0f ms/ep)", train_time, ms_per_epoch)

        # Persist LoRA adapter state to disk
        if adapter_state is not None:
            lora_dir = Path(db_path).parent / "lora_weights" / run_id
            lora_dir.mkdir(parents=True, exist_ok=True)
            torch.save(adapter_state, lora_dir / "adapter_state.pt")
            logger.info("💾 Adapter state saved to %s", lora_dir)

        # Evaluate on TEST split
        active_to_original = {i: ds_class_names.index(c) for i, c in enumerate(active_classes)}
        all_mAP_50 = []
        all_mAP_50_95 = []

        for eval_stem in test_stems:
            eval_img, eval_boxes = _load("test", eval_stem)
            raw_dets = run_inference(merged_model, config, eval_img, confidence_threshold=0.01)
            remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                        for d in raw_dets if d["class_id"] in active_to_original]

            metrics = compute_map(
                remapped, eval_boxes, eval_img.size,
                all_class_ids, MAP_IOU_THRESHOLDS,
                class_names=ds_class_names,
            )

            all_mAP_50.append(metrics["mAP_50"])
            all_mAP_50_95.append(metrics["mAP_50_95"])

            _db_execute_with_retry(conn,
                """INSERT INTO eval_results
                   (experiment_id, eval_image_stem, eval_split,
                    mAP_50, mAP_50_95, per_class_ap_json, conf_metrics_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (exp_id, eval_stem, "test",
                 metrics["mAP_50"], metrics["mAP_50_95"],
                 json.dumps(metrics["per_class_ap"]),
                 json.dumps(metrics["conf_metrics"])),
            )

        avg_mAP_50 = float(np.mean(all_mAP_50)) if all_mAP_50 else 0
        avg_mAP_50_95 = float(np.mean(all_mAP_50_95)) if all_mAP_50_95 else 0

        _db_execute_with_retry(conn,
            "UPDATE experiments SET mAP_50=?, mAP_50_95=?, status='completed' WHERE id=?",
            (avg_mAP_50, avg_mAP_50_95, exp_id),
        )
        _db_commit_with_retry(conn)
        logger.info("📊 %s  mAP@50=%.1f%%  mAP@50:95=%.1f%%",
                     run_id, avg_mAP_50*100, avg_mAP_50_95*100)

        # Live viz: check if this is new best for dataset, generate images if so
        try:
            prev_best = conn.execute(
                """SELECT MAX(mAP_50_95) FROM experiments
                   WHERE dataset_name=? AND status='completed' AND id != ?""",
                (ds_name, exp_id),
            ).fetchone()[0]
            if prev_best is None or avg_mAP_50_95 >= prev_best:
                _generate_live_viz(
                    merged_model, config, ds_name, ds_path, ds_class_names,
                    test_stems, all_class_ids, active_to_original,
                    avg_mAP_50, avg_mAP_50_95, train_time,
                    len(train_stems), len(test_stems), num_classes,
                    rank, batch_size, copy_paste, mosaic,
                    int(alpha / rank) if rank else 1, weight_decay,
                    db_path, device,
                )
        except Exception as viz_err:
            logger.warning("⚠️ Live viz generation failed: %s", viz_err)

        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error("❌ %s failed: %s", run_id, e, exc_info=True)
        try:
            _db_execute_with_retry(conn,
                "UPDATE experiments SET status='failed', error_message=? WHERE run_id=?",
                (str(e), run_id))
            _db_commit_with_retry(conn)
        except Exception:
            pass
    finally:
        conn.close()


def main_phase3(args, device):
    """Run Phase 3 benchmark: augmentation + rank + batch sweep across 20 datasets."""
    db_path = args.phase3_db or DB_PATH_PHASE3
    datasets_root = Path(args.datasets_root) if args.datasets_root else BENCHMARK_DATASETS_ROOT

    # Discover datasets
    datasets = discover_benchmark_datasets(datasets_root)
    if not datasets:
        logger.error("No datasets found in %s. Use --download-datasets first.", datasets_root)
        return

    # Select grid
    if args.smoke:
        grid = SMOKE_GRID_PHASE3
        fixed = PHASE3A_FIXED
    elif getattr(args, 'phase3b', False):
        grid = GRID_PHASE3B
        fixed = PHASE3B_FIXED
    else:
        grid = GRID_PHASE3A
        fixed = PHASE3A_FIXED

    phase_name = "phase3b" if getattr(args, 'phase3b', False) else "phase3a"

    # Generate all configs
    from itertools import product as iterproduct
    keys = sorted(grid.keys())
    all_combos = list(iterproduct(*[grid[k] for k in keys]))
    configs = [dict(zip(keys, combo)) for combo in all_combos]

    # Init DB
    conn = init_db_phase3(db_path)
    total = len(configs) * len(datasets)

    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('total_experiments', ?)",
                 (str(total),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('phase', ?)", (phase_name,))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('start_time', ?)",
                 (datetime.now().isoformat(),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('num_datasets', ?)",
                 (str(len(datasets)),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('dataset_names', ?)",
                 (json.dumps([d["name"] for d in datasets]),))
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('grid_config', ?)",
                 (json.dumps({"grid": grid, "fixed": fixed}),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("📋 Phase 3 DB: %s", db_path)
    logger.info("📊 %d datasets × %d configs = %d experiments", len(datasets), len(configs), total)

    # Start GPU monitor
    gpu_stop = None
    if device.type == "cuda":
        gpu_stop = threading.Event()
        gpu_thread = threading.Thread(target=_gpu_monitor_loop,
                                       args=(db_path, 5, gpu_stop), daemon=True)
        gpu_thread.start()

    # Load base model once
    logger.info("🔄 Loading base model...")
    base_model, config = load_base_model(device)
    logger.info("✅ Base model loaded")

    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    done = 0
    for ds_info in datasets:
        ds_name = ds_info["name"]
        num_classes = ds_info["num_classes"]
        logger.info("━━━ Dataset: %s (%d classes, %d train, %d test) ━━━",
                     ds_name, num_classes, len(ds_info["train_stems"]), len(ds_info["test_stems"]))

        # Build template model for this dataset (group_detr=1)
        fresh_config = config.model_copy(update={
            "num_classes": num_classes, "group_detr": 1,
        })
        template = build_model(config=fresh_config)
        base_state = base_model.state_dict()
        template_state = template.state_dict()
        filtered = {k: v for k, v in base_state.items()
                    if k in template_state and v.shape == template_state[k].shape}
        template.load_state_dict(filtered, strict=False)
        template.reinitialize_detection_head(num_classes + 1)
        template = template.to(device)

        criterion, weight_dict = prepare_criterion(fresh_config, num_classes, device)

        for cfg_idx, cfg in enumerate(configs):
            done += 1
            cfg_rank = cfg.get("rank", fixed.get("rank", 4))
            cfg_bs = cfg.get("batch_size", fixed.get("batch_size", 4))
            cfg_cp = cfg.get("copy_paste", fixed.get("copy_paste", False))
            cfg_mo = cfg.get("mosaic", fixed.get("mosaic", False))
            cfg_lt = cfg.get("lora_targets", fixed.get("lora_targets", "v1"))
            cfg_wu = cfg.get("warmup", fixed.get("warmup", False))
            cfg_ms = cfg.get("multi_scale", fixed.get("multi_scale", False))
            cfg_ar = cfg.get("alpha_ratio", fixed.get("alpha_ratio", 2))
            cfg_wd = cfg.get("weight_decay", fixed.get("weight_decay", 1e-4))

            logger.info("━━━ [%d/%d] %s config #%d (r=%d bs=%d cp=%s mo=%s lt=%s wu=%s ms=%s ar=%s wd=%s) ━━━",
                         done, total, ds_name, cfg_idx + 1,
                         cfg_rank, cfg_bs, cfg_cp, cfg_mo, cfg_lt, cfg_wu, cfg_ms, cfg_ar, cfg_wd)

            run_single_experiment_phase3(
                db_path, base_model, config, ds_info, device,
                rank=cfg_rank, batch_size=cfg_bs,
                copy_paste=cfg_cp, mosaic=cfg_mo,
                lora_targets=cfg_lt, warmup=cfg_wu, multi_scale=cfg_ms,
                epochs=fixed.get("epochs", 50),
                lr=fixed.get("learning_rate", 2e-3),
                alpha_ratio=cfg_ar,
                weight_decay=cfg_wd,
                template_model=template,
                filtered_state=filtered,
                fresh_config=fresh_config,
                cached_criterion=criterion,
                cached_weight_dict=weight_dict,
            )

        del template, criterion
        torch.cuda.empty_cache()

    # Stop GPU monitor
    if gpu_stop is not None:
        gpu_stop.set()

    # Record end time
    conn = _db_connect(db_path)
    _db_execute_with_retry(conn, "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('end_time', ?)",
                 (datetime.now().isoformat(),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("🎉 Phase 3 complete! Results: %s", db_path)


if __name__ == "__main__":
    main()

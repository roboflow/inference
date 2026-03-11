"""Few-shot dataset that converts training images + bounding box annotations
into DETR-compatible (image_tensor, targets) pairs."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from inference.core.utils.image_utils import load_image_bgr


class FewShotDataset(Dataset):
    """Wraps a list of training images (with bounding boxes) for LoRA fine-tuning.

    Each item is ``(image_tensor, targets_dict)`` where targets_dict has:
      - ``labels``: ``LongTensor[N]`` of class indices
      - ``boxes``:  ``FloatTensor[N, 4]`` in normalised cxcywh format

    Args:
        training_data: List of dicts, each with ``"image"`` (encodable image ref)
            and ``"boxes"`` (list of dicts with x, y, w, h, cls keys —
            pixel-space centre-x, centre-y, width, height).
        class_names: Ordered list of class names (index = class_id).
        resolution: Square input resolution the model expects.
        augment: Whether to apply random horizontal flips.
    """

    def __init__(
        self,
        training_data: list,
        class_names: List[str],
        resolution: int,
        augment: bool = True,
    ):
        self.training_data = training_data
        self.class_names = class_names
        self.class_name_to_id = {name: i for i, name in enumerate(class_names)}
        self.resolution = resolution
        self.augment = augment

    def __len__(self) -> int:
        return len(self.training_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item = self.training_data[idx]
        image_bgr = load_image_bgr(item["image"])
        h_orig, w_orig = image_bgr.shape[:2]

        # Convert BGR uint8 HWC -> RGB float CHW, resize, normalise
        image_rgb = image_bgr[:, :, ::-1].copy()
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Resize to model resolution
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Build normalised cxcywh boxes
        labels = []
        boxes = []
        for box in item["boxes"]:
            cls_name = box["cls"]
            if cls_name not in self.class_name_to_id:
                continue
            labels.append(self.class_name_to_id[cls_name])
            cx = box["x"] / w_orig
            cy = box["y"] / h_orig
            bw = box["w"] / w_orig
            bh = box["h"] / h_orig
            boxes.append([cx, cy, bw, bh])

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            image_tensor = image_tensor.flip(-1)
            if boxes_tensor.numel() > 0:
                boxes_tensor[:, 0] = 1.0 - boxes_tensor[:, 0]  # flip cx

        targets = {
            "labels": labels_tensor,
            "boxes": boxes_tensor,
        }
        return image_tensor, targets


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """Collate images into a batch tensor; targets remain a list of dicts."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

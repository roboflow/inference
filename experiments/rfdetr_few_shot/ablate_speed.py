#!/usr/bin/env python3
"""Ablation study for training speed optimizations (v2 — fixed eval).

Tests combinations of:
- group_detr: 1, 4, 13
- num_queries: 50, 100, 300
- aux_loss: True, False

Measures ms/epoch and total experiment time, plus mAP to check accuracy.

Key fix from v1: PostProcess num_select must match model's num_queries,
and matcher reverted to original batched form (compiled per-sample was buggy).
"""

import copy
import json
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "experiments/rfdetr_few_shot")
from grid_search import (
    BACKBONE_LORA_TARGETS,
    CLASS_NAMES,
    DECODER_LORA_TARGETS,
    EVAL_IMAGES,
    MAP_IOU_THRESHOLDS,
    PHASE2_FIXED,
    TRAIN_IMAGE_SETS,
    InlineFewShotDataset,
    _gpu_augment_batch,
    compute_map,
    load_base_model,
    load_image_and_labels,
)
from peft import LoraConfig, get_peft_model


def run_inference_with_num_select(model, config, pil_image, num_select=300,
                                   confidence_threshold=0.01):
    """Run inference with configurable num_select (must match model's num_queries)."""
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

    pp = PostProcess(num_select=num_select)
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


def build_template_and_criterion(base_model, config, num_classes, device,
                                  group_detr=13, num_queries=300, aux_loss=True):
    """Build template model and criterion with configurable params."""
    from inference_models.models.rfdetr.few_shot.criterion import SetCriterion
    from inference_models.models.rfdetr.few_shot.matcher import HungarianMatcher
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    fresh_config = config.model_copy(update={
        "num_classes": num_classes,
        "group_detr": group_detr,
        "num_queries": num_queries,
        "num_select": num_queries,
    })
    template = build_model(config=fresh_config)

    base_state = base_model.state_dict()
    template_state = template.state_dict()
    filtered = {k: v for k, v in base_state.items()
                if k in template_state and v.shape == template_state[k].shape}
    template.load_state_dict(filtered, strict=False)
    template.reinitialize_detection_head(num_classes + 1)
    template = template.to(device)

    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
    weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
    if aux_loss:
        for i in range(fresh_config.dec_layers - 1):
            weight_dict.update({f"{k}_{i}": v for k, v in
                                {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}.items()})

    criterion = SetCriterion(
        num_classes=num_classes + 1, matcher=matcher, weight_dict=weight_dict,
        focal_alpha=0.25, losses=["labels", "boxes", "cardinality"],
        group_detr=group_detr,
        ia_bce_loss=getattr(fresh_config, "ia_bce_loss", True),
    ).to(device)

    return template, filtered, fresh_config, criterion, weight_dict


def train_and_eval(template, filtered_state, config, dataset, num_classes, device,
                   criterion, weight_dict, num_queries=300, num_epochs=50):
    """Train with LoRA and evaluate. Returns (train_time, ms_per_epoch, mAP_50)."""
    model = copy.deepcopy(template)
    model.reinitialize_detection_head(num_classes + 1)

    lora_cfg = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0,
        target_modules=BACKBONE_LORA_TARGETS + DECODER_LORA_TARGETS,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    for name, p in peft_model.named_parameters():
        if any(kw in name for kw in ("class_embed", "bbox_embed",
                                      "enc_out_class_embed", "enc_out_bbox_embed")):
            p.requires_grad = True

    # Pre-load to GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    gpu_base_images, gpu_labels, gpu_boxes = [], [], []
    for idx in range(len(dataset)):
        base_tensor, yolo_boxes = dataset.items[idx]
        gpu_base_images.append(base_tensor.to(device))
        labels, boxes = [], []
        for b in yolo_boxes:
            cls_name = CLASS_NAMES[b["class_id"]]
            if cls_name not in dataset.cls2id:
                continue
            labels.append(dataset.cls2id[cls_name])
            boxes.append([b["cx"], b["cy"], b["w"], b["h"]])
        gpu_labels.append(torch.tensor(labels, dtype=torch.long, device=device))
        gpu_boxes.append(torch.tensor(boxes, dtype=torch.float32, device=device) if boxes
                         else torch.zeros((0, 4), device=device))

    n_images = len(gpu_base_images)
    batch_size = min(n_images, 4)

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=0.002, weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.002 * 0.01,
    )

    peft_model.train()
    criterion.train()
    indices = list(range(n_images))
    aug_level = dataset.augmentation_level
    resolution = dataset.resolution

    torch.cuda.synchronize()
    t_start = time.time()

    for epoch in range(num_epochs):
        random.shuffle(indices)
        for start in range(0, n_images, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_imgs = torch.stack([gpu_base_images[i].clone() for i in batch_idx])
            batch_boxes_t = [gpu_boxes[i].clone() for i in batch_idx]
            batch_labels_t = [gpu_labels[i].clone() for i in batch_idx]

            if dataset.augment:
                batch_imgs, batch_boxes_t, batch_labels_t = _gpu_augment_batch(
                    batch_imgs, batch_boxes_t, batch_labels_t, aug_level, resolution)

            batch_imgs = (batch_imgs - mean) / std
            targets = [{"labels": l, "boxes": b} for l, b in zip(batch_labels_t, batch_boxes_t)]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                     enabled=device.type == "cuda"):
                outputs = peft_model(batch_imgs)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * weight_dict[k]
                             for k in loss_dict if k in weight_dict)
            losses.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
        scheduler.step()

    torch.cuda.synchronize()
    train_time = time.time() - t_start
    ms_per_epoch = (train_time / num_epochs) * 1000

    # Eval — use matching num_select
    merged = peft_model.merge_and_unload().eval()
    active_classes = list(dataset.cls2id.keys())
    active_to_original = {i: CLASS_NAMES.index(c) for i, c in enumerate(active_classes)}
    all_class_ids = set(CLASS_NAMES.index(c) for c in active_classes)

    all_mAP_50 = []
    for eval_stem in EVAL_IMAGES:
        eval_img, eval_boxes = load_image_and_labels("valid", eval_stem)
        raw_dets = run_inference_with_num_select(
            merged, config, eval_img, num_select=num_queries, confidence_threshold=0.01)
        remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                    for d in raw_dets if d["class_id"] in active_to_original]
        metrics = compute_map(remapped, eval_boxes, eval_img.size, all_class_ids, MAP_IOU_THRESHOLDS)
        all_mAP_50.append(metrics["mAP_50"])

    avg_mAP_50 = float(np.mean(all_mAP_50)) if all_mAP_50 else 0

    del merged, peft_model
    torch.cuda.empty_cache()

    return train_time, ms_per_epoch, avg_mAP_50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model and data once
    base_model, config = load_base_model(device)

    stems = list(TRAIN_IMAGE_SETS.values())[0][:5]
    images_and_boxes, all_class_ids = [], set()
    for stem in stems:
        img, boxes = load_image_and_labels("train", stem)
        images_and_boxes.append((img, boxes))
        for b in boxes:
            all_class_ids.add(b["class_id"])
    active_classes = [CLASS_NAMES[i] for i in sorted(all_class_ids)]
    num_classes = len(active_classes)
    dataset = InlineFewShotDataset(images_and_boxes, active_classes, config.resolution,
                                    augment=True, augmentation_level=1)

    print(f"Classes: {num_classes}, Train images: {len(stems)}, Eval images: {len(EVAL_IMAGES)}")
    print()

    # Ablation configs: (label, group_detr, num_queries, aux_loss)
    ablations = [
        ("BASELINE (g=13, q=300, aux=T)",  13, 300, True),
        ("g=4, q=300, aux=T",               4, 300, True),
        ("g=1, q=300, aux=T",               1, 300, True),
        ("g=13, q=100, aux=T",             13, 100, True),
        ("g=4, q=100, aux=T",               4, 100, True),
        ("g=1, q=100, aux=T",               1, 100, True),
        ("g=1, q=50, aux=T",                1,  50, True),
        ("g=13, q=300, aux=F",             13, 300, False),
        ("g=4, q=300, aux=F",               4, 300, False),
        ("g=1, q=300, aux=F",               1, 300, False),
        ("g=4, q=100, aux=F",               4, 100, False),
        ("g=1, q=100, aux=F",               1, 100, False),
    ]

    results = []
    print(f"{'Config':<40} {'Train(s)':>8} {'ms/ep':>7} {'mAP50':>7} {'Speedup':>8}")
    print("=" * 75)

    baseline_ms = None

    for label, gd, nq, aux in ablations:
        print(f"  Running: {label}...", end=" ", flush=True)

        try:
            template, fstate, fconfig, criterion, wd = build_template_and_criterion(
                base_model, config, num_classes, device,
                group_detr=gd, num_queries=nq, aux_loss=aux)

            train_time, ms_ep, mAP = train_and_eval(
                template, fstate, fconfig, dataset, num_classes, device,
                criterion, wd, num_queries=nq, num_epochs=50)

            if baseline_ms is None:
                baseline_ms = ms_ep

            speedup = baseline_ms / ms_ep if ms_ep > 0 else 0
            print(f"\r{label:<40} {train_time:>7.1f}s {ms_ep:>6.0f}ms {mAP*100:>6.1f}% {speedup:>7.1f}x")
            results.append({
                "label": label, "group_detr": gd, "num_queries": nq,
                "aux_loss": aux,
                "train_time": round(train_time, 2),
                "ms_per_epoch": round(ms_ep, 1),
                "mAP_50": round(mAP * 100, 2),
                "speedup": round(speedup, 2),
            })

            del template, criterion
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({"label": label, "error": str(e)[:200]})

    print()
    print("=" * 75)
    print()
    print("=== RESULTS JSON ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

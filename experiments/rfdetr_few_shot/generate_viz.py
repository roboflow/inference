#!/usr/bin/env python3
"""Generate visualization data for ALL winning models from Phase 3 results.

Finds every config that wins at least one dataset on mAP@50:95, re-trains each
on all datasets, runs inference on one test image per dataset, and saves
annotated prediction / GT / hybrid images + a JSON manifest.
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use grid_search helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from grid_search import (
    load_base_model,
    discover_benchmark_datasets,
    load_image_and_labels_generic,
    InlineFewShotDataset,
    train_lora_fast,
    run_inference,
    compute_map,
    prepare_criterion,
    MAP_IOU_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
PALETTE = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _adaptive_sizes(img_w, img_h):
    """Return (line_width, thin_width, font_size_pred, font_size_gt) scaled to image."""
    diag = (img_w ** 2 + img_h ** 2) ** 0.5
    scale = max(diag / 640.0, 1.0)
    return (
        max(int(3 * scale), 2),
        max(int(1 * scale), 1),
        max(int(14 * scale), 10),
        max(int(12 * scale), 9),
    )


def _load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized cx,cy,w,h to absolute x1,y1,x2,y2."""
    cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def draw_predictions(pil_img, detections, class_names, conf_thresh=0.25):
    """Draw bounding boxes on image with adaptive sizes. Returns annotated PIL image."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    line_w, _, font_sz, _ = _adaptive_sizes(img_w, img_h)
    font = _load_font(font_sz)

    for det in sorted(detections, key=lambda d: d["confidence"]):
        if det["confidence"] < conf_thresh:
            continue
        x1, y1, x2, y2 = det["box"]
        cid = det["class_id"]
        color = PALETTE[cid % len(PALETTE)]
        label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
        txt = f"{label} {det['confidence']:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        bbox = draw.textbbox((x1, y1), txt, font=font)
        draw.rectangle([bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1], fill=color)
        draw.text((x1, y1), txt, fill="white", font=font)
    return img


def draw_ground_truth(pil_img, gt_boxes, class_names):
    """Draw ground truth boxes with adaptive thinner lines."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    _, thin_w, _, font_sz = _adaptive_sizes(img_w, img_h)
    font = _load_font(font_sz)

    for box in gt_boxes:
        cid = box["class_id"]
        color = PALETTE[cid % len(PALETTE)]
        label = class_names[cid] if cid < len(class_names) else f"cls{cid}"
        x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thin_w)
        draw.text((x1, max(y1 - font_sz - 2, 0)), f"GT: {label}", fill=color, font=font)
    return img


def draw_hybrid(pil_img, detections, gt_boxes, class_names, conf_thresh=0.25):
    """Draw hybrid comparison using same logic as ModelComparisonAnnotator.

    Green      = FP (predicted, no matching GT)
    Red        = FN (GT, no matching prediction)
    Transparent = TP (both predicted and GT — original image shows through)
    Dimmed     = background (neither predicted nor GT)
    """
    import cv2

    img_w, img_h = pil_img.size
    scene = np.array(pil_img)
    opacity = 0.7

    # Build pred xyxy
    pred_boxes = [det["box"] for det in detections if det["confidence"] >= conf_thresh]
    pred_xyxy = np.array(pred_boxes, dtype=np.float32).reshape(-1, 4) if pred_boxes else np.empty((0, 4), dtype=np.float32)

    # Build GT xyxy
    gt_xyxy_list = [list(yolo_to_xyxy(box, img_w, img_h)) for box in gt_boxes]
    gt_xyxy = np.array(gt_xyxy_list, dtype=np.float32).reshape(-1, 4) if gt_xyxy_list else np.empty((0, 4), dtype=np.float32)

    # Build masks
    neither = np.ones(scene.shape[:2], dtype=np.uint8)
    a_mask = np.zeros(scene.shape[:2], dtype=np.uint8)
    b_mask = np.zeros(scene.shape[:2], dtype=np.uint8)

    for box in pred_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        a_mask[y1:y2, x1:x2] = 1
        neither[y1:y2, x1:x2] = 0

    for box in gt_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        b_mask[y1:y2, x1:x2] = 1
        neither[y1:y2, x1:x2] = 0

    only_a = a_mask & (a_mask ^ b_mask)  # FP: pred only
    only_b = b_mask & (b_mask ^ a_mask)  # FN: GT only
    # Both predicted = unchanged (TP), neither = dimmed background

    def apply_overlay(base, color_rgb, mask):
        overlay = np.full_like(base, color_rgb, dtype=np.uint8)
        blended = cv2.addWeighted(base, 1 - opacity, overlay, opacity, 0)
        mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
        base[mask_3ch] = blended[mask_3ch]
        return base

    scene = apply_overlay(scene, (0, 0, 0), neither)        # black bg dimming
    scene = apply_overlay(scene, (0, 128, 0), only_a)        # green = FP
    scene = apply_overlay(scene, (255, 0, 0), only_b)        # red = FN

    return Image.fromarray(scene)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to Phase 3 results DB")
    parser.add_argument("--datasets-root", required=True, help="Path to datasets")
    parser.add_argument("--output-dir", default="viz_champion", help="Output directory")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--metric", default="mAP_50_95", choices=["mAP_50", "mAP_50_95"],
                        help="Metric to determine champions")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    metric = args.metric

    # ------------------------------------------------------------------
    # 1. Find ALL configs that won at least 1 dataset
    # ------------------------------------------------------------------
    logger.info("Finding all winning configs by %s ...", metric)
    winning_configs = conn.execute(f"""
        WITH ranked AS (
            SELECT *,
                RANK() OVER (PARTITION BY dataset_name ORDER BY {metric} DESC) as rnk
            FROM experiments WHERE status='completed'
        )
        SELECT lora_rank, batch_size, copy_paste, mosaic, alpha_ratio, weight_decay,
               COUNT(*) as wins,
               GROUP_CONCAT(dataset_name) as datasets_won,
               AVG({metric}) as avg_metric
        FROM ranked WHERE rnk = 1
        GROUP BY lora_rank, batch_size, copy_paste, mosaic, alpha_ratio, weight_decay
        ORDER BY wins DESC, avg_metric DESC
    """).fetchall()

    if not winning_configs:
        logger.error("No completed experiments found!")
        return

    logger.info("Found %d winning configs", len(winning_configs))
    for wc in winning_configs:
        logger.info("  rank=%d bs=%d cp=%d mo=%d ar=%s wd=%s → %d wins",
                     wc["lora_rank"], wc["batch_size"], wc["copy_paste"],
                     wc["mosaic"], wc["alpha_ratio"], wc["weight_decay"], wc["wins"])

    # ------------------------------------------------------------------
    # 2. Load base model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading base model on %s ...", device)
    base_model, config = load_base_model(device)

    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    datasets_root = Path(args.datasets_root)
    datasets = discover_benchmark_datasets(datasets_root)
    logger.info("Found %d datasets", len(datasets))

    # ------------------------------------------------------------------
    # 2b. Run base model (COCO) on best test image per dataset
    # ------------------------------------------------------------------
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ]
    base_coco_preds = {}  # ds_name -> list of {class_id, confidence, bbox, coco_class}
    logger.info("Running base model (COCO 80) on test images ...")
    for ds_info in datasets:
        ds_name = ds_info["name"]
        ds_path = ds_info["path"]
        test_stems = ds_info["test_stems"]
        # Pick best test image (most GT boxes)
        best_stem, best_n = test_stems[0], 0
        for stem in test_stems[:10]:
            _, gt = load_image_and_labels_generic(ds_path, "test", stem)
            if len(gt) > best_n:
                best_n = len(gt)
                best_stem = stem
        test_img, _ = load_image_and_labels_generic(ds_path, "test", best_stem)
        base_dets = run_inference(base_model, config, test_img, confidence_threshold=0.05)
        img_w, img_h = test_img.size
        base_coco_preds[ds_name] = [
            {"class_id": d["class_id"],
             "confidence": round(d["confidence"], 4),
             "bbox": [round((d["box"][0]+d["box"][2])/(2*img_w), 6),
                      round((d["box"][1]+d["box"][3])/(2*img_h), 6),
                      round((d["box"][2]-d["box"][0])/img_w, 6),
                      round((d["box"][3]-d["box"][1])/img_h, 6)],
             "coco_class": COCO_CLASSES[d["class_id"]] if d["class_id"] < len(COCO_CLASSES) else f"cls{d['class_id']}"}
            for d in base_dets
        ]
        logger.info("  %s: %d COCO detections @ ≥5%%", ds_name, len(base_coco_preds[ds_name]))

    manifest = {"metric": metric, "models": []}

    # ------------------------------------------------------------------
    # 3. For each winning config, train on all datasets and generate images
    # ------------------------------------------------------------------
    for model_idx, wc in enumerate(winning_configs):
        wc = dict(wc)
        rank = wc["lora_rank"]
        batch_size = wc["batch_size"]
        copy_paste = bool(wc["copy_paste"])
        mosaic = bool(wc["mosaic"])
        alpha_ratio = int(wc["alpha_ratio"])
        weight_decay = wc["weight_decay"]
        alpha = rank * alpha_ratio

        config_key = f"r{rank}_bs{batch_size}_cp{int(copy_paste)}_mo{int(mosaic)}_ar{alpha_ratio}_wd{weight_decay}"
        img_subdir = f"model_{model_idx}"
        (out_dir / "images" / img_subdir).mkdir(exist_ok=True)

        logger.info("━━━ Model %d: %s (%d wins) ━━━", model_idx, config_key, wc["wins"])

        # Get this config's results + ranking on every dataset from DB
        config_results = conn.execute(f"""
            SELECT e.dataset_name, e.mAP_50, e.mAP_50_95,
                   e.train_time_seconds, e.time_per_epoch_ms, e.final_loss,
                   e.num_train_images, e.dataset_num_classes,
                   (SELECT COUNT(*)+1 FROM experiments e2
                    WHERE e2.dataset_name=e.dataset_name AND e2.status='completed'
                      AND e2.{metric} > e.{metric}) as ranking,
                   (SELECT COUNT(*) FROM experiments e3
                    WHERE e3.dataset_name=e.dataset_name AND e3.status='completed') as total_configs
            FROM experiments e
            WHERE e.lora_rank=? AND e.batch_size=? AND e.copy_paste=? AND e.mosaic=?
                  AND e.alpha_ratio=? AND e.weight_decay=?
                  AND e.status='completed'
            ORDER BY e.{metric} DESC
        """, (rank, batch_size, int(copy_paste), int(mosaic),
              alpha_ratio, weight_decay)).fetchall()

        db_info_map = {}
        for r in config_results:
            db_info_map[r["dataset_name"]] = {
                "mAP_50": r["mAP_50"],
                "mAP_50_95": r["mAP_50_95"],
                "train_time_seconds": r["train_time_seconds"],
                "ranking": r["ranking"],
                "total_configs": r["total_configs"],
                "is_winner": r["ranking"] == 1,
            }

        model_entry = {
            "model_index": model_idx,
            "config_key": config_key,
            "config": {
                "rank": rank, "batch_size": batch_size,
                "copy_paste": copy_paste, "mosaic": mosaic,
                "alpha_ratio": alpha_ratio, "weight_decay": weight_decay,
                "alpha": alpha, "epochs": 50, "lr": 0.002, "lora_targets": "v1",
            },
            "wins": wc["wins"],
            "datasets": [],
        }

        for ds_info in datasets:
            ds_name = ds_info["name"]
            ds_path = ds_info["path"]
            class_names = ds_info["class_names"]
            train_stems = ds_info["train_stems"]
            test_stems = ds_info["test_stems"]
            num_classes = ds_info["num_classes"]

            logger.info("  %s (%d classes, %d train)", ds_name, num_classes, len(train_stems))

            # Build template model
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

            # Load training data
            images_and_boxes = []
            all_class_ids = set()
            for stem in train_stems:
                img, boxes = load_image_and_labels_generic(ds_path, "train", stem)
                images_and_boxes.append((img, boxes))
                for b in boxes:
                    all_class_ids.add(b["class_id"])

            active_classes = [class_names[i] for i in sorted(all_class_ids)]

            dataset = InlineFewShotDataset(
                images_and_boxes, active_classes, config.resolution,
                augment=True, augmentation_level=1,
                class_names=class_names,
            )

            # Train
            t0 = time.time()
            merged_model, loss_history, _ = train_lora_fast(
                template, filtered, fresh_config,
                dataset, num_classes, device,
                criterion, weight_dict,
                rank=rank, alpha=alpha, lr=0.002, num_epochs=50,
                lora_dropout=0.0, weight_decay=weight_decay,
                class_names=class_names,
                batch_size=batch_size, lora_targets_version="v1",
                copy_paste=copy_paste, mosaic=mosaic,
                warmup=False, multi_scale=False,
            )
            train_time = time.time() - t0
            logger.info("    Trained in %.1fs", train_time)

            # Pick best test image
            active_to_original = {i: class_names.index(c) for i, c in enumerate(active_classes)}
            best_test_stem = test_stems[0]
            best_n_gt = 0
            for stem in test_stems[:10]:
                _, gt = load_image_and_labels_generic(ds_path, "test", stem)
                if len(gt) > best_n_gt:
                    best_n_gt = len(gt)
                    best_test_stem = stem

            test_img, gt_boxes = load_image_and_labels_generic(ds_path, "test", best_test_stem)

            # Run inference
            raw_dets = run_inference(merged_model, config, test_img, confidence_threshold=0.01)
            remapped = [{**d, "class_id": active_to_original.get(d["class_id"], -1)}
                        for d in raw_dets if d["class_id"] in active_to_original]

            metrics = compute_map(
                remapped, gt_boxes, test_img.size,
                all_class_ids, MAP_IOU_THRESHOLDS,
                class_names=class_names,
            )

            # Draw predictions (adaptive)
            pred_img = draw_predictions(test_img, remapped, class_names, conf_thresh=args.conf_thresh)
            pred_path = f"images/{img_subdir}/{ds_name}_pred.jpg"
            pred_img.save(str(out_dir / pred_path), quality=85)

            # Save original (un-annotated) test image
            orig_path = f"images/{img_subdir}/{ds_name}_original.jpg"
            test_img.save(str(out_dir / orig_path), quality=90)

            # Draw ground truth (adaptive)
            gt_img = draw_ground_truth(test_img, gt_boxes, class_names)
            gt_path = f"images/{img_subdir}/{ds_name}_gt.jpg"
            gt_img.save(str(out_dir / gt_path), quality=85)

            # Draw hybrid
            hybrid_path = f"images/{img_subdir}/{ds_name}_hybrid.jpg"
            try:
                hybrid_img = draw_hybrid(test_img, remapped, gt_boxes, class_names,
                                         conf_thresh=args.conf_thresh)
                hybrid_img.save(str(out_dir / hybrid_path), quality=85)
            except Exception as hybrid_err:
                logger.warning("    ⚠️ Hybrid failed: %s", hybrid_err)
                hybrid_path = pred_path  # fallback

            db_info = db_info_map.get(ds_name, {})
            ds_entry = {
                "name": ds_name,
                "num_classes": num_classes,
                "num_train_images": len(train_stems),
                "num_test_images": len(test_stems),
                "class_names": class_names,
                "mAP_50": db_info.get("mAP_50", metrics["mAP_50"]),
                "mAP_50_95": db_info.get("mAP_50_95", metrics["mAP_50_95"]),
                "train_time_seconds": db_info.get("train_time_seconds", train_time),
                "ranking": db_info.get("ranking", -1),
                "total_configs": db_info.get("total_configs", -1),
                "is_winner": db_info.get("is_winner", False),
                "pred_image": pred_path,
                "gt_image": gt_path,
                "hybrid_image": hybrid_path,
                "original_image": orig_path,
                "test_image_stem": best_test_stem,
                "sample_mAP_50": metrics["mAP_50"],
                "sample_mAP_50_95": metrics["mAP_50_95"],
                "num_gt_boxes": len(gt_boxes),
                "num_predictions": len([d for d in remapped if d["confidence"] >= args.conf_thresh]),
                "raw_detections": [
                    {"class_id": d["class_id"], "confidence": round(d["confidence"], 4),
                     "bbox": [round((d["box"][0]+d["box"][2])/(2*test_img.width), 6),
                              round((d["box"][1]+d["box"][3])/(2*test_img.height), 6),
                              round((d["box"][2]-d["box"][0])/test_img.width, 6),
                              round((d["box"][3]-d["box"][1])/test_img.height, 6)]}
                    for d in remapped
                ],
                "gt_boxes_data": [
                    {"class_id": b["class_id"],
                     "bbox": [round(b["cx"], 6), round(b["cy"], 6),
                              round(b["w"], 6), round(b["h"], 6)]}
                    for b in gt_boxes
                ],
                "image_size": [test_img.width, test_img.height],
                "base_coco_detections": base_coco_preds.get(ds_name, []),
            }
            model_entry["datasets"].append(ds_entry)

            winner_str = "★ WINNER" if ds_entry["is_winner"] else ""
            logger.info("    mAP@50=%.1f%% mAP@50:95=%.1f%% | rank=%d/%d %s | %d preds",
                         ds_entry["mAP_50"]*100, ds_entry["mAP_50_95"]*100,
                         ds_entry["ranking"], ds_entry["total_configs"],
                         winner_str, ds_entry["num_predictions"])

            del merged_model, template, criterion
            torch.cuda.empty_cache()

        # Compute model summary stats
        ds_list = model_entry["datasets"]
        if ds_list:
            model_entry["avg_mAP_50"] = sum(d["mAP_50"] for d in ds_list) / len(ds_list)
            model_entry["avg_mAP_50_95"] = sum(d["mAP_50_95"] for d in ds_list) / len(ds_list)
            model_entry["avg_train_time"] = sum(d.get("train_time_seconds", 0) for d in ds_list) / len(ds_list)

        manifest["models"].append(model_entry)

    conn.close()

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("✅ Manifest saved to %s", manifest_path)
    logger.info("✅ %d models × %d datasets = images saved to %s/images/",
                 len(manifest["models"]), len(datasets), out_dir)


if __name__ == "__main__":
    main()

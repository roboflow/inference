#!/usr/bin/env python3
"""Evaluate SAM3 text-prompt zero-shot detection on RF20-VL-FSOD benchmark.

Runs SAM3 concept segmentation (text prompts = class names from data.yaml) on
each dataset's test split, converts masks to bounding boxes, and writes results
to the shared results DB for comparison with RF-DETR LoRA/FT baselines.

Usage:
    python eval_sam3.py --datasets-root ~/Downloads/rf20-vl-fsod
    python eval_sam3.py --smoke  # 1 dataset only
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "inference_models"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARK_DATASETS_ROOT = Path.home() / "Downloads" / "rf20-vl-fsod"
DB_PATH = Path(__file__).parent / "results_finetune_baseline.db"


# ---------------------------------------------------------------------------
# DB helpers (reused from finetune_baseline.py)
# ---------------------------------------------------------------------------

def _db_connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.row_factory = sqlite3.Row
    return conn


def _db_execute_with_retry(conn, sql, params=(), max_retries=10):
    import random
    for attempt in range(max_retries):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(1.0 + random.random() * 2)
            else:
                raise


def _db_commit_with_retry(conn, max_retries=10):
    import random
    for attempt in range(max_retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(1.0 + random.random() * 2)
            else:
                raise


# ---------------------------------------------------------------------------
# Dataset discovery (same as other eval scripts)
# ---------------------------------------------------------------------------

def discover_benchmark_datasets(root):
    datasets = []
    if not root.exists():
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
            names = [names[k] for k in sorted(names.keys())]
        test_dir = ds_dir / "test" / "images"
        if not test_dir.exists():
            continue
        test_stems = sorted(set(
            f.stem for f in test_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ))
        if not test_stems:
            continue
        datasets.append({
            "name": ds_dir.name,
            "path": ds_dir,
            "num_classes": len(names),
            "class_names": names,
            "test_stems": test_stems,
        })
    logger.info("Discovered %d benchmark datasets in %s", len(datasets), root)
    return datasets


# ---------------------------------------------------------------------------
# COCO mAP computation
# ---------------------------------------------------------------------------

def compute_dataset_map(all_preds, all_gt, active_class_ids, max_dets=500):
    """Compute COCO-style dataset-level mAP using pycocotools."""
    if not all_gt:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}

    image_ids = sorted(set(g["image_id"] for g in all_gt))
    images = [{"id": img_id} for img_id in image_ids]
    annotations = []
    for i, g in enumerate(all_gt):
        x1, y1, x2, y2 = g["box"]
        w, h = x2 - x1, y2 - y1
        annotations.append({
            "id": i + 1, "image_id": g["image_id"],
            "category_id": g["class_id"],
            "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
        })
    categories = [{"id": cid} for cid in sorted(active_class_ids)]

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt.createIndex()

    if not all_preds:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}

    coco_dets = []
    for p in all_preds:
        x1, y1, x2, y2 = p["box"]
        w, h = x2 - x1, y2 - y1
        coco_dets.append({
            "image_id": p["image_id"], "category_id": p["class_id"],
            "bbox": [x1, y1, w, h], "score": p["confidence"],
        })

    coco_dt = coco_gt.loadRes(coco_dets)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 100, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP_50": max(0.0, float(coco_eval.stats[1])),
        "mAP_50_95": max(0.0, float(coco_eval.stats[0])),
    }


# ---------------------------------------------------------------------------
# Mask to bounding box conversion
# ---------------------------------------------------------------------------

def mask_polygons_to_bbox(mask_polygons, img_w, img_h):
    """Convert SAM3 polygon mask format to a bounding box [x1, y1, x2, y2].

    mask_polygons: list of polygons, each polygon is a list of [x, y] pairs
        e.g. [[[x1,y1], [x2,y2], ...], [[x1,y1], ...]]
    Returns: [x1, y1, x2, y2] bounding box
    """
    all_x = []
    all_y = []
    for poly in mask_polygons:
        for point in poly:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                all_x.append(point[0])
                all_y.append(point[1])
            else:
                # Fallback: flat list [x1, y1, x2, y2, ...]
                all_x.extend(poly[0::2])
                all_y.extend(poly[1::2])
                break

    if not all_x:
        return None

    x1 = max(0.0, min(all_x))
    y1 = max(0.0, min(all_y))
    x2 = min(float(img_w), max(all_x))
    y2 = min(float(img_h), max(all_y))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def rle_to_bbox(rle_dict, img_w, img_h):
    """Convert RLE mask to bounding box using pycocotools."""
    from pycocotools import mask as mask_utils

    # Decode RLE to binary mask
    if isinstance(rle_dict, dict) and "counts" in rle_dict:
        if isinstance(rle_dict["counts"], str):
            # Compressed RLE
            binary_mask = mask_utils.decode(rle_dict)
        else:
            binary_mask = mask_utils.decode(rle_dict)
    else:
        return None

    # Find bounding box from mask
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_sam3_on_dataset(model, ds_info, conn, db_path):
    """Run SAM3 text-prompt detection on one dataset and write results to DB."""
    from inference.core.entities.requests.sam3 import Sam3Prompt

    ds_name = ds_info["name"]
    ds_path = ds_info["path"]
    class_names = ds_info["class_names"]
    num_classes = len(class_names)
    test_stems = ds_info["test_stems"]
    run_id = f"sam3_{ds_name}"

    # Check if already done
    existing = conn.execute(
        "SELECT status FROM experiments WHERE run_id=?", (run_id,)
    ).fetchone()
    if existing and existing["status"] == "completed":
        logger.info("  Skipping %s (already completed)", run_id)
        return

    # Insert/update experiment row
    _db_execute_with_retry(conn, """
        INSERT OR REPLACE INTO experiments
        (run_id, timestamp, dataset_name, dataset_num_classes, num_train_images,
         model_variant, method, num_epochs, status, device)
        VALUES (?, ?, ?, ?, 0, 'sam3', 'zero_shot', 0, 'running', 'cuda')
    """, (run_id, datetime.now().isoformat(), ds_name, num_classes))
    _db_commit_with_retry(conn)

    t0 = time.time()

    # Build prompts — one per class name
    prompts = [Sam3Prompt(type="text", text=cn) for cn in class_names]

    all_preds = []
    all_gt = []
    active_class_ids = set(range(num_classes))

    for img_idx, stem in enumerate(test_stems):
        image_id = img_idx + 1

        # Find image file
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = ds_path / "test" / "images" / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size
        np_img = np.array(pil_img)  # RGB uint8 numpy array

        # Run SAM3 concept segmentation
        try:
            response = model.segment_image(
                image=np_img,
                prompts=prompts,
                output_prob_thresh=0.01,  # low threshold for mAP computation
                format="polygon",
            )
        except Exception as e:
            logger.warning("  SAM3 failed on %s/%s: %s", ds_name, stem, e)
            continue

        # Convert SAM3 results to detection format
        for prompt_result in response.prompt_results:
            class_id = prompt_result.prompt_index  # maps to class_names index
            if class_id >= num_classes:
                continue
            for pred in prompt_result.predictions:
                conf = pred.confidence
                # Convert mask to bbox
                bbox = mask_polygons_to_bbox(pred.masks, img_w, img_h)
                if bbox is None:
                    continue
                all_preds.append({
                    "image_id": image_id,
                    "class_id": class_id,
                    "confidence": conf,
                    "box": bbox,
                })

        # Load GT labels
        lbl_path = ds_path / "test" / "labels" / f"{stem}.txt"
        if lbl_path.exists():
            for line in open(lbl_path).read().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                all_gt.append({
                    "image_id": image_id,
                    "class_id": cid,
                    "box": [x1, y1, x2, y2],
                })

        if (img_idx + 1) % 50 == 0:
            logger.info("  %s: %d/%d images processed (%d preds so far)",
                        ds_name, img_idx + 1, len(test_stems), len(all_preds))

    eval_time = time.time() - t0
    logger.info("  %s: %d images, %d predictions, %d GT in %.1fs",
                ds_name, len(test_stems), len(all_preds), len(all_gt), eval_time)

    # Compute dataset-level mAP
    metrics = compute_dataset_map(all_preds, all_gt, active_class_ids)
    logger.info("  %s: mAP@50:95=%.4f  mAP@50=%.4f",
                ds_name, metrics["mAP_50_95"], metrics["mAP_50"])

    # Update DB
    _db_execute_with_retry(conn, """
        UPDATE experiments SET
            train_time_seconds=?, mAP_50=?, mAP_50_95=?, status='completed'
        WHERE run_id=?
    """, (eval_time, metrics["mAP_50"], metrics["mAP_50_95"], run_id))
    _db_commit_with_retry(conn)


def main():
    parser = argparse.ArgumentParser(description="SAM3 text-prompt evaluation on RF20")
    parser.add_argument("--datasets-root", type=str, default=str(BENCHMARK_DATASETS_ROOT))
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="Only evaluate this dataset name")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    db_path = Path(args.db)
    datasets = discover_benchmark_datasets(datasets_root)

    if args.only:
        datasets = [d for d in datasets if d["name"] == args.only]
    if args.smoke:
        datasets = datasets[:1]

    if not datasets:
        logger.error("No datasets found!")
        return

    # Initialize SAM3 model
    logger.info("Loading SAM3 model...")
    os.environ.setdefault("ROBOFLOW_API_KEY", os.environ.get("ROBOFLOW_API_KEY", ""))
    from inference.models.sam3 import SegmentAnything3
    model = SegmentAnything3(model_id="sam3/sam3_final")
    logger.info("SAM3 model loaded")

    conn = _db_connect(db_path)

    for i, ds_info in enumerate(datasets):
        logger.info("[%d/%d] Evaluating SAM3 on %s (%d classes, %d test images)",
                    i + 1, len(datasets), ds_info["name"],
                    ds_info["num_classes"], len(ds_info["test_stems"]))
        try:
            evaluate_sam3_on_dataset(model, ds_info, conn, db_path)
        except Exception as e:
            logger.error("FAILED on %s: %s", ds_info["name"], e, exc_info=True)
            run_id = f"sam3_{ds_info['name']}"
            _db_execute_with_retry(conn, """
                UPDATE experiments SET status='failed', error_message=? WHERE run_id=?
            """, (str(e)[:500], run_id))
            _db_commit_with_retry(conn)
        torch.cuda.empty_cache()

    conn.close()
    logger.info("All done!")


if __name__ == "__main__":
    main()

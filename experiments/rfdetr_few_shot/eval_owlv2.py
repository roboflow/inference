#!/usr/bin/env python3
"""Evaluate OWLv2 zero-shot detection on RF20-VL-FSOD benchmark.

Runs OWLv2-large zero-shot (text prompts = class names from data.yaml) on each
dataset's test split and writes results to the shared results DB for comparison
with RF-DETR LoRA/FT baselines.

Usage:
    python eval_owlv2.py --datasets-root ~/Downloads/rf20-vl-fsod
    python eval_owlv2.py --smoke  # 1 dataset only
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


def init_db(db_path):
    """Ensure experiments table exists (may already exist from finetune_baseline)."""
    conn = _db_connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "experiments" not in tables:
        conn.execute("""CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            dataset_name TEXT,
            dataset_num_classes INTEGER,
            num_train_images INTEGER,
            model_variant TEXT,
            method TEXT,
            num_epochs INTEGER,
            learning_rate REAL,
            batch_size INTEGER,
            grad_accum_steps INTEGER,
            train_time_seconds REAL,
            final_loss REAL, loss_history TEXT,
            mAP_50 REAL, mAP_50_95 REAL,
            status TEXT DEFAULT 'pending',
            error_message TEXT, device TEXT,
            current_epoch INTEGER, current_loss REAL,
            current_map REAL, best_epoch INTEGER, best_val_map REAL,
            notes TEXT
        )""")
    if "eval_results" not in tables:
        conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            eval_image_stem TEXT, eval_split TEXT,
            mAP_50 REAL, mAP_50_95 REAL,
            per_class_ap_json TEXT, conf_metrics_json TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )""")
    if "grid_meta" not in tables:
        conn.execute("CREATE TABLE IF NOT EXISTS grid_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Dataset discovery (same as finetune_baseline.py)
# ---------------------------------------------------------------------------

def _discover_stems(images_dir: Path):
    if not images_dir.exists():
        return []
    stems = set()
    for f in images_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            stems.add(f.stem)
    return sorted(stems)


def discover_datasets(root: Path):
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
        test_stems = _discover_stems(ds_dir / "test" / "images")
        if not test_stems:
            continue
        datasets.append({
            "name": ds_dir.name,
            "path": ds_dir,
            "num_classes": len(names),
            "class_names": names,
            "test_stems": test_stems,
        })
    return datasets


# ---------------------------------------------------------------------------
# COCO mAP computation
# ---------------------------------------------------------------------------

def compute_dataset_map(all_preds, all_gt, active_class_ids, max_dets=500):
    """Compute COCO-style mAP using pycocotools."""
    if not all_gt:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}

    image_ids = sorted(set(g["image_id"] for g in all_gt))
    images = [{"id": img_id} for img_id in image_ids]
    annotations = []
    ann_id = 1
    for g in all_gt:
        x1, y1, x2, y2 = g["box"]
        w, h = x2 - x1, y2 - y1
        annotations.append({
            "id": ann_id, "image_id": g["image_id"],
            "category_id": g["class_id"],
            "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
        })
        ann_id += 1

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
    # pycocotools.summarize() hardcodes stats[0] = AP@50:95 at maxDets=100
    # and stats[1] = AP@50 at maxDets[2]. We must include 100 for stats[0].
    coco_eval.params.maxDets = [1, 100, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50_95 = max(0.0, float(coco_eval.stats[0]))
    mAP_50 = max(0.0, float(coco_eval.stats[1]))
    return {"mAP_50": mAP_50, "mAP_50_95": mAP_50_95}


# ---------------------------------------------------------------------------
# OWLv2 evaluation
# ---------------------------------------------------------------------------

def load_owlv2_model(device="cuda"):
    """Load OWLv2-large for zero-shot detection."""
    from inference_models.models.owlv2.owlv2_hf import OWLv2HF
    logger.info("Loading OWLv2-large-patch14-ensemble...")
    model = OWLv2HF.from_pretrained(
        "google/owlv2-large-patch14-ensemble",
        device=torch.device(device),
        local_files_only=False,
    )
    logger.info("OWLv2 loaded on %s", device)
    return model


def parse_yolo_label(lbl_path):
    """Parse a YOLO label file into list of dicts."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in open(lbl_path).read().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        boxes.append({
            "class_id": int(parts[0]),
            "cx": float(parts[1]), "cy": float(parts[2]),
            "w": float(parts[3]), "h": float(parts[4]),
        })
    return boxes


def eval_owlv2_on_dataset(model, ds_info, conn, db_path):
    """Run OWLv2 zero-shot on one dataset and write results to DB."""
    ds_name = ds_info["name"]
    class_names = ds_info["class_names"]
    run_id = f"owlv2_{ds_name}"

    # Skip if already done
    existing = conn.execute(
        "SELECT status FROM experiments WHERE run_id=?", (run_id,)
    ).fetchone()
    if existing and existing["status"] == "completed":
        logger.info("Skipping %s (already completed)", run_id)
        return

    # Insert or update experiment row
    _db_execute_with_retry(conn, """
        INSERT OR REPLACE INTO experiments
        (run_id, timestamp, dataset_name, dataset_num_classes, num_train_images,
         model_variant, method, num_epochs, status, device)
        VALUES (?, ?, ?, ?, ?, 'owlv2-large', 'zero_shot', 0, 'running', ?)
    """, (run_id, datetime.now().isoformat(), ds_name,
          ds_info["num_classes"], 0,  # zero-shot = no training images
          "cuda" if torch.cuda.is_available() else "cpu"))
    _db_commit_with_retry(conn)

    logger.info("Evaluating OWLv2 zero-shot on %s (%d classes, %d test images)",
                ds_name, ds_info["num_classes"], len(ds_info["test_stems"]))

    try:
        ds_path = ds_info["path"]
        active_class_ids = set(range(len(class_names)))
        all_preds = []
        all_gt = []

        t0 = time.time()
        for img_idx, stem in enumerate(ds_info["test_stems"]):
            image_id = img_idx + 1

            # Load image
            img_dir = ds_path / "test" / "images"
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            pil_img = Image.open(img_path).convert("RGB")
            img_w, img_h = pil_img.size
            img_np = np.array(pil_img)

            # OWLv2 zero-shot inference with class names as text prompts
            # Use very low confidence to get all possible detections for mAP
            detections_list = model.infer(
                img_np,
                classes=class_names,
                confidence=0.001,  # very low for mAP computation
                iou_threshold=0.7,  # loose NMS to keep more candidates
                max_detections=500,
            )
            detections = detections_list[0]  # single image

            # Convert to our format
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    cid = int(detections.class_id[i])
                    if cid not in active_class_ids:
                        continue
                    box = detections.xyxy[i].cpu().float().tolist()
                    all_preds.append({
                        "image_id": image_id,
                        "class_id": cid,
                        "confidence": float(detections.confidence[i]),
                        "box": box,
                    })

            # Load GT labels
            lbl_path = ds_path / "test" / "labels" / f"{stem}.txt"
            gt_boxes = parse_yolo_label(lbl_path)
            for b in gt_boxes:
                if b["class_id"] not in active_class_ids:
                    continue
                all_gt.append({
                    "image_id": image_id,
                    "class_id": b["class_id"],
                    "box": [
                        (b["cx"] - b["w"] / 2) * img_w,
                        (b["cy"] - b["h"] / 2) * img_h,
                        (b["cx"] + b["w"] / 2) * img_w,
                        (b["cy"] + b["h"] / 2) * img_h,
                    ],
                })

            if (img_idx + 1) % 50 == 0:
                logger.info("  %d / %d images processed", img_idx + 1, len(ds_info["test_stems"]))

        eval_time = time.time() - t0

        # Compute dataset-level mAP
        metrics = compute_dataset_map(all_preds, all_gt, active_class_ids)
        logger.info("OWLv2 %s: mAP@50=%.4f  mAP@50:95=%.4f  (%.1fs, %d preds)",
                     ds_name, metrics["mAP_50"], metrics["mAP_50_95"],
                     eval_time, len(all_preds))

        # Update DB
        _db_execute_with_retry(conn, """
            UPDATE experiments SET
                mAP_50=?, mAP_50_95=?, status='completed',
                train_time_seconds=?,
                notes=?
            WHERE run_id=?
        """, (metrics["mAP_50"], metrics["mAP_50_95"], eval_time,
              f"zero-shot with {len(class_names)} class names, {len(all_preds)} total predictions",
              run_id))
        _db_commit_with_retry(conn)

    except Exception as e:
        logger.error("FAILED: %s — %s", run_id, e, exc_info=True)
        _db_execute_with_retry(conn, """
            UPDATE experiments SET status='failed', error_message=? WHERE run_id=?
        """, (str(e)[:500], run_id))
        _db_commit_with_retry(conn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate OWLv2 zero-shot on RF20-VL-FSOD")
    parser.add_argument("--datasets-root", type=str, default=str(BENCHMARK_DATASETS_ROOT))
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--smoke", action="store_true", help="Run on 1 dataset only")
    parser.add_argument("--only", type=str, default=None, help="Run on specific dataset")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    db_path = Path(args.db)

    datasets = discover_datasets(datasets_root)
    logger.info("Discovered %d datasets in %s", len(datasets), datasets_root)

    if args.only:
        datasets = [d for d in datasets if d["name"] == args.only]
    if args.smoke:
        datasets = datasets[:1]

    conn = init_db(db_path)

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_owlv2_model(device)

    for i, ds_info in enumerate(datasets):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s (%d classes)", i + 1, len(datasets),
                     ds_info["name"], ds_info["num_classes"])
        logger.info("=" * 60)
        eval_owlv2_on_dataset(model, ds_info, conn, db_path)
        torch.cuda.empty_cache()

    logger.info("All done! Results in %s", db_path)
    conn.close()


if __name__ == "__main__":
    main()

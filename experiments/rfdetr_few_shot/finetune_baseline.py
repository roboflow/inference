#!/usr/bin/env python3
"""RF-DETR full fine-tune baseline on RF20-VL-FSOD datasets.

Trains RF-DETR Nano (optionally Medium) on each of the 20 RF20 datasets using the
official rfdetr package with default hyperparameters, then evaluates on the test
split using the same compute_map() pipeline as the LoRA experiments for an
apples-to-apples comparison.

Results are written to results_finetune_baseline.db, compatible with server.py.

Usage:
    # Full run (nano only by default)
    python finetune_baseline.py --datasets-root ~/Downloads/rf20-vl-fsod

    # Include medium model too
    python finetune_baseline.py --datasets-root ~/Downloads/rf20-vl-fsod --include-medium

    # Smoke test (1 dataset, 5 epochs)
    python finetune_baseline.py --smoke

    # Resume (skips completed experiments automatically)
    python finetune_baseline.py --datasets-root ~/Downloads/rf20-vl-fsod
"""

import argparse
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from PIL import Image

# Default FT timeout: 45 minutes per training run (most take 5-20 min)
FT_TIMEOUT_SECONDS = 2700

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "results_finetune_baseline.db"
OUTPUT_ROOT = SCRIPT_DIR / "finetune_outputs"
BENCHMARK_DATASETS_ROOT = Path.home() / "Downloads" / "rf20-vl-fsod"

# ---------------------------------------------------------------------------
# DB helpers (mirrored from grid_search.py for standalone usage)
# ---------------------------------------------------------------------------

def _db_connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def _db_execute_with_retry(conn, sql, params=(), max_retries=10):
    for attempt in range(max_retries):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait = 1.0 + random.random() * 2
                time.sleep(wait)
            else:
                raise


def _db_commit_with_retry(conn, max_retries=10):
    for attempt in range(max_retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                wait = 1.0 + random.random() * 2
                time.sleep(wait)
            else:
                raise


def init_db(db_path=None):
    """Initialize the finetune baseline results database."""
    db = db_path or DB_PATH
    conn = _db_connect(db)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        dataset_name TEXT,
        dataset_num_classes INTEGER,
        num_train_images INTEGER,
        model_variant TEXT,
        method TEXT DEFAULT 'full_finetune',
        num_epochs INTEGER,
        learning_rate REAL,
        batch_size INTEGER,
        grad_accum_steps INTEGER,
        train_time_seconds REAL,
        final_loss REAL,
        loss_history TEXT,
        mAP_50 REAL,
        mAP_50_95 REAL,
        status TEXT DEFAULT 'pending',
        error_message TEXT,
        device TEXT,
        current_epoch INTEGER,
        current_loss REAL,
        current_map REAL,
        best_epoch INTEGER,
        best_val_map REAL,
        notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT,
        eval_split TEXT,
        mAP_50 REAL,
        mAP_50_95 REAL,
        per_class_ap_json TEXT,
        conf_metrics_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS gpu_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        gpu_util_pct REAL,
        mem_used_mb REAL,
        mem_total_mb REAL,
        power_w REAL,
        temp_c REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        eval_image_stem TEXT NOT NULL,
        image_width INTEGER,
        image_height INTEGER,
        predictions_json TEXT NOT NULL,
        gt_json TEXT NOT NULL,
        class_names_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
    )""")
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_exp ON predictions(experiment_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_stem ON predictions(experiment_id, eval_image_stem)")
    except Exception:
        pass
    conn.commit()
    return conn


def experiment_done(conn, run_id):
    row = conn.execute(
        "SELECT status FROM experiments WHERE run_id=?", (run_id,)
    ).fetchone()
    return row is not None and row[0] == "completed"


def _gpu_monitor_loop(db_path, interval=5, stop_event=None):
    """Background thread: log GPU utilization to DB every `interval` seconds."""
    import subprocess as _sp
    conn = _db_connect(db_path)
    while not (stop_event and stop_event.is_set()):
        try:
            result = _sp.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
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


# ---------------------------------------------------------------------------
# Dataset discovery (mirrored from grid_search.py)
# ---------------------------------------------------------------------------

def _discover_stems(images_dir: Path) -> List[str]:
    if not images_dir.exists():
        return []
    stems = set()
    for f in images_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            stems.add(f.stem)
    return sorted(stems)


def discover_benchmark_datasets(root: Path) -> List[dict]:
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
            names = [names[k] for k in sorted(names.keys())]

        train_stems = _discover_stems(ds_dir / "train" / "images")
        test_stems = _discover_stems(ds_dir / "test" / "images")

        if not train_stems or not test_stems:
            logger.warning("Skipping %s: no train/test images", ds_dir.name)
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


# ---------------------------------------------------------------------------
# YOLO label parsing + evaluation (mirrored from grid_search.py)
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path: Path) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def load_image_and_labels_generic(dataset_path: Path, split: str, stem: str):
    img_dir = dataset_path / split / "images"
    lbl_path = dataset_path / split / "labels" / f"{stem}.txt"
    img_path = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    assert img_path is not None, f"Image not found for stem {stem} in {img_dir}"
    return Image.open(img_path).convert("RGB"), parse_yolo_label(lbl_path)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap_single_class(preds, gt_boxes, iou_threshold):
    """Compute AP for a single class at a given IoU threshold."""
    if not gt_boxes:
        return None  # No GT for this class
    if not preds:
        return 0.0

    matched = [False] * len(gt_boxes)
    tp_list = []
    fp_list = []

    for conf, box in preds:
        best_iou, best_idx = 0, -1
        for gi, gt_box in enumerate(gt_boxes):
            if matched[gi]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_threshold and best_idx >= 0:
            matched[best_idx] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum)

    # AP via all-points interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_map(predictions, gt_boxes, image_size, active_class_ids, iou_thresholds,
                class_names=None):
    """Compute mAP across classes and IoU thresholds.

    predictions: list of {class_id, confidence, box: [x1,y1,x2,y2]}
    gt_boxes: list of {class_id, cx, cy, w, h} (YOLO normalised)
    image_size: (w, h) tuple
    active_class_ids: set of class IDs present in training data
    iou_thresholds: list of IoU thresholds
    """
    w_img, h_img = image_size

    gt_by_class = {}
    for b in gt_boxes:
        cid = b["class_id"]
        if cid not in active_class_ids:
            continue
        cx, cy, bw, bh = b["cx"]*w_img, b["cy"]*h_img, b["w"]*w_img, b["h"]*h_img
        box = [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]
        gt_by_class.setdefault(cid, []).append(box)

    preds_by_class = {}
    for p in predictions:
        cid = p["class_id"]
        preds_by_class.setdefault(cid, []).append((p["confidence"], p["box"]))
    for cid in preds_by_class:
        preds_by_class[cid].sort(key=lambda x: x[0], reverse=True)

    per_class_ap = {}
    ap_by_iou = {t: [] for t in iou_thresholds}

    for cid in active_class_ids:
        cls_name = class_names[cid] if class_names else str(cid)
        cls_gt = gt_by_class.get(cid, [])
        cls_preds = preds_by_class.get(cid, [])
        per_class_ap[cls_name] = {}

        for iou_t in iou_thresholds:
            ap = compute_ap_single_class(cls_preds, cls_gt, iou_t)
            if ap is not None:
                per_class_ap[cls_name][str(iou_t)] = ap
                ap_by_iou[iou_t].append(ap)

    def mean_or_zero(lst):
        return float(np.mean(lst)) if lst else 0.0

    mAP_50 = mean_or_zero(ap_by_iou.get(0.5, []))
    coco_thresholds = [t for t in iou_thresholds if 0.5 <= t <= 0.95]
    mAP_50_95 = mean_or_zero([mean_or_zero(ap_by_iou[t]) for t in coco_thresholds]) if coco_thresholds else 0.0

    # Confidence metrics at IoU=0.5
    conf_metrics = {}
    for conf_t in [0.1, 0.3, 0.5]:
        filtered = [p for p in predictions if p["confidence"] >= conf_t]
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


def _compute_dataset_map(all_preds, all_gt_by_image, active_class_ids, class_names,
                         max_dets=500):
    """Compute COCO-style dataset-level mAP using pycocotools.

    Uses the same evaluation backend as rfdetr's training to ensure consistent
    metrics between training-time val mAP and our post-training test mAP.

    all_preds: list of {image_id, class_id, confidence, box: [x1,y1,x2,y2]}
    all_gt_by_image: list of {image_id, class_id, box: [x1,y1,x2,y2]}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build COCO GT dataset
    images = []
    annotations = []
    ann_id = 1
    image_ids = sorted(set(g["image_id"] for g in all_gt_by_image))
    for img_id in image_ids:
        images.append({"id": img_id})

    for g in all_gt_by_image:
        x1, y1, x2, y2 = g["box"]
        w, h = x2 - x1, y2 - y1
        annotations.append({
            "id": ann_id,
            "image_id": g["image_id"],
            "category_id": g["class_id"],
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        ann_id += 1

    categories = [{"id": cid} for cid in sorted(active_class_ids)]

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt.createIndex()

    # Build COCO detections
    coco_dets = []
    for p in all_preds:
        x1, y1, x2, y2 = p["box"]
        w, h = x2 - x1, y2 - y1
        coco_dets.append({
            "image_id": p["image_id"],
            "category_id": p["class_id"],
            "bbox": [x1, y1, w, h],
            "score": p["confidence"],
        })

    if not coco_dets:
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}

    coco_dt = coco_gt.loadRes(coco_dets)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # Use max_dets for ALL slots — pycocotools.summarize() hardcodes stats[0]
    # to use maxDets=100 which we don't want. Instead, we compute metrics
    # directly from the precision/recall arrays at our maxDets.
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Extract mAP directly from precision array: [TxRxKxAxM]
    # T=IoU thresholds, R=recall thresholds, K=categories, A=area ranges, M=maxDets
    precision = coco_eval.eval["precision"]  # shape [T, R, K, A, M]
    # area=all is index 0, maxDets index 2 (our max_dets)
    # mAP@50:95 = mean over all IoU thresholds (axis 0) and categories (axis 2)
    pr_all = precision[:, :, :, 0, 2]  # [T, R, K]
    pr_all[pr_all == -1] = 0
    mAP_50_95 = float(pr_all.mean()) if pr_all.size > 0 else 0.0

    # mAP@50 = IoU=0.5 is index 0
    pr_50 = precision[0, :, :, 0, 2]  # [R, K]
    pr_50[pr_50 == -1] = 0
    mAP_50 = float(pr_50.mean()) if pr_50.size > 0 else 0.0

    return {"mAP_50": mAP_50, "mAP_50_95": mAP_50_95}


# ---------------------------------------------------------------------------
# Progress callback for rfdetr training
# ---------------------------------------------------------------------------

from pytorch_lightning import Callback as _PLCallback

class ProgressDBCallback(_PLCallback):
    """PyTorch Lightning callback that writes training progress to DB.

    Hooks into the rfdetr PTL training loop to update current_epoch,
    current_loss, loss_history, and validation mAP in the DB for
    real-time dashboard monitoring.
    """

    def __init__(self, db_path, run_id, exp_id):
        self.db_path = db_path
        self.run_id = run_id
        self.exp_id = exp_id
        self.loss_history = []
        self._best_val_map = 0.0
        self._best_epoch = -1

    def _get_conn(self):
        """Open a fresh connection each time to avoid threading issues with PTL."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Get the latest training loss from logged metrics
        loss = trainer.callback_metrics.get("train/loss")
        loss_val = loss.item() if loss is not None else None
        if loss_val is not None:
            self.loss_history.append(loss_val)

        try:
            conn = self._get_conn()
            conn.execute(
                "UPDATE experiments SET current_epoch=?, current_loss=?, loss_history=? WHERE id=?",
                (epoch, loss_val, json.dumps(self.loss_history), self.exp_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("Failed to update progress in DB: %s", e)

    def on_validation_end(self, trainer, pl_module):
        # Get validation mAP@50:95 if available
        val_map = trainer.callback_metrics.get("val/mAP_50_95")
        ema_map = trainer.callback_metrics.get("val/ema_mAP_50_95")
        current_map = max(
            val_map.item() if val_map is not None else 0,
            ema_map.item() if ema_map is not None else 0,
        )
        if current_map > 0:
            # Only update best if this epoch is actually better
            if current_map > self._best_val_map:
                self._best_val_map = current_map
                self._best_epoch = trainer.current_epoch
            try:
                conn = self._get_conn()
                conn.execute(
                    "UPDATE experiments SET current_map=?, best_val_map=?, best_epoch=? WHERE id=?",
                    (current_map, self._best_val_map, self._best_epoch, self.exp_id),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.debug("Failed to update val mAP in DB: %s", e)


# ---------------------------------------------------------------------------
# Main training + evaluation logic
# ---------------------------------------------------------------------------

def make_run_id(variant_name, dataset_name):
    return f"ft_{variant_name}_{dataset_name}"


def run_finetune_and_eval(
    variant_name: str,
    model_cls,
    ds_info: dict,
    conn: sqlite3.Connection,
    db_path: Path = DB_PATH,
    output_root: Path = OUTPUT_ROOT,
    epochs: int = 100,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
):
    """Train one model variant on one dataset, then evaluate on test split."""
    ds_name = ds_info["name"]
    ds_path = ds_info["path"]
    run_id = make_run_id(variant_name, ds_name)

    # Skip if already completed
    if experiment_done(conn, run_id):
        logger.info("Skipping %s (already completed)", run_id)
        return

    logger.info("=" * 70)
    logger.info("Starting: %s", run_id)
    logger.info("  variant=%s  dataset=%s  classes=%d  train_images=%d",
                variant_name, ds_name, ds_info["num_classes"], len(ds_info["train_stems"]))
    logger.info("=" * 70)

    output_dir = str(output_root / variant_name / ds_name)
    os.makedirs(output_dir, exist_ok=True)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().isoformat()

    # Insert/update experiment row as 'running'
    _db_execute_with_retry(conn, """
        INSERT OR REPLACE INTO experiments
        (run_id, timestamp, dataset_name, dataset_num_classes, num_train_images,
         model_variant, method, num_epochs, learning_rate, batch_size,
         grad_accum_steps, status, device)
        VALUES (?, ?, ?, ?, ?, ?, 'full_finetune', ?, ?, ?, ?, 'running', ?)
    """, (
        run_id, timestamp, ds_name, ds_info["num_classes"],
        len(ds_info["train_stems"]), f"rfdetr-{variant_name}",
        epochs, lr, batch_size, grad_accum_steps, device_name,
    ))
    _db_commit_with_retry(conn)

    try:
        # ----- TRAIN -----
        t0 = time.time()
        model = model_cls()

        # Get experiment ID for progress callback
        exp_row = conn.execute(
            "SELECT id FROM experiments WHERE run_id=?", (run_id,)
        ).fetchone()
        exp_id = exp_row[0] if exp_row else None

        # Create progress callback
        progress_cb = ProgressDBCallback(db_path, run_id, exp_id)

        # Use rfdetr's training components directly so we can inject our callback
        from rfdetr.training import RFDETRDataModule, RFDETRModule, build_trainer
        config = model.get_train_config(
            dataset_dir=str(ds_path),
            dataset_file="yolo",
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=output_dir,
            progress_bar=True,
            tensorboard=True,
            project=f"rfdetr-finetune-{variant_name}",
            run=ds_name,
        )
        module = RFDETRModule(model.model_config, config)
        datamodule = RFDETRDataModule(model.model_config, config)
        trainer = build_trainer(config, model.model_config)

        # Inject our progress callback into PTL trainer
        trainer.callbacks.append(progress_cb)

        # Staleness-based watchdog: if no new epoch for FT_TIMEOUT_SECONDS, kill via SIGALRM
        _last_epoch_time = [time.time()]
        _orig_on_epoch_end = progress_cb.on_train_epoch_end
        def _patched_on_epoch_end(trainer_arg, pl_module_arg):
            _last_epoch_time[0] = time.time()
            return _orig_on_epoch_end(trainer_arg, pl_module_arg)
        progress_cb.on_train_epoch_end = _patched_on_epoch_end

        _watchdog_stop = threading.Event()
        _main_tid = threading.main_thread().ident
        def _ft_watchdog():
            while not _watchdog_stop.wait(15):
                stale = time.time() - _last_epoch_time[0]
                if stale > FT_TIMEOUT_SECONDS:
                    logger.warning("FT watchdog: no epoch progress for %.0fs, sending SIGALRM", stale)
                    signal.pthread_kill(_main_tid, signal.SIGALRM)
                    break

        ft_watchdog = threading.Thread(target=_ft_watchdog, daemon=True)
        ft_watchdog.start()
        old_handler = signal.getsignal(signal.SIGALRM)
        def _ft_alarm(signum, frame):
            raise TimeoutError(f"FT training stalled for >{FT_TIMEOUT_SECONDS}s with no epoch progress")
        signal.signal(signal.SIGALRM, _ft_alarm)

        try:
            trainer.fit(module, datamodule)
        finally:
            _watchdog_stop.set()
            signal.signal(signal.SIGALRM, old_handler)

        train_time = time.time() - t0
        logger.info("Training completed in %.1fs", train_time)

        # Load BEST checkpoint (not the last epoch weights) for evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_ckpt_path = Path(output_dir) / "checkpoint_best_total.pth"
        if best_ckpt_path.exists():
            logger.info("Loading best checkpoint from %s", best_ckpt_path)
            best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            model.model.model.load_state_dict(best_ckpt["model"], strict=True)
        else:
            logger.warning("No best checkpoint found, using last epoch weights")
            model.model.model = module.model
        model.model.model = model.model.model.to(device).eval()
        model.model.device = device

        # ----- EVALUATE on test split -----
        logger.info("Evaluating on %d test images...", len(ds_info["test_stems"]))
        class_names = ds_info["class_names"]
        active_class_ids = set(range(len(class_names)))
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

        all_preds = []
        all_gt = []
        per_image_results = []
        per_image_pred_data = []  # for predictions table

        for img_idx, stem in enumerate(ds_info["test_stems"]):
            image_id = img_idx + 1  # 1-indexed for COCO
            pil_img, gt_boxes = load_image_and_labels_generic(ds_path, "test", stem)
            img_w, img_h = pil_img.size

            # rfdetr predict returns sv.Detections with 0-indexed class_ids
            # (indices into the 91-class head; trained classes occupy slots 0..num_classes-1)
            detections = model.predict(pil_img, threshold=0.01)

            # Convert to our format: keep only predictions for the trained classes (0..nc-1)
            num_classes = len(class_names)
            preds = []
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    cid = int(detections.class_id[i])
                    if cid >= num_classes:
                        continue  # skip detections from untrained COCO head slots
                    preds.append({
                        "image_id": image_id,
                        "class_id": cid,
                        "confidence": float(detections.confidence[i]),
                        "box": detections.xyxy[i].tolist(),
                    })

            # Per-image: top 500 by confidence (match rfdetr eval_max_dets=500)
            preds.sort(key=lambda d: d["confidence"], reverse=True)
            preds = preds[:500]
            all_preds.extend(preds)

            # Convert GT from normalized YOLO to absolute xyxy for dataset-level eval
            gt_abs = []
            for b in gt_boxes:
                gt_abs.append({
                    "image_id": image_id,
                    "class_id": b["class_id"],
                    "box": [
                        (b["cx"] - b["w"]/2) * img_w,
                        (b["cy"] - b["h"]/2) * img_h,
                        (b["cx"] + b["w"]/2) * img_w,
                        (b["cy"] + b["h"]/2) * img_h,
                    ],
                })
            all_gt.extend(gt_abs)

            # Store per-image predictions for the predictions table
            per_image_pred_data.append({
                "stem": stem,
                "width": img_w, "height": img_h,
                "preds": [{"class_id": p["class_id"], "confidence": p["confidence"], "bbox": p["box"]} for p in preds],
                "gt": [{"class_id": g["class_id"], "bbox": g["box"]} for g in gt_abs],
            })

            # Per-image metrics (for eval_results table detail)
            img_metrics = compute_map(
                preds, gt_boxes, (img_w, img_h),
                active_class_ids, iou_thresholds, class_names,
            )
            per_image_results.append((stem, img_metrics))

        # Dataset-level mAP (COCO-style via pycocotools — matches rfdetr training eval)
        dataset_metrics = _compute_dataset_map(
            all_preds, all_gt, active_class_ids, class_names,
        )
        avg_mAP_50 = dataset_metrics["mAP_50"]
        avg_mAP_50_95 = dataset_metrics["mAP_50_95"]

        logger.info("Results for %s: mAP@50=%.4f  mAP@50:95=%.4f",
                     run_id, avg_mAP_50, avg_mAP_50_95)

        # Update DB with final results
        final_loss = progress_cb.loss_history[-1] if progress_cb.loss_history else None
        _db_execute_with_retry(conn, """
            UPDATE experiments SET
                train_time_seconds=?, mAP_50=?, mAP_50_95=?, status='completed',
                final_loss=?, loss_history=?
            WHERE run_id=?
        """, (train_time, avg_mAP_50, avg_mAP_50_95,
              final_loss, json.dumps(progress_cb.loss_history), run_id))
        _db_commit_with_retry(conn)

        # Write per-image eval results
        exp_row = conn.execute(
            "SELECT id FROM experiments WHERE run_id=?", (run_id,)
        ).fetchone()
        exp_id = exp_row[0] if exp_row else None

        if exp_id:
            for stem, metrics in per_image_results:
                _db_execute_with_retry(conn, """
                    INSERT INTO eval_results
                    (experiment_id, eval_image_stem, eval_split, mAP_50, mAP_50_95,
                     per_class_ap_json, conf_metrics_json)
                    VALUES (?, ?, 'test', ?, ?, ?, ?)
                """, (
                    exp_id, stem, metrics["mAP_50"], metrics["mAP_50_95"],
                    json.dumps(metrics["per_class_ap"]),
                    json.dumps(metrics["conf_metrics"]),
                ))
            _db_commit_with_retry(conn)

            # Write per-image predictions for visualization
            class_names_json = json.dumps(class_names)
            for pd in per_image_pred_data:
                _db_execute_with_retry(conn, """
                    INSERT INTO predictions
                    (experiment_id, eval_image_stem, image_width, image_height,
                     predictions_json, gt_json, class_names_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_id, pd["stem"], pd["width"], pd["height"],
                    json.dumps(pd["preds"]), json.dumps(pd["gt"]),
                    class_names_json,
                ))
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
    parser = argparse.ArgumentParser(
        description="RF-DETR full fine-tune baseline on RF20-VL-FSOD"
    )
    parser.add_argument("--datasets-root", type=str,
                        default=str(BENCHMARK_DATASETS_ROOT),
                        help="Path to RF20-VL-FSOD datasets")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Output database path")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT),
                        help="Checkpoint output root directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--include-medium", action="store_true",
                        help="Also train rfdetr-medium (slower, larger)")
    parser.add_argument("--include-2xlarge", action="store_true",
                        help="Also train rfdetr-2xlarge (much larger, ~880 resolution)")
    parser.add_argument("--include-lora-reeval", action="store_true",
                        help="Also re-evaluate best LoRA config per dataset (corrected mAP)")
    parser.add_argument("--lora-source-db", type=str,
                        default=str(SCRIPT_DIR / "results_phase3b.db"),
                        help="Source DB with LoRA results to re-evaluate")
    parser.add_argument("--only", type=str, default=None,
                        help="Only run this dataset name (for testing)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 1 dataset, 5 epochs")
    args = parser.parse_args()

    # Lazy import rfdetr (may not be installed in all envs)
    from rfdetr import RFDETRNano, RFDETRMedium
    from rfdetr import RFDETR2XLarge

    output_root = Path(args.output_root)

    datasets_root = Path(args.datasets_root)
    db_path = Path(args.db)

    # Discover datasets
    datasets = discover_benchmark_datasets(datasets_root)
    if not datasets:
        logger.error("No datasets found in %s", datasets_root)
        sys.exit(1)

    # Apply filters
    if args.only:
        datasets = [d for d in datasets if d["name"] == args.only]
        if not datasets:
            logger.error("Dataset '%s' not found", args.only)
            sys.exit(1)

    if args.smoke:
        datasets = datasets[:1]
        args.epochs = 5
        logger.info("SMOKE TEST: 1 dataset, %d epochs", args.epochs)

    # Build variant list
    variants = [("nano", RFDETRNano)]
    if args.include_medium:
        variants.append(("medium", RFDETRMedium))
    if args.include_2xlarge:
        variants.append(("2xlarge", RFDETR2XLarge))

    # Load LoRA best configs if re-eval requested
    lora_configs_by_ds = {}
    if args.include_lora_reeval:
        lora_source = Path(args.lora_source_db)
        if lora_source.exists():
            import sqlite3 as _sql
            _lconn = _sql.connect(str(lora_source))
            _lconn.row_factory = _sql.Row
            _lora_rows = _lconn.execute("""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY dataset_name ORDER BY mAP_50_95 DESC
                    ) as rn
                    FROM experiments WHERE status='completed' AND num_epochs <= 100
                )
                SELECT * FROM ranked WHERE rn = 1
            """).fetchall()
            for r in _lora_rows:
                lora_configs_by_ds[r["dataset_name"]] = dict(r)
            _lconn.close()
            logger.info("Loaded %d best LoRA configs for re-evaluation", len(lora_configs_by_ds))
        else:
            logger.warning("LoRA source DB not found: %s", lora_source)

    n_lora = len(lora_configs_by_ds) if args.include_lora_reeval else 0
    lora_variants = ["rfdetr-medium", "rfdetr-nano"]
    if args.include_2xlarge:
        lora_variants.append("rfdetr-2xlarge")
    n_lora_variants = len(lora_variants)
    total_runs = len(datasets) * len(variants) + min(n_lora, len(datasets)) * n_lora_variants
    logger.info("Will run %d experiments (%d datasets x %d FT variants + %d LoRA re-evals x %d model variants)",
                total_runs, len(datasets), len(variants),
                min(n_lora, len(datasets)), n_lora_variants)

    # Init DB
    conn = init_db(db_path)
    _db_execute_with_retry(conn,
        "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('start_time', ?)",
        (datetime.now().isoformat(),))
    _db_execute_with_retry(conn,
        "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('total_experiments', ?)",
        (str(total_runs),))
    _db_commit_with_retry(conn)

    # Start GPU monitor thread
    gpu_stop = threading.Event()
    gpu_thread = threading.Thread(
        target=_gpu_monitor_loop,
        args=(db_path, 5, gpu_stop),
        daemon=True,
    )
    gpu_thread.start()
    logger.info("GPU monitor started")

    # Run all approaches per dataset before moving to the next dataset
    completed = 0
    import subprocess
    reeval_script = str(SCRIPT_DIR / "reeval_lora_best.py")
    inference_python = os.path.expanduser("~/.pyenv/versions/inference-exp/bin/python")
    # lora_variants already set above

    for ds_info in datasets:
        ds_name = ds_info["name"]
        logger.info("=" * 70)
        logger.info("DATASET: %s  (%d classes, %d train images)",
                     ds_name, ds_info["num_classes"], len(ds_info["train_stems"]))
        logger.info("=" * 70)

        # 1) LoRA re-evals for this dataset
        if args.include_lora_reeval and ds_name in lora_configs_by_ds:
            for lora_model in lora_variants:
                short = lora_model.replace("rfdetr-", "")
                logger.info("  [LoRA %s] %s...", short, ds_name)
                reeval_cmd = [
                    inference_python, reeval_script,
                    "--only", ds_name,
                    "--source-db", args.lora_source_db,
                    "--target-db", str(db_path),
                    "--datasets-root", str(datasets_root),
                    "--model-variant", lora_model,
                    "--trials", "3",
                    "--trial-timeout", "900",
                    "--lora-epochs", "100",
                ]
                # 2XL needs smaller batch to fit in GPU memory
                if "2xlarge" in lora_model:
                    reeval_cmd.extend(["--max-batch-size", "2"])
                result = subprocess.run(reeval_cmd, capture_output=True, text=True, timeout=7200)
                if result.returncode == 0:
                    logger.info("  [LoRA %s] completed for %s", short, ds_name)
                else:
                    logger.error("  [LoRA %s] failed for %s: %s", short, ds_name,
                                 result.stderr[-500:] if result.stderr else "unknown error")
                completed += 1
                logger.info("Progress: %d / %d experiments", completed, total_runs)
                torch.cuda.empty_cache()

        # 2) Full fine-tune (nano + medium) for this dataset
        for variant_name, model_cls in variants:
            # 2XL needs smaller batch size to fit in GPU memory
            if variant_name == "2xlarge":
                ft_bs = 1
                ft_grad_accum = 16  # effective batch = 1 * 16 = 16
            else:
                ft_bs = args.batch_size
                ft_grad_accum = args.grad_accum_steps

            run_finetune_and_eval(
                variant_name=variant_name,
                model_cls=model_cls,
                ds_info=ds_info,
                conn=conn,
                db_path=db_path,
                output_root=output_root,
                epochs=args.epochs,
                batch_size=ft_bs,
                grad_accum_steps=ft_grad_accum,
                lr=args.lr,
            )
            completed += 1
            logger.info("Progress: %d / %d experiments", completed, total_runs)
            torch.cuda.empty_cache()

    # Stop GPU monitor
    gpu_stop.set()

    # Final metadata
    _db_execute_with_retry(conn,
        "INSERT OR REPLACE INTO grid_meta (key, value) VALUES ('end_time', ?)",
        (datetime.now().isoformat(),))
    _db_commit_with_retry(conn)
    conn.close()

    logger.info("All done! Results in %s", db_path)


if __name__ == "__main__":
    main()

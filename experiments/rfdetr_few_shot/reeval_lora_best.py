#!/usr/bin/env python3
"""Re-evaluate best LoRA configs with correct dataset-level COCO-style mAP.

Reads the best config per dataset from results_phase3b.db, re-trains each one
(using the same hyperparameters) N_TRIALS times (best-of-N for stability),
then evaluates with pooled dataset-level mAP instead of the incorrect
per-image-averaged mAP.

Includes a per-trial timeout watchdog to recover from hangs.

Writes corrected results into the same DB format so the dashboard can use them.

Usage:
    python reeval_lora_best.py
    python reeval_lora_best.py --smoke   # 1 dataset only
    python reeval_lora_best.py --trials 3  # best of 3
"""

import argparse
import copy
import json
import logging
import multiprocessing
import os
import random
import signal
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

# Add parent paths so we can import from grid_search
sys.path.insert(0, str(Path(__file__).parent))

from grid_search import (
    discover_benchmark_datasets,
    load_image_and_labels_generic,
    _discover_stems,
    run_inference,
    compute_map,
    compute_iou,
    compute_ap_single_class,
    train_lora_fast,
    load_base_model,
    InlineFewShotDataset,
    _db_connect,
    _db_execute_with_retry,
    _db_commit_with_retry,
    MAP_IOU_THRESHOLDS,
    BENCHMARK_DATASETS_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SOURCE_DB = SCRIPT_DIR / "results_phase3b.db"
TARGET_DB = SCRIPT_DIR / "results_lora_corrected.db"


def _compute_dataset_map(all_preds, all_gt, active_class_ids, class_names,
                         max_dets=500):
    """Compute COCO-style dataset-level mAP using pycocotools.

    Uses the same evaluation backend as rfdetr's training to ensure consistent
    metrics between training-time val mAP and our post-training test mAP.

    all_preds: list of {image_id, class_id, confidence, box: [x1,y1,x2,y2]}
    all_gt: list of {image_id, class_id, box: [x1,y1,x2,y2]}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Build COCO GT dataset
    image_ids = sorted(set(g["image_id"] for g in all_gt))
    images = [{"id": img_id} for img_id in image_ids]
    annotations = []
    ann_id = 1
    for g in all_gt:
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
    precision = coco_eval.eval["precision"]
    # area=all is index 0, maxDets index 2 (our max_dets)
    pr_all = precision[:, :, :, 0, 2]  # [T, R, K]
    pr_all[pr_all == -1] = 0
    mAP_50_95 = float(pr_all.mean()) if pr_all.size > 0 else 0.0

    # mAP@50 = IoU=0.5 is index 0
    pr_50 = precision[0, :, :, 0, 2]  # [R, K]
    pr_50[pr_50 == -1] = 0
    mAP_50 = float(pr_50.mean()) if pr_50.size > 0 else 0.0

    return {"mAP_50": mAP_50, "mAP_50_95": mAP_50_95}


def init_target_db(db_path):
    conn = _db_connect(db_path)
    # If the table already exists (e.g. created by finetune_baseline.py),
    # just add any missing columns. Otherwise create it fresh.
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
            model_variant TEXT DEFAULT 'rfdetr-medium',
            method TEXT DEFAULT 'lora',
            lora_rank INTEGER, lora_alpha INTEGER,
            num_epochs INTEGER, learning_rate REAL,
            alpha_ratio INTEGER,
            batch_size INTEGER,
            lora_targets TEXT,
            copy_paste INTEGER,
            mosaic INTEGER,
            warmup INTEGER,
            multi_scale INTEGER,
            lora_dropout REAL DEFAULT 0.0,
            weight_decay REAL DEFAULT 0.0001,
            train_time_seconds REAL,
            final_loss REAL, loss_history TEXT,
            mAP_50 REAL, mAP_50_95 REAL,
            mAP_50_old REAL, mAP_50_95_old REAL,
            status TEXT DEFAULT 'pending',
            error_message TEXT, device TEXT,
            current_epoch INTEGER, current_loss REAL,
            current_map REAL, best_epoch INTEGER, best_val_map REAL,
            notes TEXT
        )""")
    else:
        # Add missing columns to existing table
        for col, coltype in [
            ("lora_rank", "INTEGER"), ("lora_alpha", "INTEGER"),
            ("alpha_ratio", "INTEGER"), ("lora_targets", "TEXT"),
            ("copy_paste", "INTEGER"), ("mosaic", "INTEGER"),
            ("warmup", "INTEGER"), ("multi_scale", "INTEGER"),
            ("lora_dropout", "REAL"), ("weight_decay", "REAL"),
            ("mAP_50_old", "REAL"), ("mAP_50_95_old", "REAL"),
            ("method", "TEXT"), ("final_loss", "REAL"), ("loss_history", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists
    conn.execute("""CREATE TABLE IF NOT EXISTS grid_meta (
        key TEXT PRIMARY KEY, value TEXT
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


def get_best_configs(source_db):
    """Get the best LoRA config per dataset from phase3b."""
    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY dataset_name ORDER BY mAP_50_95 DESC
            ) as rn
            FROM experiments WHERE status='completed' AND num_epochs <= 100
        )
        SELECT * FROM ranked WHERE rn = 1
        ORDER BY dataset_name
    """).fetchall()
    configs = [dict(r) for r in rows]
    conn.close()
    return configs


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate best LoRA configs with correct mAP")
    parser.add_argument("--source-db", default=str(SOURCE_DB))
    parser.add_argument("--target-db", default=str(TARGET_DB))
    parser.add_argument("--datasets-root", default=str(BENCHMARK_DATASETS_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--model-variant", default="rfdetr-medium",
                        help="Model variant: rfdetr-medium or rfdetr-nano")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of trials per dataset (best-of-N for stability)")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help="Cap batch size (for large models that OOM)")
    parser.add_argument("--trial-timeout", type=int, default=600,
                        help="Timeout in seconds per trial (default: 600 = 10 min)")
    parser.add_argument("--lora-epochs", type=int, default=100,
                        help="Override num_epochs for LoRA training (default: 100 for fair comparison with FT)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover datasets
    datasets_root = Path(args.datasets_root)
    datasets = discover_benchmark_datasets(datasets_root)
    ds_by_name = {d["name"]: d for d in datasets}

    # Get best configs
    best_configs = get_best_configs(Path(args.source_db))
    if args.only:
        best_configs = [c for c in best_configs if c["dataset_name"] == args.only]
    if args.smoke:
        best_configs = best_configs[:1]

    logger.info("Re-evaluating %d best LoRA configs", len(best_configs))

    # Init target DB (don't overwrite grid_meta if DB already exists — parent process owns it)
    target_conn = init_target_db(Path(args.target_db))

    # Load model
    model_variant = args.model_variant
    logger.info("Loading %s model...", model_variant)

    from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import CONFIG_FOR_MODEL_TYPE
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model as _build_model
    from inference_models.models.rfdetr.few_shot.criterion import SetCriterion
    from inference_models.models.rfdetr.few_shot.matcher import HungarianMatcher

    # Map variant name to weights filename
    _weights_map = {
        "rfdetr-nano": "rf-detr-nano.pth",
        "rfdetr-medium": "rf-detr-medium.pth",
        "rfdetr-2xlarge": "rf-detr-xxlarge.pth",
    }
    weights_name = _weights_map.get(model_variant,
                                     f"rf-detr-{model_variant.replace('rfdetr-', '')}.pth")
    weights_path = Path(__file__).parent / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            f"Download first or copy the file there."
        )

    config = CONFIG_FOR_MODEL_TYPE[model_variant](device=device)
    weights = torch.load(str(weights_path), map_location=device, weights_only=False)["model"]
    config.num_classes = weights["class_embed.bias"].shape[0] - 1
    template_model = _build_model(config=config)
    template_model.load_state_dict(weights)
    template_model = template_model.eval().to(device)
    filtered_state = {k: v for k, v in weights.items()
                      if k in template_model.state_dict()
                      and v.shape == template_model.state_dict()[k].shape}

    n_trials = args.trials
    trial_timeout = args.trial_timeout

    for i, cfg in enumerate(best_configs):
        ds_name = cfg["dataset_name"]
        variant_short = model_variant.replace("rfdetr-", "")
        variant_suffix = f"_{variant_short}" if variant_short != "medium" else ""
        run_id = f"reeval{variant_suffix}_{ds_name}"

        # Skip if done
        existing = target_conn.execute(
            "SELECT status FROM experiments WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing and existing[0] == "completed":
            logger.info("Skipping %s (already completed)", run_id)
            continue

        ds_info = ds_by_name.get(ds_name)
        if not ds_info:
            logger.warning("Dataset %s not found, skipping", ds_name)
            continue

        logger.info("=" * 60)
        logger.info("[%d/%d] Re-evaluating %s (%d trials, best-of-%d)",
                     i+1, len(best_configs), ds_name, n_trials, n_trials)
        actual_epochs = args.lora_epochs
        logger.info("  r=%s bs=%s ep=%s (override→%d) targets=%s cp=%s mo=%s",
                     cfg["lora_rank"], cfg["batch_size"], cfg["num_epochs"],
                     actual_epochs,
                     cfg.get("lora_targets", "v1"), cfg.get("copy_paste", 0),
                     cfg.get("mosaic", 0))

        # Insert running row
        _db_execute_with_retry(target_conn, """
            INSERT OR REPLACE INTO experiments
            (run_id, timestamp, dataset_name, dataset_num_classes, num_train_images,
             model_variant, method, lora_rank, lora_alpha, num_epochs, learning_rate,
             alpha_ratio, batch_size, lora_targets, copy_paste, mosaic,
             warmup, multi_scale, lora_dropout, weight_decay,
             mAP_50_old, mAP_50_95_old, status, device)
            VALUES (?,?,?,?,?, ?,'lora',?,?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?,'running',?)
        """, (
            run_id, datetime.now().isoformat(), ds_name,
            ds_info["num_classes"], len(ds_info["train_stems"]),
            model_variant,
            cfg["lora_rank"], cfg.get("lora_alpha", cfg["lora_rank"] * cfg.get("alpha_ratio", 1)),
            actual_epochs, cfg["learning_rate"],
            cfg.get("alpha_ratio", 1), cfg["batch_size"],
            cfg.get("lora_targets", "v1"),
            cfg.get("copy_paste", 0), cfg.get("mosaic", 0),
            cfg.get("warmup", 0), cfg.get("multi_scale", 0),
            cfg.get("lora_dropout", 0.0), cfg.get("weight_decay", 0.0001),
            cfg["mAP_50"], cfg["mAP_50_95"],
            str(device),
        ))
        _db_commit_with_retry(target_conn)

        try:
            ds_path = ds_info["path"]
            ds_class_names = ds_info["class_names"]
            num_classes = ds_info["num_classes"]
            active_classes = ds_class_names[:num_classes]
            train_stems = ds_info["train_stems"]
            test_stems = ds_info["test_stems"]

            # Load training data once (shared across trials)
            images_and_boxes = []
            for stem in train_stems:
                img, boxes = load_image_and_labels_generic(ds_path, "train", stem)
                images_and_boxes.append((img, boxes))

            # Load validation data once (for best-checkpoint selection during training)
            val_stems = _discover_stems(ds_path / "valid" / "images")
            val_data = []
            for vs in val_stems:
                vimg, vboxes = load_image_and_labels_generic(ds_path, "valid", vs)
                val_data.append((vs, vimg, vboxes))
            logger.info("  Loaded %d val images for intermediate eval", len(val_data))

            # Load test data once (for final evaluation only)
            test_data = []
            for eval_stem in test_stems:
                eval_img, eval_boxes = load_image_and_labels_generic(ds_path, "test", eval_stem)
                test_data.append((eval_stem, eval_img, eval_boxes))

            # Build shared objects
            fresh_config = config.model_copy(update={"num_classes": num_classes})
            matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)
            weight_dict = {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}
            for j in range(fresh_config.dec_layers - 1):
                weight_dict.update({f"{k}_{j}": v for k, v in
                                    {"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}.items()})
            criterion = SetCriterion(
                num_classes=num_classes + 1, matcher=matcher, weight_dict=weight_dict,
                focal_alpha=0.25, losses=["labels", "boxes", "cardinality"],
                group_detr=getattr(config, "group_detr", 1),
                ia_bce_loss=getattr(config, "ia_bce_loss", True),
            ).to(device)

            rank = cfg["lora_rank"]
            alpha = cfg.get("lora_alpha", rank * cfg.get("alpha_ratio", 1))
            epochs = args.lora_epochs  # override to match FT epoch count for fair comparison

            # Get experiment ID for progress updates
            exp_row = target_conn.execute(
                "SELECT id FROM experiments WHERE run_id=?", (run_id,)
            ).fetchone()
            exp_id = exp_row[0] if exp_row else None

            # Staleness-based watchdog: kills via SIGALRM if no progress for trial_timeout seconds
            _last_progress = [time.time()]  # mutable so callback can update it

            # Progress callback: writes epoch/loss to DB AND resets watchdog
            _db_path = Path(args.target_db)
            def _progress_cb(epoch, total, loss, losses, b_epoch, b_map):
                _last_progress[0] = time.time()  # reset staleness timer
                try:
                    pconn = sqlite3.connect(str(_db_path), timeout=30)
                    pconn.execute("PRAGMA journal_mode=WAL")
                    pconn.execute(
                        "UPDATE experiments SET current_epoch=?, current_loss=?, "
                        "loss_history=?, best_epoch=?, best_val_map=?, current_map=? "
                        "WHERE id=?",
                        (epoch, loss, json.dumps(losses),
                         b_epoch if b_map > 0 else None,
                         b_map if b_map > 0 else None,
                         b_map if b_map > 0 else None,
                         exp_id),
                    )
                    pconn.commit()
                    pconn.close()
                except Exception:
                    pass

            active_to_original = {i: ds_class_names.index(c) for i, c in enumerate(active_classes)}
            all_class_ids = set(range(num_classes))

            # ─── Best-of-N trials with staleness watchdog ───
            best_metrics = None
            best_train_time = None
            best_loss_history = None
            best_best_epoch = None
            best_best_val_map = None
            best_pred_data = None
            all_trial_maps = []  # track all trial results for spread display

            for trial in range(n_trials):
                logger.info("  Trial %d/%d for %s...", trial+1, n_trials, ds_name)

                # Start watchdog thread that checks for staleness
                _last_progress[0] = time.time()
                _watchdog_stop = threading.Event()
                _main_thread = threading.main_thread()

                def _watchdog_fn():
                    while not _watchdog_stop.wait(10):  # check every 10s
                        stale = time.time() - _last_progress[0]
                        if stale > trial_timeout:
                            logger.warning("    Watchdog: no progress for %.0fs, sending SIGALRM", stale)
                            signal.pthread_kill(_main_thread.ident, signal.SIGALRM)
                            break

                watchdog = threading.Thread(target=_watchdog_fn, daemon=True)
                watchdog.start()

                old_handler = signal.getsignal(signal.SIGALRM)
                def _alarm_handler(signum, frame):
                    raise TimeoutError(f"Trial {trial+1} stalled for >{trial_timeout}s with no epoch progress")
                signal.signal(signal.SIGALRM, _alarm_handler)

                try:
                    dataset = InlineFewShotDataset(
                        images_and_boxes, active_classes, config.resolution,
                        augment=True, augmentation_level=1,
                        class_names=ds_class_names,
                    )

                    # Validation callback: eval on val set every val_freq epochs
                    # (test set used only for final measurement)
                    def _val_callback(peft_model_eval, epoch):
                        _last_progress[0] = time.time()
                        val_preds = []
                        val_gt = []
                        for vi, (vs, vimg, vboxes) in enumerate(val_data):
                            vid = vi + 1
                            vw, vh = vimg.size
                            raw = run_inference(peft_model_eval, config, vimg, confidence_threshold=0.01)
                            for d in raw:
                                cid_orig = active_to_original.get(d["class_id"], -1)
                                if cid_orig < 0:
                                    continue
                                val_preds.append({
                                    "image_id": vid, "class_id": cid_orig,
                                    "confidence": d["confidence"], "box": d["box"],
                                })
                            for b in vboxes:
                                if b["class_id"] not in all_class_ids:
                                    continue
                                val_gt.append({
                                    "image_id": vid, "class_id": b["class_id"],
                                    "box": [
                                        (b["cx"] - b["w"]/2) * vw, (b["cy"] - b["h"]/2) * vh,
                                        (b["cx"] + b["w"]/2) * vw, (b["cy"] + b["h"]/2) * vh,
                                    ],
                                })
                        if val_gt and val_preds:
                            m = _compute_dataset_map(val_preds, val_gt, all_class_ids, ds_class_names)
                            _last_progress[0] = time.time()
                            return m["mAP_50_95"]
                        return 0.0

                    t0 = time.time()
                    merged_model, loss_history, _, b_epoch, b_map, _ = train_lora_fast(
                        template_model, filtered_state, fresh_config,
                        dataset, num_classes, device,
                        criterion, weight_dict,
                        rank=rank, alpha=alpha,
                        lr=cfg["learning_rate"], num_epochs=epochs,
                        lora_dropout=cfg.get("lora_dropout", 0.0),
                        weight_decay=cfg.get("weight_decay", 0.0001),
                        class_names=ds_class_names,
                        batch_size=min(cfg["batch_size"], args.max_batch_size) if args.max_batch_size else cfg["batch_size"],
                        lora_targets_version=cfg.get("lora_targets", "v1"),
                        copy_paste=bool(cfg.get("copy_paste", 0)),
                        mosaic=bool(cfg.get("mosaic", 0)),
                        warmup=bool(cfg.get("warmup", 0)),
                        multi_scale=bool(cfg.get("multi_scale", 0)),
                        progress_callback=_progress_cb,
                        val_callback=_val_callback,
                        val_freq=25,  # validate at epochs 25, 50, 75, 100
                    )
                    train_time = time.time() - t0
                    _last_progress[0] = time.time()  # reset for eval phase

                    # Evaluate with CORRECT dataset-level mAP (pycocotools)
                    all_preds = []
                    all_gt = []
                    for img_idx, (eval_stem, eval_img, eval_boxes) in enumerate(test_data):
                        image_id = img_idx + 1  # 1-indexed for COCO
                        _last_progress[0] = time.time()  # reset per-image during eval
                        img_w, img_h = eval_img.size
                        raw_dets = run_inference(merged_model, config, eval_img, confidence_threshold=0.01)
                        remapped = [{"image_id": image_id, **d, "class_id": active_to_original.get(d["class_id"], -1)}
                                    for d in raw_dets if d["class_id"] in active_to_original]
                        # Match rfdetr eval_max_dets=500 per image
                        remapped.sort(key=lambda d: d["confidence"], reverse=True)
                        all_preds.extend(remapped[:500])
                        for b in eval_boxes:
                            if b["class_id"] not in all_class_ids:
                                continue
                            all_gt.append({
                                "image_id": image_id,
                                "class_id": b["class_id"],
                                "box": [
                                    (b["cx"] - b["w"]/2) * img_w,
                                    (b["cy"] - b["h"]/2) * img_h,
                                    (b["cx"] + b["w"]/2) * img_w,
                                    (b["cy"] + b["h"]/2) * img_h,
                                ],
                            })

                    metrics = _compute_dataset_map(all_preds, all_gt, all_class_ids, ds_class_names)
                    logger.info("    Trial %d: mAP@50:95=%.1f%%  (%.1fs)",
                                 trial+1, metrics["mAP_50_95"] * 100, train_time)
                    all_trial_maps.append(metrics["mAP_50_95"])

                    # Save per-image predictions for this trial
                    trial_pred_data = []
                    for img_idx2, (es, ei, eb) in enumerate(test_data):
                        iw, ih = ei.size
                        # Collect preds for this image from all_preds
                        img_preds = [p for p in all_preds if p["image_id"] == img_idx2 + 1]
                        img_gt = [g for g in all_gt if g["image_id"] == img_idx2 + 1]
                        trial_pred_data.append({
                            "stem": es, "width": iw, "height": ih,
                            "preds": [{"class_id": p["class_id"], "confidence": p["confidence"], "bbox": p["box"]} for p in img_preds],
                            "gt": [{"class_id": g["class_id"], "bbox": g["box"]} for g in img_gt],
                        })

                    # Keep best trial
                    if best_metrics is None or metrics["mAP_50_95"] > best_metrics["mAP_50_95"]:
                        best_metrics = metrics
                        best_train_time = train_time
                        best_loss_history = loss_history
                        best_best_epoch = b_epoch
                        best_best_val_map = b_map
                        best_pred_data = trial_pred_data
                        # Save best merged model for live inference viewer
                        save_dir = Path(__file__).parent / "lora_merged" / variant_short / ds_name
                        save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save({"model": merged_model.state_dict()},
                                   save_dir / "merged_best.pth")

                    del merged_model
                    torch.cuda.empty_cache()

                except TimeoutError as te:
                    logger.warning("    Trial %d TIMED OUT: %s", trial+1, te)
                    torch.cuda.empty_cache()
                    all_trial_maps.append(None)
                    continue
                except Exception as te:
                    logger.warning("    Trial %d FAILED: %s", trial+1, te)
                    torch.cuda.empty_cache()
                    all_trial_maps.append(None)
                    continue
                finally:
                    _watchdog_stop.set()
                    signal.signal(signal.SIGALRM, old_handler)

            if best_metrics is None:
                raise RuntimeError(f"All {n_trials} trials failed for {ds_name}")

            # Format trial spread info
            valid_maps = [m for m in all_trial_maps if m is not None]
            spread_str = ", ".join(f"{m*100:.1f}%" for m in all_trial_maps if m is not None)
            spread_min = min(valid_maps) if valid_maps else 0
            spread_max = max(valid_maps) if valid_maps else 0
            logger.info("  BEST of %d trials: mAP@50:95=%.1f%%  spread=[%s]  (was %.1f%% uncorrected)",
                         n_trials, best_metrics["mAP_50_95"] * 100, spread_str, cfg["mAP_50_95"] * 100)

            # Build notes with trial details for dashboard
            trial_details = json.dumps({
                "n_trials": n_trials,
                "trial_maps": all_trial_maps,
                "spread_min": spread_min,
                "spread_max": spread_max,
            })

            # Update DB with best trial
            _db_execute_with_retry(target_conn, """
                UPDATE experiments SET
                    train_time_seconds=?, final_loss=?, loss_history=?,
                    mAP_50=?, mAP_50_95=?, status='completed',
                    best_epoch=?, best_val_map=?,
                    notes=?
                WHERE run_id=?
            """, (
                best_train_time,
                best_loss_history[-1] if best_loss_history else None,
                json.dumps(best_loss_history),
                best_metrics["mAP_50"], best_metrics["mAP_50_95"],
                best_best_epoch, best_best_val_map,
                trial_details,
                run_id,
            ))
            _db_commit_with_retry(target_conn)

            # Write per-image predictions from best trial
            exp_row = target_conn.execute(
                "SELECT id FROM experiments WHERE run_id=?", (run_id,)
            ).fetchone()
            if exp_row and best_pred_data:
                exp_id = exp_row[0]
                # Clear any previous predictions for this experiment
                _db_execute_with_retry(target_conn,
                    "DELETE FROM predictions WHERE experiment_id=?", (exp_id,))
                class_names_json = json.dumps(ds_class_names)
                for pd in best_pred_data:
                    _db_execute_with_retry(target_conn, """
                        INSERT INTO predictions
                        (experiment_id, eval_image_stem, image_width, image_height,
                         predictions_json, gt_json, class_names_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exp_id, pd["stem"], pd["width"], pd["height"],
                        json.dumps(pd["preds"]), json.dumps(pd["gt"]),
                        class_names_json,
                    ))
                _db_commit_with_retry(target_conn)

        except Exception as e:
            logger.error("FAILED: %s — %s", run_id, e, exc_info=True)
            _db_execute_with_retry(target_conn, """
                UPDATE experiments SET status='failed', error_message=? WHERE run_id=?
            """, (str(e)[:500], run_id))
            _db_commit_with_retry(target_conn)

        torch.cuda.empty_cache()

    target_conn.close()
    logger.info("Done! Results in %s", args.target_db)


if __name__ == "__main__":
    main()

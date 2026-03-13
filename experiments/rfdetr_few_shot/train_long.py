#!/usr/bin/env python3
"""Train specific configs for extended epochs (1000).

Usage:
    python train_long.py --db results_phase3b.db
"""

import argparse
import json
import logging
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_models"))

from grid_search import (
    BENCHMARK_DATASETS_ROOT,
    InlineFewShotDataset,
    _db_commit_with_retry,
    _db_connect,
    _db_execute_with_retry,
    _gpu_monitor_loop,
    discover_benchmark_datasets,
    load_base_model,
    load_image_and_labels_generic,
    prepare_criterion,
    run_single_experiment_phase3,
    make_run_id_phase3,
    experiment_done,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "train_long.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configs to train (identified by vizConfigId hash)
LONG_CONFIGS = [
    {
        "name": "U1CFF",
        "rank": 12, "batch_size": 8,
        "copy_paste": True, "mosaic": True,
    },
    {
        "name": "TBEBO",
        "rank": 4, "batch_size": 16,
        "copy_paste": True, "mosaic": True,
    },
]

FIXED = {
    "learning_rate": 2e-3,
    "alpha_ratio": 1,
    "weight_decay": 1e-3,
    "lora_targets": "v1",
    "warmup": False,
    "multi_scale": False,
}

EPOCHS = 1000


def make_run_id_long(dataset_name, rank, batch_size, copy_paste, mosaic,
                     lora_targets, warmup, multi_scale, alpha_ratio, weight_decay, epochs):
    """Extended run ID that includes epochs to avoid collisions with 50-epoch runs."""
    base = make_run_id_phase3(dataset_name, rank, batch_size, copy_paste, mosaic,
                               lora_targets, warmup, multi_scale,
                               alpha_ratio=alpha_ratio, weight_decay=weight_decay)
    return f"{base}_ep{epochs}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="results_phase3b.db")
    parser.add_argument("--datasets-root", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db_path = Path(__file__).parent / args.db
    datasets_root = Path(args.datasets_root) if args.datasets_root else BENCHMARK_DATASETS_ROOT

    datasets = discover_benchmark_datasets(datasets_root)
    if not datasets:
        logger.error("No datasets found in %s", datasets_root)
        return

    logger.info("📋 DB: %s", db_path)
    logger.info("📊 %d datasets × %d configs × %d epochs", len(datasets), len(LONG_CONFIGS), EPOCHS)

    # Monkey-patch make_run_id_phase3 to include epochs
    import grid_search
    _orig_make_run_id = grid_search.make_run_id_phase3

    def _patched_make_run_id(dataset_name, rank, batch_size, copy_paste, mosaic,
                              lora_targets, warmup, multi_scale,
                              alpha_ratio=None, weight_decay=None):
        base = _orig_make_run_id(dataset_name, rank, batch_size, copy_paste, mosaic,
                                  lora_targets, warmup, multi_scale,
                                  alpha_ratio=alpha_ratio, weight_decay=weight_decay)
        return f"{base}_ep{EPOCHS}"

    grid_search.make_run_id_phase3 = _patched_make_run_id

    # Start GPU monitor
    gpu_stop = None
    if device.type == "cuda":
        gpu_stop = threading.Event()
        gpu_thread = threading.Thread(target=_gpu_monitor_loop,
                                       args=(db_path, 5, gpu_stop), daemon=True)
        gpu_thread.start()

    # Load base model
    logger.info("🔄 Loading base model...")
    base_model, config = load_base_model(device)
    logger.info("✅ Base model loaded")

    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model

    done = 0
    total = len(LONG_CONFIGS) * len(datasets)

    for ds_info in datasets:
        ds_name = ds_info["name"]
        num_classes = ds_info["num_classes"]
        logger.info("━━━ Dataset: %s (%d classes, %d train, %d test) ━━━",
                     ds_name, num_classes, len(ds_info["train_stems"]), len(ds_info["test_stems"]))

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

        for cfg in LONG_CONFIGS:
            done += 1
            logger.info("━━━ [%d/%d] %s config %s (r=%d bs=%d cp=%s mo=%s) %d epochs ━━━",
                         done, total, ds_name, cfg["name"],
                         cfg["rank"], cfg["batch_size"], cfg["copy_paste"], cfg["mosaic"],
                         EPOCHS)

            run_single_experiment_phase3(
                db_path, base_model, config, ds_info, device,
                rank=cfg["rank"], batch_size=cfg["batch_size"],
                copy_paste=cfg["copy_paste"], mosaic=cfg["mosaic"],
                lora_targets=FIXED["lora_targets"],
                warmup=FIXED["warmup"],
                multi_scale=FIXED["multi_scale"],
                epochs=EPOCHS,
                lr=FIXED["learning_rate"],
                alpha_ratio=FIXED["alpha_ratio"],
                weight_decay=FIXED["weight_decay"],
                template_model=template,
                filtered_state=filtered,
                fresh_config=fresh_config,
                cached_criterion=criterion,
                cached_weight_dict=weight_dict,
            )

        del template, criterion
        torch.cuda.empty_cache()

    if gpu_stop is not None:
        gpu_stop.set()

    logger.info("🎉 Long training complete!")


if __name__ == "__main__":
    main()

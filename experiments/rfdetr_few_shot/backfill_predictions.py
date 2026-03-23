#!/usr/bin/env python3
"""Backfill the predictions table for existing completed experiments.

Loads each experiment's model checkpoint, re-runs inference on test images,
and stores per-image predictions + GT in the predictions table.

Usage:
    python backfill_predictions.py [--db results_finetune_baseline.db] [--limit 50]
"""
import argparse
import json
import logging
import sqlite3
import time
import torch
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DS_ROOT = Path.home() / "Downloads" / "rf20-vl-fsod"


def _db_connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.row_factory = sqlite3.Row
    return conn


def _discover_stems(images_dir):
    if not images_dir.exists():
        return []
    return sorted(p.stem for p in images_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def load_image(ds_path, split, stem):
    img_dir = ds_path / split / "images"
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    return None


def parse_yolo_label(lbl_path):
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


def backfill_ft(conn, exp, ds_path, class_names):
    """Backfill predictions for a full_finetune experiment using rfdetr."""
    variant = exp["model_variant"]
    ds_name = exp["dataset_name"]
    exp_id = exp["id"]

    # Find checkpoint
    variant_short = variant.replace("rfdetr-", "")
    ckpt_path = SCRIPT_DIR / "finetune_outputs" / variant_short / ds_name / "checkpoint_best_total.pth"
    if not ckpt_path.exists():
        logger.warning("  Checkpoint not found: %s", ckpt_path)
        return False

    # Load model
    if "nano" in variant:
        from rfdetr import RFDETRNano as ModelCls
    elif "medium" in variant:
        from rfdetr import RFDETRMedium as ModelCls
    elif "2xlarge" in variant:
        from rfdetr import RFDETR2XLarge as ModelCls
    else:
        logger.warning("  Unknown variant: %s", variant)
        return False

    model = ModelCls()
    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=False)
    model.model.model.load_state_dict(ckpt["model"], strict=True)
    model.model.model = model.model.model.eval().cuda()
    model.model.device = torch.device("cuda")

    nc = len(class_names)
    test_stems = _discover_stems(ds_path / "test" / "images")
    class_names_json = json.dumps(class_names)

    for stem in test_stems:
        pil_img = load_image(ds_path, "test", stem)
        if pil_img is None:
            continue
        img_w, img_h = pil_img.size

        dets = model.predict(pil_img, threshold=0.01)
        preds = []
        if len(dets.xyxy) > 0:
            for i in range(len(dets.xyxy)):
                cid = int(dets.class_id[i])
                if cid >= nc:
                    continue
                preds.append({
                    "class_id": cid,
                    "confidence": float(dets.confidence[i]),
                    "bbox": dets.xyxy[i].tolist(),
                })
        preds.sort(key=lambda d: d["confidence"], reverse=True)
        preds = preds[:500]

        gt_boxes = parse_yolo_label(ds_path / "test" / "labels" / f"{stem}.txt")
        gt = []
        for b in gt_boxes:
            gt.append({
                "class_id": b["class_id"],
                "bbox": [
                    (b["cx"] - b["w"]/2) * img_w, (b["cy"] - b["h"]/2) * img_h,
                    (b["cx"] + b["w"]/2) * img_w, (b["cy"] + b["h"]/2) * img_h,
                ],
            })

        conn.execute("""INSERT INTO predictions
            (experiment_id, eval_image_stem, image_width, image_height,
             predictions_json, gt_json, class_names_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, stem, img_w, img_h, json.dumps(preds), json.dumps(gt), class_names_json))

    conn.commit()
    del model
    torch.cuda.empty_cache()
    return True


def backfill_lora(conn, exp, ds_path, class_names):
    """Backfill predictions for a LoRA experiment using inference_models."""
    import sys
    sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
    from grid_search import run_inference, load_image_and_labels_generic

    variant = exp["model_variant"]
    ds_name = exp["dataset_name"]
    exp_id = exp["id"]
    nc = len(class_names)

    from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import CONFIG_FOR_MODEL_TYPE
    from inference_models.models.rfdetr.rfdetr_base_pytorch import build_model as _build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = SCRIPT_DIR / f"rf-detr-{variant.replace('rfdetr-', '')}.pth"
    if not weights_path.exists():
        logger.warning("  Weights not found: %s", weights_path)
        return False

    config = CONFIG_FOR_MODEL_TYPE[variant](device=device)
    weights = torch.load(str(weights_path), map_location=device, weights_only=False)["model"]
    config.num_classes = weights["class_embed.bias"].shape[0] - 1

    # For LoRA we need the merged model — but we don't have it saved
    # We can only store predictions if we re-train, which is expensive
    # Instead, skip LoRA backfill for now — new runs will store predictions
    logger.info("  Skipping LoRA backfill (requires re-training). New runs will store predictions.")
    return False


def backfill_zeroshot(conn, exp, ds_path, class_names):
    """Backfill for zero-shot models (OWLv2, SAM3) — requires model re-inference."""
    variant = exp["model_variant"]
    ds_name = exp["dataset_name"]
    exp_id = exp["id"]
    nc = len(class_names)

    test_stems = _discover_stems(ds_path / "test" / "images")
    class_names_json = json.dumps(class_names)

    if "owlv2" in variant:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
        from inference_models.models.owlv2.owlv2_hf import OWLv2HF
        model = OWLv2HF.from_pretrained(
            "google/owlv2-large-patch14-ensemble",
            device=torch.device("cuda"), local_files_only=False,
        )

        for stem in test_stems:
            pil_img = load_image(ds_path, "test", stem)
            if pil_img is None:
                continue
            img_w, img_h = pil_img.size
            img_np = np.array(pil_img)

            detections_list = model.infer(
                img_np, classes=class_names,
                confidence=0.001, iou_threshold=0.7, max_detections=500,
            )
            detections = detections_list[0]

            preds = []
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i in range(len(detections.xyxy)):
                    cid = int(detections.class_id[i])
                    if cid >= nc:
                        continue
                    preds.append({
                        "class_id": cid,
                        "confidence": float(detections.confidence[i]),
                        "bbox": detections.xyxy[i].cpu().float().tolist(),
                    })
            preds.sort(key=lambda d: d["confidence"], reverse=True)
            preds = preds[:500]

            gt_boxes = parse_yolo_label(ds_path / "test" / "labels" / f"{stem}.txt")
            gt = []
            for b in gt_boxes:
                gt.append({
                    "class_id": b["class_id"],
                    "bbox": [
                        (b["cx"] - b["w"]/2) * img_w, (b["cy"] - b["h"]/2) * img_h,
                        (b["cx"] + b["w"]/2) * img_w, (b["cy"] + b["h"]/2) * img_h,
                    ],
                })

            conn.execute("""INSERT INTO predictions
                (experiment_id, eval_image_stem, image_width, image_height,
                 predictions_json, gt_json, class_names_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (exp_id, stem, img_w, img_h, json.dumps(preds), json.dumps(gt), class_names_json))

        conn.commit()
        del model
        torch.cuda.empty_cache()
        return True
    else:
        logger.info("  Skipping %s backfill (not implemented yet)", variant)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="results_finetune_baseline.db")
    parser.add_argument("--datasets-root", default=str(DS_ROOT))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max experiments to backfill")
    parser.add_argument("--method", default=None,
                        help="Only backfill this method (full_finetune, lora, zero_shot)")
    parser.add_argument("--only", default=None,
                        help="Only backfill this dataset")
    args = parser.parse_args()

    ds_root = Path(args.datasets_root)

    db_path = SCRIPT_DIR / args.db
    conn = _db_connect(db_path)

    # Create predictions table if not exists
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_exp ON predictions(experiment_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_stem ON predictions(experiment_id, eval_image_stem)")
    conn.commit()

    # Find experiments that need backfilling
    query = """
        SELECT e.id, e.run_id, e.dataset_name, e.model_variant, e.method
        FROM experiments e
        LEFT JOIN (SELECT experiment_id, COUNT(*) as cnt FROM predictions GROUP BY experiment_id) p
            ON p.experiment_id = e.id
        WHERE e.status = 'completed' AND (p.cnt IS NULL OR p.cnt = 0)
    """
    params = []
    if args.method:
        query += " AND e.method = ?"
        params.append(args.method)
    if args.only:
        query += " AND e.dataset_name = ?"
        params.append(args.only)
    query += " ORDER BY e.id"
    if args.limit:
        query += f" LIMIT {args.limit}"

    exps = conn.execute(query, params).fetchall()
    logger.info("Found %d experiments to backfill", len(exps))

    for i, exp in enumerate(exps):
        ds_path = ds_root / exp["dataset_name"]
        if not ds_path.exists():
            logger.warning("[%d/%d] Dataset not found: %s", i+1, len(exps), exp["dataset_name"])
            continue

        with open(ds_path / "data.yaml") as f:
            data = yaml.safe_load(f)
        class_names = data["names"]
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names.keys())]

        method = exp["method"] or "full_finetune"
        logger.info("[%d/%d] %s (%s %s)", i+1, len(exps), exp["run_id"], method, exp["model_variant"])

        t0 = time.time()
        if method == "full_finetune":
            ok = backfill_ft(conn, exp, ds_path, class_names)
        elif method == "lora":
            ok = backfill_lora(conn, exp, ds_path, class_names)
        elif method == "zero_shot":
            ok = backfill_zeroshot(conn, exp, ds_path, class_names)
        else:
            logger.warning("  Unknown method: %s", method)
            ok = False

        if ok:
            cnt = conn.execute("SELECT COUNT(*) FROM predictions WHERE experiment_id=?",
                               (exp["id"],)).fetchone()[0]
            logger.info("  Done: %d predictions in %.1fs", cnt, time.time() - t0)
        torch.cuda.empty_cache()

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    logger.info("Total predictions in DB: %d", total)


if __name__ == "__main__":
    main()

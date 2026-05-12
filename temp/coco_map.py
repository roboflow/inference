"""Compute coco detection + segm mAP for rfdetr-seg-nano on coco/val2017.

A single process runs one pass with the path selected by the
USE_TRITON_FOR_PREPROCESSING env var. Run it twice
(true, false) and compare — mAP should be identical to 4 decimals.

Usage:
  USE_TRITON_FOR_PREPROCESSING=true  python temp/coco_map.py
  USE_TRITON_FOR_PREPROCESSING=false python temp/coco_map.py

Also prints the Triton kernel call count: it must equal the image count
when the env is true and be 0 when it is false.
"""
import os
import time
from pathlib import Path

os.environ.setdefault(
    "DISABLED_INFERENCE_MODELS_BACKENDS",
    "torch,torch-script,onnx,hugging-face,ultralytics,mediapipe,custom",
)

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask

from inference_models import AutoModel
import inference_models.models.rfdetr.pre_processing as trt_mod

COCO_DIR = Path("/home/ubuntu/inference/coco")
ANN = COCO_DIR / "annotations" / "instances_val2017.json"
IMG_DIR = COCO_DIR / "val2017"
CONF = 0.05  # coco-eval convention: submit low-confidence detections too


def build_id_maps(model_class_names, coco):
    name_to_cat_id = {coco.loadCats([c])[0]['name']: c for c in coco.getCatIds()}
    return {
        idx: name_to_cat_id[name]
        for idx, name in enumerate(model_class_names)
        if name in name_to_cat_id
    }


def encode_rle(mask_bool):
    arr = np.asfortranarray(mask_bool.cpu().numpy().astype(np.uint8))
    rle = coco_mask.encode(arr)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def predictions_for_image(model, img_bgr, image_id, cat_map):
    pre, meta = model.pre_process(img_bgr)
    out = model.forward(pre)
    det = model.post_process(out, meta, confidence=CONF)[0]
    n = int(det.class_id.numel())
    if n == 0:
        return [], []
    xyxy = det.xyxy.cpu().numpy()
    scores = det.confidence.cpu().numpy()
    class_ids = det.class_id.cpu().numpy()
    masks = det.mask

    bbox_preds, segm_preds = [], []
    for i in range(n):
        cid = int(class_ids[i])
        if cid not in cat_map:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        bbox_preds.append({
            "image_id": image_id,
            "category_id": cat_map[cid],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(scores[i]),
        })
        if masks is not None:
            segm_preds.append({
                "image_id": image_id,
                "category_id": cat_map[cid],
                "segmentation": encode_rle(masks[i]),
                "score": float(scores[i]),
            })
    return bbox_preds, segm_preds


def run_pass(model, coco, cat_map, tag):
    bbox, segm = [], []
    ids = sorted(coco.getImgIds())
    t0 = time.perf_counter()
    for n, image_id in enumerate(ids):
        info = coco.loadImgs([image_id])[0]
        img = cv2.imread(str(IMG_DIR / info['file_name']), cv2.IMREAD_COLOR)
        if img is None:
            continue
        b, s = predictions_for_image(model, img, image_id, cat_map)
        bbox.extend(b)
        segm.extend(s)
        if (n + 1) % 500 == 0:
            print(f"  [{tag}] {n+1}/{len(ids)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return bbox, segm


def eval_and_print(coco_gt, results, iou_type):
    if not results:
        return None
    ev = COCOeval(coco_gt, coco_gt.loadRes(results), iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats.tolist()


def main():
    env_flag = os.environ.get("USE_TRITON_FOR_PREPROCESSING", "true")
    tag = f"env={env_flag}"

    triton_calls = {"count": 0}
    original = trt_mod.triton_preprocess_rfdetr_stretch
    if original is not None:
        def counting(*a, **kw):
            triton_calls["count"] += 1
            return original(*a, **kw)
        trt_mod.triton_preprocess_rfdetr_stretch = counting

    print(f"USE_TRITON_FOR_PREPROCESSING={env_flag}")
    print(f"USE_TRITON_FOR_PREPROCESSING={trt_mod.USE_TRITON_FOR_PREPROCESSING}  "
          f"_TRITON_AVAILABLE={trt_mod._TRITON_AVAILABLE}")

    coco = COCO(str(ANN))
    model = AutoModel.from_pretrained('rfdetr-seg-nano')
    cat_map = build_id_maps(model.class_names, coco)

    bbox, segm = run_pass(model, coco, cat_map, tag)
    print(f"\n  Triton kernel calls: {triton_calls['count']}")

    print(f"\n== bbox mAP ({tag}) ==")
    stats_bbox = eval_and_print(coco, bbox, 'bbox')
    print(f"\n== segm mAP ({tag}) ==")
    stats_segm = eval_and_print(coco, segm, 'segm')

    if stats_bbox:
        print(f"\n  bbox : AP={stats_bbox[0]:.4f}  AP50={stats_bbox[1]:.4f}  AP75={stats_bbox[2]:.4f}")
    if stats_segm:
        print(f"  segm : AP={stats_segm[0]:.4f}  AP50={stats_segm[1]:.4f}  AP75={stats_segm[2]:.4f}")


if __name__ == "__main__":
    main()

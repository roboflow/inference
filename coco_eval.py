import os
import json
import glob
from pathlib import Path
from typing import List, Dict
import io
import numpy as np
from PIL import Image
from tqdm import tqdm
from inference import get_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VAL_IMAGES_DIR = "val2017"
ANN_FILE = "annotations/instances_val2017.json"
RESULTS_JSON = "coco_val2017_results.json"

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] =  "['TensorrtExecutionProvider']" #"['CUDAExecutionProvider']"

def load_coco_name_to_id(ann_file: str) -> Dict[str, int]:
    """Build a mapping from COCO category name -> category_id using the GT file."""
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    return {c["name"]: c["id"] for c in cats}

def center_to_xywh(x_center: float, y_center: float, w: float, h: float):
    x_min = x_center - w / 2.0
    y_min = y_center - h / 2.0
    return [float(x_min), float(y_min), float(w), float(h)]

def main():
    print("getting model")
    model = get_model("rfdetr-medium")
    print("Got model, running evaluation on COCO val2017")

    coco_gt = COCO(ANN_FILE)
    name_to_catid = load_coco_name_to_id(ANN_FILE)

    img_id_by_file = {img["file_name"]: img["id"] for img in coco_gt.loadImgs(coco_gt.getImgIds())}

    image_paths = sorted(glob.glob(str(Path(VAL_IMAGES_DIR) / "*.jpg")))
    results: List[dict] = []

    for img_path in tqdm(image_paths, desc="Evaluating"):
        file_name = Path(img_path).name
        image_id = img_id_by_file.get(file_name)
        image_pil = Image.open(img_path).convert("RGB")
        pred = model.infer(image_pil, confidence=0.0)[0]

        for p in pred.predictions:
            score = float(p.confidence)
            cat_name = p.class_name
            category_id = int(name_to_catid[cat_name])
            x, y, w, h = float(p.x), float(p.y), float(p.width), float(p.height)
            bbox_xywh = center_to_xywh(x, y, w, h)
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [v for v in bbox_xywh],
                    "score": score
                }
            )

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f)

    print(f"Wrote detections to {RESULTS_JSON} ({len(results)} boxes).")

    if len(results) == 0:
        print("No detections produced — check class-name mapping or confidence threshold.")
        return

    coco_dt = coco_gt.loadRes(RESULTS_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()
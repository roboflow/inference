"""Compare RLE placement and rendering against a pinned source revision.

Run with the inference and inference_models packages on PYTHONPATH:
  python development/rle_pipeline/benchmark.py --results /path/to/results.json

The corpus comes from saved video predictions (polygon output), not raw SAM3
logits. It measures CPU mask operations, not GPU inference or video throughput.
"""

import argparse
import ast
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import supervision as sv
from pycocotools import mask as mask_utils
from supervision.detection.compact_mask import _rle_join_cols, _rle_split_cols

from inference.core.workflows.core_steps.common import rle_compact, tensor_native
from inference.core.workflows.core_steps.visualizations.common.annotators.compact_polygon import (
    CompactPolygonAnnotator,
)
from inference_models.models.base.types import InstancesRLEMasks

ROOT = Path(__file__).resolve().parents[2]


def baseline_functions(revision, path, module, names, extra=None):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
    )
    selected = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = dict(vars(module))
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)
    return namespace


def measure(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 240, 571])
    parser.add_argument("--baseline", default="72b5c5407")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    old_crop = baseline_functions(
        args.baseline,
        "inference/core/workflows/core_steps/common/rle_compact.py",
        rle_compact,
        {
            "_decode_coco_counts",
            "_trim_col_runs",
            "compact_mask_from_coco_rle",
            "instances_rle_to_compact_mask",
        },
        {"_rle_join_cols": _rle_join_cols, "_rle_split_cols": _rle_split_cols},
    )
    old_stitch = baseline_functions(
        args.baseline,
        "inference/core/workflows/core_steps/common/tensor_native.py",
        tensor_native,
        {
            "_column_to_runs",
            "_embed_single_mask_counts",
            "embed_rle_masks_in_larger_canvas",
        },
    )
    records = json.loads(args.results.read_text())
    for frame_index in args.frames:
        data = records[frame_index]["outputs"]["predictions"]
        h, w = data["image"]["height"], data["image"]["width"]
        rles, classes, tiles = [], [], []
        for prediction in data["predictions"]:
            polygon = (
                np.array([[p["x"], p["y"]] for p in prediction["points"]])
                .ravel()
                .tolist()
            )
            if len(polygon) < 6:
                continue
            rle = mask_utils.merge(mask_utils.frPyObjects([polygon], h, w))
            x, y, bw, bh = map(int, mask_utils.toBbox(rle))
            if bw == 0 or bh == 0:
                continue
            rles.append(rle)
            classes.append(prediction["class_id"])
            if bw <= 640 and bh <= 640 and h >= 640 and w >= 640:
                ox, oy = min(x, w - 640), min(y, h - 640)
                dense = mask_utils.decode(rle)[oy : oy + 640, ox : ox + 640]
                tile = mask_utils.encode(np.asfortranarray(dense))
                tiles.append(
                    (
                        InstancesRLEMasks.from_coco_rle_masks((640, 640), [tile]),
                        (ox, oy),
                    )
                )
        xywh = mask_utils.toBbox(rles)
        boxes = xywh.copy()
        boxes[:, 2:] += boxes[:, :2] - 1
        carrier = InstancesRLEMasks.from_coco_rle_masks((h, w), rles)
        old_time, old = measure(
            lambda: old_crop["instances_rle_to_compact_mask"](carrier, boxes),
            args.repeats,
        )
        new_time, new = measure(
            lambda: rle_compact.instances_rle_to_compact_mask(carrier, boxes),
            args.repeats,
        )
        for i in range(len(old)):
            np.testing.assert_array_equal(old.crop(i), new.crop(i))
        detections = sv.Detections(xyxy=boxes, mask=new, class_id=np.asarray(classes))
        scene = np.zeros((h, w, 3), dtype=np.uint8)
        old_draw, reference = measure(
            lambda: sv.PolygonAnnotator().annotate(scene.copy(), detections),
            args.repeats,
        )
        new_draw, rendered = measure(
            lambda: CompactPolygonAnnotator().annotate(scene.copy(), detections),
            args.repeats,
        )
        np.testing.assert_array_equal(reference, rendered)
        old_place, old_masks = measure(
            lambda: [
                old_stitch["embed_rle_masks_in_larger_canvas"](m, offset, (h, w)).masks
                for m, offset in tiles
            ],
            args.repeats,
        )
        new_place, new_masks = measure(
            lambda: [
                tensor_native.embed_rle_masks_in_larger_canvas(m, offset, (h, w)).masks
                for m, offset in tiles
            ],
            args.repeats,
        )
        assert old_masks == new_masks
        print(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "instances": len(rles),
                    "placement_instances": len(tiles),
                    "seconds": {
                        "compact_conversion": {"before": old_time, "after": new_time},
                        "polygon_render": {"before": old_draw, "after": new_draw},
                        "placement": {"before": old_place, "after": new_place},
                    },
                    "mask_and_render_parity": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

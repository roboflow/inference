"""Synthetic CPU benchmark; run from the repository root with PYTHONPATH=.:workflows:inference_models.

Measures the SAM3 block's mask conversion, not inference or video throughput.
Uses no model, network, or GPU. Peak dense allocations can exceed 2 GiB.
"""

import gc
import json
import platform
import statistics
import time
from copy import deepcopy

import numpy as np
import supervision as sv
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)
from roboflow_workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)


def _sample(*, count, height, width):
    rng = np.random.default_rng(42)
    rles = []
    boxes = []
    for _ in range(count):
        x = int(rng.integers(0, width - 80))
        y = int(rng.integers(0, height - 80))
        mask = np.zeros((height, width), dtype=np.uint8, order="F")
        mask[y : y + 60, x : x + 40] = 1
        mask[y + 10 : y + 25, x + 10 : x + 25] = 0
        rle = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)
        bx, by, bw, bh = mask_utils.toBbox(rle)
        boxes.append([bx, by, bx + bw, by + bh])
    detections = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        data={
            RLE_MASK_KEY_IN_SV_DETECTIONS: np.array(rles, dtype=object),
            IMAGE_DIMENSIONS_KEY: np.tile([height, width], (count, 1)),
        },
    )
    return detections


def _benchmark():
    block = SegmentAnything3BlockV3(None, None, None)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "supervision": sv.__version__,
                "numpy": np.__version__,
                "machine": platform.machine(),
            }
        ),
        flush=True,
    )
    for count in (100, 500):
        predictions = _sample(count=count, height=1080, width=1920)
        times = {False: [], True: []}
        for repeat in range(4):
            # Alternate order; discard one warm-up per representation.
            for compact in ((False, True) if repeat % 2 == 0 else (True, False)):
                detections = deepcopy(predictions)
                gc.collect()
                start = time.perf_counter()
                block._decode_and_cache_rle_masks(
                    [detections], use_compact_masks=compact
                )
                elapsed = time.perf_counter() - start
                if repeat:
                    times[compact].append(elapsed)
                if compact:
                    # Verify masks individually without allocating a second dense stack.
                    for i, rle in enumerate(
                        predictions.data[RLE_MASK_KEY_IN_SV_DETECTIONS]
                    ):
                        np.testing.assert_array_equal(
                            detections.mask[i], mask_utils.decode(rle)
                        )
                del detections
        dense_s, compact_s = (statistics.median(times[mode]) for mode in (False, True))
        print(
            json.dumps(
                {
                    "instances": count,
                    "shape": [1080, 1920],
                    "dense_seconds": times[False],
                    "compact_seconds": times[True],
                    "median_speedup": dense_s / compact_s,
                    "dense_stack_bytes": count * 1080 * 1920,
                    "mask_parity": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    _benchmark()

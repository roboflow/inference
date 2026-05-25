"""Convert HTTP inference API response dicts into `inference_models` native
prediction types.

Mirrors the semantics of
`convert_inference_detections_batch_to_sv_detections`
(`core_steps/common/utils.py:105`) but emits tensor-native types instead
of `sv.Detections`. Used by tensor-mode workflow blocks' remote execution
path so the value flowing out of `run_remotely` matches `run_locally`.

Per-detection fields the inference HTTP API attaches alongside
`x`/`y`/`width`/`height`/`class_id`/`confidence` (e.g. `detection_id`,
`parent_id`) land in `bboxes_metadata`. Top-level response fields
(e.g. `inference_id`) land in `image_metadata` and are later overwritten
by `attach_prediction_metadata` with the full per-prediction global
state.
"""

import uuid
from typing import Dict, Iterable, List, Optional

import torch

from inference_models.models.base.object_detection import Detections

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)


def dict_response_to_object_detections(
    response: dict,
    *,
    predictions_key: str = "predictions",
    image_key: str = "image",
) -> Detections:
    """Build a single `inference_models.Detections` from one inference
    HTTP-API response dict.

    Response dict shape (matches the format
    `convert_inference_detections_batch_to_sv_detections` consumes):
        {
          "image": {"width": int, "height": int},
          "predictions": [
              {"x": center_x, "y": center_y, "width": w, "height": h,
               "class": str, "class_id": int, "confidence": float,
               "detection_id"?: str, "parent_id"?: str, ...},
              ...
          ],
          "inference_id"?: str,
        }

    `image` width/height is preserved in `image_metadata` under the same
    `WIDTH_KEY` / `HEIGHT_KEY` constants the numpy mirror uses for
    `IMAGE_DIMENSIONS_KEY`.
    """
    raw_predictions = response.get(predictions_key) or []
    n = len(raw_predictions)
    if n == 0:
        xyxy = torch.zeros((0, 4), dtype=torch.float32)
        class_id = torch.zeros((0,), dtype=torch.int64)
        confidence = torch.zeros((0,), dtype=torch.float32)
        bboxes_metadata: Optional[List[dict]] = None
    else:
        xyxy_rows = []
        class_ids: List[int] = []
        confidences: List[float] = []
        bboxes_metadata = []
        for p in raw_predictions:
            cx, cy = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            xyxy_rows.append(
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
            )
            class_ids.append(int(p.get("class_id", 0)))
            confidences.append(float(p.get("confidence", 0.0)))
            per_box: dict = {
                DETECTION_ID_KEY: p.get(DETECTION_ID_KEY) or str(uuid.uuid4()),
                PARENT_ID_KEY: p.get(PARENT_ID_KEY, ""),
            }
            bboxes_metadata.append(per_box)
        xyxy = torch.tensor(xyxy_rows, dtype=torch.float32)
        class_id = torch.tensor(class_ids, dtype=torch.int64)
        confidence = torch.tensor(confidences, dtype=torch.float32)
    image_section = response.get(image_key) or {}
    image_metadata: dict = {}
    if WIDTH_KEY in image_section and HEIGHT_KEY in image_section:
        image_metadata[WIDTH_KEY] = image_section[WIDTH_KEY]
        image_metadata[HEIGHT_KEY] = image_section[HEIGHT_KEY]
    response_inference_id = response.get(INFERENCE_ID_KEY)
    if response_inference_id is not None:
        image_metadata[INFERENCE_ID_KEY] = response_inference_id
    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        image_metadata=image_metadata or None,
        bboxes_metadata=bboxes_metadata,
    )


def class_id_to_name_from_responses(
    responses: Iterable[dict],
    *,
    predictions_key: str = "predictions",
) -> Dict[int, str]:
    """Merge `class_id -> class_name` mapping across a batch of HTTP API
    response dicts.

    The mapping is sparse: only class_ids that appeared in at least one
    detection across the batch are included. The model has a fixed
    class table, so values across responses are consistent — first-seen
    wins on the off chance of duplicates.

    Used by tensor-mode workflow blocks' remote path so the
    `class_names` attached to each prediction's `image_metadata` is the
    same `Dict[int, str]` shape the local path produces (via
    `dict(enumerate(model_manager.get_class_names(...)))`).
    """
    mapping: Dict[int, str] = {}
    for response in responses:
        for p in response.get(predictions_key) or []:
            class_id = p.get("class_id")
            class_name = p.get("class")
            if class_id is None or class_name is None:
                continue
            class_id_int = int(class_id)
            if class_id_int not in mapping:
                mapping[class_id_int] = class_name
    return mapping

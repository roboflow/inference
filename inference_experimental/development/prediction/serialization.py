import json
import os.path
from typing import Any, Union

import supervision as sv


def serialize_results(predictions: Any) -> dict:
    if isinstance(predictions, sv.Detections):
        return sv_detections_to_json(detections=predictions)
    if isinstance(predictions, sv.KeyPoints):
        return sv_key_points_to_json(key_points=predictions)
    raise NotImplementedError(
        f"Could not serialise result of type: {type(predictions)}"
    )


def sv_detections_to_json(detections: sv.Detections) -> dict:
    return {
        "xyxy": detections.xyxy.tolist(),
        "mask": detections.mask.tolist() if detections.mask is not None else None,
        "confidence": (
            detections.confidence.tolist()
            if detections.confidence is not None
            else None
        ),
        "class_id": (
            detections.class_id.tolist() if detections.class_id is not None else None
        ),
    }


def sv_key_points_to_json(key_points: sv.KeyPoints) -> dict:
    return {
        "xy": key_points.xy.tolist(),
        "class_id": (
            key_points.class_id.tolist() if key_points.class_id is not None else None
        ),
        "confidence": (
            key_points.confidence.tolist()
            if key_points.confidence is not None
            else None
        ),
    }


def dump_json(path: str, content: Union[dict, list]) -> None:
    prent_dir = os.path.dirname(path)
    os.makedirs(prent_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f)

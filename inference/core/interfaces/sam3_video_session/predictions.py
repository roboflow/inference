from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inference.core.interfaces.sam3_video_session.entities import DEFAULT_CLASS_NAME


def mask_to_uncompressed_rle(mask: np.ndarray) -> Dict[str, Any]:
    height, width = mask.shape[:2]
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    if flat.size == 0:
        return {"counts": [0], "size": [int(height), int(width)]}
    change_indices = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_indices, [flat.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if int(flat[0]) == 1:
        counts.insert(0, 0)
    return {"counts": counts, "size": [int(height), int(width)]}


def xyxy_to_center_bounds(box: Sequence[float]) -> Optional[Dict[str, float]]:
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return None
    return {
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
    }


def class_name_for_object(
    object_id: int,
    prompt_to_object_ids: Dict[str, List[int]],
    fallback: str = DEFAULT_CLASS_NAME,
) -> str:
    for prompt, object_ids in prompt_to_object_ids.items():
        if int(object_id) in {int(item) for item in object_ids}:
            name = str(prompt).strip()
            return name or fallback
    return fallback


def serialize_frame_predictions(
    *,
    masks: np.ndarray,
    object_ids: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    prompt_to_object_ids: Dict[str, List[int]],
    threshold: float,
    width: int,
    height: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    if masks.size == 0:
        return predictions, samples
    for index, object_id in enumerate(object_ids.tolist()):
        score = float(scores[index]) if index < len(scores) else 0.0
        if score < threshold:
            continue
        mask = masks[index]
        if not np.any(mask):
            continue
        box = boxes[index] if index < len(boxes) else (0, 0, 0, 0)
        bounds = xyxy_to_center_bounds(box)
        rle_mask = mask_to_uncompressed_rle(mask)
        class_name = class_name_for_object(int(object_id), prompt_to_object_ids)
        prediction: Dict[str, Any] = {
            "tracker_id": int(object_id),
            "class_name": class_name,
            "confidence": score,
            "rle_mask": rle_mask,
        }
        if bounds is not None:
            prediction.update(bounds)
        predictions.append(prediction)
        geometry: Dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "rleMask": rle_mask,
        }
        if bounds is not None:
            geometry["bounds"] = bounds
        samples.append(
            {
                "trackId": int(object_id),
                "className": class_name,
                "confidence": score,
                "geometry": geometry,
            }
        )
    return predictions, samples

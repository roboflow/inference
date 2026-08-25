"""Read the fields blocks need from either prediction representation.

Detections reach a block either as ``supervision.Detections`` (class names in
``data["class_name"]``, numpy arrays) or, with ``ENABLE_TENSOR_DATA_REPRESENTATION``,
as ``inference_models`` ``Detections`` (torch tensors, class names in
``image_metadata["class_names"]`` keyed by class id). Blocks that only need a
few fields - which class was seen, how confident, where - use these accessors so
one implementation serves both modes without importing torch.
"""

from typing import Any, List, Tuple

import numpy as np

from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):  # torch tensor, possibly on an accelerator
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def detections_count(predictions: Any) -> int:
    return 0 if predictions is None else len(predictions)


def detections_class_names_and_confidences(
    predictions: Any,
) -> Tuple[List[str], List[float]]:
    """Per-detection class names and confidences, in detection order."""
    if detections_count(predictions) == 0:
        return [], []
    if hasattr(predictions, "data"):  # supervision
        names = [str(n) for n in predictions.data.get("class_name", [])]
        confidences = (
            [] if predictions.confidence is None else _to_numpy(predictions.confidence)
        )
        return names, [float(c) for c in confidences]
    lookup = (getattr(predictions, "image_metadata", None) or {}).get(
        CLASS_NAMES_KEY
    ) or {}
    names = []
    for class_id in _to_numpy(predictions.class_id).reshape(-1).tolist():
        class_id = int(class_id)
        names.append(str(lookup.get(class_id, lookup.get(str(class_id), class_id))))
    confidences = _to_numpy(predictions.confidence).reshape(-1).tolist()
    return names, [float(c) for c in confidences]


def detections_boxes_and_confidences(predictions: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``(xyxy of shape (n, 4), confidence of shape (n,))`` as float numpy arrays."""
    if detections_count(predictions) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float)
    xyxy = _to_numpy(predictions.xyxy).astype(float).reshape(-1, 4)
    confidence = predictions.confidence
    if confidence is None:
        confidence = np.ones((xyxy.shape[0],), dtype=float)
    return xyxy, _to_numpy(confidence).astype(float).reshape(-1)

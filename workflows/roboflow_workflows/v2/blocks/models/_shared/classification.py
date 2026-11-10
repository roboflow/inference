"""Shared result step of the single- and multi-label classification blocks.

A classification has no rows, so it gets no row identity. When the image is a
crop of a detection, the prediction carries that detection's row reference
(``image.parent_row``) for joining labels back to it.
"""

import uuid
from typing import Any, Dict

from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import with_classification_parent

PREDICTION_TYPE = "classification"


def classification_result(
    prediction: Any, image: ImageData, *, model_id: str
) -> Dict[str, Any]:
    """One image's outputs; the prediction carries the image's parent row."""
    result = {
        "predictions": with_classification_parent(prediction, image.parent_row),
        "model_id": model_id,
        "inference_id": uuid.uuid4().hex,
    }

    return result

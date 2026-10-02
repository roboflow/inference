"""Compare two native classification predictions field by field.

Exact comparison is the default: the same phase code on the same device and
input shapes must give bitwise-identical tensors. A tolerance is only for
different devices or kernels, and the record says which rule was used.
"""

from typing import Any, Collection, Dict, Optional

import torch

from inference_models.models.base.classification import ClassificationPrediction


def compare_predictions(
    expected: ClassificationPrediction,
    actual: ClassificationPrediction,
    *,
    tolerance: Optional[float] = None,
    ignore_metadata: Collection[str] = (),
) -> Dict[str, Any]:
    """Compare class ids, confidences, dtypes, devices and image metadata.

    Args:
        expected: Reference prediction.
        actual: Prediction to check.
        tolerance: Maximum absolute confidence difference; ``None`` requires
            bitwise equality. Class ids and metadata must always match.
        ignore_metadata: ``images_metadata`` keys left out, for example
            ``parent_id`` of crops, which every run names anew.

    Returns:
        ``equal`` plus the evidence: rule, max difference, class ids,
        dtypes, devices, whether the metadata matched and what was ignored.
    """
    expected_confidence = expected.confidence.detach().cpu()
    actual_confidence = actual.confidence.detach().cpu()
    shapes_match = expected_confidence.shape == actual_confidence.shape
    difference = (
        float((expected_confidence - actual_confidence).abs().max())
        if shapes_match
        else None
    )
    if not shapes_match:
        confidence_match = False
    elif tolerance is None:
        confidence_match = torch.equal(expected_confidence, actual_confidence)
    else:
        confidence_match = difference <= tolerance

    class_ids_match = torch.equal(expected.class_id.cpu(), actual.class_id.cpu())
    metadata_match = _kept(expected.images_metadata, ignore_metadata) == _kept(
        actual.images_metadata, ignore_metadata
    )
    native_types = (
        expected.confidence.dtype == actual.confidence.dtype
        and expected.class_id.dtype == actual.class_id.dtype
    )
    equal = confidence_match and class_ids_match and metadata_match and native_types
    record = {
        "equal": equal,
        "rule": "bitwise" if tolerance is None else f"abs <= {tolerance:g}",
        "max_abs_confidence_difference": difference,
        "class_ids": [expected.class_id.tolist(), actual.class_id.tolist()],
        "dtypes": [str(expected.confidence.dtype), str(actual.confidence.dtype)],
        "devices": [str(expected.confidence.device), str(actual.confidence.device)],
        "metadata_equal": metadata_match,
        "metadata_ignored": sorted(ignore_metadata),
    }

    return record


def _kept(rows: Optional[list], ignored: Collection[str]) -> Optional[list]:
    if rows is None:
        return None

    kept = [
        {key: value for key, value in row.items() if key not in ignored} for row in rows
    ]

    return kept

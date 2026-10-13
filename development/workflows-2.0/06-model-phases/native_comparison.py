"""Compare two native classification predictions field by field.

Exact comparison is the default: the same phase code on the same device and
input shapes must give bitwise-identical tensors. A tolerance is only for
different devices or kernels, and the record says which rule was used.
"""

from typing import Any, Collection, Dict, Optional

import torch
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY

from inference_models.models.base.classification import ClassificationPrediction

# V1 identity keys and the typed frame record field that repeats the same id,
# as (record section or None, field).
_RECORD_IDENTITY_FIELDS = {
    "parent_id": (None, "image_id"),
    "parent_frame_id": ("parent", "frame_id"),
    "root_parent_id": ("root", "frame_id"),
}
_IGNORED = "<ignored>"


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
            ``parent_id`` of crops, which every run names anew. An ignored
            identity key also masks the typed frame record field that repeats
            the same id. The rest of the record, including offsets, scales,
            sizes and source/time, is still compared.

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

    kept = [_kept_row(row, ignored) for row in rows]

    return kept


def _kept_row(row: dict, ignored: Collection[str]) -> dict:
    kept = {key: value for key, value in row.items() if key not in ignored}
    record = kept.get(FRAME_RECORD_KEY)
    if isinstance(record, dict):
        kept[FRAME_RECORD_KEY] = _record_without_identities(record, ignored)

    return kept


def _record_without_identities(record: dict, ignored: Collection[str]) -> dict:
    masked = dict(record)
    for key in ignored:
        if key not in _RECORD_IDENTITY_FIELDS:
            continue

        section, field = _RECORD_IDENTITY_FIELDS[key]
        if section is None:
            masked[field] = _IGNORED
        elif isinstance(masked.get(section), dict):
            masked[section] = {**masked[section], field: _IGNORED}

    return masked

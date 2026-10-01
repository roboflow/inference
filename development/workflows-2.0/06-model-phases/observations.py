"""Read delivered results into small JSON-friendly facts and check them.

Nothing here builds engine values; it only reads what the engine delivered.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from native_comparison import compare_predictions
from resnet18 import top_classes
from roboflow_workflows.execution_engine.v2.data import Batch, EntryMetadata

from inference_models.models.base.classification import ClassificationPrediction

Index = Tuple[int, ...]


class ObservationMismatch(AssertionError):
    """An example observed something other than what it documents."""


def expect(label: str, actual: Any, expected: Any) -> Any:
    """Raise when an observation differs from the documented value.

    Args:
        label: What is being checked, shown on mismatch.
        actual: Observed value.
        expected: Documented value.

    Returns:
        ``actual``, so checks can be recorded inline.

    Raises:
        ObservationMismatch: When the values differ.
    """
    if actual != expected:
        raise ObservationMismatch(f"{label}: expected {expected!r}, got {actual!r}")

    return actual


def leaves(value: Any, index: Index = ()) -> List[Tuple[Index, Any]]:
    """Flatten nested ``Batch`` values into ``(logical index, leaf)`` pairs.

    Args:
        value: Delivered output value.
        index: Index of ``value`` itself.

    Returns:
        Leaves in index order; a non-batch value is one leaf at ``index``.
    """
    if not isinstance(value, Batch):
        return [(index, value)]

    flattened = []
    for child_index, child in value.iter_with_indices():
        flattened.extend(leaves(child, tuple(child_index)))

    return flattened


def media_pts_ms(metadata: EntryMetadata, index: Index) -> Optional[float]:
    """Return the media PTS in milliseconds the engine attached at ``index``.

    Args:
        metadata: Entry metadata as delivered.
        index: Logical index; ``()`` is the whole entry.

    Returns:
        Milliseconds, or None when no temporal context applies.
    """
    timing = metadata.temporal_at(index)
    if timing is None or timing.media_coverage is None:
        return None

    stamp = timing.media_coverage
    milliseconds = stamp.ticks * stamp.time_base * 1000
    pts = int(milliseconds) if milliseconds.denominator == 1 else float(milliseconds)

    return pts


def describe_prediction(prediction: ClassificationPrediction, k: int = 3) -> Dict:
    """Summarize a native prediction: top classes and where its image came from.

    Args:
        prediction: Single-image classification prediction.
        k: Number of top classes.

    Returns:
        Top classes with rounded confidences and the provenance keys.
    """
    (metadata,) = prediction.images_metadata
    summary = {
        "top": [
            {
                "class_name": item["class_name"],
                "confidence": round(item["confidence"], 4),
            }
            for item in top_classes(prediction, k)
        ],
        "image": {
            key: metadata[key]
            for key in (
                "parent_id",
                "root_parent_id",
                "image_dimensions",
                "root_parent_coordinates",
            )
        },
    }

    return summary


def compare_outputs(
    expected: Any,
    actual: Any,
    *,
    tolerance: Optional[float] = None,
    ignore_metadata: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Compare every native prediction of two outputs with the same layout.

    Args:
        expected: Reference output value (possibly nested ``Batch``).
        actual: Output value to check.
        tolerance: ``None`` for bitwise confidences, else the absolute bound.
        ignore_metadata: Metadata keys that legitimately differ between the
            two runs, such as generated crop ids.

    Returns:
        One comparison record per logical index.

    Raises:
        ObservationMismatch: When the indices or any prediction differ.
    """
    expected_leaves = leaves(expected)
    actual_leaves = leaves(actual)
    expect(
        "logical indices",
        [index for index, _ in actual_leaves],
        [index for index, _ in expected_leaves],
    )
    records = []
    for (index, reference), (_, candidate) in zip(expected_leaves, actual_leaves):
        comparison = compare_predictions(
            reference,
            candidate,
            tolerance=tolerance,
            ignore_metadata=ignore_metadata,
        )
        record = {"index": list(index), **comparison}
        expect(
            f"prediction at {list(index)} matches ({record['rule']})",
            record["equal"],
            True,
        )
        records.append(record)

    return records


def check_own_crops(predictions: Any, crops: Any) -> int:
    """Check that each crop prediction's metadata names the crop it classified.

    Args:
        predictions: Crop predictions of one run (nested ``Batch``).
        crops: The crops of the same run, at the same logical indices.

    Returns:
        Number of predictions checked.

    Raises:
        ObservationMismatch: When a prediction names another image.
    """
    crop_ids = {index: crop.image_id for index, crop in leaves(crops)}
    checked = leaves(predictions)
    for index, prediction in checked:
        expect(
            f"prediction {list(index)} names its crop",
            prediction.images_metadata[0]["parent_id"],
            crop_ids[index],
        )

    return len(checked)


def phase_events(trace: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group the phase trace events of a run by step and logical index.

    Args:
        trace: ``RunResult.trace`` or a group result's trace.

    Returns:
        ``"step@index"`` to the phases run for that invocation, in order.
    """
    grouped: Dict[str, List[str]] = defaultdict(list)
    for event in trace:
        if event.get("event") != "phase":
            continue
        step = "/".join(event["step"])
        for index in event["indices"]:
            grouped[f"{step}@{list(index)}"].append(event["phase"])

    return dict(grouped)

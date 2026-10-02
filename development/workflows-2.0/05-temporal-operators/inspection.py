"""Turn delivered results into JSON-friendly records of layout and metadata.

This module only reads what the engine delivered. It never builds buffers,
contexts or layouts; the demo shows them exactly as the engine produced them.
"""

from typing import Any, Dict, List, Optional

from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.data import Batch, EntryMetadata, Timestamp


def timestamp_ms(timestamp: Optional[Timestamp]) -> Optional[float]:
    """Express a timestamp in milliseconds, keeping exact integers.

    Args:
        timestamp: Timestamp or None.

    Returns:
        Milliseconds, or None when there is no timestamp.
    """
    if timestamp is None:
        return None

    milliseconds = timestamp.ticks * timestamp.time_base * 1000
    value = int(milliseconds) if milliseconds.denominator == 1 else float(milliseconds)

    return value


def media_pts(metadata: EntryMetadata, index: tuple) -> Optional[float]:
    """Resolve the media PTS the engine attached at one logical index.

    Args:
        metadata: Entry metadata as delivered.
        index: Logical index path; ``()`` is the whole entry.

    Returns:
        Media PTS in milliseconds, or None when no context applies.
    """
    timing = metadata.temporal_at(index)
    pts = None if timing is None else timestamp_ms(timing.media_coverage)

    return pts


def describe_value(value: Any) -> Any:
    """Summarize one leaf payload.

    Args:
        value: Leaf payload of an entry.

    Returns:
        Image id, shape and mean for images; the value itself otherwise.
    """
    if not isinstance(value, ImageData):
        return value

    summary = {
        "image_id": value.image_id,
        "shape": list(value.tensor_image.shape),
        "mean": round(value.tensor_image.float().mean().item()),
    }

    return summary


def _leaves(value: Any, index: tuple) -> List[tuple]:
    if not isinstance(value, Batch):
        return [(index, value)]

    leaves = []
    for child_index, child in value.iter_with_indices():
        leaves.extend(_leaves(child, child_index))

    return leaves


def describe_entry(*, value: Any, layout, metadata: EntryMetadata) -> Dict[str, Any]:
    """Describe one delivered entry: axes, root context and every leaf.

    Args:
        value: Payload or nested ``Batch``.
        layout: ``EntryLayout`` of the entry.
        metadata: ``EntryMetadata`` of the entry.

    Returns:
        Axis ids/kinds, root PTS and per-leaf index, source, PTS and value.
    """
    root_sample = metadata.sample_at(())
    leaves = []
    for index, leaf in _leaves(value, ()):
        sample = metadata.sample_at(index)
        leaves.append(
            {
                "index": list(index),
                "source": None if sample is None else sample.source_id,
                "pts_ms": media_pts(metadata, index),
                "value": describe_value(leaf),
            }
        )

    record = {
        "axes": [
            {"id": axis.id, "kind": axis.kind, "stationary": axis.stationary}
            for axis in layout.axes
        ],
        "root_pts_ms": media_pts(metadata, ()),
        "root_source": None if root_sample is None else root_sample.source_id,
        "leaves": leaves,
    }

    return record


def observe_fields(result) -> Dict[str, Any]:
    """Record every selected field of a ``GroupResult`` or ``RunResult``.

    Args:
        result: Delivered group result or passive run result.

    Returns:
        Field name to status, filtered paths and entry description.
    """
    fields = {}
    for name, selection in result.selections.items():
        (entry,) = selection.values()
        status = result.statuses[entry]
        record = {
            "status": status,
            "filtered": [list(path) for path in result.filtered_paths.get(entry, ())],
        }
        if status == "complete":
            record.update(
                describe_entry(
                    value=result.outputs.data[entry],
                    layout=result.outputs.layout[entry],
                    metadata=result.outputs.metadata[entry],
                )
            )
        fields[name] = record

    return fields


def observe(result) -> Dict[str, Any]:
    """Record one ``GroupResult`` with its pulse identity and every field.

    Args:
        result: Group result delivered to a handler.

    Returns:
        Group, anchor domain, pulse, causes and per-field descriptions.
    """
    causes = [
        {"domain": cause.source, "sequence": cause.sequence} for cause in result.causes
    ]
    observation = {
        "group": result.group,
        "anchor": result.source,
        "pulse": result.pulse.sequence,
        "causes": causes,
        "fields": observe_fields(result),
    }

    return observation


def leaf_pts(observation: Dict[str, Any], field: str) -> List[Optional[float]]:
    """List the leaf PTS values of one observed field, in index order.

    Args:
        observation: Record returned by ``observe``.
        field: Field name inside the observed group.

    Returns:
        PTS in milliseconds per leaf.
    """
    values = [leaf["pts_ms"] for leaf in observation["fields"][field]["leaves"]]

    return values


def axis_kinds(observation: Dict[str, Any], field: str) -> List[str]:
    """List the axis kinds of one observed field, outermost first.

    Args:
        observation: Record returned by ``observe``.
        field: Field name inside the observed group.

    Returns:
        Axis kinds such as ``["sample", "time"]``.
    """
    kinds = [axis["kind"] for axis in observation["fields"][field]["axes"]]

    return kinds

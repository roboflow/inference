"""Turn real V2 run results into JSON documents and PNG artifacts.

Nothing here executes, groups or gates workflow data. The functions only walk
the ``Batch`` trees, layouts, metadata, statuses and trace returned by
``session.run`` and write them down.
"""

import dataclasses
import json
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from PIL import Image
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryLayout,
    EntryMetadata,
)

PayloadDescriber = Callable[[Any, str], Any]


def save_png(image: np.ndarray, path: Path) -> None:
    """Write an RGB ``uint8`` array as a PNG file.

    Args:
        image: Array of shape ``(height, width, 3)``.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(path)


def describe_layout(layout: EntryLayout) -> List[Dict[str, Any]]:
    """Serialize an entry layout.

    Args:
        layout: Layout of one named entry.

    Returns:
        One mapping per axis with ``id``, ``kind`` and ``stationary``.
    """
    axes = [
        {"id": axis.id, "kind": axis.kind, "stationary": axis.stationary}
        for axis in layout.axes
    ]

    return axes


def describe_tree(
    value: Any,
    *,
    metadata: EntryMetadata,
    describe_payload: PayloadDescriber,
    name: str,
) -> Dict[str, Any]:
    """Walk a payload or nested ``Batch`` and record groups, items and context.

    Args:
        value: Entry data: a payload or a (nested) ``Batch``.
        metadata: The entry's indexed metadata; contexts are resolved per item
            with ``sample_at``/``temporal_at``.
        describe_payload: Callback ``(payload, label) -> JSON`` for leaves.
        name: Entry name used to label artifacts.

    Returns:
        Mapping with ``groups`` (every ``Batch`` encountered, including empty
        ones, with its parent index and size) and ``items`` (every leaf with
        its full logical index, payload description and resolved contexts).
    """
    groups: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []

    def visit(node: Any, index: tuple) -> None:
        if isinstance(node, Batch):
            groups.append(
                {
                    "parent_index": list(node.parent_index),
                    "size": len(node),
                    "child_indices": [list(child) for child in node.indices],
                }
            )
            for child_index, child in node.iter_with_indices():
                visit(child, child_index)
            return

        label = f"{name}_" + "_".join(str(component) for component in index)
        if not index:
            label = name
        sample = metadata.sample_at(index)
        temporal = metadata.temporal_at(index)
        items.append(
            {
                "index": list(index),
                "payload": describe_payload(node, label),
                "sample_context": _plain(sample),
                "temporal_context": _plain(temporal),
            }
        )

    visit(value, ())
    tree = {"groups": groups, "items": items}

    return tree


def batch_to_nested(value: Any) -> Any:
    """Convert a ``Batch`` tree into nested Python lists of leaves.

    Args:
        value: Payload or nested ``Batch``.

    Returns:
        The payload itself, or a list (possibly nested) of converted children.
    """
    if isinstance(value, Batch):
        nested = [batch_to_nested(child) for child in value]
        return nested

    return value


def batch_local_indices(value: Any) -> Any:
    """Return the trailing index component of every leaf, nested like the data.

    Args:
        value: Payload or nested ``Batch``.

    Returns:
        ``None`` for a bare payload, otherwise nested lists whose innermost
        entries are the last component of each child's full index.
    """
    if not isinstance(value, Batch):
        return None

    local: List[Any] = []
    for index, child in value.iter_with_indices():
        if isinstance(child, Batch):
            local.append(batch_local_indices(child))
        else:
            local.append(index[-1])

    return local


def describe_result(
    result: Any,
    *,
    describe_payload: PayloadDescriber,
) -> Dict[str, Any]:
    """Serialize a ``RunResult`` obtained from ``session.run``.

    Args:
        result: ``RunResult`` with ``outputs`` (``WorkflowsBuffer``),
            ``selections``, ``statuses``, ``trace`` and run/session ids.
        describe_payload: Callback used for leaf payloads.

    Returns:
        JSON-friendly mapping with identity, which entry each workflow output
        selects, per-entry layout/tree/context, statuses and the trace.
    """
    outputs = result.outputs
    entries: Dict[str, Any] = {}
    for name in outputs.entry_names:
        entries[name] = {
            "layout": describe_layout(outputs.layout[name]),
            "metadata_keys": {
                "sample": [list(index) for index in outputs.metadata[name].sample],
                "temporal": [list(index) for index in outputs.metadata[name].temporal],
            },
            "tree": describe_tree(
                outputs.data[name],
                metadata=outputs.metadata[name],
                describe_payload=describe_payload,
                name=name,
            ),
        }

    document = {
        "session_id": result.session_id,
        "run_id": result.run_id,
        "lineage_id": outputs.lineage_id,
        "pulse_id": outputs.pulse_id,
        "selections": {name: dict(ports) for name, ports in result.selections.items()},
        "statuses": dict(result.statuses),
        "outputs": entries,
        "trace": list(result.trace),
    }

    return document


def image_describer(artifact_dir: Path) -> PayloadDescriber:
    """Create a leaf describer that saves image payloads as PNG files.

    Args:
        artifact_dir: Directory receiving ``<label>.png`` files.

    Returns:
        Callback usable as ``describe_payload``.
    """

    def describe(payload: Any, label: str) -> Any:
        if isinstance(payload, np.ndarray):
            path = artifact_dir / f"{label}.png"
            save_png(payload, path)
            description = {
                "kind": "image",
                "shape": list(payload.shape),
                "mean": round(float(payload.mean()), 2),
                "png": str(path),
            }
            return description

        return _plain(payload)

    return describe


def plain_describer(payload: Any, label: str) -> Any:
    """Describe a non-image leaf payload as plain JSON.

    Args:
        payload: Leaf payload.
        label: Unused label.

    Returns:
        JSON-friendly value.
    """
    return _plain(payload)


def write_json(document: Any, path: Path) -> None:
    """Write ``document`` as indented JSON.

    Args:
        document: JSON-friendly value (fractions, dataclasses, NumPy scalars
            and read-only mappings are converted).
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=_json_default) + "\n")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
        fields["type"] = type(value).__name__
        return fields
    if isinstance(value, (Mapping, MappingProxyType)):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]

    return repr(value)


def _json_default(value: Any) -> Any:
    return _plain(value)

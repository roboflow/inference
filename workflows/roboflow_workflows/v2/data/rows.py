"""Opt-in carrier of a native row prediction plus aligned numeric columns.

``PredictionRows`` pairs one native prediction with named ``(N, ...)``
tensors, such as tracker ids or embeddings, that must stay aligned with its
rows::

    rows = PredictionRows(native=detections, columns={"tracker_id": ids})
    kept = rows.select(indices=[2, 0, 2])   # native fields and columns together
    kept.frame                              # frame_of(kept.native)

Nothing wraps predictions automatically; existing kinds keep accepting the
native objects. Classification, semantic and depth results have no instance
rows and are not row carriers.

Every value has one authority. A column name may not equal a native field
name, and may not be a key of any native row-metadata dict, so a value such
as ``tracker_id`` lives either in ``bboxes_metadata`` or in a column, never in
both. Strings and other host values stay in the native row metadata. Columns
live on the native prediction's device, keep their strides and are not
copied; they are read-only by contract. The frame is read from the native
prediction, never stored twice.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
from roboflow_workflows.v2.data.coordinates import convert_prediction_output
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.kinds import Kind
from roboflow_workflows.v2.data.payload_codec import (
    decode_native,
    encode_native,
    is_native_envelope,
    tensor_dtype_problem,
)
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    Selection,
    take_rows,
    validated_row_count,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

_NATIVE_FIELD_NAMES = frozenset(
    native_field
    for carrier in (Detections, InstanceDetections, KeyPoints)
    for native_field in carrier.__dataclass_fields__
)
_ENVELOPE_TAG = "prediction_rows"


@dataclass(frozen=True, eq=False)
class PredictionRows:
    """A native row prediction with aligned numeric columns.

    Args:
        native: ``Detections``, ``InstanceDetections``, ``KeyPoints`` or a
            ``(KeyPoints, Optional[Detections])`` tuple.
        columns: Column name to ``torch.Tensor`` with ``N`` leading rows on
            the native device. Stored in a read-only mapping; the tensors are
            not copied.

    Raises:
        ContractError: When the native payload is malformed, or a column is
            not a tensor of ``N`` rows on the native device, has an
            unsupported dtype, or duplicates a native field or row-metadata
            key.
    """

    native: Any
    columns: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.columns, Mapping):
            raise ContractError(
                f"columns must be a mapping, got {type(self.columns).__name__}"
            )

        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        self.check()

    def __len__(self) -> int:
        rows = _native_row_count(self.native)

        return rows

    @property
    def device(self) -> torch.device:
        """Device of the native prediction and every column."""
        return _native_device(self.native)

    @property
    def frame(self) -> FrameRef:
        """Frame of the native prediction; see ``frames.frame_of``."""
        return frame_of(self.native)

    def check(self) -> None:
        """Validate the native payload and the column alignment.

        Reads tensor shapes, dtypes and devices and the host row metadata;
        never tensor values.

        Raises:
            ContractError: As described for the constructor.
        """
        rows = validated_row_count(self.native)
        device = _native_device(self.native)
        metadata_keys = _row_metadata_keys(self.native)
        for name, column in self.columns.items():
            _check_column(
                name,
                column,
                rows=rows,
                device=device,
                metadata_keys=metadata_keys,
            )

    def select(
        self,
        *,
        mask: Optional[Selection] = None,
        indices: Optional[Selection] = None,
    ) -> "PredictionRows":
        """Select rows of the native prediction and every column together.

        Args:
            mask: One-dimensional boolean selection, one entry per row.
            indices: One-dimensional integer positions; order and repeats are
                kept.

        Returns:
            A new carrier. One ``RowSelection`` drives every field, so the
            host work is that of ``select_predictions``, paid once.

        Raises:
            ContractError: When the selection is malformed.
        """
        selection = RowSelection.of(len(self), mask=mask, indices=indices)
        selected = PredictionRows(
            native=take_rows(self.native, selection),
            columns={
                name: selection.take(column) for name, column in self.columns.items()
            },
        )

        return selected


def _native_row_count(native: Any) -> int:
    key_points = native[0] if isinstance(native, tuple) else native
    rows = len(key_points)

    return rows


def _native_device(native: Any) -> torch.device:
    if isinstance(native, tuple):
        native = native[0]
    tensor = native.xy if isinstance(native, KeyPoints) else native.xyxy

    return tensor.device


def _row_metadata_keys(native: Any) -> frozenset:
    if isinstance(native, tuple):
        key_points, detections = native
        row_dicts = [
            *(key_points.key_points_metadata or ()),
            *(() if detections is None else detections.bboxes_metadata or ()),
        ]
    elif isinstance(native, KeyPoints):
        row_dicts = native.key_points_metadata or ()
    else:
        row_dicts = native.bboxes_metadata or ()

    keys = frozenset(key for row in row_dicts for key in row)

    return keys


def _check_column(
    name: Any,
    column: Any,
    *,
    rows: int,
    device: torch.device,
    metadata_keys: frozenset,
) -> None:
    if not isinstance(name, str) or not name:
        raise ContractError(f"Column names must be non-empty strings, got {name!r}")
    if name in _NATIVE_FIELD_NAMES:
        raise ContractError(
            f"Column '{name}' would shadow the native field of the same name"
        )
    if name in metadata_keys:
        raise ContractError(
            f"Column '{name}' is also a key of the native row metadata; keep one "
            "authority by moving the values explicitly into one of them"
        )
    if not isinstance(column, torch.Tensor):
        raise ContractError(
            f"Column '{name}' must be a torch.Tensor, got {type(column).__name__}; "
            "keep strings and other host values in the native row metadata"
        )
    if column.ndim == 0 or column.shape[0] != rows:
        raise ContractError(
            f"Column '{name}' must have {rows} leading rows, got shape "
            f"{tuple(column.shape)}"
        )
    if column.device != device:
        raise ContractError(
            f"Column '{name}' is on {column.device}, but the native prediction is "
            f"on {device}"
        )

    dtype_problem = tensor_dtype_problem(column.dtype)
    if dtype_problem is not None:
        raise ContractError(f"Column '{name}': {dtype_problem}")


# ---------------------------------------------------------------------------
# Kind
# ---------------------------------------------------------------------------


def _validate(payload: Any) -> bool:
    if not isinstance(payload, PredictionRows):
        raise ContractError(f"Expected PredictionRows, got {type(payload).__name__}")

    payload.check()

    return True


def _serialize(payload: Any) -> dict:
    _validate(payload)
    envelope = encode_native(
        {_ENVELOPE_TAG: {"native": payload.native, "columns": dict(payload.columns)}}
    )

    return envelope


def _deserialize(value: Any) -> PredictionRows:
    if isinstance(value, PredictionRows):
        return value
    if not is_native_envelope(value):
        raise ContractError(
            f"Expected PredictionRows or a native envelope, got {type(value).__name__}"
        )

    decoded = decode_native(value)
    is_tagged = isinstance(decoded, dict) and set(decoded) == {_ENVELOPE_TAG}
    body = decoded[_ENVELOPE_TAG] if is_tagged else None
    if (
        not isinstance(body, dict)
        or set(body) != {"native", "columns"}
        or not isinstance(body["columns"], dict)
    ):
        raise ContractError(
            "A prediction_rows envelope holds {'prediction_rows': {'native', "
            "'columns'}}"
        )

    rows = PredictionRows(native=body["native"], columns=body["columns"])

    return rows


def _convert_output(payload: PredictionRows, options: Mapping[str, Any]) -> Any:
    # Only the native part has geometry; numeric columns are frame-free.
    native = convert_prediction_output(payload.native, options)
    if native is payload.native:
        return payload

    converted = PredictionRows(native=native, columns=payload.columns)

    return converted


PREDICTION_ROWS_KIND = Kind(
    name="prediction_rows",
    description=(
        "PredictionRows: a native Detections, InstanceDetections, KeyPoints or "
        "(KeyPoints, Detections | None) payload with aligned (N, ...) tensor "
        "columns on its device."
    ),
    validate=_validate,
    deserialize=_deserialize,
    serialize=_serialize,
    convert_output=_convert_output,
)

__all__ = ["PREDICTION_ROWS_KIND", "PredictionRows"]

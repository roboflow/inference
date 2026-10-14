"""Semantic row identity of native predictions.

Engine indices describe graph nesting, and boxes describe geometry; neither
identifies a detection row after it was filtered, sorted or repeated. A row
key ``(namespace, id)`` does. A row-bearing prediction may carry one reserved
record in the ``image_metadata`` of its authority carrier::

    image_metadata["workflows_v2_rows"] = {
        "version": 1,
        "namespace": str,   # non-empty; one per independent result
        "rows": int,        # == len(prediction)
        "ids": None | torch.int64 tensor (rows,) on the rows' device,
    }

=================================  ===========================================
record state                       meaning
=================================  ===========================================
absent                             provenance unknown; ordinary predictions
                                   stay valid, correspondence consumers fail
``ids`` is ``None``                rows are ``0 .. rows - 1`` in order
``ids`` is a tensor                row ``i`` is source row ``ids[i]``;
                                   repeats are allowed
=================================  ===========================================

=================================  ===========================================
payload                            authority carrier
=================================  ===========================================
Detections, InstanceDetections     itself
KeyPoints, (KeyPoints, None)       KeyPoints
(KeyPoints, Detections)            Detections; the KeyPoints hold no record
=================================  ===========================================

Only shared row selection (``prediction_ops``) carries a record to a new row
set; ``rows`` catches carriers rebuilt by hand with another row count. A crop
of one row refers to it with a ``RowRef``; classifications store that
reference under ``PARENT_ROW_KEY``. Checks read shapes, dtypes and devices,
never tensor values. This module imports no other V2 data module, so image,
kind and codec modules can all depend on it.
"""

import uuid
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from roboflow_workflows.v2.data.errors import ContractError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

ROW_RECORD_KEY = "workflows_v2_rows"
PARENT_ROW_KEY = "workflows_v2_parent_row"
ROW_RECORD_VERSION = 1

_RECORD_KEYS = frozenset({"version", "namespace", "rows", "ids"})
_ROW_REF_KEYS = frozenset({"namespace", "id"})


@dataclass(frozen=True)
class RowRef:
    """One semantic source row: a row id within one result namespace.

    Args:
        namespace: Non-empty namespace of the result the row belongs to.
        id: Non-negative row id within that result.

    Raises:
        ContractError: On an empty namespace or a non-integer or negative id.
    """

    namespace: str
    id: int

    def __post_init__(self) -> None:
        _check_namespace(self.namespace, name="RowRef.namespace")
        if not _is_integer(self.id) or self.id < 0:
            raise ContractError(
                f"RowRef.id must be a non-negative integer, got {self.id!r}"
            )
        object.__setattr__(self, "id", int(self.id))

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible form ``{"namespace", "id"}``."""
        value = {"namespace": self.namespace, "id": self.id}

        return value

    @classmethod
    def from_dict(cls, value: Any) -> "RowRef":
        """Read the form written by ``to_dict``.

        Args:
            value: Mapping with exactly the keys ``namespace`` and ``id``.

        Returns:
            The reference.

        Raises:
            ContractError: When the keys or values are malformed.
        """
        if not isinstance(value, Mapping) or set(value) != _ROW_REF_KEYS:
            raise ContractError(
                "A row reference must be a dict with exactly the keys "
                f"{sorted(_ROW_REF_KEYS)}, got {_describe(value)}"
            )

        reference = cls(namespace=value["namespace"], id=value["id"])

        return reference


@dataclass(frozen=True)
class RowRecord:
    """Validated view of a prediction's row record.

    Attributes:
        namespace: Namespace shared by every row of the result.
        rows: Number of rows.
        ids: ``None`` for rows ``0 .. rows - 1`` in order, otherwise an
            ``int64`` ``(rows,)`` tensor on the rows' device. Read-only.
    """

    namespace: str
    rows: int
    ids: Optional[torch.Tensor]

    def host_ids(self) -> List[int]:
        """Row ids as Python ints, in row order.

        Returns:
            ``list(range(rows))`` without any transfer when ``ids`` is
            ``None``; otherwise the ids after one device-to-host copy.
        """
        if self.ids is None:
            return list(range(self.rows))

        ids = self.ids.cpu().tolist()

        return ids

    def host_ids_with(self, columns: torch.Tensor) -> Tuple[List[int], List[List[int]]]:
        """Read row ids together with integer row columns in one host copy.

        Use it where the columns must reach the host anyway, such as rounded
        crop rectangles, so that the ids add no transfer of their own.

        Args:
            columns: Integer ``(rows, k)`` tensor on the ids' device.

        Returns:
            The row ids and the columns as nested Python ints.

        Raises:
            ContractError: When ``columns`` is not an integer ``(rows, k)``
                tensor on the ids' device.
        """
        _check_columns(columns, rows=self.rows)
        if self.ids is None:
            host_columns = columns.cpu().tolist()
            return list(range(self.rows)), host_columns

        if columns.device != self.ids.device:
            raise ContractError(
                f"columns are on {columns.device}, but the row ids are on "
                f"{self.ids.device}"
            )
        joined = torch.cat([self.ids.unsqueeze(1), columns.to(torch.int64)], dim=1)
        host_rows = joined.cpu().tolist()
        ids = [row[0] for row in host_rows]
        host_columns = [row[1:] for row in host_rows]

        return ids, host_columns

    def refs(self) -> List[RowRef]:
        """Return one ``RowRef`` per row, in row order (see ``host_ids``)."""
        references = [RowRef(self.namespace, row_id) for row_id in self.host_ids()]

        return references


def with_row_identity(prediction: Any, *, namespace: Optional[str] = None) -> Any:
    """Attach a fresh row identity to a prediction that has none.

    Call it once per independent result: in the model block that produced
    it, or for user input once before workflow branches diverge. Two separate
    attachments can never be joined. Runs no tensor operation.

    Args:
        prediction: Detections, InstanceDetections, KeyPoints or a
            ``(KeyPoints, Detections | None)`` tuple.
        namespace: Namespace of the result; a new random one when omitted.

    Returns:
        A prediction of the same family whose authority carrier has a
        shallow-copied ``image_metadata`` with ``ids=None``. Tensors and the
        other carriers are shared; the input is not changed. Carrier fields
        are not validated here; kind checks do that.

    Raises:
        ContractError: When the payload is unsupported, a carrier already has
            a row record, or ``namespace`` is empty.
    """
    carriers = _carriers(prediction)
    for carrier in carriers:
        metadata = _image_metadata(carrier)
        if ROW_RECORD_KEY in metadata:
            raise ContractError(
                "The prediction already has a row identity "
                f"(namespace {_namespace_of(metadata[ROW_RECORD_KEY])!r}); attach "
                "one only once per result, before workflow branches diverge"
            )

    namespace = uuid.uuid4().hex if namespace is None else namespace
    _check_namespace(namespace, name="namespace")
    authority = carriers[-1]
    record = {
        "version": ROW_RECORD_VERSION,
        "namespace": namespace,
        "rows": _row_count(authority),
        "ids": None,
    }
    stamped = replace(
        authority,
        image_metadata={**_image_metadata(authority), ROW_RECORD_KEY: record},
    )
    identified = _with_authority(prediction, stamped)

    return identified


def row_record(prediction: Any) -> Optional[RowRecord]:
    """Read a prediction's row record.

    Args:
        prediction: A payload accepted by ``with_row_identity``.

    Returns:
        The validated record, or ``None`` when the prediction has none.

    Raises:
        ContractError: When the payload is unsupported or its record is
            malformed or conflicting.
    """
    carriers = _carriers(prediction)
    authority = carriers[-1]
    record = checked_row_record(
        authority.image_metadata,
        rows=_row_count(authority),
        device=_row_device(authority),
        name=f"{type(authority).__name__}.image_metadata",
    )
    if len(carriers) == 2:
        check_no_key_points_record(carriers[0].image_metadata)

    return record


def require_row_record(prediction: Any, *, consumer: str) -> RowRecord:
    """Read the row record that a correspondence consumer depends on.

    Args:
        prediction: A payload accepted by ``with_row_identity``.
        consumer: Name of the consuming block, used in the error.

    Returns:
        The validated record.

    Raises:
        ContractError: When the record is missing (also for empty
            predictions), malformed or conflicting.
    """
    record = row_record(prediction)
    if record is None:
        raise ContractError(
            f"{consumer} needs predictions with a row identity to match rows. "
            "Model blocks attach one; for other predictions apply "
            "with_row_identity once, before workflow branches diverge"
        )

    return record


def without_row_record(
    image_metadata: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Drop the row record from metadata copied onto a different row set.

    Args:
        image_metadata: Prediction image metadata, or ``None``.

    Returns:
        The same object when it holds no record, otherwise a shallow copy
        without it.
    """
    if image_metadata is None or ROW_RECORD_KEY not in image_metadata:
        return image_metadata

    stripped = {
        key: value for key, value in image_metadata.items() if key != ROW_RECORD_KEY
    }

    return stripped


def with_classification_parent(prediction: Any, parent_row: Optional[RowRef]) -> Any:
    """Record which detection row a classification was made for.

    Args:
        prediction: ``ClassificationPrediction`` of one image (stored in
            ``images_metadata[0]``) or ``MultiLabelClassificationPrediction``
            (stored in ``image_metadata``).
        parent_row: ``ImageData.parent_row`` of the classified image; ``None``
            for an image that is not a detection crop.

    Returns:
        The prediction itself when ``parent_row`` is ``None`` or already
        recorded, otherwise a copy with shallow-copied metadata.

    Raises:
        ContractError: On an unsupported payload or ``parent_row``, or when
            another parent row is already recorded.
    """
    if parent_row is None:
        return prediction
    if not isinstance(parent_row, RowRef):
        raise ContractError(
            f"parent_row must be a RowRef or None, got {type(parent_row).__name__}"
        )

    existing = classification_parent(prediction)
    if existing == parent_row:
        return prediction
    if existing is not None:
        raise ContractError(
            f"The classification already refers to row {existing}; cannot "
            f"refer it to {parent_row}"
        )

    reference = {PARENT_ROW_KEY: parent_row.to_dict()}
    if isinstance(prediction, ClassificationPrediction):
        metadata = list(prediction.images_metadata or [{}])
        metadata[0] = {**metadata[0], **reference}
        updated = replace(prediction, images_metadata=metadata)
        return updated

    updated = replace(
        prediction, image_metadata={**(prediction.image_metadata or {}), **reference}
    )

    return updated


def classification_parent(prediction: Any) -> Optional[RowRef]:
    """Read the detection row a classification was made for.

    Args:
        prediction: ``ClassificationPrediction`` of one image or
            ``MultiLabelClassificationPrediction``.

    Returns:
        The reference, or ``None`` when the classified image was not a
        detection crop.

    Raises:
        ContractError: On an unsupported payload or a malformed reference.
    """
    metadata = _classification_metadata(prediction)
    reference = checked_parent_row(
        metadata, name=f"{type(prediction).__name__} metadata"
    )

    return reference


def checked_row_record(
    image_metadata: Optional[Mapping[str, Any]],
    *,
    rows: int,
    device: torch.device,
    name: str,
) -> Optional[RowRecord]:
    """Validate the row record in one carrier's image metadata.

    Reads the ids tensor's dtype, shape and device only, never its values.

    Args:
        image_metadata: The carrier's ``image_metadata``.
        rows: Number of rows of the carrier.
        device: Device of the carrier's row tensors.
        name: Name used in errors.

    Returns:
        The record, or ``None`` when the metadata holds none.

    Raises:
        ContractError: When the record is malformed or stale.
    """
    if image_metadata is None or ROW_RECORD_KEY not in image_metadata:
        return None

    record = image_metadata[ROW_RECORD_KEY]
    name = f"{name}[{ROW_RECORD_KEY!r}]"
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ContractError(
            f"{name} must be a dict with exactly the keys {sorted(_RECORD_KEYS)}, "
            f"got {_describe(record)}"
        )
    version = record["version"]
    if not _is_integer(version) or version != ROW_RECORD_VERSION:
        raise ContractError(
            f"{name} has version {version!r}; only {ROW_RECORD_VERSION} is known"
        )
    _check_namespace(record["namespace"], name=f"{name}.namespace")
    recorded_rows = record["rows"]
    if not _is_integer(recorded_rows) or recorded_rows != rows:
        raise ContractError(
            f"{name} describes {recorded_rows!r} rows, but the prediction has "
            f"{rows}; a prediction rebuilt with other rows must drop the record "
            "(without_row_record) or select through prediction_ops"
        )
    _check_ids(record["ids"], rows=rows, device=device, name=f"{name}.ids")

    checked = RowRecord(namespace=record["namespace"], rows=rows, ids=record["ids"])

    return checked


def check_no_key_points_record(
    key_points_metadata: Optional[Mapping[str, Any]],
) -> None:
    """Reject a record on the KeyPoints of a tuple with Detections.

    The Detections hold the tuple's only record. Two copies could disagree,
    and object identity between them does not survive serialization.

    Args:
        key_points_metadata: ``KeyPoints.image_metadata``.

    Raises:
        ContractError: When the KeyPoints hold a row record.
    """
    if key_points_metadata is None or ROW_RECORD_KEY not in key_points_metadata:
        return

    raise ContractError(
        "A (KeyPoints, Detections) prediction keeps its row record on the "
        "Detections only; the KeyPoints hold one too"
    )


def checked_parent_row(
    metadata: Optional[Mapping[str, Any]], *, name: str
) -> Optional[RowRef]:
    """Validate the parent row reference in classification metadata.

    Args:
        metadata: The classification's image metadata.
        name: Name used in errors.

    Returns:
        The reference, or ``None`` when the metadata holds none.

    Raises:
        ContractError: When the reference is malformed.
    """
    if metadata is None or PARENT_ROW_KEY not in metadata:
        return None

    try:
        reference = RowRef.from_dict(metadata[PARENT_ROW_KEY])
    except ContractError as error:
        raise ContractError(f"{name}[{PARENT_ROW_KEY!r}]: {error}") from error

    return reference


def _carriers(prediction: Any) -> Tuple[Any, ...]:
    # Carriers in tuple order; the authority carrier is last.
    if isinstance(prediction, (Detections, InstanceDetections, KeyPoints)):
        return (prediction,)

    if isinstance(prediction, tuple) and len(prediction) == 2:
        key_points, detections = prediction
        if isinstance(key_points, KeyPoints) and detections is None:
            return (key_points,)
        if isinstance(key_points, KeyPoints) and isinstance(detections, Detections):
            if _row_count(key_points) != _row_count(detections):
                raise ContractError(
                    f"KeyPoints has {_row_count(key_points)} instances but its "
                    f"Detections has {_row_count(detections)} boxes"
                )
            return (key_points, detections)

    raise ContractError(
        "Row identity applies to Detections, InstanceDetections, KeyPoints or "
        f"(KeyPoints, Detections | None), got {type(prediction).__name__}"
    )


def _with_authority(prediction: Any, authority: Any) -> Any:
    if not isinstance(prediction, tuple):
        return authority

    key_points, detections = prediction
    if detections is None:
        return (authority, None)

    updated = (key_points, authority)

    return updated


def _row_tensor(carrier: Any) -> torch.Tensor:
    tensor = carrier.xy if isinstance(carrier, KeyPoints) else carrier.xyxy
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        raise ContractError(
            f"{type(carrier).__name__} has no row tensor, got "
            f"{type(tensor).__name__}"
        )

    return tensor


def _row_count(carrier: Any) -> int:
    rows = int(_row_tensor(carrier).shape[0])

    return rows


def _row_device(carrier: Any) -> torch.device:
    device = _row_tensor(carrier).device

    return device


def _image_metadata(carrier: Any) -> Mapping[str, Any]:
    metadata = carrier.image_metadata
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ContractError(
            f"{type(carrier).__name__}.image_metadata must be a dict or None, got "
            f"{type(metadata).__name__}"
        )

    return metadata


def _classification_metadata(prediction: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(prediction, MultiLabelClassificationPrediction):
        metadata = prediction.image_metadata
    elif isinstance(prediction, ClassificationPrediction):
        images_metadata = prediction.images_metadata
        if images_metadata is not None and len(images_metadata) != 1:
            raise ContractError(
                "A ClassificationPrediction of one image has one images_metadata "
                f"entry, got {len(images_metadata)}"
            )
        metadata = None if images_metadata is None else images_metadata[0]
    else:
        raise ContractError(
            "Expected inference_models.ClassificationPrediction or "
            f"MultiLabelClassificationPrediction, got {type(prediction).__name__}"
        )

    if metadata is not None and not isinstance(metadata, Mapping):
        raise ContractError(
            f"{type(prediction).__name__} metadata must be a dict, got "
            f"{type(metadata).__name__}"
        )

    return metadata


def _check_ids(ids: Any, *, rows: int, device: torch.device, name: str) -> None:
    if ids is None:
        return

    if not isinstance(ids, torch.Tensor):
        raise ContractError(
            f"{name} must be None or a torch.Tensor, got {type(ids).__name__}"
        )
    if ids.dtype != torch.int64 or ids.shape != (rows,):
        raise ContractError(
            f"{name} must be int64 with shape ({rows},), got {ids.dtype} with "
            f"shape {tuple(ids.shape)}"
        )
    if ids.device != device:
        raise ContractError(
            f"{name} is on {ids.device}, but the prediction rows are on {device}"
        )


def _check_columns(columns: Any, *, rows: int) -> None:
    is_integer = (
        isinstance(columns, torch.Tensor)
        and not columns.dtype.is_floating_point
        and not columns.dtype.is_complex
        and columns.dtype != torch.bool
    )
    if not is_integer or columns.ndim != 2 or columns.shape[0] != rows:
        raise ContractError(
            f"columns must be an integer ({rows}, k) tensor, got {_describe(columns)}"
        )


def _check_namespace(namespace: Any, *, name: str) -> None:
    if not isinstance(namespace, str) or not namespace:
        raise ContractError(f"{name} must be a non-empty string, got {namespace!r}")


def _namespace_of(record: Any) -> Any:
    namespace = record.get("namespace") if isinstance(record, Mapping) else None

    return namespace


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _describe(value: Any) -> str:
    # Never a repr of the value: it may hold tensors.
    if isinstance(value, torch.Tensor):
        return f"tensor {value.dtype} {tuple(value.shape)} on {value.device}"
    if isinstance(value, Mapping):
        return f"dict with keys {sorted(map(str, value))}"

    return type(value).__name__

"""Read detection predictions for UQL operations.

Accepted carriers, called "detections" below::

    Detections | InstanceDetections
    (KeyPoints, Detections | InstanceDetections)  boxes drive; keypoints follow rows
    PredictionRows of either                       columns follow rows
    SegmentationView                               boxes from ``view.detections``;
                                                   ``view.select`` gathers scores

Box values reach the host one column at a time, at most once per operation
call (``DetectionTable``). Masks are never read, and a view is never upscaled
to full resolution here. Size and centres are computed on the boxes' device in
their own dtype, as V1 does, then copied, so values match V1 bit for bit.

Row-metadata properties read ``bboxes_metadata[i][name]``. The V2-only
``decoded_text`` reads the ``data`` key, where code detectors store the text.

Row selection keeps the semantic row record (decision 015): ``take_rows`` and
``SegmentationView.select`` gather its ids with the rows.
"""

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Union

import torch
from roboflow_workflows.v2.blocks.query.errors import OperationFailure
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    RowSelection,
    class_names_by_id,
    row_class_name,
    row_class_names,
    take_rows,
)
from roboflow_workflows.v2.data.row_identity import without_row_record
from roboflow_workflows.v2.data.rows import PredictionRows

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

Boxes = Union[Detections, InstanceDetections]

# Per-box host copies of xyxy/class_id/confidence; stale once those change.
HOST_MIRROR_KEYS = ("_host_xyxy", "_host_class_id", "_host_confidence")
_IMAGE_METADATA_PROPERTIES = ("image_dimensions", "prediction_type")
# Property name -> row metadata key, for names that differ from their key.
_ROW_METADATA_KEYS = {"decoded_text": "data"}
_BOX_COLUMNS = {"x_min": 0, "y_min": 1, "x_max": 2, "y_max": 3}
_CORNERS = {
    "top_left": ("x_min", "y_min"),
    "top_right": ("x_max", "y_min"),
    "bottom_left": ("x_min", "y_max"),
    "bottom_right": ("x_max", "y_max"),
}
SUPPORTED_CARRIERS = (
    "Detections, InstanceDetections, a (KeyPoints, Detections) tuple, "
    "PredictionRows of them or a SegmentationView"
)


def native_of(value: Any) -> Any:
    """Return the native prediction behind a carrier.

    Args:
        value: A carrier, or any other value.

    Returns:
        ``PredictionRows.native``, a view's row-aligned ``Detections``, or
        ``value`` itself.
    """
    if isinstance(value, PredictionRows):
        return value.native
    if isinstance(value, SegmentationView):
        return value.detections

    return value


def is_detections(value: Any) -> bool:
    """Tell whether ``value`` is a carrier this module reads (boxes may be missing)."""
    native = native_of(value)
    is_carrier = isinstance(native, (Detections, InstanceDetections)) or (
        isinstance(native, tuple)
        and len(native) == 2
        and isinstance(native[0], KeyPoints)
    )

    return is_carrier


def boxes_of(value: Any) -> Boxes:
    """Return the box component of a carrier.

    Raises:
        OperationFailure: For other values, or keypoints without boxes.
    """
    if not is_detections(value):
        raise OperationFailure(f"expected {SUPPORTED_CARRIERS}")
    native = native_of(value)
    if not isinstance(native, tuple):
        return native
    if native[1] is None:
        raise OperationFailure(
            "the keypoint prediction has no bounding-box Detections, which this "
            "operation requires"
        )

    return native[1]


def row_count(value: Any) -> int:
    """Number of rows; a keypoint tuple counts skeletons, not its two parts."""
    native = native_of(value)
    if isinstance(native, tuple):
        return len(native[0])

    return len(native)


def select_rows(value: Any, selection: RowSelection) -> Any:
    """Apply one row selection to every aligned field and column of a carrier.

    The row record follows the selection; a view gathers its scores too.
    """
    if isinstance(value, SegmentationView):
        return value.select(selection)
    if isinstance(value, PredictionRows):
        selected = PredictionRows(
            native=take_rows(value.native, selection),
            columns={
                name: selection.take(column) for name, column in value.columns.items()
            },
        )
        return selected

    selected = take_rows(value, selection)

    return selected


def replace_boxes(value: Any, boxes: Boxes) -> Any:
    """Return the carrier with a new box component and everything else shared."""
    if isinstance(value, PredictionRows):
        replaced = PredictionRows(
            native=replace_boxes(value.native, boxes), columns=value.columns
        )
        return replaced
    if isinstance(value, SegmentationView):
        return _view_with_boxes(value, boxes)
    if isinstance(value, tuple):
        return value[0], boxes

    return boxes


def _view_with_boxes(view: SegmentationView, boxes: Detections) -> SegmentationView:
    # Same rows and read-only scores; only boxes, classes or metadata changed.
    if view.is_dense_fallback:
        predictions = InstanceDetections(
            xyxy=boxes.xyxy,
            class_id=boxes.class_id,
            confidence=boxes.confidence,
            mask=view.scores,
            image_metadata=boxes.image_metadata,
            bboxes_metadata=boxes.bboxes_metadata,
        )
        dense_view = SegmentationView.from_dense(predictions)
        return dense_view

    replaced = SegmentationView(
        detections=boxes,
        scores=view.scores,
        score_type=view.score_type,
        threshold=view.threshold,
        geometry=view.geometry,
        take_ownership=True,
    )

    return replaced


def without_host_mirror(rows: Optional[List[dict]]) -> Optional[List[dict]]:
    """Copy row metadata without host-mirror keys, for changed boxes or classes."""
    if rows is None:
        return None

    stripped = [
        {key: item for key, item in (row or {}).items() if key not in HOST_MIRROR_KEYS}
        for row in rows
    ]

    return stripped


def _ignored_row_class(row: Mapping[str, Any]) -> str:
    # Error suffix naming a non-string row value that was not used as a name.
    if CLASS_NAME_KEY not in row:
        return ""
    note = (
        f"; bboxes_metadata['{CLASS_NAME_KEY}'] is {row[CLASS_NAME_KEY]!r}, "
        "not a string"
    )

    return note


class _ClassNames(NamedTuple):
    """Resolved class names of one table, with the first failure of each kind.

    ``error`` (no class-name map) fails every read. ``strict_error`` (first
    row whose id the map lacks) fails only ``require_all`` reads; other reads
    get None for such rows.
    """

    names: List[Optional[str]]
    error: Optional[str]
    strict_error: Optional[str]


class DetectionTable:
    """Host columns of one carrier, each copied from the device at most once."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.boxes = boxes_of(value)
        native = native_of(value)
        self.key_points: Optional[KeyPoints] = (
            native[0] if isinstance(native, tuple) else None
        )
        self.columns = value.columns if isinstance(value, PredictionRows) else {}
        self.count = len(self.boxes)
        self.image_metadata: dict = self.boxes.image_metadata or {}
        self.row_metadata: List[dict] = [
            row or {} for row in (self.boxes.bboxes_metadata or [{}] * self.count)
        ]
        self._cache: Dict[str, list] = {}
        # Resolved on the first class-name read. Metadata does not change
        # during a table's lifetime; transforms build a new carrier and table.
        self._class_names: Optional[_ClassNames] = None

    def column(self, name: str) -> list:
        """Return a cached host list for a box column or a derived box value."""
        if name not in self._cache:
            self._cache[name] = self._read(name)

        return self._cache[name]

    def _read(self, name: str) -> list:
        xyxy = self.boxes.xyxy
        if name in _BOX_COLUMNS:
            return xyxy[:, _BOX_COLUMNS[name]].tolist()
        if name == "confidence":
            return self.boxes.confidence.tolist()
        if name == "class_id":
            return [int(value) for value in self.boxes.class_id.tolist()]
        if name == "size":
            return ((xyxy[:, 3] - xyxy[:, 1]) * (xyxy[:, 2] - xyxy[:, 0])).tolist()
        if name == "center_x":
            return (xyxy[:, 0] + (xyxy[:, 2] - xyxy[:, 0]) / 2).tolist()
        if name == "center_y":
            return (xyxy[:, 1] + (xyxy[:, 3] - xyxy[:, 1]) / 2).tolist()
        if name in self.columns:
            return self.columns[name].tolist()
        if name == "keypoints_xy":
            return self.key_points.xy.tolist()

        raise KeyError(name)

    def has_values(self, name: str) -> bool:
        """Whether a non-box property has a source: a column, keypoints or metadata."""
        has_source = (
            name in self.columns
            or (name == "keypoints_xy" and self.key_points is not None)
            or any(name in row for row in self.row_metadata)
        )

        return has_source

    def class_names(self, *, require_all: bool) -> List[Optional[str]]:
        """Effective class name per row: ``row_class_name``, else the map.

        Rows are resolved once per table, so a per-row read stays O(1).

        Args:
            require_all: Fail for a row whose name cannot be resolved; otherwise
                such rows give ``None``.
        """
        if self._class_names is None:
            self._class_names = self._resolve_class_names()
        resolved = self._class_names
        if resolved.error is not None:
            raise OperationFailure(resolved.error)
        if require_all and resolved.strict_error is not None:
            raise OperationFailure(resolved.strict_error)

        return resolved.names

    def _resolve_class_names(self) -> _ClassNames:
        names_by_id = class_names_by_id(self.image_metadata)
        class_ids = self.column("class_id")
        names = row_class_names(
            class_ids, row_metadata=self.row_metadata, names_by_id=names_by_id
        )
        unnamed = next(
            (index for index, name in enumerate(names) if name is None), None
        )
        if unnamed is None:
            resolved = _ClassNames(names=names, error=None, strict_error=None)
            return resolved

        class_id, row = class_ids[unnamed], self.row_metadata[unnamed]
        if names_by_id is None:
            error = (
                f"image_metadata['{CLASS_NAMES_KEY}'] is missing, so class "
                f"id {class_id} has no name{_ignored_row_class(row)}"
            )
            failed = _ClassNames(names=[], error=error, strict_error=None)
            return failed

        strict_error = (
            f"class id {class_id} is missing from image_metadata"
            f"['{CLASS_NAMES_KEY}'] (keys: {sorted(names_by_id)})"
            f"{_ignored_row_class(row)}"
        )
        resolved = _ClassNames(names=names, error=None, strict_error=strict_error)

        return resolved


class DetectionRow:
    """One detection of a carrier; DetectionsFilter binds it to `_`."""

    __slots__ = ("table", "index")

    def __init__(self, table: DetectionTable, index: int) -> None:
        self.table = table
        self.index = index

    def __repr__(self) -> str:
        return f"DetectionRow({self.index} of {self.table.count})"

    def property(self, name: str) -> Any:
        """Return one property of this detection, as V1 ExtractDetectionProperty does."""
        table = self.table
        index = self.index
        name = _ROW_METADATA_KEYS.get(name, name)
        if name in _BOX_COLUMNS or name in ("confidence", "class_id", "size"):
            return table.column(name)[index]
        if name == "class_name":
            return table.class_names(require_all=True)[index]
        if name == "center":
            return table.column("center_x")[index], table.column("center_y")[index]
        if name in _CORNERS:
            x_name, y_name = _CORNERS[name]
            return table.column(x_name)[index], table.column(y_name)[index]
        if name in _IMAGE_METADATA_PROPERTIES:
            return table.image_metadata.get(name)
        if name in table.columns or (
            name == "keypoints_xy" and table.key_points is not None
        ):
            return table.column(name)[index]
        value = table.row_metadata[index].get(name)
        if name == "tracker_id" and value is not None:
            value = int(value)

        return value


def detection_row(value: Any) -> DetectionRow:
    """Accept a filter row or the 7-tuple yielded by iterating native detections."""
    if isinstance(value, DetectionRow):
        return value
    if not (isinstance(value, tuple) and len(value) == 7):
        raise OperationFailure(
            "expected one detection: the `_` row of DetectionsFilter or the "
            "(xyxy, mask, class_id, confidence, tracker_id, data, image_metadata) "
            "tuple of an iterated prediction"
        )
    xyxy, _, class_id, confidence, tracker_id, data, image_metadata = value
    row_metadata = dict(data or {})
    if tracker_id is not None:
        row_metadata["tracker_id"] = tracker_id
    # The tuple carries the whole prediction's metadata. This temporary row
    # is no selection of it, so it carries no row record.
    one_row = Detections(
        xyxy=xyxy.reshape(1, 4),
        class_id=torch.as_tensor(class_id, device=xyxy.device).reshape(1),
        confidence=torch.as_tensor(confidence, device=xyxy.device).reshape(1),
        image_metadata=without_row_record(image_metadata),
        bboxes_metadata=[row_metadata],
    )
    row = DetectionRow(DetectionTable(one_row), 0)

    return row


def detections_property(value: Any, name: str) -> list:
    """Return one property per detection, as V1 DetectionsPropertyExtract does."""
    table = DetectionTable(value)
    if table.count == 0:
        return []
    if name in _BOX_COLUMNS or name in ("confidence", "class_id", "size"):
        return table.column(name)
    if name == "class_name":
        has_names = CLASS_NAMES_KEY in table.image_metadata or any(
            row_class_name(row) is not None for row in table.row_metadata
        )
        # V1 returns [] when no class names were attached at all.
        return table.class_names(require_all=True) if has_names else []
    if name == "center" or name in _CORNERS:
        return _anchors(table.boxes.xyxy, name)
    if name in _IMAGE_METADATA_PROPERTIES:
        return [table.image_metadata.get(name)] * table.count
    if not table.has_values(_ROW_METADATA_KEYS.get(name, name)):
        raise OperationFailure(
            f"property `{name}` is not a box property and no row, column or "
            "keypoint field carries it"
        )

    values = [DetectionRow(table, index).property(name) for index in range(table.count)]

    return values


def _anchors(xyxy: torch.Tensor, name: str) -> List[List[int]]:
    # V1 whole-prediction anchors are rounded integer pixels.
    if name == "center":
        xs = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        ys = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    else:
        x_name, y_name = _CORNERS[name]
        xs, ys = xyxy[:, _BOX_COLUMNS[x_name]], xyxy[:, _BOX_COLUMNS[y_name]]
    anchors = torch.stack([xs, ys], dim=1).round().long().tolist()

    return anchors

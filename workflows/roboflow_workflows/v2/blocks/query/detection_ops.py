"""UQL operations that select, reorder or edit detections.

Every row-picking operation builds one ``RowSelection`` and applies it to all
aligned fields (boxes, masks, keypoints, row metadata, PredictionRows
columns), so outputs keep their carrier type, metadata, order and repeats.

Host work per call::

    filter            class/box columns the condition reads, once each;
                      the condition itself runs in Python per row
    select, sort      none for the index (device argmax/argsort); row
                      metadata, if any, is gathered on the host by take_rows
    rename, parent    class_id column once; rename writes class_id back
    offset, shift     none
    to dictionary     whole prediction (inference response JSON boundary);
                      a grid SegmentationView is upscaled once to exact
                      full-resolution masks (cached on the view)

Box/class predicates, selection and ordering of a ``SegmentationView`` read
its boxes only; they never compute full-resolution masks.
"""

import dataclasses
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from roboflow_workflows.v2.blocks.query.detections import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DetectionRow,
    DetectionTable,
    boxes_of,
    replace_boxes,
    select_rows,
    without_host_mirror,
)
from roboflow_workflows.v2.blocks.query.errors import (
    OperationFailure,
    QueryRuntimeError,
)
from roboflow_workflows.v2.blocks.query.schema import DEFAULT_OPERAND_NAME
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.prediction_ops import RowSelection
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.data.rows import PredictionRows

RowCondition = Callable[[Mapping[str, Any]], Any]

DETECTION_ID_KEY = "detection_id"
# Fixed UUID namespace of detection ids derived from semantic row keys.
_DERIVED_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "roboflow-workflows-v2/row")


def filter_detections(
    value: Any, keep_row: RowCondition, parameters: Mapping[str, Any]
) -> Any:
    """Keep rows for which ``keep_row`` holds, binding each row to `_`.

    Every runtime parameter reaches the condition unchanged; only `_` is set
    per row. Rows are evaluated in order and all of them are evaluated.
    """
    table = DetectionTable(value)
    row_values = dict(parameters)
    kept = []
    for index in range(table.count):
        row_values[DEFAULT_OPERAND_NAME] = DetectionRow(table, index)
        try:
            if keep_row(row_values):
                kept.append(index)
        except QueryRuntimeError as error:
            raise QueryRuntimeError(
                f"detection {index}: {error.detail}",
                path=error.path,
                operation=error.operation,
                parameter=error.parameter,
            ) from error
    filtered = select_rows(value, RowSelection.of(table.count, indices=kept))

    return filtered


def select_detection(operation, value: Any) -> Any:
    boxes = boxes_of(value)
    count = len(boxes)
    if count == 0:
        index = torch.empty(0, dtype=torch.long)
    elif operation.mode == "first":
        index = torch.tensor([0])
    elif operation.mode == "last":
        index = torch.tensor([count - 1])
    elif operation.mode == "top_confidence":
        index = torch.argmax(boxes.confidence).reshape(1)
    else:
        centers_x = (boxes.xyxy[:, 0] + boxes.xyxy[:, 2]) * 0.5
        pick = torch.argmin if operation.mode == "left_most" else torch.argmax
        index = pick(centers_x).reshape(1)
    # argmin/argmax return the first of equal values, as in V1.
    selected = select_rows(value, RowSelection.of(count, indices=index))

    return selected


def sort_detections(operation, value: Any) -> Any:
    boxes = boxes_of(value)
    if len(boxes) == 0:
        return value

    xyxy = boxes.xyxy
    keys = {
        "confidence": lambda: boxes.confidence,
        "x_min": lambda: xyxy[:, 0],
        "y_min": lambda: xyxy[:, 1],
        "x_max": lambda: xyxy[:, 2],
        "y_max": lambda: xyxy[:, 3],
        "size": lambda: (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]),
        "center_x": lambda: (xyxy[:, 0] + xyxy[:, 2]) * 0.5,
        "center_y": lambda: (xyxy[:, 1] + xyxy[:, 3]) * 0.5,
    }
    # Stable: equal keys keep their input order in both directions.
    order = torch.sort(
        keys[operation.mode](), stable=True, descending=not operation.ascending
    ).indices
    sorted_value = select_rows(value, RowSelection.of(len(boxes), indices=order))

    return sorted_value


def offset_detections(operation, value: Any) -> Any:
    half_x = operation.offset_x / 2
    half_y = operation.offset_y / 2
    moved = _move_boxes(value, [-half_x, -half_y, half_x, half_y])

    return moved


def shift_detections(operation, value: Any) -> Any:
    shift = [operation.shift_x, operation.shift_y, operation.shift_x, operation.shift_y]
    moved = _move_boxes(value, shift)

    return moved


def _move_boxes(value: Any, delta: List[float]) -> Any:
    # Only boxes move; keypoints and masks stay, as in V1.
    boxes = boxes_of(value)
    xyxy = boxes.xyxy + torch.tensor(
        delta, dtype=boxes.xyxy.dtype, device=boxes.xyxy.device
    )
    moved_boxes = dataclasses.replace(
        boxes, xyxy=xyxy, bboxes_metadata=without_host_mirror(boxes.bboxes_metadata)
    )
    moved = replace_boxes(value, moved_boxes)

    return moved


def rename_detections(operation, value: Any, parameters: Mapping[str, Any]) -> Any:
    """Rename classes as V1 does; string class_map/strict name runtime parameters."""
    class_map = _parameter_value(operation.class_map, parameters, name="class_map")
    strict = _parameter_value(operation.strict, parameters, name="strict")
    if not isinstance(class_map, dict):
        raise OperationFailure(
            f"class_map must be a dictionary, got {type(class_map).__name__}",
            parameter="class_map",
        )
    if not isinstance(strict, bool):
        raise OperationFailure(
            f"strict must be a boolean, got {type(strict).__name__}", parameter="strict"
        )

    table = DetectionTable(value)
    if CLASS_NAMES_KEY not in table.image_metadata:
        if strict and table.count > 0:
            raise OperationFailure(
                "strict=True, but the detections carry no class names, so class_map "
                "coverage cannot be checked",
                parameter="strict",
            )
        return value

    names = table.class_names(require_all=True)
    if strict:
        for name in names:
            if name not in class_map:
                raise OperationFailure(
                    f"class '{name}' not found in class_map", parameter="class_map"
                )
        id_by_name = {
            name: class_id
            for class_id, name in enumerate(sorted(set(class_map.values())))
        }
    else:
        id_by_name = _non_strict_ids(
            names, table.column("class_id"), class_map, operation.new_classes_id_offset
        )
    new_names = [class_map.get(name, name) for name in names]
    new_ids = [id_by_name[name] for name in new_names]

    has_override = [CLASS_NAME_KEY in row for row in table.row_metadata]
    boxes = table.boxes
    rows = without_host_mirror(boxes.bboxes_metadata)
    if rows is not None:
        for index, row in enumerate(rows):
            if has_override[index]:
                row[CLASS_NAME_KEY] = new_names[index]
    image_metadata = dict(table.image_metadata)
    image_metadata[CLASS_NAMES_KEY] = _renamed_class_names(
        id_by_name, new_names, new_ids, class_map, has_override
    )
    renamed_boxes = dataclasses.replace(
        boxes,
        class_id=torch.tensor(
            new_ids, dtype=boxes.class_id.dtype, device=boxes.class_id.device
        ),
        image_metadata=image_metadata,
        bboxes_metadata=rows,
    )
    renamed = replace_boxes(value, renamed_boxes)

    return renamed


def _parameter_value(value: Any, parameters: Mapping[str, Any], *, name: str) -> Any:
    if not isinstance(value, str):
        return value
    if value not in parameters:
        raise OperationFailure(
            f"runtime parameter `{value}` is not provided (available: {list(parameters)})",
            parameter=name,
        )

    return parameters[value]


def _non_strict_ids(
    names: List[str], class_ids: List[int], class_map: dict, offset: int
) -> Dict[str, int]:
    # Kept classes keep their id (last row wins); new names get ids from offset.
    id_by_name = dict(zip(names, class_ids))
    new_names = sorted({name for name in class_map.values() if name not in id_by_name})
    for position, name in enumerate(new_names):
        id_by_name[name] = offset + position

    return id_by_name


def _renamed_class_names(
    id_by_name: Dict[str, int],
    new_names: List[str],
    new_ids: List[int],
    class_map: dict,
    has_override: List[bool],
) -> Dict[int, str]:
    allowed = set(new_names) | set(class_map.values())
    names_by_id = {
        class_id: name for name, class_id in id_by_name.items() if name in allowed
    }
    # A row without its own name is read through this map, so its pair wins.
    for index, overridden in enumerate(has_override):
        if overridden:
            names_by_id.setdefault(new_ids[index], new_names[index])
        else:
            names_by_id[new_ids[index]] = new_names[index]

    return names_by_id


def pick_detections_by_parent_class(operation, value: Any) -> Any:
    """Keep parents, then other rows whose box centre lies in any parent box."""
    table = DetectionTable(value)
    parents: List[int] = []
    dependents: List[int] = []
    if table.count:
        for index, name in enumerate(table.class_names(require_all=False)):
            (parents if name == operation.parent_class else dependents).append(index)
    kept = []
    if parents and dependents:
        xyxy = table.boxes.xyxy
        device = xyxy.device
        children = xyxy[torch.tensor(dependents, device=device)]
        parent_boxes = xyxy[torch.tensor(parents, device=device)]
        centers_x = ((children[:, 0] + children[:, 2]) * 0.5).unsqueeze(1)
        centers_y = ((children[:, 1] + children[:, 3]) * 0.5).unsqueeze(1)
        inside_any = (
            (centers_x >= parent_boxes[:, 0])
            & (centers_x <= parent_boxes[:, 2])
            & (centers_y >= parent_boxes[:, 1])
            & (centers_y <= parent_boxes[:, 3])
        ).any(dim=1)
        kept = [dependents[k] for k, inside in enumerate(inside_any.tolist()) if inside]
    indices = parents + kept if parents else []
    picked = select_rows(value, RowSelection.of(table.count, indices=indices))

    return picked


def detections_to_dictionary(operation, value: Any) -> dict:
    """Serialise to the V1 inference response dictionary (host/JSON boundary).

    Every output row needs a ``detection_id``. An explicit per-row value is
    kept. A row without one gets ``derived_detection_id`` of its semantic
    row key; the input is not changed. Without either, this fails rather
    than invent an identity from row positions.

    A ``SegmentationView`` is exported with exact reference masks: one
    full-resolution upscale for a grid view, cached on that view.
    """
    if isinstance(value, PredictionRows) and value.columns:
        raise OperationFailure(
            "PredictionRows columns "
            f"{sorted(value.columns)} have no field in the response dictionary; "
            "serialise the native prediction or move the columns into row metadata"
        )
    if isinstance(value, SegmentationView):
        value = mask_provider_of(value).binary(fidelity="reference").predictions
    boxes = _with_detection_ids(boxes_of(value))
    # The response format is defined by the V1 tensor serializer; import it
    # only when this operation runs.
    from roboflow_workflows.core_steps.common.serializers_tensor import (
        serialise_sv_detections,
    )

    serialized = serialise_sv_detections(boxes)

    return serialized


def derived_detection_id(namespace: str, row_id: int) -> str:
    """Return the stable response ``detection_id`` of one semantic row.

    The same row gives the same UUID string in every frame, branch and
    repeat, so a repeated row keeps one id.

    Args:
        namespace: Row record namespace of the prediction.
        row_id: Row id within that namespace.

    Returns:
        A UUID string: ``uuid5`` of ``"<namespace>/<row_id>"``.
    """
    derived = str(uuid.uuid5(_DERIVED_ID_NAMESPACE, f"{namespace}/{row_id}"))

    return derived


def _with_detection_ids(boxes: Any) -> Any:
    # Fills only missing ids, on copied row dicts; explicit ids are kept.
    rows: List[Optional[dict]] = list(boxes.bboxes_metadata or [None] * len(boxes))
    missing = [
        index for index, row in enumerate(rows) if DETECTION_ID_KEY not in (row or {})
    ]
    if not missing:
        return boxes

    record = row_record(boxes)
    if record is None:
        raise OperationFailure(
            f"detection {missing[0]} has no `{DETECTION_ID_KEY}` and the "
            "predictions have no row identity to derive one from. Model blocks "
            "attach a row identity; for other predictions set "
            f"`{DETECTION_ID_KEY}` per row or apply with_row_identity once"
        )

    row_ids = record.host_ids()
    for index in missing:
        row = dict(rows[index] or {})
        row[DETECTION_ID_KEY] = derived_detection_id(record.namespace, row_ids[index])
        rows[index] = row
    with_ids = dataclasses.replace(boxes, bboxes_metadata=rows)

    return with_ids

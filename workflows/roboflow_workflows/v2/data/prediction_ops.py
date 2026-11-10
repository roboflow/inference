"""Strict row selection for native V2 detection and keypoint payloads.

One ``RowSelection`` selects every aligned row field of a prediction together:
tensor fields, dense or RLE masks, per-row metadata dicts and keypoint tuples.
Callers that carry more ``(N, ...)`` tensors, such as score grids or row
columns, apply the same selection to them with ``RowSelection.take``.

Tensor fields are gathered with ``index_select`` on their own device and may
have any strides. Host work is limited to what the inputs require:

==========================================  ==================================
situation                                   host work
==========================================  ==================================
host list or NumPy mask / indices           none beyond reading the input
CPU tensor mask / indices                   ``nonzero`` or a min/max range
                                            check on the CPU, no device sync
CUDA indices, CUDA fields, no host rows     none
CUDA boolean mask                           one ``nonzero``: its size depends
                                            on the data, so the stream syncs
row metadata dicts or RLE masks present     selected positions copied to the
                                            host once
CPU positions, CUDA fields                  one host-to-device position copy;
                                            the source is pageable, so the
                                            host waits for the current stream
CUDA positions, CPU fields                  one device-to-host position copy
==========================================  ==================================

CUDA index tensors are not range-checked, because that needs a host sync. An
out-of-range or negative CUDA row triggers a device-side assert. It raises
``RuntimeError``, not ``ContractError``, either in the call or at a later
synchronizing operation, and it leaves the process's CUDA context unusable:
every later CUDA operation in that process fails until the process restarts.

Selection also carries the optional row record of ``row_identity``. It is the
one exception to shared ``image_metadata``: when the record changes, the
selected carrier gets a shallow copy of the metadata with a new record, and the
source metadata and record are left untouched. Ids cost no extra kernel on the
first selection, because an implicit range becomes the selection's own
positions; an explicit ids tensor is gathered once.

Row class names (decision 025) follow one rule for query, thresholding and
drawing, on host metadata only::

    name = row["class"]                     if it is a string ("" included)
           else class_names[class_id]       dict, or list indexed by id
           else None                        the caller raises its own error
"""

from dataclasses import replace
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import ROW_RECORD_KEY

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import (
    InstanceSegmentationGrid,
    InstancesRLEMasks,
)

Selection = Union[torch.Tensor, np.ndarray, Sequence]
# Keys written by producers; values match the V1 execution engine constants.
CLASS_NAMES_KEY = "class_names"
CLASS_NAME_KEY = "class"

_INTEGER_TENSOR_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class RowSelection:
    """Ordered rows picked from ``count`` aligned rows; repeats allowed.

    Build it with ``RowSelection.of``. The positions are computed once and
    kept per device, so selecting many fields costs one conversion per device.

    Attributes:
        count: Number of rows of the payload the selection applies to.
        is_identity: Whether the selection keeps every row in order. Identity
            selections return tensors unchanged (aliases).
    """

    def __init__(
        self,
        *,
        count: int,
        length: int,
        is_identity: bool,
        positions: Optional[torch.Tensor],
        host_positions: Optional[List[int]],
        caller_positions: Optional[torch.Tensor] = None,
    ) -> None:
        self.count = count
        self.is_identity = is_identity
        self._length = length
        self._positions: Dict[torch.device, torch.Tensor] = {}
        if positions is not None:
            self._positions[positions.device] = positions
        self._host_positions = host_positions
        # Positions that share storage with the caller's index tensor.
        self._caller_positions = caller_positions

    @classmethod
    def of(
        cls,
        count: int,
        *,
        mask: Optional[Selection] = None,
        indices: Optional[Selection] = None,
    ) -> "RowSelection":
        """Validate one mask or index list against ``count`` rows.

        Args:
            count: Number of rows to select from.
            mask: One-dimensional boolean selection with ``count`` entries: a
                torch tensor on any device, a NumPy array or a list of bools.
            indices: One-dimensional integer positions: a torch tensor on any
                device, a NumPy array or a list of ints. Order and repeats are
                kept. Negative and out-of-range positions are rejected, except
                in CUDA tensors, which are not range-checked: an invalid CUDA
                position raises a possibly deferred ``RuntimeError`` and
                leaves the process's CUDA context unusable (module docstring).

        Returns:
            The selection.

        Raises:
            ContractError: When the selection is malformed or both/neither
                forms are supplied.
        """
        if (mask is None) == (indices is None):
            raise ContractError("Provide exactly one of mask or indices")

        if mask is not None:
            selection = cls._from_mask(mask, count=count)
        else:
            selection = cls._from_indices(indices, count=count)

        return selection

    def __len__(self) -> int:
        return self._length

    def positions_on(self, device: torch.device) -> torch.Tensor:
        """Selected positions as an ``int64`` tensor on ``device``.

        Args:
            device: Device of the tensors to gather.

        Returns:
            ``(len(self),)`` positions, cached per device.
        """
        device = torch.device(device)
        positions = self._positions.get(device)
        if positions is not None:
            return positions

        if self._positions:
            known = next(iter(self._positions.values()))
            positions = known.to(device)
        else:
            positions = torch.tensor(
                self._host_positions, dtype=torch.int64, device=device
            )
        self._positions[device] = positions

        return positions

    def owned_positions_on(self, device: torch.device) -> torch.Tensor:
        """Selected positions that are safe to keep, such as new row ids.

        Same as ``positions_on``, except that positions sharing storage with
        the caller's index tensor are cloned, so later changes to that tensor
        cannot reach the kept values.

        Args:
            device: Device of the tensors to gather.

        Returns:
            ``(len(self),)`` ``int64`` positions; treat them as read-only.
        """
        positions = self.positions_on(device)
        if positions is self._caller_positions:
            positions = positions.clone()

        return positions

    def host_positions(self) -> List[int]:
        """Selected positions as Python ints, for host row metadata and RLE.

        Returns:
            The positions; copied from the device once and then cached.
        """
        if self._host_positions is None:
            known = next(iter(self._positions.values()))
            self._host_positions = known.cpu().tolist()

        return self._host_positions

    def take(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select rows of one ``(count, ...)`` tensor on its own device.

        Args:
            tensor: Tensor whose first axis has ``count`` rows; any strides.

        Returns:
            ``tensor`` itself for an identity selection, otherwise a new
            tensor with the same dtype and device.

        Raises:
            ContractError: When the first axis does not have ``count`` rows.
        """
        if tensor.ndim == 0 or tensor.shape[0] != self.count:
            raise ContractError(
                f"Cannot select {self.count} rows from a tensor of shape "
                f"{tuple(tensor.shape)}"
            )
        if self.is_identity:
            return tensor

        selected = tensor.index_select(0, self.positions_on(tensor.device))

        return selected

    def take_list(self, values: Sequence[Any]) -> List[Any]:
        """Select entries of a host list, such as RLE masks.

        Args:
            values: ``count`` entries.

        Returns:
            A new list of the selected entries, in selection order.
        """
        selected = [values[position] for position in self.host_positions()]

        return selected

    @classmethod
    def _from_mask(cls, mask: Selection, *, count: int) -> "RowSelection":
        _check_mask(mask, rows=count)
        if isinstance(mask, torch.Tensor):
            positions = torch.nonzero(mask.detach()).reshape(-1)
            selection = cls(
                count=count,
                length=int(positions.shape[0]),
                is_identity=int(positions.shape[0]) == count,
                positions=positions,
                host_positions=None,
            )
            return selection

        host_positions = np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()
        selection = cls(
            count=count,
            length=len(host_positions),
            is_identity=len(host_positions) == count,
            positions=None,
            host_positions=host_positions,
        )

        return selection

    @classmethod
    def _from_indices(cls, indices: Selection, *, count: int) -> "RowSelection":
        if isinstance(indices, torch.Tensor):
            positions = _checked_index_tensor(indices, rows=count)
            # CUDA identity would need a value read; only CPU checks it.
            is_identity = (
                positions.device.type == "cpu"
                and positions.shape[0] == count
                and torch.equal(positions, torch.arange(count))
            )
            shares_storage = positions.data_ptr() == indices.data_ptr()
            selection = cls(
                count=count,
                length=int(positions.shape[0]),
                is_identity=is_identity,
                positions=positions,
                host_positions=None,
                caller_positions=positions if shares_storage else None,
            )
            return selection

        host_positions = _checked_host_indices(indices, rows=count)
        selection = cls(
            count=count,
            length=len(host_positions),
            is_identity=host_positions == list(range(count)),
            positions=None,
            host_positions=host_positions,
        )

        return selection


def select_predictions(
    prediction: Any,
    *,
    mask: Optional[Selection] = None,
    indices: Optional[Selection] = None,
) -> Any:
    """Select aligned native rows with exactly one mask or ordered index list.

    Args:
        prediction: Detections, InstanceDetections, KeyPoints, or a tuple of
            KeyPoints and optional Detections. Fields are validated first.
        mask: One-dimensional boolean selection with exactly one entry per row.
        indices: One-dimensional integer positions; may reorder or repeat rows.
            Negative and out-of-range positions are rejected, except in CUDA
            tensors (see ``RowSelection.of``).

    Returns:
        A carrier of the original family with all tensor fields, masks, row IDs
        and metadata selected together. Identity selections share tensors;
        selected row dictionaries are copied and image metadata is shared,
        except that a changed row record goes into a shallow copy. Tensor
        fields remain on their device. See the module docstring for the host
        work of each selection form.

    Raises:
        ContractError: If the payload or selection is malformed, unsupported,
            or both/neither selection forms are supplied.
    """
    if (mask is None) == (indices is None):
        raise ContractError("Provide exactly one of mask or indices")

    rows = validated_row_count(prediction)
    selection = RowSelection.of(rows, mask=mask, indices=indices)
    selected = _take_prediction(prediction, selection)

    return selected


def take_rows(prediction: Any, selection: RowSelection) -> Any:
    """Apply a prepared selection to every aligned field of a prediction.

    Use this when one selection must also cover other aligned tensors, such
    as scores or row columns, so that all of them pick the same rows.

    Args:
        prediction: A payload accepted by ``select_predictions``.
        selection: Selection built for the prediction's row count.

    Returns:
        The selected prediction, as ``select_predictions`` returns it.

    Raises:
        ContractError: When the payload is malformed or the selection was
            built for another row count.
    """
    rows = validated_row_count(prediction)
    if selection.count != rows:
        raise ContractError(
            f"Selection was built for {selection.count} rows; the prediction "
            f"has {rows}"
        )

    selected = _take_prediction(prediction, selection)

    return selected


def validated_row_count(prediction: Any) -> int:
    """Validate a native row payload and return its number of rows.

    Reads tensor shape, dtype and device only.

    Args:
        prediction: Detections, InstanceDetections, KeyPoints, or a tuple of
            KeyPoints and optional Detections.

    Returns:
        Number of rows (boxes or skeletons).

    Raises:
        ContractError: When the payload is malformed or unsupported.
    """
    if isinstance(prediction, Detections):
        OBJECT_DETECTION_PREDICTION_KIND.check(prediction)
        return len(prediction)

    if isinstance(prediction, InstanceDetections):
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(prediction)
        return len(prediction)

    if isinstance(prediction, KeyPoints):
        KEYPOINT_DETECTION_PREDICTION_KIND.check((prediction, None))
        return len(prediction)

    if isinstance(prediction, tuple):
        KEYPOINT_DETECTION_PREDICTION_KIND.check(prediction)
        return len(prediction[0])

    raise ContractError(
        "select_predictions supports Detections, InstanceDetections, KeyPoints "
        f"or (KeyPoints, optional Detections), got {type(prediction).__name__}"
    )


def row_class_name(row: Mapping[str, Any]) -> Optional[str]:
    """Return a row's own class name: its ``class`` value if that is a string.

    Every consumer of class names uses this rule, so a detection is filtered,
    thresholded and drawn under one name. A non-string value names nothing;
    the ``class_names`` map names the row instead.

    Args:
        row: One row of ``bboxes_metadata``.

    Returns:
        The string (``""`` included), or ``None``.
    """
    if CLASS_NAME_KEY not in row:
        return None
    name = row[CLASS_NAME_KEY]
    own_name = name if isinstance(name, str) else None

    return own_name


def class_names_by_id(
    image_metadata: Optional[Mapping[str, Any]],
) -> Optional[Mapping[int, Any]]:
    """Return ``image_metadata["class_names"]`` as an id -> name mapping.

    Producers write a dict; a list is read as indexed by class id, so a
    negative id names nothing.

    Args:
        image_metadata: A prediction's ``image_metadata``; may be ``None``.

    Returns:
        The mapping, or ``None`` when there is no ``class_names``.
    """
    names = (image_metadata or {}).get(CLASS_NAMES_KEY)
    if names is None or isinstance(names, Mapping):
        return names

    by_id = dict(enumerate(names))

    return by_id


def row_class_names(
    class_ids: Sequence[int],
    *,
    row_metadata: Sequence[Mapping[str, Any]],
    names_by_id: Optional[Mapping[int, Any]],
) -> List[Optional[Any]]:
    """Name each row: ``row_class_name``, else ``names_by_id[class_id]``.

    One pass over host data; each row's ``class`` is looked up once. Callers
    pass class ids already on the host and raise their own error for rows
    left unnamed.

    Args:
        class_ids: ``N`` host integer class ids, matched exactly.
        row_metadata: ``N`` row dicts (``bboxes_metadata``).
        names_by_id: ``class_names_by_id`` of the rows' ``image_metadata``.

    Returns:
        ``N`` names; a map value is returned as stored. ``None`` where neither
        the row nor the map names the row (a map value of ``None`` included).
    """
    names = []
    for row, class_id in zip(row_metadata, class_ids):
        name = row_class_name(row)
        if name is None and names_by_id is not None:
            name = names_by_id.get(class_id)
        names.append(name)

    return names


def _take_prediction(prediction: Any, selection: RowSelection) -> Any:
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        if detections is None:
            record = _selected_row_record(
                key_points.image_metadata, selection, device=key_points.xy.device
            )
            selected = (_take_key_points(key_points, selection, record=record), None)
            return selected

        # Detections hold the tuple's only record.
        record = _selected_row_record(
            detections.image_metadata, selection, device=detections.xyxy.device
        )
        selected = (
            _take_key_points(key_points, selection, record=record),
            _take_boxes(detections, selection, record=record),
        )
        return selected

    if isinstance(prediction, KeyPoints):
        record = _selected_row_record(
            prediction.image_metadata, selection, device=prediction.xy.device
        )
        selected = _take_key_points(prediction, selection, record=record)
        return selected

    record = _selected_row_record(
        prediction.image_metadata, selection, device=prediction.xyxy.device
    )
    selected = _take_boxes(prediction, selection, record=record)

    return selected


def _selected_row_record(
    image_metadata: Optional[dict], selection: RowSelection, *, device: torch.device
) -> Optional[dict]:
    """The row record of the selected rows; ``None`` keeps metadata as is."""
    record = None if image_metadata is None else image_metadata.get(ROW_RECORD_KEY)
    if record is None or selection.is_identity:
        return None

    if not len(selection):
        ids = None
    elif record["ids"] is None:
        # Implicit rows 0..n-1: the selected ids are the positions themselves.
        ids = selection.owned_positions_on(device)
    else:
        ids = selection.take(record["ids"])
    selected = {**record, "rows": len(selection), "ids": ids}

    return selected


def _with_row_record(
    image_metadata: Optional[dict], record: Optional[dict]
) -> Optional[dict]:
    # Shallow copy; only carriers that held the source record receive it.
    if record is None or image_metadata is None or ROW_RECORD_KEY not in image_metadata:
        return image_metadata

    updated = {**image_metadata, ROW_RECORD_KEY: record}

    return updated


def _take_boxes(
    detections: Union[Detections, InstanceDetections],
    selection: RowSelection,
    *,
    record: Optional[dict],
) -> Union[Detections, InstanceDetections]:
    fields = {
        "xyxy": selection.take(detections.xyxy),
        "class_id": selection.take(detections.class_id),
        "confidence": selection.take(detections.confidence),
        "image_metadata": _with_row_record(detections.image_metadata, record),
        "bboxes_metadata": _take_row_dicts(detections.bboxes_metadata, selection),
    }
    if not isinstance(detections, InstanceDetections):
        selected = Detections(**fields)
        return selected

    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        selected_mask = InstancesRLEMasks(
            image_size=mask.image_size,
            mask_size=mask.mask_size,
            masks=selection.take_list(mask.masks),
        )
    elif isinstance(mask, InstanceSegmentationGrid):
        selected_mask = replace(mask, scores=selection.take(mask.scores))
    else:
        selected_mask = selection.take(mask)
    selected = replace(detections, **fields, mask=selected_mask)

    return selected


def _take_key_points(
    key_points: KeyPoints, selection: RowSelection, *, record: Optional[dict]
) -> KeyPoints:
    covariance = key_points.covariance
    detection_confidence = key_points.detection_confidence
    selected = KeyPoints(
        xy=selection.take(key_points.xy),
        class_id=selection.take(key_points.class_id),
        confidence=selection.take(key_points.confidence),
        image_metadata=_with_row_record(key_points.image_metadata, record),
        key_points_metadata=_take_row_dicts(key_points.key_points_metadata, selection),
        covariance=None if covariance is None else selection.take(covariance),
        detection_confidence=(
            None
            if detection_confidence is None
            else selection.take(detection_confidence)
        ),
    )

    return selected


def _take_row_dicts(
    rows: Optional[Sequence[dict]], selection: RowSelection
) -> Optional[List[dict]]:
    # Copies, so a block mutating a selected row cannot change the source.
    if rows is None:
        return None

    selected = [dict(row) for row in selection.take_list(rows)]

    return selected


def _check_mask(mask: Selection, *, rows: int) -> None:
    if isinstance(mask, (torch.Tensor, np.ndarray)):
        expected_dtype = torch.bool if isinstance(mask, torch.Tensor) else np.bool_
        if mask.ndim != 1 or mask.dtype != expected_dtype:
            raise ContractError(
                "mask must be one-dimensional with boolean dtype, "
                f"got shape {tuple(mask.shape)} and dtype {mask.dtype}"
            )
    elif not isinstance(mask, (list, tuple)) or not all(
        isinstance(value, (bool, np.bool_)) for value in mask
    ):
        raise ContractError("mask must be a one-dimensional boolean sequence")

    if len(mask) != rows:
        raise ContractError(
            f"mask has {len(mask)} entries; expected {rows} prediction rows"
        )


def _checked_index_tensor(indices: torch.Tensor, *, rows: int) -> torch.Tensor:
    if indices.ndim != 1:
        raise ContractError("indices must be one-dimensional")
    if indices.dtype not in _INTEGER_TENSOR_DTYPES:
        raise ContractError(f"indices require integer dtype, got {indices.dtype}")

    positions = indices.detach().to(torch.int64)
    if positions.device.type == "cpu" and positions.numel():
        low, high = int(positions.min()), int(positions.max())
        if low < 0 or high >= rows:
            raise ContractError(
                f"prediction indices must lie in [0, {rows}), got [{low}, {high}]"
            )

    return positions


def _checked_host_indices(indices: Selection, *, rows: int) -> List[int]:
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1:
            raise ContractError("indices must be one-dimensional")
        if indices.dtype.kind not in "iu":
            raise ContractError(f"indices require integer dtype, got {indices.dtype}")
        indices = indices.tolist()

    if not isinstance(indices, (list, tuple)) or not all(
        isinstance(index, Integral) and not isinstance(index, (bool, np.bool_))
        for index in indices
    ):
        raise ContractError("indices must be a one-dimensional integer sequence")

    positions = [int(index) for index in indices]
    for index in positions:
        if not 0 <= index < rows:
            raise ContractError(f"prediction index {index} is outside [0, {rows})")

    return positions

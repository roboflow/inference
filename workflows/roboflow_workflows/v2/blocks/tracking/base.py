"""Shared pieces of the per-source tracker blocks.

A tracker block keeps one tracker and one instance cache per video source.
The tracker library works on host ``sv.Detections``; the block's outputs are
rows of the original native prediction::

    image ──► SourceTrackers.state_for ──► SourceState(tracker, instances)
                (source key, max_sources, frame rate; created on first frame)

    predictions ──► host_boxes ──► sv.Detections + input row index ──► tracker
    (any device)    (xyxy, class_id, confidence: one copy per field)     │
                                                                          ▼
    kept_rows ◄── sv.Detections: kept input rows, tracker ids
        │
        ├─ InstanceCache.split: records ids and returns new/seen positions
        └─ tracking_result: selects demanded original rows using those positions
             ├─ tracked_detections      masks, keypoints, row identity,
             ├─ new_instances           metadata, dtype and device
             └─ already_seen_instances  follow the selected input rows

Masks are never decoded or upscaled for tracking: dense and RLE masks are
selected as stored, and a ``SegmentationView`` is selected with
``SegmentationView.select``, keeping its low-resolution scores.

A concrete block supplies the tracker constructor and the per-frame update
call and explicitly advances the instance cache before building outputs.
Policies shared by every tracker block:

* The source is ``image.video_metadata.video_identifier``; there is no
  default source.
* A source's tracker and cache are created on its first frame with that
  call's parameters. Later values, literal or selected, do not reconfigure
  them; a processing reset replaces the block and drops every source.
* ``max_sources`` bounds the sources. The first frame of one source too many
  raises ``TrackerCapacityError`` before any state is created or updated.
  Live sources are never evicted: a block has no callback when a source ends.
* The nominal frame rate is ``frame_rate`` when set, otherwise
  ``video_metadata.fps``, which must then be finite and positive. No rate is
  guessed.
* Only processed frames advance a tracker. An empty prediction is a frame
  without detections and ages tracks; a skipped call is not seen.
* The instance cache remembers at most ``instances_cache_size`` reported ids
  per source, evicting the oldest first; an evicted id counts as new again.
"""

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import supervision as sv
import torch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    select_predictions,
    validated_row_count,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KeyPointPrediction,
)
from roboflow_workflows.v2.engine.declaration import Output

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

DEFAULT_MAX_SOURCES = 256
DEFAULT_INSTANCES_CACHE_SIZE = 16384

TRACKED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

TRACKER_OUTPUTS = MappingProxyType(
    {
        "tracked_detections": Output(
            *TRACKED_KINDS,
            source="image",
            description=(
                "Rows the tracker kept, in tracker order, same family as the "
                "input, with tracker_id in bboxes_metadata."
            ),
        ),
        "new_instances": Output(
            *TRACKED_KINDS,
            source="image",
            description=(
                "Rows of tracked_detections whose tracker_id this source's "
                "instance cache had not seen."
            ),
        ),
        "already_seen_instances": Output(
            *TRACKED_KINDS,
            source="image",
            description=(
                "Rows of tracked_detections whose tracker_id is in this "
                "source's instance cache."
            ),
        ),
    }
)

ROW_INDEX_KEY = "input_row"

TrackedPrediction = Union[
    Detections, InstanceDetections, SegmentationView, KeyPointPrediction
]

_HOST_BOX_DTYPES = (torch.float32, torch.float64)


class TrackerCapacityError(RuntimeError):
    """A new source would exceed the tracker block's ``max_sources``."""


class InstanceCache:
    """FIFO set of tracker ids one source has reported, as in V1.

    An id not in the cache is new and is inserted; when the cache is full the
    oldest inserted id is dropped first. A dropped id that returns is new
    again.

    Args:
        size: Most ids remembered; at least 1.

    Raises:
        ValueError: When ``size`` is smaller than 1.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"instances_cache_size must be at least 1, got {size}")

        self.size = size
        self._order: Deque[int] = deque()
        self._ids: Set[int] = set()

    def __len__(self) -> int:
        return len(self._ids)

    def split(self, tracker_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Record one frame's ids and split their positions into new and seen.

        Args:
            tracker_ids: Host ids of the kept rows, in output order.

        Returns:
            Positions into ``tracker_ids`` of new ids and of already seen
            ids, each in order.
        """
        new_positions = []
        seen_positions = []
        for position, tracker_id in enumerate(tracker_ids):
            if tracker_id in self._ids:
                seen_positions.append(position)
                continue

            self._insert(tracker_id)
            new_positions.append(position)

        return new_positions, seen_positions

    def _insert(self, tracker_id: int) -> None:
        if len(self._order) >= self.size:
            self._ids.discard(self._order.popleft())
        self._order.append(tracker_id)
        self._ids.add(tracker_id)


@dataclass
class SourceState:
    """State of one source: its tracker and its instance cache.

    Attributes:
        tracker: The library tracker, created on the source's first frame.
        instances: Ids this source already reported.
    """

    tracker: Any
    instances: InstanceCache


class SourceTrackers(Mapping):
    """Bounded per-source states of one tracker block instance.

    Read-only mapping of source key to ``SourceState``; ``state_for`` is the
    only way to add a source.

    Args:
        block_type: Block type named in error messages.
    """

    def __init__(self, *, block_type: str) -> None:
        self._block_type = block_type
        self._states: Dict[str, SourceState] = {}

    def __getitem__(self, source: str) -> SourceState:
        return self._states[source]

    def __iter__(self) -> Iterator[str]:
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def state_for(
        self,
        image: ImageData,
        *,
        max_sources: int,
        frame_rate: Optional[float],
        instances_cache_size: int,
        create_tracker: Callable[[float], Any],
    ) -> SourceState:
        """Return the frame's source state, creating it on the first frame.

        For a known source every argument except ``image`` is ignored. For a
        new source the checks run in this order and none leaves state behind:
        capacity, frame rate, cache size, tracker construction.

        Args:
            image: Frame whose ``video_metadata`` names the source.
            max_sources: Most sources kept at once.
            frame_rate: Nominal fps for a new source; ``None`` uses
                ``video_metadata.fps``.
            instances_cache_size: Cache size for a new source.
            create_tracker: Builds a tracker from the nominal frame rate.

        Returns:
            The source's state.

        Raises:
            ValueError: When the image has no video identifier, or a new
                source has no finite positive frame rate, or the cache size is
                below 1.
            TrackerCapacityError: When a new source would exceed
                ``max_sources``.
        """
        source = self._source_of(image)
        state = self._states.get(source)
        if state is not None:
            return state

        if len(self._states) >= max_sources:
            raise TrackerCapacityError(
                f"{self._block_type}: source {source!r} would be source "
                f"{len(self._states) + 1} of max_sources={max_sources}; "
                "raise max_sources or reset processing to drop ended sources"
            )

        rate = self._frame_rate_of(image, source=source, frame_rate=frame_rate)
        instances = InstanceCache(instances_cache_size)
        state = SourceState(tracker=create_tracker(rate), instances=instances)
        self._states[source] = state

        return state

    def _source_of(self, image: ImageData) -> str:
        metadata = image.video_metadata
        if metadata is None or not metadata.video_identifier:
            raise ValueError(
                f"{self._block_type} needs video_metadata.video_identifier"
            )

        return metadata.video_identifier

    def _frame_rate_of(
        self, image: ImageData, *, source: str, frame_rate: Optional[float]
    ) -> float:
        # Direct calls bypass Params validation, so check here too.
        if frame_rate is not None:
            if not _is_positive_finite(frame_rate):
                raise ValueError(
                    f"{self._block_type}: frame_rate must be a finite positive "
                    f"number or null, got {frame_rate!r}"
                )

            return float(frame_rate)

        fps = image.video_metadata.fps
        if not _is_positive_finite(fps):
            raise ValueError(
                f"{self._block_type}: source {source!r} reports "
                f"video_metadata.fps={fps!r}, not a finite positive nominal "
                "rate (a live stream may not report one); set the step's "
                "frame_rate to the source's nominal fps"
            )

        return float(fps)


def host_boxes(prediction: TrackedPrediction) -> sv.Detections:
    """Copy the boxes a tracker needs to the host, tagged with input rows.

    One device-to-host copy each for ``xyxy``, ``class_id`` and
    ``confidence``. Masks and keypoints stay where they are. Boxes in a
    reduced float precision are cast to float32 on their device first.

    Args:
        prediction: Detections, InstanceDetections, a SegmentationView or a
            (KeyPoints, Detections) tuple; not modified.

    Returns:
        ``sv.Detections`` whose ``data[ROW_INDEX_KEY]`` holds each row's
        input position, so the rows a tracker keeps can be mapped back.

    Raises:
        ValueError: When a keypoint prediction has no boxes.
        ContractError: When the prediction is malformed or unsupported.
    """
    rows = _row_count(prediction)
    boxes = _boxes_of(prediction)
    host = sv.Detections(
        xyxy=_to_host_float(boxes.xyxy),
        class_id=boxes.class_id.detach().cpu().numpy(),
        confidence=_to_host_float(boxes.confidence),
        data={ROW_INDEX_KEY: np.arange(rows, dtype=np.int64)},
    )

    return host


def kept_rows(tracked: sv.Detections) -> Tuple[np.ndarray, np.ndarray]:
    """Read the input rows a tracker kept and their tracker ids.

    Rows without a tracker id or with id ``-1`` (unconfirmed tracks in some
    tracker libraries) are dropped.

    Args:
        tracked: The tracker's output for ``host_boxes``' detections.

    Returns:
        ``int64`` input rows and ``int64`` tracker ids, aligned, in the
        tracker's output order.
    """
    empty = np.empty((0,), dtype=np.int64)
    rows = tracked.data.get(ROW_INDEX_KEY) if tracked.data else None
    if rows is None or tracked.tracker_id is None or len(tracked) == 0:
        return empty, empty

    tracker_ids = np.asarray(tracked.tracker_id, dtype=np.int64)
    confirmed = tracker_ids != -1
    kept = (
        np.asarray(rows, dtype=np.int64)[confirmed],
        tracker_ids[confirmed],
    )

    return kept


def tracking_result(
    prediction: TrackedPrediction,
    *,
    rows: np.ndarray,
    tracker_ids: np.ndarray,
    new_positions: List[int],
    seen_positions: List[int],
    wanted_outputs: Set[str],
) -> Dict[str, Any]:
    """Select demanded tracker outputs without changing source state.

    Args:
        prediction: The frame's original prediction; not modified.
        rows: Kept input rows from ``kept_rows``.
        tracker_ids: Their tracker ids, aligned with ``rows``.
        new_positions: Positions of newly reported ids in the kept rows.
        seen_positions: Positions of previously reported ids in the kept rows.
        wanted_outputs: Names of the outputs to construct; empty builds none.

    Returns:
        Requested outputs among ``tracked_detections``, ``new_instances`` and
        ``already_seen_instances``, as predictions of the input's family.
        New and seen positions partition the tracked rows and keep their order.
    """
    ids = tracker_ids.tolist()
    positions = {
        "tracked_detections": np.arange(len(rows), dtype=np.int64),
        "new_instances": np.asarray(new_positions, dtype=np.int64),
        "already_seen_instances": np.asarray(seen_positions, dtype=np.int64),
    }
    result = {}
    for name, selected_positions in positions.items():
        if name not in wanted_outputs:
            continue

        result[name] = _tracked_rows(
            prediction,
            rows=rows[selected_positions],
            ids=[ids[position] for position in selected_positions],
        )

    return result


def _tracked_rows(
    prediction: TrackedPrediction, *, rows: np.ndarray, ids: List[int]
) -> TrackedPrediction:
    if isinstance(prediction, SegmentationView):
        selected = prediction.select(RowSelection.of(len(prediction), indices=rows))
        rows_metadata = _with_tracker_ids(selected.detections.bboxes_metadata, ids)
        tracked_view = selected.with_row_metadata(rows_metadata)
        return tracked_view

    # A fresh selection: setting its metadata cannot reach the input.
    tracked = select_predictions(prediction, indices=rows)
    boxes = _boxes_of(tracked)
    boxes.bboxes_metadata = _with_tracker_ids(boxes.bboxes_metadata, ids)

    return tracked


def _row_count(prediction: TrackedPrediction) -> int:
    if isinstance(prediction, SegmentationView):
        INSTANCE_SEGMENTATION_VIEW_KIND.check(prediction)
        return len(prediction)

    rows = validated_row_count(prediction)

    return rows


def _boxes_of(
    prediction: TrackedPrediction,
) -> Union[Detections, InstanceDetections]:
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    if not isinstance(prediction, tuple):
        return prediction

    boxes = prediction[1]
    if boxes is None:
        raise ValueError(
            "Tracking needs boxes: this keypoint prediction has KeyPoints "
            "without Detections"
        )

    return boxes


def _to_host_float(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach()
    if tensor.dtype not in _HOST_BOX_DTYPES:
        tensor = tensor.to(torch.float32)
    host = tensor.cpu().numpy()

    return host


def _with_tracker_ids(
    existing: Optional[List[dict]], tracker_ids: List[int]
) -> Optional[List[dict]]:
    # New dicts, so no row dict of any input or selection changes.
    rows_metadata = []
    for row, tracker_id in enumerate(tracker_ids):
        row_metadata = dict(existing[row]) if existing and existing[row] else {}
        row_metadata["tracker_id"] = tracker_id
        rows_metadata.append(row_metadata)

    return rows_metadata or None


def _is_positive_finite(value: Any) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    return is_number and math.isfinite(value) and value > 0

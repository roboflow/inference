"""Adapter shared by the blocks that wrap trackers from the ``trackers`` package.

The SORT, ByteTrack, OC-SORT and BoT-SORT blocks each subclass
``TrackersPackageBlock``; it holds what they share::

    run(knobs) ──► TrackersPackageBlock.track
                     ├─ host_boxes(predictions)           boxes to host, once
                     ├─ SourceTrackers.state_for(image)   first frame: tracker
                     │                                    from knobs + rate
                     ├─ update_tracker(tracker, boxes)    library call (CPU)
                     ├─ kept_rows + InstanceCache.split   state advances here
                     └─ tracking_result(wanted outputs)   no state change

Shared source, cache, capacity and frame-rate policies live in ``base``. The
V1 ``detections`` input is ``predictions`` here. Unlike V1, a source without a
finite positive ``video_metadata.fps`` needs ``frame_rate``; V1 assumed 30.

The library trackers run on the host CPU over a box-only ``sv.Detections``.
Masks, keypoints and row metadata stay on the input's device and are selected,
not copied to the host. ``trackers`` is imported when a block is created, so
the catalogue imports without it.

Tracker ids come from a counter that the ``trackers`` package shares between
all trackers of one algorithm in the process, as in V1. Ids are therefore
unique per algorithm across sources and blocks, but a source's first id is not
necessarily 0. The blocks never call the library's ``reset``, which would
rewind that shared counter for every source.
"""

import importlib
from typing import Any, Callable, Dict, Optional

import supervision as sv
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
    DEFAULT_MAX_SOURCES,
    TRACKED_KINDS,
    TRACKER_OUTPUTS,
    SourceTrackers,
    TrackedPrediction,
    host_boxes,
    kept_rows,
    tracking_result,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine import NoExecutionContextError, get_execution_context
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref

FIRST_FRAME_NOTE = (
    "Read once per source, at its first frame; a later value, selected or "
    "edited, does not retune it until a processing reset."
)

__all__ = [
    "FIRST_FRAME_NOTE",
    "TrackersPackageBlock",
    "fraction_knob",
    "frame_count_knob",
]


def fraction_knob(default: float, *, name: str, description: str) -> Any:
    """Describe a tracker knob in [0, 1], literal or selected.

    Args:
        default: The V1 default.
        name: Field name, used for the selector example.
        description: What the knob does; the first-frame note is appended.

    Returns:
        A pydantic ``Field`` for a ``StrictFloat | Ref(FLOAT_KIND)`` field.
    """
    field = Field(
        default=default,
        ge=0,
        le=1,
        description=f"{description} {FIRST_FRAME_NOTE}",
        examples=[default, f"$inputs.{name}"],
    )

    return field


def frame_count_knob(default: int, *, name: str, description: str) -> Any:
    """Describe a tracker knob counting frames (at least 1), literal or selected.

    Args:
        default: The V1 default.
        name: Field name, used for the selector example.
        description: What the knob does; the first-frame note is appended.

    Returns:
        A pydantic ``Field`` for a ``StrictInt | Ref(INTEGER_KIND)`` field.
    """
    field = Field(
        default=default,
        ge=1,
        description=f"{description} {FIRST_FRAME_NOTE}",
        examples=[default, f"$inputs.{name}"],
    )

    return field


class TrackersPackageBlock(Block):
    """Per-source tracking with one ``trackers`` package tracker per source.

    Abstract: a concrete block sets ``type``, ``library_class``, its ``Params``
    knobs and a ``run`` that passes those knobs to ``track``. Every output has
    the input's family and image metadata: rows the tracker kept, in tracker
    order, each with its ``tracker_id`` in ``bboxes_metadata``. Rows the
    library reports with id ``-1`` (not yet confirmed) are dropped.
    ``new_instances`` and ``already_seen_instances`` split the kept rows by
    whether the source reported the id before. The input is not changed.

    Raises:
        ImportError: On construction, when the ``trackers`` package is absent.
    """

    library_class: str = ""
    outputs = TRACKER_OUTPUTS

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and, unless "
                "frame_rate is set, gives its nominal fps."
            )
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Detections, instance segmentations (dense, RLE or a "
                "segmentation view) or keypoints with boxes of this frame "
                "(V1 input 'detections')."
            )
        )
        instances_cache_size: StrictInt = Field(
            default=DEFAULT_INSTANCES_CACHE_SIZE,
            ge=1,
            description=(
                "Tracker ids each source remembers to tell new_instances from "
                "already_seen_instances; the oldest is dropped first, and a "
                f"dropped id counts as new again. {FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_INSTANCES_CACHE_SIZE],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources tracked at once. A frame of one more source "
                "fails with TrackerCapacityError; no tracker is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )
        frame_rate: Optional[StrictFloat] = Field(
            default=None,
            gt=0,
            allow_inf_nan=False,
            description=(
                "Nominal source fps that scales lost_track_buffer, for every "
                "source. Null uses video_metadata.fps and fails when a source "
                "does not report a finite positive fps (e.g. live RTSP); V1 "
                "assumed 30. Not the processing rate; dropped frames do not "
                "age tracks."
            ),
            examples=[None, 30.0],
        )

    def __init__(self):
        self._tracker_class = _load_library_class(
            self.library_class, block_type=self.type
        )
        self._trackers = SourceTrackers(block_type=self.type)

    def track(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        max_sources: int,
        frame_rate: Optional[float],
        instances_cache_size: int,
        tracker_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update this source's tracker and instance cache with one frame.

        Both updates happen before outputs are selected, so state advances
        even when no output is demanded.

        Args:
            image: Frame naming the source in ``video_metadata``.
            predictions: Detections of this frame; not modified.
            max_sources: Most sources this block keeps a tracker for.
            frame_rate: Nominal fps for a new source's tracker; None uses
                ``video_metadata.fps``.
            instances_cache_size: Tracker ids a new source's cache keeps.
            tracker_settings: Library constructor arguments other than
                ``frame_rate``; read only for a new source.

        Returns:
            Demanded outputs among ``tracked_detections``, ``new_instances``
            and ``already_seen_instances``. Direct calls return all three.

        Raises:
            ValueError: When the image has no video identifier, when a
                keypoint prediction has no boxes, when ``frame_rate`` is not a
                finite positive number, or when it is None and
                ``video_metadata.fps`` is not one.
            TrackerCapacityError: When this is the first frame of a source and
                ``max_sources`` sources already have trackers.
        """
        # Copied first: a malformed prediction leaves no source state behind.
        boxes = host_boxes(predictions)

        def create_tracker(rate: float) -> Any:
            tracker = self._tracker_class(frame_rate=rate, **tracker_settings)

            return tracker

        state = self._trackers.state_for(
            image,
            max_sources=max_sources,
            frame_rate=frame_rate,
            instances_cache_size=instances_cache_size,
            create_tracker=create_tracker,
        )

        tracked = self.update_tracker(state.tracker, boxes, image=image)
        rows, tracker_ids = kept_rows(tracked)
        new_positions, seen_positions = state.instances.split(tracker_ids.tolist())

        result = tracking_result(
            predictions,
            rows=rows,
            tracker_ids=tracker_ids,
            new_positions=new_positions,
            seen_positions=seen_positions,
            wanted_outputs=self._wanted_outputs(),
        )

        return result

    def update_tracker(
        self, tracker: Any, boxes: sv.Detections, *, image: ImageData
    ) -> sv.Detections:
        """Advance a library tracker by one frame.

        Override when the library needs frame pixels as well as boxes.

        Args:
            tracker: This source's library tracker.
            boxes: Host boxes from ``host_boxes``.
            image: The frame; unused here.

        Returns:
            The library's tracked detections.
        """
        tracked = tracker.update(boxes)

        return tracked

    def _wanted_outputs(self) -> set[str]:
        # wants() requires a running call. Detect a plain run() through the
        # public context API; errors in an engine call must still propagate.
        try:
            get_execution_context()
        except NoExecutionContextError:
            return set(self.outputs)

        wanted = {name for name in self.outputs if self.wants(name)}

        return wanted


def _load_library_class(name: str, *, block_type: str) -> Callable[..., Any]:
    try:
        trackers = importlib.import_module("trackers")
    except ImportError as error:
        raise ImportError(
            f"{block_type} needs the 'trackers' package, a roboflow-workflows "
            "dependency: pip install 'trackers~=2.4.0'"
        ) from error

    library_class = getattr(trackers, name)

    return library_class

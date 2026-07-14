"""Shared base classes for tracker workflow blocks (tensor-native sibling).

Tensor-native counterpart of ``_base.py``. Under
``ENABLE_TENSOR_DATA_REPRESENTATION`` the loader swaps each concrete tracker
block to its ``*_tensor.py`` sibling; those siblings import ``TrackerBlockBase``
/ ``TRACKER_PREDICTION_KINDS`` / ``tracker_describe_outputs`` from here instead
of from ``_base.py``.

The only representation-specific change versus ``_base.py`` is the native
input/output handling in ``_run_tracker``: detection predictions are native
``inference_models`` objects (``Detections`` / ``InstanceDetections`` / the
``(KeyPoints, Optional[Detections])`` keypoint tuple) rather than
``sv.Detections``. The third-party tracker libraries (``trackers`` package) are
``sv.Detections``-based, so ``_run_tracker`` materialises a minimal
``sv.Detections`` (bounding boxes only, with a stashed row index) as the
transport to/from the tracker, then maps the surviving rows back onto the
ORIGINAL native input — preserving masks / keypoints / all native metadata —
and writes the assigned ``tracker_id`` as a same-device tensor field. Block
outputs are always native
``inference_models`` objects; ``sv.Detections`` is used only as the algorithm
boundary to the tracker library. Metadata receives IDs only when an explicit
legacy iteration or serialization boundary requests Python values.

Each concrete tracker block (ByteTrack, BoT-SORT, SORT, OC-SORT) inherits from
``TrackerBlockBase`` and implements ``_create_tracker`` and ``get_manifest``.
Sub-classes may override ``_tracker_update`` when the underlying tracker needs
extra per-frame context (e.g. a video frame for camera motion compensation).
``_tracker_update`` / ``_create_tracker`` stay ``sv.Detections``-based /
library-based and identical to ``_base.py`` (the third-party trackers are
``sv``-based) — only ``_run_tracker`` does the native↔sv conversion.
"""

import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import supervision as sv
import torch

from inference.core import logger
from inference.core.workflows.core_steps.common.tensor_native import (
    split_key_point_prediction,
    take_prediction_by_indices,
)
from inference.core.workflows.core_steps.trackers.batch_scheduler import (
    get_tracker_batch_scheduler,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

OUTPUT_KEY: str = "tracked_detections"

#: Key under which the per-detection row index into the original native input is
#: stashed inside the transport ``sv.Detections.data`` dict. The third-party
#: tracker libraries index back into the input ``sv.Detections`` (preserving the
#: ``.data`` dict) so this row index travels through ``tracker.update`` and lets
#: us slice the ORIGINAL native input by the surviving rows afterwards.
_TRACKER_ROW_INDEX_KEY: str = "__tracker_row_index__"

#: Detection kinds accepted as tracker input and declared on tracker output.
#: Trackers only use bounding boxes for association and preserve all other
#: fields (masks, keypoints, custom data) via native indexing back into the
#: original ``inference_models`` prediction.
TRACKER_PREDICTION_KINDS = [
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
]


class InstanceCache:
    """Device-resident FIFO cache for exact new/seen tracker classification.

    Used to categorize tracked detections as new (first appearance) or
    already seen (reappearance) across video frames.
    """

    def __init__(self, size: int):
        self._size = max(1, size)
        self._ids: Optional[torch.Tensor] = None
        self._valid: Optional[torch.Tensor] = None
        self._write_index: Optional[torch.Tensor] = None

    def record_instances(self, tracker_ids: torch.Tensor) -> torch.Tensor:
        """Record an ID tensor and return an exact same-device seen mask.

        Tracktor outputs guarantee unique IDs within a frame. That contract lets
        membership, insertion ranks, wraparound positions, and final ring writes
        run as vector operations rather than one launch sequence per object.
        """
        tracker_ids = tracker_ids.to(dtype=torch.long).reshape(-1)
        self._ensure_device(tracker_ids.device)
        if tracker_ids.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=tracker_ids.device)
        assert self._ids is not None
        assert self._valid is not None
        assert self._write_index is not None
        cache_ids = self._ids[: self._size]
        valid = self._valid[: self._size]
        seen = (tracker_ids[:, None].eq(cache_ids[None, :]) & valid[None, :]).any(dim=1)
        new = ~seen
        insertion_rank = torch.cumsum(new.to(dtype=torch.long), dim=0) - 1
        insertion_slot = torch.remainder(
            self._write_index + insertion_rank,
            self._size,
        )
        sentinel = torch.full_like(insertion_slot, self._size)
        scatter_slot = torch.where(new, insertion_slot, sentinel)

        # Multiple complete wraps can target one slot. Select the greatest
        # insertion rank so the final ring contains the newest ID for that slot.
        last_rank = torch.full(
            (self._size + 1,),
            -1,
            dtype=torch.long,
            device=tracker_ids.device,
        )
        last_rank.scatter_reduce_(
            0,
            scatter_slot,
            torch.where(new, insertion_rank, torch.full_like(insertion_rank, -1)),
            reduce="amax",
            include_self=True,
        )
        selected_ids = tracker_ids.index_select(0, last_rank.clamp_min(0))
        updated = last_rank >= 0
        self._ids.copy_(torch.where(updated, selected_ids, self._ids))
        self._valid.logical_or_(updated)
        self._write_index.copy_(
            torch.remainder(self._write_index + new.sum(dtype=torch.long), self._size)
        )
        return seen

    def _ensure_device(self, device: torch.device) -> None:
        """Initialize fixed cache tensors or reject cross-device video state."""
        if self._ids is not None:
            if self._ids.device != device:
                raise ValueError("tracker cache device changed for an active video")
            return
        self._ids = torch.zeros(self._size + 1, dtype=torch.long, device=device)
        self._valid = torch.zeros(self._size + 1, dtype=torch.bool, device=device)
        self._write_index = torch.zeros((), dtype=torch.long, device=device)

    def record_instance(self, tracker_id: int) -> bool:
        """Retain the legacy scalar API as an explicit host-facing adapter.

        Returns:
            True if the tracker_id was already in the cache (seen before),
            False if this is its first appearance.
        """
        seen = self.record_instances(torch.tensor([tracker_id], dtype=torch.long))
        return bool(seen[0].item())


# Native prediction shapes accepted by tracker blocks (object detection,
# instance segmentation, RLE instance segmentation, or the keypoint-detection
# tuple). Mirrors TRACKER_PREDICTION_KINDS.
TensorNativeTrackerPrediction = Union[
    Detections,
    InstanceDetections,
    Tuple[KeyPoints, Optional[Detections]],
]


class TrackerBlockBase(WorkflowBlock):
    """Common run-loop shared by every tracker block.

    Sub-classes implement ``_create_tracker`` and ``get_manifest``.  Override
    ``_tracker_update`` only when the tracker API requires additional context
    beyond ``sv.Detections`` (e.g. BoT-SORT with camera motion compensation).
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, Any] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]: ...

    @abstractmethod
    def _create_tracker(self, fps: int, **kwargs: Any) -> Any:
        """Instantiate the concrete tracker with algorithm-specific params."""
        ...

    def _tracker_update(
        self,
        tracker: Any,
        detections: sv.Detections,
        image: WorkflowImageData,
    ) -> sv.Detections:
        """Invoke the tracker for one frame.

        Must call ``tracker.update`` only with arguments that library trackers
        define for the per-frame step (typically detections, optionally a frame
        tensor).  Do **not** pass workflow/block kwargs used in ``_create_tracker``.
        """
        return tracker.update(detections)

    def _tracker_batch_frame(
        self,
        tracker: Any,
        image: WorkflowImageData,
    ) -> Any | None:
        """Return optional per-frame context passed to batched Tracktors."""
        return None

    @staticmethod
    def _can_batch_tracker_update(detections: sv.Detections) -> bool:
        """Return whether this call can enter the CUDA micro-batch scheduler."""
        if os.getenv("TRACKTORS_DISABLE_BATCHING", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return False
        boxes = detections.xyxy
        if not isinstance(boxes, torch.Tensor) or boxes.device.type != "cuda":
            return False
        try:
            import tracktors
        except ImportError:
            return False
        return callable(getattr(tracktors, "update_batch", None))

    def _run_tracker(
        self,
        image: WorkflowImageData,
        detections: TensorNativeTrackerPrediction,
        instances_cache_size: int,
        **tracker_kwargs: Any,
    ) -> BlockResult:
        """Run one frame through the tracker.

        Note: tracker parameters (``tracker_kwargs``) are only used when the
        tracker is **first created** for a given ``video_identifier``.
        Changing parameter values on subsequent frames has no effect because
        the tracker instance is cached for the lifetime of the video stream.

        Tensor-native note: ``detections`` is a native ``inference_models``
        prediction. Its bounding-box tensors are wrapped by ``sv.Detections``
        without moving them off-device. Surviving rows are mapped back onto the
        original native input so masks, keypoints, and metadata are preserved.
        """
        video_id, tracker, bbox, sv_input = self._prepare_tracker_input(
            image=image,
            detections=detections,
            tracker_kwargs=tracker_kwargs,
        )

        if self._can_batch_tracker_update(sv_input):
            tracked_sv = get_tracker_batch_scheduler().update(
                tracker,
                sv_input,
                frame=self._tracker_batch_frame(tracker, image),
            )
        else:
            tracked_sv = self._tracker_update(tracker, sv_input, image)
        tracked_detections, tracker_ids_tensor = self._recover_tracker_output(
            detections=detections,
            bbox=bbox,
            tracked_sv=tracked_sv,
        )
        return self._build_tracker_result(
            video_id=video_id,
            tracked_detections=tracked_detections,
            tracker_ids=tracker_ids_tensor,
            instances_cache_size=instances_cache_size,
        )

    def _run_tracker_auto(
        self,
        image: Union[WorkflowImageData, Batch[WorkflowImageData]],
        detections: Union[
            TensorNativeTrackerPrediction,
            Batch[TensorNativeTrackerPrediction],
        ],
        instances_cache_size: int,
        **tracker_kwargs: Any,
    ) -> Union[BlockResult, List[BlockResult]]:
        """Dispatch direct callers scalarly and execution-engine SIMD as a batch."""
        image_is_batch = isinstance(image, Batch)
        detections_are_batch = isinstance(detections, Batch)
        if image_is_batch != detections_are_batch:
            raise ValueError("tracker image and detections must both be batched")
        if image_is_batch and detections_are_batch:
            return self._run_tracker_batch(
                images=image,
                detections=detections,
                instances_cache_size=instances_cache_size,
                **tracker_kwargs,
            )
        return self._run_tracker(
            image=image,
            detections=detections,
            instances_cache_size=instances_cache_size,
            **tracker_kwargs,
        )

    def _run_tracker_batch(
        self,
        images: Batch[WorkflowImageData],
        detections: Batch[TensorNativeTrackerPrediction],
        instances_cache_size: int,
        **tracker_kwargs: Any,
    ) -> List[BlockResult]:
        """Run one aligned SIMD batch with a single Tracktors batch invocation."""
        if len(images) != len(detections) or images.indices != detections.indices:
            raise ValueError("tracker image and detection batches must be aligned")
        prepared = [
            self._prepare_tracker_input(
                image=image,
                detections=prediction,
                tracker_kwargs=tracker_kwargs,
            )
            for image, prediction in zip(images, detections)
        ]
        trackers = [item[1] for item in prepared]
        sv_inputs = [item[3] for item in prepared]
        can_batch = len({id(tracker) for tracker in trackers}) == len(trackers) and all(
            self._can_batch_tracker_update(item) for item in sv_inputs
        )
        if can_batch:
            tracked_outputs = get_tracker_batch_scheduler().execute_batch(
                trackers,
                sv_inputs,
                frames=[
                    self._tracker_batch_frame(tracker, image)
                    for tracker, image in zip(trackers, images)
                ],
                timestamps=[None] * len(trackers),
            )
        else:
            tracked_outputs = [
                self._tracker_update(tracker, sv_input, image)
                for tracker, sv_input, image in zip(trackers, sv_inputs, images)
            ]
        if len(tracked_outputs) != len(prepared):
            raise RuntimeError("tracktors.update_batch returned the wrong batch size")

        recovered = [
            self._recover_tracker_output(
                detections=prediction,
                bbox=prepared_item[2],
                tracked_sv=tracked_sv,
            )
            for prediction, prepared_item, tracked_sv in zip(
                detections,
                prepared,
                tracked_outputs,
            )
        ]
        return [
            self._build_tracker_result(
                video_id=prepared_item[0],
                tracked_detections=tracked_detections,
                tracker_ids=tracker_ids,
                instances_cache_size=instances_cache_size,
            )
            for prepared_item, (tracked_detections, tracker_ids) in zip(
                prepared,
                recovered,
            )
        ]

    def _prepare_tracker_input(
        self,
        image: WorkflowImageData,
        detections: TensorNativeTrackerPrediction,
        tracker_kwargs: Dict[str, Any],
    ) -> Tuple[str, Any, Union[Detections, InstanceDetections], sv.Detections]:
        """Resolve per-video state and wrap native bounding boxes for Tracktors."""
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 30
            logger.warning(
                f"fps not available in VideoMetadata for {self.__class__.__name__}, "
                "defaulting to 30 fps for tracker initialisation"
            )
        video_id = metadata.video_identifier
        if video_id not in self._trackers:
            self._trackers[video_id] = self._create_tracker(
                fps=fps,
                **tracker_kwargs,
            )
        tracker = self._trackers[video_id]
        _, bbox = split_key_point_prediction(detections)
        row_count = int(bbox.xyxy.shape[0])
        sv_input = sv.Detections(
            xyxy=bbox.xyxy,
            confidence=bbox.confidence,
            class_id=bbox.class_id,
            data={
                _TRACKER_ROW_INDEX_KEY: torch.arange(
                    row_count,
                    dtype=torch.long,
                    device=bbox.xyxy.device,
                )
            },
        )
        return video_id, tracker, bbox, sv_input

    @staticmethod
    def _recover_tracker_output(
        detections: TensorNativeTrackerPrediction,
        bbox: Union[Detections, InstanceDetections],
        tracked_sv: sv.Detections,
    ) -> Tuple[TensorNativeTrackerPrediction, torch.Tensor]:
        """Map tracker rows back to native tensors and attach device-resident IDs."""
        if tracked_sv.tracker_id is not None and len(tracked_sv) > 0:
            tracked_sv = tracked_sv[tracked_sv.tracker_id != -1]
        has_rows = (
            tracked_sv.data
            and _TRACKER_ROW_INDEX_KEY in tracked_sv.data
            and tracked_sv.tracker_id is not None
            and len(tracked_sv) > 0
        )
        if has_rows:
            surviving = torch.as_tensor(
                tracked_sv.data[_TRACKER_ROW_INDEX_KEY],
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
            tracker_ids = torch.as_tensor(
                tracked_sv.tracker_id,
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
        else:
            surviving = torch.empty(0, dtype=torch.long, device=bbox.xyxy.device)
            tracker_ids = torch.empty(0, dtype=torch.long, device=bbox.xyxy.device)
        tracked_detections = take_prediction_by_indices(detections, surviving)
        _bbox_component(tracked_detections).tracker_id = tracker_ids
        return tracked_detections, tracker_ids

    def _build_tracker_result(
        self,
        video_id: str,
        tracked_detections: TensorNativeTrackerPrediction,
        tracker_ids: torch.Tensor,
        instances_cache_size: int,
    ) -> BlockResult:
        """Classify exact FIFO new/seen outputs at the explicit cache boundary."""
        if video_id not in self._per_video_cache:
            self._per_video_cache[video_id] = InstanceCache(size=instances_cache_size)
        seen = self._per_video_cache[video_id].record_instances(tracker_ids)
        not_seen_indices = torch.nonzero(~seen, as_tuple=False).reshape(-1)
        seen_indices = torch.nonzero(seen, as_tuple=False).reshape(-1)
        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": take_prediction_by_indices(
                tracked_detections,
                not_seen_indices,
            ),
            "already_seen_instances": take_prediction_by_indices(
                tracked_detections,
                seen_indices,
            ),
        }


def _bbox_component(
    prediction: TensorNativeTrackerPrediction,
) -> Union[Detections, InstanceDetections]:
    """Return the bounding-box-bearing component of a tracker prediction (the
    Detections/InstanceDetections itself, or the bbox element of the keypoint
    tuple)."""
    _, bbox = split_key_point_prediction(prediction)
    return bbox


def tracker_describe_outputs() -> List[OutputDefinition]:
    """Output definitions shared by all tracker blocks.

    Trackers preserve all detection fields (masks, keypoints, custom data) —
    they only use bounding boxes for association then index back into the
    original native prediction.  The output kinds therefore mirror the input
    kinds accepted by every tracker manifest.
    """
    return [
        OutputDefinition(name=OUTPUT_KEY, kind=TRACKER_PREDICTION_KINDS),
        OutputDefinition(name="new_instances", kind=TRACKER_PREDICTION_KINDS),
        OutputDefinition(
            name="already_seen_instances",
            kind=TRACKER_PREDICTION_KINDS,
        ),
    ]

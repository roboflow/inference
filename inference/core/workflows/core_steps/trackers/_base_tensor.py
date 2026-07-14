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
from inference.core.workflows.core_steps.trackers.instance_cache_kernels import (
    InstanceCacheKernelResult,
    instance_cache_hash_capacity,
    run_triton_instance_cache,
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
        self._count: Optional[torch.Tensor] = None
        self._ring_hash_slots: Optional[torch.Tensor] = None
        self._hash_keys: Optional[torch.Tensor] = None
        self._hash_values: Optional[torch.Tensor] = None
        self._batch_arena: Optional["_InstanceCacheBatchArena"] = None

    def record_instances(self, tracker_ids: torch.Tensor) -> torch.Tensor:
        """Record an ID tensor and return an exact same-device seen mask.

        CUDA uses one exact sequential FIFO/hash program. The fallback preserves
        the same ordering for duplicate IDs and interleaved eviction.
        """
        tracker_ids = tracker_ids.to(dtype=torch.long).reshape(-1)
        self._ensure_device(tracker_ids.device)
        if tracker_ids.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=tracker_ids.device)
        fused = self._record_instances_triton(tracker_ids)
        if fused is not None:
            return fused.seen
        return self._record_instances_fallback(tracker_ids)

    def _record_instances_triton(
        self,
        tracker_ids: torch.Tensor,
    ) -> Optional[InstanceCacheKernelResult]:
        """Run the single-stream exact kernel when CUDA Triton is available."""
        assert self._ids is not None
        assert self._valid is not None
        assert self._write_index is not None
        assert self._count is not None
        assert self._ring_hash_slots is not None
        assert self._hash_keys is not None
        assert self._hash_values is not None
        metadata = torch.tensor(
            [[0, tracker_ids.numel()]],
            dtype=torch.long,
            device=tracker_ids.device,
        )
        cache_rows = torch.zeros(
            1,
            dtype=torch.long,
            device=tracker_ids.device,
        )
        return run_triton_instance_cache(
            tracker_ids,
            metadata,
            cache_rows,
            self._ids.reshape(1, -1),
            self._valid.reshape(1, -1),
            self._ring_hash_slots.reshape(1, -1),
            self._write_index.reshape(1),
            self._count.reshape(1),
            self._hash_keys.reshape(1, -1),
            self._hash_values.reshape(1, -1),
            cache_size=self._size,
            max_inputs=tracker_ids.numel(),
        )

    def _record_instances_fallback(
        self,
        tracker_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exact sequential FIFO semantics with ordinary Torch operators."""
        assert self._ids is not None
        assert self._valid is not None
        assert self._write_index is not None
        assert self._count is not None
        seen = torch.empty(
            tracker_ids.numel(),
            dtype=torch.bool,
            device=tracker_ids.device,
        )
        for position in range(tracker_ids.numel()):
            tracker_id = tracker_ids[position]
            was_seen = (
                self._ids[: self._size].eq(tracker_id) & self._valid[: self._size]
            ).any()
            seen[position] = was_seen
            if bool(was_seen.item()):
                continue
            write_index = int(self._write_index.item())
            self._ids[write_index] = tracker_id
            self._valid[write_index] = True
            self._write_index.fill_((write_index + 1) % self._size)
            self._count.add_(1).clamp_max_(self._size)
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
        self._count = torch.zeros((), dtype=torch.int32, device=device)
        self._ring_hash_slots = torch.full(
            (self._size + 1,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        hash_capacity = instance_cache_hash_capacity(self._size)
        self._hash_keys = torch.empty(
            hash_capacity,
            dtype=torch.long,
            device=device,
        )
        self._hash_values = torch.full(
            (hash_capacity,),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def record_instance(self, tracker_id: int) -> bool:
        """Retain the legacy scalar API as an explicit host-facing adapter.

        Returns:
            True if the tracker_id was already in the cache (seen before),
            False if this is its first appearance.
        """
        seen = self.record_instances(torch.tensor([tracker_id], dtype=torch.long))
        return bool(seen[0].item())


class _InstanceCacheBatchArena:
    """Persistent row-major cache state for a cohort of video streams.

    Each ``InstanceCache`` remains the source of truth and is rebound to a row
    view when it first joins the arena. On CUDA, one Triton program per stream
    performs exact sequential FIFO lookup, eviction, hashing, and stable output
    partitioning without scanning the full cache or launching per-stream work.
    """

    def __init__(self, size: int, device: torch.device) -> None:
        self._size = size
        self._device = device
        self._hash_capacity = instance_cache_hash_capacity(size)
        self._capacity = 0
        self._ids = torch.empty((0, size + 1), dtype=torch.long, device=device)
        self._valid = torch.empty((0, size + 1), dtype=torch.bool, device=device)
        self._write_index = torch.empty(0, dtype=torch.long, device=device)
        self._count = torch.empty(0, dtype=torch.int32, device=device)
        self._ring_hash_slots = torch.empty(
            (0, size + 1),
            dtype=torch.int32,
            device=device,
        )
        self._hash_keys = torch.empty(
            (0, self._hash_capacity),
            dtype=torch.long,
            device=device,
        )
        self._hash_values = torch.empty(
            (0, self._hash_capacity),
            dtype=torch.int32,
            device=device,
        )
        self._row_indices = torch.empty(0, dtype=torch.long, device=device)
        self._cache_rows: Dict[int, int] = {}
        self._caches: List[InstanceCache] = []

    def _grow(self, minimum_capacity: int) -> None:
        if minimum_capacity <= self._capacity:
            return
        new_capacity = max(minimum_capacity, max(4, self._capacity * 2))
        new_ids = torch.zeros(
            (new_capacity, self._size + 1),
            dtype=torch.long,
            device=self._device,
        )
        new_valid = torch.zeros(
            (new_capacity, self._size + 1),
            dtype=torch.bool,
            device=self._device,
        )
        new_write_index = torch.zeros(
            new_capacity,
            dtype=torch.long,
            device=self._device,
        )
        new_count = torch.zeros(
            new_capacity,
            dtype=torch.int32,
            device=self._device,
        )
        new_ring_hash_slots = torch.full(
            (new_capacity, self._size + 1),
            -1,
            dtype=torch.int32,
            device=self._device,
        )
        new_hash_keys = torch.empty(
            (new_capacity, self._hash_capacity),
            dtype=torch.long,
            device=self._device,
        )
        new_hash_values = torch.full(
            (new_capacity, self._hash_capacity),
            -1,
            dtype=torch.int32,
            device=self._device,
        )
        if self._capacity:
            new_ids[: self._capacity].copy_(self._ids)
            new_valid[: self._capacity].copy_(self._valid)
            new_write_index[: self._capacity].copy_(self._write_index)
            new_count[: self._capacity].copy_(self._count)
            new_ring_hash_slots[: self._capacity].copy_(self._ring_hash_slots)
            new_hash_keys[: self._capacity].copy_(self._hash_keys)
            new_hash_values[: self._capacity].copy_(self._hash_values)
        self._ids = new_ids
        self._valid = new_valid
        self._write_index = new_write_index
        self._count = new_count
        self._ring_hash_slots = new_ring_hash_slots
        self._hash_keys = new_hash_keys
        self._hash_values = new_hash_values
        self._row_indices = torch.arange(
            new_capacity,
            dtype=torch.long,
            device=self._device,
        )
        self._capacity = new_capacity
        for row, cache in enumerate(self._caches):
            self._bind_views(cache, row)

    def _bind_views(self, cache: InstanceCache, row: int) -> None:
        cache._ids = self._ids[row]
        cache._valid = self._valid[row]
        cache._write_index = self._write_index[row]
        cache._count = self._count[row]
        cache._ring_hash_slots = self._ring_hash_slots[row]
        cache._hash_keys = self._hash_keys[row]
        cache._hash_values = self._hash_values[row]
        cache._batch_arena = self

    def _ensure_caches(self, caches: List[InstanceCache]) -> List[int]:
        rows: List[int] = []
        for cache in caches:
            if cache._size != self._size:
                raise ValueError("tracker cache size changed for an active video")
            if cache._batch_arena not in (None, self):
                raise ValueError("tracker cache arena changed for an active video")
            cache_key = id(cache)
            row = self._cache_rows.get(cache_key)
            if row is None:
                row = len(self._caches)
                old_ids = cache._ids
                old_valid = cache._valid
                old_write_index = cache._write_index
                old_count = cache._count
                old_ring_hash_slots = cache._ring_hash_slots
                old_hash_keys = cache._hash_keys
                old_hash_values = cache._hash_values
                if old_ids is not None and old_ids.device != self._device:
                    raise ValueError("tracker cache device changed for an active video")
                self._grow(row + 1)
                if old_ids is not None:
                    self._ids[row].copy_(old_ids)
                    assert old_valid is not None
                    assert old_write_index is not None
                    assert old_count is not None
                    assert old_ring_hash_slots is not None
                    assert old_hash_keys is not None
                    assert old_hash_values is not None
                    self._valid[row].copy_(old_valid)
                    self._write_index[row].copy_(old_write_index)
                    self._count[row].copy_(old_count)
                    self._ring_hash_slots[row].copy_(old_ring_hash_slots)
                    self._hash_keys[row].copy_(old_hash_keys)
                    self._hash_values[row].copy_(old_hash_values)
                self._cache_rows[cache_key] = row
                self._caches.append(cache)
                self._bind_views(cache, row)
            rows.append(row)
        return rows

    def record_instances(
        self,
        caches: List[InstanceCache],
        tracker_ids: List[torch.Tensor],
    ) -> InstanceCacheKernelResult:
        """Classify a ragged cohort with stable device new/seen positions."""
        if len(caches) != len(tracker_ids):
            raise ValueError("cache and tracker ID batches must be aligned")
        rows = self._ensure_caches(caches)
        batch_size = len(caches)
        counts = [int(ids.numel()) for ids in tracker_ids]
        max_count = max(counts, default=0)
        if max_count == 0:
            empty_bool = torch.empty(0, dtype=torch.bool, device=self._device)
            empty_indices = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=self._device,
            )
            empty_counts = torch.zeros(
                (batch_size, 2),
                dtype=torch.int32,
                device=self._device,
            )
            return InstanceCacheKernelResult(
                seen=empty_bool,
                new_indices=empty_indices,
                seen_indices=empty_indices,
                partition_counts=empty_counts,
            )
        normalized_ids = [
            ids.to(device=self._device, dtype=torch.long).reshape(-1)
            for ids in tracker_ids
        ]
        flat_ids = torch.cat(normalized_ids)
        offsets = []
        next_offset = 0
        for count in counts:
            offsets.append(next_offset)
            next_offset += count
        stream_metadata = torch.tensor(
            list(zip(offsets, counts)),
            dtype=torch.long,
            device=self._device,
        )
        contiguous_start = rows[0] if rows else 0
        contiguous = rows == list(range(contiguous_start, contiguous_start + len(rows)))
        if contiguous:
            row_tensor = self._row_indices[
                contiguous_start : contiguous_start + batch_size
            ]
        else:
            row_tensor = torch.tensor(rows, dtype=torch.long, device=self._device)
        fused = run_triton_instance_cache(
            flat_ids,
            stream_metadata,
            row_tensor,
            self._ids,
            self._valid,
            self._ring_hash_slots,
            self._write_index,
            self._count,
            self._hash_keys,
            self._hash_values,
            cache_size=self._size,
            max_inputs=max_count,
        )
        if fused is not None:
            return fused
        seen_parts = [
            cache._record_instances_fallback(ids)
            for cache, ids in zip(caches, normalized_ids)
        ]
        seen = torch.cat(seen_parts)
        new_indices = torch.empty(
            (batch_size, max_count),
            dtype=torch.long,
            device=self._device,
        )
        seen_indices = torch.empty_like(new_indices)
        partition_counts = torch.empty(
            (batch_size, 2),
            dtype=torch.int32,
            device=self._device,
        )
        for stream, stream_seen in enumerate(seen_parts):
            stream_new_indices = torch.nonzero(
                ~stream_seen,
                as_tuple=False,
            ).reshape(-1)
            stream_seen_indices = torch.nonzero(
                stream_seen,
                as_tuple=False,
            ).reshape(-1)
            new_indices[stream, : len(stream_new_indices)].copy_(stream_new_indices)
            seen_indices[stream, : len(stream_seen_indices)].copy_(stream_seen_indices)
            partition_counts[stream, 0] = len(stream_new_indices)
            partition_counts[stream, 1] = len(stream_seen_indices)
        return InstanceCacheKernelResult(
            seen=seen,
            new_indices=new_indices,
            seen_indices=seen_indices,
            partition_counts=partition_counts,
        )


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
        self._instance_cache_batch_arenas: Dict[
            Tuple[int, torch.device], _InstanceCacheBatchArena
        ] = {}

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

        recovered = self._recover_tracker_outputs_batch(
            detections=list(detections),
            bboxes=[item[2] for item in prepared],
            tracked_outputs=tracked_outputs,
        )
        return self._build_tracker_results_batch(
            video_ids=[item[0] for item in prepared],
            tracked_detections=[item[0] for item in recovered],
            tracker_ids=[item[1] for item in recovered],
            instances_cache_size=instances_cache_size,
        )

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
        has_rows = (
            tracked_sv.data
            and _TRACKER_ROW_INDEX_KEY in tracked_sv.data
            and tracked_sv.tracker_id is not None
            and len(tracked_sv) > 0
        )
        if has_rows:
            tracker_rows = torch.as_tensor(
                tracked_sv.data[_TRACKER_ROW_INDEX_KEY],
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
            all_tracker_ids = torch.as_tensor(
                tracked_sv.tracker_id,
                dtype=torch.long,
                device=bbox.xyxy.device,
            )
            confirmed = all_tracker_ids != -1
            surviving = tracker_rows[confirmed]
            tracker_ids = all_tracker_ids[confirmed]
        else:
            surviving = torch.empty(0, dtype=torch.long, device=bbox.xyxy.device)
            tracker_ids = torch.empty(0, dtype=torch.long, device=bbox.xyxy.device)
        tracked_detections = take_prediction_by_indices(detections, surviving)
        _bbox_component(tracked_detections).tracker_id = tracker_ids
        return tracked_detections, tracker_ids

    @staticmethod
    def _recover_tracker_outputs_batch(
        detections: List[TensorNativeTrackerPrediction],
        bboxes: List[Union[Detections, InstanceDetections]],
        tracked_outputs: List[sv.Detections],
    ) -> List[Tuple[TensorNativeTrackerPrediction, torch.Tensor]]:
        """Recover a SIMD cohort with one dynamic-cardinality boundary.

        Tracker rows and IDs remain device tensors. Rows with ``tracker_id ==
        -1`` are stably partitioned for the whole cohort, then one small vector
        of confirmed counts supplies the exact public prediction slice bounds.
        """
        if not (len(detections) == len(bboxes) == len(tracked_outputs)):
            raise ValueError("tracker recovery batches must be aligned")
        if not detections:
            return []
        device = bboxes[0].xyxy.device
        row_batches: List[torch.Tensor] = []
        id_batches: List[torch.Tensor] = []
        for bbox, tracked_sv in zip(bboxes, tracked_outputs):
            if bbox.xyxy.device != device:
                raise ValueError("tracker recovery batch must use one tensor device")
            has_rows = (
                tracked_sv.data
                and _TRACKER_ROW_INDEX_KEY in tracked_sv.data
                and tracked_sv.tracker_id is not None
                and len(tracked_sv) > 0
            )
            if has_rows:
                rows = torch.as_tensor(
                    tracked_sv.data[_TRACKER_ROW_INDEX_KEY],
                    dtype=torch.long,
                    device=device,
                ).reshape(-1)
                ids = torch.as_tensor(
                    tracked_sv.tracker_id,
                    dtype=torch.long,
                    device=device,
                ).reshape(-1)
                if rows.shape[0] != ids.shape[0]:
                    raise ValueError("tracker output rows and IDs must be aligned")
            else:
                rows = torch.empty(0, dtype=torch.long, device=device)
                ids = torch.empty(0, dtype=torch.long, device=device)
            row_batches.append(rows)
            id_batches.append(ids)

        counts = [int(ids.numel()) for ids in id_batches]
        max_count = max(counts, default=0)
        if max_count == 0:
            return [
                TrackerBlockBase._recover_tracker_output(
                    detections=prediction,
                    bbox=bbox,
                    tracked_sv=tracked_sv,
                )
                for prediction, bbox, tracked_sv in zip(
                    detections,
                    bboxes,
                    tracked_outputs,
                )
            ]
        count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
        batch_size = len(detections)
        batch_rows = torch.repeat_interleave(
            torch.arange(batch_size, device=device), count_tensor
        )
        offsets = torch.cumsum(count_tensor, dim=0) - count_tensor
        positions = torch.arange(sum(counts), device=device) - torch.repeat_interleave(
            offsets, count_tensor
        )
        padded_rows = torch.zeros(
            (batch_size, max_count), dtype=torch.long, device=device
        )
        padded_ids = torch.zeros_like(padded_rows)
        padded_rows[batch_rows, positions] = torch.cat(row_batches)
        padded_ids[batch_rows, positions] = torch.cat(id_batches)
        valid = torch.arange(max_count, device=device)[None, :] < count_tensor[:, None]
        confirmed = valid & padded_ids.ne(-1)
        partition_key = torch.where(
            valid,
            (~confirmed).to(dtype=torch.long),
            torch.full_like(confirmed, 2, dtype=torch.long),
        )
        stable_order = torch.argsort(partition_key, dim=1, stable=True)
        confirmed_counts = confirmed.sum(dim=1).detach().cpu().tolist()
        recovered: List[Tuple[TensorNativeTrackerPrediction, torch.Tensor]] = []
        for stream_index, prediction in enumerate(detections):
            confirmed_count = int(confirmed_counts[stream_index])
            selected = stable_order[stream_index, :confirmed_count]
            surviving = padded_rows[stream_index].index_select(0, selected)
            tracker_ids = padded_ids[stream_index].index_select(0, selected)
            tracked_detections = take_prediction_by_indices(prediction, surviving)
            _bbox_component(tracked_detections).tracker_id = tracker_ids
            recovered.append((tracked_detections, tracker_ids))
        return recovered

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

    def _build_tracker_results_batch(
        self,
        video_ids: List[str],
        tracked_detections: List[TensorNativeTrackerPrediction],
        tracker_ids: List[torch.Tensor],
        instances_cache_size: int,
    ) -> List[BlockResult]:
        """Classify and split one SIMD cohort with a single cache transaction.

        Exact public prediction lengths require Python slice bounds. We export
        one two-column partition-count tensor after the batched stable partition;
        numeric detections and tracker IDs remain on their original device.
        """
        if not (len(video_ids) == len(tracked_detections) == len(tracker_ids)):
            raise ValueError("tracker result batches must be aligned")
        if not tracker_ids:
            return []
        device = tracker_ids[0].device
        if any(ids.device != device for ids in tracker_ids):
            raise ValueError("tracker result batch must use one tensor device")
        caches: List[InstanceCache] = []
        for video_id in video_ids:
            if video_id not in self._per_video_cache:
                self._per_video_cache[video_id] = InstanceCache(
                    size=instances_cache_size
                )
            caches.append(self._per_video_cache[video_id])
        cache_sizes = {cache._size for cache in caches}
        unique_caches = len({id(cache) for cache in caches}) == len(caches)
        if len(cache_sizes) != 1 or not unique_caches:
            return [
                self._build_tracker_result(
                    video_id=video_id,
                    tracked_detections=prediction,
                    tracker_ids=ids,
                    instances_cache_size=instances_cache_size,
                )
                for video_id, prediction, ids in zip(
                    video_ids,
                    tracked_detections,
                    tracker_ids,
                )
            ]
        arena_key = (next(iter(cache_sizes)), device)
        arena = self._instance_cache_batch_arenas.get(arena_key)
        if arena is None:
            arena = _InstanceCacheBatchArena(
                size=arena_key[0],
                device=device,
            )
            self._instance_cache_batch_arenas[arena_key] = arena
        cache_result = arena.record_instances(
            caches=caches,
            tracker_ids=tracker_ids,
        )
        partition_counts = cache_result.partition_counts.detach().cpu().tolist()
        results: List[BlockResult] = []
        for stream_index, prediction in enumerate(tracked_detections):
            new_count, seen_count = partition_counts[stream_index]
            new_indices = cache_result.new_indices[
                stream_index,
                : int(new_count),
            ]
            seen_indices = cache_result.seen_indices[
                stream_index,
                : int(seen_count),
            ]
            results.append(
                {
                    OUTPUT_KEY: prediction,
                    "new_instances": take_prediction_by_indices(
                        prediction,
                        new_indices,
                    ),
                    "already_seen_instances": take_prediction_by_indices(
                        prediction,
                        seen_indices,
                    ),
                }
            )
        return results


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

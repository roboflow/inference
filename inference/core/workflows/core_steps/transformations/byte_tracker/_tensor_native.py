"""Tensor-native compatibility helpers for the deprecated ByteTrack blocks."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import supervision as sv
import torch

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

_INPUT_INDEX_KEY = "__byte_tracker_input_index__"

TensorDetections = Union[Detections, InstanceDetections]


@dataclass(frozen=True)
class MetadataPartitions:
    """Host metadata and positions for V3's device-resident cache partitions."""

    new_metadata: Optional[List[dict]]
    seen_metadata: Optional[List[dict]]
    new_positions: List[int]
    seen_positions: List[int]


def update_tensor_byte_tracker(
    tracker: sv.ByteTrack,
    detections: TensorDetections,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update SuperiorVision ByteTrack without moving numeric input off-device."""
    device = detections.xyxy.device
    row_count = int(detections.xyxy.shape[0])
    tracker_input = sv.Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        data={
            _INPUT_INDEX_KEY: torch.arange(
                row_count,
                dtype=torch.long,
                device=device,
            )
        },
    )
    tracked = tracker.update_with_detections(tracker_input)
    has_rows = False
    if tracked.data:
        has_rows = all(
            (
                _INPUT_INDEX_KEY in tracked.data,
                tracked.tracker_id is not None,
                len(tracked) > 0,
            )
        )
    if not has_rows:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    kept_indices = torch.as_tensor(
        tracked.data[_INPUT_INDEX_KEY],
        dtype=torch.long,
        device=device,
    ).reshape(-1)
    tracker_ids = torch.as_tensor(
        tracked.tracker_id,
        dtype=torch.long,
        device=device,
    ).reshape(-1)
    if kept_indices.shape != tracker_ids.shape:
        raise ValueError("ByteTrack output rows and tracker IDs must be aligned")
    return kept_indices, tracker_ids


def recover_tensor_byte_tracker_output(
    detections: TensorDetections,
    kept_indices: torch.Tensor,
    tracker_ids: torch.Tensor,
    seen: Optional[torch.Tensor] = None,
) -> Tuple[TensorDetections, Optional[MetadataPartitions]]:
    """Gather tracked rows and materialize legacy metadata with one host export."""
    if kept_indices.shape != tracker_ids.shape:
        raise ValueError("ByteTrack row indices and tracker IDs must be aligned")
    if seen is not None and seen.shape != tracker_ids.shape:
        raise ValueError("ByteTrack cache mask and tracker IDs must be aligned")

    export_columns = [kept_indices, tracker_ids]
    if seen is not None:
        export_columns.append(seen.to(dtype=torch.long))
    packed_rows = (
        torch.stack(export_columns, dim=1).detach().cpu().tolist()
        if tracker_ids.numel() > 0
        else []
    )

    source_metadata = detections.bboxes_metadata
    tracked_metadata: List[dict] = []
    new_metadata: List[dict] = []
    seen_metadata: List[dict] = []
    new_positions: List[int] = []
    seen_positions: List[int] = []
    host_source_rows: List[int] = []
    for output_position, packed_row in enumerate(packed_rows):
        source_row = int(packed_row[0])
        tracker_id = int(packed_row[1])
        host_source_rows.append(source_row)
        metadata = (
            dict(source_metadata[source_row] or {})
            if source_metadata is not None
            else {}
        )
        metadata["tracker_id"] = tracker_id
        tracked_metadata.append(metadata)
        if seen is None:
            continue
        if bool(packed_row[2]):
            seen_positions.append(output_position)
            seen_metadata.append(dict(metadata))
        else:
            new_positions.append(output_position)
            new_metadata.append(dict(metadata))

    selector = kept_indices.to(device=detections.xyxy.device, dtype=torch.long)
    identity_selection = host_source_rows == list(range(len(detections)))
    common_kwargs = {
        "xyxy": (
            detections.xyxy
            if identity_selection
            else detections.xyxy.index_select(0, selector)
        ),
        "class_id": (
            detections.class_id
            if identity_selection
            else detections.class_id.index_select(0, selector)
        ),
        "confidence": (
            detections.confidence
            if identity_selection
            else detections.confidence.index_select(0, selector)
        ),
        "image_metadata": detections.image_metadata,
        "bboxes_metadata": tracked_metadata or None,
        "tracker_id": tracker_ids,
    }
    if isinstance(detections, InstanceDetections):
        masks = detections.mask
        if identity_selection:
            selected_masks = masks
        elif isinstance(masks, InstancesRLEMasks):
            selected_masks = InstancesRLEMasks(
                image_size=masks.image_size,
                masks=[masks.masks[index] for index in host_source_rows],
            )
        else:
            selected_masks = masks.index_select(0, selector.to(masks.device))
        tracked_detections: TensorDetections = InstanceDetections(
            mask=selected_masks,
            **common_kwargs,
        )
    else:
        tracked_detections = Detections(**common_kwargs)

    if seen is None:
        return tracked_detections, None
    partitions = MetadataPartitions(
        new_metadata=new_metadata or None,
        seen_metadata=seen_metadata or None,
        new_positions=new_positions,
        seen_positions=seen_positions,
    )
    return tracked_detections, partitions


def select_tracked_partition(
    detections: TensorDetections,
    mask: torch.Tensor,
    host_positions: List[int],
    metadata: Optional[List[dict]],
) -> TensorDetections:
    """Select a V3 cache partition without exporting its device mask."""
    selector = mask.to(device=detections.xyxy.device, dtype=torch.bool)
    common_kwargs = {
        "xyxy": detections.xyxy[selector],
        "class_id": detections.class_id[selector],
        "confidence": detections.confidence[selector],
        "image_metadata": detections.image_metadata,
        "bboxes_metadata": metadata,
        "tracker_id": (
            detections.tracker_id[selector]
            if detections.tracker_id is not None
            else None
        ),
    }
    if isinstance(detections, InstanceDetections):
        masks = detections.mask
        if isinstance(masks, InstancesRLEMasks):
            selected_masks = InstancesRLEMasks(
                image_size=masks.image_size,
                masks=[masks.masks[index] for index in host_positions],
            )
        else:
            selected_masks = masks[selector.to(masks.device)]
        return InstanceDetections(mask=selected_masks, **common_kwargs)
    return Detections(**common_kwargs)

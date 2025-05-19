from typing import Literal, Union

from pydantic import ConfigDict, Field

import supervision as sv

from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    Selector,
)

from ..base import BaseReIDTrackerBlock, BaseReIDTrackerBlockManifest

SHORT_DESCRIPTION = "Track objects with DeepSORT and appearance embeddings."
LONG_DESCRIPTION = (
    "The `DeepSortTrackerBlockV1` augments SORT tracking with a re-identification "
    "model to compute appearance embeddings for improved object association."
)

__all__ = ["DeepSortTrackerBlockV1"]


class DeepSortTrackerBlockManifest(BaseReIDTrackerBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "DeepSORT Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "mdi-target",
                "blockPriority": 0,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/deep_sort_tracker@v1"]

    # DeepSORT defaults
    minimum_consecutive_frames: Union[int, Selector(kind=[INTEGER_KIND])] = 3
    minimum_iou_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.3


class DeepSortTrackerBlockV1(BaseReIDTrackerBlock):
    """DeepSORT tracker workflow block (ReID‑based)."""

    @classmethod
    def get_manifest(cls):
        return DeepSortTrackerBlockManifest

    # ---------------------------------------------------------------

    def _instantiate_tracker(
        self,
        *,
        video_id: str,
        frame_rate: float,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_consecutive_frames: int,
        minimum_iou_threshold: float,
        embedding_model: str,
        appearance_threshold: float,
        appearance_weight: float,
        distance_metric: str,
        device: str | None,
        **kwargs,
    ):
        from trackers.core.deepsort.tracker import DeepSORTTracker

        reid_model = self._get_reid_model(embedding_model, device)

        return DeepSORTTracker(
            reid_model=reid_model,
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate or 30.0,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            appearance_threshold=appearance_threshold,
            appearance_weight=appearance_weight,
            distance_metric=distance_metric,
        )

    # ---------------------------------------------------------------

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_consecutive_frames: int,
        minimum_iou_threshold: float,
        instances_cache_size: int,
        embedding_model: str,
        appearance_threshold: float,
        appearance_weight: float,
        distance_metric: str,
        device: str | None = None,
    ):
        # Call parent run to get basic tracking outputs
        outputs = super().run(
            image=image,
            detections=detections,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            instances_cache_size=instances_cache_size,
            embedding_model=embedding_model,
            appearance_threshold=appearance_threshold,
            appearance_weight=appearance_weight,
            distance_metric=distance_metric,
            device=device,
        )

        # Attach appearance embeddings to tracked detections ----------------
        reid_model = self._get_reid_model(embedding_model, device)
        tracked_detections: sv.Detections = outputs["tracked_detections"]
        embeddings = reid_model.extract_features(tracked_detections, image.numpy_image)
        tracked_detections.embedding = embeddings  # type: ignore[attr-defined]

        return outputs

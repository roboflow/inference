"""Action recognition with a held snapshot of the latest analysis window."""

import math
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type

from pydantic import ConfigDict
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    LONG_DESCRIPTION as V1_LONG_DESCRIPTION,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as V1BlockManifest,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    _ActionRecognitionBookkeeping as V1Bookkeeping,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import logger
from roboflow_workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    DICTIONARY_KIND,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest

LONG_DESCRIPTION = V1_LONG_DESCRIPTION + """
`latest_predictions` holds the actions of the latest model call until the next
call, as multi-label classification with a fixed confidence of 1.0. Connect it
to Classification Label Visualization to show them on the video.

For a temporally explicit overlay, connect `window` to Action Recognition
Visualization. It contains `status` (collecting, ready or error), all `classes`
from the latest call, inclusive `start_frame`/`end_frame` of the actual sampled
window, the declared source `fps` (null when assumed), and `video_identifier`.
The snapshot is held between calls; a failure clears its classes and bounds
and keeps error status until a successful call. Ready with no classes means
no action was reported under the class filter. The visualizer optionally uses
`timeline` for accumulated intervals clipped to this window. Times derived
from frames/FPS are estimates, not source PTS or inference latency.
"""


@dataclass
class _ActionRecognitionBookkeeping(V1Bookkeeping):
    latest_actions: List[str] = field(default_factory=list)
    analysis_status: str = "collecting"
    window_start_frame: Optional[int] = None
    window_end_frame: Optional[int] = None
    source_fps_is_fallback: bool = False


class BlockManifest(V1BlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            **V1BlockManifest.model_config["json_schema_extra"],
            "version": "v2",
            "long_description": LONG_DESCRIPTION,
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_action_recognition_model@v2"]

    # Both versions execute the same model with the same resource requirements.
    discover_dependent_resources = V1BlockManifest.discover_dependent_resources
    discover_work_operations = V1BlockManifest.discover_work_operations
    get_actual_restrictions = V1BlockManifest.get_actual_restrictions

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return super().describe_outputs() + [
            OutputDefinition(
                name="latest_predictions", kind=[CLASSIFICATION_PREDICTION_KIND]
            ),
            OutputDefinition(name="window", kind=[DICTIONARY_KIND]),
        ]


class ActionRecognitionModelBlockV2(ActionRecognitionModelBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @staticmethod
    def _create_bookkeeping(signature: Tuple[Tuple[str, ...], float, float]):
        return _ActionRecognitionBookkeeping(signature=signature)

    def _resolve_source_fps(
        self, metadata: VideoMetadata, bookkeeping: _ActionRecognitionBookkeeping
    ) -> float:
        fps = super()._resolve_source_fps(metadata=metadata, bookkeeping=bookkeeping)
        declared_fps = metadata.fps
        bookkeeping.source_fps_is_fallback = not (
            declared_fps is not None
            and declared_fps > 0
            and math.isfinite(declared_fps)
        )
        return fps

    def _classify_buffer(
        self,
        model,
        bookkeeping: _ActionRecognitionBookkeeping,
        block_filter: Optional[List[str]],
        id_vocabulary: Optional[List[str]],
        effective_sample_fps: float,
        sampling_stride: float,
    ) -> str:
        if not bookkeeping.sampled:
            return ""
        frames = self._prepare_frames_for_model(
            [frame for _, frame in bookkeeping.sampled]
        )
        try:
            segments = model.infer(
                frames=frames,
                class_names=block_filter,
                fps=effective_sample_fps,
            )
        except Exception as error:
            logger.warning(
                "Action Recognition Model call failed: %s",
                error,
                exc_info=True,
            )
            # Stale labels would look like a fresh result on screen.
            bookkeeping.latest_actions = []
            bookkeeping.analysis_status = "error"
            bookkeeping.window_start_frame = None
            bookkeeping.window_end_frame = None
            return str(error)
        # Separates "the model output one range" from "the block merged
        # several".
        logger.debug(
            "Action Recognition model call over sampled frames "
            "[%s, %s] returned %d pre-merge segment(s): %s",
            bookkeeping.sampled[0][0],
            bookkeeping.sampled[-1][0],
            len(segments),
            [
                (segment.start_frame_idx, segment.end_frame_idx, segment.class_name)
                for segment in segments
            ],
        )
        # Same-class ranges merge only when no sampled frame lies in the
        # gap. ceil(stride) is that rule exactly: consecutive samples sit
        # floor/ceil(stride) apart; the next sampled gap is ~2x the stride.
        self._merge_segments(
            bookkeeping=bookkeeping,
            segments=segments,
            block_filter=block_filter,
            id_vocabulary=id_vocabulary,
            stride=max(1, math.ceil(sampling_stride)),
        )
        bookkeeping.analysis_status = "ready"
        bookkeeping.window_start_frame = bookkeeping.sampled[0][0]
        bookkeeping.window_end_frame = bookkeeping.sampled[-1][0]
        # Sorted, so an action keeps its label slot while it persists across calls.
        bookkeeping.latest_actions = sorted(
            {
                segment.class_name
                for segment in segments
                if block_filter is None or segment.class_name in block_filter
            }
        )
        return ""

    def _build_output(
        self,
        image: WorkflowImageData,
        bookkeeping: _ActionRecognitionBookkeeping,
        error_status: str,
    ) -> dict:
        return {
            **super()._build_output(
                image=image, bookkeeping=bookkeeping, error_status=error_status
            ),
            "latest_predictions": self._build_latest_predictions(
                image=image, actions=bookkeeping.latest_actions
            ),
            "window": {
                "status": bookkeeping.analysis_status,
                "classes": list(bookkeeping.latest_actions),
                "start_frame": bookkeeping.window_start_frame,
                "end_frame": bookkeeping.window_end_frame,
                "fps": (
                    None
                    if bookkeeping.source_fps_is_fallback
                    else bookkeeping.source_fps
                ),
                "video_identifier": image.video_metadata.video_identifier,
            },
        }

    def _build_latest_predictions(
        self, image: WorkflowImageData, actions: List[str]
    ) -> Any:
        # Multi-label so every action gets a label. Position ids stay dense, which
        # the tensor confidence vector needs; captions also have no vocabulary id.
        height, width = image._read_shape_without_materialization()
        return {
            "image": {"height": height, "width": width},
            "predictions": {
                action: {"confidence": 1.0, "class_id": class_id}
                for class_id, action in enumerate(actions)
            },
            "predicted_classes": list(actions),
            "prediction_type": "classification",
            "parent_id": image.parent_metadata.parent_id,
            "root_parent_id": image.workflow_root_ancestor_metadata.parent_id,
        }

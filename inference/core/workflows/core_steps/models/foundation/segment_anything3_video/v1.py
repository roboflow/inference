"""SAM3 Video Tracker workflow block.

Wraps ``inference_models``'s ``SAM3Video`` — the HuggingFace streaming
port of SAM3's open-vocabulary *concept* tracker — so it can be driven
from a workflow powered by ``InferencePipeline``.  The pipeline
delivers one frame at a time with ``WorkflowImageData.video_metadata``;
this block keeps one state_dict per ``video_identifier``.

Unlike the SAM2 video block (which needs an upstream detector to seed
box prompts, and a ``prompt_mode`` policy to decide when to re-seed),
SAM3 concept prompts are registered on the session once and the model
runs fused detect-and-track on every frame: objects entering the scene
that match a concept are picked up automatically with fresh tracker
ids.  There is therefore no re-prompt scheduling — the session is only
re-seeded when the stream restarts or ``class_names`` changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
)
from inference.core.workflows.core_steps.models.foundation._streaming_video_common import (
    concept_frame_to_sv_detections,
    normalise_class_names,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Segment and track objects across video frames from text prompts with "
    "SAM3's streaming concept tracker."
)

_EMPTY_MASKS = np.zeros((0, 0, 0), dtype=bool)
_EMPTY_IDS = np.zeros((0,), dtype=np.int64)
_EMPTY_SCORES = np.zeros((0,), dtype=np.float32)
_EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)

LONG_DESCRIPTION = """
Run Segment Anything 3 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Provide the concepts to track as text in `class_names` (e.g.
`["person", "forklift"]`) — no upstream detector is needed.  SAM3 runs
fused detection and tracking on every frame, so objects matching a
concept that enter the scene mid-stream are picked up automatically and
assigned fresh `tracker_id`s.  Each emitted mask carries the prompt it
matched as its class name and the model's detection score as its
confidence.

The block multiplexes a single SAM3 streaming model across many video
streams by keying state on `video_metadata.video_identifier`; a session
is re-seeded only when the source stream restarts or `class_names`
changes.  For detector-driven (box-prompted) video tracking, use the
SAM2 Video Tracker block instead.

Intended for use with `InferencePipeline`, which delivers one frame at
a time and tags each frame with video metadata.
"""


@dataclass
class _ConceptSessionBookkeeping:
    """Per-video session state for the concept tracker.

    ``prompt_signature`` records the exact concept set the live session
    was seeded with, so a change in ``class_names`` (e.g. driven by a
    workflow parameter) re-seeds instead of silently tracking stale
    concepts.
    """

    state_dict: Optional[dict] = None
    last_frame_number: int = -1
    prompt_signature: Tuple[str, ...] = field(default_factory=tuple)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM3 Video Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "SAM3",
                "segment anything 3",
                "video",
                "tracking",
                "open vocabulary",
                "META",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "fa-brands fa-meta",
                "blockPriority": 9.39,
                "needsGPU": True,
                "inference": True,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3_video@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_names: Union[
        List[str], str, Selector(kind=[LIST_OF_VALUES_KIND, STRING_KIND])
    ] = Field(
        description=(
            "Concepts to segment and track, as a list of phrases (or a "
            "single comma-separated string).  Each emitted mask carries "
            "the concept it matched as its class name."
        ),
        examples=[["person", "forklift"]],
        json_schema_extra={"always_visible": True},
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="sam3video",
        description="Streaming SAM3 model id resolved by `inference_models`.",
        examples=["sam3video"],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.5,
        description=(
            "Minimum detection score for emitted masks.  Scores come "
            "from SAM3's per-object concept detection head."
        ),
        examples=[0.5],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; the streaming SAM3 video model needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return ["sam3video"]


class SegmentAnything3VideoBlockV1(WorkflowBlock):
    """Stateful SAM3 streaming concept tracking block."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "SAM3 Video Tracker only supports LOCAL workflow step "
        "execution.  Remote execution would ship each frame to a "
        "separate process and break the per-video SAM3 session "
        "that holds the temporal memory.  Set "
        "WORKFLOWS_STEP_EXECUTION_MODE=local (or run on a "
        "dedicated deployment) to use this block."
    )

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model = None  # lazily loaded
        self._current_model_id: Optional[str] = None
        self._sessions: Dict[str, _ConceptSessionBookkeeping] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            from inference_models import AutoModel

            self._model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
            )
            self._current_model_id = model_id
            # Switching model invalidates every session we held.
            self._sessions.clear()
        return self._model

    def run(
        self,
        images: Batch[WorkflowImageData],
        class_names: Union[List[str], str],
        model_id: str,
        threshold: float,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        model = self._get_model(model_id=model_id)
        class_list = normalise_class_names(class_names)
        prompt_signature = tuple(class_list)

        batch_detections: List[sv.Detections] = []
        for single_image in images:
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            session = self._sessions.setdefault(video_id, _ConceptSessionBookkeeping())
            stream_restarted = (
                session.last_frame_number >= 0
                and frame_number < session.last_frame_number
            )
            if stream_restarted or session.prompt_signature != prompt_signature:
                session.state_dict = None

            frame_np = single_image.numpy_image

            if not class_list:
                detections = concept_frame_to_sv_detections(
                    masks=_EMPTY_MASKS,
                    object_ids=_EMPTY_IDS,
                    scores=_EMPTY_SCORES,
                    boxes=_EMPTY_BOXES,
                    prompt_to_object_ids={},
                    class_names=class_list,
                    image=single_image,
                    threshold=threshold,
                )
            else:
                if session.state_dict is None:
                    result = model.prompt(
                        image=frame_np,
                        text=class_list,
                        clear_old_prompts=True,
                    )
                    session.prompt_signature = prompt_signature
                else:
                    result = model.track(image=frame_np, state_dict=session.state_dict)
                session.state_dict = result.state_dict
                detections = concept_frame_to_sv_detections(
                    masks=result.masks,
                    object_ids=result.object_ids,
                    scores=result.scores,
                    boxes=result.boxes,
                    prompt_to_object_ids=result.prompt_to_object_ids,
                    class_names=class_list,
                    image=single_image,
                    threshold=threshold,
                )

            session.last_frame_number = frame_number
            batch_detections.append(detections)

        batch_detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=batch_detections,
            prediction_type="instance-segmentation",
        )
        batch_detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=batch_detections,
        )
        return [{"predictions": pred} for pred in batch_detections]

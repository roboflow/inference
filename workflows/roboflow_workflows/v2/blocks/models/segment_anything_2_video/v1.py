"""Track box or point prompted objects locally with SAM2 video.

Canonical type: ``roboflow_core/segment_anything_2_video@v1``. Source
sessions, episodes and their limits are ``VideoTracker``'s
(``_shared/sam_video.py``).
"""

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from roboflow_workflows.v2.blocks.models._shared.sam_video import (
    VideoTracker,
    _is_selector,
    _positive_int,
    _session_frame_limit,
)
from roboflow_workflows.v2.blocks.models._shared.segment_anything import PREDICTION_TYPE
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    models_cache,
)
from roboflow_workflows.v2.blocks.models.sam3_interactive.v1 import _labeled_points
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


SAM2_VIDEO_MODEL_ID = "sam2video/small"


class SegmentAnything2VideoBlock(Block):
    """Track box or point prompted objects locally with SAM2 video.

    A source without prompts emits no rows until it can be seeded. Re-prompts
    start a new episode, which restarts backend tracker IDs; every frame always
    gets fresh row identity.
    """

    type = "roboflow_core/segment_anything_2_video@v1"
    metadata = {
        "name": "SAM2 Video Tracker",
        "port_of": "roboflow_core/segment_anything_2_video@v1",
        "section": "video",
    }
    outputs = {
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="image",
            description=(
                "Serializable InstanceDetections with a dense (N, H, W) bool mask in "
                "this image's pixel coordinates: segmentation's full-resolution "
                "result, same rows. Use it as a workflow output; wire segmentation "
                "to blocks."
            ),
        ),
        "segmentation": Output(
            INSTANCE_SEGMENTATION_VIEW_KIND,
            source="image",
            description=(
                "SegmentationView of the masks: wire it to blocks. Not "
                "serializable; use predictions as a workflow output."
            ),
        ),
        "model_id": Output(
            STRING_KIND,
            source="image",
            description="Model id that served this image.",
        ),
        "inference_id": Output(
            STRING_KIND,
            source="image",
            description=(
                "Fresh id of this image's result; for results with rows, also the "
                "namespace of their row identity."
            ),
        ),
    }
    # Source sessions persist across calls: one whole invocation at a time.
    phase_overlap = False
    # Prompt lineage may contain caller-owned objects retained for future frames.
    retains_inputs = ("boxes",)
    retains_outputs = ("predictions", "segmentation")

    class Params(BlockParams):
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description=(
                "RGB video frames; members of each source are processed in order."
            )
        )
        boxes: Optional[
            Ref(
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_VIEW_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                batch="always",  # noqa: F821
            )
        ] = Field(
            default=None,
            description=(
                "Visual prompt rows predicted on these frames; read only when "
                "prompting."
            ),
        )
        points: Optional[List[Any]] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Visual points for one extra object: {x, y, positive} or "
                "[x, y, positive]; positive defaults to true. Absolute finite "
                "coordinates inside the image."
            ),
            examples=[[{"x": 100, "y": 120, "positive": True}]],
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM2_VIDEO_MODEL_ID,
            description="Local streaming model id or cached package path.",
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Exact model package to load; None lets the loader choose. List "
                    "a model's package ids with "
                    'inference_models.AutoModel.describe_model("<model_id>").'
                ),
            )
        )
        prompt_mode: Literal["first_frame", "every_n_frames", "every_frame"] = Field(
            default="first_frame",
            description=(
                "Visual mode: seed once, re-seed after N tracked frames, or re-seed "
                "each frame."
            ),
        )
        prompt_interval: StrictInt | Ref(INTEGER_KIND) = Field(
            default=30,
            description=(
                "Positive number of tracking calls between scheduled visual prompts."
            ),
        )
        confidence: Annotated[StrictFloat, Field(ge=0, le=1)] | Ref(FLOAT_KIND) = Field(
            default=0.0,
            description=(
                "Minimum emitted score: visual prompt confidence or concept "
                "detection score."
            ),
        )
        max_sources: StrictInt = Field(
            default=256,
            ge=1,
            description=(
                "Maximum retained sources, stopped/expired ones included; exceeding "
                "it fails without evicting another stream."
            ),
        )
        max_session_frames: StrictInt | Literal["unbounded"] = Field(
            description=(
                "Required positive integer: successful backend frames per session, "
                "or 'unbounded'. New sessions start a new count. No safe numerical "
                "default is established."
            ),
            examples=[300, "unbounded"],
        )
        on_session_limit: Literal["error", "restart"] = Field(
            default="error",
            description=(
                "At max_session_frames: 'error' stops the source until reset_source; "
                "'restart' starts a new tracking episode once the source can seed."
            ),
        )
        reset_source: StrictBool | Ref(BOOLEAN_KIND, batch="if_varying") = Field(
            default=False,
            description=(
                "True drops this frame's source state before processing it, also "
                "recovering a stopped or expired source. Every true frame resets again."
            ),
        )
        max_idle_pulses: Optional[StrictInt] = Field(
            default=None,
            ge=1,
            description=(
                "Allow this many intervening block invocations without the source; "
                "release its handle on the next absent pulse and retain an expired "
                "entry until reset or cleanup. Null disables idle release."
            ),
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "SegmentAnything2VideoBlock.Params":
            if not _is_selector(self.prompt_interval):
                _positive_int(self.prompt_interval, name="prompt_interval")
            if not _is_selector(self.points):
                _labeled_points(self.points, block="SAM video")
            _session_frame_limit(self.max_session_frames)

            return self

    def __init__(self, models: Mapping[Any, Any] | ModelCache):
        models = models_cache(models, block=self.type)
        self._tracker = VideoTracker(models, block=self.type)

    @classmethod
    def discover_dependent_resources(cls, params: Params) -> List[DependentResource]:
        """Declare the selected model in the models resource.

        Args:
            params: Configured block parameters.

        Returns:
            The local streaming model dependency.
        """
        dependencies = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": PREDICTION_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return dependencies

    def run(
        self,
        image: Batch[ImageData],
        boxes: Optional[Batch] = None,
        points: Optional[List[Any]] = None,
        model_id: str = SAM2_VIDEO_MODEL_ID,
        model_package_id: Optional[str] = None,
        prompt_mode: Literal[
            "first_frame", "every_n_frames", "every_frame"
        ] = "first_frame",
        prompt_interval: int = 30,
        confidence: float = 0.0,
        max_sources: int = 256,
        *,
        max_session_frames: int | str,
        on_session_limit: Literal["error", "restart"] = "error",
        reset_source: bool | Batch = False,
        max_idle_pulses: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Process sequential frames using independent source handles.

        Args:
            image: RGB frames with video metadata or source sample context.
            boxes: Aligned box prompts on frames which re-prompt.
            points: Labeled points for one additional object.
            model_id: Model id or existing local model package path.
            model_package_id: Exact optional package id.
            prompt_mode: Visual re-prompt policy.
            prompt_interval: Tracking calls between scheduled prompts.
            confidence: Minimum inherited visual prompt confidence.
            max_sources: Maximum sources to retain, stopped/expired ones included.
            max_session_frames: Positive backend-frame budget per session,
                or "unbounded".
            on_session_limit: "error" stops the source; "restart" starts a
                new episode.
            reset_source: One flag, or flags aligned with image; true drops
                that frame's source state first.
            max_idle_pulses: Invocations a source may miss before its handle is
                released; the expired entry remains until reset/cleanup. None
                disables expiry.

        Returns:
            Demanded dense/view predictions and model/inference identifiers.
            Prediction image_metadata["video_episode"] names the episode.

        Raises:
            ValueError: For invalid inputs or malformed backend results.
            RuntimeError: When an idle-expired source requires explicit reset/cleanup.
            VideoSourceCapacityError: When new sources exceed the bound.
            VideoSessionLimitError: When a source reaches its frame budget
                under "error", or sends frames while stopped.
        """
        results = self._tracker.track(
            image,
            reader=self,
            boxes=boxes,
            points=points,
            model_id=model_id,
            model_package_id=model_package_id,
            confidence=confidence,
            prompt_mode=prompt_mode,
            prompt_interval=prompt_interval,
            max_sources=max_sources,
            max_session_frames=max_session_frames,
            on_session_limit=on_session_limit,
            reset_source=reset_source,
            max_idle_pulses=max_idle_pulses,
            mode="visual",
        )

        return results

    def reset_state(self) -> None:
        """Drop all source handles for direct callers.

        Engine processing resets instead create a fresh block instance. Cached
        model weights remain owned by the host's models resource.
        """
        self._tracker.reset()

    def release_source(self, source_id: str) -> bool:
        """Drop one source's handle and state, stopped/expired entries included.

        Frees its max_sources slot at once. Other sources and cached model
        weights are untouched. Call it between invocations, like run().

        Args:
            source_id: Identity used for the source's frames.

        Returns:
            Whether the source was retained.
        """
        released = self._tracker.release_source(source_id)

        return released

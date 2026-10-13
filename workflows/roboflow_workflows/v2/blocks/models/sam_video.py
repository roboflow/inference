"""Local SAM2 visual and SAM3 visual/concept video tracking.

ModelCache owns reusable model weights. Each block instance owns separate opaque
backend handles per source. Handles are never workflow state or output. The
engine creates a fresh instance on a processing reset or a new session and
serializes this ordinary run stage in pulse order; batch members run in order.
Callers outside the engine must serialize calls themselves. Backend locks also
protect models shared across steps/sessions, but do not order source frames.

Visual re-prompts start new backend sessions and restart tracker IDs. V1's
every_n_frames counter is retained: after prompting, N tracking calls occur
before the next call re-prompts. A due frame without prompts keeps tracking.
An unseeded source returns empty results until prompts arrive. Concept mode
prompts once per concept set and tracks new matching objects continuously.
A fresh backend session numbers its first frame 0, so prompts attach there;
source frame numbers only order frames and detect rollback.

Image and frame-number checks run before a source's frame is processed. Any
later error on that frame discards the source, including an invalid prompt on
a due frame: opaque sessions can mutate before throwing and cannot be rolled
back. Previously processed batch members remain advanced. Discarding drops this
block's references only; memory is freed when nothing else references it.
Source capacity is bounded, backend history is not; no model unloading,
backend cleanup or long-session memory bound is promised.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from pydantic import Field, StrictInt, StrictStr, model_validator
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    UnitFloat,
    model_dependency,
    models_cache,
)
from roboflow_workflows.v2.blocks.models.sam_interactive import _labeled_points
from roboflow_workflows.v2.blocks.models.segment_anything import (
    FOREGROUND,
    PACKAGE_DESCRIPTION,
    PREDICTION_TYPE,
    SEGMENTATION_OUTPUTS,
    _mask_boxes,
    _metadata,
    _prompt_boxes,
    _require_rgb,
    _require_same_frame,
    _segmentation_result,
    _unit_float,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.context import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    DependentResource,
    Ref,
)

SAM2_VIDEO_MODEL_ID = "sam2video/small"
SAM3_CONCEPT_MODEL_ID = "sam3video"
SAM3_VISUAL_MODEL_ID = "sam3trackervideo"
PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]


class VideoSourceCapacityError(RuntimeError):
    """A new source exceeds max_sources; reset processing or raise the bound."""


class _VideoParams(BlockParams):
    image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description="RGB video frames; members of each source are processed in order."
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
        description="Visual prompt rows predicted on these frames; read only when prompting.",
    )
    points: Optional[List[Any]] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = Field(
        default=None,
        description=(
            "Visual points for one extra object: {x, y, positive} or [x, y, positive]; "
            "positive defaults to true. Absolute finite coordinates inside the image."
        ),
        examples=[[{"x": 100, "y": 120, "positive": True}]],
    )
    model_id: StrictStr | Ref(STRING_KIND) = Field(
        default=SAM2_VIDEO_MODEL_ID,
        description="Local streaming model id or cached package path.",
    )
    model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
        default=None, description=PACKAGE_DESCRIPTION
    )
    prompt_mode: PromptMode = Field(
        default="first_frame",
        description="Visual mode: seed once, re-seed after N tracked frames, or re-seed each frame.",
    )
    prompt_interval: StrictInt | Ref(INTEGER_KIND) = Field(
        default=30,
        description="Positive number of tracking calls between scheduled visual prompts.",
    )
    confidence: UnitFloat | Ref(FLOAT_KIND) = Field(
        default=0.0,
        description="Minimum emitted score: visual prompt confidence or concept detection score.",
    )
    max_sources: StrictInt = Field(
        default=256,
        ge=1,
        description="Maximum retained sources; exceeding it fails without evicting another stream.",
    )

    @model_validator(mode="after")
    def _check_literals(self):
        if not _is_selector(self.prompt_interval):
            _positive_int(self.prompt_interval, name="prompt_interval")
        if not _is_selector(self.points):
            _labeled_points(self.points, block="SAM video")

        return self


class _PromptObject(NamedTuple):
    class_id: int
    confidence: float
    row: dict


@dataclass
class _SourceSession:
    configuration: tuple
    size_hw: Tuple[int, int]
    handle: Any = None
    last_frame: Optional[int] = None
    frames_since_prompt: int = 0
    visual_metadata: Dict[int, _PromptObject] = field(default_factory=dict)
    class_names: dict = field(default_factory=dict)


class _VideoBlock(Block):
    outputs = SEGMENTATION_OUTPUTS
    phase_overlap = False
    # Prompt lineage may contain caller-owned objects retained for future frames.
    retains_inputs = ("boxes",)
    retains_outputs = ("predictions", "segmentation")

    def __init__(self, models: Mapping[Any, Any] | ModelCache):
        self._models = models_cache(models, block=self.type)
        self._sources: Dict[str, _SourceSession] = {}

    def reset_state(self) -> None:
        """Drop all source handles for direct callers.

        Engine processing resets instead create a fresh block instance. Cached
        model weights remain owned by the host's models resource.
        """
        self._sources.clear()

    def _run_video(
        self,
        image: Batch[ImageData],
        *,
        boxes: Optional[Batch],
        points: Optional[List[Any]],
        model_id: str,
        model_package_id: Optional[str],
        confidence: float,
        prompt_mode: str,
        prompt_interval: int,
        max_sources: int,
        mode: str,
        class_names: Any = None,
    ) -> List[Dict[str, Any]]:
        confidence = _unit_float(confidence, name="confidence", block=self.type)
        _positive_int(prompt_interval, name="prompt_interval")
        _positive_int(max_sources, name="max_sources")
        if prompt_mode not in ("first_frame", "every_n_frames", "every_frame"):
            raise ValueError(f"{self.type}: unsupported prompt_mode {prompt_mode!r}")
        if mode not in ("visual", "concept"):
            raise ValueError(f"{self.type}: tracking_mode must be concept or visual")
        if boxes is not None and (
            len(boxes) != len(image) or boxes.indices != image.indices
        ):
            raise ValueError(
                f"{self.type}: boxes and image batches must align in length and indices"
            )

        names = _concept_names(class_names) if mode == "concept" else []
        point_list = (
            _labeled_points(points, block=self.type) if mode == "visual" else []
        )
        request = ModelRequest(model_id, model_package_id)
        configuration = (request, mode, tuple(names), prompt_mode, prompt_interval)
        results = []
        for position, member in enumerate(image):
            _require_rgb(member, block=self.type)
            key, frame = self._source_identity(member, index=image.indices[position])
            session = self._sources.get(key)
            if session is None and len(self._sources) >= max_sources:
                raise VideoSourceCapacityError(
                    f"{self.type}: max_sources={max_sources} exceeded; reset processing "
                    "to drop ended sources, or increase max_sources"
                )
            restarted = (
                session is not None
                and frame is not None
                and session.last_frame is not None
                and frame < session.last_frame
            )
            if (
                session is None
                or session.configuration != configuration
                or session.size_hw != member.size_hw
                or restarted
            ):
                session = _SourceSession(
                    configuration=configuration, size_hw=member.size_hw
                )
                self._sources[key] = session

            # Inputs read only for visual prompt policy; tracking ignores current boxes/points.
            due = (
                session.handle is None
                or prompt_mode == "every_frame"
                or (
                    prompt_mode == "every_n_frames"
                    and session.frames_since_prompt >= prompt_interval
                )
            )
            try:
                if mode == "concept":
                    predictions = self._concept_frame(
                        session,
                        member,
                        request=request,
                        names=names,
                        confidence=confidence,
                    )
                else:
                    prompt = (
                        _prompt_boxes(boxes[position], block=self.type)
                        if due and boxes is not None
                        else None
                    )
                    predictions = self._visual_frame(
                        session,
                        member,
                        request=request,
                        prompt=prompt,
                        points=point_list if due else [],
                        due=due,
                        confidence=confidence,
                    )
                session.last_frame = frame
                inference_id = str(uuid.uuid4())
                predictions = with_row_identity(predictions, namespace=inference_id)
                result = _segmentation_result(
                    predictions,
                    reader=self,
                    model_id=model_id,
                    inference_id=inference_id,
                )
                results.append(result)
            except Exception:
                # Includes invalid prompts and malformed results after a backend call.
                self._sources.pop(key, None)
                raise

        return results

    def _source_identity(self, image: ImageData, *, index: tuple):
        metadata = image.video_metadata
        frame = None if metadata is None else metadata.frame_number
        if frame is not None and (type(frame) is not int or frame < 0):
            raise ValueError(
                f"{self.type}: frame_number must be a non-negative integer or None"
            )
        if metadata is not None and metadata.video_identifier:
            return metadata.video_identifier, frame

        try:
            context = self.execution_context
        except NoExecutionContextError:
            return "default", frame

        sample = context.sample_at(index, field="image")
        key = sample.source_id if sample is not None else "default"

        return key, frame

    def _visual_frame(self, session, image, *, request, prompt, points, due, confidence):
        has_prompts = prompt is not None and len(prompt) > 0 or bool(points)
        if due and has_prompts:
            _require_same_frame(prompt, image, block=self.type)
            for x, y, _ in points:
                if not (0 <= x < image.size_hw[1] and 0 <= y < image.size_hw[0]):
                    raise ValueError(
                        f"{self.type}: point {(x, y)} is outside image {image.size_hw}"
                    )
            bboxes, metadata, names = _visual_prompts(
                prompt, points=points, image=image
            )
            session.handle = None
            session.visual_metadata.clear()
            session.class_names = names
            model = self._models.get(request)
            # The fresh HF session numbers its first streamed frame 0, not the source frame.
            result = model.prompt(
                image=_backend_image(image),
                bboxes=bboxes,
                points=points or None,
                state_dict=None,
                clear_old_prompts=True,
                frame_idx=0,
            )
            masks, ids, handle = _visual_result(result, image=image)
            # The pinned HF visual prompt assigns obj_ids=range(prompt_count).
            # Seed frames can return no visible masks, or only a subset of ids.
            session.visual_metadata = dict(enumerate(metadata))
            if any(int(object_id) not in session.visual_metadata for object_id in ids):
                raise ValueError(
                    f"{self.type}: visual prompt returned an unprompted object id"
                )
            session.handle = handle
            session.frames_since_prompt = 0
        elif session.handle is not None:
            model = self._models.get(request)
            result = model.track(image=_backend_image(image), state_dict=session.handle)
            masks, ids, session.handle = _visual_result(result, image=image)
            if any(int(object_id) not in session.visual_metadata for object_id in ids):
                raise ValueError(
                    f"{self.type}: visual tracker returned an unprompted object id"
                )
            session.frames_since_prompt += 1
        else:
            masks = torch.zeros((0, *image.size_hw), dtype=torch.bool)
            ids = torch.zeros(0, dtype=torch.int64)

        objects = [session.visual_metadata[int(i)] for i in ids]
        scores = torch.tensor([o.confidence for o in objects], dtype=torch.float32)
        classes = torch.tensor([o.class_id for o in objects], dtype=torch.int64)
        rows = [{**o.row, "tracker_id": int(i)} for o, i in zip(objects, ids)]
        predictions = _frame_predictions(
            masks,
            ids=ids,
            scores=scores,
            classes=classes,
            rows=rows,
            names=session.class_names,
            image=image,
            confidence=confidence,
        )

        return predictions

    def _concept_frame(self, session, image, *, request, names, confidence):
        if not names:
            masks = torch.zeros((0, *image.size_hw), dtype=torch.bool)
            ids = torch.zeros(0, dtype=torch.int64)
            scores = torch.zeros(0)
            boxes = torch.zeros((0, 4))
            classes = torch.zeros(0, dtype=torch.int64)
        else:
            model = self._models.get(request)
            if session.handle is None:
                result = model.prompt(
                    image=_backend_image(image),
                    text=names,
                    state_dict=None,
                    clear_old_prompts=True,
                )
            else:
                result = model.track(
                    image=_backend_image(image), state_dict=session.handle
                )
            masks, ids, scores, boxes, mapping, session.handle = _concept_result(
                result, image=image, names=names
            )
            by_id = {
                object_id: names.index(name)
                for name, object_ids in mapping.items()
                for object_id in object_ids
            }
            # V1 defaults an unmapped object to class 0; an explicit fallback class
            # keeps the V2 class-name table truthful instead of naming it as prompt 0.
            classes = torch.tensor(
                [by_id.get(int(i), len(names)) for i in ids], dtype=torch.int64
            )

        predictions = _frame_predictions(
            masks,
            ids=ids,
            scores=scores,
            classes=classes,
            rows=[{"tracker_id": int(i)} for i in ids],
            names={**dict(enumerate(names)), len(names): FOREGROUND},
            image=image,
            confidence=confidence,
            boxes=boxes,
        )

        return predictions


class SegmentAnything2VideoBlock(_VideoBlock):
    """Track box or point prompted objects locally with SAM2 video.

    A source without prompts emits no rows until it can be seeded. Re-prompts
    restart backend tracker IDs; every frame always gets fresh row identity.
    """

    type = "v2/segment_anything_2_video@v1"
    metadata = {
        "name": "SAM2 Video Tracker",
        "port_of": "roboflow_core/segment_anything_2_video@v1",
        "section": "video",
    }

    class Params(_VideoParams):
        pass

    @classmethod
    def discover_dependent_resources(cls, params: Params) -> List[DependentResource]:
        """Declare the selected model in the models resource.

        Args:
            params: Configured block parameters.

        Returns:
            The local streaming model dependency.
        """
        dependencies = model_dependency(params, task_type=PREDICTION_TYPE)

        return dependencies

    def run(
        self,
        image: Batch[ImageData],
        boxes: Optional[Batch] = None,
        points: Optional[List[Any]] = None,
        model_id: str = SAM2_VIDEO_MODEL_ID,
        model_package_id: Optional[str] = None,
        prompt_mode: PromptMode = "first_frame",
        prompt_interval: int = 30,
        confidence: float = 0.0,
        max_sources: int = 256,
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
            max_sources: Maximum source handles to retain.

        Returns:
            Demanded dense/view predictions and model/inference identifiers.

        Raises:
            ValueError: For invalid inputs or malformed backend results.
            VideoSourceCapacityError: When a new source exceeds the bound.
        """
        result = self._run_video(
            image,
            boxes=boxes,
            points=points,
            model_id=model_id,
            model_package_id=model_package_id,
            confidence=confidence,
            prompt_mode=prompt_mode,
            prompt_interval=prompt_interval,
            max_sources=max_sources,
            mode="visual",
        )

        return result


class SegmentAnything3VideoBlock(_VideoBlock):
    """Track concepts continuously, or visual prompts, with local SAM3 video."""

    type = "v2/sam3_video@v1"
    metadata = {
        "name": "SAM3 Video Tracker",
        "port_of": "roboflow_core/sam3_video@v1",
        "section": "video",
    }

    class Params(_VideoParams):
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM3_CONCEPT_MODEL_ID,
            description="Local SAM3 concept video model id or cached package path.",
        )
        visual_model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM3_VISUAL_MODEL_ID,
            description="Local SAM3 visual video model id or cached package path.",
        )
        tracking_mode: Literal["concept", "visual"] | Ref(STRING_KIND) = Field(
            default="concept",
            description="Select text concept tracking or box/point visual tracking.",
        )
        class_names: Optional[List[StrictStr] | StrictStr] | Ref(
            LIST_OF_VALUES_KIND, STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Concept names as a list or comma-separated string; empty returns no objects.",
        )
        confidence: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description="Minimum concept score or inherited visual prompt confidence.",
        )

        @model_validator(mode="after")
        def _check_concepts(self):
            if self.tracking_mode == "concept" and self.class_names is None:
                raise ValueError(
                    "SAM3 concept mode requires class_names (an empty list is allowed)"
                )
            if not _is_selector(self.class_names):
                _concept_names(self.class_names)

            return self

    @classmethod
    def discover_dependent_resources(cls, params: Params) -> List[DependentResource]:
        """Declare model dependencies selected by the tracking mode.

        A selected mode declares both models, so the models resource can load
        and retain both. model_package_id applies to each declared model.

        Args:
            params: Configured parameters; a selected mode can use either model.

        Returns:
            Local models required by the literal or selected tracking mode.
        """
        identifiers = (
            [params.model_id, params.visual_model_id]
            if _is_selector(params.tracking_mode)
            else [
                (
                    params.visual_model_id
                    if params.tracking_mode == "visual"
                    else params.model_id
                )
            ]
        )
        dependencies = [
            DependentResource(
                resource_type="model",
                identifier=identifier,
                details={
                    "resource": "models",
                    "task_type": PREDICTION_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
            for identifier in identifiers
        ]

        return dependencies

    def run(
        self,
        image: Batch[ImageData],
        boxes: Optional[Batch] = None,
        points: Optional[List[Any]] = None,
        model_id: str = SAM3_CONCEPT_MODEL_ID,
        model_package_id: Optional[str] = None,
        prompt_mode: PromptMode = "first_frame",
        prompt_interval: int = 30,
        confidence: float = 0.5,
        max_sources: int = 256,
        tracking_mode: str = "concept",
        class_names: Any = None,
        visual_model_id: str = SAM3_VISUAL_MODEL_ID,
    ) -> List[Dict[str, Any]]:
        """Process sequential SAM3 concept or visual video frames.

        Args:
            image: RGB frames with source identity.
            boxes: Aligned visual prompt rows.
            points: Labeled visual points for one object.
            model_id: Concept video model id or cached package path.
            model_package_id: Exact optional package id for the selected model.
            prompt_mode: Visual re-prompt policy; concept mode prompts once.
            prompt_interval: Tracking calls between visual re-prompts.
            confidence: Minimum concept score or visual prompt confidence.
            max_sources: Maximum retained sources.
            tracking_mode: concept or visual, matching the backend contract.
            class_names: Concept prompts; empty emits no rows.
            visual_model_id: Visual tracker id or cached package path.

        Returns:
            Demanded dense/view predictions and model/inference identifiers.

        Raises:
            ValueError: For invalid inputs or malformed backend results.
            VideoSourceCapacityError: When a new source exceeds the bound.
        """
        selected = visual_model_id if tracking_mode == "visual" else model_id
        result = self._run_video(
            image,
            boxes=boxes,
            points=points,
            model_id=selected,
            model_package_id=model_package_id,
            confidence=confidence,
            prompt_mode=prompt_mode,
            prompt_interval=prompt_interval,
            max_sources=max_sources,
            mode=tracking_mode,
            class_names=class_names,
        )

        return result


def _is_selector(value):
    selector = isinstance(value, str) and bool(re.fullmatch(DATA_SELECTOR_PATTERN, value))

    return selector


def _positive_int(value, *, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"SAM video: {name} must be a positive integer")


def _concept_names(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [name.strip() for name in value.split(",") if name.strip()]
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(n, str) or not n.strip() for n in value
    ):
        raise ValueError("SAM video: class_names must contain non-empty strings")
    names = [name.strip() for name in value]
    if len(set(names)) != len(names):
        raise ValueError(
            "SAM video: concept names must be unique to map object ids unambiguously"
        )

    return names


def _backend_image(image):
    # The pinned backend reads HWC RGB arrays, not ImageData's CHW tensors.
    pixels = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()

    return pixels


def _visual_prompts(prompt, *, points, image):
    names = {} if prompt is None else class_names_by_id(prompt.image_metadata)
    bboxes, metadata = [], []
    if prompt is not None:
        xyxy = prompt.xyxy.detach().cpu()
        if xyxy.ndim != 2 or xyxy.shape[1] != 4 or not torch.isfinite(xyxy).all():
            raise ValueError("SAM video: visual boxes must be finite (N, 4) xyxy")
        height, width = image.size_hw
        if (
            (xyxy[:, :2] < 0).any()
            or (xyxy[:, 2] > width).any()
            or (xyxy[:, 3] > height).any()
            or (xyxy[:, 2:] <= xyxy[:, :2]).any()
        ):
            raise ValueError(
                "SAM video: visual boxes must have positive area inside the frame"
            )
        count = len(xyxy)
        if (
            prompt.class_id.shape != (count,)
            or prompt.confidence.shape != (count,)
            or (
                prompt.bboxes_metadata is not None
                and len(prompt.bboxes_metadata) != count
            )
        ):
            raise ValueError(
                "SAM video: box classes, confidences and row metadata must align with boxes"
            )
        if prompt.class_id.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("SAM video: prompt class ids must be integers")
        bboxes = [tuple(row) for row in xyxy.tolist()]
        classes = prompt.class_id.tolist()
        scores = prompt.confidence.tolist()
        for index, (class_id, score) in enumerate(zip(classes, scores)):
            if class_id < 0:
                raise ValueError("SAM video: prompt class ids must be non-negative")
            _unit_float(score, name="box confidence", block="SAM video")
            row = (
                {}
                if prompt.bboxes_metadata is None
                else without_host_mirrors(prompt.bboxes_metadata[index])
            )
            # A track is a new result row, while its original prompt stays lineage.
            metadata.append(
                _PromptObject(int(class_id), float(score), {"prompt_metadata": row})
            )
    if points:
        point_class = max([*names, *(m.class_id for m in metadata)], default=-1) + 1
        names[point_class] = FOREGROUND
        metadata.append(_PromptObject(point_class, 1.0, {}))

    return bboxes, metadata, names


def _mask_ids(masks, object_ids, *, image):
    masks = torch.as_tensor(np.asarray(masks))
    ids = torch.as_tensor(np.asarray(object_ids))
    if masks.dtype != torch.bool or masks.ndim != 3:
        raise ValueError("SAM video: backend masks must be binary bool (N, H, W)")
    if ids.ndim != 1 or ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(
            "SAM video: object_ids must be a one-dimensional integer array"
        )
    if (
        masks.shape[0] != len(ids)
        or len(set(ids.tolist())) != len(ids)
        or (ids < 0).any()
    ):
        raise ValueError(
            "SAM video: masks and unique non-negative object_ids must align"
        )
    # Visual HF backend's no-mask sentinel is (0, 0, 0), unlike concept (0,H,W).
    if masks.shape[0] == 0 and tuple(masks.shape[1:]) == (0, 0):
        masks = torch.zeros((0, *image.size_hw), dtype=torch.bool)
    if tuple(masks.shape[1:]) != image.size_hw:
        raise ValueError(
            f"SAM video: backend mask size {tuple(masks.shape[1:])} differs from frame {image.size_hw}"
        )

    return masks, ids.to(torch.int64)


def _handle(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("SAM video: backend must return a non-empty opaque state_dict")

    return value


def _visual_result(result, *, image):
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(
            "SAM video: visual backend must return (masks, object_ids, state_dict)"
        )
    masks, ids = _mask_ids(result[0], result[1], image=image)
    handle = _handle(result[2])

    return masks, ids, handle


def _concept_result(result, *, image, names):
    fields = (
        "masks",
        "object_ids",
        "scores",
        "boxes",
        "prompt_to_object_ids",
        "state_dict",
    )
    if isinstance(result, tuple) or any(not hasattr(result, key) for key in fields):
        raise ValueError(
            "SAM video: concept backend must return a SAM3VideoFrameResult record"
        )
    masks, ids = _mask_ids(result.masks, result.object_ids, image=image)
    scores = torch.as_tensor(np.asarray(result.scores), dtype=torch.float32)
    boxes = torch.as_tensor(np.asarray(result.boxes), dtype=torch.float32)
    if (
        scores.shape != ids.shape
        or not torch.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "SAM video: concept scores must align with ids and lie in 0..1"
        )
    if boxes.shape != (len(ids), 4) or not torch.isfinite(boxes).all():
        raise ValueError("SAM video: concept boxes must be finite (N, 4)")
    mapping = result.prompt_to_object_ids
    if not isinstance(mapping, Mapping) or any(name not in names for name in mapping):
        raise ValueError("SAM video: concept mapping must name registered prompts")
    claimed = set()
    for object_ids in mapping.values():
        if not isinstance(object_ids, (list, tuple)) or any(
            type(i) is not int or i < 0 for i in object_ids
        ):
            raise ValueError(
                "SAM video: concept mapping ids must be non-negative integers"
            )
        if any(i in claimed for i in object_ids) or len(set(object_ids)) != len(
            object_ids
        ):
            raise ValueError(
                "SAM video: concept mapping assigns one object to multiple prompts"
            )
        # Mapping can contain retained tracks absent from this frame's masks.
        claimed.update(object_ids)
    handle = _handle(result.state_dict)

    return masks, ids, scores, boxes, mapping, handle


def _frame_predictions(
    masks, *, ids, scores, classes, rows, names, image, confidence, boxes=None
):
    keep = (scores >= confidence) & masks.flatten(1).any(dim=1)
    device = image.device
    selected_masks = masks[keep].to(device)
    xyxy = _mask_boxes(selected_masks) if boxes is None else boxes[keep].to(device)
    selected_rows = [row for row, selected in zip(rows, keep.tolist()) if selected]
    predictions = InstanceDetections(
        xyxy=xyxy,
        mask=selected_masks,
        confidence=scores[keep].to(device),
        class_id=classes[keep].to(device),
        bboxes_metadata=selected_rows,
        image_metadata=_metadata(image, class_names=names),
    )

    return predictions

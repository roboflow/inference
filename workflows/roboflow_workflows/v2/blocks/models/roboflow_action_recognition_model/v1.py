"""Classify one bounded raw constant-rate clip with a local action model.

The caller or Window owns history. Results describe this clip, never a rolling
source timeline. Model-relative segment indices refer to ``sampled_frames``;
that mapping retains source identity, input position and exact timestamps.
"""

import math
from numbers import Real
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union
import torch
from pydantic import Field, StrictFloat, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import models_cache
from roboflow_workflows.v2.data.batch import Batch, Timestamp
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase
from inference_models.models.base.action_recognition import (
    SLIDING_WINDOW_MODE,
    UNTRAINED_MAX_FRAMES,
    WHOLE_VIDEO_MODE,
    VideoSampling,
    effective_max_frame_side,
    plan_windows,
)


_RAW_FRAMES = 1024

_FRAME_PIXELS = 8_294_400

_CLIP_BYTES = 256 * 1024 * 1024

_SAMPLED_FRAMES = 256

class _PreparedClip(NamedTuple):
    model: Any
    request: ModelRequest
    tensors: List[torch.Tensor]
    sample_fps: float
    vocabulary: Optional[Tuple[str, ...]]
    class_filter: Optional[Tuple[str, ...]]
    mapping: List[Dict[str, Any]]
    source_fps: float
    raw_frame_count: int

def _positive_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"action_recognition: {name} must be finite and positive")

    return float(value)

def _positive_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"action_recognition: {name} must be a positive integer")

    return value

def _names(value: Any, *, name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"action_recognition: {name} must be a nonempty list or null")
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise ValueError(f"action_recognition: {name} must contain nonempty names")
    if len(set(value)) != len(value):
        raise ValueError(f"action_recognition: {name} must not contain duplicates")

    return tuple(value)

def _sampling(model: Any) -> VideoSampling:
    sampling = getattr(model, "video_sampling", None)
    if not isinstance(sampling, VideoSampling) or not callable(
        getattr(model, "infer", None)
    ):
        raise TypeError(
            "action_recognition: model needs VideoSampling and infer(frames, fps)"
        )

    _positive_number(sampling.sample_fps, name="model sample_fps")
    _positive_count(sampling.min_frames, name="model min_frames")
    if sampling.mode not in (SLIDING_WINDOW_MODE, WHOLE_VIDEO_MODE):
        raise ValueError(
            f"action_recognition: unsupported sampling mode {sampling.mode!r}"
        )
    if sampling.mode == SLIDING_WINDOW_MODE:
        _positive_number(sampling.window_seconds, name="model window_seconds")
    if sampling.max_frames is not None:
        _positive_count(sampling.max_frames, name="model max_frames")
        if sampling.max_frames < sampling.min_frames:
            raise ValueError("action_recognition: model max_frames is below min_frames")
    if sampling.max_frame_side is not None:
        _positive_count(sampling.max_frame_side, name="model max_frame_side")

    return sampling

def _stamp_dict(stamp: Timestamp) -> Dict[str, Any]:
    result = {
        "ticks": stamp.ticks,
        "time_base": [stamp.time_base.numerator, stamp.time_base.denominator],
        "clock_id": stamp.clock_id,
    }

    return result

def _validate_clip(
    frames: Batch[ImageData],
    *,
    fps: float,
    max_raw_frames: int,
    max_frame_pixels: int,
    max_clip_bytes: int,
) -> List[Dict[str, Any]]:
    if not isinstance(frames, Batch) or not 2 <= len(frames) <= max_raw_frames:
        raise ValueError(
            f"action_recognition: raw clip needs 2..{max_raw_frames} frames; "
            "one repeated source frame cannot show motion"
        )
    if frames.indices != frames.known_indices:
        raise ValueError(
            "action_recognition: filtered inner positions leave holes in the raw clip"
        )
    positions = [index[-1] for index in frames.indices]
    if positions != list(range(positions[0], positions[0] + len(frames))):
        raise ValueError(
            "action_recognition: raw time positions must be contiguous and oldest first"
        )
    if frames.metadata is None:
        raise ValueError(
            "action_recognition: source and media/capture timestamps are required"
        )

    contexts = [frames.metadata.temporal_at(index) for index in frames.indices]
    # Observed time may follow processing speed, so it cannot establish source FPS.
    if all(
        context is not None and isinstance(context.media_coverage, Timestamp)
        for context in contexts
    ):
        clock_field = "media_coverage"
    elif all(
        context is not None and isinstance(context.capture_coverage, Timestamp)
        for context in contexts
    ):
        clock_field = "capture_coverage"
    else:
        raise ValueError(
            "action_recognition: every frame needs a media or capture point timestamp on one clock"
        )

    first_source = None
    first_stamp = None
    first_signature = None
    seen_images = set()
    bytes_total = 0
    if any(not isinstance(frame, ImageData) for frame in frames):
        raise TypeError("action_recognition: raw clip members must be ImageData")
    video_present = [frame.video_metadata is not None for frame in frames]
    if any(video_present) and not all(video_present):
        raise ValueError("action_recognition: mixed video metadata is unsupported")

    mapping = []
    for row, (index, frame, context) in enumerate(
        zip(frames.indices, frames, contexts)
    ):
        source = frames.metadata.sample_at(index)
        if source is None:
            raise ValueError(
                "action_recognition: every frame needs known source identity"
            )
        identity = (source.source_id, source.source_type)
        stamp = getattr(context, clock_field)
        if first_source is None:
            first_source, first_stamp = identity, stamp
        if identity != first_source:
            raise ValueError(
                "action_recognition: contradictory source identities within one clip"
            )
        elapsed = float(stamp.seconds - first_stamp.seconds)
        if stamp.clock_id != first_stamp.clock_id or not math.isclose(
            elapsed, row / fps, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError(
                "action_recognition: timestamps must have uniform cadence matching source_fps (1 microsecond tolerance)"
            )

        tensor = frame.tensor_image
        if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.dtype != torch.uint8:
            raise ValueError("action_recognition: frames must be uint8 CHW RGB tensors")
        signature = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if first_signature is None:
            first_signature = signature
        if signature != first_signature:
            raise ValueError(
                "action_recognition: mixed frame shape/dtype/device is unsupported"
            )
        if frame.height * frame.width > max_frame_pixels:
            raise ValueError(
                "action_recognition: raw frame exceeds max_frame_pixels before copying"
            )
        bytes_total += tensor.numel() * tensor.element_size()
        if bytes_total > max_clip_bytes:
            raise ValueError(
                "action_recognition: raw clip exceeds max_clip_bytes before copying"
            )
        if frame.image_id in seen_images:
            raise ValueError(
                "action_recognition: raw frames need distinct image identities"
            )
        seen_images.add(frame.image_id)
        video = frame.video_metadata
        if video is not None:
            if video.video_identifier != source.source_id:
                raise ValueError(
                    "action_recognition: image video identity contradicts source context"
                )
            if video.fps is not None and not math.isclose(
                _positive_number(video.fps, name="image fps"),
                fps,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("action_recognition: image fps contradicts source_fps")
            if (
                row
                and video.frame_number
                != frames.content[row - 1].video_metadata.frame_number + 1
            ):
                raise ValueError(
                    "action_recognition: source frame numbers must be contiguous"
                )
        mapping.append(
            {
                "input_frame_idx": row,
                "input_position": list(index),
                "image_id": frame.image_id,
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_frame_number": None if video is None else video.frame_number,
                "timestamp_field": clock_field,
                "timestamp": _stamp_dict(stamp),
            }
        )

    return mapping

def _microseconds(seconds: float) -> int:
    scaled = seconds * 1_000_000
    if not math.isfinite(scaled) or scaled <= 0:
        raise ValueError(
            "action_recognition: sampling interval cannot be represented in finite microseconds"
        )
    ticks = int(round(scaled))

    return ticks

def _plan_clip(
    frame_count: int, *, fps: float, sampling: VideoSampling, max_sampled_frames: int
):
    duration_us = _microseconds(frame_count / fps)
    if sampling.mode == SLIDING_WINDOW_MODE:
        limit_us = _microseconds(sampling.window_seconds)
    elif sampling.max_frames is None:
        limit_us = _microseconds(UNTRAINED_MAX_FRAMES / sampling.sample_fps)
    else:
        limit_us = None
    if duration_us <= 0 or (limit_us is not None and duration_us > limit_us):
        raise ValueError(
            "action_recognition: raw clip must fit exactly one compatible sample interval; shorten Window size (including any dropped tail)"
        )

    ceiling = (
        sampling.max_frames if sampling.max_frames is not None else UNTRAINED_MAX_FRAMES
    )
    budget = min(ceiling, max_sampled_frames)
    if sampling.min_frames > budget:
        raise ValueError(
            "action_recognition: model min_frames exceeds sample budget before planning"
        )
    if sampling.max_frames is None and frame_count < sampling.min_frames:
        raise ValueError(
            "action_recognition: insufficient raw content for untrained min_frames"
        )
    # The pinned planner allocates its index tuple eagerly. Bound its count
    # first using its public trained/untrained count policy; do not create a
    # potentially enormous tuple and only then inspect its length.
    rate = (
        sampling.sample_fps
        if sampling.max_frames is not None
        else min(sampling.sample_fps, fps)
    )
    nominal_count = duration_us / 1_000_000 * rate
    if not math.isfinite(nominal_count):
        raise ValueError(
            "action_recognition: model sampling count exceeds finite sample budget"
        )
    count = (
        max(sampling.min_frames, min(ceiling, int(round(nominal_count))))
        if sampling.max_frames is not None
        else max(sampling.min_frames, int(round(nominal_count)))
    )
    if count > budget:
        raise ValueError(
            "action_recognition: selected frame count exceeds sample budget before planning"
        )

    # Public planning is the sole source of selected indices and effective rate.
    windows = plan_windows(frame_count=frame_count, source_fps=fps, sampling=sampling)
    if len(windows) != 1:
        raise ValueError(
            "action_recognition: public planner must produce exactly one WindowSpec"
        )
    window = windows[0]
    indices = window.frame_indices
    ceiling = (
        sampling.max_frames if sampling.max_frames is not None else UNTRAINED_MAX_FRAMES
    )
    if not sampling.min_frames <= len(indices) <= min(ceiling, max_sampled_frames):
        raise ValueError(
            "action_recognition: selected frame count violates model or adapter sample budget"
        )
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < frame_count
        for index in indices
    ):
        raise ValueError(
            "action_recognition: public planner returned invalid frame indices"
        )
    if len(set(indices)) < 2:
        raise ValueError(
            "action_recognition: insufficient distinct selected frames for motion"
        )
    if tuple(sorted(indices)) != indices:
        raise ValueError(
            "action_recognition: public planner returned unordered frame indices"
        )
    _positive_number(window.sample_fps, name="effective sample_fps")

    return window

def _model_tensor(image: ImageData, *, max_side: Optional[int]) -> torch.Tensor:
    tensor = image.tensor_image
    if max_side is None or max(image.size_hw) <= max_side:
        copied = tensor.clone(memory_format=torch.contiguous_format)
        return copied

    scale = max_side / max(image.size_hw)
    height, width = (max(1, round(side * scale)) for side in image.size_hw)
    # Same on-device area path as V1 tensor action. At noninteger shrink
    # ratios, cv2 uses different area weights as well as byte rounding.
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0).to(torch.float32), size=(height, width), mode="area"
    ).squeeze(0)
    copied = resized.round_().clamp_(0, 255).to(torch.uint8)

    return copied

class ActionRecognitionBlock(Block):
    """Classify an explicit raw clip using a host-owned model cache.

    Args:
        models: ModelCache or exact request-to-model mapping. Cache loading is
            coalesced; inference uses the inherited scheduler concurrency policy.
    """

    type = "roboflow_core/roboflow_action_recognition_model@v1"
    outputs = {
        "predictions": Output(
            DICTIONARY_KIND,
            source="frames",
            context_policy="last",
            description="Clip-local inclusive sampled-index segments plus exact sampled-to-source mapping; last chronological frame context.",
        ),
    }

    class Params(BlockParams):
        frames: Group(IMAGE_KIND, temporal=True) = Field(
            description="One oldest-first raw constant-rate clip with known source and media/capture point timestamps."
        )
        source_fps: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            gt=0,
            allow_inf_nan=False,
            description="Explicit finite positive source rate; never inferred from processing time.",
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            description="Local action model ID/path or selector in the models resource."
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description="Exact local model package per parent clip; None lets the loader choose.",
            )
        )
        class_filter: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description="Optional exact output class subset; fixed models also receive it as their infer restriction. Open models are filtered after free-form inference. Empty lists are invalid.",
        )
        max_raw_frames: StrictInt = Field(
            default=_RAW_FRAMES,
            gt=0,
            description="Maximum raw arrivals per clip, checked before copies.",
        )
        max_frame_pixels: StrictInt = Field(
            default=_FRAME_PIXELS,
            gt=0,
            description="Maximum pixels per raw RGB frame, checked before resizing.",
        )
        max_clip_bytes: StrictInt = Field(
            default=_CLIP_BYTES,
            gt=0,
            description="Byte ceiling separately for retained raw tensors and selected pre-resize copies; excludes upstream/processor allocations.",
        )
        max_sampled_frames: StrictInt = Field(
            default=_SAMPLED_FRAMES,
            gt=0,
            description="Additional finite selected-frame ceiling, independent of model budget.",
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Describe the exact local model request.

        Args:
            params: Validated step configuration.

        Returns:
            Model dependency including package selection.
        """
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": "action-recognition",
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    @phase
    def network_input(
        self,
        frames: Batch[ImageData],
        source_fps: float,
        model_id: str,
        model_package_id: Optional[str],
        class_filter: Optional[List[str]],
        max_raw_frames: int,
        max_frame_pixels: int,
        max_clip_bytes: int,
        max_sampled_frames: int,
    ) -> _PreparedClip:
        """Validate raw metadata and bounds, plan once, then copy selected frames.

        Args:
            frames: One parent raw clip with temporal metadata.
            source_fps: Constant source rate.
            model_id: Local model ID/path.
            model_package_id: Exact package or None.
            class_filter: Output vocabulary subset or None.
            max_raw_frames: Raw arrival bound.
            max_frame_pixels: Per-raw-frame pixel bound.
            max_clip_bytes: Raw and selected byte bounds.
            max_sampled_frames: Additional model input count bound.

        Returns:
            Isolated model inputs and exact source mapping.

        Raises:
            ValueError: On incompatible cadence, vocabulary, sampling or bounds.
        """
        fps = _positive_number(source_fps, name="source_fps")
        for name, value in (
            ("max_raw_frames", max_raw_frames),
            ("max_frame_pixels", max_frame_pixels),
            ("max_clip_bytes", max_clip_bytes),
            ("max_sampled_frames", max_sampled_frames),
        ):
            _positive_count(value, name=name)
        mapping = _validate_clip(
            frames,
            fps=fps,
            max_raw_frames=max_raw_frames,
            max_frame_pixels=max_frame_pixels,
            max_clip_bytes=max_clip_bytes,
        )
        selected_filter = _names(class_filter, name="class_filter")
        request = ModelRequest(model_id, model_package_id)
        model = self._models.get(request)
        sampling = _sampling(model)
        vocabulary = _names(model.class_names, name="model class_names")
        if vocabulary is not None and selected_filter is not None:
            unknown = set(selected_filter) - set(vocabulary)
            if unknown:
                raise ValueError(
                    f"action_recognition: unknown fixed classes {sorted(unknown)!r}; available {list(vocabulary)!r}"
                )
        window = _plan_clip(
            len(frames),
            fps=fps,
            sampling=sampling,
            max_sampled_frames=max_sampled_frames,
        )
        selected_bytes = sum(
            frames.content[i].tensor_image.numel()
            * frames.content[i].tensor_image.element_size()
            for i in window.frame_indices
        )
        if selected_bytes > max_clip_bytes:
            raise ValueError(
                "action_recognition: selected copies exceed max_clip_bytes before materialization"
            )

        tensors = [
            _model_tensor(
                frames.content[i], max_side=effective_max_frame_side(sampling)
            )
            for i in window.frame_indices
        ]
        selected_mapping = [
            {"sampled_frame_idx": sample, **mapping[i]}
            for sample, i in enumerate(window.frame_indices)
        ]
        prepared = _PreparedClip(
            model,
            request,
            tensors,
            window.sample_fps,
            vocabulary,
            selected_filter,
            selected_mapping,
            fps,
            len(frames),
        )

        return prepared

    @phase
    def raw_output(self, network_input: _PreparedClip) -> Any:
        """Call the selected model once synchronously.

        Args:
            network_input: Isolated selected tensors and contract.

        Returns:
            Backend segments in sampled-index space.
        """
        clip = network_input
        # Open Cosmos ignores prompt vocabularies. Apply a truthful output
        # filter here after unconstrained inference rather than claiming one.
        names = (
            list(clip.class_filter)
            if clip.vocabulary is not None and clip.class_filter is not None
            else None
        )
        raw = clip.model.infer(
            frames=clip.tensors, class_names=names, fps=clip.sample_fps
        )

        return raw

    @phase
    def result(self, raw_output: Any, network_input: _PreparedClip) -> Dict[str, Any]:
        """Validate segments and return JSON-friendly clip evidence.

        Args:
            raw_output: Backend segment list.
            network_input: Prepared clip and mapping.

        Returns:
            Clip predictions; inclusive bounds reference sampled_frames.

        Raises:
            ValueError: On malformed backend ranges or vocabulary.
        """
        clip = network_input
        if not isinstance(raw_output, (list, tuple)):
            raise ValueError("action_recognition: backend must return a segment list")
        ids = (
            {}
            if clip.vocabulary is None
            else {name: i for i, name in enumerate(clip.vocabulary)}
        )
        segments = []
        for segment in raw_output:
            start = getattr(segment, "start_frame_idx", None)
            end = getattr(segment, "end_frame_idx", None)
            label = getattr(segment, "class_name", None)
            if any(
                isinstance(i, bool) or not isinstance(i, int) for i in (start, end)
            ) or not 0 <= start <= end < len(clip.mapping):
                raise ValueError(
                    "action_recognition: backend inclusive segment range is outside selected frames"
                )
            if (
                not isinstance(label, str)
                or not label.strip()
                or (clip.vocabulary is not None and label not in ids)
            ):
                raise ValueError(
                    "action_recognition: backend class is invalid or outside fixed vocabulary"
                )
            if clip.class_filter is not None and label not in clip.class_filter:
                continue
            segments.append(
                {
                    "start_frame_idx": start,
                    "end_frame_idx": end,
                    "class_name": label,
                    "class_id": ids.get(label, -1),
                }
            )
        predictions = {
            "segments": segments,
            "sampled_frames": clip.mapping,
            "source_fps": clip.source_fps,
            "sample_fps": clip.sample_fps,
            "raw_frame_count": clip.raw_frame_count,
            "model_id": clip.request.model_id,
            "model_package_id": clip.request.model_package_id,
            "index_space": "clip_sampled_frames",
            "range_end": "inclusive",
        }
        result = {"predictions": predictions}

        return result

    def run(
        self,
        *,
        frames: Batch[ImageData],
        source_fps: float,
        model_id: str,
        model_package_id: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
        max_raw_frames: int = _RAW_FRAMES,
        max_frame_pixels: int = _FRAME_PIXELS,
        max_clip_bytes: int = _CLIP_BYTES,
        max_sampled_frames: int = _SAMPLED_FRAMES,
    ) -> Dict[str, Any]:
        """Execute the same preparation, inference and result phases directly.

        Args:
            frames: One raw constant-rate clip.
            source_fps: Explicit nominal source FPS.
            model_id: Local model ID/path.
            model_package_id: Exact package or None.
            class_filter: Optional exact output subset.
            max_raw_frames: Raw arrival ceiling.
            max_frame_pixels: Raw frame pixel ceiling.
            max_clip_bytes: Raw and selected-copy byte ceilings.
            max_sampled_frames: Selected count ceiling.

        Returns:
            One clip prediction dictionary with exact source mapping.
        """
        prepared = self.network_input(
            frames,
            source_fps,
            model_id,
            model_package_id,
            class_filter,
            max_raw_frames,
            max_frame_pixels,
            max_clip_bytes,
            max_sampled_frames,
        )
        raw = self.raw_output(prepared)
        result = self.result(raw, prepared)

        return result

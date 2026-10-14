"""``VideoTracker``: local SAM2 visual and SAM3 visual/concept video tracking.

ModelCache owns reusable model weights. Each block instance owns one tracker,
which owns separate opaque backend handles per source. Handles are never
workflow state or output. The engine creates a fresh instance on a processing
reset or a new session and serializes the block's run in pulse order; batch
members run in order. Callers outside the engine must serialize calls
themselves. Backend locks also protect models shared across steps/sessions,
but do not order source frames.

Visual re-prompts start new backend sessions and restart tracker IDs. V1's
every_n_frames counter is retained: after prompting, N tracking calls occur
before the next call re-prompts. A due frame without prompts keeps tracking.
An unseeded source returns empty results until prompts arrive. Concept mode
prompts once per concept set and tracks new matching objects continuously.
A fresh backend session numbers its first frame 0, so prompts attach there;
source frame numbers only order frames and detect rollback.

Lifetime: each actual backend session is an episode with its own id. The
required max_session_frames bounds its successful backend frames; the call
that would exceed it is never made. on_session_limit="error" releases the
handle and stops that source until reset_source; "restart" starts a new
episode once the source can seed. Tracker ids are local to an episode. An
invocation is one pulse; max_idle_pulses releases idle handles but retains
expired entries until explicit reset or cleanup. Both stopped and expired
entries count against max_sources.

All members are checked before any is processed: image, frame number, source
identity and stopped/expired sources. Those errors change no state. Budget
exhaustion releases the handle and retains a stopped entry. Other errors during
frame processing discard that source, including an invalid due prompt: opaque
sessions can mutate before throwing and cannot be rolled back. Earlier
members remain advanced and later members are not processed. Discarding drops
this block's references only; memory is freed when nothing else references it.
Frame and source bounds do not bound bytes, objects per frame or the cache.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.models._shared.segment_anything import (
    FOREGROUND,
    _mask_boxes,
    _metadata,
    _prompt_boxes,
    _require_same_frame,
    _segmentation_result,
    _unit_float,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import require_rgb
from roboflow_workflows.v2.blocks.models.sam3_interactive.v1 import _labeled_points
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.context import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import DATA_SELECTOR_PATTERN, Block


UNBOUNDED = "unbounded"

VIDEO_EPISODE_KEY = "video_episode"

class VideoSourceCapacityError(RuntimeError):
    """A new source exceeds max_sources; release ended sources or raise the bound."""

class VideoSessionLimitError(RuntimeError):
    """A source reached max_session_frames under on_session_limit="error".

    Its backend handle is already released. The source refuses frames until a
    frame arrives with reset_source=true, or until it is released. source_id
    names the stopped source.
    """

    def __init__(self, message: str, *, source_id: str):
        """Create the error for one stopped source.

        Args:
            message: Human-readable explanation and recovery action.
            source_id: Source whose frames are refused.
        """
        super().__init__(message)
        self.source_id = source_id

class _PromptObject(NamedTuple):
    class_id: int
    confidence: float
    row: dict

@dataclass
class _SourceSession:
    configuration: tuple
    size_hw: Tuple[int, int]
    last_pulse: int
    handle: Any = None
    last_frame: Optional[int] = None
    frames_since_prompt: int = 0
    visual_metadata: Dict[int, _PromptObject] = field(default_factory=dict)
    class_names: dict = field(default_factory=dict)
    episode_id: Optional[str] = None
    # Successful backend frames in the current episode (backend session).
    episode_frames: int = 0
    # Reached max_session_frames under "error"; only a reset or release clears it.
    stopped: bool = False
    # Idle expiry retains the slot and refuses re-entry until reset or cleanup.
    expired: bool = False

    def end_episode(self) -> None:
        # Drop the handle before any new session is created for this source.
        self.handle = None
        self.visual_metadata = {}
        self.class_names = {}
        self.episode_id = None
        self.episode_frames = 0
        self.frames_since_prompt = 0

    def start_episode(self, handle) -> None:
        self.handle = handle
        self.episode_id = str(uuid.uuid4())
        self.episode_frames = 1
        self.frames_since_prompt = 0

    def continue_episode(self, handle) -> None:
        self.handle = handle
        self.episode_frames += 1
        self.frames_since_prompt += 1

class VideoTracker:
    """Per-source backend sessions of one SAM video block and their frame steps.

    Args:
        models: The block's model cache.
        block: Block type, for error messages.
    """

    def __init__(self, models: ModelCache, *, block: str):
        self._models = models
        self._block = block
        self._sources: Dict[str, _SourceSession] = {}
        # One pulse per track() call that passes the batch checks.
        self._pulse = 0

    def reset(self) -> None:
        """Drop every source's handle and retained state."""
        self._sources.clear()

    def release_source(self, source_id: str) -> bool:
        """Drop one source's handle and retained state.

        Args:
            source_id: Identity used for the source's frames.

        Returns:
            Whether the source was retained.
        """
        released = self._sources.pop(source_id, None) is not None

        return released

    def track(
        self,
        image: Batch[ImageData],
        *,
        reader: Block,
        boxes: Optional[Batch],
        points: Optional[List[Any]],
        model_id: str,
        model_package_id: Optional[str],
        confidence: float,
        prompt_mode: str,
        prompt_interval: int,
        max_sources: int,
        max_session_frames: Any,
        on_session_limit: str,
        reset_source: Any,
        max_idle_pulses: Optional[int],
        mode: str,
        class_names: Any = None,
    ) -> List[Dict[str, Any]]:
        """Process one invocation's frames in batch order; see the module docstring.

        Args:
            image: RGB frames with video metadata or source sample context.
            reader: The calling block: its execution context names sources
                and its demanded outputs are built.
            boxes: Aligned visual prompt rows, or ``None``.
            points: Labeled visual points for one extra object.
            model_id: Model id of the selected backend.
            model_package_id: Exact optional package id.
            confidence: Minimum emitted score.
            prompt_mode: Visual re-prompt policy.
            prompt_interval: Tracking calls between scheduled visual prompts.
            max_sources: Maximum retained sources.
            max_session_frames: Positive backend-frame budget per session, or
                "unbounded".
            on_session_limit: "error" or "restart".
            reset_source: One flag, or a ``Batch`` of flags aligned with image.
            max_idle_pulses: Missed invocations before release, or ``None``.
            mode: "visual" or "concept".
            class_names: Concept prompts (concept mode only).

        Returns:
            One segmentation result per frame, in batch order.
        """
        confidence = _unit_float(confidence, name="confidence", block=self._block)
        _positive_int(prompt_interval, name="prompt_interval")
        _positive_int(max_sources, name="max_sources")
        limit = _session_frame_limit(max_session_frames)
        if on_session_limit not in ("error", "restart"):
            raise ValueError(
                f"{self._block}: on_session_limit must be error or restart"
            )
        if max_idle_pulses is not None:
            _positive_int(max_idle_pulses, name="max_idle_pulses")
        if prompt_mode not in ("first_frame", "every_n_frames", "every_frame"):
            raise ValueError(f"{self._block}: unsupported prompt_mode {prompt_mode!r}")
        if mode not in ("visual", "concept"):
            raise ValueError(f"{self._block}: tracking_mode must be concept or visual")
        if boxes is not None and (
            len(boxes) != len(image) or boxes.indices != image.indices
        ):
            raise ValueError(
                f"{self._block}: boxes and image batches must align in length and "
                "indices"
            )

        names = _concept_names(class_names) if mode == "concept" else []
        point_list = (
            _labeled_points(points, block=self._block) if mode == "visual" else []
        )
        resets = self._member_resets(reset_source, image=image)
        identities = self._preflight(image, resets=resets, reader=reader)
        self._start_pulse(
            [key for key, _ in identities],
            max_idle_pulses=max_idle_pulses,
            max_sources=max_sources,
        )

        request = ModelRequest(model_id, model_package_id)
        configuration = (request, mode, tuple(names), prompt_mode, prompt_interval)
        results = []
        for position, member in enumerate(image):
            key, frame = identities[position]
            transition = None
            if resets[position]:
                # Drops the old handle (and a stop) before any new session starts.
                self._sources.pop(key, None)
                transition = "reset"
            session = self._sources.get(key)
            rolled_back = (
                session is not None
                and frame is not None
                and session.last_frame is not None
                and frame < session.last_frame
            )
            if (
                session is None
                or session.configuration != configuration
                or session.size_hw != member.size_hw
                or rolled_back
            ):
                session = _SourceSession(
                    configuration=configuration,
                    size_hw=member.size_hw,
                    last_pulse=self._pulse,
                )
                self._sources[key] = session

            at_limit = (
                session.handle is not None
                and limit is not None
                and session.episode_frames >= limit
            )
            restart = at_limit and on_session_limit == "restart"
            # Inputs read only for visual prompt policy; tracking ignores current boxes/points.
            scheduled = (
                session.handle is None
                or prompt_mode == "every_frame"
                or (
                    prompt_mode == "every_n_frames"
                    and session.frames_since_prompt >= prompt_interval
                )
            )
            due = scheduled or restart
            try:
                prompt = (
                    _prompt_boxes(boxes[position], block=self._block)
                    if mode == "visual" and due and boxes is not None
                    else None
                )
                member_points = point_list if mode == "visual" and due else []
                # A scheduled visual re-prompt starts a fresh session within the limit.
                reprompts = scheduled and _has_prompts(prompt, member_points)
                if at_limit and not reprompts:
                    if not restart:
                        session.end_episode()
                        session.stopped = True
                        raise VideoSessionLimitError(
                            f"{self._block}: source {key!r} reached max_session_frames="
                            f"{limit}; its session was released. Send reset_source="
                            "true with its next frame to start a new episode",
                            source_id=key,
                        )
                    session.end_episode()
                    transition = "session_limit"

                if mode == "concept":
                    predictions = self._concept_frame(
                        session,
                        member,
                        request=request,
                        names=names,
                        confidence=confidence,
                    )
                else:
                    predictions = self._visual_frame(
                        session,
                        member,
                        request=request,
                        prompt=prompt,
                        points=member_points,
                        due=due,
                        confidence=confidence,
                    )
                session.last_frame = frame
                if transition is None and session.episode_frames == 1:
                    transition = "new_session"
                predictions.image_metadata[VIDEO_EPISODE_KEY] = {
                    "episode_id": session.episode_id,
                    "episode_frame": session.episode_frames,
                    "transition": transition,
                }
                inference_id = str(uuid.uuid4())
                predictions = with_row_identity(predictions, namespace=inference_id)
                result = _segmentation_result(
                    predictions,
                    reader=reader,
                    model_id=model_id,
                    inference_id=inference_id,
                )
                results.append(result)
            except VideoSessionLimitError:
                # Keep the stopped entry: it refuses frames until reset or release.
                raise
            except Exception:
                # Includes invalid prompts and malformed results after a backend call.
                self._sources.pop(key, None)
                raise

        return results

    def _member_resets(self, reset_source, *, image) -> List[bool]:
        if isinstance(reset_source, Batch):
            if len(reset_source) != len(image) or reset_source.indices != image.indices:
                raise ValueError(
                    f"{self._block}: reset_source and image batches must align"
                )
            resets = list(reset_source)
        else:
            resets = [reset_source] * len(image)
        if any(type(value) is not bool for value in resets):
            raise ValueError(f"{self._block}: reset_source must be true or false")

        return resets

    def _preflight(self, image, *, resets, reader) -> List[Tuple[str, Optional[int]]]:
        """Check every member before any is processed; changes no state."""
        stopped = {key for key, session in self._sources.items() if session.stopped}
        expired = {key for key, session in self._sources.items() if session.expired}
        identities = []
        for position, member in enumerate(image):
            require_rgb(member, block=self._block)
            key, frame = self._source_identity(
                member, index=image.indices[position], reader=reader
            )
            if resets[position]:
                stopped.discard(key)
                expired.discard(key)
            elif key in stopped:
                raise VideoSessionLimitError(
                    f"{self._block}: source {key!r} stopped at max_session_frames; "
                    "send reset_source=true with its frame, or release the source",
                    source_id=key,
                )
            elif key in expired:
                raise RuntimeError(
                    f"{self._block}: source {key!r} expired at max_idle_pulses; send "
                    "reset_source=true with its frame, or release the source"
                )
            identities.append((key, frame))

        return identities

    def _start_pulse(self, keys, *, max_idle_pulses, max_sources) -> None:
        """Count this invocation, expire idle handles, then check retained capacity."""
        self._pulse += 1
        present = set(keys)
        if max_idle_pulses is not None:
            idle = [
                key
                for key, session in self._sources.items()
                if key not in present
                and not session.stopped
                and not session.expired
                and self._pulse - session.last_pulse > max_idle_pulses
            ]
            for key in idle:
                session = self._sources[key]
                session.end_episode()
                session.expired = True
        for key in present & self._sources.keys():
            self._sources[key].last_pulse = self._pulse

        added = present - self._sources.keys()
        if len(self._sources) + len(added) > max_sources:
            raise VideoSourceCapacityError(
                f"{self._block}: max_sources={max_sources} exceeded by {len(added)} "
                "new source(s); release ended, stopped or expired sources "
                "(release_source or a processing reset), or increase max_sources"
            )

    def _source_identity(self, image: ImageData, *, index: tuple, reader: Block):
        metadata = image.video_metadata
        frame = None if metadata is None else metadata.frame_number
        if frame is not None and (type(frame) is not int or frame < 0):
            raise ValueError(
                f"{self._block}: frame_number must be a non-negative integer or None"
            )
        if metadata is not None and metadata.video_identifier:
            return metadata.video_identifier, frame

        try:
            context = reader.execution_context
        except NoExecutionContextError:
            return "default", frame

        sample = context.sample_at(index, field="image")
        key = sample.source_id if sample is not None else "default"

        return key, frame

    def _visual_frame(
        self, session, image, *, request, prompt, points, due, confidence
    ):
        if due and _has_prompts(prompt, points):
            _require_same_frame(prompt, image, block=self._block)
            for x, y, _ in points:
                if not (0 <= x < image.size_hw[1] and 0 <= y < image.size_hw[0]):
                    raise ValueError(
                        f"{self._block}: point {(x, y)} is outside image "
                        f"{image.size_hw}"
                    )
            bboxes, metadata, names = _visual_prompts(
                prompt, points=points, image=image
            )
            session.end_episode()
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
                    f"{self._block}: visual prompt returned an unprompted object id"
                )
            session.start_episode(handle)
        elif session.handle is not None:
            model = self._models.get(request)
            result = model.track(image=_backend_image(image), state_dict=session.handle)
            masks, ids, handle = _visual_result(result, image=image)
            if any(int(object_id) not in session.visual_metadata for object_id in ids):
                raise ValueError(
                    f"{self._block}: visual tracker returned an unprompted object id"
                )
            session.continue_episode(handle)
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
            classes = torch.zeros(0, dtype=torch.int64)
        else:
            model = self._models.get(request)
            starts = session.handle is None
            if starts:
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
            masks, ids, scores, mapping, handle = _concept_result(
                result, image=image, names=names
            )
            if starts:
                session.start_episode(handle)
            else:
                session.continue_episode(handle)
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
        )

        return predictions

def _is_selector(value):
    selector = isinstance(value, str) and bool(
        re.fullmatch(DATA_SELECTOR_PATTERN, value)
    )

    return selector

def _positive_int(value, *, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"SAM video: {name} must be a positive integer")

def _session_frame_limit(value):
    # None means the caller chose unbounded sessions explicitly.
    if value == UNBOUNDED:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(
            "SAM video: max_session_frames must be a positive integer or 'unbounded'"
        )

    return value

def _has_prompts(prompt, points):
    present = prompt is not None and len(prompt) > 0 or bool(points)

    return present

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
    # Backend boxes are inclusive occupied-pixel extrema (masks_to_boxes). They are
    # validated as part of the record, but output boxes are half-open mask extents.
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

    return masks, ids, scores, mapping, handle

def _frame_predictions(masks, *, ids, scores, classes, rows, names, image, confidence):
    keep = (scores >= confidence) & masks.flatten(1).any(dim=1)
    device = image.device
    selected_masks = masks[keep].to(device)
    xyxy = _mask_boxes(selected_masks)
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

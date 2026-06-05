"""Shared helpers for SAM2/SAM3 streaming video tracker workflow blocks.

Both blocks multiplex a single ``inference_models``-backed streaming
model across many videos by keying ``state_dict``s on
``video_identifier``.  They follow the same decision logic on every
frame: reset a session if the source stream restarted, and re-prompt
only on the frames requested by ``prompt_mode``.  Everything that is
independent of "SAM2 vs SAM3" lives here so each concrete block is just
a thin wrapper around ``inference_models.AutoModel``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import numpy as np
import supervision as sv

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

DETECTIONS_CLASS_NAME_FIELD = "class_name"

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]


@dataclass
class VideoSessionBookkeeping:
    """Per-video bookkeeping that lives alongside the model's opaque
    ``state_dict``.

    We store the last state returned from the model so the next call
    can continue the same session; ``obj_id_metadata`` holds the
    detector-provided class name / id / parent detection id for each
    prompted track so the emitted masks inherit them.
    """

    state_dict: Optional[dict] = None
    last_frame_number: int = -1
    frames_since_prompt: int = 0
    obj_id_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class BoxPromptMetadata:
    """Class info carried from an upstream detector to the emitted mask."""

    class_id: int
    class_name: str
    confidence: float
    parent_id: Optional[str]


def extract_box_prompts(
    boxes_for_image: Optional[sv.Detections],
) -> Tuple[List[Tuple[float, float, float, float]], List[BoxPromptMetadata]]:
    """Flatten an ``sv.Detections`` into xyxy tuples + per-box metadata.

    Empty / missing input returns two empty lists; class_name defaults
    to "foreground" when the detection doesn't carry one.
    """
    if boxes_for_image is None or len(boxes_for_image) == 0:
        return [], []

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    metas: List[BoxPromptMetadata] = []
    for xyxy, _mask, confidence, class_id, _tracker_id, data in boxes_for_image:
        x1, y1, x2, y2 = xyxy
        boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        class_name = (
            data.get(DETECTIONS_CLASS_NAME_FIELD, "foreground")
            if isinstance(data, dict)
            else "foreground"
        )
        parent_id = data.get("detection_id") if isinstance(data, dict) else None
        metas.append(
            BoxPromptMetadata(
                class_id=int(class_id) if class_id is not None else 0,
                class_name=str(class_name),
                confidence=float(confidence) if confidence is not None else 1.0,
                parent_id=str(parent_id) if parent_id is not None else None,
            )
        )
    return boxes_xyxy, metas


def decide_prompt_vs_track(
    session: VideoSessionBookkeeping,
    frame_number: int,
    prompt_mode: PromptMode,
    prompt_interval: int,
    has_prompts: bool,
) -> Tuple[bool, bool]:
    """Return ``(should_reset, should_prompt)`` for a single frame.

    - A reset fires when the source stream's ``frame_number`` rolls
      back (or this is the first frame we've seen for this video).
    - ``should_prompt`` is gated on prompt availability: there's no
      point issuing a prompt call with nothing to prompt on.
    """
    fresh_session = session.last_frame_number < 0 or session.state_dict is None
    reset = fresh_session or frame_number < session.last_frame_number

    if prompt_mode == "every_frame":
        return reset, has_prompts
    if prompt_mode == "every_n_frames":
        due = reset or session.frames_since_prompt >= max(1, prompt_interval)
        return reset, due and has_prompts
    # first_frame
    return reset, reset and has_prompts


def masks_to_sv_detections(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    image: WorkflowImageData,
    obj_id_metadata: Dict[int, BoxPromptMetadata],
    threshold: float,
) -> sv.Detections:
    """Assemble one ``sv.Detections`` of instance-seg predictions.

    Emits one detection per SAM-assigned object (preserving the
    one-to-one mapping with ``tracker_id``).  Masks without any positive
    pixels are dropped.
    """
    h, w = image.numpy_image.shape[:2]
    if masks.shape[0] == 0:
        return _empty_detections(h, w)

    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names: List[str] = []
    tracker_ids: List[int] = []
    detection_ids: List[str] = []
    parent_ids: List[str] = []
    kept_masks: List[np.ndarray] = []

    for mask, obj_id in zip(masks, obj_ids.tolist()):
        meta = obj_id_metadata.get(int(obj_id))
        confidence = meta.confidence if meta is not None else 1.0
        if confidence < threshold:
            continue
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        xyxy.append(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ]
        )
        confidences.append(float(confidence))
        class_ids.append(meta.class_id if meta is not None else 0)
        class_names.append(meta.class_name if meta is not None else "foreground")
        tracker_ids.append(int(obj_id))
        parent = meta.parent_id if meta is not None else None
        parent_ids.append(str(parent) if parent is not None else "")
        detection_ids.append(str(uuid4()))
        kept_masks.append(mask.astype(bool))

    if not kept_masks:
        return _empty_detections(h, w)

    detections = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        mask=np.stack(kept_masks, axis=0),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
        tracker_id=np.asarray(tracker_ids, dtype=int),
    )
    detections.data[DETECTIONS_CLASS_NAME_FIELD] = np.asarray(class_names, dtype=object)
    detections[DETECTION_ID_KEY] = np.asarray(detection_ids, dtype=object)
    detections[PARENT_ID_KEY] = np.asarray(parent_ids, dtype=object)
    detections[IMAGE_DIMENSIONS_KEY] = np.asarray([[h, w]] * len(detections), dtype=int)
    return detections


def _empty_detections(h: int, w: int) -> sv.Detections:
    empty = sv.Detections.empty()
    empty[DETECTION_ID_KEY] = np.array([], dtype=object)
    empty[PARENT_ID_KEY] = np.array([], dtype=object)
    empty[IMAGE_DIMENSIONS_KEY] = np.zeros((0, 2), dtype=int)
    empty.data[DETECTIONS_CLASS_NAME_FIELD] = np.array([], dtype=object)
    return empty


def build_obj_id_metadata_from_boxes(
    obj_ids: np.ndarray,
    box_metas: List[BoxPromptMetadata],
) -> Dict[int, BoxPromptMetadata]:
    """Align SAM-assigned object ids with the detector-provided metadata.

    The model hands us object ids in the same order as the prompts we
    issued; we zip them together so later frames (which only have
    ``obj_ids``) can still be labelled.
    """
    return dict(zip([int(i) for i in obj_ids.tolist()], box_metas))


def build_obj_id_metadata_from_text(
    obj_ids: np.ndarray,
    class_names: List[str],
) -> Dict[int, BoxPromptMetadata]:
    """For text-prompt sessions where we don't have per-object class
    info, fall back to a single class name (if only one was supplied)
    or "foreground" (if multiple or none).
    """
    label = class_names[0] if len(class_names) == 1 and class_names[0] else "foreground"
    return {
        int(oid): BoxPromptMetadata(
            class_id=0, class_name=label, confidence=1.0, parent_id=None
        )
        for oid in obj_ids.tolist()
    }


def normalise_class_names(
    class_names: Optional[Any],
) -> List[str]:
    """Accept a list, comma-separated string, or None and return a list."""
    if class_names is None:
        return []
    if isinstance(class_names, str):
        return [c.strip() for c in class_names.split(",") if c.strip()]
    return [c for c in class_names if c]

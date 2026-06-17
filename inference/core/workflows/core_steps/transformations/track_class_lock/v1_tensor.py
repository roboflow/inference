"""Tensor-native sibling of track_class_lock/v1.py.

Loaded INSTEAD of v1.py when ENABLE_TENSOR_DATA_REPRESENTATION is set. Reuses the
exact same class name / block type@version (``TrackClassLockBlockV1``,
``roboflow_core/track_class_lock@v1``) so it is a drop-in swap in loader.py.

The underlying voting/lock logic is numpy/``sv.Detections``-based and inherently
STATEFUL across engine runs (per-video vote tallies and locks persist between
frames). Rather than re-implement that delicate state machine on torch tensors,
this block:

1. materialises the native ``inference_models.Detections`` input to an
   ``sv.Detections`` at the boundary (xyxy/class_id/confidence pulled from torch
   via ``.detach().cpu().numpy()``; tracker_id + class_name + detection_id carried
   from ``bboxes_metadata`` / ``image_metadata[CLASS_NAMES_KEY]`` into sv ``.data``
   so the existing voting/lock logic reads them unchanged);
2. runs the SAME lock/vote logic as v1 (the per-video state dict ``self._per_video_state``
   persists across runs, exactly like the numpy block — this is the persistence the
   test asserts);
3. re-packs the result back into a native ``inference_models.Detections``: the
   locked/relabelled ``class``, ``class_id``, ``confidence`` and a boolean
   ``class_locked`` flag are written into ``bboxes_metadata[i]``, and
   ``image_metadata[CLASS_NAMES_KEY]`` is rebuilt so any newly introduced class id
   (a relabelled class) resolves to a name downstream.

Keypoint input arrives as a ``(KeyPoints, Detections)`` tuple; the bbox component
drives the voting and the (possibly relabelled) tuple is returned so keypoints
survive. Instance-segmentation masks are carried through unchanged.

The voting/lock math, re-attachment logic and parameter validation are imported
from v1.py verbatim — only the input/output marshalling differs.
"""

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import List, Optional, Set, Tuple, Type, Union

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.tensor_native import (
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.transformations.track_class_lock.v1 import (
    MAX_TRACKED_VIDEOS,
    BlockManifest,
    _find_lock_to_inherit,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

OUTPUT_KEY: str = "tracked_detections"
CLASS_LOCKED_KEY: str = "class_locked"

# sentinel name used internally when a detection has no resolvable class name; the
# numpy block falls back to ``str(class_id)`` in this case, and so do we.
_CLASS_NAME_DATA_KEY = "class_name"

TensorNativeDetectionsLike = Union[Detections, InstanceDetections]
TensorNativeTrackInput = Union[
    Detections,
    InstanceDetections,
    Tuple[KeyPoints, Optional[Detections]],
]


def _resolve_class_names(detections: TensorNativeDetectionsLike) -> List[Optional[str]]:
    """Per-box class name, mirroring how the numpy block reads
    ``detections.data["class_name"]``.

    Native predictions carry the per-box class name on ``bboxes_metadata[i]["class"]``
    (the ``CLASS_NAME_KEY`` convention) and/or a ``class_id -> name`` map on
    ``image_metadata[CLASS_NAMES_KEY]``. We prefer the per-box value, fall back to the
    image map keyed by ``class_id``, and finally to ``None`` (the numpy block then
    uses ``str(class_id)`` for voting).
    """
    n = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata or [{} for _ in range(n)]
    class_names_map = (detections.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    class_id = detections.class_id.detach().to("cpu").numpy()
    resolved: List[Optional[str]] = []
    for index in range(n):
        per_box = (bboxes_metadata[index] or {}).get(CLASS_NAME_KEY)
        if per_box is not None:
            resolved.append(str(per_box))
            continue
        mapped = class_names_map.get(int(class_id[index]))
        resolved.append(str(mapped) if mapped is not None else None)
    return resolved


def _to_supervision_boundary(detections: TensorNativeDetectionsLike) -> sv.Detections:
    """Materialise a native ``Detections``/``InstanceDetections`` to ``sv.Detections``
    carrying tracker_id + class_name + detection_id in ``.data`` so the v1 voting
    logic runs unchanged. Masks are intentionally dropped — the voting/lock logic
    only uses xyxy/class_id/confidence/tracker_id/class_name.
    """
    n = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata or [{} for _ in range(n)]
    xyxy = detections.xyxy.detach().to("cpu").numpy().astype(np.float64)
    class_id = detections.class_id.detach().to("cpu").numpy()
    confidence = detections.confidence.detach().to("cpu").numpy()
    tracker_ids = np.array(
        [(bboxes_metadata[i] or {}).get(TRACKER_ID_KEY, -1) for i in range(n)],
        dtype=np.int64,
    )
    detection_ids = np.array(
        [(bboxes_metadata[i] or {}).get(DETECTION_ID_KEY, "") for i in range(n)],
        dtype=object,
    )
    class_names = np.array(
        [name if name is not None else "" for name in _resolve_class_names(detections)],
        dtype=object,
    )
    data = {
        _CLASS_NAME_DATA_KEY: class_names,
        DETECTION_ID_KEY: detection_ids,
    }
    return sv.Detections(
        xyxy=xyxy if n else np.empty((0, 4), dtype=np.float64),
        class_id=class_id if n else np.empty((0,), dtype=int),
        confidence=confidence if n else np.empty((0,), dtype=np.float32),
        tracker_id=tracker_ids if n else np.empty((0,), dtype=np.int64),
        data=data,
    )


def _vote_and_lock(
    self: "TrackClassLockBlockV1",
    image: WorkflowImageData,
    dets: sv.Detections,
    min_votes: int,
    vote_confidence: float,
    lead_margin: int,
    switch_after: int,
    state_ttl: int,
    reattach_window: int,
    reattach_iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the v1 majority-vote / lock state machine over a boundary
    ``sv.Detections``. Mutates ``dets.class_id``/``dets.confidence`` and the
    ``class_name`` column in place (matching v1) and returns
    ``(class_names, class_id, locked_flags)`` for repacking into the native output.

    This is the EXACT logic from ``TrackClassLockBlockV1.run`` in v1.py, lifted so
    the cross-run ``self._per_video_state`` persistence is identical.
    """
    video_id = image.video_metadata.video_identifier
    video_state = self._per_video_state.setdefault(
        video_id, {"tracks": {}, "frame": 0}
    )
    self._per_video_state.move_to_end(video_id)
    while len(self._per_video_state) > MAX_TRACKED_VIDEOS:
        self._per_video_state.popitem(last=False)
    video_state["frame"] += 1
    frame = video_state["frame"]
    tracks = video_state["tracks"]

    n = len(dets)
    locked_flags = np.zeros(n, dtype=bool)

    if dets.confidence is None and n > 0:
        dets.confidence = np.ones(n, dtype=np.float32)

    class_names = dets.data.get(_CLASS_NAME_DATA_KEY)
    if class_names is not None:
        class_names = np.asarray(class_names).astype(object)
        dets.data[_CLASS_NAME_DATA_KEY] = class_names
    active_tids: Set[int] = set()
    if dets.tracker_id is not None:
        active_tids = {int(t) for t in dets.tracker_id if t is not None and int(t) >= 0}
    for i in range(n):
        tid = dets.tracker_id[i]
        if tid is None or int(tid) < 0:
            continue
        tid = int(tid)
        if tid not in tracks:
            inherited = _find_lock_to_inherit(
                tracks=tracks,
                xyxy=dets.xyxy[i],
                frame=frame,
                reattach_window=reattach_window,
                reattach_iou=reattach_iou,
                active_tids=active_tids,
            )
            if inherited is not None:
                tracks[tid] = tracks.pop(inherited)
        st = tracks.setdefault(
            tid,
            {
                "votes": defaultdict(int),
                "conf_sum": defaultdict(float),
                "class_ids": {},
                "locked": None,
                "last_seen": frame,
                "challenger": None,
                "streak": 0,
                "streak_conf": 0.0,
                "last_xyxy": None,
            },
        )
        st["last_seen"] = frame
        st["last_xyxy"] = np.array(dets.xyxy[i], copy=True)

        if class_names is not None and str(class_names[i]) != "":
            cname = str(class_names[i])
        elif dets.class_id is not None:
            cname = str(dets.class_id[i])
        else:
            continue
        conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
        qualifying = conf >= vote_confidence

        if st["locked"] is None:
            if qualifying:
                st["votes"][cname] += 1
                st["conf_sum"][cname] += conf
                if dets.class_id is not None:
                    st["class_ids"][cname] = int(dets.class_id[i])
            if st["votes"]:
                ranked = sorted(
                    st["votes"].items(), key=lambda kv: kv[1], reverse=True
                )
                top_c, top_v = ranked[0]
                runner_v = ranked[1][1] if len(ranked) > 1 else 0
                if top_v >= min_votes and top_v - runner_v >= lead_margin:
                    st["locked"] = top_c
        else:
            if qualifying and cname != st["locked"]:
                if cname == st["challenger"]:
                    st["streak"] += 1
                    st["streak_conf"] += conf
                else:
                    st["challenger"] = cname
                    st["streak"] = 1
                    st["streak_conf"] = conf
                    if dets.class_id is not None:
                        st["class_ids"][cname] = int(dets.class_id[i])
                if st["streak"] >= switch_after:
                    new = st["challenger"]
                    st["locked"] = new
                    st["votes"] = defaultdict(int, {new: st["streak"]})
                    st["conf_sum"] = defaultdict(float, {new: st["streak_conf"]})
                    st["challenger"], st["streak"], st["streak_conf"] = None, 0, 0.0
            else:
                st["challenger"], st["streak"], st["streak_conf"] = None, 0, 0.0

        if st["locked"] is not None:
            lc = st["locked"]
            if class_names is not None:
                class_names[i] = lc
            if dets.class_id is not None and lc in st["class_ids"]:
                dets.class_id[i] = st["class_ids"][lc]
            denom = max(st["votes"][lc], 1)
            dets.confidence[i] = min(1.0, st["conf_sum"][lc] / denom)
            locked_flags[i] = True

    stale = [t for t, st in tracks.items() if frame - st["last_seen"] > state_ttl]
    for t in stale:
        del tracks[t]

    out_class_names = (
        np.asarray(class_names) if class_names is not None else np.empty((n,), dtype=object)
    )
    out_class_id = (
        dets.class_id.copy()
        if dets.class_id is not None
        else np.empty((n,), dtype=int)
    )
    return out_class_names, out_class_id, locked_flags


def _repack_native(
    detections: TensorNativeDetectionsLike,
    class_names: np.ndarray,
    class_id: np.ndarray,
    confidence: np.ndarray,
    locked_flags: np.ndarray,
) -> TensorNativeDetectionsLike:
    """Write the locked/relabelled class, confidence and ``class_locked`` flag back
    into a COPY of the native prediction. ``bboxes_metadata[i]`` receives the new
    ``class`` name, ``class_locked`` flag (and ``class_id``/``confidence`` mirrors);
    ``image_metadata[CLASS_NAMES_KEY]`` is rebuilt so any relabelled class id resolves
    to a name. The xyxy/mask tensors are untouched; class_id/confidence tensors are
    rebuilt from the (possibly mutated) numpy values on the original device/dtype.
    """
    result = deepcopy(detections)
    n = int(result.xyxy.shape[0])
    new_class_id = result.class_id.new_tensor(class_id.tolist()) if n else result.class_id
    new_confidence = (
        result.confidence.new_tensor(confidence.tolist()) if n else result.confidence
    )
    result.class_id = new_class_id
    result.confidence = new_confidence

    class_names_map = dict((result.image_metadata or {}).get(CLASS_NAMES_KEY) or {})
    bboxes_metadata = result.bboxes_metadata or [{} for _ in range(n)]
    new_bboxes_metadata: List[dict] = []
    for index in range(n):
        entry = dict(bboxes_metadata[index] or {})
        cid = int(class_id[index])
        name = class_names[index] if index < len(class_names) else None
        if name is not None and str(name) != "":
            entry[CLASS_NAME_KEY] = str(name)
            class_names_map[cid] = str(name)
        entry[CLASS_LOCKED_KEY] = bool(locked_flags[index])
        new_bboxes_metadata.append(entry)
    result.bboxes_metadata = new_bboxes_metadata if n else None

    image_metadata = dict(result.image_metadata or {})
    image_metadata[CLASS_NAMES_KEY] = class_names_map
    result.image_metadata = image_metadata
    return result


class TrackClassLockBlockV1(WorkflowBlock):
    def __init__(self):
        self._per_video_state: "OrderedDict[str, dict]" = OrderedDict()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: TensorNativeTrackInput,
        min_votes: int,
        vote_confidence: float,
        lead_margin: int,
        switch_after: int,
        state_ttl: int,
        reattach_window: int,
        reattach_iou: float,
    ) -> BlockResult:
        # selector-provided params bypass the manifest's pydantic Field bounds,
        # so the same constraints are re-checked here at runtime (verbatim from v1)
        if min_votes < 1:
            raise ValueError(f"`min_votes` must be >= 1, got {min_votes}")
        if not 0.0 <= vote_confidence <= 1.0:
            raise ValueError(
                f"`vote_confidence` must be within [0.0, 1.0], got {vote_confidence}"
            )
        if lead_margin < 0:
            raise ValueError(f"`lead_margin` must be >= 0, got {lead_margin}")
        if switch_after < 1:
            raise ValueError(f"`switch_after` must be >= 1, got {switch_after}")
        if state_ttl < 1:
            raise ValueError(f"`state_ttl` must be >= 1, got {state_ttl}")
        if reattach_window < 0:
            raise ValueError(f"`reattach_window` must be >= 0, got {reattach_window}")
        if not 0.0 <= reattach_iou <= 1.0:
            raise ValueError(
                f"`reattach_iou` must be within [0.0, 1.0], got {reattach_iou}"
            )
        if reattach_window > state_ttl:
            raise ValueError(
                f"`reattach_window` ({reattach_window}) must not exceed `state_ttl` "
                f"({state_ttl}) - lost tracks are purged after `state_ttl` frames, so "
                "re-attachment beyond that point can never happen."
            )

        key_points, bbox_detections = split_key_point_prediction(detections)

        n = int(bbox_detections.xyxy.shape[0])
        if n > 0 and not _has_tracker_ids(bbox_detections):
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires "
                "detections to be tracked"
            )

        sv_dets = _to_supervision_boundary(bbox_detections)
        class_names, class_id, locked_flags = _vote_and_lock(
            self,
            image=image,
            dets=sv_dets,
            min_votes=min_votes,
            vote_confidence=vote_confidence,
            lead_margin=lead_margin,
            switch_after=switch_after,
            state_ttl=state_ttl,
            reattach_window=reattach_window,
            reattach_iou=reattach_iou,
        )
        confidence = (
            sv_dets.confidence
            if sv_dets.confidence is not None
            else np.ones(n, dtype=np.float32)
        )
        native_output = _repack_native(
            detections=bbox_detections,
            class_names=class_names,
            class_id=class_id,
            confidence=confidence,
            locked_flags=locked_flags,
        )
        if key_points is not None:
            return {OUTPUT_KEY: (key_points, native_output)}
        return {OUTPUT_KEY: native_output}


def _has_tracker_ids(detections: TensorNativeDetectionsLike) -> bool:
    """Native equivalent of ``detections.tracker_id is not None``: at least one box
    carries a ``tracker_id`` in ``bboxes_metadata``."""
    if detections.bboxes_metadata is None:
        return False
    return any(
        (entry or {}).get(TRACKER_ID_KEY) is not None
        for entry in detections.bboxes_metadata
    )

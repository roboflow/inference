from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Set, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    RuntimeRestriction,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"
LONG_DESCRIPTION = """
Lock the class label of each tracked object by majority voting, eliminating class
flicker in video workflows where a model alternates between similar classes for the
same physical object.

## How This Block Works

This block maintains per-track voting state, keyed by the video_identifier embedded
in the image's video metadata:

1. Pre-lock, every qualifying frame (confidence >= vote_confidence) counts as a vote
   for the predicted class. A class becomes locked once it collects min_votes votes
   AND leads the runner-up class by at least lead_margin votes.
2. Post-lock, the locked class is written into every subsequent detection of that
   track. Reported confidence is the running mean of counted votes (clamped to 1.0).
3. A locked class can only change after switch_after CONSECUTIVE qualifying frames
   of the same challenger class. Challenger evidence is streak-scoped: both the
   streak counter and its confidence sum reset whenever the streak breaks, and on a
   successful switch the new class's tallies are seeded from the streak values only,
   so reported confidence never exceeds 1.0.
4. When a NEW tracker id appears where a locked track recently disappeared (within
   reattach_window frames, bounding box IoU >= reattach_iou), the new track inherits
   the lost track's lock and voting state. This makes locks survive tracker id
   switches caused by short detection gaps or occlusions. Only locked tracks are
   inherited, and a track still present in the current frame is never inherited.
   Set reattach_window to 0 to disable re-attachment.
5. State for tracks unseen for state_ttl frames is purged.

Each detection is annotated with a boolean `class_locked` flag in detections.data.

## Requirements

Detections must carry tracker_id (wire this block after a tracking block such as
Byte Tracker). The image's video_metadata is used to maintain separate state per
video stream.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Track Class Lock",
            "version": "v1",
            "short_description": "Lock the class of each tracked object by majority voting to eliminate class flicker.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "fas fa-lock",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/track_class_lock@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image with embedded video metadata. The video_metadata contains video_identifier used to maintain separate voting state for different videos.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block.",
        examples=["$steps.byte_tracker.tracked_detections"],
    )
    min_votes: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=10,
        description="Cumulative qualifying votes a class needs before the initial lock is acquired. Higher values delay locking but make the initial decision more reliable.",
        examples=[10, "$inputs.min_votes"],
    )
    vote_confidence: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.8,
        description="Minimum prediction confidence for a frame to count, both for pre-lock votes and post-lock challenger streaks. Frames below this threshold are ignored.",
        examples=[0.8, "$inputs.vote_confidence"],
    )
    lead_margin: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=3,
        description="Number of votes by which the top class must lead the runner-up before locking. Prevents premature locks when two classes are contested.",
        examples=[3, "$inputs.lead_margin"],
    )
    switch_after: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=15,
        description="Number of CONSECUTIVE qualifying frames of the same challenger class required to change an existing lock. Any interruption resets the streak.",
        examples=[15, "$inputs.switch_after"],
    )
    state_ttl: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=300,
        description="Number of frames after which state of unseen tracks is purged.",
        examples=[300, "$inputs.state_ttl"],
    )
    reattach_window: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=30,
        description="When a NEW tracker id appears where a locked track disappeared within this many frames, the new track inherits the lost track's lock and votes. Bridges tracker id switches caused by short detection gaps. Set to 0 to disable re-attachment.",
        examples=[30, "$inputs.reattach_window"],
    )
    reattach_iou: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.3,
        description="Minimum IoU between a new detection's bounding box and a recently lost locked track's last known bounding box for the lock to be inherited. Higher values require the object to reappear closer to where it vanished.",
        examples=[0.3, "$inputs.reattach_iou"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]


class TrackClassLockBlockV1(WorkflowBlock):
    def __init__(self):
        self._per_video_state: Dict[str, dict] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        min_votes: int,
        vote_confidence: float,
        lead_margin: int,
        switch_after: int,
        state_ttl: int,
        reattach_window: int,
        reattach_iou: float,
    ) -> BlockResult:
        if len(detections) > 0 and detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        video_state = self._per_video_state.setdefault(
            image.video_metadata.video_identifier, {"tracks": {}, "frame": 0}
        )
        video_state["frame"] += 1
        frame = video_state["frame"]
        tracks = video_state["tracks"]

        dets = deepcopy(detections)
        n = len(dets)
        locked_flags = np.zeros(n, dtype=bool)

        class_names = dets.data.get("class_name")
        if class_names is not None:
            # fixed-width numpy string arrays silently truncate longer names
            # on assignment - switch to object dtype before relabelling
            class_names = np.asarray(class_names).astype(object)
            dets.data["class_name"] = class_names
        active_tids: Set[int] = set()
        if dets.tracker_id is not None:
            active_tids = {
                int(t) for t in dets.tracker_id if t is not None and int(t) >= 0
            }
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

            cname = (
                str(class_names[i])
                if class_names is not None
                else str(dets.class_id[i])
            )
            conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
            qualifying = conf >= vote_confidence

            if st["locked"] is None:
                # pre-lock: cumulative voting (any order)
                if qualifying:
                    st["votes"][cname] += 1
                    st["conf_sum"][cname] += conf
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
                # post-lock: strict consecutive challenge; challenger evidence
                # is streak-scoped only, so a switch can never produce
                # confidence above 1.0
                if qualifying and cname != st["locked"]:
                    if cname == st["challenger"]:
                        st["streak"] += 1
                        st["streak_conf"] += conf
                    else:
                        st["challenger"] = cname
                        st["streak"] = 1
                        st["streak_conf"] = conf
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
                dets.class_id[i] = st["class_ids"][lc]
                denom = max(st["votes"][lc], 1)
                dets.confidence[i] = min(1.0, st["conf_sum"][lc] / denom)
                locked_flags[i] = True

        dets.data["class_locked"] = locked_flags

        stale = [t for t, st in tracks.items() if frame - st["last_seen"] > state_ttl]
        for t in stale:
            del tracks[t]

        return {OUTPUT_KEY: dets}


def _find_lock_to_inherit(
    tracks: Dict[int, dict],
    xyxy: np.ndarray,
    frame: int,
    reattach_window: int,
    reattach_iou: float,
    active_tids: Set[int],
) -> Optional[int]:
    best_tid: Optional[int] = None
    best_iou = 0.0
    for tid, st in tracks.items():
        if tid in active_tids or st["locked"] is None or st["last_xyxy"] is None:
            continue
        gap = frame - st["last_seen"]
        if gap < 1 or gap > reattach_window:
            continue
        iou = _box_iou(xyxy, st["last_xyxy"])
        if iou >= reattach_iou and iou > best_iou:
            best_tid, best_iou = tid, iou
    return best_tid


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = float(inter_w * inter_h)
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union

"""``roboflow_core/track_class_lock@v1``: lock each track's class by majority vote.

Port of V1 ``roboflow_core/track_class_lock@v1``; its tensor variant is the
reference. The V1 ``detections`` input is ``predictions`` here. Per source
and tracker id, one call is one frame::

    before lock   a frame with confidence >= vote_confidence votes for its
                  class; lock when top votes >= min_votes and
                  top - runner-up >= lead_margin
    after lock    write the locked class, its class id and the mean
                  confidence of its counted votes (at most 1.0);
                  switch_after consecutive qualifying frames of one other
                  class move the lock
    new id        inherits a lost, locked track whose last box has
                  IoU >= reattach_iou, lost 1..reattach_window frames ago
    unseen        purged after state_ttl frames

A row votes under its class name (decision 025: its own ``"class"``, else
``class_names[class_id]``). A row without a name, or with ``""``, votes under
``str(class_id)``; locking such a key writes that string as the row's name,
as V1 does. Rows without ``tracker_id`` pass through unlocked; a non-empty
prediction where no row has one is an error.

Frames are counted per source. An empty prediction is a frame and ages
tracks; a filtered or missing prediction skips the call and is not counted.
Every parameter applies from the call that passes it.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine). ``max_sources`` (default 256) bounds the
sources; the first frame of one source too many raises
``TrackerCapacityError`` before any state changes. Sources are never
evicted; a processing reset replaces the block and drops them all.

The output keeps the input's family, rows, row order, boxes, masks (a
segmentation view stays low resolution), keypoints and row identity. Only
``class_id``, ``confidence``, the row ``"class"`` and ``"class_locked"``
change, and ``image_metadata["class_names"]`` maps each written id to its
name. Stale host mirrors of class id and confidence are dropped.

CPU boundary: the vote state machine is host Python over host copies of
``xyxy``, ``class_id`` and ``confidence`` (one device-to-host copy each), and
the new class ids and confidences return in one host-to-device copy each.

Differences from V1:

* V1 evicted the least recently seen of 256 videos; V2 raises
  ``TrackerCapacityError`` instead of silently forgetting a live source.
* A segmentation view is accepted and stays lazy.
* A row's non-string ``"class"`` value does not name it (decision 025).
* An empty prediction is returned as given; V1 added an empty
  ``class_names`` map.
"""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import Field, StrictFloat, StrictInt, model_validator
from roboflow_workflows.v2.blocks.query.detections import (
    replace_boxes,
    without_host_mirror,
)
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_MAX_SOURCES,
    TRACKED_KINDS,
    TrackedPrediction,
    TrackerCapacityError,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    class_names_by_id,
    row_class_names,
)
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections


__all__ = [
    "CLASS_LOCKED_KEY",
    "DEFAULT_SOURCE",
    "MAX_STATE_TTL",
    "MAX_TRACKS_PER_SOURCE",
    "TrackClassLockBlock",
]

CLASS_LOCKED_KEY = "class_locked"

DEFAULT_SOURCE = "default"

MAX_STATE_TTL = 1_000_000

MAX_TRACKS_PER_SOURCE = 4096

MAX_REATTACH_CANDIDATES = 256

MAX_CLASSES_PER_TRACK = 256

TRACKER_ID_KEY = "tracker_id"

@dataclass
class _Track:
    """Voting state of one tracker id."""

    last_seen: int
    last_xyxy: np.ndarray
    votes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidence_sums: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    class_ids: Dict[str, int] = field(default_factory=dict)
    locked: Optional[str] = None
    challenger: Optional[str] = None
    streak: int = 0
    streak_confidence: float = 0.0

    def end_challenge(self) -> None:
        self.challenger, self.streak, self.streak_confidence = None, 0, 0.0

@dataclass
class _SourceTracks:
    """Frame counter and tracks of one source."""

    frame: int = 0
    tracks: Dict[int, _Track] = field(default_factory=dict)

@dataclass(frozen=True)
class _Rules:
    min_votes: int
    vote_confidence: float
    lead_margin: int
    switch_after: int

class TrackClassLockBlock(Block):
    """Replace each tracked row's class by its track's majority-vote lock.

    Example with ``min_votes=2, lead_margin=1, switch_after=2`` and every
    confidence 0.9, one track::

        frame        1      2      3      4      5      6
        model says   car    car    truck  car    truck  truck
        output       car    car*   car*   car*   car*   truck*   (* locked)

    Frame 2 locks ``car`` (2 votes, lead 2). Frame 3 starts a ``truck``
    streak, frame 4 breaks it, frames 5-6 complete it and move the lock.
    """

    type = "roboflow_core/track_class_lock@v1"
    metadata = {
        "name": "Track Class Lock",
        "port_of": "roboflow_core/track_class_lock@v1",
        "section": "video",
    }
    outputs = {
        "tracked_detections": Output(
            *TRACKED_KINDS,
            source="predictions",
            description=(
                "The input rows in input order, with locked class, class id "
                "and confidence, and a boolean class_locked in "
                "bboxes_metadata."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Frame whose video_metadata names the source."
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        min_votes: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            ge=1,
            description="Qualifying votes a class needs before the first lock.",
            examples=[10, "$inputs.min_votes"],
        )
        vote_confidence: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.8,
            ge=0,
            le=1,
            description=(
                "Minimum confidence for a frame to count, both as a vote and "
                "in a challenger streak."
            ),
            examples=[0.8, "$inputs.vote_confidence"],
        )
        lead_margin: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            ge=0,
            description="Votes by which the top class must lead the runner-up to lock.",
            examples=[3, "$inputs.lead_margin"],
        )
        switch_after: StrictInt | Ref(INTEGER_KIND) = Field(
            default=15,
            ge=1,
            description=(
                "Consecutive qualifying frames of one other class that move a "
                "lock; any other frame resets the streak."
            ),
            examples=[15, "$inputs.switch_after"],
        )
        state_ttl: StrictInt | Ref(INTEGER_KIND) = Field(
            default=300,
            ge=1,
            le=MAX_STATE_TTL,
            description="Frames after which the state of an unseen track is purged.",
            examples=[300, "$inputs.state_ttl"],
        )
        reattach_window: StrictInt | Ref(INTEGER_KIND) = Field(
            default=30,
            ge=0,
            description=(
                "A new tracker id inherits a locked track lost at most this "
                "many frames ago; 0 disables it. At most state_ttl."
            ),
            examples=[30, "$inputs.reattach_window"],
        )
        reattach_iou: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.3,
            ge=0,
            le=1,
            description=(
                "Minimum IoU between the new row's box and the lost track's "
                "last box for the lock to be inherited."
            ),
            examples=[0.3, "$inputs.reattach_iou"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with voting state. A frame of one more source "
                "fails with TrackerCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

        @model_validator(mode="after")
        def _reattach_within_ttl(self) -> "TrackClassLockBlock.Params":
            # Selected values are checked again in run().
            if isinstance(self.state_ttl, int) and isinstance(
                self.reattach_window, int
            ):
                _check_reattach_window(self.reattach_window, self.state_ttl)

            return self

    def __init__(self) -> None:
        self._sources: Dict[str, _SourceTracks] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        min_votes: int,
        vote_confidence: float,
        lead_margin: int,
        switch_after: int,
        state_ttl: int,
        reattach_window: int,
        reattach_iou: float,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Vote with one frame of this source and write the locked classes.

        Args:
            image: Frame naming the source.
            predictions: Tracked rows of this frame; not modified.
            min_votes: Votes needed for the first lock.
            vote_confidence: Minimum confidence of a counted frame.
            lead_margin: Lead over the runner-up needed for the first lock.
            switch_after: Consecutive challenger frames that move a lock.
            state_ttl: Frames an unseen track's state is kept.
            reattach_window: Frames a lost locked track can be inherited.
            reattach_iou: Box IoU needed to inherit a lost track.
            max_sources: Most sources with state.

        Returns:
            ``tracked_detections``: the input family with locked classes.

        Raises:
            ValueError: When ``reattach_window`` exceeds ``state_ttl``, a
                keypoint prediction has no boxes, or rows exist but none has
                a ``tracker_id``.
            TrackerCapacityError: When this is the first frame of a source
                and ``max_sources`` sources already have state.
        """
        _check_reattach_window(reattach_window, state_ttl)
        boxes = _boxes_of(predictions)
        rows_metadata = boxes.bboxes_metadata or [{} for _ in range(len(boxes))]
        tracker_ids = [_tracker_id(row) for row in rows_metadata]
        if tracker_ids and all(tracker_id is None for tracker_id in tracker_ids):
            raise ValueError(
                f"{self.type} needs tracked predictions: no row has a "
                "tracker_id in bboxes_metadata; connect a tracker first"
            )

        source = self._source_state(image, max_sources=max_sources)
        source.frame += 1
        rules = _Rules(min_votes, vote_confidence, lead_margin, switch_after)
        if not tracker_ids:
            _forget_stale(source, state_ttl=state_ttl)
            return {"tracked_detections": predictions}

        xyxy = boxes.xyxy.detach().cpu().double().numpy()
        class_ids = boxes.class_id.detach().cpu().tolist()
        confidences = boxes.confidence.detach().cpu().tolist()
        names = row_class_names(
            class_ids,
            row_metadata=rows_metadata,
            names_by_id=class_names_by_id(boxes.image_metadata),
        )
        names = [str(name) if name not in (None, "") else "" for name in names]
        locked = _vote_frame(
            source,
            tracker_ids=tracker_ids,
            xyxy=xyxy,
            names=names,
            class_ids=class_ids,
            confidences=confidences,
            rules=rules,
            reattach_window=reattach_window,
            reattach_iou=reattach_iou,
        )
        _forget_stale(source, state_ttl=state_ttl)

        relabelled = _relabelled_boxes(
            boxes,
            rows_metadata=rows_metadata,
            names=names,
            class_ids=class_ids,
            confidences=confidences,
            locked=locked,
        )
        result = {"tracked_detections": replace_boxes(predictions, relabelled)}

        return result

    def _source_state(self, image: ImageData, *, max_sources: int) -> _SourceTracks:
        source = _source_key(self, image)
        state = self._sources.get(source)
        if state is not None:
            return state

        if len(self._sources) >= max_sources:
            raise TrackerCapacityError(
                f"{self.type}: source {source!r} would be source "
                f"{len(self._sources) + 1} of max_sources={max_sources}; "
                "raise max_sources or reset processing to drop ended sources"
            )

        state = _SourceTracks()
        self._sources[source] = state

        return state

def _vote_frame(
    source: _SourceTracks,
    *,
    tracker_ids: List[Optional[int]],
    xyxy: np.ndarray,
    names: List[str],
    class_ids: List[int],
    confidences: List[float],
    rules: _Rules,
    reattach_window: int,
    reattach_iou: float,
) -> List[bool]:
    """Update every row's track and relabel locked rows in the host lists.

    ``names``, ``class_ids`` and ``confidences`` are rewritten in place for
    locked rows. Returns the per-row ``class_locked`` flags.
    """
    frame, tracks = source.frame, source.tracks
    active = {tracker_id for tracker_id in tracker_ids if tracker_id is not None}
    # Tracks created in this frame have gap 0 and never qualify, so the
    # candidates are fixed for the whole frame (V1).
    candidates = _inheritance_candidates(
        tracks, frame=frame, reattach_window=reattach_window, active=active
    )
    locked = [False] * len(tracker_ids)
    for row, tracker_id in enumerate(tracker_ids):
        if tracker_id is None:
            continue

        if tracker_id not in tracks and candidates:
            inherited = _best_candidate(
                candidates, xyxy=xyxy[row], reattach_iou=reattach_iou
            )
            if inherited is not None:
                lost_id = candidates.pop(inherited)[0]
                tracks[tracker_id] = tracks.pop(lost_id)

        track = tracks.setdefault(
            tracker_id, _Track(last_seen=frame, last_xyxy=xyxy[row].copy())
        )
        track.last_seen = frame
        track.last_xyxy = xyxy[row].copy()
        vote_key = names[row] or str(class_ids[row])
        _update_track(
            track,
            name=vote_key,
            class_id=class_ids[row],
            confidence=confidences[row],
            rules=rules,
        )
        if track.locked is None:
            continue

        names[row] = track.locked
        class_ids[row] = track.class_ids.get(track.locked, class_ids[row])
        votes = max(track.votes.get(track.locked, 0), 1)
        confidences[row] = min(1.0, track.confidence_sums[track.locked] / votes)
        locked[row] = True

    return locked

def _update_track(
    track: _Track, *, name: str, class_id: int, confidence: float, rules: _Rules
) -> None:
    qualifying = confidence >= rules.vote_confidence
    if track.locked is None:
        if qualifying and _class_admissible(track, name):
            track.votes[name] += 1
            track.confidence_sums[name] += confidence
            _record_class_id(track, name, class_id)
        if track.votes:
            # Only the top two tallies matter; ties keep first-voted order.
            ranked = heapq.nlargest(2, track.votes.items(), key=lambda item: item[1])
            top_name, top_votes = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0
            if (
                top_votes >= rules.min_votes
                and top_votes - runner_up >= rules.lead_margin
            ):
                track.locked = top_name
        return

    if not qualifying or name == track.locked:
        track.end_challenge()
        return

    if name == track.challenger:
        track.streak += 1
        track.streak_confidence += confidence
    else:
        track.challenger, track.streak = name, 1
        track.streak_confidence = confidence
        _record_class_id(track, name, class_id)
    if track.streak < rules.switch_after:
        return

    # Seeded from the streak only, so the reported confidence stays <= 1.0.
    track.locked = track.challenger
    track.votes = defaultdict(int, {track.locked: track.streak})
    track.confidence_sums = defaultdict(float, {track.locked: track.streak_confidence})
    track.end_challenge()

def _class_admissible(track: _Track, name: str) -> bool:
    # A new class is admitted only below the per-track class cap (V1).
    return name in track.votes or len(track.votes) < MAX_CLASSES_PER_TRACK

def _record_class_id(track: _Track, name: str, class_id: int) -> None:
    # At the cap, drop ids no vote, lock or challenger refers to (V1).
    class_ids = track.class_ids
    if name not in class_ids and len(class_ids) >= MAX_CLASSES_PER_TRACK:
        referenced = set(track.votes) | {track.locked, track.challenger, name}
        for stale in [key for key in class_ids if key not in referenced]:
            del class_ids[stale]
    class_ids[name] = class_id

def _inheritance_candidates(
    tracks: Dict[int, _Track], *, frame: int, reattach_window: int, active: Set[int]
) -> List[Tuple[int, _Track]]:
    if reattach_window <= 0:
        return []

    candidates = [
        (tracker_id, track)
        for tracker_id, track in tracks.items()
        if tracker_id not in active
        and track.locked is not None
        and 1 <= frame - track.last_seen <= reattach_window
    ]
    if len(candidates) > MAX_REATTACH_CANDIDATES:
        candidates.sort(key=lambda candidate: candidate[1].last_seen, reverse=True)
        del candidates[MAX_REATTACH_CANDIDATES:]

    return candidates

def _best_candidate(
    candidates: List[Tuple[int, _Track]], *, xyxy: np.ndarray, reattach_iou: float
) -> Optional[int]:
    best, best_iou = None, 0.0
    for position, (_, track) in enumerate(candidates):
        iou = _box_iou(xyxy, track.last_xyxy)
        if iou >= reattach_iou and iou > best_iou:
            best, best_iou = position, iou

    return best

def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    inter_w = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = float(inter_w * inter_h)
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0

    return iou

def _forget_stale(source: _SourceTracks, *, state_ttl: int) -> None:
    tracks = source.tracks
    for tracker_id in [
        tracker_id
        for tracker_id, track in tracks.items()
        if source.frame - track.last_seen > state_ttl
    ]:
        del tracks[tracker_id]

    excess = len(tracks) - MAX_TRACKS_PER_SOURCE
    if excess > 0:
        oldest = sorted(tracks, key=lambda tracker_id: tracks[tracker_id].last_seen)
        for tracker_id in oldest[:excess]:
            del tracks[tracker_id]

def _relabelled_boxes(
    boxes: Detections,
    *,
    rows_metadata: List[dict],
    names: List[str],
    class_ids: List[int],
    confidences: List[float],
    locked: List[bool],
) -> Detections:
    names_map = dict(class_names_by_id(boxes.image_metadata) or {})
    relabelled_rows = without_host_mirror(rows_metadata)
    for row, name in enumerate(names):
        if name:
            relabelled_rows[row][CLASS_NAME_KEY] = name
            names_map[class_ids[row]] = name
        relabelled_rows[row][CLASS_LOCKED_KEY] = locked[row]

    relabelled = replace(
        boxes,
        class_id=boxes.class_id.new_tensor(class_ids),
        confidence=boxes.confidence.new_tensor(confidences),
        image_metadata={**(boxes.image_metadata or {}), CLASS_NAMES_KEY: names_map},
        bboxes_metadata=relabelled_rows,
    )

    return relabelled

def _boxes_of(prediction: TrackedPrediction) -> Detections | InstanceDetections:
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    if not isinstance(prediction, tuple):
        return prediction

    if prediction[1] is None:
        raise ValueError(
            "Track class lock needs boxes: this keypoint prediction has "
            "KeyPoints without Detections"
        )

    return prediction[1]

def _tracker_id(row: Optional[dict]) -> Optional[int]:
    # V1 skips rows without an id or with a negative one.
    tracker_id = (row or {}).get(TRACKER_ID_KEY)
    if tracker_id is None or int(tracker_id) < 0:
        return None

    return int(tracker_id)

def _source_key(block: Block, image: ImageData) -> str:
    metadata = image.video_metadata
    if metadata is not None and metadata.video_identifier:
        return metadata.video_identifier

    # The image's own context: another input without one must not drop it.
    # A plain run() has no context; errors in an engine call still propagate.
    try:
        context = block.execution_context
    except NoExecutionContextError:
        return DEFAULT_SOURCE

    sample = context.sample_at(context.indices[0], field="image")
    key = sample.source_id if sample is not None else DEFAULT_SOURCE

    return key

def _check_reattach_window(reattach_window: int, state_ttl: int) -> None:
    if reattach_window > state_ttl:
        raise ValueError(
            f"reattach_window ({reattach_window}) must not exceed state_ttl "
            f"({state_ttl}): lost tracks are purged after state_ttl frames, "
            "so they could never be re-attached"
        )

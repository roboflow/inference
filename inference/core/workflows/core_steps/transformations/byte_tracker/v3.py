from collections import deque
from typing import Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"
SHORT_DESCRIPTION = (
    "Track and update object positions across video frames using ByteTrack."
)
LONG_DESCRIPTION = """
Track objects across video frames using the ByteTrack algorithm to maintain consistent object identities, handle occlusions and temporary disappearances, associate detections with existing tracks, assign unique track IDs, categorize instances as new or previously seen, and enable object behavior analysis, movement tracking, first-appearance detection, and video analytics workflows.

## How This Block Works

This block maintains object tracking across sequential video frames by associating detections from each frame with existing tracks and creating new tracks for new objects, while also categorizing instances based on whether they've been seen before. The block:

1. Receives detection predictions for the current frame and an image with embedded video metadata
2. Extracts video metadata from the image (including frame rate and video identifier):
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts fps (frames per second) for tracker configuration
   - Extracts video_identifier to maintain separate tracking state for different videos
   - Handles missing fps gracefully (defaults to 0 and logs a warning instead of failing)
3. Initializes or retrieves a ByteTrack tracker for the video:
   - Creates a new tracker instance for each unique video (identified by video_identifier)
   - Stores trackers in memory to maintain tracking state across frames
   - Configures tracker with frame rate from metadata and user-specified parameters
   - Reuses existing tracker for subsequent frames of the same video
4. Initializes or retrieves an instance cache for the video:
   - Creates a cache to track which track IDs have been seen before
   - Maintains separate cache for each video using video_identifier
   - Configures cache size using instances_cache_size parameter
   - Uses FIFO (First-In-First-Out) strategy to manage cache capacity
5. Merges multiple detection batches if provided:
   - Combines detections from multiple sources into a single detection set
   - Ensures all detections are processed together for consistent tracking
6. Updates tracks using ByteTrack algorithm:
   - **Track Association**: Matches current frame detections to existing tracks using IoU (Intersection over Union) matching
   - **Track Activation**: Creates new tracks for detections with confidence above track_activation_threshold that don't match existing tracks
   - **Track Matching**: Associates detections to tracks when IoU exceeds minimum_matching_threshold
   - **Track Persistence**: Maintains tracks that don't have matches using lost_track_buffer to handle temporary occlusions
   - **Track Validation**: Only outputs tracks that have been present for at least minimum_consecutive_frames consecutive frames
7. Categorizes tracked instances as new or already seen:
   - For each tracked detection with a track_id, checks the instance cache
   - **New Instances**: Track IDs not found in cache are marked as new (first appearance)
   - **Already Seen Instances**: Track IDs found in cache are marked as already seen (reappearance)
   - Updates cache with new track IDs, managing cache size with FIFO eviction
8. Handles tracking challenges:
   - **Occlusions**: Maintains tracks when objects are temporarily hidden (using lost_track_buffer frames)
   - **Missed Detections**: Keeps tracks alive through frames with missing detections
   - **False Positives**: Filters out tracks that don't persist long enough (minimum_consecutive_frames)
   - **Track Fragmentation**: Reduces track splits by maintaining buffer for lost objects
9. Assigns unique track IDs to each object:
   - Each tracked object receives a consistent track_id that persists across frames
   - Track IDs are assigned when tracks are activated and maintained throughout the video
   - Enables tracking individual objects across the entire video sequence
10. Returns three sets of tracked detections:
    - **tracked_detections**: All tracked detections with track IDs (same as v2)
    - **new_instances**: Detections with track IDs that are appearing for the first time (each track ID appears only once when first generated)
    - **already_seen_instances**: Detections with track IDs that have been seen before (track IDs appear each time the tracker associates them with detections)

ByteTrack is an efficient multi-object tracking algorithm that performs tracking-by-detection, associating detections across frames without requiring appearance features. It uses a two-stage association strategy: first matching high-confidence detections to tracks, then matching low-confidence detections to remaining tracks and lost tracks. The algorithm maintains a buffer for lost tracks, allowing it to recover tracks when objects temporarily disappear due to occlusions or detection failures. The instance categorization feature enables detection of first appearances (new objects entering the scene) versus reappearances (objects returning after occlusion or leaving frame), which is useful for counting, behavior analysis, and event detection. The configurable parameters allow fine-tuning tracking behavior: track_activation_threshold controls when new tracks are created (higher = more conservative), lost_track_buffer controls occlusion handling (higher = better occlusion recovery), minimum_matching_threshold controls association quality (higher = stricter matching), minimum_consecutive_frames filters short-lived false tracks (higher = fewer false tracks), and instances_cache_size controls how many track IDs to remember for new/seen categorization (higher = longer memory).

## Common Use Cases

- **Video Analytics**: Track objects across video frames for behavior analysis and movement patterns (e.g., track people movement in videos, monitor vehicle paths, analyze object trajectories), enabling video analytics workflows
- **First Appearance Detection**: Identify new objects entering the scene for counting and event detection (e.g., detect new people entering area, identify new vehicles appearing, track first-time appearances), enabling new instance detection workflows
- **Traffic Monitoring**: Track vehicles and objects in traffic scenes with appearance tracking (e.g., track vehicles across frames, monitor vehicle paths, count unique vehicles with consistent IDs, detect new vehicles entering scene), enabling traffic monitoring workflows
- **Surveillance Systems**: Maintain object identities and detect new entries for security monitoring (e.g., track individuals in surveillance footage, detect new people entering area, monitor object movements, maintain object identities), enabling surveillance tracking workflows
- **Retail Analytics**: Track customers and products with entry detection for retail insights (e.g., track customer paths, detect new customers entering store, monitor shopping behavior, analyze foot traffic patterns), enabling retail analytics workflows
- **Object Counting**: Accurately count unique objects by tracking first appearances (e.g., count unique visitors by tracking new instances, count vehicles entering intersection, track unique object appearances), enabling accurate counting workflows

## Connecting to Other Blocks

This block receives an image with video metadata and detection predictions, and produces tracked_detections, new_instances, and already_seen_instances:

- **After object detection, instance segmentation, or keypoint detection blocks** to track detected objects across video frames (e.g., track detected objects in video, add track IDs to detections, maintain object identities across frames), enabling detection-to-tracking workflows
- **Using new_instances output** to detect and process first appearances (e.g., count new objects, trigger actions on first appearance, detect new entries, initialize tracking for new objects), enabling new instance detection workflows
- **Using already_seen_instances output** to process reappearances and returning objects (e.g., handle returning objects, process reappearances, filter for existing objects), enabling reappearance handling workflows
- **Before video analysis blocks** that require consistent object identities (e.g., analyze tracked object behavior, process object trajectories, work with tracked object data), enabling tracking-to-analysis workflows
- **Before visualization blocks** to display tracked objects with consistent colors or labels (e.g., visualize tracked objects, display track IDs, show object paths, highlight new instances), enabling tracking visualization workflows
- **Before logic blocks** like Continue If to make decisions based on track information or instance status (e.g., continue if object is new, filter based on track IDs, make decisions using tracking data, handle new vs returning objects), enabling tracking-based decision workflows

## Version Differences

**Enhanced from v2:**

- **Instance Categorization**: Adds two new outputs (`new_instances` and `already_seen_instances`) that categorize tracked objects based on whether their track IDs have been seen before, enabling first-appearance detection and reappearance tracking
- **Instance Cache**: Introduces an instance cache system that remembers previously seen track IDs across frames, allowing distinction between new objects entering the scene and objects reappearing after occlusion or leaving frame
- **Keypoint Detection Support**: Adds support for keypoint detection predictions in addition to object detection and instance segmentation, expanding tracking capabilities to keypoint-based detection models
- **Configurable Cache Size**: Adds `instances_cache_size` parameter to control how many track IDs are remembered in the cache, balancing memory usage with tracking history length
- **Enhanced Outputs**: Returns three outputs instead of one - `tracked_detections` (all tracked objects), `new_instances` (first appearances), and `already_seen_instances` (reappearances)

## Requirements

This block requires detection predictions (object detection, instance segmentation, or keypoint detection) and an image with embedded video metadata containing frame rate (fps) and video identifier information. The image's video_metadata should include a valid fps value for optimal tracking performance, though the block will continue with fps=0 if missing. The block maintains tracking state and instance cache across frames for each video, so it should be used in video workflows where frames are processed sequentially. For optimal tracking performance, detections should be provided consistently across frames. The algorithm works best with stable detection performance and handles temporary detection gaps through the lost_track_buffer mechanism. The instance cache maintains a history of seen track IDs with FIFO eviction when the cache size limit is reached.
"""


class ByteTrackerBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Byte Tracker",
            "version": "v3",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 0,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/byte_tracker@v3"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Input image containing embedded video metadata (fps and video_identifier) required for ByteTrack initialization and tracking state management. The block extracts video_metadata from the WorkflowImageData object. The fps value is used to configure the tracker, and the video_identifier is used to maintain separate tracking state and instance cache for different videos. If fps is missing or invalid, the block defaults to 0 and logs a warning but continues operation. If processing multiple videos, each video should have a unique video_identifier in its metadata to maintain separate tracking states and caches. The block maintains persistent trackers and instance caches across frames for each video using the video_identifier.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Detection predictions (object detection, instance segmentation, or keypoint detection) for the current video frame to be tracked. The block associates these detections with existing tracks or creates new tracks. Supports object detection, instance segmentation, and keypoint detection predictions. Detections should be provided for each frame in sequence to maintain consistent tracking. If multiple detection batches are provided, they will be merged before tracking. The detections must include bounding boxes and class names (and keypoints if keypoint detection). After tracking, the output will include the same detections enhanced with track_id information, allowing identification of the same object across frames.",
        examples=["$steps.object_detection_model.predictions"],
    )
    track_activation_threshold: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.25,
        description="Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common.",
        examples=[0.25, "$inputs.confidence"],
    )
    lost_track_buffer: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=30,
        description="Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios.",
        examples=[30, "$inputs.lost_track_buffer"],
    )
    minimum_matching_threshold: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.8,
        description="IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects.",
        examples=[0.8, "$inputs.min_matching_threshold"],
    )
    minimum_consecutive_frames: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=1,
        description="Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable.",
        examples=[1, "$inputs.min_consecutive_frames"],
    )
    instances_cache_size: int = Field(
        default=16384,
        description="Maximum number of track IDs to remember in the instance cache for determining if instances are new or already seen. Must be a positive integer. Default is 16384. The cache uses FIFO (First-In-First-Out) eviction - when the cache is full, the oldest track ID is removed to make room for new ones. Increasing this value (e.g., 32768-65536) maintains longer history of seen track IDs, allowing detection of reappearances after longer gaps, but uses more memory. Decreasing this value (e.g., 8192) reduces memory usage but may lose history of track IDs that appeared earlier, causing reappearing objects to be classified as new. Adjust based on video length and object reappearance patterns: use higher values for long videos or frequent reappearances, lower values for short videos or rare reappearances.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
            OutputDefinition(
                name="new_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="already_seen_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ByteTrackerBlockV3(WorkflowBlock):
    def __init__(
        self,
    ):
        self._trackers: Dict[str, sv.ByteTrack] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ByteTrackerBlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        instances_cache_size: int = 16384,
    ) -> BlockResult:
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 0
            logger.warning(
                f"Malformed fps in VideoMetadata, {self.__class__.__name__} requires fps in order to initialize ByteTrack"
            )
        if metadata.video_identifier not in self._trackers:
            self._trackers[metadata.video_identifier] = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                frame_rate=fps,
            )
        tracker = self._trackers[metadata.video_identifier]
        tracked_detections = tracker.update_with_detections(
            sv.Detections.merge(detections[i] for i in range(len(detections)))
        )
        if metadata.video_identifier not in self._per_video_cache:
            self._per_video_cache[metadata.video_identifier] = InstanceCache(
                size=instances_cache_size
            )
        cache = self._per_video_cache[metadata.video_identifier]
        not_seen_instances_mask, seen_instances_mask = [], []
        for tracker_id in tracked_detections.tracker_id.tolist():
            already_seen = cache.record_instance(tracker_id=tracker_id)
            not_seen_instances_mask.append(not already_seen)
            seen_instances_mask.append(already_seen)
        not_seen_instances_detections = tracked_detections[not_seen_instances_mask]
        already_seen_instances_detections = tracked_detections[seen_instances_mask]
        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": not_seen_instances_detections,
            "already_seen_instances": already_seen_instances_detections,
        }


class InstanceCache:

    def __init__(self, size: int):
        size = max(1, size)
        self._cache_inserts_track = deque(maxlen=size)
        self._cache = set()

    def record_instance(self, tracker_id: int) -> bool:
        in_cache = tracker_id in self._cache
        if not in_cache:
            self._cache_new_tracker_id(tracker_id=tracker_id)
        return in_cache

    def _cache_new_tracker_id(self, tracker_id: int) -> None:
        while len(self._cache) >= self._cache_inserts_track.maxlen:
            to_drop = self._cache_inserts_track.popleft()
            self._cache.remove(to_drop)
        self._cache_inserts_track.append(tracker_id)
        self._cache.add(tracker_id)

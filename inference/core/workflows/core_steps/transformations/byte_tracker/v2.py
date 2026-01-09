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
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
    WorkflowImageSelector,
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
Track objects across video frames using the ByteTrack algorithm to maintain consistent object identities, handle occlusions and temporary disappearances, associate detections with existing tracks, assign unique track IDs, and enable object behavior analysis, movement tracking, and video analytics workflows.

## How This Block Works

This block maintains object tracking across sequential video frames by associating detections from each frame with existing tracks and creating new tracks for new objects. The block:

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
4. Merges multiple detection batches if provided:
   - Combines detections from multiple sources into a single detection set
   - Ensures all detections are processed together for consistent tracking
5. Updates tracks using ByteTrack algorithm:
   - **Track Association**: Matches current frame detections to existing tracks using IoU (Intersection over Union) matching
   - **Track Activation**: Creates new tracks for detections with confidence above track_activation_threshold that don't match existing tracks
   - **Track Matching**: Associates detections to tracks when IoU exceeds minimum_matching_threshold
   - **Track Persistence**: Maintains tracks that don't have matches using lost_track_buffer to handle temporary occlusions
   - **Track Validation**: Only outputs tracks that have been present for at least minimum_consecutive_frames consecutive frames
6. Handles tracking challenges:
   - **Occlusions**: Maintains tracks when objects are temporarily hidden (using lost_track_buffer frames)
   - **Missed Detections**: Keeps tracks alive through frames with missing detections
   - **False Positives**: Filters out tracks that don't persist long enough (minimum_consecutive_frames)
   - **Track Fragmentation**: Reduces track splits by maintaining buffer for lost objects
7. Assigns unique track IDs to each object:
   - Each tracked object receives a consistent track_id that persists across frames
   - Track IDs are assigned when tracks are activated and maintained throughout the video
   - Enables tracking individual objects across the entire video sequence
8. Returns tracked detections with track IDs:
   - Outputs detection predictions enhanced with track_id information
   - Each detection includes its assigned track_id for identifying the same object across frames
   - Maintains all original detection properties (bounding boxes, confidence, class names) plus tracking information

ByteTrack is an efficient multi-object tracking algorithm that performs tracking-by-detection, associating detections across frames without requiring appearance features. It uses a two-stage association strategy: first matching high-confidence detections to tracks, then matching low-confidence detections to remaining tracks and lost tracks. The algorithm maintains a buffer for lost tracks, allowing it to recover tracks when objects temporarily disappear due to occlusions or detection failures. The configurable parameters allow fine-tuning tracking behavior: track_activation_threshold controls when new tracks are created (higher = more conservative), lost_track_buffer controls occlusion handling (higher = better occlusion recovery), minimum_matching_threshold controls association quality (higher = stricter matching), and minimum_consecutive_frames filters short-lived false tracks (higher = fewer false tracks).

## Common Use Cases

- **Video Analytics**: Track objects across video frames for behavior analysis and movement patterns (e.g., track people movement in videos, monitor vehicle paths, analyze object trajectories), enabling video analytics workflows
- **Traffic Monitoring**: Track vehicles and objects in traffic scenes for traffic analysis (e.g., track vehicles across frames, monitor vehicle paths, count vehicles with consistent IDs), enabling traffic monitoring workflows
- **Surveillance Systems**: Maintain object identities across video frames for security monitoring (e.g., track individuals in surveillance footage, monitor object movements, maintain object identities), enabling surveillance tracking workflows
- **Sports Analysis**: Track players and objects in sports videos for performance analysis (e.g., track player movements, analyze player trajectories, monitor ball positions), enabling sports analysis workflows
- **Retail Analytics**: Track customers and products across video frames for retail insights (e.g., track customer paths, monitor shopping behavior, analyze foot traffic patterns), enabling retail analytics workflows
- **Object Behavior Analysis**: Track objects to analyze their behavior and interactions over time (e.g., analyze object interactions, study movement patterns, track object relationships), enabling behavior analysis workflows

## Connecting to Other Blocks

This block receives an image with video metadata and detection predictions, and produces tracked_detections with track IDs:

- **After object detection or instance segmentation blocks** to track detected objects across video frames (e.g., track detected objects in video, add track IDs to detections, maintain object identities across frames), enabling detection-to-tracking workflows
- **Before video analysis blocks** that require consistent object identities (e.g., analyze tracked object behavior, process object trajectories, work with tracked object data), enabling tracking-to-analysis workflows
- **Before visualization blocks** to display tracked objects with consistent colors or labels (e.g., visualize tracked objects, display track IDs, show object paths), enabling tracking visualization workflows
- **Before logic blocks** like Continue If to make decisions based on track information (e.g., continue if object is tracked, filter based on track IDs, make decisions using tracking data), enabling tracking-based decision workflows
- **Before counting or aggregation blocks** to count tracked objects accurately (e.g., count unique tracked objects, aggregate track statistics, process track data), enabling tracking-to-counting workflows
- **In video processing pipelines** where object tracking is part of a larger video analysis workflow (e.g., track objects in video pipelines, maintain identities in processing chains, enable video analytics), enabling video tracking pipeline workflows

## Version Differences

**Enhanced from v1:**

- **Simplified Input**: Uses `image` input that contains embedded video metadata instead of requiring a separate `metadata` field, simplifying workflow connections and reducing input complexity
- **Graceful FPS Handling**: Handles missing or invalid fps values gracefully by defaulting to 0 and logging a warning instead of raising an error, making the block more resilient to incomplete metadata
- **Improved Integration**: Better integration with image-based workflows since video metadata is accessed directly from the image object rather than requiring separate metadata input

## Requirements

This block requires detection predictions (object detection or instance segmentation) and an image with embedded video metadata containing frame rate (fps) and video identifier information. The image's video_metadata should include a valid fps value for optimal tracking performance, though the block will continue with fps=0 if missing. The block maintains tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For optimal tracking performance, detections should be provided consistently across frames. The algorithm works best with stable detection performance and handles temporary detection gaps through the lost_track_buffer mechanism.
"""


class ByteTrackerBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Byte Tracker",
            "version": "v2",
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
    type: Literal["roboflow_core/byte_tracker@v2"]
    image: WorkflowImageSelector = Field(
        description="Input image containing embedded video metadata (fps and video_identifier) required for ByteTrack initialization and tracking state management. The block extracts video_metadata from the WorkflowImageData object. The fps value is used to configure the tracker, and the video_identifier is used to maintain separate tracking state for different videos. If fps is missing or invalid, the block defaults to 0 and logs a warning but continues operation. If processing multiple videos, each video should have a unique video_identifier in its metadata to maintain separate tracking states. The block maintains persistent trackers across frames for each video using the video_identifier. This version simplifies input by embedding metadata in the image object rather than requiring a separate metadata field.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Detection predictions (object detection or instance segmentation) for the current video frame to be tracked. The block associates these detections with existing tracks or creates new tracks. Detections should be provided for each frame in sequence to maintain consistent tracking. If multiple detection batches are provided, they will be merged before tracking. The detections must include bounding boxes and class names. After tracking, the output will include the same detections enhanced with track_id information, allowing identification of the same object across frames.",
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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ByteTrackerBlockV2(WorkflowBlock):
    def __init__(
        self,
    ):
        self._trackers: Dict[str, sv.ByteTrack] = {}

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
        return {OUTPUT_KEY: tracked_detections}

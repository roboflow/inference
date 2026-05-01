
# Byte Tracker

!!! warning "Deprecated"

    This block is deprecated. Use the [ByteTrack Tracker](byte_track_tracker.md) block instead.



## v3

??? "Class: `ByteTrackerBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/byte_tracker/v3.py">inference.core.workflows.core_steps.transformations.byte_tracker.v3.ByteTrackerBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



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


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/byte_tracker@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `track_activation_threshold` | `float` | Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios.. | ✅ |
| `minimum_matching_threshold` | `float` | IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects.. | ✅ |
| `minimum_consecutive_frames` | `int` | Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable.. | ✅ |
| `instances_cache_size` | `int` | Maximum number of track IDs to remember in the instance cache for determining if instances are new or already seen. Must be a positive integer. Default is 16384. The cache uses FIFO (First-In-First-Out) eviction - when the cache is full, the oldest track ID is removed to make room for new ones. Increasing this value (e.g., 32768-65536) maintains longer history of seen track IDs, allowing detection of reappearances after longer gaps, but uses more memory. Decreasing this value (e.g., 8192) reduces memory usage but may lose history of track IDs that appeared earlier, causing reappearing objects to be classified as new. Adjust based on video length and object reappearance patterns: use higher values for long videos or frequent reappearances, lower values for short videos or rare reappearances.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Byte Tracker` in version `v3`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Contours`](image_contours.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`SIFT`](sift.md), [`VLM As Detector`](vlm_as_detector.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Consensus`](detections_consensus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Byte Tracker` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image containing embedded video metadata (fps and video_identifier) required for ByteTrack initialization and tracking state management. The block extracts video_metadata from the WorkflowImageData object. The fps value is used to configure the tracker, and the video_identifier is used to maintain separate tracking state and instance cache for different videos. If fps is missing or invalid, the block defaults to 0 and logs a warning but continues operation. If processing multiple videos, each video should have a unique video_identifier in its metadata to maintain separate tracking states and caches. The block maintains persistent trackers and instance caches across frames for each video using the video_identifier..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions (object detection, instance segmentation, or keypoint detection) for the current video frame to be tracked. The block associates these detections with existing tracks or creates new tracks. Supports object detection, instance segmentation, and keypoint detection predictions. Detections should be provided for each frame in sequence to maintain consistent tracking. If multiple detection batches are provided, they will be merged before tracking. The detections must include bounding boxes and class names (and keypoints if keypoint detection). After tracking, the output will include the same detections enhanced with track_id information, allowing identification of the same object across frames..
        - `track_activation_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios..
        - `minimum_matching_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable..

    - output
    
        - `tracked_detections` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `new_instances` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `already_seen_instances` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Byte Tracker` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/byte_tracker@v3",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "track_activation_threshold": 0.25,
	    "lost_track_buffer": 30,
	    "minimum_matching_threshold": 0.8,
	    "minimum_consecutive_frames": 1,
	    "instances_cache_size": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `ByteTrackerBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/byte_tracker/v2.py">inference.core.workflows.core_steps.transformations.byte_tracker.v2.ByteTrackerBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



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


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/byte_tracker@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `track_activation_threshold` | `float` | Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios.. | ✅ |
| `minimum_matching_threshold` | `float` | IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects.. | ✅ |
| `minimum_consecutive_frames` | `int` | Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Byte Tracker` in version `v2`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Seg Preview`](seg_preview.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Contours`](image_contours.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`SORT Tracker`](sort_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Icon Visualization`](icon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Blur Visualization`](blur_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Consensus`](detections_consensus.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Velocity`](velocity.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detection Offset`](detection_offset.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SORT Tracker`](sort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Byte Tracker` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image containing embedded video metadata (fps and video_identifier) required for ByteTrack initialization and tracking state management. The block extracts video_metadata from the WorkflowImageData object. The fps value is used to configure the tracker, and the video_identifier is used to maintain separate tracking state for different videos. If fps is missing or invalid, the block defaults to 0 and logs a warning but continues operation. If processing multiple videos, each video should have a unique video_identifier in its metadata to maintain separate tracking states. The block maintains persistent trackers across frames for each video using the video_identifier. This version simplifies input by embedding metadata in the image object rather than requiring a separate metadata field..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions (object detection or instance segmentation) for the current video frame to be tracked. The block associates these detections with existing tracks or creates new tracks. Detections should be provided for each frame in sequence to maintain consistent tracking. If multiple detection batches are provided, they will be merged before tracking. The detections must include bounding boxes and class names. After tracking, the output will include the same detections enhanced with track_id information, allowing identification of the same object across frames..
        - `track_activation_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios..
        - `minimum_matching_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable..

    - output
    
        - `tracked_detections` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Byte Tracker` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/byte_tracker@v2",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "track_activation_threshold": 0.25,
	    "lost_track_buffer": 30,
	    "minimum_matching_threshold": 0.8,
	    "minimum_consecutive_frames": 1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `ByteTrackerBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/byte_tracker/v1.py">inference.core.workflows.core_steps.transformations.byte_tracker.v1.ByteTrackerBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Track objects across video frames using the ByteTrack algorithm to maintain consistent object identities, handle occlusions and temporary disappearances, associate detections with existing tracks, assign unique track IDs, and enable object behavior analysis, movement tracking, and video analytics workflows.

## How This Block Works

This block maintains object tracking across sequential video frames by associating detections from each frame with existing tracks and creating new tracks for new objects. The block:

1. Receives detection predictions for the current frame and video metadata (including frame rate and video identifier)
2. Initializes or retrieves a ByteTrack tracker for the video:
   - Creates a new tracker instance for each unique video (identified by video_identifier)
   - Stores trackers in memory to maintain tracking state across frames
   - Configures tracker with frame rate from metadata and user-specified parameters
   - Reuses existing tracker for subsequent frames of the same video
3. Merges multiple detection batches if provided:
   - Combines detections from multiple sources into a single detection set
   - Ensures all detections are processed together for consistent tracking
4. Updates tracks using ByteTrack algorithm:
   - **Track Association**: Matches current frame detections to existing tracks using IoU (Intersection over Union) matching
   - **Track Activation**: Creates new tracks for detections with confidence above track_activation_threshold that don't match existing tracks
   - **Track Matching**: Associates detections to tracks when IoU exceeds minimum_matching_threshold
   - **Track Persistence**: Maintains tracks that don't have matches using lost_track_buffer to handle temporary occlusions
   - **Track Validation**: Only outputs tracks that have been present for at least minimum_consecutive_frames consecutive frames
5. Handles tracking challenges:
   - **Occlusions**: Maintains tracks when objects are temporarily hidden (using lost_track_buffer frames)
   - **Missed Detections**: Keeps tracks alive through frames with missing detections
   - **False Positives**: Filters out tracks that don't persist long enough (minimum_consecutive_frames)
   - **Track Fragmentation**: Reduces track splits by maintaining buffer for lost objects
6. Assigns unique track IDs to each object:
   - Each tracked object receives a consistent track_id that persists across frames
   - Track IDs are assigned when tracks are activated and maintained throughout the video
   - Enables tracking individual objects across the entire video sequence
7. Returns tracked detections with track IDs:
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

This block receives detection predictions and video metadata, and produces tracked_detections with track IDs:

- **After object detection or instance segmentation blocks** to track detected objects across video frames (e.g., track detected objects in video, add track IDs to detections, maintain object identities across frames), enabling detection-to-tracking workflows
- **Before video analysis blocks** that require consistent object identities (e.g., analyze tracked object behavior, process object trajectories, work with tracked object data), enabling tracking-to-analysis workflows
- **Before visualization blocks** to display tracked objects with consistent colors or labels (e.g., visualize tracked objects, display track IDs, show object paths), enabling tracking visualization workflows
- **Before logic blocks** like Continue If to make decisions based on track information (e.g., continue if object is tracked, filter based on track IDs, make decisions using tracking data), enabling tracking-based decision workflows
- **Before counting or aggregation blocks** to count tracked objects accurately (e.g., count unique tracked objects, aggregate track statistics, process track data), enabling tracking-to-counting workflows
- **In video processing pipelines** where object tracking is part of a larger video analysis workflow (e.g., track objects in video pipelines, maintain identities in processing chains, enable video analytics), enabling video tracking pipeline workflows

## Requirements

This block requires detection predictions (object detection or instance segmentation) and video metadata with frame rate (fps) information. The video metadata must include a valid fps value for ByteTrack initialization. The block maintains tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For optimal tracking performance, detections should be provided consistently across frames. The algorithm works best with stable detection performance and handles temporary detection gaps through the lost_track_buffer mechanism.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/byte_tracker@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `track_activation_threshold` | `float` | Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios.. | ✅ |
| `minimum_matching_threshold` | `float` | IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects.. | ✅ |
| `minimum_consecutive_frames` | `int` | Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Byte Tracker` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Seg Preview`](seg_preview.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Contours`](image_contours.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`SORT Tracker`](sort_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Icon Visualization`](icon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Blur Visualization`](blur_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Consensus`](detections_consensus.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Velocity`](velocity.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detection Offset`](detection_offset.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SORT Tracker`](sort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Byte Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `metadata` (*[`video_metadata`](../kinds/video_metadata.md)*): Video metadata containing frame rate (fps) and video identifier information required for ByteTrack initialization and tracking state management. The fps value is used to configure the tracker, and the video_identifier is used to maintain separate tracking state for different videos. The metadata must include valid fps information - ByteTrack requires frame rate to initialize. If processing multiple videos, each video's metadata should have a unique video_identifier to maintain separate tracking states. The block maintains persistent trackers across frames for each video using the video_identifier..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions (object detection or instance segmentation) for the current video frame to be tracked. The block associates these detections with existing tracks or creates new tracks. Detections should be provided for each frame in sequence to maintain consistent tracking. If multiple detection batches are provided, they will be merged before tracking. The detections must include bounding boxes and class names. After tracking, the output will include the same detections enhanced with track_id information, allowing identification of the same object across frames..
        - `track_activation_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for activating new tracks from detections. Must be between 0.0 and 1.0. Default is 0.25. Only detections with confidence above this threshold can create new tracks. Increasing this threshold (e.g., 0.3-0.5) improves tracking accuracy and stability by only creating tracks from high-confidence detections, but might miss true detections with lower confidence. Decreasing this threshold (e.g., 0.15-0.2) increases tracking completeness by accepting lower-confidence detections, but risks introducing noise and instability from false positives. Adjust based on detection model performance: use lower values if detections are reliable, higher values if false positives are common..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to maintain a track when it's lost (no matching detections). Must be a positive integer. Default is 30 frames. When an object temporarily disappears (due to occlusion, missed detection, or leaving frame), the track is maintained for this many frames before being considered lost. Increasing this value (e.g., 50-100) enhances occlusion handling and significantly reduces track fragmentation or disappearance caused by brief detection gaps, but increases memory usage. Decreasing this value (e.g., 10-20) reduces memory usage but may cause tracks to disappear during short occlusions. Adjust based on occlusion frequency: use higher values for frequent occlusions, lower values for stable tracking scenarios..
        - `minimum_matching_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): IoU (Intersection over Union) threshold for matching detections to existing tracks. Must be between 0.0 and 1.0. Default is 0.8. Detections are associated with tracks when their bounding box IoU exceeds this threshold. Increasing this threshold (e.g., 0.85-0.95) improves tracking accuracy by requiring stronger spatial overlap, but risks track fragmentation when objects move quickly or detection boxes vary. Decreasing this threshold (e.g., 0.6-0.75) improves tracking completeness by accepting looser matches, but risks false positive associations and track drift. Adjust based on object movement speed and detection stability: use higher values for stable objects, lower values for fast-moving objects..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Minimum number of consecutive frames an object must be tracked before the track is considered valid and output. Must be a positive integer. Default is 1 (all tracks are immediately valid). Only tracks that persist for at least this many consecutive frames are included in the output. Increasing this value (e.g., 3-5) prevents the creation of accidental tracks from false detections or double detections, filtering out short-lived spurious tracks, but risks missing shorter legitimate tracks. Decreasing this value (e.g., 1) includes all tracks immediately, maximizing completeness but potentially including false tracks. Adjust based on false positive rate: use higher values if false detections are common, lower values if detections are reliable..

    - output
    
        - `tracked_detections` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Byte Tracker` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/byte_tracker@v1",
	    "metadata": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "track_activation_threshold": 0.25,
	    "lost_track_buffer": 30,
	    "minimum_matching_threshold": 0.8,
	    "minimum_consecutive_frames": 1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

